"""Proxy Mesh control plane (MVP).

Nodes enroll with a handshake (hardware and model kinds) and send heartbeats. The control
plane creates rounds naming participants, the model to bake, the dataset and the metrics it wants;
nodes pull their tasks from GET /tasks/{node_id}, ack them, and report back through the Egress
Gate (numbers-and-ids-only payload). Submissions land in SQLite; the global top-k pools them on
held-out-calibrated scores and gates every node on held-out precision at its cutoff (pooling.py).

No auth (hackathon build): any caller can act as any node_id, see CONTRACT.md §2.1.

Run:  uvicorn control_plane.app:app --port 8100
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import ValidationError

from .campaigns import CampaignError, CampaignService
from . import metrics as metrics_module
from .oracle import MockOracle, StaticOracle
from .partitioning import chunk_ids_for_partition
from .pooling import HeldoutCurve, NodeSubmission, select_topk
from .schemas import (
    CHUNK_ID_RE, HANDSHAKE_RESPONSE_EXAMPLE, HEARTBEAT_RESPONSE_EXAMPLE, TASK_EXAMPLE, TASKS_EXAMPLE,
    CreateCampaign, CreateRound, Handshake, HandshakeResponse, Heartbeat, HeartbeatResponse, ImportParquet,
    IngestChunks, SubmitPayload, TasksResponse, TaskView, example_response, submit_openapi_body,
)
from .storage import Store

log = logging.getLogger("proxy_mesh.control_plane")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_BODY_BYTES = int(os.getenv("PM_MAX_BODY_BYTES", str(32 * 1024 * 1024)))
MAX_PARQUET_BYTES = int(os.getenv("PM_MAX_PARQUET_BYTES", str(512 * 1024 * 1024)))
MAX_PARQUET_ROWS = int(os.getenv("PM_MAX_PARQUET_ROWS", "500000"))
PARQUET_UPLOAD_TTL_S = int(os.getenv("PM_PARQUET_UPLOAD_TTL_S", "3600"))
OFFLINE_AFTER_MISSED_BEATS = 3
DASHBOARD_HTML = (Path(__file__).with_name("dashboard.html")).read_text()


class Settings:
    def __init__(self) -> None:
        self.db_path = Path(os.getenv("PM_DB_PATH", "data/control_plane.sqlite3"))
        self.audit_log = Path(os.getenv("PM_AUDIT_LOG", "data/submissions.jsonl"))
        # Reliability Gate: Wilson 95% lower bound of held-out precision above a node's top-k cutoff.
        self.min_precision = float(os.getenv("PM_GATE_MIN_PRECISION", "0.2"))
        self.heartbeat_s = float(os.getenv("PM_HEARTBEAT_S", "3"))
        self.telemetry_retention_s = float(os.getenv("PM_TELEMETRY_RETENTION_S", str(7 * 24 * 3600)))
        self.telemetry_history_limit = int(os.getenv("PM_TELEMETRY_HISTORY_LIMIT", "240"))
        if self.telemetry_retention_s <= 0:
            raise ValueError("PM_TELEMETRY_RETENTION_S must be greater than zero")
        if not 1 <= self.telemetry_history_limit <= 5000:
            raise ValueError("PM_TELEMETRY_HISTORY_LIMIT must be between 1 and 5000")
        # Server-master campaigns (campaigns.py): the oracle judges the pool. Default is a hash-based
        # MockOracle; PM_GOLDEN_LABELS_PATH swaps in a StaticOracle over real precomputed labels (e.g.
        # tools/load_golden_shard.py pulling Llama-3 scores from FineWeb-Edu annotations) — same
        # interface either way, a real judge (a live LLM call) plugs into the same seam later.
        self.oracle_good_rate = float(os.getenv("PM_MOCK_ORACLE_GOOD_RATE", "0.1"))
        self.golden_labels_path = os.getenv("PM_GOLDEN_LABELS_PATH")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    store = Store(settings.db_path)
    if settings.golden_labels_path:
        with open(settings.golden_labels_path) as fh:
            oracle = StaticOracle(json.load(fh))
        log.info("oracle: static, %d golden labels from %s", len(oracle._labels), settings.golden_labels_path)
    else:
        oracle = MockOracle(good_rate=settings.oracle_good_rate)
    campaigns = CampaignService(store, oracle)
    settings.audit_log.parent.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Proxy Mesh control plane", version="0.11.0")
    app.state.settings = settings
    app.state.store = store
    app.state.campaigns = campaigns
    parquet_upload_dir = tempfile.TemporaryDirectory(prefix="proxy-mesh-parquet-")
    staged_parquet: Dict[str, Dict[str, Any]] = {}
    staged_parquet_lock = threading.Lock()
    app.state.parquet_upload_dir = parquet_upload_dir

    def cleanup_staged_parquet() -> None:
        cutoff = time.time() - PARQUET_UPLOAD_TTL_S
        with staged_parquet_lock:
            expired = [token for token, item in staged_parquet.items() if item["created_at"] < cutoff]
            for token in expired:
                Path(staged_parquet.pop(token)["path"]).unlink(missing_ok=True)

    def parquet_field_flags(field: pa.Field) -> Dict[str, bool]:
        string = pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
        numeric = (pa.types.is_boolean(field.type) or pa.types.is_integer(field.type)
                   or pa.types.is_floating(field.type) or pa.types.is_decimal(field.type))
        return {"can_text": string, "can_id": string, "can_label": numeric}

    def preferred_column(names: List[str], candidates: List[str]) -> Optional[str]:
        lowered = {name.lower(): name for name in names}
        return next((lowered[candidate] for candidate in candidates if candidate in lowered), None)

    # --- helpers ------------------------------------------------------------------
    def reliability(agg_stats: Dict[str, Any]) -> str:
        """The gate depends on the budget allocation, so it is decided at top-k; here only whether it can be."""
        return "pending" if HeldoutCurve.from_stats(agg_stats) else "unknown"

    def node_or_404(node_id: str) -> Dict[str, Any]:
        node = store.get_node(node_id)
        if not node:
            raise HTTPException(404, "unknown node_id: handshake first")
        return node

    def node_view(node: Dict[str, Any]) -> Dict[str, Any]:
        last = node["last_heartbeat"]
        online = last is not None and time.time() - last <= OFFLINE_AFTER_MISSED_BEATS * settings.heartbeat_s
        return {**node, "online": online}

    def task_view(task: Dict[str, Any]) -> Dict[str, Any]:
        spec = task["spec"] or {}
        operation = spec.get("operation") or {
            "train": "fresh", "score": True, "input_checkpoint_id": None,
            "output_checkpoint_id": task["round_id"],
        }
        return {
            "round_id": task["round_id"],
            "status": task["status"],
            "model": spec.get("model"),
            "operation": operation,
            "dataset_id": task["dataset_id"],
            "metrics": spec.get("metrics", []),
            "params": {**spec.get("params", {}), **task.get("participant_params", {})},
            "budget_k": task["budget_k"],
            "assigned_at": task["assigned_at"],
            "accepted_at": task["accepted_at"],
            "ack_url": f"/tasks/{task['node_id']}/{task['round_id']}/ack",
            "submit_url": f"/tasks/{task['node_id']}/{task['round_id']}/submit",
        }

    def round_view(round_row: Dict[str, Any]) -> Dict[str, Any]:
        subs = store.submissions_for_round(round_row["round_id"])
        return {
            **round_row,
            "participants": store.participants_for_round(round_row["round_id"]),
            "nodes": [
                {
                    "node_id": s["node_id"],
                    "revision": s["revision"],
                    "received_at": s["received_at"],
                    "n_scores": s["n_scores"],
                    "eval_spearman": s["eval_spearman"],
                    "trust": s["trust"],
                    "agg_stats": s["agg_stats"],
                }
                for s in subs
            ],
            "nodes_with_heldout": sum(1 for s in subs if s["trust"] == "pending"),
        }

    # --- service ------------------------------------------------------------------
    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        """Prometheus text format — nodes, rounds, submissions (every numeric agg_stats a node sent,
        as-is), campaigns, shards. See metrics.py. Point Grafana/curl straight at this, no auth."""
        body = metrics_module.render(store, settings.heartbeat_s, time.time())
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/metrics.json")
    def metrics_json() -> Dict[str, Any]:
        """Same read as /metrics, nested instead of flat — what dashboard.html polls."""
        return metrics_module.snapshot(store, settings.heartbeat_s, time.time())

    @app.get("/telemetry/history")
    def telemetry_history(
        node_id: Optional[str] = Query(default=None, description="Return one registered node only"),
        since: Optional[float] = Query(default=None, ge=0, description="Unix timestamp, inclusive"),
        limit: int = Query(default=settings.telemetry_history_limit, ge=1, le=5000,
                           description="Newest samples to return per node"),
    ) -> Dict[str, Any]:
        """Persisted heartbeat samples, oldest-first per node, for restoring telemetry graphs."""
        if node_id is not None:
            node_or_404(node_id)
        return {
            "generated_at": time.time(),
            "retention_s": settings.telemetry_retention_s,
            "limit_per_node": limit,
            "nodes": store.heartbeat_history(limit_per_node=limit, since=since, node_id=node_id),
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        """A control room the control plane serves itself: polls /metrics.json every couple of
        seconds and draws it — nodes, rounds, campaign progress. No Prometheus/Grafana to stand up
        for a demo. See dashboard.html."""
        return DASHBOARD_HTML

    # --- nodes: handshake, heartbeat, registry --------------------------------------
    @app.post("/nodes/handshake", status_code=201, response_model=HandshakeResponse,
              responses=example_response(HANDSHAKE_RESPONSE_EXAMPLE, 201))
    def handshake(body: Handshake) -> Dict[str, Any]:
        specs = body.model_dump(exclude={"name", "node_id"})
        if body.node_id:  # re-register: keep id, refresh specs
            node_or_404(body.node_id)
            node_id = body.node_id
            store.update_node(node_id, body.name, specs)
        else:
            node_id = f"{body.name}-{secrets.token_hex(3)}"
            store.register_node(node_id, body.name, specs)
        log.info("handshake node=%s kinds=%s", node_id, body.model_kinds)
        return {"node_id": node_id, "heartbeat_interval_s": settings.heartbeat_s,
                "tasks_url": f"/tasks/{node_id}"}

    @app.post("/nodes/{node_id}/heartbeat", response_model=HeartbeatResponse,
              responses=example_response(HEARTBEAT_RESPONSE_EXAMPLE))
    def heartbeat(node_id: str, body: Heartbeat) -> Dict[str, Any]:
        node_or_404(node_id)
        store.record_heartbeat(
            node_id,
            body.model_dump(exclude_none=True),
            settings.telemetry_retention_s,
        )
        return {"ok": True, "next_heartbeat_s": settings.heartbeat_s,
                "pending_tasks": len(store.tasks_for_node(node_id))}

    @app.get("/nodes")
    def list_nodes() -> List[Dict[str, Any]]:
        return [node_view(n) for n in store.list_nodes()]

    # --- rounds and tasks -----------------------------------------------------------
    @app.post("/rounds", status_code=201)
    def create_round(body: CreateRound) -> Dict[str, Any]:
        for p in body.participants:
            node = store.get_node(p.node_id)
            if not node:
                raise HTTPException(422, f"participant {p.node_id!r} is not a registered node")
            if body.model.kind not in node["specs"]["model_kinds"]:
                raise HTTPException(422, f"node {p.node_id!r} cannot bake model kind {body.model.kind!r}")
        operation = (
            body.operation.model_dump()
            if body.operation is not None
            else {"train": "fresh", "score": True, "input_checkpoint_id": None,
                  "output_checkpoint_id": body.round_id}
        )
        if operation["train"] != "skip" and operation.get("output_checkpoint_id") is None:
            operation["output_checkpoint_id"] = body.round_id
        spec = {
            "model": body.model.model_dump(), "operation": operation,
            "metrics": body.metrics, "params": body.params,
        }
        participants = [p.model_dump() for p in body.participants]
        if not store.create_round(body.round_id, body.budget_k, body.note, spec, participants):
            raise HTTPException(409, f"round {body.round_id!r} already exists")
        log.info("round=%s created model=%s participants=%s", body.round_id, spec["model"],
                 [p["node_id"] for p in participants])
        return round_view(store.get_round(body.round_id))

    @app.get("/tasks/{node_id}", response_model=TasksResponse, responses=example_response(TASKS_EXAMPLE))
    def get_tasks(node_id: str) -> Dict[str, Any]:
        """Open rounds the control plane wants this node in, not yet submitted."""
        node_or_404(node_id)
        return {"node_id": node_id, "tasks": [task_view(t) for t in store.tasks_for_node(node_id)]}

    @app.post("/tasks/{node_id}/{round_id}/ack", response_model=TaskView,
              responses=example_response({**TASK_EXAMPLE, "status": "accepted", "accepted_at": 1789245612.5}))
    def ack_task(node_id: str, round_id: str) -> Dict[str, Any]:
        """Node reports it has started working on the round. Idempotent."""
        node_or_404(node_id)
        task = store.get_task(node_id, round_id)
        if not task:
            raise HTTPException(404, "no such task for this node")
        if task["round_status"] != "open":
            raise HTTPException(409, "round is closed")
        store.mark_accepted(node_id, round_id)
        log.info("round=%s node=%s accepted", round_id, node_id)
        return task_view(store.get_task(node_id, round_id))

    @app.get("/rounds/{round_id}")
    def get_round(round_id: str) -> Dict[str, Any]:
        row = store.get_round(round_id)
        if not row:
            raise HTTPException(404, "round not found")
        return round_view(row)

    @app.post("/rounds/{round_id}/close")
    def close_round(round_id: str) -> Dict[str, Any]:
        if not store.get_round(round_id):
            raise HTTPException(404, "round not found")
        store.set_round_status(round_id, "closed")
        return round_view(store.get_round(round_id))

    @app.post("/tasks/{node_id}/{round_id}/submit", status_code=201,
              openapi_extra={"requestBody": submit_openapi_body()},
              description="node_id and round_id in the path must match the body.")
    async def submit(node_id: str, round_id: str, request: Request) -> JSONResponse:
        """Egress Gate: the node reports chunk ids, numeric scores and numeric metrics for its task."""
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(413, "payload too large")
        try:
            raw = json.loads(body)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"invalid JSON: {exc}") from exc
        try:
            payload = SubmitPayload.model_validate(raw)
        except ValidationError as exc:
            # Egress contract violation: report where, never echo the offending values back.
            errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
            return JSONResponse(status_code=422, content={"detail": errors})

        if (payload.node_id, payload.round_id) != (node_id, round_id):
            raise HTTPException(400, "node_id/round_id in path and payload differ")
        node_or_404(node_id)
        task = store.get_task(node_id, round_id)
        if task is None:
            raise HTTPException(404, "no such task for this node")
        if task["round_status"] != "open":
            raise HTTPException(409, "round is closed")

        agg_stats = payload.agg_stats.model_dump(exclude_none=True)
        missing = [m for m in (task["spec"] or {}).get("metrics", []) if m not in agg_stats]
        if missing:
            return JSONResponse(status_code=422, content={"detail": [{
                "loc": ["body", "agg_stats"], "msg": f"missing requested metrics: {missing}",
                "type": "missing_metrics"}]})

        rho = payload.agg_stats.eval_spearman
        trust = reliability(agg_stats)
        payload_sha256 = hashlib.sha256(body).hexdigest()
        result = store.upsert_submission(
            round_id=round_id,
            node_id=payload.node_id,
            scores=[s.model_dump() for s in payload.scores],
            agg_stats=agg_stats,
            eval_spearman=rho,
            trust=trust,
            payload_sha256=payload_sha256,
        )
        store.mark_submitted(payload.node_id, round_id)
        # Audit log keeps metadata only; scores live in SQLite.
        with settings.audit_log.open("a") as fh:
            fh.write(json.dumps({
                "ts": time.time(), "round_id": round_id, "node_id": payload.node_id,
                "revision": result["revision"], "n_scores": len(payload.scores),
                "eval_spearman": rho, "trust": trust, "payload_sha256": payload_sha256,
            }) + "\n")
        log.info("round=%s node=%s rev=%d n=%d rho=%s trust=%s", round_id, payload.node_id,
                 result["revision"], len(payload.scores), rho, trust)
        return JSONResponse(status_code=201, content={
            "accepted": True,
            "round_id": round_id,
            "node_id": payload.node_id,
            "revision": result["revision"],
            "n_scores": len(payload.scores),
            "trust": trust,
            "gate_min_precision": settings.min_precision,
            "payload_sha256": payload_sha256,
        })

    @app.get("/rounds/{round_id}/topk")
    def topk(
        round_id: str,
        k: Optional[int] = Query(default=None, ge=1, description="defaults to round budget_k"),
        include_untrusted: bool = Query(default=False, description="report the gate, do not enforce it"),
        include_unknown: bool = Query(default=False, description="pool nodes without a held-out curve "
                                                                 "(impossible with normalize=calibrated)"),
        normalize: str = Query(default="calibrated", pattern="^(calibrated|zscore|rank|none)$"),
    ) -> Dict[str, Any]:
        """Pool per-node scores into one global ranking and apply the Reliability Gate.

        `calibrated` scores a chunk by its node's held-out precision in the chunk's rank band, so the
        budget follows the expected number of good docs; `zscore` / `rank` give every node the same
        share. A node is trusted when the Wilson lower bound of held-out precision above its cutoff
        reaches PM_GATE_MIN_PRECISION; failing nodes are dropped and the budget reallocated.
        """
        rnd = store.get_round(round_id)
        if not rnd:
            raise HTTPException(404, "round not found")
        k = k or rnd["budget_k"]
        nodes = []
        for sub in store.submissions_for_round(round_id):
            rows = store.scores_for_submission(sub["id"])
            nodes.append(NodeSubmission(sub["node_id"], [r["chunk_id"] for r in rows], [r["score"] for r in rows],
                                        HeldoutCurve.from_stats(sub["agg_stats"])))
        result = select_topk(nodes, k, normalize, settings.min_precision,
                             enforce_gate=not include_untrusted, include_unknown=include_unknown)
        return {"round_id": round_id, "k": k, "normalize": normalize,
                "gate_min_precision": settings.min_precision, **result}

    # --- server-master: shards (chunk pool) and campaigns ---------------------------------------
    @app.post(
        "/imports/parquet/inspect",
        status_code=201,
        openapi_extra={"requestBody": {"required": True, "content": {
            "application/vnd.apache.parquet": {"schema": {"type": "string", "format": "binary"}},
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
        }}},
    )
    async def inspect_parquet(request: Request) -> Dict[str, Any]:
        """Stage a raw Parquet file and return its schema so the operator can map columns."""
        cleanup_staged_parquet()
        declared_size = request.headers.get("content-length")
        if declared_size:
            try:
                if int(declared_size) > MAX_PARQUET_BYTES:
                    raise HTTPException(413, f"Parquet file exceeds {MAX_PARQUET_BYTES} bytes")
            except ValueError as exc:
                raise HTTPException(400, "invalid Content-Length") from exc

        token = secrets.token_hex(16)
        path = Path(parquet_upload_dir.name) / f"{token}.parquet"
        size = 0
        try:
            with path.open("wb") as target:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_PARQUET_BYTES:
                        raise HTTPException(413, f"Parquet file exceeds {MAX_PARQUET_BYTES} bytes")
                    target.write(chunk)
            if size == 0:
                raise HTTPException(422, "Parquet file is empty")
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            n_rows = parquet.metadata.num_rows
            if n_rows == 0:
                raise HTTPException(422, "Parquet file contains no rows")
            if n_rows > MAX_PARQUET_ROWS:
                raise HTTPException(422, f"Parquet file has {n_rows} rows; limit is {MAX_PARQUET_ROWS}")
            if len(schema.names) != len(set(schema.names)):
                raise HTTPException(422, "Parquet file has duplicate column names")
        except HTTPException:
            path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, f"invalid Parquet file: {exc}") from exc

        columns = [{"name": field.name, "type": str(field.type), **parquet_field_flags(field)}
                   for field in schema]
        text_names = [column["name"] for column in columns if column["can_text"]]
        id_names = [column["name"] for column in columns if column["can_id"]]
        label_names = [column["name"] for column in columns if column["can_label"]]
        text_column = preferred_column(text_names, ["text", "content", "document", "body"])
        if text_column is None and text_names:
            text_column = text_names[0]
        id_column = preferred_column(id_names, ["chunk_id", "content_hash", "sha256"])
        label_column = preferred_column(label_names, ["label", "score", "target", "quality_score"])
        threshold_mode = bool(label_column and "score" in label_column.lower())
        filename = Path(unquote(request.headers.get("x-filename", "dataset.parquet"))).name[:128]
        staged = {
            "path": str(path), "filename": filename, "size_bytes": size,
            "n_rows": n_rows, "schema": schema, "created_at": time.time(),
        }
        with staged_parquet_lock:
            staged_parquet[token] = staged
        return {
            "upload_token": token,
            "filename": filename,
            "size_bytes": size,
            "n_rows": n_rows,
            "columns": columns,
            "defaults": {
                "text_column": text_column,
                "id_column": id_column,
                "label_column": label_column,
                "label_mode": "threshold" if threshold_mode else "binary",
                "label_threshold": 3 if threshold_mode else None,
            },
            "expires_in_s": PARQUET_UPLOAD_TTL_S,
        }

    @app.get("/shards")
    def list_shards() -> Dict[str, Any]:
        """Dataset inventory for operators and the built-in dataset browser."""
        return {"shards": store.list_shards()}

    @app.post("/shards/{shard_id}/chunks", status_code=201)
    def ingest_chunks(shard_id: str, body: IngestChunks) -> Dict[str, Any]:
        """Operator loads (or extends) a shard the server holds. Nodes never see this text directly —
        they read it back via GET /shards/{shard_id}/chunks to train/score, same as any node call."""
        n_new = store.add_chunks(shard_id, [c.model_dump() for c in body.chunks])
        return {"shard_id": shard_id, "received": len(body.chunks), "new": n_new,
                "total": len(store.chunk_ids_for_shard(shard_id))}

    @app.post("/shards/{shard_id}/imports/parquet", status_code=201)
    def import_parquet(shard_id: str, body: ImportParquet) -> Dict[str, Any]:
        """Import mapped Parquet columns into a shard; optional labels become server-held oracle labels."""
        cleanup_staged_parquet()
        with staged_parquet_lock:
            staged = staged_parquet.get(body.upload_token)
        if not staged:
            raise HTTPException(404, "Parquet upload token not found or expired")

        schema: pa.Schema = staged["schema"]
        fields = {field.name: field for field in schema}
        text_field = fields.get(body.text_column)
        if text_field is None:
            raise HTTPException(422, f"text column {body.text_column!r} not found")
        if not parquet_field_flags(text_field)["can_text"]:
            raise HTTPException(422, f"text column {body.text_column!r} must be a string")
        if body.id_column:
            id_field = fields.get(body.id_column)
            if id_field is None:
                raise HTTPException(422, f"id column {body.id_column!r} not found")
            if not parquet_field_flags(id_field)["can_id"]:
                raise HTTPException(422, f"id column {body.id_column!r} must be a string")
        if body.label_column:
            label_field = fields.get(body.label_column)
            if label_field is None:
                raise HTTPException(422, f"label column {body.label_column!r} not found")
            if not parquet_field_flags(label_field)["can_label"]:
                raise HTTPException(422, f"label column {body.label_column!r} must be numeric or boolean")
        if body.label_threshold is not None and not math.isfinite(body.label_threshold):
            raise HTTPException(422, "label threshold must be finite")

        selected_columns = list(dict.fromkeys(filter(None, [
            body.text_column, body.id_column, body.label_column,
        ])))
        stats = {"rows_read": 0, "duplicates_in_file": 0, "skipped_empty": 0,
                 "skipped_oversized": 0, "unlabeled": 0}
        seen: set[str] = set()

        def rows():
            parquet = pq.ParquetFile(staged["path"])
            for batch in parquet.iter_batches(columns=selected_columns, batch_size=2000):
                values = batch.to_pydict()
                for index in range(batch.num_rows):
                    stats["rows_read"] += 1
                    text = values[body.text_column][index]
                    if text is None or not text.strip():
                        stats["skipped_empty"] += 1
                        continue
                    if len(text) > 20_000:
                        stats["skipped_oversized"] += 1
                        continue
                    if body.id_column:
                        raw_id = values[body.id_column][index]
                        chunk_id = raw_id if isinstance(raw_id, str) else ""
                        if not CHUNK_ID_RE.fullmatch(chunk_id):
                            raise ValueError(
                                f"row {stats['rows_read']} has an invalid chunk id in {body.id_column!r}"
                            )
                    else:
                        chunk_id = hashlib.sha256(text.encode()).hexdigest()
                    if chunk_id in seen:
                        stats["duplicates_in_file"] += 1
                        continue
                    seen.add(chunk_id)

                    label = None
                    if body.label_column:
                        raw_label = values[body.label_column][index]
                        if raw_label is None:
                            stats["unlabeled"] += 1
                        else:
                            numeric_label = float(raw_label)
                            if not math.isfinite(numeric_label):
                                raise ValueError(
                                    f"row {stats['rows_read']} has a non-finite label in {body.label_column!r}"
                                )
                            if body.label_threshold is None:
                                if numeric_label not in (0, 1):
                                    raise ValueError(
                                        f"row {stats['rows_read']} label is {numeric_label}; "
                                        "select threshold conversion for non-binary labels"
                                    )
                                label = int(numeric_label)
                            else:
                                label = int(numeric_label >= body.label_threshold)
                    yield chunk_id, text, label

        source = (f"parquet:{staged['filename']}:{body.label_column}"
                  if body.label_column else f"parquet:{staged['filename']}")
        try:
            result = store.import_chunks(shard_id, rows(), label_source=source)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(422, f"could not import Parquet file: {exc}") from exc

        with staged_parquet_lock:
            completed = staged_parquet.pop(body.upload_token, None)
        if completed:
            Path(completed["path"]).unlink(missing_ok=True)
        return {"shard_id": shard_id, "filename": staged["filename"], "source": source,
                **result, **stats}

    @app.get("/shards/{shard_id}/chunks")
    def get_chunks(
        shard_id: str,
        partition: Optional[int] = Query(default=None, ge=0),
        n_partitions: Optional[int] = Query(default=None, ge=1),
    ) -> Dict[str, Any]:
        if (partition is None) != (n_partitions is None):
            raise HTTPException(422, "partition and n_partitions must be provided together")
        if partition is not None and partition >= n_partitions:
            raise HTTPException(422, "partition must be smaller than n_partitions")
        ids = store.chunk_ids_for_shard(shard_id)
        if partition is not None:
            ids = chunk_ids_for_partition(ids, partition, n_partitions)
        texts = store.chunk_texts(ids)
        return {"shard_id": shard_id, "chunks": [{"chunk_id": c, "text": texts[c]} for c in ids]}

    @app.get("/shards/{shard_id}/preview")
    def preview_shard(
        shard_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=25, ge=1, le=100),
        q: Optional[str] = Query(default=None, max_length=200,
                                 description="Case-insensitive substring in chunk id or text"),
    ) -> Dict[str, Any]:
        result = store.shard_preview(shard_id, offset=offset, limit=limit, query=q or None)
        if result is None:
            raise HTTPException(404, "shard not found")
        return {"shard_id": shard_id, "offset": offset, "limit": limit, "query": q, **result}

    @app.get("/shards/{shard_id}/labels")
    def get_labels(shard_id: str) -> Dict[str, Any]:
        """Every label the oracle has produced so far for this shard, across every campaign that used
        it — labels are chunk-scoped, not campaign-scoped. A node trains on these."""
        return {"shard_id": shard_id, "labels": store.labels_for_shard(shard_id)}

    @app.post("/campaigns", status_code=201)
    def create_campaign(body: CreateCampaign) -> Dict[str, Any]:
        """Create a throughput-sharded or mixture-of-experts campaign and its first round."""
        try:
            create = campaigns.create_sharded if body.mode == "sharded" else campaigns.create_experts
            return create(body.campaign_id, body.shard_id, body.model.model_dump(), body.metrics,
                          body.node_ids, body.schedule, body.train_mode, body.strategy, body.k_frac,
                          body.good_min, body.seed)
        except CampaignError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/campaigns/{campaign_id}")
    def get_campaign(campaign_id: str) -> Dict[str, Any]:
        row = store.get_campaign(campaign_id)
        if not row:
            raise HTTPException(404, "campaign not found")
        return {**row, "current_round": store.current_campaign_round(campaign_id)}

    @app.post("/campaigns/{campaign_id}/advance")
    def advance_campaign(campaign_id: str) -> Dict[str, Any]:
        """Consume all current submissions, label each domain further, or finalize the campaign."""
        try:
            result = campaigns.advance(campaign_id)
        except CampaignError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"campaign_id": campaign_id, "status": result.status, "detail": result.detail,
                "round_id": result.round_id}

    return app


app = create_app()
