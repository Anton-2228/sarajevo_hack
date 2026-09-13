"""A real control plane, in-process.

A genuine socket rather than a mocked transport, because the things most likely
to be wrong -- timeouts, retries, dropped connections, multi-megabyte bodies,
header handling -- are exactly the things a mock cannot exercise.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

# The contract has no auth at all; the fake mirrors that.

# Contract 0.7.0. The shard is the corpus: the fake serves texts and labels the
# same way the live server does, including the parts that are easy to get wrong
# -- an unknown shard answers 200 with an empty list rather than 404, and
# `partition` without `n_partitions` is a 422.
DEFAULT_SHARD = "wave1"

# Contract 0.9.0 / METRICS_GUIDE: the closed set `stage` accepts.
STAGES = ("downloading", "training", "scoring", "uploading")


def chunk_id_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def partition_of(chunk_id: str, n_partitions: int) -> int:
    """CONTRACT.md 4.7, as the server computes it. Verified against the live one."""
    return int(chunk_id[:8], 16) % n_partitions


@dataclass
class FakeState:
    """Everything the fake server knows, and everything the tests assert on."""

    url: str = ""
    nodes: dict[str, dict] = field(default_factory=dict)
    tasks: dict[str, dict] = field(default_factory=dict)
    submissions: dict[str, dict] = field(default_factory=dict)
    revisions: dict[str, int] = field(default_factory=dict)
    requests: list[tuple[str, str]] = field(default_factory=list)
    # shard_id -> [{chunk_id, text}] and shard_id -> {chunk_id: int}
    shards: dict[str, list[dict]] = field(default_factory=dict)
    labels: dict[str, dict[str, int]] = field(default_factory=dict)
    campaigns: dict[str, dict] = field(default_factory=dict)
    # Every heartbeat body the fake accepted, for telemetry assertions.
    heartbeats: list[dict] = field(default_factory=list)
    # path -> list of (status, body) to serve before behaving normally
    scripted: dict[str, list[tuple[int, Any]]] = field(default_factory=dict)
    delays: dict[str, float] = field(default_factory=dict)
    handshake_count: int = 0
    heartbeat_count: int = 0
    pending_tasks: int = 0

    def seed_shard(
        self,
        shard_id: str = DEFAULT_SHARD,
        n_chunks: int = 60,
        n_labels: int = 40,
    ) -> str:
        """A shard with enough labelled classes to train and hold out on.

        Labels are integers, as the oracle produces them, and spread over ten
        values so a stratified split has something to work with.
        """
        chunks = []
        labels: dict[str, int] = {}
        for i in range(n_chunks):
            text = f"document {i} about topic {i % 10} with some filler words here"
            chunk_id = chunk_id_of(text)
            chunks.append({"chunk_id": chunk_id, "text": text})
            if i < n_labels:
                labels[chunk_id] = i % 10
        self.shards[shard_id] = chunks
        self.labels[shard_id] = labels
        return shard_id

    def label_every_partition(self, shard_id: str, n_partitions: int, per: int = 24) -> None:
        """Make sure each domain has labels, so a campaign node can train.

        Seeding by index alone can leave a partition empty, and a node with no
        labels in its own domain fails the round for a reason that has nothing
        to do with what the test is checking.
        """
        labels = self.labels.setdefault(shard_id, {})
        counts = {p: 0 for p in range(n_partitions)}
        for index, chunk in enumerate(self.shards.get(shard_id, [])):
            owner = partition_of(chunk["chunk_id"], n_partitions)
            if counts[owner] < per:
                labels[chunk["chunk_id"]] = index % 10
                counts[owner] += 1

    def add_task(
        self, node_id: str, round_id: str, *, seed: bool = True, **overrides: Any
    ) -> dict:
        # A task needs a shard that exists: the corpus is server-held now, so a
        # round pointing at nothing is a round the node must refuse. Pass
        # seed=False to build exactly that round.
        dataset_id = overrides.get("dataset_id", DEFAULT_SHARD)
        if seed and dataset_id not in self.shards:
            self.seed_shard(dataset_id)
        task = {
            "round_id": round_id,
            "status": "assigned",
            "model": {"kind": "classifier", "id": "quality-clf-v1"},
            # Contract 0.9.0: every task carries an explicit lifecycle
            # instruction. The default is the one a first round gets.
            "operation": {
                "train": "fresh",
                "score": True,
                "input_checkpoint_id": None,
                "output_checkpoint_id": f"{round_id}-ckpt",
            },
            "dataset_id": dataset_id,
            "metrics": ["eval_spearman", "n_dedup_dropped"],
            "params": {"proxy_lr": 0.5},
            "budget_k": 100,
            "assigned_at": time.time(),
            "accepted_at": None,
            "ack_url": f"/tasks/{node_id}/{round_id}/ack",
            "submit_url": f"/tasks/{node_id}/{round_id}/submit",
        }
        task.update(overrides)
        self.tasks[round_id] = task
        return task

    def add_campaign_task(
        self,
        node_id: str,
        round_id: str,
        *,
        mode: str = "sharded",
        partition: int = 0,
        n_partitions: int = 2,
        n_labels: int = 24,
        **overrides: Any,
    ) -> dict:
        """A campaign round: routing rides in `params`, with a string `mode`."""
        shard_id = overrides.pop("dataset_id", DEFAULT_SHARD)
        if shard_id not in self.shards:
            self.seed_shard(shard_id)
        self.label_every_partition(shard_id, n_partitions, per=n_labels)
        params = {
            "proxy_lr": 0.5,
            "mode": mode,
            "partition": partition,
            "n_partitions": n_partitions,
            "n_labels": n_labels,
        }
        params.update(overrides.pop("params", {}))
        return self.add_task(
            node_id, round_id, dataset_id=shard_id, params=params, **overrides
        )

    def script(self, path: str, *responses: tuple[int, Any]) -> None:
        """Serve these responses, in order, before resuming normal behaviour."""
        self.scripted[path] = list(responses)

    def closed_rounds(self) -> list[str]:
        return [path for _, path in self.requests if path.endswith("/close")]


HELDOUT_KEYS = (
    "ho_n", "ho_good",
    *(f"ho_{part}_q{q}" for q in ("01", "02", "05", "10", "20", "30", "50")
      for part in ("n", "good")),
)


def _check_agg_stats(agg: dict, scores: list) -> dict | None:
    """The validation the live server actually performs, verified against it."""
    if "n_chunks" not in agg:
        return {"loc": ["agg_stats"], "msg": "n_chunks is required", "type": "value_error"}
    if agg["n_chunks"] < len(scores):
        return {"loc": [], "msg": "agg_stats.n_chunks is smaller than len(scores)",
                "type": "value_error"}
    for key, value in agg.items():
        if value is None:
            # Only these two are declared nullable.
            if key in ("eval_spearman", "proxy_lr"):
                continue
            return {"loc": ["agg_stats"],
                    "msg": f"agg_stats[{key!r}] must be a number (no strings/objects)",
                    "type": "value_error"}
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return {"loc": ["agg_stats"],
                    "msg": f"agg_stats[{key!r}] must be a number (no strings/objects)",
                    "type": "value_error"}

    present = [k for k in HELDOUT_KEYS if k in agg]
    if present and len(present) != len(HELDOUT_KEYS):
        return {"loc": ["agg_stats"],
                "msg": f"held-out curve needs all of {list(HELDOUT_KEYS)!r}",
                "type": "value_error"}
    if present:
        if agg["ho_n"] < 1 or agg["ho_good"] > agg["ho_n"]:
            return {"loc": ["agg_stats"], "msg": "bad held-out totals", "type": "value_error"}
        last_n = last_good = 0
        for q in ("01", "02", "05", "10", "20", "30", "50"):
            n, good = agg[f"ho_n_q{q}"], agg[f"ho_good_q{q}"]
            if good > n or n < last_n or good < last_good or n > agg["ho_n"] or good > agg["ho_good"]:
                return {"loc": ["agg_stats"],
                        "msg": f"held-out curve is not monotone at q{q}", "type": "value_error"}
            last_n, last_good = n, good
    return None


class _Handler(BaseHTTPRequestHandler):
    state: FakeState

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: A003 - silence the stdlib access log
        return

    # -- dispatch ---------------------------------------------------------

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        path = self.path.split("?")[0]
        self._query = parse_qs(urlparse(self.path).query)
        self.state.requests.append((method, path))

        # Drain the body before anything else. On a keep-alive connection an
        # unread request body is parsed as the next request line, which
        # corrupts every later call on that socket -- and a scripted response
        # that skips its own routing is exactly where that gets forgotten.
        self._cached_body = self._read_body()

        delay = self.state.delays.get(path)
        if delay:
            time.sleep(delay)

        scripted = self.state.scripted.get(path)
        if scripted:
            status, body = scripted.pop(0)
            self._send(status, body)
            return

        try:
            status, body = self._route(method, path)
        except Exception as error:  # noqa: BLE001 - the fake must not hang a test
            status, body = 500, {"detail": f"fake server error: {error}"}
        self._send(status, body)

    def _route(self, method: str, path: str) -> tuple[int, Any]:
        parts = [p for p in path.split("/") if p]

        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/nodes" and method == "GET":
            return 200, list(self.state.nodes.values())
        if path == "/nodes/handshake":
            return self._handshake()
        if len(parts) == 3 and parts[0] == "nodes" and parts[2] == "heartbeat":
            return self._heartbeat(parts[1])
        if len(parts) == 2 and parts[0] == "tasks" and method == "GET":
            return self._get_tasks(parts[1])
        if len(parts) == 4 and parts[0] == "tasks" and parts[3] == "ack":
            return self._ack(parts[1], parts[2])
        if len(parts) == 4 and parts[0] == "tasks" and parts[3] == "submit":
            return self._submit(parts[1], parts[2])
        if len(parts) == 3 and parts[0] == "shards" and parts[2] == "chunks":
            if method == "GET":
                return self._get_chunks(parts[1])
            return self._put_chunks(parts[1])
        if len(parts) == 3 and parts[0] == "shards" and parts[2] == "labels":
            return self._get_labels(parts[1])
        if path == "/rounds" and method == "POST":
            return 201, {"round_id": (self._body() or {}).get("round_id")}
        if len(parts) == 2 and parts[0] == "rounds":
            if parts[1] not in self.state.tasks:
                # Matches the live server: {"detail": "round not found"}, which
                # must NOT read as a lost enrolment.
                return 404, {"detail": "round not found"}
            return 200, {"round_id": parts[1], "submissions": len(self.state.submissions)}
        if path == "/campaigns" and method == "POST":
            return self._create_campaign()
        if len(parts) == 2 and parts[0] == "campaigns" and method == "GET":
            campaign = self.state.campaigns.get(parts[1])
            if campaign is None:
                return 404, {"detail": "campaign not found"}
            return 200, campaign

        return 404, {"detail": "not found"}

    # -- shards (contract 0.7.0) ------------------------------------------

    def _get_chunks(self, shard_id: str) -> tuple[int, Any]:
        """The node's corpus. Note the two behaviours that trip clients up."""
        raw_partition = self._query.get("partition", [None])[0]
        raw_n = self._query.get("n_partitions", [None])[0]

        # Both or neither: the live server 422s on one alone.
        if (raw_partition is None) != (raw_n is None):
            return 422, {
                "detail": [
                    {
                        "loc": ["query", "partition"],
                        "msg": "partition and n_partitions must be given together",
                        "type": "value_error",
                    }
                ]
            }

        # An unknown shard is 200 with an empty list, NOT a 404. Emptiness is
        # the only signal a node gets that its workload does not exist.
        chunks = self.state.shards.get(shard_id, [])

        if raw_partition is not None:
            try:
                partition, n_partitions = int(raw_partition), int(raw_n)
            except ValueError:
                return 422, {"detail": [{"loc": ["query"], "msg": "not an integer",
                                         "type": "int_parsing"}]}
            if n_partitions < 1 or not 0 <= partition < n_partitions:
                return 422, {
                    "detail": [
                        {
                            "loc": ["query", "partition"],
                            "msg": "partition out of range for n_partitions",
                            "type": "value_error",
                        }
                    ]
                }
            chunks = [
                c for c in chunks
                if partition_of(c["chunk_id"], n_partitions) == partition
            ]

        return 200, {"shard_id": shard_id, "chunks": chunks}

    def _get_labels(self, shard_id: str) -> tuple[int, Any]:
        # Shard-wide, campaign-agnostic, and with nothing marking a held-out
        # split -- carving one out is the node's job.
        return 200, {"shard_id": shard_id, "labels": self.state.labels.get(shard_id, {})}

    def _put_chunks(self, shard_id: str) -> tuple[int, Any]:
        payload = self._body() or {}
        incoming = payload.get("chunks")
        if not isinstance(incoming, list) or not 1 <= len(incoming) <= 5_000:
            return 422, {"detail": [{"loc": ["body", "chunks"], "msg": "1..5000 chunks",
                                     "type": "value_error"}]}
        existing = self.state.shards.setdefault(shard_id, [])
        known = {c["chunk_id"] for c in existing}
        new = 0
        for chunk in incoming:
            # An already known chunk_id is not overwritten.
            if chunk["chunk_id"] in known:
                continue
            existing.append({"chunk_id": chunk["chunk_id"], "text": chunk["text"]})
            known.add(chunk["chunk_id"])
            new += 1
        return 201, {
            "shard_id": shard_id,
            "received": len(incoming),
            "new": new,
            "total": len(existing),
        }

    def _create_campaign(self) -> tuple[int, Any]:
        """Enough of a campaign to prove the node drives the right calls.

        Creating one places the first slice of labels and opens round 1, exactly
        as CONTRACT.md 4.8 describes.
        """
        body = self._body() or {}
        campaign_id = body.get("campaign_id")
        shard_id = body.get("shard_id")
        node_ids = body.get("node_ids") or []
        schedule = body.get("schedule") or [1]
        mode = body.get("mode", "sharded")
        n_partitions = len(node_ids)

        if not campaign_id or not shard_id or not node_ids:
            return 422, {"detail": [{"loc": ["body"], "msg": "missing fields",
                                     "type": "value_error"}]}

        self.state.label_every_partition(shard_id, n_partitions, per=schedule[0])
        round_id = f"{campaign_id}-r1"
        for index, node_id in enumerate(node_ids):
            self.state.add_campaign_task(
                node_id,
                round_id,
                mode=mode,
                partition=index,
                n_partitions=n_partitions,
                n_labels=schedule[0],
                dataset_id=shard_id,
                model=body.get("model"),
                metrics=body.get("metrics") or [],
            )
        self.state.campaigns[campaign_id] = {
            "campaign_id": campaign_id,
            "spec": body,
            "rounds_done": 0,
            "status": "running",
            "current_round": round_id,
        }
        return 201, self.state.campaigns[campaign_id]

    # -- endpoints --------------------------------------------------------

    def _handshake(self) -> tuple[int, Any]:
        payload = self._body() or {}
        self.state.handshake_count += 1

        # additionalProperties:false, and contract 0.7.0 dropped `datasets`.
        # Verified against the live server: sending it fails enrolment outright,
        # so the fake has to be just as strict or it would hide the break.
        extra = set(payload) - {"name", "node_id", "hardware", "software", "model_kinds"}
        if extra:
            return 422, {
                "detail": [
                    {"loc": ["body", key], "msg": "Extra inputs are not permitted",
                     "type": "extra_forbidden"}
                    for key in sorted(extra)
                ]
            }

        node_id = payload.get("node_id") or f"{payload.get('name', 'node')}-fake"
        self.state.nodes[node_id] = {"node_id": node_id, "name": payload.get("name")}
        return 201, {
            "node_id": node_id,
            "heartbeat_interval_s": 0.05,
            "tasks_url": f"/tasks/{node_id}",
        }

    def _heartbeat(self, node_id: str) -> tuple[int, Any]:
        if node_id not in self.state.nodes:
            return 404, {"detail": "unknown node_id: handshake first"}

        # Validate the way the live server does, so a telemetry mistake shows up
        # here rather than as a node that silently looks offline.
        payload = self._body() or {}
        extra = set(payload) - {"status", "stage", "round_id", "load"}
        if extra:
            return 422, {
                "detail": [
                    {"loc": ["body", key], "msg": "Extra inputs are not permitted",
                     "type": "extra_forbidden"}
                    for key in sorted(extra)
                ]
            }
        stage = payload.get("stage")
        if stage is not None and stage not in STAGES:
            return 422, {
                "detail": [
                    {"loc": ["body", "stage"], "msg": f"stage must be one of {STAGES}",
                     "type": "enum"}
                ]
            }
        load = payload.get("load") or {}
        if len(load) > 32:
            return 422, {"detail": [{"loc": ["body", "load"], "msg": "at most 32 gauges",
                                     "type": "value_error"}]}
        for key, value in load.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return 422, {
                    "detail": [
                        {"loc": ["body", "load", key],
                         "msg": f"load[{key!r}] must be a finite number",
                         "type": "value_error"}
                    ]
                }
            if value != value or value in (float("inf"), float("-inf")):
                return 422, {
                    "detail": [
                        {"loc": ["body", "load", key],
                         "msg": f"load[{key!r}] must be finite", "type": "value_error"}
                    ]
                }

        self.state.heartbeats.append(payload)
        self.state.heartbeat_count += 1
        open_tasks = sum(1 for t in self.state.tasks.values() if t["status"] != "submitted")
        return 200, {
            "ok": True,
            "next_heartbeat_s": 0.05,
            "pending_tasks": self.state.pending_tasks or open_tasks,
        }

    def _get_tasks(self, node_id: str) -> tuple[int, Any]:
        if node_id not in self.state.nodes:
            return 404, {"detail": "unknown node_id: handshake first"}
        # The real server lists rounds "not yet submitted".
        open_tasks = [t for t in self.state.tasks.values() if t["status"] != "submitted"]
        return 200, {"node_id": node_id, "tasks": open_tasks}

    def _ack(self, node_id: str, round_id: str) -> tuple[int, Any]:
        task = self.state.tasks.get(round_id)
        if task is None:
            return 404, {"detail": "round not found"}
        # Idempotent, per the server's docstring.
        task["status"] = "accepted"
        task["accepted_at"] = time.time()
        return 200, task

    def _submit(self, node_id: str, round_id: str) -> tuple[int, Any]:
        payload = self._body() or {}
        task = self.state.tasks.get(round_id)
        if task is None:
            return 404, {"detail": "round not found"}
        if task.get("closed"):
            return 409, {"detail": "round is closed"}
        if payload.get("node_id") != node_id or payload.get("round_id") != round_id:
            return 400, {"detail": "node_id/round_id in path and body differ"}

        extra = set(payload) - {"node_id", "round_id", "scores", "agg_stats"}
        if extra:
            return 422, {
                "detail": [
                    {"loc": [k], "msg": "Extra inputs are not permitted", "type": "extra_forbidden"}
                    for k in sorted(extra)
                ]
            }

        agg = payload.get("agg_stats") or {}
        problem = _check_agg_stats(agg, payload.get("scores") or [])
        if problem:
            return 422, {"detail": [problem]}

        missing = [m for m in task.get("metrics", []) if m not in agg]
        if missing:
            return 422, {
                "detail": [
                    {
                        "loc": ["body", "agg_stats"],
                        "msg": f"missing requested metrics: {sorted(missing)!r}",
                        "type": "missing_metrics",
                    }
                ]
            }

        # A resubmission into an open round replaces the previous one.
        revision = self.state.revisions.get(round_id, 0) + 1
        self.state.revisions[round_id] = revision
        task["status"] = "submitted"
        self.state.submissions[round_id] = payload
        return 201, {
            "accepted": True,
            "round_id": round_id,
            "node_id": node_id,
            "revision": revision,
            "n_scores": len(payload.get("scores") or []),
            "trust": "pending" if "ho_n" in agg else "unknown",
        }

    # -- io ---------------------------------------------------------------

    def _body(self) -> dict | None:
        return self._cached_body

    def _read_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return None

    def _send(self, status: int, body: Any) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture
def fake_server():
    state = FakeState()
    handler = type("BoundHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
