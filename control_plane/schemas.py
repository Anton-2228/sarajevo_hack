"""Wire contracts between nodes and the control plane.

Four contracts, each owned by one side:
- handshake  node -> CP   who the node is: hardware, software, model kinds it can bake;
- heartbeat  node -> CP   node is alive, idle/busy, numeric load;
- task       CP -> node   GET /tasks/{node_id}: round, model to bake, dataset, metrics wanted;
- submit     node -> CP   Egress Gate: chunk ids, numeric scores, numeric metrics.

No auth in this build (hackathon speed): no enrollment token, no per-node secret, no request
signing — any caller can act as any node_id. Fine on a trusted local/demo network; do not expose
this control plane on an untrusted one without adding auth back.

The Egress Gate contract is deliberately narrow: a node may send chunk identifiers (hashes),
numeric scores and numeric aggregate statistics — nothing else. Unknown fields are
rejected, string values are only allowed where they are identifiers, and identifiers
are constrained to a charset with no whitespace, so free text cannot ride along.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
# Short human-readable labels in node specs (GPU model, versions). No newlines.
LABEL_RE = re.compile(r"^[A-Za-z0-9_.:\-+/() ]{1,64}$")
# chunk_id must look like a content hash (hex, 16..128 chars), not like text.
CHUNK_ID_RE = re.compile(r"^[a-f0-9]{16,128}$")

MAX_SCORES = 200_000
MAX_AGG_KEYS = 64
NODE_NAME_MAX = 48

# Held-out curve (see pooling.py): top-quantiles at which a node reports, as counts, how its proxy ranks
# its oracle-labelled held-out split. node_sdk/client.py keeps a copy of this tuple.
HELDOUT_QUANTILES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def q_key(q: float) -> str:
    return f"{round(q * 100):02d}"


HELDOUT_KEYS = ("ho_n", "ho_good") + tuple(
    f"ho_{what}_q{q_key(q)}" for q in HELDOUT_QUANTILES for what in ("n", "good"))

ModelKind = Literal["lora", "classifier"]


def _ident(v: str) -> str:
    if not ID_RE.match(v):
        raise ValueError("must match [A-Za-z0-9_.:-]{1,64}")
    return v


def _label(v: Optional[str]) -> Optional[str]:
    if v is not None and not LABEL_RE.match(v):
        raise ValueError("must be a short label [A-Za-z0-9_.:-+/() ]{1,64}")
    return v


def _hash(v: str) -> str:
    if not CHUNK_ID_RE.match(v):
        raise ValueError("chunk_id must be a lowercase hex hash (16-128 chars)")
    return v


def _numeric_map(values: Dict[str, Any], what: str, max_keys: int) -> Dict[str, Any]:
    if len(values) > max_keys:
        raise ValueError(f"{what} has more than {max_keys} keys")
    for key, value in values.items():
        if not ID_RE.match(key):
            raise ValueError(f"{what} key {key!r} is not a valid identifier")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{what}[{key!r}] must be a number (no strings/objects)")
        if not math.isfinite(value):
            raise ValueError(f"{what}[{key!r}] must be finite")
    return values


# --- handshake / heartbeat (node -> CP) ---------------------------------------------
class GPU(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    vram_gb: float = Field(ge=0)

    _model_label = field_validator("model")(_label)


class Hardware(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_cores: int = Field(ge=1)
    ram_gb: float = Field(gt=0)
    gpus: List[GPU] = Field(default_factory=list, max_length=64)
    disk_free_gb: Optional[float] = Field(default=None, ge=0)


class Handshake(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "name": "bank-a",
        "hardware": {"cpu_cores": 16, "ram_gb": 64, "gpus": [{"model": "A100", "vram_gb": 80}],
                     "disk_free_gb": 500},
        "software": {"agent_version": "0.2.0", "python": "3.12"},
        "model_kinds": ["classifier", "lora"],
    }]})

    name: str = Field(description="Human label, e.g. bank-a; node_id is derived from it")
    node_id: Optional[str] = Field(
        default=None,
        description="Set only to re-register an already enrolled node (keeps id and secret)")
    hardware: Hardware
    software: Dict[str, str] = Field(default_factory=dict)
    model_kinds: List[ModelKind] = Field(min_length=1, description="What this node can bake")

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        _ident(v)
        if len(v) > NODE_NAME_MAX:
            raise ValueError(f"name is longer than {NODE_NAME_MAX} chars")
        return v

    _node_ident = field_validator("node_id")(lambda v: v if v is None else _ident(v))

    @field_validator("software")
    @classmethod
    def _software(cls, v: Dict[str, str]) -> Dict[str, str]:
        if len(v) > 32:
            raise ValueError("software has more than 32 keys")
        for key, value in v.items():
            _ident(key)
            _label(value)
        return v


class Heartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "status": "busy",
        "stage": "training",
        "round_id": "campaign-r1",
        "load": {
            "progress_pct": 42.5, "docs_processed": 8500, "docs_total": 20000,
            "docs_per_sec": 127.4, "eta_s": 90, "train_loss": 0.31,
            "cpu_pct": 73.5, "ram_pct": 61.0, "gpu_util_pct": 92.0, "gpu_mem_pct": 78.0,
        },
    }]})

    status: Literal["idle", "busy"] = "idle"
    stage: Optional[Literal["downloading", "training", "scoring", "uploading"]] = Field(
        default=None,
        description="Current task phase, if busy",
    )
    round_id: Optional[str] = Field(default=None, description="Round being worked on, if busy")
    load: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Up to 32 numeric telemetry gauges. Recommended keys: progress_pct, docs_processed, "
            "docs_total, docs_per_sec, eta_s, train_loss, cpu_pct, ram_pct, gpu_util_pct, gpu_mem_pct"
        ),
    )

    _round_ident = field_validator("round_id")(lambda v: v if v is None else _ident(v))
    _load_numeric = field_validator("load")(lambda v: _numeric_map(v, "load", 32))


# --- rounds / tasks (CP -> node) -----------------------------------------------------
class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ModelKind
    id: str = Field(description="Model recipe/family id; concrete weights are named by operation checkpoints")

    _id_ident = field_validator("id")(_ident)


class TaskOperation(BaseModel):
    """An explicit model lifecycle instruction carried by every task."""

    model_config = ConfigDict(extra="forbid")

    train: Literal["fresh", "continue", "skip"] = "fresh"
    score: Literal[True] = Field(default=True, description="Tasks always produce scores in this contract")
    input_checkpoint_id: Optional[str] = Field(
        default=None, description="Node-local checkpoint to load for continue/skip")
    output_checkpoint_id: Optional[str] = Field(
        default=None, description="Node-local checkpoint name to save after training")

    _checkpoint_idents = field_validator("input_checkpoint_id", "output_checkpoint_id")(
        lambda v: v if v is None else _ident(v))

    @model_validator(mode="after")
    def _checkpoint_contract(self) -> "TaskOperation":
        if self.train == "fresh" and self.input_checkpoint_id is not None:
            raise ValueError("fresh training cannot have input_checkpoint_id")
        if self.train in ("continue", "skip") and self.input_checkpoint_id is None:
            raise ValueError(f"{self.train} requires input_checkpoint_id")
        return self


class Participant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    dataset_id: str = Field(description="Server-held shard or workload assigned to this node")
    params: Dict[str, Any] = Field(default_factory=dict,
                                   description="Numeric only; overrides round params for this node, e.g. cutoff_q")

    _idents = field_validator("node_id", "dataset_id")(_ident)
    _params_numeric = field_validator("params")(lambda v: _numeric_map(v, "participant params", MAX_AGG_KEYS))


class CreateRound(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "round_id": "r1",
        "budget_k": 1000,
        "model": {"kind": "classifier", "id": "quality-clf-v1"},
        "operation": {"train": "fresh", "score": True, "output_checkpoint_id": "r1"},
        "metrics": ["eval_spearman", "n_dedup_dropped"],
        "params": {"proxy_lr": 1e-5},
        "participants": [{"node_id": "worker-a-3f9c1d", "dataset_id": "wave1"}],
    }]})

    round_id: str
    budget_k: int = Field(default=1000, ge=1, description="global top-k to select")
    note: Optional[str] = Field(default=None, max_length=280)
    model: ModelRef
    operation: Optional[TaskOperation] = Field(
        default=None,
        description="Model lifecycle; defaults to fresh train+score with output_checkpoint_id=round_id",
    )
    metrics: List[str] = Field(min_length=1, max_length=MAX_AGG_KEYS,
                               description="agg_stats keys every participant must report")
    params: Dict[str, Any] = Field(default_factory=dict, description="Numeric only, e.g. proxy_lr")
    participants: List[Participant] = Field(min_length=1)

    _round_ident = field_validator("round_id")(_ident)
    _params_numeric = field_validator("params")(lambda v: _numeric_map(v, "params", MAX_AGG_KEYS))

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, v: List[str]) -> List[str]:
        for m in v:
            _ident(m)
        if len(v) != len(set(v)):
            raise ValueError("duplicate metric")
        return v

    @model_validator(mode="after")
    def _unique_participants(self) -> "CreateRound":
        ids = [p.node_id for p in self.participants]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate node_id in participants")
        return self


# --- responses to the node -------------------------------------------------------------
class HandshakeResponse(BaseModel):
    node_id: str
    heartbeat_interval_s: float
    tasks_url: str


class HeartbeatResponse(BaseModel):
    ok: bool
    next_heartbeat_s: float
    pending_tasks: int = Field(description="Open tasks not yet submitted; poll /tasks when > 0")


class TaskView(BaseModel):
    round_id: str
    status: Literal["assigned", "accepted", "submitted"]
    model: Optional[ModelRef] = Field(description="Model recipe/family to execute")
    operation: TaskOperation = Field(description="Whether to train fresh, continue, or only score")
    dataset_id: str = Field(description="Server-held shard or workload assigned by the control plane")
    metrics: List[str] = Field(description="agg_stats keys the submit must contain")
    params: Dict[str, Any] = Field(description="Round parameters merged with per-node task routing")
    budget_k: int
    assigned_at: float
    accepted_at: Optional[float]
    ack_url: str
    submit_url: str


class TasksResponse(BaseModel):
    node_id: str
    tasks: List[TaskView]


TASK_EXAMPLE = {
    "round_id": "r1",
    "status": "assigned",
    "model": {"kind": "classifier", "id": "quality-clf-v1"},
    "operation": {"train": "fresh", "score": True, "input_checkpoint_id": None,
                  "output_checkpoint_id": "r1"},
    "dataset_id": "wave1",
    "metrics": ["eval_spearman", "n_dedup_dropped"],
    "params": {"proxy_lr": 1e-5},
    "budget_k": 1000,
    "assigned_at": 1789245600.12,
    "accepted_at": None,
    "ack_url": "/tasks/bank-a-3f9c1d/r1/ack",
    "submit_url": "/tasks/bank-a-3f9c1d/r1/submit",
}
TASKS_EXAMPLE = {"node_id": "bank-a-3f9c1d", "tasks": [TASK_EXAMPLE]}
HANDSHAKE_RESPONSE_EXAMPLE = {
    "node_id": "bank-a-3f9c1d", "heartbeat_interval_s": 3.0, "tasks_url": "/tasks/bank-a-3f9c1d",
}
HEARTBEAT_RESPONSE_EXAMPLE = {"ok": True, "next_heartbeat_s": 3.0, "pending_tasks": 1}


def example_response(example: dict, status: int = 200) -> dict:
    return {status: {"content": {"application/json": {"example": example}}}}


# --- submit (node -> CP, Egress Gate) ----------------------------------------------
class ChunkScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(description="Hex content hash of the curated chunk")
    score: float

    _chunk_id_hash = field_validator("chunk_id")(_hash)

    @field_validator("score")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("score must be finite")
        return v


def _check_heldout_curve(stats: Dict[str, Any]) -> None:
    present = [k for k in HELDOUT_KEYS if k in stats]
    if not present:
        return
    if len(present) != len(HELDOUT_KEYS):
        raise ValueError(f"held-out curve needs all of {list(HELDOUT_KEYS)}")
    if any(stats[k] < 0 or stats[k] != int(stats[k]) for k in HELDOUT_KEYS):
        raise ValueError("held-out curve values must be non-negative integer counts")
    n, good = stats["ho_n"], stats["ho_good"]
    if n < 1 or good > n:
        raise ValueError("held-out curve needs ho_n >= 1 and ho_good <= ho_n")
    prev_n = prev_good = 0
    for q in HELDOUT_QUANTILES:
        top_n, top_good = stats[f"ho_n_q{q_key(q)}"], stats[f"ho_good_q{q_key(q)}"]
        if not (prev_n <= top_n <= n and prev_good <= top_good <= good and top_good <= top_n):
            raise ValueError("held-out curve counts must be cumulative over quantiles and within totals")
        prev_n, prev_good = top_n, top_good


class AggStats(BaseModel):
    """Numeric-only aggregates. The held-out curve (HELDOUT_KEYS) feeds calibration and the Reliability
    Gate; `eval_spearman` is informational."""

    model_config = ConfigDict(extra="allow")

    n_chunks: int = Field(ge=0)
    eval_spearman: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0,
        description="Spearman rho between proxy ranking and held-out eval on this node",
    )
    proxy_lr: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _extras_numeric(self) -> "AggStats":
        extra = self.model_extra or {}
        _numeric_map(extra, "agg_stats", MAX_AGG_KEYS - 3)
        _check_heldout_curve(extra)
        return self


class SubmitPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    round_id: str
    scores: List[ChunkScore] = Field(min_length=1, max_length=MAX_SCORES)
    agg_stats: AggStats

    _idents = field_validator("node_id", "round_id")(_ident)

    @model_validator(mode="after")
    def _unique_chunks(self) -> "SubmitPayload":
        ids = [s.chunk_id for s in self.scores]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk_id in scores")
        if self.agg_stats.n_chunks < len(ids):
            raise ValueError("agg_stats.n_chunks is smaller than len(scores)")
        return self


SUBMIT_EXAMPLE = {
    "node_id": "bank-a-3f9c1d",
    "round_id": "r1",
    "scores": [
        {"chunk_id": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "score": 0.91},
        {"chunk_id": "60303ae22b998861bce3b28f33eec1be758a213c86c93c076dbe9f558c11c752", "score": 0.12},
    ],
    "agg_stats": {"n_chunks": 2, "eval_spearman": 0.96, "proxy_lr": 1e-5, "n_dedup_dropped": 0},
}


def submit_openapi_body() -> dict:
    """OpenAPI requestBody for /submit: the handler reads raw bytes, so FastAPI can't infer it."""
    schema = SubmitPayload.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            ref = node.get("$ref", "")
            if ref.startswith("#/$defs/"):
                return inline(defs[ref.rsplit("/", 1)[-1]])
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return {"required": True,
            "content": {"application/json": {"schema": inline(schema), "example": SUBMIT_EXAMPLE}}}


# --- server-master: chunk ingestion and campaigns (see campaigns.py) ----------------------------
class ChunkIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str = Field(min_length=1, max_length=20_000)

    _chunk_id_hash = field_validator("chunk_id")(_hash)


class IngestChunks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: List[ChunkIn] = Field(min_length=1, max_length=5_000)

    @model_validator(mode="after")
    def _unique(self) -> "IngestChunks":
        ids = [c.chunk_id for c in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk_id in chunks")
        return self


class ImportParquet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_token: str = Field(pattern=r"^[a-f0-9]{32}$")
    text_column: str = Field(min_length=1, max_length=256)
    id_column: Optional[str] = Field(default=None, min_length=1, max_length=256)
    label_column: Optional[str] = Field(default=None, min_length=1, max_length=256)
    label_threshold: Optional[float] = Field(
        default=None,
        description="When set, imported label is 1 for values >= threshold and 0 otherwise; "
                    "without it the label column must contain only 0/1",
    )

    @model_validator(mode="after")
    def _mapping(self) -> "ImportParquet":
        selected = [self.text_column, self.id_column, self.label_column]
        selected = [value for value in selected if value is not None]
        if len(selected) != len(set(selected)):
            raise ValueError("text, id and label columns must be different")
        if self.label_threshold is not None and self.label_column is None:
            raise ValueError("label_threshold requires label_column")
        return self


class CreateCampaign(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "campaign_id": "c1", "shard_id": "wave1", "mode": "sharded",
        "model": {"kind": "classifier", "id": "quality-clf-v1"},
        "metrics": [], "node_ids": ["node-a-1a2b3c", "node-b-4d5e6f"], "schedule": [500, 1000, 1500],
        "train_mode": "fresh", "strategy": "cutoff", "k_frac": 0.1,
    }]})

    campaign_id: str
    shard_id: str
    mode: Literal["sharded", "experts"] = "sharded"
    model: ModelRef
    metrics: List[str] = Field(default_factory=list, max_length=MAX_AGG_KEYS)
    node_ids: List[str] = Field(min_length=1, max_length=64,
                                description="node_ids[i] owns partition/domain i")
    schedule: List[int] = Field(min_length=1, max_length=20,
                                description="cumulative oracle labels per partition, strictly increasing")
    train_mode: Literal["fresh", "continue"] = Field(
        default="fresh",
        description="Retrain from the recipe each round or continue from the preceding round checkpoint",
    )
    strategy: Literal["random", "cutoff", "qbc"] = "cutoff"
    k_frac: float = Field(default=0.1, gt=0, le=1, description="target top-k as a share of the shard")
    good_min: Optional[int] = Field(default=None, description="label >= this counts as good at finalize")
    seed: int = 0

    _idents = field_validator("campaign_id", "shard_id")(_ident)

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, v: List[str]) -> List[str]:
        for m in v:
            _ident(m)
        if len(v) != len(set(v)):
            raise ValueError("duplicate metric")
        return v

    @field_validator("node_ids")
    @classmethod
    def _node_idents(cls, v: List[str]) -> List[str]:
        for n in v:
            _ident(n)
        if len(v) != len(set(v)):
            raise ValueError("duplicate node_id")
        return v

    @field_validator("schedule")
    @classmethod
    def _schedule(cls, v: List[int]) -> List[int]:
        if any(b <= 0 for b in v) or list(v) != sorted(set(v)):
            raise ValueError("schedule must be a positive, strictly increasing list of cumulative label counts")
        return v
