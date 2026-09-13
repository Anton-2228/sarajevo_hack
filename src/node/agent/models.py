"""The wire types of the control plane, as dataclasses.

Two rules run through the whole module.

*Unknown fields are kept, not dropped.* Every inbound type carries an `extra`
dict holding whatever the server sent that we do not model. The protocol is
still moving, and preserving what we do not understand is what makes the next
field arrive for free instead of as a rewrite.

*Outbound payloads omit absent optionals.* Most request schemas are
`additionalProperties: false`, so the safest body is the smallest one that
still says everything.

Contract 0.7.0 moved the corpus to the server. A node no longer declares the
datasets it holds -- the handshake schema forbids the key outright -- and a
task instead names a server-held shard plus, in a campaign, the routing that
says which slice of it is this node's. `Routing` is that half of the task.

Contract 0.9.0 added the other half: `TaskOperation`. The model lifecycle is now
an instruction, not an inference. Every task says outright whether to train
fresh, continue from a checkpoint, or skip training and only score -- so the
guessing this module used to do is gone, and the METRICS_GUIDE says so in as
many words: do not key off `round_id`, `model.id`, `n_labels`, or whether a file
happens to exist on disk.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# CONTRACT.md 2: the server rejects anything outside these, so it is cheaper to
# know locally than to learn from a 422 after a training run.
ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9_.:\-+/() ]{1,64}$")
HASH_RE = re.compile(r"^[a-f0-9]{16,128}$")

MAX_NAME = 48
MAX_AGG_STATS_KEYS = 64
MAX_SOFTWARE_KEYS = 32
MAX_LOAD_KEYS = 32


def is_id(value: str) -> bool:
    return bool(ID_RE.match(value))


def as_id(value: str, *, limit: int = 64) -> str:
    """Coerce a local string into something the server will accept as an ID.

    Hostnames and filenames are the two sources here, and neither is bound by
    the server's alphabet.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.:\-]", "-", value).strip("-")
    return cleaned[:limit] or "node"


def as_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:\-+/() ]", "-", value).strip()
    return cleaned[:64] or "-"


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


@dataclass(frozen=True)
class GPU:
    model: str
    vram_gb: float

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "vram_gb": self.vram_gb}


@dataclass(frozen=True)
class Hardware:
    cpu_cores: int
    ram_gb: float
    gpus: list[GPU] = field(default_factory=list)
    disk_free_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "cpu_cores": self.cpu_cores,
                "ram_gb": self.ram_gb,
                "gpus": [g.to_dict() for g in self.gpus],
                "disk_free_gb": self.disk_free_gb,
            }
        )


@dataclass(frozen=True)
class HandshakeRequest:
    """Compute capabilities, and nothing else.

    There is deliberately no `datasets` here. Contract 0.7.0 made the corpus
    server-held, and the handshake schema is `additionalProperties: false`, so
    the dataset list an earlier build sent is now a 422 that fails enrolment
    outright -- the node would never get as far as a round.
    """

    name: str
    hardware: Hardware
    model_kinds: list[str]
    # Set only to re-register an already enrolled node: the server then keeps
    # our id instead of minting a new one.
    node_id: str | None = None
    software: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "name": self.name,
                "node_id": self.node_id,
                "hardware": self.hardware.to_dict(),
                "software": self.software,
                "model_kinds": list(self.model_kinds),
            }
        )


@dataclass(frozen=True)
class HandshakeResponse:
    """What enrolment gives back.

    No `node_secret`, no `auth_mode`, and correspondingly no signature on
    submit: contract 0.5.0 removed authentication outright (CONTRACT.md 2.1).
    Knowing a `node_id` -- which `GET /nodes` publishes -- is all it takes to
    act as that node.
    """

    node_id: str
    heartbeat_interval_s: float
    tasks_url: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HandshakeResponse:
        return cls(
            node_id=str(payload["node_id"]),
            heartbeat_interval_s=float(payload["heartbeat_interval_s"]),
            tasks_url=str(payload["tasks_url"]),
        )


# METRICS_GUIDE: the phases the dashboard knows, and the only values `stage`
# accepts. `None` is what an idle node sends -- the field is for an active task.
STAGE_DOWNLOADING = "downloading"
STAGE_TRAINING = "training"
STAGE_SCORING = "scoring"
STAGE_UPLOADING = "uploading"
STAGES = (STAGE_DOWNLOADING, STAGE_TRAINING, STAGE_SCORING, STAGE_UPLOADING)

# The `load` gauges the built-in dashboard draws. Extra numeric keys are allowed
# and reach Prometheus, but only these get their own graph, so they are worth
# spelling exactly.
DASHBOARD_LOAD_KEYS = (
    "progress_pct",
    "docs_processed",
    "docs_total",
    "docs_per_sec",
    "eta_s",
    "train_loss",
    "cpu_pct",
    "ram_pct",
    "gpu_util_pct",
    "gpu_mem_pct",
)


@dataclass(frozen=True)
class HeartbeatRequest:
    """Liveness plus a snapshot of what the node is doing right now.

    `stage` arrived in contract 0.9.0 and is constrained to `STAGES`; an idle
    node omits it, along with `round_id` and any stale `load` -- the
    METRICS_GUIDE is explicit that finishing a task means sending `idle` with
    none of the previous task's numbers still attached.
    """

    status: str = "idle"
    round_id: str | None = None
    stage: str | None = None
    load: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "status": self.status,
                "round_id": self.round_id,
                "stage": self.stage,
                "load": self.load,
            }
        )


@dataclass(frozen=True)
class HeartbeatResponse:
    ok: bool
    next_heartbeat_s: float
    pending_tasks: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HeartbeatResponse:
        return cls(
            ok=bool(payload["ok"]),
            next_heartbeat_s=float(payload["next_heartbeat_s"]),
            pending_tasks=int(payload["pending_tasks"]),
        )


@dataclass(frozen=True)
class ModelRef:
    """The recipe to execute, not a set of weights. Weights are node-local."""

    kind: str
    id: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.id}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ModelRef | None:
        if not payload:
            return None
        return cls(kind=str(payload.get("kind", "")), id=str(payload.get("id", "")))


# CONTRACT 0.9.0 / METRICS_GUIDE: what `operation.train` may say.
TRAIN_FRESH = "fresh"
TRAIN_CONTINUE = "continue"
TRAIN_SKIP = "skip"
TRAIN_POLICIES = (TRAIN_FRESH, TRAIN_CONTINUE, TRAIN_SKIP)


@dataclass(frozen=True)
class TaskOperation:
    """The model lifecycle instruction every task carries.

    This is a command, not a hint, and it is the whole answer to "should I
    train?". The METRICS_GUIDE names the heuristics it replaces -- `round_id`,
    `model.id`, `n_labels`, the presence of a local file -- and says to use none
    of them.

    | `train`    | what the node does                                          |
    |------------|-------------------------------------------------------------|
    | `fresh`    | build from the `model` recipe, train, save to output         |
    | `continue` | load input checkpoint, train on this round's data, save out  |
    | `skip`     | load input checkpoint, run no trainer, score with it         |

    Checkpoints never leave the node: the control plane passes names, and the
    namespace is `(node_id, checkpoint_id)`.
    """

    train: str = TRAIN_FRESH
    score: bool = True
    input_checkpoint_id: str | None = None
    output_checkpoint_id: str | None = None
    # Set when the server named a policy this build does not know. Kept rather
    # than dropped so the node can say why it fell back to training.
    unknown_train: str | None = None

    @property
    def trains(self) -> bool:
        return self.train in (TRAIN_FRESH, TRAIN_CONTINUE)

    @property
    def continues(self) -> bool:
        return self.train == TRAIN_CONTINUE

    @property
    def needs_input_checkpoint(self) -> bool:
        """`continue` and `skip` both start from one."""
        return self.train in (TRAIN_CONTINUE, TRAIN_SKIP)

    def describe(self) -> str:
        parts = [f"train={self.train}"]
        if self.input_checkpoint_id:
            parts.append(f"from={self.input_checkpoint_id}")
        if self.output_checkpoint_id:
            parts.append(f"to={self.output_checkpoint_id}")
        return " ".join(parts)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> TaskOperation:
        # A task without `operation` is a pre-0.9.0 server, or a fake in a test.
        # Training fresh is the reading that always produces a usable model.
        if not isinstance(payload, dict):
            return cls()

        raw = payload.get("train", TRAIN_FRESH)
        train = str(raw) if raw is not None else TRAIN_FRESH
        unknown = None
        if train not in TRAIN_POLICIES:
            # Never guess a lifecycle: fall back to the one policy that needs
            # nothing to already exist, and keep the name so it can be reported.
            unknown, train = train, TRAIN_FRESH

        def checkpoint(key: str) -> str | None:
            value = payload.get(key)
            return str(value) if isinstance(value, str) and value else None

        return cls(
            train=train,
            score=bool(payload.get("score", True)),
            input_checkpoint_id=checkpoint("input_checkpoint_id"),
            output_checkpoint_id=checkpoint("output_checkpoint_id"),
            unknown_train=unknown,
        )


# CONTRACT.md 4.8. `sharded` splits one pool for throughput; `experts` gives
# every node the whole pool and differs only in which labels it may train on.
CAMPAIGN_MODES = ("sharded", "experts")

# Params whose value is a string by design, so quarantining them as
# "non-numeric" would be reporting the contract as a fault.
_STRING_PARAMS = frozenset({"mode"})


@dataclass(frozen=True)
class Routing:
    """Which slice of a server-held shard is this node's work.

    A plain operator round carries none of this and the node takes the whole
    shard. A campaign round carries all four keys inside `params`
    (CONTRACT.md 3.3.1), and the two modes differ in a way that matters:

    - `sharded`: train on, and score, only partition `i`. Scoring anything else
      would put another node's chunks into a ranking the server concatenates
      without deduplicating.
    - `experts`: train only on the labels of domain `i`, but score *every*
      chunk in the pool. `advance` checks that coverage and rejects a
      submission with ids missing or extra.
    """

    mode: str = ""
    partition: int | None = None
    n_partitions: int | None = None
    n_labels: int | None = None

    @property
    def is_campaign(self) -> bool:
        return self.mode in CAMPAIGN_MODES

    @property
    def scores_whole_pool(self) -> bool:
        """True when the submission must cover the full shard, not one slice."""
        return self.mode != "sharded"

    def describe(self) -> str:
        if not self.is_campaign:
            return "plain round (whole shard)"
        return (
            f"{self.mode} partition {self.partition}/{self.n_partitions}"
            + (f", {self.n_labels} labels" if self.n_labels is not None else "")
        )

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> tuple[Routing, list[str]]:
        """Parse routing out of `task.params`, reporting what does not add up.

        Problems are returned rather than raised: the caller decides, and for a
        campaign it has to be fatal. Guessing a partition would submit scores
        for documents this node was never assigned.
        """
        problems: list[str] = []

        raw_mode = params.get("mode")
        mode = ""
        if raw_mode is not None:
            if isinstance(raw_mode, str) and raw_mode in CAMPAIGN_MODES:
                mode = raw_mode
            else:
                problems.append(
                    f"task params name an unknown campaign mode {raw_mode!r}; "
                    f"expected one of {', '.join(CAMPAIGN_MODES)}"
                )

        def whole(key: str) -> int | None:
            if key not in params:
                return None
            value = params[key]
            if isinstance(value, bool):
                problems.append(f"params.{key} is a bool, not an integer")
                return None
            # A JSON number reaches us as float, and the server sends these as
            # integers; 2.0 is the same partition as 2, 2.5 is not a partition.
            try:
                number = float(value)
            except (TypeError, ValueError):
                problems.append(f"params.{key}={value!r} is not a number")
                return None
            if number != int(number):
                problems.append(f"params.{key}={value!r} is not a whole number")
                return None
            return int(number)

        partition = whole("partition")
        n_partitions = whole("n_partitions")
        n_labels = whole("n_labels")

        if mode and (partition is None or n_partitions is None):
            problems.append(
                f"campaign mode {mode!r} needs both partition and n_partitions in params"
            )
        if n_partitions is not None and n_partitions < 1:
            problems.append(f"params.n_partitions must be at least 1, got {n_partitions}")
        if partition is not None:
            if partition < 0:
                problems.append(f"params.partition must not be negative, got {partition}")
            elif n_partitions is not None and partition >= n_partitions:
                problems.append(
                    f"params.partition={partition} is out of range for "
                    f"n_partitions={n_partitions}"
                )

        return cls(
            mode=mode, partition=partition, n_partitions=n_partitions, n_labels=n_labels
        ), problems


# Fields we model explicitly; everything else the server sends lands in `extra`.
_TASK_KNOWN = {
    "round_id",
    "status",
    "model",
    "operation",
    "dataset_id",
    "metrics",
    "params",
    "budget_k",
    "assigned_at",
    "accepted_at",
    "ack_url",
    "submit_url",
}


@dataclass(frozen=True)
class TaskView:
    """One round this node should work on.

    `params` is the numeric view, because that is what may be echoed back into
    `agg_stats`, which takes numbers only. `raw_params` is what the server
    actually sent -- a campaign puts a string `mode` in there -- and `routing`
    is that parsed into the slice of the shard this node owns.
    """

    round_id: str
    status: str
    dataset_id: str
    metrics: list[str] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)
    budget_k: int = 0
    assigned_at: float = 0.0
    ack_url: str = ""
    submit_url: str = ""
    model: ModelRef | None = None
    operation: TaskOperation = field(default_factory=TaskOperation)
    accepted_at: float | None = None
    raw_params: dict[str, Any] = field(default_factory=dict)
    routing: Routing = field(default_factory=Routing)
    routing_problems: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def shard_id(self) -> str:
        """`dataset_id` names a server-held shard now, not anything local."""
        return self.dataset_id

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskView:
        extra = {k: v for k, v in payload.items() if k not in _TASK_KNOWN}
        raw_params = dict(payload.get("params") or {})

        # Params are declared `additionalProperties: true`, and a campaign uses
        # that for a string `mode`. Anything else that will not convert is
        # quarantined rather than crashed on -- it is a field we were never
        # going to read.
        params: dict[str, float] = {}
        rejected: dict[str, Any] = {}
        for key, value in raw_params.items():
            if key in _STRING_PARAMS:
                continue
            if isinstance(value, bool):
                rejected[key] = value
                continue
            try:
                params[key] = float(value)
            except (TypeError, ValueError):
                rejected[key] = value
        if rejected:
            extra["params_nonnumeric"] = rejected

        routing, routing_problems = Routing.from_params(raw_params)

        return cls(
            round_id=str(payload["round_id"]),
            status=str(payload.get("status", "assigned")),
            dataset_id=str(payload.get("dataset_id", "")),
            metrics=[str(m) for m in (payload.get("metrics") or [])],
            params=params,
            budget_k=int(payload.get("budget_k") or 0),
            assigned_at=float(payload.get("assigned_at") or 0.0),
            ack_url=str(payload.get("ack_url", "")),
            submit_url=str(payload.get("submit_url", "")),
            model=ModelRef.from_dict(payload.get("model")),
            operation=TaskOperation.from_dict(payload.get("operation")),
            accepted_at=(
                float(payload["accepted_at"])
                if payload.get("accepted_at") is not None
                else None
            ),
            raw_params=raw_params,
            routing=routing,
            routing_problems=routing_problems,
            extra=extra,
        )


@dataclass(frozen=True)
class TaskMode:
    """What to do about the model this round, resolved into one object."""

    train: bool
    source: str
    policy: str = TRAIN_FRESH
    input_checkpoint_id: str | None = None
    output_checkpoint_id: str | None = None

    @property
    def continues(self) -> bool:
        return self.policy == TRAIN_CONTINUE


def resolve_mode(task: TaskView, forced: str = "auto") -> TaskMode:
    """Decide what to do with the model, from the task's own instruction.

    Contract 0.9.0 settled this: `task.operation.train` is a command, so `auto`
    -- the default -- simply obeys it. The guesswork that used to live here is
    gone, and the METRICS_GUIDE is explicit that a node must not reconstruct it
    from `round_id`, `model.id`, `n_labels` or what is lying on disk.

    `--mode train` and `--mode score` remain as operator overrides for a node
    being debugged by hand. They override the policy but keep the checkpoint
    names, because those are how the server tracks lineage either way.
    """
    operation = task.operation

    if forced == "train":
        # Forced training is a fresh build: continuing would need the server's
        # checkpoint to be the one we think it is, which an override cannot know.
        return TaskMode(
            train=True,
            source="--mode train",
            policy=TRAIN_FRESH,
            output_checkpoint_id=operation.output_checkpoint_id,
        )
    if forced == "score":
        return TaskMode(
            train=False,
            source="--mode score",
            policy=TRAIN_SKIP,
            input_checkpoint_id=operation.input_checkpoint_id,
        )

    source = f"operation.train={operation.train}"
    if operation.unknown_train:
        source = f"operation.train={operation.unknown_train!r} unknown, training fresh"
    return TaskMode(
        train=operation.trains,
        source=source,
        policy=operation.train,
        input_checkpoint_id=operation.input_checkpoint_id,
        output_checkpoint_id=operation.output_checkpoint_id,
    )


@dataclass(frozen=True)
class ChunkScore:
    chunk_id: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "score": self.score}


# The server caps a submit at this many scores.
MAX_SCORES = 200_000


@dataclass(frozen=True)
class SubmitPayload:
    node_id: str
    round_id: str
    scores: list[ChunkScore]
    agg_stats: dict[str, Any]

    def body(self) -> dict[str, Any]:
        """The whole submit payload.

        Exactly these four keys: the schema is `additionalProperties: false`,
        so a leftover `signature` from the pre-0.5.0 contract is now a 422.
        """
        return {
            "node_id": self.node_id,
            "round_id": self.round_id,
            "scores": [s.to_dict() for s in self.scores],
            "agg_stats": dict(self.agg_stats),
        }


@dataclass
class TaskProgress:
    """Where a round has got to. For the console now, a GUI later."""

    round_id: str
    phase: str
    started_at: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "phase": self.phase,
            "started_at": self.started_at,
            "detail": dict(self.detail),
        }


@dataclass
class AgentView:
    """Everything a front end needs to render the agent, and nothing more.

    Deliberately plain data with no I/O: this is the seam a GUI attaches to, so
    it must be possible to hand one of these across a thread or a socket.
    """

    server_url: str
    status: str = "starting"  # starting | idle | busy | degraded | stopping
    node_id: str | None = None
    budget_cores: int = 0
    budget_ram_gb: float = 0.0
    current: TaskProgress | None = None
    completed: int = 0
    failed: int = 0
    last_error: str | None = None
    last_heartbeat_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_url": self.server_url,
            "status": self.status,
            "node_id": self.node_id,
            "budget_cores": self.budget_cores,
            "budget_ram_gb": self.budget_ram_gb,
            "current": self.current.to_dict() if self.current else None,
            "completed": self.completed,
            "failed": self.failed,
            "last_error": self.last_error,
            "last_heartbeat_ok": self.last_heartbeat_ok,
        }
