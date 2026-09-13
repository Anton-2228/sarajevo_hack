"""What the agent remembers between runs.

The important piece is the payload stash. The finished submit body is written
to disk *before* the POST and removed only once delivery is confirmed, so a
crash in that window leaves a complete payload that the next start can simply
send. Training is the only expensive step in a round; never having to repeat it
is worth one file write.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IDENTITY_FILE = "identity.json"
JOURNAL_FILE = "journal.json"
MODELS_DIR = "models"
PAYLOAD_DIR = "outbox"
MANIFEST_FILE = "manifest.json"

# Phases a round moves through. `ready` is the one that matters on resume.
PHASE_ACKED = "acked"
PHASE_TRAINING = "training"
PHASE_SCORING = "scoring"
PHASE_READY = "ready"
PHASE_SUBMITTED = "submitted"
PHASE_FAILED = "failed"

MAX_TASK_ATTEMPTS = 3

# Windows refuses these as file names, with or without an extension, in any case.
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(raw: str) -> str:
    """Turn a server-supplied id into a directory name every OS will accept.

    `round_id` and `model.id` come off the wire and become paths. A round
    called `con`, or `a:b`, or one ending in a dot, is an unhandled exception at
    mkdir time on Windows -- and a round called `../..` is worse than that.
    """
    cleaned = _UNSAFE.sub(lambda m: f"%{ord(m.group()):02x}", raw)
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        return "_empty"
    if cleaned.split(".")[0].lower() in _WINDOWS_RESERVED:
        cleaned = "_" + cleaned
    return cleaned[:120]


def atomic_write_text(path: Path, text: str) -> None:
    """Write so that a crash leaves either the old file or the new one.

    The temp file lives in the destination directory because a rename across
    filesystems is not atomic -- and on Windows, is not even permitted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp{os.getpid()}")

    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

    # os.replace, never os.rename: rename onto an existing file fails on Windows.
    os.replace(temp, path)


@dataclass
class Identity:
    """Who this node is to the control plane.

    A `node_id` and nothing else. Contract 0.5.0 has no per-node secret and no
    request signing (CONTRACT.md 2.1), so there is no credential to protect
    here -- and `GET /nodes` publishes every node_id anyway.
    """

    node_id: str
    server_url: str
    name: str
    heartbeat_interval_s: float = 15.0
    registered_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Identity:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class TaskRecord:
    round_id: str
    phase: str
    attempts: int = 0
    updated_at: float = field(default_factory=time.time)
    model_key: str | None = None
    dataset_id: str | None = None
    last_error: str | None = None
    # Contract 0.9.0: which lifecycle the round was told to run, and the
    # checkpoint it produced. Both are here so a restart can tell "this round
    # already trained" from "this round never started".
    train_policy: str | None = None
    checkpoint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskRecord:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


class AgentState:
    """The state directory, as an object."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- identity ---------------------------------------------------------

    def load_identity(self, server_url: str | None = None) -> Identity | None:
        path = self.root / IDENTITY_FILE
        if not path.is_file():
            return None
        try:
            identity = Identity.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None
        # A different control plane means a different enrolment: reusing a
        # node_id across servers just produces confusing 404s.
        if server_url is not None and identity.server_url != server_url:
            return None
        return identity

    def save_identity(self, identity: Identity) -> None:
        atomic_write_text(
            self.root / IDENTITY_FILE,
            json.dumps(identity.to_dict(), indent=2, sort_keys=True),
        )

    def clear_identity(self) -> None:
        (self.root / IDENTITY_FILE).unlink(missing_ok=True)

    # -- checkpoints ------------------------------------------------------
    #
    # Contract 0.9.0 made these first-class: `operation.input_checkpoint_id` and
    # `output_checkpoint_id` name weights the node keeps to itself. The control
    # plane never receives them -- it only passes the names, and the namespace is
    # (node_id, checkpoint_id), so this directory is the whole of that namespace.

    def model_dir(self, key: str) -> Path:
        return self.root / MODELS_DIR / safe_name(key)

    # Checkpoints and models share one directory: a checkpoint *is* a saved
    # model, and giving them separate trees would only invite the two to
    # disagree about which holds the weights a round was told to load.
    checkpoint_dir = model_dir

    def has_checkpoint(self, checkpoint_id: str | None) -> bool:
        if not checkpoint_id:
            return False
        return self.checkpoint_dir(checkpoint_id).is_dir()

    def checkpoints(self) -> list[str]:
        root = self.root / MODELS_DIR
        return sorted(p.name for p in root.glob("*") if p.is_dir()) if root.is_dir() else []

    def write_manifest(self, checkpoint_id: str, manifest: dict[str, Any]) -> Path:
        """Record what a checkpoint actually is, beside the weights.

        The METRICS_GUIDE asks for at least the round, the model recipe, the
        chunk ids it was trained on and how far the round got. The point is
        restart: a familiar `round_id` coming back should be recovered from this
        rather than retrained, and certainly rather than submitted twice.
        """
        path = self.checkpoint_dir(checkpoint_id) / MANIFEST_FILE
        atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True))
        return path

    def read_manifest(self, checkpoint_id: str | None) -> dict[str, Any] | None:
        if not checkpoint_id:
            return None
        path = self.checkpoint_dir(checkpoint_id) / MANIFEST_FILE
        if not path.is_file():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return manifest if isinstance(manifest, dict) else None

    # -- journal ----------------------------------------------------------

    def journal(self) -> dict[str, TaskRecord]:
        path = self.root / JOURNAL_FILE
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        records = {}
        for round_id, payload in (raw or {}).items():
            try:
                records[round_id] = TaskRecord.from_dict(payload)
            except TypeError:
                continue
        return records

    def record(self, rec: TaskRecord) -> None:
        rec.updated_at = time.time()
        journal = self.journal()
        journal[rec.round_id] = rec
        atomic_write_text(
            self.root / JOURNAL_FILE,
            json.dumps({k: v.to_dict() for k, v in journal.items()}, indent=2, sort_keys=True),
        )

    def get_record(self, round_id: str) -> TaskRecord | None:
        return self.journal().get(round_id)

    def forget(self, round_id: str) -> None:
        journal = self.journal()
        if journal.pop(round_id, None) is not None:
            atomic_write_text(
                self.root / JOURNAL_FILE,
                json.dumps(
                    {k: v.to_dict() for k, v in journal.items()}, indent=2, sort_keys=True
                ),
            )

    # -- outbox -----------------------------------------------------------

    def payload_path(self, round_id: str) -> Path:
        return self.root / PAYLOAD_DIR / f"{safe_name(round_id)}.json"

    def stash_payload(self, round_id: str, body: dict[str, Any]) -> Path:
        path = self.payload_path(round_id)
        atomic_write_text(path, json.dumps(body, allow_nan=False))
        return path

    def load_payload(self, round_id: str) -> dict[str, Any] | None:
        path = self.payload_path(round_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def drop_payload(self, round_id: str) -> None:
        self.payload_path(round_id).unlink(missing_ok=True)

    # -- housekeeping -----------------------------------------------------

    def prune(self, keep_days: float = 14.0) -> int:
        """Forget rounds that finished long enough ago to be uninteresting."""
        cutoff = time.time() - keep_days * 86400
        journal = self.journal()
        stale = [
            round_id
            for round_id, rec in journal.items()
            if rec.phase == PHASE_SUBMITTED and rec.updated_at < cutoff
        ]
        for round_id in stale:
            journal.pop(round_id)
            self.drop_payload(round_id)
        if stale:
            atomic_write_text(
                self.root / JOURNAL_FILE,
                json.dumps(
                    {k: v.to_dict() for k, v in journal.items()}, indent=2, sort_keys=True
                ),
            )
        return len(stale)

    def describe(self) -> dict[str, Any]:
        """For `node-agent info`. Secret omitted, deliberately."""
        identity = self.load_identity()
        return {
            "state_dir": str(self.root),
            "exists": self.root.is_dir(),
            "node_id": identity.node_id if identity else None,
            "models": sorted(p.name for p in (self.root / MODELS_DIR).glob("*"))
            if (self.root / MODELS_DIR).is_dir()
            else [],
            "journal_entries": len(self.journal()),
            "pending_payloads": sorted(
                p.stem for p in (self.root / PAYLOAD_DIR).glob("*.json")
            )
            if (self.root / PAYLOAD_DIR).is_dir()
            else [],
        }
