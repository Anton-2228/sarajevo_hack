"""Everything the agent needs to know before it starts, in one frozen object.

Settings resolve flag > NODE_AGENT_* environment variable > default. The env
layer exists so that a GUI, a container or a systemd unit can configure the
agent without assembling an argv, and so that a secret-ish value like the
server URL can come from the environment on a shared machine.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

# A cloudflared quick tunnel: ephemeral by nature. Override with --server or
# NODE_AGENT_SERVER when it moves, which it does.
DEFAULT_SERVER = "https://emily-coordinator-billy-randy.trycloudflare.com"
ENV_PREFIX = "NODE_AGENT_"

SCORE_SCALES = ("raw", "unit")
# `auto` first: it is the default, and since contract 0.9.0 the only one that
# reflects what the control plane actually asked for.
MODES = ("auto", "train", "score")
UNKNOWN_METRIC_POLICIES = ("omit", "zero", "null", "fail")
LOG_LEVELS = ("debug", "info", "warning", "error")

# Below this a training run has no room to do anything useful, and the failure
# would look like a mysterious crash rather than a budget that was never viable.
MIN_RAM_GB = 0.5


@dataclass(frozen=True)
class AgentConfig:
    server_url: str = DEFAULT_SERVER
    name: str = ""

    cores: int = 1
    ram_gb: float = 2.0
    enforce_limits: bool = True

    state_dir: Path = field(default_factory=lambda: default_state_dir())

    # There is deliberately no dataset setting. Contract 0.7.0 made every
    # corpus server-held, so there is nothing local to point at and nothing to
    # substitute: a round names a shard and the node reads it from /shards.
    #
    # `auto` means "do what task.operation says", which since contract 0.9.0 is
    # an explicit instruction rather than something to infer. train/score remain
    # as manual overrides for debugging a single node.
    mode: str = "auto"
    model_kinds: list[str] = field(default_factory=lambda: ["classifier"])
    score_scale: str = "raw"
    unknown_metric: str = "omit"

    poll_interval_s: float = 10.0
    heartbeat_interval_s: float | None = None
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 60.0
    max_retries: int = 4

    once: bool = False
    max_tasks: int | None = None
    dry_run: bool = False
    shutdown_grace_s: float = 120.0
    reset_identity: bool = False

    log_level: str = "info"
    log_file: Path | None = None

    def validate(self) -> list[str]:
        """Return every problem at once, so one run fixes them all."""
        problems: list[str] = []
        if not self.server_url.startswith(("http://", "https://")):
            problems.append(f"--server must be an http(s) URL, got {self.server_url!r}")
        if not self.name:
            problems.append("--name must not be empty")
        else:
            from node.agent.models import MAX_NAME, is_id

            if len(self.name) > MAX_NAME or not is_id(self.name):
                problems.append(
                    f"--name must be at most {MAX_NAME} characters of "
                    f"[A-Za-z0-9_.:-], got {self.name!r}"
                )
        # 0 means "work it out from the machine", which `resources.resolve_budget`
        # does. Only a value the operator actually typed can be wrong.
        if self.cores < 0:
            problems.append(f"--cores cannot be negative, got {self.cores}")
        if self.ram_gb < 0:
            problems.append(f"--ram cannot be negative, got {self.ram_gb}")
        elif 0 < self.ram_gb < MIN_RAM_GB:
            problems.append(f"--ram must be at least {MIN_RAM_GB} GiB, got {self.ram_gb}")
        if self.mode not in MODES:
            problems.append(f"--mode must be one of {', '.join(MODES)}")
        if self.score_scale not in SCORE_SCALES:
            problems.append(f"--score-scale must be one of {', '.join(SCORE_SCALES)}")
        if self.unknown_metric not in UNKNOWN_METRIC_POLICIES:
            problems.append(
                f"--unknown-metric must be one of {', '.join(UNKNOWN_METRIC_POLICIES)}"
            )
        if self.poll_interval_s <= 0:
            problems.append("--poll-interval must be positive")
        if self.max_tasks is not None and self.max_tasks < 1:
            problems.append("--max-tasks must be at least 1")
        if not self.model_kinds:
            problems.append("--model-kinds must name at least one kind")
        return problems


def default_state_dir() -> Path:
    """Where identity, models and the task journal live, per OS convention."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_STATE_HOME")
        root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "node-agent"


def default_name() -> str:
    """A handshake name the server will accept.

    CONTRACT.md 3.1: the ID alphabet, at most 48 characters. Hostnames respect
    neither, and a 422 on the very first call is a poor introduction.
    """
    from node.agent.models import MAX_NAME, as_id

    host = socket.gethostname().split(".")[0].strip().lower()
    return as_id(f"node-{host}", limit=MAX_NAME)


def from_args(ns: argparse.Namespace, environ: dict[str, str] | None = None) -> AgentConfig:
    """Build a config from parsed flags, falling back to env then defaults."""
    env = os.environ if environ is None else environ

    def pick(flag: str, env_name: str, default, cast=None):
        value = getattr(ns, flag, None)
        if value is None:
            raw = env.get(ENV_PREFIX + env_name)
            if raw is None or raw == "":
                return default
            value = raw
        return cast(value) if cast is not None else value

    state_dir = Path(pick("state_dir", "STATE_DIR", default_state_dir(), Path)).expanduser()

    model_kinds = pick("model_kinds", "MODEL_KINDS", "classifier")
    if isinstance(model_kinds, str):
        model_kinds = [k.strip() for k in model_kinds.split(",") if k.strip()]

    return AgentConfig(
        server_url=str(pick("server", "SERVER", DEFAULT_SERVER)).rstrip("/"),
        name=str(pick("name", "NAME", default_name())),
        cores=int(pick("cores", "CORES", 0, int) or 0),
        ram_gb=float(pick("ram", "RAM_GB", 0.0, float) or 0.0),
        enforce_limits=not getattr(ns, "no_enforce_limits", False),
        state_dir=state_dir,
        mode=str(pick("mode", "MODE", "auto")),
        model_kinds=list(model_kinds),
        score_scale=str(pick("score_scale", "SCORE_SCALE", "raw")),
        unknown_metric=str(pick("unknown_metric", "UNKNOWN_METRIC", "omit")),
        poll_interval_s=float(pick("poll_interval", "POLL_INTERVAL", 10.0, float)),
        heartbeat_interval_s=pick("heartbeat_interval", "HEARTBEAT_INTERVAL", None, float),
        connect_timeout_s=float(pick("connect_timeout", "CONNECT_TIMEOUT", 5.0, float)),
        read_timeout_s=float(pick("read_timeout", "READ_TIMEOUT", 60.0, float)),
        max_retries=int(pick("max_retries", "MAX_RETRIES", 4, int)),
        once=getattr(ns, "once", False),
        max_tasks=pick("max_tasks", "MAX_TASKS", None, int),
        dry_run=getattr(ns, "dry_run", False),
        shutdown_grace_s=float(pick("shutdown_grace", "SHUTDOWN_GRACE", 120.0, float)),
        reset_identity=getattr(ns, "reset_identity", False),
        log_level=str(pick("log_level", "LOG_LEVEL", "info")).lower(),
        log_file=Path(pick("log_file", "LOG_FILE", None, Path)).expanduser()
        if pick("log_file", "LOG_FILE", None)
        else None,
    )
