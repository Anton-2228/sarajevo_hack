"""What the window remembers between launches, and how it becomes an AgentConfig.

A JSON file next to `identity.json`, not QSettings: the agent already owns a
per-OS state directory, the file is inspectable by the same person debugging
the CLI, it survives being moved by a packager, and it is testable without Qt.
QSettings would have put half of this in the Windows registry, where nothing
else about the node lives.

Nothing here imports Qt, and nothing here touches the network.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from node.agent.config import (
    DEFAULT_SERVER,
    ENV_PREFIX,
    AgentConfig,
    default_name,
    default_state_dir,
)
from node.agent.resources import HardwareInfo
from node.agent.state import atomic_write_text
from node.gui.presets import MEDIUM, PRESET_LEVELS, preset_budget

LOG = logging.getLogger("node.gui")

SETTINGS_FILE = "gui.json"

# Bump when a field changes meaning rather than merely appearing. A file from an
# older version falls back to defaults instead of being half-read.
SETTINGS_VERSION = 1


@dataclass
class GuiSettings:
    """Everything the window persists. Deliberately small.

    The operator configures a share of their machine and nothing else; the rest
    of `AgentConfig` stays on its defaults, which are the ones the CLI uses too.
    """

    preset: str = MEDIUM
    # True means `cores`/`ram_gb` below win over the preset.
    advanced: bool = False
    cores: int = 0  # 0 = let resolve_budget decide
    ram_gb: float = 0.0  # 0 = let resolve_budget decide

    autostart: bool = False
    # The "it is still running in the tray" balloon: once, ever.
    tray_hint_shown: bool = False
    window_geometry_hex: str | None = None

    # Rounds handled by the previous run, shown while idle.
    last_run_completed: int = 0
    last_run_failed: int = 0
    last_run_finished_at: float | None = None


def settings_path(state_dir: Path | None = None) -> Path:
    return (state_dir or default_state_dir()) / SETTINGS_FILE


def load(path: Path | None = None) -> GuiSettings:
    """Read the settings. A missing, corrupt or older file yields defaults.

    Never raises: a node that refuses to open its window because a JSON file
    lost a brace is worse than one that opens with the defaults.
    """
    target = path or settings_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return GuiSettings()
    except (OSError, ValueError) as error:
        LOG.warning("could not read %s (%s); using defaults", target, error)
        return GuiSettings()

    if not isinstance(raw, dict) or raw.get("version") != SETTINGS_VERSION:
        return GuiSettings()

    known = {f.name: f for f in fields(GuiSettings)}
    values: dict[str, Any] = {}
    for name, value in raw.items():
        field_def = known.get(name)
        if field_def is None:  # "version", or a key from a future release
            continue
        try:
            values[name] = _coerce(field_def.type, value)
        except (TypeError, ValueError):
            LOG.warning("ignoring %s=%r in %s", name, value, target)

    settings = GuiSettings(**values)
    if settings.preset not in PRESET_LEVELS:
        settings.preset = MEDIUM
    if settings.cores < 0:
        settings.cores = 0
    if settings.ram_gb < 0:
        settings.ram_gb = 0.0
    return settings


def save(settings: GuiSettings, path: Path | None = None) -> None:
    """Write the settings atomically. Logs and swallows I/O failures."""
    target = path or settings_path()
    payload = {"version": SETTINGS_VERSION, **asdict(settings)}
    try:
        atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=True))
    except OSError as error:
        LOG.warning("could not write %s (%s)", target, error)


def resolved_budget(settings: GuiSettings, hw: HardwareInfo) -> tuple[int, float]:
    """The cores and RAM this configuration asks for. The one place the preset resolves."""
    if settings.advanced:
        return settings.cores, settings.ram_gb
    budget = preset_budget(settings.preset, hw)
    return budget.cores, budget.ram_gb


def server_url() -> str:
    """The control plane. Known up front, so it has no field in the window.

    `DEFAULT_SERVER` is an ephemeral tunnel that moves; the environment variable
    is the escape hatch, and it is the same one the CLI reads.
    """
    return (os.environ.get(ENV_PREFIX + "SERVER") or DEFAULT_SERVER).rstrip("/")


def node_name() -> str:
    return os.environ.get(ENV_PREFIX + "NAME") or default_name()


def build_config(
    settings: GuiSettings,
    hw: HardwareInfo,
    *,
    state_dir: Path | None = None,
    reset_identity: bool = False,
) -> AgentConfig:
    """Settings plus the machine -> the config the agent runs on.

    Pure: `AgentConfig.validate()` is safe to call on the result from a
    `textChanged` handler, because neither this nor validation touches disk.
    """
    cores, ram_gb = resolved_budget(settings, hw)
    return AgentConfig(
        server_url=server_url(),
        name=node_name(),
        cores=cores,
        ram_gb=ram_gb,
        state_dir=state_dir or default_state_dir(),
        # One-shot, never persisted: a `reset_identity` read back off disk would
        # mint a new node on every launch.
        reset_identity=reset_identity,
        log_level=os.environ.get(ENV_PREFIX + "LOG_LEVEL", "info").lower(),
    )


def problems_by_field(cfg: AgentConfig) -> tuple[dict[str, list[str]], list[str]]:
    """Split `AgentConfig.validate()` messages into per-widget and leftovers.

    Every message starts with the flag it belongs to (`--ram must be at least
    ...`), so the leading token is a reliable key. Anything we have no widget
    for goes into the second list and is shown as a summary rather than
    silently dropped -- a new check upstream must not become an invisible
    disabled Start button.
    """
    by_field: dict[str, list[str]] = {}
    other: list[str] = []
    for problem in cfg.validate():
        flag = problem.split(" ", 1)[0]
        field = FLAG_TO_FIELD.get(flag)
        if field is None:
            other.append(problem)
        else:
            by_field.setdefault(field, []).append(problem)
    return by_field, other


FLAG_TO_FIELD = {
    "--server": "server_url",
    "--name": "name",
    "--cores": "cores",
    "--ram": "ram_gb",
    "--mode": "mode",
    "--score-scale": "score_scale",
    "--unknown-metric": "unknown_metric",
    "--poll-interval": "poll_interval_s",
    "--max-tasks": "max_tasks",
    "--model-kinds": "model_kinds",
}


def _coerce(declared: Any, value: Any) -> Any:
    """Field types are strings under `from __future__ import annotations`."""
    name = declared if isinstance(declared, str) else getattr(declared, "__name__", "")

    if name.startswith("bool"):
        if not isinstance(value, bool):
            raise TypeError(f"expected a bool, got {type(value).__name__}")
        return value
    if name.startswith("int"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"expected an int, got {type(value).__name__}")
        return value
    if name.startswith("float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"expected a number, got {type(value).__name__}")
        return float(value)
    if name.startswith("str"):
        if not isinstance(value, str):
            raise TypeError(f"expected a string, got {type(value).__name__}")
        return value
    # `float | None`, `str | None`
    if value is None:
        return None
    if "float" in name:
        return float(value)
    return str(value)
