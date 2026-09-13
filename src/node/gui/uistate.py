"""The four states the window can be in, and what the buttons say in each.

Pure, so the state machine can be tested without a display. The window owns the
phase; everything visual is derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# How long a graceful stop is allowed to look like it is working before the
# window offers the hammer. Long enough that an ordinary stop between rounds
# never shows it, short enough that a stop during training does.
FORCE_AFTER_S = 10.0


class UiPhase(Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


@dataclass(frozen=True)
class ButtonSpec:
    label: str
    enabled: bool
    tooltip: str = ""


def button_spec(
    phase: UiPhase, *, valid: bool = True, first_problem: str = ""
) -> ButtonSpec:
    """What the primary button shows. `valid` only matters while idle."""
    if phase is UiPhase.IDLE:
        return ButtonSpec(
            label="Start",
            enabled=valid,
            tooltip="" if valid else first_problem,
        )
    if phase is UiPhase.STARTING:
        return ButtonSpec(
            label="Cancel start",
            enabled=True,
            # `AgentLoop.run` does not check the stop flag until after health()
            # and the handshake, so a cancel here waits out a network timeout.
            tooltip="The cancel takes effect after the current network call (up to ~40 s)",
        )
    if phase is UiPhase.RUNNING:
        return ButtonSpec(label="Stop", enabled=True)
    return ButtonSpec(
        label="Stopping…",
        enabled=False,
        tooltip="The node is finishing the current round",
    )


def controls_editable(phase: UiPhase) -> bool:
    """The budget cannot change under a running agent; it is read once at start."""
    return phase is UiPhase.IDLE


def shows_busy_bar(phase: UiPhase) -> bool:
    return phase in (UiPhase.STARTING, UiPhase.STOPPING)


def shows_live_panel(phase: UiPhase) -> bool:
    return phase in (UiPhase.RUNNING, UiPhase.STOPPING)


# Which pill variant renders each of the agent's statuses. Keyed on
# `AgentView.status` rather than on the window's phase, because the agent has
# more to say than four states. The hex values live in `theme`; this stays
# free of Qt so it can be tested without a display.
PILL_VARIANTS = {
    "starting": "warn",
    "idle": "good",
    "busy": "good",
    "degraded": "warn",
    "stopping": "off",
    "offline": "off",
    "error": "crit",
}

DEFAULT_VARIANT = "off"

STATUS_LABELS = {
    "starting": "starting",
    "idle": "waiting for tasks",
    "busy": "training",
    "degraded": "connection unstable",
    "stopping": "stopping",
    "offline": "stopped",
    "error": "error",
}


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def status_variant(status: str) -> str:
    return PILL_VARIANTS.get(status, DEFAULT_VARIANT)
