"""How the agent tells someone what it is doing.

The loop and the runner never call `print`. They call a `Reporter`, and the
console implementation is only one of the things that can be on the other end
-- a GUI is the next one, and it should not require touching the loop to add.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from node.agent.models import AgentView, TaskProgress

LOG = logging.getLogger("node.agent")


class Reporter(Protocol):
    """The agent's entire output surface."""

    def state(self, view: AgentView) -> None:
        """The agent as a whole changed: enrolled, went busy, is stopping."""

    def task(self, progress: TaskProgress) -> None:
        """A round moved to a new phase."""

    def note(self, level: int, message: str, **fields: Any) -> None:
        """Something worth saying that is not a state change."""


class NullReporter:
    """Swallows everything. For tests, and for callers that render elsewhere."""

    def state(self, view: AgentView) -> None:
        return None

    def task(self, progress: TaskProgress) -> None:
        return None

    def note(self, level: int, message: str, **fields: Any) -> None:
        return None


class ConsoleReporter:
    """Logs to stderr through the stdlib, leaving stdout free for `--json`."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or LOG

    def state(self, view: AgentView) -> None:
        self._log.info(
            "agent %s node=%s completed=%d failed=%d",
            view.status,
            view.node_id or "-",
            view.completed,
            view.failed,
        )

    def task(self, progress: TaskProgress) -> None:
        self._log.info(
            "round %s: %s%s",
            progress.round_id,
            progress.phase,
            _render(progress.detail),
        )

    def note(self, level: int, message: str, **fields: Any) -> None:
        self._log.log(level, "%s%s", message, _render(fields))


def _render(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    return " " + " ".join(f"{k}={_value(v)}" for k, v in sorted(fields.items()))


def _value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    # Long payload dumps belong at DEBUG with their own truncation, not spliced
    # into a one-line INFO record.
    return text if len(text) <= 120 else text[:117] + "..."
