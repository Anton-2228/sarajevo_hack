"""What happened to each round this run, assembled from what the agent says.

The agent reports a round moving through three phases -- `acked`, `running`,
`done` -- and nothing at all when one fails: `_fail` only bumps `view.failed`
and sets `view.last_error`. So a failure has to be *inferred*, by noticing that
the counter moved while a round was still open. That inference is the reason
this is a separate, pure module with tests rather than a few lines inside a
widget.

No Qt here, and no I/O.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from node.agent.models import AgentView, TaskProgress

ACKED = "acked"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# The phases a round walks through, in order. A failure stops it wherever it was.
STEPS = (ACKED, RUNNING, DONE)

STEP_LABELS = {
    ACKED: "accepted",
    RUNNING: "training",
    DONE: "submitted",
    FAILED: "error",
}

# Enough to scroll back through a long session without the window growing
# without bound. Older rounds are in the journal on disk, not here.
DEFAULT_LIMIT = 60


@dataclass
class RoundEntry:
    """One round, as far as the window knows."""

    round_id: str
    phase: str = ACKED
    started_at: float = field(default_factory=time.time)
    n_scores: int | None = None
    seconds: float | None = None
    error: str | None = None
    mode: str | None = None

    @property
    def finished(self) -> bool:
        return self.phase in (DONE, FAILED)

    @property
    def step_index(self) -> int:
        """How many of the three steps are behind it."""
        if self.phase == DONE:
            return len(STEPS)
        if self.phase == FAILED:
            return 0
        return STEPS.index(self.phase) + 1 if self.phase in STEPS else 0

    def elapsed(self, now: float | None = None) -> float:
        if self.seconds is not None:
            return self.seconds
        return max(0.0, (now if now is not None else time.time()) - self.started_at)


class RoundTracker:
    """Turns the agent's two event streams into one list of rounds."""

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self._limit = limit
        self._order: list[str] = []  # oldest first
        self._by_id: dict[str, RoundEntry] = {}
        self._failed_seen = 0

    def reset(self) -> None:
        self._order.clear()
        self._by_id.clear()
        self._failed_seen = 0

    def entries(self) -> list[RoundEntry]:
        """Newest first, which is the order a person reads them in."""
        return [self._by_id[r] for r in reversed(self._order)]

    def current(self) -> RoundEntry | None:
        for entry in self.entries():
            if not entry.finished:
                return entry
        return None

    def on_task(self, progress: TaskProgress) -> RoundEntry:
        entry = self._by_id.get(progress.round_id)
        if entry is None:
            entry = RoundEntry(
                round_id=progress.round_id, started_at=progress.started_at
            )
            self._by_id[progress.round_id] = entry
            self._order.append(progress.round_id)
            self._trim()

        entry.phase = progress.phase
        detail = progress.detail or {}
        if "mode" in detail:
            entry.mode = str(detail["mode"])
        if "n_scores" in detail:
            entry.n_scores = int(detail["n_scores"])
        if "seconds" in detail:
            entry.seconds = float(detail["seconds"])
        if entry.phase == DONE and entry.seconds is None:
            entry.seconds = max(0.0, time.time() - entry.started_at)
        return entry

    def on_state(self, view: AgentView) -> RoundEntry | None:
        """Notice a failure the agent never announced.

        `view.failed` is the only trace of one. When it moves, whichever round
        is still open is the one that died -- the loop handles rounds strictly
        one at a time, so there is never a second candidate.
        """
        if view.failed <= self._failed_seen:
            self._failed_seen = view.failed
            return None
        self._failed_seen = view.failed

        entry = self.current()
        if entry is None and view.current is not None:
            # The round failed before it ever reached us as a task event.
            entry = self.on_task(view.current)
        if entry is None:
            return None

        entry.phase = FAILED
        entry.error = view.last_error
        if entry.seconds is None:
            entry.seconds = max(0.0, time.time() - entry.started_at)
        return entry

    def _trim(self) -> None:
        while len(self._order) > self._limit:
            self._by_id.pop(self._order.pop(0), None)


def describe(entry: RoundEntry) -> str:
    """The one line under a round: what it produced, or why it stopped."""
    if entry.phase == FAILED:
        return entry.error or "the round failed"
    if entry.phase == DONE:
        parts = []
        if entry.n_scores is not None:
            parts.append(f"{entry.n_scores} scores")
        if entry.seconds is not None:
            parts.append(f"{entry.seconds:.0f} s")
        return " · ".join(parts) or "submitted"
    if entry.phase == RUNNING:
        return f"training · {entry.mode}" if entry.mode else "training"
    return "accepted, preparing"


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    return f"{total // 60}:{total % 60:02d}"
