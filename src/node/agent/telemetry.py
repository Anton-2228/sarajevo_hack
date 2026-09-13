"""The snapshot the heartbeat sends, and the only thing the work loop updates.

The shape of this follows the METRICS_GUIDE directly. The heavy loop updates a
local snapshot as often as it likes; the heartbeat thread reads it on the
server's own schedule and posts it. Nothing here does I/O, and nothing in the
training or scoring path waits on the network.

Three rules from the guide are enforced rather than documented:

*An idle node carries nothing over.* Finishing a task clears `round_id`, `stage`
and every gauge from the task that just ended -- a stale `progress_pct` of 100
on an idle node is a lie the dashboard would draw.

*`load` is numbers only.* Up to 32 finite gauges. A string, a bool, a NaN or an
infinity is a 422, and losing liveness to a telemetry detail would be a
remarkably silly way to drop off the mesh.

*No identifiers, no paths, no text.* Gauge names are a fixed vocabulary, never
built from a document, a file or a host. That is both a privacy rule and a
Prometheus cardinality one.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from node.agent import resources
from node.agent.models import (
    MAX_LOAD_KEYS,
    STAGES,
    HeartbeatRequest,
    is_id,
)

# A throughput figure from the last few seconds is what an operator wants; an
# average since the task began goes flat and stops saying anything.
THROUGHPUT_WINDOW_S = 30.0


class Telemetry:
    """Thread-safe current state of this node, as the heartbeat will report it.

    One writer (the work loop) and one reader (the heartbeat thread), so the
    lock is held only for dict copies -- never across a network call or a
    training step.
    """

    def __init__(self, budget: resources.Budget) -> None:
        self._budget = budget
        self._lock = threading.Lock()
        self._status = "idle"
        self._round_id: str | None = None
        self._stage: str | None = None
        self._gauges: dict[str, float] = {}
        # (timestamp, docs) pairs inside the throughput window.
        self._marks: list[tuple[float, int]] = []
        self._docs_total = 0

    # -- writes, from the work loop ---------------------------------------

    def idle(self) -> None:
        """Between tasks. Drops the finished task's round, stage and gauges."""
        with self._lock:
            self._status = "idle"
            self._round_id = None
            self._stage = None
            self._gauges = {}
            self._marks = []
            self._docs_total = 0

    def begin(self, round_id: str, stage: str = "downloading") -> None:
        with self._lock:
            self._status = "busy"
            self._round_id = round_id
            self._gauges = {}
            self._marks = []
            self._docs_total = 0
        self.stage(stage)

    def stage(self, stage: str, *, total: int | None = None) -> None:
        """Move to a new phase. Progress restarts: it is per-phase, not per-task."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        with self._lock:
            self._stage = stage
            self._status = "busy"
            # Each phase measures its own progress, so the counters that belong
            # to the previous one would otherwise read as this one's.
            for key in ("progress_pct", "docs_processed", "docs_per_sec", "eta_s"):
                self._gauges.pop(key, None)
            self._marks = []
            if total is not None:
                self._docs_total = max(int(total), 0)
                self._gauges["docs_total"] = float(self._docs_total)

    def progress(self, processed: int, total: int | None = None) -> None:
        """How far through the current phase, with rate and ETA derived from it."""
        now = time.monotonic()
        with self._lock:
            if total is not None:
                self._docs_total = max(int(total), 0)
            total_docs = self._docs_total
            processed = max(int(processed), 0)

            self._gauges["docs_processed"] = float(processed)
            if total_docs:
                self._gauges["docs_total"] = float(total_docs)
                self._gauges["progress_pct"] = round(
                    min(100.0, 100.0 * processed / total_docs), 2
                )

            self._marks.append((now, processed))
            cutoff = now - THROUGHPUT_WINDOW_S
            while len(self._marks) > 2 and self._marks[0][0] < cutoff:
                self._marks.pop(0)

            rate = _rate(self._marks)
            if rate is not None:
                self._gauges["docs_per_sec"] = round(rate, 2)
                if total_docs and rate > 0:
                    remaining = max(total_docs - processed, 0)
                    self._gauges["eta_s"] = round(remaining / rate, 1)

    def gauge(self, name: str, value: float) -> None:
        """One extra number. Dropped if it is not a finite, ID-shaped gauge."""
        if not is_id(name):
            return
        number = _finite(value)
        if number is None:
            return
        with self._lock:
            self._gauges[name] = number

    def train_loss(self, value: float) -> None:
        self.gauge("train_loss", value)

    # -- reads, from the heartbeat thread ---------------------------------

    def snapshot(self) -> HeartbeatRequest:
        """What to send right now. Called on the server's interval, not ours."""
        with self._lock:
            status = self._status
            round_id = self._round_id
            stage = self._stage
            gauges = dict(self._gauges)

        if status == "busy":
            # Sampled at send time rather than carried in the snapshot: these
            # describe the machine now, not when the last batch finished.
            gauges.update(resources.current_load(self._budget))

        return HeartbeatRequest(
            status=status,
            round_id=round_id,
            stage=stage,
            load=_clean(gauges),
        )


def _rate(marks: list[tuple[float, int]]) -> float | None:
    if len(marks) < 2:
        return None
    (first_t, first_n), (last_t, last_n) = marks[0], marks[-1]
    elapsed = last_t - first_t
    if elapsed <= 0 or last_n < first_n:
        return None
    return (last_n - first_n) / elapsed


def _finite(value: Any) -> float | None:
    """`load` takes finite numbers and nothing else -- a bool is not a number."""
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clean(gauges: dict[str, Any]) -> dict[str, float]:
    """Make `load` something the server will accept: ≤ 32 finite numbers.

    Over the limit the dashboard's own keys are kept first, because they are the
    ones anybody is looking at; the node's extra diagnostics give way.
    """
    from node.agent.models import DASHBOARD_LOAD_KEYS

    cleaned: dict[str, float] = {}
    for name, value in gauges.items():
        if not is_id(name):
            continue
        number = _finite(value)
        if number is not None:
            cleaned[name] = number

    if len(cleaned) <= MAX_LOAD_KEYS:
        return cleaned

    kept = {k: cleaned[k] for k in DASHBOARD_LOAD_KEYS if k in cleaned}
    for name, value in cleaned.items():
        if len(kept) >= MAX_LOAD_KEYS:
            break
        kept.setdefault(name, value)
    return kept
