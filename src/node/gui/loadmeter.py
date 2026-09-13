"""Live CPU and memory numbers for the window, without fighting the heartbeat.

`resources.current_load` calls `psutil.cpu_percent(interval=None)`, whose "time
of the last sample" is a *module global*. The heartbeat thread already calls it
on the server's cadence; if the window called it too, each would consume the
other's delta and both would report nonsense. So the CPU figure here is derived
from this object's own `cpu_times` snapshot, and only the memory arithmetic --
which is stateless -- is borrowed from `resources`.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil

from node.agent.resources import Budget


@dataclass(frozen=True)
class Load:
    cpu_pct: float
    ram_pct: float
    agent_rss_mb: float
    tree_rss_mb: float
    budget_ram_pct: float


class LoadMeter:
    """Samples the machine on the GUI thread. Cheap enough for a 2 s timer."""

    def __init__(self) -> None:
        self._previous = psutil.cpu_times()
        self._process = psutil.Process()

    def sample(self, budget: Budget | None) -> Load:
        return Load(
            cpu_pct=self._cpu_percent(),
            ram_pct=float(psutil.virtual_memory().percent),
            **self._memory(budget),
        )

    def _cpu_percent(self) -> float:
        """System-wide CPU use since the previous sample, from our own delta."""
        try:
            current = psutil.cpu_times()
        except (psutil.Error, OSError):
            return 0.0

        previous, self._previous = self._previous, current
        busy = _busy(current) - _busy(previous)
        total = _total(current) - _total(previous)
        if total <= 0:
            # First call, or a clock that did not move. Zero is honest here.
            return 0.0
        return round(min(100.0, max(0.0, 100.0 * busy / total)), 1)

    def _memory(self, budget: Budget | None) -> dict[str, float]:
        try:
            rss = self._process.memory_info().rss
            tree = rss + sum(
                child.memory_info().rss
                for child in self._process.children(recursive=True)
            )
        except (psutil.Error, OSError):
            rss = tree = 0

        ram_bytes = budget.ram_bytes if budget else 0
        return {
            "agent_rss_mb": round(rss / 1024**2, 1),
            "tree_rss_mb": round(tree / 1024**2, 1),
            "budget_ram_pct": round(100.0 * tree / ram_bytes, 1) if ram_bytes else 0.0,
        }


def _busy(times: object) -> float:
    """Everything that is not idle. `iowait` counts as idle: the CPU was free."""
    total = _total(times)
    idle = getattr(times, "idle", 0.0) + getattr(times, "iowait", 0.0)
    return total - idle


def _total(times: object) -> float:
    # Fields differ by platform (no `iowait` on Windows, no `interrupt` on
    # Linux), so sum whatever this one has rather than naming them.
    return float(sum(getattr(times, field) for field in times._fields))  # type: ignore[attr-defined]
