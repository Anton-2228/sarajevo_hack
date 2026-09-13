"""The presets have to mean something on every machine, not just this one."""

from __future__ import annotations

import pytest

from node.agent.config import MIN_RAM_GB
from node.agent.resources import HardwareInfo, resolve_budget
from node.gui.presets import LARGE, MEDIUM, PRESET_LEVELS, SMALL, preset_budget

# Deliberately includes the two machines that break naive arithmetic: a
# single-core box (25% of 1 core is zero) and a tiny-RAM one (25% of 1 GiB is
# below the minimum a training run can do anything with).
MACHINES = [
    HardwareInfo(logical_cores=1, total_ram_gb=1.0, disk_free_gb=10.0),
    HardwareInfo(logical_cores=2, total_ram_gb=2.0, disk_free_gb=10.0),
    HardwareInfo(logical_cores=4, total_ram_gb=8.0, disk_free_gb=50.0),
    HardwareInfo(logical_cores=8, total_ram_gb=16.0, disk_free_gb=50.0),
    HardwareInfo(logical_cores=16, total_ram_gb=64.0, disk_free_gb=500.0),
    HardwareInfo(logical_cores=128, total_ram_gb=1024.0, disk_free_gb=5000.0),
]

IDS = [f"{m.logical_cores}c/{m.total_ram_gb:g}g" for m in MACHINES]


@pytest.mark.parametrize("hw", MACHINES, ids=IDS)
@pytest.mark.parametrize("level", PRESET_LEVELS)
def test_budget_fits_the_machine(hw: HardwareInfo, level: str) -> None:
    budget = preset_budget(level, hw)

    assert 1 <= budget.cores <= hw.logical_cores
    assert budget.ram_gb >= MIN_RAM_GB
    assert budget.ram_gb <= hw.total_ram_gb


@pytest.mark.parametrize("hw", MACHINES, ids=IDS)
def test_levels_are_monotonic(hw: HardwareInfo) -> None:
    small = preset_budget(SMALL, hw)
    medium = preset_budget(MEDIUM, hw)
    large = preset_budget(LARGE, hw)

    assert small.cores <= medium.cores <= large.cores
    assert small.ram_gb <= medium.ram_gb <= large.ram_gb


@pytest.mark.parametrize("hw", MACHINES, ids=IDS)
@pytest.mark.parametrize("level", PRESET_LEVELS)
def test_a_preset_never_trips_the_safety_net(hw: HardwareInfo, level: str) -> None:
    """The number in the window must be the number the agent uses.

    `resolve_budget` clamps and warns; a preset that made it warn would mean the
    window is promising something the agent then quietly changes. The one
    warning it may still emit is about core affinity, which depends on the
    *real* machine rather than on `hw`, so only budget warnings are asserted.
    """
    budget = preset_budget(level, hw)
    _, warnings = resolve_budget(budget.cores, budget.ram_gb, hw)

    assert [w for w in warnings if "asked for" in w] == []


@pytest.mark.parametrize("hw", MACHINES, ids=IDS)
def test_large_leaves_a_core_for_the_operator(hw: HardwareInfo) -> None:
    budget = preset_budget(LARGE, hw)
    if hw.logical_cores > 1:
        assert budget.cores == hw.logical_cores - 1
    else:
        assert budget.cores == 1


def test_an_unknown_level_falls_back_to_medium() -> None:
    hw = HardwareInfo(logical_cores=8, total_ram_gb=16.0, disk_free_gb=50.0)

    assert preset_budget("enormous", hw) == preset_budget(MEDIUM, hw)


def test_preset_budget_does_no_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure function of its argument: it must not look at the real machine."""
    import psutil

    monkeypatch.setattr(psutil, "cpu_count", lambda **_: pytest.fail("probed the CPU"))
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: pytest.fail("probed memory")
    )

    preset_budget(MEDIUM, HardwareInfo(logical_cores=4, total_ram_gb=8.0, disk_free_gb=1.0))
