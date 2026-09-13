"""How much of the machine each preset hands over.

The operator picks a share of their own laptop, not a pair of numbers, so the
three levels have to mean something on a 2-core netbook and on a 128-core
server alike. Everything here is a pure function of `HardwareInfo`: no probing,
no I/O, no Qt -- which is what makes the "a preset never trips a budget
warning" invariant testable against `resources.resolve_budget` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from node.agent.config import MIN_RAM_GB
from node.agent.resources import HardwareInfo
from node.gui.text import cores as cores_text

SMALL = "small"
MEDIUM = "medium"
LARGE = "large"
PRESET_LEVELS = (SMALL, MEDIUM, LARGE)

# (core share, RAM share, RAM ceiling in GiB). A `None` core share means every
# core but one -- "large" should not be a fraction, it should be the machine.
_SHARES: dict[str, tuple[float | None, float, float]] = {
    SMALL: (0.25, 0.25, 2.0),
    MEDIUM: (0.50, 0.50, 4.0),
    LARGE: (None, 0.75, 16.0),
}

# Never hand over the whole box. The operator is still using this machine, and
# a node that swaps their editor out is a node they will uninstall.
HEADROOM_GB = 1.0

LABELS: dict[str, str] = {
    SMALL: "Small",
    MEDIUM: "Medium",
    LARGE: "Large",
}

DESCRIPTIONS: dict[str, str] = {
    SMALL: "A quarter of the machine. The node is all but invisible in the background.",
    MEDIUM: "Half the machine. Sensible while the laptop is still being used.",
    LARGE: "Everything but one core. For a dedicated machine.",
}


@dataclass(frozen=True)
class PresetBudget:
    """What a preset resolves to on this particular machine."""

    cores: int
    ram_gb: float

    def summary(self) -> str:
        return f"{cores_text(self.cores)} · {self.ram_gb:g} GiB"


def preset_budget(level: str, hw: HardwareInfo) -> PresetBudget:
    """Resolve a preset against the machine. Pure; never raises on a known level."""
    try:
        core_share, ram_share, ram_ceiling = _SHARES[level]
    except KeyError:
        core_share, ram_share, ram_ceiling = _SHARES[MEDIUM]

    if core_share is None:
        cores = max(1, hw.logical_cores - 1)
    else:
        cores = max(1, int(hw.logical_cores * core_share))
    cores = min(cores, max(1, hw.logical_cores))

    ram_gb = min(hw.total_ram_gb * ram_share, ram_ceiling)
    # On a small machine the headroom clamp is what actually decides, and it
    # can push below the share -- deliberately. Better a node that fits.
    ram_gb = min(ram_gb, max(MIN_RAM_GB, hw.total_ram_gb - HEADROOM_GB))
    ram_gb = max(ram_gb, MIN_RAM_GB)

    return PresetBudget(cores=cores, ram_gb=round(ram_gb, 1))


def describe_machine(hw: HardwareInfo) -> str:
    return f"{cores_text(hw.logical_cores)} / {hw.total_ram_gb:.1f} GiB"
