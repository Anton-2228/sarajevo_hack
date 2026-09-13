"""Plurals, because "1 cores" in a window reads as a bug."""

from __future__ import annotations


def plural(n: int, one: str, many: str) -> str:
    """Pick the form for `n`: 1 core, 2 cores."""
    return one if abs(n) == 1 else many


def cores(n: int) -> str:
    return f"{n} {plural(n, 'core', 'cores')}"


def rounds(n: int) -> str:
    return f"{n} {plural(n, 'round', 'rounds')}"
