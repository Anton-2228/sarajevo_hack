"""Choose which chunks in one partition/domain the oracle should label next."""
from __future__ import annotations

from typing import Tuple

import numpy as np

STRATEGIES: Tuple[str, ...] = ("random", "cutoff", "qbc")


def acquire(strategy: str, budget: int, labeled: np.ndarray, mean: np.ndarray, std: np.ndarray,
           k_frac: float, rng: np.random.Generator) -> np.ndarray:
    """New pool indices (0..n-1) to label, never repeating `labeled`.

    ``mean`` contains one domain expert's scores for this partition and ``std`` is currently zero.
    ``k_frac`` is the campaign's target top-k fraction, around which cutoff/qbc sample.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown acquisition strategy {strategy!r}, expected one of {STRATEGIES}")
    n = len(mean)
    budget = min(budget, n - len(labeled))
    if budget <= 0:
        return np.empty(0, dtype=int)
    free = np.ones(n, bool)
    free[labeled] = False
    if strategy == "random":
        return rng.choice(np.flatnonzero(free), budget, replace=False)

    order = np.argsort(-mean, kind="stable")
    frac = np.empty(n)
    frac[order] = (np.arange(n) + 1) / n
    third = budget // 3
    if strategy == "cutoff":
        near = [i for i in np.argsort(np.abs(frac - k_frac), kind="stable") if free[i]][:third]
        free[near] = False
        top = [i for i in order if free[i]][:third]
        free[top] = False
        picked = near + top
    else:  # qbc: most disagreement among nodes, in a band around the cutoff
        band = np.flatnonzero(free & (frac >= k_frac / 3) & (frac <= k_frac * 3))
        picked = list(band[np.argsort(-std[band], kind="stable")][:budget - third])
        free[picked] = False
    rest = rng.choice(np.flatnonzero(free), budget - len(picked), replace=False)
    return np.concatenate([np.array(picked, dtype=int), rest])
