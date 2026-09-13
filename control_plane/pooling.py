"""Global top-k across nodes: held-out calibration, budget allocation and the Reliability Gate.

Every node reports, as counts in agg_stats, how its proxy ranks the assigned CP-labelled held-out split:
`ho_n`, `ho_good`, and for each top-quantile q in HELDOUT_QUANTILES the number of held-out docs in its
top q (`ho_n_qXX`) and how many of them are good (`ho_good_qXX`). Counts rather than rates, so a
precise estimate can be told from a lucky handful. From that curve the control plane

- calibrates: a chunk ranked in the top fraction f of its node is scored by the share of good docs
  expected at f — smoothed per-band held-out precision, interpolated between band midpoints (a step per
  band hands out whole bands at once and lost up to 3.5 points of top-k quality in simulation). Pooling
  on this estimate spends the budget where good docs are expected, so a node with a richer corpus or a
  sharper proxy gets more of it — z-scoring instead hands every node the same share;
- gates: once the budget is allocated, a node's cutoff is q = selected / n_scores, and the node is
  trusted when the Wilson lower bound of its held-out precision above q — never fewer than
  MIN_HELDOUT_AT_CUTOFF held-out docs — reaches `min_precision`. Untrusted nodes are dropped and the
  budget is reallocated until no remaining node fails.
"""
from __future__ import annotations

import math
import statistics
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schemas import HELDOUT_QUANTILES, q_key

PRIOR_STRENGTH = 5.0        # pseudo-docs pulling a band's precision toward the node's base rate
WILSON_Z = 1.96
MIN_HELDOUT_AT_CUTOFF = 30  # a node with few or no slots is not judged on a handful of held-out docs
BAND_MIDPOINTS = tuple((lo + hi) / 2 for lo, hi in zip((0.0,) + HELDOUT_QUANTILES, HELDOUT_QUANTILES + (1.0,)))


def wilson_lower(good: float, n: float, z: float = WILSON_Z) -> float:
    if n <= 0:
        return 0.0
    p = min(max(good / n, 0.0), 1.0)
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z * z / n))


def _non_increasing(values: List[float], weights: List[float]) -> List[float]:
    """Weighted isotonic fit (pool adjacent violators): a band ranked higher is not expected to be worse."""
    blocks: List[List[float]] = []  # [value, weight, bands merged]
    for value, weight in zip(values, weights):
        blocks.append([value, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            last = blocks[-1]
            last[0] = (last[0] * last[1] + v2 * w2) / (last[1] + w2)
            last[1] += w2
            last[2] += c2
    return [b[0] for b in blocks for _ in range(int(b[2]))]


@dataclass(frozen=True)
class HeldoutCurve:
    points: Tuple[Tuple[float, float, float], ...]  # (q, held-out docs in top q, good among them), q = 0..1

    @classmethod
    def from_stats(cls, stats: Dict[str, Any]) -> Optional["HeldoutCurve"]:
        if "ho_n" not in stats:
            return None
        inner = tuple((q, float(stats[f"ho_n_q{q_key(q)}"]), float(stats[f"ho_good_q{q_key(q)}"]))
                      for q in HELDOUT_QUANTILES)
        return cls(((0.0, 0.0, 0.0),) + inner + ((1.0, float(stats["ho_n"]), float(stats["ho_good"])),))

    @property
    def n(self) -> float:
        return self.points[-1][1]

    @property
    def base_rate(self) -> float:
        _, n, good = self.points[-1]
        return good / n

    def at(self, q: float) -> Tuple[float, float]:
        """(held-out docs, good docs) in the top q, linear between reported quantiles."""
        q = min(max(q, 0.0), 1.0)
        for (q0, n0, g0), (q1, n1, g1) in zip(self.points, self.points[1:]):
            if q <= q1:
                t = (q - q0) / (q1 - q0)
                return n0 + t * (n1 - n0), g0 + t * (g1 - g0)
        return self.points[-1][1], self.points[-1][2]

    def band_precision(self) -> List[float]:
        """Smoothed share of good docs per band: top 1%, 1-2%, 2-5%, ..., 50-100%."""
        base = self.base_rate
        values, weights = [], []
        for (_, n0, g0), (_, n1, g1) in zip(self.points, self.points[1:]):
            values.append((g1 - g0 + PRIOR_STRENGTH * base) / (n1 - n0 + PRIOR_STRENGTH))
            weights.append(n1 - n0 + PRIOR_STRENGTH)
        return _non_increasing(values, weights)

    def expected_precision(self, fracs: Sequence[float]) -> List[float]:
        """Share of good docs expected at each rank fraction: band precisions interpolated between band
        midpoints (flat beyond the outer ones), still non-increasing in f."""
        bands, out = self.band_precision(), []
        for f in fracs:
            j = bisect_left(BAND_MIDPOINTS, f)
            if j == 0 or j == len(BAND_MIDPOINTS):
                out.append(bands[min(j, len(bands) - 1)])
            else:
                m0, m1 = BAND_MIDPOINTS[j - 1], BAND_MIDPOINTS[j]
                out.append(bands[j - 1] + (f - m0) / (m1 - m0) * (bands[j] - bands[j - 1]))
        return out


@dataclass
class NodeSubmission:
    node_id: str
    chunk_ids: List[str]
    scores: List[float]
    curve: Optional[HeldoutCurve]


def _pooling_scores(node: NodeSubmission, normalize: str) -> Tuple[List[float], List[float]]:
    """(pooling score, share of the node's chunks ranked at or above it) for every chunk."""
    n = len(node.scores)
    frac = [0.0] * n
    for rank, i in enumerate(sorted(range(n), key=node.scores.__getitem__, reverse=True)):
        frac[i] = (rank + 1) / n
    if normalize == "calibrated":
        return node.curve.expected_precision(frac), frac
    if normalize == "zscore":
        mu, sd = statistics.fmean(node.scores), statistics.pstdev(node.scores) or 1.0
        return [(v - mu) / sd for v in node.scores], frac
    if normalize == "rank":
        denom = max(n - 1, 1)
        return [(n - n * f) / denom for f in frac], frac
    return list(node.scores), frac


def _judge(node: NodeSubmission, selected: int, min_precision: float) -> Dict[str, Any]:
    q = selected / len(node.scores)
    view: Dict[str, Any] = {"node_id": node.node_id, "selected": selected, "n_scores": len(node.scores),
                            "cutoff_q": round(q, 6)}
    if node.curve is None:
        return {**view, "trust": "unknown"}
    n_top, good_top = node.curve.at(max(q, min(1.0, MIN_HELDOUT_AT_CUTOFF / node.curve.n)))
    lower = wilson_lower(good_top, n_top)
    return {**view,
            "trust": "trusted" if lower >= min_precision else "untrusted",
            "heldout_at_cutoff": round(n_top, 2),
            "precision_at_cutoff": round(good_top / n_top, 4) if n_top else None,
            "precision_lower_bound": round(lower, 4),
            "heldout_base_rate": round(node.curve.base_rate, 4)}


def select_topk(nodes: Sequence[NodeSubmission], k: int, normalize: str = "calibrated",
                min_precision: float = 0.2, enforce_gate: bool = True, include_unknown: bool = False
                ) -> Dict[str, Any]:
    skipped: List[Dict[str, Any]] = []
    active: List[NodeSubmission] = []
    for node in nodes:
        if node.curve is None and (normalize == "calibrated" or not include_unknown):
            skipped.append({"node_id": node.node_id, "trust": "unknown", "reason": "no held-out curve"})
        else:
            active.append(node)
    pooling = {node.node_id: _pooling_scores(node, normalize) for node in active}

    while True:
        pooled = sorted(((value, frac, node.node_id, i) for node in active
                         for i, (value, frac) in enumerate(zip(*pooling[node.node_id]))),
                        key=lambda item: (-item[0], item[1]))
        selected = pooled[:k]
        counts = Counter(item[2] for item in selected)
        gate = [_judge(node, counts.get(node.node_id, 0), min_precision) for node in active]
        failing = {g["node_id"] for g in gate if g["trust"] == "untrusted"}
        if not enforce_gate or not failing:
            break
        skipped += [{**g, "reason": "held-out precision at cutoff below gate"} for g in gate if g["node_id"] in failing]
        active = [node for node in active if node.node_id not in failing]

    by_id = {node.node_id: node for node in active}
    return {
        "nodes_used": [node.node_id for node in active],
        "nodes_skipped": skipped,
        "gate": gate,
        "pool_size": len(pooled),
        "selected_per_node": dict(counts),
        "selected": [{"node_id": nid, "chunk_id": by_id[nid].chunk_ids[i], "raw_score": by_id[nid].scores[i],
                      "norm_score": value} for value, _, nid, i in selected],
    }
