"""Server-side label source for campaigns.

Server-master: the server holds the pool and judges it, nodes only train/score (see campaigns.py).
This is the seam where a real judge plugs in later — an LLM call, a queue for human review, whatever.
For the hackathon contour, `MockOracle` stands in: deterministic given a seed, so a campaign run is
reproducible without any real judging.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Protocol


class Oracle(Protocol):
    name: str

    def label(self, chunk_ids: Iterable[str], texts: Dict[str, str]) -> Dict[str, int]:
        """Judge each chunk. Returns an integer label per id (e.g. 0..N, or 0/1)."""
        ...


class MockOracle:
    """Deterministic stand-in: label is a hash of (chunk_id, seed) compared to `good_rate`.

    Ignores `texts` entirely — good for wiring the campaign loop end-to-end before a real judge (LLM
    call, human review queue, ...) is plugged in behind the same `Oracle` interface.
    """

    name = "mock"

    def __init__(self, good_rate: float = 0.1, seed: str = "mock"):
        self.good_rate = good_rate
        self.seed = seed

    def label(self, chunk_ids: Iterable[str], texts: Dict[str, str]) -> Dict[str, int]:
        out = {}
        for cid in chunk_ids:
            h = int(hashlib.sha256(f"{self.seed}:{cid}".encode()).hexdigest(), 16)
            out[cid] = int((h % 10_000) / 10_000 < self.good_rate)
        return out


class StaticOracle:
    """Oracle backed by real judgments computed up front (e.g. tools/load_golden_shard.py pulling
    Llama-3 scores from FineWeb-Edu annotations) — a mocked *call* (no live LLM), but real labels and
    real text, so a demo run's numbers mean the same thing the earlier simulator runs did.
    """

    name = "static"

    def __init__(self, labels: Dict[str, int]):
        self._labels = labels

    def label(self, chunk_ids: Iterable[str], texts: Dict[str, str]) -> Dict[str, int]:
        chunk_ids = list(chunk_ids)
        missing = [c for c in chunk_ids if c not in self._labels]
        if missing:
            raise KeyError(f"StaticOracle has no golden label for {len(missing)} chunk(s), "
                           f"e.g. {missing[:3]} — was this shard loaded via load_golden_shard.py?")
        return {c: self._labels[c] for c in chunk_ids}
