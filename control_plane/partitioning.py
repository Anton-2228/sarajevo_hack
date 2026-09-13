"""Deterministic, stateless partitioning for server-held chunk pools."""
from __future__ import annotations

from typing import List


def partition_of(chunk_id: str, n_partitions: int) -> int:
    """Return the stable partition for a validated hexadecimal chunk id."""
    if n_partitions <= 0:
        raise ValueError("n_partitions must be positive")
    return int(chunk_id[:8], 16) % n_partitions


def chunk_ids_for_partition(pool: List[str], partition: int, n_partitions: int) -> List[str]:
    """Filter ``pool`` without storing partition membership in the database."""
    if not 0 <= partition < n_partitions:
        raise ValueError("partition must be between 0 and n_partitions - 1")
    return [chunk_id for chunk_id in pool if partition_of(chunk_id, n_partitions) == partition]
