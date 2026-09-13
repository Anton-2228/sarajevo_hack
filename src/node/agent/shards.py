"""The corpus, fetched from the control plane.

Contract 0.7.0 moved the data. A task no longer names something this node
already holds; it names a *server-held shard* and the node reads the texts and
the current oracle labels out of it:

    GET /shards/{shard_id}/chunks?partition=&n_partitions=
    GET /shards/{shard_id}/labels

Three things about this are worth knowing before reading the code.

*Chunk ids come from the server, never from us.* `datasets.chunk_id_for` hashes
local text to invent an id; a shard already has one, and recomputing it would
produce a ranking keyed by ids the server cannot match to its pool.

*Labels are shard-wide and unsplit.* One map covers every campaign over the
shard, and nothing in it marks a document as train or held-out (CONTRACT.md
4.7). Carving out an honest evaluation slice is therefore the node's job, and
`runner` does it -- the server has no opinion to respect here.

*Partition membership is a formula, not a negotiation.* `int(chunk_id[:8], 16)
% n_partitions`, computed identically on both sides, which is what lets a node
filter the shard-wide label map down to its own domain without another call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from node.agent.datasets import Chunk
from node.agent.models import HASH_RE, Routing, TaskView

LOG = logging.getLogger("node.agent.shards")


class CorpusUnavailable(Exception):
    """The shard this round names cannot be turned into work.

    Always permanent. An empty or unreadable shard is not something a retrain
    fixes, and a node that guesses past it submits scores for documents the
    control plane never assigned it.
    """


def partition_of(chunk_id: str, n_partitions: int) -> int:
    """Which partition owns this chunk. CONTRACT.md 4.7, verified against the server.

    Stateless by design: the first 8 hex characters of the id, modulo the
    partition count. Both sides compute it, so the node can filter the
    shard-wide label map to its own domain locally.
    """
    if n_partitions < 1:
        raise ValueError(f"n_partitions must be at least 1, got {n_partitions}")
    return int(chunk_id[:8], 16) % n_partitions


@dataclass
class Corpus:
    """What one round has to work with: what to train on, and what to score.

    The two are not the same list. In `experts` the scoring set is the whole
    pool while the training labels are one domain's; in `sharded` both are the
    node's own partition.
    """

    shard_id: str
    to_score: list[Chunk]
    to_train: list[Chunk]
    routing: Routing
    # Size of the pool this submission is answerable for -- the partition in
    # `sharded`, the whole shard otherwise. Becomes agg_stats.n_chunks.
    pool_size: int = 0
    n_labels_available: int = 0
    n_dedup_dropped: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def n_score(self) -> int:
        return len(self.to_score)

    @property
    def n_train(self) -> int:
        return len(self.to_train)


class CorpusSource(Protocol):
    """Where `runner` gets a round's documents. One method, so a test can fake it."""

    def fetch(self, task: TaskView) -> Corpus: ...


class ShardCorpusSource:
    """The real one: reads the shard the task names off the control plane."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def fetch(self, task: TaskView) -> Corpus:
        # A campaign whose routing does not parse is unworkable. Scoring the
        # whole pool "just in case" is the one thing that must not happen: in
        # `sharded` the server concatenates rankings without deduplicating, so
        # a wrong partition corrupts every other node's share of the budget.
        if task.routing_problems:
            raise CorpusUnavailable(
                f"round {task.round_id} has unusable routing: "
                + "; ".join(task.routing_problems)
            )

        shard_id = task.shard_id
        if not shard_id:
            raise CorpusUnavailable(f"round {task.round_id} names no shard to score")

        routing = task.routing
        warnings: list[str] = []

        # `sharded` asks for its slice and gets exactly that; everything else
        # reads the pool whole, because that is what it must cover.
        if routing.mode == "sharded":
            raw = self._client.get_shard_chunks(
                shard_id,
                partition=routing.partition,
                n_partitions=routing.n_partitions,
            )
        else:
            raw = self._client.get_shard_chunks(shard_id)

        chunks, n_duplicate, n_malformed = _as_chunks(raw)
        if n_malformed:
            warnings.append(
                f"shard {shard_id} returned {n_malformed} chunk(s) the contract's "
                "HASH/text rules reject; they were left out"
            )
        if n_duplicate:
            warnings.append(
                f"shard {shard_id} returned {n_duplicate} chunk_id(s) twice; "
                "duplicates in a submission are a 422, so only the first was kept"
            )
        if not chunks:
            # The server answers an unknown shard with 200 and an empty list
            # rather than a 404, so emptiness is the only signal there is.
            raise CorpusUnavailable(
                f"shard {shard_id!r} came back with no usable chunks"
                + (f" for partition {routing.partition}" if routing.mode == "sharded" else "")
                + " -- either it was never loaded or this node was routed to an empty slice"
            )

        labels = self._client.get_shard_labels(shard_id)
        if not isinstance(labels, dict):
            raise CorpusUnavailable(f"shard {shard_id} labels are not a map: {type(labels)}")

        to_train, label_warnings = _training_set(chunks, labels, routing)
        warnings.extend(label_warnings)

        if not to_train:
            raise CorpusUnavailable(
                f"shard {shard_id} has no labels this node may train on"
                + (f" in domain {routing.partition}" if routing.is_campaign else "")
                + f" ({len(labels)} label(s) exist for the shard)"
            )

        if routing.n_labels is not None and len(to_train) < routing.n_labels:
            # The round says how many labels its schedule has placed in this
            # domain. Fewer locally means the pool and the label map disagree,
            # which is worth saying out loud rather than quietly training small.
            warnings.append(
                f"round expects {routing.n_labels} label(s) in this domain but only "
                f"{len(to_train)} of them match chunks in the shard"
            )

        corpus = Corpus(
            shard_id=shard_id,
            to_score=chunks,
            to_train=to_train,
            routing=routing,
            pool_size=len(chunks),
            n_labels_available=len(labels),
            n_dedup_dropped=n_duplicate,
            warnings=warnings,
        )
        LOG.info(
            "fetched shard %s: %d to score, %d labelled (%s)",
            shard_id,
            corpus.n_score,
            corpus.n_train,
            routing.describe(),
        )
        return corpus


def _as_chunks(payload: Any) -> tuple[list[Chunk], int, int]:
    """Turn the `chunks` array into `Chunk`s, dropping what cannot be scored.

    Returns the chunks, how many were duplicate ids, and how many were
    malformed. The two are counted apart because the first is `n_dedup_dropped`,
    a metric rounds ask for, and the second is a fault in the shard.
    """
    records = payload.get("chunks") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return [], 0, 0

    chunks: list[Chunk] = []
    seen: set[str] = set()
    n_duplicate = 0
    n_malformed = 0
    for record in records:
        if not isinstance(record, dict):
            n_malformed += 1
            continue
        chunk_id = record.get("chunk_id")
        text = record.get("text")
        # A malformed id cannot be matched back to the pool, and a duplicate one
        # in a submission is a 422, so neither is worth carrying any further.
        if (
            not isinstance(chunk_id, str)
            or not HASH_RE.match(chunk_id)
            or not isinstance(text, str)
            or not text.strip()
        ):
            n_malformed += 1
            continue
        if chunk_id in seen:
            n_duplicate += 1
            continue
        seen.add(chunk_id)
        chunks.append(Chunk(chunk_id=chunk_id, text=text))
    return chunks, n_duplicate, n_malformed


def _training_set(
    chunks: list[Chunk], labels: dict[str, Any], routing: Routing
) -> tuple[list[Chunk], list[str]]:
    """The labelled chunks this node is allowed to learn from.

    The label map is shard-wide, so in a campaign it has to be narrowed to this
    node's domain -- that narrowing is the whole difference between an expert
    and a node that has quietly trained on everybody's labels.
    """
    warnings: list[str] = []
    n_partitions = routing.n_partitions
    own = routing.partition
    restrict = routing.is_campaign and n_partitions is not None and own is not None

    to_train: list[Chunk] = []
    unusable = 0
    foreign = 0

    for chunk in chunks:
        raw = labels.get(chunk.chunk_id)
        if raw is None:
            continue
        if restrict and partition_of(chunk.chunk_id, n_partitions) != own:
            # `experts` sees the whole pool but may only train on its own
            # domain. In `sharded` the fetch already filtered, so this is free.
            foreign += 1
            continue
        label = _as_label(raw)
        if label is None:
            unusable += 1
            continue
        to_train.append(Chunk(chunk_id=chunk.chunk_id, text=chunk.text, label=label))

    if unusable:
        warnings.append(f"{unusable} label(s) were not numbers and were ignored")
    if foreign:
        LOG.debug("ignored %d label(s) outside domain %s", foreign, own)
    return to_train, warnings


def _as_label(value: Any) -> str | None:
    """Oracle labels are integers on the wire; fastText wants a label string.

    Kept as the integer's own text ("1", not "1.0") so that the core's numeric
    label handling -- which is what makes `expected_score` an ordinal quality
    score rather than a class index -- still recognises it.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else repr(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return str(int(number)) if number == int(number) else repr(number)
    return None
