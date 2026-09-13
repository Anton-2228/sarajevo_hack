"""Local JSONL corpora, for the cases where the server is not the source.

Contract 0.7.0 moved the real thing: a round names a server-held shard and the
node reads the texts and labels out of `/shards/…` (see `shards`). Nothing here
is on that path any more, and in particular the node no longer tells the server
what it holds -- the handshake schema forbids the key.

What is left has two uses. `selftest` synthesizes a corpus and *uploads* it as a
shard, so the node can manufacture end-to-end work for itself; and the core's
own `node-clf` workflow reads JSONL directly.

A round is scored over *chunks*. One JSONL record is one chunk, identified by
the hash of its text rather than by any id the file happens to carry, so two
nodes holding the same document agree on its identity without coordinating --
which is also what makes a locally built shard's ids survive the round trip
through the server.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from node.agent.models import as_id
from node.core.dataset import LABEL_FIELDS, TEXT_FIELDS, load_jsonl
from node.core.types import LabeledSample, Sample

LOG = logging.getLogger("node.agent.datasets")


class UnknownDatasetError(LookupError):
    """The round named a dataset this node does not hold."""


def chunk_id_for(text: str) -> str:
    """The hash of the raw text, exactly as it was read.

    Not `normalize_text(text)`: normalization is a fastText implementation
    detail, and a LoRA node scoring the same corpus has to arrive at the same
    chunk ids without knowing anything about our tokenizer.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    label: str | None = None

    def as_sample(self) -> Sample:
        return Sample(sample_id=self.chunk_id, text=self.text)

    def as_labeled(self) -> LabeledSample:
        if self.label is None:
            raise ValueError(f"chunk {self.chunk_id[:12]} has no label")
        return LabeledSample(sample_id=self.chunk_id, text=self.text, label=self.label)


@dataclass(frozen=True)
class LocalDataset:
    dataset_id: str
    path: Path
    chunks: list[Chunk] = field(default_factory=list)
    domain: str | None = None
    language: str | None = None
    n_dedup_dropped: int = 0
    n_unparsed: int = 0

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def labeled(self) -> list[Chunk]:
        return [c for c in self.chunks if c.label is not None]

    def as_shard_chunks(self) -> list[dict[str, str]]:
        """Shaped for `POST /shards/{id}/chunks`. What selftest uploads."""
        return [{"chunk_id": c.chunk_id, "text": c.text} for c in self.chunks]


def _first(record: dict, names: tuple[str, ...]) -> object | None:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def load_dataset(path: Path, dataset_id: str | None = None) -> LocalDataset:
    """Read one JSONL file into chunks, deduplicating by content.

    Field names are read leniently through the core's aliases, so the agent
    inherits the same tolerance `node-clf` already documents.
    """
    records = load_jsonl(path)

    chunks: list[Chunk] = []
    seen: set[str] = set()
    dropped = 0
    unparsed = 0

    for record in records:
        raw_text = _first(record, TEXT_FIELDS)
        if not isinstance(raw_text, str) or not raw_text.strip():
            unparsed += 1
            continue

        # Duplicate chunk_ids in a submit would corrupt the server's pooled
        # ranking, so they are dropped here and counted -- `n_dedup_dropped` is
        # a metric the control plane's own example asks participants to report.
        digest = chunk_id_for(raw_text)
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)

        raw_label = _first(record, LABEL_FIELDS)
        label = None if raw_label is None else str(raw_label)
        chunks.append(Chunk(chunk_id=digest, text=raw_text, label=label))

    return LocalDataset(
        # A filename is not bound by the server's ID alphabet, and a dataset
        # whose id it rejects would fail the handshake for the whole node.
        dataset_id=dataset_id or as_id(path.stem),
        path=path,
        chunks=chunks,
        n_dedup_dropped=dropped,
        n_unparsed=unparsed,
    )


class DatasetRegistry:
    """Everything under one directory, keyed by filename stem."""

    def __init__(self, root: Path, *, allow_substitution: bool = True) -> None:
        self.root = Path(root)
        self.allow_substitution = allow_substitution
        self._cache: dict[str, LocalDataset] = {}
        self._scanned = False

    def discover(self, *, refresh: bool = False) -> list[LocalDataset]:
        if self._scanned and not refresh:
            return list(self._cache.values())

        self._cache.clear()
        if self.root.is_dir():
            for path in sorted(self.root.glob("*.jsonl")):
                try:
                    dataset = load_dataset(path)
                except (OSError, ValueError) as error:
                    LOG.warning("skipping %s: %s", path.name, error)
                    continue
                if dataset.n_chunks:
                    self._cache[dataset.dataset_id] = dataset
        self._scanned = True
        return list(self._cache.values())

    def get(self, dataset_id: str) -> tuple[LocalDataset, bool]:
        """Return the dataset and whether it is a stand-in for the one asked for."""
        self.discover()
        if dataset_id in self._cache:
            return self._cache[dataset_id], False

        if not self.allow_substitution:
            raise UnknownDatasetError(
                f"no local dataset {dataset_id!r}; have {sorted(self._cache) or 'none'}"
            )
        if not self._cache:
            raise UnknownDatasetError(f"no local datasets at all under {self.root}")

        # Only reachable off the round path now -- a server-held shard is never
        # substituted, because scoring chunk ids the control plane did not assign
        # corrupts a sharded campaign's merge for every other node.
        stand_in = max(self._cache.values(), key=lambda d: d.n_chunks)
        LOG.warning(
            "round asked for dataset %r which this node does not hold; "
            "substituting %r (%d chunks) -- provisioning is still mocked",
            dataset_id,
            stand_in.dataset_id,
            stand_in.n_chunks,
        )
        return stand_in, True



# -- the synthetic stand-in ------------------------------------------------
#
# Ten labels, each tied to its own vocabulary, mixed into shared filler. The
# signal is deliberately learnable: a poor score here means the pipeline is
# broken, not that the task is hard.

SAMPLES_PER_LABEL = 30
GOLDEN_SEED = 20260912

TOPICS: dict[str, list[str]] = {
    "1": ["volcano", "basalt", "eruption", "magma", "caldera"],
    "2": ["sonata", "orchestra", "violin", "conductor", "symphony"],
    "3": ["mortgage", "dividend", "portfolio", "annuity", "brokerage"],
    "4": ["antibiotic", "diagnosis", "surgeon", "vaccine", "clinic"],
    "5": ["compiler", "runtime", "kernel", "debugger", "toolchain"],
    "6": ["harvest", "irrigation", "orchard", "topsoil", "pasture"],
    "7": ["marathon", "striker", "referee", "stadium", "tournament"],
    "8": ["monsoon", "glacier", "sediment", "estuary", "tundra"],
    "9": ["parliament", "referendum", "treaty", "diplomat", "coalition"],
    "10": ["telescope", "asteroid", "nebula", "orbit", "spectrograph"],
}

FILLER = [
    "the", "report", "notes", "that", "a", "recent", "review", "of", "this",
    "subject", "was", "published", "last", "week", "and", "several", "readers",
    "asked", "for", "more", "detail", "about", "it", "which", "seems",
    "reasonable", "given", "how", "often", "the", "question", "comes", "up",
]

TEMPLATES = [
    "the {a} was discussed at length , and the {b} came up again .",
    "according to the summary , {a} and {b} remain the central concerns .",
    "readers asked about the {a} ; the answer involves the {b} as well .",
    "a short note on {a} , with some remarks on {b} at the end .",
    "the {a} matters here , though the {b} is what most people notice first .",
]


def synthesize_golden(
    out: Path, samples_per_label: int = SAMPLES_PER_LABEL, seed: int = GOLDEN_SEED
) -> Path:
    """Write a deterministic synthetic golden set.

    The order of random calls is load-bearing: it is what makes regenerating
    `data/tiny_golden.jsonl` reproduce the committed file byte for byte.
    """
    rng = random.Random(seed)
    records = []

    for label, markers in TOPICS.items():
        for index in range(samples_per_label):
            first, second = rng.sample(markers, 2)
            sentence = rng.choice(TEMPLATES).format(a=first, b=second)
            padding = " ".join(rng.choice(FILLER) for _ in range(rng.randint(8, 18)))
            records.append(
                {
                    "id": f"doc-{label}-{index:03d}",
                    "text": f"{sentence} {padding}",
                    "label": label,
                }
            )

    rng.shuffle(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out


def ensure_dataset(root: Path, dataset_id: str = "tiny_golden") -> LocalDataset:
    """Guarantee the registry has something to work with.

    Without this, a fresh checkout on a new machine enrols as a node that holds
    no data and can do nothing -- which is a confusing way to discover that a
    setup step was missed.
    """
    root = Path(root)
    existing = DatasetRegistry(root).discover()
    if existing:
        return max(existing, key=lambda d: d.n_chunks)

    path = root / f"{dataset_id}.jsonl"
    LOG.info("no local datasets under %s; synthesizing %s", root, path.name)
    synthesize_golden(path)
    return load_dataset(path, dataset_id)
