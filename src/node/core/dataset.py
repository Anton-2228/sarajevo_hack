"""Reading what the server sends and shaping it for fastText.

The wire format is not yet agreed with the server's author, so parsing accepts
a few plausible field names. When the contract is pinned down, the aliases
below are the only thing that needs to change.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

from node.core.text import normalize_label, normalize_text, to_fasttext_line
from node.core.types import LabeledSample, Sample

TEXT_FIELDS = ("text", "content", "document", "body")
ID_FIELDS = ("id", "sample_id", "doc_id", "uid")
LABEL_FIELDS = ("label", "score", "target", "class")


def _first_present(record: dict, names: Sequence[str]) -> str | None:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip() != "":
            return str(value)
    return None


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return records


def parse_labeled(records: Iterable[dict]) -> tuple[list[LabeledSample], int]:
    """Parse round-1 golden records. Returns the samples and a skipped count."""
    samples: list[LabeledSample] = []
    skipped = 0
    for position, record in enumerate(records):
        text = _first_present(record, TEXT_FIELDS)
        label = _first_present(record, LABEL_FIELDS)
        if text is None or label is None:
            skipped += 1
            continue
        samples.append(
            LabeledSample(
                sample_id=_first_present(record, ID_FIELDS) or f"idx-{position}",
                text=text,
                label=normalize_label(label),
            )
        )
    return samples, skipped


def parse_unlabeled(records: Iterable[dict]) -> tuple[list[Sample], int]:
    """Parse round-2/3 records. Returns the samples and a skipped count."""
    samples: list[Sample] = []
    skipped = 0
    for position, record in enumerate(records):
        text = _first_present(record, TEXT_FIELDS)
        if text is None:
            skipped += 1
            continue
        samples.append(
            Sample(
                sample_id=_first_present(record, ID_FIELDS) or f"idx-{position}",
                text=text,
            )
        )
    return samples, skipped


class WriteStats(NamedTuple):
    lines: int
    tokens: int


def write_training_file(samples: Sequence[LabeledSample], path: str | Path) -> WriteStats:
    """Write samples in fastText's supervised format.

    Samples whose text normalizes to nothing are dropped -- there is no signal
    in them -- so ``lines`` can be smaller than ``len(samples)``. The token
    count is returned because fastText's hashing bucket count has to be sized
    against it.
    """
    lines = 0
    tokens = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for sample in samples:
            text = normalize_text(sample.text)
            if not text:
                continue
            handle.write(to_fasttext_line(sample.label, text) + "\n")
            lines += 1
            tokens += text.count(" ") + 1
    return WriteStats(lines, tokens)


def stratified_split(
    samples: Sequence[LabeledSample], fraction: float, seed: int
) -> tuple[list[LabeledSample], list[LabeledSample]]:
    """Split by label so every class keeps representation on both sides.

    A class with a single sample stays entirely in train: holding it out would
    make it unlearnable and unmeasurable at the same time.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")

    grouped: dict[str, list[LabeledSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    train: list[LabeledSample] = []
    holdout: list[LabeledSample] = []

    for label in sorted(grouped):
        group = list(grouped[label])
        rng.shuffle(group)

        n_holdout = int(round(len(group) * fraction))
        if len(group) >= 2:
            # Guarantee presence on both sides for any class that can afford it.
            n_holdout = min(max(n_holdout, 1), len(group) - 1)
        else:
            n_holdout = 0

        holdout.extend(group[:n_holdout])
        train.extend(group[n_holdout:])

    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout
