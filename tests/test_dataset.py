"""Parsing what the server sends, and shaping it for fastText."""

import json
import re
from collections import Counter

import pytest

from node.core.dataset import (
    load_jsonl,
    parse_labeled,
    parse_unlabeled,
    stratified_split,
    write_training_file,
)
from node.core.types import LabeledSample

TRAINING_LINE = re.compile(r"^__label__\S+ \S.*$")


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return path


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"a": 1}\n\n\n{"a": 2}\n', encoding="utf-8")
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_load_jsonl_reports_the_offending_line(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        load_jsonl(path)


def test_field_aliases_are_accepted():
    # The wire format is not agreed yet, so parsing stays lenient.
    samples, skipped = parse_labeled(
        [
            {"id": "a", "text": "one", "label": "3"},
            {"doc_id": "b", "content": "two", "score": "4"},
        ]
    )
    assert skipped == 0
    assert [s.sample_id for s in samples] == ["a", "b"]
    assert [s.label for s in samples] == ["3", "4"]


def test_records_without_text_or_label_are_skipped():
    samples, skipped = parse_labeled(
        [{"text": "kept", "label": "1"}, {"text": "no label"}, {"label": "2"}]
    )
    assert len(samples) == 1
    assert skipped == 2


def test_missing_ids_get_positional_fallbacks():
    samples, _ = parse_unlabeled([{"text": "a"}, {"text": "b"}])
    assert [s.sample_id for s in samples] == ["idx-0", "idx-1"]


def test_training_file_format(tmp_path):
    samples = [
        LabeledSample("a", "First document.", "1"),
        LabeledSample("b", "Second\ndocument with a newline.", "2"),
    ]
    path = tmp_path / "train.txt"
    stats = write_training_file(samples, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert stats.lines == len(lines) == 2
    assert all(TRAINING_LINE.match(line) for line in lines)
    assert stats.tokens > 0


def test_empty_samples_never_reach_the_training_file(tmp_path):
    samples = [
        LabeledSample("a", "real content", "1"),
        LabeledSample("b", "   \n\t ", "2"),
        LabeledSample("c", "", "3"),
    ]
    path = tmp_path / "train.txt"
    stats = write_training_file(samples, path)

    assert stats.lines == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_stratified_split_keeps_every_class_on_both_sides():
    samples = [
        LabeledSample(f"{label}-{i}", "text here", str(label))
        for label in range(1, 11)
        for i in range(10)
    ]
    train, holdout = stratified_split(samples, fraction=0.2, seed=17)

    assert len(train) + len(holdout) == len(samples)
    assert set(Counter(s.label for s in train)) == {str(i) for i in range(1, 11)}
    assert set(Counter(s.label for s in holdout)) == {str(i) for i in range(1, 11)}
    assert all(count == 2 for count in Counter(s.label for s in holdout).values())


def test_singleton_class_stays_in_train():
    # Holding out the only example of a class makes it both unlearnable and
    # unmeasurable, which helps nobody.
    samples = [LabeledSample("solo", "text", "rare")] + [
        LabeledSample(f"c-{i}", "text", "common") for i in range(10)
    ]
    train, holdout = stratified_split(samples, fraction=0.5, seed=1)

    assert "rare" in {s.label for s in train}
    assert "rare" not in {s.label for s in holdout}


def test_split_is_deterministic_for_a_seed():
    samples = [
        LabeledSample(f"{label}-{i}", "text", str(label))
        for label in range(3)
        for i in range(8)
    ]
    first = stratified_split(samples, 0.25, seed=5)
    second = stratified_split(samples, 0.25, seed=5)
    assert [s.sample_id for s in first[0]] == [s.sample_id for s in second[0]]


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_degenerate_fractions(fraction):
    with pytest.raises(ValueError):
        stratified_split([LabeledSample("a", "t", "1")], fraction, seed=0)
