"""Local datasets -- the half of the protocol the server never sees."""

import hashlib
import json
from pathlib import Path

import pytest

from node.agent import datasets
from node.agent.datasets import DatasetRegistry, UnknownDatasetError, chunk_id_for

REPO_GOLDEN = Path(__file__).resolve().parents[2] / "data" / "tiny_golden.jsonl"


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return path


def test_chunk_id_is_sha256_of_the_raw_text():
    # Pinned against a literal, because this identity has to agree with nodes
    # that share none of our code.
    text = "the volcano was discussed at length"
    assert chunk_id_for(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_chunk_id_does_not_normalize():
    # Normalization is a fastText detail. A LoRA node hashing the same document
    # must land on the same id without knowing our tokenizer.
    assert chunk_id_for("Hello, World") != chunk_id_for("hello , world")


def test_load_reads_field_aliases(tmp_path):
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [
            {"id": "a", "text": "alpha text", "label": "3"},
            {"doc_id": "b", "content": "beta text", "score": "7"},
            {"uid": "c", "body": "gamma text"},
        ],
    )
    dataset = datasets.load_dataset(path)
    assert dataset.n_chunks == 3
    assert {c.label for c in dataset.chunks} == {"3", "7", None}
    assert len(dataset.labeled) == 2


def test_duplicate_chunks_are_dropped_and_counted(tmp_path):
    # Duplicate chunk_ids in a submit would corrupt the server's pooled
    # ranking, and n_dedup_dropped is a metric its own example asks for.
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [
            {"id": "a", "text": "same text", "label": "1"},
            {"id": "b", "text": "same text", "label": "2"},
            {"id": "c", "text": "other text", "label": "3"},
        ],
    )
    dataset = datasets.load_dataset(path)
    assert dataset.n_chunks == 2
    assert dataset.n_dedup_dropped == 1


def test_records_without_usable_text_are_counted_not_fatal(tmp_path):
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [{"id": "a", "text": "real"}, {"id": "b"}, {"id": "c", "text": "   "}],
    )
    dataset = datasets.load_dataset(path)
    assert dataset.n_chunks == 1
    assert dataset.n_unparsed == 2


def test_a_local_corpus_is_shaped_for_shard_upload(tmp_path):
    """Contract 0.7.0 left this module one job on the wire: seeding a shard.

    The node no longer tells the server what it holds -- the handshake forbids
    the key -- so the only way local text reaches the control plane is as an
    explicit upload, which is what `selftest` does.
    """
    path = write_jsonl(tmp_path / "d.jsonl", [{"text": "a customer record"}])
    chunks = datasets.load_dataset(path).as_shard_chunks()
    assert [sorted(c) for c in chunks] == [["chunk_id", "text"]]
    # The id is the content hash, so it survives the round trip through the
    # server unchanged -- the node reads back the ids it uploaded.
    assert chunks[0]["chunk_id"] == datasets.chunk_id_for("a customer record")


def test_registry_keys_datasets_by_filename_stem(tmp_path):
    write_jsonl(tmp_path / "alpha.jsonl", [{"text": "one"}])
    write_jsonl(tmp_path / "beta.jsonl", [{"text": "two"}, {"text": "three"}])

    registry = DatasetRegistry(tmp_path)
    assert {d.dataset_id for d in registry.discover()} == {"alpha", "beta"}

    dataset, substituted = registry.get("beta")
    assert dataset.n_chunks == 2
    assert substituted is False


def test_substitution_picks_the_largest_and_says_so(tmp_path):
    write_jsonl(tmp_path / "small.jsonl", [{"text": "one"}])
    write_jsonl(tmp_path / "big.jsonl", [{"text": "a"}, {"text": "b"}, {"text": "c"}])

    dataset, substituted = DatasetRegistry(tmp_path).get("not-provisioned-yet")
    assert dataset.dataset_id == "big"
    assert substituted is True


def test_substitution_can_be_refused(tmp_path):
    write_jsonl(tmp_path / "small.jsonl", [{"text": "one"}])
    registry = DatasetRegistry(tmp_path, allow_substitution=False)
    with pytest.raises(UnknownDatasetError, match="not-provisioned"):
        registry.get("not-provisioned")


def test_empty_registry_always_raises(tmp_path):
    with pytest.raises(UnknownDatasetError, match="no local datasets"):
        DatasetRegistry(tmp_path).get("anything")


def test_synthesize_reproduces_the_committed_golden_set(tmp_path):
    # The generator moved out of scripts/ into the package. This is the proof
    # the move changed nothing: byte for byte, or it is a different dataset.
    regenerated = datasets.synthesize_golden(tmp_path / "tiny_golden.jsonl")
    assert regenerated.read_bytes() == REPO_GOLDEN.read_bytes()


def test_synthesized_set_is_balanced_and_learnable(tmp_path):
    dataset = datasets.load_dataset(datasets.synthesize_golden(tmp_path / "g.jsonl"))
    labels = [c.label for c in dataset.labeled]
    assert len(dataset.chunks) == 300
    assert set(labels) == {str(i) for i in range(1, 11)}
    assert all(labels.count(str(i)) == 30 for i in range(1, 11))


def test_ensure_dataset_bootstraps_an_empty_machine(tmp_path):
    # A fresh checkout must not enrol as a node that holds nothing.
    dataset = datasets.ensure_dataset(tmp_path / "datasets")
    assert dataset.n_chunks == 300
    assert (tmp_path / "datasets" / "tiny_golden.jsonl").is_file()


def test_ensure_dataset_leaves_existing_data_alone(tmp_path):
    write_jsonl(tmp_path / "mine.jsonl", [{"text": "one", "label": "1"}])
    dataset = datasets.ensure_dataset(tmp_path)
    assert dataset.dataset_id == "mine"
    assert not (tmp_path / "tiny_golden.jsonl").exists()
