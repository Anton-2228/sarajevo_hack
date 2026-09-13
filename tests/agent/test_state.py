"""Identity, journal and outbox -- what survives a restart."""

import json

import pytest

from node.agent import state as state_mod
from node.agent.state import AgentState, Identity, TaskRecord, safe_name


@pytest.fixture
def state(tmp_path):
    return AgentState(tmp_path / "state")


def test_safe_name_passes_ordinary_ids_through():
    assert safe_name("r1") == "r1"
    assert safe_name("quality-clf-v1") == "quality-clf-v1"


@pytest.mark.parametrize("reserved", ["con", "CON", "nul", "com1", "LPT9", "aux.txt"])
def test_safe_name_defuses_windows_device_names(reserved):
    # These are not creatable as files on Windows at all, in any directory.
    assert safe_name(reserved).startswith("_")


@pytest.mark.parametrize(
    "hostile", ["../../etc/passwd", "..", ".", "..\\..\\windows", "/absolute", "a/b"]
)
def test_safe_name_yields_one_harmless_path_component(hostile):
    # round_id comes off the wire and becomes a directory name. The property
    # that matters is that the result addresses exactly one name inside the
    # state dir, and is never a relative-path element.
    from pathlib import Path, PurePosixPath, PureWindowsPath

    name = safe_name(hostile)
    assert name not in (".", "..")
    assert PurePosixPath(name).name == name
    assert PureWindowsPath(name).name == name
    assert len(Path(name).parts) == 1


def test_safe_name_defuses_illegal_characters_and_trailing_dots():
    assert ":" not in safe_name("a:b")
    assert not safe_name("trailing.").endswith(".")
    assert not safe_name("trailing ").endswith(" ")
    assert safe_name("") == "_empty"
    assert safe_name("...") == "_empty"


def test_identity_round_trips(state):
    identity = Identity(node_id="n1", server_url="https://example.test", name="node-x")
    state.save_identity(identity)
    loaded = state.load_identity("https://example.test")
    assert loaded is not None
    assert loaded.node_id == "n1"
    assert loaded.name == "node-x"


def test_identity_is_not_reused_across_servers(state):
    state.save_identity(Identity(node_id="n1", server_url="https://a.test", name="x"))
    # A different control plane means a different enrolment; reusing the id
    # would just produce confusing 404s.
    assert state.load_identity("https://b.test") is None
    assert state.load_identity("https://a.test") is not None




def test_corrupt_identity_reads_as_absent_not_as_a_crash(state, tmp_path):
    state.root.mkdir(parents=True)
    (state.root / "identity.json").write_text("{not json", encoding="utf-8")
    assert state.load_identity() is None


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "x.json"
    state_mod.atomic_write_text(target, '{"a": 1}')
    state_mod.atomic_write_text(target, '{"a": 2}')
    assert json.loads(target.read_text()) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_journal_records_and_updates(state):
    state.record(TaskRecord(round_id="r1", phase="acked"))
    assert state.get_record("r1").phase == "acked"

    state.record(TaskRecord(round_id="r1", phase="submitted", attempts=1))
    assert state.get_record("r1").phase == "submitted"
    assert len(state.journal()) == 1

    state.forget("r1")
    assert state.get_record("r1") is None


def test_payload_survives_a_crash_before_sending(state):
    # The reason the stash exists: training is the expensive step, and a
    # finished payload on disk means a crash never costs a retrain.
    body = {"node_id": "n1", "round_id": "r1", "scores": [], "agg_stats": {"n_chunks": 0}}
    state.stash_payload("r1", body)

    reopened = AgentState(state.root)
    assert reopened.load_payload("r1") == body

    reopened.drop_payload("r1")
    assert reopened.load_payload("r1") is None


def test_payload_with_nan_is_refused_locally(state):
    # A bare NaN is invalid JSON; failing here beats an opaque remote parse error.
    with pytest.raises(ValueError):
        state.stash_payload("r1", {"agg_stats": {"eval_spearman": float("nan")}})


def test_model_dir_is_keyed_safely(state):
    assert state.model_dir("quality-clf-v1").name == "quality-clf-v1"
    assert state.model_dir("con").name == "_con"


def test_record_stamps_the_current_time(state):
    # Callers do not get to choose: the journal's timestamps are what prune
    # trusts, so they always reflect when the write actually happened.
    state.record(TaskRecord(round_id="r1", phase="acked", updated_at=0.0))
    assert state.get_record("r1").updated_at > 0.0


def test_prune_drops_only_old_submitted_rounds(state, tmp_path):
    state.record(TaskRecord(round_id="old", phase="submitted"))
    state.record(TaskRecord(round_id="fresh", phase="submitted"))
    state.record(TaskRecord(round_id="broken", phase="failed"))

    # Age two entries the way fourteen days would.
    journal = json.loads((state.root / "journal.json").read_text())
    journal["old"]["updated_at"] = 0.0
    journal["broken"]["updated_at"] = 0.0
    (state.root / "journal.json").write_text(json.dumps(journal))

    assert state.prune(keep_days=14) == 1
    # The failed round stays: it is evidence, not clutter.
    assert set(state.journal()) == {"fresh", "broken"}


def test_describe_reports_the_identity(state):
    # Contract 0.5.0 has no node secret, so there is nothing here to hide --
    # the node_id is public through GET /nodes anyway.
    state.save_identity(Identity(node_id="n1", server_url="s", name="x"))
    described = json.dumps(state.describe())
    assert "n1" in described
