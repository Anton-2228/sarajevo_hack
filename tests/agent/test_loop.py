"""The agent loop, end to end against a real (fake) control plane."""

import json
import math
import threading
from dataclasses import replace

import pytest

from node.agent import runner as runner_mod
from node.agent.config import AgentConfig
from node.agent.loop import AgentLoop
from node.agent.reporting import NullReporter
from node.agent.state import (
    PHASE_FAILED,
    PHASE_READY,
    PHASE_SUBMITTED,
    AgentState,
    TaskRecord,
)
from node.core.types import ScoredSample, TrainReport

LABELS = [str(i) for i in range(1, 11)]


class StubClassifier:
    def __init__(self):
        self.labels = list(LABELS)

    def score(self, samples):
        return [
            ScoredSample(
                sample_id=s.sample_id,
                label="5",
                probs={label: 0.1 for label in self.labels},
                entropy=1.0,
                margin=0.0,
                expected_score=5.0 + (i % 5),
            )
            for i, s in enumerate(samples)
        ]

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        return directory


@pytest.fixture
def stub_training(monkeypatch):
    """Real everything, except the minutes fastText would take.

    `load` is stubbed alongside `train` because contract 0.9.0's `skip` and
    `continue` policies load a checkpoint, and the stub's `save` writes only a
    directory -- the real loader would rightly refuse it.
    """
    calls = {"n": 0, "loaded": 0}

    def fake_load(directory):
        calls["loaded"] += 1
        return StubClassifier()

    monkeypatch.setattr(runner_mod.FastTextClassifier, "load", staticmethod(fake_load))

    def fake_train(samples, config=None):
        calls["n"] += 1
        report = TrainReport(
            n_samples=len(samples),
            n_skipped=0,
            n_train=len(samples),
            n_eval=0,
            labels=list(LABELS),
            autotuned=False,
            accuracy=0.9,
            macro_f1=0.9,
            mae=0.2,
            qwk=0.9,
        )
        return StubClassifier(), report

    monkeypatch.setattr(runner_mod.FastTextClassifier, "train", staticmethod(fake_train))
    return calls


@pytest.fixture
def make_loop(tmp_path, fake_server):
    def build(**overrides):
        settings = {
            "server_url": fake_server.url,
            "name": "test-node",
            "cores": 1,
            "ram_gb": 1.0,
            "state_dir": tmp_path / "state",
            "poll_interval_s": 0.05,
            "heartbeat_interval_s": 0.05,
            "max_retries": 1,
            "once": True,
        }
        settings.update(overrides)
        return AgentLoop(AgentConfig(**settings), reporter=NullReporter())

    return build


def test_a_complete_round(make_loop, fake_server, stub_training):
    loop = make_loop()
    node_id = "test-node-fake"
    fake_server.add_task(node_id, "r1")

    assert loop.run() == 0

    assert "r1" in fake_server.submissions
    submitted = fake_server.submissions["r1"]
    assert submitted["node_id"] == node_id
    assert submitted["round_id"] == "r1"
    # n_chunks counts the corpus and must be at least the number submitted.
    assert submitted["agg_stats"]["n_chunks"] >= len(submitted["scores"])
    # Exactly the contract's four keys: a leftover signature is now a 422.
    assert set(submitted) == {"node_id", "round_id", "scores", "agg_stats"}


def test_every_required_metric_is_present(make_loop, fake_server, stub_training):
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1", metrics=["eval_spearman", "n_dedup_dropped"])
    loop.run()
    stats = fake_server.submissions["r1"]["agg_stats"]
    assert "eval_spearman" in stats
    assert "n_dedup_dropped" in stats


def test_the_agent_acks_before_it_submits(make_loop, fake_server, stub_training):
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1")
    loop.run()
    paths = [p for m, p in fake_server.requests]
    assert paths.index("/tasks/test-node-fake/r1/ack") < paths.index(
        "/tasks/test-node-fake/r1/submit"
    )


def test_rounds_are_handled_strictly_one_at_a_time(make_loop, fake_server, stub_training):
    loop = make_loop(once=False, max_tasks=2)
    fake_server.add_task("test-node-fake", "r1", assigned_at=1.0)
    fake_server.add_task("test-node-fake", "r2", assigned_at=2.0)

    assert loop.run() == 0

    paths = [p for m, p in fake_server.requests]
    # Oldest first, and r2 is not touched until r1 is delivered.
    assert paths.index("/tasks/test-node-fake/r1/submit") < paths.index(
        "/tasks/test-node-fake/r2/ack"
    )


def test_the_agent_never_closes_a_round(make_loop, fake_server, stub_training):
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1")
    loop.run()
    assert fake_server.closed_rounds() == []


def test_already_submitted_rounds_are_not_redone(make_loop, fake_server, stub_training):
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1", status="submitted")
    assert loop.run() == 0
    assert stub_training["n"] == 0


def test_handshake_declares_only_what_the_node_can_do(make_loop, fake_server, stub_training):
    loop = make_loop()
    loop.run()
    node = fake_server.nodes["test-node-fake"]
    assert node["name"] == "test-node"
    # No GPU is claimed: there is no torch here, so a lora round would fail.
    assert loop._hardware.gpus == []


def test_dry_run_touches_nothing(make_loop, fake_server, stub_training):
    loop = make_loop(dry_run=True)
    fake_server.add_task("test-node-fake", "r1")
    loop.run()

    assert fake_server.submissions == {}
    paths = [p for m, p in fake_server.requests]
    assert "/tasks/test-node-fake/r1/ack" not in paths
    # The pipeline did run, though -- that is the point of a dry run.
    assert stub_training["n"] == 1


def test_a_lost_enrolment_is_recovered(make_loop, fake_server, stub_training, tmp_path):
    # Write an identity for a node the server has never heard of.
    from node.agent.state import Identity

    AgentState(tmp_path / "state").save_identity(
        Identity(node_id="ghost-node", server_url=fake_server.url, name="test-node")
    )
    loop = make_loop()
    fake_server.add_task("ghost-node", "r1")

    assert loop.run() == 0
    # Re-registered under the stored id, so the round addressed to it survives.
    assert loop._identity.node_id == "ghost-node"
    assert "r1" in fake_server.submissions


def test_a_crash_before_sending_costs_no_retraining(
    make_loop, fake_server, stub_training, tmp_path
):
    # Exactly the state a crash in that window leaves behind.
    state = AgentState(tmp_path / "state")
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1")
    loop._enrol()

    from node.agent.models import ChunkScore, SubmitPayload

    payload = SubmitPayload(
        "test-node-fake", "r1", [ChunkScore("a" * 64, 5.0)],
        {"n_chunks": 1, "eval_spearman": 0.5, "n_dedup_dropped": 0},
    )
    state.stash_payload("r1", payload.body())
    state.record(TaskRecord(round_id="r1", phase=PHASE_READY))

    resumed = make_loop()
    assert resumed.run() == 0

    assert "r1" in fake_server.submissions
    # The whole point: the payload was already paid for.
    assert stub_training["n"] == 0
    assert state.get_record("r1").phase == PHASE_SUBMITTED
    assert state.load_payload("r1") is None


def test_a_permanently_broken_round_is_not_retried_forever(
    make_loop, fake_server, stub_training, tmp_path
):
    state = AgentState(tmp_path / "state")
    state.record(TaskRecord(round_id="r1", phase=PHASE_FAILED, attempts=3))

    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1")
    assert loop.run() == 0

    assert stub_training["n"] == 0
    assert "r1" not in fake_server.submissions


def test_once_exits_even_when_every_round_is_skipped(
    make_loop, fake_server, stub_training, tmp_path
):
    # The round is offered but skipped, so --once must still terminate.
    # Without this the agent polls the same dead round forever.
    AgentState(tmp_path / "state").record(
        TaskRecord(round_id="r1", phase=PHASE_FAILED, attempts=3)
    )
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1")
    assert loop.run() == 0


def test_once_exits_when_there_is_no_work(make_loop, fake_server, stub_training):
    assert make_loop().run() == 0


def test_an_empty_shard_fails_the_round_without_killing_the_agent(
    make_loop, fake_server, stub_training, tmp_path
):
    # An unknown shard answers 200 with an empty list, so the node has to treat
    # emptiness as the failure. Nothing is substituted: scoring chunk ids the
    # control plane never assigned would corrupt a sharded campaign's merge.
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1", dataset_id="never-loaded", seed=False)

    assert loop.run() == 0
    assert "r1" not in fake_server.submissions
    assert loop.view.failed == 1
    record = AgentState(tmp_path / "state").get_record("r1")
    # Permanent: retrying cannot conjure a corpus.
    assert record.phase == PHASE_FAILED
    assert record.attempts >= 3


def test_the_agent_survives_a_poll_failure(make_loop, fake_server, stub_training):
    loop = make_loop(once=False, max_tasks=1)
    fake_server.add_task("test-node-fake", "r1")
    fake_server.script("/tasks/test-node-fake", (500, {"detail": "boom"}))

    assert loop.run() == 0
    assert "r1" in fake_server.submissions


def test_a_metric_this_node_cannot_compute_is_retried_as_zero(
    make_loop, fake_server, stub_training
):
    # Default policy omits it, the server answers missing_metrics, and only
    # then do we send a number we do not believe -- loudly.
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1", metrics=["mystery_metric"])

    assert loop.run() == 0
    assert fake_server.submissions["r1"]["agg_stats"]["mystery_metric"] == 0.0
    # Two attempts: the honest one, then the one the server insisted on.
    submits = [p for m, p in fake_server.requests if p.endswith("/r1/submit")]
    assert len(submits) == 2


def test_a_closed_round_is_failed_permanently(make_loop, fake_server, stub_training, tmp_path):
    loop = make_loop()
    task = fake_server.add_task("test-node-fake", "r1")
    task["closed"] = True

    assert loop.run() == 0
    record = AgentState(tmp_path / "state").get_record("r1")
    # There is no reopening a round, so retrying would only burn the machine.
    assert record.phase == PHASE_FAILED
    assert record.attempts >= 3


def test_the_heldout_curve_reaches_the_server(make_loop, fake_server, stub_training):
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1")
    loop.run()

    from node.agent.runner import HELDOUT_KEYS

    agg = fake_server.submissions["r1"]["agg_stats"]
    assert set(HELDOUT_KEYS) <= set(agg)
    # The fake applies the same validation the live server does, verified
    # against it, so acceptance means the curve is well formed.
    assert agg["ho_n"] >= 1


def test_stopping_between_rounds_exits_cleanly(make_loop, fake_server, stub_training):
    loop = make_loop(once=False)
    fake_server.add_task("test-node-fake", "r1")

    def stop_soon():
        loop.request_stop()

    threading.Timer(0.4, stop_soon).start()
    assert loop.run() == 0
    assert "r1" in fake_server.submissions





def test_an_unreachable_server_fails_fast(make_loop, tmp_path):
    loop = make_loop()
    loop.config = replace(loop.config, server_url="http://127.0.0.1:1")
    loop._client.base_url = "http://127.0.0.1:1"
    assert loop.run() == 1


def test_heartbeats_keep_flowing(make_loop, fake_server, stub_training):
    loop = make_loop(once=False, max_tasks=1)
    fake_server.add_task("test-node-fake", "r1")
    loop.run()
    assert fake_server.heartbeat_count >= 1


def test_an_explicit_interval_is_respected_and_the_servers_is_floored(make_loop):
    # A control plane asking for a 1 ms heartbeat must not talk a fleet of nodes
    # into hammering it. An interval the operator set is their own call.
    from node.agent.loop import MIN_SERVER_HEARTBEAT_S, Heartbeater

    loop = make_loop()
    beat = Heartbeater(
        loop._client,
        "n1",
        loop._telemetry,
        interval_s=0.05,
        wake=threading.Event(),
        reenrol=threading.Event(),
    )
    assert beat._interval == 0.05
    assert MIN_SERVER_HEARTBEAT_S >= 1.0


# -- telemetry (contract 0.9.0 / METRICS_GUIDE) ---------------------------


def test_heartbeats_carry_the_stage_and_round_of_the_active_task(
    make_loop, fake_server, stub_training
):
    loop = make_loop(once=False, max_tasks=1)
    fake_server.add_task("test-node-fake", "r1")
    loop.run()

    busy = [b for b in fake_server.heartbeats if b.get("status") == "busy"]
    assert busy, "no busy heartbeat reached the server"
    assert all(b.get("round_id") == "r1" for b in busy)
    # The fake rejects anything outside the server's enum, so reaching here at
    # all proves the stage names are the contract's.
    assert {b.get("stage") for b in busy} <= {
        "downloading", "training", "scoring", "uploading"
    }


def test_the_last_heartbeat_of_a_round_is_idle_and_carries_nothing_over(
    make_loop, fake_server, stub_training
):
    # METRICS_GUIDE: finishing a task means idle, with no round, no stage and
    # none of the finished task's gauges still attached.
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1")
    loop.run()

    idle = [b for b in fake_server.heartbeats if b.get("status") == "idle"]
    assert idle
    last = idle[-1]
    assert last.get("round_id") is None
    assert last.get("stage") is None
    assert last.get("load") in ({}, None)


def test_every_load_gauge_the_server_saw_was_a_finite_number(
    make_loop, fake_server, stub_training
):
    # The fake validates this the way the live server does; a 422 here would
    # mean the node had silently dropped off the mesh over telemetry.
    loop = make_loop(once=False, max_tasks=1)
    fake_server.add_task("test-node-fake", "r1")
    loop.run()

    assert fake_server.heartbeats
    for beat in fake_server.heartbeats:
        load = beat.get("load") or {}
        assert len(load) <= 32
        for key, value in load.items():
            assert isinstance(value, (int, float)) and not isinstance(value, bool), key
            assert math.isfinite(value), key


def test_the_checkpoint_a_round_produced_is_journalled(
    make_loop, fake_server, stub_training, tmp_path
):
    # So a restart can tell "this round already trained" from "never started".
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1")
    loop.run()

    record = AgentState(tmp_path / "state").get_record("r1")
    assert record.train_policy == "fresh"
    assert record.checkpoint_id == "r1-ckpt"


def test_a_skip_round_is_driven_by_the_operation_not_a_heuristic(
    make_loop, fake_server, stub_training, monkeypatch
):
    loop = make_loop()
    # Round one trains and leaves a checkpoint behind.
    fake_server.add_task("test-node-fake", "r1")
    loop.run()
    assert stub_training["n"] == 1

    # Round two is told to score from it and must not train again.
    loop2 = make_loop()
    fake_server.tasks.clear()
    fake_server.add_task(
        "test-node-fake",
        "r2",
        operation={
            "train": "skip",
            "score": True,
            "input_checkpoint_id": "r1-ckpt",
            "output_checkpoint_id": "r1-ckpt",
        },
    )
    loop2.run()

    assert "r2" in fake_server.submissions
    assert stub_training["n"] == 1, "a skip round retrained"
    assert stub_training["loaded"] == 1
    stats = fake_server.submissions["r2"]["agg_stats"]
    assert stats["train_policy_skip"] == 1.0
    assert stats["checkpoint_loaded"] == 1.0
    # And it still submits scores: operation.score is always true.
    assert fake_server.submissions["r2"]["scores"]


def test_the_submitted_payload_is_json_clean(make_loop, fake_server, stub_training):
    loop = make_loop()
    fake_server.add_task("test-node-fake", "r1")
    loop.run()
    assert json.loads(json.dumps(fake_server.submissions["r1"], allow_nan=False))
