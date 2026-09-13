"""The wire types, checked against the control plane's own documented examples.

The payloads below are copied verbatim from the server's openapi.json. That is
the point: when its author renames a field, this fails here rather than five
minutes into a round on demo day.
"""

import json

import pytest

from node.agent import models
from node.agent.models import (
    GPU,
    ChunkScore,
    Hardware,
    HandshakeRequest,
    HandshakeResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    ModelRef,
    Routing,
    SubmitPayload,
    TaskOperation,
    TaskView,
    resolve_mode,
)

# openapi.json -> components.schemas.Handshake.examples[0], contract 0.7.0.
# Note what is absent: `datasets`. The schema is additionalProperties:false and
# no longer declares the key, so sending it fails enrolment with a 422.
HANDSHAKE_EXAMPLE = {
    "hardware": {
        "cpu_cores": 16,
        "disk_free_gb": 500,
        "gpus": [{"model": "A100", "vram_gb": 80}],
        "ram_gb": 64,
    },
    "model_kinds": ["classifier", "lora"],
    "name": "bank-a",
    "software": {"agent_version": "0.2.0", "python": "3.12"},
}

TASK_EXAMPLE = {
    "round_id": "r1",
    "status": "assigned",
    "model": {"kind": "classifier", "id": "quality-clf-v1"},
    "dataset_id": "support-tickets-2025",
    "metrics": ["eval_spearman", "n_dedup_dropped"],
    "params": {"proxy_lr": 1e-05, "cutoff_q": 0.1},
    "budget_k": 1000,
    "assigned_at": 1757000000.0,
    "accepted_at": None,
    "ack_url": "/tasks/bank-a-3f9c1d/r1/ack",
    "submit_url": "/tasks/bank-a-3f9c1d/r1/submit",
}

# CONTRACT.md 3.3.1: a campaign round carries routing inside params, with `mode`
# as a string and the rest as whole numbers.
CAMPAIGN_PARAMS = {
    "proxy_lr": 1e-05,
    "mode": "sharded",
    "partition": 0,
    "n_partitions": 2,
    "n_labels": 500,
}


def test_handshake_request_reproduces_the_documented_example():
    request = HandshakeRequest(
        name="bank-a",
        hardware=Hardware(
            cpu_cores=16,
            ram_gb=64,
            gpus=[GPU(model="A100", vram_gb=80)],
            disk_free_gb=500,
        ),
        model_kinds=["classifier", "lora"],
        software={"agent_version": "0.2.0", "python": "3.12"},
    )
    assert request.to_dict() == HANDSHAKE_EXAMPLE


def test_handshake_omits_absent_optionals():
    # Handshake is additionalProperties:false, so the smallest correct body is
    # the safest one -- and node_id present-but-null would read as a
    # re-registration request.
    body = HandshakeRequest(
        name="n", hardware=Hardware(cpu_cores=1, ram_gb=1.0), model_kinds=["classifier"]
    ).to_dict()
    assert "node_id" not in body
    assert "disk_free_gb" not in body["hardware"]


def test_handshake_never_mentions_a_dataset():
    """Verified against the live server: the key is a 422, not an ignored extra.

    It is the hardest kind of break to read at runtime -- enrolment fails, so
    the node never reaches a round and nothing in the logs mentions datasets.
    """
    body = HandshakeRequest(
        name="bank-a",
        hardware=Hardware(cpu_cores=1, ram_gb=1.0),
        model_kinds=["classifier"],
    ).to_dict()
    assert set(body) <= {"name", "node_id", "hardware", "software", "model_kinds"}
    assert "datasets" not in body
    assert not hasattr(HandshakeRequest, "datasets")


def test_task_view_parses_the_full_shape():
    task = TaskView.from_dict(TASK_EXAMPLE)
    assert task.round_id == "r1"
    assert task.status == "assigned"
    assert task.model == ModelRef(kind="classifier", id="quality-clf-v1")
    assert task.dataset_id == "support-tickets-2025"
    assert task.metrics == ["eval_spearman", "n_dedup_dropped"]
    assert task.params == {"proxy_lr": 1e-05, "cutoff_q": 0.1}
    assert task.budget_k == 1000
    assert task.accepted_at is None
    assert task.extra == {}


def test_task_view_keeps_fields_we_do_not_model():
    # This is how the future "train or just score" field survives the trip.
    task = TaskView.from_dict({**TASK_EXAMPLE, "stage": "score_only", "priority": 7})
    assert task.extra["stage"] == "score_only"
    assert task.extra["priority"] == 7


def test_task_view_tolerates_a_null_model():
    task = TaskView.from_dict({**TASK_EXAMPLE, "model": None})
    assert task.model is None


def test_non_numeric_params_are_quarantined_not_fatal():
    # TaskView.params is typed as numbers, but a round is created with
    # additionalProperties:true, so a string can genuinely arrive here.
    task = TaskView.from_dict(
        {**TASK_EXAMPLE, "params": {"proxy_lr": 0.5, "tag": "fast", "flag": True}}
    )
    assert task.params == {"proxy_lr": 0.5}
    assert task.extra["params_nonnumeric"] == {"tag": "fast", "flag": True}


def test_a_campaigns_string_mode_is_not_a_fault():
    # Contract 0.7.0 puts a string `mode` in params by design. Reporting it as a
    # quarantined oddity would be calling the contract a bug.
    task = TaskView.from_dict({**TASK_EXAMPLE, "params": CAMPAIGN_PARAMS})
    assert "params_nonnumeric" not in task.extra
    assert task.raw_params["mode"] == "sharded"
    # Still absent from the numeric view, which is what may be echoed into
    # agg_stats -- and agg_stats takes numbers only.
    assert "mode" not in task.params
    assert task.params["partition"] == 0.0


def test_numeric_strings_in_params_are_accepted():
    task = TaskView.from_dict({**TASK_EXAMPLE, "params": {"proxy_lr": "0.25"}})
    assert task.params == {"proxy_lr": 0.25}


def test_handshake_response_is_just_an_identity():
    # Contract 0.5.0 removed node_secret and auth_mode along with all auth.
    response = HandshakeResponse.from_dict(
        {
            "node_id": "bank-a-3f9c1d",
            "heartbeat_interval_s": 15.0,
            "tasks_url": "/tasks/bank-a-3f9c1d",
        }
    )
    assert response.node_id == "bank-a-3f9c1d"
    assert response.heartbeat_interval_s == 15.0
    assert not hasattr(response, "node_secret")


def test_heartbeat_round_trip():
    body = HeartbeatRequest(
        status="busy", round_id="r1", stage="training", load={"cpu_pct": 41.5}
    ).to_dict()
    assert body == {
        "status": "busy",
        "round_id": "r1",
        "stage": "training",
        "load": {"cpu_pct": 41.5},
    }

    # An idle heartbeat has no round and no stage, and both are omitted rather
    # than sent as null -- the schema is additionalProperties:false.
    idle = HeartbeatRequest().to_dict()
    assert "round_id" not in idle
    assert "stage" not in idle

    response = HeartbeatResponse.from_dict(
        {"ok": True, "next_heartbeat_s": 20.0, "pending_tasks": 2}
    )
    assert (response.ok, response.next_heartbeat_s, response.pending_tasks) == (True, 20.0, 2)


def test_submit_body_is_exactly_the_four_contract_keys():
    payload = SubmitPayload(
        node_id="n1",
        round_id="r1",
        scores=[ChunkScore("abc", 7.5)],
        agg_stats={"n_chunks": 1, "eval_spearman": None},
    )
    body = payload.body()
    # additionalProperties:false, and the pre-0.5.0 signature is now a 422.
    assert set(body) == {"node_id", "round_id", "scores", "agg_stats"}
    assert body["scores"] == [{"chunk_id": "abc", "score": 7.5}]
    # eval_spearman is one of the two keys the schema declares nullable.
    assert body["agg_stats"]["eval_spearman"] is None


def test_submit_body_is_json_serializable_without_nan():
    payload = SubmitPayload(
        node_id="n1", round_id="r1", scores=[ChunkScore("a", 1.0)], agg_stats={"n_chunks": 1}
    )
    assert json.loads(json.dumps(payload.body(), allow_nan=False))


@pytest.mark.parametrize("forced", ["train", "auto"])
def test_todays_tasks_always_train(forced):
    # Requirement for now: always train. Under `auto` that has to fall out of
    # the data rather than being hardcoded, because no field says otherwise yet.
    mode = resolve_mode(TaskView.from_dict(TASK_EXAMPLE), forced)
    assert mode.train is True


def test_mode_forced_to_score():
    mode = resolve_mode(TaskView.from_dict(TASK_EXAMPLE), "score")
    assert mode.train is False


@pytest.mark.parametrize(
    "train, trains",
    [("fresh", True), ("continue", True), ("skip", False)],
)
def test_auto_obeys_the_operation_it_was_given(train, trains):
    """The field this client spent three contracts anticipating finally exists.

    It is an instruction, so `auto` follows it and nothing else -- no inference
    from round_id, model.id, n_labels or what is on disk.
    """
    task = TaskView.from_dict({**TASK_EXAMPLE, "operation": {"train": train}})
    mode = resolve_mode(task, "auto")
    assert mode.train is trains
    assert mode.policy == train
    assert train in mode.source


@pytest.mark.parametrize("campaign_mode", ["sharded", "experts"])
def test_a_campaign_mode_is_never_read_as_score_only(campaign_mode):
    """`params.mode` is the campaign mode and says nothing about training.

    An expert trains every round. Reading this as "just score" would run a
    campaign against a model the node may never have built.
    """
    task = TaskView.from_dict(
        {**TASK_EXAMPLE, "params": {**CAMPAIGN_PARAMS, "mode": campaign_mode}}
    )
    assert resolve_mode(task, "auto").train is True


def test_the_heuristics_the_metrics_guide_forbids_are_gone():
    # The guide is explicit: do not key off these. A server that still sends one
    # must not change what the node does -- `operation` is the only authority.
    task = TaskView.from_dict(
        {
            **TASK_EXAMPLE,
            "operation": {"train": "fresh"},
            "task_type": "inference",
            "params": {"score_only": 1, "train": 0, "n_labels": 500},
        }
    )
    assert resolve_mode(task, "auto").train is True


def test_max_scores_matches_the_schema_cap():
    assert models.MAX_SCORES == 200_000


# -- routing (contract 0.7.0) ----------------------------------------------


def test_routing_is_parsed_out_of_campaign_params():
    task = TaskView.from_dict({**TASK_EXAMPLE, "params": CAMPAIGN_PARAMS})
    assert task.routing == Routing(
        mode="sharded", partition=0, n_partitions=2, n_labels=500
    )
    assert task.routing.is_campaign
    assert task.routing_problems == []


def test_a_plain_round_has_no_routing_and_takes_the_whole_shard():
    task = TaskView.from_dict(TASK_EXAMPLE)
    assert task.routing.is_campaign is False
    assert task.routing.scores_whole_pool is True
    assert task.routing_problems == []


def test_sharded_scores_one_slice_and_experts_the_whole_pool():
    # The difference the server enforces: `advance` rejects an expert whose
    # submission misses pool chunks, and concatenates sharded rankings as-is.
    assert Routing(mode="sharded", partition=0, n_partitions=2).scores_whole_pool is False
    assert Routing(mode="experts", partition=0, n_partitions=2).scores_whole_pool is True


def test_whole_number_floats_are_accepted_as_partitions():
    # JSON numbers arrive as floats; 2.0 is the same partition as 2.
    routing, problems = Routing.from_params(
        {"mode": "experts", "partition": 1.0, "n_partitions": 4.0, "n_labels": 10.0}
    )
    assert problems == []
    assert (routing.partition, routing.n_partitions, routing.n_labels) == (1, 4, 10)


@pytest.mark.parametrize(
    "params, complaint",
    [
        ({"mode": "sharded"}, "needs both partition and n_partitions"),
        ({"mode": "shardedd", "partition": 0, "n_partitions": 2}, "unknown campaign mode"),
        ({"mode": "experts", "partition": 2, "n_partitions": 2}, "out of range"),
        ({"mode": "experts", "partition": -1, "n_partitions": 2}, "must not be negative"),
        ({"mode": "experts", "partition": 0, "n_partitions": 0}, "at least 1"),
        ({"mode": "sharded", "partition": 0.5, "n_partitions": 2}, "not a whole number"),
        ({"mode": "sharded", "partition": "x", "n_partitions": 2}, "not a number"),
    ],
)
def test_unusable_routing_is_reported_rather_than_guessed(params, complaint):
    # Guessing a partition would submit scores for documents this node was never
    # assigned, which in a sharded campaign corrupts every other node's share.
    _, problems = Routing.from_params(params)
    assert any(complaint in p for p in problems), problems


# -- the model lifecycle instruction (contract 0.9.0) ----------------------


def test_operation_parses_the_documented_shape():
    task = TaskView.from_dict(
        {
            **TASK_EXAMPLE,
            "operation": {
                "train": "continue",
                "score": True,
                "input_checkpoint_id": "c1-r1",
                "output_checkpoint_id": "c1-r2",
            },
        }
    )
    assert task.operation == TaskOperation(
        train="continue",
        score=True,
        input_checkpoint_id="c1-r1",
        output_checkpoint_id="c1-r2",
    )
    assert task.operation.trains is True
    assert task.operation.continues is True
    assert task.operation.needs_input_checkpoint is True


def test_a_task_without_an_operation_trains_fresh():
    # A pre-0.9.0 server, or any task missing the field: training is the only
    # reading that does not depend on something already existing on disk.
    task = TaskView.from_dict(TASK_EXAMPLE)
    assert task.operation.train == "fresh"
    assert task.operation.trains is True
    assert task.operation.needs_input_checkpoint is False


def test_skip_needs_a_checkpoint_and_does_not_train():
    operation = TaskOperation.from_dict(
        {"train": "skip", "input_checkpoint_id": "c1-r1"}
    )
    assert operation.trains is False
    assert operation.needs_input_checkpoint is True


def test_an_unknown_train_policy_falls_back_to_fresh_and_says_so():
    # Never guess a lifecycle: fresh is the one policy that needs nothing to
    # already exist, and the unrecognised name is kept so it can be reported.
    operation = TaskOperation.from_dict({"train": "distill"})
    assert operation.train == "fresh"
    assert operation.unknown_train == "distill"

    task = TaskView.from_dict({**TASK_EXAMPLE, "operation": {"train": "distill"}})
    mode = resolve_mode(task, "auto")
    assert mode.train is True
    assert "distill" in mode.source and "unknown" in mode.source


def test_operation_is_modelled_not_quarantined_as_an_unknown_field():
    task = TaskView.from_dict({**TASK_EXAMPLE, "operation": {"train": "skip"}})
    assert "operation" not in task.extra


def test_forced_modes_keep_the_checkpoint_names():
    # An operator override changes the policy, not the lineage: the server tracks
    # checkpoints by these names either way.
    operation = {
        "train": "continue",
        "input_checkpoint_id": "in",
        "output_checkpoint_id": "out",
    }
    task = TaskView.from_dict({**TASK_EXAMPLE, "operation": operation})

    forced_train = resolve_mode(task, "train")
    assert (forced_train.train, forced_train.policy) == (True, "fresh")
    assert forced_train.output_checkpoint_id == "out"

    forced_score = resolve_mode(task, "score")
    assert (forced_score.train, forced_score.policy) == (False, "skip")
    assert forced_score.input_checkpoint_id == "in"


def test_the_stage_vocabulary_is_the_servers():
    # The enum is closed: anything else is a 422 on a heartbeat, which would be
    # an absurd way to look offline.
    assert models.STAGES == ("downloading", "training", "scoring", "uploading")
