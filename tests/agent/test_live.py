"""Tests that talk to the real control plane.

Opt-in: `NODE_AGENT_LIVE=1 uv run pytest -m integration`. A default run has to
work offline, on a machine that has never enrolled as a node.

Most of what is here is a *drift canary*. The agent is built against a contract
that someone else owns and can change, and the failure mode we care about is
discovering that five minutes into a round on demo day rather than here.

The tunnel URL is ephemeral. Override it with NODE_AGENT_SERVER when it moves.
"""

import os
import uuid

import pytest
import requests

from node.agent.config import DEFAULT_SERVER

pytestmark = pytest.mark.integration

SERVER = os.environ.get("NODE_AGENT_SERVER", DEFAULT_SERVER).rstrip("/")


@pytest.fixture(scope="module", autouse=True)
def require_live():
    if os.environ.get("NODE_AGENT_LIVE") != "1":
        pytest.skip("set NODE_AGENT_LIVE=1 to run live tests")
    try:
        requests.get(f"{SERVER}/health", timeout=10).raise_for_status()
    except requests.RequestException as error:
        pytest.skip(f"control plane unreachable at {SERVER}: {error}")


@pytest.fixture(scope="module")
def spec():
    response = requests.get(f"{SERVER}/openapi.json", timeout=20)
    response.raise_for_status()
    return response.json()


def schema(spec, name):
    return spec["components"]["schemas"][name]


def test_health():
    assert requests.get(f"{SERVER}/health", timeout=10).json() == {"status": "ok"}


def submit_schema(spec):
    return spec["paths"]["/tasks/{node_id}/{round_id}/submit"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"]


def test_submit_takes_exactly_four_keys_and_no_signature(spec):
    # Request signing was removed and never came back. A leftover `signature` is
    # a 422, so this is what stops us from re-adding one.
    submit = submit_schema(spec)
    assert set(submit["required"]) == {"node_id", "round_id", "scores", "agg_stats"}
    assert set(submit["properties"]) == {"node_id", "round_id", "scores", "agg_stats"}
    assert submit.get("additionalProperties") is False
    assert submit["properties"]["scores"]["maxItems"] == 200_000


def test_handshake_returns_no_credential(spec):
    response = schema(spec, "HandshakeResponse")
    assert set(response["required"]) == {"node_id", "heartbeat_interval_s", "tasks_url"}
    assert "node_secret" not in response["properties"]
    assert "auth_mode" not in response["properties"]


def test_handshake_still_refuses_to_hear_about_our_datasets(spec):
    """Contract 0.7.0 made the corpus server-held and dropped the key.

    The handshake is `additionalProperties: false`, so an agent that still sends
    `datasets` fails *enrolment* -- it never reaches a round, and nothing in its
    logs mentions datasets. That makes this the most expensive break to rediscover
    at runtime and the cheapest one to catch here.
    """
    handshake = schema(spec, "Handshake")
    assert handshake.get("additionalProperties") is False
    assert "datasets" not in handshake["properties"]
    assert set(handshake["properties"]) == {
        "name", "node_id", "hardware", "software", "model_kinds"
    }

    from node.agent.models import HandshakeRequest, Hardware

    body = HandshakeRequest(
        name=f"schema-canary-{uuid.uuid4().hex[:6]}",
        hardware=Hardware(cpu_cores=1, ram_gb=1.0),
        model_kinds=["classifier"],
    ).to_dict()
    assert set(body) <= set(handshake["properties"])


def test_chunk_score_shape_is_unchanged(spec):
    chunk = submit_schema(spec)["properties"]["scores"]["items"]
    assert set(chunk["required"]) == {"chunk_id", "score"}


def test_agg_stats_bounds_still_match_what_we_send(spec):
    agg = submit_schema(spec)["properties"]["agg_stats"]

    assert agg["required"] == ["n_chunks"]
    # We clamp eval_spearman to exactly this range; if it moves, the clamp
    # in node.core.metrics.spearman has to move with it.
    spearman = agg["properties"]["eval_spearman"]["anyOf"][0]
    assert (spearman["minimum"], spearman["maximum"]) == (-1.0, 1.0)
    # And this is why a non-positive proxy_lr goes back as null.
    assert agg["properties"]["proxy_lr"]["anyOf"][0]["exclusiveMinimum"] == 0


def test_task_view_field_names_are_unchanged(spec):
    task = schema(spec, "TaskView")
    assert set(task["required"]) == {
        "round_id",
        "status",
        "model",
        "operation",
        "dataset_id",
        "metrics",
        "params",
        "budget_k",
        "assigned_at",
        "accepted_at",
        "ack_url",
        "submit_url",
    }
    assert set(task["properties"]["status"]["enum"]) == {
        "assigned",
        "accepted",
        "submitted",
    }


def test_the_train_or_score_field_arrived_and_is_the_shape_we_obey(spec):
    """Contract 0.9.0 settled what this client spent three versions guessing.

    `operation.train` is a three-valued instruction, not a boolean -- which is
    why `resolve_mode` switched from inference to obedience and `--mode auto`
    became the default.
    """
    from node.agent.models import TRAIN_POLICIES

    operation = schema(spec, "TaskOperation")
    assert set(operation["properties"]["train"]["enum"]) == set(TRAIN_POLICIES)
    assert set(operation["properties"]) == {
        "train", "score", "input_checkpoint_id", "output_checkpoint_id"
    }
    assert operation.get("additionalProperties") is False
    # Tasks always produce scores in this contract; if that stops being const,
    # the runner has to learn to submit nothing.
    assert operation["properties"]["score"].get("const") is True


def test_the_heartbeat_stage_enum_is_the_one_we_send(spec):
    # A stage outside this set is a 422 on a heartbeat, which would make the node
    # look offline for a reason nothing in its own logs would explain.
    from node.agent.models import STAGES

    stage = schema(spec, "Heartbeat")["properties"]["stage"]
    assert set(stage["anyOf"][0]["enum"]) == set(STAGES)


def test_the_heartbeat_takes_exactly_the_keys_we_send(spec):
    heartbeat = schema(spec, "Heartbeat")
    assert heartbeat.get("additionalProperties") is False
    assert set(heartbeat["properties"]) == {"status", "stage", "round_id", "load"}


def test_the_campaign_train_mode_policy_exists_without_binding_the_node(spec):
    """`train_mode` is the server's policy for generating rounds.

    The node must not mirror it: each task's own `operation` is the authority,
    and a node that reimplemented this would disagree with the server the moment
    the policy changed.
    """
    train_mode = schema(spec, "CreateCampaign")["properties"]["train_mode"]
    assert set(train_mode["enum"]) == {"fresh", "continue"}



def test_nothing_is_gated_by_a_credential():
    # CONTRACT.md 2.1: no enrollment token, no node secret, no signing. An
    # earlier build did demand a token; this is what tells us if one returns.
    body = {
        "name": f"canary-{uuid.uuid4().hex[:6]}",
        "hardware": {"cpu_cores": 1, "ram_gb": 1},
        "model_kinds": ["classifier"],
    }
    response = requests.post(f"{SERVER}/nodes/handshake", json=body, timeout=20)
    assert response.status_code == 201, response.text
    enrolled = response.json()
    assert set(enrolled) == {"node_id", "heartbeat_interval_s", "tasks_url"}


def test_the_heldout_curve_keys_are_the_ones_we_send():
    """The server names them in its own rejection message.

    `HELDOUT_KEYS` is not in the OpenAPI document, so the only authority is the
    422 from a partial curve -- which lists all sixteen.
    """
    from node.agent.runner import HELDOUT_KEYS

    body = {
        "name": f"curve-canary-{uuid.uuid4().hex[:6]}",
        "hardware": {"cpu_cores": 1, "ram_gb": 1},
        "model_kinds": ["classifier"],
    }
    node_id = requests.post(f"{SERVER}/nodes/handshake", json=body, timeout=20).json()["node_id"]
    round_id = f"canary-{uuid.uuid4().hex[:8]}"
    requests.post(
        f"{SERVER}/rounds",
        json={
            "round_id": round_id,
            "budget_k": 1,
            "model": {"kind": "classifier", "id": "canary"},
            "metrics": ["eval_spearman"],
            "participants": [{"node_id": node_id, "dataset_id": "canary-set"}],
        },
        timeout=20,
    ).raise_for_status()

    # A deliberately partial curve, so the server enumerates what it wants.
    rejected = requests.post(
        f"{SERVER}/tasks/{node_id}/{round_id}/submit",
        json={
            "node_id": node_id,
            "round_id": round_id,
            "scores": [{"chunk_id": "a" * 64, "score": 1.0}],
            "agg_stats": {"n_chunks": 1, "eval_spearman": 0.5, "ho_n": 10, "ho_good": 3},
        },
        timeout=20,
    )
    assert rejected.status_code == 422
    message = str(rejected.json()["detail"])
    for key in HELDOUT_KEYS:
        assert f"'{key}'" in message, key


def test_a_null_extra_metric_is_refused():
    # Which is why the default policy omits a metric it cannot compute rather
    # than sending null for it.
    body = {
        "name": f"null-canary-{uuid.uuid4().hex[:6]}",
        "hardware": {"cpu_cores": 1, "ram_gb": 1},
        "model_kinds": ["classifier"],
    }
    node_id = requests.post(f"{SERVER}/nodes/handshake", json=body, timeout=20).json()["node_id"]
    round_id = f"null-canary-{uuid.uuid4().hex[:8]}"
    requests.post(
        f"{SERVER}/rounds",
        json={
            "round_id": round_id,
            "budget_k": 1,
            "model": {"kind": "classifier", "id": "canary"},
            "metrics": ["eval_spearman"],
            "participants": [{"node_id": node_id, "dataset_id": "canary-set"}],
        },
        timeout=20,
    ).raise_for_status()

    rejected = requests.post(
        f"{SERVER}/tasks/{node_id}/{round_id}/submit",
        json={
            "node_id": node_id,
            "round_id": round_id,
            "scores": [{"chunk_id": "a" * 64, "score": 1.0}],
            "agg_stats": {"n_chunks": 1, "eval_spearman": 0.5, "mystery": None},
        },
        timeout=20,
    )
    assert rejected.status_code == 422
    assert "must be a number" in str(rejected.json()["detail"])


def test_unknown_node_is_still_reported_the_way_we_detect_it():
    response = requests.get(f"{SERVER}/tasks/{uuid.uuid4().hex}", timeout=10)
    assert response.status_code == 404
    # api.UnknownNodeError keys off this exact phrase to trigger re-enrolment.
    assert "unknown node_id" in response.json()["detail"]


# -- server-held shards (contract 0.7.0) ----------------------------------


@pytest.fixture(scope="module")
def live_shard():
    """A small shard of our own on the live server, for the read canaries.

    Skipped rather than failed when the server accepts the upload but does not
    store it -- observed on this deployment as a `201` reporting
    `received: 40, new: 0, total: 0`, non-deterministically, for payloads that
    had just succeeded. That is a server-side defect, and these tests are drift
    canaries about the schema and the partition formula; reporting it as a client
    failure would point at the wrong thing.
    """
    import hashlib

    shard_id = f"canary-shard-{uuid.uuid4().hex[:8]}"
    chunks = []
    for i in range(40):
        text = f"canary document {i} about basalt and eruption {i}"
        chunks.append(
            {"chunk_id": hashlib.sha256(text.encode()).hexdigest(), "text": text}
        )
    response = requests.post(
        f"{SERVER}/shards/{shard_id}/chunks", json={"chunks": chunks}, timeout=30
    )
    response.raise_for_status()
    stored = response.json()

    served = requests.get(f"{SERVER}/shards/{shard_id}/chunks", timeout=30).json()
    if len(served.get("chunks", [])) != len(chunks):
        pytest.skip(
            f"server did not persist the canary shard: POST said {stored}, "
            f"GET returned {len(served.get('chunks', []))} of {len(chunks)} chunks"
        )
    return shard_id, chunks


def test_a_shard_read_has_the_shape_the_node_parses(live_shard):
    shard_id, chunks = live_shard
    payload = requests.get(f"{SERVER}/shards/{shard_id}/chunks", timeout=30).json()
    assert set(payload) == {"shard_id", "chunks"}
    assert len(payload["chunks"]) == len(chunks)
    assert set(payload["chunks"][0]) == {"chunk_id", "text"}


def test_the_partition_formula_is_the_one_we_compute_locally(live_shard):
    """`int(chunk_id[:8], 16) % n_partitions`, on both sides.

    The node filters the shard-wide label map to its own domain with this. If the
    server's placement ever diverged, an expert would train on the wrong domain
    and nothing would report an error.
    """
    from node.agent.shards import partition_of

    shard_id, chunks = live_shard
    for n_partitions in (2, 3):
        seen = set()
        for partition in range(n_partitions):
            served = requests.get(
                f"{SERVER}/shards/{shard_id}/chunks",
                params={"partition": partition, "n_partitions": n_partitions},
                timeout=30,
            ).json()["chunks"]
            ids = {c["chunk_id"] for c in served}
            assert ids == {
                c["chunk_id"]
                for c in chunks
                if partition_of(c["chunk_id"], n_partitions) == partition
            }
            assert not (ids & seen), "partitions overlap"
            seen |= ids
        # And together they are the whole shard, with nothing dropped.
        assert seen == {c["chunk_id"] for c in chunks}


def test_an_unknown_shard_is_empty_rather_than_a_404():
    """The trap this client has to handle explicitly.

    A missing workload looks exactly like a successful read of nothing, so
    emptiness is the only signal there is -- `shards.CorpusUnavailable` exists
    because of this line.
    """
    missing = f"no-such-shard-{uuid.uuid4().hex[:8]}"
    chunks = requests.get(f"{SERVER}/shards/{missing}/chunks", timeout=20)
    assert chunks.status_code == 200
    assert chunks.json()["chunks"] == []

    labels = requests.get(f"{SERVER}/shards/{missing}/labels", timeout=20)
    assert labels.status_code == 200
    assert labels.json()["labels"] == {}


def test_one_partition_parameter_without_the_other_is_refused(live_shard):
    # Which is why the client raises locally rather than sending a half-query.
    shard_id, _ = live_shard
    assert (
        requests.get(
            f"{SERVER}/shards/{shard_id}/chunks", params={"partition": 0}, timeout=20
        ).status_code
        == 422
    )
    assert (
        requests.get(
            f"{SERVER}/shards/{shard_id}/chunks",
            params={"partition": 2, "n_partitions": 2},
            timeout=20,
        ).status_code
        == 422
    )


def test_labels_are_integers_keyed_by_chunk_id(live_shard):
    shard_id, _ = live_shard
    payload = requests.get(f"{SERVER}/shards/{shard_id}/labels", timeout=30).json()
    assert set(payload) == {"shard_id", "labels"}
    for chunk_id, label in payload["labels"].items():
        assert isinstance(label, int) and not isinstance(label, bool)
        assert len(chunk_id) == 64


def test_a_round_still_refuses_a_non_numeric_param(live_shard):
    """Only a campaign may put a string `mode` in params.

    An operator round is numeric-only, so a client that tried to fabricate
    campaign routing through `POST /rounds` would be rejected -- the routing has
    to come from a real campaign.
    """
    shard_id, _ = live_shard
    body = {
        "name": f"param-canary-{uuid.uuid4().hex[:6]}",
        "hardware": {"cpu_cores": 1, "ram_gb": 1},
        "model_kinds": ["classifier"],
    }
    node_id = requests.post(
        f"{SERVER}/nodes/handshake", json=body, timeout=20
    ).json()["node_id"]

    rejected = requests.post(
        f"{SERVER}/rounds",
        json={
            "round_id": f"param-canary-{uuid.uuid4().hex[:8]}",
            "budget_k": 1,
            "model": {"kind": "classifier", "id": "canary"},
            "metrics": ["eval_spearman"],
            "params": {"mode": "sharded"},
            "participants": [{"node_id": node_id, "dataset_id": shard_id}],
        },
        timeout=20,
    )
    assert rejected.status_code == 422
    assert "must be a number" in str(rejected.json()["detail"])


def test_a_heartbeat_with_a_stage_and_gauges_is_accepted():
    """The telemetry the METRICS_GUIDE asks for, end to end.

    A 422 here would make a working node look offline, which is the kind of
    break that gets blamed on the network.
    """
    body = {
        "name": f"beat-canary-{uuid.uuid4().hex[:6]}",
        "hardware": {"cpu_cores": 1, "ram_gb": 1},
        "model_kinds": ["classifier"],
    }
    node_id = requests.post(
        f"{SERVER}/nodes/handshake", json=body, timeout=20
    ).json()["node_id"]

    accepted = requests.post(
        f"{SERVER}/nodes/{node_id}/heartbeat",
        json={
            "status": "busy",
            "stage": "training",
            "round_id": "canary-r1",
            "load": {
                "progress_pct": 42.5,
                "docs_processed": 8500,
                "docs_total": 20000,
                "docs_per_sec": 127.4,
                "eta_s": 90,
                "cpu_pct": 73.5,
                "ram_pct": 61.0,
            },
        },
        timeout=20,
    )
    assert accepted.status_code == 200, accepted.text
    assert set(accepted.json()) == {"ok", "next_heartbeat_s", "pending_tasks"}

    # An idle beat carries nothing over, and is equally acceptable.
    idle = requests.post(
        f"{SERVER}/nodes/{node_id}/heartbeat", json={"status": "idle"}, timeout=20
    )
    assert idle.status_code == 200, idle.text

    # And a stage outside the enum is refused, which is why we validate locally.
    assert (
        requests.post(
            f"{SERVER}/nodes/{node_id}/heartbeat",
            json={"status": "busy", "stage": "thinking"},
            timeout=20,
        ).status_code
        == 422
    )


def test_the_telemetry_history_the_node_can_read_back():
    history = requests.get(
        f"{SERVER}/telemetry/history", params={"limit": 5}, timeout=30
    )
    assert history.status_code == 200
    payload = history.json()
    assert isinstance(payload, dict)
