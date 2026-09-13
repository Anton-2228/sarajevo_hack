"""The control plane client, against a real server on a real socket."""

import pytest

from node.agent.api import (
    ApiError,
    ControlPlaneClient,
    TransientError,
    UnknownNodeError,
    ValidationRejected,
)
from node.agent.models import (
    Hardware,
    HandshakeRequest,
    HeartbeatRequest,
    SubmitPayload,
    ChunkScore,
)


@pytest.fixture
def client(fake_server):
    return ControlPlaneClient(fake_server.url, connect_timeout=2, read_timeout=5)


def enrol(client) -> str:
    response = client.handshake(
        HandshakeRequest(
            name="test", hardware=Hardware(cpu_cores=2, ram_gb=4.0), model_kinds=["classifier"]
        )
    )
    return response.node_id


def test_health_and_handshake(client, fake_server):
    assert client.health() == {"status": "ok"}
    node_id = enrol(client)
    assert node_id in fake_server.nodes



def test_tasks_ack_and_submit_round_trip(client, fake_server):
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1")

    tasks = client.get_tasks(node_id)
    assert [t.round_id for t in tasks] == ["r1"]

    acked = client.ack(node_id, "r1")
    assert acked.status == "accepted"

    payload = SubmitPayload(
        node_id=node_id,
        round_id="r1",
        scores=[ChunkScore("a" * 64, 7.0)],
        agg_stats={"n_chunks": 1, "eval_spearman": 0.5, "n_dedup_dropped": 0},
    )
    client.submit(node_id, "r1", payload.body())

    assert "r1" in fake_server.submissions
    # A submitted round drops off /tasks, exactly as the real server documents.
    assert client.get_tasks(node_id) == []


def test_ack_is_idempotent(client, fake_server):
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1")
    assert client.ack(node_id, "r1").status == "accepted"
    assert client.ack(node_id, "r1").status == "accepted"


def test_transient_failures_are_retried(client, fake_server):
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1")
    fake_server.script(
        f"/tasks/{node_id}", (503, {"detail": "busy"}), (502, {"detail": "gateway"})
    )

    tasks = client.get_tasks(node_id)
    assert [t.round_id for t in tasks] == ["r1"]
    assert sum(1 for m, p in fake_server.requests if p == f"/tasks/{node_id}") == 3


def test_retries_are_bounded(fake_server):
    client = ControlPlaneClient(fake_server.url, max_retries=1, read_timeout=5)
    node_id = enrol(client)
    fake_server.script(
        f"/tasks/{node_id}",
        (503, {"detail": "busy"}),
        (503, {"detail": "busy"}),
        (503, {"detail": "busy"}),
    )
    with pytest.raises(TransientError):
        client.get_tasks(node_id)


def test_client_errors_are_not_retried(client, fake_server):
    node_id = enrol(client)
    fake_server.script(f"/tasks/{node_id}", (400, {"detail": "malformed"}))
    with pytest.raises(ApiError) as caught:
        client.get_tasks(node_id)
    assert caught.value.status == 400
    # A 400 will be a 400 again; retrying it just wastes the server's time.
    assert sum(1 for m, p in fake_server.requests if p == f"/tasks/{node_id}") == 1


def test_unknown_node_is_its_own_error(client, fake_server):
    with pytest.raises(UnknownNodeError):
        client.get_tasks("never-enrolled")


def test_a_plain_404_does_not_look_like_a_lost_enrolment(client, fake_server):
    # Otherwise a mistyped base URL would trigger an endless re-handshake loop.
    with pytest.raises(ApiError) as caught:
        client.get_round("no-such-round")
    assert not isinstance(caught.value, UnknownNodeError)


def test_validation_errors_name_the_offending_fields(client, fake_server):
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1")
    with pytest.raises(ValidationRejected) as caught:
        # additionalProperties:false -- the pre-0.5.0 signature is now a 422.
        client.submit(node_id, "r1", {"node_id": node_id, "round_id": "r1",
                                      "scores": [], "agg_stats": {}, "signature": "x"})
    assert "signature" in caught.value.fields



def test_resubmitting_into_an_open_round_replaces_and_bumps_the_revision(
    client, fake_server
):
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1", metrics=[])
    payload = SubmitPayload(node_id, "r1", [ChunkScore("a" * 64, 1.0)], {"n_chunks": 1})

    first = client.submit(node_id, "r1", payload.body())
    second = client.submit(node_id, "r1", payload.body())
    assert first["revision"] == 1
    assert second["revision"] == 2


def test_a_closed_round_is_a_permanent_refusal(client, fake_server):
    from node.agent.api import RoundClosed

    node_id = enrol(client)
    task = fake_server.add_task(node_id, "r1", metrics=[])
    task["closed"] = True
    payload = SubmitPayload(node_id, "r1", [ChunkScore("a" * 64, 1.0)], {"n_chunks": 1})
    with pytest.raises(RoundClosed):
        client.submit(node_id, "r1", payload.body())


def test_submit_is_not_blindly_resent_after_a_lost_response(client, fake_server):
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1")
    payload = SubmitPayload(node_id, "r1", [ChunkScore("a" * 64, 1.0)], {"n_chunks": 1})
    signed = payload.body()

    # The submit lands, then the connection drops before we hear about it.
    fake_server.tasks["r1"]["status"] = "submitted"
    fake_server.submissions["r1"] = signed
    fake_server.script(f"/tasks/{node_id}/r1/submit", (503, {"detail": "gone"}))

    assert client.submit(node_id, "r1", signed) == {}
    submits = [p for m, p in fake_server.requests if p.endswith("/r1/submit")]
    # Exactly one attempt: the ambiguity was resolved by asking, not resending.
    assert len(submits) == 1


def test_the_client_cannot_close_a_round(client, fake_server):
    # Requirement: never close a round. Enforced by not having the capability.
    assert not hasattr(client, "close_round")
    assert not any("close" in name for name in dir(client))


def test_nothing_the_client_does_touches_close(client, fake_server):
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1", metrics=[])
    client.get_tasks(node_id)
    client.ack(node_id, "r1")
    payload = SubmitPayload(node_id, "r1", [ChunkScore("a" * 64, 1.0)], {"n_chunks": 1})
    client.submit(node_id, "r1", payload.body())
    client.get_round("r1")

    assert fake_server.closed_rounds() == []


def test_heartbeat_reports_pending_work(client, fake_server):
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1")
    response = client.heartbeat(node_id, HeartbeatRequest(status="idle", load={"cpu_pct": 1.0}))
    assert response.ok is True
    assert response.pending_tasks == 1




def test_clone_is_a_separate_session(client, fake_server):
    # requests.Session is not documented thread-safe, and the heartbeat thread
    # runs alongside a submit that can take minutes.
    enrol(client)
    twin = client.clone()
    assert twin._session is not client._session
    assert twin.base_url == client.base_url


def test_outgoing_nan_fails_locally(client, fake_server):
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1")
    # requests' own encoder would emit a bare NaN token and fail remotely with
    # something unreadable.
    with pytest.raises(ValueError):
        client.submit(
            node_id, "r1", {"node_id": node_id, "round_id": "r1", "x": float("nan")}
        )


def test_read_timeout_is_retried(fake_server):
    client = ControlPlaneClient(fake_server.url, read_timeout=0.2, max_retries=3)
    node_id = enrol(client)
    fake_server.add_task(node_id, "r1")
    fake_server.delays[f"/tasks/{node_id}"] = 0.5

    import threading

    def clear_delay():
        fake_server.delays.pop(f"/tasks/{node_id}", None)

    threading.Timer(0.8, clear_delay).start()
    assert [t.round_id for t in client.get_tasks(node_id)] == ["r1"]
