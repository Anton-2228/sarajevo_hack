import hashlib
import json
import threading

import pytest
from fastapi.testclient import TestClient

from control_plane.app import Settings, create_app
from node_sdk.client import ControlPlane, build_payload, heldout_curve

HW = {"cpu_cores": 8, "ram_gb": 32, "gpus": [{"model": "RTX 4090", "vram_gb": 24}]}


def h(i, node="n"):
    return hashlib.sha256(f"{node}:{i}".encode()).hexdigest()


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    def _make(**env):
        monkeypatch.setenv("PM_DB_PATH", str(tmp_path / "cp.sqlite3"))
        monkeypatch.setenv("PM_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return TestClient(create_app(Settings()))
    return _make


def enroll(c, name="bank-a", kinds=("classifier", "lora")):
    r = c.post("/nodes/handshake", json={
        "name": name, "hardware": HW, "model_kinds": list(kinds)})
    assert r.status_code == 201, r.text
    return r.json()


def make_round(c, rid, participants, metrics=("eval_spearman",), kind="classifier", budget_k=1000):
    return c.post("/rounds", json={
        "round_id": rid, "budget_k": budget_k, "model": {"kind": kind, "id": "quality-clf-v1"},
        "metrics": list(metrics), "params": {"proxy_lr": 1e-5},
        "participants": [{"node_id": n, "dataset_id": d} for n, d in participants]})


def setup_round(c, names, rid="r1", **round_kw):
    nodes = {name: enroll(c, name) for name in names}
    r = make_round(c, rid, [(n["node_id"], "ds1") for n in nodes.values()], **round_kw)
    assert r.status_code == 201, r.text
    return {name: n["node_id"] for name, n in nodes.items()}


def curve(kind="sharp", n=200, good_top=None):
    """Held-out curve stats. sharp: the 20% good docs rank on top; blunt: good docs spread evenly;
    good_top=m: the m best-ranked docs are the good ones."""
    if good_top is not None:
        good = [i < good_top for i in range(n)]
    else:
        good = [i < n // 5 for i in range(n)] if kind == "sharp" else [i % 5 == 0 for i in range(n)]
    return heldout_curve([n - i for i in range(n)], good)


def payload(node_id, rnd="r1", n=10, rho=0.95, offset=0.0, heldout="sharp", **extra):
    scores = [(h(i, node_id), i / n + offset) for i in range(n)]
    held = curve(heldout) if isinstance(heldout, str) else (heldout or {})
    stats = {"n_chunks": n, **held, **extra}
    if rho is not None:
        stats["eval_spearman"] = rho
    return build_payload(node_id, rnd, scores, stats)


def submit(c, p):
    return c.post(f"/tasks/{p['node_id']}/{p['round_id']}/submit", json=p)


def test_handshake_heartbeat_registry(make_client):
    c = make_client()
    info = enroll(c, "bank-a")
    assert info["node_id"].startswith("bank-a-")
    assert info["tasks_url"] == f"/tasks/{info['node_id']}"

    node = c.get("/nodes").json()[0]
    assert node["online"] is False
    assert node["specs"]["hardware"]["gpus"][0]["model"] == "RTX 4090"
    assert "datasets" not in node["specs"]
    assert c.post("/nodes/handshake", json={
        "name": "legacy", "hardware": HW, "model_kinds": ["classifier"], "datasets": [],
    }).status_code == 422

    r = c.post(f"/nodes/{info['node_id']}/heartbeat", json={"status": "idle", "load": {"cpu_pct": 12.5}})
    assert r.json() == {"ok": True, "next_heartbeat_s": 3.0, "pending_tasks": 0}
    assert c.get("/nodes").json()[0]["online"] is True
    assert c.post("/nodes/nope/heartbeat", json={}).status_code == 404
    assert c.post(f"/nodes/{info['node_id']}/heartbeat", json={"load": {"cpu": "high"}}).status_code == 422


def test_create_round_validates_participants(make_client):
    c = make_client()
    a = enroll(c, "bank-a", kinds=("lora",))["node_id"]
    assert make_round(c, "r1", [("ghost-000000", "ds1")]).status_code == 422
    assert make_round(c, "r1", [(a, "ds1")], kind="classifier").status_code == 422  # can't bake
    assert make_round(c, "r-other", [(a, "other-ds")], kind="lora").status_code == 201
    assert make_round(c, "r1", [(a, "ds1"), (a, "ds1")], kind="lora").status_code == 422
    assert make_round(c, "r1", [(a, "ds1")], kind="lora", metrics=()).status_code == 422
    assert make_round(c, "r1", [(a, "ds1")], kind="lora").status_code == 201
    assert make_round(c, "r1", [(a, "ds1")], kind="lora").status_code == 409


def test_task_lifecycle(make_client):
    c = make_client()
    a = enroll(c, "bank-a")["node_id"]
    b = enroll(c, "telco-b")["node_id"]
    assert make_round(c, "r1", [(a, "ds1")], metrics=("eval_spearman", "n_dedup_dropped")).status_code == 201

    tasks = c.get(f"/tasks/{a}").json()["tasks"]
    assert len(tasks) == 1
    t = tasks[0]
    assert (t["round_id"], t["status"], t["dataset_id"]) == ("r1", "assigned", "ds1")
    assert t["model"] == {"kind": "classifier", "id": "quality-clf-v1"}
    assert t["operation"] == {
        "train": "fresh", "score": True, "input_checkpoint_id": None,
        "output_checkpoint_id": "r1",
    }
    assert t["metrics"] == ["eval_spearman", "n_dedup_dropped"] and t["params"] == {"proxy_lr": 1e-5}
    assert t["submit_url"] == f"/tasks/{a}/r1/submit" and "deadline_at" not in t
    assert c.get(f"/tasks/{b}").json()["tasks"] == []
    assert c.post(f"/nodes/{a}/heartbeat", json={}).json()["pending_tasks"] == 1

    assert c.post(f"/tasks/{a}/r1/ack").json()["status"] == "accepted"
    assert c.post(f"/tasks/{a}/r1/ack").json()["status"] == "accepted"  # idempotent
    assert c.post(f"/tasks/{b}/r1/ack").status_code == 404

    assert submit(c, payload(a, n_dedup_dropped=3)).status_code == 201
    assert c.get(f"/tasks/{a}").json()["tasks"] == []
    parts = c.get("/rounds/r1").json()["participants"]
    assert parts[0]["status"] == "submitted" and parts[0]["accepted_at"] is not None

    assert make_round(c, "r2", [(a, "ds1")]).status_code == 201
    c.post("/rounds/r2/close")
    assert c.get(f"/tasks/{a}").json()["tasks"] == []
    assert c.post(f"/tasks/{a}/r2/ack").status_code == 409


def test_task_operation_validates_checkpoint_lifecycle(make_client):
    c = make_client()
    node_id = enroll(c, "bank-a")["node_id"]
    base = {
        "model": {"kind": "classifier", "id": "quality-clf-v1"},
        "metrics": ["n_chunks"],
        "participants": [{"node_id": node_id, "dataset_id": "ds1"}],
    }

    missing_input = c.post("/rounds", json={
        **base, "round_id": "bad-continue", "operation": {"train": "continue"},
    })
    assert missing_input.status_code == 422
    fresh_with_input = c.post("/rounds", json={
        **base, "round_id": "bad-fresh",
        "operation": {"train": "fresh", "input_checkpoint_id": "older"},
    })
    assert fresh_with_input.status_code == 422

    response = c.post("/rounds", json={
        **base, "round_id": "score-existing",
        "operation": {"train": "skip", "input_checkpoint_id": "checkpoint-v7"},
    })
    assert response.status_code == 201, response.text
    task = c.get(f"/tasks/{node_id}").json()["tasks"][0]
    assert task["operation"] == {
        "train": "skip", "score": True, "input_checkpoint_id": "checkpoint-v7",
        "output_checkpoint_id": None,
    }


def test_submit_requires_registration_assignment_and_metrics(make_client):
    c = make_client()
    ids = setup_round(c, ["bank-a"], metrics=("eval_spearman", "n_dedup_dropped"))
    a = ids["bank-a"]
    outsider = enroll(c, "telco-b")["node_id"]

    assert submit(c, payload("ghost-000000", n_dedup_dropped=1)).status_code == 404
    assert submit(c, payload(outsider, n_dedup_dropped=1)).status_code == 404
    assert submit(c, payload(a, rnd="r9", n_dedup_dropped=1)).status_code == 404
    p = payload(a, n_dedup_dropped=1)
    assert c.post(f"/tasks/{a}/other/submit", json=p).status_code == 400
    assert c.post(f"/tasks/{outsider}/r1/submit", json=p).status_code == 400
    r = submit(c, payload(a))
    assert r.status_code == 422 and "n_dedup_dropped" in r.text
    assert submit(c, p).status_code == 201
    c.post("/rounds/r1/close")
    assert submit(c, p).status_code == 409


def test_round_spec_has_no_deadline(make_client):
    c = make_client()
    a = enroll(c, "bank-a")["node_id"]
    body = {"round_id": "r1", "model": {"kind": "lora", "id": "l1"}, "metrics": ["n_chunks"],
            "participants": [{"node_id": a, "dataset_id": "ds1"}]}
    assert c.post("/rounds", json={**body, "deadline_at": 1789999999}).status_code == 422
    assert "deadline_at" not in c.post("/rounds", json=body).json()["spec"]


def test_removed_endpoints_are_gone(make_client):
    c = make_client()
    a = enroll(c, "bank-a")["node_id"]
    assert c.get("/status").status_code == 404
    assert c.get(f"/nodes/{a}").status_code in (404, 405)


def test_reliability_gate_uses_heldout_precision_at_cutoff(make_client):
    c = make_client()
    ids = setup_round(c, ["good", "bad", "mystery"], metrics=("n_chunks",))
    assert submit(c, payload(ids["good"])).json()["trust"] == "pending"
    assert submit(c, payload(ids["bad"], heldout="blunt", rho=0.99)).json()["trust"] == "pending"  # rho is not the gate
    assert submit(c, payload(ids["mystery"], heldout=None)).json()["trust"] == "unknown"

    top = c.get("/rounds/r1/topk", params={"k": 5}).json()
    assert top["normalize"] == "calibrated" and top["nodes_used"] == [ids["good"]]
    skipped = {n["node_id"]: n for n in top["nodes_skipped"]}
    assert skipped[ids["bad"]]["trust"] == "untrusted" and skipped[ids["bad"]]["precision_lower_bound"] < 0.2
    assert skipped[ids["mystery"]]["reason"] == "no held-out curve"
    assert all(item["node_id"] == ids["good"] for item in top["selected"])
    (gate,) = top["gate"]  # budget reallocated to the good node after the bad one was dropped
    assert gate["trust"] == "trusted" and gate["cutoff_q"] == 0.5 and gate["precision_lower_bound"] >= 0.2

    loose = c.get("/rounds/r1/topk", params={"k": 5, "include_untrusted": "true"}).json()
    assert {g["node_id"]: g["trust"] for g in loose["gate"]} == {ids["good"]: "trusted", ids["bad"]: "untrusted"}


@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("ho_good_q10"),                     # incomplete
    lambda s: s.update(ho_good_q05=s["ho_n_q05"] + 1),  # more good docs than docs in that top
    lambda s: s.update(ho_n_q20=s["ho_n_q10"] - 1),     # not cumulative
    lambda s: s.update(ho_n=1.5),                       # not a count
])
def test_heldout_curve_must_be_consistent_counts(make_client, mutate):
    c = make_client()
    a = setup_round(c, ["bank-a"])["bank-a"]
    stats = curve("sharp")
    mutate(stats)
    r = submit(c, payload(a, heldout=stats))
    assert r.status_code == 422 and "held-out curve" in r.text


def test_calibrated_pooling_gives_budget_to_the_node_with_more_good_docs(make_client):
    c = make_client()
    ids = setup_round(c, ["rich", "poor"])
    submit(c, payload(ids["rich"], heldout=curve(good_top=100)))  # half of held-out is good, ranked on top
    submit(c, payload(ids["poor"], heldout=curve(good_top=20)))   # a tenth is good
    top = c.get("/rounds/r1/topk", params={"k": 10}).json()
    per = top["selected_per_node"]
    assert per[ids["rich"]] > per.get(ids["poor"], 0) >= 1
    assert {g["trust"] for g in top["gate"]} == {"trusted"}
    z = c.get("/rounds/r1/topk", params={"k": 10, "normalize": "zscore", "include_untrusted": "true"}).json()
    assert z["selected_per_node"] == {ids["rich"]: 5, ids["poor"]: 5}


def test_participant_params_override_round_params(make_client):
    c = make_client()
    a = enroll(c, "bank-a")["node_id"]
    part = {"node_id": a, "dataset_id": "ds1", "params": {"cutoff_q": 0.07}}
    body = {"round_id": "r1", "model": {"kind": "classifier", "id": "m"}, "metrics": ["n_chunks"],
            "params": {"proxy_lr": 1e-5, "cutoff_q": 0.1}, "participants": [part]}
    bad = {**body, "round_id": "r0", "participants": [{**part, "params": {"cutoff_q": "top"}}]}
    assert c.post("/rounds", json=bad).status_code == 422
    assert c.post("/rounds", json=body).status_code == 201
    assert c.get(f"/tasks/{a}").json()["tasks"][0]["params"] == {"proxy_lr": 1e-5, "cutoff_q": 0.07}
    assert c.get("/rounds/r1").json()["participants"][0]["params"] == {"cutoff_q": 0.07}


@pytest.mark.parametrize("mutate, loc_part", [
    (lambda p: p.update(raw_text="Customer John Doe, IBAN DE89..."), "raw_text"),
    (lambda p: p["scores"][0].update(chunk_id="this is a sentence"), "chunk_id"),
    (lambda p: p["scores"][0].update(text="leak"), "text"),
    (lambda p: p["agg_stats"].update(sample="some raw text"), "agg_stats"),
    (lambda p: p["agg_stats"].update(nested={"a": 1}), "agg_stats"),
    (lambda p: p.update(scores=[]), "scores"),
])
def test_egress_contract_rejects_non_numeric_content(make_client, mutate, loc_part):
    c = make_client()
    a = setup_round(c, ["bank-a"])["bank-a"]
    p = payload(a)
    mutate(p)
    r = submit(c, p)
    assert r.status_code == 422
    assert loc_part in json.dumps(r.json()["detail"])
    assert "John Doe" not in r.text and "raw text" not in r.text  # values never echoed


def test_no_auth_anyone_can_act_as_any_node(make_client):
    # Hackathon build: no enrollment token, no per-node secret, no request signing (CONTRACT.md §2.1).
    # This test documents the consequence rather than hiding it: a node_id is enough to act as that
    # node, from any caller.
    c = make_client()
    a = enroll(c, "bank-a")["node_id"]
    assert c.get(f"/tasks/{a}").status_code == 200  # no token needed
    assert c.post(f"/nodes/{a}/heartbeat", json={}).status_code == 200  # no token needed

    # "someone else" re-handshakes as the same node_id with zero prior knowledge beyond it
    again = c.post("/nodes/handshake", json={"name": "impostor", "node_id": a, "hardware": HW,
                                             "model_kinds": ["classifier"]}).json()
    assert again["node_id"] == a  # accepted — no secret was ever required

    assert make_round(c, "r1", [(a, "ds1")]).status_code == 201
    assert submit(c, payload(a)).status_code == 201  # no signature field, none checked


def test_resubmit_replaces_and_bumps_revision(make_client):
    c = make_client()
    a = setup_round(c, ["bank-a"])["bank-a"]
    submit(c, payload(a, n=10))
    r = submit(c, payload(a, n=4))
    assert r.json()["revision"] == 2
    assert [n["n_scores"] for n in c.get("/rounds/r1").json()["nodes"]] == [4]
    # budget_k 1000 takes all 4 chunks: cutoff q=1, precision = base rate 20%, below the gate
    assert c.get("/rounds/r1/topk", params={"include_untrusted": "true"}).json()["pool_size"] == 4


def test_topk_normalizes_across_nodes(make_client):
    c = make_client()
    ids = setup_round(c, ["quiet", "loud"])
    # node "loud" has scores shifted by +100; without normalization it would take all of top-k
    submit(c, payload(ids["quiet"], n=10))
    submit(c, payload(ids["loud"], n=10, offset=100.0))
    loose = {"k": 10, "include_untrusted": "true"}
    raw = c.get("/rounds/r1/topk", params={**loose, "normalize": "none"}).json()
    assert raw["selected_per_node"] == {ids["loud"]: 10}
    for mode in ("calibrated", "zscore", "rank"):
        top = c.get("/rounds/r1/topk", params={**loose, "normalize": mode}).json()
        assert top["selected_per_node"] == {ids["quiet"]: 5, ids["loud"]: 5}, mode


def test_concurrent_heartbeats_and_task_polls(make_client):
    # regression: shared SQLite connection used from worker threads without the lock -> HTTP 500
    from concurrent.futures import ThreadPoolExecutor

    c = make_client()
    a = setup_round(c, ["bank-a"])["bank-a"]

    def hit(i):
        if i % 2:
            return c.post(f"/nodes/{a}/heartbeat", json={"status": "busy", "round_id": "r1"}).status_code
        return c.get(f"/tasks/{a}").status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(hit, range(400))) == {200}


def test_openapi_documents_submit_body(make_client):
    c = make_client()
    op = c.get("/openapi.json").json()["paths"]["/tasks/{node_id}/{round_id}/submit"]["post"]
    schema = op["requestBody"]["content"]["application/json"]["schema"]
    assert set(schema["properties"]) == {"node_id", "round_id", "scores", "agg_stats"}


def test_sdk_heartbeat_loop_adopts_server_interval():
    cp = ControlPlane("http://unused", state_file=None)
    second_beat = threading.Event()
    calls = []

    def heartbeat(**_state):
        calls.append(1)
        if len(calls) >= 2:
            second_beat.set()
        return {"next_heartbeat_s": 0.01}

    cp.heartbeat = heartbeat
    stop = cp.start_heartbeat()
    try:
        assert second_beat.wait(0.5)
        assert cp.heartbeat_interval_s == 0.01
    finally:
        stop.set()
