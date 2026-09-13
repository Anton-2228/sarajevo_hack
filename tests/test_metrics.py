"""GET /metrics: Prometheus text over live store state — nodes, rounds, submissions, campaigns, shards."""
import hashlib

import pytest
from fastapi.testclient import TestClient

from control_plane.app import Settings, create_app
from node_sdk.client import build_payload


def parse(text):
    """{metric_name: [(labels_dict, value), ...]}"""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_labels, value = line.rsplit(" ", 1)
        if "{" in name_labels:
            name, tags = name_labels[:-1].split("{", 1)
            labels = dict(kv.split("=", 1) for kv in _split_tags(tags))
            labels = {k: v.strip('"') for k, v in labels.items()}
        else:
            name, labels = name_labels, {}
        out.setdefault(name, []).append((labels, float(value)))
    return out


def _split_tags(tags):
    # tags are like: a="1",b="two,three" — split on commas outside quotes
    parts, buf, in_quotes = [], "", False
    for ch in tags:
        if ch == '"':
            in_quotes = not in_quotes
        if ch == "," and not in_quotes:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        parts.append(buf)
    return parts


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PM_DB_PATH", str(tmp_path / "cp.sqlite3"))
    monkeypatch.setenv("PM_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    return TestClient(create_app(Settings()))


def test_metrics_is_prometheus_text_and_starts_empty(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    m = parse(r.text)
    assert m["proxy_mesh_nodes_total"] == [({}, 0.0)]


def test_metrics_reflects_node_round_and_submission(client):
    c = client
    node = c.post("/nodes/handshake", json={"name": "bank-a", "hardware": {"cpu_cores": 4, "ram_gb": 8},
                                            "model_kinds": ["classifier"]}).json()
    a = node["node_id"]
    heartbeat_load = {
        "progress_pct": 42.5, "docs_processed": 8500, "docs_total": 20000,
        "docs_per_sec": 127.4, "cpu_pct": 73.5, "custom_temperature": 61.2,
    }
    c.post(f"/nodes/{a}/heartbeat", json={
        "status": "busy", "stage": "training", "round_id": "r1", "load": heartbeat_load,
    })
    c.post("/rounds", json={"round_id": "r1", "model": {"kind": "classifier", "id": "m"},
                            "metrics": ["custom_metric"],
                            "participants": [{"node_id": a, "dataset_id": "ds1"}]})

    scores = [(hashlib.sha256(f"c{i}".encode()).hexdigest(), i / 10) for i in range(10)]
    payload = build_payload(a, "r1", scores, {"n_chunks": 10, "custom_metric": 42.5})
    c.post(f"/tasks/{a}/r1/submit", json=payload)

    m = parse(c.get("/metrics").text)
    assert m["proxy_mesh_nodes_total"] == [({}, 1.0)]
    assert m["proxy_mesh_node_online"] == [({"node_id": a, "name": "bank-a"}, 1.0)]
    assert m["proxy_mesh_node_busy"] == [({"node_id": a, "name": "bank-a"}, 1.0)]
    assert m["proxy_mesh_node_stage"] == [(
        {"node_id": a, "name": "bank-a", "stage": "training"}, 1.0,
    )]
    load = {lab["metric"]: value for lab, value in m["proxy_mesh_node_load"]
            if lab["node_id"] == a}
    assert load == heartbeat_load
    assert ({"status": "open"}, 1.0) in m["proxy_mesh_rounds_total"]
    assert ({"round_id": "r1", "status": "submitted"}, 1.0) in m["proxy_mesh_round_participants"]

    # the demo node's own custom_metric shows up as its own labelled series, no server allowlist
    custom = {(lab["metric"]): val for lab, val in m["proxy_mesh_submission_metric"]
             if lab["round_id"] == "r1" and lab["node_id"] == a}
    assert custom["custom_metric"] == 42.5
    assert custom["n_chunks"] == 10

    snapshot = c.get("/metrics.json").json()["nodes"][0]
    assert snapshot["status"] == "busy"
    assert snapshot["stage"] == "training"
    assert snapshot["round_id"] == "r1"
    assert snapshot["load"] == heartbeat_load
    assert snapshot["heartbeat_age_s"] >= 0


def test_metrics_reflects_campaign_and_shard_progress(client):
    c = client
    chunks = [{"chunk_id": hashlib.sha256(f"g{i}".encode()).hexdigest(), "text": f"doc {i}"} for i in range(20)]
    c.post("/shards/s1/chunks", json={"chunks": chunks})
    node_ids = []
    for i in range(2):
        r = c.post("/nodes/handshake", json={"name": f"n{i}", "hardware": {"cpu_cores": 1, "ram_gb": 1},
                                             "model_kinds": ["classifier"]})
        node_ids.append(r.json()["node_id"])
    c.post("/campaigns", json={"campaign_id": "c1", "shard_id": "s1", "model": {"kind": "classifier", "id": "m"},
                               "metrics": [], "node_ids": node_ids, "schedule": [5], "strategy": "random",
                               "k_frac": 0.2})

    m = parse(c.get("/metrics").text)
    assert ({"shard_id": "s1"}, 20.0) in m["proxy_mesh_shard_chunks_total"]
    assert ({"shard_id": "s1"}, 10.0) in m["proxy_mesh_shard_labels_total"]  # 5 per partition
    assert ({"campaign_id": "c1", "shard_id": "s1"}, 1.0) in m["proxy_mesh_campaign_rounds_done"]
    assert ({"campaign_id": "c1", "shard_id": "s1"}, 0.0) in m["proxy_mesh_campaign_rounds_completed"]
    assert ({"campaign_id": "c1", "shard_id": "s1"}, 0.0) in m["proxy_mesh_campaign_partitions_submitted"]
    assert ({"campaign_id": "c1", "shard_id": "s1"}, 10.0) in m["proxy_mesh_campaign_labels"]
    assert ({"campaign_id": "c1", "shard_id": "s1"}, 10.0) in m["proxy_mesh_campaign_labels_target"]
    assert ({"campaign_id": "c1", "shard_id": "s1"}, 1.0) in m["proxy_mesh_campaign_schedule_len"]
    assert ({"status": "running"}, 1.0) in m["proxy_mesh_campaigns_total"]

    campaign = c.get("/metrics.json").json()["campaigns"][0]
    assert campaign["mode"] == "sharded"
    assert campaign["n_partitions"] == 2
    assert campaign["partitions_submitted"] == 0
    assert campaign["current_step"] == 1
    assert campaign["completed_rounds"] == 0
    assert campaign["n_labels"] == campaign["n_labels_target"] == 10


def test_telemetry_history_is_persisted_and_limited_per_node(tmp_path, monkeypatch):
    db_path = tmp_path / "cp.sqlite3"
    monkeypatch.setenv("PM_DB_PATH", str(db_path))
    monkeypatch.setenv("PM_AUDIT_LOG", str(tmp_path / "audit.jsonl"))

    first = TestClient(create_app(Settings()))
    node_id = first.post("/nodes/handshake", json={
        "name": "worker", "hardware": {"cpu_cores": 2, "ram_gb": 4},
        "model_kinds": ["classifier"],
    }).json()["node_id"]
    assert first.post(f"/nodes/{node_id}/heartbeat", json={
        "status": "busy", "stage": "downloading", "round_id": "r1",
        "load": {"progress_pct": 10.0},
    }).status_code == 200
    assert first.post(f"/nodes/{node_id}/heartbeat", json={
        "status": "busy", "stage": "training", "round_id": "r1",
        "load": {"progress_pct": 25.0, "train_loss": 0.8},
    }).status_code == 200

    # A new app instance over the same SQLite file models a control-plane restart.
    second = TestClient(create_app(Settings()))
    history = second.get(f"/telemetry/history?node_id={node_id}&limit=1")
    assert history.status_code == 200
    body = history.json()
    assert body["limit_per_node"] == 1
    assert body["retention_s"] == 7 * 24 * 3600
    assert body["nodes"][node_id][0]["stage"] == "training"
    assert body["nodes"][node_id][0]["load"] == {"progress_pct": 25.0, "train_loss": 0.8}

    all_samples = second.get("/telemetry/history?limit=10").json()["nodes"][node_id]
    assert [sample["stage"] for sample in all_samples] == ["downloading", "training"]
    assert second.get("/telemetry/history?node_id=missing").status_code == 404


def test_heartbeat_rejects_unknown_stage(client):
    node_id = client.post("/nodes/handshake", json={
        "name": "worker", "hardware": {"cpu_cores": 2, "ram_gb": 4},
        "model_kinds": ["classifier"],
    }).json()["node_id"]
    response = client.post(f"/nodes/{node_id}/heartbeat", json={
        "status": "busy", "stage": "thinking", "load": {},
    })
    assert response.status_code == 422
