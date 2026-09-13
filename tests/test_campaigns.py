"""End-to-end tests for sharded and mixture-of-experts campaigns."""
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from control_plane.app import Settings, create_app
from control_plane.oracle import StaticOracle
from control_plane.partitioning import chunk_ids_for_partition, partition_of
from node_sdk.client import build_payload

HW = {"cpu_cores": 4, "ram_gb": 8}


def chunk_id(i: int) -> str:
    return hashlib.sha256(f"chunk:{i}".encode()).hexdigest()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PM_DB_PATH", str(tmp_path / "cp.sqlite3"))
    monkeypatch.setenv("PM_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("PM_MOCK_ORACLE_GOOD_RATE", "0.2")
    return TestClient(create_app(Settings()))


def enroll(c, name):
    response = c.post("/nodes/handshake", json={
        "name": name, "hardware": HW, "model_kinds": ["classifier"],
    })
    assert response.status_code == 201, response.text
    return response.json()["node_id"]


def load_shard(c, shard_id, n):
    chunks = [{"chunk_id": chunk_id(i), "text": f"document {i}"} for i in range(n)]
    response = c.post(f"/shards/{shard_id}/chunks", json={"chunks": chunks})
    assert response.status_code == 201, response.text
    return response.json()


def task_for(c, node_id, round_id):
    tasks = c.get(f"/tasks/{node_id}").json()["tasks"]
    return next(task for task in tasks if task["round_id"] == round_id)


def chunks_for_task(c, task):
    params = task["params"]
    path = f"/shards/{task['dataset_id']}/chunks"
    if params["mode"] == "sharded":
        path += f"?partition={params['partition']}&n_partitions={params['n_partitions']}"
    return c.get(path).json()["chunks"]


def submit_node_scores(c, node_id, round_id, seed=0, chunks=None, score_overrides=None):
    if chunks is None:
        chunks = chunks_for_task(c, task_for(c, node_id, round_id))
    scores = []
    for item in chunks:
        cid = item["chunk_id"]
        score = (int(hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest(), 16) % 1000) / 1000
        if score_overrides and cid in score_overrides:
            score = score_overrides[cid]
        scores.append((cid, score))
    payload = build_payload(node_id, round_id, scores, {"n_chunks": len(scores)})
    response = c.post(f"/tasks/{node_id}/{round_id}/submit", json=payload)
    assert response.status_code == 201, response.text
    return {item[0] for item in scores}


def run_to_completion(c, campaign_id, node_ids, max_steps=10):
    for _ in range(max_steps):
        campaign = c.get(f"/campaigns/{campaign_id}").json()
        if campaign["status"] == "done":
            return campaign
        round_id = campaign["current_round"]
        for i, node_id in enumerate(node_ids):
            submit_node_scores(c, node_id, round_id, seed=i)
        response = c.post(f"/campaigns/{campaign_id}/advance")
        assert response.status_code == 200, response.text
        assert response.json()["status"] in ("advanced", "done")
    raise AssertionError(f"campaign {campaign_id} did not finish within {max_steps} advances")


def test_partitioning_is_deterministic_and_balanced():
    pool = [chunk_id(i) for i in range(10_000)]
    first = [partition_of(cid, 8) for cid in pool]
    assert first == [partition_of(cid, 8) for cid in pool]
    sizes = [len(chunk_ids_for_partition(pool, i, 8)) for i in range(8)]
    assert max(sizes) - min(sizes) < len(pool) * 0.05
    with pytest.raises(ValueError, match="positive"):
        partition_of(pool[0], 0)


def test_partitioned_chunk_access_is_stable_disjoint_and_complete(client):
    c = client
    load_shard(c, "s1", 101)
    partitions = []
    for i in range(4):
        path = f"/shards/s1/chunks?partition={i}&n_partitions=4"
        first = [item["chunk_id"] for item in c.get(path).json()["chunks"]]
        second = [item["chunk_id"] for item in c.get(path).json()["chunks"]]
        assert first == second
        partitions.append(set(first))
    assert set.union(*partitions) == {chunk_id(i) for i in range(101)}
    assert sum(map(len, partitions)) == len(set.union(*partitions))
    assert c.get("/shards/s1/chunks?partition=0").status_code == 422
    assert c.get("/shards/s1/chunks?n_partitions=4").status_code == 422
    assert c.get("/shards/s1/chunks?partition=4&n_partitions=4").status_code == 422


def test_sharded_campaign_scores_disjoint_partitions_and_merges_without_loss(client):
    c = client
    shard_id, n_pool = "wave1", 120
    load_shard(c, shard_id, n_pool)
    node_ids = [enroll(c, f"node-{i}") for i in range(3)]
    response = c.post("/campaigns", json={
        "campaign_id": "camp1", "shard_id": shard_id, "mode": "sharded",
        "model": {"kind": "classifier", "id": "clf-v1"}, "metrics": [], "node_ids": node_ids,
        "schedule": [1], "strategy": "cutoff", "k_frac": 1.0, "good_min": 0, "seed": 0,
    })
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["spec"]["mode"] == "sharded"
    assert created["spec"]["train_mode"] == "fresh"
    assert created["spec"]["n_partitions"] == 3
    assert len(c.get(f"/shards/{shard_id}/labels").json()["labels"]) == 3

    round_id = c.get("/campaigns/camp1").json()["current_round"]
    scored = []
    for i, node_id in enumerate(node_ids):
        task = task_for(c, node_id, round_id)
        assert task["params"] == {"n_labels": 1, "partition": i, "n_partitions": 3, "mode": "sharded"}
        assert task["operation"] == {
            "train": "fresh", "score": True, "input_checkpoint_id": None,
            "output_checkpoint_id": "camp1-r1",
        }
        scored.append(submit_node_scores(c, node_id, round_id, seed=i))
    assert set.union(*scored) == {chunk_id(i) for i in range(n_pool)}
    assert sum(map(len, scored)) == n_pool

    assert c.post("/campaigns/camp1/advance").json()["status"] == "done"
    result = c.get("/campaigns/camp1").json()["result"]
    assert len(result["selected"]) == n_pool
    assert set(result["selected"]) == {chunk_id(i) for i in range(n_pool)}


def test_sharded_schedule_is_per_partition(client):
    c = client
    load_shard(c, "s1", 400)
    node_ids = [enroll(c, f"node-{i}") for i in range(3)]
    response = c.post("/campaigns", json={
        "campaign_id": "c-schedule", "shard_id": "s1", "mode": "sharded",
        "model": {"kind": "classifier", "id": "m"}, "metrics": [], "node_ids": node_ids,
        "schedule": [20, 40, 60], "strategy": "cutoff", "k_frac": 0.1, "good_min": 1,
    })
    assert response.status_code == 201, response.text
    assert len(c.get("/shards/s1/labels").json()["labels"]) == 60
    campaign = run_to_completion(c, "c-schedule", node_ids)
    assert campaign["rounds_done"] == 3
    assert len(c.get("/shards/s1/labels").json()["labels"]) == 180
    assert len(campaign["result"]["selected"]) == 40


def test_advance_waits_for_every_node(client):
    c = client
    load_shard(c, "s1", 100)
    node_ids = [enroll(c, "a"), enroll(c, "b")]
    c.post("/campaigns", json={
        "campaign_id": "c2", "shard_id": "s1", "mode": "sharded",
        "model": {"kind": "classifier", "id": "m"}, "metrics": [], "node_ids": node_ids,
        "schedule": [10, 20], "strategy": "random", "k_frac": 0.1,
    })
    round_id = c.get("/campaigns/c2").json()["current_round"]
    submit_node_scores(c, node_ids[0], round_id)
    response = c.post("/campaigns/c2/advance")
    assert response.json()["status"] == "waiting"
    assert node_ids[1] in response.json()["detail"]
    submit_node_scores(c, node_ids[1], round_id)
    assert c.post("/campaigns/c2/advance").json()["status"] == "advanced"


def test_continue_campaign_chains_round_checkpoints(client):
    c = client
    load_shard(c, "continue-shard", 40)
    node_id = enroll(c, "continuing-node")
    response = c.post("/campaigns", json={
        "campaign_id": "continue-campaign", "shard_id": "continue-shard", "mode": "sharded",
        "model": {"kind": "classifier", "id": "m"}, "metrics": [], "node_ids": [node_id],
        "schedule": [5, 10], "train_mode": "continue", "strategy": "random", "k_frac": 0.1,
    })
    assert response.status_code == 201, response.text

    first = task_for(c, node_id, "continue-campaign-r1")
    assert first["operation"] == {
        "train": "fresh", "score": True, "input_checkpoint_id": None,
        "output_checkpoint_id": "continue-campaign-r1",
    }
    submit_node_scores(c, node_id, first["round_id"])
    assert c.post("/campaigns/continue-campaign/advance").json()["status"] == "advanced"

    second = task_for(c, node_id, "continue-campaign-r2")
    assert second["operation"] == {
        "train": "continue", "score": True,
        "input_checkpoint_id": "continue-campaign-r1",
        "output_checkpoint_id": "continue-campaign-r2",
    }


def test_experts_must_each_score_the_whole_pool(client):
    c = client
    load_shard(c, "s1", 80)
    node_ids = [enroll(c, "a"), enroll(c, "b")]
    response = c.post("/campaigns", json={
        "campaign_id": "experts", "shard_id": "s1", "mode": "experts",
        "model": {"kind": "classifier", "id": "m"}, "metrics": [], "node_ids": node_ids,
        "schedule": [2], "strategy": "random", "k_frac": 0.1,
    })
    assert response.status_code == 201, response.text
    round_id = c.get("/campaigns/experts").json()["current_round"]
    task = task_for(c, node_ids[0], round_id)
    assert task["params"]["mode"] == "experts"
    assert len(chunks_for_task(c, task)) == 80

    partition = c.get("/shards/s1/chunks?partition=0&n_partitions=2").json()["chunks"]
    submit_node_scores(c, node_ids[0], round_id, chunks=partition)
    submit_node_scores(c, node_ids[1], round_id)
    response = c.post("/campaigns/experts/advance")
    assert response.status_code == 422
    assert "expected chunks" in response.json()["detail"]

    full_pool = c.get("/shards/s1/chunks").json()["chunks"]
    submit_node_scores(c, node_ids[0], round_id, chunks=full_pool)
    assert c.post("/campaigns/experts/advance").json()["status"] == "done"


def test_experts_finalize_by_mean_score(client):
    c = client
    load_shard(c, "s1", 80)
    node_ids = [enroll(c, "a"), enroll(c, "b")]
    c.post("/campaigns", json={
        "campaign_id": "mean", "shard_id": "s1", "mode": "experts",
        "model": {"kind": "classifier", "id": "m"}, "metrics": [], "node_ids": node_ids,
        "schedule": [1], "strategy": "random", "k_frac": 0.1,
    })
    labels = c.get("/shards/s1/labels").json()["labels"]
    candidates = [chunk_id(i) for i in range(80) if chunk_id(i) not in labels]
    one_strong, both_good = candidates[:2]
    round_id = c.get("/campaigns/mean").json()["current_round"]
    zeroes = {chunk_id(i): 0.0 for i in range(80)}
    submit_node_scores(c, node_ids[0], round_id,
                       score_overrides={**zeroes, one_strong: 1.0, both_good: 0.6})
    submit_node_scores(c, node_ids[1], round_id,
                       score_overrides={**zeroes, one_strong: 0.0, both_good: 0.6})
    assert c.post("/campaigns/mean/advance").json()["status"] == "done"
    selected = c.get("/campaigns/mean").json()["result"]["selected"]
    assert both_good in selected
    assert selected.index(both_good) < selected.index(one_strong)


def test_create_campaign_rejects_bad_input(client):
    c = client
    load_shard(c, "s1", 50)
    node_id = enroll(c, "a")
    base = {
        "campaign_id": "c3", "shard_id": "s1", "mode": "sharded",
        "model": {"kind": "classifier", "id": "m"}, "metrics": [], "node_ids": [node_id],
        "strategy": "random", "k_frac": 0.1,
    }
    assert c.post("/campaigns", json={**base, "schedule": [60]}).status_code == 422
    assert c.post("/campaigns", json={**base, "schedule": [20, 10]}).status_code == 422
    assert c.post("/campaigns", json={**base, "schedule": [20], "node_ids": ["ghost-000000"]}).status_code == 422
    assert c.post("/campaigns", json={**base, "schedule": [20], "mode": "committee"}).status_code == 422
    assert c.post("/campaigns", json={**base, "schedule": [20]}).status_code == 201
    assert c.post("/campaigns", json={**base, "schedule": [20]}).status_code == 422


def test_static_oracle_serves_real_labels_not_hash_mock():
    oracle = StaticOracle({chunk_id(0): 1, chunk_id(1): 0})
    assert oracle.label([chunk_id(0), chunk_id(1)], texts={}) == {chunk_id(0): 1, chunk_id(1): 0}
    with pytest.raises(KeyError, match="golden label"):
        oracle.label([chunk_id(0), chunk_id(99)], texts={})


def test_golden_labels_path_wires_static_oracle_into_campaign(tmp_path, monkeypatch):
    golden = {chunk_id(i): int(i < 5) for i in range(20)}
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden))
    monkeypatch.setenv("PM_DB_PATH", str(tmp_path / "cp.sqlite3"))
    monkeypatch.setenv("PM_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("PM_GOLDEN_LABELS_PATH", str(golden_path))
    c = TestClient(create_app(Settings()))

    load_shard(c, "g1", 20)
    node_ids = [enroll(c, "a")]
    response = c.post("/campaigns", json={
        "campaign_id": "cg", "shard_id": "g1", "mode": "sharded",
        "model": {"kind": "classifier", "id": "m"}, "metrics": [], "node_ids": node_ids,
        "schedule": [20], "strategy": "random", "k_frac": 0.25, "good_min": 1, "seed": 1,
    })
    assert response.status_code == 201, response.text
    campaign = run_to_completion(c, "cg", node_ids)
    assert c.get("/shards/g1/labels").json()["labels"] == golden
    assert sorted(campaign["result"]["selected"]) == sorted(chunk_id(i) for i in range(5))
    assert campaign["result"]["n_known_good"] == 5


def test_ingest_chunks_rejects_non_hash_ids_and_dedupes(client):
    c = client
    assert c.post("/shards/s1/chunks", json={"chunks": [{"chunk_id": "not a hash", "text": "x"}]}).status_code == 422
    response = c.post("/shards/s1/chunks", json={"chunks": [{"chunk_id": chunk_id(0), "text": "a"}]})
    assert response.json() == {"shard_id": "s1", "received": 1, "new": 1, "total": 1}
    response = c.post("/shards/s1/chunks", json={"chunks": [{"chunk_id": chunk_id(0), "text": "a"}]})
    assert response.json()["new"] == 0 and response.json()["total"] == 1


def test_shard_inventory_and_searchable_preview(client):
    c = client
    load_shard(c, "browser-data", 5)
    labeled_id = chunk_id(2)
    c.app.state.store.add_labels({labeled_id: 4}, source="static-test")

    inventory = c.get("/shards").json()["shards"]
    assert len(inventory) == 1
    assert inventory[0]["shard_id"] == "browser-data"
    assert inventory[0]["n_chunks"] == 5 and inventory[0]["n_labels"] == 1

    page = c.get("/shards/browser-data/preview?offset=1&limit=2").json()
    assert page["total"] == 5 and page["offset"] == 1 and len(page["chunks"]) == 2
    labeled = c.get("/shards/browser-data/preview?q=document%202").json()
    assert labeled["total"] == 1
    assert labeled["chunks"][0]["chunk_id"] == labeled_id
    assert labeled["chunks"][0]["label"] == 4
    assert labeled["chunks"][0]["source"] == "static-test"
    assert c.get("/shards/missing/preview").status_code == 404


def test_parquet_inspect_and_threshold_import(client, tmp_path):
    parquet_path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({
        "text": ["alpha document", "beta document", "alpha document", None, "x" * 20_001],
        "score": [4, 2, 1, 5, 5],
    }), parquet_path)

    inspected = client.post(
        "/imports/parquet/inspect",
        content=parquet_path.read_bytes(),
        headers={"Content-Type": "application/vnd.apache.parquet", "X-Filename": "sample.parquet"},
    )
    assert inspected.status_code == 201, inspected.text
    upload = inspected.json()
    assert upload["n_rows"] == 5
    assert upload["defaults"] == {
        "text_column": "text", "id_column": None, "label_column": "score",
        "label_mode": "threshold", "label_threshold": 3,
    }
    assert {column["name"] for column in upload["columns"]} == {"text", "score"}

    imported = client.post("/shards/from-parquet/imports/parquet", json={
        "upload_token": upload["upload_token"],
        "text_column": "text",
        "id_column": None,
        "label_column": "score",
        "label_threshold": 3,
    })
    assert imported.status_code == 201, imported.text
    result = imported.json()
    assert result["rows_read"] == 5
    assert result["received"] == 2 and result["new"] == 2 and result["labeled"] == 2
    assert result["duplicates_in_file"] == 1
    assert result["skipped_empty"] == 1 and result["skipped_oversized"] == 1

    labels = client.get("/shards/from-parquet/labels").json()["labels"]
    assert labels[hashlib.sha256(b"alpha document").hexdigest()] == 1
    assert labels[hashlib.sha256(b"beta document").hexdigest()] == 0
    preview = client.get("/shards/from-parquet/preview?q=alpha").json()["chunks"][0]
    assert preview["source"] == "parquet:sample.parquet:score"
    assert client.post("/shards/again/imports/parquet", json={
        "upload_token": upload["upload_token"], "text_column": "text",
    }).status_code == 404


def test_parquet_inspect_rejects_invalid_file(client):
    response = client.post(
        "/imports/parquet/inspect", content=b"not parquet",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422
