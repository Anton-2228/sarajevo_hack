"""Run a sharded or mixture-of-experts campaign end to end over real HTTP.

The mock workers follow each task's routing parameters. Their deterministic score uses document
length and lexical shape, so the demo responds to actual text even though it does not train a model.

    ./run.sh &
    .venv/bin/python tools/demo_campaign.py --mode sharded
    .venv/bin/python tools/demo_campaign.py --mode experts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List


def http(base: str, method: str, path: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
    request = urllib.request.Request(
        base + path,
        method=method,
        headers={"Content-Type": "application/json"},
        data=None if body is None else json.dumps(body).encode(),
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} -> {exc.code}: {exc.read().decode()}") from exc


def chunk_id(shard_id: str, i: int) -> str:
    return hashlib.sha256(f"demo-chunk:{shard_id}:{i}".encode()).hexdigest()


def mock_score(expert: int, item: Dict[str, str]) -> float:
    """Text-aware stand-in for training plus inference, with a distinct view per expert."""
    text = item["text"]
    words = text.split()
    length_signal = min(math.log1p(len(words)) / math.log(500), 1.0)
    lexical_signal = min(sum(len(word) for word in words) / max(len(words) * 12, 1), 1.0)
    weight = 0.35 + 0.15 * (expert % 3)
    jitter = (int(hashlib.sha256(f"{expert}:{item['chunk_id']}".encode()).hexdigest(), 16) % 1000) / 1000
    return (1 - weight) * length_signal + weight * lexical_signal + 0.02 * jitter


def task_for(base: str, node_id: str, round_id: str) -> Dict[str, Any]:
    tasks = http(base, "GET", f"/tasks/{node_id}")["tasks"]
    return next(task for task in tasks if task["round_id"] == round_id)


def chunks_for_task(base: str, task: Dict[str, Any]) -> List[Dict[str, str]]:
    params = task["params"]
    path = f"/shards/{task['dataset_id']}/chunks"
    if params["mode"] == "sharded":
        path += f"?partition={params['partition']}&n_partitions={params['n_partitions']}"
    return http(base, "GET", path)["chunks"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("PM_URL", "http://127.0.0.1:8100"))
    parser.add_argument("--shard-id", default="demo")
    parser.add_argument("--n-pool", type=int, default=500, help="ignored with --skip-load")
    parser.add_argument("--skip-load", action="store_true", help="use an already loaded shard")
    parser.add_argument("--mode", choices=["sharded", "experts"], default="sharded")
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--schedule", default="30,60,100", help="cumulative labels per partition")
    parser.add_argument("--strategy", default="cutoff", choices=["random", "cutoff", "qbc"])
    args = parser.parse_args()
    base = args.url.rstrip("/")

    if args.skip_load:
        n_pool = len(http(base, "GET", f"/shards/{args.shard_id}/chunks")["chunks"])
        print(f"1. using {n_pool} chunks already in shard {args.shard_id!r}")
    else:
        print(f"1. loading {args.n_pool} chunks into shard {args.shard_id!r} ...")
        chunks = [
            {
                "chunk_id": chunk_id(args.shard_id, i),
                "text": f"synthetic document {i} " + "useful context " * (1 + i % 20),
            }
            for i in range(args.n_pool)
        ]
        print("  ", http(base, "POST", f"/shards/{args.shard_id}/chunks", {"chunks": chunks}))

    print(f"2. enrolling {args.nodes} mock nodes ...")
    node_ids: List[str] = []
    for i in range(args.nodes):
        response = http(base, "POST", "/nodes/handshake", {
            "name": f"demo-node-{i}",
            "hardware": {"cpu_cores": 4, "ram_gb": 8},
            "model_kinds": ["classifier"],
        })
        node_ids.append(response["node_id"])
        print(f"   {response['node_id']}")

    schedule = [int(value) for value in args.schedule.split(",")]
    campaign_id = "demo-" + hashlib.sha256(os.urandom(8)).hexdigest()[:8]
    print(f"3. creating {args.mode} campaign {campaign_id!r} ...")
    http(base, "POST", "/campaigns", {
        "campaign_id": campaign_id,
        "shard_id": args.shard_id,
        "mode": args.mode,
        "model": {"kind": "classifier", "id": "demo-clf"},
        "metrics": [],
        "node_ids": node_ids,
        "schedule": schedule,
        "strategy": args.strategy,
        "k_frac": 0.1,
        "good_min": 1,
    })
    print(f"   seeded {schedule[0]} labels per partition ({schedule[0] * args.nodes} total)")

    step = 0
    while True:
        campaign = http(base, "GET", f"/campaigns/{campaign_id}")
        if campaign["status"] == "done":
            break
        round_id = campaign["current_round"]
        step += 1
        print(f"4.{step} round {round_id!r}: nodes score their assigned scope ...")
        for expert, node_id in enumerate(node_ids):
            task = task_for(base, node_id, round_id)
            chunks = chunks_for_task(base, task)
            scores = [{"chunk_id": item["chunk_id"], "score": mock_score(expert, item)} for item in chunks]
            payload = {
                "node_id": node_id,
                "round_id": round_id,
                "scores": scores,
                "agg_stats": {"n_chunks": len(scores)},
            }
            http(base, "POST", f"/tasks/{node_id}/{round_id}/submit", payload)
            print(f"      node {expert}: {len(scores)} chunks ({task['params']['mode']})")
        result = http(base, "POST", f"/campaigns/{campaign_id}/advance")
        n_labels = len(http(base, "GET", f"/shards/{args.shard_id}/labels")["labels"])
        print(f"      -> {result['status']}: {result['detail']} (labels: {n_labels})")

    print(f"\n5. done after {campaign['rounds_done']} rounds, {campaign['result']['n_labels']} oracle labels")
    print(
        f"   selected {len(campaign['result']['selected'])} chunks "
        f"({campaign['result']['n_known_good']} already known good, rest by score)"
    )


if __name__ == "__main__":
    main()
