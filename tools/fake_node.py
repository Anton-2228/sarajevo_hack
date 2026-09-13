"""Simulated node: handshake -> heartbeat -> poll tasks -> ack -> score -> submit.

Generates synthetic scores for whatever workload a task assigns. The proxy score is signal + noise,
so higher `--noise` means a worse proxy. A synthetic held-out split yields the curve the Reliability
Gate judges, so a noisy node ends up untrusted.

    python tools/fake_node.py --name bank-a  --noise 0.3
    python tools/fake_node.py --name gov-c   --noise 2.0 --once

It prints its node_id; create a round naming it (see README) and watch it pick the task up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from node_sdk.client import ControlPlane, ControlPlaneError, heldout_curve  # noqa: E402


def draw(args, rng):
    relevant = rng.random() < args.relevant
    return relevant, (1.0 if relevant else 0.0) + rng.gauss(0, args.noise)


def score_dataset(args, task, rng):
    scores, n_rel = [], 0
    for i in range(args.n):
        relevant, score = draw(args, rng)
        n_rel += relevant
        chunk_hash = hashlib.sha256(f"{args.name}:{task['dataset_id']}:{i}".encode()).hexdigest()
        scores.append((chunk_hash, score))
    heldout = [draw(args, rng) for _ in range(args.heldout)]

    known = {
        "n_chunks": args.n,
        "eval_spearman": args.rho,
        "proxy_lr": task["params"].get("proxy_lr", 1e-5),
        "n_dedup_dropped": rng.randint(0, 50),
        "score_mean": sum(s for _, s in scores) / len(scores),
    }
    stats = {"n_chunks": args.n}
    for metric in task["metrics"]:
        if metric not in known:
            print(f"  metric {metric!r} unknown to fake node, reporting a random value", file=sys.stderr)
        stats[metric] = known.get(metric, rng.random())
    stats.update(heldout_curve([s for _, s in heldout], [rel for rel, _ in heldout]))
    return scores, stats, n_rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("PM_URL", "http://127.0.0.1:8100"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--relevant", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.9, help="eval_spearman to report (informational)")
    ap.add_argument("--heldout", type=int, default=500, help="synthetic oracle-labelled held-out docs")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between task polls")
    ap.add_argument("--once", action="store_true", help="exit after the first submitted task")
    ap.add_argument("--state", default=None, help="identity file (default .pm_node_<name>.json)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cp = ControlPlane(args.url, state_file=args.state or f".pm_node_{args.name}.json")
    cp.handshake(
        name=args.name,
        hardware={"cpu_cores": os.cpu_count() or 1, "ram_gb": 16, "gpus": [], "disk_free_gb": 100},
        model_kinds=["classifier", "lora"],
        software={"agent": "fake_node", "python": platform.python_version()},
    )
    print(f"registered: node_id={cp.node_id}; polling {args.url} for tasks")

    beat = {"status": "idle"}
    cp.start_heartbeat(lambda: dict(beat))
    rng = random.Random(f"{args.name}:{args.seed}")
    while True:
        try:
            for task in cp.tasks():
                rid = task["round_id"]
                cp.ack(rid)
                beat.update(status="busy", round_id=rid)
                print(f"round {rid}: bake {task['model']['kind']} {task['model']['id']} "
                      f"on {task['dataset_id']}, metrics={task['metrics']}")
                scores, stats, n_rel = score_dataset(args, task, rng)
                print(json.dumps(cp.submit(rid, scores, stats), indent=2))
                print(f"  (local ground truth, never sent: {n_rel} relevant of {args.n})", file=sys.stderr)
                if args.once:
                    return
        except (ControlPlaneError, URLError) as exc:  # keep polling through server restarts
            print(f"poll failed, retrying: {exc}", file=sys.stderr)
        finally:
            beat.clear()
            beat["status"] = "idle"
        time.sleep(args.poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
