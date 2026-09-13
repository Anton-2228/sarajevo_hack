"""Load a shard from FineWeb-Edu annotations, real text and real Llama-3 judgments, no LLM call needed.

Loads a control plane shard with real documents and writes a golden.json (chunk_id -> 0/1, good = the
oracle's educational score >= 3) that `control_plane.oracle.StaticOracle` reads via
PM_GOLDEN_LABELS_PATH. A campaign against this shard then gets real numbers — comparable to the earlier
tools/proxy_rounds_sim.py / tools/al_committee_sim.py runs — while the "oracle call" stays a mock (no
LLM, no cost, fully reproducible): the judgment was already made once, by Llama-3-70B, when this dataset
was built.

    ./run.sh &
    .venv/bin/python tools/load_golden_shard.py --shards data/fineweb_edu_ann/*.parquet \
        --shard-id demo --n 2000 --url http://127.0.0.1:8100 --out data/golden_demo.json
    PM_GOLDEN_LABELS_PATH=$(pwd)/data/golden_demo.json ./run.sh    # restart the CP with this oracle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq

MAX_TEXT_CHARS = 20_000  # control_plane.schemas.ChunkIn.text limit
ORACLE_GOOD = 3          # FineWeb-Edu: score >= 3 counts as good


def load(shards: List[str], n: int, seed: int) -> Tuple[List[Tuple[str, str, int]], int]:
    """[(chunk_id, text, oracle_score)], deduped, capped at n. Returns (rows, n_skipped_oversized).

    Reads each shard in batches via ParquetFile.iter_batches and stops as soon as it has enough rows
    (n plus a small margin for dedup/oversized skips) — never materializes a whole shard as Python
    objects. `pq.read_table(...).to_pylist()` on a full ~117k-row shard turns compact Arrow buffers into
    hundreds of thousands of individual Python str objects (~2-3 GB here); with tools/al_committee_sim.py
    doing the same at a bigger scale plus several concurrent proxy subprocesses, that combination is what
    took the machine down earlier this session. Reading only what's needed avoids the failure mode
    regardless of how this loader ends up being called (this run is a single process, but there is no
    reason to build the habit of full-shard reads only to slice afterwards).

    Trade-off: rows come from the front of each shard's row order, not a uniform sample of the whole
    file — fine for a demo pool, not a claim of an unbiased sample.
    """
    seen, rows, n_skipped = set(), [], 0
    target = n + n // 5 + 50
    for shard in shards:
        if len(rows) >= target:
            break
        for batch in pq.ParquetFile(shard).iter_batches(columns=["text", "score"], batch_size=2000):
            for text, score in zip(batch.column("text").to_pylist(), batch.column("score").to_pylist()):
                if len(text) > MAX_TEXT_CHARS:
                    n_skipped += 1
                    continue
                cid = hashlib.sha256(text.encode()).hexdigest()
                if cid in seen:
                    continue
                seen.add(cid)
                rows.append((cid, text, int(score)))
            if len(rows) >= target:
                break
    random.Random(seed).shuffle(rows)
    return rows[:n], n_skipped


def http(base: str, method: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, method=method, headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} -> {exc.code}: {exc.read().decode()}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--shard-id", default="golden")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--url", default=None, help="control plane base URL; omit to only write --out")
    ap.add_argument("--batch", type=int, default=1000, help="chunks per POST /shards/.../chunks call")
    ap.add_argument("--out", type=Path, default=Path("data/golden.json"),
                    help="chunk_id -> 0/1 map for PM_GOLDEN_LABELS_PATH")
    args = ap.parse_args()

    print("loading corpus ...")
    rows, n_skipped = load(args.shards, args.n, args.seed)
    if n_skipped:
        print(f"skipped {n_skipped} docs over {MAX_TEXT_CHARS} chars (ChunkIn.text limit)")
    print(f"{len(rows)} chunks, {sum(1 for *_, s in rows if s >= ORACLE_GOOD) / len(rows):.1%} good "
         f"(score >= {ORACLE_GOOD})")

    golden = {cid: int(score >= ORACLE_GOOD) for cid, _, score in rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden))
    print(f"wrote {args.out} ({len(golden)} labels)")

    if args.url:
        base = args.url.rstrip("/")
        for i in range(0, len(rows), args.batch):
            batch = rows[i:i + args.batch]
            chunks = [{"chunk_id": cid, "text": text} for cid, text, _ in batch]
            resp = http(base, "POST", f"/shards/{args.shard_id}/chunks", {"chunks": chunks})
            print(f"  posted {i + len(batch)}/{len(rows)} -> shard total {resp['total']}")
        print(f"\nshard {args.shard_id!r} ready with {len(rows)} chunks on {base}.")
    print(f"\nRestart the control plane with PM_GOLDEN_LABELS_PATH={args.out.resolve()} so its oracle "
         f"answers with these real labels, then create a campaign against shard {args.shard_id!r}:\n"
         f"  .venv/bin/python tools/demo_campaign.py --shard-id {args.shard_id} --skip-load "
         f"--n-pool {len(rows)}")


if __name__ == "__main__":
    main()
