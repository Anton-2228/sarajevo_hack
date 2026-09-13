"""Does decentralized compute pay off for active learning? A committee of fastText proxies vs one model.

Server-master setting: the server holds the pool (HuggingFaceFW/fineweb-edu-llama3-annotations) and the
oracle (Llama-3-70B edu score, good = score >= 3); nodes only train proxies on the labels they are given
and score the pool. Every strategy gets the same label budget, grown along --schedule; every proxy run
(train + score the pool) is one isolated process — the unit of work a node would execute.

  single-random     1 model, random new labels
  single-cutoff     1 model, new labels: 1/3 nearest the top-k cutoff, 1/3 best-ranked, 1/3 random
  committee-random  N models on bootstrap resamples of the labels (own seeds), ranked by their mean
  committee-cutoff  N models, cutoff thirds on the mean score
  committee-qbc     N models, 2/3 with the largest disagreement (std across members) among docs ranked
                    within [k/3, 3k] of the pool, 1/3 random

Metrics: precision in the top 10% of a fixed eval split that is never labelled, Spearman there, and the
share of good docs in the pool top-k when the server fills it with the good docs it already has labels
for, then with the best-scored unlabelled ones.

    .venv/bin/python tools/al_committee_sim.py --shards data/fineweb_edu_ann/*.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxy_rounds_sim import LOCAL_ID, local_text, run_proxy_jobs, spearman  # noqa: E402

ORACLE_GOOD = 3
STRATEGIES = ("single-random", "single-cutoff", "committee-random", "committee-cutoff", "committee-qbc")
METRICS = {"eval_p10": "precision in the top 10% of the eval split",
           "eval_rho": "Spearman on the eval split",
           "select": "share of good docs in the pool top-k (known-good labels first)"}


def load(shards: List[str]) -> Tuple[List[str], np.ndarray]:
    seen, texts, scores = set(), [], []
    for shard in shards:
        table = pq.read_table(shard, columns=["text", "score"])
        for text, score in zip(table.column("text").to_pylist(), table.column("score").to_pylist()):
            h = hashlib.sha256(text.encode()).digest()
            if h not in seen:  # exact duplicates would leak labels from pool into eval
                seen.add(h)
                texts.append(text)
                scores.append(int(score))
    return texts, np.array(scores)


def acquire(strategy: str, labeled: np.ndarray, batch: int, mean: np.ndarray, std: np.ndarray, k_frac: float,
            rng: np.random.Generator) -> np.ndarray:
    n = len(mean)
    free = np.ones(n, bool)
    free[labeled] = False
    if strategy.endswith("random"):
        return rng.choice(np.flatnonzero(free), batch, replace=False)
    order = np.argsort(-mean, kind="stable")
    frac = np.empty(n)
    frac[order] = (np.arange(n) + 1) / n
    third = batch // 3
    if strategy.endswith("cutoff"):
        near = [i for i in np.argsort(np.abs(frac - k_frac), kind="stable") if free[i]][:third]
        free[near] = False
        top = [i for i in order if free[i]][:third]
        free[top] = False
        picked = near + top
    else:  # qbc
        band = np.flatnonzero(free & (frac >= k_frac / 3) & (frac <= k_frac * 3))
        picked = list(band[np.argsort(-std[band], kind="stable")][:batch - third])
        free[picked] = False
    rest = rng.choice(np.flatnonzero(free), batch - len(picked), replace=False)
    return np.concatenate([np.array(picked, dtype=int), rest])


def evaluate(pool_mean: np.ndarray, eval_mean: np.ndarray, labeled: np.ndarray, pool_y: np.ndarray,
             eval_y: np.ndarray, k_frac: float) -> Dict[str, float]:
    eval_top = np.argsort(-eval_mean, kind="stable")[:round(len(eval_y) * k_frac)]
    k = round(len(pool_y) * k_frac)
    known_good = labeled[pool_y[labeled] >= ORACLE_GOOD][:k]
    unlabeled = np.ones(len(pool_y), bool)
    unlabeled[labeled] = False
    candidates = np.flatnonzero(unlabeled)
    fill = candidates[np.argsort(-pool_mean[candidates], kind="stable")][:k - len(known_good)]
    return {"eval_p10": float((eval_y[eval_top] >= ORACLE_GOOD).mean()),
            "eval_rho": spearman(eval_mean, eval_y),
            "select": float(len(known_good) + (pool_y[fill] >= ORACLE_GOOD).sum()) / k}


def run_seed(seed: int, texts: List[str], y: np.ndarray, args, workdir: Path) -> Dict[Tuple[str, int], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(texts))
    eval_idx = perm[:args.eval_size]
    pool_idx = perm[args.eval_size:args.eval_size + args.pool_limit] if args.pool_limit else perm[args.eval_size:]
    pool_lines = [local_text(texts[i]) for i in pool_idx]
    pool_y, eval_y = y[pool_idx], y[eval_idx]
    n_pool = len(pool_idx)
    score_path = workdir / f"seed{seed}.score.txt"
    score_path.write_text("".join(line + "\n" for line in pool_lines)
                          + "".join(local_text(texts[i]) + "\n" for i in eval_idx))
    print(f"seed {seed}: pool {n_pool} (good {(pool_y >= ORACLE_GOOD).mean():.1%}), eval {len(eval_y)}", flush=True)

    schedule = [int(s) for s in args.schedule.split(",")]
    init = rng.choice(n_pool, schedule[0], replace=False)
    labeled = {s: init.copy() for s in STRATEGIES}
    prev: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    results = {}
    for step, budget in enumerate(schedule):
        t0 = time.time()
        if step:
            for s in STRATEGIES:
                arng = np.random.default_rng([seed, step, zlib.crc32(s.encode())])
                new = acquire(s, labeled[s], budget - schedule[step - 1], *prev[s], args.k_frac, arng)
                labeled[s] = np.concatenate([labeled[s], new])
        tasks, jobs = [], []
        for s in STRATEGIES:
            members = 1 if s.startswith("single") else args.committee
            for m in range(members):
                idx = labeled[s]
                if members > 1:  # bagging: each member sees its own bootstrap resample of the labels
                    idx = idx[np.random.default_rng([seed, step, m, zlib.crc32(s.encode())]).integers(0, len(idx), len(idx))]
                path = workdir / f"seed{seed}_{s}_{step}_{m}.train.txt"
                path.write_text("".join(f"__label__{min(int(pool_y[i]), 4)} {pool_lines[i]}\n" for i in idx))
                tasks.append((s, path))
                jobs.append((str(path), str(score_path), seed * 10_000 + step * 100 + m))
        outs = run_proxy_jobs(jobs, args.procs)  # a fresh, dedicated subprocess per proxy run
        by_strategy: Dict[str, List[np.ndarray]] = {s: [] for s in STRATEGIES}
        for (s, path), (scores, _lr) in zip(tasks, outs):
            by_strategy[s].append(scores)
            path.unlink()
        for s in STRATEGIES:
            runs = np.stack(by_strategy[s])
            pool_runs, eval_runs = runs[:, :n_pool], runs[:, n_pool:]
            prev[s] = (pool_runs.mean(0), pool_runs.std(0))
            results[(s, budget)] = evaluate(prev[s][0], eval_runs.mean(0), labeled[s], pool_y, eval_y, args.k_frac)
        print(f"  labels {budget:>5}: " + "  ".join(f"{s} {results[(s, budget)]['eval_p10']:.1%}/{results[(s, budget)]['select']:.1%}"
                                                  for s in STRATEGIES) + f"  ({time.time() - t0:.0f}s, {len(jobs)} proxy runs)", flush=True)
    score_path.unlink()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--workdir", type=Path, default=Path("data/committee_sim"))
    ap.add_argument("--eval-size", type=int, default=10_000)
    ap.add_argument("--pool-limit", type=int, default=30_000, help="cap the pool; 0 = everything left after "
                                                                   "eval (~450k docs — raise gradually, it "
                                                                   "multiplies by --committee x --procs in memory")
    ap.add_argument("--schedule", default="500,1000,1500,2000,3000,4000", help="cumulative oracle labels per step")
    ap.add_argument("--committee", type=int, default=5)
    ap.add_argument("--k-frac", type=float, default=0.1, help="top-k as a share of the pool")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--procs", type=int, default=2, help="proxy runs in parallel (stand-ins for nodes); "
                                                        "raise gradually and watch memory, not all at once")
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    texts, y = load(args.shards)
    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = [run_seed(seed, texts, y, args, args.workdir) for seed in seeds]

    schedule = [int(s) for s in args.schedule.split(",")]
    lines = [f"# Committee vs single proxy — active learning, server holds data and oracle", "",
             f"Pool: FineWeb-Edu llama3 annotations minus a {args.eval_size}-doc eval split, good = score ≥ {ORACLE_GOOD}; "
             f"top-k = {args.k_frac:.0%} of the pool. Proxy recipe `{LOCAL_ID}`, committee of {args.committee} bagged "
             f"models. Seeds {args.seeds}: mean ± half-range. Proxy runs per step: single 1, committee {args.committee}.", ""]
    for metric, title in METRICS.items():
        lines += [f"## {title}", "", "| labels | " + " | ".join(STRATEGIES) + " |", "|---|" + "---|" * len(STRATEGIES)]
        for budget in schedule:
            cells = []
            for s in STRATEGIES:
                vals = np.array([r[(s, budget)][metric] for r in per_seed])
                fmt = (lambda v: f"{v:.3f}") if metric == "eval_rho" else (lambda v: f"{v:.1%}")
                cells.append(fmt(vals.mean()) + (f" ±{(vals.max() - vals.min()) / 2 * (1 if metric == 'eval_rho' else 100):.{3 if metric == 'eval_rho' else 1}f}"
                                                 if len(vals) > 1 else ""))
            lines.append(f"| {budget} | " + " | ".join(cells) + " |")
        lines.append("")
    report = "\n".join(lines)
    out = args.workdir / f"report-{int(time.time())}.md"
    out.write_text(report)
    print("\n" + report + f"\nwritten to {out}")


if __name__ == "__main__":
    main()
