"""Proxy rounds on FineWeb-Edu annotations through the control plane: global DCLM v0, then local fastText.

Nodes are URL TLD groups (.edu, .gov, .org, .com, rest) of HuggingFaceFW/fineweb-edu-llama3-annotations;
the Llama-3-70B educational score (0-5) plays the oracle, good = score >= 3. Five rounds on one split:

  v0          global mlfoundations/fasttext-oh-eli5, frozen, identical on every node
  v1-random   node buys `label_budget` random oracle labels and trains its own fastText
  v1-al       same budget: 1/3 around this node's cutoff in the v0 top-k, 1/3 from its v0 top, 1/3 random
  v2-random   `label_budget` more random labels on top of v1-random's
  v2-al       `label_budget` more, the same thirds around this node's cutoff in the v1-al top-k

Every node reports its held-out precision curve (node_sdk.client.heldout_curve). The control plane pools
on calibrated scores, so the budget follows expected good docs, and gates each node on held-out precision
at its cutoff. A node's cutoff in the previous round's allocation reaches it as participants[].params.cutoff_q.

Node side sees oracle labels only for its held-out split and the chunks it paid to label. Local weights
never leave the node (a fastText .bin stores its training vocabulary). God view (simulation only, never
available in production) knows the oracle for every doc and scores what the global top-k actually picked.

    ./run.sh &
    .venv/bin/pip install -r tools/requirements-sim.txt
    .venv/bin/python tools/proxy_rounds_sim.py --shards data/fineweb_edu_ann/*.parquet \
        --model data/models/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fasttext
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from node_sdk.client import ControlPlane, heldout_curve  # noqa: E402

# Global model registry. model.id carries a sha256 prefix, so a node can check it holds exactly the
# weights the round asks for (the contract has no digest field yet).
MODEL_SHA256 = "f8da53846636ce5ed2836f95e2fffada6aade035e2169af1d6c099bda2674915"
V0_ID = f"dclm-oh-eli5:{MODEL_SHA256[:12]}"
MODEL_URL = ("https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/"
             "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin")
DCLM_THRESHOLD = 0.018112  # DCLM-Baseline cut (keeps ~top 10% of RefinedWeb)
ORACLE_GOOD = 3            # FineWeb-Edu keeps score >= 3

# What every node trains locally. The id pins the recipe, not the weights: those stay on the node.
LOCAL_RECIPE = {
    "text": "first 2000 chars (the oracle judged a ~2k-char extract), lowercased, whitespace collapsed",
    "labels": "oracle 0..4, 5 merged into 4",
    "score": "expected label",
    # lr 0.1-0.2 + bigrams + dim 32 underfit on 2k docs (rho ~0); unigrams at lr 0.5 reach rho ~0.54,
    # close to a tf-idf ridge baseline (0.59). bucket=0: no n-grams, so no hash rows (the Python
    # binding does not zero it the way the CLI does, and 2M rows x dim would be allocated for nothing).
    "hp": {"loss": "softmax", "lr": 0.5, "epoch": 25, "dim": 32, "wordNgrams": 1, "minCount": 1,
           "minn": 0, "maxn": 0, "bucket": 0},
    "on_nan": "retry at lr/2, at most twice; the lr used is reported as train_lr",
}
_RECIPE_SHA = hashlib.sha256(json.dumps(LOCAL_RECIPE, sort_keys=True).encode()).hexdigest()[:12]
LOCAL_ID = f"local-ft:{_RECIPE_SHA}"

DOMAINS = {"edu": "univ", "gov": "gov", "org": "ngo", "com": "commerce", "misc": "misc"}
BASE_METRICS = ["eval_spearman", "n_heldout", "score_mean", "n_dedup_dropped"]
LOCAL_METRICS = BASE_METRICS + ["n_oracle_labels", "n_new_labels", "n_train_pos", "train_lr"]
SELECTIONS = {  # top-k query variants the operator compares after every round
    "calibrated + gate": {},
    "calibrated, no gate": {"include_untrusted": "true"},
    "z-score, no gate": {"normalize": "zscore", "include_untrusted": "true", "include_unknown": "true"},
}


# --- shared helpers -------------------------------------------------------------------------
def rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (oracle scores are 0-5, so ties are the norm)."""
    order = np.argsort(a, kind="mergesort")
    _, inv, counts = np.unique(a[order], return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    ranks = np.empty(len(a))
    ranks[order] = (starts + (counts - 1) / 2.0)[inv]
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n: int = 200, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [spearman(x[idx], y[idx]) for idx in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def domain_of(url: str) -> str:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    tld = host.rsplit(".", 1)[-1]
    if tld == "edu" or ".edu." in host or ".ac." in host:
        return "edu"
    if tld in ("gov", "mil") or ".gov." in host:
        return "gov"
    return tld if tld in ("org", "com") else "misc"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- node side: only node-local data from here to "operator" ---------------------------------
@dataclass
class NodeData:
    name: str
    domain: str
    chunk_ids: List[str]
    texts: List[str]
    heldout_texts: List[str]
    heldout_oracle: np.ndarray  # labels the node paid its local oracle for
    n_dedup_dropped: int
    _oracle: np.ndarray         # simulated local LLM judge over `texts`: reach it only via ask_oracle
    labels_spent: int = 0
    state: Dict[Any, Any] = field(default_factory=dict)  # node-local memory across rounds

    @property
    def dataset_id(self) -> str:
        return f"fwedu-{self.domain}"

    def ask_oracle(self, idx: np.ndarray) -> np.ndarray:
        self.labels_spent += len(idx)
        return self._oracle[idx]


def load_v0_model(path: Path):
    digest = sha256_file(path)
    if digest != MODEL_SHA256:
        raise SystemExit(f"{path}: sha256 {digest[:12]} does not match registry {MODEL_SHA256[:12]}")
    fasttext.FastText.eprint = lambda *_: None  # silence the load_model deprecation notice
    return fasttext.load_model(str(path))


def dclm_prob(model, texts: List[str]) -> np.ndarray:
    """DCLM scoring: newlines joined, P(__label__hq)."""
    labels, probs = model.predict([" ".join(t.strip().splitlines()) for t in texts], k=1)
    return np.array([p[0] if lab[0] == "__label__hq" else 1.0 - p[0] for lab, p in zip(labels, probs)])


def cached_v0(model, node: NodeData, cache_dir: Path) -> np.ndarray:
    """Re-score only when (model weights, corpus) change — the bake key, not the round."""
    key = hashlib.sha256((MODEL_SHA256 + "\n" + "\n".join(node.chunk_ids)).encode()).hexdigest()[:16]
    path = cache_dir / f"v0_{node.domain}_{key}.npy"
    if path.exists():
        return np.load(path)
    scores = dclm_prob(model, node.texts)
    np.save(path, scores)
    return scores


def local_text(text: str) -> str:
    return " ".join(text[:2000].lower().split())


WORKER = Path(__file__).resolve().with_name("fasttext_worker.py")


def run_proxy_jobs(jobs: List[Tuple[str, str, int]], max_parallel: int = 1) -> List[Tuple[np.ndarray, float]]:
    """(train_path, score_path, seed) -> [(expected-label score per line of score_path, lr used), ...].

    Each job runs `fasttext_worker.py` as its own subprocess, up to max_parallel concurrently: a fresh
    OS process, not a `multiprocessing` worker — see fasttext_worker.py for why. `seed` only affects
    fastText's internal initialization; `thread=1` inside the worker keeps a given seed reproducible."""
    # keyed by job index, not (score_path, seed): several jobs commonly share both (e.g. one member
    # each of several strategies at the same step), which would collide on one output file
    out_paths = [f"{score_path}.job{i}.out.npy" for i, (_, score_path, _) in enumerate(jobs)]
    pending, running, results = list(enumerate(jobs)), [], [None] * len(jobs)
    while pending or running:
        while pending and len(running) < max_parallel:
            i, (train_path, score_path, seed) = pending.pop(0)
            proc = subprocess.Popen([sys.executable, str(WORKER), train_path, score_path, str(seed), out_paths[i]],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            running.append((i, proc))
        i, proc = running.pop(0)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"fasttext_worker failed for job {jobs[i]}:\n{stderr}")
        results[i] = (np.load(out_paths[i]), float(stdout.strip()))
        Path(out_paths[i]).unlink()
    return results


def bake_local(train_texts: List[str], labels: np.ndarray, score_texts: List[str], work: Path, seed: int
               ) -> Tuple[np.ndarray, float]:
    """Train the local recipe on `train_texts`/`labels` and score `score_texts`."""
    train_path, score_path = work.parent / f"{work.name}.train.txt", work.parent / f"{work.name}.score.txt"
    train_path.write_text("".join(f"__label__{min(int(y), 4)} {local_text(t)}\n" for t, y in zip(train_texts, labels)))
    score_path.write_text("".join(local_text(t) + "\n" for t in score_texts))
    try:
        (result,) = run_proxy_jobs([(str(train_path), str(score_path), seed)])
        return result
    finally:
        train_path.unlink()
        score_path.unlink()


def pick_labels(n: int, budget: int, strategy: int, prev: np.ndarray, cutoff_q: float, taken: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """New chunk indices to label, never re-labelling `taken`; all labels stay within half the corpus.

    0 = random. 1 = thirds: nearest to this node's cutoff quantile in the previous round's ranking,
    best-ranked there, random."""
    budget = min(budget, n // 2 - len(taken))
    if budget <= 0:
        return np.empty(0, dtype=int)
    free = np.ones(n, bool)
    free[taken] = False
    if strategy == 0:
        return rng.choice(np.flatnonzero(free), budget, replace=False)
    order = np.argsort(-prev, kind="stable")
    frac = np.empty(n)
    frac[order] = (np.arange(n) + 1) / n
    near = np.array([i for i in np.argsort(np.abs(frac - cutoff_q), kind="stable") if free[i]][:budget // 3], dtype=int)
    free[near] = False
    top = np.array([i for i in order if free[i]][:budget // 3], dtype=int)
    free[top] = False
    rest = rng.choice(np.flatnonzero(free), budget - len(near) - len(top), replace=False)
    return np.concatenate([near, top, rest])


def node_scores(node: NodeData, task: Dict[str, Any], v0_model, cache_dir: Path, label_seed: int
                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Bake the round's model on this node: (corpus scores, held-out scores, extra metrics)."""
    model_id, params = task["model"]["id"], task["params"]
    if model_id == V0_ID:
        if "v0" not in node.state:
            if v0_model is None:
                raise RuntimeError(f"{node.name}: v0 weights not loaded")
            node.state["v0"] = (cached_v0(v0_model, node, cache_dir), dclm_prob(v0_model, node.heldout_texts))
        scores, held = node.state["v0"]
        return scores, held, {"frac_above_dclm_threshold": float((scores >= params["dclm_threshold"]).mean())}

    if model_id != LOCAL_ID:
        raise RuntimeError(f"{node.name}: unknown model {model_id!r}")
    if "v0" not in node.state:
        raise RuntimeError(f"{node.name}: local rounds need this node's v0 scores first")
    strategy, step, budget = int(params["al_strategy"]), int(params["al_step"]), int(params["label_budget"])
    chain = node.state.setdefault(("chain", strategy), {"idx": np.empty(0, dtype=int), "y": np.empty(0, dtype=int),
                                                        "steps": {}})
    if step not in chain["steps"]:
        if step > 1 and step - 1 not in chain["steps"]:
            raise RuntimeError(f"{node.name}: al_step {step} before al_step {step - 1}")
        if strategy == 1 and "cutoff_q" not in params:
            raise RuntimeError(f"{node.name}: al_strategy 1 needs cutoff_q")
        prev = node.state["v0"][0] if step == 1 else chain["steps"][step - 1]["scores"]
        rng = np.random.default_rng([label_seed, strategy, step, zlib.crc32(node.domain.encode())])
        new = pick_labels(len(node.texts), budget, strategy, prev, params.get("cutoff_q", 0.0), chain["idx"], rng)
        if len(new) == 0 and step > 1:  # label cap reached: nothing new to learn from
            chain["steps"][step] = {**chain["steps"][step - 1], "n_new": 0}
        else:
            chain["idx"] = np.concatenate([chain["idx"], new])
            chain["y"] = np.concatenate([chain["y"], node.ask_oracle(new)])
            all_scores, lr = bake_local([node.texts[i] for i in chain["idx"]], chain["y"],
                                        node.texts + node.heldout_texts, cache_dir / node.domain, label_seed * 100 + step)
            n = len(node.texts)
            chain["steps"][step] = {"scores": all_scores[:n], "held": all_scores[n:], "n_labels": len(chain["idx"]),
                                    "n_new": len(new), "n_pos": int((chain["y"] >= ORACLE_GOOD).sum()), "lr": lr}
    s = chain["steps"][step]
    return s["scores"], s["held"], {"n_oracle_labels": s["n_labels"] + len(s["held"]), "n_new_labels": s["n_new"],
                                    "n_train_pos": s["n_pos"], "train_lr": s["lr"]}


def run_node_task(cp: ControlPlane, node: NodeData, round_id: str, v0_model, cache_dir: Path,
                  label_seed: int) -> Dict[str, Any]:
    task = next((t for t in cp.tasks() if t["round_id"] == round_id), None)
    if task is None:
        raise RuntimeError(f"{node.name}: no task for round {round_id}")
    cp.ack(round_id)

    t0 = time.time()
    scores, held, extra = node_scores(node, task, v0_model, cache_dir, label_seed)
    curve = heldout_curve(held.tolist(), (node.heldout_oracle >= ORACLE_GOOD).tolist())
    local = {
        "n_chunks": len(scores),
        "eval_spearman": spearman(held, node.heldout_oracle),
        "n_heldout": len(held),
        "score_mean": float(scores.mean()),
        "n_dedup_dropped": node.n_dedup_dropped,
        **extra,
    }
    stats = {m: local[m] for m in task["metrics"]} | {"n_chunks": local["n_chunks"]} | curve
    resp = cp.submit(round_id, zip(node.chunk_ids, scores.tolist()), stats)
    lo, hi = bootstrap_ci(held, node.heldout_oracle)
    print(f"  {node.name:<9} n={len(scores):>6}  rho={local['eval_spearman']:.3f} [{lo:.2f}, {hi:.2f}]"
          f"  held-out p@10%={curve['ho_good_q10'] / curve['ho_n_q10']:.2f}  {resp['trust']}  {time.time() - t0:.1f}s")
    return local | {"rho_lo": lo, "rho_hi": hi}


# --- simulation setup: split the public corpus into nodes ------------------------------------
def build_nodes(shards: List[str], heldout: int, min_docs: int, seed: int
                ) -> Tuple[List[NodeData], Dict[str, int]]:
    by_domain: Dict[str, List[Tuple[str, int]]] = {d: [] for d in DOMAINS}
    for shard in shards:
        table = pq.read_table(shard, columns=["text", "score", "metadata"])
        urls = table.column("metadata").combine_chunks().field("url").to_pylist()
        for text, score, url in zip(table.column("text").to_pylist(), table.column("score").to_pylist(), urls):
            by_domain[domain_of(url or "")].append((text, int(score)))

    nodes, truth = [], {}  # truth: chunk_id -> oracle score, god view only
    for domain, docs in by_domain.items():
        if len(docs) < min_docs:
            print(f"skip domain {domain}: {len(docs)} docs < {min_docs}")
            continue
        # per-domain seed: a node's held-out split must not depend on which other nodes exist
        random.Random(f"{seed}:{domain}").shuffle(docs)
        n_held = min(heldout, len(docs) // 5)
        held, rest = docs[:n_held], docs[n_held:]
        seen, ids, texts, oracle = set(), [], [], []
        for text, score in rest:
            cid = hashlib.sha256(text.encode()).hexdigest()
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            texts.append(text)
            oracle.append(score)
            truth.setdefault(cid, score)
        nodes.append(NodeData(
            name=DOMAINS[domain], domain=domain, chunk_ids=ids, texts=texts,
            heldout_texts=[t for t, _ in held], heldout_oracle=np.array([s for _, s in held]),
            n_dedup_dropped=len(rest) - len(ids), _oracle=np.array(oracle)))
    return nodes, truth


# --- operator ----------------------------------------------------------------------------------
def http(base: str, method: str, path: str, body: Optional[dict] = None) -> dict:
    req = urllib.request.Request(base + path, method=method, headers={"Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def judge(chunk_ids: List[str], truth: Dict[str, int]) -> Tuple[float, float]:
    """God view: mean oracle score and share of oracle >= 3 in a selection."""
    if not chunk_ids:
        return float("nan"), float("nan")
    s = np.array([truth[c] for c in chunk_ids])
    return float(s.mean()), float((s >= ORACLE_GOOD).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.getenv("PM_URL", "http://127.0.0.1:8100"))
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--model", required=True, type=Path, help=f"local copy of {MODEL_URL}")
    ap.add_argument("--heldout", type=int, default=2000, help="oracle-labelled held-out docs per node (max 20%%)")
    ap.add_argument("--label-budget", type=int, default=2000, help="new oracle labels per node per local round "
                                                                   "(all labels stay within half the corpus)")
    ap.add_argument("--budget-frac", type=float, default=0.1, help="global top-k as a share of all chunks")
    ap.add_argument("--min-docs", type=int, default=500, help="skip domains smaller than this")
    ap.add_argument("--workdir", type=Path, default=Path("data/proxy_sim"))
    ap.add_argument("--round-prefix", default=None)
    ap.add_argument("--seed", type=int, default=0, help="held-out split")
    ap.add_argument("--label-seed", type=int, default=0, help="which chunks get labelled, fastText init")
    args = ap.parse_args()
    cache_dir = args.workdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")

    print("loading corpus ...")
    nodes, truth = build_nodes(args.shards, args.heldout, args.min_docs, args.seed)
    print("loading v0 model (sha256 check) ...")
    v0_model = load_v0_model(args.model)

    clients: Dict[str, ControlPlane] = {}
    for node in nodes:
        cp = ControlPlane(base, state_file=str(args.workdir / f".pm_node_{node.name}.json"))
        cp.handshake(
            name=node.name,
            hardware={"cpu_cores": os.cpu_count() or 1, "ram_gb": 16},
            model_kinds=["classifier"],
            software={"agent": "proxy_rounds_sim", "fasttext": "0.9.3"},
        )
        clients[node.name] = cp
    name_of = {cp.node_id: name for name, cp in clients.items()}

    total = sum(len(n.chunk_ids) for n in nodes)
    budget_k = max(1, round(total * args.budget_frac))
    prefix = args.round_prefix or f"fwedu-{int(time.time())}"

    def run_round(key: str, model_id: str, params: Dict[str, float], metrics: List[str], note: str,
                  node_params: Optional[Dict[str, Dict[str, float]]] = None):
        round_id = f"{prefix}-{key}"
        http(base, "POST", "/rounds", {
            "round_id": round_id, "budget_k": budget_k, "note": note,
            "model": {"kind": "classifier", "id": model_id}, "metrics": metrics, "params": params,
            "participants": [{"node_id": clients[n.name].node_id, "dataset_id": n.dataset_id,
                              **({"params": node_params[n.name]} if node_params else {})} for n in nodes],
        })
        print(f"round {round_id}: {model_id} params={params}" + (f" cutoffs={node_params}" if node_params else ""))
        local = {n.name: run_node_task(clients[n.name], n, round_id, v0_model, cache_dir, args.label_seed)
                 for n in nodes}
        topk = {label: http(base, "GET", f"/rounds/{round_id}/topk?" + urllib.parse.urlencode({"k": budget_k, **q}))
                for label, q in SELECTIONS.items()}
        return local, topk

    def cutoffs(topk: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Each node's cutoff quantile in a round's calibrated allocation, gate not enforced."""
        gate = {g["node_id"]: g for g in topk["calibrated, no gate"]["gate"]}
        return {n.name: {"cutoff_q": round(max(gate[clients[n.name].node_id]["cutoff_q"], 1 / len(n.chunk_ids)), 6)}
                for n in nodes}

    results: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    results["v0"] = run_round("v0", V0_ID, {"dclm_threshold": DCLM_THRESHOLD},
                              BASE_METRICS + ["frac_above_dclm_threshold"], "global DCLM fastText, frozen")
    v0_model = None  # nodes keep their v0 scores; the 2.4 GB weights are no longer needed

    def local_round(key: str, strategy: int, step: int, note: str, node_params=None) -> None:
        params = {"label_budget": args.label_budget, "al_strategy": strategy, "al_step": step}
        results[key] = run_round(key, LOCAL_ID, params, LOCAL_METRICS, note, node_params)

    local_round("v1-random", 0, 1, "local fastText, random oracle labels")
    local_round("v1-al", 1, 1, "local fastText, labels around each node's v0 cutoff", cutoffs(results["v0"][1]))
    local_round("v2-random", 0, 2, "v1-random + more random labels")
    local_round("v2-al", 1, 2, "v1-al + labels around each node's v1-al cutoff", cutoffs(results["v1-al"][1]))

    # --- report (node-local view, then god view) ----------------------------------------------
    keys = list(results)
    local_keys = keys[1:]
    gate_min = results["v0"][1]["calibrated + gate"]["gate_min_precision"]
    lines = [f"# Proxy rounds — `{prefix}`", "",
             f"{total} chunks in {len(nodes)} nodes, budget_k={budget_k} ({args.budget_frac:.0%}). "
             f"Oracle = Llama-3-70B edu score, good = score ≥ {ORACLE_GOOD}. Held-out ≤{args.heldout}/node, "
             f"+{args.label_budget} oracle labels/node per local round (capped at half the corpus). "
             f"Gate: Wilson 95% lower bound of held-out precision at the node's cutoff ≥ {gate_min}.", "",
             f"v0 `{V0_ID}`, local recipe `{LOCAL_ID}`: `{json.dumps(LOCAL_RECIPE['hp'])}`", "",
             "## ρ on held-out (node-local view)", "",
             "| node | chunks | held-out good | " + " | ".join(keys) + " |",
             "|---|---|---|" + "---|" * len(keys)]
    for n in nodes:
        cells = []
        for k in keys:
            m = results[k][0][n.name]
            lr = m.get("train_lr", LOCAL_RECIPE["hp"]["lr"])
            cells.append(f"{m['eval_spearman']:.3f} ±{(m['rho_hi'] - m['rho_lo']) / 2:.2f}"
                         + (f" (lr {lr:g})" if lr != LOCAL_RECIPE["hp"]["lr"] else ""))
        lines.append(f"| {n.name} | {len(n.chunk_ids)} | {(n.heldout_oracle >= ORACLE_GOOD).mean():.1%} | "
                     + " | ".join(cells) + " |")

    lines += ["", "## Gate and allocation (calibrated + gate): slots · held-out precision at cutoff [lower bound]", "",
              "| node | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for n in nodes:
        nid, cells = clients[n.name].node_id, []
        for k in keys:
            top = results[k][1]["calibrated + gate"]
            entry = next(g for g in top["gate"] + top["nodes_skipped"] if g["node_id"] == nid)
            mark = "✓" if entry["trust"] == "trusted" else "✗"
            cells.append(f"{mark} {top['selected_per_node'].get(nid, 0)} · {entry['precision_at_cutoff']:.2f} "
                         f"[{entry['precision_lower_bound']:.2f}]" if entry.get("precision_at_cutoff") is not None
                         else f"{mark} {entry.get('reason', '')}")
        lines.append(f"| {n.name} | " + " | ".join(cells) + " |")

    lines += ["", "## Oracle labels used for training, cumulative per chain", "",
              "| node | " + " | ".join(local_keys) + " | bought in total (+ held-out) |", "|---|" + "---|" * (len(local_keys) + 1)]
    for n in nodes:
        cells = [str(results[k][0][n.name]["n_oracle_labels"] - results[k][0][n.name]["n_heldout"]) for k in local_keys]
        lines.append(f"| {n.name} | " + " | ".join(cells) + f" | {n.labels_spent} (+{len(n.heldout_texts)}) |")

    pool = np.array(list(truth.values()))
    best = np.sort(pool)[::-1][:budget_k]
    labels = list(SELECTIONS)
    lines += ["", "## Global top-k: share of good docs (god view)", "",
              "| round | " + " | ".join(labels) + " | trusted nodes | slots per node (calibrated + gate) |",
              "|---|" + "---|" * (len(labels) + 2),
              f"| random k | " + " | ".join(f"{(pool >= ORACLE_GOOD).mean():.1%}" for _ in labels) + " | — | — |",
              f"| oracle top-k (ceiling) | " + " | ".join(f"{(best >= ORACLE_GOOD).mean():.1%}" for _ in labels)
              + " | — | — |"]
    for k in keys:
        topks = results[k][1]
        shares = []
        for label in labels:
            chosen = [s["chunk_id"] for s in topks[label]["selected"]]
            shares.append(f"{judge(chosen, truth)[1]:.1%}" if chosen else "nothing selected")
        gated = topks["calibrated + gate"]
        per_node = ", ".join(f"{name_of[nid]} {c}" for nid, c in
                             sorted(gated["selected_per_node"].items(), key=lambda kv: -kv[1]))
        lines.append(f"| {k} | " + " | ".join(shares) + f" | {len(gated['nodes_used'])}/{len(nodes)} | {per_node} |")

    report = "\n".join(lines) + "\n"
    out = args.workdir / f"report-{prefix}.md"
    out.write_text(report)
    print("\n" + report + f"\nwritten to {out}")


if __name__ == "__main__":
    main()
