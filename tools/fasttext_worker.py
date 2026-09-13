"""Standalone fastText train+score worker: trains on `train_path`, scores every line of `score_path`.

Run as a subprocess (a fresh OS process via `python tools/fasttext_worker.py ...`), never imported and
called in-process or via `multiprocessing`: fastText 0.9.3's Python binding non-deterministically raises
`RuntimeError: Encountered NaN` when training runs inside a multiprocessing worker — a `Pool` (even with
maxtasksperchild=1, even one task at a time), or a plain `Process` that re-imports a module with enough
of its own transitive imports (reproduced with this project's tools modules; not reproduced with a
bare `python -c` call or the fastText CLI) — while the identical training data trains cleanly as a bare
script every time. A real subprocess sidesteps whatever in multiprocessing's machinery triggers it.

    python tools/fasttext_worker.py <train.txt> <score.txt> <seed> <out.npy>

`train.txt`: fastText-format lines (`__label__N text...`). `score.txt`: one text-to-score per line, same
preprocessing as the training text — read and scored in batches, never all at once, so a large pool
doesn't multiply memory by every concurrent proxy run. Writes `out.npy` (float64 expected-label score
per score.txt line) and prints the learning rate actually used (may be halved on retry) as the only
stdout line.
"""
import sys

import numpy as np

# Must match proxy_rounds_sim.LOCAL_RECIPE["hp"] (kept independent: this script has no project imports,
# so it survives even if the caller's import chain is what triggers the bug above).
HP = {"loss": "softmax", "lr": 0.5, "epoch": 25, "dim": 32, "wordNgrams": 1, "minCount": 1,
      "minn": 0, "maxn": 0, "bucket": 0}
SCORE_BATCH = 5_000  # lines held in memory at once while scoring


def score_file(model, score_path: str) -> np.ndarray:
    scores, batch = [], []

    def flush():
        if not batch:
            return
        labels, probs = model.predict(batch, k=-1)
        scores.extend(sum(int(lab.removeprefix("__label__")) * p for lab, p in zip(ls, ps))
                      for ls, ps in zip(labels, probs))
        batch.clear()

    with open(score_path) as fh:
        for line in fh:
            batch.append(line.rstrip("\n"))
            if len(batch) >= SCORE_BATCH:
                flush()
    flush()
    return np.array(scores)


def main() -> None:
    train_path, score_path, seed, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    import fasttext  # imported here, not at module level: keep this script's own footprint minimal

    fasttext.FastText.eprint = lambda *_: None  # silence the load_model/train deprecation notice
    hp = dict(HP)
    for attempt in range(4):
        try:
            model = fasttext.train_supervised(input=train_path, thread=1, seed=seed, verbose=0, **hp)
            break
        except RuntimeError as exc:
            if "NaN" not in str(exc) or attempt == 3:
                raise
            hp["lr"] /= 2
    np.save(out_path, score_file(model, score_path))
    print(hp["lr"])


if __name__ == "__main__":
    main()
