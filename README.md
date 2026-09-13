# sarajevo_hack — an active-learning node

A node in an active-learning fleet. A large LLM scores documents; the node
distils those scores into a cheap fastText classifier and scores new data in
the LLM's place.

Three programs live here:

- **`node-gui`** — the desktop app. The same node behind a window: pick how much
  of the machine to give it, press Start, close the window and it keeps working
  from the tray.
- **`node-agent`** — the node. Turns the machine into a worker: enrols with the
  control plane, polls for rounds, trains, scores, submits, within a CPU and
  RAM budget you set on the command line.
- **`node-clf`** — the classifier core on its own, for training and scoring
  from files without a server.

The control plane drives the rounds, and since contract 0.9.0 it says outright
what each one is for. Every task carries an `operation`:

| `operation.train` | Node does | Model |
|---|---|---|
| `fresh` | trains from the `model` recipe | saved as `output_checkpoint_id` |
| `continue` | trains on this round's data | loaded from `input_checkpoint_id`, saved as the output |
| `skip` | no trainer at all | loaded from `input_checkpoint_id` |

Every task scores, in all three cases (`operation.score` is always true). So the
model has to outlive a single round, and the agent keys it on the checkpoint id
the server named rather than guessing from `model.id` or from what happens to be
on disk — the METRICS_GUIDE is explicit that a node must not infer this.

The corpus is server-held (contract 0.7.0): a task names a shard, and the node
reads the texts and the current oracle labels from `/shards/…`. Nothing is
provisioned locally.

## Setup

Needs [uv](https://docs.astral.sh/uv/). The pinned Python is 3.12 — not a
preference: `fasttext-community` publishes wheels for 3.9–3.13 only, and the
system interpreter on the dev machine is 3.14. Linux and Windows are both
supported; fastText ships `win_amd64` wheels for 3.10–3.13.

```bash
uv sync
uv run pytest          # offline; live tests are opt-in, see below
```

## Running a node

```bash
uv run node-agent info --cores 2 --ram 4      # what would happen, no network
uv run node-agent run  --cores 2 --ram 4      # enrol and work until stopped
```

No credentials are needed. The contract has no enrollment token, no per-node
secret and no request signing (CONTRACT.md §2.1) — a `node_id` is the whole of
a node's identity, and `GET /nodes` publishes every one of them. Fine on a
trusted demo network; it is the control plane's call, not ours.

`--cores` and `--ram` are the promise the node makes about the machine, and the
agent keeps it three ways (see *Resource budgets* below). Everything else has a
usable default; `node-agent run --help` lists the rest. Useful ones:

| Flag | What it is for |
|---|---|
| `--once` / `--max-tasks N` | one round, or N, then exit. CI and demos |
| `--dry-run` | run the whole pipeline but never ack or submit |
| `--mode auto\|train\|score` | `auto` obeys `operation.train`; the others force one policy for debugging |
| `--log-level debug` | request bodies, resolved hyperparameters, applied caps |

There is no dataset flag. The corpus is server-held, so a round names a shard and
the node reads it — see *Server-held workloads* below.

Other subcommands:

- **`node-agent selftest`** — uploads a synthetic shard, starts a one-node
  campaign over it, and runs the round start to finish. The operator endpoints
  are open, so this needs nothing from the server's operator. It is also the
  demo. It has to be a campaign rather than a bare round: oracle labels are
  placed by a campaign's schedule, and a shard no campaign has touched has
  nothing to train on.
- **`node-agent reset --identity|--models|--journal`** — forget local state.

The agent **never closes a round**. That is the operator's call, and it is
enforced by `ControlPlaneClient` having no such method rather than by anyone
remembering not to call it.

### Where state lives

Identity, trained models, the task journal and unsent payloads go to
`%LOCALAPPDATA%\node-agent` on Windows, `$XDG_STATE_HOME/node-agent` (else
`~/.local/state/node-agent`) elsewhere. `--state-dir` overrides it. There is no
credential in there to protect — just the `node_id` the server assigned, the
task journal, the trained models and any payload still waiting to be sent.

## Running a node from the desktop app

```bash
uv sync --extra gui
uv run node-gui
```

PyQt5, Linux and Windows. The window is the same node as `node-agent run`, with
the same state directory — so an operator who starts here and an operator who
starts from the terminal are the same node, not two.

It is styled from the control room dashboard's own `:root` tokens — the same
palette, stat tiles, status pills and step bars — so the node and the panel it
reports to read as one product. Light and dark follow the desktop.

- **Power presets.** Small, medium and large mean a quarter, a half and
  all-but-one-core of *this* machine, resolved against what it actually has.
  Each segment's tooltip shows the numbers before you commit to them.
- **Configure…** sets exact cores and RAM. Nothing else is behind that button:
  the server URL is known up front (override it with `NODE_AGENT_SERVER`) and
  everything else keeps the agent's own defaults.
- **Round progress.** Every round of the run gets a card — its id, a three-step
  bar for acked → training → submitted, a live timer, and what it produced
  (`1480 scores · 52 s`). A round that fails turns red with the reason on it;
  the agent never reports a failure as an event, so it is inferred from the
  counter moving while a round is still open ([`gui/rounds.py`](src/node/gui/rounds.py)).
- **Closing the window** leaves the node running in the tray. Quitting from the
  tray menu stops it — and kills the training child on the way out, so nothing
  is left burning cores after the window is gone.
- **Stop** is graceful: the loop finishes the round it is on, which can take
  minutes inside a fastText training run. After ten seconds the window offers a
  forced stop, which kills the training child; that round is then marked failed
  and retried later rather than lost.

There is no log pane. The agent's log goes to `node-gui.log` in the state
directory (rotating, 5 MB × 3) and to stderr when there is one.

On a desktop with no system tray — GNOME under Wayland without an AppIndicator
extension — the window says so and closing it quits, because a running node
with no way back to its window is worse than a stopped one.

The GUI drives `AgentLoop` directly on a worker thread rather than shelling out
to `node-agent`, so it gets the agent's `Reporter` seam, its `AgentView`
snapshots and its `request_stop()` for free.

## Using the core alone

```bash
# Round 1 -- train on golden + teacher scores, persist to state/demo
uv run node-clf train --golden data/tiny_golden.jsonl --model-dir state/demo

# Rounds 2 and 3 -- score with the model round 1 left behind
uv run node-clf score --model-dir state/demo --input data/new.jsonl --out scores.json
```

`--json` makes `train` emit the report as JSON on stdout (fastText's progress
goes to stderr, so the JSON stays clean). `node-clf train --help` lists the rest.

## Input and output

This is the `node-clf` file interface, not the wire format — the agent reads its
corpus from `/shards/…`, where chunks are `{chunk_id, text}` and labels are
integers keyed by `chunk_id`. Input here is JSONL, with field names read
leniently — `text`/`content`/`document`/`body`,
`label`/`score`/`target`/`class`, `id`/`sample_id`/`doc_id`/`uid`:

```json
{"id": "doc-1", "text": "...", "label": "3"}
```

Each scored document comes back as:

```json
{
  "sample_id": "doc-1",
  "label": "3",
  "probs": {"1": 0.01, "2": 0.04, "3": 0.88, "...": "..."},
  "expected_score": 3.12,
  "entropy": 0.47,
  "margin": 0.81
}
```

The full distribution is returned deliberately. Active learning works by
letting the server choose what to label next, and that decision needs the
classifier's uncertainty — `entropy` and `margin` are there for exactly that.
An argmax label alone throws it away.

`expected_score` is `Σ pₖ·k`, which assumes the ten labels are an ordinal
scale rather than ten unrelated categories. It is `null` for non-numeric
labels.

## Things worth knowing about the agent

**Resource budgets are kept three ways.** `--cores 2 --ram 4` is a promise, and
one lever alone would not keep it:

1. *The work is shaped to fit.* The n-gram table's ceiling is derived from the
   RAM budget, so the round produces a model the node can hold. This is the
   lever that should do all the work — it prevents rather than punishes.
2. *The machinery is constrained.* `TrainConfig.thread`, the `OMP_NUM_THREADS`
   family, and a CPU affinity mask. On Linux the mask starts from
   `sched_getaffinity`, so the agent composes correctly inside a cpuset or a
   container instead of claiming cores that were never ours.
3. *It is capped hard.* `RLIMIT_AS` on Linux, a job object with
   `JOB_OBJECT_LIMIT_PROCESS_MEMORY` on Windows — applied by the training
   child to itself, since its pid never reaches the agent (it is spawned inside
   `classifier._run_isolated`). The budget reaches it through the environment,
   which the child inherits.

A breach is a dead training run, not a dead machine, and it looks the same on
both platforms. Measured on the 300-document set with `--ram 2`: the training
child peaked at 291 MiB against a 1.6 GiB share, and the model was 11.4 MiB.

**The held-out curve is the payload that actually matters.** `agg_stats`
carries the sixteen counters of CONTRACT.md §3.5.1 — `ho_n`, `ho_good`, and
`ho_n_qXX` / `ho_good_qXX` for q ∈ {1, 2, 5, 10, 20, 30, 50}% — and they are
what calibrated pooling and the Reliability Gate run on. A node that omits them
is scored `unknown` and dropped from the top-k by default, so the round is
wasted. They travel as counts and nothing else: no documents, no scores, no
labels. A partial curve is a 422, so the agent emits all sixteen or none.

"Good" means the oracle's verdict. The round is meant to supply the rubric;
until it does, the node uses the midpoint of the label scale, overridable per
round with a `good_label_min` param.

**`eval_spearman` is measured on data the shipped weights never saw.** This is
subtle enough to be worth stating. `TrainConfig.retrain_on_full` defaults to
true, so the model returned by `train()` has seen the core's *own* eval split;
computing the correlation there would report memorisation. Since
`eval_spearman` is exactly what the server's reliability gate consumes, the
agent holds out 15% of the labels itself, trains on the rest, and measures on
its own slice. The holdout is seeded from the round id via CRC32 rather than
`hash()` — Python's string hash is salted per process, so `hash()` would
reshuffle the split on every restart and break crash resume.

**A crash never costs a retrain.** The finished submit body is written to disk
before the POST and deleted only once delivery is confirmed. A crash in that
window leaves a complete payload that the next start simply sends. Training is
the only expensive step in a round, so this is the piece of robustness that
actually pays.

**A lost submit response is resolved by asking, not resending.** Resubmitting
into an open round is safe — it replaces the previous one and bumps its
revision — but the body can be megabytes, so on a dropped connection the agent
re-reads `/tasks` (which lists rounds "not yet submitted") and treats absence
as delivery. A 409 means the operator closed the round: permanent, never
retried.

**A metric the node cannot compute is left out, not invented.** The server then
answers `missing_metrics`, and only at that point does the agent resubmit with
zeros for exactly those keys, saying loudly that it did. That order puts the
honest attempt first. Sending `null` instead is not an option: the server
rejects a null for any key but `eval_spearman` and `proxy_lr`.

**Failures are classified, not just counted.** Unusable data (no labels, one
class) is permanent and never retried; fastText's stochastic NaN is transient;
a dead training worker is read as the RAM budget and retried once. One poisoned
round can never take the node down.

**Server-held workloads.** Contract 0.7.0 moved the corpus to the control plane.
A task names a shard; the node reads the texts from
`GET /shards/{id}/chunks` and the current oracle labels from
`GET /shards/{id}/labels`, and joins them on `chunk_id`. Chunk ids come from the
server and are never recomputed — a locally hashed id would not match the pool
the server is ranking.

Nothing is substituted. An empty or unreadable shard fails the round
permanently, because scoring chunk ids the control plane did not assign is worse
than failing: in a sharded campaign the server concatenates rankings without
deduplicating, so one wrong partition corrupts every other node's share of the
budget. Note the trap this guards against — an unknown shard answers `200` with
an empty list, not `404`, so emptiness is the only signal there is.

**Campaigns: sharded and experts.** A campaign round carries its routing in
`params` — `mode`, `partition`, `n_partitions`, `n_labels` — and the two modes
differ in a way that matters:

- `sharded`: the node fetches, trains on and scores **only** its own partition.
- `experts`: the node trains on its own domain's labels but must return a score
  for **every** chunk in the pool; `advance` rejects a submission with ids
  missing. So truncation to the 200 000-score cap is refused in this mode — a
  failed round is better than a payload that can only be rejected.

Partition membership is a formula computed identically on both sides,
`int(chunk_id[:8], 16) % n_partitions`, which is what lets the node narrow the
shard-wide label map to its own domain without another call. A live test checks
the server still agrees. Routing that does not parse fails the round rather than
being guessed.

**The model lifecycle is instructed, not inferred.** `task.operation` says
`fresh`, `continue` or `skip`, and `--mode auto` simply obeys it; the heuristics
this agent used to carry are gone, as the METRICS_GUIDE requires. Checkpoints
stay on the node — the control plane passes only names, namespaced
`(node_id, checkpoint_id)` — and each one gets a `manifest.json` beside the
weights recording the round, the recipe, the chunk ids trained on and how far the
round got, so a restart can recognise a familiar round instead of retraining it
or submitting it twice.

One divergence is declared rather than hidden: **`continue` is executed as a full
retrain**, because fastText supervised models have no warm start. Since campaign
labels are cumulative, retraining on the round's whole label set is what a
continuation would converge to here — but it is not what was asked, so the
submission carries `train_continued_as_fresh: 1.0` and the agent says so in its
log. `skip` against a checkpoint this node does not hold trains instead and
reports `train_fallback_no_checkpoint: 1.0`, on the grounds that scores from a
new model beat no scores at all.

**Telemetry follows the METRICS_GUIDE.** The work loop updates a local snapshot;
the heartbeat thread posts it on the server's own interval, never more often. A
busy beat carries `status`, `round_id`, the `stage`
(`downloading` → `training` → `scoring` → `uploading`) and numeric `load`
gauges — `progress_pct`, `docs_processed`, `docs_total`, `docs_per_sec`, `eta_s`,
`cpu_pct`, `ram_pct`. Finishing a task sends `idle` with the round, the stage and
every gauge from it cleared, plus one immediate beat so the dashboard does not
show a finished round as still running.

Progress is per phase, not per task, so scoring does not start at the 100% that
training ended on. `load` is capped at the 32 finite numbers the schema allows,
and gauge names are a fixed vocabulary — never built from a document, a path or a
host, which is both the guide's privacy rule and its Prometheus cardinality one.
`train_loss` is omitted: fastText does not report one through this wrapper, and a
made-up number on that graph would be worse than a gap.

### Live tests

```bash
NODE_AGENT_LIVE=1 uv run pytest -m integration
NODE_AGENT_SERVER=https://... NODE_AGENT_LIVE=1 uv run pytest -m integration
```

Skipped by default so `uv run pytest` stays offline, and skipped rather than
failed when the tunnel is down. Most of them are a drift canary: they assert
that the live `openapi.json` still has the field names, the ±1 bound on
`eval_spearman` and the `exclusiveMinimum: 0` on `proxy_lr` this agent was
built against, that nothing has become credential-gated again, and that the
server's own rejection message still names exactly the sixteen `ho_*` keys we
send. When the server's author changes something, that is where it surfaces.

## Things worth knowing

**Ordinal metrics.** If the labels are a 1–10 scale, accuracy is misleading:
it punishes an off-by-one exactly as hard as an off-by-seven. The training
report therefore also carries MAE and quadratic weighted kappa, which see the
distance between classes. They are omitted when labels are not numeric.

**Hyperparameters scale with the data.** The data volume is not known up front,
and fastText's stock defaults are tuned for large corpora:

- *Epochs.* 25 epochs scores **0.20** accuracy on the 260-sample smoke set;
  300 scores **0.975**. What stays roughly constant across corpus sizes is the
  update budget, not the pass count, so epochs are derived as
  `target_updates / n_samples`, clamped to 5–300. `--epoch` overrides.
- *Buckets.* The stock 2,000,000 n-gram buckets cost `dim × 4` bytes each —
  an **800 MB** model for 300 documents. Buckets are sized against the corpus
  token count instead, which brings the same model to **14 MB**.

**Autotuning** kicks in at 2000+ samples, searching against its own split so
the reported metrics stay honest (train / tune / eval, three ways). It
optimizes validation score and ignores model size — 2500 documents produced a
**262 MB** model. `--max-model-size 50M` caps it, at a real cost in accuracy:
on the same data a 10M cap took accuracy from 1.00 to 0.69, because the
quantization search eats into the time budget. Give it a longer
`--autotune-seconds` if you cap the size.

**Determinism.** fastText trains with asynchronous SGD across threads, so runs
are not bit-reproducible. `--threads 1` makes them so; the tests use it.

**Training runs in a spawned process.** fastText aborts at random with
`RuntimeError: Encountered NaN`, and the cause is accumulated state in a
long-lived interpreter rather than anything in the parameters. What the
investigation ruled out:

- **not the hyperparameters** — sweeps over learning rate (0.05–1.0), epochs
  (50–400), bucket counts (5k–2M) and three corpus sizes all came back clean;
  over 500 trainings at settings that fail inside the test suite;
- **not a memory leak** — RSS plateaus and stays flat over repeated cycles;
- **not the published wheel or its `-O3 -funroll-loops` flags** — a build from
  the C++ sources with gcc 16 fails at *exactly the same rate*, 3 runs in 8.

What does reproduce it is depth into a process: failures only ever appear after
many trainings in one interpreter, never in a fresh one. So `train()` runs the
whole fastText pipeline in a separate interpreter that has never loaded
fastText (`isolate_training`, default on) and loads the finished model back.
The suite went from 6 failures in 16 runs to 0 in 30.

The worker is a plain `subprocess` running `python -m node.core._worker`, not
`multiprocessing`. The "spawn" start method re-imports the parent's `__main__`
in the child, which fails outright when `__main__` is not a file — a REPL, a
notebook, `python -c` — and spawns processes without end when a script calls
`train()` outside an `if __name__ == "__main__"` guard. The core has to be safe
to embed, so it avoids that machinery entirely.

Isolation costs a process start per training — the test suite went from 8 to
11 seconds — and it is worth it. Turn it off only to patch fastText in-process,
as a few tests do.

A retry on this one error is kept as well (`train_attempts`, default 3); any
other `RuntimeError` propagates immediately.

Quantization has its own version of the problem: product quantization runs
k-means over the weight subvectors and raises the same NaN on small corpora.
A retry is not appropriate there, so failure degrades instead — the full-size
model ships and the report carries a warning.

**Never pass `thread=0` to fastText.** It divides work by the thread count and
kills the process with SIGFPE, deterministically — not an exception, a signal.
`TrainConfig.thread` uses 0 to mean "one per core" and `resolve_thread()`
turns it into a real number before it reaches fastText.

## Layout

```
src/node/core/          the classifier
  text.py         normalization -- defuses fastText's line-format footguns
  dataset.py      lenient parsing, training files, stratified splits
  metrics.py      accuracy, macro-F1, MAE, QWK, Spearman (numpy only)
  classifier.py   train / score / save / load
  limits.py       self-imposed RAM and CPU caps, applied by the training child
  _worker.py      the training subprocess

src/node/agent/         the node
  cli.py          node-agent: run / info / selftest / reset
  loop.py         heartbeat thread + poll -> ack -> run -> submit
  runner.py       one round: fetch, split, train, score, aggregate (no HTTP)
  api.py          the control plane client (with no way to close a round)
  shards.py       the server-held corpus: chunks, labels, partition routing
  telemetry.py    the heartbeat snapshot: stages and load gauges
  datasets.py     local JSONL, chunk ids, the synthetic shard selftest uploads
  resources.py    hardware detection, budgets, derived hyperparameters
  state.py        identity, task journal, outbox, checkpoints and manifests
  models.py       wire types; unknown fields preserved in `extra`
  reporting.py    the seam the GUI attaches to

src/node/gui/           the desktop app (PyQt5, optional extra)
  app.py          node-gui: bootstrap, single instance, the quit sequence
  worker.py       AgentLoop on a QThread; the Reporter -> Qt signal bridge
  mainwindow.py   the window and its four-state machine
  advanced.py     exact cores and RAM
  tray.py         the tray icon, and the desktops that have none
  widgets.py      stat tiles, status pills, step bars, round cards
  theme.py        the dashboard's design tokens, as a Qt stylesheet
  rounds.py       one round's progress; infers the failures nobody reports
  presets.py      small / medium / large, as a pure function of the machine
  settings.py     gui.json next to identity.json; settings -> AgentConfig
  uistate.py      what the buttons say in each state
  loadmeter.py    live CPU and RAM that do not fight the heartbeat over psutil
  logsetup.py     the `node` logger -> a rotating file in the state directory
  icon.py         the status-coloured icon, painted rather than shipped

src/node/cli.py         node-clf
scripts/make_tiny_golden.py   regenerates the synthetic smoke dataset
```

`runner.py` does no I/O beyond local disk and `api.py` does nothing but I/O, so
a round can be tested without a server and a server without a round.

`data/tiny_golden.jsonl` is 300 synthetic English documents across 10 labels
with the signal planted deliberately: a weak score there means the pipeline is
broken, not that the task is hard. Expect accuracy ≈ 0.97.

## Open questions for the server's author

In rough order of how much they block us.

1. **`POST /campaigns` returns 500.** Reproducible on the live control plane with
   only the required fields, on 0.7.0 through 0.11.0, for every `mode`,
   `strategy` and `schedule` we tried. Since a campaign is what places oracle
   labels, a shard no campaign has touched has nothing to train on — so this
   currently blocks any real labelled round, and `node-agent selftest` stops
   there. Everything before it works against the live server: enrolment, shard
   upload, shard reads, partition routing, heartbeats with stages and gauges.
2. **`POST /shards/{id}/chunks` sometimes stores nothing.** It answers `201` with
   `received: 40, new: 0, total: 0`, and the subsequent `GET` returns an empty
   shard — non-deterministically, for payloads that had just succeeded with the
   same shape. A node cannot tell this from a shard that was never loaded,
   because an unknown shard also reads as `200` with an empty list. The live
   shard canaries skip with the counts when they hit it rather than reporting it
   as a client failure.
2. **The "good document" rubric.** The held-out curve counts documents the
   node's oracle called good, and §3.5.1 says the rubric comes from the round —
   but no field carries it. A campaign has `good_min` and applies it server-side
   at finalize; we use it when a round passes it, else the midpoint of the label
   scale, overridable with `good_label_min`. Since the gate compares nodes on
   precision, every node needs the *same* rubric or the comparison is
   meaningless. Could `good_min` be put into task `params`?
3. **`continue` against a backend with no warm start.** fastText supervised
   models cannot resume from weights, so we execute `operation.train=continue` as
   a full retrain on the round's cumulative labels and report
   `train_continued_as_fresh: 1.0`. Is that the intended reading for a
   classifier node, or should such a node decline `continue` outright?
4. **`proxy_lr`.** The spec's example is `1e-05`, which is a *neural* learning
   rate; fastText's own default is 0.5 and below ~0.05 it does not train at
   all. We clamp, echo the requested value back as `proxy_lr`, and report what
   we actually used as `proxy_lr_effective`. Were those round params written
   with a LoRA node in mind?
5. **Score semantics.** We submit `expected_score` = `Σ pₖ·k`, an ordinal
   quality estimate on the label scale, on the reading that `budget_k` selects
   the *best* chunks. All three real `topk` normalizers are scale-invariant so
   the range should not matter — but if `normalize=none` is ever used, it does.
6. **The held-out split.** CONTRACT.md §4.7 says `/shards/…/labels` does not mark
   one and an external workflow must supply it. Nothing in the task names one
   either, so the node carves its own 15% out of the labels it was given. Every
   node doing that independently is fine for calibration but means the gate
   compares curves measured on different slices.
7. Whether the labels are an ordinal scale or unrelated categories — this
   decides whether MAE, QWK and `expected_score` mean anything.
