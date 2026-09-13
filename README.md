# Proxy Mesh — control plane (MVP)

Control plane for distributed proxy-model work. Nodes contribute compute only; the control plane
assigns server-held workloads, either sharding a large pool for throughput or combining domain
experts for quality. It receives Egress-Gate payloads, stores them in SQLite and produces a global top-k.

Full contract (fields, limits, error codes): [CONTRACT.md](CONTRACT.md). Node-side telemetry setup:
[METRICS_GUIDE.md](METRICS_GUIDE.md).

```
node                                                 control plane
 │ POST /nodes/handshake  (compute capabilities)  ──▶ registry: node_id
 │ POST /nodes/{id}/heartbeat   every N sec       ──▶ online / idle / busy
 │                                                    POST /rounds  (operator: participants,
 │                                                                   model, dataset, metrics)
 │ GET  /tasks/{id}                               ◀── rounds this node should work on
 │ GET  /shards/{dataset}/chunks + /labels       ◀── assigned texts and oracle labels
 │ POST /tasks/{id}/{round}/ack                   ──▶ participant: assigned → accepted
 │   … local pipeline: raw → curated → proxy score …
 │ POST /tasks/{id}/{round}/submit  (Egress Gate) ──▶ participant: → submitted
 │                                                    GET /rounds/{round}/topk: calibrated pooling,
 │                                                        Reliability Gate on held-out precision
```

The node never accepts inbound connections: it only calls out. That is what a bank or telco
perimeter allows, and it means only the control plane needs a reachable address.
Node lifecycle API lives under `/nodes/…` and `/tasks/…`; nodes also read their assigned
server-held workloads through `GET /shards/…`. Operator API is under `/rounds/…`, `/campaigns/…`,
and the write side of `/shards/…`. Observability is exposed by `/health`, `/metrics*`, and `/dashboard`.

## Who owns what

| Control plane | Node |
|---|---|
| Registry of nodes and their liveness; server-held shards and task assignment | Declares hardware, software and model kinds it can bake |
| Creates rounds: participants, dataset, model recipe, explicit train/score operation, required metrics and params | Pulls its tasks, follows `operation`, trains or loads the named checkpoint, then scores |
| Provides server-held workloads and campaign labels; calibrated pooling, Reliability Gate and cross-node top-k | Fetches its assigned texts and labels, trains/scores, and optionally reports a curve from a separately prepared labelled evaluation split |

## Run

```bash
./run.sh                    # creates .venv on first run, listens on 0.0.0.0:8100
.venv/bin/pytest -q         # tests
```

Swagger UI: http://127.0.0.1:8100/docs

Exposing via a tunnel: point it at IPv4 explicitly — `cloudflared tunnel --url http://127.0.0.1:8100`
(`localhost` may resolve to `::1`, which `0.0.0.0` does not listen on). `PM_HOST=::` listens on IPv6 only.

## Endpoints

| Method | Path | Who | Purpose |
|---|---|---|---|
| POST | `/nodes/handshake` | node | Enroll (or re-register with `node_id`); returns `node_id`, `heartbeat_interval_s` |
| POST | `/nodes/{node_id}/heartbeat` | node | `{status: idle\|busy, stage?, round_id?, load{}}`; returns `pending_tasks` |
| GET  | `/tasks/{node_id}` | node | Open rounds assigned to this node, not yet submitted |
| POST | `/tasks/{node_id}/{round_id}/ack` | node | "Started working"; idempotent |
| POST | `/tasks/{node_id}/{round_id}/submit` | node | Egress Gate payload |
| POST | `/rounds` | operator | Create a round: participants, model, metrics, params |
| GET  | `/rounds/{round_id}` | operator | Round spec, participants with status, submissions |
| POST | `/rounds/{round_id}/close` | operator | Stop accepting submissions; tasks disappear |
| GET  | `/rounds/{round_id}/topk?k=&normalize=calibrated\|zscore\|rank\|none&include_untrusted=&include_unknown=` | operator | Global top-k across nodes |
| GET  | `/nodes` | operator | Registry with specs, last heartbeat, `online` |
| GET | `/shards` | operator | Dataset inventory with document and label counts |
| POST | `/shards/{shard_id}/chunks` | operator | Load a server-held campaign pool |
| POST | `/imports/parquet/inspect` | operator | Stage a raw Parquet file and inspect its columns |
| POST | `/shards/{shard_id}/imports/parquet` | operator | Import mapped Parquet text, IDs and optional labels |
| GET | `/shards/{shard_id}/preview?offset=&limit=&q=` | operator | Searchable, paginated documents joined to labels |
| GET | `/shards/{shard_id}/chunks?partition=&n_partitions=` | node | Read the whole pool or one deterministic partition |
| GET | `/shards/{shard_id}/labels` | node | Current oracle labels for training |
| POST | `/campaigns` | operator | Start a `sharded` or `experts` campaign |
| GET | `/campaigns/{campaign_id}` | operator | Campaign progress and final selection |
| POST | `/campaigns/{campaign_id}/advance` | operator | Consume a completed round and continue/finalize |
| GET  | `/health` | anyone | Liveness |
| GET  | `/metrics` | anyone | Prometheus metrics, including generic heartbeat `load` gauges |
| GET  | `/metrics.json` | anyone | Dashboard-shaped metrics and latest client telemetry |
| GET  | `/telemetry/history?node_id=&since=&limit=` | anyone | Persisted heartbeat history, oldest-first per node |
| GET  | `/dashboard` | anyone | Built-in control room and dataset browser/import UI |

Request/response shapes, limits and error codes: [CONTRACT.md](CONTRACT.md).

## Node SDK

`node_sdk/client.py` is stdlib-only — the node vendors this one file and never imports server code:

```python
from node_sdk.client import ControlPlane, heldout_curve

cp = ControlPlane("https://cp.example", state_file=".pm_node.json")
cp.handshake(name="bank-a", hardware={...}, model_kinds=["classifier"],
             software={"agent_version": "0.11.0"})
beat = {"status": "idle"}
cp.start_heartbeat(lambda: beat)
for task in cp.tasks():
    cp.ack(task["round_id"])
    beat.update(status="busy", round_id=task["round_id"])
    op = task["operation"]
    model = new_model(task["model"]) if op["train"] == "fresh" else load_checkpoint(op["input_checkpoint_id"])
    if op["train"] != "skip":
        train(model, task["dataset_id"], task["params"])
        save_checkpoint(model, op["output_checkpoint_id"])
    scores, stats, held_scores, held_good = score(model, task["dataset_id"], task["params"])
    cp.submit(task["round_id"], scores, {**stats, **heldout_curve(held_scores, held_good)})
    beat.clear(); beat["status"] = "idle"
```

`task.operation` is authoritative: `fresh` initializes the `model.id` recipe without an input
checkpoint, `continue` loads `input_checkpoint_id` and keeps training, and `skip` loads it and only
scores. Training operations save to `output_checkpoint_id`. Checkpoint ids are node-local; namespace
them by `(node_id, checkpoint_id)`. A repeated `round_id` is a resume/retry, not a new training run.

While a task is running, update the heartbeat state with numeric telemetry. The dashboard treats
these names as stable; any additional numeric keys are still exported through Prometheus:

```python
beat.update(status="busy", stage="training", round_id=task["round_id"], load={
    "progress_pct": 42.5,
    "docs_processed": 8500,
    "docs_total": 20000,
    "docs_per_sec": 127.4,
    "eta_s": 90,
    "train_loss": 0.31,
    "cpu_pct": 73.5,
    "ram_pct": 61.0,
    "gpu_util_pct": 92.0,
    "gpu_mem_pct": 78.0,
})
```

The control plane stores every heartbeat in SQLite, so client graphs survive browser and server
restarts. A copy-ready, thread-safe worker example and metric semantics are in
[METRICS_GUIDE.md](METRICS_GUIDE.md).

The task carries addressing, not text: `dataset_id` names the server-held shard and campaign tasks
also carry `mode`, `partition`, and `n_partitions` in `params`. Fetch the assigned corpus and current
training labels before running the pipeline:

```bash
curl "$CP/shards/$DATASET/chunks?partition=$PARTITION&n_partitions=$N_PARTITIONS" -o chunks.json
curl "$CP/shards/$DATASET/labels" -o labels.json
```

For `sharded`, train and score the returned partition. For `experts`, filter labels to the node's
partition for training but fetch and score `/shards/$DATASET/chunks` without query parameters. Labels
are keyed by `chunk_id`; join them to chunks locally. The stdlib SDK currently wraps lifecycle and
submission calls only, so shard reads use these HTTP endpoints directly.

## Dataset workspace

The `Datasets` tab at `/dashboard` lists every server-held shard, document/oracle-label counts and
label coverage. Selecting one opens a server-paginated document browser with case-insensitive search
over chunk ID and text, so large shards are never loaded into the browser in one response.

`Import dataset` accepts JSON (`[{"text": ...}]` or `{"chunks": [...]}`), JSONL, line-delimited TXT,
and Parquet. JSON/JSONL/TXT objects may provide a valid lowercase-hex `chunk_id`; otherwise the
browser uses SHA-256 of the text, removes duplicate IDs and uploads in batches of 5,000.

Parquet is streamed to the server once for schema inspection. The UI then lets the operator choose a
string text column, use a string ID column or generate SHA-256 IDs, and optionally import a numeric
oracle-label column. Labels can already be binary `0/1` or converted with a threshold (FineWeb-Edu's
default auto-mapping is `score >= 3`). The final import is atomic; duplicate rows are skipped, empty or
over-20,000-character texts are reported as skipped, and imported labels show their Parquet source in
the dataset browser.

## Auth

None (hackathon build): no enrollment token, no per-node secret, no request signing — any caller can
act as any `node_id`. Fine on a trusted local/demo network; do not expose this control plane on an
untrusted one without adding auth back. Details: CONTRACT.md §2.1.

## Pooling and the Reliability Gate

When a workflow has a separately prepared labelled evaluation split, a node may report, as counts,
how its proxy ranks that split: `agg_stats.ho_n`, `ho_good`, and for each top-quantile q ∈
{1, 2, 5, 10, 20, 30, 50}% the held-out docs in the top q (`ho_n_qXX`) and the good ones among them
(`ho_good_qXX`). `node_sdk.client.heldout_curve` builds them. The current API does not allocate or
identify a held-out split by itself; it trusts a complete curve supplied by the node.

- **Calibrated pooling** (`normalize=calibrated`, default): a chunk in the top fraction f of its node is scored
  by the share of good docs expected at f on that node's held-out curve, so the budget goes where good docs are
  expected and a node with a richer corpus or a sharper proxy gets more of it. `zscore` / `rank` give every node
  the same share of its submitted scores; `none` pools raw scores.
- **Reliability Gate**: after allocation a node's cutoff is q = selected / n_scores. The node is `trusted` when
  the Wilson 95% lower bound of held-out precision above q (judged on at least 30 held-out docs) reaches
  `PM_GATE_MIN_PRECISION` (0.2). Untrusted nodes are dropped and the budget is reallocated; nodes without a
  curve are `unknown`. `include_untrusted=true` reports the gate without enforcing it.

At submit, `trust` is `pending` (curve present, decided at top-k) or `unknown`; `eval_spearman` is informational.
Details: [control_plane/pooling.py](control_plane/pooling.py), CONTRACT.md §3.5.1 and §4.4.

## Sharded and expert campaigns

`POST /campaigns` takes `mode: "sharded" | "experts"`, `train_mode: "fresh" | "continue"` (default
`fresh`) and an ordered `node_ids` list. Chunk placement
is stateless: `int(chunk_id[:8], 16) % len(node_ids)`, so `node_ids[i]` always owns partition/domain `i`.
The cumulative `schedule` is per partition, not global.

- `sharded`: node `i` trains and scores only partition `i`. Its task points to
  `GET /shards/{id}/chunks?partition=i&n_partitions=N`; the server z-score normalizes and concatenates
  the non-overlapping rankings.
- `experts`: node `i` trains on labels from domain `i`, then scores the full shard. The server requires
  every expert to cover every pool chunk and ranks by the mean score across experts.

Task routing is explicit in `task.params`: `mode`, `partition`, `n_partitions`, plus `n_labels` for the
current round. Model lifecycle is explicit in `task.operation`; with campaign `continue`, round 1 is
necessarily `fresh` and every later round consumes the preceding round checkpoint. `/labels` returns
the shard-wide map; each node filters its training labels with the same
partition function. Run both paths locally with `tools/demo_campaign.py --mode sharded|experts`.

## Try it without a real node

```bash
./run.sh &
.venv/bin/python tools/fake_node.py --name bank-a --noise 0.3   # prints node_id; --noise 3.0 fails the gate
curl -s localhost:8100/nodes | python3 -m json.tool
curl -s -X POST localhost:8100/rounds -H 'Content-Type: application/json' -d '{
  "round_id": "r1", "budget_k": 500, "model": {"kind": "classifier", "id": "quality-clf-v1"},
  "metrics": ["eval_spearman", "n_dedup_dropped"], "params": {"proxy_lr": 1e-5},
  "participants": [{"node_id": "<node_id>", "dataset_id": "synthetic-wave1"}]}'
# the fake node picks the task up, acks and submits within --poll seconds
curl -s localhost:8100/rounds/r1 | python3 -m json.tool
curl -s "localhost:8100/rounds/r1/topk?k=500" | python3 -m json.tool | head -40
```

## Simulation on public data

`tools/proxy_rounds_sim.py` runs rounds end to end through a control plane on
[FineWeb-Edu annotations](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations), split into
nodes by URL TLD, with the Llama-3-70B score as the oracle: global DCLM fastText (v0), then local fastText trained on
random or active-learning labels (v1, v2). The report measures top-k quality against the oracle. It needs the
[DCLM fastText model](https://huggingface.co/mlfoundations/fasttext-oh-eli5) and the parquet shards on disk (the
script downloads nothing) and `.venv/bin/pip install -r tools/requirements-sim.txt`.

```bash
.venv/bin/python tools/proxy_rounds_sim.py --url http://127.0.0.1:8100 --shards data/fineweb_edu_ann/*.parquet --model data/models/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin
```

## Config

| Env | Default |
|---|---|
| `PM_DB_PATH` | `data/control_plane.sqlite3` |
| `PM_AUDIT_LOG` | `data/submissions.jsonl` (metadata only, no scores) |
| `PM_GATE_MIN_PRECISION` | `0.2` (Wilson lower bound of held-out precision at a node's cutoff) |
| `PM_HEARTBEAT_S` | `3` (node offline after 3 missed beats) |
| `PM_TELEMETRY_RETENTION_S` | `604800` (7 days of persisted heartbeat samples) |
| `PM_TELEMETRY_HISTORY_LIMIT` | `240` (default samples per node returned by history API) |
| `PM_MAX_BODY_BYTES` | 32 MiB |
| `PM_MAX_PARQUET_BYTES` | 512 MiB per staged Parquet file |
| `PM_MAX_PARQUET_ROWS` | 500000 rows per staged Parquet file |
| `PM_PARQUET_UPLOAD_TTL_S` | 3600 seconds before an unused staged upload expires |
| `PM_GOLDEN_LABELS_PATH` | unset; JSON map `chunk_id -> label`, read at startup, enables `StaticOracle` |
| `PM_MOCK_ORACLE_GOOD_RATE` | `0.1`; fallback `MockOracle` positive rate |
| `PM_HOST` / `PM_PORT` | `0.0.0.0` / `8100` (run.sh) |

Out of scope: queueing / push (nodes poll over HTTP), TLS termination, operator auth, long-term telemetry beyond SQLite.
