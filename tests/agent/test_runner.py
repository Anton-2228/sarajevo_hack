"""The round pipeline, with training stubbed so the whole thing runs instantly."""

import json
import math

import pytest

from node.agent import runner as runner_mod
from node.agent.datasets import chunk_id_for, load_dataset
from node.agent.models import Routing, TaskMode, TaskView, resolve_mode
from node.agent.reporting import NullReporter
from node.agent.resources import Budget
from node.agent.runner import TaskFailed, TaskRunner
from node.agent.shards import Corpus, CorpusUnavailable
from node.agent.state import AgentState
from node.core.types import ScoredSample, TrainReport

LABELS = [str(i) for i in range(1, 11)]

TASK = {
    "round_id": "r1",
    "status": "accepted",
    "model": {"kind": "classifier", "id": "quality-clf-v1"},
    "dataset_id": "d",
    "metrics": ["eval_spearman", "n_dedup_dropped"],
    "params": {"proxy_lr": 0.5, "cutoff_q": 0.1},
    "budget_k": 100,
    "assigned_at": 1.0,
    "ack_url": "/a",
    "submit_url": "/s",
}


class FakeCorpusSource:
    """Stands in for `/shards/…`. The runner does no HTTP of its own."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.asked_for = []

    def fetch(self, task):
        self.asked_for.append(task.round_id)
        if isinstance(self.corpus, Exception):
            raise self.corpus
        return self.corpus


class StubClassifier:
    """Predicts the true label, with a controllable amount of error."""

    def __init__(self, truth: dict[str, str], noise: float = 0.0):
        self.labels = list(LABELS)
        self._truth = truth
        self._noise = noise
        self.report = None
        self.saved_to = None

    def score(self, samples):
        out = []
        for i, sample in enumerate(samples):
            true_label = self._truth.get(sample.sample_id, "5")
            value = float(true_label)
            if self._noise:
                value = max(1.0, min(10.0, value + (self._noise if i % 2 else -self._noise)))
            probs = {label: 0.0 for label in self.labels}
            probs[true_label] = 1.0
            out.append(
                ScoredSample(
                    sample_id=sample.sample_id,
                    label=true_label,
                    probs=probs,
                    entropy=0.0,
                    margin=1.0,
                    expected_score=value,
                )
            )
        return out

    def save(self, directory):
        self.saved_to = directory
        directory.mkdir(parents=True, exist_ok=True)
        return directory


def make_dataset(tmp_path, records):
    path = tmp_path / "datasets" / "d.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return path


def balanced_records(per_label=6):
    return [
        {"text": f"document {label} number {i}", "label": label}
        for label in LABELS
        for i in range(per_label)
    ]


def make_corpus(tmp_path, records, routing=None, score_only_partition=False):
    """A `Corpus` shaped the way `shards` would hand one over.

    Built through the real local loader so that dedup counting and label
    parsing stay honest; only the transport is faked.
    """
    dataset = load_dataset(make_dataset(tmp_path, records), "d")
    routing = routing or Routing()
    to_score = list(dataset.chunks)
    to_train = list(dataset.labeled)

    if routing.is_campaign and routing.n_partitions:
        from node.agent.shards import partition_of

        own = [
            c
            for c in dataset.chunks
            if partition_of(c.chunk_id, routing.n_partitions) == routing.partition
        ]
        # `sharded` is handed only its slice; `experts` sees the pool but may
        # train on its own domain alone.
        to_train = [c for c in own if c.label is not None]
        if score_only_partition:
            to_score = own

    return Corpus(
        shard_id="d",
        to_score=to_score,
        to_train=to_train,
        routing=routing,
        pool_size=len(to_score),
        n_labels_available=len(dataset.labeled),
        n_dedup_dropped=dataset.n_dedup_dropped,
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A runner whose training is a stub; everything else is the real thing."""

    def build(records=None, noise=0.0, corpus=None, **kwargs):
        records = balanced_records() if records is None else records
        truth = {
            chunk_id_for(r["text"]): str(r["label"]) for r in records if r.get("label")
        }
        if corpus is None:
            corpus = make_corpus(tmp_path, records)

        trained_with = {}

        def fake_train(samples, config=None):
            trained_with["calls"] = trained_with.get("calls", 0) + 1
            trained_with["samples"] = list(samples)
            trained_with["config"] = config
            report = TrainReport(
                n_samples=len(samples),
                n_skipped=0,
                n_train=len(samples),
                n_eval=0,
                labels=list(LABELS),
                autotuned=False,
                accuracy=0.9,
                macro_f1=0.88,
                mae=0.3,
                qwk=0.95,
            )
            return StubClassifier(truth, noise), report

        monkeypatch.setattr(
            runner_mod.FastTextClassifier, "train", staticmethod(fake_train)
        )

        # Loading a checkpoint is what `continue` and `skip` turn on, so the stub
        # has to be loadable as well as trainable.
        loaded_from = []

        def fake_load(directory):
            loaded_from.append(directory)
            return StubClassifier(truth, noise)

        monkeypatch.setattr(
            runner_mod.FastTextClassifier, "load", staticmethod(fake_load)
        )
        trained_with["loaded_from"] = loaded_from

        run = TaskRunner(
            state=AgentState(tmp_path / "state"),
            corpus_source=FakeCorpusSource(corpus),
            budget=Budget(cores=2, ram_bytes=2 * 1024**3, core_ids=[0, 1]),
            reporter=NullReporter(),
            **kwargs,
        )
        return run, trained_with

    return build


def run_task(harness_build, task_overrides=None, mode=None, **kwargs):
    run, trained = harness_build(**kwargs)
    task = TaskView.from_dict({**TASK, **(task_overrides or {})})
    # Default to what the task itself says, which is what the loop does.
    return run.run(task, mode or resolve_mode(task, "auto")), trained


def test_every_chunk_is_scored(harness):
    outcome, _ = run_task(harness)
    assert len(outcome.scores) == 60
    # n_chunks counts the corpus; the server requires n_chunks >= len(scores).
    assert outcome.agg_stats["n_chunks"] == 60
    assert outcome.agg_stats["n_chunks"] >= len(outcome.scores)
    assert len({s.chunk_id for s in outcome.scores}) == 60


def test_scores_are_the_ordinal_quality_estimate(harness):
    outcome, _ = run_task(harness)
    # The stub predicts the true label, so scores must span the 1..10 scale.
    values = [s.score for s in outcome.scores]
    assert min(values) == 1.0
    assert max(values) == 10.0


def test_unit_scale_rescales_to_zero_one(harness):
    outcome, _ = run_task(harness, score_scale="unit")
    values = [s.score for s in outcome.scores]
    assert min(values) == 0.0
    assert max(values) == 1.0


def test_the_model_never_sees_the_slice_it_is_measured_on(harness):
    # retrain_on_full means the core's own eval slice ends up in the shipped
    # weights, so eval_spearman has to be measured on a slice we withheld.
    outcome, trained = run_task(harness)
    trained_ids = {s.sample_id for s in trained["samples"]}
    assert len(trained_ids) < 60
    assert outcome.agg_stats["n_eval"] == 60 - len(trained_ids)


def test_holdout_is_reproducible_across_processes(harness):
    # Python's str hash is salted per interpreter; using it would reshuffle the
    # holdout on every restart and break crash-resume.
    first, _ = run_task(harness)
    second, _ = run_task(harness)
    assert first.agg_stats["n_eval"] == second.agg_stats["n_eval"]
    assert runner_mod.stable_seed("r1") == runner_mod.stable_seed("r1")
    assert runner_mod.stable_seed("r1") != runner_mod.stable_seed("r2")


def test_perfect_proxy_scores_spearman_one(harness):
    outcome, _ = run_task(harness)
    assert outcome.agg_stats["eval_spearman"] == 1.0


def test_spearman_stays_within_the_servers_bounds(harness):
    outcome, _ = run_task(harness, noise=3.0)
    rho = outcome.agg_stats["eval_spearman"]
    assert rho is None or -1.0 <= rho <= 1.0


def test_required_metrics_are_always_present(harness):
    outcome, _ = run_task(
        harness, {"metrics": ["eval_spearman", "n_dedup_dropped", "proxy_lr"]}
    )
    for key in ("eval_spearman", "n_dedup_dropped", "proxy_lr"):
        assert key in outcome.agg_stats


def test_proxy_lr_is_echoed_verbatim(harness):
    outcome, _ = run_task(harness, {"params": {"proxy_lr": 0.25}})
    assert outcome.agg_stats["proxy_lr"] == 0.25


def test_a_clamped_learning_rate_reports_both_numbers(harness):
    # The server gets the value it asked for back, plus what we actually used.
    outcome, trained = run_task(harness, {"params": {"proxy_lr": 1e-05}})
    assert outcome.agg_stats["proxy_lr"] == 1e-05
    assert outcome.agg_stats["proxy_lr_effective"] == 0.05
    assert trained["config"].lr == 0.05
    assert any("1e-05" in w for w in outcome.warnings)


def test_non_positive_proxy_lr_becomes_null(harness):
    # The schema has exclusiveMinimum: 0, so the number itself cannot be sent.
    outcome, _ = run_task(harness, {"params": {"proxy_lr": 0.0}})
    assert outcome.agg_stats["proxy_lr"] is None


def test_heldout_curve_uses_the_contracts_exact_keys(harness):
    # CONTRACT.md 3.5.1 names all sixteen. This is what calibrated pooling and
    # the Reliability Gate run on; a node without it is scored `unknown`.
    outcome, _ = run_task(harness)
    stats = outcome.agg_stats
    assert set(runner_mod.HELDOUT_KEYS) <= set(stats)
    assert len(runner_mod.HELDOUT_KEYS) == 16


def test_heldout_curve_is_all_counts_and_nothing_else(harness):
    outcome, _ = run_task(harness)
    for key in runner_mod.HELDOUT_KEYS:
        value = outcome.agg_stats[key]
        assert isinstance(value, int) and not isinstance(value, bool), key
        assert value >= 0, key


def test_heldout_curve_satisfies_the_servers_monotonicity_rules(harness):
    outcome, noise = run_task(harness, noise=2.0)
    stats = outcome.agg_stats
    ho_n, ho_good = stats["ho_n"], stats["ho_good"]
    assert ho_n >= 1
    assert ho_good <= ho_n

    last_n = last_good = 0
    for _, suffix in runner_mod.HELDOUT_QUANTILES:
        n, good = stats[f"ho_n_q{suffix}"], stats[f"ho_good_q{suffix}"]
        assert good <= n
        assert n >= last_n and good >= last_good
        assert n <= ho_n and good <= ho_good
        last_n, last_good = n, good


def test_heldout_curve_counts_are_ceil_of_q_times_n(harness):
    outcome, _ = run_task(harness)
    ho_n = outcome.agg_stats["ho_n"]
    for q, suffix in runner_mod.HELDOUT_QUANTILES:
        assert outcome.agg_stats[f"ho_n_q{suffix}"] == min(ho_n, math.ceil(q * ho_n))


def test_a_perfect_proxy_puts_only_good_docs_on_top(harness):
    # The stub predicts the truth exactly, so the top of the ranking is all
    # "good" (label above the midpoint of the 1..10 scale).
    outcome, _ = run_task(harness)
    assert outcome.agg_stats["ho_good_q10"] == outcome.agg_stats["ho_n_q10"]


def test_the_curve_is_omitted_entirely_when_there_is_no_holdout(harness):
    # A partial curve is a 422, so it is all sixteen keys or none.
    records = [{"text": f"doc {i}", "label": "3"} for i in range(10)]
    outcome, _ = run_task(harness, records=records)
    assert not any(k in outcome.agg_stats for k in runner_mod.HELDOUT_KEYS)


def test_unknown_metric_is_left_out_by_default(harness):
    # A null for an extra agg_stats key is rejected outright by the server, so
    # omitting is the only honest option: it lets the server say whether it
    # really needs the number before we invent one.
    outcome, _ = run_task(harness, {"metrics": ["something_we_cannot_know"]})
    assert "something_we_cannot_know" not in outcome.agg_stats
    assert any("something_we_cannot_know" in w for w in outcome.warnings)


def test_a_null_is_never_sent_for_a_key_that_forbids_it(harness):
    outcome, _ = run_task(
        harness, {"metrics": ["mystery"]}, unknown_metric="null"
    )
    # The policy asked for null; _finalize drops it, because only
    # eval_spearman and proxy_lr are nullable in the schema.
    assert "mystery" not in outcome.agg_stats


def test_nullable_metrics_may_still_be_null(harness):
    records = [{"text": f"doc {i}", "label": "3"} for i in range(10)]
    outcome, _ = run_task(harness, records=records)
    assert outcome.agg_stats["eval_spearman"] is None


def test_unknown_metric_policy_zero(harness):
    outcome, _ = run_task(
        harness, {"metrics": ["mystery"]}, unknown_metric="zero"
    )
    assert outcome.agg_stats["mystery"] == 0.0


def test_unknown_metric_policy_omit(harness):
    outcome, _ = run_task(harness, {"metrics": ["mystery"]}, unknown_metric="omit")
    assert "mystery" not in outcome.agg_stats


def test_unknown_metric_policy_fail(harness):
    with pytest.raises(TaskFailed):
        run_task(harness, {"metrics": ["mystery"]}, unknown_metric="fail")


def test_dedup_is_counted_and_reported(harness):
    records = balanced_records() + [{"text": "document 1 number 0", "label": "1"}]
    outcome, _ = run_task(harness, records=records)
    assert outcome.agg_stats["n_dedup_dropped"] == 1
    assert outcome.agg_stats["n_chunks"] == 60


def test_an_unavailable_shard_is_a_permanent_failure(harness):
    # Nothing is substituted any more: scoring chunk ids the control plane did
    # not assign would corrupt a sharded campaign's merge for everyone else.
    run, _ = harness(corpus=CorpusUnavailable("shard 'nope' came back empty"))
    task = TaskView.from_dict({**TASK, "dataset_id": "nope"})
    with pytest.raises(TaskFailed) as caught:
        run.run(task, TaskMode(train=True, source="test"))
    assert caught.value.permanent is True
    assert "nope" in str(caught.value)


def test_the_shard_warnings_reach_the_payload_authors(tmp_path, harness):
    corpus = make_corpus(tmp_path, balanced_records())
    corpus.warnings.append("shard d returned 3 chunk_id(s) twice")
    outcome, _ = run_task(harness, corpus=corpus)
    assert any("twice" in w for w in outcome.warnings)


def test_unlabeled_data_is_a_permanent_failure(harness):
    records = [{"text": f"doc {i}"} for i in range(10)]
    with pytest.raises(TaskFailed) as caught:
        run_task(harness, records=records)
    assert caught.value.permanent is True


def test_a_single_label_trains_on_everything_and_reports_null(harness):
    # Nothing can be held out meaningfully, and a null is honester than a
    # correlation computed over two documents.
    records = [{"text": f"doc {i}", "label": "3"} for i in range(10)]
    outcome, _ = run_task(harness, records=records)
    assert outcome.agg_stats["eval_spearman"] is None
    assert outcome.agg_stats["n_eval"] == 0


def test_payload_is_json_clean(harness):
    outcome, _ = run_task(harness)
    # allow_nan=False is what the client will use; it must not blow up there.
    encoded = json.dumps(
        {"scores": [s.to_dict() for s in outcome.scores], "agg_stats": outcome.agg_stats},
        allow_nan=False,
    )
    assert json.loads(encoded)


def test_non_finite_and_non_numeric_metrics_are_dropped():
    warnings = []
    cleaned = runner_mod._finalize(
        {"a": float("nan"), "b": float("inf"), "c": 1.0, "d": "text",
         "eval_spearman": None, "other": None, "bad key!": 1.0},
        warnings,
    )
    # Only the two schema-nullable keys may be null; everything else must be a
    # finite number or be left out.
    assert cleaned == {"c": 1.0, "eval_spearman": None}
    assert len(warnings) == 5


def test_agg_stats_is_trimmed_to_the_key_limit():
    warnings = []
    stats = {f"filler_{i}": float(i) for i in range(80)}
    stats["n_chunks"] = 5
    stats.update({k: 1 for k in runner_mod.HELDOUT_KEYS})
    cleaned = runner_mod._finalize(stats, warnings)
    assert len(cleaned) == runner_mod.MAX_AGG_KEYS
    # The keys the gate actually runs on survive the trim.
    assert "n_chunks" in cleaned
    assert set(runner_mod.HELDOUT_KEYS) <= set(cleaned)


def test_truncation_keeps_the_highest_scores(harness, monkeypatch):
    # The server selects a global top-k, so the tail is what it never looks at.
    monkeypatch.setattr(runner_mod, "MAX_SCORES", 10)
    outcome, _ = run_task(harness)
    kept = [s.score for s in outcome.scores]
    assert len(kept) == 10
    assert kept == sorted(kept, reverse=True)
    # Six documents score 10 and six score 9, so the top ten is exactly those.
    assert min(kept) == 9.0
    # n_chunks is the corpus, and the server enforces n_chunks >= len(scores).
    assert outcome.agg_stats["n_chunks"] == 60
    assert outcome.agg_stats["n_scores"] == 10


def test_the_model_is_persisted_under_its_server_given_id(harness, tmp_path):
    outcome, _ = run_task(harness)
    assert outcome.model_key == "quality-clf-v1"
    assert (tmp_path / "state" / "models" / "quality-clf-v1").is_dir()


def test_a_hostile_model_id_cannot_escape_the_state_dir(harness, tmp_path):
    outcome, _ = run_task(harness, {"model": {"kind": "classifier", "id": "../../pwned"}})
    # The key stays the server's own id, for the journal and the logs...
    assert outcome.model_key == "../../pwned"

    # ...and the directory it resolves to stays inside the state dir.
    models = (tmp_path / "state" / "models").resolve()
    resolved = AgentState(tmp_path / "state").model_dir(outcome.model_key).resolve()
    assert resolved.parent == models
    assert resolved.is_dir()
    assert not (tmp_path.parent / "pwned").exists()


def test_budget_is_reported_back_as_telemetry(harness):
    outcome, _ = run_task(harness)
    assert outcome.agg_stats["budget_cores"] == 2
    assert outcome.agg_stats["budget_ram_gb"] == 2.0


def test_training_is_configured_from_the_budget(harness):
    _, trained = run_task(harness)
    config = trained["config"]
    assert config.thread == 2
    assert config.isolate_training is True
    assert config.max_bucket < 2_000_000


# -- campaigns (contract 0.7.0) -------------------------------------------


SHARDED = {"params": {"mode": "sharded", "partition": 0, "n_partitions": 2, "n_labels": 20}}
EXPERTS = {"params": {"mode": "experts", "partition": 0, "n_partitions": 2, "n_labels": 20}}


def test_sharded_scores_its_own_partition_and_nothing_else(tmp_path, harness):
    # The server concatenates sharded rankings without deduplicating, so a score
    # for another node's chunk corrupts its share of the budget.
    from node.agent.shards import partition_of

    routing = Routing(mode="sharded", partition=0, n_partitions=2, n_labels=20)
    corpus = make_corpus(tmp_path, balanced_records(), routing, score_only_partition=True)
    outcome, _ = run_task(harness, SHARDED, corpus=corpus)

    assert 0 < len(outcome.scores) < 60
    assert all(partition_of(s.chunk_id, 2) == 0 for s in outcome.scores)
    # n_chunks answers for the partition, not the whole shard.
    assert outcome.agg_stats["n_chunks"] == len(outcome.scores)
    assert outcome.agg_stats["partition"] == 0.0
    assert outcome.agg_stats["n_partitions"] == 2.0


def test_an_expert_covers_the_whole_pool_but_trains_on_one_domain(tmp_path, harness):
    from node.agent.shards import partition_of

    routing = Routing(mode="experts", partition=0, n_partitions=2, n_labels=20)
    corpus = make_corpus(tmp_path, balanced_records(), routing)
    outcome, trained = run_task(harness, EXPERTS, corpus=corpus)

    # Every chunk in the pool is scored: `advance` rejects missing ids.
    assert len(outcome.scores) == 60
    assert outcome.agg_stats["n_chunks"] == 60
    # But training saw only its own domain's labels.
    trained_ids = {s.sample_id for s in trained["samples"]}
    assert trained_ids
    assert all(partition_of(cid, 2) == 0 for cid in trained_ids)


def test_an_expert_refuses_to_truncate_rather_than_submit_a_gap(tmp_path, harness, monkeypatch):
    # Dropping the tail would fail `advance` for the whole campaign, so failing
    # this one round loudly is the better outcome.
    monkeypatch.setattr(runner_mod, "MAX_SCORES", 10)
    routing = Routing(mode="experts", partition=0, n_partitions=2, n_labels=20)
    corpus = make_corpus(tmp_path, balanced_records(), routing)
    with pytest.raises(TaskFailed) as caught:
        run_task(harness, EXPERTS, corpus=corpus)
    assert caught.value.permanent is True
    assert "every one of them" in str(caught.value)


def test_a_sharded_round_may_still_truncate(tmp_path, harness, monkeypatch):
    # Only `experts` owes full coverage; a sharded ranking is a top-k feed.
    monkeypatch.setattr(runner_mod, "MAX_SCORES", 10)
    routing = Routing(mode="sharded", partition=0, n_partitions=2, n_labels=20)
    corpus = make_corpus(tmp_path, balanced_records(), routing, score_only_partition=True)
    outcome, _ = run_task(harness, SHARDED, corpus=corpus)
    assert len(outcome.scores) == 10


def test_the_campaigns_label_count_is_not_overwritten_by_the_class_count(harness):
    # params.n_labels means "labels placed in this domain"; the number of
    # distinct classes is a different question and gets a different key.
    outcome, _ = run_task(harness, {"metrics": ["n_labels"], **SHARDED})
    assert outcome.agg_stats["n_labels"] == 20.0
    assert outcome.agg_stats["n_label_classes"] == 10


def test_an_explicit_good_min_sets_the_curves_threshold(harness):
    # A campaign judges "good" by good_min; when a round passes it, the curve we
    # report has to agree with the selection we are measured against.
    strict, _ = run_task(harness, {"params": {"good_min": 10}})
    loose, _ = run_task(harness, {"params": {"good_min": 1}})
    assert strict.agg_stats["ho_good"] < loose.agg_stats["ho_good"]
    assert loose.agg_stats["ho_good"] == loose.agg_stats["ho_n"]


# -- the model lifecycle instruction (contract 0.9.0) ---------------------


def op(train, **extra):
    return {"operation": {"train": train, **extra}}


def test_fresh_trains_and_saves_under_the_checkpoint_the_server_named(harness, tmp_path):
    outcome, trained = run_task(
        harness, op("fresh", output_checkpoint_id="c1-r1-ckpt")
    )
    assert trained["calls"] == 1
    assert trained["loaded_from"] == []
    # Saved where the *next* round's input_checkpoint_id will look for it.
    assert outcome.model_key == "c1-r1-ckpt"
    assert outcome.checkpoint_id == "c1-r1-ckpt"
    assert (tmp_path / "state" / "models" / "c1-r1-ckpt").is_dir()
    assert outcome.agg_stats["train_policy_fresh"] == 1.0
    assert outcome.agg_stats["checkpoint_saved"] == 1.0


def test_skip_loads_the_checkpoint_and_runs_no_trainer(harness, tmp_path):
    # Make the checkpoint exist first.
    run_task(harness, op("fresh", output_checkpoint_id="prev"))

    outcome, trained = run_task(
        harness, op("skip", input_checkpoint_id="prev", output_checkpoint_id="prev")
    )
    assert trained.get("calls") is None  # a fresh harness; nothing trained
    assert trained["loaded_from"] == [tmp_path / "state" / "models" / "prev"]
    assert outcome.agg_stats["train_policy_skip"] == 1.0
    assert outcome.agg_stats["checkpoint_loaded"] == 1.0
    assert outcome.agg_stats["checkpoint_saved"] == 0.0
    # It still scores: operation.score is always true.
    assert len(outcome.scores) == 60


def test_skip_without_the_checkpoint_trains_rather_than_stranding_the_round(harness):
    # Told to score from weights this node does not hold -- a fresh machine, or
    # state cleared between rounds. Scores from a new model beat no scores, but
    # the divergence has to be on the record.
    outcome, trained = run_task(harness, op("skip", input_checkpoint_id="never-saved"))
    assert trained["calls"] == 1
    assert outcome.agg_stats["train_fallback_no_checkpoint"] == 1.0
    assert any("does not hold" in w for w in outcome.warnings)


def test_continue_is_declared_when_it_is_executed_as_a_retrain(harness):
    """fastText has no warm start, so `continue` cannot be taken literally.

    Retraining on the round's full (cumulative) label set is what a continuation
    would converge to here, but it is not what was asked -- so it is reported
    rather than quietly substituted.
    """
    run_task(harness, op("fresh", output_checkpoint_id="r1"))
    outcome, trained = run_task(
        harness, op("continue", input_checkpoint_id="r1", output_checkpoint_id="r2")
    )
    assert trained["calls"] == 1
    assert outcome.agg_stats["train_policy_continue"] == 1.0
    assert outcome.agg_stats["train_continued_as_fresh"] == 1.0
    assert outcome.agg_stats["checkpoint_saved"] == 1.0
    assert any("no warm start" in w for w in outcome.warnings)
    # The lineage the server tracks is still honoured: output is a new name.
    assert outcome.checkpoint_id == "r2"


def test_continue_says_so_when_the_input_checkpoint_is_missing(harness):
    outcome, _ = run_task(
        harness, op("continue", input_checkpoint_id="gone", output_checkpoint_id="r2")
    )
    assert any("does not hold" in w for w in outcome.warnings)
    assert outcome.agg_stats["train_continued_as_fresh"] == 1.0


def test_a_manifest_is_written_beside_the_weights(harness, tmp_path):
    import json

    outcome, _ = run_task(harness, op("fresh", output_checkpoint_id="ckpt-a"))
    manifest_path = tmp_path / "state" / "models" / "ckpt-a" / "manifest.json"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text())
    # Enough to recognise a repeat of this round after a restart, rather than
    # retraining it or submitting it twice (METRICS_GUIDE).
    assert manifest["round_id"] == "r1"
    assert manifest["model_id"] == "quality-clf-v1"
    assert manifest["train_policy"] == "fresh"
    assert manifest["n_scores"] == len(outcome.scores)
    assert len(manifest["train_chunk_ids"]) > 0
    assert all(len(cid) == 64 for cid in manifest["train_chunk_ids"])


def test_no_manifest_is_written_when_nothing_was_trained(harness, tmp_path):
    run_task(harness, op("fresh", output_checkpoint_id="prev"))
    (tmp_path / "state" / "models" / "prev" / "manifest.json").unlink()

    outcome, _ = run_task(harness, op("skip", input_checkpoint_id="prev"))
    assert outcome.checkpoint_id is None
    assert not (tmp_path / "state" / "models" / "prev" / "manifest.json").exists()


def test_the_runner_reports_its_stages_in_order(harness, tmp_path):
    class RecordingProgress:
        def __init__(self):
            self.stages = []

        def stage(self, stage, *, total=None):
            self.stages.append(stage)

        def progress(self, processed, total=None):
            pass

        def gauge(self, name, value):
            pass

    progress = RecordingProgress()
    run, _ = harness()
    run._progress = progress
    task = TaskView.from_dict(TASK)
    run.run(task, resolve_mode(task, "auto"))

    # Uploading belongs to the loop, which owns the POST.
    assert progress.stages == ["downloading", "training", "scoring"]


def test_a_corrupt_checkpoint_is_trained_over_rather_than_crashed_on(
    harness, tmp_path, monkeypatch
):
    run_task(harness, op("fresh", output_checkpoint_id="half-written"))

    def broken_load(directory):
        raise RuntimeError("truncated model file")

    run, trained = harness()
    monkeypatch.setattr(
        runner_mod.FastTextClassifier, "load", staticmethod(broken_load)
    )
    task = TaskView.from_dict({**TASK, **op("skip", input_checkpoint_id="half-written")})
    outcome = run.run(task, resolve_mode(task, "auto"))

    # Somebody else's crash must not become this round's failure.
    assert trained["calls"] == 1
    assert outcome.agg_stats["train_fallback_no_checkpoint"] == 1.0


def test_a_single_valued_holdout_does_not_claim_perfect_precision(tmp_path):
    """The threshold comes from every label the shard gave us, not the slice.

    A small slice can easily come out all-zeros. Read off that slice alone, the
    midpoint would land on the only value present, every held-out document would
    count as good, and the node would report perfect precision to the
    Reliability Gate on the strength of an accident.
    """
    records = [{"text": f"bad doc {i}", "label": "0"} for i in range(40)]
    records += [{"text": f"good doc {i}", "label": "1"} for i in range(4)]
    corpus = make_corpus(tmp_path, records)

    ctx = runner_mod.MetricContext(
        task=TaskView.from_dict(TASK),
        corpus=corpus,
        report=None,
        held_pred=[0.1, 0.2, 0.3, 0.4],
        held_true=[0.0, 0.0, 0.0, 0.0],  # the accident
        n_scores=4,
        n_corpus=44,
        train_s=0.0,
        score_s=0.0,
        budget=Budget(cores=1, ram_bytes=1024**3, core_ids=[0]),
        effective={},
        peak_rss_mb=None,
    )
    # Midpoint of 0..1 across the shard's labels, not of the all-zero slice.
    assert runner_mod.good_threshold(ctx) == 0.5
    assert runner_mod.heldout_curve(ctx)["ho_good"] == 0
