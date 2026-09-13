"""Executing one round: corpus in, scores and aggregates out.

`TaskRunner` does no HTTP itself. It takes a `TaskView` and a `CorpusSource`
and gives back a `RunOutcome`, which is what makes the whole pipeline testable
without a server and reusable from a GUI. Talking to the control plane is
`loop`'s job, and fetching the shard is `shards`'.

Two subtleties are worth reading before the code.

*The runner holds out its own evaluation slice* rather than trusting the
classifier's internal one. `TrainConfig.retrain_on_full` defaults to True, so
the model that comes back has seen the core's eval split. Measuring on that
slice would report memorisation -- and the held-out curve built from it is
precisely what the server's Reliability Gate consumes. So the runner splits
first, trains on its own training half, and measures on documents the shipped
weights never saw. Nothing in `/shards/…/labels` marks a held-out split
(CONTRACT.md 4.7), so this is the only place one exists.

*What gets scored is not what gets trained on.* In an `experts` campaign the
node trains on one domain's labels and must return a score for every chunk in
the pool; `advance` rejects a submission with ids missing. That is why the
corpus carries two lists and why truncation is refused in that mode.

Contract 0.9.0 added `task.operation`, and the runner obeys it rather than
deciding for itself: `fresh` trains from the recipe, `continue` carries a
checkpoint forward, `skip` loads one and only scores. The one place this backend
cannot take the instruction literally is `continue` -- fastText supervised
models have no warm start -- so it retrains on the round's full label set and
says so in `agg_stats`, which is the honest version of a continuation here.
"""

from __future__ import annotations

import logging
import math
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from node.agent import resources
from node.agent.datasets import Chunk
from node.agent.models import (
    MAX_SCORES,
    STAGE_DOWNLOADING,
    STAGE_SCORING,
    STAGE_TRAINING,
    TRAIN_CONTINUE,
    TRAIN_FRESH,
    TRAIN_SKIP,
    ChunkScore,
    TaskMode,
    TaskView,
    is_id,
)
from node.agent.reporting import Reporter
from node.agent.resources import Budget, MemoryWatchdog
from node.agent.shards import Corpus, CorpusSource, CorpusUnavailable
from node.agent.state import AgentState
from node.core import metrics
from node.core.classifier import FastTextClassifier
from node.core.dataset import stratified_split
from node.core.types import Sample, TrainConfig, TrainReport

LOG = logging.getLogger("node.agent.runner")

EVAL_FRACTION = 0.15
SCORE_BATCH = 5_000

# CONTRACT.md 3.5.1 fixes both the quantiles and the key spelling. All sixteen
# keys travel together or none of them do; a partial curve is a 422.
HELDOUT_QUANTILES = ((0.01, "01"), (0.02, "02"), (0.05, "05"), (0.10, "10"),
                     (0.20, "20"), (0.30, "30"), (0.50, "50"))
HELDOUT_KEYS = ("ho_n", "ho_good", *(
    f"ho_{part}_q{suffix}" for _, suffix in HELDOUT_QUANTILES for part in ("n", "good")
))

# agg_stats keys must match the contract's ID format, and there may be at most
# this many of them.
MAX_AGG_KEYS = 64
# The only two keys the schema declares nullable; everything else must be a
# finite number or be left out entirely.
NULLABLE_METRICS = frozenset({"eval_spearman", "proxy_lr"})


class TaskFailed(Exception):
    """A round that cannot be completed. `permanent` says whether to retry."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


@dataclass
class RunOutcome:
    scores: list[ChunkScore]
    agg_stats: dict[str, Any]
    model_key: str
    report: TrainReport | None = None
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    # Where the wall clock went, phase by phase. `download_s` is the network
    # half of the round and everything else is this machine's own work, which
    # is the split an operator sizing a node actually needs to see.
    timings: dict[str, float] = field(default_factory=dict)
    # The checkpoint this round left behind, for the journal and the next round.
    checkpoint_id: str | None = None


@dataclass
class TrainNotes:
    """What actually happened to the model, versus what was asked.

    `operation` is a command, so any divergence from it has to be visible: the
    flags here become `agg_stats` numbers, and the warnings reach the operator's
    log. Silently doing something else would make the server's lineage wrong.
    """

    policy: str
    loaded_checkpoint: bool = False
    saved_checkpoint: bool = False
    input_checkpoint_present: bool = False
    # `continue` asked for a warm start this backend cannot do.
    continued_as_fresh: bool = False
    # `skip` named a checkpoint this node does not hold.
    fell_back_to_fresh: bool = False
    warnings: list[str] = field(default_factory=list)


class NullProgress:
    """The telemetry seam, as a no-op. Keeps the runner usable without a loop."""

    def stage(self, stage: str, *, total: int | None = None) -> None:
        pass

    def progress(self, processed: int, total: int | None = None) -> None:
        pass

    def gauge(self, name: str, value: float) -> None:
        pass


def stable_seed(text: str, base: int = 17) -> int:
    """A seed that is the same in every process.

    Python's str hash is salted per interpreter, so using it here would give a
    different holdout after every restart -- and crash-resume would no longer
    reproduce the split it was interrupted in the middle of.
    """
    return base + zlib.crc32(text.encode("utf-8")) % 100_000


class TaskRunner:
    def __init__(
        self,
        *,
        state: AgentState,
        corpus_source: CorpusSource,
        budget: Budget,
        reporter: Reporter,
        score_scale: str = "raw",
        unknown_metric: str = "omit",
        enforce_limits: bool = True,
        verbose: int = 0,
        progress: Any | None = None,
    ) -> None:
        self._state = state
        self._corpus_source = corpus_source
        self._budget = budget
        self._reporter = reporter
        self._score_scale = score_scale
        self._unknown_metric = unknown_metric
        self._enforce_limits = enforce_limits
        self._verbose = verbose
        self._progress = progress or NullProgress()
        self._peak_rss_mb: float | None = None

    def run(self, task: TaskView, mode: TaskMode) -> RunOutcome:
        started = time.time()
        warnings: list[str] = []

        self._progress.stage(STAGE_DOWNLOADING)
        download_started = time.time()
        corpus = self._fetch_corpus(task)
        download_s = time.time() - download_started
        warnings.extend(corpus.warnings)
        self._reporter.note(
            logging.INFO,
            "fetched server-held workload",
            round_id=task.round_id,
            shard_id=corpus.shard_id,
            routing=corpus.routing.describe(),
            n_score=corpus.n_score,
            n_train=corpus.n_train,
        )

        split_started = time.time()
        fit, held_out = self._split(corpus, task.round_id)
        split_s = time.time() - split_started
        config, config_warnings, effective = resources.train_config_for(
            self._budget, task.params, verbose=self._verbose
        )
        warnings.extend(config_warnings)

        # Where the weights land. `operation.output_checkpoint_id` is the
        # server's chosen name and takes precedence; without one, the model
        # recipe's id is the stable fallback. The server's ids are kept
        # unescaped here because they go into the journal and the logs, where
        # they must match what the operator sees -- making them safe as a
        # directory name is `AgentState.checkpoint_dir`'s job.
        model_key = (
            mode.output_checkpoint_id
            or (task.model.id if task.model else None)
            or f"round-{task.round_id}"
        )
        classifier, report, train_s, train_notes = self._obtain_model(
            task, mode, fit, config, model_key
        )
        warnings.extend(train_notes.warnings)

        self._progress.stage(STAGE_SCORING, total=corpus.n_score)
        scored_started = time.time()
        scores = self._score_corpus(classifier, corpus)
        score_s = time.time() - scored_started

        eval_started = time.time()
        held_pred, held_true = self._evaluate_holdout(classifier, held_out)
        eval_s = time.time() - eval_started

        metrics_started = time.time()
        context = MetricContext(
            task=task,
            corpus=corpus,
            report=report,
            held_pred=held_pred,
            held_true=held_true,
            n_scores=len(scores),
            n_corpus=corpus.pool_size,
            train_s=train_s,
            score_s=score_s,
            budget=self._budget,
            effective=effective,
            peak_rss_mb=self._peak_rss_mb,
            mode=mode,
            train_notes=train_notes,
            download_s=download_s,
        )
        agg_stats, metric_warnings = compute_metrics(
            task.metrics, context, self._unknown_metric
        )
        metrics_s = time.time() - metrics_started
        warnings.extend(metric_warnings)

        self._reporter.note(
            logging.INFO,
            "scored corpus",
            round_id=task.round_id,
            n_chunks=len(scores),
            **resources.summarize([s.score for s in scores]),
        )

        # The manifest beside the weights, so a restart can recognise this round
        # instead of retraining it or submitting it twice (METRICS_GUIDE).
        if train_notes.saved_checkpoint:
            self._write_manifest(task, mode, corpus, model_key, report, len(scores))

        total_s = time.time() - started
        timings = {
            "download_s": round(download_s, 3),
            "split_s": round(split_s, 3),
            "train_s": round(train_s, 3),
            "score_s": round(score_s, 3),
            "eval_s": round(eval_s, 3),
            "metrics_s": round(metrics_s, 3),
            "total_s": round(total_s, 3),
            # What is left is checkpoint I/O and the manifest -- small, but it
            # has to land somewhere or the parts stop summing to the whole.
            "other_s": round(
                max(
                    total_s
                    - (download_s + split_s + train_s + score_s + eval_s + metrics_s),
                    0.0,
                ),
                3,
            ),
        }
        timings["compute_s"] = round(total_s - download_s, 3)
        self._reporter.note(
            logging.INFO,
            "round timing",
            round_id=task.round_id,
            n_score=corpus.n_score,
            n_train=corpus.n_train,
            download_s=timings["download_s"],
            compute_s=timings["compute_s"],
            download_share=round(download_s / total_s, 3) if total_s > 0 else 0.0,
            train_s=timings["train_s"],
            score_s=timings["score_s"],
            eval_s=timings["eval_s"],
        )

        return RunOutcome(
            scores=scores,
            agg_stats=agg_stats,
            model_key=model_key,
            report=report,
            warnings=warnings,
            duration_s=total_s,
            timings=timings,
            checkpoint_id=model_key if train_notes.saved_checkpoint else None,
        )

    # -- steps ------------------------------------------------------------

    def _write_manifest(
        self,
        task: TaskView,
        mode: TaskMode,
        corpus: Corpus,
        checkpoint_id: str,
        report: TrainReport | None,
        n_scores: int,
    ) -> None:
        try:
            self._state.write_manifest(
                checkpoint_id,
                {
                    "round_id": task.round_id,
                    "model_id": task.model.id if task.model else None,
                    "model_kind": task.model.kind if task.model else None,
                    "train_policy": mode.policy,
                    "input_checkpoint_id": mode.input_checkpoint_id,
                    "shard_id": corpus.shard_id,
                    "routing": corpus.routing.describe(),
                    # The chunk ids the weights actually saw. Enough to tell a
                    # repeat of this round from a genuinely new one.
                    "train_chunk_ids": sorted(c.chunk_id for c in corpus.to_train),
                    "n_train": report.n_train if report else 0,
                    "n_scores": n_scores,
                    "status": "scored",
                    "written_at": time.time(),
                },
            )
        except OSError as error:
            # A manifest is for recovery, not correctness. Losing the round over
            # it would be the wrong trade.
            LOG.warning("could not write manifest for %s: %s", checkpoint_id, error)

    def _fetch_corpus(self, task: TaskView) -> Corpus:
        try:
            return self._corpus_source.fetch(task)
        except CorpusUnavailable as error:
            # An empty shard or unusable routing. Neither improves on a retry,
            # and both are the control plane's to fix.
            raise TaskFailed(str(error), permanent=True) from error

    def _split(self, corpus: Corpus, round_id: str) -> tuple[list, list[Chunk]]:
        labeled = corpus.to_train
        if not labeled:
            raise TaskFailed(
                f"shard {corpus.shard_id!r} has no labeled chunks to train on",
                permanent=True,
            )

        samples = [c.as_labeled() for c in labeled]
        by_id = {c.chunk_id: c for c in labeled}

        if len({s.label for s in samples}) < 2:
            # Nothing to hold out meaningfully, and the core would refuse anyway.
            return samples, []

        try:
            fit, held = stratified_split(
                samples, EVAL_FRACTION, seed=stable_seed(round_id)
            )
        except ValueError:
            return samples, []

        if len(held) < 2 or len({s.label for s in fit}) < 2:
            # Too small to measure honestly. Reporting a null metric beats
            # reporting one computed on two documents.
            return samples, []

        return fit, [by_id[s.sample_id] for s in held]

    def _obtain_model(
        self, task: TaskView, mode: TaskMode, fit: list, config: TrainConfig, model_key: str
    ) -> tuple[FastTextClassifier, TrainReport | None, float, TrainNotes]:
        """Carry out `task.operation`: fresh, continue, or skip.

        The instruction is obeyed, not second-guessed -- with one documented
        exception, `continue`, which this backend cannot do literally.
        """
        notes = TrainNotes(policy=mode.policy)
        model_dir = self._state.checkpoint_dir(model_key)

        if mode.policy == TRAIN_SKIP:
            loaded = self._load_checkpoint(mode.input_checkpoint_id, model_key)
            if loaded is not None:
                notes.loaded_checkpoint = True
                return loaded, None, 0.0, notes
            # Told to score with a checkpoint this node does not have -- a fresh
            # machine, or state cleared between rounds. Training gives the round
            # something real to submit; failing it gives the campaign nothing.
            notes.fell_back_to_fresh = True
            notes.warnings.append(
                f"operation said train=skip from checkpoint "
                f"{mode.input_checkpoint_id or model_key!r}, which this node does not "
                "hold; trained fresh instead so the round still produces scores"
            )
            self._reporter.note(
                logging.WARNING,
                "no such checkpoint; training fresh instead of scoring only",
                checkpoint_id=mode.input_checkpoint_id or model_key,
            )

        elif mode.policy == TRAIN_CONTINUE:
            # fastText supervised training has no warm start: there is no API to
            # resume from saved weights. Since campaign labels are cumulative,
            # retraining on the round's full label set is what a continuation
            # would converge to anyway -- but it is not what we were asked, so
            # it is declared rather than quietly substituted.
            notes.continued_as_fresh = True
            if self._state.has_checkpoint(mode.input_checkpoint_id):
                notes.input_checkpoint_present = True
            else:
                notes.warnings.append(
                    f"operation said train=continue from checkpoint "
                    f"{mode.input_checkpoint_id!r}, which this node does not hold"
                )
            notes.warnings.append(
                "train=continue was executed as a full retrain on this round's "
                "labels: fastText supervised models have no warm start "
                "(reported as train_continued_as_fresh)"
            )

        if mode.policy == TRAIN_CONTINUE and notes.input_checkpoint_present:
            self._reporter.note(
                logging.INFO,
                "continuing from checkpoint by retraining on the full label set",
                from_checkpoint=mode.input_checkpoint_id,
                to_checkpoint=model_key,
            )

        self._progress.stage(STAGE_TRAINING, total=len(fit))
        self._reporter.note(
            logging.INFO,
            "training",
            round_id=task.round_id,
            policy=mode.policy,
            n_train=len(fit),
            lr=config.lr,
            epoch=config.resolve_epoch(len(fit)),
            thread=config.thread,
            max_bucket=config.max_bucket,
        )

        env = resources.child_env(self._budget, enforce=self._enforce_limits)
        started = time.time()
        self._peak_rss_mb = None
        try:
            with resources.applied_env(env), MemoryWatchdog(
                self._budget, self._reporter
            ) as watchdog:
                classifier, report = FastTextClassifier.train(fit, config)
            self._peak_rss_mb = watchdog.peak_rss_mb
        except ValueError as error:
            # The core validates its input up front: no samples, one label, or
            # everything empty after normalization. None of that improves on a
            # retry.
            raise TaskFailed(f"training data unusable: {error}", permanent=True) from error
        except MemoryError as error:
            # The self-imposed cap fired and the worker forwarded the C++
            # bad_alloc back across the process boundary.
            raise TaskFailed(
                f"training exceeded the {self._budget.ram_gb:.1f} GiB RAM budget: {error}"
            ) from error
        except RuntimeError as error:
            if "produced nothing" in str(error):
                # The same cap, but hit where the worker could not report it --
                # the process died outright. Also what an OOM killer looks like.
                raise TaskFailed(
                    f"training process died, most likely on the RAM budget: {error}"
                ) from error
            raise TaskFailed(f"training failed: {error}") from error

        duration = time.time() - started
        self._progress.progress(len(fit), len(fit))

        # Saved under the name the server asked for, so the next round's
        # `input_checkpoint_id` finds it.
        classifier.save(model_dir)
        notes.saved_checkpoint = True

        # No `train_loss` gauge: fastText's supervised trainer does not report a
        # final loss through this wrapper, and the METRICS_GUIDE's key is
        # optional. An invented number on that graph would be worse than a gap.

        self._reporter.note(
            logging.INFO,
            "trained",
            round_id=task.round_id,
            checkpoint_id=model_key,
            accuracy=report.accuracy,
            macro_f1=report.macro_f1,
            mae=report.mae,
            qwk=report.qwk,
            duration_s=round(duration, 2),
        )
        for warning in report.warnings:
            self._reporter.note(logging.WARNING, "core: " + warning)
        return classifier, report, duration, notes

    def _load_checkpoint(
        self, checkpoint_id: str | None, fallback_key: str
    ) -> FastTextClassifier | None:
        """Load node-local weights by the name the control plane gave them.

        Falls back to the recipe id when no input checkpoint was named, which is
        how a pre-0.9.0 round or a hand-driven `--mode score` still finds the
        model this node saved last time.
        """
        tried: set[str] = set()
        for candidate in (checkpoint_id, fallback_key):
            if not candidate or candidate in tried:
                continue
            tried.add(candidate)
            directory = self._state.checkpoint_dir(candidate)
            if not directory.is_dir():
                continue
            try:
                classifier = FastTextClassifier.load(directory)
            except (OSError, ValueError, RuntimeError) as error:
                # A half-written checkpoint from a killed run. Say so and let the
                # caller train instead of dying on someone else's crash.
                LOG.warning("checkpoint %s will not load: %s", candidate, error)
                continue
            self._reporter.note(
                logging.INFO, "loaded checkpoint", checkpoint_id=candidate
            )
            return classifier
        return None

    def _score_corpus(
        self, classifier: FastTextClassifier, corpus: Corpus
    ) -> list[ChunkScore]:
        """Score the assigned slice, keeping only what the server will accept.

        Batched because the corpus can be large and a `ScoredSample` carries a
        full probability distribution -- holding 200k of them just to read one
        float off each would be a needless spike inside a capped process.
        """
        chunks = corpus.to_score
        values = _label_range(classifier.labels)
        scores: list[ChunkScore] = []

        for start in range(0, len(chunks), SCORE_BATCH):
            batch = chunks[start : start + SCORE_BATCH]
            scored = classifier.score([Sample(c.chunk_id, c.text) for c in batch])
            for sample in scored:
                scores.append(
                    ChunkScore(
                        chunk_id=sample.sample_id,
                        score=_score_value(sample, classifier.labels, values, self._score_scale),
                    )
                )
            # Per batch, not per document: the snapshot is cheap but the
            # heartbeat only ships it on the server's interval anyway.
            self._progress.progress(len(scores), len(chunks))

        if len(scores) > MAX_SCORES:
            if corpus.routing.mode == "experts":
                # An expert owes a score for every chunk in the pool: `advance`
                # refuses a submission with ids missing, which would strand the
                # whole campaign. Better to fail this round loudly than to
                # deliver a payload that can only be rejected.
                raise TaskFailed(
                    f"shard {corpus.shard_id!r} has {len(scores)} chunks but an experts "
                    f"campaign must score every one of them and submit caps at "
                    f"{MAX_SCORES}; this pool is too large for this node's contract",
                    permanent=True,
                )
            # Otherwise the server is selecting a global top-k, so the tail of
            # our ranking is the part it would never look at.
            scores.sort(key=lambda s: s.score, reverse=True)
            LOG.warning(
                "corpus has %d chunks; submitting the top %d", len(scores), MAX_SCORES
            )
            scores = scores[:MAX_SCORES]
        return scores

    def _evaluate_holdout(
        self, classifier: FastTextClassifier, held_out: Sequence[Chunk]
    ) -> tuple[list[float], list[float]]:
        """Predicted score and true score for documents the model never saw."""
        if not held_out:
            return [], []

        values = _label_range(classifier.labels)
        scored = classifier.score([Sample(c.chunk_id, c.text) for c in held_out])

        predicted: list[float] = []
        truth: list[float] = []
        for chunk, sample in zip(held_out, scored, strict=True):
            true_value = _as_float(chunk.label)
            if true_value is None:
                continue
            predicted.append(_score_value(sample, classifier.labels, values, "raw"))
            truth.append(true_value)
        return predicted, truth


# -- score extraction ------------------------------------------------------


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _label_range(labels: Sequence[str]) -> tuple[float, float] | None:
    values = metrics.numeric_label_values(labels)
    if not values:
        return None
    return min(values), max(values)


def _score_value(
    sample: Any, labels: Sequence[str], values: tuple[float, float] | None, scale: str
) -> float:
    """One number per chunk: how good the proxy thinks this document is."""
    raw = sample.expected_score
    if raw is None:
        # Non-numeric labels have no ordinal meaning, so fall back to the rank
        # of the predicted class. Reported via score_is_ordinal.
        raw = float(labels.index(sample.label)) if sample.label in labels else 0.0

    if scale == "unit" and values is not None:
        low, high = values
        if high > low:
            return float(min(max((raw - low) / (high - low), 0.0), 1.0))
    return float(raw)


# -- aggregate statistics --------------------------------------------------


@dataclass
class MetricContext:
    task: TaskView
    corpus: Corpus
    report: TrainReport | None
    held_pred: list[float]
    held_true: list[float]
    n_scores: int
    n_corpus: int
    train_s: float
    score_s: float
    budget: Budget
    effective: dict[str, float]
    peak_rss_mb: float | None
    mode: TaskMode | None = None
    train_notes: TrainNotes | None = None
    # Time spent reading the shard off the control plane, kept apart from the
    # two compute numbers so a slow round can be blamed on the right half.
    download_s: float = 0.0


def compute_metrics(
    required: Sequence[str], ctx: MetricContext, policy: str
) -> tuple[dict[str, Any], list[str]]:
    """Build agg_stats: what we always report, plus whatever the round demands."""
    warnings: list[str] = []
    stats: dict[str, Any] = _base_metrics(ctx)

    for key in required:
        if key in stats:
            continue
        value, how = _resolve_metric(key, ctx)
        if how is not None:
            stats[key] = value
            LOG.debug("metric %s resolved via %s", key, how)
            continue

        # Nothing we compute, nothing the round set for us, nothing in the
        # training report. Never invent a plausible number.
        warnings.append(f"round requires metric {key!r}; this node cannot compute it")
        if policy == "fail":
            raise TaskFailed(f"cannot compute required metric {key!r}", permanent=True)
        if policy == "omit":
            # The server answers with `missing_metrics`, and the loop retries
            # once with zeros. That order puts the lie second, not first.
            continue
        if policy == "zero":
            stats[key] = 0.0
        else:
            stats[key] = None

    return _finalize(stats, warnings), warnings


def _base_metrics(ctx: MetricContext) -> dict[str, Any]:
    stats: dict[str, Any] = {
        # The corpus, not the submission. The server enforces
        # n_chunks >= len(scores), and the two differ once a corpus is large
        # enough to be truncated at MAX_SCORES.
        "n_chunks": max(ctx.n_corpus, ctx.n_scores),
        "n_scores": ctx.n_scores,
        "eval_spearman": metrics.spearman(ctx.held_pred, ctx.held_true)
        if len(ctx.held_pred) >= 2
        else None,
        "n_dedup_dropped": ctx.corpus.n_dedup_dropped,
        # The held-out size again, but always present: `ho_n` only exists when
        # there is a curve to attach it to.
        "n_eval": len(ctx.held_true),
        "download_duration_s": round(ctx.download_s, 3),
        "train_duration_s": round(ctx.train_s, 3),
        "score_duration_s": round(ctx.score_s, 3),
        "budget_cores": ctx.budget.cores,
        "budget_ram_gb": round(ctx.budget.ram_gb, 2),
    }
    if ctx.peak_rss_mb is not None:
        stats["peak_rss_mb"] = ctx.peak_rss_mb

    # Routing, echoed as numbers so the operator can see which slice this
    # submission answers for without cross-referencing the campaign.
    routing = ctx.corpus.routing
    if routing.is_campaign:
        stats["partition"] = float(routing.partition or 0)
        stats["n_partitions"] = float(routing.n_partitions or 0)
    stats["n_shard_labels"] = ctx.corpus.n_labels_available
    stats["n_train_labels"] = ctx.corpus.n_train

    # What the model lifecycle instruction asked for, and whether it was carried
    # out as given. `operation` is a command, so a divergence has to be on the
    # record rather than only in this node's log.
    notes = ctx.train_notes
    if notes is not None:
        stats["train_policy_fresh"] = 1.0 if notes.policy == TRAIN_FRESH else 0.0
        stats["train_policy_continue"] = 1.0 if notes.policy == TRAIN_CONTINUE else 0.0
        stats["train_policy_skip"] = 1.0 if notes.policy == TRAIN_SKIP else 0.0
        stats["checkpoint_loaded"] = 1.0 if notes.loaded_checkpoint else 0.0
        stats["checkpoint_saved"] = 1.0 if notes.saved_checkpoint else 0.0
        if notes.continued_as_fresh:
            stats["train_continued_as_fresh"] = 1.0
        if notes.fell_back_to_fresh:
            stats["train_fallback_no_checkpoint"] = 1.0

    if "proxy_lr" in ctx.task.params:
        requested = ctx.task.params["proxy_lr"]
        # The schema has exclusiveMinimum: 0, so a non-positive value has to go
        # as null rather than as the number we were given.
        stats["proxy_lr"] = requested if requested > 0 else None
    stats.update(ctx.effective)

    if ctx.report is not None:
        stats["n_train"] = ctx.report.n_train
        # Not `n_labels`: contract 0.7.0 gave that name to the campaign's
        # cumulative label count for this domain, which arrives in params and is
        # echoed back by `_resolve_metric`. Reporting the number of distinct
        # classes under the same key would answer a different question.
        stats["n_label_classes"] = len(ctx.report.labels)

    stats.update(heldout_curve(ctx))
    _add_eval_quality(stats, ctx)
    return stats


def _add_eval_quality(stats: dict[str, Any], ctx: MetricContext) -> None:
    """Accuracy-style numbers on our own holdout, not the core's internal one."""
    if len(ctx.held_true) < 2:
        return
    errors = [abs(p - t) for p, t in zip(ctx.held_pred, ctx.held_true, strict=True)]
    stats["eval_mae"] = round(sum(errors) / len(errors), 4)
    stats["eval_exact"] = round(sum(1 for e in errors if e < 0.5) / len(errors), 4)


def good_threshold(ctx: MetricContext) -> float:
    """At or above this the oracle's label counts as good.

    A campaign has its own `good_min` and applies it server-side at finalize
    (CONTRACT.md 4.8); it is not promised in `params`, but if a round does pass
    it we use it, so the curve we report agrees with the selection we are judged
    against. Failing that, the midpoint of the label scale is the only
    defensible reading -- and for the binary labels an oracle produces it lands
    exactly where it should.
    """
    for key in ("good_min", "good_label_min"):
        override = ctx.task.params.get(key)
        if override is not None:
            return float(override)

    # Taken over every label the shard gave us, not just the held-out slice.
    # A small slice can easily come out single-valued, and reading the midpoint
    # off that would put the threshold *at* the only value present -- scoring
    # every held-out document "good" and reporting perfect precision to the
    # Reliability Gate on the strength of an accident.
    scale = [
        value
        for value in (_as_float(c.label) for c in ctx.corpus.to_train)
        if value is not None
    ] or ctx.held_true
    if not scale:
        return 0.0
    return (min(scale) + max(scale)) / 2.0


def heldout_curve(ctx: MetricContext) -> dict[str, int]:
    """The sixteen counters of CONTRACT.md 3.5.1, or nothing at all.

    This is what calibrated pooling and the Reliability Gate actually run on --
    a node that omits it is scored `unknown` and dropped from the top-k by
    default. Only counts leave the node: no documents, no scores, no labels.

    A partial curve is a 422, so all sixteen keys are built together or the
    whole thing is left out.
    """
    ho_n = len(ctx.held_true)
    if ho_n < 1 or len(ctx.held_pred) != ho_n:
        return {}

    threshold = good_threshold(ctx)
    # Ranked by what the proxy predicted; "good" is the oracle's verdict.
    ranked_good = [
        1 if true_value >= threshold else 0
        for _, true_value in sorted(
            zip(ctx.held_pred, ctx.held_true, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ]

    curve: dict[str, int] = {"ho_n": ho_n, "ho_good": sum(ranked_good)}
    for q, suffix in HELDOUT_QUANTILES:
        # ceil(q * ho_n), exactly as the contract defines it. Nested prefixes
        # make both counters non-decreasing in q for free, which is one of the
        # server's validation rules.
        take = min(ho_n, math.ceil(q * ho_n))
        curve[f"ho_n_q{suffix}"] = take
        curve[f"ho_good_q{suffix}"] = sum(ranked_good[:take])
    return curve


def _resolve_metric(key: str, ctx: MetricContext) -> tuple[Any, str | None]:
    """Try the round's own params, then the training report."""
    if key in ctx.task.params:
        # "Report my own parameter back", which covers proxy_lr, cutoff_q and
        # anything else the round set for us.
        return ctx.task.params[key], "echoed from task.params"

    if ctx.report is not None and hasattr(ctx.report, key):
        value = getattr(ctx.report, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value, "TrainReport attribute"

    return None, None


def _finalize(stats: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Make agg_stats something the server will actually accept.

    Three rules from the contract, each of which is a 422 if broken: keys match
    the ID format, values are finite numbers (only `eval_spearman` and
    `proxy_lr` may be null), and there are at most 64 of them.
    """
    cleaned: dict[str, Any] = {}
    for key, value in stats.items():
        if not is_id(key):
            warnings.append(f"dropping metric {key!r}: not a valid agg_stats key")
            continue

        if value is None:
            if key in NULLABLE_METRICS:
                cleaned[key] = None
            else:
                # A null here is rejected outright, so the honest move is to
                # leave the key out rather than to invent a number for it.
                warnings.append(f"dropping metric {key!r}: null is not accepted for it")
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            warnings.append(f"dropping metric {key!r}: agg_stats takes numbers only")
            continue
        if not math.isfinite(value):
            warnings.append(f"dropping metric {key!r}: value was {value}")
            continue
        cleaned[key] = value

    if len(cleaned) > MAX_AGG_KEYS:
        # Keep the ones the contract and the gate actually run on.
        protected = {"n_chunks", *HELDOUT_KEYS, "eval_spearman", "proxy_lr"}
        keep = {k: v for k, v in cleaned.items() if k in protected}
        for key, value in cleaned.items():
            if len(keep) >= MAX_AGG_KEYS:
                break
            keep.setdefault(key, value)
        warnings.append(
            f"agg_stats had {len(cleaned)} keys; trimmed to the {MAX_AGG_KEYS} the server allows"
        )
        cleaned = keep

    return cleaned
