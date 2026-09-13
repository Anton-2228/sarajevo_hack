"""The node's classifier: trained once in round 1, reloaded to score in rounds 2 and 3."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import fasttext
import numpy as np

from node.core import metrics
from node.core.dataset import WriteStats, stratified_split, write_training_file
from node.core.limits import AppliedLimits
from node.core.text import normalize_text, strip_label_prefix
from node.core.types import (
    ClassReport,
    LabeledSample,
    Sample,
    ScoredSample,
    TrainConfig,
    TrainReport,
)

META_FILENAME = "meta.json"


def _sorted_labels(labels: Sequence[str]) -> list[str]:
    """Order labels numerically when they are numbers, lexicographically otherwise.

    Consistent ordering matters: it fixes the column order of the confusion
    matrix and of every probability vector sent to the server.
    """
    unique = set(labels)
    values = metrics.numeric_label_values(sorted(unique))
    if values is not None:
        return sorted(unique, key=float)
    return sorted(unique)


def _resolved_params(model: Any, config: TrainConfig, stats: WriteStats) -> dict[str, Any]:
    """Read back the hyperparameters a trained model actually used.

    After autotuning these differ from the requested ones, and we need them to
    retrain on the full dataset. The accessor is internal to fastText, so a
    failure here falls back to the configured defaults rather than crashing.
    """
    try:
        args = model.f.getArgs()
        loss = getattr(args.loss, "name", None) or str(args.loss).rsplit(".", 1)[-1]
        return {
            "lr": args.lr,
            "dim": args.dim,
            "epoch": args.epoch,
            "wordNgrams": args.wordNgrams,
            "minCount": args.minCount,
            "bucket": args.bucket,
            "minn": args.minn,
            "maxn": args.maxn,
            "neg": args.neg,
            "ws": args.ws,
            "lrUpdateRate": args.lrUpdateRate,
            "t": args.t,
            "loss": loss,
            "thread": config.resolve_thread(),
            "verbose": config.verbose,
        }
    except Exception:  # noqa: BLE001 - internal API, degrade to defaults
        params = config.fasttext_kwargs(stats.lines, stats.tokens)
        params["verbose"] = config.verbose
        return params


def _train_with_retry(build: Callable[[], Any], attempts: int) -> Any:
    """Run a fastText training, retrying the stochastic "Encountered NaN" abort.

    Only that specific failure is retried; anything else is a real error and
    propagates immediately.
    """
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            return build()
        except RuntimeError as error:
            if "NaN" not in str(error):
                raise
            last_error = error
    raise RuntimeError(
        f"fastText training hit NaN on {attempts} consecutive attempts"
    ) from last_error


def build_scored(
    sample_id: str,
    vector: np.ndarray,
    labels: Sequence[str],
    label_values: Sequence[float] | None = None,
) -> ScoredSample:
    """Turn a probability vector into the payload the server receives.

    Pure, so the uncertainty arithmetic can be checked against hand-built
    distributions without training anything.
    """
    order = np.argsort(vector)[::-1]
    top = int(order[0])
    margin = float(vector[top] - vector[int(order[1])]) if vector.size > 1 else 1.0

    expected: float | None = None
    if label_values is not None:
        expected = float(np.dot(vector, np.asarray(label_values, dtype=np.float64)))

    return ScoredSample(
        sample_id=sample_id,
        label=labels[top],
        probs={label: float(p) for label, p in zip(labels, vector, strict=True)},
        entropy=metrics.entropy(np.asarray(vector, dtype=np.float64)),
        margin=margin,
        expected_score=expected,
    )


class FastTextClassifier:
    """Wraps a supervised fastText model with the node's scoring contract."""

    def __init__(
        self,
        model: Any,
        labels: Sequence[str],
        config: TrainConfig,
        report: TrainReport | None = None,
        quantized: bool = False,
    ) -> None:
        self._model = model
        self.labels = list(labels)
        self.config = config
        self.report = report
        self.quantized = quantized
        self._label_values = metrics.numeric_label_values(self.labels)

    # ------------------------------------------------------------------ train

    @classmethod
    def train(
        cls, samples: Sequence[LabeledSample], config: TrainConfig | None = None
    ) -> tuple[FastTextClassifier, TrainReport]:
        """Fit on the teacher's labelled golden set. This is round 1.

        Returns the classifier and a report describing how well it reproduces
        the teacher on a held-out slice. Call `save()` afterwards so rounds 2
        and 3 can pick the model up.
        """
        config = config or TrainConfig()

        # Validate here rather than in the worker, so these errors surface
        # directly instead of through a process boundary.
        if not samples:
            raise ValueError("no training samples")
        if len({s.label for s in samples}) < 2:
            raise ValueError(
                f"need at least 2 distinct labels, got {_sorted_labels([s.label for s in samples])}"
            )

        holder = tempfile.TemporaryDirectory(prefix="node-model-")
        try:
            if config.isolate_training:
                result = _run_isolated(list(samples), config, holder.name)
            else:
                result = _fit(list(samples), config, holder.name)

            model = fasttext.load_model(str(Path(holder.name) / result.model_filename))
            report = _with_limits(result.report, result.limits)
            classifier = cls(
                model,
                result.labels,
                config,
                report=report,
                quantized=result.quantized,
            )
            return classifier, report
        finally:
            holder.cleanup()



    def _evaluate(self, samples: Sequence[LabeledSample], labels: Sequence[str]) -> TrainReport:
        """Score a held-out slice and turn it into a report."""
        if not samples:
            # Every class had a single sample, so nothing could be held out.
            # Report no metrics rather than metrics of zero.
            return TrainReport(
                n_samples=0,
                n_skipped=0,
                n_train=0,
                n_eval=0,
                labels=list(labels),
                autotuned=False,
                accuracy=None,
                macro_f1=None,
                mae=None,
                qwk=None,
                metrics_measured_on="nothing (dataset too small to hold out)",
            )

        scored = self.score([Sample(s.sample_id, s.text) for s in samples])
        y_true = [s.label for s in samples]
        y_pred = [s.label for s in scored]

        matrix = metrics.confusion_matrix(y_true, y_pred, labels)
        per_class = metrics.per_class_reports(matrix, labels)

        values = metrics.numeric_label_values(labels)
        mae: float | None = None
        qwk: float | None = None
        if values is not None:
            mapping = dict(zip(labels, values, strict=True))
            mae = metrics.mean_absolute_error(y_true, y_pred, mapping)
            qwk = metrics.quadratic_weighted_kappa(matrix, values)

        return TrainReport(
            n_samples=len(samples),
            n_skipped=0,
            n_train=0,
            n_eval=len(samples),
            labels=list(labels),
            autotuned=False,
            accuracy=metrics.accuracy(matrix),
            macro_f1=metrics.macro_f1(per_class),
            mae=mae,
            qwk=qwk,
            per_class=per_class,
        )

    # ------------------------------------------------------------------ score

    def score(self, samples: Sequence[Sample]) -> list[ScoredSample]:
        """Score every sample, preserving order and count.

        A document that normalizes to nothing still gets a result -- a uniform
        distribution, meaning "no information" -- because dropping it would
        silently break the one-row-per-document contract with the server.
        """
        if not samples:
            return []

        texts = [normalize_text(s.text) for s in samples]
        scorable = [i for i, text in enumerate(texts) if text]

        predictions: dict[int, dict[str, float]] = {}
        if scorable:
            label_lists, prob_lists = self._model.predict(
                [texts[i] for i in scorable], k=len(self.labels)
            )
            for position, index in enumerate(scorable):
                predictions[index] = {
                    strip_label_prefix(label): float(probability)
                    for label, probability in zip(
                        label_lists[position], prob_lists[position], strict=True
                    )
                }

        uniform = 1.0 / len(self.labels)
        results: list[ScoredSample] = []
        for index, sample in enumerate(samples):
            raw = predictions.get(index)
            if raw is None:
                vector = np.full(len(self.labels), uniform)
            else:
                vector = np.array([raw.get(label, 0.0) for label in self.labels])
                total = vector.sum()
                vector = vector / total if total > 0 else np.full(len(self.labels), uniform)

            results.append(self._build_scored(sample.sample_id, vector))
        return results

    def _build_scored(self, sample_id: str, vector: np.ndarray) -> ScoredSample:
        return build_scored(sample_id, vector, self.labels, self._label_values)

    # ------------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> Path:
        """Persist model and metadata so a later round can pick it up."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_filename = "model.ftz" if self.quantized else "model.bin"
        self._model.save_model(str(directory / model_filename))

        meta = {
            "model_file": model_filename,
            "quantized": self.quantized,
            "labels": self.labels,
            "config": asdict(self.config),
            "report": self.report.to_dict() if self.report else None,
        }
        (directory / META_FILENAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> FastTextClassifier:
        """Reopen a model saved by an earlier round, ready to score."""
        directory = Path(directory)
        meta_path = directory / META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(f"no {META_FILENAME} in {directory}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model = fasttext.load_model(str(directory / meta["model_file"]))

        config = TrainConfig(**meta.get("config", {}))
        report_data = meta.get("report")
        report = None
        if report_data:
            report_data = dict(report_data)
            report_data["per_class"] = [
                ClassReport(**entry) for entry in report_data.get("per_class", [])
            ]
            report = TrainReport(**report_data)

        return cls(
            model,
            meta["labels"],
            config,
            report=report,
            quantized=meta.get("quantized", False),
        )


@dataclass(frozen=True)
class _FitResult:
    """What the training worker hands back across the process boundary."""

    model_filename: str
    labels: list[str]
    report: TrainReport
    quantized: bool
    # Resource caps the worker imposed on itself, or None when training ran
    # in-process and so had none. Defaulted, because `_fit` constructs this
    # without knowing anything about limits.
    limits: AppliedLimits | None = None


def _with_limits(report: TrainReport, limits: AppliedLimits | None) -> TrainReport:
    """Fold the worker's self-imposed caps into the report.

    Caps that were applied are telemetry, so they go in `params`. Caps that
    could not be applied are a weaker promise than the node advertised, which
    is exactly what `warnings` is for.
    """
    if limits is None:
        return report
    return replace(
        report,
        params={**report.params, "limits": list(limits.applied)},
        warnings=[*report.warnings, *(f"limit not applied: {f}" for f in limits.failures)],
    )


def _run_isolated(
    samples: list[LabeledSample], config: TrainConfig, out_dir: str
) -> _FitResult:
    """Run the fit in a brand-new interpreter that has never loaded fastText.

    See `node.core._worker` for why this is a subprocess rather than
    `multiprocessing`.
    """
    with tempfile.TemporaryDirectory(prefix="node-ipc-") as exchange:
        spec = Path(exchange) / "spec.pkl"
        result = Path(exchange) / "result.pkl"
        spec.write_bytes(pickle.dumps((samples, config, out_dir)))

        process = subprocess.run(
            [sys.executable, "-m", "node.core._worker", str(spec), str(result)],
            capture_output=not config.verbose,
            text=True,
        )

        if not result.exists():
            # The worker died without reporting -- a signal, most likely.
            detail = (process.stderr or "").strip().splitlines()[-3:]
            raise RuntimeError(
                f"training worker exited with {process.returncode} and produced nothing"
                + (f": {' | '.join(detail)}" if detail else "")
            )

        status, payload = pickle.loads(result.read_bytes())

    if status == "error":
        raise payload
    return payload


def _fit(
    samples: list[LabeledSample], config: TrainConfig, out_dir: str
) -> _FitResult:
    """Do the whole fastText pipeline and leave the model in `out_dir`.

    Module-level and picklable-by-reference so it can be the target of a
    spawned process.
    """
    started = time.monotonic()
    labels = _sorted_labels([s.label for s in samples])
    autotune = len(samples) >= config.autotune_min_samples

    # With autotuning the search consumes a validation split, so measuring on
    # that same split would report the search's own optimism. A third slice
    # keeps the reported number honest.
    if autotune:
        rest, eval_split = stratified_split(
            samples, config.holdout_fraction, config.split_seed
        )
        fit_split, tune_split = stratified_split(
            rest, config.holdout_fraction, config.split_seed + 1
        )
    else:
        fit_split, eval_split = stratified_split(
            samples, config.holdout_fraction, config.split_seed
        )
        tune_split = []

    with tempfile.TemporaryDirectory(prefix="node-fasttext-") as workdir:
        work = Path(workdir)
        train_path = work / "train.txt"
        stats = write_training_file(fit_split, train_path)
        if stats.lines == 0:
            raise ValueError("every training sample was empty after normalization")

        if autotune and tune_split:
            tune_path = work / "tune.txt"
            write_training_file(tune_split, tune_path)
            extra: dict[str, Any] = {}
            if config.autotune_model_size:
                extra["autotuneModelSize"] = config.autotune_model_size
            model = _train_with_retry(
                lambda: fasttext.train_supervised(
                    input=str(train_path),
                    autotuneValidationFile=str(tune_path),
                    autotuneDuration=config.autotune_seconds,
                    thread=config.resolve_thread(),
                    verbose=config.verbose,
                    **extra,
                ),
                config.train_attempts,
            )
        else:
            autotune = False
            kwargs = config.fasttext_kwargs(stats.lines, stats.tokens)
            model = _train_with_retry(
                lambda: fasttext.train_supervised(
                    input=str(train_path), **kwargs, verbose=config.verbose
                ),
                config.train_attempts,
            )

        params = _resolved_params(model, config, stats)
        report = FastTextClassifier(model, labels, config)._evaluate(eval_split, labels)

        # A size-capped autotune hands back an already-quantized model;
        # retraining it would throw the quantization away and blow the budget
        # the caller asked for.
        size_capped = bool(autotune and config.autotune_model_size)
        retrain = config.retrain_on_full and not size_capped

        # Ship a model that has seen everything; the report above describes how
        # this procedure performs, not these exact weights.
        if retrain:
            full_path = work / "full.txt"
            write_training_file(samples, full_path)
            model = _train_with_retry(
                lambda: fasttext.train_supervised(input=str(full_path), **params),
                config.train_attempts,
            )

        quantized = size_capped
        warnings: list[str] = []
        if config.quantize and not size_capped:
            source = work / ("full.txt" if retrain else "train.txt")
            try:
                model.quantize(
                    input=str(source), retrain=config.quantize_retrain, qnorm=True
                )
                quantized = True
            except RuntimeError as error:
                # fastText's product quantization runs k-means over the weight
                # subvectors and raises "Encountered NaN" when a small corpus
                # leaves it with empty clusters. Shipping a working full-size
                # model beats losing the training run over an optional size
                # optimization.
                warnings.append(f"quantization failed, model left unquantized: {error}")

        model_filename = "model.ftz" if quantized else "model.bin"
        model.save_model(str(Path(out_dir) / model_filename))

    report = replace(
        report,
        n_samples=len(samples),
        n_skipped=sum(1 for s in samples if not normalize_text(s.text)),
        n_train=len(fit_split),
        n_eval=len(eval_split),
        labels=labels,
        autotuned=autotune,
        warnings=warnings,
        params=params,
        duration_s=round(time.monotonic() - started, 3),
        model_trained_on="full" if retrain else "train-split",
    )
    return _FitResult(model_filename, labels, report, quantized)
