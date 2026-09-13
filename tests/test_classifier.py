"""Training, scoring, and the persistence that makes the three-round flow work."""

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from node.core.classifier import FastTextClassifier, build_scored
from node.core.types import LabeledSample, Sample, TrainConfig

LABELS_10 = [str(i) for i in range(1, 11)]


def in_process(config: TrainConfig) -> TrainConfig:
    """Training normally happens in a spawned process, where a monkeypatch
    applied in the parent has no effect. Tests that patch fastText itself have
    to keep the work here."""
    return dataclasses.replace(config, isolate_training=False)


# --------------------------------------------------------------- pure scoring


def test_peaked_distribution_is_confident():
    vector = np.array([0.01] * 9 + [0.91])
    scored = build_scored("s", vector, LABELS_10, [float(x) for x in LABELS_10])

    assert scored.label == "10"
    assert scored.margin == pytest.approx(0.90)
    assert scored.expected_score == pytest.approx(0.01 * 45 + 0.91 * 10)
    assert scored.entropy < 0.7


def test_uniform_distribution_is_maximally_uncertain():
    vector = np.full(10, 0.1)
    scored = build_scored("s", vector, LABELS_10, [float(x) for x in LABELS_10])

    assert scored.margin == pytest.approx(0.0)
    assert scored.entropy == pytest.approx(np.log(10))
    # The mean of 1..10 -- no information means the midpoint.
    assert scored.expected_score == pytest.approx(5.5)


def test_expected_score_is_absent_for_non_numeric_labels():
    scored = build_scored("s", np.array([0.7, 0.3]), ["good", "bad"], None)
    assert scored.expected_score is None
    assert scored.label == "good"


def test_expected_score_differs_from_argmax_on_a_split_distribution():
    # The point of returning a distribution: argmax says 1, but the mass sits
    # between 1 and 10, and the server should be able to see that.
    vector = np.array([0.34] + [0.0] * 8 + [0.33])
    vector = np.append(vector[:9], [0.33])
    scored = build_scored("s", vector, LABELS_10, [float(x) for x in LABELS_10])

    assert scored.label == "1"
    assert scored.expected_score > 3.0


# -------------------------------------------------------------------- training


def test_learns_a_separable_task(separable_samples, fast_config):
    # If this drops, the defaults underfit: fastText's stock epoch count scores
    # 0.20 on data this easy.
    classifier, report = FastTextClassifier.train(separable_samples, fast_config)

    assert report.accuracy >= 0.8
    assert report.macro_f1 >= 0.8
    assert report.n_train + report.n_eval == len(separable_samples)
    assert report.labels == LABELS_10


def test_report_carries_ordinal_metrics(separable_samples, fast_config):
    _, report = FastTextClassifier.train(separable_samples, fast_config)

    assert report.mae is not None and report.mae < 1.0
    assert report.qwk is not None and report.qwk > 0.8
    assert {r.label for r in report.per_class} == set(LABELS_10)
    assert sum(r.support for r in report.per_class) == report.n_eval


def test_ordinal_metrics_absent_for_non_numeric_labels(fast_config):
    samples = [
        LabeledSample(f"{label}-{i}", f"{label} marker word here", label)
        for label in ("spam", "ham")
        for i in range(20)
    ]
    _, report = FastTextClassifier.train(samples, fast_config)

    assert report.mae is None
    assert report.qwk is None
    assert report.accuracy is not None


def test_epoch_count_adapts_to_dataset_size():
    config = TrainConfig()
    # A constant update budget: tiny corpora get many passes, large ones few.
    assert config.resolve_epoch(200) == config.max_epoch
    assert config.resolve_epoch(4_000) == 50
    assert config.resolve_epoch(10_000_000) == config.min_epoch
    # An explicit value always wins.
    assert dataclasses.replace(config, epoch=7).resolve_epoch(200) == 7


def test_bucket_count_adapts_to_corpus_size():
    config = TrainConfig()
    # fastText's stock 2M buckets would be an 800 MB model for a tiny corpus.
    assert config.resolve_bucket(1_000) == config.min_bucket
    assert config.resolve_bucket(100_000) == 500_000
    assert config.resolve_bucket(10_000_000) == config.max_bucket
    assert dataclasses.replace(config, bucket=123).resolve_bucket(1_000) == 123


def test_training_rejects_a_single_class(fast_config):
    samples = [LabeledSample(str(i), "some text", "1") for i in range(10)]
    with pytest.raises(ValueError, match="at least 2 distinct labels"):
        FastTextClassifier.train(samples, fast_config)


def test_training_rejects_an_empty_dataset(fast_config):
    with pytest.raises(ValueError, match="no training samples"):
        FastTextClassifier.train([], fast_config)


def test_training_rejects_an_all_empty_corpus(fast_config):
    samples = [LabeledSample(str(i), "  \n ", str(i % 2)) for i in range(10)]
    with pytest.raises(ValueError, match="empty after normalization"):
        FastTextClassifier.train(samples, fast_config)


def test_skipped_count_reports_unusable_samples(separable_samples, fast_config):
    samples = separable_samples + [
        LabeledSample("blank-1", "   ", "1"),
        LabeledSample("blank-2", "\n\n", "2"),
    ]
    _, report = FastTextClassifier.train(samples, fast_config)
    assert report.n_skipped == 2


# --------------------------------------------------------------------- scoring


def test_score_returns_a_complete_distribution(separable_samples, fast_config):
    classifier, _ = FastTextClassifier.train(separable_samples, fast_config)
    scored = classifier.score([Sample("x", "volcano basalt eruption the report")])

    probabilities = scored[0].probs
    assert set(probabilities) == set(LABELS_10)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in probabilities.values())


def test_score_preserves_count_and_order(separable_samples, fast_config):
    classifier, _ = FastTextClassifier.train(separable_samples, fast_config)
    samples = [Sample(f"id-{i}", f"{markers} filler") for i, markers in enumerate(
        ["volcano basalt", "sonata violin", "compiler kernel"]
    )]

    scored = classifier.score(samples)
    assert [s.sample_id for s in scored] == ["id-0", "id-1", "id-2"]


def test_empty_documents_still_get_a_row(separable_samples, fast_config):
    # Dropping them would silently break the one-row-per-document contract
    # with the server.
    classifier, _ = FastTextClassifier.train(separable_samples, fast_config)
    scored = classifier.score(
        [Sample("a", "volcano basalt"), Sample("b", "   \n "), Sample("c", "sonata violin")]
    )

    assert [s.sample_id for s in scored] == ["a", "b", "c"]
    blank = scored[1]
    # No information means a uniform distribution, i.e. maximal uncertainty.
    assert blank.entropy == pytest.approx(np.log(10))
    assert blank.margin == pytest.approx(0.0)


def test_score_of_an_empty_batch_is_empty(separable_samples, fast_config):
    classifier, _ = FastTextClassifier.train(separable_samples, fast_config)
    assert classifier.score([]) == []


def test_only_trained_labels_are_ever_predicted(separable_samples, fast_config):
    classifier, _ = FastTextClassifier.train(separable_samples, fast_config)
    scored = classifier.score([Sample(str(i), "entirely unrelated wording") for i in range(5)])

    assert all(s.label in LABELS_10 for s in scored)


# ----------------------------------------------------------------- persistence


def test_save_load_roundtrip_preserves_scores(separable_samples, fast_config, tmp_path):
    # This is what makes the three-round protocol work: round 1 trains and
    # saves, rounds 2 and 3 load and score.
    classifier, report = FastTextClassifier.train(separable_samples, fast_config)
    probes = [Sample(f"p-{i}", text) for i, text in enumerate(
        ["volcano basalt eruption", "mortgage dividend", "telescope nebula", "unrelated text"]
    )]
    before = classifier.score(probes)

    classifier.save(tmp_path / "model")
    reloaded = FastTextClassifier.load(tmp_path / "model")
    after = reloaded.score(probes)

    assert [s.sample_id for s in after] == [s.sample_id for s in before]
    assert [s.label for s in after] == [s.label for s in before]
    for old, new in zip(before, after, strict=True):
        assert new.probs == pytest.approx(old.probs)
        assert new.expected_score == pytest.approx(old.expected_score)


def test_reloaded_classifier_keeps_labels_and_report(separable_samples, fast_config, tmp_path):
    classifier, report = FastTextClassifier.train(separable_samples, fast_config)
    classifier.save(tmp_path / "model")
    reloaded = FastTextClassifier.load(tmp_path / "model")

    assert reloaded.labels == classifier.labels
    assert reloaded.report is not None
    assert reloaded.report.accuracy == pytest.approx(report.accuracy)
    assert [r.label for r in reloaded.report.per_class] == [r.label for r in report.per_class]


def test_training_really_runs_in_another_process(separable_samples, fast_config, monkeypatch):
    # The guarantee itself: with isolation on, a fastText patched in this
    # process is not the one that trains.
    import node.core.classifier as module

    def must_not_run(*args, **kwargs):
        raise AssertionError("training ran in the parent process")

    monkeypatch.setattr(module.fasttext, "train_supervised", must_not_run)

    classifier, report = FastTextClassifier.train(separable_samples, fast_config)
    assert report.accuracy >= 0.8
    assert classifier.labels == LABELS_10


def test_training_works_when_main_is_not_a_file():
    # Regression guard. multiprocessing's "spawn" re-imports the parent's
    # __main__ in the child, which fails outright from a REPL, a notebook or
    # `python -c`, and spawns endlessly from a script without an
    # `if __name__ == "__main__"` guard. The worker is a plain subprocess so
    # that none of that applies -- this test is what keeps it that way.
    program = (
        "from node.core import FastTextClassifier, LabeledSample, TrainConfig;"
        "g=[LabeledSample(f'd{i}',t,l) for i,(t,l) in "
        "enumerate([('alpha beta','1'),('gamma delta','2')]*10)];"
        "c,r=FastTextClassifier.train(g, TrainConfig(thread=1));"
        "print('OK', r.accuracy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK"), result.stdout


def test_worker_failure_surfaces_as_an_error(separable_samples, fast_config, monkeypatch):
    # If the worker dies without reporting -- a signal, say -- the parent must
    # raise rather than sail on with no model.
    import node.core.classifier as module

    def dead_worker(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=-8, stdout="", stderr="boom")

    monkeypatch.setattr(module.subprocess, "run", dead_worker)

    with pytest.raises(RuntimeError, match="produced nothing"):
        FastTextClassifier.train(separable_samples, fast_config)


def test_load_without_metadata_fails_loudly(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        FastTextClassifier.load(empty)


def test_training_retries_the_stochastic_nan_abort(
    separable_samples, fast_config, monkeypatch
):
    # fastText aborts with "Encountered NaN" at random, on parameters that
    # succeed a moment later. It survives a build from C++ sources, so it is
    # upstream and stochastic, and a retry is the only practical answer.
    import node.core.classifier as module

    real = module.fasttext.train_supervised
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Encountered NaN.")
        return real(*args, **kwargs)

    monkeypatch.setattr(module.fasttext, "train_supervised", flaky)

    classifier, report = FastTextClassifier.train(separable_samples, in_process(fast_config))
    assert len(calls) >= 2
    assert report.accuracy >= 0.8


def test_errors_other_than_nan_are_not_retried(separable_samples, fast_config, monkeypatch):
    import node.core.classifier as module

    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("input file cannot be opened")

    monkeypatch.setattr(module.fasttext, "train_supervised", broken)

    with pytest.raises(RuntimeError, match="cannot be opened"):
        FastTextClassifier.train(separable_samples, in_process(fast_config))
    assert len(calls) == 1


def test_persistent_nan_gives_up_with_a_clear_message(
    separable_samples, fast_config, monkeypatch
):
    import node.core.classifier as module

    calls = []

    def always_nan(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("Encountered NaN.")

    monkeypatch.setattr(module.fasttext, "train_supervised", always_nan)

    with pytest.raises(RuntimeError, match="consecutive attempts"):
        FastTextClassifier.train(separable_samples, in_process(fast_config))
    assert len(calls) == fast_config.train_attempts


def test_quantization_failure_does_not_lose_the_model(separable_samples, monkeypatch):
    # fastText's product quantization raises "Encountered NaN" on small
    # corpora. Losing a successful training run to an optional size
    # optimization would be the worse outcome, so it degrades instead.
    import fasttext.FastText as fasttext_module

    def explode(self, *args, **kwargs):
        raise RuntimeError("Encountered NaN.")

    monkeypatch.setattr(fasttext_module._FastText, "quantize", explode)

    config = TrainConfig(
        thread=1, verbose=0, retrain_on_full=False, quantize=True, isolate_training=False
    )
    classifier, report = FastTextClassifier.train(separable_samples, config)

    assert classifier.quantized is False
    assert any("quantization failed" in warning for warning in report.warnings)

    scored = classifier.score([Sample("a", "volcano basalt eruption")])
    assert sum(scored[0].probs.values()) == pytest.approx(1.0)


def test_quantization_shrinks_the_saved_model(separable_samples, fast_config, tmp_path):
    plain, _ = FastTextClassifier.train(separable_samples, fast_config)
    plain.save(tmp_path / "plain")

    quantized, _ = FastTextClassifier.train(
        separable_samples, dataclasses.replace(fast_config, quantize=True)
    )
    quantized.save(tmp_path / "quantized")

    if not quantized.quantized:
        pytest.skip("fastText declined to quantize this corpus; degradation is covered above")

    plain_size = (tmp_path / "plain" / "model.bin").stat().st_size
    quantized_size = (tmp_path / "quantized" / "model.ftz").stat().st_size
    assert quantized_size < plain_size

    scored = FastTextClassifier.load(tmp_path / "quantized").score(
        [Sample("a", "volcano basalt eruption")]
    )
    assert sum(scored[0].probs.values()) == pytest.approx(1.0)
