"""Metrics, including the ordinal ones that accuracy alone would hide."""

import numpy as np
import pytest

from node.core import metrics

LABELS = ["1", "2", "3"]
VALUES = [1.0, 2.0, 3.0]


def test_numeric_label_values():
    assert metrics.numeric_label_values(["1", "10"]) == [1.0, 10.0]
    assert metrics.numeric_label_values(["good", "bad"]) is None


def test_confusion_matrix_orientation():
    # Rows are truth, columns predictions.
    matrix = metrics.confusion_matrix(["1", "1", "2"], ["1", "2", "2"], LABELS)
    assert matrix[0, 0] == 1
    assert matrix[0, 1] == 1
    assert matrix[1, 1] == 1
    assert matrix.sum() == 3


def test_accuracy_and_macro_f1_on_perfect_predictions():
    labels = ["1", "2", "3", "1"]
    matrix = metrics.confusion_matrix(labels, labels, LABELS)
    reports = metrics.per_class_reports(matrix, LABELS)

    assert metrics.accuracy(matrix) == 1.0
    assert metrics.macro_f1(reports) == 1.0


def test_macro_f1_ignores_classes_with_no_support():
    # Class "3" never appears; counting it as a zero would punish the model for
    # something the evaluation set, not the model, is responsible for.
    matrix = metrics.confusion_matrix(["1", "2"], ["1", "2"], LABELS)
    reports = metrics.per_class_reports(matrix, LABELS)

    assert [r.support for r in reports] == [1, 1, 0]
    assert metrics.macro_f1(reports) == 1.0


def test_per_class_precision_and_recall():
    # Truth 1,1,2 predicted 1,2,2 -> for class "2": one correct, one spurious.
    matrix = metrics.confusion_matrix(["1", "1", "2"], ["1", "2", "2"], LABELS)
    reports = {r.label: r for r in metrics.per_class_reports(matrix, LABELS)}

    assert reports["1"].precision == 1.0
    assert reports["1"].recall == pytest.approx(0.5)
    assert reports["2"].precision == pytest.approx(0.5)
    assert reports["2"].recall == 1.0


def test_mean_absolute_error_uses_label_distance():
    mapping = dict(zip(LABELS, VALUES, strict=True))
    # |1-2| + |2-3| + |3-1| = 4, over 3 samples.
    error = metrics.mean_absolute_error(["1", "2", "3"], ["2", "3", "1"], mapping)
    assert error == pytest.approx(4 / 3)


def test_accuracy_cannot_tell_a_near_miss_from_a_far_one():
    # The reason ordinal metrics exist: both of these are 0% accurate.
    near = metrics.confusion_matrix(["1", "2"], ["2", "3"], LABELS)
    far = metrics.confusion_matrix(["1", "2"], ["3", "3"], LABELS)
    assert metrics.accuracy(near) == metrics.accuracy(far) == 0.0

    mapping = dict(zip(LABELS, VALUES, strict=True))
    assert metrics.mean_absolute_error(["1", "2"], ["2", "3"], mapping) < metrics.mean_absolute_error(
        ["1", "2"], ["3", "3"], mapping
    )


def test_qwk_is_one_for_perfect_agreement():
    labels = ["1", "2", "3"] * 4
    matrix = metrics.confusion_matrix(labels, labels, LABELS)
    assert metrics.quadratic_weighted_kappa(matrix, VALUES) == pytest.approx(1.0)


def test_qwk_penalizes_distance():
    truth = ["1", "2", "3", "1", "2", "3"]
    near = ["2", "3", "2", "1", "2", "3"]
    far = ["3", "3", "1", "3", "1", "1"]

    kappa_near = metrics.quadratic_weighted_kappa(
        metrics.confusion_matrix(truth, near, LABELS), VALUES
    )
    kappa_far = metrics.quadratic_weighted_kappa(
        metrics.confusion_matrix(truth, far, LABELS), VALUES
    )
    assert kappa_near > kappa_far


def test_qwk_is_none_when_undefined():
    # A single class present means there is no expected disagreement to
    # normalize against.
    matrix = metrics.confusion_matrix(["1", "1"], ["1", "1"], LABELS)
    assert metrics.quadratic_weighted_kappa(matrix, VALUES) is None
    assert metrics.quadratic_weighted_kappa(np.zeros((3, 3), dtype=np.int64), VALUES) is None


def test_entropy_is_highest_for_a_uniform_distribution():
    uniform = np.full(10, 0.1)
    peaked = np.array([0.91] + [0.01] * 9)

    assert metrics.entropy(uniform) == pytest.approx(np.log(10))
    assert metrics.entropy(uniform) > metrics.entropy(peaked)
    assert metrics.entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
