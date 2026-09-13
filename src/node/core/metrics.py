"""Evaluation metrics, on numpy alone.

Nominal metrics (accuracy, macro-F1) treat the ten labels as unrelated
categories. If the labels are actually a 1..10 scale -- which is what a "score"
usually means -- that view is misleading, because accuracy punishes an
off-by-one exactly as hard as an off-by-seven. So whenever the labels parse as
numbers we also report MAE and quadratic weighted kappa, which see the distance
between classes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from node.core.types import ClassReport


def numeric_label_values(labels: Sequence[str]) -> list[float] | None:
    """Return the labels as numbers, or None if any of them is not numeric."""
    values: list[float] = []
    for label in labels:
        try:
            values.append(float(label))
        except (TypeError, ValueError):
            return None
    return values


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    """Rows are true labels, columns predicted, both ordered as `labels`."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred, strict=True):
        # A prediction outside the known label set cannot happen with fastText,
        # but an unknown *true* label can if evaluation data drifts.
        if true in index and pred in index:
            matrix[index[true], index[pred]] += 1
    return matrix


def accuracy(matrix: np.ndarray) -> float:
    total = matrix.sum()
    return float(np.trace(matrix) / total) if total else 0.0


def per_class_reports(matrix: np.ndarray, labels: Sequence[str]) -> list[ClassReport]:
    reports: list[ClassReport] = []
    for i, label in enumerate(labels):
        true_positive = int(matrix[i, i])
        support = int(matrix[i, :].sum())
        predicted = int(matrix[:, i].sum())

        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0

        reports.append(
            ClassReport(
                label=label,
                support=support,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
            )
        )
    return reports


def macro_f1(reports: Sequence[ClassReport]) -> float:
    """Unweighted mean F1 over classes that actually occur in the evaluation set.

    Classes with no support are excluded rather than counted as zero: a label
    absent from a small holdout would otherwise drag the score down for a
    reason that says nothing about the model.
    """
    present = [r.f1 for r in reports if r.support > 0]
    return float(np.mean(present)) if present else 0.0


def mean_absolute_error(
    y_true: Sequence[str], y_pred: Sequence[str], label_values: dict[str, float]
) -> float:
    errors = [
        abs(label_values[t] - label_values[p])
        for t, p in zip(y_true, y_pred, strict=True)
        if t in label_values and p in label_values
    ]
    return float(np.mean(errors)) if errors else 0.0


def quadratic_weighted_kappa(matrix: np.ndarray, values: Sequence[float]) -> float | None:
    """Agreement corrected for chance, penalizing errors by squared distance.

    Returns None when it is undefined -- a single class present, or zero
    expected disagreement.
    """
    total = matrix.sum()
    if total == 0 or len(values) < 2:
        return None

    scale = max(values) - min(values)
    if scale == 0:
        return None

    grid = np.asarray(values, dtype=np.float64)
    weights = (grid[:, None] - grid[None, :]) ** 2 / scale**2

    observed = matrix.astype(np.float64) / total
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)).astype(np.float64)
    expected /= total**2

    denominator = float((weights * expected).sum())
    if denominator == 0:
        return None
    return float(1.0 - (weights * observed).sum() / denominator)


def entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in nats. Higher means the classifier is less sure."""
    p = probabilities[probabilities > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def rankdata(values: Sequence[float]) -> np.ndarray:
    """1-based ranks, ties sharing their mean rank.

    Mergesort because it is stable: equal values keep their input order, which
    makes the tie groups contiguous and the result reproducible.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(1, array.size + 1, dtype=np.float64)

    # Average the ranks within each run of equal values.
    sorted_values = array[order]
    start = 0
    for stop in range(1, array.size + 1):
        if stop == array.size or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = (start + stop + 1) / 2.0
            start = stop
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation between two sequences, or None when it is undefined.

    Pearson over the ranks rather than the 6*sum(d^2) shortcut: the shortcut
    assumes no ties, and on a ten-point score scale ties are the normal case,
    not the exception.

    Undefined -- and so None -- for fewer than two points, or when either side
    is constant, because a flat sequence has no ordering to correlate with.
    The result is clamped to [-1, 1]: the server's `eval_spearman` is bounded
    there, and floating-point error on a perfect correlation really does
    produce 1.0000000000000002.
    """
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if a.size < 2:
        return None
    # argsort sorts NaN to the end rather than propagating it, so a corrupt
    # input would otherwise come back as a perfectly respectable correlation.
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None

    rank_a, rank_b = rankdata(a), rankdata(b)
    centered_a = rank_a - rank_a.mean()
    centered_b = rank_b - rank_b.mean()

    denominator = float(np.sqrt((centered_a**2).sum() * (centered_b**2).sum()))
    if denominator == 0.0:
        return None

    rho = float((centered_a * centered_b).sum() / denominator)
    if not np.isfinite(rho):
        return None
    return float(np.clip(rho, -1.0, 1.0))
