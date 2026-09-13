"""Rank correlation -- the node's honesty signal about its own proxy scores.

The server feeds `eval_spearman` into its Reliability Gate, so a wrong value
here is worse than no value: it buys trust the node has not earned.
"""

import numpy as np
import pytest

from node.core import metrics


def test_rankdata_without_ties():
    assert list(metrics.rankdata([10.0, 30.0, 20.0])) == [1.0, 3.0, 2.0]


def test_rankdata_averages_ties():
    # Positions 2 and 3 tie, so both take (2 + 3) / 2.
    assert list(metrics.rankdata([5.0, 7.0, 7.0, 9.0])) == [1.0, 2.5, 2.5, 4.0]


def test_rankdata_all_tied():
    assert list(metrics.rankdata([4.0, 4.0, 4.0])) == [2.0, 2.0, 2.0]


def test_rankdata_empty():
    assert metrics.rankdata([]).size == 0


def test_perfect_agreement_is_exactly_one():
    # Not "approximately 1.0": the server bounds eval_spearman at 1.0, and an
    # unclamped Pearson over ranks returns 1.0000000000000002 often enough to
    # fail a submit for a reason nobody would think to look for.
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert metrics.spearman(x, x) == 1.0


def test_perfect_disagreement_is_minus_one():
    assert metrics.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0


def test_monotone_but_nonlinear_still_scores_one():
    # This is the point of a *rank* correlation: the proxy only has to order
    # the documents the same way, not reproduce the teacher's scale.
    predicted = [1.0, 2.0, 3.0, 4.0]
    truth = [1.0, 1.5, 9.0, 100.0]
    assert metrics.spearman(predicted, truth) == 1.0


def test_known_value_with_ties():
    # Ranks: x -> [1, 2.5, 2.5, 4], y -> [1.5, 1.5, 3, 4].
    # Centered: x -> [-1.5, 0, 0, 1.5], y -> [-1, -1, 0.5, 1.5].
    # numerator 1.5 + 0 + 0 + 2.25 = 3.75; denominator sqrt(4.5 * 4.5) = 4.5.
    rho = metrics.spearman([1.0, 2.0, 2.0, 3.0], [5.0, 5.0, 6.0, 7.0])
    assert rho == pytest.approx(3.75 / 4.5)


def test_undefined_cases_return_none():
    assert metrics.spearman([], []) is None
    assert metrics.spearman([1.0], [2.0]) is None
    # A constant sequence has no ordering, so there is nothing to correlate.
    assert metrics.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert metrics.spearman([1.0, 2.0, 3.0], [7.0, 7.0, 7.0]) is None


def test_nan_input_returns_none_rather_than_nan():
    # A NaN would sail through json.dumps as a bare `NaN` token and be rejected
    # remotely; None is a value the schema actually permits.
    assert metrics.spearman([1.0, 2.0, float("nan")], [1.0, 2.0, 3.0]) is None


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        metrics.spearman([1.0, 2.0], [1.0, 2.0, 3.0])


def test_result_always_within_server_bounds():
    rng = np.random.default_rng(20260913)
    for _ in range(200):
        n = int(rng.integers(2, 40))
        # Small integer range on purpose: it manufactures ties, which is where
        # a naive implementation drifts outside [-1, 1].
        x = rng.integers(0, 5, size=n).astype(float)
        y = rng.integers(0, 5, size=n).astype(float)
        rho = metrics.spearman(x, y)
        assert rho is None or -1.0 <= rho <= 1.0
