import pytest

from control_plane import schemas
from control_plane.pooling import HeldoutCurve, NodeSubmission, select_topk, wilson_lower
from node_sdk import client


def test_sdk_curve_matches_contract():
    assert client.HELDOUT_QUANTILES == schemas.HELDOUT_QUANTILES
    stats = client.heldout_curve([0.9, 0.8, 0.1, 0.5], [True, False, False, True])
    assert set(stats) == set(schemas.HELDOUT_KEYS)
    assert (stats["ho_n"], stats["ho_good"]) == (4, 2)
    assert (stats["ho_n_q01"], stats["ho_good_q01"]) == (1, 1)
    assert (stats["ho_n_q50"], stats["ho_good_q50"]) == (2, 1)
    assert client.heldout_curve(list(range(2000)), [True] * 2000)["ho_n_q10"] == 200  # no float ceil drift


def test_wilson_lower_bound():
    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(10, 10) == pytest.approx(0.722, abs=1e-3)
    assert wilson_lower(50, 100) == pytest.approx(0.404, abs=1e-3)


def test_curve_interpolates_between_reported_quantiles():
    curve = HeldoutCurve.from_stats(client.heldout_curve(list(range(100, 0, -1)), [i < 30 for i in range(100)]))
    assert curve.at(0.1) == (10, 10)
    assert curve.at(0.4) == pytest.approx((40, 30))
    assert curve.at(1.0) == (100, 30)
    assert curve.base_rate == 0.3


def test_band_precision_is_smoothed_and_non_increasing():
    good = [i % 3 == 0 or i < 10 for i in range(200)]
    bands = HeldoutCurve.from_stats(client.heldout_curve(list(range(200, 0, -1)), good)).band_precision()
    assert len(bands) == len(schemas.HELDOUT_QUANTILES) + 1
    assert all(a >= b for a, b in zip(bands, bands[1:]))
    assert all(0 < b < 1 for b in bands)


def test_expected_precision_interpolates_bands_monotonically():
    curve = HeldoutCurve.from_stats(client.heldout_curve(list(range(200, 0, -1)), [i < 40 for i in range(200)]))
    fracs = [i / 100 for i in range(1, 101)]
    values = curve.expected_precision(fracs)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert len(set(round(v, 9) for v in values)) > len(schemas.HELDOUT_QUANTILES) + 1  # not a step per band


def test_gate_does_not_judge_a_node_on_a_handful_of_heldout_docs():
    # "small" has 40 held-out docs; its best-ranked one is bad, the next 12 are good. It gets no slots,
    # so its top 1% is a single bad doc — the gate must look at 30 docs instead (12/30 good).
    sharp = client.heldout_curve(list(range(200, 0, -1)), [i < 40 for i in range(200)])
    small = client.heldout_curve(list(range(40, 0, -1)), [1 <= i <= 12 for i in range(40)])
    nodes = [NodeSubmission("big", [f"{i:016x}" for i in range(10)], [float(i) for i in range(10)],
                            HeldoutCurve.from_stats(sharp)),
             NodeSubmission("small", [f"{i + 100:016x}" for i in range(10)], [float(i) for i in range(10)],
                            HeldoutCurve.from_stats(small))]
    top = select_topk(nodes, k=1, min_precision=0.2)
    gate = {g["node_id"]: g for g in top["gate"]}
    assert top["selected_per_node"] == {"big": 1}
    assert gate["small"]["trust"] == "trusted" and gate["small"]["heldout_at_cutoff"] == 30
