"""The heartbeat snapshot: stages, gauges, and the rules the guide imposes."""

import math
import threading

import pytest

from node.agent.models import MAX_LOAD_KEYS, STAGES
from node.agent.resources import Budget
from node.agent.telemetry import Telemetry

BUDGET = Budget(cores=2, ram_bytes=2 * 1024**3, core_ids=[0, 1])


@pytest.fixture
def telemetry():
    return Telemetry(BUDGET)


def test_an_idle_node_sends_no_round_and_no_stage(telemetry):
    beat = telemetry.snapshot()
    assert beat.status == "idle"
    assert beat.round_id is None
    assert beat.stage is None
    # Omitted rather than null: the schema is additionalProperties:false.
    body = beat.to_dict()
    assert "round_id" not in body
    assert "stage" not in body


def test_a_busy_node_names_its_round_and_phase(telemetry):
    telemetry.begin("r1", "downloading")
    beat = telemetry.snapshot()
    assert (beat.status, beat.round_id, beat.stage) == ("busy", "r1", "downloading")
    # Machine gauges are sampled at send time, not carried in the snapshot.
    assert "cpu_pct" in beat.load
    assert "ram_pct" in beat.load


def test_finishing_a_task_drops_the_finished_rounds_numbers(telemetry):
    """METRICS_GUIDE: idle means idle, with none of the last task attached.

    A stale progress_pct of 100 sitting on an idle node is a line the dashboard
    would draw as if the work were still there.
    """
    telemetry.begin("r1", "scoring")
    telemetry.progress(500, 500)
    assert telemetry.snapshot().load["progress_pct"] == 100.0

    telemetry.idle()
    beat = telemetry.snapshot()
    assert beat.round_id is None and beat.stage is None
    assert beat.load == {}


def test_progress_is_per_phase_not_per_task(telemetry):
    # Training to 100% then moving to scoring must not start scoring at 100%.
    telemetry.begin("r1", "training")
    telemetry.progress(100, 100)
    assert telemetry.snapshot().load["progress_pct"] == 100.0

    telemetry.stage("scoring", total=1000)
    load = telemetry.snapshot().load
    assert "progress_pct" not in load
    assert "docs_processed" not in load
    assert load["docs_total"] == 1000.0


def test_progress_reports_the_guides_stable_keys(telemetry):
    telemetry.begin("r1", "scoring")
    telemetry.stage("scoring", total=1000)
    telemetry.progress(250)
    telemetry.progress(500)
    load = telemetry.snapshot().load

    assert load["docs_processed"] == 500.0
    assert load["docs_total"] == 1000.0
    assert load["progress_pct"] == 50.0
    # Rate and ETA need two marks to exist at all.
    assert load["docs_per_sec"] >= 0.0
    assert load["eta_s"] >= 0.0


def test_progress_never_exceeds_one_hundred_percent(telemetry):
    telemetry.begin("r1", "scoring")
    telemetry.progress(1500, 1000)
    assert telemetry.snapshot().load["progress_pct"] == 100.0


def test_an_unknown_stage_is_refused_locally(telemetry):
    # The server's enum is closed, and a 422 on a heartbeat would make the node
    # look offline for a reason nothing in the logs would explain.
    with pytest.raises(ValueError, match="stage must be one of"):
        telemetry.stage("uploadingg")
    for stage in STAGES:
        telemetry.stage(stage)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "73.5", True, None])
def test_load_refuses_anything_that_is_not_a_finite_number(telemetry, value):
    telemetry.begin("r1")
    telemetry.gauge("suspect", value)
    load = telemetry.snapshot().load
    assert "suspect" not in load
    assert all(isinstance(v, float) and math.isfinite(v) for v in load.values())


def test_a_gauge_name_that_is_not_an_id_is_dropped(telemetry):
    # Also a cardinality rule: gauge names are a fixed vocabulary, never built
    # from a document, a path or a host.
    telemetry.begin("r1")
    telemetry.gauge("processing file 123!", 1.0)
    assert not any(" " in k for k in telemetry.snapshot().load)


def test_load_is_trimmed_to_the_thirty_two_the_server_allows(telemetry):
    telemetry.begin("r1", "training")
    telemetry.stage("training", total=100)
    telemetry.progress(50)
    for i in range(60):
        telemetry.gauge(f"extra_{i}", float(i))

    load = telemetry.snapshot().load
    assert len(load) <= MAX_LOAD_KEYS
    # The keys the dashboard actually draws survive the trim; the node's own
    # diagnostics are what give way.
    for key in ("progress_pct", "docs_processed", "docs_total", "cpu_pct", "ram_pct"):
        assert key in load


def test_the_snapshot_can_be_read_while_the_work_loop_writes(telemetry):
    """One writer, one reader, and the reader is the heartbeat thread.

    A snapshot torn halfway through an update would post a progress_pct that
    belongs to neither phase.
    """
    telemetry.begin("r1", "scoring")
    stop = threading.Event()
    seen = []

    def reader():
        while not stop.is_set():
            beat = telemetry.snapshot()
            seen.append(beat.load.get("docs_processed"))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for i in range(1, 2001):
            telemetry.progress(i, 2000)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert seen
    assert telemetry.snapshot().load["docs_processed"] == 2000.0
