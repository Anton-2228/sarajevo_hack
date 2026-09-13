"""A failed round is never announced; it has to be inferred. Hence these."""

from __future__ import annotations

import pytest

from node.agent.models import AgentView, TaskProgress
from node.gui.rounds import (
    ACKED,
    DONE,
    FAILED,
    RUNNING,
    RoundEntry,
    RoundTracker,
    describe,
    format_elapsed,
)


def _task(round_id: str, phase: str, started: float = 100.0, **detail) -> TaskProgress:
    return TaskProgress(round_id=round_id, phase=phase, started_at=started, detail=detail)


def _view(**overrides) -> AgentView:
    base = {"server_url": "https://example.test", "status": "busy"}
    return AgentView(**{**base, **overrides})


def test_a_round_walks_the_three_phases() -> None:
    tracker = RoundTracker()

    tracker.on_task(_task("r1", ACKED))
    assert tracker.current() is not None
    assert tracker.current().step_index == 1

    tracker.on_task(_task("r1", RUNNING, mode="train"))
    assert tracker.current().step_index == 2
    assert tracker.current().mode == "train"

    tracker.on_task(_task("r1", DONE, n_scores=1480, seconds=52.0))
    entry = tracker.entries()[0]
    assert entry.phase == DONE
    assert entry.step_index == 3
    assert entry.finished
    assert tracker.current() is None


def test_the_same_round_is_updated_not_duplicated() -> None:
    tracker = RoundTracker()
    for phase in (ACKED, RUNNING, DONE):
        tracker.on_task(_task("r1", phase))

    assert len(tracker.entries()) == 1


def test_newest_first() -> None:
    tracker = RoundTracker()
    tracker.on_task(_task("r1", DONE))
    tracker.on_task(_task("r2", ACKED))

    assert [e.round_id for e in tracker.entries()] == ["r2", "r1"]


def test_a_failure_is_inferred_from_the_counter() -> None:
    """The agent bumps view.failed and says nothing else about the round."""
    tracker = RoundTracker()
    tracker.on_task(_task("r1", RUNNING))

    entry = tracker.on_state(_view(failed=1, last_error="training worker exited with -9"))

    assert entry is not None
    assert entry.round_id == "r1"
    assert entry.phase == FAILED
    assert entry.error == "training worker exited with -9"
    assert entry.finished


def test_a_steady_failed_counter_changes_nothing() -> None:
    """Every poll carries the same totals; only a change means a new failure."""
    tracker = RoundTracker()
    tracker.on_task(_task("r1", RUNNING))
    tracker.on_state(_view(failed=1))

    assert tracker.on_state(_view(failed=1)) is None
    assert tracker.on_state(_view(failed=1)) is None
    assert len(tracker.entries()) == 1


def test_a_failure_does_not_touch_an_already_finished_round() -> None:
    tracker = RoundTracker()
    tracker.on_task(_task("r1", DONE, n_scores=10))
    tracker.on_task(_task("r2", RUNNING))

    tracker.on_state(_view(failed=1, last_error="boom"))

    by_id = {e.round_id: e for e in tracker.entries()}
    assert by_id["r1"].phase == DONE
    assert by_id["r2"].phase == FAILED


def test_a_failure_with_no_task_event_still_lands() -> None:
    """A round can die in the ack, before `running` is ever reported."""
    tracker = RoundTracker()

    entry = tracker.on_state(
        _view(failed=1, last_error="round closed", current=_task("r9", ACKED))
    )

    assert entry is not None
    assert entry.round_id == "r9"
    assert entry.phase == FAILED


def test_a_failure_with_nothing_open_is_ignored() -> None:
    """A counter already non-zero at the first poll must not invent a round."""
    tracker = RoundTracker()

    assert tracker.on_state(_view(failed=2)) is None
    assert tracker.entries() == []


def test_the_list_is_capped() -> None:
    tracker = RoundTracker(limit=3)
    for i in range(6):
        tracker.on_task(_task(f"r{i}", DONE))

    assert [e.round_id for e in tracker.entries()] == ["r5", "r4", "r3"]


def test_reset_clears_everything_including_the_failure_baseline() -> None:
    tracker = RoundTracker()
    tracker.on_task(_task("r1", RUNNING))
    tracker.on_state(_view(failed=1))
    tracker.reset()

    assert tracker.entries() == []
    tracker.on_task(_task("r2", RUNNING))
    assert tracker.on_state(_view(failed=1)) is not None


def test_elapsed_freezes_once_the_round_is_over() -> None:
    tracker = RoundTracker()
    tracker.on_task(_task("r1", DONE, seconds=52.0))

    entry = tracker.entries()[0]
    assert entry.elapsed(now=1e9) == 52.0


def test_elapsed_runs_while_the_round_does() -> None:
    tracker = RoundTracker()
    tracker.on_task(_task("r1", RUNNING, started=100.0))

    assert tracker.entries()[0].elapsed(now=130.0) == 30.0


@pytest.mark.parametrize(
    "phase, detail, expected",
    [
        (ACKED, {}, "accepted, preparing"),
        (RUNNING, {"mode": "train"}, "training · train"),
        (DONE, {"n_scores": 1480, "seconds": 52.0}, "1480 scores · 52 s"),
        (DONE, {"seconds": 52.0}, "52 s"),
    ],
)
def test_describe(phase: str, detail: dict, expected: str) -> None:
    tracker = RoundTracker()
    tracker.on_task(_task("r1", phase, **detail))

    assert describe(tracker.entries()[0]) == expected


def test_describe_a_bare_completion() -> None:
    """Unreachable through on_task, which fills in the duration -- but describe
    is also handed entries that came the other way."""
    assert describe(RoundEntry(round_id="r1", phase=DONE)) == "submitted"


def test_describe_a_failure_prefers_the_real_error() -> None:
    tracker = RoundTracker()
    tracker.on_task(_task("r1", RUNNING))
    tracker.on_state(_view(failed=1, last_error="control plane unreachable"))

    assert describe(tracker.entries()[0]) == "control plane unreachable"


@pytest.mark.parametrize(
    "seconds, expected", [(0, "0:00"), (9, "0:09"), (75, "1:15"), (3725, "1:02:05")]
)
def test_format_elapsed(seconds: float, expected: str) -> None:
    assert format_elapsed(seconds) == expected
