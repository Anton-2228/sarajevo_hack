"""The view the loop hands out is live and mutable. This is the guard.

`snapshot_view` lives in a Qt module, but it imports nothing from Qt itself,
so this test runs without PyQt5 installed -- it is the one piece of the
threading design that can be proven without a display.
"""

from __future__ import annotations

import pytest

from node.agent.models import AgentView, TaskProgress

pytest.importorskip("PyQt5", reason="GUI extra not installed")

from node.gui.worker import snapshot_view  # noqa: E402


def _view() -> AgentView:
    return AgentView(
        server_url="https://example.test",
        status="busy",
        node_id="node-a",
        budget_cores=2,
        budget_ram_gb=4.0,
        current=TaskProgress(
            round_id="r1", phase="running", started_at=1.0, detail={"mode": "train"}
        ),
        completed=3,
        failed=1,
    )


def test_the_snapshot_survives_the_loop_moving_on() -> None:
    live = _view()
    taken = snapshot_view(live)

    live.status = "stopping"
    live.completed = 99
    live.failed = 42
    live.last_error = "something later"
    live.node_id = "node-b"

    assert taken.status == "busy"
    assert taken.completed == 3
    assert taken.failed == 1
    assert taken.last_error is None
    assert taken.node_id == "node-a"


def test_the_nested_progress_is_copied_too() -> None:
    live = _view()
    taken = snapshot_view(live)

    assert live.current is not None
    live.current.phase = "done"
    live.current.detail["n_scores"] = 500

    assert taken.current is not None
    assert taken.current.phase == "running"
    assert "n_scores" not in taken.current.detail


def test_replacing_the_current_round_does_not_reach_back() -> None:
    live = _view()
    taken = snapshot_view(live)

    live.current = TaskProgress(round_id="r2", phase="acked", started_at=2.0)

    assert taken.current is not None
    assert taken.current.round_id == "r1"


def test_a_view_with_no_round_snapshots_cleanly() -> None:
    live = AgentView(server_url="https://example.test")

    assert snapshot_view(live).current is None
