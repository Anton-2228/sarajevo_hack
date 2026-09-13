"""The agent reports its view only after a round; between rounds we ask for it.

Without the poll a node that enrolled and then sat waiting for work showed no
node_id and no status at all until its first round landed -- or until it was
stopped, which is when `AgentLoop.run`'s `finally` finally reports one.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui

pytest.importorskip("PyQt5", reason="GUI extra not installed")

from node.agent.config import AgentConfig  # noqa: E402
from node.agent.models import AgentView, TaskProgress  # noqa: E402
from node.gui.worker import AgentController, AgentWorker  # noqa: E402


class _StubLoop:
    """Stands in for AgentLoop: the worker only ever reads `.view` off it."""

    def __init__(self, view: AgentView) -> None:
        self.view = view
        self.stopped = False

    def request_stop(self) -> None:
        self.stopped = True


def _view(**overrides) -> AgentView:
    base = {"server_url": "https://example.test", "status": "idle", "node_id": "node-a"}
    return AgentView(**{**base, **overrides})


@pytest.fixture
def worker(qapp) -> AgentWorker:
    return AgentWorker(AgentConfig(name="node-test"))


def test_no_loop_yet_means_no_view(worker: AgentWorker) -> None:
    assert worker.current_view() is None


def test_the_view_comes_back_as_a_snapshot(worker: AgentWorker) -> None:
    live = _view(current=TaskProgress("r1", "running", 0.0, {"mode": "train"}))
    worker._loop = _StubLoop(live)

    taken = worker.current_view()
    live.status = "busy"
    live.node_id = "node-b"
    assert live.current is not None
    live.current.detail["mode"] = "score"

    assert taken is not None
    assert taken.status == "idle"
    assert taken.node_id == "node-a"
    assert taken.current is not None
    assert taken.current.detail["mode"] == "train"


def test_polling_emits_the_view_once_it_changes(qapp, worker: AgentWorker) -> None:
    controller = AgentController()
    controller._worker = worker
    seen: list[AgentView] = []
    controller.state.connect(seen.append)

    controller._poll_view()  # no loop yet
    assert seen == []

    loop = _StubLoop(_view())
    worker._loop = loop
    controller._poll_view()
    assert [v.node_id for v in seen] == ["node-a"]

    # Unchanged: nothing new to say.
    controller._poll_view()
    assert len(seen) == 1

    loop.view.status = "busy"
    controller._poll_view()
    assert [v.status for v in seen] == ["idle", "busy"]


def test_a_pushed_state_is_not_repeated_by_the_next_poll(qapp, worker) -> None:
    """The reporter push and the poll must not double up on the same view."""
    controller = AgentController()
    controller._worker = worker
    seen: list[AgentView] = []
    controller.state.connect(seen.append)

    loop = _StubLoop(_view(completed=3))
    worker._loop = loop

    controller._on_state(worker.current_view())
    assert len(seen) == 1

    controller._poll_view()
    assert len(seen) == 1


def test_a_stop_during_construction_is_not_lost(worker: AgentWorker) -> None:
    """Cancel can land before the loop exists; the worker has to hold on to it."""
    worker.request_stop()
    loop = _StubLoop(_view())

    # What AgentWorker.run does once construction returns.
    with worker._lock:
        worker._loop = loop
        if worker._stop_requested:
            loop.request_stop()

    assert loop.stopped
