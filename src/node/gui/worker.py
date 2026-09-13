"""Running the agent off the GUI thread, and getting its state back safely.

The agent is synchronous and blocking by design ("no asyncio, because there is
exactly one concurrent thing happening"), so the window runs `AgentLoop.run()`
on a QThread and listens on the `Reporter` seam the agent already exposes.

Two rules hold this together, and breaking either one is silent rather than
loud:

1. `AgentLoop` is *constructed* on the worker thread, not handed to it.
   `__init__` detects hardware, which walks the CPU topology and stats the disk
   -- enough work on a cold machine to freeze the window if it ran on the GUI
   thread.
2. `request_stop` is a plain method called directly from the GUI thread, never
   a queued slot. `thread.started.connect(worker.run)` means the worker's event
   loop is blocked for the entire life of the agent, so a queued slot would sit
   in the queue until the thing it was meant to stop had already finished.
"""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from dataclasses import dataclass, field, replace

import psutil
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot

from node.agent import resources
from node.agent.config import AgentConfig
from node.agent.models import AgentView, TaskProgress
from node.agent.reporting import ConsoleReporter
from node.agent.resources import Budget
from node.gui.uistate import UiPhase

LOG = logging.getLogger("node.gui")

# How long to wait for the worker thread to unwind after the agent has returned.
# It has nothing left to do by then; this is only to catch a wedged thread.
_THREAD_JOIN_MS = 5000

# How often the GUI asks the agent what it looks like. A lock and a dataclass
# copy, so the cost is nil; a second is short enough that enrolling feels
# immediate and long enough not to be busywork.
_VIEW_POLL_MS = 1000


@dataclass(frozen=True)
class BudgetInfo:
    """What the agent actually got, once the request met the machine."""

    budget: Budget
    warnings: list[str] = field(default_factory=list)


def snapshot_view(view: AgentView) -> AgentView:
    """Copy the view before it crosses a thread.

    The loop mutates its `AgentView` in place (`self.view.status = ...`) and
    `reporter.state(self.view)` hands out the live object. Emit that and the
    window renders whatever the loop did next, not what it was told about.
    This is the only place the copy happens; nothing else holds `loop.view`.
    """
    current = view.current
    return replace(
        view,
        current=replace(current, detail=dict(current.detail)) if current else None,
    )


class QtReporter:
    """Implements `node.agent.reporting.Reporter`. Lives on the worker thread.

    `state` and `task` arrive from the agent thread; `note` arrives from the
    agent thread *and* from the memory watchdog's. All three are safe: the
    first two emit immutable snapshots over a queued connection, and the third
    goes through the stdlib logger, whose handler does not touch a widget.
    """

    def __init__(self, worker: AgentWorker) -> None:
        self._worker = worker
        # Notes keep flowing into the `node` logger, which is what feeds both
        # the log pane and the rotating file. One path, no duplication.
        self._console = ConsoleReporter()

    def state(self, view: AgentView) -> None:
        self._worker.state_changed.emit(snapshot_view(view))

    def task(self, progress: TaskProgress) -> None:
        self._worker.task_changed.emit(replace(progress, detail=dict(progress.detail)))

    def note(self, level: int, message: str, **fields: object) -> None:
        self._console.note(level, message, **fields)


class AgentWorker(QObject):
    """Owns one `AgentLoop`, from construction to exit, on the worker thread."""

    started_ok = pyqtSignal(object)  # BudgetInfo
    state_changed = pyqtSignal(object)  # AgentView snapshot
    task_changed = pyqtSignal(object)  # TaskProgress snapshot
    start_failed = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, config: AgentConfig) -> None:
        super().__init__()
        self._config = config
        self._loop = None
        self._stop_requested = False
        self._lock = threading.Lock()

    @pyqtSlot()
    def run(self) -> None:
        """Everything the agent does, on the worker thread. Never raises."""
        from node.agent.loop import AgentLoop

        code = 1
        try:
            hw = resources.detect_hardware(self._config.state_dir.parent)
            budget, warnings = resources.resolve_budget(
                self._config.cores, self._config.ram_gb, hw
            )
            # The reporter is attached before construction so the budget
            # warnings `AgentLoop.__init__` emits reach the log pane.
            loop = AgentLoop(self._config, reporter=QtReporter(self))
            with self._lock:
                self._loop = loop
                if self._stop_requested:
                    # Cancel arrived while we were building. Honour it now, or
                    # it is lost and the agent runs unstoppably.
                    loop.request_stop()
            self.started_ok.emit(BudgetInfo(budget=budget, warnings=warnings))
            code = loop.run()
        except Exception as error:  # noqa: BLE001 - a bad path, a full disk, a dead dataset
            LOG.exception("the agent worker died")
            self.start_failed.emit(f"{type(error).__name__}: {error}")
        finally:
            with self._lock:
                self._loop = None
            self.finished.emit(code)

    def request_stop(self) -> None:
        """Called directly from the GUI thread. Not a slot -- see the module docstring.

        `AgentLoop.request_stop` only sets two `threading.Event`s, which is safe
        across threads; the lock is here for the `self._loop` reference alone.
        """
        with self._lock:
            self._stop_requested = True
            loop = self._loop
        if loop is not None:
            loop.request_stop()

    def current_view(self) -> AgentView | None:
        """A snapshot of the agent's view, for the GUI to poll. Also not a slot.

        The agent *mutates* its view at enrolment, on every heartbeat and as
        each round starts and ends, but it only *reports* it after a round and
        at shutdown (loop.py:372 and :230). A node that enrols and then waits
        for work would therefore tell the window nothing at all -- no node_id,
        no status -- until its first round landed. Polling closes that gap
        without touching the agent.

        Reading fields another thread may be assigning is safe under the GIL:
        no value can be torn. The snapshot may mix a status from just before a
        change with a counter from just after, which is what a sample is.
        """
        with self._lock:
            loop = self._loop
        return snapshot_view(loop.view) if loop is not None else None


class AgentController(QObject):
    """The window's whole view of the agent: start it, stop it, hear about it."""

    phase_changed = pyqtSignal(object)  # UiPhase
    state = pyqtSignal(object)  # AgentView snapshot
    task = pyqtSignal(object)  # TaskProgress snapshot
    budget_ready = pyqtSignal(object)  # BudgetInfo
    failed = pyqtSignal(str)
    stopped = pyqtSignal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: AgentWorker | None = None
        self._phase = UiPhase.IDLE
        # Set by force_stop so the window can say "you abandoned this round"
        # instead of showing the misleading error the agent will record.
        self.forced = False

        # The agent reports its view only after a round; between rounds we ask.
        self._last_view: AgentView | None = None
        self._view_poll = QTimer(self)
        self._view_poll.setInterval(_VIEW_POLL_MS)
        self._view_poll.timeout.connect(self._poll_view)

    # -- state ------------------------------------------------------------

    @property
    def phase(self) -> UiPhase:
        return self._phase

    def is_running(self) -> bool:
        return self._phase is not UiPhase.IDLE

    def _set_phase(self, phase: UiPhase) -> None:
        if phase is not self._phase:
            self._phase = phase
            self.phase_changed.emit(phase)

    # -- lifecycle --------------------------------------------------------

    def start(self, config: AgentConfig) -> None:
        if self.is_running():
            return

        self.forced = False
        thread = QThread()
        thread.setObjectName("agent")
        worker = AgentWorker(config)
        worker.moveToThread(thread)

        # Explicitly queued. Auto-connection would already resolve to queued,
        # but saying so documents the invariant these slots depend on.
        worker.started_ok.connect(self._on_started, Qt.QueuedConnection)
        worker.state_changed.connect(self._on_state, Qt.QueuedConnection)
        worker.task_changed.connect(self.task, Qt.QueuedConnection)
        worker.start_failed.connect(self._on_failed, Qt.QueuedConnection)
        worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        thread.started.connect(worker.run)

        self._thread = thread
        self._worker = worker
        self._last_view = None
        self._set_phase(UiPhase.STARTING)
        thread.start()
        self._view_poll.start()

    def request_stop(self) -> None:
        """Graceful. May take minutes: the loop checks its flag between rounds,
        not inside a training subprocess."""
        if self._worker is None:
            return
        self._set_phase(UiPhase.STOPPING)
        self._worker.request_stop()

    def force_stop(self) -> None:
        """Ask nicely, then kill the training child.

        The child dies without writing its result file, `_run_isolated` turns
        that into a RuntimeError, and the runner marks the round failed but
        *retryable* -- nothing is lost, and the next run picks it up.
        """
        self.forced = True
        self.request_stop()
        self.kill_children()

    def kill_children(self) -> int:
        """Kill the training process tree. Returns how many were killed.

        Also called unconditionally on quit: a fastText worker orphaned by an
        exiting GUI keeps burning every budgeted core, and it is in no job
        object on Windows and no killable process group on Linux.
        """
        try:
            # Snapshot before signalling anything: killing a parent reparents
            # the rest, and then we can no longer find them.
            children = psutil.Process().children(recursive=True)
        except (psutil.Error, OSError):
            return 0
        if not children:
            return 0

        LOG.warning("killing %d child process(es)", len(children))
        for child in children:
            with suppress(psutil.Error, OSError):
                child.terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            with suppress(psutil.Error, OSError):
                child.kill()
        return len(children)

    def wait(self, msec: int) -> bool:
        """Only for the quit path; never call this from a handler that must return."""
        thread = self._thread
        return True if thread is None else thread.wait(msec)

    # -- slots ------------------------------------------------------------

    def _on_started(self, info: BudgetInfo) -> None:
        self._set_phase(UiPhase.RUNNING)
        self.budget_ready.emit(info)

    def _on_failed(self, message: str) -> None:
        self.failed.emit(message)

    def _on_state(self, view: AgentView) -> None:
        """A push from the agent. Recorded so the poll does not repeat it."""
        self._last_view = view
        self.state.emit(view)

    def _poll_view(self) -> None:
        worker = self._worker
        if worker is None:
            return
        view = worker.current_view()
        if view is not None and view != self._last_view:
            self._last_view = view
            self.state.emit(view)

    def _on_finished(self, code: int) -> None:
        self._view_poll.stop()
        self._last_view = None
        thread, self._thread = self._thread, None
        worker, self._worker = self._worker, None

        if thread is not None:
            thread.quit()
            if not thread.wait(_THREAD_JOIN_MS):
                # Never terminate() a thread sitting in C code -- that corrupts
                # the interpreter. Leak it; the process is about to move on.
                LOG.warning("the agent thread did not finish in time")
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()

        self._set_phase(UiPhase.IDLE)
        self.stopped.emit(code)
