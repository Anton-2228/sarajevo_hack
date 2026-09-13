"""One QApplication, offscreen, and a controller that never starts an agent."""

from __future__ import annotations

import os

# Must be set before Qt is imported, or the platform plugin is already chosen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false")

import pytest  # noqa: E402

from node.agent.config import AgentConfig  # noqa: E402
from node.agent.resources import Budget, HardwareInfo  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """A second QApplication aborts the process, so exactly one, for the session."""
    pytest.importorskip("PyQt5", reason="GUI extra not installed")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def hw() -> HardwareInfo:
    return HardwareInfo(
        logical_cores=8, total_ram_gb=16.0, disk_free_gb=100.0, python="3.12"
    )


@pytest.fixture
def controller(qapp):
    """Stands in for AgentController: same signals, same methods, no thread.

    The window must be drivable without an AgentLoop -- which would detect
    hardware, synthesize a dataset and open sockets -- so every test here talks
    to this instead.
    """
    from PyQt5.QtCore import QObject, pyqtSignal

    from node.gui.uistate import UiPhase
    from node.gui.worker import BudgetInfo

    class FakeController(QObject):
        phase_changed = pyqtSignal(object)
        state = pyqtSignal(object)
        task = pyqtSignal(object)
        budget_ready = pyqtSignal(object)
        failed = pyqtSignal(str)
        stopped = pyqtSignal(int)

        def __init__(self) -> None:
            super().__init__()
            self._phase = UiPhase.IDLE
            self.forced = False
            self.started_with: list[AgentConfig] = []
            self.stop_calls = 0
            self.force_calls = 0
            self.kill_calls = 0

        @property
        def phase(self) -> UiPhase:
            return self._phase

        def is_running(self) -> bool:
            return self._phase is not UiPhase.IDLE

        def set_phase(self, phase: UiPhase) -> None:
            self._phase = phase
            self.phase_changed.emit(phase)

        def start(self, config: AgentConfig) -> None:
            self.started_with.append(config)
            self.set_phase(UiPhase.STARTING)

        def finish_starting(self, cores: int = 2, ram_gb: float = 4.0) -> None:
            self.set_phase(UiPhase.RUNNING)
            self.budget_ready.emit(
                BudgetInfo(budget=Budget(cores=cores, ram_bytes=int(ram_gb * 1024**3)))
            )

        def request_stop(self) -> None:
            self.stop_calls += 1
            self.set_phase(UiPhase.STOPPING)

        def force_stop(self) -> None:
            self.force_calls += 1
            self.forced = True
            self.request_stop()

        def kill_children(self) -> int:
            self.kill_calls += 1
            return 0

        def wait(self, msec: int) -> bool:
            return True

        def finish(self, code: int = 0) -> None:
            self.set_phase(UiPhase.IDLE)
            self.stopped.emit(code)

    return FakeController()
