"""The window, driven end to end against a fake controller.

No agent, no thread, no socket: this asserts the wiring, which is the part that
cannot be checked by reading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt5", reason="GUI extra not installed")

from node.agent.models import AgentView, TaskProgress  # noqa: E402
from node.agent.resources import HardwareInfo  # noqa: E402
from node.gui.presets import LARGE, MEDIUM, SMALL, preset_budget  # noqa: E402
from node.gui.settings import GuiSettings  # noqa: E402
from node.gui.uistate import UiPhase  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qapp, hw: HardwareInfo, controller, tmp_path: Path):
    from node.gui.mainwindow import MainWindow

    # tray=None: QSystemTrayIcon under the offscreen platform is not worth
    # exercising, and the no-tray path is the one with real logic in it.
    made = MainWindow(GuiSettings(), hw, controller, state_dir=tmp_path, tray=None)
    # Shown, because isVisible() on a child is False for as long as its window
    # has never been shown -- which would make every visibility assertion pass
    # for the wrong reason.
    made.show()
    yield made
    # Without this, closing a window whose node is still running asks the
    # operator to confirm -- a modal dialog with nobody to click it.
    made.begin_quit()
    made.close()
    made.deleteLater()


def _view(**overrides) -> AgentView:
    base = {
        "server_url": "https://example.test",
        "status": "busy",
        "node_id": "node-test",
        "budget_cores": 2,
        "budget_ram_gb": 4.0,
    }
    return AgentView(**{**base, **overrides})


def _task(round_id: str, phase: str, **detail) -> TaskProgress:
    return TaskProgress(round_id=round_id, phase=phase, started_at=0.0, detail=detail)


def _running(window, controller):
    window._primary.click()
    controller.finish_starting()


# -- opening ---------------------------------------------------------------


def test_opens_idle_and_ready_to_start(window) -> None:
    assert window._phase is UiPhase.IDLE
    assert window._primary.text() == "Start"
    assert window._primary.isEnabled()
    assert window._pages.currentIndex() == 0
    assert window._empty.text() == "The node is not running"


def test_there_is_no_log_pane(window) -> None:
    """It was removed on purpose; the log goes to a file now."""
    assert not hasattr(window, "_log")


def test_the_presets_describe_this_machine(window, hw) -> None:
    for level in (SMALL, MEDIUM, LARGE):
        assert preset_budget(level, hw).summary() in window._preset_buttons[level].toolTip()


# -- starting --------------------------------------------------------------


def test_start_hands_the_selected_preset_to_the_controller(window, controller, hw) -> None:
    window._preset_buttons[SMALL].setChecked(True)
    window._primary.click()

    assert len(controller.started_with) == 1
    config = controller.started_with[0]
    expected = preset_budget(SMALL, hw)
    assert (config.cores, config.ram_gb) == (expected.cores, expected.ram_gb)


def test_advanced_values_win_over_the_preset(window, controller) -> None:
    window._settings.advanced = True
    window._settings.cores = 3
    window._settings.ram_gb = 1.5
    window._refresh_power()
    window._primary.click()

    config = controller.started_with[0]
    assert (config.cores, config.ram_gb) == (3, 1.5)


def test_an_invalid_budget_disables_start(window) -> None:
    window._settings.advanced = True
    window._settings.ram_gb = 0.2  # below MIN_RAM_GB
    window._refresh_power()

    assert not window._primary.isEnabled()
    assert "--ram" in window._primary.toolTip()


def test_the_button_follows_the_phase(window, controller) -> None:
    window._primary.click()
    assert window._primary.text() == "Cancel start"

    controller.finish_starting()
    assert window._primary.text() == "Stop"

    window._primary.click()
    assert controller.stop_calls == 1
    assert not window._primary.isEnabled()

    controller.finish(0)
    assert window._phase is UiPhase.IDLE
    assert window._primary.text() == "Start"
    assert window._primary.isEnabled()


def test_the_budget_cannot_be_changed_under_a_running_agent(window, controller) -> None:
    _running(window, controller)

    assert not window._advanced_button.isEnabled()
    assert not window._preset_buttons[MEDIUM].isEnabled()


def test_a_start_failure_is_surfaced(window, controller, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    seen = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *a, **k: seen.append(a) or QMessageBox.Ok
    )
    window._primary.click()
    controller.failed.emit("ApiError: control plane unreachable")

    assert seen


# -- the header ------------------------------------------------------------


def test_the_pill_follows_the_phase_before_the_agent_says_anything(
    window, controller
) -> None:
    """It used to be painted only on the way into IDLE, so it said "stopped"
    for the whole run."""
    assert window._pill.text() == "stopped"

    window._primary.click()
    assert window._pill.text() == "starting"

    controller.finish_starting()
    assert window._pill.text() == "starting"  # running, nothing reported yet

    controller.state.emit(_view(status="idle"))
    assert window._pill.text() == "waiting for tasks"

    controller.state.emit(_view(status="busy"))
    assert window._pill.text() == "training"


def test_the_pill_shows_stopping_even_while_the_round_finishes(window, controller) -> None:
    _running(window, controller)
    controller.state.emit(_view(status="busy"))
    window._primary.click()

    # The agent keeps reporting "busy" until the round it is on is done.
    controller.state.emit(_view(status="busy"))

    assert window._pill.text() == "stopping"


def test_the_node_id_appears_as_soon_as_a_view_arrives(window, controller) -> None:
    _running(window, controller)
    assert "not enrolled" in window._identity.text()

    controller.state.emit(_view(status="idle", last_heartbeat_ok=True))

    assert "node-test" in window._identity.text()
    assert "connection fine" in window._identity.text()


# -- stat tiles ------------------------------------------------------------


def test_the_tiles_count_rounds(window, controller) -> None:
    _running(window, controller)

    controller.state.emit(_view(completed=7, failed=1))

    assert window._stat_done._value.text() == "7"
    assert window._stat_failed._value.text() == "1"
    assert window._stat_failed._value.property("tone") == "crit"


def test_the_failed_tile_is_not_red_at_zero(window, controller) -> None:
    _running(window, controller)
    controller.state.emit(_view(completed=3, failed=0))

    assert not window._stat_failed._value.property("tone")


def test_the_budget_tiles_show_what_the_agent_actually_got(window, controller) -> None:
    window._primary.click()
    controller.finish_starting(cores=3, ram_gb=2.5)

    assert window._stat_cores._value.text() == "3"
    assert window._stat_ram._value.text() == "2.5"


# -- round progress --------------------------------------------------------


def test_a_round_gets_a_card_that_walks_its_phases(window, controller) -> None:
    _running(window, controller)

    controller.task.emit(_task("r-42", "acked"))
    assert window._pages.currentIndex() == 1
    card = window._cards["r-42"]
    assert card._pill.text() == "accepted"

    controller.task.emit(_task("r-42", "running", mode="train"))
    assert card._pill.text() == "training"
    assert "train" in card._detail.text()

    controller.task.emit(_task("r-42", "done", n_scores=1480, seconds=52.0))
    assert card._pill.text() == "submitted"
    assert "1480" in card._detail.text()
    assert card._elapsed.text() == "0:52"


def test_one_card_per_round_newest_on_top(window, controller) -> None:
    _running(window, controller)
    controller.task.emit(_task("r-1", "done", seconds=1.0))
    controller.task.emit(_task("r-2", "running"))

    assert set(window._cards) == {"r-1", "r-2"}
    assert window._round_layout.itemAt(0).widget().round_id == "r-2"
    assert window._rounds_count.text() == "this run: 2"


def test_a_failed_round_is_inferred_and_shown(window, controller) -> None:
    """The agent never reports a failure as a task event; only the counter moves."""
    _running(window, controller)
    controller.task.emit(_task("r-9", "running"))

    controller.state.emit(_view(failed=1, last_error="training worker exited with -9"))

    card = window._cards["r-9"]
    assert card._pill.text() == "error"
    assert "training worker" in card._detail.text()


def test_rounds_are_cleared_between_runs(window, controller) -> None:
    _running(window, controller)
    controller.task.emit(_task("r-1", "done", seconds=1.0))
    controller.finish(0)

    window._primary.click()

    assert window._cards == {}
    assert window._pages.currentIndex() == 0


def test_the_empty_text_explains_where_the_node_is(window, controller) -> None:
    window._primary.click()
    assert window._empty.text() == "Connecting to the server"

    controller.finish_starting()
    controller.state.emit(_view(status="idle"))
    assert window._empty.text() == "Waiting for tasks from the server"


# -- stopping --------------------------------------------------------------


def test_the_force_button_is_hidden_until_a_stop_drags_on(window, controller) -> None:
    _running(window, controller)
    window._primary.click()

    assert not window._force.isVisible()

    window._maybe_offer_force()  # what the 10 s timer calls
    assert window._force.isVisible()


def test_the_force_button_stays_hidden_once_the_stop_lands(window, controller) -> None:
    _running(window, controller)
    window._primary.click()
    controller.finish(0)

    window._maybe_offer_force()

    assert not window._force.isVisible()


def test_the_last_run_is_remembered(window, controller, tmp_path: Path) -> None:
    from node.gui import settings as settings_mod

    _running(window, controller)
    controller.state.emit(_view(completed=5, failed=2))
    controller.finish(0)

    saved = settings_mod.load(settings_mod.settings_path(tmp_path))
    assert saved.last_run_completed == 5
    assert saved.last_run_failed == 2
    assert "5 rounds" in window._empty.text()
