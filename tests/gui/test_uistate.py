"""The window's four states, without a window."""

from __future__ import annotations

import pytest

from node.gui.uistate import (
    UiPhase,
    button_spec,
    controls_editable,
    shows_busy_bar,
    shows_live_panel,
    status_label,
    status_variant,
)


def test_idle_offers_start() -> None:
    spec = button_spec(UiPhase.IDLE)

    assert spec.label == "Start"
    assert spec.enabled


def test_idle_with_a_problem_disables_start_and_explains_why() -> None:
    spec = button_spec(UiPhase.IDLE, valid=False, first_problem="--ram must be at least 0.5")

    assert not spec.enabled
    assert "--ram" in spec.tooltip


def test_running_offers_stop() -> None:
    assert button_spec(UiPhase.RUNNING).label == "Stop"
    assert button_spec(UiPhase.RUNNING).enabled


def test_stopping_is_not_clickable_again() -> None:
    """Two stop requests are not twice as fast, and the second reads as a hang."""
    spec = button_spec(UiPhase.STOPPING)

    assert not spec.enabled


def test_starting_can_be_cancelled_but_warns_about_the_wait() -> None:
    spec = button_spec(UiPhase.STARTING)

    assert spec.enabled
    assert spec.tooltip


@pytest.mark.parametrize("phase", list(UiPhase))
def test_only_idle_lets_the_budget_change(phase: UiPhase) -> None:
    assert controls_editable(phase) is (phase is UiPhase.IDLE)


def test_the_busy_bar_shows_only_during_transitions() -> None:
    assert shows_busy_bar(UiPhase.STARTING)
    assert shows_busy_bar(UiPhase.STOPPING)
    assert not shows_busy_bar(UiPhase.IDLE)
    assert not shows_busy_bar(UiPhase.RUNNING)


def test_the_live_panel_shows_once_there_is_an_agent() -> None:
    assert shows_live_panel(UiPhase.RUNNING)
    assert shows_live_panel(UiPhase.STOPPING)
    assert not shows_live_panel(UiPhase.IDLE)


@pytest.mark.parametrize(
    "status", ["starting", "idle", "busy", "degraded", "stopping", "offline"]
)
def test_every_agent_status_has_a_variant_and_a_label(status: str) -> None:
    """`AgentView.status` comes from the loop; an unhandled one must not be blank."""
    assert status_variant(status) in ("good", "warn", "crit", "off")
    assert status_label(status) != ""


def test_an_unexpected_status_still_renders() -> None:
    assert status_variant("something-new") == "off"
    assert status_label("something-new") == "something-new"
