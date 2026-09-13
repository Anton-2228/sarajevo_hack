"""The advanced dialog, and the one thing about the theme it depends on.

Both tests here guard the same complaint -- "the cores and RAM fields do not
work" -- from its two sides: fields that looked dead because they were disabled
until a checkbox above them was found, and arrows that were invisible because a
styled QSpinBox draws none of its own.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PyQt5", reason="GUI extra not installed")

from node.agent.resources import HardwareInfo  # noqa: E402
from node.gui.advanced import AdvancedDialog  # noqa: E402
from node.gui.settings import GuiSettings  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def dialog(qapp, hw: HardwareInfo):
    made = AdvancedDialog(GuiSettings(), hw)
    made.show()
    yield made
    made.close()
    made.deleteLater()


def test_the_fields_are_live_before_the_checkbox_is_ticked(dialog) -> None:
    assert dialog._cores.isEnabled()
    assert dialog._ram.isEnabled()


def test_typing_a_value_is_itself_the_request_for_exact_values(dialog) -> None:
    assert not dialog._exact.isChecked()

    dialog._cores.setValue(3)

    assert dialog._exact.isChecked()
    assert dialog.result_settings().cores == 3


def test_restoring_defaults_goes_back_to_the_preset(dialog) -> None:
    dialog._ram.setValue(4.0)
    assert dialog._exact.isChecked()

    dialog._restore_defaults()

    settings = dialog.result_settings()
    assert (settings.advanced, settings.cores, settings.ram_gb) == (False, 0, 0.0)


def test_a_budget_too_small_to_train_blocks_ok(dialog) -> None:
    from PyQt5.QtWidgets import QDialogButtonBox

    dialog._ram.setValue(0.2)  # below MIN_RAM_GB

    assert dialog._error.isVisible()
    assert "--ram" in dialog._error.text()
    assert not dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_the_spin_boxes_get_arrows_to_click(qapp) -> None:
    """Styling a QSpinBox costs it the native arrows; the sheet must supply them.

    Without this the dialog still works and looks broken, which is worse than
    either.
    """
    from node.gui import theme

    sheet = theme.stylesheet()
    urls = re.findall(r'image: url\("([^"]+)"\)', sheet)

    assert len(urls) == 4  # up, down, and the two at the end of the range
    for url in urls:
        assert Path(url).is_file()
