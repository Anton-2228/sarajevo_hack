"""Exact cores and RAM, for the operator who already knows what they want.

Everything else about the node -- the server, the directories, the transport --
is either known up front or has a default worth keeping, so it is not here.
"""

from __future__ import annotations

from dataclasses import replace

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from node.agent.config import MIN_RAM_GB
from node.agent.resources import HardwareInfo
from node.gui import settings as settings_mod
from node.gui.presets import describe_machine
from node.gui.settings import GuiSettings

AUTO = "auto"


class AdvancedDialog(QDialog):
    """Edits `cores` and `ram_gb` on a copy; the caller takes `result_settings`."""

    def __init__(
        self,
        settings: GuiSettings,
        hw: HardwareInfo,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced settings")
        self.setModal(True)
        self._hw = hw
        self._settings = replace(settings)
        # Set while the dialog moves the spin boxes itself, so that restoring
        # the defaults does not read as the operator typing a value.
        self._restoring = False

        self._exact = QCheckBox("Set exact values instead of a preset")
        self._exact.setChecked(settings.advanced)

        self._cores = QSpinBox()
        # Oversubscription is allowed on purpose -- the agent warns rather than
        # refusing, and someone benchmarking may want it.
        self._cores.setRange(0, max(2, hw.logical_cores * 2))
        self._cores.setSpecialValueText(AUTO)
        self._cores.setValue(settings.cores)
        self._cores.setToolTip(
            f"0 — let the agent decide. This machine has {hw.logical_cores}"
        )

        self._ram = QDoubleSpinBox()
        self._ram.setRange(0.0, max(1.0, round(hw.total_ram_gb, 1)))
        self._ram.setSingleStep(0.5)
        self._ram.setDecimals(1)
        self._ram.setSpecialValueText(AUTO)
        self._ram.setSuffix(" GiB")
        self._ram.setValue(settings.ram_gb)
        self._ram.setToolTip(
            f"0 — let the agent decide. Training needs at least {MIN_RAM_GB} GiB. "
            f"This machine has {hw.total_ram_gb:.1f} GiB"
        )

        self._error = QLabel()
        self._error.setObjectName("error")
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #c0392b;")
        self._error.hide()

        hint = QLabel(
            "With this off the preset chosen in the window decides, and these "
            "two are only a note of what to go back to."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")

        form = QFormLayout()
        form.addRow(self._exact)
        form.addRow("", hint)
        form.addRow("Cores:", self._cores)
        form.addRow("Memory:", self._ram)

        machine = QLabel(f"This machine: {describe_machine(hw)}")
        machine.setStyleSheet("color: palette(mid);")

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(machine)
        layout.addWidget(self._error)
        layout.addStretch(1)
        layout.addWidget(self._buttons)

        self._exact.toggled.connect(self._revalidate)
        self._cores.valueChanged.connect(self._on_value_edited)
        self._ram.valueChanged.connect(self._on_value_edited)
        self._revalidate()

    def result_settings(self) -> GuiSettings:
        """The edited settings. Only meaningful after `exec_()` returned Accepted."""
        return replace(
            self._settings,
            advanced=self._exact.isChecked(),
            cores=self._cores.value(),
            ram_gb=round(self._ram.value(), 1),
        )

    def _on_value_edited(self) -> None:
        """Typing a number is how someone asks for exact values.

        The spin boxes are live whether or not the box above them is ticked:
        two greyed-out fields that wake up only for a checkbox someone has to
        find first are indistinguishable from two broken ones.
        """
        if not self._restoring:
            self._exact.setChecked(True)
        self._revalidate()

    def _restore_defaults(self) -> None:
        self._restoring = True
        try:
            self._cores.setValue(0)
            self._ram.setValue(0.0)
            self._exact.setChecked(False)
        finally:
            self._restoring = False
        self._revalidate()

    def _revalidate(self) -> None:
        # A static suffix would leave the box reading "1 cores". At 0 the special
        # value text replaces the suffix too, so only 1 is worth a case.
        self._cores.setSuffix(" core" if self._cores.value() == 1 else " cores")

        # `AgentConfig.validate()` reports every problem at once and touches no
        # disk, so it is safe to run on every keystroke.
        problems = settings_mod.build_config(self.result_settings(), self._hw).validate()
        if problems:
            self._error.setText("\n".join(problems))
            self._error.show()
        else:
            self._error.hide()
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(not problems)
