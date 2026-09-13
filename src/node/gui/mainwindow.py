"""The window: pick a share of the machine, start the node, watch the rounds.

Laid out after the control room dashboard -- the same tokens, the same stat
tiles, pills and step bars -- so an operator moving between the panel and the
node is looking at one product rather than two.

The controller is injected rather than constructed here, so the window can be
driven by a fake in tests: no thread, no network, no dataset synthesis.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from node.agent.config import AgentConfig
from node.agent.models import AgentView, TaskProgress
from node.agent.resources import HardwareInfo
from node.gui import settings as settings_mod
from node.gui.advanced import AdvancedDialog
from node.gui.loadmeter import LoadMeter
from node.gui.presets import (
    DESCRIPTIONS,
    LABELS,
    PRESET_LEVELS,
    describe_machine,
    preset_budget,
)
from node.gui.rounds import RoundTracker
from node.gui.settings import GuiSettings
from node.gui.text import cores as cores_text
from node.gui.text import rounds as rounds_text
from node.gui.uistate import (
    FORCE_AFTER_S,
    UiPhase,
    button_spec,
    controls_editable,
    shows_busy_bar,
)
from node.gui.widgets import Pill, RoundCard, StatTile, card, label, repolish, rule

LOG = logging.getLogger("node.gui")

_TICK_MS = 1000
_LOAD_MS = 2000


class MainWindow(QMainWindow):
    """Everything the operator sees. Owns the phase; the rest is derived."""

    quit_requested = pyqtSignal()

    def __init__(
        self,
        settings: GuiSettings,
        hw: HardwareInfo,
        controller,
        state_dir: Path,
        tray=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Node")
        self.setMinimumSize(520, 620)
        self.resize(560, 760)

        self._settings = settings
        self._hw = hw
        self._controller = controller
        self._state_dir = state_dir
        self._tray = tray

        self._phase = UiPhase.IDLE
        self._view: AgentView | None = None
        self._budget = None
        self._meter = LoadMeter()
        self._tracker = RoundTracker()
        self._cards: dict[str, RoundCard] = {}
        self._quitting = False

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)
        self._load_timer = QTimer(self)
        self._load_timer.timeout.connect(self._tick_load)

        self._build()
        self._wire()

        self._apply_phase(UiPhase.IDLE)
        self._refresh_power()

    # -- construction -----------------------------------------------------

    def _build(self) -> None:
        page = QWidget()
        page.setObjectName("page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 18, 20, 14)
        layout.setSpacing(16)

        layout.addWidget(self._build_header())
        layout.addWidget(rule())
        layout.addLayout(self._build_stats())
        layout.addWidget(self._build_power())
        layout.addLayout(self._build_controls())
        layout.addWidget(self._build_rounds(), 1)
        layout.addWidget(self._build_meters())

        self.setCentralWidget(page)
        self.statusBar().showMessage(str(self._state_dir))

    def _build_header(self) -> QWidget:
        holder = QWidget()
        outer = QVBoxLayout(holder)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)

        self._pill = Pill("stopped", "off")

        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(label("NODE AGENT", "eyebrow"))
        top.addStretch(1)
        top.addWidget(self._pill)

        self._title = label("Node", "title")

        self._identity = label("not enrolled", "subtle")
        self._identity.setTextInteractionFlags(Qt.TextSelectableByMouse)
        server = settings_mod.server_url()
        self._identity.setToolTip(server)

        outer.addLayout(top)
        outer.addWidget(self._title)
        outer.addWidget(self._identity)
        return holder

    def _build_stats(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self._stat_done = StatTile("rounds", "0")
        self._stat_failed = StatTile("failed", "0")
        self._stat_cores = StatTile("cores", "—")
        self._stat_ram = StatTile("RAM, GiB", "—")
        for tile in (self._stat_done, self._stat_failed, self._stat_cores, self._stat_ram):
            row.addWidget(tile)
        return row

    def _build_power(self) -> QWidget:
        box = card()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        head = QHBoxLayout()
        head.addWidget(label("Power", "h2"))
        head.addStretch(1)
        head.addLayout(self._build_segments())
        layout.addLayout(head)

        self._effective = label("", "subtle")
        self._effective.setWordWrap(True)

        self._advanced_button = QPushButton("Configure…")
        self._advanced_button.setAutoDefault(False)

        foot = QHBoxLayout()
        foot.setSpacing(10)
        foot.addWidget(self._effective, 1)
        foot.addWidget(self._advanced_button)
        layout.addLayout(foot)
        return box

    def _build_segments(self) -> QHBoxLayout:
        """Three joined buttons rather than radios -- the panel has no radios."""
        row = QHBoxLayout()
        row.setSpacing(0)

        self._preset_group = QButtonGroup(self)
        self._preset_group.setExclusive(True)
        self._preset_buttons: dict[str, QPushButton] = {}

        for index, level in enumerate(PRESET_LEVELS):
            button = QPushButton(LABELS[level])
            button.setObjectName("segment")
            button.setCheckable(True)
            button.setAutoDefault(False)
            button.setCursor(Qt.PointingHandCursor)
            if index == 0:
                button.setProperty("edge", "left")
            elif index == len(PRESET_LEVELS) - 1:
                button.setProperty("edge", "right")
            button.setChecked(level == self._settings.preset)
            self._preset_group.addButton(button)
            self._preset_buttons[level] = button
            row.addWidget(button)
        return row

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self._primary = QPushButton("Start")
        self._primary.setObjectName("primary")
        self._primary.setDefault(True)
        self._primary.setCursor(Qt.PointingHandCursor)

        self._force = QPushButton("Abort round")
        self._force.setObjectName("danger")
        self._force.setAutoDefault(False)
        self._force.hide()

        row.addWidget(self._primary, 1)
        row.addWidget(self._force)
        return row

    def _build_rounds(self) -> QWidget:
        holder = QWidget()
        outer = QVBoxLayout(holder)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        head = QHBoxLayout()
        head.addWidget(label("Rounds", "h2"))
        head.addStretch(1)
        self._rounds_count = label("", "countTag")
        head.addWidget(self._rounds_count)
        outer.addLayout(head)

        self._empty = label("", "empty")
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # A page of its own with a stretch under it, so the dashed placeholder
        # stays the compact bar the dashboard uses instead of growing to fill
        # everything the rounds will eventually occupy.
        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.addWidget(self._empty)
        empty_layout.addStretch(1)

        self._round_list = QWidget()
        self._round_layout = QVBoxLayout(self._round_list)
        self._round_layout.setContentsMargins(0, 0, 0, 0)
        self._round_layout.setSpacing(8)
        self._round_layout.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._round_list)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._pages = QStackedWidget()
        self._pages.addWidget(empty_page)
        self._pages.addWidget(self._scroll)
        outer.addWidget(self._pages, 1)
        return holder

    def _build_meters(self) -> QWidget:
        box = card()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(8)

        self._cpu = QProgressBar()
        self._cpu.setRange(0, 100)
        self._cpu.setTextVisible(False)
        self._ram = QProgressBar()
        self._ram.setRange(0, 100)
        self._ram.setTextVisible(False)
        self._ram.setToolTip("Share of the RAM budget allotted to the node")

        bars = QHBoxLayout()
        bars.setSpacing(12)
        bars.addWidget(label("CPU", "statLabel"))
        bars.addWidget(self._cpu, 1)
        bars.addWidget(label("RAM", "statLabel"))
        bars.addWidget(self._ram, 1)

        self._meter_text = label("—", "subtle")
        layout.addLayout(bars)
        layout.addWidget(self._meter_text)
        return box

    def _wire(self) -> None:
        self._primary.clicked.connect(self._on_primary)
        self._force.clicked.connect(self._on_force)
        self._advanced_button.clicked.connect(self._open_advanced)
        self._preset_group.buttonToggled.connect(self._on_preset_toggled)

        self._controller.phase_changed.connect(self._apply_phase)
        self._controller.state.connect(self.on_state)
        self._controller.task.connect(self.on_task)
        self._controller.budget_ready.connect(self.on_budget)
        self._controller.failed.connect(self.on_failed)
        self._controller.stopped.connect(self.on_stopped)

        if self._tray is not None:
            self._tray.show_requested.connect(self.show_window)
            self._tray.start_requested.connect(self.start_node)
            self._tray.stop_requested.connect(self._controller.request_stop)
            self._tray.quit_requested.connect(self.quit_requested)
            self._tray.availability_changed.connect(self._on_tray_appeared)

    # -- settings ---------------------------------------------------------

    def collect_settings(self) -> GuiSettings:
        return replace(
            self._settings, window_geometry_hex=bytes(self.saveGeometry()).hex()
        )

    def build_config(self, *, reset_identity: bool = False) -> AgentConfig:
        return settings_mod.build_config(
            self._settings,
            self._hw,
            state_dir=self._state_dir,
            reset_identity=reset_identity,
        )

    def restore_geometry(self) -> None:
        raw = self._settings.window_geometry_hex
        if not raw:
            return
        try:
            self.restoreGeometry(bytes.fromhex(raw))
        except (ValueError, TypeError):
            # A monitor that vanished, or a file someone edited. Not worth a word.
            pass

    # -- actions ----------------------------------------------------------

    def _on_preset_toggled(self, button: QPushButton, checked: bool) -> None:
        if not checked:
            return
        for level, candidate in self._preset_buttons.items():
            if candidate is button:
                self._settings.preset = level
                # Choosing a preset means wanting the preset, not the exact
                # numbers typed three sessions ago.
                self._settings.advanced = False
                break
        self._refresh_power()

    def _open_advanced(self) -> None:
        dialog = AdvancedDialog(self._settings, self._hw, self)
        if dialog.exec_() != AdvancedDialog.Accepted:
            return
        self._settings = dialog.result_settings()
        self._refresh_power()

    def _on_primary(self) -> None:
        if self._phase is UiPhase.IDLE:
            self.start_node()
        elif self._phase in (UiPhase.RUNNING, UiPhase.STARTING):
            self._controller.request_stop()

    def start_node(self) -> None:
        if self._phase is not UiPhase.IDLE:
            return
        cfg = self.build_config()
        problems = cfg.validate()
        if problems:
            QMessageBox.warning(self, "Cannot start", "\n".join(problems))
            return
        self._clear_rounds()
        self._controller.start(cfg)

    def _on_force(self) -> None:
        entry = self._tracker.current()
        round_id = entry.round_id if entry else "the current round"
        answer = QMessageBox.question(
            self,
            "Abort round",
            f"Abort training of round {round_id} and stop now?\n\n"
            "The round is not lost: it will be marked failed and retried later.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._force.setEnabled(False)
            self._controller.force_stop()

    def show_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    # -- agent events -----------------------------------------------------

    def _apply_phase(self, phase: UiPhase) -> None:
        self._phase = phase

        problems = self.build_config().validate()
        spec = button_spec(
            phase, valid=not problems, first_problem=problems[0] if problems else ""
        )
        self._primary.setText(spec.label)
        self._primary.setEnabled(spec.enabled)
        self._primary.setToolTip(spec.tooltip)
        tone = "stop" if phase in (UiPhase.RUNNING, UiPhase.STARTING) else None
        if self._primary.property("tone") != tone:
            self._primary.setProperty("tone", tone)
            repolish(self._primary)

        editable = controls_editable(phase)
        for button in self._preset_buttons.values():
            button.setEnabled(editable)
        self._advanced_button.setEnabled(editable)

        busy = shows_busy_bar(phase)

        if phase is UiPhase.STOPPING:
            self._force.setEnabled(True)
            QTimer.singleShot(int(FORCE_AFTER_S * 1000), self._maybe_offer_force)
        else:
            self._force.hide()

        if phase is UiPhase.IDLE:
            self._tick_timer.stop()
            self._load_timer.stop()
            self.statusBar().showMessage(str(self._state_dir))
        else:
            self._tick_timer.start(_TICK_MS)
            self._load_timer.start(_LOAD_MS)
            # Clear on the way into RUNNING, or "Connecting to the server…" sits
            # there for the rest of the session.
            self.statusBar().showMessage(
                "Connecting to the server…"
                if phase is UiPhase.STARTING
                else "Stopping the node…"
                if busy
                else str(self._state_dir)
            )

        self._repaint_pill()
        self._refresh_stats()
        self._refresh_rounds()
        self._update_tray()

    def _maybe_offer_force(self) -> None:
        """Only if the graceful stop really is stuck on a training subprocess."""
        if self._phase is UiPhase.STOPPING:
            self._force.show()
            entry = self._tracker.current()
            where = entry.round_id if entry else "the current round"
            self.statusBar().showMessage(
                f"Waiting for training of {where} to finish. This can take a few minutes."
            )

    def on_state(self, view: AgentView) -> None:
        self._view = view
        node_id = view.node_id or "not enrolled"
        heartbeat = "connection fine" if view.last_heartbeat_ok else "connection failing"
        self._identity.setText(f"{node_id} · {heartbeat}")

        entry = self._tracker.on_state(view)
        if entry is not None:
            self._sync_card(entry)

        self._repaint_pill()
        self._refresh_stats()
        self._refresh_rounds()
        self._update_tray()

    def on_task(self, progress: TaskProgress) -> None:
        self._sync_card(self._tracker.on_task(progress))
        self._refresh_rounds()

    def on_budget(self, info) -> None:
        self._budget = info.budget
        self._stat_cores.set_value(str(info.budget.cores))
        self._stat_ram.set_value(f"{info.budget.ram_gb:.1f}")
        for warning in info.warnings:
            LOG.warning(warning)

    def on_failed(self, message: str) -> None:
        self.statusBar().showMessage(message, 15_000)
        QMessageBox.critical(self, "The node did not start", message)

    def on_stopped(self, code: int) -> None:
        """Remember what this run managed, then let the window settle."""
        if self._view is not None:
            self._settings.last_run_completed = self._view.completed
            self._settings.last_run_failed = self._view.failed
            self._settings.last_run_finished_at = time.time()
            settings_mod.save(self.collect_settings(), self._settings_path())

        self._view = None
        self._budget = None
        self._stat_cores.set_value("—")
        self._stat_ram.set_value("—")
        self._refresh_rounds()

        if code != 0 and not self._quitting:
            self.statusBar().showMessage(f"The node exited with code {code}", 10_000)

    # -- rounds -----------------------------------------------------------

    def _clear_rounds(self) -> None:
        self._tracker.reset()
        for widget in self._cards.values():
            self._round_layout.removeWidget(widget)
            widget.deleteLater()
        self._cards.clear()
        self._refresh_rounds()

    def _sync_card(self, entry) -> None:
        widget = self._cards.get(entry.round_id)
        if widget is None:
            widget = RoundCard(entry)
            self._cards[entry.round_id] = widget
            # Newest on top; the stretch is always the last item.
            self._round_layout.insertWidget(0, widget)
            self._drop_evicted()
        else:
            widget.update_entry(entry)

    def _drop_evicted(self) -> None:
        """The tracker caps its history; the widgets must follow it."""
        alive = {e.round_id for e in self._tracker.entries()}
        for round_id in [r for r in self._cards if r not in alive]:
            widget = self._cards.pop(round_id)
            self._round_layout.removeWidget(widget)
            widget.deleteLater()

    def _refresh_rounds(self) -> None:
        entries = self._tracker.entries()
        self._rounds_count.setText(f"this run: {len(entries)}" if entries else "")

        has_rounds = bool(entries)
        self._pages.setCurrentIndex(1 if has_rounds else 0)
        if not has_rounds:
            self._empty.setText(self._empty_text())

    def _empty_text(self) -> str:
        if self._phase is UiPhase.IDLE:
            finished_at = self._settings.last_run_finished_at
            if finished_at:
                when = datetime.fromtimestamp(finished_at).strftime("%d %b, %H:%M")
                return (
                    f"Last run: {rounds_text(self._settings.last_run_completed)}"
                    f", {when}"
                )
            return "The node is not running"
        if self._view is not None and self._view.node_id:
            return "Waiting for tasks from the server"
        return "Connecting to the server"

    # -- timers -----------------------------------------------------------

    def _tick(self) -> None:
        now = time.time()
        for entry in self._tracker.entries():
            widget = self._cards.get(entry.round_id)
            if widget is not None and not entry.finished:
                widget.update_entry(entry, now)

    def _tick_load(self) -> None:
        load = self._meter.sample(self._budget)
        self._cpu.setValue(int(load.cpu_pct))
        self._ram.setValue(int(min(100.0, load.budget_ram_pct)))
        self._meter_text.setText(
            f"CPU {load.cpu_pct:.0f}%  ·  node processes {load.tree_rss_mb:.0f} MB  ·  "
            f"machine RAM used {load.ram_pct:.0f}%"
        )

    # -- rendering --------------------------------------------------------

    def _refresh_power(self) -> None:
        for level, button in self._preset_buttons.items():
            budget = preset_budget(level, self._hw)
            button.setToolTip(f"{DESCRIPTIONS[level]}\n{budget.summary()}")

        cores, ram_gb = settings_mod.resolved_budget(self._settings, self._hw)
        if self._settings.advanced:
            self._advanced_button.setText("Configured by hand")
            self._preset_group.setExclusive(False)
            for button in self._preset_buttons.values():
                button.setChecked(False)
            self._preset_group.setExclusive(True)
        else:
            self._advanced_button.setText("Configure…")
            self._preset_buttons[self._settings.preset].setChecked(True)

        shown_cores = cores_text(cores) if cores else "cores automatic"
        shown_ram = f"{ram_gb:g} GiB" if ram_gb else "RAM automatic"
        self._effective.setText(
            f"{shown_cores} · {shown_ram}   of   {describe_machine(self._hw)}"
        )
        if self._phase is UiPhase.IDLE:
            self._stat_cores.set_value(str(cores) if cores else "auto")
            self._stat_ram.set_value(f"{ram_gb:g}" if ram_gb else "auto")
            self._apply_phase(UiPhase.IDLE)

    def _refresh_stats(self) -> None:
        view = self._view
        completed = view.completed if view else self._settings.last_run_completed
        failed = view.failed if view else self._settings.last_run_failed
        self._stat_done.set_value(str(completed))
        self._stat_failed.set_value(str(failed), tone="crit" if failed else "")

    def _repaint_pill(self) -> None:
        """One rule for the pill, so it cannot be left showing a stale state.

        The phase wins while we are in transition -- during a stop the agent may
        still report "busy" for the round it is finishing, and a green "training"
        beside a greyed-out "Stopping…" button reads as a hang.
        """
        if self._phase is UiPhase.IDLE:
            self._pill.set_status("offline")
        elif self._phase is UiPhase.STOPPING:
            self._pill.set_status("stopping")
        elif self._view is None:
            # Running, but nothing reported yet: still connecting and enrolling.
            self._pill.set_status("starting")
        else:
            self._pill.set_status(self._view.status)

    def _update_tray(self) -> None:
        if self._tray is None:
            return
        view = self._view
        status = "offline" if self._phase is UiPhase.IDLE else "starting"
        if self._phase is UiPhase.STOPPING:
            status = "stopping"
        elif view is not None:
            status = view.status
        self._tray.update(
            self._phase,
            status,
            view.completed if view else 0,
            view.failed if view else 0,
        )

    # -- window -----------------------------------------------------------

    def _on_tray_appeared(self, available: bool) -> None:
        if available:
            self.statusBar().showMessage(
                "A system tray appeared — the window will now hide into it", 5000
            )

    def closeEvent(self, event) -> None:
        if self._quitting:
            event.accept()
            return

        # Without a tray, hiding would leave no way back to the window. Quitting
        # is the lesser evil; an unreachable running node is the worse one.
        if self._tray is None or not self._tray.available:
            if self._controller.is_running() and not self._confirm_quit():
                event.ignore()
                return
            event.accept()
            self.quit_requested.emit()
            return

        event.ignore()
        self.hide()
        if not self._settings.tray_hint_shown:
            self._settings.tray_hint_shown = True
            settings_mod.save(self.collect_settings(), self._settings_path())
            title = "The node keeps running"
            message = (
                "The app is now in the tray. To stop the node, "
                "choose \u201cQuit\u201d from the tray menu."
            )
            if not self._tray.notify(title, message):
                # Some desktops accept showMessage and render nothing at all.
                QMessageBox.information(self, title, message)

    def _confirm_quit(self) -> bool:
        answer = QMessageBox.question(
            self,
            "Quit",
            "The node is running. Quit and stop it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def begin_quit(self) -> None:
        """Told by the app that we are on the way out; stop resisting close."""
        self._quitting = True

    def _settings_path(self) -> Path:
        return settings_mod.settings_path(self._state_dir)
