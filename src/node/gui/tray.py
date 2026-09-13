"""The tray icon, and what to do on the desktops that do not have one.

Closing the window leaves the node running, which is only safe if the operator
can get the window back. On GNOME under Wayland there is often no StatusNotifier
host at all unless an extension provides one, and when there is one it
frequently registers a second or two after the app starts. So availability is
re-checked once, and while there is no tray the window never hides.
"""

from __future__ import annotations

import logging

from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QAction, QMenu, QSystemTrayIcon, QWidget

from node.gui.icon import node_icon
from node.gui.uistate import UiPhase, status_label

LOG = logging.getLogger("node.gui")

# The StatusNotifier host may not be up when we are. One late re-check is
# enough; polling forever would be noise.
_RECHECK_MS = 5000


class TrayController(QObject):
    """Owns the tray icon and reports what the operator picked from its menu."""

    show_requested = pyqtSignal()
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    availability_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.available = QSystemTrayIcon.isSystemTrayAvailable()

        self._icon = QSystemTrayIcon(node_icon("idle"), self)
        self._icon.setToolTip("Node stopped")
        self._icon.activated.connect(self._on_activated)

        menu = QMenu()
        self._show = QAction("Show window", menu)
        self._show.triggered.connect(self.show_requested)
        font = self._show.font()
        font.setBold(True)
        self._show.setFont(font)

        self._start = QAction("Start", menu)
        self._start.triggered.connect(self.start_requested)
        self._stop = QAction("Stop", menu)
        self._stop.triggered.connect(self.stop_requested)
        self._quit = QAction("Quit", menu)
        self._quit.triggered.connect(self.quit_requested)

        menu.addAction(self._show)
        menu.addSeparator()
        menu.addAction(self._start)
        menu.addAction(self._stop)
        menu.addSeparator()
        menu.addAction(self._quit)
        menu.setDefaultAction(self._show)

        # Kept alive on the controller: a QMenu owned only by a local would be
        # collected and the tray would show an empty context menu.
        self._menu = menu
        self._icon.setContextMenu(menu)

        # Shown regardless: harmless if nothing renders it, and it appears by
        # itself if a host turns up later.
        self._icon.show()
        if not self.available:
            LOG.info("no system tray; the window will not hide on close")
            QTimer.singleShot(_RECHECK_MS, self._recheck)

    # -- updates ----------------------------------------------------------

    def update(self, phase: UiPhase, status: str, completed: int, failed: int) -> None:
        self._start.setEnabled(phase is UiPhase.IDLE)
        self._stop.setEnabled(phase is UiPhase.RUNNING)

        if phase is UiPhase.IDLE:
            self._icon.setIcon(node_icon("offline"))
            self._icon.setToolTip("Node stopped")
            return

        self._icon.setIcon(node_icon(status))
        self._icon.setToolTip(
            f"Node: {status_label(status)} · {completed} done / {failed} failed"
        )

    def notify(self, title: str, message: str) -> bool:
        """Show a balloon. Returns False when the desktop cannot, so the caller
        can fall back to a dialog rather than saying nothing at all."""
        if not self.available or not self._icon.supportsMessages():
            return False
        self._icon.showMessage(title, message, QSystemTrayIcon.Information, 5000)
        return True

    def _on_activated(self, reason: int) -> None:
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.show_requested.emit()

    def _recheck(self) -> None:
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.available = True
            self._icon.show()
            LOG.info("a system tray appeared; closing the window will now hide it")
            self.availability_changed.emit(True)
