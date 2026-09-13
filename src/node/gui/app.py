"""Bootstrapping the window, and the one sequence that must not go wrong: quit.

Quitting has to stop the node, and stopping the node is not instant. So the
quit path never blocks the GUI thread -- it asks, waits on a timer, escalates,
and as an absolute last resort exits hard. The hard exit is only defensible
because the agent writes every finished payload to its outbox before the
network call and resends it on the next start; nothing is lost by it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import sys
from pathlib import Path

from PyQt5.QtCore import QLockFile, Qt, QTimer
from PyQt5.QtNetwork import QLocalServer, QLocalSocket
from PyQt5.QtWidgets import QApplication, QProgressDialog

from node.agent import resources
from node.agent.config import ENV_PREFIX, default_state_dir
from node.gui import settings as settings_mod
from node.gui import theme
from node.gui.icon import node_icon
from node.gui.logsetup import default_log_file, install_logging
from node.gui.mainwindow import MainWindow
from node.gui.tray import TrayController
from node.gui.uistate import UiPhase
from node.gui.worker import AgentController

LOG = logging.getLogger("node.gui")

# A second instance would share identity.json and race the first over the task
# journal, heartbeating as the same node from two processes.
LOCK_FILE = "gui.lock"
_LOCK_STALE_MS = 30_000

# How long a graceful quit is given before the training child is killed.
_QUIT_GRACE_MS = 30_000
# And how long after that before we stop being polite about it.
_QUIT_HARD_MS = 5_000


def main(argv: list[str] | None = None) -> int:
    # Both must be set before the QApplication exists, or they do nothing.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Node")
    app.setOrganizationName("sarajevo")
    # Wayland matches a window to its icon through this, not through the title.
    app.setDesktopFileName("node-gui")
    app.setWindowIcon(node_icon("idle"))
    # The window closing must not end the process; the tray outlives it.
    app.setQuitOnLastWindowClosed(False)

    state_dir = _state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)

    lock = QLockFile(str(state_dir / LOCK_FILE))
    lock.setStaleLockTime(_LOCK_STALE_MS)
    if not lock.tryLock(100):
        # Launching again is how someone asks for the window back when it is
        # hidden in the tray, so raise the running one rather than saying no.
        _raise_running_instance(state_dir)
        return 0

    settings = settings_mod.load(settings_mod.settings_path(state_dir))
    install_logging(
        os.environ.get(ENV_PREFIX + "LOG_LEVEL", "info").lower(),
        default_log_file(state_dir),
    )
    app.setStyleSheet(theme.stylesheet())

    hw = resources.detect_hardware(state_dir.parent)
    controller = AgentController()
    tray = TrayController()
    window = MainWindow(settings, hw, controller, state_dir=state_dir, tray=tray)

    quitter = Quitter(app, window, controller, state_dir)
    window.quit_requested.connect(quitter.begin)
    server = _listen_for_second_instance(state_dir, window)

    app.aboutToQuit.connect(lambda: _on_about_to_quit(window, controller, state_dir))
    _install_signal_handlers(app, quitter)

    window.restore_geometry()
    window.show()
    LOG.info("node-gui started; state in %s", state_dir)

    if settings.autostart:
        QTimer.singleShot(0, window.start_node)

    try:
        return app.exec_()
    finally:
        if server is not None:
            server.close()
        lock.unlock()


class Quitter:
    """The shutdown sequence, as an object so its timers have an owner."""

    def __init__(self, app: QApplication, window: MainWindow, controller, state_dir: Path) -> None:
        self._app = app
        self._window = window
        self._controller = controller
        self._state_dir = state_dir
        self._dialog: QProgressDialog | None = None
        self._started = False

    def begin(self) -> None:
        if self._started:
            return
        self._started = True
        self._window.begin_quit()

        if not self._controller.is_running():
            self._app.quit()
            return

        self._dialog = QProgressDialog(
            "Stopping the node…", "Force quit", 0, 0, self._window
        )
        self._dialog.setWindowTitle("Quit")
        self._dialog.setWindowModality(Qt.WindowModal)
        self._dialog.canceled.connect(self._escalate)
        self._dialog.show()

        self._controller.stopped.connect(self._finish)
        self._controller.request_stop()

        if self._controller.phase is UiPhase.STOPPING:
            self._window.statusBar().showMessage("Stopping the node…")
        QTimer.singleShot(_QUIT_GRACE_MS, self._escalate)

    def _escalate(self) -> None:
        """The round is still training. Kill it -- it will be retried."""
        if not self._controller.is_running():
            return
        LOG.warning("the node did not stop in time; killing the training child")
        self._controller.force_stop()
        QTimer.singleShot(_QUIT_HARD_MS, self._hard_exit)

    def _finish(self, _code: int = 0) -> None:
        if self._dialog is not None:
            self._dialog.close()
        self._app.quit()

    def _hard_exit(self) -> None:
        """Last resort. Safe because every finished payload is already on disk:
        the agent stashes it before the POST and resends it on the next start.
        Do not "fix" this into a wait -- a hung quit is what it exists to avoid.
        """
        if not self._controller.is_running():
            return
        LOG.error("the node is still not stopping; exiting hard")
        self._controller.kill_children()
        os._exit(0)


def _on_about_to_quit(window: MainWindow, controller, state_dir: Path) -> None:
    """Nothing of ours may outlive the process.

    A fastText worker orphaned by an exiting GUI keeps every budgeted core busy
    forever: it is in no job object on Windows and in no killable process group
    on Linux, so killing the tree here is mandatory rather than tidy.
    """
    try:
        controller.kill_children()
    except Exception:  # noqa: BLE001 - we are leaving; nothing may raise here
        LOG.exception("could not kill the child processes")
    try:
        settings_mod.save(window.collect_settings(), settings_mod.settings_path(state_dir))
    except Exception:  # noqa: BLE001
        LOG.exception("could not save the settings")


def _install_signal_handlers(app: QApplication, quitter: Quitter) -> None:
    """Ctrl+C and SIGTERM should stop the node, not strand it.

    Legal here only because this is the main thread -- which is also why the
    GUI constructs `AgentLoop` itself and never calls `loop.run_agent`, whose
    own handlers would raise off the main thread.
    """

    def handler(signum, frame):  # noqa: ARG001
        quitter.begin()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    # Qt's event loop sits in C, so Python-level handlers only run when the
    # interpreter gets control. An idle timer hands it over periodically.
    keepalive = QTimer(app)
    keepalive.start(200)
    keepalive.timeout.connect(lambda: None)


def _socket_name(state_dir: Path) -> str:
    """Per state directory, so two nodes on one machine do not talk to each other."""
    digest = hashlib.sha1(str(state_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"sarajevo-node-gui-{digest}"


def _listen_for_second_instance(state_dir: Path, window: MainWindow) -> QLocalServer | None:
    name = _socket_name(state_dir)
    # A crashed instance leaves its socket file behind on POSIX, and listen()
    # then fails forever. The lock file, not this, is what decides who is first.
    QLocalServer.removeServer(name)

    server = QLocalServer()
    if not server.listen(name):
        LOG.debug("could not listen on %s: %s", name, server.errorString())
        return None

    def on_connection() -> None:
        connection = server.nextPendingConnection()
        if connection is not None:
            connection.disconnected.connect(connection.deleteLater)
            window.show_window()

    server.newConnection.connect(on_connection)
    return server


def _raise_running_instance(state_dir: Path) -> None:
    socket = QLocalSocket()
    socket.connectToServer(_socket_name(state_dir))
    if socket.waitForConnected(500):
        socket.write(b"show")
        socket.flush()
        socket.waitForBytesWritten(500)
        socket.disconnectFromServer()
    else:
        LOG.warning("node-gui is already running, but its window did not respond")


def _state_dir() -> Path:
    raw = os.environ.get(ENV_PREFIX + "STATE_DIR")
    return Path(raw).expanduser() if raw else default_state_dir()


if __name__ == "__main__":
    raise SystemExit(main())
