"""Where the agent's log goes now that the window does not show it.

A rotating file in the state directory, plus stderr when there is one. The file
matters more than the pane did: it survives the session, it can be attached to
a bug report, and it is the same format the CLI writes.

`cli.configure_logging` cannot be reused here -- it adds a StreamHandler on
`sys.stderr` unconditionally, and under `pythonw.exe`, which the Windows entry
point binds to, `sys.stderr` is None.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"

LOG_FILE = "node-gui.log"
MAX_BYTES = 5 * 1024**2
BACKUPS = 3


def default_log_file(state_dir: Path) -> Path:
    return state_dir / LOG_FILE


def install_logging(level: str, log_file: Path | None = None) -> None:
    """Point the `node` logger at a file and, if we have one, a console."""
    root = logging.getLogger("node")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)

    if sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            rotating = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8"
            )
            rotating.setFormatter(formatter)
            root.addHandler(rotating)
        except OSError as error:
            root.warning("could not open the log file %s (%s)", log_file, error)

    if not root.handlers:
        # Windowed, and no writable state directory. Without this the stdlib
        # prints "No handlers could be found" on the first record.
        root.addHandler(logging.NullHandler())
