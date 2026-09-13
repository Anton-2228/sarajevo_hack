"""The desktop front end: one window that turns this machine into a node.

The agent was built with this in mind -- `reporting.Reporter` is a protocol,
`AgentView` is plain data, and `AgentLoop.request_stop` exists for a Stop
button -- so nothing under `node.agent` or `node.core` had to change to add it.

Importing this package must not import Qt. `presets`, `settings` and `uistate`
are pure and testable on a machine with no display and no PyQt5 installed; the
widget modules pull Qt in when they are imported, which is only from
`node.gui.app`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
