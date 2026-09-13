"""The dashboard's components, as widgets.

Stat tiles, status pills, the three-segment step bar and a round card -- the
same pieces the control room uses, so the node and the panel read as one thing.
Styling lives entirely in `theme.stylesheet`; these only set object names and
the dynamic properties that sheet selects on.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from node.gui.rounds import FAILED, RUNNING, STEPS, RoundEntry, describe, format_elapsed
from node.gui.uistate import status_label, status_variant


def repolish(widget: QWidget) -> None:
    """Qt does not re-evaluate a stylesheet when a dynamic property changes."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def rule() -> QFrame:
    line = QFrame()
    line.setObjectName("rule")
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    return line


def label(text: str = "", name: str = "") -> QLabel:
    made = QLabel(text)
    if name:
        made.setObjectName(name)
    return made


class Pill(QLabel):
    """A rounded status chip: `good`, `warn`, `crit` or `off`."""

    def __init__(self, text: str = "", tone: str = "off") -> None:
        super().__init__(text)
        self.setObjectName("pill")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        if self.property("tone") != tone:
            self.setProperty("tone", tone)
            repolish(self)

    def set_status(self, status: str) -> None:
        """Render one of the agent's own statuses."""
        self.setText(status_label(status))
        self.set_tone(status_variant(status))


class StatTile(QFrame):
    """One big tabular number with a small uppercase label under it."""

    def __init__(self, caption: str, value: str = "—") -> None:
        super().__init__()
        self.setObjectName("card")
        self._value = label(value, "statValue")
        self._caption = label(caption.upper(), "statLabel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)
        layout.addWidget(self._value)
        layout.addWidget(self._caption)

    def set_value(self, value: str, tone: str = "") -> None:
        self._value.setText(value)
        if self._value.property("tone") != tone:
            self._value.setProperty("tone", tone or None)
            repolish(self._value)


class StepBar(QWidget):
    """One thin segment per phase: grey ahead, amber behind, teal when done."""

    def __init__(self, steps: int = len(STEPS)) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._segments: list[QFrame] = []
        for _ in range(steps):
            segment = QFrame()
            segment.setObjectName("step")
            segment.setFixedHeight(5)
            segment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._segments.append(segment)
            layout.addWidget(segment)

        self.setFixedHeight(5)
        self.set_progress(0)

    def set_progress(self, done: int, *, failed: bool = False, complete: bool = False) -> None:
        for index, segment in enumerate(self._segments):
            if failed and index < max(done, 1):
                state = "failed"
            elif index >= done:
                state = "todo"
            elif complete:
                state = "done"
            else:
                # Amber for a round still under way -- the dashboard's own
                # `.step.done` colour -- and teal only once it is submitted.
                state = "active"
            if segment.property("state") != state:
                segment.setProperty("state", state)
                repolish(segment)


class RoundCard(QFrame):
    """One round: its id, where it has got to, and what it produced."""

    def __init__(self, entry: RoundEntry) -> None:
        super().__init__()
        self.setObjectName("card")
        self.round_id = entry.round_id

        self._id = label(entry.round_id, "roundId")
        self._id.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._pill = Pill()
        self._steps = StepBar()
        self._detail = label("", "subtle")
        self._detail.setWordWrap(True)
        self._elapsed = label("", "mono")
        self._elapsed.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(self._id, 1)
        head.addWidget(self._pill)

        foot = QHBoxLayout()
        foot.setSpacing(8)
        foot.addWidget(self._detail, 1)
        foot.addWidget(self._elapsed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        layout.addLayout(head)
        layout.addWidget(self._steps)
        layout.addLayout(foot)

        self.update_entry(entry)

    def update_entry(self, entry: RoundEntry, now: float | None = None) -> None:
        failed = entry.phase == FAILED
        self._steps.set_progress(
            entry.step_index if not failed else 1,
            failed=failed,
            complete=entry.finished and not failed,
        )

        if failed:
            self._pill.setText("error")
            self._pill.set_tone("crit")
        elif entry.finished:
            self._pill.setText("submitted")
            self._pill.set_tone("good")
        elif entry.phase == RUNNING:
            self._pill.setText("training")
            self._pill.set_tone("warn")
        else:
            self._pill.setText("accepted")
            self._pill.set_tone("off")

        text = describe(entry)
        self._detail.setText(text)
        self._detail.setToolTip(text if failed else "")
        self._detail.setObjectName("error" if failed else "subtle")
        repolish(self._detail)

        self._elapsed.setText(format_elapsed(entry.elapsed(now)))
