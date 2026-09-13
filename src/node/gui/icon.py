"""The tray and window icon, drawn rather than shipped.

A painted icon has no asset to lose in a wheel, needs no QtSvg, renders sharply
at 16 px (the Windows tray) and 22 px (Linux panels), works under the offscreen
platform in tests, and gets the status colour for free.
"""

from __future__ import annotations

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap

from node.gui.theme import status_colour

# Every size a tray or a task switcher is likely to ask for. Qt picks the
# closest and scales; giving it exact pixmaps avoids a blurry 16 px.
SIZES = (16, 22, 24, 32, 48, 64)

_CACHE: dict[tuple[str, str], QIcon] = {}


def node_icon(status: str = "idle") -> QIcon:
    """A rounded square in the status colour with an N in it. Cached per status."""
    hex_colour = status_colour(status)
    # Keyed on the colour too: the desktop can switch to dark while we run, and
    # a cache keyed on the status alone would keep serving the old palette.
    icon = _CACHE.get((status, hex_colour))
    if icon is None:
        colour = QColor(hex_colour)
        icon = QIcon()
        for size in SIZES:
            icon.addPixmap(_draw(size, colour))
        _CACHE[(status, hex_colour)] = icon
    return icon


def _draw(size: int, colour: QColor) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        inset = max(1.0, size * 0.06)
        body = QRectF(inset, inset, size - 2 * inset, size - 2 * inset)
        radius = size * 0.22

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(colour))
        painter.drawRoundedRect(body, radius, radius)

        font = QFont()
        font.setBold(True)
        # setPixelSize, not setPointSize: point sizes depend on the screen DPI
        # and a 16 px tray icon would get a letter that does not fit.
        font.setPixelSize(max(6, int(size * 0.62)))
        painter.setFont(font)
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(body, Qt.AlignCenter, "N")
    finally:
        # A QPainter still active when its QPixmap is destroyed warns on every
        # call and leaks the paint engine.
        painter.end()

    return pixmap
