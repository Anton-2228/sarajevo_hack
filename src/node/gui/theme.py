"""The control room's design tokens, ported to Qt.

Values are lifted verbatim from the dashboard's `:root` block so the node and
the panel it reports to look like one product. Qt stylesheets have no custom
properties, so the tokens live in a dict and the sheet is formatted from it.

Archivo and IBM Plex are web fonts and will usually not be installed, so every
family degrades to whatever this machine actually has.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PyQt5.QtCore import QPointF, QStandardPaths, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFontDatabase,
    QPainter,
    QPalette,
    QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import QApplication

from node.gui.uistate import status_variant

LOG = logging.getLogger("node.gui")

LIGHT: dict[str, str] = {
    "bg": "#F4F6F8",
    "surface": "#FFFFFF",
    "surface2": "#EAEEF2",
    "border": "#D8DFE6",
    "ink": "#131A21",
    "muted": "#5B6B78",
    "faint": "#8A97A3",
    "accent": "#0E8F79",
    "accent2": "#B4750A",
    "good": "#1E9E5A",
    "goodBg": "#E4F5EB",
    "warn": "#B4750A",
    "warnBg": "#FBF0DD",
    "crit": "#C1443C",
    "critBg": "#FBE7E5",
    "offBg": "#EDF0F2",
}

DARK: dict[str, str] = {
    "bg": "#0B0F13",
    "surface": "#121820",
    "surface2": "#1A222B",
    "border": "#26313C",
    "ink": "#E7EDF2",
    "muted": "#93A2AF",
    "faint": "#64727E",
    "accent": "#4FD9B8",
    "accent2": "#F0B24C",
    "good": "#46D67F",
    "goodBg": "#123322",
    "warn": "#F0B24C",
    "warnBg": "#3A2E12",
    "crit": "#F17369",
    "critBg": "#3A1917",
    "offBg": "#1C232C",
}

# First match wins. The web fonts are listed first so a machine that does have
# them looks exactly like the dashboard.
DISPLAY_FAMILIES = ("Archivo", "Inter", "Noto Sans", "DejaVu Sans")
BODY_FAMILIES = ("IBM Plex Sans", "Inter", "Noto Sans", "DejaVu Sans")
MONO_FAMILIES = ("IBM Plex Mono", "JetBrains Mono", "Noto Sans Mono", "DejaVu Sans Mono")

# Which token carries a pill variant's foreground. `off` is the dashboard grey.
_VARIANT_INK = {"good": "good", "warn": "warn", "crit": "crit", "off": "faint"}


def status_colour(status: str, t: dict[str, str] | None = None) -> str:
    """The solid colour for a status, for things a stylesheet cannot paint.

    The tray icon is drawn with QPainter, so it needs the hex rather than a
    pill variant -- but it must be the same hex the pill would have used.
    """
    t = t or tokens()
    return t[_VARIANT_INK[status_variant(status)]]


def is_dark(app: QApplication | None = None) -> bool:
    """Follow the desktop rather than forcing a theme on it."""
    app = app or QApplication.instance()
    if app is None:
        return False
    window = app.palette().color(QPalette.Window)
    return window.lightness() < 128


def tokens(dark: bool | None = None) -> dict[str, str]:
    return DARK if (is_dark() if dark is None else dark) else LIGHT


def pick_family(candidates: tuple[str, ...], *, fixed: bool = False) -> str:
    available = set(QFontDatabase().families())
    for family in candidates:
        if family in available:
            return family
    system = QFontDatabase.systemFont(
        QFontDatabase.FixedFont if fixed else QFontDatabase.GeneralFont
    )
    return system.family()


def fonts() -> dict[str, str]:
    return {
        "display": pick_family(DISPLAY_FAMILIES),
        "body": pick_family(BODY_FAMILIES),
        "mono": pick_family(MONO_FAMILIES, fixed=True),
    }


def stylesheet(t: dict[str, str] | None = None, f: dict[str, str] | None = None) -> str:
    t = t or tokens()
    f = f or fonts()
    arrows = {f"arrow_{k}": v for k, v in spin_arrows(t).items()}
    return _SHEET.format(**t, **arrows, **{f"font_{k}": v for k, v in f.items()})


# -- spin box arrows ------------------------------------------------------
#
# Styling a QSpinBox at all costs it its native arrows: Qt draws a sub-control
# itself once the widget is styled, and it has nothing to draw with. A
# stylesheet cannot paint a triangle either -- the CSS trick of a zero-sized
# box with two transparent borders comes out of Qt as a solid square -- so the
# arrows arrive as images, painted here for the same reasons the tray icon is
# (no asset to lose in a wheel, the right colour for the current palette).

ARROW_W = 9
ARROW_H = 6
_ARROWS: dict[str, dict[str, str]] = {}


def spin_arrows(t: dict[str, str] | None = None) -> dict[str, str]:
    """`{"up": url(...), ...}` for the sheet, or `none` if painting failed.

    Cached per palette: the desktop can switch to dark while the app runs.
    """
    t = t or tokens()
    key = f"{t['muted']}/{t['border']}"
    cached = _ARROWS.get(key)
    if cached is None:
        cached = {
            f"{name}{suffix}": _arrow_url(name, colour)
            for name in ("up", "down")
            for suffix, colour in (("", t["muted"]), ("_off", t["border"]))
        }
        _ARROWS[key] = cached
    return cached


def _arrow_url(direction: str, colour: str) -> str:
    """Paint a caret to the cache directory and return it as a QSS `url(...)`.

    Two files: Qt picks `name@2x.png` itself on a high-DPI screen, and a 9x6
    pixmap stretched by the compositor is a smudge.

    `none` on failure rather than an empty string: `image: ;` is a parse error,
    and Qt drops the entire application stylesheet over one of those.
    """
    try:
        # GenericCacheLocation, not CacheLocation: the latter is named after
        # whatever QApplication happens to be called, which under a test runner
        # is a directory nobody would recognise as ours.
        cache = Path(
            QStandardPaths.writableLocation(QStandardPaths.GenericCacheLocation)
            or tempfile.gettempdir()
        ) / "node-gui"
        cache.mkdir(parents=True, exist_ok=True)
        stem = f"spin-{direction}-{colour.lstrip('#').lower()}"
        for scale, suffix in ((1, ""), (2, "@2x")):
            target = cache / f"{stem}{suffix}.png"
            # Content is fully determined by the name, so an existing file is
            # the file we would write. Repainting it every launch is waste.
            if not target.exists() and not _paint_arrow(direction, colour, scale).save(
                str(target), "PNG"
            ):
                return "none"
        # Forward slashes and quotes: a Windows path is full of backslashes,
        # which a stylesheet reads as escapes.
        return f'url("{(cache / (stem + ".png")).as_posix()}")'
    except OSError as error:
        LOG.warning("could not paint the spin box arrows (%s)", error)
        return "none"


def _paint_arrow(direction: str, colour: str, scale: int) -> QPixmap:
    pixmap = QPixmap(ARROW_W * scale, ARROW_H * scale)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = ARROW_W * scale, ARROW_H * scale
        points = (
            [QPointF(0, h), QPointF(w, h), QPointF(w / 2, 0)]
            if direction == "up"
            else [QPointF(0, 0), QPointF(w, 0), QPointF(w / 2, h)]
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(colour)))
        painter.drawPolygon(QPolygonF(points))
    finally:
        painter.end()
    return pixmap


# `#id` selectors rather than classes: Qt has no class attribute, and dynamic
# properties need an explicit repolish on every change.
_SHEET = """
QWidget {{
    background: transparent;
    color: {ink};
    font-family: "{font_body}";
    font-size: 13px;
}}
QMainWindow, QDialog, #page {{ background: {bg}; }}

/* -- type ------------------------------------------------------------- */
#eyebrow {{
    font-family: "{font_mono}";
    font-size: 10px;
    letter-spacing: 1.4px;
    color: {faint};
}}
#title {{
    font-family: "{font_display}";
    font-size: 22px;
    font-weight: 800;
    color: {ink};
}}
#h2 {{
    font-family: "{font_display}";
    font-size: 14px;
    font-weight: 700;
    color: {ink};
}}
#subtle, #countTag {{
    font-family: "{font_mono}";
    font-size: 11px;
    color: {faint};
}}
#mono {{ font-family: "{font_mono}"; font-size: 12px; color: {muted}; }}
#roundId {{ font-family: "{font_mono}"; font-size: 13px; font-weight: 600; color: {ink}; }}
#error {{ font-family: "{font_mono}"; font-size: 11px; color: {crit}; }}

/* -- surfaces --------------------------------------------------------- */
#card {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 12px;
}}
#rule {{ background: {border}; max-height: 1px; border: none; }}
#empty {{
    border: 1px dashed {border};
    border-radius: 10px;
    padding: 22px;
    font-family: "{font_mono}";
    font-size: 12px;
    color: {faint};
}}

/* -- stat tiles ------------------------------------------------------- */
#statValue {{
    font-family: "{font_mono}";
    font-size: 24px;
    font-weight: 600;
    color: {ink};
}}
#statValue[tone="crit"] {{ color: {crit}; }}
#statLabel {{
    font-family: "{font_mono}";
    font-size: 10px;
    letter-spacing: 0.8px;
    color: {faint};
}}

/* -- pills ------------------------------------------------------------ */
#pill {{
    font-family: "{font_mono}";
    font-size: 11px;
    font-weight: 500;
    border-radius: 9px;
    padding: 3px 10px;
    background: {offBg};
    color: {faint};
}}
#pill[tone="good"] {{ background: {goodBg}; color: {good}; }}
#pill[tone="warn"] {{ background: {warnBg}; color: {warn}; }}
#pill[tone="crit"] {{ background: {critBg}; color: {crit}; }}
#pill[tone="off"]  {{ background: {offBg};  color: {faint}; }}

#tag {{
    font-family: "{font_mono}";
    font-size: 10px;
    background: {surface2};
    color: {muted};
    border-radius: 5px;
    padding: 2px 7px;
}}

/* -- progress steps --------------------------------------------------- */
#step {{ border-radius: 3px; background: {surface2}; min-height: 5px; max-height: 5px; }}
#step[state="active"] {{ background: {accent2}; }}
#step[state="done"]   {{ background: {accent}; }}
#step[state="failed"] {{ background: {crit}; }}

/* -- buttons ---------------------------------------------------------- */
QPushButton {{
    font-family: "{font_body}";
    font-size: 13px;
    background: {surface};
    color: {ink};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 7px 14px;
}}
QPushButton:hover {{ background: {surface2}; }}
QPushButton:disabled {{ color: {faint}; background: {surface}; }}

#primary {{
    font-family: "{font_display}";
    font-size: 15px;
    font-weight: 700;
    background: {accent};
    color: {surface};
    border: 1px solid {accent};
    border-radius: 10px;
    padding: 11px 18px;
}}
#primary:hover {{ background: {accent}; border-color: {ink}; }}
#primary:disabled {{ background: {surface2}; color: {faint}; border-color: {border}; }}
#primary[tone="stop"] {{ background: {crit}; border-color: {crit}; color: #FFFFFF; }}

#danger {{ color: {crit}; border-color: {crit}; }}
#danger:hover {{ background: {critBg}; }}

/* Segmented preset picker: one rounded group, dividers between. */
#segment {{
    font-family: "{font_mono}";
    font-size: 12px;
    background: {surface2};
    color: {muted};
    border: 1px solid {border};
    border-radius: 0px;
    padding: 6px 14px;
}}
#segment:hover {{ color: {ink}; }}
#segment:checked {{ background: {accent}; color: {surface}; border-color: {accent}; }}
#segment:disabled {{ color: {faint}; }}
#segment[edge="left"]  {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
#segment[edge="right"] {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}

/* -- meters ----------------------------------------------------------- */
QProgressBar {{
    font-family: "{font_mono}";
    font-size: 10px;
    color: {faint};
    background: {surface2};
    border: none;
    border-radius: 4px;
    min-height: 8px;
    max-height: 8px;
    text-align: right;
}}
QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}
QProgressBar#busy::chunk {{ background: {accent2}; border-radius: 4px; }}

/* -- scrolling -------------------------------------------------------- */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {border}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {faint}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

/* -- dialogs ---------------------------------------------------------- */
QSpinBox, QDoubleSpinBox {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 6px 24px 6px 8px;
    color: {ink};
    selection-background-color: {accent};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {accent}; }}
QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {surface2};
    color: {faint};
}}
/* The buttons have to be given a size and the arrows an image; see
   `spin_arrows` for why Qt will not draw either of them for us here. */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    background: transparent;
    border: none;
    width: 22px;
    height: 15px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {surface2};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: {arrow_up}; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: {arrow_down}; }}
/* `:off` is the end of the range: the click does nothing, so say so. */
QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off,
QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:off {{
    image: {arrow_up_off};
}}
QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off,
QDoubleSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:off {{
    image: {arrow_down_off};
}}
QCheckBox {{ spacing: 8px; }}
QStatusBar {{ color: {faint}; font-family: "{font_mono}"; font-size: 10px; }}
QStatusBar::item {{ border: none; }}
QToolTip {{
    background: {surface};
    color: {ink};
    border: 1px solid {border};
    padding: 5px;
}}
"""
