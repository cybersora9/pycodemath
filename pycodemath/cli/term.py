"""Terminal styling for the presentation commands — the closed brand palette.

Four colours and nothing else (no glow, no shadow, no other hues):

    ink  #0e0e12   the panel the logo is drawn on
    krew #e11d33   accent: the mark, headings, step numbers, the VERIFIED badge
    hot  #ff3a52   REFUTED
    kosc #f6f2f3   light strokes and text, only ever on ink or krew

``kosc`` is never used on the terminal's own background: it would vanish on a
light theme. Plain text keeps the terminal's default colour.

Colour depth, decided once per stream (``color_mode``):

* ``mono``      — ``NO_COLOR`` is set (https://no-color.org), stdout is not a
                  TTY, or ``TERM=dumb``: not one escape byte is written;
* ``truecolor`` — ``COLORTERM`` is ``truecolor``/``24bit``, or Windows Terminal
                  (``WT_SESSION``);
* ``256``       — every other TTY: each palette entry is mapped to the nearest
                  xterm-256 colour (``nearest_256``).
"""

from __future__ import annotations

import os
import sys
from typing import IO, Mapping

PALETTE: dict[str, tuple[int, int, int]] = {
    "ink": (0x0E, 0x0E, 0x12),
    "krew": (0xE1, 0x1D, 0x33),
    "hot": (0xFF, 0x3A, 0x52),
    "kosc": (0xF6, 0xF2, 0xF3),
}


def color_mode(stream: "IO[str] | None" = None, env: "Mapping[str, str] | None" = None) -> str:
    """``"truecolor"``, ``"256"`` or ``"mono"`` for ``stream`` (default stdout)."""
    env = os.environ if env is None else env
    stream = sys.stdout if stream is None else stream
    if env.get("NO_COLOR", ""):
        return "mono"
    try:
        tty = stream.isatty()
    except (AttributeError, ValueError):
        tty = False
    if not tty or env.get("TERM", "") == "dumb":
        return "mono"
    if env.get("COLORTERM", "").lower() in ("truecolor", "24bit") or env.get("WT_SESSION"):
        return "truecolor"
    return "256"


def _cube_levels() -> list[int]:
    return [0, 95, 135, 175, 215, 255]


def nearest_256(rgb: tuple[int, int, int]) -> int:
    """The xterm-256 index closest to ``rgb`` (6x6x6 cube or the grey ramp)."""
    best, best_d = 16, 1 << 30
    levels = _cube_levels()
    for i, r in enumerate(levels):
        for j, g in enumerate(levels):
            for k, b in enumerate(levels):
                d = (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2
                if d < best_d:
                    best, best_d = 16 + 36 * i + 6 * j + k, d
    for n in range(24):
        v = 8 + 10 * n
        d = (v - rgb[0]) ** 2 + (v - rgb[1]) ** 2 + (v - rgb[2]) ** 2
        if d < best_d:
            best, best_d = 232 + n, d
    return best


class Style:
    """Wraps text in the escapes of one colour mode (``mono`` = identity)."""

    def __init__(self, mode: str = "mono") -> None:
        if mode not in ("truecolor", "256", "mono"):
            raise ValueError(f"unknown colour mode {mode!r}")
        self.mode = mode

    @property
    def color(self) -> bool:
        return self.mode != "mono"

    def _code(self, name: str, ground: int) -> str:
        if self.mode == "truecolor":
            r, g, b = PALETTE[name]
            return f"{ground};2;{r};{g};{b}"
        return f"{ground};5;{nearest_256(PALETTE[name])}"

    def paint(self, text: str, fg: "str | None" = None, bg: "str | None" = None,
              bold: bool = False) -> str:
        if not self.color or not text:
            return text
        codes = []
        if bold:
            codes.append("1")
        if fg:
            codes.append(self._code(fg, 38))
        if bg:
            codes.append(self._code(bg, 48))
        if not codes:
            return text
        return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"

    def bold(self, text: str) -> str:
        return self.paint(text, bold=True)

    def accent(self, text: str) -> str:
        return self.paint(text, fg="krew", bold=True)

    def refuted(self, text: str) -> str:
        return self.paint(text, fg="hot", bold=True)

    def verified(self, text: str) -> str:
        return self.paint(f" {text} ", fg="kosc", bg="krew", bold=True) if self.color else text


def logo_rows(grid: "tuple[str, ...]", style: Style) -> list[str]:
    """Half-block rows of a pixel grid from ``_logo`` (``.``/``r``/``k``).

    In colour the mark sits on an ink panel one column wider on each side, so
    its light stroke reads on any terminal theme; in mono the shapes alone.
    """
    names = {"r": "krew", "k": "kosc", ".": "ink"}
    rows = []
    for top, bottom in zip(grid[0::2], grid[1::2]):
        if not style.color:
            rows.append("".join(
                " ▀▄█"[(t != ".") + 2 * (b != ".")] for t, b in zip(top, bottom)))
            continue
        # runs of one (glyph, fg, bg) share one escape sequence
        runs: list[list] = []
        for t, b in zip("." + top + ".", "." + bottom + "."):
            key = (" ", None, "ink") if t == "." and b == "." else ("▀", names[t], names[b])
            if runs and runs[-1][0] == key:
                runs[-1][1] += 1
            else:
                runs.append([key, 1])
        cells = [style.paint(glyph * n, fg=fg, bg=bg) for (glyph, fg, bg), n in runs]
        rows.append("".join(cells))
    return rows


def logo_width(grid: "tuple[str, ...]", style: Style) -> int:
    return len(grid[0]) + (2 if style.color else 0)


def visible_len(text: str) -> int:
    """Length on screen: ANSI escapes do not count."""
    import re

    return len(re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text))
