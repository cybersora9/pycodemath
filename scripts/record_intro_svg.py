"""Record ``pycodemath intro`` into an animated SVG (assets/intro.svg).

    python scripts/record_intro_svg.py [--out assets/intro.svg]

Reproducible: the intro runs for real (its numbers come from the engine) into
a recording stream that claims to be a truecolor TTY, so the animation and its
timing are the CLI's own. Each finished line is stamped with the moment it was
written; the SVG shows it from that moment (CSS animation, plays once). With
``prefers-reduced-motion`` the SVG shows the final frame at once, the same rule
the CLI follows. The spinner is not recorded, only the lines it resolves into.

Half-block cells (the mark) are drawn as rectangles, not as the ``▀▄█``
glyphs, so the logo is exact whatever monospace font the viewer has; an empty
half stays transparent, as in the terminal. The page background is ink and
default text is kosc — the closed palette, nothing else.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pycodemath.cli import intro  # noqa: E402
from pycodemath.cli.term import PALETTE  # noqa: E402

CW, LH, FS, PAD = 8.4, 18.0, 14, 16


class Recorder:
    """A truecolor "TTY" that keeps every committed line with its timestamp."""

    def __init__(self) -> None:
        self.start = time.monotonic()
        self.buf = ""
        self.lines: list[tuple[float, str]] = []

    def isatty(self) -> bool:
        return True

    def flush(self) -> None:
        pass

    def write(self, text: str) -> int:
        for part in re.split(r"(\r|\n)", text):
            if part == "\r":
                self.buf = ""
            elif part == "\n":
                self.lines.append((time.monotonic() - self.start, self.buf))
                self.buf = ""
            else:
                self.buf += part
        return len(text)


def _hex(name: str) -> str:
    r, g, b = PALETTE[name]
    return f"#{r:02x}{g:02x}{b:02x}"


_RGB = {tuple(v): k for k, v in PALETTE.items()}


def cells(line: str) -> list[tuple[str, "str | None", "str | None", bool]]:
    """``(char, fg, bg, bold)`` per column, from the SGR escapes of ``line``."""
    out = []
    fg = bg = None
    bold = False
    for tok in re.split(r"(\x1b\[[0-9;?]*[A-Za-z])", line):
        m = re.fullmatch(r"\x1b\[([0-9;]*)m", tok)
        if m:
            codes = [int(c) for c in m[1].split(";") if c] or [0]
            i = 0
            while i < len(codes):
                c = codes[i]
                if c == 0:
                    fg = bg = None
                    bold = False
                elif c == 1:
                    bold = True
                elif c in (38, 48) and codes[i + 1] == 2:
                    name = _RGB[tuple(codes[i + 2:i + 5])]
                    if c == 38:
                        fg = name
                    else:
                        bg = name
                    i += 4
                i += 1
            continue
        if tok.startswith("\x1b"):
            continue
        out += [(ch, fg, bg, bold) for ch in tok]
    return out


def line_svg(row: int, line: str) -> str:
    y = PAD + row * LH
    parts: list[str] = []
    text_runs: list[tuple[int, str, str, bool]] = []
    for col, (ch, fg, bg, bold) in enumerate(cells(line)):
        x = PAD + col * CW
        if ch in "▀▄█":
            top = fg if ch in "▀█" else bg
            bottom = fg if ch in "▄█" else bg
            if top:
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{CW + 0.2:.1f}" height="{LH / 2 + 0.2:.1f}" fill="{_hex(top)}"/>')
            if bottom:
                parts.append(f'<rect x="{x:.1f}" y="{y + LH / 2:.1f}" width="{CW + 0.2:.1f}" height="{LH / 2:.1f}" fill="{_hex(bottom)}"/>')
            continue
        if bg:
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{CW + 0.2:.1f}" height="{LH:.1f}" fill="{_hex(bg)}"/>')
        style = (fg or "kosc", bold)
        if text_runs and text_runs[-1][0] + len(text_runs[-1][1]) == col and (text_runs[-1][2], text_runs[-1][3]) == style:
            start, s, f, b = text_runs[-1]
            text_runs[-1] = (start, s + ch, f, b)
        else:
            text_runs.append((col, ch, *style))
    for col, s, fg, bold in text_runs:
        if not s.strip():
            continue
        weight = ' font-weight="700"' if bold else ""
        parts.append(f'<text x="{PAD + col * CW:.1f}" y="{y + LH * 0.75:.1f}" fill="{_hex(fg)}"{weight} '
                     f'xml:space="preserve">{escape(s)}</text>')
    return "".join(parts)


def render(lines: list[tuple[float, str]]) -> str:
    width = PAD * 2 + 80 * CW
    height = PAD * 2 + len(lines) * LH
    groups = []
    for row, (t, line) in enumerate(lines):
        groups.append(f'<g class="l" style="animation-delay:{t:.2f}s">{line_svg(row, line)}</g>')
    title = "pycodemath intro"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{title}">\n'
        f"<title>{title} — generated by scripts/record_intro_svg.py</title>\n"
        "<style>\n"
        "text{font-family:ui-monospace,'JetBrains Mono','Cascadia Code',Consolas,'DejaVu Sans Mono',monospace;"
        f"font-size:{FS}px;font-variant-ligatures:none}}\n"
        ".l{opacity:0;animation:show .01s linear forwards}\n"
        "@keyframes show{to{opacity:1}}\n"
        "@media (prefers-reduced-motion: reduce){.l{opacity:1;animation:none}}\n"
        "</style>\n"
        f'<rect width="100%" height="100%" rx="8" fill="{_hex("ink")}"/>\n'
        + "\n".join(groups)
        + "\n</svg>\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "assets" / "intro.svg"
    rec = Recorder()
    intro.show(rec, env={"COLORTERM": "truecolor", "TERM": "xterm-256color"})  # type: ignore[arg-type]
    out.write_text(render(rec.lines), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({len(rec.lines)} lines, {rec.lines[-1][0]:.1f} s)")


if __name__ == "__main__":
    main()
