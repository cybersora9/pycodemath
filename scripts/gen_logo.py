"""Rasterize the Pycodemath mark (assets/pycodemath-sygnet.svg) into terminal text art.

Run once, offline, whenever the SVG changes:

    python scripts/gen_logo.py            # rewrites pycodemath/cli/_logo.py
    python scripts/gen_logo.py --preview  # prints every size, nothing written

No dependencies: the generator reads the SVG with ``xml.etree`` and understands
exactly what the mark uses — ``<path>`` with ``M``/``C``/``L`` commands stroked
with round caps and joins, and a filled ``<rect>`` with ``rx``. Anything else
in the file is an error, so a redesigned mark cannot be silently mis-drawn.

Each terminal cell holds two square pixels (the upper and lower half of a
``▀`` block), so the output is a grid of pixel letters: ``.`` background,
``r`` krew (#e11d33), ``k`` kosc (#f6f2f3). The CLI turns that grid into
half-block characters at runtime — truecolor, 256-color or monochrome — so the
committed art is independent of the terminal. Each pixel is the colour that
covers most of a 6x6 supersample grid, if it covers at least the size's threshold (``SIZES``).
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "assets" / "pycodemath-sygnet.svg"
OUT = ROOT / "pycodemath" / "cli" / "_logo.py"

#: name -> (pixel height = rows * 2, coverage threshold). ``mini`` goes in the
#: 3-line REPL banner, ``full`` in ``pycodemath intro``. The strokes are 9 units
#: in a 98-unit-tall mark, i.e. about one pixel at these sizes, so the smaller
#: grid needs a lower threshold or the integral breaks into dots. Chosen by eye
#: from ``--preview`` at heights 6-16 and thresholds 0.2-0.4.
SIZES = {"mini": (6, 0.2), "full": (14, 0.3)}
SUPER = 6
COLOURS = {"#e11d33": "r", "#f6f2f3": "k"}


def _flatten(d: str) -> list[tuple[float, float]]:
    tokens = re.findall(r"[MCL]|-?\d+(?:\.\d+)?", d)
    pts: list[tuple[float, float]] = []
    i, cmd = 0, ""
    cur = (0.0, 0.0)
    while i < len(tokens):
        if tokens[i] in "MCL":
            cmd = tokens[i]
            i += 1
            continue
        nums = [float(t) for t in tokens[i:i + (6 if cmd == "C" else 2)]]
        i += len(nums)
        if cmd in ("M", "L"):
            cur = (nums[0], nums[1])
            pts.append(cur)
        elif cmd == "C":
            p0, p1, p2, p3 = cur, (nums[0], nums[1]), (nums[2], nums[3]), (nums[4], nums[5])
            for s in range(1, 41):
                t = s / 40
                u = 1 - t
                pts.append((
                    u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
                ))
            cur = p3
        else:
            raise SystemExit(f"unsupported path command {cmd!r} in {SVG.name}")
    return pts


def _seg_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    n = dx * dx + dy * dy
    t = 0.0 if n == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / n))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _shapes() -> list[tuple[str, object]]:
    """``(pixel letter, inside(x, y) -> bool)`` in paint order, from the SVG."""
    tree = ET.parse(SVG)
    shapes: list[tuple[str, object]] = []
    for el in tree.iter():
        tag = el.tag.split("}")[-1]
        if tag == "path":
            colour = COLOURS[el.get("stroke", "").lower()]
            half = float(el.get("stroke-width", "1")) / 2
            pts = _flatten(el.get("d", ""))
            segs = list(zip(pts, pts[1:]))
            shapes.append((colour, lambda x, y, segs=segs, half=half:
                           any(_seg_dist((x, y), a, b) <= half for a, b in segs)))
        elif tag == "rect":
            colour = COLOURS[el.get("fill", "").lower()]
            x0, y0 = float(el.get("x", 0)), float(el.get("y", 0))
            w, h, r = float(el.get("width")), float(el.get("height")), float(el.get("rx", 0))

            def inside(x: float, y: float, x0=x0, y0=y0, w=w, h=h, r=r) -> bool:
                if not (x0 <= x <= x0 + w and y0 <= y <= y0 + h):
                    return False
                cx = min(max(x, x0 + r), x0 + w - r)
                cy = min(max(y, y0 + r), y0 + h - r)
                return math.hypot(x - cx, y - cy) <= r

            shapes.append((colour, inside))
        elif tag in ("svg", "g", "title"):
            continue
        else:
            raise SystemExit(f"unsupported element <{tag}> in {SVG.name}")
    return shapes


def _bbox(shapes: list[tuple[str, object]]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for gy in range(0, 121):
        for gx in range(0, 185):
            if any(f(gx, gy) for _, f in shapes):  # type: ignore[operator]
                xs.append(gx)
                ys.append(gy)
    return min(xs) - 0.5, min(ys) - 0.5, max(xs) + 0.5, max(ys) + 0.5


def rasterize(height: int, threshold: float = 0.3) -> list[str]:
    shapes = _shapes()
    x0, y0, x1, y1 = _bbox(shapes)
    scale = (y1 - y0) / height
    width = round((x1 - x0) / scale)
    xoff = x0 + ((x1 - x0) - width * scale) / 2
    rows = []
    for py in range(height):
        row = []
        for px in range(width):
            votes: dict[str, int] = {}
            for sy in range(SUPER):
                for sx in range(SUPER):
                    x = xoff + (px + (sx + 0.5) / SUPER) * scale
                    y = y0 + (py + (sy + 0.5) / SUPER) * scale
                    for colour, f in reversed(shapes):  # the last painted wins
                        if f(x, y):  # type: ignore[operator]
                            votes[colour] = votes.get(colour, 0) + 1
                            break
            best = max(votes, key=votes.__getitem__) if votes else "."
            row.append(best if votes.get(best, 0) >= threshold * SUPER * SUPER else ".")
        rows.append("".join(row))
    return rows


def halfblocks(rows: list[str]) -> list[str]:
    out = []
    for top, bottom in zip(rows[0::2], rows[1::2]):
        out.append("".join(
            " ▀▄█"[(t != ".") + 2 * (b != ".")] for t, b in zip(top, bottom)).rstrip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", action="store_true", help="print, do not write")
    args = ap.parse_args()
    grids = {name: rasterize(h, th) for name, (h, th) in SIZES.items()}
    if args.preview:
        for name, grid in grids.items():
            print(f"{name} ({len(grid[0])}x{len(grid)} px)")
            print("\n".join(grid))
            print("\n".join(halfblocks(grid)))
            print()
        return
    lines = [
        '"""GENERATED by scripts/gen_logo.py from assets/pycodemath-sygnet.svg — do not edit.',
        "",
        "Pixel grids of the mark: ``.`` background, ``r`` krew #e11d33, ``k`` kosc",
        "#f6f2f3. Two pixel rows make one terminal row (``▀`` half blocks); the",
        "monochrome rendering is shown under each grid.",
        '"""',
        "",
    ]
    for name, grid in grids.items():
        lines.append(f"# {name}: {len(grid[0])} x {len(grid)} px -> {len(grid[0])} cols x {len(grid) // 2} rows")
        for art in halfblocks(grid):
            lines.append(f"#   {art}")
        lines.append(f"{name.upper()} = (")
        lines += [f'    "{row}",' for row in grid]
        lines.append(")")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
