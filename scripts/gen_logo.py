"""Rasterize the Pycodemath mark (assets/pycodemath-sygnet.svg) into terminal text art.

Run once, offline, whenever the SVG or the approved look changes:

    python scripts/gen_logo.py            # rewrites pycodemath/cli/_logo.py
    python scripts/gen_logo.py --preview  # prints the grid and the art, writes nothing

Needs Pillow (``pip install pillow``) — HERE only. The committed ``_logo.py`` is
plain strings, so the package never imports Pillow at run time.

The approved look (owner, PR #2): 36 columns wide, stroke width 12 SVG units
(the file draws 9; the terminal needs it thicker), transparent background.
The geometry is read from the SVG — the integral's cubic path, the chevron's
polyline and the cursor's rounded rect — and anything else in the file is an
error, so a redesigned mark cannot be silently mis-drawn. Then, exactly as the
approved sketch did:

1. each colour is drawn into its own 8x supersampled mask (``ImageDraw.line``
   with ``joint="curve"``, round caps as discs at the ends and, for the
   chevron, at its three points; the cursor as ``rounded_rectangle``);
2. each mask is box-downsampled to ``(36, round(36 * 120 / 180))`` pixels;
3. a pixel is on where coverage > 128/255; kosc wins where both are on;
4. empty pixel rows are trimmed in pairs, top and bottom (one terminal row is
   two pixel rows), which leaves 10 rows.

The grid letters are ``.`` empty, ``r`` krew (#e11d33), ``k`` kosc (#f6f2f3);
``term.logo_rows`` turns each pair of pixel rows into half blocks at run time.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "assets" / "pycodemath-sygnet.svg"
OUT = ROOT / "pycodemath" / "cli" / "_logo.py"

COLS = 36          #: terminal columns of the full mark
STROKE = 12        #: stroke width in SVG units (the SVG itself says 9)
SUPER = 8          #: supersampling factor
BEZIER_STEPS = 40  #: line segments per cubic
THRESHOLD = 128    #: a pixel is on above this coverage (0-255)
COLOURS = {"#e11d33": ("krew", "r"), "#f6f2f3": ("kosc", "k")}

Point = tuple[float, float]


def _numbers(text: str) -> list[float]:
    return [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _path(d: str) -> tuple[str, list[list[Point]]]:
    """``("C", [[p0, p1, p2, p3], ...])`` for ``M C C ...`` or ``("L", [[points]])`` for ``M L L ...``."""
    tokens = re.findall(r"[A-Za-z]|-?\d+(?:\.\d+)?", d)
    if not tokens or tokens[0] != "M":
        raise SystemExit(f"a path in {SVG.name} must start with M: {d!r}")
    cmds = {t for t in tokens if t.isalpha()} - {"M"}
    if cmds not in ({"C"}, {"L"}) or tokens.count("M") != 1:
        raise SystemExit(f"unsupported path in {SVG.name} (only M+C or M+L): {d!r}")
    nums = [float(t) for t in tokens if not t.isalpha()]
    pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    if cmds == {"L"}:
        return "L", [pts]
    if (len(pts) - 1) % 3:
        raise SystemExit(f"a cubic path in {SVG.name} needs 3 points per C: {d!r}")
    return "C", [[pts[i], pts[i + 1], pts[i + 2], pts[i + 3]] for i in range(0, len(pts) - 1, 3)]


def _bezier(p0: Point, p1: Point, p2: Point, p3: Point, n: int = BEZIER_STEPS) -> list[Point]:
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
    return out


def masks(cols: int = COLS, stroke: float = STROKE) -> dict[str, "object"]:
    """One downsampled ``L`` image per colour name, ``cols`` wide."""
    from PIL import Image, ImageDraw

    root = ET.parse(SVG).getroot()
    vx, vy, vw, vh = _numbers(root.get("viewBox", ""))
    W, H = round(vw * SUPER), round(vh * SUPER)
    out = {name: Image.new("L", (W, H), 0) for name, _ in COLOURS.values()}

    def tx(p: Point) -> Point:
        return ((p[0] - vx) * SUPER, (p[1] - vy) * SUPER)

    r = stroke * SUPER / 2

    def disc(draw: "ImageDraw.ImageDraw", p: Point) -> None:
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)

    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("svg", "g", "title"):
            continue
        if tag == "path":
            name = COLOURS[el.get("stroke", "").lower()][0]
            draw = ImageDraw.Draw(out[name])
            kind, parts = _path(el.get("d", ""))
            if kind == "C":
                pts = [tx(p) for seg in parts for p in _bezier(*seg)]
                ends = [pts[0], pts[-1]]
            else:
                pts = [tx(p) for p in parts[0]]
                ends = pts
            draw.line(pts, fill=255, width=int(stroke * SUPER), joint="curve")
            for p in ends:
                disc(draw, p)
        elif tag == "rect":
            name = COLOURS[el.get("fill", "").lower()][0]
            x, y = tx((float(el.get("x", 0)), float(el.get("y", 0))))
            w, h = float(el.get("width", 0)) * SUPER, float(el.get("height", 0)) * SUPER
            ImageDraw.Draw(out[name]).rounded_rectangle(
                [x, y, x + w, y + h], radius=float(el.get("rx", 0)) * SUPER, fill=255)
        else:
            raise SystemExit(f"unsupported element <{tag}> in {SVG.name}")
    size = (cols, round(cols * vh / vw))
    return {name: img.resize(size, Image.BOX) for name, img in out.items()}


def rasterize(cols: int = COLS, stroke: float = STROKE) -> list[str]:
    """The pixel grid (``.``/``r``/``k``), empty row pairs trimmed top and bottom."""
    m = masks(cols, stroke)
    krew, kosc = m["krew"], m["kosc"]
    w, h = krew.size  # type: ignore[attr-defined]
    rows = []
    for y in range(h):
        row = ""
        for x in range(w):
            if kosc.getpixel((x, y)) > THRESHOLD:  # type: ignore[attr-defined]
                row += "k"
            elif krew.getpixel((x, y)) > THRESHOLD:  # type: ignore[attr-defined]
                row += "r"
            else:
                row += "."
        rows.append(row)
    if len(rows) % 2:
        rows.append("." * w)
    pairs = [rows[i:i + 2] for i in range(0, len(rows), 2)]
    while pairs and not any(c != "." for row in pairs[0] for c in row):
        pairs.pop(0)
    while pairs and not any(c != "." for row in pairs[-1] for c in row):
        pairs.pop()
    return [row for pair in pairs for row in pair]


def halfblocks(rows: list[str]) -> list[str]:
    return ["".join(" ▀▄█"[(t != ".") + 2 * (b != ".")] for t, b in zip(top, bottom)).rstrip()
            for top, bottom in zip(rows[0::2], rows[1::2])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", action="store_true", help="print, do not write")
    args = ap.parse_args()
    grid = rasterize()
    if args.preview:
        print("\n".join(grid))
        print("\n".join(halfblocks(grid)))
        return
    lines = [
        '"""GENERATED by scripts/gen_logo.py from assets/pycodemath-sygnet.svg — do not edit.',
        "",
        f"The mark at {COLS} columns, stroke {STROKE}: ``.`` empty (transparent), ``r`` krew",
        "#e11d33, ``k`` kosc #f6f2f3. Two pixel rows make one terminal row (half",
        "blocks, ``term.logo_rows``); the monochrome rendering is shown below.",
        '"""',
        "",
        f"# full: {len(grid[0])} x {len(grid)} px -> {len(grid[0])} cols x {len(grid) // 2} rows",
    ]
    lines += [f"#   {art}".rstrip() for art in halfblocks(grid)]
    lines.append("FULL = (")
    lines += [f'    "{row}",' for row in grid]
    lines += [")", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
