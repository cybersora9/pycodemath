"""``python -m pycodemath.lint [paths...]`` — the command line (ruff-like).

Exit codes (as ruff): 0 = no findings, 1 = findings, 2 = usage / configuration /
internal error. ``--exit-zero`` turns 1 into 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

__all__ = ["main", "build_parser"]  # _entry: console script for I2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pycodemath.lint",
        description="Find the places where the mathematics in your Python code computes the wrong "
                    "value — each with an executed witness, and a proved fix where one exists.",
        epilog="Rules: python -m pycodemath.lint --list-rules; one rule: --explain PCM101. "
               "Configuration: [tool.pycodemath.lint] in pyproject.toml.",
    )
    p.add_argument("paths", nargs="*", default=None, help="files or directories (default: .)")
    p.add_argument("--format", "--output-format", dest="format", choices=("text", "json", "sarif", "concise"),
                   default="text", help="output format (default: text)")
    p.add_argument("--concise", action="store_true", help="one line per finding (no witness/fix lines)")
    p.add_argument("--select", type=_codes, default=None, metavar="CODES",
                   help="comma-separated codes or prefixes to run (replaces the default set)")
    p.add_argument("--extend-select", type=_codes, default=None, metavar="CODES")
    p.add_argument("--ignore", type=_codes, default=None, metavar="CODES")
    p.add_argument("--config", type=Path, default=None, help="pyproject.toml to read (default: nearest)")
    p.add_argument("--precision-bits", type=float, default=None, metavar="K",
                   help="PCM101 threshold in bits of error (default 26, or the config's)")
    p.add_argument("--statistics", action="store_true", help="count findings per rule")
    p.add_argument("--exit-zero", action="store_true", help="exit 0 even when there are findings")
    p.add_argument("--no-cache", action="store_true", help="do not read or write .pcm_lint_cache")
    p.add_argument("--no-execute", action="store_true",
                   help="PCM3xx: do not run functions (a refutation that needs execution stays undecided)")
    p.add_argument("--output", "-o", type=Path, default=None, help="write the report to this file")
    p.add_argument("--list-rules", action="store_true", help="print the rule table and exit")
    p.add_argument("--explain", metavar="CODE", help="print the documentation of one rule and exit")
    p.add_argument("--generate-docs", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--show-files", action="store_true", help="print each file as it is analysed (stderr)")
    return p


def _codes(text: str) -> list[str]:
    return [c.strip() for c in text.split(",") if c.strip()]


def main(argv: "Sequence[str] | None" = None) -> int:
    from .config import ConfigError, load_config
    from .rules import RULES, rules_markdown

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: 2 on a usage error, 0 for --help
        return int(exc.code or 0)
    if args.generate_docs:
        target = Path(__file__).with_name("RULES.md")
        target.write_text(rules_markdown(), encoding="utf-8")
        print(f"wrote {target}")
        return 0
    if args.list_rules:
        for r in RULES.values():
            print(f"{r.code}  {r.name:24} {r.level:7} {'on ' if r.default else 'off'}  {r.summary}")
        return 0
    if args.explain:
        code = args.explain.strip().upper()
        if code not in RULES:
            print(f"error: unknown rule {code!r}", file=sys.stderr)
            return 2
        md = rules_markdown()
        start = md.index(f"### {code}")
        nxt = md.find("\n### ", start + 1)
        nxt2 = md.find("\n## ", start + 1)
        ends = [e for e in (nxt, nxt2) if e != -1]
        print(md[start:min(ends) if ends else len(md)].strip())
        return 0
    paths = args.paths or ["."]
    for p in paths:
        if not Path(p).exists():
            print(f"error: {p}: no such file or directory", file=sys.stderr)
            return 2
    try:
        config = load_config(args.config, start=paths[0] if Path(paths[0]).is_dir() else Path(paths[0]).parent)
        config = config.with_cli(select=args.select, extend_select=args.extend_select, ignore=args.ignore,
                                 execute=False if args.no_execute else None,
                                 cache=False if args.no_cache else None)
        if args.precision_bits is not None:
            if not 1 <= args.precision_bits <= 64:
                raise ConfigError("--precision-bits must be from 1 to 64")
            from dataclasses import replace

            config = replace(config, precision_bits=float(args.precision_bits))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    from .engine import lint_paths
    from .report import format_json, format_statistics, format_text, summary_line, use_color
    from .sarif import format_sarif

    def progress(r: object) -> None:
        print(f"  {getattr(r, 'path', '')} ({getattr(r, 'seconds', 0):.2f} s)", file=sys.stderr)

    try:
        result = lint_paths(paths, config, on_file=progress if args.show_files else None)
    except KeyboardInterrupt:
        return 2
    except Exception as exc:  # an internal error is exit 2, never a silent pass
        print(f"error: internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for path, err in result.errors:
        print(f"warning: {path}: {err}", file=sys.stderr)
    findings = result.findings
    out = sys.stdout if args.output is None else None
    fmt = "concise" if args.concise else args.format
    if fmt == "json":
        text = format_json(findings)
    elif fmt == "sarif":
        text = format_sarif(findings, root_uri=Path(config.root).resolve().as_uri() + "/")
    else:
        color = out is not None and use_color(sys.stdout)
        if args.statistics:  # as ruff: the table instead of the findings
            text = format_statistics(result)
        else:
            text = format_text(findings, color=color, concise=fmt == "concise")
        text += summary_line(result) + "\n"
    if args.statistics and fmt in ("json", "sarif"):
        sys.stderr.write(format_statistics(result))
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.exit_zero:
        return 0
    if findings:
        return 1
    return 2 if result.errors else 0  # a file that was not analysed is not a pass


def _entry() -> None:
    """Console-script entry point (``pycodemath-lint``): exit with ``main``'s code."""
    sys.exit(main())


if __name__ == "__main__":
    _entry()
