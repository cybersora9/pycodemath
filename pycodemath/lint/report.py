"""Text and JSON output of ``pycodemath lint`` (SARIF is in ``sarif.py``).

Text follows ruff's ``full`` format: ``path:line:col: CODE message``, and under it
the witness and the fix, indented — the evidence is the point of this linter, so it
is shown by default (``--concise`` prints the first line only). Colour only on a
TTY and never when ``NO_COLOR`` is set (https://no-color.org).

    >>> from pycodemath.lint.engine import lint_source
    >>> fs = lint_source("y = 2^10 + 1\\n", "c.py")
    >>> print(format_text(fs, color=False).splitlines()[0])
    c.py:1:5: PCM502 `2 ^ 10` is bitwise XOR, not a power: 2 ^ 10 = 8, but 2 ** 10 = 1024
"""

from __future__ import annotations

import json
import os
import sys
from typing import IO, Any, Sequence

from .engine import Finding, LintResult
from .rules import RULES

__all__ = ["format_text", "format_json", "format_statistics", "use_color", "summary_line"]

_RED, _YELLOW, _CYAN, _BOLD, _DIM, _RESET = "\033[31m", "\033[33m", "\033[36m", "\033[1m", "\033[2m", "\033[0m"


def use_color(stream: IO[str]) -> bool:
    """Colour when writing to a terminal, unless ``NO_COLOR`` is set (or ``FORCE_COLOR``)."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def _paint(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


def format_text(findings: Sequence[Finding], *, color: bool = False, concise: bool = False) -> str:
    out: list[str] = []
    for f in findings:
        lvl = RULES[f.code].level
        code_color = _RED if lvl == "error" else _YELLOW if lvl == "warning" else _CYAN
        head = (f"{_paint(f.path, _BOLD, color)}:{f.line}:{f.col}: "
                f"{_paint(f.code, _BOLD + code_color, color)} {f.message}")
        out.append(head)
        if concise:
            continue
        if f.witness is not None:
            w = f.witness
            inputs = ", ".join(f"{k}={v}" for k, v in w.inputs.items()) or "(constant)"
            out.append(_paint(f"    witness: {inputs} -> code gives {w.observed}, "
                              f"{'intended' if w.expected_program else 'exact'} {w.expected}"
                              + (f"  [{w.dtype}]" if w.dtype else ""), _DIM, color))
        if f.fix is not None:
            tag = "fix (proved)" if f.fix.proof else "suggestion"
            out.append(_paint(f"    {tag}: {f.fix.replacement}", _DIM, color))
            out.append(_paint(f"      {f.fix.description}", _DIM, color))
    return "\n".join(out) + ("\n" if out else "")


def summary_line(result: LintResult) -> str:
    n = len(result.findings)
    files = len(result.files)
    errs = len(result.errors)
    msg = f"Found {n} finding{'s' if n != 1 else ''} in {files} file{'s' if files != 1 else ''}" if n \
        else (f"No findings in {files} file{'s' if files != 1 else ''}" if errs
              else f"All checks passed ({files} file{'s' if files != 1 else ''})")
    if errs:
        msg += f"; {errs} file{'s' if errs != 1 else ''} not analysed"
    return msg + f" in {result.seconds:.2f} s."


def format_statistics(result: LintResult) -> str:
    """ruff ``--statistics``: count, code, name — most frequent first."""
    rows = result.statistics()
    if not rows:
        return ""
    width = max(len(str(c)) for _, c in rows)
    return "\n".join(f"{c:>{width}}\t{code}\t{RULES[code].name}" for code, c in rows) + "\n"


def format_json(findings: Sequence[Finding]) -> str:
    """A JSON list, one object per finding (ruff's ``--output-format json`` shape + witness)."""
    data: list[dict[str, Any]] = [f.to_dict() for f in findings]
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _stderr(text: str) -> None:
    sys.stderr.write(text)
