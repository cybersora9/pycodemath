"""SARIF 2.1.0 output — what GitHub code scanning (``upload-sarif``) reads.

One ``run`` with ``tool.driver`` listing every rule (id, name, short/full
description, ``helpUri``, default level), and one ``result`` per finding:
``ruleId``/``ruleIndex``, ``level``, ``message.text``, a ``physicalLocation`` relative
to ``%SRCROOT%``, ``partialFingerprints`` (stable across line moves: rule +
function + the expression), the witness in ``properties`` and — when a fix was
proved or suggested — ``fixes`` with one ``artifactChange``. A witness is also
given as a ``codeFlow`` step at the reported location: CodeQL uses code flows for
the path to a bug; ours has one step, and its message is the executed input.

``tests/test_lint_guards.py::test_sarif_validates_against_schema`` validates the
output against the OASIS schema (``tests/data/lint/sarif-schema-2.1.0.json``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from .engine import Finding
from .rules import RULES, rule_url

__all__ = ["to_sarif", "format_sarif", "SARIF_SCHEMA_URI"]

SARIF_SCHEMA_URI = "https://json.schemastore.org/sarif-2.1.0.json"


def _version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except ImportError:  # pragma: no cover
        return "0"


def _rule(code: str) -> dict[str, Any]:
    r = RULES[code]
    full = r.why
    return {
        "id": r.code,
        "name": "".join(p.capitalize() for p in r.name.split("-")),
        "shortDescription": {"text": r.summary[0].upper() + r.summary[1:]},
        "fullDescription": {"text": full},
        "helpUri": rule_url(r.code),
        "help": {"text": f"{r.summary}. Witness: {r.witness}.",
                 "markdown": f"**{r.code} {r.name}** — {r.summary}.\n\n{r.why}\n\n*Witness:* {r.witness}."},
        "defaultConfiguration": {"level": r.level, "enabled": r.default},
        "properties": {"tags": ["math", r.family], "precision": "very-high"},
    }


def _region(line: int, col: int, end_line: int, end_col: int) -> dict[str, int]:
    reg = {"startLine": max(1, line), "startColumn": max(1, col)}
    if end_line >= line:
        reg["endLine"] = end_line
        if end_line > line or end_col > col:
            reg["endColumn"] = max(1, end_col)
    return reg


def _fingerprint(f: Finding) -> str:
    w = f.witness.program if f.witness is not None else f.message
    h = hashlib.sha256(f"{f.code}|{f.path}|{f.function}|{w}".encode()).hexdigest()
    return h[:32]


def _result(f: Finding, index: dict[str, int]) -> dict[str, Any]:
    loc = {"physicalLocation": {"artifactLocation": {"uri": f.path, "uriBaseId": "%SRCROOT%"},
                                "region": _region(f.line, f.col, f.end_line, f.end_col)}}
    res: dict[str, Any] = {
        "ruleId": f.code,
        "ruleIndex": index[f.code],
        "level": RULES[f.code].level,
        "message": {"text": f.message},
        "locations": [loc],
        "partialFingerprints": {"pcmLint/v1": _fingerprint(f)},
    }
    props: dict[str, Any] = {}
    if f.function:
        props["function"] = f.function
    if f.witness is not None:
        props["witness"] = f.witness.to_dict()
        res["codeFlows"] = [{"threadFlows": [{"locations": [{"location": dict(
            loc, message={"text": f"witness: {f.witness.describe()}"})}]}]}]
    if props:
        res["properties"] = props
    if f.fix is not None:
        res["fixes"] = [{
            "description": {"text": f.fix.description + ("" if f.fix.proof else " (suggestion, not an equivalence)")},
            "artifactChanges": [{
                "artifactLocation": {"uri": f.path, "uriBaseId": "%SRCROOT%"},
                "replacements": [{
                    "deletedRegion": _region(f.fix.line, f.fix.col, f.fix.end_line, f.fix.end_col),
                    "insertedContent": {"text": f.fix.replacement},
                }],
            }],
        }]
    return res


def to_sarif(findings: Sequence[Finding], *, root_uri: "str | None" = None) -> dict[str, Any]:
    """The SARIF log as a dict. ``root_uri`` (``file:///...//``) anchors ``%SRCROOT%``."""
    codes = list(RULES)
    index = {c: i for i, c in enumerate(codes)}
    run: dict[str, Any] = {
        "tool": {"driver": {
            "name": "pycodemath-lint",
            "informationUri": "https://github.com/cybersora9/pycodemath",
            "version": _version(),
            "semanticVersion": _version(),
            "rules": [_rule(c) for c in codes],
        }},
        "results": [_result(f, index) for f in findings],
        "columnKind": "unicodeCodePoints",
    }
    if root_uri:
        run["originalUriBaseIds"] = {"%SRCROOT%": {"uri": root_uri if root_uri.endswith("/") else root_uri + "/"}}
    return {"$schema": SARIF_SCHEMA_URI, "version": "2.1.0", "runs": [run]}


def format_sarif(findings: Sequence[Finding], *, root_uri: "str | None" = None) -> str:
    return json.dumps(to_sarif(findings, root_uri=root_uri), indent=2, ensure_ascii=False) + "\n"
