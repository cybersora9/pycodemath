"""Guards of module L1 (``pycodemath lint``): witnesses, SARIF, suppression — and the evaluation set.

``test_every_lint_finding_has_a_witness``: every finding on the evaluation set
(``tests/data/lint/{dev,held}``) carries a witness, and replaying it HERE — the
program executed again with the code's own operations, the exact value recomputed
by SymPy — shows a result different from the exact one.
``test_sarif_validates_against_schema``: the SARIF log validates against the OASIS
2.1.0 schema. ``test_noqa_comment_silences_exactly_one_rule``: ``# pcm:
ignore[CODE]`` removes that code on that line and nothing else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pycodemath.lint.config import Config
from pycodemath.lint.engine import lint_source
from pycodemath.lint.rules import RULES
from pycodemath.lint.sarif import to_sarif

DATA = Path(__file__).parent / "data" / "lint"
CASES = sorted((DATA / "dev").glob("*.py")) + sorted((DATA / "held").glob("*.py"))
_EXPECT = re.compile(r"#\s*expect\s+((?:PCM\d{3}\s*)+)")
_SELECT = re.compile(r"^#\s*select:\s*(.+)$", re.M)


def _config_for(text: str) -> Config:
    m = _SELECT.search(text)
    extra = tuple(c.strip() for c in m.group(1).split(",")) if m else ()
    return Config(extend_select=extra, cache=False)


def _expected(text: str) -> set[tuple[int, str]]:
    out = set()
    for i, line in enumerate(text.splitlines(), 1):
        m = _EXPECT.search(line)
        if m:
            out |= {(i, c) for c in m.group(1).split()}
    return out


_RESULTS: dict[Path, list] = {}


def _lint(path: Path) -> list:
    if path not in _RESULTS:
        text = path.read_text(encoding="utf-8")
        _RESULTS[path] = lint_source(text, path.name, _config_for(text))
    return _RESULTS[path]


def test_evaluation_set_is_large_enough() -> None:
    assert len(CASES) >= 40
    negatives = [c for c in CASES if not _expected(c.read_text(encoding="utf-8"))]
    assert len(negatives) >= len(CASES) // 3  # "half correct-but-tempting", give or take


@pytest.mark.parametrize("path", CASES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_lint_finding_has_a_witness(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    for f in _lint(path):
        rule = RULES[f.code]
        if rule.family == "meta" or f.code == "PCM302":
            assert f.witness is None
            continue
        assert f.witness is not None, f"{f.code} at line {f.line} has no witness"
        rep = f.witness.replay(source if f.witness.compare == "call" else None)
        assert rep.differs, f"{f.code} at line {f.line}: replay agrees with the exact value ({rep})"
        assert rep.observed == f.witness.observed or f.witness.compare == "call", (rep, f.witness)


@pytest.mark.parametrize("path", CASES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_evaluation_case(path: Path) -> None:
    """Exactly the expected findings: no false alarm (precision) and no miss (recall)."""
    got = {(f.line, f.code) for f in _lint(path)}
    assert got == _expected(path.read_text(encoding="utf-8"))


def test_sarif_validates_against_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((DATA / "sarif-schema-2.1.0.json").read_text(encoding="utf-8"))
    findings = [f for p in CASES for f in _lint(p)]
    assert len(findings) >= 20
    log = to_sarif(findings, root_uri="file:///repo/")
    jsonschema.Draft4Validator.check_schema(schema)
    errors = sorted(jsonschema.Draft4Validator(schema).iter_errors(log), key=lambda e: list(e.path))
    assert not errors, [f"{list(e.path)}: {e.message}" for e in errors[:5]]
    run = log["runs"][0]
    ids = [r["id"] for r in run["tool"]["driver"]["rules"]]
    assert ids == list(RULES)
    for res in run["results"]:
        assert ids[res["ruleIndex"]] == res["ruleId"]
        assert res["locations"][0]["physicalLocation"]["region"]["startLine"] >= 1
        assert run["tool"]["driver"]["rules"][res["ruleIndex"]]["helpUri"].endswith("#" + res["ruleId"].lower())
        assert "witness" in res["properties"]


def test_noqa_comment_silences_exactly_one_rule() -> None:
    base = (
        "def gross(price: float) -> float:\n"
        "    return round(price * 1.23, 2){c}\n"
        "\n"
        "EPS = 10^-6\n"
    )
    plain = {(f.line, f.code) for f in lint_source(base.format(c=""), "m.py")}
    assert plain == {(2, "PCM401"), (2, "PCM503"), (4, "PCM502")}
    one = {(f.line, f.code) for f in lint_source(base.format(c="  # pcm: ignore[PCM503]"), "m.py")}
    assert one == {(2, "PCM401"), (4, "PCM502")}
    noqa = {(f.line, f.code) for f in lint_source(base.format(c="  # noqa: PCM401"), "m.py")}
    assert noqa == {(2, "PCM503"), (4, "PCM502")}
    # a bare flake8/ruff "# noqa" is not ours, and silences nothing here
    bare = {(f.line, f.code) for f in lint_source(base.format(c="  # noqa"), "m.py")}
    assert bare == plain
    both = {(f.line, f.code) for f in lint_source(base.format(c="  # pcm: ignore"), "m.py")}
    assert both == {(4, "PCM502")}
