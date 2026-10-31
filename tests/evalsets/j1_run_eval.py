"""Run the J1 evaluation set: python tests/evalsets/j1_run_eval.py [dev|holdout|all]

Prints one line per failing case and a summary table. ``evaluate`` is also used
by tests/test_units.py, so the measured numbers and the test suite cannot drift.

Scoring:
* convert — exact value equal (SymPy, symbolic difference 0) or the expected
  exception class; a certified conversion must also have round_trip True.
* check / names — status equal; for REFUTED the expected location must appear in
  the reported location / operands, and a counterexample must be present; an
  expected warning kind must be among the warnings.
* false REFUTED = REFUTED where the expectation was verified/undecided.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "pycodemath" / "code" / "data" / "J1_units_eval.jsonl"


def load(split: str = "all") -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [c for c in cases if split == "all" or c["split"] == split]


def _warned(warnings: Any, kind: "str | None") -> bool:
    if kind is None:
        return True
    return any(kind in w.kind for w in warnings)


def evaluate(case: dict[str, Any]) -> tuple[bool, str, bool]:
    """-> (passed, observed, false_refutation)."""
    from pycodemath.engine import units as U

    kind = case["kind"]
    if kind == "convert":
        try:
            c = U.convert(case["q"], case["to"])
        except Exception as exc:  # noqa: BLE001 - scoring the class
            name = type(exc).__name__
            return name == case.get("error"), name, False
        if "error" in case:
            return False, f"converted to {c.value_text}", False
        ok = sp.simplify(sp.sympify(case["expected"]) - c.value) == 0 and c.certificate["round_trip"]
        return bool(ok), c.value_text, False
    expected = case["expected"]
    try:
        if kind == "check":
            r = U.check_dimensions(case["lhs"], case["rhs"], case["units"], values=case["values"])
            status, where, warnings = r.status.value, f"{r.location} | {r.detail}", r.warnings
            has_cx = r.counterexample is not None
        else:
            nu = U.units_from_names(f"({case['lhs']}) - ({case['rhs']})")
            if case["lhs"] in nu.units:
                r = U.check_dimensions(case["lhs"], case["rhs"], nu, values="numbers")
                status, where, warnings = r.status.value, f"{r.location} | {r.detail}", r.warnings
                has_cx = r.counterexample is not None
            else:
                try:
                    rep = U.dimension_of(case["rhs"], nu, values="numbers")
                    status, where, warnings, has_cx = "verified", "", rep.warnings, False
                    if rep.dimension is None and rep.unknown:
                        status = "undecided"
                except U.DimensionError as exc:
                    status, where, warnings = "refuted", f"{exc.location} | {exc}", ()
                    has_cx = exc.witness is not None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", False
    false_ref = status == "refuted" and expected != "refuted"
    ok = status == expected
    if ok and status == "refuted":
        ok = has_cx and (case.get("location") is None or case["location"] in where)
    if ok:
        ok = _warned(warnings, case.get("warning"))
    return ok, f"{status} [{where}]", false_ref


def main(split: str = "all") -> int:
    cases = load(split)
    by: Counter[tuple[str, str]] = Counter()
    passed: Counter[tuple[str, str]] = Counter()
    false_refuted = 0
    refuted_with_location = refuted = 0
    for c in cases:
        ok, observed, fr = evaluate(c)
        key = (c["split"], c["kind"])
        by[key] += 1
        passed[key] += ok
        false_refuted += fr
        if observed.startswith("refuted"):
            refuted += 1
            refuted_with_location += "None |" not in observed
        if not ok:
            print(f"FAIL {c['id']} {c['kind']}: {c.get('q') or c.get('lhs')} -> "
                  f"{c.get('to') or c.get('rhs')}: expected {c.get('expected') or c.get('error')}, "
                  f"got {observed}")
    print("\n| split | kind | passed |\n|---|---|---|")
    for key in sorted(by):
        print(f"| {key[0]} | {key[1]} | {passed[key]}/{by[key]} |")
    print(f"| **{split}** | **all** | **{sum(passed.values())}/{sum(by.values())}** |")
    print(f"\nfalse REFUTED: {false_refuted}; REFUTED with a location: "
          f"{refuted_with_location}/{refuted}")
    return 0 if sum(passed.values()) == sum(by.values()) and false_refuted == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "dev"))
