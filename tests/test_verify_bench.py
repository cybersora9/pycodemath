"""The VERIFY benchmark (V6): its data are reproducible and its scoring is honest.

The full run is a separate command (``python -m pycodemath.bench.verify_bench``,
minutes); the gate only checks that the committed data set IS the generator's
output, that every injected error is real (by an evaluation that does not go
through ``pycodemath.verify``), that UNDECIDED never scores as a catch, and one
smoke run of 10 derivations — twice, for determinism. The smoke set avoids every
``boundary`` case: those are the ones that can spend the budget, and a budget that
fires in the gate is finding A (see ``tests/test_verify_fuzz.py``).
"""

from __future__ import annotations

import json
from dataclasses import replace

import mpmath
import pytest
import sympy as sp

from pycodemath.bench import verify_bench as vb
from pycodemath.frontend.parser import parse

X, Y = sp.symbols("x y")
#: Evaluation points of the ground-truth check: positive, rational, away from
#: the removable singularity of ``(kx)^n / x^m`` at 0.
_POINTS = [{X: sp.Rational(n, 7), Y: sp.Rational(n + 5, 11)} for n in (3, 9, 17, 26, 38)]

#: Smoke set: one pair each from five error types and all three derivation kinds.
_SMOKE = [
    "sign-01-ok", "sign-01-err",          # expressions, the plan's example
    "chain-01-ok", "chain-01-err",        # derivative: certify_diff + check_steps
    "constant-03-ok", "constant-03-err",  # equations, proportional steps
    "divide-02-ok", "divide-02-err",      # equations, a lost root
    "square-01-ok", "square-01-err",      # equations, a gained root
]


@pytest.fixture(scope="module")
def records() -> "list[dict]":
    return vb.load()


# --- the data set ------------------------------------------------------------
def test_committed_data_is_the_generators_output(records: "list[dict]") -> None:
    # A data file edited by hand (or a generator changed without --regenerate)
    # would make the report's numbers unreproducible.
    assert records == vb.generate()
    assert vb.generate() == vb.generate()
    assert vb.generate(seed=vb.SEED + 1) != records


def test_data_set_shape(records: "list[dict]") -> None:
    assert len(records) == 2 * len(vb.ERROR_TYPES) * vb.PAIRS_PER_TYPE == 210
    assert len({r["id"] for r in records}) == len(records)
    for etype in vb.ERROR_TYPES:
        mine = [r for r in records if r["error_type"] == etype]
        assert sum(r["label"] == "wrong" for r in mine) == vb.PAIRS_PER_TYPE
        assert sum(r["label"] == "correct" for r in mine) == vb.PAIRS_PER_TYPE
    for r in records:
        steps = len(r["steps"]) + (r["kind"] == "derivative")  # step 1 = d/dx f
        if r["label"] == "wrong":
            assert 2 <= r["error_step"] <= steps
        else:
            assert r["error_step"] is None and "witness" not in r
        assert r["boundary"] in (None, "transcendental", "cubic-irreducible", "quartic")


def _value(e: sp.Expr, point: dict) -> sp.Expr:
    # SymPy numbers, not complex(): exp(8x^3) at x = 38/7 overflows a double
    return sp.N(e.xreplace(point), 40)


def _equal(a: sp.Expr, b: sp.Expr) -> bool:
    return all(
        abs(_value(a, p) - _value(b, p)) <= sp.Float("1e-20") * (1 + abs(_value(a, p)))
        for p in _POINTS
    )


def _solves(step: str, v: sp.Expr) -> bool:
    for alternative in step.split(" or "):
        left, right = (parse(t).sy for t in alternative.split("="))
        if abs(_value(left, {X: v}) - _value(right, {X: v})) < sp.Float("1e-25"):
            return True
    return False


def test_every_injected_error_is_real_and_nothing_else_is(records: "list[dict]") -> None:
    """Ground truth by evaluation, not by ``pycodemath.verify``: in an expression
    or derivative chain exactly the injected step changes the value; in an
    equation chain the witness solves exactly one side of the injected step."""
    for r in records:
        if r["kind"] == "equations":
            if r["label"] == "correct":
                continue  # equivalence by construction; a false alarm shows as FP
            k, steps = r["error_step"], r["steps"]
            if "witness" in r:
                w = parse(r["witness"]["x"]).sy
            else:  # square/sqrt-quartic: the positive root of x^2 + sqrt(x) = a
                a = int(steps[0].rsplit("- ", 1)[1])
                with mpmath.workdps(45):
                    root = mpmath.findroot(lambda t: t**2 + mpmath.sqrt(t) - a, 1)
                w = sp.Float(root, 45)
            assert _solves(steps[k - 2], w) != _solves(steps[k - 1], w), r["id"]
            continue
        chain = [parse(t).sy for t in r["steps"]]
        if r["kind"] == "derivative":
            chain.insert(0, sp.diff(parse(r["function"]).sy, X))
        for n in range(2, len(chain) + 1):
            injected = r["label"] == "wrong" and n == r["error_step"]
            assert _equal(chain[n - 2], chain[n - 1]) != injected, (r["id"], n)


def test_sums_are_written_as_a_model_would() -> None:
    assert vb._sum((3, "x^2"), (-1, "x"), (2, "")) == "3*x^2 - x + 2"
    assert vb._sum((-2, "x"), (0, "")) == "-2*x"
    assert vb._sum((0, "x")) == "0"
    assert vb._q(vb.Fraction(-7, 3)) == "-7/3"


# --- scoring -----------------------------------------------------------------
def _fake(rid: str, label: str, status: str, first: "int | None" = None) -> "tuple[dict, vb.Outcome]":
    rec = {"id": rid, "label": label, "error_type": "sign", "template": "t",
           "boundary": None, "error_step": 2 if label == "wrong" else None}
    return rec, vb.Outcome(rid, status, first, ((2, status, "m"),), 0.01)


def test_undecided_is_never_a_catch() -> None:
    pairs = [
        _fake("a", "wrong", "refuted", 2),     # caught, right step
        _fake("b", "wrong", "refuted", 3),     # caught, wrong step
        _fake("c", "wrong", "undecided"),      # missed
        _fake("d", "wrong", "verified"),       # missed — a false proof
        _fake("e", "correct", "refuted", 2),   # false alarm
        _fake("f", "correct", "undecided"),
        _fake("g", "correct", "verified"),
        _fake("h", "correct", "verified"),
    ]
    s = vb.summarize([r for r, _ in pairs], [o for _, o in pairs])["overall"]
    assert (s["tp"], s["fp"], s["wrong"], s["correct"]) == (2, 1, 4, 4)
    assert s["recall"] == 2 / 4  # UNDECIDED and VERIFIED both count as misses
    assert s["precision"] == 2 / 3
    assert s["step_accuracy"] == 1 / 2
    assert (s["wrong_verified"], s["wrong_undecided"]) == (1, 1)
    assert (s["correct_verified"], s["correct_undecided"]) == (2, 1)
    assert s["undecided_rate"] == 2 / 8


# --- the smoke run -----------------------------------------------------------
def test_smoke_run_is_deterministic(records: "list[dict]") -> None:
    by_id = {r["id"]: r for r in records}
    subset = [by_id[i] for i in _SMOKE]
    assert all(r["boundary"] is None for r in subset)
    first = vb.run(subset)
    second = vb.run(subset)
    strip = [replace(o, seconds=0.0) for o in first]
    assert strip == [replace(o, seconds=0.0) for o in second]
    assert all(o.status in ("verified", "refuted", "undecided") for o in first)
    assert not any(m == "time-budget" for o in first for _, _, m in o.steps)
    s = vb.summarize(subset, first)["overall"]
    assert s["derivations"] == 10 and s["wrong"] == s["correct"] == 5
    assert s["tp"] + s["wrong_verified"] + s["wrong_undecided"] + s["wrong_error"] == 5
    assert json.loads(json.dumps(s))  # the --json output is plain JSON
