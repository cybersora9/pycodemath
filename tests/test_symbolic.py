"""Round 2 tests (Block A): contract leaks in the symbolic engine.

Contract: boundaries always raise PycodemathError, never a raw SymPy
exception; the result contains zero junk (unevaluated Integral/Sum).
Expectations verified against bare sympy (python -c), not via pycodemath:
- sp.solve(sin(x)-x, x) -> NotImplementedError ("multiple generators")
- sp.integrate(x**x, x) -> unevaluated Integral(x**x, x)
"""

from __future__ import annotations

import pytest

from pycodemath import parse
from pycodemath.core.errors import PycodemathError
from pycodemath.engine import symbolic
from pycodemath.engine.symbolic import _point


# --- solve: raw sympy exception leak ---------------------------------------
def test_solve_unsolvable_raises_pycodemath():
    # bare sympy raises NotImplementedError here — the contract requires PycodemathError
    with pytest.raises(PycodemathError, match="solve"):
        symbolic.solve(parse("sin(x) - x"), "x")


def test_solve_happy_path_no_regression():
    roots = symbolic.solve(parse("x^2 - 9"), "x")
    vals = sorted(int(str(r)) for r in roots)
    assert vals == [-3, 3]


# --- solve: "real only" filter (round 3) -----------------------------------
def test_solve_real_filters_complex_roots():
    # x^2 + 1 = 0 has only complex roots (±i); real=True → empty
    assert symbolic.solve(parse("x^2 + 1"), "x", real=True) == []
    # and by default (real=False) complex roots pass through — no regression
    assert len(symbolic.solve(parse("x^2 + 1"), "x")) == 2


def test_solve_real_keeps_real_roots():
    # the filter does not touch real roots
    roots = symbolic.solve(parse("x^2 - 9"), "x", real=True)
    assert sorted(int(str(r)) for r in roots) == [-3, 3]


def test_solve_real_mixed_roots_keeps_only_real():
    # x^3 - x = 0 → {-1, 0, 1}, all real; real=True cuts nothing
    roots = symbolic.solve(parse("x^3 - x"), "x", real=True)
    assert sorted(int(str(r)) for r in roots) == [-1, 0, 1]


# --- integrate: an unevaluated Integral is junk in the result ---------------
def test_integrate_no_closed_form_raises():
    # bare sympy returns Integral(x**x, x) — for us it's a clear refusal
    with pytest.raises(PycodemathError, match="closed form"):
        symbolic.integrate(parse("x^x"), "x")


def test_integrate_happy_path_no_regression():
    # ∫ 2x dx = x^2 (closed form from tables, not from pycodemath)
    assert symbolic.integrate(parse("2*x"), "x").equivalent(parse("x^2"))


def test_integrate_via_expr_api_guarded():
    # the guard sits at the source (Expr.integrate), so it also catches the direct API
    with pytest.raises(PycodemathError, match="closed form"):
        parse("x^x").integrate("x")


# --- _point: a bool must not pass as an int ---------------------------------
def test_point_rejects_bool():
    with pytest.raises(PycodemathError, match="point/bound"):
        _point(True)
    with pytest.raises(PycodemathError, match="point/bound"):
        _point(False)


def test_limit_rejects_bool_point():
    with pytest.raises(PycodemathError):
        symbolic.limit(parse("1/x"), "x", True)


def test_point_still_accepts_numbers():
    # 0 and 0.5 pass as before (regression test for the bool guard)
    assert str(_point(0)) == "0"
    assert float(_point(0.5)) == 0.5
