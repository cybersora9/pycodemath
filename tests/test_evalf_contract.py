"""``Expr.evalf`` / ``Expr.compiled`` back inside the typed-failure contract (VERIFY V0).

Finding B of the v0.4 review (``dev-journal/PROMPT_V04_NEXTGEN.md`` §3): the only
``try`` in ``evalf`` wrapped ``float(result)``, so a point that is undefined BELOW
the top level of the expression escaped from inside SymPy as a raw exception.
Measured before the fix on 3000 fuzzed expressions (seed 1, points x=y=0.5 and
x=y=0): 2 raw TypeError + 98 raw ZeroDivisionError from ``evalf``, 65 silent
``nan`` returns, and 86 raw printer errors from building ``compiled``.

Independent source for every expected outcome: the mathematics of the input
(the point is a pole, or the expression is ``zoo``/``nan`` before any number is
substituted), not what pycodemath happens to return today. ``subs`` is the
second path: it returns the same fact symbolically, and must keep doing so.
"""

from __future__ import annotations

import math
import random

import pytest
import sympy as sp

from pycodemath import parse
from pycodemath.core.errors import DomainError, PycodemathError


# --- regression: the exact inputs the probe and the fuzz found -------------

@pytest.mark.parametrize(
    "source, values",
    [
        # finding B as the probe (dev-journal/sondy_v04) prints it: zoo nested
        # under cos — evalf_trig unpacks ComplexInfinity -> TypeError
        ("exp(cos(y/0))", {"y": 0.5}),
        ("cos(y/0)", {"y": 0.5}),
        # the same shape the fuzz reached through tan(0) = 0 in a denominator
        ("sin(((y**pi)/tan(0)))", {"y": 0.5}),
        # a pole hit by the substituted value — evalf_pow divides by an mpf
        # zero -> ZeroDivisionError from mpmath; the most ordinary input there is
        ("1/x", {"x": 0.0}),
        ("(((pi-2)/pi)-sqrt((1/x)))", {"x": 0.0}),
        ("(asin(sqrt(2))/y)", {"y": 0.0}),
    ],
)
def test_undefined_point_below_the_top_is_a_domain_error(source, values):
    with pytest.raises(DomainError):
        parse(source).evalf(**values)


@pytest.mark.parametrize(
    "source, values",
    [
        ("sin(1/0)", {}),
        ("0/0", {}),
        ("exp(Abs((0/0)))", {}),
        ("(0/tan((0/2)))", {}),
        # nan fed in by the caller is not a number to evaluate at either
        ("sin(x)", {"x": float("nan")}),
    ],
)
def test_nan_is_refused_not_returned(source, values):
    # float(sympy.nan) passes through silently; before the fix these returned
    # nan while 1/0 — the same undefined point, zoo at the top — refused
    with pytest.raises(DomainError):
        parse(source).evalf(**values)


def test_top_level_zoo_still_refuses_as_before():
    # the four inputs the probe marks as already correct stay correct
    for source, values in [("y/0", {"y": 1.0}), ("1/0", {}), ("log(0)", {}),
                           ("x/(x-x)", {"x": 2.0})]:
        with pytest.raises(DomainError):
            parse(source).evalf(**values)


def test_infinity_is_a_real_limit_and_is_kept():
    # |1/0| is +oo, a signed real limit, not an undefined value — the fix
    # refuses nan, not inf
    assert parse("Abs(1/0)").evalf() == math.inf
    assert parse("x").evalf(x=float("inf")) == math.inf


def test_defined_points_are_unchanged():
    assert parse("1/x").evalf(x=4.0) == 0.25
    assert parse("exp(cos(y))").evalf(y=0.0) == pytest.approx(math.e)


def test_subs_and_evalf_give_one_answer():
    # subs is the symbolic path: zoo / nan are SymPy values and it returns them;
    # evalf promises a real float and refuses the same point. One mathematical
    # fact (undefined here), two representations fixed by each method's contract.
    for source, values in [("exp(cos(y/0))", {"y": 0.5}), ("1/x", {"x": 0.0}),
                           ("sin(1/0)", {})]:
        e = parse(source)
        symbolic = e.subs(values).sy
        assert symbolic.has(sp.zoo, sp.nan)
        with pytest.raises(DomainError):
            e.evalf(**values)


# --- compiled: building refuses typed, calling keeps its documented contract --

@pytest.mark.parametrize("source", ["y/0", "exp(cos(y/0))", "(atan((-1/0))**E)", "sin(1/0)"])
def test_compiling_an_undefined_expression_is_a_domain_error(source):
    # before: KeyError('ComplexInfinity') / PrintMethodNotImplementedError
    # (AccumBounds) from the NumPy printer
    e = parse(source)
    with pytest.raises(DomainError):
        e.compiled(e.symbol_names())


def test_calling_a_compiled_function_keeps_plain_domain_exceptions():
    # the hot-loop contract (Expr.compiled docstring): math's ValueError, turned
    # into data by engine.numerics._DOMAIN_ERRORS — not changed by this module
    fn = parse("log(x)").compiled(["x"])
    with pytest.raises(ValueError):
        fn(-1.0)


# --- fuzz: nothing outside the PycodemathError hierarchy ------------------

_ATOMS = ["x", "y", "1", "2", "0", "-1", "1/2", "pi", "E", "sqrt(2)"]
_UNARY = ["sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "sign", "asin", "atan"]
_BINARY = ["+", "-", "*", "/", "**"]


def _expression(rng: random.Random, depth: int = 0) -> str:
    # the probe's generator (dev-journal/sondy_v04/odtworz_znaleziska.py) on a
    # private Random instance, so the sequence does not depend on test order
    if depth > 2 or rng.random() < 0.32:
        return rng.choice(_ATOMS)
    if rng.random() < 0.42:
        return f"{rng.choice(_UNARY)}({_expression(rng, depth + 1)})"
    return f"({_expression(rng, depth + 1)}{rng.choice(_BINARY)}{_expression(rng, depth + 1)})"


def test_fuzz_evalf_and_compiled_fail_only_with_typed_errors():
    # 400 expressions, the probe's seed; x=y=0.5 is where the probe looks and
    # x=y=0 is where poles live. Before the fix this seed gave 4 raw TypeError,
    # 13 raw ZeroDivisionError, 11 raw KeyError and 8 nan returns; after: 0.
    # ~4 s.
    rng = random.Random(20260906)
    raw: list[tuple[str, str, str]] = []
    nans: list[tuple[str, str]] = []
    for _ in range(400):
        source = _expression(rng)
        e = parse(source)
        for label, op in [
            ("evalf@0.5", lambda: e.evalf(x=0.5, y=0.5)),
            ("evalf@0", lambda: e.evalf(x=0.0, y=0.0)),
            ("compiled", lambda: e.compiled(e.symbol_names())),
        ]:
            try:
                result = op()
            except PycodemathError:
                continue
            except Exception as exc:  # the contract under test
                raw.append((label, source, type(exc).__name__))
                continue
            if isinstance(result, float) and math.isnan(result):
                nans.append((label, source))
    assert raw == []
    assert nans == []
