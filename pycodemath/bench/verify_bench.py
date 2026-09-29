"""VERIFY benchmark — how often does ``check_steps`` catch a model's typical slip?

The data are derivations of the kind a language model writes, generated from a
seed (so every run of every machine sees the same ~200): for each of seven error
types, PAIRS of one correct derivation and its twin with one injected mistake —

========== ==============================================================
``sign``    a lost minus (``-(x-b)^2`` expanded with ``+b^2``, ``ax+b=c ->
            ax=c+b``, ``cos 2x = 1 + 2 sin^2 x``)
``chain``   a wrong chain rule (the inner derivative forgotten, half-done,
            or replaced by the inner function)
``power``   a wrong power (``(x^a)^b = x^(a+b)``, ``d/dx x^-n = -n x^(-n+1)``,
            ``(kx)^n = k x^n``)
``constant`` a lost constant (a dropped ``ab`` in ``(x+a)(x+b)``, a factor
            ``k`` lost in the product rule, ``a(x+b) = ax + b``)
``divide``  dividing by an expression that can be 0 (``x^2 = ax -> x = a``)
``square``  squaring both sides, which adds a root
``formula`` a wrong short multiplication formula (``(a+b)^2 = a^2+b^2``)
========== ==============================================================

After the injected step the wrong derivation CARRIES the mistake forward, as a
model does, so the injected step is the only one a checker may reject. The
correct twin shares the template (and, where the mathematics allows, the
parameters); for ``square`` the twin squares legitimately (both sides known
non-negative), so it exercises the same move without the error.

Nothing is selected by how the verifier does: the templates were fixed before
the first measured run, every generated derivation is scored, and the ones that
touch a known limit of the verifier carry a ``boundary`` tag (``transcendental``
— a ``solveset`` ConditionSet; ``cubic-irreducible`` — three real roots only
reachable through complex radicals; ``quartic`` — Ferrari radicals) so a report
can show the numbers with and without them. They are not removed.

HOW A DERIVATION IS CHECKED — public API only, semantics of ``verify`` untouched:
``expressions`` / ``equations`` go through ``check_steps``; a ``derivative``
(``d/dx f = g1 = g2``, as the parser has no ``diff``) checks step 2 with
``certify_diff(f, x, g1)`` and the rest with ``check_steps([g1, g2, ...])``,
renumbered. ``budget`` is the allowance for the whole derivation.

SCORING. A derivation is flagged when the verdict is REFUTED. Recall counts every
wrong derivation — an UNDECIDED one is a miss, never a hit; precision is over
REFUTED verdicts; step accuracy is the share of caught errors whose
``first_error`` is the injected step. UNDECIDED is reported on its own, split by
label, and a wrong derivation VERIFIED (a false proof) is counted separately.

    python -m pycodemath.bench.verify_bench               # full run, text report
    python -m pycodemath.bench.verify_bench --json out.json
    python -m pycodemath.bench.verify_bench --seed 7      # control run, fresh data
    python -m pycodemath.bench.verify_bench --regenerate  # rewrite the data file

Timing is ``time.perf_counter`` (``time.monotonic`` ticks every 15.6 ms on
Windows, which would round most derivations to 0 or 16 ms).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..core.errors import PycodemathError
from ..verify import VerdictStatus, certify_diff, check_steps

__all__ = [
    "ERROR_TYPES",
    "SEED",
    "PAIRS_PER_TYPE",
    "DATA",
    "BUDGET",
    "generate",
    "load",
    "write",
    "run_one",
    "run",
    "summarize",
    "Outcome",
]

#: The seed of the committed data set. Changing it changes every number.
SEED = 20260929
#: Pairs (correct + wrong) per error type: 7 x 15 x 2 = 210 derivations.
PAIRS_PER_TYPE = 15
#: Allowance for ONE derivation, seconds. Not the library default (120 s): a
#: full run would then be bounded by 210 x 120 s. Budget-stopped steps are
#: UNDECIDED with ``method="time-budget"`` and counted as such.
BUDGET = 30.0
#: The committed data set, next to this module.
DATA = Path(__file__).with_name("data") / "derivations.jsonl"

ERROR_TYPES = ("sign", "chain", "power", "constant", "divide", "square", "formula")


# --- writing numbers and sums ------------------------------------------------
def _q(v: "Fraction | int") -> str:
    """An exact rational as the parser reads it back: ``5``, ``-5``, ``7/3``."""
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _sum(*terms: "tuple[Fraction | int, str]") -> str:
    """``coef*mono`` terms joined with their signs; zero terms dropped.

    ``_sum((3, "x^2"), (-1, "x"), (2, ""))`` -> ``"3*x^2 - x + 2"``.
    """
    out: "list[str]" = []
    for coef, mono in terms:
        c = Fraction(coef)
        if c == 0:
            continue
        mag = abs(c)
        if not mono:
            body = _q(mag)
        elif mag == 1:
            body = mono
        else:
            body = f"{_q(mag)}*{mono}"
        if not out:
            out.append(body if c > 0 else f"-{body}")
        else:
            out.append(("+ " if c > 0 else "- ") + body)
    return " ".join(out) if out else "0"


def _pow(n: int, name: str = "x") -> str:
    """``x^n``, with ``x^1`` written ``x``."""
    return name if n == 1 else f"{name}^{n}"


def _shift(v: int, name: str = "x") -> str:
    """``x + v`` / ``x - |v|`` / ``x`` — a monic linear factor."""
    return _sum((1, name), (v, ""))


# --- templates ---------------------------------------------------------------
_SAME = "same as the wrong twin"


@dataclass
class _Pair:
    """One template instance: the correct derivation and its wrong twin."""

    template: str
    kind: str
    correct: "list[str]"
    wrong: "list[str]"
    error_step: int
    boundary: "str | None" = None
    #: for ``derivative``: the differentiated function (step 1)
    function: "str | None" = None
    #: for ``equations``: a value solving exactly one of the two steps around
    #: the injected error — the ground truth that the error is real
    witness: "str | None" = None
    #: the correct twin's boundary when its parameters differ (``square``)
    correct_boundary: "str | None" = _SAME


_Rng = random.Random
_Template = Callable[[_Rng, int], _Pair]


# sign ------------------------------------------------------------------------
def _sign_squares(rng: _Rng, i: int) -> _Pair:
    """``(x+a)^2 - (x-b)^2`` expanded with ``+b^2``: the plan's own example."""
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    start = f"(x + {a})^2 - (x - {b})^2"
    ok = [
        start,
        f"x^2 + {2 * a}*x + {a * a} - x^2 + {2 * b}*x - {b * b}",
        _sum((2 * a + 2 * b, "x"), (a * a - b * b, "")),
    ]
    bad = [
        start,
        f"x^2 + {2 * a}*x + {a * a} - x^2 + {2 * b}*x + {b * b}",
        _sum((2 * a + 2 * b, "x"), (a * a + b * b, "")),
    ]
    step = 2
    if rng.random() < 0.5:
        # a harmless first rewrite, so the slip sits at step 3
        pre = f"(x + {a})*(x + {a}) - (x - {b})*(x - {b})"
        ok.insert(1, pre)
        bad.insert(1, pre)
        step = 3
    return _Pair("sign/squares", "expressions", ok, bad, step)


def _sign_linear(rng: _Rng, i: int) -> _Pair:
    """``a*x + b = c -> a*x = c + b``: a term moved without its sign flipped."""
    a, b, c = rng.randint(2, 9), rng.randint(1, 9), rng.randint(1, 20)
    if rng.random() < 0.5:
        start = f"{a}*x + {b} = {c}"
        right, slip = Fraction(c - b), Fraction(c + b)
    else:
        start = f"{a}*x - {b} = {c}"
        right, slip = Fraction(c + b), Fraction(c - b)
    ok = [start, f"{a}*x = {_q(right)}", f"x = {_q(right / a)}"]
    bad = [start, f"{a}*x = {_q(slip)}", f"x = {_q(slip / a)}"]
    return _Pair("sign/linear", "equations", ok, bad, 2, witness=_q(right / a))


def _sign_trig(rng: _Rng, i: int) -> _Pair:
    """``cos 2x`` through the wrong-signed double-angle formula."""
    a = rng.randint(3, 9)
    if rng.random() < 0.5:
        start = f"cos(2*x) + {a}*sin(x)^2"
        ok = [start, f"1 - 2*sin(x)^2 + {a}*sin(x)^2", f"1 + {a - 2}*sin(x)^2"]
        bad = [start, f"1 + 2*sin(x)^2 + {a}*sin(x)^2", f"1 + {a + 2}*sin(x)^2"]
    else:
        start = f"cos(2*x) + {a}*cos(x)^2"
        ok = [start, f"2*cos(x)^2 - 1 + {a}*cos(x)^2", f"{a + 2}*cos(x)^2 - 1"]
        bad = [start, f"2*cos(x)^2 + 1 + {a}*cos(x)^2", f"{a + 2}*cos(x)^2 + 1"]
    return _Pair("sign/trig", "expressions", ok, bad, 2)


# chain -----------------------------------------------------------------------
def _chain_sin(rng: _Rng, i: int) -> _Pair:
    """``d/dx sin(ax^2+b)``: inner derivative half-done or replaced by the inner."""
    a, b = rng.randint(2, 9), rng.randint(1, 9)
    u = f"{a}*x^2 + {b}"
    f = f"sin({u})"
    ok = [f"cos({u})*({2 * a}*x)", f"{2 * a}*x*cos({u})"]
    if rng.random() < 0.5:  # d/dx (a x^2) = 2x — the coefficient forgotten
        bad = [f"cos({u})*(2*x)", f"2*x*cos({u})"]
    else:  # multiplied by the inner function instead of its derivative
        bad = [f"cos({u})*({u})", f"({u})*cos({u})"]
    return _Pair("chain/sin", "derivative", ok, bad, 2, function=f)


def _chain_power(rng: _Rng, i: int) -> _Pair:
    """``d/dx (bx+c)^n = n (bx+c)^(n-1)``: the inner derivative ``b`` forgotten."""
    b, c, n = rng.randint(2, 9), rng.randint(1, 9), rng.randint(3, 7)
    f = f"({b}*x + {c})^{n}"
    ok = [f"{n}*({b}*x + {c})^{n - 1}*{b}", f"{n * b}*({b}*x + {c})^{n - 1}"]
    # already as simple as it gets: the wrong derivation is one step
    bad = [f"{n}*({b}*x + {c})^{n - 1}"]
    return _Pair("chain/power", "derivative", ok, bad, 2, function=f)


def _chain_log_exp(rng: _Rng, i: int) -> _Pair:
    """``d/dx log(ax^2+b)`` without the inner derivative / ``exp(ax^3)`` with the inner."""
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    if rng.random() < 0.5:
        u = f"{a}*x^2 + {b}" if a > 1 else f"x^2 + {b}"
        f = f"log({u})"
        ok = [f"1/({u})*({2 * a}*x)", f"{2 * a}*x/({u})"]
        bad = [f"1/({u})"]
    else:
        u = f"{a}*x^3" if a > 1 else "x^3"
        f = f"exp({u})"
        ok = [f"exp({u})*({3 * a}*x^2)", f"{3 * a}*x^2*exp({u})"]
        bad = [f"exp({u})*({u})", f"{a}*x^3*exp({u})" if a > 1 else f"x^3*exp({u})"]
    return _Pair("chain/log-exp", "derivative", ok, bad, 2, function=f)


# power -----------------------------------------------------------------------
def _power_tower(rng: _Rng, i: int) -> _Pair:
    """``(x^a)^b = x^(a+b)``."""
    while True:
        a, b = rng.randint(2, 5), rng.randint(2, 5)
        if a + b != a * b:
            break
    c = rng.randint(1, 5)
    start = f"(x^{a})^{b}*x^{c}"
    ok = [start, f"x^{a * b}*x^{c}", f"x^{a * b + c}"]
    bad = [start, f"x^{a + b}*x^{c}", f"x^{a + b + c}"]
    return _Pair("power/tower", "expressions", ok, bad, 2)


def _power_rule(rng: _Rng, i: int) -> _Pair:
    """``d/dx k/x^n = -nk/x^(n-1)``: the exponent moved the wrong way."""
    k, n = rng.randint(1, 9), rng.randint(2, 5)
    f = f"{k}/x^{n}"
    ok = [f"-{n}*{k}*x^(-{n} - 1)", f"-{n * k}/x^{n + 1}"]
    bad = [f"-{n}*{k}*x^(-{n} + 1)", f"-{n * k}/{_pow(n - 1)}"]
    return _Pair("power/rule", "derivative", ok, bad, 2, function=f)


def _power_coefficient(rng: _Rng, i: int) -> _Pair:
    """``(kx)^n = k x^n``: the power not applied to the coefficient."""
    k, n = rng.randint(2, 5), rng.randint(2, 4)
    m = rng.randint(1, n - 1)
    start = f"({k}*x)^{n}/{_pow(m)}"
    ok = [start, f"{k ** n}*x^{n}/{_pow(m)}", f"{k ** n}*{_pow(n - m)}"]
    bad = [start, f"{k}*x^{n}/{_pow(m)}", f"{k}*{_pow(n - m)}"]
    return _Pair("power/coefficient", "expressions", ok, bad, 2)


# constant --------------------------------------------------------------------
def _constant_product(rng: _Rng, i: int) -> _Pair:
    """``(x+a)(x+b)`` expanded without ``ab`` — at the expansion or at collecting."""
    a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
    start = f"(x + {a})*(x + {b}) + {c}*x"
    s2 = f"x^2 + {a}*x + {b}*x + {a * b} + {c}*x"
    s3 = _sum((1, "x^2"), (a + b + c, "x"), (a * b, ""))
    s3_bad = _sum((1, "x^2"), (a + b + c, "x"))
    ok = [start, s2, s3]
    if rng.random() < 0.5:
        bad, step = [start, f"x^2 + {a}*x + {b}*x + {c}*x", s3_bad], 2
    else:
        bad, step = [start, s2, s3_bad], 3
    return _Pair("constant/product", "expressions", ok, bad, step)


def _constant_product_rule(rng: _Rng, i: int) -> _Pair:
    """``d/dx k x e^x`` (or ``k x sin x``) with ``k`` lost in one term."""
    k = rng.randint(2, 9)
    if rng.random() < 0.5:
        f = f"{k}*x*exp(x)"
        ok = [f"{k}*exp(x) + {k}*x*exp(x)", f"{k}*(x + 1)*exp(x)"]
        bad = [f"exp(x) + {k}*x*exp(x)", f"({k}*x + 1)*exp(x)"]
    else:
        f = f"{k}*x*sin(x)"
        ok = [f"{k}*sin(x) + {k}*x*cos(x)", f"{k}*(sin(x) + x*cos(x))"]
        bad = [f"sin(x) + {k}*x*cos(x)"]
    return _Pair("constant/product-rule", "derivative", ok, bad, 2, function=f)


def _constant_distribute(rng: _Rng, i: int) -> _Pair:
    """``a(x + b) = c -> a*x + b = c``: the factor not carried to the constant."""
    a, b, c = rng.randint(2, 9), rng.randint(1, 9), rng.randint(1, 30)
    start = f"{a}*(x + {b}) = {c}"
    ok = [start, f"{a}*x + {a * b} = {c}", f"x = {_q(Fraction(c - a * b, a))}"]
    bad = [start, f"{a}*x + {b} = {c}", f"x = {_q(Fraction(c - b, a))}"]
    return _Pair(
        "constant/distribute", "equations", ok, bad, 2,
        witness=_q(Fraction(c - a * b, a)),
    )


# divide ----------------------------------------------------------------------
#: ``x^3 - q x + p`` irreducible over Q with three real roots (discriminant
#: ``4q^3 - 27p^2 > 0``, no rational root) — the casus irreducibilis.
_IRREDUCIBLE_CUBICS = ((3, 1), (4, 1), (5, 1), (6, 3), (7, 7))


def _divide_polynomial(rng: _Rng, i: int) -> _Pair:
    """``x^2 = ax -> x = a``, ``x^3 = a^2 x``, ``x^4 + px = qx^2``: root 0 lost."""
    variant = i % 3
    if variant == 0:
        a = rng.randint(1, 9)
        start = f"x^2 = {a}*x"
        ok = [start, f"x^2 - {a}*x = 0", f"x*(x - {a}) = 0", f"x = 0 or x = {a}"]
        bad = [start, f"x = {a}"]
        return _Pair("divide/polynomial", "equations", ok, bad, 2, witness="0")
    if variant == 1:
        a = rng.randint(1, 6)
        start = f"x^3 = {a * a}*x"
        ok = [start, f"x*(x - {a})*(x + {a}) = 0", f"x = 0 or x = {a} or x = -{a}"]
        bad = [start, f"x^2 = {a * a}", f"x = {a} or x = -{a}"]
        return _Pair("divide/polynomial", "equations", ok, bad, 2, witness="0")
    q, p = _IRREDUCIBLE_CUBICS[rng.randrange(len(_IRREDUCIBLE_CUBICS))]
    start = f"x^4 + {p}*x = {q}*x^2" if p > 1 else f"x^4 + x = {q}*x^2"
    cubic = f"x^3 - {q}*x + {p}"
    ok = [start, f"x*({cubic}) = 0", f"x = 0 or {cubic} = 0"]
    bad = [start, f"x^3 + {p} = {q}*x", f"{cubic} = 0"]
    return _Pair(
        "divide/polynomial", "equations", ok, bad, 2,
        boundary="cubic-irreducible", witness="0",
    )


def _divide_common_factor(rng: _Rng, i: int) -> _Pair:
    """``(x-a)(x+b) = c(x-a) -> x + b = c``: the root ``x = a`` lost."""
    while True:
        a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
        if c - b != a:
            break
    start = f"(x - {a})*(x + {b}) = {c}*(x - {a})"
    ok = [
        start,
        f"(x - {a})*(x + {b}) - {c}*(x - {a}) = 0",
        f"(x - {a})*({_shift(b - c)}) = 0",
        f"x = {a} or x = {c - b}",
    ]
    bad = [start, f"x + {b} = {c}", f"x = {c - b}"]
    return _Pair("divide/common-factor", "equations", ok, bad, 2, witness=str(a))


def _divide_transcendental(rng: _Rng, i: int) -> _Pair:
    """``x e^x = ax -> e^x = a``; ``x e^x = x(x+a) -> e^x = x + a`` (ConditionSet)."""
    a = rng.randint(2, 9)
    if i % 2 == 0:
        start = f"x*exp(x) = {a}*x"
        ok = [start, f"x*(exp(x) - {a}) = 0", f"x = 0 or x = log({a})"]
        bad = [start, f"exp(x) = {a}", f"x = log({a})"]
        return _Pair("divide/transcendental", "equations", ok, bad, 2, witness="0")
    start = f"x*exp(x) = x*(x + {a})"
    ok = [start, f"x*(exp(x) - x - {a}) = 0", f"x = 0 or exp(x) = x + {a}"]
    bad = [start, f"exp(x) = x + {a}"]
    return _Pair(
        "divide/transcendental", "equations", ok, bad, 2,
        boundary="transcendental", witness="0",
    )


# square ----------------------------------------------------------------------
def _square_sqrt_linear(rng: _Rng, i: int) -> _Pair:
    """``sqrt(x + p) = x - q`` squared: the root with ``x < q`` is extraneous.

    The squared equation has roots ``r1 + r2 = 2q + 1``. The wrong derivation
    keeps both with ``r2 < q`` (so ``r1 >= q + 2``); the twin is the case ``r1 = q, r2 = q + 1``,
    where both roots are genuine and squaring changes nothing.
    """
    q = rng.randint(1, 6)
    # r1 = q + 1 would give r2 = q, a GENUINE root (x - q = 0): no error at all
    r1 = rng.randint(q + 2, q + 5)
    r2 = 2 * q + 1 - r1

    def chain(p: int, s: int, t: int) -> "list[str]":
        return [
            f"sqrt({_shift(p)}) = x - {q}",
            f"{_shift(p)} = (x - {q})^2",
            f"{_sum((1, 'x^2'), (-(2 * q + 1), 'x'), (q * q - p, ''))} = 0",
            f"x = {s} or x = {t}",
        ]

    bad = chain(q * q - r1 * r2, r1, r2)
    ok = chain(-q, q, q + 1)
    return _Pair("square/sqrt-linear", "equations", ok, bad, 2, witness=str(r2))


def _square_sqrt_quadratic(rng: _Rng, i: int) -> _Pair:
    """``sqrt(x^2 + p) = x - q`` has no solution; squared, it has one.

    With ``p, q > 0``: ``sqrt(x^2+p) > |x| >= x > x - q`` — no root, while the
    squared, linear equation has ``x = (q^2 - p) / 2q``. The twin ``= x + q``
    has that root genuinely (``x + q = (p + q^2) / 2q > 0``).
    """
    p, q = rng.randint(1, 20), rng.randint(1, 6)

    def chain(sign: int) -> "list[str]":
        r = Fraction(q * q - p, 2 * q) * (-sign)
        rhs = f"x + {q}" if sign > 0 else f"x - {q}"
        return [
            f"sqrt(x^2 + {p}) = {rhs}",
            f"x^2 + {p} = ({rhs})^2",
            f"{_sum((2 * q * sign, 'x'))} = {_q(p - q * q)}",
            f"x = {_q(r)}",
        ]

    bad = chain(-1)
    return _Pair(
        "square/sqrt-quadratic", "equations", chain(+1), bad, 2,
        witness=bad[-1].split("= ")[1],
    )


def _square_sqrt_quartic(rng: _Rng, i: int) -> _Pair:
    """``sqrt(x) = x^2 - a`` squared: a quartic whose root with ``x^2 < a`` is extraneous.

    ``x^2 + sqrt(x) = a`` has a positive solution for every ``a > 0``, and there
    ``x^2 - a < 0``, so squaring always gains it. The twin ``sqrt(x) = x^2 + a``
    squares an equation whose right side is positive: an equivalence.
    """
    a = rng.randint(1, 6)

    def chain(s: int) -> "list[str]":
        rhs = f"x^2 + {a}" if s > 0 else f"x^2 - {a}"
        return [
            f"sqrt(x) = {rhs}",
            f"x = ({rhs})^2",
            f"{_sum((1, 'x^4'), (2 * a * s, 'x^2'), (-1, 'x'), (a * a, ''))} = 0",
        ]

    bad = chain(-1)
    return _Pair(
        "square/sqrt-quartic", "equations", chain(+1), bad, 2,
        boundary=_quartic_boundary(-a), correct_boundary=_quartic_boundary(a),
        witness=None,  # an algebraic number; checked numerically by the tests
    )


def _quartic_boundary(a: int) -> "str | None":
    """``quartic`` when ``x^4 + 2a x^2 - x + a^2`` is irreducible over Q, else
    ``cubic-irreducible`` for an irreducible cubic factor with three real roots."""
    import sympy as sp

    x = sp.Symbol("x")
    degrees = [
        (sp.degree(f, x), f)
        for f, _ in sp.factor_list(x**4 + 2 * a * x**2 - x + a * a)[1]
    ]
    if any(d == 4 for d, _ in degrees):
        return "quartic"
    for d, f in degrees:
        if d == 3 and sp.discriminant(f, x) > 0:
            return "cubic-irreducible"
    return None


# formula ---------------------------------------------------------------------
def _formula_two_squares(rng: _Rng, i: int) -> _Pair:
    """``(px+qy)^2 - (px-qy)^2`` with ``(a-b)^2 = a^2 - b^2``."""
    p, q = rng.randint(1, 5), rng.randint(1, 5)
    px = f"{p}*x" if p > 1 else "x"
    qy = f"{q}*y" if q > 1 else "y"
    start = f"({px} + {qy})^2 - ({px} - {qy})^2"
    ok = [
        start,
        f"{_sum((p * p, 'x^2'), (2 * p * q, 'x*y'), (q * q, 'y^2'))} "
        f"- ({_sum((p * p, 'x^2'), (-2 * p * q, 'x*y'), (q * q, 'y^2'))})",
        _sum((4 * p * q, "x*y")),
    ]
    bad = [
        start,
        f"{_sum((p * p, 'x^2'), (2 * p * q, 'x*y'), (q * q, 'y^2'))} "
        f"- ({_sum((p * p, 'x^2'), (-q * q, 'y^2'))})",
        _sum((2 * p * q, "x*y"), (2 * q * q, "y^2")),
    ]
    step = 2
    if rng.random() < 0.5:
        pre = f"({px} + {qy})*({px} + {qy}) - ({px} - {qy})^2"
        ok.insert(1, pre)
        bad.insert(1, pre)
        step = 3
    return _Pair("formula/two-squares", "expressions", ok, bad, step)


def _formula_difference(rng: _Rng, i: int) -> _Pair:
    """``x^2 - a^2 = (x - a)^2``."""
    a = rng.randint(1, 9)
    start = f"x^2 - {a * a} + (x + {a})^2"
    ok = [start, f"(x - {a})*(x + {a}) + (x + {a})^2", f"2*x*(x + {a})"]
    bad = [start, f"(x - {a})^2 + (x + {a})^2", f"2*x^2 + {2 * a * a}"]
    return _Pair("formula/difference", "expressions", ok, bad, 2)


def _formula_cube(rng: _Rng, i: int) -> _Pair:
    """``(x - a)^3`` as ``x^3 - a^3`` or with the wrong sign pattern."""
    a = rng.randint(1, 6)
    start = f"(x - {a})^3 + {3 * a}*x^2"
    ok = [
        start,
        f"x^3 - {3 * a}*x^2 + {3 * a * a}*x - {a ** 3} + {3 * a}*x^2",
        f"x^3 + {3 * a * a}*x - {a ** 3}",
    ]
    if rng.random() < 0.5:
        bad = [start, f"x^3 - {a ** 3} + {3 * a}*x^2", f"x^3 + {3 * a}*x^2 - {a ** 3}"]
    else:
        bad = [
            start,
            f"x^3 - {3 * a}*x^2 - {3 * a * a}*x - {a ** 3} + {3 * a}*x^2",
            f"x^3 - {3 * a * a}*x - {a ** 3}",
        ]
    return _Pair("formula/cube", "expressions", ok, bad, 2)


TEMPLATES: "dict[str, tuple[_Template, ...]]" = {
    "sign": (_sign_squares, _sign_linear, _sign_trig),
    "chain": (_chain_sin, _chain_power, _chain_log_exp),
    "power": (_power_tower, _power_rule, _power_coefficient),
    "constant": (_constant_product, _constant_product_rule, _constant_distribute),
    "divide": (_divide_polynomial, _divide_common_factor, _divide_transcendental),
    "square": (_square_sqrt_linear, _square_sqrt_quadratic, _square_sqrt_quartic),
    "formula": (_formula_two_squares, _formula_difference, _formula_cube),
}


# --- the data set ------------------------------------------------------------
def generate(
    seed: int = SEED, pairs_per_type: int = PAIRS_PER_TYPE
) -> "list[dict[str, Any]]":
    """The derivations, as JSON-ready dicts — a pure function of the arguments.

    Templates of a type are used in turn (with 15 pairs and 3 templates, 5
    each). One ``random.Random`` per type, seeded from ``seed`` and the type's
    name, so adding a type does not reshuffle the others.
    """
    records: "list[dict[str, Any]]" = []
    for etype in ERROR_TYPES:
        rng = random.Random(f"{seed}/{etype}")
        templates = TEMPLATES[etype]
        for n in range(pairs_per_type):
            pair = templates[n % len(templates)](rng, n // len(templates))
            pid = f"{etype}-{n + 1:02d}"
            for label, steps in (("correct", pair.correct), ("wrong", pair.wrong)):
                wrong = label == "wrong"
                rec: "dict[str, Any]" = {
                    "id": f"{pid}-{'err' if wrong else 'ok'}",
                    "pair": pid,
                    "error_type": etype,
                    "template": pair.template,
                    "label": label,
                    "kind": pair.kind,
                    "steps": list(steps),
                    "error_step": pair.error_step if wrong else None,
                    "boundary": (
                        pair.boundary if wrong or pair.correct_boundary == _SAME
                        else pair.correct_boundary
                    ),
                }
                if pair.kind == "derivative":
                    rec["function"] = pair.function
                    rec["var"] = "x"
                if wrong and pair.witness is not None:
                    rec["witness"] = {"x": pair.witness}
                records.append(rec)
    return records


def write(records: "Sequence[dict[str, Any]]", path: Path = DATA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def load(path: Path = DATA) -> "list[dict[str, Any]]":
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- running -----------------------------------------------------------------
@dataclass(frozen=True)
class Outcome:
    """What the verifier said about one derivation.

    ``status`` is ``"verified"`` / ``"refuted"`` / ``"undecided"``, or
    ``"error"`` when the verifier raised (``error`` then names it). ``steps``
    holds ``[number, status, method]`` for every checked step.
    """

    id: str
    status: str
    first_error: "int | None"
    steps: "tuple[tuple[int, str, str], ...]"
    seconds: float
    error: "str | None" = None


def run_one(rec: "dict[str, Any]", budget: float = BUDGET) -> Outcome:
    """Check one derivation within ``budget`` seconds (see the module docstring)."""
    t0 = time.perf_counter()
    rows: "list[tuple[int, str, str]]" = []
    try:
        if rec["kind"] == "derivative":
            v = certify_diff(rec["function"], rec["var"], rec["steps"][0], budget=budget)
            rows.append((2, v.status.value, v.method))
            if len(rec["steps"]) > 1:
                left = max(budget - (time.perf_counter() - t0), 1e-3)
                r = check_steps(rec["steps"], budget=left)
                rows += [(c.number + 1, c.verdict.status.value, c.verdict.method) for c in r.checks]
        else:
            r = check_steps(rec["steps"], budget=budget)
            rows = [(c.number, c.verdict.status.value, c.verdict.method) for c in r.checks]
    except PycodemathError as exc:
        return Outcome(rec["id"], "error", None, tuple(rows),
                       time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")
    seconds = time.perf_counter() - t0
    refuted = VerdictStatus.REFUTED.value
    first = next((n for n, s, _ in rows if s == refuted), None)
    if first is not None:
        status = refuted
    elif all(s == VerdictStatus.VERIFIED.value for _, s, _ in rows):
        status = VerdictStatus.VERIFIED.value
    else:
        status = VerdictStatus.UNDECIDED.value
    return Outcome(rec["id"], status, first, tuple(rows), seconds)


def run(
    records: "Iterable[dict[str, Any]]",
    budget: float = BUDGET,
    progress: "Callable[[dict[str, Any], Outcome], None] | None" = None,
) -> "list[Outcome]":
    out = []
    for rec in records:
        o = run_one(rec, budget)
        if progress is not None:
            progress(rec, o)
        out.append(o)
    return out


# --- scoring -----------------------------------------------------------------
def _ratio(num: int, den: int) -> "float | None":
    return None if den == 0 else num / den


def _group(
    records: "Sequence[dict[str, Any]]", outcomes: "Sequence[Outcome]"
) -> "dict[str, Any]":
    wrong = [(r, o) for r, o in zip(records, outcomes) if r["label"] == "wrong"]
    right = [(r, o) for r, o in zip(records, outcomes) if r["label"] == "correct"]

    def count(pairs: "list[tuple[dict[str, Any], Outcome]]", status: str) -> int:
        return sum(o.status == status for _, o in pairs)

    tp = count(wrong, "refuted")
    fp = count(right, "refuted")
    located = sum(o.first_error == r["error_step"] for r, o in wrong if o.status == "refuted")
    times = sorted(o.seconds for o in outcomes)
    undecided = count(wrong, "undecided") + count(right, "undecided")
    return {
        "derivations": len(outcomes),
        "wrong": len(wrong),
        "correct": len(right),
        "tp": tp,
        "fp": fp,
        "wrong_verified": count(wrong, "verified"),
        "wrong_undecided": count(wrong, "undecided"),
        "wrong_error": count(wrong, "error"),
        "correct_verified": count(right, "verified"),
        "correct_undecided": count(right, "undecided"),
        "correct_error": count(right, "error"),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, len(wrong)),
        "step_accuracy": _ratio(located, tp),
        "located": located,
        "undecided": undecided,
        "undecided_rate": _ratio(undecided, len(outcomes)),
        "budget_stops": sum(
            any(m == "time-budget" for _, _, m in o.steps) for o in outcomes
        ),
        "median_s": statistics.median(times) if times else None,
        "p90_s": times[min(len(times) - 1, math.ceil(0.9 * len(times)) - 1)] if times else None,
        "max_s": times[-1] if times else None,
        "total_s": sum(times),
    }


def summarize(
    records: "Sequence[dict[str, Any]]", outcomes: "Sequence[Outcome]"
) -> "dict[str, Any]":
    """Overall, per error type, per template, and with/without boundary cases."""
    by_id = {o.id: o for o in outcomes}
    recs = [r for r in records if r["id"] in by_id]
    outs = [by_id[r["id"]] for r in recs]

    def sub(field: str, value: "str | None") -> "dict[str, Any]":
        pairs = [(r, o) for r, o in zip(recs, outs) if r[field] == value]
        return _group([r for r, _ in pairs], [o for _, o in pairs])

    return {
        "overall": _group(recs, outs),
        "by_type": {t: sub("error_type", t) for t in ERROR_TYPES},
        "by_template": {
            name: sub("template", name)
            for name in dict.fromkeys(r["template"] for r in recs)
        },
        "without_boundary": sub("boundary", None),
        "by_boundary": {
            b: sub("boundary", b)
            for b in sorted({r["boundary"] for r in recs if r["boundary"]})
        },
    }


# --- report ------------------------------------------------------------------
def _pct(v: "float | None") -> str:
    return "  n/a" if v is None else f"{100 * v:5.1f}%"


def _row(name: str, g: "dict[str, Any]") -> str:
    return (
        f"{name:<24} {g['derivations']:>4} {g['tp']:>3}/{g['wrong']:<3} {g['fp']:>3} "
        f"{_pct(g['precision'])} {_pct(g['recall'])} {_pct(g['step_accuracy'])} "
        f"{g['wrong_verified']:>4} {g['wrong_undecided']:>4} {g['correct_verified']:>4} "
        f"{g['correct_undecided']:>4} {_pct(g['undecided_rate'])} {g['budget_stops']:>3} "
        f"{1e3 * (g['median_s'] or 0):>8.0f} {1e3 * (g['p90_s'] or 0):>8.0f} "
        f"{g['max_s'] or 0:>7.1f}"
    )


_HEADER = (
    f"{'group':<24} {'n':>4} {'TP/wrong':>7} {'FP':>3} {'prec':>6} {'recall':>6} "
    f"{'step':>6} {'W:ver':>5} {'W:und':>4} {'C:ver':>4} {'C:und':>4} {'undec':>6} "
    f"{'bud':>3} {'med ms':>8} {'p90 ms':>8} {'max s':>7}"
)


def report(summary: "dict[str, Any]") -> str:
    lines = [_HEADER, "-" * len(_HEADER), _row("ALL", summary["overall"])]
    lines.append(_row("ALL without boundary", summary["without_boundary"]))
    for b, g in summary["by_boundary"].items():
        lines.append(_row(f"boundary: {b}", g))
    lines.append("")
    for t, g in summary["by_type"].items():
        lines.append(_row(t, g))
    lines.append("")
    for t, g in summary["by_template"].items():
        lines.append(_row(t, g))
    return "\n".join(lines)


def main(argv: "Sequence[str] | None" = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m pycodemath.bench.verify_bench")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--seed", type=int, default=None,
                    help="generate a fresh data set from this seed instead of "
                         "reading --data (a control run: is the number the seed's luck?)")
    ap.add_argument("--regenerate", action="store_true",
                    help="write the data file from the generator and stop")
    ap.add_argument("--budget", type=float, default=BUDGET,
                    help=f"seconds per derivation (default {BUDGET:g})")
    ap.add_argument("--only", default=None,
                    help="comma-separated error types or ids to run")
    ap.add_argument("--json", type=Path, default=None,
                    help="write every outcome and the summary here")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="one line per derivation as it finishes")
    args = ap.parse_args(argv)

    if args.regenerate:
        records = generate()
        write(records, args.data)
        print(f"wrote {len(records)} derivations to {args.data}")
        return 0
    records = generate(args.seed) if args.seed is not None else load(args.data)
    if args.only:
        wanted = set(args.only.split(","))
        records = [r for r in records
                   if r["error_type"] in wanted or r["id"] in wanted or r["template"] in wanted]

    def progress(rec: "dict[str, Any]", o: Outcome) -> None:
        if args.verbose:
            print(f"{rec['id']:<16} {rec['template']:<24} {o.status:<9} "
                  f"step={o.first_error!s:<4} want={rec['error_step']!s:<4} "
                  f"{o.seconds:7.2f}s {o.error or ''}", flush=True)

    t0 = time.perf_counter()
    outcomes = run(records, args.budget, progress)
    wall = time.perf_counter() - t0
    summary = summarize(records, outcomes)
    source = f"seed {args.seed}" if args.seed is not None else str(args.data)
    print(f"{len(records)} derivations ({source}), budget {args.budget:g} s each, "
          f"wall {wall:.1f} s\n")
    print(report(summary))
    if args.json is not None:
        args.json.write_text(json.dumps({
            "source": source,
            "budget": args.budget,
            "wall_s": wall,
            "summary": summary,
            "outcomes": [asdict(o) for o in outcomes],
        }, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
