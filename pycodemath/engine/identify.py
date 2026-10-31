"""``identify`` — which closed form is this number? Only answers it can prove (K2).

``mpmath.identify(0.13436424411240122)`` answers
``((85/24) + (-11/9)*pi + (-25/36)*e + (241/72)*log(2))`` for a number drawn at
random, and ``sympy.nsimplify`` answers something for every input. A recogniser
that always finds "something" is a numerology machine: with enough formulas in
the search, any number matches one of them to the digits given. This one says
which closed form it found ONLY when that cannot be a coincidence, and says
UNDECIDED otherwise — with the candidates and how far each one is.

HOW A CANDIDATE IS FOUND. Integer relations by PSLQ (``mpmath.pslq``,
``mpmath.findpoly``, both in a private context) over fixed families:

* rational numbers ``p/q``;
* ``a + b*c`` for one constant ``c`` of the basis or a derived constant
  (``pi**2``, ``log(3)``, ...) — applied to ``x``, ``1/x``, ``x**2`` and ``log x``;
* ``a + b*c1 + c*c2`` for two basis constants;
* products of powers ``± 2^a 3^b 5^c 7^d pi^f e^g`` (a relation among logarithms);
* algebraic numbers of degree 2 to 6 (``findpoly``).

The basis — pi, e, log 2, sqrt 2, sqrt 3, zeta(3), Euler's gamma, Catalan's
constant, the golden ratio — and the derived constants live in
``pycodemath/code/data/K2_constants.json``.

WHEN IS A MATCH NOT A COINCIDENCE — THE THRESHOLD. A relation with ``m`` integer
coefficients of size at most ``h`` is one of about ``(2h+1)^m`` relations of
that shape; over ``F`` families searched, a number known to ``n`` significant
digits satisfies one of them BY CHANCE with probability about
``F * (2h+1)^m * 10^-n``. Its base-10 logarithm, negated, is the candidate's
``margin``: ``n - m*log10(2h+1) - log10(F)``. This is the information-theory
bound of integer-relation finding (Bailey & Borwein: a relation of ``m`` terms
with ``d``-digit coefficients needs about ``m*d`` digits to be meaningful) with
the look-elsewhere factor ``F`` added. A candidate is RECOGNISED (VERIFIED) only
when

1. at least ``IDENTIFY_MIN_DIGITS = 50`` significant digits were given,
2. its margin is at least ``IDENTIFY_MIN_MARGIN = 20`` (chance below 1e-20),
3. its value, enclosed RIGOROUSLY by ``approx`` to ``n + 5`` digits, agrees
   with every given digit (rounded or truncated: within one unit in the last
   place).

Below 50 digits a match is reported as a candidate "consistent with the n digits
given" and the status stays UNDECIDED: 16 digits (a double) leave room for
about 10^16 / F formulas, and the families here hold far more than that.

WHAT VERIFIED MEANS for a decimal: the closed form agrees with all the digits
given, proved by enclosure, and a chance match is excluded by the margin. It is
NOT a proof that the quantity those digits came from equals the closed form —
digits cannot prove that. For an EXACT input (``identify("zeta(2)")``,
``identify("acos(-1)**2/6")``) the candidate is checked by ``check_equal``, and
VERIFIED there means a symbolic proof of equality.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Optional, Sequence

import mpmath
import sympy as sp

from ..core.budget import guard, time_budget
from ..core.errors import DomainError, ParseError
from ..core.ir import Expr
from ..verify.verdict import Verdict, VerdictStatus
from .precision import (
    Approx, PrecisionUndecided, _check_budget, _quiet, approx, to_sympy,
)

__all__ = [
    "Candidate",
    "Identification",
    "identify",
    "IDENTIFY_MIN_DIGITS",
    "IDENTIFY_MIN_MARGIN",
    "default_basis",
    "recheck_identify",
]

#: Fewer significant digits than this can never be VERIFIED (module docstring).
IDENTIFY_MIN_DIGITS = 50
#: A recognised candidate's chance of being a coincidence is below 10^-this.
IDENTIFY_MIN_MARGIN = 20.0
#: Candidates below this margin are coincidences to be expected and are not shown.
_SHOW_MARGIN = 3.0
#: Digits computed for an exact input before searching.
_EXACT_DIGITS = 60

#: Largest integer coefficient sought, per family.
_H_LINEAR = 1000
_H_POWERS = 60
_H_POLY = 1000
_MAX_DEGREE = 6
_MAX_SHOWN = 5

_INTEGER_TEXT = re.compile(r"^\s*[+-]?\d[\d_]*\s*$")
_DECIMAL_TEXT = re.compile(r"^\s*[+-]?(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?\s*$")


# --- the basis --------------------------------------------------------------------

@lru_cache(maxsize=1)
def _data() -> dict:
    text = resources.files("pycodemath.code.data").joinpath("K2_constants.json").read_text("utf-8")
    return json.loads(text)  # type: ignore[no-any-return]


def default_basis() -> tuple[str, ...]:
    """The constants searched by default (``K2_constants.json``, key ``basis``)."""
    return tuple(_data()["basis"])


def _derived() -> tuple[str, ...]:
    return tuple(_data()["derived"])


# --- results ----------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One closed form that matches the digits, with how well and how surprisingly.

    ``expr``                 the closed form (SymPy text)
    ``value``                its decimal value (certified, 20 digits)
    ``deviation``            given number minus candidate, as a short decimal
    ``agree_digits``         leading significant digits on which they agree
    ``description_digits``   ``m*log10(2h+1) + log10(F)``: how many digits it takes
                             to pick this formula out of the families searched
    ``margin``               ``given digits - description_digits``; chance of a
                             coincidence about ``10^-margin``
    ``family``               which search found it
    ``consistent``           agrees with every given digit (within 1 ulp)
    ``rigorous``             the agreement was decided by a rigorous enclosure
    """

    expr: str
    value: str
    deviation: str
    agree_digits: int
    description_digits: float
    margin: float
    family: str
    consistent: bool
    rigorous: bool

    def to_dict(self) -> dict:
        return {
            "expr": self.expr, "value": self.value, "deviation": self.deviation,
            "agree_digits": self.agree_digits,
            "description_digits": self.description_digits, "margin": self.margin,
            "family": self.family, "consistent": self.consistent,
            "rigorous": self.rigorous,
        }


@dataclass(frozen=True)
class Identification:
    """The answer of ``identify``: VERIFIED with a closed form, or UNDECIDED.

    ``value`` is set exactly when ``status`` is VERIFIED. ``candidates`` are the
    matches worth showing (margin >= 3), best first, whatever the status.
    ``digits`` is how many significant digits of the input were used,
    ``searched`` how many PSLQ problems were run (the ``F`` of the margin).
    REFUTED is never returned: no number refutes a closed form it was not asked
    about.
    """

    status: VerdictStatus
    value: "str | None"
    input: str
    digits: int
    candidates: tuple[Candidate, ...]
    method: str
    detail: str
    warnings: tuple[str, ...] = ()
    basis: tuple[str, ...] = ()
    searched: int = 0
    kind: str = "identify"

    @property
    def verified(self) -> bool:
        return self.status is VerdictStatus.VERIFIED

    @property
    def verdict(self) -> Verdict:
        """The status as a ``Verdict`` — what an X1 certificate records."""
        return Verdict(self.status, self.method, None, self.detail)

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.input} = {self.value}  [VERIFIED: {self.detail}]"
        head = f"{self.input}: UNDECIDED — {self.detail}"
        lines = [head] + [
            f"  candidate {c.expr}  (deviation {c.deviation}, agrees on {c.agree_digits} "
            f"digits, margin {c.margin:g})" for c in self.candidates
        ]
        return "\n".join(lines + [f"  warning: {w}" for w in self.warnings])

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "status": self.status.value,
            "value": self.value,
            "input": self.input,
            "digits": self.digits,
            "candidates": [c.to_dict() for c in self.candidates],
            "method": self.method,
            "detail": self.detail,
            "warnings": list(self.warnings),
            "basis": list(self.basis),
            "searched": self.searched,
        }


# --- reading the input --------------------------------------------------------------

@dataclass
class _Target:
    x: Fraction            # the number searched for (exact)
    n: int                 # significant digits it is known to
    tol: Fraction          # the scale of the uncertainty (one unit in the last place)
    text: str              # the input as shown back
    below: Fraction = Fraction(0)   # true value in [x - below, x + above]
    above: Fraction = Fraction(0)
    exact: "sp.Expr | None" = None   # set for an exact input
    rigorous_input: bool = True
    notes: list[str] = field(default_factory=list)


def _sig_digits(text: str) -> tuple[Fraction, int, int]:
    """(value, significant digits written, decimal exponent of the last digit)."""
    t = text.strip().replace("_", "")
    mant, _, exp = t.lower().partition("e")
    e = int(exp) if exp else 0
    sign = -1 if mant.startswith("-") else 1
    mant = mant.lstrip("+-")
    whole, _, frac = mant.partition(".")
    digits = (whole + frac).lstrip("0")
    value = Fraction(Decimal(t)) if t else Fraction(0)
    _ = sign
    return value, len(digits), e - len(frac)


def _target(x: Any, digits: "int | None") -> _Target:
    if isinstance(x, bool):
        raise DomainError("identify: expected a number or an expression, not a bool")
    text: "str | None" = None
    float_input = False
    if isinstance(x, str):
        if not x.strip():
            raise ParseError("identify: empty input")
        if _DECIMAL_TEXT.match(x) and not _INTEGER_TEXT.match(x):
            text = x.strip()
    elif isinstance(x, float):
        if not math.isfinite(x):
            raise DomainError(f"identify: {x!r} is not a finite number")
        text, float_input = repr(x), True
    elif isinstance(x, Decimal):
        if not x.is_finite():
            raise DomainError(f"identify: {x!r} is not a finite number")
        text = str(x)
    elif isinstance(x, sp.Float):
        text = sp.sstr(x, full_prec=True)
    elif isinstance(x, mpmath.mpf):
        text = mpmath.nstr(x, max(1, int(x.context.prec * math.log10(2)) - 1),
                           strip_zeros=False, min_fixed=-math.inf, max_fixed=math.inf)
    if text is None:
        return _exact_target(x, digits)
    value, n, last = _sig_digits(text)
    if digits is not None:
        if isinstance(digits, bool) or not isinstance(digits, int) or digits < 1:
            raise DomainError(f"identify: digits must be a positive integer, not {digits!r}")
        if digits > n:
            raise DomainError(
                f"identify: digits={digits} but only {n} significant digits were given"
            )
        last += n - digits
        n = digits
    ulp = Fraction(10) ** last
    # The digits were ROUNDED (true value within half a unit) or TRUNCATED (true
    # value between x and one unit further from zero): the union of the two.
    # Not "one unit either way" — that would accept 1 for 1.000...0001.
    far, near = ulp, ulp / 2
    t = _Target(value, n, ulp, text,
                below=near if value >= 0 else far, above=far if value >= 0 else near)
    if float_input:
        # a double carries 53 bits; its shortest repr may claim one digit more
        slack = abs(value) * Fraction(1, 2 ** 51)
        t.tol = max(ulp, slack)
        t.below, t.above = max(t.below, slack), max(t.above, slack)
        t.notes.append("a Python float: its digits are a double's, at most ~16 significant")
    return t


def _exact_target(x: Any, digits: "int | None") -> _Target:
    sy = to_sympy(x, "identify")
    n = max(_EXACT_DIGITS, digits or 0)
    a = approx(sy, n)
    if a.imag is not None:
        raise DomainError(f"identify: {sp.sstr(sy)} is not a real number (≈ {a.value})")
    if a.real == "0":
        value = Fraction(0)
        tol = Fraction(0)
    else:
        value, _n, last = _sig_digits(a.real)
        tol = Fraction(10) ** last  # correctly rounded: within half of this
    t = _Target(value, n, tol, sp.sstr(sy), exact=sy, rigorous_input=a.rigorous,
                below=tol / 2, above=tol / 2)
    if not a.rigorous:
        t.notes.append(f"the input's digits were found by {a.method}, not a proof")
    return t


# --- the search -----------------------------------------------------------------------

@dataclass
class _Found:
    expr: sp.Expr
    family: str
    coeffs: tuple[int, ...]


def _families(basis: Sequence[str]) -> tuple[list[tuple], int]:
    """The PSLQ problems, and how many there are (``F`` of the margin)."""
    derived = [d for d in _derived() if d not in basis]
    singles = list(basis) + derived
    pair_pool = list(basis) + [c for c in ("pi**2", "log(2)**2") if c not in basis]
    problems: list[tuple] = [("rational",)]
    for transform in ("x", "1/x", "x**2", "log(x)"):
        for c in singles:
            problems.append(("linear", transform, (c,)))
    for c1, c2 in itertools.combinations(pair_pool, 2):
        problems.append(("linear", "x", (c1, c2)))
    problems.append(("powers",))
    for deg in range(2, _MAX_DEGREE + 1):
        problems.append(("algebraic", deg))
    return problems, len(problems)


_POWER_BASE = ("2", "3", "5", "7", "pi")


def _search(t: _Target, basis: Sequence[str]) -> tuple[list[_Found], int]:
    problems, count = _families(basis)
    ctx = mpmath.MPContext()
    ctx.dps = t.n + 15
    x = ctx.mpf(t.x.numerator) / t.x.denominator
    if x == 0:
        return [], count
    # Scale far-off magnitudes so the relations stay small: x = 10^k * m.
    k = _scale_k(t.x)
    m = x / ctx.mpf(10) ** k
    rel_unc = ctx.mpf(t.tol.numerator) / t.tol.denominator / abs(x) + ctx.mpf(10) ** (-(t.n + 5))
    constants = {name: _const_value(ctx, name) for name in set(
        [c for p in problems if p[0] == "linear" for c in p[2]])}
    found: list[_Found] = []
    for p in problems:
        try:
            if p[0] == "rational":
                found += _pslq_linear(ctx, m, "x", (), rel_unc, k)
            elif p[0] == "linear":
                found += _pslq_linear(ctx, m, p[1], p[2], rel_unc, k, constants)
            elif p[0] == "powers":
                found += _pslq_powers(ctx, m, rel_unc, k)
            else:
                found += _findpoly(ctx, m, p[1], rel_unc, k)
        except (ValueError, ZeroDivisionError, ArithmeticError):
            continue
    return found, count


def _scale_k(x: Fraction) -> int:
    """The power of ten taken out of ``x`` before searching (0 near 1)."""
    if x == 0:
        return 0
    a = abs(x)
    e = int(math.floor((a.numerator.bit_length() - a.denominator.bit_length()) * math.log10(2)))
    while Fraction(10) ** e > a:
        e -= 1
    while Fraction(10) ** (e + 1) <= a:
        e += 1
    return 0 if -3 <= e <= 4 else e


@lru_cache(maxsize=64)
def _const_sympy(name: str) -> sp.Expr:
    return to_sympy(name, "identify basis")


def _const_value(ctx: Any, name: str) -> Any:
    sy = _const_sympy(name)
    return ctx.mpmathify(str(sp.N(sy, ctx.dps + 10)))


def _scaled(expr: sp.Expr, k: int) -> sp.Expr:
    return expr * sp.Integer(10) ** k if k else expr


def _pslq_linear(ctx: Any, m: Any, transform: str, consts: tuple, rel_unc: Any, k: int,
                 values: "dict | None" = None) -> list[_Found]:
    if transform == "x":
        t, unc = m, rel_unc * abs(m)
    elif transform == "1/x":
        t, unc = 1 / m, rel_unc / abs(m)
    elif transform == "x**2":
        t, unc = m * m, 2 * rel_unc * m * m
    else:
        if m <= 0:
            return []
        t, unc = ctx.log(m), rel_unc
    vec = [t, ctx.mpf(1)] + [values[c] for c in consts] if values else [t, ctx.mpf(1)]
    tol = max(unc * _H_LINEAR * 10, ctx.mpf(10) ** (-(ctx.dps - 8)))
    rel = ctx.pslq(vec, tol=tol, maxcoeff=_H_LINEAR, maxsteps=4000)
    if rel is None or rel[0] == 0:
        return []
    a0 = rel[0]
    tsy = -(sp.Integer(rel[1]) + sum(
        (sp.Integer(c) * _const_sympy(n) for c, n in zip(rel[2:], consts)), sp.Integer(0)
    )) / a0
    if transform == "x":
        msy = tsy
    elif transform == "1/x":
        if tsy == 0:
            return []
        msy = 1 / tsy
    elif transform == "x**2":
        if tsy.is_negative:
            return []
        msy = sp.sqrt(tsy) * (1 if m > 0 else -1)
    else:
        msy = sp.exp(tsy)
    family = "rational" if not consts and transform == "x" else (
        f"linear in {transform} over {', '.join(consts)}")
    return [_Found(_scaled(msy, k), family, tuple(int(r) for r in rel))]


def _pslq_powers(ctx: Any, m: Any, rel_unc: Any, k: int) -> list[_Found]:
    sign = -1 if m < 0 else 1
    am = abs(m)
    logs = [ctx.log(am)] + [ctx.log(ctx.mpmathify(b) if b != "pi" else ctx.pi)
                            for b in _POWER_BASE] + [ctx.mpf(1)]
    tol = max(rel_unc * _H_POWERS * 10, ctx.mpf(10) ** (-(ctx.dps - 8)))
    rel = ctx.pslq(logs, tol=tol, maxcoeff=_H_POWERS, maxsteps=6000)
    if rel is None or rel[0] == 0:
        return []
    a0 = rel[0]
    expr: sp.Expr = sp.Integer(sign)
    for c, b in zip(rel[1:-1], _POWER_BASE):
        if c:
            expr *= (sp.pi if b == "pi" else sp.Integer(int(b))) ** sp.Rational(-c, a0)
    if rel[-1]:
        expr *= sp.exp(sp.Rational(-rel[-1], a0))
    return [_Found(_scaled(expr, k), "products of powers of 2, 3, 5, 7, pi, e",
                   tuple(int(r) for r in rel))]


def _findpoly(ctx: Any, m: Any, degree: int, rel_unc: Any, k: int) -> list[_Found]:
    unc = rel_unc * max(1, abs(m)) ** degree * degree
    tol = max(unc * _H_POLY * 10, ctx.mpf(10) ** (-(ctx.dps - 8)))
    coeffs = ctx.findpoly(m, degree, maxcoeff=_H_POLY, tol=tol, maxsteps=6000)
    if coeffs is None or len(coeffs) != degree + 1:
        return []  # a lower degree already found it
    X = sp.Symbol("x")
    poly = sp.Poly(sum(sp.Integer(c) * X ** (degree - i) for i, c in enumerate(coeffs)), X)
    if not poly.is_irreducible:
        return []
    roots: list[sp.Expr] = []
    if degree <= 4:
        try:
            roots = [r for r in sp.roots(poly, multiple=True) if r.is_real]
        except (NotImplementedError, sp.PolynomialError):
            roots = []
    if not roots:
        roots = [r for r in poly.all_roots() if r.is_real]
    target = float(m)
    best = min(roots, key=lambda r: abs(float(sp.N(r, 30)) - target), default=None)
    if best is None:
        return []
    return [_Found(_scaled(best, k), f"algebraic of degree {degree}",
                   tuple(int(c) for c in coeffs))]


# --- judging the candidates -------------------------------------------------------------

def _describe(f: _Found, searched: int, k: int) -> float:
    h = max(abs(c) for c in f.coeffs)
    m = len(f.coeffs)
    extra = math.log10(2 * abs(k) + 1) if k else 0.0
    return m * math.log10(2 * h + 1) + math.log10(searched) + extra


def _short_decimal(q: Fraction, sig: int = 3) -> str:
    if q == 0:
        return "0"
    return mpmath.nstr(mpmath.mpf(q.numerator) / q.denominator, sig)


def _judge(t: _Target, found: list[_Found], searched: int, k: int) -> list[Candidate]:
    out: dict[str, Candidate] = {}
    for f in found:
        try:
            expr = sp.nsimplify(f.expr) if f.expr.has(sp.Float) else f.expr
            text = sp.sstr(expr)
            if text in out:
                continue
            a = approx(expr, t.n + 5)
        except (PrecisionUndecided, DomainError, ParseError, ValueError, TypeError):
            continue
        if a.imag is not None:
            continue
        cv, _n, last = _sig_digits(a.real) if a.real != "0" else (Fraction(0), 0, 0)
        half = Fraction(10) ** last / 2 if a.real != "0" else Fraction(0)
        diff = t.x - cv
        # the candidate's whole enclosure [cv - half, cv + half] must lie where the
        # true value can be; for an exact input both sides are roundings of
        # numbers, so each may sit half a unit off
        slack = half if t.exact is not None else Fraction(0)
        consistent = (cv - half >= t.x - t.below - slack) and (cv + half <= t.x + t.above + slack)
        if t.x != 0 and abs(diff) > 0:
            agree = int(math.floor(math.log10(abs(t.x)) - math.log10(abs(diff))))
            agree = max(0, min(t.n, agree))
        else:
            agree = t.n
        desc = _describe(f, searched, k)
        margin = round(t.n - desc, 1)
        cand = Candidate(
            text, approx(expr, 20).value, _short_decimal(diff) if abs(diff) > half else
            f"below the last digit (|d| <= {_short_decimal(t.tol, 2)})",
            agree, round(desc, 1), margin, f.family, consistent, a.rigorous and t.rigorous_input,
        )
        prev = out.get(text)
        if prev is None or cand.description_digits < prev.description_digits:
            out[text] = cand
    ranked = sorted(out.values(), key=lambda c: (not c.consistent, -c.margin))
    return ranked


def _numerology_warning(n: int, searched: int) -> str:
    return (
        f"only {n} significant digits: the {searched} families searched hold far more "
        f"than 10^{n} formulas with small coefficients, so a match to {n} digits "
        f"happens by chance (the 'strong law of small numbers' — numerology); "
        f"give at least {IDENTIFY_MIN_DIGITS} digits to recognise a constant"
    )


def _identify(t: _Target, basis: tuple[str, ...]) -> Identification:
    if t.x == 0 and t.exact is None:
        return Identification(
            VerdictStatus.UNDECIDED, None, t.text, 0, (), "pslq+enclosure",
            "a decimal zero has no significant digits: it is consistent with 0 and with "
            "every number smaller than its last place", tuple(t.notes), basis, 0,
        )
    found, searched = _search(t, basis)
    k = _scale_k(t.x)
    cands = _judge(t, found, searched, k)
    shown = tuple(c for c in cands if c.margin >= _SHOW_MARGIN)[:_MAX_SHOWN]
    dropped = len(cands) - len([c for c in cands if c.margin >= _SHOW_MARGIN])
    warns = list(t.notes)
    if dropped:
        warns.append(
            f"{dropped} relation(s) with margin below {_SHOW_MARGIN:g} not shown: a match "
            f"that good is expected by chance"
        )
    if t.exact is not None:
        return _decide_exact(t, shown, basis, searched, tuple(warns))
    if t.n < IDENTIFY_MIN_DIGITS:
        warns.append(_numerology_warning(t.n, searched))
    best = shown[0] if shown else None
    if (
        best is not None and best.consistent and best.rigorous
        and t.n >= IDENTIFY_MIN_DIGITS and best.margin >= IDENTIFY_MIN_MARGIN
    ):
        detail = (
            f"{best.expr} agrees with all {t.n} given digits (rigorous enclosure); "
            f"chance of a coincidence ~1e-{best.margin:g} over {searched} families"
        )
        return Identification(VerdictStatus.VERIFIED, best.expr, t.text, t.n, shown,
                              "pslq+enclosure", detail, tuple(warns), basis, searched)
    if best is None and dropped:
        detail = (
            f"the {dropped} relation(s) found match {t.n} digits no better than chance "
            f"(margin < {_SHOW_MARGIN:g})"
        )
    elif best is None:
        detail = f"no closed form in the {searched} families matches the {t.n} digits"
    elif t.n < IDENTIFY_MIN_DIGITS:
        detail = (
            f"best candidate {best.expr} is consistent with the {t.n} digits given, "
            f"but {t.n} < {IDENTIFY_MIN_DIGITS} digits cannot exclude a coincidence"
        )
    elif not best.consistent:
        detail = f"best candidate {best.expr} differs from the given digits"
    else:
        detail = (
            f"best candidate {best.expr} has margin {best.margin:g} < "
            f"{IDENTIFY_MIN_MARGIN:g}: a chance match is not excluded"
        )
    return Identification(VerdictStatus.UNDECIDED, None, t.text, t.n, shown,
                          "pslq+enclosure", detail, tuple(warns), basis, searched)


def _decide_exact(t: _Target, shown: tuple[Candidate, ...], basis: tuple[str, ...],
                  searched: int, warns: tuple[str, ...]) -> Identification:
    from ..verify.equal import check_equal

    assert t.exact is not None
    if t.exact.is_Rational:
        return Identification(VerdictStatus.VERIFIED, sp.sstr(t.exact), t.text, t.n, shown,
                              "exact", "the input is a rational number", warns, basis, searched)
    refuted = 0
    for c in shown:
        if not c.consistent:
            continue
        verdict = check_equal(Expr(t.exact), c.expr)
        if verdict.verified:
            return Identification(
                VerdictStatus.VERIFIED, c.expr, t.text, t.n, shown, "check_equal",
                f"check_equal proves {t.text} == {c.expr} ({verdict.detail})",
                warns, basis, searched,
            )
        if verdict.refuted:
            refuted += 1
    best = next((c for c in shown if c.consistent), None)
    if best is None:
        detail = f"no closed form in the {searched} families matches {t.n} digits of the input"
    else:
        detail = (
            f"{best.expr} agrees with the input to {t.n} digits but check_equal could not "
            f"prove the identity"
        )
    if refuted:
        detail += f"; {refuted} candidate(s) refuted by check_equal"
    return Identification(VerdictStatus.UNDECIDED, None, t.text, t.n, shown,
                          "check_equal", detail, warns, basis, searched)


def identify(
    x: Any,
    *,
    digits: "int | None" = None,
    basis: "Iterable[str] | None" = None,
    budget: "float | None" = None,
) -> Identification:
    """Recognise a closed form for ``x`` — VERIFIED only when it cannot be chance.

    ``x`` is a decimal (text ``"1.6449340668482264"``, ``float``, ``Decimal``,
    SymPy ``Float``, mpmath ``mpf``) or an exact expression (``"zeta(2)"``,
    an ``Expr``). ``digits`` uses only the first ``digits`` significant digits of
    a decimal. ``basis`` replaces the default constants (names or expressions,
    e.g. ``["pi", "log(3)"]``). See the module docstring for the families, the
    threshold and what VERIFIED means.

        >>> identify("1.6449340668482264").status.value          # 17 digits
        'undecided'
        >>> identify("1.6449340668482264").candidates[0].expr
        'pi**2/6'
        >>> identify("zeta(2)").value
        'pi**2/6'
    """
    if basis is None:
        names = default_basis()
    else:
        if isinstance(basis, str):
            raise DomainError("identify: basis must be a list of constants, not one string")
        names = tuple(dict.fromkeys(str(b) for b in basis))
        for b in names:
            to_sympy(b, "identify basis")
    seconds = None if budget is None else _check_budget(budget, "identify")

    def run() -> Identification:
        return _identify(_target(x, digits), names)

    if seconds is None:
        return _quiet(lambda: guard("identify", x, run))  # type: ignore[no-any-return]

    def bounded() -> Identification:
        with time_budget(seconds, "identify", x):
            return run()

    return _quiet(bounded)  # type: ignore[no-any-return]


def recheck_identify(x: str, value: "str | None", *, digits: "int | None" = None,
                     basis: "Iterable[str] | None" = None,
                     budget: "float | None" = None) -> Verdict:
    """Re-derive a recorded ``identify`` result — the ``run`` of an X1 ``identify``
    kind. VERIFIED only when the search, run again on the recorded input, VERIFIES
    the recorded closed form; UNDECIDED when it does not decide; REFUTED (no point)
    when it verifies a DIFFERENT value than the one recorded.

        >>> recheck_identify("zeta(2)", "pi**2/6").status.value
        'verified'
    """
    res = identify(x, digits=digits, basis=basis, budget=budget)
    if not res.verified or value is None:
        return Verdict(VerdictStatus.UNDECIDED, res.method, None, res.detail)
    if res.value != value:
        same = approx(res.value or "0", 70).value == approx(value, 70).value
        if not same:
            return Verdict(VerdictStatus.REFUTED, res.method, {},
                           f"the input is recognised as {res.value}, not {value}")
    return Verdict(VerdictStatus.VERIFIED, res.method, None, res.detail)
