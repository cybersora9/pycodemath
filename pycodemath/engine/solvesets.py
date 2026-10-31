"""Solution SETS with certificates (series FULL, module K5): inequalities, systems, periodic families.

Every calculator "solves" ``x^2 - 3x + 2 < 0``. This module answers it AND proves the
answer, or refuses with a typed reason. Three public entry points, each returning a
frozen result with ``to_dict()`` (``kind`` = the X1 certificate kind):

``solve_inequality``   one real variable; polynomial, rational, ``abs``, roots, ``log``,
                       ``exp`` (and trigonometric ones on a bounded window)
``solve_system``       linear (exact, rank, parametrisation, proof of inconsistency) and
                       polynomial (Groebner basis, completeness by counting) systems
``periodic_solutions`` ``sin(x) = 1/2`` -> ``x = pi/6 + 2*pi*k, x = 5*pi/6 + 2*pi*k``

--- HOW AN INEQUALITY ANSWER IS PROVED (the sign decomposition) ----------------------

SymPy's ``solveset`` proposes an answer; it is NOT trusted (it answers ``sin(x) > 0``
with ``Interval.open(0, pi)``, measured on SymPy 1.14). The answer that is returned is
the one this module PROVES, by cutting the real line at a finite set of exact points:

* every real zero of ``h = lhs - rhs``,
* every point where the natural domain can change (zeros of denominators, of the
  arguments of ``log``/``sqrt``/fractional powers, poles of ``tan``),
* every endpoint SymPy proposed (extra cuts are harmless: each is checked).

On each open CELL between two consecutive points, ``h`` is continuous and has no zero,
so it has ONE sign there (intermediate value theorem) and the whole cell is in or out.
That sign is established in one of two ways, recorded per cell as ``method``:

``exact-roots``          ``h`` is a rational function with rational coefficients: the
                         cuts contain ALL real roots of its numerator and denominator
                         (exact real-root isolation, ``Poly.real_roots``), so one exact
                         rational sample decides the sign of the whole cell.
``interval-arithmetic``  otherwise: a rigorous enclosure of ``h`` over the cell (mpmath
                         interval arithmetic, outward rounding, adaptive bisection)
                         excludes 0. Near an endpoint where ``h`` vanishes or blows up,
                         the enclosure of a derivative excludes 0 instead, and the
                         one-sided limit at the endpoint (SymPy ``limit``) fixes the sign.

Each cut point itself is decided EXACTLY: membership in the domain (signs of the
constraint functions there) and the sign of ``h`` (zero by exact simplification or a
minimal polynomial, otherwise a nonzero ``evalf`` with error control). A cell or point
that cannot be decided makes the whole call a ``SolveSetUndecided`` — never a guess.
The certificate carries every cell (with its exact rational sample) and every point,
so it can be re-checked by anyone with a CAS.

The SymPy answer is kept beside ours: ``sympy_agrees`` says whether it was right and,
when it was not, ``sympy_witness`` is a point where it is wrong.

--- WHAT IS NOT CLAIMED ---------------------------------------------------------------

Several variables in an inequality; trigonometric inequalities on the whole real line
(infinitely many cells — give ``within=(a, b)``); systems mixing polynomial and
transcendental equations (``UnsupportedFormError`` with route ``solve_nd``); positive-
dimensional NONLINEAR systems (same route). Values of a variable are REAL in
inequalities and periodic families; systems report complex solutions and tag which
are real.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence

import mpmath
import sympy as sp
from mpmath import iv

from ..core.budget import under_budget
from ..core.errors import (
    DomainError,
    ParseError,
    PycodemathError,
    TimeBudgetError,
    UnsupportedFormError,
)
from ..frontend.parser import parse
from sympy.printing.str import StrPrinter

__all__ = [
    "SolveSetUndecided",
    "EndpointCheck",
    "CellCheck",
    "InequalityResult",
    "SystemResult",
    "PeriodicFamily",
    "PeriodicResult",
    "solve_inequality",
    "solve_system",
    "periodic_solutions",
]


class SolveSetUndecided(PycodemathError):
    """No answer could be PROVED, so none is returned.

    Raised when a cell's sign cannot be certified (an enclosure that never excludes 0
    within the effort limit — typically a zero the cut points do not know about), when
    the sign at a cut point cannot be decided exactly, or when the cut points
    themselves cannot be found (a transcendental equation SymPy does not solve on an
    unbounded line). ``reason`` repeats the cause as one short machine-stable phrase.
    """

    def __init__(self, message: str, *, reason: str = "", route: "str | None" = None) -> None:
        super().__init__(message, route=route)
        self.reason = reason


# --- small helpers ----------------------------------------------------------------

#: Functions the decomposition knows how to enclose and whose domains it knows.
_ALLOWED_FUNCS = (sp.exp, sp.log, sp.Abs, sp.sin, sp.cos, sp.tan, sp.atan, sp.cot, sp.sec, sp.csc)

#: Working precision (bits) of the interval enclosures and the bisection limits.
_IV_PREC = 113
_MAX_PIECES = 4000
_MAX_DEPTH = 40

#: Digits used to ORDER exact cut points and to read off a nonzero sign.
_ORDER_DPS = 60
_ZERO_EPS = sp.Float("1e-45", 60)

_SYM_OPS: tuple[tuple[str, str], ...] = (
    ("<=", "<="), (">=", ">="), ("≤", "<="), ("≥", ">="), ("=<", "<="), ("=>", ">="),
    ("<", "<"), (">", ">"),
)
_REL = {"<": sp.StrictLessThan, "<=": sp.LessThan, ">": sp.StrictGreaterThan, ">=": sp.GreaterThan}


class _SetPrinter(StrPrinter):
    """``str`` that reads back with ``sympify``: a finite set prints as
    ``FiniteSet(1, 2)``, not ``{1, 2}`` (which ``sympify`` turns into a Python set)."""

    def _print_FiniteSet(self, s: sp.FiniteSet) -> str:
        return "FiniteSet(" + ", ".join(self._print(a) for a in s.args) + ")"


_PRINTER = _SetPrinter({"order": None})


def _s(e: Any) -> str:
    """Exact SymPy text of a value or a set (reads back with ``sympify``)."""
    return _PRINTER.doprint(e)


def _approx(e: sp.Expr, digits: int = 15) -> str:
    if e in (sp.oo, -sp.oo):
        return str(e)
    return sp.sstr(sp.N(e, digits))


def _exact(e: sp.Expr, who: str) -> sp.Expr:
    """Decimals mean the decimal WRITTEN (``0.1`` is ``1/10``, as in ``check_equal``),
    and an undefined or infinite value inside the input is refused up front."""
    floats = e.atoms(sp.Float)
    if floats:
        e = e.xreplace({f: sp.nsimplify(f, rational=True) for f in floats})
    if e.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise DomainError(f"{who}: the input contains an undefined or infinite value ({e})")
    return e


def _real_symbol(var: str) -> sp.Symbol:
    if not isinstance(var, str) or not var.isidentifier():
        raise DomainError(f"variable must be a name, not {var!r}")
    return sp.Symbol(var, real=True)


def _to_real(e: sp.Expr, x: sp.Symbol) -> sp.Expr:
    """Re-express ``e`` (parsed with a plain symbol) in the REAL symbol ``x``."""
    return e.xreplace({sp.Symbol(x.name): x})


def _parse_expr(text: "str | sp.Basic | int | Fraction", what: str) -> sp.Expr:
    if isinstance(text, sp.Basic):
        return sp.sympify(text)
    if isinstance(text, bool):
        raise DomainError(f"{what}: not a number: {text!r}")
    if isinstance(text, int):
        return sp.Integer(text)
    if isinstance(text, Fraction):
        return sp.Rational(text.numerator, text.denominator)
    if isinstance(text, float):
        if not math.isfinite(text):
            return sp.oo if text > 0 else -sp.oo if text < 0 else sp.nan
        return sp.Rational(str(text))  # the decimal WRITTEN, as check_equal reads it
    if not isinstance(text, str):
        raise DomainError(f"{what}: expected text or a SymPy expression, not {type(text).__name__}")
    return parse(text).sy


def _check_constant(v: sp.Expr, what: str) -> sp.Expr:
    if v.free_symbols:
        raise DomainError(f"{what} must be a number, not {v}")
    if v in (sp.oo, -sp.oo):
        return v
    if v.has(sp.nan, sp.zoo) or v.is_real is False or not v.is_finite:
        raise DomainError(f"{what} must be a real number or ±oo, not {v}")
    return v


def _check_functions(h: sp.Expr, who: str) -> None:
    """Refuse shapes the decomposition cannot enclose — before any work."""
    for node in sp.preorder_traversal(h):
        if isinstance(node, sp.Function) and not isinstance(node, _ALLOWED_FUNCS):
            raise UnsupportedFormError(
                f"{who}: the function {node.func.__name__} is not supported here "
                f"(supported: polynomials, rational functions, abs, roots, exp, log, "
                f"sin, cos, tan, cot, sec, csc, atan)"
            )


# --- exact decisions at a point ------------------------------------------------------

def _is_zero(v: sp.Expr) -> "bool | None":
    """Exact zero test: True / False / None (not decidable here)."""
    if v == 0:
        return True
    if v.is_number and v.is_zero is not None:
        return bool(v.is_zero)
    if v.is_number and v.is_algebraic and not v.has(sp.Abs):
        try:
            m = sp.minimal_polynomial(v, sp.Dummy("z"))
            return bool(m.is_Symbol)
        except (NotImplementedError, ValueError, sp.PolynomialError):
            pass
    try:
        eq = v.equals(0)
    except Exception:  # noqa: BLE001 — SymPy's equals raises a zoo of errors
        eq = None
    if eq is not None:
        return bool(eq)
    s = sp.simplify(v)
    if s == 0:
        return True
    return None


def _sign_at(v: sp.Expr) -> "int | None":
    """Sign of a closed-form real number: -1, 0, 1; None if undecidable or not real.

    A value that is visibly away from 0 at 30 digits gets its sign from ``evalf``
    (SymPy's evalf tracks its own error and ``strict`` refuses digits it cannot vouch
    for); only a value that LOOKS like 0 goes to the exact zero test, which for sums
    of ``CRootOf`` means a minimal polynomial of high degree — measured at minutes for
    two quintic roots, where the numerical sign takes milliseconds.
    """
    if v.has(sp.zoo, sp.nan) or v in (sp.oo, -sp.oo):
        return None
    if v == 0:
        return 0
    for dps in (30, 60, 120):
        try:
            n = v.evalf(dps, strict=True, maxn=4 * dps)
        except (mpmath.libmp.NoConvergence, sp.core.evalf.PrecisionExhausted, ValueError, TypeError):
            continue
        re, im = n.as_real_imag()
        if im != 0 and abs(im) > _ZERO_EPS:
            return None  # complex: not a real value
        if abs(re) > _ZERO_EPS:
            return 1 if re > 0 else -1
        break
    z = _is_zero(v)
    if z:
        return 0
    return None


def _num(e: sp.Expr, dps: int = _ORDER_DPS) -> mpmath.mpf:
    if e == sp.oo:
        return mpmath.inf
    if e == -sp.oo:
        return -mpmath.inf
    with mpmath.workdps(dps):
        return mpmath.mpf(sp.N(e, dps))


def _lt(a: sp.Expr, b: sp.Expr) -> "bool | None":
    """Exact ``a < b`` for real closed forms (None: equal or undecidable)."""
    if a == b:
        return False
    if a == -sp.oo or b == sp.oo:
        return True
    if a == sp.oo or b == -sp.oo:
        return False
    s = _sign_at(b - a)
    if s is None:
        return None
    return s > 0


# --- domain constraints ----------------------------------------------------------------

@dataclass(frozen=True)
class _Constraint:
    g: sp.Expr
    need: Literal["pos", "nonneg", "nonzero"]

    def text(self, x: sp.Symbol) -> str:
        op = {"pos": ">", "nonneg": ">=", "nonzero": "!="}[self.need]
        return f"{_s(self.g)} {op} 0"

    def holds(self, sign: int) -> bool:
        return {"pos": sign > 0, "nonneg": sign >= 0, "nonzero": sign != 0}[self.need]


def _constraints(h: sp.Expr, x: sp.Symbol) -> list[_Constraint]:
    """The natural REAL domain of ``h`` as sign conditions on sub-expressions."""
    out: list[_Constraint] = []

    def add(g: sp.Expr, need: Literal["pos", "nonneg", "nonzero"]) -> None:
        if not g.has(x):
            return
        c = _Constraint(sp.factor_terms(g), need)
        if c not in out:
            out.append(c)

    for node in sp.preorder_traversal(h):
        if isinstance(node, sp.Pow):
            b, e = node.args
            if not b.has(x) and not e.has(x):
                continue
            if e.has(x):
                if b.has(x):
                    add(b, "pos")
                continue
            if e.is_Integer:
                if e < 0:
                    add(b, "nonzero")
            elif e.is_Rational:
                add(b, "pos" if e < 0 else "nonneg")
            else:
                add(b, "pos")
        elif isinstance(node, sp.log):
            add(node.args[0], "pos")
        elif isinstance(node, (sp.tan, sp.sec)):
            add(sp.cos(node.args[0]), "nonzero")
        elif isinstance(node, (sp.cot, sp.csc)):
            # SymPy itself rewrites tan(pi/4 - u) as cot(u + pi/4): the shifted
            # expressions at a cell's end must stay enclosable
            add(sp.sin(node.args[0]), "nonzero")
    return out


# --- zeros (cut points) ----------------------------------------------------------------

def _rational_parts(h: sp.Expr, x: sp.Symbol) -> "tuple[sp.Poly, sp.Poly] | None":
    """``h = P/Q`` with P, Q in QQ[x], or None (not rational over QQ)."""
    if not h.is_rational_function(x):
        return None
    num, den = sp.fraction(sp.cancel(sp.together(h)))
    try:
        P = sp.Poly(num, x)
        Q = sp.Poly(den, x)
    except sp.PolynomialError:
        return None
    if P.domain not in (sp.ZZ, sp.QQ) or Q.domain not in (sp.ZZ, sp.QQ):
        return None
    return P, Q


def _poly_real_roots(P: sp.Poly) -> list[sp.Expr]:
    """All real roots of a QQ polynomial, exact and complete (distinct).

    Irreducible factors of degree <= 2 give radicals (``sqrt(3)``); higher ones give
    ``CRootOf`` (the count per factor from Sturm's theorem, ``count_roots``) — the
    radicals of a cubic or quartic are unreadable and slow to compute with.
    """
    if P.is_zero or P.degree() <= 0:
        return []
    roots: list[sp.Expr] = []
    for f, _mult in P.factor_list()[1]:
        if f.degree() <= 2:
            new = [r for r in sp.roots(f, filter="R")]
        else:
            new = [sp.CRootOf(f.as_expr(), i) for i in range(f.count_roots())]
        for r in new:
            if r not in roots:
                roots.append(r)
    return roots


def _zeros_in(g: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr) -> tuple[list[sp.Expr], bool]:
    """Real zeros of ``g`` in [lo, hi]: (points, complete?).

    ``complete`` is True only when the list is PROVED complete (exact real-root
    isolation of a rational function). Otherwise the points come from SymPy's
    ``solveset`` and the cell certification is what guarantees nothing was missed.
    """
    parts = _rational_parts(g, x)
    if parts is not None:
        P, _ = parts
        pts = [r for r in _poly_real_roots(P) if _inside(r, lo, hi)]
        return pts, True
    window = sp.Interval(lo, hi)
    try:
        sol = sp.solveset(sp.Eq(g, 0), x, window)
    except (NotImplementedError, ValueError, TypeError) as exc:
        raise SolveSetUndecided(
            f"cannot find the zeros of {g} (SymPy refused)", reason="zeros-not-found"
        ) from exc
    if sol == sp.S.EmptySet:
        return [], False
    if isinstance(sol, sp.FiniteSet):
        pts = [p for p in sol if p.is_real is not False and _inside(p, lo, hi)]
        return pts, False
    raise SolveSetUndecided(
        f"the zeros of {g} on {window} are not a finite set SymPy can list "
        f"({type(sol).__name__}) — give a bounded window",
        reason="zeros-not-finite",
    )


def _inside(p: sp.Expr, lo: sp.Expr, hi: sp.Expr) -> bool:
    """``lo <= p <= hi`` for an exact real ``p`` (False for a non-real one)."""
    if p.is_real is False or p.has(sp.I):
        return False
    lo, hi = sp.sympify(lo), sp.sympify(hi)
    a = True if p == lo else _lt(lo, p)
    b = True if p == hi else _lt(p, hi)
    if a is None and _is_zero(p - lo):
        a = True
    if b is None and _is_zero(hi - p):
        b = True
    if a is None or b is None:
        raise SolveSetUndecided(f"cannot order {p} against [{lo}, {hi}]", reason="order")
    return a and b


def _sorted_unique(points: Iterable[sp.Expr]) -> list[sp.Expr]:
    """Exact points in increasing order; equal ones (proved equal) merged."""
    pts = list(points)
    keyed = sorted(pts, key=lambda p: _num(p))
    out: list[sp.Expr] = []
    for p in keyed:
        if out:
            q = out[-1]
            if abs(_num(p) - _num(q)) < mpmath.mpf(10) ** -40:
                z = _is_zero(p - q)
                if z:
                    # keep the simpler spelling
                    if sp.count_ops(p) < sp.count_ops(q):
                        out[-1] = p
                    continue
                raise SolveSetUndecided(
                    f"cannot separate the cut points {q} and {p}", reason="order"
                )
        out.append(p)
    return out


# --- interval enclosures ---------------------------------------------------------------

class _NoEnclosure(Exception):
    """An enclosure could not be formed (unsupported node, empty domain)."""


@contextmanager
def _iv_prec() -> Iterator[None]:
    """mpmath's interval context has no ``workprec``: set and restore ``iv.prec``."""
    old = iv.prec
    iv.prec = _IV_PREC
    try:
        yield
    finally:
        iv.prec = old


def _iv_const(c: sp.Expr) -> Any:
    if c.is_Integer:
        return iv.mpf(int(c))
    if c.is_Rational:
        return iv.mpf(int(c.p)) / int(c.q)
    if c == sp.pi:
        return iv.pi
    if c == sp.E:
        return iv.e
    raise _NoEnclosure(c)


def _iv_atan(X: Any) -> Any:
    def one(v: Any, down: bool) -> mpmath.mpf:
        if v == mpmath.inf:
            r = mpmath.pi / 2
        elif v == -mpmath.inf:
            r = -mpmath.pi / 2
        else:
            with mpmath.workprec(_IV_PREC + 30):
                r = mpmath.atan(v)
        slack = mpmath.mpf(2) ** (-_IV_PREC + 4)
        return r - slack if down else r + slack

    return iv.mpf([one(X.a, True), one(X.b, False)])


def _enc(e: sp.Expr, x: sp.Symbol, X: Any) -> Any:
    """Outward-rounded enclosure of ``e`` over ``X`` restricted to the real domain.

    Arguments of ``sqrt``/``log``/fractional powers are CLIPPED at 0: callers only
    enclose over cells already proved to lie inside the domain, where those arguments
    are >= 0 (> 0 for ``log``), so values below 0 in an argument's enclosure are
    over-estimation, never real values.
    """
    if e == x:
        return X
    if not e.has(x):
        if e.is_Number or e in (sp.pi, sp.E):
            return _iv_const(e)
    if isinstance(e, sp.Add):
        acc = _enc(e.args[0], x, X)
        for a in e.args[1:]:
            acc = acc + _enc(a, x, X)
        return acc
    if isinstance(e, sp.Mul):
        acc = _enc(e.args[0], x, X)
        for a in e.args[1:]:
            acc = acc * _enc(a, x, X)
        return acc
    if isinstance(e, sp.Pow):
        b, p = e.args
        if p.is_Integer:
            B = _enc(b, x, X)
            n = int(p)
            if n >= 0:
                return B ** n
            return 1 / (B ** (-n))
        if p.is_Rational:
            B = _clip0(_enc(b, x, X))
            r = iv.mpf(int(p.p)) / int(p.q)
            if p > 0:
                return B ** r
            return 1 / (B ** (-r))
        if not b.has(x) and b.is_positive:
            return iv.exp(_enc(p, x, X) * iv.log(_iv_const(b) if b.is_Number or b in (sp.pi, sp.E) else _enc(b, x, X)))
        B = _clip0(_enc(b, x, X))
        return iv.exp(_enc(p, x, X) * iv.log(B))
    if isinstance(e, sp.exp):
        return iv.exp(_enc(e.args[0], x, X))
    if isinstance(e, sp.log):
        return iv.log(_clip0(_enc(e.args[0], x, X)))
    if isinstance(e, sp.Abs):
        return abs(_enc(e.args[0], x, X))
    if isinstance(e, sp.sin):
        return iv.sin(_enc(e.args[0], x, X))
    if isinstance(e, sp.cos):
        return iv.cos(_enc(e.args[0], x, X))
    if isinstance(e, sp.tan):
        return iv.tan(_enc(e.args[0], x, X))
    if isinstance(e, sp.atan):
        return _iv_atan(_enc(e.args[0], x, X))
    if isinstance(e, sp.cot):
        A = _enc(e.args[0], x, X)
        return iv.cos(A) / iv.sin(A)
    if isinstance(e, sp.sec):
        return 1 / iv.cos(_enc(e.args[0], x, X))
    if isinstance(e, sp.csc):
        return 1 / iv.sin(_enc(e.args[0], x, X))
    raise _NoEnclosure(e)


def _clip0(B: Any) -> Any:
    if B.b < 0:
        raise _NoEnclosure("argument below zero on a cell said to be in the domain")
    if B.a < 0:
        return iv.mpf([0, B.b])
    return B


def _enc_sign(forms: Sequence[sp.Expr], x: sp.Symbol, X: Any) -> "int | None":
    """+1 / -1 if some form provably keeps that sign over X, else None."""
    for f in forms:
        try:
            with _iv_prec():
                s = _struct_sign(f, x, X)
        except (_NoEnclosure, ZeroDivisionError, ValueError, mpmath.libmp.ComplexResult):
            continue
        if s is not None:
            return s
    return None


def _struct_sign(e: sp.Expr, x: sp.Symbol, X: Any) -> "int | None":
    """Sign of ``e`` over X from its STRUCTURE first, its enclosure second.

    ``exp`` of a real argument is positive however wide the argument's enclosure; a
    product has the product of its factors' signs; a sum of terms of one strict sign
    has that sign. This is what proves ``x*exp(-x) > 0`` on [M, oo), where every
    enclosure of the product touches 0 (``inf * 0``) although no value does.
    """
    if isinstance(e, sp.exp):
        _enc(e.args[0], x, X)  # the argument must still be enclosable (real)
        return 1
    if isinstance(e, sp.Mul):
        sign = 1
        for a in e.args:
            sa = _struct_sign(a, x, X)
            if sa is None:
                return _plain_sign(e, x, X)
            sign *= sa
        return sign
    if isinstance(e, sp.Pow) and e.args[1].is_Integer:
        sb = _struct_sign(e.args[0], x, X)
        if sb is None:
            return _plain_sign(e, x, X)
        return sb ** abs(int(e.args[1])) if int(e.args[1]) % 2 else 1
    if isinstance(e, sp.Add):
        signs = set()
        for a in e.args:
            sa = _struct_sign(a, x, X)
            if sa is None:
                return _plain_sign(e, x, X)
            signs.add(sa)
        if len(signs) == 1:
            return signs.pop()
    return _plain_sign(e, x, X)


def _plain_sign(e: sp.Expr, x: sp.Symbol, X: Any) -> "int | None":
    Y = _enc(e, x, X)
    if Y.a > 0:
        return 1
    if Y.b < 0:
        return -1
    return None


def _forms(h: sp.Expr) -> list[sp.Expr]:
    """Equal spellings of ``h`` whose enclosures differ (dependency problem).

    Besides ``factor``/``together``/``expand``: for ``h = c*sqrt(A) + R`` the
    conjugate form ``(c**2*A - R**2) / (c*sqrt(A) - R)``. It is equal to ``h`` wherever
    its denominator is nonzero, and where the denominator vanishes so does its
    numerator — so an enclosure that excludes 0 from the numerator also proves the
    denominator nonzero, and the form is sound wherever its sign is read.
    """
    out = [h]
    cands = []
    for fn in (sp.factor, sp.together, sp.expand):
        try:
            cands.append(fn(h))
        except Exception:  # noqa: BLE001
            continue
    conj = _conjugate_form(h)
    if conj is not None:
        cands.append(conj)
    for g in cands:
        if g not in out and sp.count_ops(g) < 4 * sp.count_ops(h) + 20:
            out.append(g)
    return out


def _conjugate_form(h: sp.Expr) -> "sp.Expr | None":
    if not isinstance(h, sp.Add):
        return None
    roots = [t for t in h.args if _sqrt_term(t) is not None]
    if len(roots) != 1:
        return None
    t = roots[0]
    c, A = _sqrt_term(t)  # type: ignore[misc]
    R = h - t
    if R.has(sp.sqrt(A)):
        return None
    return sp.Mul(sp.expand(c**2 * A - R**2), sp.Pow(c * sp.sqrt(A) - R, -1), evaluate=False)


def _sqrt_term(t: sp.Expr) -> "tuple[sp.Expr, sp.Expr] | None":
    c, rest = t.as_coeff_Mul()
    if isinstance(rest, sp.Pow) and rest.args[1] == sp.Rational(1, 2) and rest.args[0].free_symbols:
        return c, rest.args[0]
    return None


def _iv_point(p: sp.Expr) -> Any:
    """A tight enclosure of an exact real number."""
    if p == sp.oo:
        return iv.mpf(mpmath.inf)
    if p == -sp.oo:
        return iv.mpf(-mpmath.inf)
    with mpmath.workdps(_ORDER_DPS):
        v = mpmath.mpf(sp.N(p, _ORDER_DPS))
    slack = abs(v) * mpmath.mpf(10) ** -(_ORDER_DPS - 5) + mpmath.mpf(10) ** -(_ORDER_DPS - 5)
    return iv.mpf([v - slack, v + slack])


def _bisect_sign(forms: Sequence[sp.Expr], x: sp.Symbol, lo: mpmath.mpf, hi: mpmath.mpf) -> "int | None":
    """Prove one sign of ``h`` on the closed [lo, hi] by adaptive bisection."""
    stack: list[tuple[mpmath.mpf, mpmath.mpf, int]] = [(lo, hi, 0)]
    sign: "int | None" = None
    pieces = 0
    while stack:
        a, b, depth = stack.pop()
        pieces += 1
        if pieces > _MAX_PIECES:
            return None
        s = _enc_sign(forms, x, iv.mpf([a, b]))
        if s is not None:
            if sign is None:
                sign = s
            elif s != sign:
                return None  # cannot happen for a zero-free cell: refuse, do not guess
            continue
        if depth >= _MAX_DEPTH:
            return None
        m = (a + b) / 2
        stack.append((m, b, depth + 1))
        stack.append((a, m, depth + 1))
    return sign


def _one_sided_limit(h: sp.Expr, u: sp.Symbol) -> "sp.Expr | None":
    try:
        L = sp.limit(h, u, 0, "+")
    except (NotImplementedError, ValueError, sp.PoleError, TypeError):
        return None
    if L.has(sp.Limit, sp.nan, sp.zoo) or isinstance(L, sp.AccumBounds):
        return None
    return L


def _end_sign(hu: sp.Expr, u: sp.Symbol, delta: sp.Rational) -> "int | None":
    """Sign of ``hu(u)`` on (0, delta], ``u`` the distance from a cell's endpoint.

    1. the enclosure over [0, delta] excludes 0;
    2. otherwise for m = 1..3: the m-th derivative's enclosure excludes 0 and the
       one-sided limits of hu, hu', ..., hu^(m-1) at 0+ are all 0 — then every lower
       derivative is strictly monotone from 0, so they all share the m-th sign;
    3. otherwise m = 1 with a nonzero limit L: hu monotone; increasing from L >= 0 is
       positive, decreasing from L <= 0 is negative.
    """
    U = iv.mpf([0, _num(delta)])
    forms = _forms(hu)
    s = _enc_sign(forms, u, U)
    if s is not None:
        return s
    derivs = [hu]
    for m in range(1, 4):
        dm = sp.diff(derivs[-1], u)
        derivs.append(dm)
        sm = _enc_sign(_forms(dm), u, U)
        if sm is None:
            continue
        limits = [_one_sided_limit(d, u) for d in derivs[:-1]]
        if all(L is not None and _is_zero(L) for L in limits):
            return sm
        if m == 1 and limits[0] is not None:
            L = limits[0]
            sL: "int | None" = (1 if L == sp.oo else -1) if L in (sp.oo, -sp.oo) else _sign_at(L)
            if sL is None:
                return None
            if sm > 0 and sL >= 0:
                return 1
            if sm < 0 and sL <= 0:
                return -1
        return None
    return None


def _tail_sign(h: sp.Expr, x: sp.Symbol, M: sp.Rational) -> "int | None":
    """Sign of ``h`` on [M, +oo): direct enclosure, or monotone + limit at +oo."""
    T = iv.mpf([_num(M), mpmath.inf])
    s = _enc_sign(_forms(h), x, T)
    if s is not None:
        return s
    d = sp.diff(h, x)
    sd = _enc_sign(_forms(d), x, T)
    if sd is None:
        return None
    sM = _sign_at(h.subs(x, M))
    if sM is None:
        return None
    try:
        L = sp.limit(h, x, sp.oo)
    except (NotImplementedError, ValueError, sp.PoleError, TypeError):
        L = None
    if sd > 0 and sM > 0:
        return 1
    if sd < 0 and sM < 0:
        return -1
    if L is None or L.has(sp.Limit, sp.nan, sp.zoo) or isinstance(L, sp.AccumBounds):
        return None
    sL = 1 if L == sp.oo else -1 if L == -sp.oo else _sign_at(L)
    if sL is None:
        return None
    if sd > 0 and sL <= 0:
        return -1
    if sd < 0 and sL >= 0:
        return 1
    return None


def _interval_cell_sign(h: sp.Expr, x: sp.Symbol, a: sp.Expr, b: sp.Expr) -> "int | None":
    """Prove the sign of ``h`` on the open cell (a, b) — see the module docstring."""
    u = sp.Symbol("u", positive=True)
    if a == -sp.oo and b == sp.oo:
        # split at 0: (-oo, 0] and [0, oo) — neither end is a cut point, so the point
        # 0 itself is covered by the interior enclosure of the two tails below
        left = _tail_cover(h.subs(x, -x), x, sp.Integer(0))
        right = _tail_cover(h, x, sp.Integer(0))
        if left is None or right is None or left != right:
            return None
        return left
    if b == sp.oo:
        return _ray_sign(h, x, a)
    if a == -sp.oo:
        return _ray_sign(h.subs(x, -x), x, -b)
    width = b - a
    delta = _nice_fraction(width / 4)
    hl = sp.expand(h.subs(x, a + u)) if _small(h) else h.subs(x, a + u)
    hr = sp.expand(h.subs(x, b - u)) if _small(h) else h.subs(x, b - u)
    sl = _end_sign(hl, u, delta)
    sr = _end_sign(hr, u, delta)
    if sl is None or sr is None or sl != sr:
        return None
    lo = _iv_point(a + delta).a
    hi = _iv_point(b - delta).b
    si = _bisect_sign(_forms(h), x, lo, hi)
    if si is None or si != sl:
        return None
    return sl


def _ray_sign(h: sp.Expr, x: sp.Symbol, a: sp.Expr) -> "int | None":
    """Sign of ``h`` on the open ray (a, +oo), ``a`` finite."""
    u = sp.Symbol("u", positive=True)
    delta = sp.Rational(1, 2)
    hl = sp.expand(h.subs(x, a + u)) if _small(h) else h.subs(x, a + u)
    sl = _end_sign(hl, u, delta)
    if sl is None:
        return None
    start = a + delta
    rest = _tail_cover(h, x, start)
    if rest is None or rest != sl:
        return None
    return sl


def _tail_cover(h: sp.Expr, x: sp.Symbol, start: sp.Expr) -> "int | None":
    """Sign on [start, +oo): bisection on [start, M] and a tail argument beyond M."""
    s0 = _num(start)
    for k in range(6):
        M = sp.Integer(int(mpmath.ceil(abs(s0)))) + 2 ** (2 * k + 1)
        st = _tail_sign(h, x, M)
        if st is None:
            continue
        lo = _iv_point(start).a
        si = _bisect_sign(_forms(h), x, lo, _num(M))
        if si is None or si != st:
            return None
        return st
    return None


def _small(h: sp.Expr) -> bool:
    return sp.count_ops(h) < 60


def _nice_fraction(v: sp.Expr) -> sp.Rational:
    """A simple positive rational not above ``v`` (v > 0)."""
    f = _num(v)
    for d in (1, 2, 4, 8, 16, 64, 256, 1024, 2**16, 2**32):
        n = int(mpmath.floor(f * d))
        if n >= 1:
            return sp.Rational(n, d)
    return sp.Rational(1, 2**40)


def _rational_between(a: sp.Expr, b: sp.Expr) -> sp.Rational:
    """A short exact rational strictly inside (a, b)."""
    if a == -sp.oo and b == sp.oo:
        return sp.Integer(0)
    if a == -sp.oo:
        return sp.Integer(int(mpmath.floor(_num(b)))) - 1
    if b == sp.oo:
        return sp.Integer(int(mpmath.ceil(_num(a)))) + 1
    na, nb = _num(a), _num(b)
    mid = (na + nb) / 2
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64, 100, 128, 1000] + [2**k for k in range(8, 200, 8)]:
        lo_n = int(mpmath.floor(na * d)) + 1
        hi_n = int(mpmath.ceil(nb * d)) - 1
        if lo_n > hi_n:
            continue
        n = min(max(int(mpmath.nint(mid * d)), lo_n), hi_n)
        q = sp.Rational(n, d)
        if _lt(a, q) and _lt(q, b):
            return q
    raise SolveSetUndecided(f"no rational sample found in ({a}, {b})", reason="sample")


# --- the decomposition -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EndpointCheck:
    """One cut point, decided exactly.

    ``in_domain``  every domain constraint holds at the point (exact signs)
    ``sign``       sign of ``lhs - rhs`` there (None outside the domain)
    ``member``     the point belongs to the solution set
    ``exact``      always True — kept explicit so a reader of ``to_dict`` sees it
    """

    point: str
    approx: str
    in_domain: bool
    sign: "int | None"
    member: bool
    exact: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"point": self.point, "approx": self.approx, "in_domain": self.in_domain,
                "sign": self.sign, "member": self.member, "exact": self.exact}


@dataclass(frozen=True, slots=True)
class CellCheck:
    """One open cell (lo, hi) between consecutive cut points.

    ``sample`` is an exact rational inside the cell; ``sample_sign`` is the exact sign
    of ``lhs - rhs`` there. ``sign`` is the PROVED sign on the whole cell, by
    ``method``: ``exact-roots``, ``interval-arithmetic`` or ``outside-domain``.
    """

    lo: str
    hi: str
    sample: str
    in_domain: bool
    sign: "int | None"
    sample_sign: "int | None"
    member: bool
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "sample": self.sample, "in_domain": self.in_domain,
                "sign": self.sign, "sample_sign": self.sample_sign, "member": self.member,
                "method": self.method}


@dataclass(frozen=True)
class _Decomposition:
    points: list[EndpointCheck]
    cells: list[CellCheck]
    point_values: list[sp.Expr]
    cell_bounds: list[tuple[sp.Expr, sp.Expr]]
    domain: sp.Set
    method: str


def _deabs(h: sp.Expr, x: sp.Symbol, a: sp.Expr, b: sp.Expr, q: sp.Rational) -> sp.Expr:
    """``h`` on the cell (a, b) with every ``Abs(g)`` replaced by ``±g``.

    Each ``g`` keeps one sign on the cell — its zeros are cut points — and that sign is
    PROVED (exactly for a rational ``g``, by enclosure otherwise) before ``Abs`` is
    dropped. Innermost first, so a nested ``abs`` is resolved before its parent.
    """
    while True:
        inner = [n for n in sp.preorder_traversal(h)
                 if isinstance(n, sp.Abs) and not n.args[0].has(sp.Abs) and n.args[0].has(x)]
        if not inner:
            return h
        node = inner[0]
        g = node.args[0]
        if _rational_parts(g, x) is not None and not _roots_inside(g, x, a, b):
            sg = _sign_at(g.subs(x, q))
        else:
            sg = _interval_cell_sign(g, x, a, b)
        if not sg:
            raise SolveSetUndecided(f"cannot prove the sign of {g} on ({a}, {b})", reason="cell-abs")
        h = h.xreplace({node: g if sg > 0 else -g})


def _roots_inside(g: sp.Expr, x: sp.Symbol, a: sp.Expr, b: sp.Expr) -> list[sp.Expr]:
    """Real roots of the numerator/denominator of a rational ``g`` strictly in (a, b)."""
    parts = _rational_parts(g, x)
    if parts is None:
        return []
    out = []
    for P in parts:
        for r in _poly_real_roots(P):
            if _lt(a, r) and _lt(r, b):
                out.append(r)
    return out


def _constraint_sign_on_cell(g: sp.Expr, x: sp.Symbol, a: sp.Expr, b: sp.Expr, q: sp.Rational) -> "int | None":
    if _rational_parts(g, x) is not None and not _roots_inside(g, x, a, b):
        return _sign_at(g.subs(x, q))
    return _interval_cell_sign(g, x, a, b)


def _decompose(h: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr, extra: Iterable[sp.Expr],
               keep: Callable[[int], bool]) -> _Decomposition:
    """Sign decomposition of ``h`` on [lo, hi] (``lo``/``hi`` may be infinite).

    ``keep(sign)`` says whether a sign satisfies the relation. Finite window ends are
    themselves cut points (closed window). If a cell, once its ``abs`` are resolved,
    turns out rational with a root inside it, that root becomes a cut point and the
    decomposition is redone (at most three rounds).
    """
    cons = _constraints(h, x)
    cuts: list[sp.Expr] = []
    zh, complete = _zeros_in(h, x, lo, hi)
    cuts += zh
    # Roots from exact isolation of a numerator ARE zeros (the fraction was cancelled,
    # so the denominator does not vanish there): no need to re-prove h(p) = 0 — which,
    # for a quartic's radicals, is a minimal-polynomial computation measured at seconds.
    known_zero: dict[sp.Expr, set[sp.Expr]] = {h: set(zh) if complete else set()}
    for c in cons:
        zg, cg = _zeros_in(c.g, x, lo, hi)
        cuts += zg
        known_zero[c.g] = set(zg) if cg else set()
    for node in sp.preorder_traversal(h):
        if isinstance(node, sp.Abs) and node.args[0].has(x):
            cuts += _zeros_in(node.args[0], x, lo, hi)[0]
    for p in extra:
        if p.is_finite and p.is_real is not False and not p.has(sp.I) and _inside(p, lo, hi):
            cuts.append(p)
    for end in (lo, hi):
        if end.is_finite:
            cuts.append(end)
    for _round in range(3):
        pts = _sorted_unique(cuts)
        seq = ([lo] if not lo.is_finite else []) + pts + ([hi] if not hi.is_finite else [])
        found: list[sp.Expr] = []
        cells: list[CellCheck] = []
        cell_bounds: list[tuple[sp.Expr, sp.Expr]] = []
        methods: set[str] = set()
        for a, b in zip(seq, seq[1:]):
            q = _rational_between(a, b)
            in_dom = True
            for c in cons:
                sg = _constraint_sign_on_cell(c.g, x, a, b, q)
                if sg is None or (c.need != "nonzero" and sg == 0):
                    raise SolveSetUndecided(
                        f"cannot prove the sign of {c.g} on ({a}, {b})", reason="cell-domain")
                if not c.holds(sg):
                    in_dom = False
                    break
            if not in_dom:
                cells.append(CellCheck(_s(a), _s(b), _s(q), False, None, None, False, "outside-domain"))
                cell_bounds.append((a, b))
                continue
            hc = _deabs(h, x, a, b, q)
            inside = _roots_inside(hc, x, a, b)
            if inside:
                found += inside
                continue
            s_q = _sign_at(hc.subs(x, q))
            if s_q is None or s_q == 0:
                raise SolveSetUndecided(f"{h} vanishes or is undecidable at the sample {q}",
                                        reason="cell-sample")
            sign: int
            if _rational_parts(hc, x) is not None:
                sign, method = s_q, "exact-roots"
            else:
                sign_c = _interval_cell_sign(hc, x, a, b)
                if sign_c is None:
                    raise SolveSetUndecided(
                        f"cannot prove that {h} keeps one sign on ({a}, {b}) — a zero may be "
                        f"missing from the cut points", reason="cell-sign")
                if sign_c != s_q:
                    raise SolveSetUndecided(f"inconsistent signs on ({a}, {b})", reason="cell-sign")
                sign, method = sign_c, "interval-arithmetic"
            methods.add(method)
            cells.append(CellCheck(_s(a), _s(b), _s(q), True, sign, s_q, keep(sign), method))
            cell_bounds.append((a, b))
        if not found:
            break
        cuts = pts + found
    else:
        raise SolveSetUndecided("the cut points did not stabilise", reason="cuts")

    point_checks: list[EndpointCheck] = []
    for p in pts:
        in_dom = True
        for c in cons:
            sg = 0 if p in known_zero[c.g] else _sign_at(c.g.subs(x, p))
            if sg is None:
                raise SolveSetUndecided(f"cannot decide the sign of {c.g} at {p}", reason="point-sign")
            if not c.holds(sg):
                in_dom = False
                break
        psign: "int | None" = None
        if in_dom:
            psign = 0 if p in known_zero[h] else _sign_at(h.subs(x, p))
            if psign is None:
                raise SolveSetUndecided(f"cannot decide the sign of {h} at {p}", reason="point-sign")
        point_checks.append(EndpointCheck(_s(p), _approx(p), in_dom, psign,
                                          bool(in_dom and psign is not None and keep(psign))))
    dom_parts: list[sp.Set] = []
    for (a, b), cell in zip(cell_bounds, cells):
        if cell.in_domain:
            dom_parts.append(sp.Interval.open(a, b))
    for p, pc in zip(pts, point_checks):
        if pc.in_domain:
            dom_parts.append(sp.FiniteSet(p))
    domain = sp.Union(*dom_parts) if dom_parts else sp.S.EmptySet
    method = "interval-arithmetic" if "interval-arithmetic" in methods else "exact-roots"
    return _Decomposition(point_checks, cells, pts, cell_bounds, domain, method)


def _assemble(dec: _Decomposition) -> sp.Set:
    parts: list[sp.Set] = []
    for (a, b), c in zip(dec.cell_bounds, dec.cells):
        if c.member:
            parts.append(sp.Interval.open(a, b))
    for p, pc in zip(dec.point_values, dec.points):
        if pc.member:
            parts.append(sp.FiniteSet(p))
    return sp.Union(*parts) if parts else sp.S.EmptySet


def _intervals(S: sp.Set) -> list[tuple[str, str, bool, bool]]:
    """(lo, hi, lo_closed, hi_closed) for each connected piece; a point is lo == hi."""
    if S == sp.S.EmptySet:
        return []
    pieces = S.args if isinstance(S, sp.Union) else (S,)
    out: list[tuple[str, str, bool, bool]] = []
    for P in pieces:
        if isinstance(P, sp.Interval):
            out.append((_s(P.start), _s(P.end), not P.left_open and P.start.is_finite,
                        not P.right_open and P.end.is_finite))
        elif isinstance(P, sp.FiniteSet):
            for p in P:
                out.append((_s(p), _s(p), True, True))
        elif P == sp.S.Reals:
            out.append(("-oo", "oo", False, False))
        else:  # pragma: no cover — _assemble only builds the shapes above
            raise SolveSetUndecided(f"unexpected set shape {P}", reason="shape")
    out.sort(key=lambda t: _num(sp.sympify(t[0])))
    return out


def _approx_pieces(iv_list: Sequence[tuple[str, str, bool, bool]]) -> list[tuple[str, str, bool, bool]]:
    def short(t: str) -> str:
        v = sp.sympify(t)
        if v.is_Rational or v in (sp.oo, -sp.oo):
            return t
        return sp.sstr(sp.N(v, 10))
    return [(short(a), short(b), lc, hc) for a, b, lc, hc in iv_list]


def _pretty(iv_list: Sequence[tuple[str, str, bool, bool]]) -> str:
    if not iv_list:
        return "∅"
    parts = []
    for lo, hi, lc, hc in iv_list:
        if lo == hi:
            parts.append("{" + lo + "}")
        elif lo == "-oo" and hi == "oo":
            parts.append("ℝ")
        else:
            parts.append(("[" if lc else "(") + f"{lo}, {hi}" + ("]" if hc else ")"))
    return " ∪ ".join(parts)


# --- inequalities ----------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InequalityResult:
    """A PROVED solution set of a one-variable real inequality.

    ``solution``     the set, as SymPy text (``Interval.open(1, 2)``)
    ``pretty``       the same for a human: ``(1, 2)``, ``(-oo, -2) ∪ [1, oo)``, ``∅``
    ``approx``       ``pretty`` with every irrational endpoint as 10 significant digits
                     (``(-1.167303978, oo)`` for a ``CRootOf``)
    ``intervals``    ``(lo, hi, lo_closed, hi_closed)`` per piece; a point has lo == hi
    ``domain``       the natural real domain inside ``within``, as SymPy text
    ``conditions``   the constraints that define it (``x + 2 != 0``, ``x > 0``)
    ``endpoints``    every cut point, decided exactly (``EndpointCheck``)
    ``cells``        every open cell between cut points, with an exact sample (``CellCheck``)
    ``method``       ``exact-roots`` or ``interval-arithmetic`` (the weaker one used)
    ``sympy_*``      SymPy's own ``solveset`` answer and, if it is wrong, a witness
    """

    relation: str
    var: str
    solution: str
    pretty: str
    intervals: tuple[tuple[str, str, bool, bool], ...]
    domain: str
    conditions: tuple[str, ...]
    within: "tuple[str, str] | None"
    endpoints: tuple[EndpointCheck, ...]
    cells: tuple[CellCheck, ...]
    method: str
    sympy_answer: "str | None"
    sympy_agrees: "bool | None"
    sympy_witness: "str | None"
    approx: str = ""
    proof: str = "proof"
    kind: str = "inequality"

    @property
    def verified(self) -> bool:
        return True

    def as_set(self) -> sp.Set:
        """The solution as a SymPy set over a real symbol named ``var``."""
        return sp.sympify(self.solution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "relation": self.relation,
            "var": self.var,
            "solution": self.solution,
            "pretty": self.pretty,
            "approx": self.approx,
            "intervals": [list(t) for t in self.intervals],
            "domain": self.domain,
            "conditions": list(self.conditions),
            "within": list(self.within) if self.within else None,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "cells": [c.to_dict() for c in self.cells],
            "method": self.method,
            "proof": self.proof,
            "verdict": "verified",
            "sympy": {"answer": self.sympy_answer, "agrees": self.sympy_agrees,
                      "witness": self.sympy_witness},
        }


def _split_relation(text: str) -> tuple[str, str, str]:
    """``"lhs OP rhs"`` -> (lhs, op, rhs); exactly one relational operator."""
    for sym, op in _SYM_OPS:
        if sym in text:
            lhs, rest = text.split(sym, 1)
            for other, _ in _SYM_OPS:
                if other in rest:
                    raise ParseError(
                        "solve_inequality: one relation at a time — write a chained "
                        "inequality 'a < f < b' as two calls and intersect"
                    )
            if not lhs.strip() or not rest.strip():
                raise ParseError(f"solve_inequality: a side of '{text}' is empty")
            return lhs, op, rest
    if "!=" in text or "=" in text:
        raise ParseError("solve_inequality: expected one of <, <=, >, >= — use solve for equations")
    raise ParseError(f"solve_inequality: no relation (<, <=, >, >=) in '{text}'")


def _relation_parts(relation: "str | sp.Rel", x: sp.Symbol) -> tuple[sp.Expr, str, str]:
    if isinstance(relation, sp.Rel):
        ops = {sp.StrictLessThan: "<", sp.LessThan: "<=", sp.StrictGreaterThan: ">",
               sp.GreaterThan: ">="}
        op = ops.get(type(relation))
        if op is None:
            raise DomainError(f"solve_inequality: unsupported relation {relation}")
        h = _to_real(relation.lhs - relation.rhs, x)
        return h, op, f"{_s(relation.lhs)} {op} {_s(relation.rhs)}"
    if not isinstance(relation, str):
        raise DomainError(f"solve_inequality: expected text, not {type(relation).__name__}")
    lhs, op, rhs = _split_relation(relation)
    L = _to_real(parse(lhs).sy, x)
    R = _to_real(parse(rhs).sy, x)
    return L - R, op, f"{_s(L)} {op} {_s(R)}"


_KEEP: dict[str, Callable[[int], bool]] = {
    "<": lambda s: s < 0, "<=": lambda s: s <= 0, ">": lambda s: s > 0, ">=": lambda s: s >= 0,
}


def _sympy_cross_check(theirs: "sp.Set | None", dec: _Decomposition) -> tuple["str | None", "bool | None", "str | None"]:
    """SymPy's own answer and where (if anywhere) it is wrong."""
    if theirs is None:
        return "error", None, None
    if isinstance(theirs, sp.ConditionSet) or theirs.has(sp.ConditionSet):
        return _s(theirs), None, None
    probes = [sp.sympify(c.sample) for c in dec.cells] + list(dec.point_values)
    members = [c.member for c in dec.cells] + [p.member for p in dec.points]
    for p, m in zip(probes, members):
        try:
            c = theirs.contains(p)
        except Exception:  # noqa: BLE001
            return _s(theirs), None, None
        if c not in (sp.true, sp.false):
            return _s(theirs), None, None
        if bool(c) != m:
            return _s(theirs), False, _s(p)
    return _s(theirs), True, None


@under_budget("solve_inequality")
def solve_inequality(
    relation: "str | sp.Rel",
    var: str = "x",
    *,
    within: "tuple[Any, Any] | None" = None,
) -> InequalityResult:
    """Solve a real inequality in one variable and PROVE the answer.

        >>> r = solve_inequality("x^2 - 3*x + 2 < 0")
        >>> r.pretty, r.method
        ('(1, 2)', 'exact-roots')
        >>> solve_inequality("log(x) > 1").solution
        'Interval.open(E, oo)'

    ``within=(a, b)`` restricts the variable to the closed window [a, b] (needed for
    trigonometric inequalities, which have infinitely many cells on the line).
    Raises ``SolveSetUndecided`` when no proof is found, ``UnsupportedFormError`` for
    functions outside the supported list, ``ParseError``/``DomainError`` for bad input.
    """
    x = _real_symbol(var)
    h, op, shown = _relation_parts(relation, x)
    h = _exact(h, "solve_inequality")
    extra_syms = h.free_symbols - {x}
    if extra_syms:
        raise UnsupportedFormError(
            f"solve_inequality: one variable only; also found {sorted(map(str, extra_syms))}"
        )
    if h.has(sp.I):
        raise DomainError("solve_inequality: complex numbers have no order")
    _check_functions(h, "solve_inequality")
    if within is None:
        lo, hi = -sp.oo, sp.oo
        window: sp.Set = sp.S.Reals
        within_s = None
    else:
        if not isinstance(within, (tuple, list)) or len(within) != 2:
            raise DomainError("solve_inequality: within must be a pair (a, b)")
        lo = _check_constant(_parse_expr(within[0], "within"), "within")
        hi = _check_constant(_parse_expr(within[1], "within"), "within")
        if not _lt(lo, hi):
            raise DomainError(f"solve_inequality: empty window [{lo}, {hi}]")
        window = sp.Interval(lo, hi)
        within_s = (_s(lo), _s(hi))
    if h.has(sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc) and not (lo.is_finite and hi.is_finite):
        raise SolveSetUndecided(
            "solve_inequality: a trigonometric inequality on an unbounded range has "
            "infinitely many cells — give within=(a, b)", reason="periodic-unbounded")
    keep = _KEEP[op]
    if h == 0:
        # the two sides are the same expression after SymPy's automatic evaluation:
        # the relation reads 0 op 0 at every point of the window
        whole = window if keep(0) else sp.S.EmptySet
        pcs = _intervals(whole) if whole != sp.S.Reals else [("-oo", "oo", False, False)]
        return InequalityResult(shown, var, _s(whole), _pretty(pcs), tuple(pcs), _s(window), (),
                                within_s, (), (), "exact-roots", None, None, None, _pretty(pcs))
    # SymPy's own answer is a PROPOSAL of cut points, needed only when h is not a
    # rational function (whose real roots are found exactly and completely). It is not
    # free: measured, solveset on a rational inequality with a quartic numerator spent
    # the whole 120 s budget building casus-irreducibilis radicals. So for a rational
    # h it is asked only when every factor is at most quadratic, as a cross-check.
    parts = _rational_parts(h, x)
    cheap = parts is None or max(parts[0].degree(), parts[1].degree()) <= 2
    extra: list[sp.Expr] = []
    proposed: "sp.Set | None" = None
    if cheap:
        try:
            proposed = sp.solveset(_REL[op](h, 0), x, window)
            if parts is None and _plain_set(proposed):
                extra = [b for b in proposed.boundary if b.is_finite]
        except Exception:  # noqa: BLE001 — SymPy failing to propose is not a failure here
            proposed = None
    dec = _decompose(h, x, lo, hi, extra, keep)
    ours = _assemble(dec)
    pieces = _intervals(ours)
    if cheap:
        sym_ans, agrees, witness = _sympy_cross_check(proposed, dec)
    else:
        sym_ans, agrees, witness = None, None, None
    cons = _constraints(h, x)
    return InequalityResult(
        relation=shown,
        var=var,
        solution=_s(ours),
        pretty=_pretty(pieces),
        intervals=tuple(pieces),
        domain=_s(dec.domain),
        conditions=tuple(c.text(x) for c in cons),
        within=within_s,
        endpoints=tuple(dec.points),
        cells=tuple(dec.cells),
        method=dec.method,
        sympy_answer=sym_ans,
        sympy_agrees=agrees,
        sympy_witness=witness,
        approx=_pretty(_approx_pieces(pieces)),
    )


def _plain_set(S: sp.Set) -> bool:
    if S == sp.S.EmptySet or isinstance(S, (sp.Interval, sp.FiniteSet)) or S == sp.S.Reals:
        return True
    return isinstance(S, sp.Union) and all(_plain_set(a) for a in S.args)


# --- systems -------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SystemResult:
    """A PROVED solution set of a system of equations.

    ``status``      ``unique`` (linear, one point), ``infinite`` (linear, an affine
                    family), ``finite`` (polynomial, finitely many points), ``none``
    ``solutions``   exact values per variable (SymPy text); for ``infinite`` they are
                    in terms of ``parameters`` (free variables of the reduced system)
    ``approx``      decimal values (15 digits) of each solution
    ``real``        per solution: all coordinates real? (None: not decided)
    ``rank`` / ``rank_augmented``   linear systems: ranks of A and [A | b]
    ``method``      ``linear`` or ``groebner``
    ``proof``       how the set was proved — the substitution of every solution, plus:
                    linear: rank argument (n - rank free parameters); ``none``: a left
                    null vector y with y·A = 0 and y·b != 0, or 1 in the Groebner basis;
                    polynomial: the number of distinct solutions equals the dimension of
                    the quotient by the RADICAL ideal (Seidenberg), so none is missing
    """

    equations: tuple[str, ...]
    vars: tuple[str, ...]
    status: Literal["unique", "infinite", "finite", "none"]
    solutions: tuple[dict[str, str], ...]
    approx: tuple[dict[str, str], ...]
    real: tuple["bool | None", ...]
    parameters: tuple[str, ...]
    rank: "int | None"
    rank_augmented: "int | None"
    method: str
    proof: dict[str, Any]
    checks: tuple[dict[str, Any], ...]
    kind: str = "system"

    @property
    def verified(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "equations": list(self.equations),
            "vars": list(self.vars),
            "status": self.status,
            "solutions": [dict(s) for s in self.solutions],
            "approx": [dict(s) for s in self.approx],
            "real": list(self.real),
            "parameters": list(self.parameters),
            "rank": self.rank,
            "rank_augmented": self.rank_augmented,
            "method": self.method,
            "proof": self.proof,
            "checks": [dict(c) for c in self.checks],
            "verdict": "verified",
        }


def _parse_equation(text: "str | sp.Basic") -> tuple[sp.Expr, str]:
    """``"lhs = rhs"`` (or an expression meaning ``= 0``) -> (lhs - rhs, shown)."""
    if isinstance(text, sp.Equality):
        return text.lhs - text.rhs, f"{_s(text.lhs)} = {_s(text.rhs)}"
    if isinstance(text, sp.Basic):
        return sp.sympify(text), f"{_s(text)} = 0"
    if not isinstance(text, str):
        raise DomainError(f"equation must be text, not {type(text).__name__}")
    t = text.replace("==", "=")
    if any(op in t for op in ("<", ">", "≤", "≥", "!=")):
        raise ParseError(f"'{text}': an equation is expected here, not an inequality")
    if t.count("=") > 1:
        raise ParseError(f"'{text}': one '=' per equation")
    if "=" in t:
        lhs, rhs = t.split("=")
        if not lhs.strip() or not rhs.strip():
            raise ParseError(f"'{text}': a side of the equation is empty")
        L, R = parse(lhs).sy, parse(rhs).sy
        return L - R, f"{_s(L)} = {_s(R)}"
    E = parse(t).sy
    return E, f"{_s(E)} = 0"


def _crootof_zero(v: sp.Expr) -> "bool | None":
    """Exact zero test for a polynomial expression in ONE ``CRootOf`` r of q:
    ``v(r) = 0`` iff q divides the numerator of ``v(t)`` — reduced modulo q only when
    q is irreducible (``CRootOf`` builds it from an irreducible factor)."""
    roots = {a for a in v.atoms(sp.CRootOf)}
    if len(roots) != 1:
        return None
    r = roots.pop()
    t = sp.Dummy("t")
    e = v.xreplace({r: t})
    if e.free_symbols - {t}:
        return None
    num, den = sp.fraction(sp.together(e))
    q = sp.Poly(r.poly.as_expr().subs(r.poly.gen, t), t)
    try:
        rem = sp.Poly(num, t).rem(q)
    except sp.PolynomialError:
        return None
    if rem.is_zero:
        return True
    return False if q.is_irreducible else None


def _complex_zero(v: sp.Expr) -> "bool | None":
    """Zero test for a possibly complex closed form: a value that is visibly nonzero
    at 30 digits is nonzero (SymPy's evalf tracks its error); only a value that LOOKS
    like zero goes to an exact test. ``maxn`` bounds evalf, which otherwise raises its
    precision without end on an exact zero."""
    if v.has(sp.CRootOf):
        z = _crootof_zero(v)
        if z is not None:
            return z
    try:
        n = v.evalf(30, maxn=120)
    except (TypeError, ValueError):
        n = None
    if n is not None and n.is_number and abs(n) > sp.Float("1e-20"):
        return False
    if not v.has(sp.CRootOf):
        z = _is_zero(sp.expand(v))
        if z is not None:
            return z
    return None


def _is_real_value(v: sp.Expr) -> "bool | None":
    if v.is_real is not None:
        return bool(v.is_real)
    im = sp.im(sp.expand_complex(v))
    z = _complex_zero(im)
    return z


def _approx_dict(sol: dict[sp.Symbol, sp.Expr]) -> dict[str, str]:
    out = {}
    for k, v in sol.items():
        n = sp.N(v, 15)
        out[str(k)] = sp.sstr(n)
    return out


def _check_solution(polys: Sequence[sp.Expr], dens: Sequence[sp.Expr], sol: dict[sp.Symbol, sp.Expr],
                    params: Sequence[sp.Symbol]) -> dict[str, Any]:
    """Substitute a solution into every equation: the residual must be exactly 0."""
    how = "exact"
    for p in polys:
        r = p.xreplace(sol) if params else p.subs(sol)
        if params:
            if sp.expand(sp.together(r)) != 0 and sp.simplify(r) != 0:
                raise SolveSetUndecided(f"the solution {sol} does not satisfy {p} = 0", reason="substitution")
            continue
        z = _complex_zero(r)
        if z is False:
            raise SolveSetUndecided(f"the solution {sol} does not satisfy {p} = 0", reason="substitution")
        if z is None:
            how = "numeric-60-digits"
    for d in dens:
        r = d.subs(sol)
        if not params and _complex_zero(r) is not False:
            raise SolveSetUndecided(f"the solution {sol} makes the denominator {d} vanish", reason="denominator")
    return {"solution": {str(k): _s(v) for k, v in sol.items()}, "residual": "0", "how": how}


def _linear_system(polys: list[sp.Expr], X: list[sp.Symbol], shown: tuple[str, ...]) -> SystemResult:
    A, b = sp.linear_eq_to_matrix(polys, X)
    Ab = A.row_join(b)
    r, ra = A.rank(), Ab.rank()
    names = tuple(str(v) for v in X)
    if r < ra:
        # a left null vector of A that does not annihilate b: the combination
        # sum_i y_i * eq_i reads 0 = y.b != 0
        for y in A.T.nullspace():
            yb = (y.T * b)[0]
            if sp.simplify(yb) != 0:
                den = sp.ilcm(*[sp.fraction(sp.nsimplify(c))[1] for c in y]) if all(c.is_Rational for c in y) else 1
                y = (y * den).applyfunc(sp.simplify)
                yb = sp.simplify((y.T * b)[0])
                assert (y.T * A).is_zero_matrix
                return SystemResult(
                    shown, names, "none", (), (), (), (), r, ra, "linear",
                    {"argument": "rank(A) < rank([A|b]): the combination of the equations with "
                                 "the coefficients y gives 0 = y·b",
                     "y": [_s(c) for c in y], "combination": f"0 = {_s(yb)}"},
                    (),
                )
        raise SolveSetUndecided("inconsistent system without a certificate", reason="linear")  # pragma: no cover
    sol_set = sp.linsolve((A, b), X)
    sol = next(iter(sol_set))
    mapping = dict(zip(X, sol))
    params = sorted({s for v in sol for s in v.free_symbols if s in X}, key=str)
    check = _check_solution(polys, [], mapping, params)
    status: Literal["unique", "infinite"] = "unique" if r == len(X) else "infinite"
    if len(params) != len(X) - r:  # pragma: no cover — linsolve's own contract
        raise SolveSetUndecided("parametrisation does not match the rank", reason="linear")
    approx = (_approx_dict(mapping),) if status == "unique" else ()
    return SystemResult(
        shown, names, status, ({str(k): _s(v) for k, v in mapping.items()},), approx,
        (True,) if status == "unique" else (None,), tuple(str(p) for p in params), r, ra, "linear",
        {"argument": f"rank(A) = rank([A|b]) = {r}; the solution set is an affine space of "
                     f"dimension {len(X) - r} = n - rank, and the substituted family satisfies "
                     f"every equation identically in its {len(params)} parameter(s)"},
        (check,),
    )


def _standard_monomials(G: sp.GroebnerBasis, n: int) -> int:
    """Number of monomials outside the leading-term ideal (zero-dimensional G)."""
    lms = [sp.Poly(g, *G.gens).monoms(order=G.order)[0] for g in G.exprs]
    bounds = []
    for i in range(n):
        pure = [m[i] for m in lms if all(m[j] == 0 for j in range(n) if j != i) and m[i] > 0]
        if not pure:
            raise SolveSetUndecided("the ideal is not zero-dimensional", reason="dimension")
        bounds.append(min(pure))
    total = 1
    for d in bounds:
        total *= d
    if total > 20000:
        raise SolveSetUndecided("too many standard monomials to count", reason="dimension")
    count = 0

    def rec(i: int, e: list[int]) -> None:
        nonlocal count
        if i == n:
            if not any(all(e[j] >= m[j] for j in range(n)) for m in lms):
                count += 1
            return
        for k in range(bounds[i]):
            e.append(k)
            rec(i + 1, e)
            e.pop()

    rec(0, [])
    return count


def _shape_solutions(G: sp.GroebnerBasis, X: list[sp.Symbol]) -> "list[dict[sp.Symbol, sp.Expr]] | None":
    """Solutions of a lex basis ``[x1 - p1(xn), ..., q(xn)]`` (shape position), else None."""
    last = X[-1]
    exprs = list(G.exprs)
    uni = [g for g in exprs if g.free_symbols <= {last}]
    if len(uni) != 1 or len(exprs) != len(X):
        return None
    q = sp.Poly(uni[0], last)
    if q.domain not in (sp.ZZ, sp.QQ):
        return None
    sol_of: dict[sp.Symbol, sp.Expr] = {}
    for v in X[:-1]:
        lin = [g for g in exprs if g.has(v) and sp.Poly(g, *X).degree(v) == 1
               and g.free_symbols <= {v, last}]
        if len(lin) != 1:
            return None
        a, b = sp.Poly(lin[0], v).all_coeffs()
        if a.has(last) or a == 0:
            return None
        sol_of[v] = -b / a
    out: list[dict[sp.Symbol, sp.Expr]] = []
    for r in q.all_roots():
        if any(r == t[last] for t in out):
            continue
        d = {v: sp.expand(e.subs(last, r)) for v, e in sol_of.items()}
        d[last] = r
        out.append(d)
    return out


def _distinct_solution_count(polys: list[sp.Expr], X: list[sp.Symbol]) -> int:
    """Number of distinct complex solutions of a zero-dimensional system (exact).

    For each variable v, the generator of I ∩ Q[v] is the last element of a lex basis
    with v last; adding the squarefree part of each such polynomial to I gives the
    radical of I (Seidenberg's lemma, characteristic 0). The dimension of Q[X]/radical
    is the number of distinct solutions.
    """
    extra = []
    for v in X:
        order = [w for w in X if w != v] + [v]
        Gv = sp.groebner(polys, *order, order="lex")
        uni = [g for g in Gv.exprs if g.free_symbols <= {v}]
        if not uni:
            raise SolveSetUndecided("no elimination polynomial", reason="dimension")
        extra.append(sp.sqf_part(uni[-1]))
    G = sp.groebner(list(polys) + extra, *X, order="grevlex")
    return _standard_monomials(G, len(X))


def _polynomial_system(polys: list[sp.Expr], dens: list[sp.Expr], X: list[sp.Symbol],
                       shown: tuple[str, ...], real: bool) -> SystemResult:
    names = tuple(str(v) for v in X)
    G = sp.groebner(polys, *X, order="lex")
    if list(G.exprs) == [1]:
        return SystemResult(
            shown, names, "none", (), (), (), (), None, None, "groebner",
            {"argument": "1 is in the ideal generated by the equations (reduced Groebner basis "
                         "[1]), so they have no common solution, not even a complex one",
             "groebner_basis": ["1"]},
            (),
        )
    if not G.is_zero_dimensional:
        raise UnsupportedFormError(
            "solve_system: the polynomial system has infinitely many solutions (positive-"
            "dimensional) — a parametrisation is not available; find one solution "
            "numerically instead (solve_nd)", route="solve_nd")
    count = _distinct_solution_count(polys, X)
    raw = sp.solve(polys, X, dict=True)
    sols: list[dict[sp.Symbol, sp.Expr]] = [s for s in raw if set(s) == set(X)]
    if len(sols) < count:
        # SymPy's solve can come back short — measured: [] for x^5 - x + 1 = 0, y = x.
        # A lex basis in shape position gives every solution from the roots of its
        # univariate polynomial (CRootOf: complete, exact).
        shaped = _shape_solutions(G, X)
        if shaped is not None:
            sols = shaped
    # distinct solutions of the polynomial equations: each pair PROVED equal or different
    distinct: list[dict[sp.Symbol, sp.Expr]] = []
    for s in sols:
        dup = False
        for t in distinct:
            diffs = [_complex_zero(s[v] - t[v]) for v in X]
            if all(d is True for d in diffs):
                dup = True
                break
            if not any(d is False for d in diffs):
                raise SolveSetUndecided("cannot tell two solutions apart", reason="distinct")
        if not dup:
            distinct.append(s)
    if len(distinct) != count:
        raise SolveSetUndecided(
            f"found {len(distinct)} distinct solutions but the radical ideal has {count}: "
            f"some solutions are missing", reason="completeness")
    # then drop the points where a denominator of the ORIGINAL equations vanishes
    keep = [s for s in distinct if all(_complex_zero(d.subs(s)) is False for d in dens)]
    checks = [_check_solution(polys, dens, s, []) for s in keep]
    reals = [all(_is_real_value(v) for v in s.values()) if all(_is_real_value(v) is not None for v in s.values()) else None
             for s in keep]
    if real:
        pairs = [(s, r, c) for s, r, c in zip(keep, reals, checks) if r]
        if any(r is None for r in reals):
            raise SolveSetUndecided("could not decide which solutions are real", reason="real")
        keep, reals, checks = [p[0] for p in pairs], [True] * len(pairs), [p[2] for p in pairs]
    status: Literal["finite", "none"] = "finite" if keep else "none"
    order = sorted(range(len(keep)), key=lambda i: tuple(
        (float(sp.re(sp.N(keep[i][v], 20))), float(sp.im(sp.N(keep[i][v], 20)))) for v in X))
    keep = [keep[i] for i in order]
    reals = [reals[i] for i in order]
    checks = [checks[i] for i in order]
    return SystemResult(
        shown, names, status,
        tuple({str(k): _s(v) for k, v in s.items()} for s in keep),
        tuple(_approx_dict(s) for s in keep),
        tuple(reals), (), None, None, "groebner",
        {"argument": "every solution substituted gives 0; the number of distinct complex "
                     "solutions equals the dimension of Q[vars]/radical(I) (Seidenberg), so "
                     "the list is complete" + ("; only the real ones are listed" if real else ""),
         "distinct_complex_solutions": count,
         "groebner_basis": [_s(g) for g in G.exprs]},
        tuple(checks),
    )


@under_budget("solve_system")
def solve_system(
    equations: "Sequence[str | sp.Basic] | str",
    vars: "Sequence[str] | str | None" = None,
    *,
    real: bool = False,
) -> SystemResult:
    """Solve a system of equations and PROVE the solution set complete.

        >>> solve_system(["x + y = 3", "x - y = 1"], ["x", "y"]).solutions
        ({'x': '2', 'y': '1'},)
        >>> solve_system(["x + y = 3", "x + y = 1"]).status
        'none'

    ``equations`` is a list (or one comma-separated string) of ``lhs = rhs`` texts;
    ``vars`` defaults to every free symbol, sorted. ``real=True`` keeps only real
    solutions of a polynomial system. A system that is not polynomial in its
    variables, or a nonlinear one with infinitely many solutions, raises
    ``UnsupportedFormError`` with route ``solve_nd``.
    """
    if isinstance(equations, str):
        equations = [e for e in _split_top(equations) if e.strip()]
    eqs = list(equations)
    if not eqs:
        raise DomainError("solve_system: no equations")
    parsed = [_parse_equation(e) for e in eqs]
    exprs = [_exact(p[0], "solve_system") for p in parsed]
    shown = tuple(p[1] for p in parsed)
    free = set().union(*(e.free_symbols for e in exprs))
    if vars is None:
        X = sorted(free, key=str)
    else:
        names = [v.strip() for v in vars.split(",")] if isinstance(vars, str) else list(vars)
        if not names or any(not isinstance(v, str) or not v.isidentifier() for v in names):
            raise DomainError(f"solve_system: bad variable list {vars!r}")
        if len(set(names)) != len(names):
            raise DomainError("solve_system: a variable is listed twice")
        X = [sp.Symbol(v) for v in names]
        other = free - set(X)
        if other:
            raise UnsupportedFormError(
                f"solve_system: symbolic parameters are not supported ({sorted(map(str, other))}) "
                f"— list them as variables or substitute numbers")
    if not X:
        raise DomainError("solve_system: no variables")
    polys: list[sp.Expr] = []
    dens: list[sp.Expr] = []
    for e in exprs:
        num, den = sp.fraction(sp.together(e))
        if any(not num.is_polynomial(*X) for _ in [0]) or not den.is_polynomial(*X):
            raise UnsupportedFormError(
                "solve_system: the system is not polynomial in its variables "
                "(transcendental functions) — solve it numerically (solve_nd)", route="solve_nd")
        if den.has(*X):
            dens.append(den)
        num = sp.expand(num)
        if num == 0:
            continue  # 0 = 0: no constraint
        polys.append(num)
    if not polys:
        raise UnsupportedFormError("solve_system: every equation is an identity (0 = 0)",
                                   route="solve_nd")
    if any(e.has(sp.I) for e in polys) and real:
        raise DomainError("solve_system: real=True with complex coefficients")
    linear = all(sp.Poly(p, *X).total_degree() <= 1 for p in polys)
    if linear and not dens:
        return _linear_system(polys, X, shown)
    for p in polys:
        dom = sp.Poly(p, *X).domain
        if dom not in (sp.ZZ, sp.QQ):
            raise UnsupportedFormError(
                f"solve_system: nonlinear equations need rational coefficients (got {dom}) "
                f"— solve numerically (solve_nd)", route="solve_nd")
    return _polynomial_system(polys, dens, X, shown, real)


def _split_top(text: str) -> list[str]:
    """Split on commas/semicolons that are not inside parentheses."""
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch in ",;" and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


# --- periodic families -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PeriodicFamily:
    """``base + k*period`` for every integer k; ``base`` lies in [0, period)."""

    base: str
    period: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"base": self.base, "period": self.period, "text": self.text}


@dataclass(frozen=True, slots=True)
class PeriodicResult:
    """A PROVED solution set of a periodic equation: finitely many families.

    ``families``     the readable families, merged (``k*pi/2`` rather than four
                     families of period ``2*pi``)
    ``period``       a proved period P of the equation; ``bases`` are ALL solutions in
                     [0, P) — the whole solution set is ``bases + P*Z``
    ``text``         ``x = pi/6 + 2*pi*k, x = 5*pi/6 + 2*pi*k, k ∈ ℤ``
    ``checks``       substitution of every family member for k = -2..2 (exact)
    ``periodicity``  how ``f(x + P) = f(x)`` was proved (``check_equal`` detail)
    ``endpoints``/``cells``  the sign decomposition of [0, P]: every zero there is a
                     cut point, every cell between them is proved zero-free
    """

    equation: str
    var: str
    status: Literal["families", "empty"]
    families: tuple[PeriodicFamily, ...]
    period: str
    bases: tuple[str, ...]
    text: str
    checks: tuple[dict[str, Any], ...]
    periodicity: str
    endpoints: tuple[EndpointCheck, ...]
    cells: tuple[CellCheck, ...]
    method: str
    sympy_answer: "str | None"
    proof: str = "proof"
    kind: str = "periodic"

    @property
    def verified(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "equation": self.equation,
            "var": self.var,
            "status": self.status,
            "families": [f.to_dict() for f in self.families],
            "period": self.period,
            "bases": list(self.bases),
            "text": self.text,
            "checks": [dict(c) for c in self.checks],
            "periodicity": self.periodicity,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "cells": [c.to_dict() for c in self.cells],
            "method": self.method,
            "proof": self.proof,
            "verdict": "verified",
            "sympy": {"answer": self.sympy_answer},
        }


def _mod(v: sp.Expr, T: sp.Expr) -> sp.Expr:
    """``v`` reduced into [0, T), exactly."""
    k = sp.floor(sp.nsimplify(v / T)) if (v / T).is_rational else sp.floor(v / T)
    r = sp.simplify(v - k * T)
    return r


def _family_text(var: str, base: sp.Expr, period: sp.Expr) -> str:
    k = sp.Symbol("k")
    term = sp.sstr(period * k)
    if base == 0:
        return f"{var} = {term}"
    return f"{var} = {sp.sstr(base)} + {term}"


def _same_point(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(_is_zero(sp.simplify(a - b)))


@under_budget("periodic_solutions")
def periodic_solutions(equation: "str | sp.Basic", var: str = "x") -> PeriodicResult:
    """All real solutions of a periodic equation as families ``a + k*T, k ∈ ℤ``.

        >>> r = periodic_solutions("sin(x) = 1/2")
        >>> r.text
        'x = pi/6 + 2*pi*k, x = 5*pi/6 + 2*pi*k, k ∈ ℤ'

    The period P is found by SymPy and PROVED (``f(x + P) = f(x)`` VERIFIED by
    ``check_equal``); the solutions in [0, P) are proved COMPLETE by the sign
    decomposition of ``solve_inequality``; every family is substituted for
    k = -2..2. Equations that are not periodic raise ``UnsupportedFormError``.
    """
    from ..core.ir import Expr
    from ..verify.equal import check_equal

    x = _real_symbol(var)
    e, shown = _parse_equation(equation)
    f = _exact(_to_real(e, x), "periodic_solutions")
    if f == 0:
        raise DomainError(f"periodic_solutions: {shown} holds for every {var} (an identity)")
    if f.free_symbols - {x}:
        raise UnsupportedFormError(
            f"periodic_solutions: one variable only; also found {sorted(map(str, f.free_symbols - {x}))}")
    if not f.has(x):
        raise DomainError(f"periodic_solutions: the equation does not contain {var}")
    _check_functions(f, "periodic_solutions")
    T = sp.periodicity(f, x)
    if T is None or T == 0 or not T.is_positive:
        raise UnsupportedFormError(
            f"periodic_solutions: {shown} is not periodic in {var} — use solve, or root for a "
            f"numerical solution", route="root")
    plain = sp.Symbol(var)
    v = check_equal(Expr(f.subs(x, x + T).xreplace({x: plain})), Expr(f.xreplace({x: plain})),
                    vars=[var])
    if not v.verified:
        raise SolveSetUndecided(f"cannot prove that {T} is a period of {shown}: {v.detail}",
                                reason="periodicity")
    try:
        proposed = sp.solveset(f, x, sp.Interval.Ropen(0, T))
        extra = list(proposed) if isinstance(proposed, sp.FiniteSet) else []
    except Exception:  # noqa: BLE001
        extra = []
    dec = _decompose(f, x, sp.Integer(0), T, extra, lambda s: s == 0)
    zeros = [p for p, pc in zip(dec.point_values, dec.points) if pc.member and not _same_point(p, T)]
    try:
        theirs = _s(sp.solveset(f, x, sp.S.Reals))
    except Exception as exc:  # noqa: BLE001
        theirs = f"error: {type(exc).__name__}"
    if not zeros:
        return PeriodicResult(shown, var, "empty", (), _s(T), (), f"no real solutions", (),
                              v.detail, tuple(dec.points), tuple(dec.cells), dec.method, theirs)
    n = len(zeros)
    best = 1
    for d in range(n, 1, -1):
        if n % d:
            continue
        shift = T / d
        if all(any(_same_point(_mod(z + shift, T), w) for w in zeros) for z in zeros):
            best = d
            break
    P = sp.simplify(T / best)
    bases = [z for z in zeros if _lt(z, P) or z == 0]
    fams = tuple(PeriodicFamily(_s(b), _s(P), _family_text(var, b, P)) for b in bases)
    checks: list[dict[str, Any]] = []
    for b in bases:
        for k in range(-2, 3):
            pt = b + k * P
            s = _sign_at(f.subs(x, pt))
            if s != 0:
                raise SolveSetUndecided(f"{pt} does not satisfy {shown}", reason="family-check")
            checks.append({"k": k, var: _s(pt), "residual": "0"})
    text = ", ".join(fm.text.split(" = ", 1)[0] + " = " + fm.text.split(" = ", 1)[1] for fm in fams) + ", k ∈ ℤ"
    return PeriodicResult(shown, var, "families", fams, _s(T), tuple(_s(z) for z in zeros), text,
                          tuple(checks), v.detail, tuple(dec.points), tuple(dec.cells), dec.method, theirs)
