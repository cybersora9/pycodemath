"""``certify`` — check an engine result by a route that did not produce it.

A certificate answers one question about one result: IS IT RIGHT? The answer is a
``Verdict`` with the same three words as ``check_equal`` and the same discipline:
VERIFIED only when an argument closes, REFUTED only with evidence confirmed twice,
UNDECIDED otherwise. What each operation's claim IS, and the route that checks it:

==============  ===================================  ==============================
operation       the claim certified                  the independent route
==============  ===================================  ==============================
``integrate``   ``d/dx F == f``                      finite differences of ``F``,
                (the constant of integration         then ``check_equal`` against
                vanishes by definition)              the symbolic derivative
``diff``        ``g == d/dx f``                      the same two routes
``dsolve``      every ``Y`` satisfies ``Y' = f(t,    the same two routes, per
                Y)`` — not that they are ALL the     solution, with ``C1``... as
                solutions                            sampled symbols
``solve``       every root is a root AND none is     substitution (``check_equal``);
                missing (real roots only when        root COUNT for a polynomial
                ``real=True``)                       (degree, or Sturm when real),
                                                     a sign-change scan otherwise
``limit``       ``lim f == L`` from the given side   an approach sequence at 30 and
                                                     60 digits, then the leading
                                                     term of the expansion
``nintegrate``  ``|value - I| <= error_estimate``    a second quadrature (mpmath
                (or ``rtol`` for a bare float)       tanh-sinh at 30 and 45 digits),
                                                     then the symbolic definite
                                                     integral when it has one
==============  ===================================  ==============================

WHY FINITE DIFFERENCES FIRST, for the three derivative claims. Re-deriving with the
engine that produced the answer is not a check of it: if ``Expr.diff`` is wrong, the
re-derivation agrees with the wrong answer. That is not hypothetical — the v0.3
audit found ``Abs``/``sign``/``Min``/``Max`` differentiating wrongly, silently. So
the first phase touches no symbolic calculus at all: it evaluates ``F`` at
``x0 ± h`` (exact rational points, two step sizes, 50 and 70 digits) and compares
the difference quotient with the claim. Only that phase may REFUTE on its own. The
symbolic phase (``check_equal`` against ``Expr.diff``) supplies the PROOF; when it
disagrees with the claim, the disagreement is reported only if finite differences
at the same point confirm it — otherwise the two routes contradict each other and
the verdict is UNDECIDED. A point where the difference quotient is not trustworthy
(a jump, a kink, a value with no digits left after cancellation) is skipped, never
counted either way.

WHAT IS NOT CLAIMED. ``dsolve`` is certified solution by solution: a family that
misses a singular solution (``y' = y^2`` has ``y = 0`` besides ``-1/(C1 + t)``) is
still VERIFIED, because every function it returns does solve the equation — the
table says so, and so does every such verdict's ``detail``. ``solve`` completeness
is proved only for rational functions with rational coefficients (from degree 4 on,
the listed roots are matched against ``nroots`` first — evidence, never a proof —
see ``certify_solve``); everywhere else
the scan can find a missing real root (REFUTED) but never prove there is none, so
the best it gives is UNDECIDED "completeness unconfirmed".

THE BUDGET — V1's rule, kept. Alone, a certificate arms ONE ``time_budget`` around
all of its phases, and running out becomes UNDECIDED with ``method="time-budget"``.
Inside a caller's ``time_budget`` it arms nothing (a tighter budget nested inside
another poisons the outer when it fires — measured in V1: an outer
``time_budget(10)`` refused at 0.26 s), so the enclosing allowance governs the
symbolic phases and its expiry refuses the enclosing block; ``budget`` then bounds
only the numeric loops, checked between points. ``check_equal`` called from here
always sees an armed budget, so it never arms one of its own either.

PRECISION. Every numeric value here is converted and combined inside
``mpmath.workdps``: ``mpmath.mpf(x)`` rounds to the GLOBAL precision (15 digits by
default), and a difference quotient with ``h = 1e-20`` computed at 15 digits is
noise. (``equal._evaluate`` converted under the global precision too until V5,
which capped ``check_equal``'s own sampling at double precision; it now converts
inside ``workdps`` itself, and the ``workdps`` this module still passes around
its calls is harmless.)
"""

from __future__ import annotations

import inspect
import math
import time
from typing import Any, Callable, Sequence

import mpmath
import sympy as sp

from ..core.budget import DEFAULT_TIME_BUDGET, time_budget
from ..core.errors import DomainError, TimeBudgetError
from ..core.ir import Expr
from ..core.result import QuadratureResult
from ..frontend.parser import parse
from .equal import (
    _as_written,
    _caller_budget_active,
    _evaluate,
    _format_point,
    _parse_written,
    _points,
    _show,
    _undefined_as_written,
    _value,
    _where,
    check_equal,
)
from .verdict import Verdict, VerdictStatus

__all__ = [
    "certify",
    "certify_integrate",
    "certify_diff",
    "certify_solve",
    "certify_limit",
    "certify_dsolve",
    "certify_nintegrate",
    "CERTIFIERS",
]

VERIFIED, REFUTED, UNDECIDED = (
    VerdictStatus.VERIFIED,
    VerdictStatus.REFUTED,
    VerdictStatus.UNDECIDED,
)

#: Points at which a derivative claim is checked by finite differences — V1's
#: readable points first (so a counterexample reads ``x = 2``), then seeded random.
_FD_SAMPLES = 16
#: (step, digits) of the two difference quotients. At ``h = 1e-15`` and 50 digits
#: the truncation error is ~1e-31·F''' and the cancellation costs 15 digits; at
#: ``h = 1e-20`` and 70 digits, ~1e-41·F''' and 20 digits. A real derivative gives
#: the same number twice to ~30 digits; noise from cancellation does not.
_FD_STEPS = ((sp.Rational(1, 10**15), 50), (sp.Rational(1, 10**20), 70))
#: The two quotients must agree to this (relative) before either is believed.
_FD_SELF = mpmath.mpf("1e-15")
#: Forward and backward quotients must agree to this: at a kink or a jump they
#: differ by O(1), at a smooth point by O(h·F'') ~ 1e-20.
_FD_SIDES = mpmath.mpf("1e-10")
#: A trusted quotient and the claimed derivative differ when they are further apart
#: than this (relative) — ten orders of margin over the quotient's own accuracy.
_FD_TOL = mpmath.mpf("1e-12")

#: Half-width of the real interval the root scan covers, at least; widened to twice
#: the largest listed root, so a scan never ends inside the listed ones.
_SCAN_RADIUS = 10
#: Grid cells of the scan. 2000 cells on [-10, 10] are 0.01 wide: two roots closer
#: than that inside one cell cancel their sign changes and are not seen — the scan
#: can only ever REFUTE, so a missed pair costs evidence, never a false verdict.
_SCAN_CELLS = 2000
#: Bisection steps on a bracket: 2^-90 of a 0.01 cell is ~1e-29.
_SCAN_BISECT = 90
#: A listed root and a found one are the same when this close (relative).
_SAME_ROOT = mpmath.mpf("1e-8")
#: From this degree of the numerator on, ``certify_solve`` checks the listed roots
#: numerically first (substitution without ``simplify``, then against ``nroots``)
#: and only then tries to PROVE them. A Ferrari root of an irreducible quartic,
#: substituted, is an exact 0 with no digits to evaluate, and ``simplify`` of it did
#: not finish in the whole 120 s default (V5: 4 of 4 probed quartics; V6: every
#: squared ``sqrt(x) = x^2 - a``). Cubics stay on the old route: Cardano roots
#: substitute and simplify in under a second (V2 finding 4).
_NUMERIC_DEGREE = 4
#: Digits ``nroots`` is asked for; listed roots are compared at 30.
_NROOTS_DPS = 50
#: A relative residual above this, reproduced at 30 and 60 digits, is not a root.
#: A root's residual at 30 digits is ~1e-30; ten orders of margin.
_SUBSTITUTED = mpmath.mpf("1e-20")

#: The approach sequence of ``limit``: distances 10^-k from the limit point (or
#: magnitudes 10^k for ±oo).
_APPROACH = (6, 10, 14, 18, 22, 26, 30)
#: The sequence has SETTLED when its last two values agree to this (relative).
_SETTLED = mpmath.mpf("1e-10")
#: A settled value and the claim differ when further apart than this (relative).
_LIMIT_TOL = mpmath.mpf("1e-6")
#: ... and a sequence DIVERGES when it has grown past this times ``max(1, |L|)``.
_DIVERGED = mpmath.mpf("1e12")

#: Relative accuracy a bare ``float`` from ``nintegrate`` is held to.
NINTEGRATE_RTOL = 1e-8
#: Rounding a double-precision quadrature may add on top of its method error,
#: relative to ``∫|f|``: a Simpson sum of ~200 terms accumulates ~1e-14; this is
#: 10x that, so ``∫ sin over [0, 2π] = 1e-17`` is not refuted for not being 0.
_FLOAT_SLACK = mpmath.mpf("1e-13")


# --- the run: one clock, one ledger of evidence ----------------------------------
class _OutOfTime(Exception):
    """The certificate's own deadline passed between two numeric points."""


class _Run:
    """One certificate's deadline and the evidence gathered so far — which is what
    an out-of-time verdict quotes, so a caller learns what WAS established."""

    def __init__(self, total: float) -> None:
        self.total = total
        self.deadline = time.perf_counter() + total  # inf stays inf
        self.phase = "setup"
        self.evidence: list[str] = []
        #: What was established by the numeric phases, returned instead of a bare
        #: "time-budget" when the budget runs out in a symbolic proof after them.
        self.fallback: "Verdict | None" = None

    def tick(self) -> None:
        if time.perf_counter() >= self.deadline:
            raise _OutOfTime

    def equal(self, a: sp.Expr, b: sp.Expr) -> Verdict:
        """``check_equal`` on this run's clock. A budget is always armed around a
        certificate, so ``check_equal`` arms none of its own (module docstring)."""
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0:
            raise _OutOfTime
        return check_equal(Expr(a), Expr(b), budget=remaining)

    def so_far(self) -> str:
        return "; ".join(self.evidence)


def _check_budget(budget: "float | None", operation: str) -> float:
    if budget is None:
        return DEFAULT_TIME_BUDGET
    if not isinstance(budget, (int, float)) or isinstance(budget, bool):
        raise DomainError(f"certify {operation}: budget must be a number, not {budget!r}")
    if math.isnan(budget) or budget <= 0:
        raise DomainError(f"certify {operation}: budget must be positive, got {budget!r}")
    return float(budget)


def _certify(
    operation: str, budget: "float | None", work: Callable[[_Run], Verdict]
) -> Verdict:
    run = _Run(_check_budget(budget, operation))
    if _caller_budget_active():
        # The ENCLOSING budget governs the symbolic phases; its expiry is the
        # caller's and propagates. Only this run's own clock becomes a verdict.
        try:
            return work(run)
        except _OutOfTime:
            return _out_of_time(run)
    try:
        with time_budget(run.total, f"certify {operation}"):
            return work(run)
    except (_OutOfTime, TimeBudgetError):
        return _out_of_time(run)


def _out_of_time(run: _Run) -> Verdict:
    if run.fallback is not None:
        known = run.fallback
        return Verdict(
            known.status, known.method, known.counterexample,
            f"{known.detail}; the {run.total:g}s time budget ran out during "
            f"{run.phase}",
        )
    so_far = run.so_far()
    return Verdict(
        UNDECIDED, "time-budget", None,
        f"the {run.total:g}s time budget ran out during {run.phase}"
        + (f"; {so_far}" if so_far else ""),
    )


def _attempt(fn: Callable[[], Any]) -> Any:
    """``fn()``, or ``None`` when SymPy/mpmath raises — a route that is not
    available is a route not taken, never an error. A budget running out is not
    such a failure and passes through."""
    try:
        return fn()
    except (TimeBudgetError, _OutOfTime):
        raise
    except Exception:  # noqa: BLE001 — see the docstring
        return None


# --- inputs ------------------------------------------------------------------
def _sym(value: object, what: str, operation: str) -> sp.Expr:
    if isinstance(value, Expr):
        sy = value.sy
    elif isinstance(value, str):
        sy = _parse_written(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        sy = sp.sympify(value)
    else:
        raise DomainError(
            f"certify {operation}: {what} must be an expression or text, "
            f"not {type(value).__name__}"
        )
    if not isinstance(sy, sp.Expr):
        raise DomainError(f"certify {operation}: {what} is not a scalar expression")
    return _as_written(sy, f"certify {operation}: {what}")


def _var(name: object, what: str, operation: str) -> sp.Symbol:
    if not isinstance(name, str) or not name.isidentifier():
        raise DomainError(f"certify {operation}: {what} must be a name, not {name!r}")
    return sp.Symbol(name)


def _sym_list(values: object, what: str, operation: str) -> list[sp.Expr]:
    if isinstance(values, (str, Expr)) or not isinstance(values, (list, tuple)):
        raise DomainError(
            f"certify {operation}: {what} must be a list, not {type(values).__name__}"
        )
    return [_sym(v, f"each of the {what}", operation) for v in values]


def _sorted_symbols(*exprs: sp.Expr) -> tuple[sp.Symbol, ...]:
    free: set = set().union(*(e.free_symbols for e in exprs))
    return tuple(sorted(free, key=lambda s: s.name))


def _read_point(counterexample: "dict[str, str]") -> dict:
    return {sp.Symbol(k): parse(v).sy for k, v in counterexample.items()}


# --- numbers -----------------------------------------------------------------
def _num(sy: sp.Expr, point: dict, dps: int, *, exact: "bool | None" = None) -> Any:
    """The value of ``sy`` at ``point`` as an ``mpc`` carrying ``dps`` digits, or
    ``None`` where it has none worth trusting (V1's ``_evaluate`` rules). ``exact``
    None tries exact substitution first, then ``evalf(subs=)``; ``False`` never
    substitutes exactly (``(1 + 1/x)^x`` at ``x = 10^30`` would be built as an exact
    rational with 10^30 digits — a C-level computation no budget can interrupt)."""
    order = (True, False) if exact is None else (exact,)
    with mpmath.workdps(dps):
        for how in order:
            value = _evaluate(sy, point, dps, exact=how)
            if value is not None:
                return value
            if how and _undefined_as_written(sy, point):
                # 0/0 as written: whatever ``evalf(subs=)`` returns next is noise
                # with full precision claimed (see ``equal._undefined_as_written``)
                return None
    return None


def _apart(a: Any, b: Any, rel: Any, floor: Any = 0) -> bool:
    """Are two values further apart than ``rel`` (relative) plus ``floor``? Call it
    inside the ``workdps`` the values were computed at."""
    return bool(abs(a - b) > rel * max(abs(a), abs(b)) + floor)


def _approx(v: Any) -> str:
    """A 15-digit rendering the parser reads back (``-3.14159265358979``,
    ``-0.662358978622373 + 0.562279512062301*I``)."""
    re, im = mpmath.nstr(v.real, 15), mpmath.nstr(abs(v.imag), 15)
    if v.imag == 0:
        return re
    return f"{re} {'-' if v.imag < 0 else '+'} {im}*I"


def _mpf(value: sp.Rational) -> Any:
    return mpmath.mpf(int(value.p)) / int(value.q)


# --- derivative claims: integrate, diff, dsolve --------------------------------
def _difference_quotient(F: sp.Expr, x: sp.Symbol, point: dict) -> "tuple[Any, Any] | None":
    """``(F'(x0), noise)`` by central differences at two step sizes — or ``None``
    when the quotient cannot be trusted there (see the module docstring).

    Trusted only if (a) the two central quotients agree, so cancellation noise is
    ruled out, and (b) the gap between the one-sided quotients shrinks with ``h``
    (by ~1e5 from step 1e-15 to 1e-20 at a smooth point; it stays O(1) at a kink and
    GROWS at a jump). ``noise`` is the absolute resolution of the answer.
    """
    x0 = point[x]
    central, sides = [], []
    for h, dps in _FD_STEPS:
        with mpmath.workdps(dps):
            up = _num(F, {**point, x: x0 + h}, dps)
            down = _num(F, {**point, x: x0 - h}, dps)
            mid = _num(F, point, dps)
            if up is None or down is None or mid is None:
                return None
            step = _mpf(h)
            central.append((up - down) / (2 * step))
            sides.append(abs((up - mid) / step - (mid - down) / step))
    with mpmath.workdps(_FD_STEPS[-1][1]):
        noise = mpmath.mpf("1e-25") * max(1, abs(mid))
        if _apart(central[0], central[1], _FD_SELF, noise):
            return None
        if sides[1] > mpmath.mpf("1e-3") * sides[0] + noise:
            return None
        return central[1], noise


def _derivative_claim(
    run: _Run, F: sp.Expr, x: sp.Symbol, g: sp.Expr, *, what: str, left: str, right: str
) -> Verdict:
    """Is ``d/dx F == g``? Finite differences first (may refute), then
    ``check_equal`` against ``Expr.diff`` (may prove). ``what`` names the claim in
    ``detail``; ``left`` and ``right`` name the two sides."""
    syms = _sorted_symbols(F, g, x)
    points = _points(syms, _FD_SAMPLES, 0)
    agreed = 0
    run.phase = "finite differences"
    for point in points:
        run.tick()
        quotient = _difference_quotient(F, x, point)
        claimed = _num(g, point, 60)
        if quotient is None or claimed is None:
            continue
        q, noise = quotient
        with mpmath.workdps(70):
            if not _apart(q, claimed, _FD_TOL, noise):
                agreed += 1
                continue
        return Verdict(
            REFUTED, "finite-differences", _format_point(point),
            f"{_where(point)}: {left} is {_show(q)} by finite differences, "
            f"{right} is {_show(claimed)}",
        )
    fd = f"finite differences agree at {agreed} of {len(points)} points"
    run.evidence.append(f"{what}: {fd}")

    run.phase = "symbolic differentiation"
    dF = _attempt(lambda: Expr(F).diff(x.name).sy)
    if dF is None:
        return Verdict(
            UNDECIDED, "finite-differences", None,
            f"{what}: {fd}, and no symbolic derivative to prove it with",
        )
    v = run.equal(dF, g)
    if v.status is VERIFIED:
        return Verdict(
            VERIFIED, "symbolic", None,
            f"{what}: the symbolic derivative matches ({v.detail}); {fd}",
        )
    if v.status is UNDECIDED:
        return Verdict(UNDECIDED, v.method, None, f"{what}: {fd}; {v.detail}")
    # REFUTED by the symbolic route — which re-derives with the engine under test.
    # Reported only where finite differences agree with THAT derivative.
    assert v.counterexample is not None
    where = _read_point(v.counterexample)
    for base in points:
        point = {**base, **where}
        quotient = _difference_quotient(F, x, point)
        derived = _num(dF, point, 60)
        claimed = _num(g, point, 60)
        if quotient is None or derived is None or claimed is None:
            continue
        q, noise = quotient
        with mpmath.workdps(70):
            if not _apart(q, derived, _FD_TOL, noise) and _apart(q, claimed, _FD_TOL, noise):
                return Verdict(
                    REFUTED, "symbolic", _format_point(point),
                    f"{_where(point)}: {left} is {_show(derived)} (symbolic, "
                    f"confirmed by finite differences), {right} is {_show(claimed)}",
                )
        break
    return Verdict(
        UNDECIDED, "finite-differences", None,
        f"{what}: the symbolic derivative disagrees with the claim ({v.detail}), "
        f"but finite differences do not confirm it; {fd}",
    )


def certify_integrate(
    expr: "Expr | str | int | float",
    var: str,
    result: "Expr | str | int | float",
    *,
    budget: "float | None" = None,
) -> Verdict:
    """Certify ``∫ expr d(var) = result``: is ``d/d(var) result == expr``?

    The constant of integration vanishes under the derivative, so ``x^2 + 5`` is as
    good an antiderivative of ``2*x`` as ``x^2``. See the module docstring for the
    two routes and for ``budget``.
    """
    op = "integrate"
    f, x, F = _sym(expr, "the integrand", op), _var(var, "var", op), _sym(result, "the result", op)
    return _certify(op, budget, lambda run: _derivative_claim(
        run, F, x, f, what=f"d/d{x.name} of the result",
        left="the result's derivative", right="the integrand",
    ))


def certify_diff(
    expr: "Expr | str | int | float",
    var: str,
    result: "Expr | str | int | float",
    *,
    budget: "float | None" = None,
) -> Verdict:
    """Certify ``d/d(var) expr = result`` — finite differences of ``expr`` against
    ``result``, then a symbolic proof. See the module docstring."""
    op = "diff"
    f, x, g = _sym(expr, "the expression", op), _var(var, "var", op), _sym(result, "the result", op)
    return _certify(op, budget, lambda run: _derivative_claim(
        run, f, x, g, what=f"d/d{x.name}",
        left="the derivative", right="the claimed derivative",
    ))


def certify_dsolve(
    expr: "Expr | str | int | float",
    func: str,
    var: str,
    solutions: "Sequence[Expr | str | int | float]",
    *,
    budget: "float | None" = None,
) -> Verdict:
    """Certify that every function in ``solutions`` solves ``func'(var) = expr``.

    ``expr`` is written as ``engine.ode.dsolve`` takes it — the unknown as a plain
    symbol ``func``. Each solution is checked by substitution (finite differences of
    ``Y``, then a symbolic proof), with its constants ``C1``… sampled as variables,
    so the claim is an identity in them too. NOT claimed: that the solutions are
    all there are (see the module docstring).
    """
    op = "dsolve"
    rhs = _sym(expr, "the right-hand side", op)
    y, t = _var(func, "func", op), _var(var, "var", op)
    if y == t:
        raise DomainError("certify dsolve: func and var must differ")
    sols = _sym_list(solutions, "solutions", op)
    if not sols:
        raise DomainError("certify dsolve: no solutions to check")
    for Y in sols:
        if Y.has(y):
            raise DomainError(
                f"certify dsolve: solution {Y} contains {y} — only explicit "
                f"solutions {y}({t}) = <expression without {y}> can be checked"
            )

    def work(run: _Run) -> Verdict:
        proved, undecided = 0, None
        for Y in sols:
            v = _derivative_claim(
                run, Y, t, rhs.xreplace({y: Y}), what=f"{y}({t}) = {Y}",
                left=f"{y}'", right=f"the right-hand side {rhs}",
            )
            if v.status is REFUTED:
                return Verdict(
                    REFUTED, v.method, v.counterexample,
                    f"{y}({t}) = {Y} does not solve {y}' = {rhs}: {v.detail}",
                )
            if v.status is VERIFIED:
                proved += 1
            elif undecided is None:
                undecided = v
        if undecided is not None:
            return Verdict(
                UNDECIDED, undecided.method, None,
                f"{proved} of {len(sols)} solutions proved; {undecided.detail}",
            )
        return Verdict(
            VERIFIED, "symbolic", None,
            f"each of the {len(sols)} solution(s) satisfies {y}' = {rhs} "
            f"(symbolic proof; finite differences agree) — that they are ALL the "
            f"solutions is not certified",
        )

    return _certify(op, budget, work)


# --- solve ---------------------------------------------------------------------
def _rational_parts(f: sp.Expr, x: sp.Symbol) -> "tuple[Any, Any] | None":
    """``(numerator, denominator)`` as polynomials in ``x`` with rational
    coefficients — or ``None`` when ``f`` is not such a rational function, or is
    identically zero."""
    num, den = sp.fraction(sp.together(f))
    try:
        pn, pd = sp.Poly(num, x), sp.Poly(den, x)
    except sp.PolynomialError:
        return None
    if not all(p.domain.is_ZZ or p.domain.is_QQ for p in (pn, pd)) or pn.is_zero:
        return None
    return pn, pd


def _algebraic_root(parts: "tuple[Any, Any]", r: sp.Expr, x: sp.Symbol) -> bool:
    """Is the constant ``r`` PROVABLY a root of numerator/denominator? Its minimal
    polynomial ``m`` over the rationals is irreducible, so ``r`` is a root of a
    rational polynomial exactly when ``m`` divides it. The first route for a
    ``CRootOf``, whose minimal polynomial is at hand: ``simplify`` does not reduce
    ``CRootOf(x^5 - x - 1, 4)^5 - CRootOf(...) - 1`` — measured: substitution ran out
    the whole 120 s default budget on that quintic. Only the FALLBACK for anything
    else: on the three Cardano roots of ``x^3 - x - 1`` it costs ~3 s each (the
    factorisations inside ``minimal_polynomial``), substitution well under 1 s."""
    pn, pd = parts
    m = sp.Poly(sp.minimal_polynomial(r, x), x)
    return bool(pn.rem(m).is_zero and not pd.rem(m).is_zero)


def _root_count(parts: "tuple[Any, Any]", real: bool) -> "tuple[int, Any, str]":
    """How many distinct (real) roots numerator/denominator has: roots of the
    square-free numerator that are not roots of the denominator — counted by
    degree, or by Sturm sequence when ``real``. ``(count, that polynomial, how)``."""
    pn, pd = parts
    s = pn.sqf_part()
    s = s.exquo(s.gcd(pd))
    if real:
        return int(s.count_roots()), s, "Sturm count"
    return int(max(s.degree(), 0)), s, "degree"


def _unlisted_root(s: Any, listed: list, real: bool) -> "str | None":
    """A root of ``s`` that none of ``listed`` is — the smallest in modulus."""
    try:
        found = s.nroots(n=30)
    except Exception:  # noqa: BLE001 — no located root, no counterexample
        return None
    with mpmath.workdps(30):
        values = []
        for c in found:
            re, im = c.as_real_imag()
            values.append(mpmath.mpc(mpmath.mpf(re), mpmath.mpf(im)))
        for c in sorted(values, key=abs):
            if real and abs(c.imag) > mpmath.mpf("1e-20") * max(1, abs(c)):
                continue
            if all(_apart(c, v, _SAME_ROOT, _SAME_ROOT) for v in listed):
                return _approx(c)
    return None


def _scan(run: _Run, f: sp.Expr, x: sp.Symbol, listed: list) -> "tuple[int, str | None] | None":
    """Look for real roots by sign changes on ``[-R, R]``: ``(roots found, the
    smallest-modulus one NOT listed, or None)`` — or ``None`` if ``f`` cannot be
    evaluated this way. A bracket counts only if bisection closes on a genuine zero
    (a pole or a jump also changes sign), and an unlisted one only if the sign
    change reproduces at 30 and 60 digits with SymPy's own evaluation."""
    fn = _attempt(lambda: sp.lambdify(x, f, "mpmath"))
    if fn is None:
        return None

    def real_value(s: Any) -> Any:
        v = _attempt(lambda: mpmath.mpmathify(fn(s)))
        if v is None or not mpmath.isfinite(v):
            return None
        if isinstance(v, mpmath.mpc):
            if abs(v.imag) > mpmath.mpf("1e-12") * max(1, abs(v.real)):
                return None
            v = v.real
        return v

    with mpmath.workdps(20):
        reals = [v.real for v in listed if abs(v.imag) <= mpmath.mpf("1e-20") * max(1, abs(v))]
        radius = max([mpmath.mpf(_SCAN_RADIUS)] + [2 * abs(r) for r in reals])
        cell = 2 * radius / _SCAN_CELLS
        # offset by a golden-ratio fraction of a cell: no grid point on 0, pi, ...
        grid = [-radius + cell * (i + mpmath.mpf("0.381966011250105")) for i in range(_SCAN_CELLS)]
        values = []
        for s in grid:
            run.tick()
            values.append(real_value(s))
    found, unlisted = 0, []
    for (a, fa), (b, fb) in zip(zip(grid, values), zip(grid[1:], values[1:])):
        if fa is None or fb is None or (fa > 0) == (fb > 0):
            continue
        run.tick()
        with mpmath.workdps(40):
            lo, hi, flo = mpmath.mpf(a), mpmath.mpf(b), fa
            for _ in range(_SCAN_BISECT):
                mid = (lo + hi) / 2
                fm = real_value(mid)
                if fm is None:
                    break
                if (fm > 0) == (flo > 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            c = (lo + hi) / 2
            fc = real_value(c)
            if fc is None or abs(fc) > mpmath.mpf("1e-12") * max(1, abs(fa), abs(fb)):
                continue  # a pole or a jump, not a root
            found += 1
            if any(not _apart(c, r, _SAME_ROOT, _SAME_ROOT) for r in reals):
                continue
            unlisted.append((c, lo, hi))
    for c, lo, hi in sorted(unlisted, key=lambda u: abs(u[0])):
        ends = sp.Rational(mpmath.nstr(lo, 40)), sp.Rational(mpmath.nstr(hi, 40))
        if all(_sign_change(f, x, ends, dps) for dps in (30, 60)):
            with mpmath.workdps(40):
                return found, _approx(mpmath.mpc(c))
    return found, None


def _sign_change(f: sp.Expr, x: sp.Symbol, ends: tuple, dps: int) -> bool:
    va, vb = (_num(f, {x: e}, dps) for e in ends)
    if va is None or vb is None:
        return False
    with mpmath.workdps(dps):
        tiny = mpmath.mpf(10) ** (-dps // 2)
        if abs(va.imag) > tiny * max(1, abs(va)) or abs(vb.imag) > tiny * max(1, abs(vb)):
            return False
        return bool((va.real > 0) != (vb.real > 0) and va.real != 0 and vb.real != 0)


def _residual(pn: Any, v: Any, dps: int) -> Any:
    """``|pn(v)|`` relative to ``sum |a_k| |v|^k`` — the size the terms cancel from."""
    with mpmath.workdps(dps):
        value, scale = mpmath.mpc(0), mpmath.mpf(0)
        for c in pn.all_coeffs():
            a = _mpf(c)
            value, scale = value * v + a, scale * abs(v) + abs(a)
        return abs(value) / scale if scale else mpmath.mpf(0)


def _substituted(parts: "tuple[Any, Any]", r: sp.Expr) -> "str | None":
    """Is the listed ``r`` NOT a root? The reason when the numerator's relative
    residual at ``r``'s value is far from 0 at 30 digits and again at 60, by the
    same amount; ``None`` otherwise — then what proves ``r`` is left to
    ``_algebraic_root`` (see ``_NUMERIC_DEGREE``).

    Horner on ``r``'s value, not the substituted expression: a Ferrari root's
    exact 0 has no digits, and ``evalf`` searched for them for 0.33 s per try
    (0.7 s per root with the exact retry) — 2.7 s for a quartic before V8's
    candidates were even compared, against 10 ms for the root's own value."""
    found = []
    for dps in (30, 60):
        v = _root_value(r, dps)
        if v is None:
            return None
        found.append(_residual(parts[0], v, dps))
    low, high = found
    with mpmath.workdps(30):
        if low <= _SUBSTITUTED or high <= _SUBSTITUTED or _apart(low, high, mpmath.mpf("1e-6")):
            return None
        return (
            f"the numerator is not 0 there: relative residual "
            f"{mpmath.nstr(high, 6)} at 60 digits ({mpmath.nstr(low, 6)} at 30)"
        )


def _located(
    s: Any, real: bool, pending: list, n: int, expected: int, how: str, kind: str,
) -> str:
    """What ``nroots`` says about the roots not yet proved — evidence, never a
    proof: a listed value that agrees with a located root to ``_SAME_ROOT`` is
    still only a number that is close to a root. Completeness is stated only when
    ``nroots`` located as many roots as the square-free numerator's degree (every
    root simple, so that is the count with multiplicity) — and, for real roots, as
    many real ones as Sturm counts."""
    found = _attempt(lambda: s.nroots(n=_NROOTS_DPS, maxsteps=200))
    if not found or len(found) != s.degree():
        k = len(found) if found else 0
        return (
            f"completeness unconfirmed: nroots located {k} of the "
            f"{s.degree()} roots, so the {len(pending)} unproved ones could not be "
            f"compared"
        )
    with mpmath.workdps(30):
        located = []
        for c in found:
            re, im = c.as_real_imag()
            located.append(mpmath.mpc(mpmath.mpf(re), mpmath.mpf(im)))
        if real:
            located = [
                c for c in located
                if abs(c.imag) <= mpmath.mpf("1e-20") * max(1, abs(c))
            ]
        matched = 0
        for v in pending:
            near = [c for c in located if not _apart(v, c, _SAME_ROOT, _SAME_ROOT)]
            if near:
                located.remove(near[0])
                matched += 1
    text = (
        f"{matched} of the {len(pending)} not proved by substitution agree to 8 "
        f"digits with roots located by nroots — numeric candidates, not a proof"
    )
    if real and len(located) + matched != expected:
        return text + (
            f"; completeness unconfirmed: nroots finds {len(located) + matched} "
            f"real roots, Sturm counts {expected}"
        )
    return text + (
        f"; the equation has exactly {expected} distinct {kind}roots ({how} of "
        f"the square-free numerator), {n} listed"
    )


def _root_value(r: sp.Expr, dps: int) -> Any:
    """``_num`` for a listed root — except a ``CRootOf``, whose ``evalf`` refines
    complex isolating rectangles (measured: 2.4 s per complex root of
    ``x^5 - x - 1`` at 30 digits) while ``eval_approx`` runs a secant method checked
    against the same root bounds (3 ms)."""
    if isinstance(r, sp.CRootOf):
        approx = _attempt(lambda: r.eval_approx(dps))
        if approx is not None:
            with mpmath.workdps(dps):
                re, im = approx.as_real_imag()
                return mpmath.mpc(mpmath.mpf(re), mpmath.mpf(im))
    return _num(r, {}, dps)


def certify_solve(
    expr: "Expr | str | int | float",
    var: str,
    roots: "Sequence[Expr | str | int | float]",
    *,
    real: bool = False,
    budget: "float | None" = None,
) -> Verdict:
    """Certify that ``roots`` are exactly the solutions of ``expr = 0`` in ``var``.

    Two claims, both checked: every listed value IS a root (substitution through
    ``check_equal``; a value where the equation is undefined is not a root), and no
    root is MISSING — all roots, or all real roots with ``real=True`` (the REPL's
    default, mirroring ``engine.symbolic.solve``). A listed value that is not real
    under ``real=True`` is refuted too.

    Completeness is PROVED only for a rational function with rational coefficients
    (root count of the square-free numerator — degree, or Sturm when ``real``);
    otherwise a sign-change scan can find a missing real root (REFUTED) but not
    prove there is none, so the verdict stays UNDECIDED "completeness unconfirmed".
    With parameters (``x^2 - a``) completeness is not attempted.

    From numerator degree 4 on (``_NUMERIC_DEGREE``), a listed root is first
    checked by its value — refuted when the numerator's residual there is far from
    0 at 30 and 60 digits — and compared with the roots ``nroots`` locates; only
    then is it PROVED, through its minimal polynomial (``_algebraic_root``: ~5 s
    per Ferrari root, where ``simplify`` of the substitution did not finish in
    120 s). Should the budget run out in those proofs, the verdict is the numeric
    one — UNDECIDED ``method="numeric-roots"``, saying how many listed roots agree
    with located ones and whether the count is complete — never VERIFIED.
    """
    op = "solve"
    f, x = _sym(expr, "the equation", op), _var(var, "var", op)
    rs = _sym_list(roots, "roots", op)
    if not isinstance(real, bool):
        raise DomainError(f"certify solve: real must be True or False, not {real!r}")
    kind = "real " if real else ""

    def work(run: _Run) -> Verdict:
        proved = 0
        values: list = []
        parts = None if f.free_symbols - {x} else _attempt(lambda: _rational_parts(f, x))
        numeric = parts is not None and parts[0].degree() >= _NUMERIC_DEGREE
        pending: list = []  # (root, value): not refuted numerically, not yet proved

        def algebraic_root(r: sp.Expr) -> bool:
            return parts is not None and bool(_attempt(lambda: _algebraic_root(parts, r, x)))

        for r in rs:
            run.phase = f"substituting {x} = {r}"
            if real and not r.free_symbols:
                value = _root_value(r, 60)
                with mpmath.workdps(60):
                    if value is not None and _apart(value, value.real, mpmath.mpf("1e-30")):
                        return Verdict(
                            REFUTED, "numeric-sampling", {x.name: str(r)},
                            f"{x} = {r} is not real ({_show(value)}), "
                            f"but real roots were asked for",
                        )
            if not r.free_symbols:
                values.append(_root_value(r, 30))
            if isinstance(r, sp.CRootOf) and algebraic_root(r):
                proved += 1
                continue
            at = f.subs(x, r)
            if at.has(sp.nan, sp.zoo) or at in (sp.oo, -sp.oo):
                return Verdict(
                    REFUTED, "symbolic", {x.name: str(r)},
                    f"{x} = {r} is not a root: the equation is undefined there",
                )
            if numeric and not r.free_symbols:
                # See ``_NUMERIC_DEGREE``: the value, not ``simplify``, first.
                if at == 0:
                    proved += 1
                    continue
                assert parts is not None  # ``numeric`` implies it
                why = _substituted(parts, r)
                if why is None:
                    pending.append((r, values[-1]))
                    continue
                return Verdict(
                    REFUTED, "numeric-substitution", {x.name: str(r)},
                    f"{x} = {r} is not a root: {why}",
                )
            else:
                v = run.equal(at, sp.Integer(0))
            if v.status is REFUTED:
                assert v.counterexample is not None
                return Verdict(
                    REFUTED, v.method, {x.name: str(r), **v.counterexample},
                    f"{x} = {r} is not a root: {v.detail}",
                )
            if v.status is VERIFIED or (not r.free_symbols and algebraic_root(r)):
                proved += 1
        every = proved == len(rs)
        run.evidence.append(f"{proved} of {len(rs)} listed roots proved to be roots")

        run.phase = "the completeness check"
        how_roots = "symbolic" if every else "numeric-sampling"
        if f.free_symbols - {x}:
            return Verdict(
                UNDECIDED, how_roots, None,
                f"{run.so_far()}; completeness unconfirmed: the equation has parameters",
            )
        if any(v is None for v in values):
            return Verdict(
                UNDECIDED, how_roots, None,
                f"{run.so_far()}; completeness unconfirmed: a listed root has no value",
            )
        with mpmath.workdps(30):
            for i, a in enumerate(values):
                if any(not _apart(a, b, mpmath.mpf("1e-20")) for b in values[i + 1:]):
                    return Verdict(
                        UNDECIDED, how_roots, None,
                        f"{run.so_far()}; two listed roots coincide",
                    )

        if parts is not None:
            expected, s, how = _root_count(parts, real)
            n = len(rs)
            if n < expected:
                missing = _unlisted_root(s, values, real)
                if missing is None:
                    return Verdict(
                        UNDECIDED, "root-count", None,
                        f"{run.so_far()}; {expected} distinct {kind}roots exist "
                        f"({how}) and {n} are listed, but the missing ones could not "
                        f"be located",
                    )
                return Verdict(
                    REFUTED, "root-count", {x.name: missing},
                    f"the equation has {expected} distinct {kind}roots ({how} of the "
                    f"square-free numerator), {n} listed: {x} ≈ {missing} is missing",
                )
            if pending:
                run.phase = "locating the roots numerically (nroots)"
                located = _located(s, real, [v for _, v in pending], n, expected, how, kind)

                def numeric_verdict(extra: str = "") -> Verdict:
                    return Verdict(
                        UNDECIDED, "numeric-roots", None,
                        f"{proved} of {n} listed roots proved to be roots; "
                        f"{located}{extra}",
                    )

                run.fallback = numeric_verdict()
                for r, _ in pending:
                    run.tick()
                    run.phase = f"proving {x} = {r} a root (its minimal polynomial)"
                    if algebraic_root(r):
                        proved += 1
                        run.fallback = numeric_verdict()
                run.fallback = None
                every = proved == n
                if not every:
                    return numeric_verdict(
                        f"; the minimal-polynomial proof did not close for "
                        f"{n - proved} of them"
                    )
            if n == expected and every:
                return Verdict(
                    VERIFIED, "symbolic", None,
                    f"all {n} listed roots proved to be roots, and the equation "
                    f"has exactly {expected} distinct {kind}roots ({how} of the "
                    f"square-free numerator)",
                )
            return Verdict(
                UNDECIDED, how_roots, None,
                f"{run.so_far()}; {expected} distinct {kind}roots exist ({how}), "
                f"{n} listed — not every listed root is proved",
            )

        run.phase = "the sign-change scan"
        scanned = _scan(run, f, x, values)
        if scanned is None:
            return Verdict(
                UNDECIDED, how_roots, None,
                f"{run.so_far()}; completeness unconfirmed: the equation cannot be "
                f"scanned numerically",
            )
        found, missing = scanned
        if missing is not None:
            return Verdict(
                REFUTED, "sign-scan", {x.name: missing},
                f"a root is missing: the expression changes sign at {x} ≈ {missing} "
                f"(confirmed at 30 and 60 digits), where no listed root lies",
            )
        return Verdict(
            UNDECIDED, "sign-scan", None,
            f"{run.so_far()}; completeness unconfirmed: a sign-change scan found "
            f"{found} real root(s), all listed, but cannot rule out others (even "
            f"multiplicity, complex, or outside the scanned interval)",
        )

    return _certify(op, budget, work)


# --- limit ---------------------------------------------------------------------
def _is_infinite(e: sp.Expr) -> bool:
    return e in (sp.oo, -sp.oo)


def certify_limit(
    expr: "Expr | str | int | float",
    var: str,
    to: "Expr | str | int | float",
    result: "Expr | str | int | float",
    *,
    dir: str = "+",
    budget: "float | None" = None,
) -> Verdict:
    """Certify ``lim_{var → to, dir} expr = result`` (``to``/``result`` may be ±oo).

    Numerically: the expression at distances 10^-6 … 10^-30 from ``to`` on the
    ``dir`` side (at 10^6 … 10^30 for ±oo), at 30 digits, never exact substitution.
    A sequence that SETTLES somewhere else, or grows past 10^12 when a finite value
    was claimed (or settles when ±oo was), refutes — re-evaluated at 60 digits
    first; ``counterexample`` is then the closest point tried. A slow sequence
    (``1/log(1/x)``) neither settles nor diverges there and refutes nothing.

    Proof: the leading term ``c·t^p`` of the expression in the distance ``t > 0``
    — ``c`` for ``p = 0``, ``0`` for ``p > 0``, ``±oo`` for ``p < 0`` and a ``c`` of
    known sign — compared with ``result`` through ``check_equal``. A leading term
    that disagrees with a sequence that settled ON the claim is a contradiction, and
    UNDECIDED. Parameters are sampled at four points.
    """
    op = "limit"
    f, x = _sym(expr, "the expression", op), _var(var, "var", op)
    p, L = _sym(to, "the limit point", op), _sym(result, "the result", op)
    if dir not in ("+", "-"):
        raise DomainError(f"certify limit: dir must be '+' or '-', not {dir!r}")
    if p.free_symbols or p.has(sp.nan, sp.zoo):
        raise DomainError(f"certify limit: the limit point must be a number or ±oo, not {p}")
    if L.has(sp.nan, sp.zoo):
        raise DomainError(f"certify limit: {L} is not a limit value")

    t = sp.Dummy("t", positive=True)
    if p is sp.oo:
        xt, side = 1 / t, f"{x} → oo"
    elif p is sp.S.NegativeInfinity:
        xt, side = -1 / t, f"{x} → -oo"
    else:
        xt, side = (p + t if dir == "+" else p - t), f"{x} → {p}{dir}"
    g = f.subs(x, xt)
    params = _sorted_symbols(g, L)
    params = tuple(s for s in params if s != t)
    param_points = _points(params, 4, 0)

    def work(run: _Run) -> Verdict:
        run.phase = "the approach sequence"
        agreed = 0
        for pp in param_points:
            run.tick()
            seq = [_num(g, {**pp, t: sp.Rational(1, 10**k)}, 30, exact=False) for k in _APPROACH]
            last = sp.Rational(1, 10 ** _APPROACH[-1])
            claimed = None if _is_infinite(L) else _num(L, pp, 30)
            if any(v is None for v in seq[-3:]) or (claimed is None and not _is_infinite(L)):
                continue
            with mpmath.workdps(30):
                v5, v6, v7 = seq[-3:]
                settled = not _apart(v7, v6, _SETTLED, _SETTLED)
                bound = _DIVERGED * max(1, abs(claimed) if claimed is not None else 1)
                diverged = abs(v7) > bound and abs(v7) > abs(v6) > abs(v5)
                if _is_infinite(L):
                    sign_ok = abs(v7.imag) < abs(v7.real) and (v7.real > 0) == (L is sp.oo)
                    wrong = settled or (diverged and not sign_ok)
                    agrees = diverged and sign_ok
                else:
                    wrong = (settled and _apart(v7, claimed, _LIMIT_TOL, _LIMIT_TOL)) or diverged
                    agrees = settled and not wrong
            if not wrong:
                agreed += agrees
                continue
            # confirm at 60 digits before accusing
            again = _num(g, {**pp, t: last}, 60, exact=False)
            if again is None:
                continue
            with mpmath.workdps(60):
                if not _apart(again, v7, mpmath.mpf("1e-20"), 0):
                    point = {**pp, x: xt.subs(t, last)}
                    why = "grows without bound" if diverged else f"settles at {_show(again)}"
                    return Verdict(
                        REFUTED, "approach-sequence", _format_point(point),
                        f"as {side} the expression {why} (at {_where(point)[3:]}: "
                        f"{_show(again)}), not the claimed {L}",
                    )
        numeric = (
            f"the approach sequence agrees with the claim at {agreed} of "
            f"{len(param_points)} parameter point(s)"
        )
        run.evidence.append(numeric)

        run.phase = "the leading term"
        lead = _attempt(lambda: g.as_leading_term(t))
        candidate = None
        if lead is not None:
            c, e = lead.as_coeff_exponent(t)
            if not c.has(t) and e.is_number and e.is_extended_real:
                if e == 0:
                    candidate = c
                elif e > 0:
                    candidate = sp.Integer(0)
                elif c.is_extended_positive:
                    candidate = sp.oo
                elif c.is_extended_negative:
                    candidate = -sp.oo
        if candidate is None:
            return Verdict(
                UNDECIDED, "approach-sequence", None,
                f"{numeric}; the leading term gives no closed form to prove it with",
            )
        contradicted = agreed == len(param_points)
        if _is_infinite(candidate) or _is_infinite(L):
            if candidate == L:
                return Verdict(VERIFIED, "leading-term", None, f"the leading term as {side} gives {candidate}; {numeric}")
            if contradicted:
                return _contradiction(candidate, numeric)
            return Verdict(
                REFUTED, "leading-term", _format_point(param_points[0]),
                f"the leading term as {side} gives {candidate}, not the claimed {L}",
            )
        v = run.equal(candidate, L)
        if v.status is VERIFIED:
            return Verdict(VERIFIED, "leading-term", None, f"the leading term as {side} gives {candidate}, equal to the claim; {numeric}")
        if v.status is REFUTED and not contradicted:
            assert v.counterexample is not None
            return Verdict(
                REFUTED, "leading-term", v.counterexample,
                f"the leading term as {side} gives {candidate}, not the claimed {L}: {v.detail}",
            )
        if v.status is REFUTED:
            return _contradiction(candidate, numeric)
        return Verdict(UNDECIDED, v.method, None, f"{numeric}; leading term {candidate}: {v.detail}")

    return _certify(op, budget, work)


def _contradiction(candidate: sp.Expr, numeric: str) -> Verdict:
    return Verdict(
        UNDECIDED, "leading-term", None,
        f"the leading term gives {candidate}, but {numeric} — the two routes "
        f"contradict each other",
    )


# --- nintegrate ------------------------------------------------------------------
def _reference(fn: Callable[..., Any], a: sp.Expr, b: sp.Expr) -> "tuple[Any, Any, Any] | None":
    """``(I, u, ∫|f|)``: tanh-sinh at 45 digits, its uncertainty (the distance to the
    30-digit run plus its own error estimate), and the scale of the integrand."""
    runs = []
    for dps in (30, 45):
        lo, hi = _num(a, {}, dps), _num(b, {}, dps)
        if lo is None or hi is None:
            return None
        with mpmath.workdps(dps):
            runs.append(mpmath.quad(fn, [lo.real, hi.real], error=True))
    with mpmath.workdps(20):
        lo, hi = _num(a, {}, 20), _num(b, {}, 20)
        assert lo is not None and hi is not None
        scale = mpmath.quad(lambda s: abs(fn(s)), [lo.real, hi.real])
    with mpmath.workdps(45):
        (i30, _), (i45, e45) = runs
        return mpmath.mpmathify(i45), abs(i30 - i45) + abs(e45), abs(mpmath.mpmathify(scale))


def certify_nintegrate(
    expr: "Expr | str | int | float",
    var: str,
    a: "Expr | str | int | float",
    b: "Expr | str | int | float",
    result: "QuadratureResult | float",
    *,
    rtol: float = NINTEGRATE_RTOL,
    budget: "float | None" = None,
) -> Verdict:
    """Certify a numerical integral: is ``|value - ∫_a^b expr| <= claimed error``?

    ``result`` is what ``integrate_num`` returned: a ``QuadratureResult`` claims its
    own ``error_estimate`` (the engine documents that estimate as able to LIE — this
    is the check that catches it), a bare ``float`` is held to ``rtol`` relative.
    Either way a rounding allowance of 1e-13·∫|f| is added.

    Route 1, which may refute: mpmath's tanh-sinh quadrature at 30 and 45 digits;
    a gap larger than the claim plus ten times the reference's own uncertainty is
    REFUTED (``counterexample`` is ``{}`` — the claim has no free variable). Route 2,
    which may prove: SymPy's definite integral, evaluated at 60 digits — used only
    where it agrees with route 1, since a definite integral through a discontinuous
    antiderivative can be wrong. A failed run with ``error_estimate = inf`` claims
    nothing and gets UNDECIDED with ``method="none"``.
    """
    op = "nintegrate"
    f, x = _sym(expr, "the integrand", op), _var(var, "var", op)
    if f.free_symbols - {x}:
        raise DomainError(f"certify nintegrate: the integrand may depend only on {x}")
    # A float bound is the binary number the engine integrated over, exactly — not
    # V1's "decimal as written", which rounds 3.141592653589793 to 15 digits.
    lo, hi = (
        sp.Rational(v) if isinstance(v, float) else _sym(v, name, op)
        for v, name in ((a, "a"), (b, "b"))
    )
    for bound in (lo, hi):
        value = _num(bound, {}, 30) if not bound.free_symbols else None
        if value is None or value.imag != 0:
            raise DomainError(f"certify nintegrate: bound {bound} is not a finite real number")
    if isinstance(result, QuadratureResult):
        value, claim, source = result.value, result.error_estimate, "error estimate"
    elif isinstance(result, (int, float)) and not isinstance(result, bool):
        if not isinstance(rtol, (int, float)) or isinstance(rtol, bool) or not 0 < rtol < math.inf:
            raise DomainError(f"certify nintegrate: rtol must be positive and finite, not {rtol!r}")
        value, claim, source = float(result), abs(float(result)) * rtol, f"rtol {rtol:g}"
    else:
        raise DomainError(
            f"certify nintegrate: result must be a QuadratureResult or a number, "
            f"not {type(result).__name__}"
        )
    if not math.isfinite(value):
        raise DomainError(f"certify nintegrate: the value {value!r} is not finite")
    if not claim < math.inf:
        return Verdict(
            UNDECIDED, "none", None,
            f"the result claims no accuracy (error estimate {claim}), so there is "
            f"nothing to certify",
        )

    def work(run: _Run) -> Verdict:
        run.phase = "the reference quadrature"
        fn = _attempt(lambda: sp.lambdify(x, f, "mpmath"))
        ref = None if fn is None else _attempt(lambda: _reference(fn, lo, hi))
        if ref is None:
            return Verdict(
                UNDECIDED, "none", None,
                "no independent quadrature of this integrand was possible",
            )
        integral, u, scale = ref
        with mpmath.workdps(45):
            allowed = claim + _FLOAT_SLACK * scale
            gap = abs(value - integral)
            if gap > allowed + 10 * u:
                return Verdict(
                    REFUTED, "reference-quadrature", {},
                    f"an independent quadrature (mpmath tanh-sinh, 30 and 45 digits) "
                    f"gives {_show(integral)} ± {mpmath.nstr(u, 2)}; the result "
                    f"{value!r} is off by {mpmath.nstr(gap, 3)}, more than its "
                    f"{source} {claim:.3g} allows",
                )
            numeric = (
                f"an independent quadrature agrees to {mpmath.nstr(gap, 3)} "
                f"(within {mpmath.nstr(allowed, 3)}, reference ± {mpmath.nstr(u, 2)})"
            )
        run.evidence.append(numeric)

        run.phase = "the symbolic definite integral"
        exact = _attempt(lambda: sp.integrate(f, (x, lo, hi)))
        if exact is None or exact.has(sp.Integral, sp.nan, sp.zoo, sp.oo, -sp.oo):
            return Verdict(
                UNDECIDED, "reference-quadrature", None,
                f"{numeric}; no closed form to prove it with",
            )
        closed = _num(exact, {}, 60)
        if closed is None:
            return Verdict(UNDECIDED, "reference-quadrature", None, f"{numeric}; the closed form {exact} has no value")
        with mpmath.workdps(60):
            if abs(closed - integral) > 10 * u + mpmath.mpf("1e-40") * max(1, scale):
                return Verdict(
                    UNDECIDED, "reference-quadrature", None,
                    f"{numeric}; but the symbolic integral {exact} = {_show(closed)} "
                    f"disagrees with it — the two routes contradict each other",
                )
            gap = abs(value - closed)
            if gap <= allowed:
                return Verdict(
                    VERIFIED, "symbolic", None,
                    f"the exact integral is {exact} = {_show(closed)}; the result "
                    f"{value!r} is within its {source} ({mpmath.nstr(gap, 3)} ≤ "
                    f"{mpmath.nstr(allowed, 3)}); {numeric}",
                )
            return Verdict(
                REFUTED, "symbolic", {},
                f"the exact integral is {exact} = {_show(closed)} (confirmed by an "
                f"independent quadrature); the result {value!r} is off by "
                f"{mpmath.nstr(gap, 3)}, more than its {source} {claim:.3g} allows",
            )

    return _certify(op, budget, work)


# --- one entry point -----------------------------------------------------------
#: Operation name (the REPL command token) -> its certificate. Each takes the
#: engine call's own arguments in the engine's order, then the result.
CERTIFIERS: dict[str, Callable[..., Verdict]] = {
    "integrate": certify_integrate,
    "diff": certify_diff,
    "solve": certify_solve,
    "limit": certify_limit,
    "dsolve": certify_dsolve,
    "nintegrate": certify_nintegrate,
}


def certify(operation: str, *args: Any, **kwargs: Any) -> Verdict:
    """Certify one engine result: ``certify("integrate", "2*x", "x", "x^2")``.

    ``operation`` is the REPL command token; the remaining arguments are the
    matching ``certify_<operation>``'s — the engine call's arguments in the engine's
    order, then the result. An unknown operation or arguments that do not fit are a
    ``DomainError``, like every other refusal of input here.
    """
    fn = CERTIFIERS.get(operation) if isinstance(operation, str) else None
    if fn is None:
        raise DomainError(
            f"certify: no certificate for {operation!r} — available: "
            f"{', '.join(CERTIFIERS)}"
        )
    try:
        inspect.signature(fn).bind(*args, **kwargs)
    except TypeError as exc:
        raise DomainError(f"certify {operation}: {exc}") from exc
    return fn(*args, **kwargs)
