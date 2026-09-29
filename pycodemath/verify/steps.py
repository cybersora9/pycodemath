"""``check_steps`` — which step of a derivation is the first wrong one, and where.

A derivation is a chain of steps, checked PAIRWISE: step ``n`` against step
``n - 1``, every pair on its own. A later step is therefore judged on what it
claims about its predecessor, not on whether the chain got there correctly — so
one slip is reported once, at the step that made it, and the steps after it can
still come back VERIFIED. Steps are numbered from 1 as written; step 1 is the
starting point and is never checked itself.

TWO KINDS OF CHAIN, told apart by the text:

* EXPRESSIONS — ``(x+1)^2 - (x-1)^2 = x^2 + 2x + 1 - x^2 + 2x + 1 = 4x + 2``
  (one line, or one step per line, each continuation line starting with ``=``).
  Neighbouring steps must be EQUAL, and equality is ``check_equal``'s, with its
  semantics unchanged (principal branch, where both sides are defined, decimals as
  written). That example refutes step 2 at ``x = 0``: 0 vs 2.
* EQUATIONS — one equation per line, or several on a line separated by an arrow
  (``->``, ``=>``, ``<=>``, ``→``, ``⇒``, ``⇔``); a step may be a disjunction,
  ``x = 2 or x = -2``. Neighbouring steps must have the SAME SOLUTION SET over the
  REAL numbers (values still on the principal branch, so ``sqrt(x) = -1`` has no
  real solution). A step that changes the solution set is REFUTED, with a real
  number that solves one of the two steps and not the other as the counterexample
  — also when the step only GAINS roots (squaring both sides: every old solution
  survives, and the step is still not an equivalence). Such a step carries a
  machine-readable ``effect`` (``"gains-roots"`` / ``"loses-roots"``) and a
  ``warning`` naming the cause when it is recognised: squaring both sides, or
  multiplying / dividing both sides by an expression that is 0 at the root.

HOW AN EQUATION STEP IS DECIDED, cheapest first:

1. PROPORTIONAL — ``left - right`` of the two steps differ by a nonzero constant
   factor and have proportional denominators (so the same poles): VERIFIED,
   ``method="symbolic"``. This is every "add the same to both sides / multiply by
   a nonzero number" step, in any number of variables, and costs one ``cancel``.
2. LOCATED ROOTS (V8) — a step whose ``left - right`` is a rational function with
   rational coefficients and a numerator of degree 4 or more: its real roots
   located by ``nroots``, as many as the Sturm count, each bracketed by an exact
   sign change of the numerator within 1e-30. A located root at which the other
   step is false (at both ends of the bracket and in the middle, each refuted at
   two precisions) refutes the step, ``method="located-root"``. A located root
   never proves anything: finding none, the step goes on to 3 exactly as before.
   Measured on V6's squared ``sqrt(x) = x^2 - 6``: REFUTED in 60 ms, where the
   Ferrari roots from ``solveset`` ran out the whole 30 s budget in ``simplify``.
3. CANDIDATES — the real roots SymPy's ``solveset`` finds for EITHER step, plus
   ``samples`` deterministic sample points, are each substituted into BOTH steps.
   A candidate that solves one and not the other refutes the step. Substitution
   is exact first, then ``check_equal`` on the two numbers, so a counterexample
   is confirmed at two precisions and is never ``solveset``'s word alone — which
   matters, because ``solveset`` is not on the principal branch everywhere:
   measured, ``solveset(log(x^2) - 2*log(x), x, Reals)`` is ``Reals``, while at
   ``x = -1`` the two sides are 0 and ``2*pi*I``.
4. SOLUTION SETS — no candidate disagreed and both steps' solution sets are
   finite with every root CONFIRMED in both steps, or the two sets are the same
   object (``Interval(0, oo)``, an ``ImageSet`` of a period): VERIFIED,
   ``method="solution-sets"``. This one rests on ``solveset`` having found the
   whole set; nothing cheaper can show that no root was missed.
5. Otherwise UNDECIDED, with the number of candidates checked in ``detail``.

Measured cost (warm, median of 5): the plan's ``2x + 3 = 7 -> 2x = 4 -> x = 2``,
two proportional steps, 3.4 ms; a step that needs the candidates 35-65 ms
(``x^2 = 2x -> x = 2``: 35 ms, squaring ``sqrt(x) = x - 2``: 65 ms) — ~40
``check_equal`` calls on numbers. Proportionality goes first because it is 10-20x
cheaper and proves the commonest step outright.

With several variables and no proportionality, the step is solved for one of
them (the first of ``vars``, else the first alphabetically that occurs in both
steps) at a few sample values of the others. A disagreement there is a genuine
counterexample; agreement at every sampled value is UNDECIDED, never VERIFIED.

THE BUDGET, AND AN ENCLOSING ``time_budget`` — the rule V1 measured. ``budget``
is the allowance for the WHOLE derivation (default ``DEFAULT_TIME_BUDGET``); what
is left of it goes to each step in turn, and a step reached with nothing left is
UNDECIDED with ``method="time-budget"``, as is a step the budget stops. Inside a
caller's ``with time_budget(...)`` this module arms NOTHING: a tighter budget
nested in another poisons the outer when it fires (an outer ``time_budget(10)``
refused at 0.26 s). There, ``budget`` is honoured only BETWEEN pieces of work
(sample points, solution sets, parameter values) and inside ``check_equal``'s
sampling; one piece can overrun it (a single ``solveset`` was measured at 6.5 s
against ``budget=0.1``), and the caller's allowance governs that — its expiry
refuses the caller's block exactly as it always does.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import mpmath
import sympy as sp

from ..core.budget import DEFAULT_TIME_BUDGET, time_budget
from ..core.errors import DomainError, TimeBudgetError
from ..core.ir import Expr
from .equal import (
    SAMPLES,
    _as_written,
    _caller_budget_active,
    _parse_written,
    _points,
    check_equal,
)
from .verdict import Verdict, VerdictStatus

__all__ = ["StepCheck", "StepsResult", "check_steps"]

#: Arrows that separate equation steps written on one line. Longest first, so
#: ``<=>`` is not read as ``<`` followed by ``=>``.
_ARROW = re.compile(r"<=>|<->|==>|=>|->|⟺|⇔|⟹|⇒|→")
#: The word joining the alternatives of a disjunction, ``x = 2 or x = -2``.
_OR = re.compile(r"\s+or\s+|∨", re.IGNORECASE)
#: Relations this module does not check. Refused, not misread as an equation.
_INEQUALITY = re.compile(r"[<>≤≥≠]|!=")

#: Values of the other variables a multi-variable equation step is solved at.
_PARAMETER_POINTS = 6
#: Sample points added to the roots when a step is solved at one parameter value
#: — fewer than for a single-variable step, as they are paid once per value.
_PARAMETER_SAMPLES = 8
#: From this degree of a step's numerator on, its real roots are first located by
#: ``nroots`` (see ``_numeric_roots``) — a refutation found there costs
#: milliseconds, while the same root from ``solveset`` is a Ferrari expression
#: whose substitution ``check_equal`` could not settle in the budget (V6: all 5
#: misses, squared ``sqrt(x) = x^2 - a``, 30 s each).
_NUMERIC_DEGREE = 4
#: Half-width of the bracket around a located root, relative to ``max(1, |root|)``.
_BRACKET = mpmath.mpf("1e-30")


@dataclass(frozen=True)
class StepCheck:
    """The verdict on ONE step, taken against the step before it.

    ``number`` is the step's position in the derivation, from 1 (so it is never
    1: the first step is the starting point). ``effect`` is set only on a REFUTED
    equation step whose error is one-sided — ``"gains-roots"`` when every old
    solution survives and new ones appear, ``"loses-roots"`` when no new solution
    appears and old ones vanish — and ``warning`` says it in words, with the cause
    when it is recognised. Branch on ``effect``; ``warning`` is for a human.
    """

    number: int
    text: str
    verdict: Verdict
    effect: "str | None" = None
    warning: "str | None" = None


@dataclass(frozen=True)
class StepsResult:
    """The verdicts on a derivation: one ``StepCheck`` per step after the first.

    ``kind`` is ``"expressions"`` or ``"equations"``; ``steps`` holds every step
    as parsed (``steps[0]`` is step 1). ``first_error`` is the NUMBER of the first
    REFUTED step, ``counterexample`` its counterexample, and ``status`` the
    derivation's: REFUTED if any step is, VERIFIED if every step is, UNDECIDED
    otherwise.
    """

    kind: str
    steps: "tuple[str, ...]"
    checks: "tuple[StepCheck, ...]"

    @property
    def failed(self) -> "StepCheck | None":
        """The first REFUTED step, or ``None``."""
        return next((c for c in self.checks if c.verdict.refuted), None)

    @property
    def first_error(self) -> "int | None":
        failed = self.failed
        return None if failed is None else failed.number

    @property
    def counterexample(self) -> "dict[str, str] | None":
        failed = self.failed
        return None if failed is None else failed.verdict.counterexample

    @property
    def status(self) -> VerdictStatus:
        if self.failed is not None:
            return VerdictStatus.REFUTED
        if all(c.verdict.verified for c in self.checks):
            return VerdictStatus.VERIFIED
        return VerdictStatus.UNDECIDED


# --- reading the derivation --------------------------------------------------
def _split(steps: "str | Sequence[str]") -> "tuple[str, list[str]]":
    """``(kind, step texts)``. See the module docstring for the two notations."""
    if isinstance(steps, str):
        pieces = [
            p.strip()
            for line in steps.splitlines()
            for p in _ARROW.split(line)
        ]
        arrows = bool(_ARROW.search(steps))
    elif isinstance(steps, Sequence) and all(isinstance(s, str) for s in steps):
        pieces = [s.strip() for s in steps]
        arrows = False
    else:
        raise DomainError("check_steps: steps must be text or a list of texts")
    pieces = [p.replace("==", "=") for p in pieces if p]
    for p in pieces:
        if _INEQUALITY.search(p):
            raise DomainError(
                f"check_steps: {p!r} is an inequality — only = is checked"
            )

    continued = any(p.startswith("=") for p in pieces)
    has_equals = any("=" in p for p in pieces)
    if continued or len(pieces) == 1 or not has_equals:
        # An expression chain: every '=' separates two neighbouring steps. A
        # single line is always one — ``x^2 = 4`` alone claims an identity.
        if arrows:
            raise DomainError(
                "check_steps: an arrow separates equations, and these steps "
                "are expressions — continue an expression chain with '='"
            )
        if continued and not all(p.startswith("=") for p in pieces[1:]):
            raise DomainError(
                "check_steps: in an expression chain every line after the "
                "first starts with '='"
            )
        if has_equals:
            texts = [t.strip() for t in " ".join(pieces).lstrip("=").split("=")]
        else:
            texts = pieces
        kind = "expressions"
    else:
        for n, p in enumerate(pieces, 1):
            for alternative in _OR.split(p):
                if alternative.count("=") != 1:
                    raise DomainError(
                        f"check_steps: step {n} ({p!r}) is not one equation — "
                        f"each step of an equation chain has exactly one '='"
                    )
        texts, kind = pieces, "equations"

    if any(not t for t in texts):
        raise DomainError("check_steps: a step is empty (two '=' in a row?)")
    if len(texts) < 2:
        raise DomainError(
            "check_steps: a derivation needs at least two steps to check"
        )
    return kind, texts


def _expr(text: str) -> sp.Expr:
    # An unmatched ')' is the parser's own ParseError since 29.09 (V3 finding 2
    # was translated here from SymPy's IndexError until then).
    sy = _parse_written(text)
    if not isinstance(sy, sp.Expr):
        raise DomainError(f"check_steps: {text!r} is not a scalar expression")
    return _as_written(sy, f"check_steps: {text!r}")


@dataclass(frozen=True)
class _Equation:
    """One equation step: the alternatives ``left = right`` it is a disjunction of."""

    sides: "tuple[tuple[sp.Expr, sp.Expr], ...]"

    @classmethod
    def read(cls, text: str) -> "_Equation":
        sides = []
        for alternative in _OR.split(text):
            left, right = alternative.split("=")
            sides.append((_expr(left), _expr(right)))
        return cls(tuple(sides))

    @property
    def zero(self) -> sp.Expr:
        """An expression that vanishes exactly on the solutions (where defined)."""
        return sp.Mul(*(left - right for left, right in self.sides))

    @property
    def free(self) -> "set[sp.Basic]":
        return set().union(*(sp.Add(a, b).free_symbols for a, b in self.sides))

    def at(self, point: dict) -> "_Equation":
        return _Equation(
            tuple((a.xreplace(point), b.xreplace(point)) for a, b in self.sides)
        )

    def holds(self, point: dict) -> "bool | None":
        """Does ``point`` solve this step? ``None`` when that cannot be settled."""
        answers = [_holds(a, b, point) for a, b in self.sides]
        if True in answers:
            return True
        return None if None in answers else False


def _undefined(value: sp.Expr) -> bool:
    return bool(value.has(sp.nan, sp.zoo, sp.oo, -sp.oo))


def _holds(left: sp.Expr, right: sp.Expr, point: dict) -> "bool | None":
    try:
        a, b = left.xreplace(point), right.xreplace(point)
    except Exception:  # noqa: BLE001 — SymPy could not evaluate it: unsettled
        return None
    if _undefined(a) or _undefined(b):
        return False  # a side with no value there: not a solution
    if a - b == 0:
        return True
    # Both sides are numbers now: check_equal decides at one point, by proof or by
    # a difference confirmed at 60 digits. Arms nothing — this always runs inside
    # the step's budget or the caller's.
    status = check_equal(Expr(a), Expr(b), samples=1).status
    if status is VerdictStatus.VERIFIED:
        return True
    return False if status is VerdictStatus.REFUTED else None


def _variables(
    free: "set[sp.Basic]", vars: "Iterable[str] | str | None"
) -> "tuple[sp.Symbol, ...]":
    if vars is None:
        return tuple(sorted(free, key=lambda s: s.name))  # type: ignore[attr-defined]
    names = vars.replace(",", " ").split() if isinstance(vars, str) else list(vars)
    if not all(isinstance(n, str) and n for n in names):
        raise DomainError("check_steps: vars must be variable names")
    chosen = tuple(sp.Symbol(n) for n in dict.fromkeys(names))
    missing = sorted(s.name for s in free - set(chosen))  # type: ignore[attr-defined]
    if missing:
        raise DomainError(
            f"check_steps: {', '.join(missing)} not listed in vars — a symbol "
            f"that is not sampled cannot be evaluated"
        )
    return chosen


# --- one equation step -------------------------------------------------------
def _constant_ratio(f: sp.Expr, g: sp.Expr) -> "sp.Expr | None":
    """``c`` with ``f == c*g`` and the same poles, or ``None`` if there is none."""
    try:
        nf, df = sp.fraction(sp.together(f))
        ng, dg = sp.fraction(sp.together(g))
        if nf == 0 or ng == 0:
            return None
        ratios = (sp.cancel(nf / ng), sp.cancel(df / dg))
        for q in ratios:
            if q.free_symbols or not q.is_finite or q.is_zero is not False:
                return None
        return sp.cancel(ratios[0] / ratios[1])
    except Exception:  # noqa: BLE001 — no proof this way; the solution sets decide
        return None


def _factor_zero_at(f: sp.Expr, g: sp.Expr, point: dict) -> "sp.Expr | None":
    """``f / g`` when it is a non-constant expression vanishing at ``point`` — the
    divisor (or multiplier) that took a root away (or brought one in)."""
    # Rational only: the question is "which factor was cancelled", and a quotient
    # such as log(x^2)/(2*log(x)) (measured: 0 at x = -1) is not a factor anyone
    # divided by — naming it would be a guess dressed up as a cause.
    try:
        q = sp.cancel(f / g)
        if not q.free_symbols or not q.is_rational_function(*q.free_symbols):
            return None
        return q if q.xreplace(point) == 0 else None
    except Exception:  # noqa: BLE001 — a cause not found is simply not named
        return None


def _candidates(solutions: "sp.Set | None") -> "list[sp.Expr]":
    """Real numbers worth substituting: the roots of a finite set, a few members of
    a periodic family, the ends and an inner point of an interval."""
    found: "list[sp.Expr]" = []
    if solutions is None:
        return found
    if isinstance(solutions, sp.FiniteSet):
        found.extend(solutions.args)  # type: ignore[arg-type]
    elif isinstance(solutions, (sp.Union, sp.Intersection, sp.Complement)):
        for part in solutions.args:
            found.extend(_candidates(part))  # type: ignore[arg-type]
    elif isinstance(solutions, sp.ImageSet) and solutions.base_sets == (sp.S.Integers,):
        found.extend(solutions.lamda(n) for n in (0, 1, -1))
    elif isinstance(solutions, sp.Interval):
        lo, hi = solutions.start, solutions.end
        found.extend(e for e in (lo, hi) if e.is_finite)
        if lo.is_finite and hi.is_finite:
            found.append((lo + hi) / 2)
        elif lo.is_finite:
            found.append(lo + 1)
        elif hi.is_finite:
            found.append(hi - 1)
    return [
        c for c in found
        if isinstance(c, sp.Expr) and not c.free_symbols and c.is_extended_real
        and c.is_finite
    ]


class _OutOfTime(Exception):
    """The step's deadline passed between two pieces of work (checked, not armed)."""


def _tick(deadline: float) -> None:
    # Cooperative: inside a caller's time_budget nothing of ours is armed, so this
    # is all that holds ``budget`` there — between pieces of work. One piece can
    # still overrun it: a single ``solveset`` on a four-variable trig step was
    # measured at 6.5 s against a 0.1 s ``budget``; only the caller's own
    # allowance can stop that, and it does.
    if time.perf_counter() >= deadline:
        raise _OutOfTime


def _solve(eq: _Equation, v: sp.Symbol) -> "sp.Set | None":
    try:
        return sp.solveset(eq.zero, v, sp.S.Reals)
    except Exception:  # noqa: BLE001 — no solution set is a fallback, not a failure
        return None  # (the budget's interrupt is a BaseException and passes)


def _numeric_roots(
    eq: _Equation, v: sp.Symbol
) -> "list[tuple[sp.Rational, ...]] | None":
    """The real solutions of a step whose ``left - right`` is a rational function
    of ``v`` with rational coefficients and a numerator of degree
    ``_NUMERIC_DEGREE`` or more — as exact rationals ``(mid, lo, hi)``, each
    bracketing ONE root: ``nroots`` locates the roots, and a sign change of the
    square-free numerator between ``lo`` and ``hi``, computed exactly, proves a
    root lies there. A root that is a rational with a small denominator comes back
    as that exact rational alone, ``(r,)``, once the numerator vanishes there
    exactly — so a counterexample reads ``x = -2``, not ``x = -2.00000000000000``.
    ``None`` when the step is not of that shape, or ``nroots``
    does not locate exactly as many real roots as the Sturm count (so no root
    rests on ``nroots`` alone, and none is missed); ``[]`` when Sturm says there
    are none. A located root is a candidate, never a proof of anything else."""
    try:
        num, den = sp.fraction(sp.together(eq.zero))
        pn, pd = sp.Poly(num, v), sp.Poly(den, v)
        if (
            not all(p.domain.is_ZZ or p.domain.is_QQ for p in (pn, pd))
            or pn.degree() < _NUMERIC_DEGREE
        ):
            return None
        sq = pn.sqf_part()
        sq = sq.exquo(sq.gcd(pd))
        count = int(sq.count_roots())
        if count == 0:
            return []
        found = sq.nroots(n=50, maxsteps=200)
        with mpmath.workdps(50):
            reals = []
            for c in found:
                re, im = (mpmath.mpf(part) for part in c.as_real_imag())
                if abs(im) <= _BRACKET * max(1, abs(re)):
                    reals.append(re)
            if len(reals) != count:
                return None
            brackets: "list[tuple[sp.Rational, ...]]" = []
            for c in sorted(reals):
                exact = sp.Rational(mpmath.nstr(c, 40)).limit_denominator(1000)
                if sq.eval(exact) == 0:
                    brackets.append((exact,))
                    continue
                d = _BRACKET * max(1, abs(c))
                lo, mid, hi = (sp.Rational(mpmath.nstr(t, 60)) for t in (c - d, c, c + d))
                a, b = sq.eval(lo), sq.eval(hi)
                if a == 0 or b == 0 or (a > 0) == (b > 0):
                    return None
                brackets.append((mid, lo, hi))
        return brackets
    except Exception:  # noqa: BLE001 — no located roots is a route not taken
        return None


def _located_refutation(
    before: _Equation, after: _Equation, v: sp.Symbol, n: int, deadline: float,
    extra: "dict | None",
) -> "_Outcome | None":
    """A root located by ``_numeric_roots`` in one step at which the other step is
    FALSE — at both ends of the root's bracket and at its middle, each by
    ``holds`` (so each point refuted at two precisions); for an exact rational
    root, at that root, which must solve its own step by ``holds`` too. Between two points where
    the other step's sides differ by more than 1e-20 relative, 2e-30 apart, it has
    no root unless it is steeper than 1e10 there. Only refutes; ``None`` when
    nothing is found, and the solution sets are then compared as before V8."""
    for eq, other, effect, which in (
        (after, before, "gains-roots", (n, n - 1)),
        (before, after, "loses-roots", (n - 1, n)),
    ):
        for bracket in _numeric_roots(eq, v) or ():
            _tick(deadline)
            exact = len(bracket) == 1
            if exact and eq.holds({v: bracket[0]}) is not True:
                continue
            if not all(other.holds({v: p}) is False for p in bracket):
                continue
            at = bracket[0] if exact else sp.Float(bracket[0], 15)
            found = {**(extra or {}), v: at}
            point = dict(sorted(found.items(), key=lambda kv: kv[0].name))
            why = (
                f"{v} = {at}, located by nroots, solves step {which[0]} exactly"
                if exact else
                f"{v} ≈ {at} is a root of step {which[0]} located by nroots and "
                f"bracketed by an exact sign change within 1e-30, and step "
                f"{which[1]} is false at both ends of the bracket and at its middle"
            )
            return _Outcome(
                Verdict(
                    VerdictStatus.REFUTED, "located-root",
                    {s.name: str(val) for s, val in point.items()},
                    f"{_solves(point, *which)} — {why}",
                ),
                effect, point,
            )
    return None


def _readable(point: dict) -> str:
    return ", ".join(f"{s.name} = {v}" for s, v in point.items())


def _solves(point: dict, first: int, second: int) -> str:
    if not point:  # no variables: each step is simply true or false
        return f"step {first} is true and step {second} is false"
    return f"{_readable(point)} solves step {first} but not step {second}"


@dataclass(frozen=True)
class _Outcome:
    """What comparing two equation steps found: a verdict, and for a one-sided
    refutation, which side it was on and at which point."""

    verdict: Verdict
    effect: "str | None" = None
    point: "dict | None" = None


def _compare(
    before: _Equation, after: _Equation, v: "sp.Symbol | None",
    n: int, samples: int, seed: int, deadline: float,
    extra: "dict | None" = None,
) -> _Outcome:
    """Solution sets of two steps in ONE variable ``v`` (or none): a point that
    solves exactly one of them, a proof they coincide, or neither. ``extra`` holds
    the values the other variables were fixed at, reported with a counterexample."""
    s1: "sp.Set | None" = None
    s2: "sp.Set | None" = None
    roots: "list[dict]" = []
    points: "list[dict]" = [{}]
    if v is not None:
        located = _located_refutation(before, after, v, n, deadline, extra)
        if located is not None:
            return located
        _tick(deadline)
        s1 = _solve(before, v)
        _tick(deadline)
        s2 = _solve(after, v)
        roots = [{v: r} for r in dict.fromkeys(_candidates(s1) + _candidates(s2))]
        points = roots + _points((v,), samples, seed)

    lost = gained = None
    settled = []  # per point: both steps' answers are known and agree
    for point in points:
        _tick(deadline)
        h1, h2 = before.holds(point), after.holds(point)
        if h1 is True and h2 is False:
            lost = lost or point
        elif h2 is True and h1 is False:
            gained = gained or point
        settled.append(None not in (h1, h2) and h1 is h2)
        if point in roots and not (h1 is True and h2 is True):
            settled[-1] = False  # a root solveset reported that is not confirmed

    if lost is not None or gained is not None:
        found = {**(extra or {}), **(lost if lost is not None else gained)}  # type: ignore[dict-item]
        point = dict(sorted(found.items(), key=lambda kv: kv[0].name))
        which = (n - 1, n) if lost is not None else (n, n - 1)
        effect = None
        if lost is None:
            effect = "gains-roots"
        elif gained is None:
            effect = "loses-roots"
        return _Outcome(
            Verdict(
                VerdictStatus.REFUTED, "solution-sets",
                {s.name: str(val) for s, val in point.items()},
                _solves(point, *which),
            ),
            effect if point else None, point,
        )

    if all(settled):
        if v is None:
            return _Outcome(Verdict(
                VerdictStatus.VERIFIED, "symbolic", None,
                "both steps are " + ("true" if before.holds({}) else "false"),
            ))
        finite = isinstance(s1, sp.FiniteSet) and isinstance(s2, sp.FiniteSet)
        same = s1 is not None and s1 == s2 and not isinstance(s1, sp.ConditionSet)
        if finite or same:
            return _Outcome(Verdict(
                VerdictStatus.VERIFIED, "solution-sets", None,
                f"both steps have the real solution set {s1}",
            ))
    return _Outcome(Verdict(
        VerdictStatus.UNDECIDED, "solution-sets", None,
        f"no point of {len(points)} checked solves one step and not the other, "
        f"and the solution sets are not proved equal",
    ))


def _warning(
    before: _Equation, after: _Equation, outcome: _Outcome, n: int
) -> "str | None":
    if outcome.effect is None or outcome.point is None:
        return None
    point = outcome.point
    f, g = before.zero, after.zero
    if outcome.effect == "loses-roots":
        text = f"step {n} loses the root {_readable(point)}"
        divisor = _factor_zero_at(f, g, point)
        if divisor is not None:
            text += f" — both sides were divided by {divisor}, which is 0 there"
        elif any(_undefined(a.xreplace(point)) or _undefined(b.xreplace(point))
                 for a, b in after.sides):
            text += f" — step {n} is undefined there"
        return text
    text = f"step {n} gains the root {_readable(point)}"
    if len(before.sides) == 1:
        left, right = before.sides[0]
        if _constant_ratio(left**2 - right**2, g) is not None:
            return text + (
                " — squaring both sides adds roots: check each one against "
                f"step {n - 1}"
            )
    multiplier = _factor_zero_at(g, f, point)
    if multiplier is not None:
        text += f" — both sides were multiplied by {multiplier}, which is 0 there"
    return text


def _compare_at_parameters(
    before: _Equation, after: _Equation, present: "list[sp.Symbol]",
    n: int, samples: int, seed: int, deadline: float,
) -> _Outcome:
    """Several variables: solve for one at sampled values of the others. Can
    refute; agreement at every sampled value is evidence, not proof."""
    both = [s for s in present if s in before.free and s in after.free]
    v = (both or present)[0]
    others = tuple(s for s in present if s != v)
    agreed = 0
    for values in _points(others, _PARAMETER_POINTS, seed):
        here = _compare(
            before.at(values), after.at(values), v, n,
            min(samples, _PARAMETER_SAMPLES), seed, deadline, extra=values,
        )
        if here.verdict.refuted:
            return here
        agreed += here.verdict.verified
    return _Outcome(Verdict(
        VerdictStatus.UNDECIDED, "solution-sets", None,
        f"the solution sets in {v} agree at {agreed} of {_PARAMETER_POINTS} "
        f"sampled values of {', '.join(s.name for s in others)}, without a "
        f"proof for all values",
    ))


def _check_equation(
    before: _Equation, after: _Equation, syms: "tuple[sp.Symbol, ...]",
    n: int, text: str, samples: int, seed: int, deadline: float,
) -> StepCheck:
    free = before.free | after.free
    present = [s for s in syms if s in free]
    ratio = _constant_ratio(before.zero, after.zero)
    _tick(deadline)
    if ratio is not None:
        how = "equals" if ratio == 1 else f"is {ratio} times"
        verdict = Verdict(
            VerdictStatus.VERIFIED, "symbolic", None,
            f"left - right of step {n - 1} {how} that of step {n}, with the "
            f"same poles, so the solution sets coincide",
        )
        return StepCheck(n, text, verdict)

    if len(present) <= 1:
        v = present[0] if present else None
        outcome = _compare(before, after, v, n, samples, seed, deadline)
    else:
        outcome = _compare_at_parameters(
            before, after, present, n, samples, seed, deadline
        )
    warning = _warning(before, after, outcome, n)
    return StepCheck(n, text, outcome.verdict, outcome.effect, warning)


# --- the entry point ---------------------------------------------------------
def _check_budget(budget: "float | None") -> float:
    if budget is None:
        return DEFAULT_TIME_BUDGET
    if not isinstance(budget, (int, float)) or isinstance(budget, bool):
        raise DomainError(f"check_steps: budget must be a number, not {budget!r}")
    if math.isnan(budget) or budget <= 0:
        raise DomainError(f"check_steps: budget must be positive, got {budget!r}")
    return float(budget)


def _out_of_time(
    total: float, n: int, started: bool, evidence: str = ""
) -> Verdict:
    when = "while step" if started else "before step"
    return Verdict(
        VerdictStatus.UNDECIDED, "time-budget", None,
        f"the {total:g}s time budget ran out {when} {n} was decided"
        + (f"; {evidence}" if evidence else ""),
    )


def check_steps(
    steps: "str | Sequence[str]",
    *,
    vars: "Iterable[str] | str | None" = None,
    budget: "float | None" = None,
    samples: int = SAMPLES,
    seed: int = 0,
) -> StepsResult:
    """Check a derivation step by step: a verdict per step, the first wrong one,
    and a counterexample for it.

        >>> r = check_steps('''(x+1)^2 - (x-1)^2
        ...                    = x^2 + 2x + 1 - x^2 + 2x + 1
        ...                    = 4x + 2''')
        >>> r.first_error, r.counterexample
        (2, {'x': '0'})

    ``steps`` is the derivation as text (one step per line; see the module
    docstring for expression vs equation chains) or as a list of step texts.
    ``vars``, ``samples`` and ``seed`` mean what they mean for ``check_equal``;
    ``vars`` must cover every variable of every step. ``budget`` is the allowance
    for the whole derivation, in seconds.

    Every failure is a ``PycodemathError``: a step that does not parse is a
    ``ParseError``; an inequality, a malformed chain or a bad argument is a
    ``DomainError``. A budget that runs out is an UNDECIDED step, not an error —
    except inside a caller's ``time_budget``, whose expiry is the caller's.
    """
    total = _check_budget(budget)
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise DomainError(
            f"check_steps: samples must be a positive integer, not {samples!r}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DomainError(f"check_steps: seed must be an integer, not {seed!r}")

    kind, texts = _split(steps)
    # Everything is parsed before anything is checked: a typo in the last step is
    # a ParseError now, not after the budget has been spent on the first ones.
    exprs: "list[sp.Expr]" = []
    equations: "list[_Equation]" = []
    if kind == "expressions":
        exprs = [_expr(t) for t in texts]
        free = set().union(*(e.free_symbols for e in exprs))
    else:
        equations = [_Equation.read(t) for t in texts]
        free = set().union(*(e.free for e in equations))
    syms = _variables(free, vars)
    names = [s.name for s in syms]

    inside = _caller_budget_active()
    deadline = time.perf_counter() + total  # inf stays inf
    checks = []
    for i in range(1, len(texts)):
        n = i + 1
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            checks.append(StepCheck(n, texts[i], _out_of_time(total, n, False)))
            continue
        if kind == "expressions":
            verdict = check_equal(
                Expr(exprs[i - 1]), Expr(exprs[i]), vars=names,
                budget=remaining, samples=samples, seed=seed,
            )
            if verdict.method == "time-budget":
                # check_equal quotes the budget it was handed — what was LEFT of
                # ours; the derivation's allowance is the number the caller chose.
                evidence = verdict.detail.partition("; ")[2]
                verdict = _out_of_time(total, n, True, evidence)
            checks.append(StepCheck(n, texts[i], verdict))
            continue
        pair = (
            equations[i - 1], equations[i], syms, n, texts[i], samples, seed,
            deadline,
        )
        try:
            if inside:
                # The caller's budget governs, and its expiry is the caller's:
                # its TimeBudgetError is deliberately NOT caught here.
                check = _check_equation(*pair)
            else:
                try:
                    with time_budget(remaining, "check_steps"):
                        check = _check_equation(*pair)
                except TimeBudgetError:
                    raise _OutOfTime from None
        except _OutOfTime:
            check = StepCheck(n, texts[i], _out_of_time(total, n, True))
        checks.append(check)
    return StepsResult(kind, tuple(texts), tuple(checks))
