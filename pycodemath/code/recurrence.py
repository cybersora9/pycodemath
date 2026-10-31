"""C5 — recurrences: solved with their conditions, never from memory.

* ``solve_recurrence("T(n) = T(n-1) + T(n-2), T(0) = 0, T(1) = 1")`` — a linear
  recurrence with constant shifts goes to SymPy's ``rsolve`` UNDER A BUDGET; the
  answer is then checked three ways: substituted back into the recurrence
  (symbolic zero), against the initial conditions, and against the values the
  recurrence itself produces for the first 30+ arguments (exact arithmetic). Its
  Θ comes from the dominant term of the exponential polynomial (a proof, not a fit).
* ``T(n) = a T(n/b) + f(n)`` goes to ``master_theorem`` — the CLRS theorem with
  the generalised case 2 (``f = Θ(n^c log^k n)``, ``k >= 0``), every condition
  checked by a limit, including the REGULARITY condition of case 3. Outside the
  theorem it raises ``MasterTheoremInapplicable`` with the reason; ``solve_recurrence``
  then tries ``akra_bazzi`` (several sub-problem sizes, the ``n/log n`` gap…), whose
  own conditions (polynomial growth of ``g``) are checked too; failing both, the
  answer is UNDECIDED with both reasons. Never a silent answer outside a theorem.
* ``recurrence_from_code(func)`` reads ``T(n)`` from a recursive function (its base
  case and its self-calls), solves it, and certifies the READING by counting real
  calls in the ``isolated`` worker.

    >>> r = solve_recurrence("T(n) = 2*T(n/2) + n")
    >>> r.theta, r.method
    (n*log(n), 'master theorem, case 2')
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import mpmath
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..core.budget import guard
from ..core.errors import DomainError, ParseError, PycodemathError, TimeBudgetError
from ..verify.verdict import VerdictStatus
from .frontend import _load
from .loops import (
    CountCertificate,
    LoopShapeError,
    _ToDict,
    _count_worker,
    _exec_spec,
    _limit,
    _smooth,
    theta_of,
)

__all__ = [
    "solve_recurrence",
    "master_theorem",
    "akra_bazzi",
    "recurrence_from_code",
    "RecurrenceSolution",
    "MasterTheoremResult",
    "AkraBazziResult",
    "CodeRecurrence",
    "MasterTheoremInapplicable",
    "AkraBazziInapplicable",
]


class MasterTheoremInapplicable(DomainError):
    """``T(n) = a T(n/b) + f(n)`` is outside the master theorem's conditions.

    ``reason`` says which condition failed (``"gap: …"``, ``"regularity …"``,
    ``"a must be a constant"``…). ``alternative`` names the API that may still
    answer (``"akra_bazzi"``) or is ``None``. ``route`` is ``None``: there is no
    numerical command for an asymptotic question.
    """

    def __init__(self, reason: str, alternative: "str | None" = "akra_bazzi") -> None:
        super().__init__(f"master theorem does not apply: {reason}")
        self.reason = reason
        self.alternative = alternative

    def __reduce__(self):
        return (type(self), (self.reason, self.alternative))


class AkraBazziInapplicable(DomainError):
    """The Akra–Bazzi theorem's conditions could not be established."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Akra–Bazzi does not apply: {reason}")
        self.reason = reason

    def __reduce__(self):
        return (type(self), (self.reason,))


N = sp.Symbol("n", positive=True)  # asymptotic variable (real, > 0)
NI = sp.Symbol("n", integer=True)  # the recurrence's own argument


# --------------------------------------------------------------------------- #
# results                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MasterTheoremResult(_ToDict):
    """``T(n) = a T(n/b) + f(n)`` is Θ(``theta``) by case ``case`` of the master
    theorem; ``checks`` are the conditions, each with the limit that proves it."""

    kind = "recurrence"
    a: sp.Expr
    b: sp.Expr
    f: sp.Expr
    critical_exponent: sp.Expr
    case: int
    theta: sp.Expr
    checks: tuple[str, ...]
    method: str = "master theorem"


@dataclass(frozen=True)
class AkraBazziResult(_ToDict):
    """``T(n) = sum a_i T(b_i n + h_i) + g(n)`` is Θ(``theta``) by Akra–Bazzi:
    ``p`` solves ``sum a_i b_i**p = 1``; ``theta`` = Θ(n^p (1 + ∫ g(u)/u^(p+1) du))."""

    kind = "recurrence"
    terms: tuple[tuple[sp.Expr, sp.Expr], ...]
    g: sp.Expr
    p: sp.Expr
    p_exact: bool
    integral: "sp.Expr | None"
    theta: sp.Expr
    checks: tuple[str, ...]
    method: str = "Akra–Bazzi"


@dataclass(frozen=True)
class RecurrenceSolution(_ToDict):
    """The answer of ``solve_recurrence``.

    ``main`` is the exact closed form (``None`` when none was found or none is
    claimed); ``exact_on_powers`` the exact value for ``n = b**k`` of a
    divide-and-conquer recurrence when its initial value is given. ``theta`` is
    proved by ``method``. ``verdict``: VERIFIED when every claimed value is proved
    (substitution + initial conditions + the recurrence's own values), UNDECIDED
    with ``reason`` otherwise.
    """

    kind = "recurrence"
    recurrence: str
    form: str  # "linear" | "divide-and-conquer" | "unknown"
    main: "sp.Expr | None"
    valid_for: str
    exact_on_powers: "sp.Expr | None"
    theta: "sp.Expr | None"
    method: str
    verdict: VerdictStatus
    reason: str
    checks: tuple[str, ...]
    exact: bool = True
    certificate: "CountCertificate | None" = None


@dataclass(frozen=True)
class CodeRecurrence(_ToDict):
    """``recurrence_from_code``: the recurrence a recursive function obeys.

    ``measure`` is ``"calls"`` (how many times the function is entered for one
    top-level call). ``recurrence`` is the reading (base case and self-calls),
    ``solution`` its solution, ``certificate`` the counted calls of a REAL
    execution compared with the recurrence's values, exactly.
    """

    kind = "recurrence"
    function: str
    measure: str
    recurrence: str
    base: str
    solution: RecurrenceSolution
    certificate: "CountCertificate | None"
    verdict: VerdictStatus

    @property
    def main(self) -> "sp.Expr | None":
        return self.solution.main

    @property
    def theta(self) -> "sp.Expr | None":
        return self.solution.theta

    @property
    def exact(self) -> bool:
        return self.solution.exact


# --------------------------------------------------------------------------- #
# limits and growth                                                            #
# --------------------------------------------------------------------------- #


def _lim(e: sp.Expr) -> "sp.Expr | None":
    return _limit(e.subs(NI, N), N)


def _order(f: sp.Expr) -> "sp.Expr | None":
    """``lim log f / log n`` — the polynomial order (``oo`` for exponential).

    Fallback for an oscillating ``f`` (``n*(2 - cos n)``) where SymPy cannot take
    the log-limit: the integer ``K`` with ``f/n^K`` confined to a positive bounded
    interval (then ``log f / log n -> K`` exactly)."""
    L = _lim(sp.expand_log(sp.log(f), force=True) / sp.log(N))
    if L is not None and not L.has(sp.AccumBounds) and L is not sp.nan:
        return L
    for K in range(-3, 6):
        R = _lim(f / N**K)
        if isinstance(R, sp.AccumBounds) and _cmp(R.min, sp.Integer(0)) == 1 and R.max.is_finite:
            return sp.Integer(K)
    return None


def _sign_eventually(f: sp.Expr) -> "int | None":
    L = _lim(f)
    if L is not None and L.has(sp.AccumBounds):
        # oscillating (n*(2 - cos n)): positive if f/n^K stays in a positive interval
        for K in (0, 1, 2, 3):
            R = _lim(f / N**K)
            if isinstance(R, sp.AccumBounds) and _cmp(R.min, sp.Integer(0)) == 1:
                return 1
            if isinstance(R, sp.AccumBounds) and _cmp(R.max, sp.Integer(0)) == -1:
                return -1
        return None
    if L is None or L is sp.nan:
        return None
    if L == sp.oo or (L.is_positive is True):
        return 1
    if L == -sp.oo or (L.is_negative is True):
        return -1
    if L == 0:
        # tends to 0: positive if f * n**K tends to a positive limit for some K
        for K in (1, 2, 4, 8):
            L2 = _lim(f * N**K)
            if L2 is not None and not L2.has(sp.AccumBounds) and L2 != 0:
                return 1 if (L2 == sp.oo or L2.is_positive) else -1
    return None


def _cmp(a: sp.Expr, b: sp.Expr) -> "int | None":
    """Exact sign of ``a - b`` (``None`` when undecided)."""
    if a == sp.oo and b != sp.oo:
        return 1
    d = sp.simplify(a - b)
    if d == 0:
        return 0
    if d.is_positive:
        return 1
    if d.is_negative:
        return -1
    try:
        v = sp.N(d, 60)
        if abs(v) > sp.Float("1e-50"):
            return 1 if v > 0 else -1
    except Exception:  # noqa: BLE001
        pass
    return None


def _fmt_theta(e: sp.Expr) -> sp.Expr:
    return sp.powsimp(sp.expand_log(e, force=True)).subs(N, NI)


# --------------------------------------------------------------------------- #
# master theorem                                                              #
# --------------------------------------------------------------------------- #


def master_theorem(a: Any, b: Any, f: Any, *, var: "str | sp.Symbol" = "n") -> MasterTheoremResult:
    """Θ of ``T(n) = a T(n/b) + f(n)`` by the master theorem — or a refusal.

    Conditions checked (each by a symbolic limit): ``a >= 1`` and ``b > 1``
    constants; ``f`` asymptotically positive; then with ``c = log_b a`` and the
    order ``p = lim log f / log n``:

    * case 1 — ``p < c``: ``f = O(n^(c-ε))`` with ``ε = (c-p)/2``: Θ(n^c);
    * case 2 — ``p = c`` and ``f = Θ(n^c log^k n)`` with ``k >= 0``: Θ(n^c log^(k+1) n);
    * case 3 — ``p > c`` AND regularity ``lim a f(n/b)/f(n) < 1``: Θ(f).

    ``p = c`` with ``k < 0`` (``n/log n``) is the GAP: refused. Floors and ceilings
    in ``T(n/b)`` do not change the answer (CLRS §4.6.2).

        >>> master_theorem(2, 2, "n").theta
        n*log(n)
        >>> try:
        ...     master_theorem(2, 2, "n/log(n)")
        ... except MasterTheoremInapplicable as exc:
        ...     print(exc.reason[:4], exc.alternative)
        gap: akra_bazzi
    """
    n = sp.Symbol(str(var), positive=True)
    a_ = sp.sympify(a, locals={str(var): n})
    b_ = sp.sympify(b, locals={str(var): n})
    f_ = sp.sympify(f, locals={str(var): n}).subs({n: N, NI: N})
    a_, b_ = a_.subs({n: N, NI: N}), b_.subs({n: N, NI: N})
    if a_.free_symbols or b_.free_symbols:
        raise MasterTheoremInapplicable("a and b must be a constant (they depend on n)", None)
    if not (a_.is_real and _cmp(a_, sp.Integer(1)) in (0, 1)):
        raise MasterTheoremInapplicable(f"needs a >= 1 (a = {a_})")
    if not (b_.is_real and _cmp(b_, sp.Integer(1)) == 1):
        raise MasterTheoremInapplicable(f"needs b > 1 (b = {b_})", None)
    if f_.free_symbols - {N}:
        raise MasterTheoremInapplicable(f"f has symbols other than n: {f_}", None)
    s = _sign_eventually(f_)
    if s != 1:
        raise MasterTheoremInapplicable(
            "f must be asymptotically positive" + (" (it is eventually negative)" if s == -1 else
                                                   " (its sign could not be decided)"), None if s == -1 else "akra_bazzi")
    c = sp.simplify(sp.log(a_) / sp.log(b_))
    c = sp.nsimplify(c) if c.is_Float else c
    checks = [f"a = {a_} >= 1, b = {b_} > 1 (constants)", "f(n) > 0 for all large n", f"c = log_b(a) = {c}"]
    p = _order(f_)
    if p is None or p.has(sp.AccumBounds) or p is sp.nan:
        raise MasterTheoremInapplicable("the growth order of f could not be determined (lim log f / log n)")
    checks.append(f"lim log f / log n = {p}")
    rel = _cmp(p, c)
    if rel is None:
        raise MasterTheoremInapplicable(f"could not compare the order of f ({p}) with log_b a ({c})")
    if rel < 0:
        eps = (c - p) / 2
        checks.append(f"case 1: f = O(n^(c - ε)) with ε = {sp.simplify(eps)} > 0")
        return MasterTheoremResult(a_, b_, f_.subs(N, NI), c, 1, _fmt_theta(N**c), tuple(checks),
                                   "master theorem, case 1")
    if rel > 0:
        if p != sp.oo:
            checks.append(f"f = Ω(n^(c + ε)) with ε = {sp.simplify((p - c) / 2)} > 0")
        else:
            checks.append("f grows faster than every power of n")
        R = _lim(a_ * f_.subs(N, N / b_) / f_)
        if R is None or R is sp.nan:
            raise MasterTheoremInapplicable("regularity condition a f(n/b) <= k f(n) (k < 1) could not be proved: "
                                            "the limit of a f(n/b)/f(n) was not found")
        if R.has(sp.AccumBounds):
            hi = R.max if isinstance(R, sp.AccumBounds) else None
            raise MasterTheoremInapplicable(
                f"regularity condition fails: a f(n/b)/f(n) oscillates in {R}" +
                (" and reaches values >= 1" if hi is not None and _cmp(hi, sp.Integer(1)) in (0, 1) else ""))
        if _cmp(R, sp.Integer(1)) != -1:
            raise MasterTheoremInapplicable(f"regularity condition fails: lim a f(n/b)/f(n) = {R}, not < 1")
        checks.append(f"case 3 regularity: lim a f(n/b)/f(n) = {R} < 1, so a f(n/b) <= k f(n) "
                      f"for k = {(R + 1) / 2} and all large n")
        g, L, why = theta_of(f_.subs(N, NI), NI)
        theta = g if g is not None else f_.subs(N, NI)
        return MasterTheoremResult(a_, b_, f_.subs(N, NI), c, 3, theta, tuple(checks), "master theorem, case 3")
    # p == c: case 2 needs f = Θ(n^c log^k n), k >= 0
    k = _lim(sp.expand_log(sp.log(f_ / N**c), force=True) / sp.log(sp.log(N)))
    if k is None or k.has(sp.AccumBounds) or k in (sp.oo, -sp.oo) or k is sp.nan:
        raise MasterTheoremInapplicable(f"f has the critical order n^{c} but is not of the form n^c log^k n")
    L = _lim(f_ / (N**c * sp.log(N) ** k))
    if L is None or L.has(sp.AccumBounds) or not (L.is_positive and L.is_finite):
        raise MasterTheoremInapplicable(f"f is not Θ(n^{c} log^{k} n) (the limit of the ratio is {L})")
    checks.append(f"lim f / (n^c log^k n) = {L} with k = {k}")
    if _cmp(k, sp.Integer(0)) == -1:
        raise MasterTheoremInapplicable(
            f"gap: f = Θ(n^{c} log^{k} n) with k = {k} < 0 lies between case 1 and case 2")
    checks.append(f"case 2 (k = {k} >= 0)")
    return MasterTheoremResult(a_, b_, f_.subs(N, NI), c, 2, _fmt_theta(N**c * sp.log(N) ** (k + 1)),
                               tuple(checks), "master theorem, case 2")


# --------------------------------------------------------------------------- #
# Akra–Bazzi                                                                  #
# --------------------------------------------------------------------------- #


def akra_bazzi(terms: Sequence[tuple[Any, Any]], g: Any, *, var: "str | sp.Symbol" = "n") -> AkraBazziResult:
    """Θ of ``T(x) = sum a_i T(b_i x + h_i(x)) + g(x)`` by the Akra–Bazzi theorem.

    ``terms`` are ``(a_i, b_i)`` with constants ``a_i > 0``, ``0 < b_i < 1``
    (floors/ceilings and ``h_i = O(x/log² x)`` do not change the result).
    Checked: ``g >= 0`` eventually and of POLYNOMIAL GROWTH — proved here by
    regular variation (``lim x g'(x)/g(x)`` finite) or by ``g/x^q`` staying in a
    positive bounded interval. ``p`` solves ``sum a_i b_i**p = 1`` (exactly when
    SymPy can, else by bisection to 40 digits with a certified bracket); then
    Θ(n^p (1 + ∫_2^n g(u)/u^(p+1) du)).

        >>> akra_bazzi([(1, sp.Rational(1, 3)), (1, sp.Rational(2, 3))], "n").theta
        n*log(n)
        >>> akra_bazzi([(2, sp.Rational(1, 2))], "n/log(n)").theta
        n*log(log(n))
    """
    n = sp.Symbol(str(var), positive=True)
    g_ = sp.sympify(g, locals={str(var): n}).subs({n: N, NI: N})
    ts = [(sp.sympify(a), sp.sympify(b)) for a, b in terms]
    if not ts:
        raise AkraBazziInapplicable("no recursive term")
    for a, b in ts:
        if a.free_symbols or b.free_symbols:
            raise AkraBazziInapplicable("a_i and b_i must be constants")
        if _cmp(a, sp.Integer(0)) != 1:
            raise AkraBazziInapplicable(f"needs a_i > 0 (a = {a})")
        if not (_cmp(b, sp.Integer(0)) == 1 and _cmp(b, sp.Integer(1)) == -1):
            raise AkraBazziInapplicable(f"needs 0 < b_i < 1 (b = {b})")
    checks = [f"terms {[(str(a), str(b)) for a, b in ts]}: a_i > 0, 0 < b_i < 1"]
    if g_ != 0:
        s = _sign_eventually(g_)
        if s != 1:
            raise AkraBazziInapplicable("g must be non-negative for large n" +
                                        (" (it is eventually negative)" if s == -1 else " (sign undecided)"))
        checks.append("g(n) > 0 for all large n")
    # p
    P = sp.Symbol("p", real=True)
    eq = sp.Add(*[a * b**P for a, b in ts]) - 1
    p_exact = True
    p: sp.Expr
    if len(ts) == 1:
        a, b = ts[0]
        p = sp.simplify(-sp.log(a) / sp.log(b))
    else:
        sols = []
        try:
            sols = guard("solve", eq, lambda: sp.solve(eq, P))
        except Exception:  # noqa: BLE001
            sols = []
        real = [s for s in sols if s.is_real and sp.simplify(eq.subs(P, s)) == 0]
        if not real:  # small exact candidates first: (1/3)^p + (2/3)^p = 1 at p = 1
            real = [sp.Rational(c, 2) for c in range(-6, 13)
                    if sp.simplify(eq.subs(P, sp.Rational(c, 2))) == 0][:1]
        if len(real) == 1:
            p = real[0]
        else:
            p_exact = False
            p = _bisect_p(ts)
    checks.append(f"p = {p} solves sum a_i b_i^p = 1" + ("" if p_exact else " (bisection, bracket < 1e-40)"))
    # polynomial growth of g
    if g_ != 0:
        growth = _polynomial_growth(g_)
        if growth is None:
            raise AkraBazziInapplicable("g's polynomial-growth condition could not be proved")
        checks.append(growth)
    if g_ == 0:
        return AkraBazziResult(tuple(ts), g_.subs(N, NI), p, p_exact, sp.Integer(0), _fmt_theta(N**p),
                               tuple(checks))
    u = sp.Symbol("u", positive=True)
    integral: "sp.Expr | None" = None
    if p_exact:
        integrand = g_.subs(N, u) / u ** (p + 1)
        try:
            integral = guard("integrate", integrand, lambda: sp.integrate(integrand, (u, 2, N)))
        except Exception:  # noqa: BLE001
            integral = None
        if integral is not None and integral.has(sp.Integral):
            integral = None
    if integral is not None:
        total = N**p * (1 + integral)
        th, L, why = theta_of(total.subs(N, NI), NI)
        if th is None:
            raise AkraBazziInapplicable(f"Θ of n^p(1 + ∫ g/u^(p+1)) not found: {why}")
        checks.append(f"∫_2^n g(u)/u^(p+1) du = {integral}")
        return AkraBazziResult(tuple(ts), g_.subs(N, NI), p, p_exact, integral, th, tuple(checks))
    # no closed integral: decide by the order of g against p
    q = _order(g_)
    if q is None or q.has(sp.AccumBounds) or q is sp.nan:
        raise AkraBazziInapplicable("the integral has no closed form and g's order is unknown")
    rel = _cmp(q, p)
    if rel == -1:
        checks.append(f"g = O(n^(p - ε)) (order {q} < p): the integral converges")
        return AkraBazziResult(tuple(ts), g_.subs(N, NI), p, p_exact, None, _fmt_theta(N**p), tuple(checks))
    rho = _lim(N * sp.diff(g_, N) / g_)
    if rel == 1 and rho is not None and not rho.has(sp.AccumBounds) and rho.is_finite:
        # Karamata: g regularly varying with index q > p, so the integrand has index
        # q - p - 1 > -1 and the integral is ~ g(n) n^-p / (q - p): the total is Θ(g)
        th, L, why = theta_of(g_.subs(N, NI), NI)
        if th is not None:
            checks.append(f"g regularly varying with index {rho} > p: ∫ g(u)/u^(p+1) du ~ g(n) n^-p/(q - p) "
                          f"(Karamata), so T = Θ(g)")
            return AkraBazziResult(tuple(ts), g_.subs(N, NI), p, p_exact, None, th, tuple(checks))
    raise AkraBazziInapplicable(f"the integral has no closed form and g's order ({q}) is not below p ({p})")


def _bisect_p(ts: Sequence[tuple[sp.Expr, sp.Expr]]) -> sp.Expr:
    """The unique real root of ``sum a_i b_i**p = 1`` (strictly decreasing in p)."""
    with mpmath.workdps(60):
        fa = [(mpmath.mpf(sp.N(a, 70)), mpmath.mpf(sp.N(b, 70))) for a, b in ts]

        def F(x: Any) -> Any:
            return sum(a * b**x for a, b in fa) - 1

        lo, hi = mpmath.mpf(-1), mpmath.mpf(1)
        while F(lo) < 0:
            lo *= 2
        while F(hi) > 0:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            if F(mid) > 0:
                lo = mid
            else:
                hi = mid
        return sp.Float(mpmath.nstr((lo + hi) / 2, 45), 45)


def _polynomial_growth(g: sp.Expr) -> "str | None":
    """Why ``g`` satisfies Akra–Bazzi's polynomial-growth condition, or ``None``."""
    rho = _lim(N * sp.diff(g, N) / g)
    if rho is not None and not rho.has(sp.AccumBounds) and rho.is_finite:
        return (f"polynomial growth: lim n g'(n)/g(n) = {rho} (finite), so g is regularly varying "
                f"and g(u)/g(n) is bounded above and below for u in [b n, n]")
    q = _order(g)
    if q is not None and not q.has(sp.AccumBounds) and q.is_finite:
        R = _lim(g / N**q)
        if isinstance(R, sp.AccumBounds) and _cmp(R.min, sp.Integer(0)) == 1 and R.max.is_finite:
            return f"polynomial growth: g/n^{q} stays in {R} (positive, bounded)"
        if R is not None and not R.has(sp.AccumBounds) and R.is_positive and R.is_finite:
            return f"polynomial growth: g ~ {R} n^{q}"
    return None


# --------------------------------------------------------------------------- #
# parsing a recurrence                                                        #
# --------------------------------------------------------------------------- #

_T = sp.Function("T")


def _split_top(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch in ",;" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _parse_side(text: str, fname: str, var: str) -> sp.Expr:
    t = re.sub(r"([\w.]+|\([^()]*\))\s*//\s*([\w.]+|\([^()]*\))", r"floor((\1)/(\2))", text)
    t = t.replace("⌊", "floor(").replace("⌋", ")").replace("⌈", "ceiling(").replace("⌉", ")")
    local = {fname: _T, var: NI, "log": sp.log, "lg": lambda x: sp.log(x, 2), "floor": sp.floor,
             "ceil": sp.ceiling, "ceiling": sp.ceiling, "sqrt": sp.sqrt}
    try:
        return sp.sympify(parse_expr(t, local_dict=local,
                                     transformations=standard_transformations
                                     + (implicit_multiplication_application, convert_xor)))
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"could not read {text!r} as part of a recurrence: {exc}") from exc


@dataclass
class _Rec:
    rhs: sp.Expr
    shifts: list[tuple[sp.Expr, int]]  # (coefficient, k) for T(n - k)
    scales: list[tuple[sp.Expr, sp.Expr, sp.Expr]]  # (a, b, h) for T(b n + h), 0 < b < 1
    g: sp.Expr
    init: dict[int, sp.Expr]
    text: str


def _parse_recurrence(rec: "str | sp.Basic", initial: "Mapping[int, Any] | None", fname: str, var: str) -> _Rec:
    init: dict[int, sp.Expr] = {}
    if isinstance(rec, str):
        parts = _split_top(rec)
        main = None
        for part in parts:
            if "=" not in part:
                raise ParseError(f"recurrence part without '=': {part!r}")
            lhs_t, rhs_t = part.split("=", 1)
            lhs, rhs = _parse_side(lhs_t.strip(), fname, var), _parse_side(rhs_t.strip(), fname, var)
            if lhs == _T(NI):
                if main is not None:
                    raise ParseError("two equations for T(n)")
                main = rhs
            elif isinstance(lhs, AppliedUndef) and lhs.args[0].is_Integer and not rhs.has(_T):
                init[int(lhs.args[0])] = rhs
            else:
                raise ParseError(f"cannot read {part!r}: expected T(n) = … or T(k) = value")
        if main is None:
            raise ParseError("no equation for T(n)")
        rhs = main
        text = rec
    else:
        e = rec
        if isinstance(e, sp.Eq):
            e = e.rhs if e.lhs == _T(NI) else e.lhs - e.rhs + _T(NI)
        rhs = sp.sympify(e)
        text = f"T(n) = {rhs}"
    for k, v in (initial or {}).items():
        init[int(k)] = sp.sympify(v)
    shifts: list[tuple[sp.Expr, int]] = []
    scales: list[tuple[sp.Expr, sp.Expr, sp.Expr]] = []
    g = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(rhs)):
        apps = [a for a in term.atoms(AppliedUndef) if a.func == _T]
        if not apps:
            g += term
            continue
        if len(apps) != 1:
            raise ParseError(f"term {term} multiplies two T(...) — not a linear recurrence")
        app = apps[0]
        coeff = sp.simplify(term / app)
        if coeff.has(_T):
            raise ParseError(f"term {term} is not linear in T")
        arg = app.args[0]
        d = sp.simplify(arg - NI)
        smooth, thetas = _smooth(arg, NI)
        if d.is_Integer:
            if d >= 0:
                raise ParseError(f"T({arg}) on the right refers to T(n) or later — not a recurrence")
            shifts.append((coeff, int(-d)))
            continue
        # b = lim arg/n; floors and ceilings are their argument plus a bounded theta,
        # which the theorems allow (CLRS 4.6.2, Leighton's h_i = O(n/log^2 n))
        bcoef = _lim(smooth.subs({t: 0 for t in thetas}) / NI)
        if bcoef is None or bcoef.free_symbols or not (bcoef.is_positive and bcoef < 1):
            raise ParseError(f"T({arg}): the argument is neither n - k nor b·n with 0 < b < 1")
        h = sp.expand(smooth.subs({t: 0 for t in thetas}) - bcoef * NI)
        if h.has(NI) and _lim(h / (N / sp.log(N) ** 2)) != 0:
            raise ParseError(f"T({arg}): the perturbation {h} is not O(n/log² n)")
        scales.append((coeff, bcoef, h))
    return _Rec(rhs, shifts, scales, sp.simplify(g), init, text)


# --------------------------------------------------------------------------- #
# solve_recurrence                                                            #
# --------------------------------------------------------------------------- #


def solve_recurrence(
    recurrence: "str | sp.Basic",
    initial: "Mapping[int, Any] | None" = None,
    *,
    func: str = "T",
    var: str = "n",
    budget: float = 20.0,
) -> RecurrenceSolution:
    """Solve a recurrence; every result checked, every gap said.

        >>> r = solve_recurrence("T(n) = T(n-1) + T(n-2), T(0) = 0, T(1) = 1")
        >>> r.verdict.value, r.theta
        ('verified', (1/2 + sqrt(5)/2)**n)
        >>> solve_recurrence("T(n) = 2*T(n/2) + n/log(n)").theta
        n*log(log(n))
    """
    rec = _parse_recurrence(recurrence, initial, func, var)
    if rec.shifts and rec.scales:
        return RecurrenceSolution(rec.text, "unknown", None, "False", None, None, "none", VerdictStatus.UNDECIDED,
                                  "mixes T(n - k) and T(n/b): neither rsolve nor the master theorem applies",
                                  (), False)
    if rec.scales:
        return _solve_dc(rec, budget)
    if rec.shifts:
        return _solve_linear(rec, budget)
    raise ParseError("the right-hand side has no T(...) term")


def _solve_linear(rec: _Rec, budget: float) -> RecurrenceSolution:
    order = max(k for _, k in rec.shifts)
    const = all(not c.has(NI) for c, _ in rec.shifts)
    eq = _T(NI) - rec.rhs
    init = {_T(sp.Integer(k)): v for k, v in rec.init.items()}
    checks: list[str] = []
    try:
        from ..core.budget import time_budget

        with time_budget(budget, "rsolve", eq):
            sol = guard("rsolve", eq, lambda: sp.rsolve(eq, _T(NI), init or None))
    except TimeBudgetError as exc:
        return RecurrenceSolution(rec.text, "linear", None, "False", None, None, "rsolve", VerdictStatus.UNDECIDED,
                                  f"rsolve ran out of its budget ({exc})", (), False)
    except Exception as exc:  # noqa: BLE001 — SymPy's own failure is data
        sol = None
        checks.append(f"rsolve raised {type(exc).__name__}")
    if sol is None and order == 1 and len(rec.shifts) == 1 and const and len(rec.init) == 1:
        # first order T(n) = c T(n-1) + g(n): T(n) = c^(n-n0) T(n0) + sum_{k=n0+1}^{n} c^(n-k) g(k)
        (c1, _), (n0, v0) = rec.shifts[0], next(iter(rec.init.items()))
        k = sp.Symbol("k", integer=True)
        try:
            ssum = guard("summation", rec.g, lambda: sp.summation(c1 ** (NI - k) * rec.g.subs(NI, k), (k, n0 + 1, NI)))
            if not ssum.has(sp.Sum):
                sol = c1 ** (NI - n0) * v0 + ssum
                checks.append("rsolve gave nothing; solved by the first-order summation formula")
        except Exception:  # noqa: BLE001
            pass
    if sol is None:
        return RecurrenceSolution(rec.text, "linear", None, "False", None, None, "rsolve", VerdictStatus.UNDECIDED,
                                  "rsolve found no closed form" + ("" if const else " (non-constant coefficients)"),
                                  tuple(checks), False)
    if isinstance(sol, list):
        sol = sol[0]
    sol = sp.simplify(sol)
    # 1. substitution
    back = rec.rhs.replace(lambda e: isinstance(e, AppliedUndef) and e.func == _T,
                           lambda e: sol.subs(NI, e.args[0]))
    resid = sp.simplify(sp.expand(back - sol))
    if resid != 0:
        resid = sp.simplify(sp.expand_func(resid).rewrite(sp.exp))
    sub_ok = resid == 0 or resid.equals(0) is True
    checks.append("substitution into the recurrence: " + ("residual 0" if sub_ok else f"residual {resid} (NOT 0)"))
    # 2. initial conditions
    init_ok = True
    for k, v in rec.init.items():
        if not _exact_equal(sol.subs(NI, k), v):
            init_ok = False
            checks.append(f"T({k}) = {v} NOT satisfied (gives {sp.simplify(sol.subs(NI, k))})")
    if rec.init:
        checks.append(f"initial conditions {sorted(rec.init)}: " + ("satisfied" if init_ok else "NOT satisfied"))
    free = sorted(s.name for s in sol.free_symbols if s != NI)
    # 3. the recurrence's own values
    values_ok, upto = True, None
    if len(rec.init) >= order and not free:
        start = min(rec.init)
        vals = dict(rec.init)
        top = start + order + 30
        for m in range(start, top + 1):
            if m in vals:
                continue
            prev = [m - k for _, k in rec.shifts]
            if any(p not in vals for p in prev):
                values_ok = False
                break
            vals[m] = sp.expand(rec.rhs.replace(lambda e: isinstance(e, AppliedUndef) and e.func == _T,
                                                lambda e: vals[int(e.args[0].subs(NI, m))]).subs(NI, m))
        for m, v in vals.items():
            if not _exact_equal(sol.subs(NI, m), v):
                values_ok = False
                checks.append(f"value at n = {m}: recurrence gives {v}, the formula {sp.simplify(sol.subs(NI, m))}")
                break
        upto = top
        checks.append(f"values n = {start}..{top} computed from the recurrence: "
                      + ("all equal the formula (exact)" if values_ok else "MISMATCH"))
    theta, how = _theta_exppoly(sol)
    via = ", exponential-polynomial dominance" if theta is not None else ""
    if theta is None and not free:
        th, L, why = theta_of(sol, NI)
        if th is not None:
            theta, how, via = th, f"lim T(n)/({th}) = {L}", ", limit"
    if theta is not None:
        checks.append(f"Θ({theta}): {how}")
    ok = sub_ok and init_ok and values_ok
    first_n = min(rec.init) if rec.init else None
    valid = f"n >= {first_n}" if first_n is not None else "True"
    reason = "closed form proved by substitution" + (", initial conditions and the first values" if rec.init else "")
    if free:
        reason += f"; general solution with free constants {', '.join(free)}"
    if not ok:
        reason = "the closed form from rsolve did NOT pass every check — see checks"
    return RecurrenceSolution(rec.text, "linear", sol if ok else None, valid if ok else "False", None, theta,
                              "rsolve" + via,
                              VerdictStatus.VERIFIED if ok else VerdictStatus.UNDECIDED, reason, tuple(checks), ok)


def _exact_equal(a: sp.Expr, b: sp.Expr) -> bool:
    d = sp.expand(sp.sympify(a) - sp.sympify(b))
    if d == 0:
        return True
    d = sp.simplify(sp.radsimp(d))
    if d == 0:
        return True
    return bool(d.equals(0))


def _theta_exppoly(sol: sp.Expr) -> "tuple[sp.Expr | None, str]":
    """Θ of ``sum c_j n^k_j r_j^n`` by its strictly dominant term (``|r|``, then ``k``)."""
    grouped: dict[sp.Expr, sp.Expr] = {}
    for t in sp.Add.make_args(sp.expand(sol, power_base=False, mul=True, multinomial=True)):
        coeff, shape = t.as_independent(NI, as_Add=False)
        shape = sp.powsimp(shape)
        grouped[shape] = grouped.get(shape, sp.Integer(0)) + coeff
    terms = [sp.simplify(c) * sh for sh, c in grouped.items() if sp.simplify(c) != 0]
    keyed: list[tuple[sp.Expr, int, sp.Expr]] = []
    for t in terms:
        if not t.has(NI):
            keyed.append((sp.Integer(1), 0, t))
            continue
        base, deg = sp.Integer(1), 0
        for f in sp.Mul.make_args(t):
            if not f.has(NI):
                continue
            if f == NI:
                deg += 1
            elif f.is_Pow and f.base == NI and f.exp.is_Integer and f.exp > 0:
                deg += int(f.exp)
            elif f.is_Pow and not f.base.has(NI):
                ex = sp.expand(f.exp)
                alpha = ex.coeff(NI)
                rest = sp.simplify(ex - alpha * NI)
                if rest.has(NI) or not alpha.is_Rational:
                    return None, ""
                base *= sp.Abs(f.base) ** alpha
            elif isinstance(f, sp.exp):
                ex = sp.expand(f.args[0])
                alpha = ex.coeff(NI)
                if sp.simplify(ex - alpha * NI).has(NI):
                    return None, ""
                base *= sp.exp(sp.re(alpha))
            else:
                return None, ""
        keyed.append((sp.simplify(base), deg, t))
    if not keyed:
        return None, ""
    best = keyed[0]
    for item in keyed[1:]:
        c = _cmp(item[0], best[0])
        if c is None:
            return None, ""
        if c > 0 or (c == 0 and item[1] > best[1]):
            best = item
    ties = [k for k in keyed if _cmp(k[0], best[0]) == 0 and k[1] == best[1]]
    if len(ties) != 1:
        return None, ""
    g = (best[0] ** NI if best[0] != 1 else sp.Integer(1)) * NI ** best[1]
    g = sp.powsimp(g)
    return g, f"dominant term {sp.simplify(best[2])} of the exponential polynomial"


def _solve_dc(rec: _Rec, budget: float) -> RecurrenceSolution:
    checks: list[str] = []
    a_total = sp.Add(*[c for c, _, _ in rec.scales])
    bs = {b for _, b, _ in rec.scales}
    hs = [h for _, _, h in rec.scales]
    theta = None
    method = ""
    reasons: list[str] = []
    if len(bs) == 1 and all(h == 0 for h in hs):
        b = next(iter(bs))
        try:
            m = master_theorem(a_total, 1 / b, rec.g)
            theta, method = m.theta, m.method
            checks.extend(m.checks)
        except MasterTheoremInapplicable as exc:
            reasons.append(str(exc))
        except PycodemathError as exc:
            reasons.append(str(exc))
    elif len(bs) == 1:
        reasons.append("master theorem not used: T(n/b + h) with h != 0")
    else:
        reasons.append("master theorem not used: sub-problems of different sizes")
    if theta is None:
        try:
            ab = akra_bazzi([(c, b) for c, b, _ in rec.scales], rec.g)
            theta, method = ab.theta, "Akra–Bazzi"
            checks.extend(ab.checks)
            if any(h != 0 for h in hs):
                checks.append(f"perturbations {hs} are O(n/log² n): Akra–Bazzi allows them")
        except (AkraBazziInapplicable, PycodemathError) as exc:
            reasons.append(str(exc))
    exact_powers = None
    if len(rec.scales) == 1 and all(h == 0 for h in hs):
        exact_powers = _exact_on_powers(rec, checks)
    if theta is None:
        return RecurrenceSolution(rec.text, "divide-and-conquer", None, "False", exact_powers, None, "none",
                                  VerdictStatus.UNDECIDED, "; ".join(reasons), tuple(checks), False)
    return RecurrenceSolution(rec.text, "divide-and-conquer", exact_powers, _powers_text(rec), exact_powers, theta,
                              method, VerdictStatus.VERIFIED,
                              f"Θ({theta}) by {method}" + (f" (master theorem refused: {reasons[0]})" if reasons else ""),
                              tuple(checks), exact_powers is not None)


def _powers_text(rec: _Rec) -> str:
    b = rec.scales[0][1]
    return f"n = {sp.Integer(1) / b}**k" if rec.init else "False"


def _exact_on_powers(rec: _Rec, checks: list[str]) -> "sp.Expr | None":
    """Exact ``T(b^k)`` from ``S(k) = a S(k-1) + f(b^k)`` when ``T(b^j)`` is given."""
    a, bb, _ = rec.scales[0]
    B = sp.Integer(1) / bb
    if not B.is_Integer or not rec.init:
        return None
    base_k = None
    for k0, v in sorted(rec.init.items()):
        if k0 >= 1:
            j = sp.log(k0, B)
            j = sp.nsimplify(j)
            if j.is_Integer:
                base_k, base_v = int(j), v
                break
    if base_k is None:
        return None
    K = sp.Symbol("k", integer=True, nonnegative=True)
    S = sp.Function("S")
    gk = rec.g.subs(NI, B**K)
    try:
        sol = guard("rsolve", gk, lambda: sp.rsolve(S(K) - a * S(K - 1) - gk, S(K), {S(base_k): base_v}))
    except Exception:  # noqa: BLE001
        return None
    if sol is None:
        return None
    sol = sp.simplify(sol)
    # check by iteration in k, exactly
    val = base_v
    for k in range(base_k + 1, base_k + 25):
        val = sp.expand(a * val + rec.g.subs(NI, B**k))
        if not _exact_equal(sol.subs(K, k), val):
            checks.append(f"exact value at n = {B}**{k} did not match — not reported")
            return None
    n = NI
    out = sp.expand(sp.expand_log(sp.simplify(sol.subs(K, sp.log(n, B))), force=True))
    checks.append(f"exact for n = {B}**k (k >= {base_k}): checked against the recurrence for 24 values of k")
    return out


# --------------------------------------------------------------------------- #
# recurrence_from_code                                                        #
# --------------------------------------------------------------------------- #


def recurrence_from_code(
    func_or_source: "Callable[..., Any] | str",
    *,
    name: "str | None" = None,
    n_max: int = 30,
    budget: float = 30.0,
) -> CodeRecurrence:
    """The call-count recurrence of a recursive function, solved and certified.

    Reads the base case (``if n <= 1: return …`` before any self-call) and the
    self-calls (``f(n - 1)``, ``f(n // 2)``…, unconditional, outside loops) of a
    function of one integer parameter. ``C(n) = 1`` in the base region and
    ``C(n) = 1 + sum C(arg_i)`` elsewhere. The READING is certified by running the
    instrumented function and counting its real calls for n = 0..``n_max``.

        >>> src = '''
        ... def fib(n: int) -> int:
        ...     if n < 2:
        ...         return n
        ...     return fib(n - 1) + fib(n - 2)
        ... '''
        >>> r = recurrence_from_code(src)       # doctest: +SKIP
        >>> r.recurrence, r.theta                # doctest: +SKIP
        ('C(n) = C(n - 1) + C(n - 2) + 1 for n >= 2; C(n) = 1 for n <= 1', (1/2 + sqrt(5)/2)**n)
    """
    started = time.perf_counter()
    src = _load(func_or_source, name)
    fdef = src.func
    params = [a.arg for a in fdef.args.posonlyargs + fdef.args.args]
    if not params:
        raise LoopShapeError("recursive function without parameters", fdef.lineno)
    pname = params[0]
    body = list(fdef.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    base_conds: list[sp.Basic] = []
    rd = _ArgReader(pname, fdef.name)
    i = 0
    while i < len(body):
        head = body[i]
        if not (isinstance(head, ast.If) and not _self_calls(head, fdef.name, test_only=True)
                and head.body and isinstance(head.body[-1], (ast.Return, ast.Raise))
                and not _self_calls(ast.Module(body=head.body, type_ignores=[]), fdef.name) and not head.orelse):
            break
        base_conds.append(rd.cond(head.test))
        i += 1
    rest: list[ast.stmt] = body[i:]
    first = rest[0] if len(rest) == 1 else None
    if isinstance(first, ast.If) and first.orelse and not base_conds:
        # if COND: return base  else: recursive part   (or the other way round)
        st = first
        a_calls = _self_calls(ast.Module(body=st.body, type_ignores=[]), fdef.name)
        b_calls = _self_calls(ast.Module(body=st.orelse, type_ignores=[]), fdef.name)
        if not a_calls and b_calls:
            base_conds.append(rd.cond(st.test))
            rest = st.orelse
        elif a_calls and not b_calls:
            base_conds.append(sp.Not(rd.cond(st.test)))
            rest = st.body
    if not base_conds:
        raise LoopShapeError("recursion without a base case 'if …: return' before the self-calls", fdef.lineno)
    calls = []
    for stmt in rest:
        for node in ast.walk(stmt):
            if isinstance(node, (ast.For, ast.While, ast.If, ast.IfExp, ast.BoolOp, ast.Lambda,
                                 ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp, ast.Try)):
                if _self_calls(node, fdef.name):
                    raise LoopShapeError("self-call under a loop, a condition or short-circuit", node.lineno,
                                         "the number of calls would depend on more than n")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == fdef.name:
                if not node.args:
                    arg = next((k.value for k in node.keywords if k.arg == pname), None)
                    if arg is None:
                        raise LoopShapeError("self-call without the size argument", node.lineno)
                else:
                    arg = node.args[0]
                calls.append(rd.expr(arg))
    if not calls:
        raise LoopShapeError("no self-call after the base case", fdef.lineno)
    base = sp.Or(*base_conds) if len(base_conds) > 1 else base_conds[0]
    C = sp.Function("C")
    rhs = sp.Add(*[C(a) for a in calls]) + 1
    base_text = str(base).replace(pname, "n")
    n = NI
    rec_text = f"C(n) = {sp.sstr(rhs)} where not ({base_text}); C(n) = 1 where {base_text}"
    values = _evaluate(base, calls, n, -3, max(n_max, 40) + 30)
    shifts = [sp.simplify(n - a) for a in calls]
    sol: RecurrenceSolution
    if all(s.is_Integer and s > 0 for s in shifts):
        sol = _code_linear(rec_text, base, calls, values, n)
    else:
        rec_str = "T(n) = " + " + ".join(f"T({sp.sstr(a)})" for a in calls) + " + 1"
        try:
            sol = solve_recurrence(rec_str)
        except ParseError as exc:
            sol = RecurrenceSolution(rec_text, "unknown", None, "False", None, None, "none",
                                     VerdictStatus.UNDECIDED, str(exc), (), False)
        sol = RecurrenceSolution(rec_text, sol.form, None, "False", None, sol.theta, sol.method, sol.verdict,
                                 sol.reason + "; exact values: from the recurrence itself (certified below)",
                                 sol.checks, False)
    cert = _certify_calls(src, pname, values, n_max, budget)
    verdict = sol.verdict
    if cert is None or not cert.agree:
        verdict = VerdictStatus.UNDECIDED
    del started
    return CodeRecurrence(fdef.name, "calls", rec_text, base_text, sol, cert, verdict)


def _self_calls(node: ast.AST, fname: str, test_only: bool = False) -> bool:
    target = node.test if test_only and isinstance(node, ast.If) else node
    return any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == fname
               for x in ast.walk(target))


class _ArgReader:
    """Reads ``n - 1``, ``n // 2``, ``(n + 1) // 2``, ``n >> 1`` and conditions on ``n``."""

    def __init__(self, pname: str, fname: str) -> None:
        self.pname = pname
        self.fname = fname

    def expr(self, node: ast.expr) -> sp.Expr:
        if isinstance(node, ast.Name) and node.id == self.pname:
            return NI
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return sp.Integer(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.expr(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = self.expr(node.left), self.expr(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.FloorDiv) and b.is_Integer and b > 0:
                return sp.floor(a / b)
            if isinstance(node.op, ast.RShift) and b.is_Integer and b >= 0:
                return sp.floor(a / 2**b)
        raise LoopShapeError(f"recursive argument {ast.unparse(node)!r}", getattr(node, "lineno", None),
                             f"supported: expressions in {self.pname} with + - * // >>")

    def cond(self, node: ast.expr) -> sp.Basic:
        if isinstance(node, ast.BoolOp):
            parts = [self.cond(v) for v in node.values]
            return sp.And(*parts) if isinstance(node.op, ast.And) else sp.Or(*parts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return sp.Not(self.cond(node.operand))
        if isinstance(node, ast.Compare):
            terms = [node.left, *node.comparators]
            out = []
            for op, l, r in zip(node.ops, terms, terms[1:]):
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(r, (ast.Tuple, ast.List, ast.Set)):
                    a = self.expr(l)
                    c = sp.Or(*[sp.Eq(a, self.expr(e)) for e in r.elts])
                    out.append(c if isinstance(op, ast.In) else sp.Not(c))
                    continue
                rel = {ast.Lt: sp.Lt, ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge,
                       ast.Eq: sp.Eq, ast.NotEq: sp.Ne}.get(type(op))
                if rel is None:
                    raise LoopShapeError(f"base-case comparison {type(op).__name__}", node.lineno)
                out.append(rel(self.expr(l), self.expr(r)))
            return sp.And(*out)
        if isinstance(node, ast.Name) and node.id == self.pname:
            return sp.Ne(NI, 0)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) is False:
            pass
        raise LoopShapeError(f"base-case condition {ast.unparse(node)!r}", getattr(node, "lineno", None))


def _evaluate(base: sp.Basic, calls: Sequence[sp.Expr], n: sp.Symbol, lo: int, hi: int) -> dict[int, "int | None"]:
    """``C(m)`` from the recurrence itself for ``lo <= m <= hi`` (``None``: does not
    terminate). Bottom-up from ``lo - 64``: every argument must be SMALLER than
    ``m`` (else the recursion does not descend there) and already known."""
    start = lo - 64
    memo: dict[int, "int | None"] = {}
    for m in range(start, hi + 1):
        if bool(base.subs(n, m)):
            memo[m] = 1
            continue
        total: "int | None" = 1
        for a in calls:
            v = int(a.subs(n, m))
            if v >= m or v < start or memo.get(v) is None:
                total = None
                break
            total += memo[v]  # type: ignore[operator]
        memo[m] = total
    return {m: memo[m] for m in range(lo, hi + 1)}


def _code_linear(rec_text: str, base: sp.Basic, calls: Sequence[sp.Expr], values: Mapping[int, "int | None"],
                 n: sp.Symbol) -> RecurrenceSolution:
    order = max(int(n - a) for a in calls)
    rec_region = [m for m, v in values.items() if v is not None and not bool(base.subs(n, m))]
    if not rec_region:
        return RecurrenceSolution(rec_text, "linear", None, "False", None, None, "none", VerdictStatus.UNDECIDED,
                                  "no argument reaches the recursive case", (), False)
    # the recursive region must be a ray n >= n0 (beyond the tested values too)
    n0 = min(rec_region)
    top = max(values)
    if any(values.get(m) is None or bool(base.subs(n, m)) for m in range(n0, top + 1)):
        return RecurrenceSolution(rec_text, "linear", None, "False", None, None, "none", VerdictStatus.UNDECIDED,
                                  "the recursive case is not a ray n >= n0 (base case inside it)", (), False)
    init = {m: values[m] for m in range(n0 - order, n0)}
    if any(v is None for v in init.values()):
        return RecurrenceSolution(rec_text, "linear", None, "False", None, None, "none", VerdictStatus.UNDECIDED,
                                  "the initial values do not terminate", (), False)
    rec = "T(n) = " + " + ".join(f"T(n - {int(n - a)})" for a in calls) + " + 1, " + \
        ", ".join(f"T({m}) = {v}" for m, v in init.items())
    sol = solve_recurrence(rec)
    checks = list(sol.checks)
    main = sol.main
    if main is not None:
        bad = [m for m in range(n0 - order, top + 1) if values.get(m) is not None
               and not _exact_equal(main.subs(NI, m), values[m])]  # type: ignore[arg-type]
        checks.append(f"closed form vs the recurrence for n = {n0 - order}..{top}: "
                      + ("equal" if not bad else f"differs at {bad[:3]}"))
        if bad:
            main = None
    valid = f"n >= {n0 - order}"
    low = [m for m, v in values.items() if v is not None and m < n0 - order]
    if low and all(values[m] == 1 for m in low):
        valid += f" (C(n) = 1 below: base case)"
    return RecurrenceSolution(rec_text, "linear", main, valid if main is not None else "False", None, sol.theta,
                              sol.method, sol.verdict if main is not None else VerdictStatus.UNDECIDED,
                              sol.reason, tuple(checks), main is not None)


def _certify_calls(src: Any, pname: str, values: Mapping[int, "int | None"], n_max: int,
                   budget: float) -> "CountCertificate | None":
    from ..core.backstop import isolated
    from ..core.errors import IsolationError

    spec, blocked = _exec_spec(src)
    if blocked is not None:
        return None
    started = time.perf_counter()
    pts = [m for m in range(-3, max(n_max, 30) + 1) if values.get(m) is not None and values[m] <= 2_000_000]  # type: ignore[operator]
    skipped = [(f"n={m}", "the recurrence does not terminate there" if values.get(m) is None else "over 2e6 calls")
               for m in range(-3, max(n_max, 30) + 1) if m not in pts]
    per_call = min(3.0, max(0.25, budget / 6))
    try:
        res = isolated(_count_worker, spec, [{pname: m} for m in pts], per_call,
                       budget=max(2.0, min(budget, per_call * len(pts) + 2.0)))
    except (TimeBudgetError, IsolationError, EOFError, OSError) as exc:
        return CountCertificate(0, False, "==", "", None, (("all", type(exc).__name__),), None,
                                round(time.perf_counter() - started, 3))
    agree, witness = 0, None
    for m, (status, val) in zip(pts, res):
        if status != "ok":
            skipped.append((f"n={m}", f"{status}: {val if status == 'raise' else 'budget'}"))
            continue
        counted = val.get(0, 0)
        if counted == values[m]:
            agree += 1
        elif witness is None:
            witness = {"n": m, "counted": counted, "recurrence": values[m]}
    return CountCertificate(agree + (1 if witness else 0), witness is None and agree > 0, "==",
                            f"n in [{pts[0] if pts else 0}, {pts[-1] if pts else 0}]", witness,
                            tuple(skipped[:20]), None, round(time.perf_counter() - started, 3))

