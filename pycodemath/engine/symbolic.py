"""Symbolic engine (module 1) — algebra + analysis.

Operates on IR (``Expr``). Delegates to SymPy, but the API belongs to Pycodemath.
MVP: simplify, expand, derivative, integral, solve.
Module P2: limit, series, summation.

--- EVERY ENTRY POINT HERE IS BOUNDED IN TIME (module 9) -----------------------

Each public function below runs under a wall-clock budget —
``core.budget.DEFAULT_TIME_BUDGET`` (120 s) unless the caller opened
``pycodemath.time_budget(...)`` around the call — and raises
``TimeBudgetError`` if SymPy is still working when it runs out.

This exists because the alternative was not a slow call but a call that never
returns. Measured: ``integrate 1/(x^5+x+1) dx`` did not finish in 500 s, and
neither did ``integrate log(x)/(x^5+x+1) dx`` or ``series exp(sin(exp(x))) n 14``.
Bare SymPy does not finish them either — the cost is inherited, not caused here —
but a library that hands an agent a tool call which hangs forever gives it nothing
to branch on: no result, no exception, no payload. Modules 2-8 typed every failure
that RETURNS; this is the one that did not.

The refusal is deliberately NOT the same outcome as "no closed form". That message
is a fact about the PROBLEM and is reached, correctly, in 14.5 s for
``integrate exp(-x^2)*log(x)/(x^2+1) dx``. ``TimeBudgetError`` is a fact about our
PATIENCE, and the distinction is the same discipline module 8 applied to
``converged``: a caller must be able to tell "no answer exists" from "I did not let
it look long enough".

See ``core.budget`` for the mechanism, the alternatives it rejected and the
measurement that rejected each, and for what an asynchronous interrupt cannot do.
"""

from __future__ import annotations

import sympy as sp

from ..core.budget import under_budget
from ..core.errors import (
    DivergenceError,
    DomainError,
    NoClosedFormError,
    NonConvergenceError,
    UnsupportedFormError,
)
from ..core.ir import Expr
from ..frontend.parser import parse


@under_budget("simplify")
def simplify(expr: Expr) -> Expr:
    return expr.simplify()


@under_budget("expand")
def expand(expr: Expr) -> Expr:
    return expr.expand()


@under_budget("diff")
def diff(expr: Expr, var: str) -> Expr:
    return expr.diff(var)


@under_budget("integrate")
def integrate(expr: Expr, var: str) -> Expr:
    return expr.integrate(var)


def _is_real_value(s: sp.Basic) -> bool:
    """Is the root real? Filters out junk complex roots that ``sympy.solve``
    returns for transcendental equations (e.g. ``cos(x) = 2``). Resolves
    three-valued: explicitly complex → reject, explicitly real → keep. When
    SymPy cannot decide (``is_real`` = None), we check the imaginary part; if
    that too is ambiguous, we KEEP the root — we do not pretend to know more
    than the engine, and the filter only cuts obvious junk."""
    if s.is_real is True:
        return True
    if s.is_real is False:
        return False
    im = sp.simplify(sp.im(s))
    if im.is_zero is True:
        return True
    if im.is_zero is False:
        return False
    return True  # ambiguous — we do not discard


@under_budget("solve")
def solve(expr: Expr, var: str, real: bool = False) -> list[Expr]:
    """Solve the equation ``expr = 0`` for ``var``. Returns a list of roots.

    ``sympy.solve`` can be inconsistent in the shape of its result: usually a
    list of roots, but sometimes a dict ``{var: value}`` or a list of dicts
    (parametric solutions). We normalize these shapes into a list of values
    for ``var``; a shape that cannot be interpreted as a root with respect to
    ``var`` (e.g. a tuple from a system solution) we do NOT let through as a
    raw ``SympifyError`` from the ``Expr`` constructor — we return a
    ``PycodemathError``.

    ``real=True`` filters out complex roots (``_is_real_value``). Defaults to
    ``False`` (the full API also returns complex ones — no regression); the
    REPL calls with ``real=True``, because transcendental equations returned
    junk complex roots there. We filter AFTER solving, on the same
    (non-real) symbol as the expression — otherwise ``sp.solve`` would look
    for a different symbol.
    """
    sym = sp.Symbol(var)
    try:
        raw = sp.solve(expr.sy, sym)
    except (
        NotImplementedError,
        sp.PolynomialError,
        sp.SympifyError,
        ValueError,
        AttributeError,
    ) as exc:
        # SymPy REFUSED before searching — no algorithm for this shape (module 10).
        # Its own words for that ("multiple generators [x, exp(x)]") used to be
        # pasted into the message; they stay reachable through ``__cause__`` and are
        # out of the text, because an agent must not have to read them to learn that
        # a numerical root finder is the way forward. That is what ``route`` says.
        raise UnsupportedFormError(
            f"solve: no symbolic method for this equation in '{var}' — "
            f"find a numerical root instead (root <expr> for {var} at <x0>)",
            route="root",
        ) from exc
    if isinstance(raw, dict):  # single solution as a map {sym: value}
        raw = [raw]

    roots: list[Expr] = []
    for r in raw:
        if isinstance(r, dict):  # list of maps (parametric solutions)
            if sym in r:
                roots.append(Expr(r[sym]))
            # a map without our variable carries no root for var — we skip it
        elif isinstance(r, (tuple, list)):  # system tuple — not for a single variable
            # A DomainError since module 10, not a bare PycodemathError: nothing
            # went wrong mathematically — a single-variable solver was handed a
            # question whose answer is a system solution, which is the same kind of
            # refusal as a non-square matrix. Defensive, and knowingly so: nine
            # probes over ``sp.solve(expr, one_symbol)`` never produced a tuple, so
            # ``tests/test_symbolic.py`` reaches it by substituting the SymPy call.
            raise DomainError(
                f"solve: unexpected solution shape for '{var}' "
                f"(parametric/system result) — use solve_nd for systems",
                route="solve_nd",
            )
        else:
            roots.append(Expr(r))
    if real:
        roots = [rt for rt in roots if _is_real_value(rt.sy)]
    return roots


# --- module P2: limit / series / summation ---------------------------------
def _point(value: "Expr | str | int | float") -> sp.Basic:
    """Point/interval bound: Expr, text (via the parser — the same whitelist
    as the whole input) or a number. We let nothing else through, so as not to
    open a side door into sympify with the full namespace."""
    if isinstance(value, Expr):
        return value.sy
    if isinstance(value, str):
        return parse(value).sy
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return sp.sympify(value)
    raise DomainError(f"invalid point/bound: {value!r}")


@under_budget("limit")
def limit(
    expr: Expr, var: str, to: "Expr | str | int | float", dir: str = "+"
) -> Expr:
    """Limit of ``expr`` as ``var → to`` (``to`` also accepts ``oo``/``-oo``).

    ``dir``: "+" (from the right, default) or "-" (from the left). A
    nonexistent limit (oscillation → AccumBounds) or an undefined one (nan/zoo,
    an unevaluated ``Limit``) is a clear refusal, not a raw SymPy object.
    """
    if dir not in ("+", "-"):
        raise DomainError(f"limit: dir must be '+' or '-', not {dir!r}")
    sym = sp.Symbol(var)
    point = _point(to)
    try:
        res = sp.limit(expr.sy, sym, point, dir)
    except (sp.PoleError, NotImplementedError, ValueError) as exc:
        # SymPy raised: no METHOD for this shape at this point (module 10). Its own
        # wording is kept on ``__cause__`` rather than in the message — it varies
        # by SymPy version and carries no instruction a caller can act on.
        raise UnsupportedFormError(
            f"limit: no method for the limit at {point} — "
            f"expand the expression or approach a different point"
        ) from exc
    if isinstance(res, sp.AccumBounds):
        raise NonConvergenceError(
            f"limit: the limit at {point} does not exist (the expression oscillates)"
        )
    if res.has(sp.Limit):
        # SymPy RAN and handed back the question unanswered — the same outcome an
        # unevaluated ``Integral`` or ``Sum`` is, and split off from the nan/zoo
        # case below for that reason (module 10). No route: the grammar has no
        # numerical limit.
        raise NoClosedFormError(
            f"limit: no closed form for the limit at {point} — "
            f"the engine could not evaluate it"
        )
    if res.has(sp.nan) or res.has(sp.zoo):
        # ``zoo`` is complex infinity and ``nan`` is undefined: no limit exists.
        # ``summation`` below tests exactly this predicate and calls it a
        # DivergenceError; module 10 made ``limit`` agree with its own neighbour
        # instead of filing the same fact under a class that says nothing.
        raise DivergenceError(f"limit: the limit at {point} is undefined")
    return Expr(res)


@under_budget("series")
def series(
    expr: Expr, var: str, at: "Expr | str | int | float" = 0, n: int = 6
) -> Expr:
    """Series expansion (Taylor/Laurent) around ``at`` up to order ``n``.

    Returns a polynomial WITHOUT the ``O(...)`` term (``removeO``) — a
    deliberate decision: the result is an ordinary ``Expr`` that round-trips
    through the parser (``O`` is not in the whitelist and makes no sense
    outside the series context). The truncation order is documented by the
    parameter ``n``, not the result.
    """
    if not isinstance(n, int) or n < 1:
        raise DomainError(
            f"series: n must be a positive integer, not {n!r}"
        )
    sym = sp.Symbol(var)
    point = _point(at)
    try:
        res = expr.sy.series(sym, point, n).removeO()
    except (sp.PoleError, NotImplementedError, ValueError) as exc:
        # As in ``limit``: SymPy raised, so it has no expansion METHOD here. Its
        # message ("Asymptotic expansion of besselj around [0, oo] is not
        # implemented", complete with a leading newline) stays on ``__cause__``.
        raise UnsupportedFormError(
            f"series: no expansion method around {point} — "
            f"expand around a different point"
        ) from exc
    return Expr(res)


@under_budget("summation")
def summation(
    expr: Expr,
    var: str,
    lo: "Expr | str | int | float",
    hi: "Expr | str | int | float",
) -> Expr:
    """Symbolic summation of ``expr`` over ``var`` from ``lo`` to ``hi``.

    The bounds may be numbers, symbols (``n``) or ``oo``. A divergent sum or
    one without a closed form is a ``PycodemathError`` with a hint, not
    ``oo``/an unevaluated ``Sum`` in the result (contract: zero junk).
    """
    sym = sp.Symbol(var)
    a, b = _point(lo), _point(hi)
    try:
        res = sp.summation(expr.sy, (sym, a, b))
    except (NotImplementedError, ValueError) as exc:
        raise UnsupportedFormError(
            "summation: no method for this sum — try finite bounds (partial sum)"
        ) from exc
    # First the no-closed-form case: an unevaluated Sum with an oo bound
    # CARRIES oo inside, so the divergence check would catch it with a
    # misleading message (the sum may be convergent, just without a closed form).
    # NO ROUTE, and that is the informative part: the command grammar has no
    # numerical infinite sum, so unlike an integral this one has nothing else to
    # try. The message keeps saying so; module 10 made the ABSENCE readable as
    # data (``route is None``) rather than only as English.
    if res.has(sp.Sum):
        raise NoClosedFormError(
            "summation: no closed form — try finite bounds "
            "(partial sum) or compute numerically"
        )
    if isinstance(res, sp.AccumBounds) or res.has(
        sp.oo, -sp.oo, sp.zoo, sp.nan
    ):
        raise DivergenceError(
            "summation: divergent sum — narrow the bounds to finite ones"
        )
    return Expr(res)
