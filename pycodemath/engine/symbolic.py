"""Symbolic engine (module 1) — algebra + analysis.

Operates on IR (``Expr``). Delegates to SymPy, but the API belongs to Pycodemath.
MVP: simplify, expand, derivative, integral, solve.
Module P2: limit, series, summation.
"""

from __future__ import annotations

import sympy as sp

from ..core.errors import PycodemathError
from ..core.ir import Expr
from ..frontend.parser import parse


def simplify(expr: Expr) -> Expr:
    return expr.simplify()


def expand(expr: Expr) -> Expr:
    return expr.expand()


def diff(expr: Expr, var: str) -> Expr:
    return expr.diff(var)


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
        raise PycodemathError(
            f"solve: cannot solve the equation for '{var}': {exc}"
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
            raise PycodemathError(
                f"solve: unexpected solution shape for '{var}' "
                f"(parametric/system result) — use solve_nd for systems"
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
    raise PycodemathError(f"invalid point/bound: {value!r}")


def limit(
    expr: Expr, var: str, to: "Expr | str | int | float", dir: str = "+"
) -> Expr:
    """Limit of ``expr`` as ``var → to`` (``to`` also accepts ``oo``/``-oo``).

    ``dir``: "+" (from the right, default) or "-" (from the left). A
    nonexistent limit (oscillation → AccumBounds) or an undefined one (nan/zoo,
    an unevaluated ``Limit``) is a clear refusal, not a raw SymPy object.
    """
    if dir not in ("+", "-"):
        raise PycodemathError(f"limit: dir must be '+' or '-', not {dir!r}")
    sym = sp.Symbol(var)
    point = _point(to)
    try:
        res = sp.limit(expr.sy, sym, point, dir)
    except (sp.PoleError, NotImplementedError, ValueError) as exc:
        raise PycodemathError(
            f"limit: cannot determine the limit at {point}: {exc}"
        ) from exc
    if isinstance(res, sp.AccumBounds):
        raise PycodemathError(
            f"limit: the limit at {point} does not exist (the expression oscillates)"
        )
    if res.has(sp.Limit) or res.has(sp.nan) or res.has(sp.zoo):
        raise PycodemathError(f"limit: the limit at {point} is undefined")
    return Expr(res)


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
        raise PycodemathError(
            f"series: n must be a positive integer, not {n!r}"
        )
    sym = sp.Symbol(var)
    point = _point(at)
    try:
        res = expr.sy.series(sym, point, n).removeO()
    except (sp.PoleError, NotImplementedError, ValueError) as exc:
        raise PycodemathError(
            f"series: cannot expand around {point}: {exc}"
        ) from exc
    return Expr(res)


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
        raise PycodemathError(f"summation: cannot compute the sum: {exc}") from exc
    # First the no-closed-form case: an unevaluated Sum with an oo bound
    # CARRIES oo inside, so the divergence check would catch it with a
    # misleading message (the sum may be convergent, just without a closed form).
    if res.has(sp.Sum):
        raise PycodemathError(
            "summation: no closed form — try finite bounds "
            "(partial sum) or compute numerically"
        )
    if isinstance(res, sp.AccumBounds) or res.has(
        sp.oo, -sp.oo, sp.zoo, sp.nan
    ):
        raise PycodemathError(
            "summation: divergent sum — narrow the bounds to finite ones"
        )
    return Expr(res)
