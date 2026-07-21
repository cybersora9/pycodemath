"""Numerical engine (module 4) — where symbolic methods fall short.

Operates on IR (``Expr``). When an expression has no closed-form root,
integral or minimum — or when we simply want a number — we drop down to
numerical methods. Derivatives are computed SYMBOLICALLY (``Expr.diff``), and
we iterate through COMPILED functions ``Expr.compiled`` (lambdify): compile
once before the loop, iterate without SymPy overhead. ``Expr.evalf`` remains the
contract for a single evaluation in IR — here, inside loops, it would be orders
of magnitude too slow.

1D methods (module 4):
  root_find     — root f(x)=0: Newton with a fallback to bisection.
  integrate_num — definite integral: composite Simpson quadrature on samples.
  minimize      — 1D minimum: gradient descent by default; method="newton"
                  or "bfgs" (module 15) gives 2nd-order methods with a
                  backtracking Armijo line search.

Multidimensional methods (module 5, 2nd-order methods — module 15):
  gradient      — vector of partial derivatives (symbolically, IR Matrix).
  jacobian      — matrix of derivatives of a system of functions (IR Matrix).
  hessian       — matrix of second derivatives (IR Matrix).
  root_find_nd  — system of nonlinear equations: multidimensional Newton (J·Δ = -F).
  minimize_nd   — minimum of a function of several variables: gradient descent
                  by default; method="newton" (Hessian symbolically,
                  H·d = -∇f) or "bfgs" (approximation of the inverse Hessian)
                  — both with a backtracking Armijo line search instead of a fixed lr.

Divergence / bad domain / singular Jacobian -> ``PycodemathError``
with a readable message (consistent with ``evalf`` and the parser).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from ..core.errors import PycodemathError
from ..core.ir import Expr, Matrix


# --- shared sampling layer ---------------------------------------------
#: Exceptions from a compiled function meaning "outside the domain" (math
#: raises ValueError; division by zero and overflow — their own types).
_DOMAIN_ERRORS = (ValueError, TypeError, ZeroDivisionError, OverflowError)


def _scalar(expr: Expr, var: str) -> Callable[[float], float]:
    """Return a COMPILED function ``x -> float`` (``Expr.compiled``).

    Validates that the expression is a function of the single variable ``var`` —
    otherwise the compiled form would leave unsubstituted symbols — here we give
    a readable message right away.
    """
    extra = [s for s in expr.symbol_names() if s != var]
    if extra:
        raise PycodemathError(
            f"a numerical method requires a function of the single variable {var!r}, "
            f"but the expression also depends on {', '.join(extra)}"
        )
    return expr.compiled([var])


def _safe(f: Callable[[float], float], x: float) -> "float | None":
    """Sample ``f(x)``; return ``None`` when the result is outside the domain.

    We catch the compiled form's domain exceptions (e.g. ``math.sqrt(-1)`` ->
    ValueError), complex results and infinite/NaN values — iterations treat
    ``None`` as a "not possible here" signal, instead of tearing down the whole
    method.
    """
    try:
        y = f(x)
    except _DOMAIN_ERRORS:
        return None
    if isinstance(y, complex):
        return None
    y = float(y)
    return y if math.isfinite(y) else None


def _sample(f: Callable[[float], float], x: float) -> float:
    """Like ``_safe``, but outside the domain it raises a readable ``PycodemathError``.

    Used where a missing value means an input error (e.g. quadrature over a
    point outside the domain), not an opportunity to change strategy.
    """
    y = _safe(f, x)
    if y is None:
        raise PycodemathError(
            f"cannot sample the function at x={x:g} "
            f"(outside the domain or complex result)"
        )
    return y


# --- shared DIVERGENCE threshold (R4–R7, unified in R8) -----------------
#: Scale of the divergence threshold. Every iterative solver (``root_find``,
#: ``root_find_nd``, ``_descent_min`` and ``minimize``/``minimize_nd`` in the
#: gradient-descent variant) treats an iterate as an escape to infinity when its
#: norm exceeds ``_DIVERGE_SCALE·max(1, ‖x0‖)``. Rounds R4–R7 introduced the same
#: number FIVE separate times — R8 lifts it into a single source, so the family
#: of detections cannot quietly drift apart under future tuning (one solver with
#: a different threshold = a regression trap).
_DIVERGE_SCALE: float = 1e12


def _diverge_limit(x0_norm: float) -> float:
    """Norm threshold above which we treat an iterate as DIVERGING.

    ``x0_norm`` is the norm of the starting point: ``abs(x0)`` in 1D, ``‖x0‖`` in
    many dimensions. Returns ``_DIVERGE_SCALE·max(1.0, x0_norm)`` — generous (a
    convergent run to a distant but finite solution has time to return a result
    before touching it). Bit-identical to the former ``1e12·max(1.0, x0_norm)``
    scattered across the solvers: R8 only centralizes the formula, it does not
    change a single number.
    """
    return _DIVERGE_SCALE * max(1.0, x0_norm)


def _diverge_limit_for(x0: "float | np.ndarray") -> float:
    """Divergence threshold matched to the DIMENSIONALITY of the starting point ``x0`` (R10).

    Closes the centralization from R8/R9. R8 lifted the threshold SCALE
    (``_diverge_limit``), R9 lifted the PREDICATE (``_has_diverged``) — but each of
    the five solvers still computed the start NORM itself, manually matched to the
    dimension: ``abs(float(x0))`` in 1D vs ``float(np.linalg.norm(x))`` in nd. This
    choice ("which norm formula fits this dimensionality") was open-coded five
    times — a risk that someone writes the vector norm for a scalar solver (or vice
    versa) under future tuning. R10 lifts it HERE: scalar (``ndim == 0``) →
    ``abs``; vector → ``‖x0‖``. Bit-identical to the former five calls — the same
    norm, the same ``_diverge_limit``, not a single new number."""
    arr = np.asarray(x0, dtype=float)
    if arr.ndim == 0:
        return _diverge_limit(abs(float(arr)))
    return _diverge_limit(float(np.linalg.norm(arr)))


def _has_diverged(x: "float | np.ndarray", diverge_limit: float) -> bool:
    """Is the iterate ``x`` DIVERGING — one predicate for ALL solvers (R9).

    ``x`` may be a scalar (1D: ``root_find``, ``_descent_min``) or a vector/ndarray
    (nd: ``root_find_nd``, ``minimize_nd``). Returns ``True`` when the iterate has
    escaped to infinity/NaN OR when its norm has exceeded ``diverge_limit`` (see
    ``_diverge_limit``).

    Closes the centralization from R8. R8 unified the THRESHOLD (a single source
    ``_diverge_limit``), but the divergence PREDICATE itself — ``not isfinite(...)
    or norm > limit`` — still lived open-coded in five solvers with minor
    differences (1D: ``abs`` + ``math.isfinite``; nd: ``np.all(np.isfinite)`` +
    ``np.linalg.norm``). R9 lifts it HERE, so the detection family cannot quietly
    drift apart in the details under future tuning. Bit-identical to each of the
    former versions: the same conditions, the same ``or``, the same norm."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        # scalar — former 1D: ``not math.isfinite(x) or abs(x) > limit``
        v = float(arr)
        return not math.isfinite(v) or abs(v) > diverge_limit
    # vector — former nd: ``not np.all(np.isfinite(x)) or norm(x) > limit``
    if not bool(np.all(np.isfinite(arr))):
        return True
    return float(np.linalg.norm(arr)) > diverge_limit


# --- shared STAGNATION threshold (R11) ---------------------------------
#: The opposite side of DIVERGENCE. Sometimes a solver neither converges nor
#: explodes (``_has_diverged`` stays silent) — it simply STALLS: successive
#: iterations move neither the iterate nor the function value, and the residual
#: stays above ``tol``. Formerly such a run ground through ``max_iter`` and ended
#: with the GENERIC "does not converge (unbounded below?)" — misleading, because
#: the function may well be bounded (e.g. ``x**4`` with too large an ``lr`` falls
#: into a period-2 cycle between ±x0, with ``f`` stuck at a constant). R11 gives
#: this a SEPARATE, apt STALL message.
#:
#: Design note: a "vanished step" alone (‖x_next−x‖ ≈ 0) is NOT a safe signal in
#: gradient descent — convergence to a DISTANT minimum (‖x*‖≫1) inherently takes a
#: small step relative to ‖x‖, yet still makes real progress. That is why the gd
#: solvers detect stagnation by the LACK OF RELATIVE PROGRESS in the function value
#: (see ``minimize``/``minimize_nd``), while the ``_has_stagnated`` below — the
#: "iterate stopped moving" predicate — is a shared primitive, ready for a solver
#: for which the vanishing step IS decisive (the twin of ``_has_diverged`` on the
#: other side of the scale).
_STAG_SCALE: float = 4.0
#: How many CONSECUTIVE iterations without progress (with residual > tol) we treat
#: as a stall — a single step without improvement (e.g. a momentary overshoot of a
#: convergent run) must not overturn the result, only a persistent streak.
_STAG_PATIENCE: int = 8
#: float64 machine epsilon — the scale of "rounding noise" below which a
#: difference carries no information about progress.
_EPS: float = float(np.finfo(np.float64).eps)


def _has_stagnated(
    x_next: "float | np.ndarray",
    x: "float | np.ndarray",
    scale: float = _STAG_SCALE,
) -> bool:
    """Did the step ``x → x_next`` VANISH at machine precision — the STAGNATION
    primitive, the twin of ``_has_diverged`` on the other side of the scale (R11).

    Returns ``True`` when ``‖x_next − x‖ ≤ scale · eps · max(1, ‖x‖)``: the step
    length has dropped to the level of rounding noise, so the iterate practically
    stands still. Scalar (1D) and vector (nd) — like ``_has_diverged`` (the norm
    via ``np.linalg.norm`` works for both). The ``max(1, ‖x‖)`` term gives a safe
    RELATIVE threshold (without it, it would collapse to zero near zero),
    consistent with the divergence-threshold convention.

    This predicate alone does NOT determine an error and — deliberately — is NOT
    wired into the gd solvers: convergence to a distant minimum can be
    "step-stagnant" while still making progress (see the note at ``_STAG_SCALE``).
    It is a primitive available to a solver for which the vanishing step is
    decisive."""
    a = np.asarray(x_next, dtype=float)
    b = np.asarray(x, dtype=float)
    step = float(np.linalg.norm(a - b))
    base = float(np.linalg.norm(b))
    return step <= scale * _EPS * max(1.0, base)


def _value_progressed(fx: float, f_next: float, scale: float = _STAG_SCALE) -> bool:
    """Did the gd step reduce the function VALUE significantly: ``fx − f_next >
    scale·eps·|fx|`` (R12: predicate lifted from ``minimize``/``minimize_nd``, the
    mirror of the ``_has_diverged``/``_diverge_limit_for`` centralization from R8–R10).

    This — and NOT the step-wise ``_has_stagnated`` — is the predicate the gd
    solvers actually use to detect a stall: convergence to a DISTANT minimum takes
    a small step relative to ‖x‖ (``_has_stagnated`` would raise a false alarm), but
    still cuts f by >> eps of relative improvement, so here the streak resets and
    convergent runs stay bit-identical. The negation over ``_STAG_PATIENCE``
    consecutive steps with residual > tol = STAGNATION (cycle/oscillation with too
    large an lr). ``fx``/``f_next`` are scalar — gd computes a one-dimensional
    objective value in both solvers."""
    return fx - f_next > scale * _EPS * abs(fx)


# --- R13: FINDING — why the stagnation family STOPS at gd -------------
# R4–R10 closed DIVERGENCE across all five solvers, so the natural reflex is:
# "close STAGNATION the same way on ``root_find`` and ``_descent_min``". Probing
# (the R11 pattern) says: NO — and not out of laziness, but because in neither of
# them is there anything to fix. Wiring in detection would be dead code posing as
# safety.
#
# The bug R11 fixed has EXACTLY one cause: a FIXED ``lr``. Too large a step throws
# gd into a period-2 cycle (``x^4`` bounces ±x0, ``f`` stuck at a constant), the
# run grinds through ``max_iter`` and ends with the MISLEADING "unbounded below?" —
# even though the function has a minimum.
#
#   * ``_descent_min`` (newton/bfgs) — has NO ``lr`` to overshoot. The step is
#     chosen by the backtracking Armijo, which accepts it ONLY when ``f`` truly
#     decreases (``cand_f ≤ fx + 1e-4·α·gᵀd`` with ``gᵀd < 0``), so ``f`` is
#     strictly monotone — a "f stuck" cycle is STRUCTURALLY impossible. When no α
#     descends, the line search ends the run RIGHT AWAY (without grinding). Probe:
#     20 runs (newton/bfgs × ``x^4``, Rosenbrock, ill-conditioned, distant/flat
#     minimum, kink, unbounded functions) — the longest streak of no value progress
#     = 0 at ``_STAG_PATIENCE`` = 8, zero steps with rising ``f``. The only run
#     grinding through ``max_iter`` is ``-x-y``: constant gradient, the iterate
#     marches LINEARLY (norm ~1e2 against a threshold of 1e12, so rightly does not
#     "diverge"), and ``f`` drops by a constant each step — the function REALLY is
#     unbounded below, so the generic message is TRUE there. There is no misleading
#     case to fix.
#   * ``root_find`` (1D) — a Newton cycle exists (``x^3-2x+2`` from x0=0 bounces
#     0→1→0), but does NOT reach the message: bisection is the fallback and, given a
#     real sign change, rescues the run, returning the correct root (−1.769). And
#     when no real root exists (``x^2+1``), today's "no root nearby?" is simply
#     TRUE. Stagnation would not add a single more accurate diagnosis.
#
# That is why stagnation is wired ONLY into gd (``minimize``/``minimize_nd``) — the
# only solvers with a fixed step. This is the same decision as the deliberately
# unwired ``_has_stagnated`` in R11: the primitive exists, but forcing it where the
# solver's structure already rules out the problem adds the risk of a false alarm
# for zero diagnosis. The R13 tests guard this FINDING behaviorally (Newton cycle →
# root; Armijo → no stagnation), and the anti-drift counter from R12 (1 definition +
# 2 gd solvers) guards it mechanically: wiring the predicate into a third solver
# would break that test.


# --- roots -------------------------------------------------------------
def root_find(
    expr: Expr,
    var: str,
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Find a root ``f(x) = 0`` starting from ``x0``.

    First Newton (derivative computed SYMBOLICALLY via ``Expr.diff``, evaluated
    numerically). When Newton fails to deliver — the derivative vanishes, the
    step runs outside the domain, or there is no convergence within ``max_iter``
    — we drop down to bisection over an interval with a sign change found around
    ``x0``.

    No real root nearby -> ``PycodemathError``.
    """
    f = _scalar(expr, var)
    df = _scalar(expr.diff(var), var)

    # DIVERGENCE threshold — consistent with ``root_find_nd`` (module R4). Newton
    # with a bad start can blow up: the step ``fx/dfx`` grows exponentially, the
    # iterate escapes to infinity and would quietly fall back to bisection, which —
    # finding no sign-change interval in range — ended with the misleading "no real
    # root nearby?". This is the same trap R4 fixed for root_find_nd: divergence
    # confused with an actual absence of a root. We track ``diverged`` and, when the
    # iterate exceeds the threshold (scaled by |x0|, generous — a convergent run to
    # a large finite root has time to return a result), we distinguish DIVERGENCE
    # from a real absence of a root in the message. We keep bisection as a fallback
    # for a REAL sign change (Newton may diverge despite an existing root — then the
    # interval catches it and we return a result).
    diverge_limit = _diverge_limit_for(x0)
    diverged = False
    x = float(x0)
    for _ in range(max_iter):
        fx = _safe(f, x)
        if fx is None:
            break
        if abs(fx) < tol:
            return x
        dfx = _safe(df, x)
        if dfx is None or dfx == 0.0:
            break  # derivative vanishes/diverges — Newton cannot proceed
        x = x - fx / dfx
        if _has_diverged(x, diverge_limit):
            # iterate escaped to inf/NaN OR its norm exploded before reaching
            # overflow — one shared predicate (R9), bit-identical to the former
            # two branches (not isfinite / abs>limit, both: diverged+break)
            diverged = True
            break

    # Fallback: a real sign change in range rescues the run even after divergence.
    root = _bisect(f, float(x0), tol)
    if root is not None:
        return root
    if diverged:
        raise PycodemathError(
            f"root_find DIVERGES from x0={x0} for {expr}: the Newton step escaped "
            f"to |x|>{diverge_limit:.3g} and there is no sign change within reach of "
            f"bisection — this is divergence, not proof of no root. Provide a starting "
            f"point closer to the expected root."
        )
    raise PycodemathError(
        f"root_find does not converge for {expr} at x0={x0} "
        f"(no real root nearby?)"
    )


def _find_bracket(
    f: Callable[[float], float],
    x0: float,
    step: float = 0.5,
    max_steps: int = 2000,
) -> "tuple[float, float] | None":
    """Find an interval ``[lo, hi]`` with a sign change, marching from ``x0``.

    We go separately right and left, checking successive samples — this way we
    catch a root even for an even function (a symmetric window would have the
    same sign at both ends). ``None`` when there is no sign change in range.
    """
    f0 = _safe(f, x0)
    for sign in (1.0, -1.0):
        prev_x, prev_f = x0, f0
        for k in range(1, max_steps + 1):
            xk = x0 + sign * k * step
            fk = _safe(f, xk)
            if prev_f is not None and fk is not None and prev_f * fk <= 0.0:
                return (prev_x, xk) if prev_x <= xk else (xk, prev_x)
            prev_x, prev_f = xk, fk
    return None


def _bisect(
    f: Callable[[float], float],
    x0: float,
    tol: float,
    max_iter: int = 200,
) -> "float | None":
    """Bisection over an interval found around ``x0`` (fallback for Newton).

    ``None`` when there is no sign-change interval — the signal of no real root
    in range.
    """
    bracket = _find_bracket(f, x0)
    if bracket is None:
        return None
    lo, hi = bracket
    flo = _safe(f, lo)
    if flo is None:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = _safe(f, mid)
        if fm is None:
            return None
        if abs(fm) < tol or 0.5 * (hi - lo) < tol:
            return mid
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return mid


# --- integration -------------------------------------------------------
def integrate_num(
    expr: Expr,
    var: str,
    a: float,
    b: float,
    n: int = 100,
) -> float:
    """Definite integral ``∫_a^b f(var) d(var)`` by composite Simpson quadrature.

    A numerical alternative to ``symbolic.integrate`` when the integral has no
    closed form. Simpson is exact for polynomials of degree ≤ 3 regardless of
    ``n``; for the rest ``n`` controls accuracy (even — we round up). A point
    outside the domain -> ``PycodemathError``.
    """
    if n <= 0:
        raise PycodemathError("the number of subintervals n must be positive")
    if n % 2:
        n += 1  # Simpson requires an even number of subintervals

    a, b = float(a), float(b)
    if a == b:
        return 0.0

    f = _scalar(expr, var)
    h = (b - a) / n
    total = _sample(f, a) + _sample(f, b)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * _sample(f, a + i * h)
    return h / 3.0 * total


# --- minimization ------------------------------------------------------
def minimize(
    expr: Expr,
    var: str,
    x0: float,
    lr: float = 0.1,
    tol: float = 1e-9,
    max_iter: int = 10000,
    method: str = "gd",
) -> float:
    """Find the argument of the 1D minimum.

    By default (``method="gd"``) gradient descent: the gradient is the SYMBOLIC
    derivative (``Expr.diff``), the step ``lr`` is halved on overshoot (historical
    behavior, results bit-identical to versions before module 15).
    ``method="newton"`` / ``"bfgs"`` are 2nd-order methods with an Armijo line
    search (the shared ``_descent_min`` core with ``minimize_nd``) — Newton hits
    the minimum of a quadratic in one step; ``lr`` does not apply to them. Returns
    the ``x`` minimizing ``expr``.

    Function unbounded below / no convergence / unknown method ->
    ``PycodemathError``.
    """
    _check_method(method, "minimize")
    if method != "gd":
        e = Expr(expr)
        _check_vars([e], [var], "minimize")
        return _descent_min(e, [var], [x0], tol, int(max_iter), method, "minimize")[0]
    f = _scalar(expr, var)
    grad = _scalar(expr.diff(var), var)

    # DIVERGENCE threshold — closes the detection family (R4 root_find_nd, R5
    # 2nd-order minimize, R6 1D root_find) for the last missing solver: historical
    # gradient descent. A function unbounded below has no minimum — gd finds a
    # descent at every step, so the iterate marches to infinity and f goes to -inf.
    # Without this the loop ground through max_iter steps and ended with the generic
    # "does not converge", confusing DIVERGENCE with slow oscillation. We detect the
    # norm explosion early and give an unambiguous error. The threshold is generous
    # (scaled by |x0|), and the check fires ONLY on explosion: a convergent run
    # keeps the iterate near the minimum and never touches it -> results
    # bit-identical to the behavior before R7.
    diverge_limit = _diverge_limit_for(x0)
    x = float(x0)
    fx = _sample(f, x)
    stag = 0  # consecutive accepted steps without a significant (relative) drop in f (R11)
    for _ in range(max_iter):
        g = _safe(grad, x)
        if g is None:
            break
        if abs(g) < tol:
            return x
        x_next = x - lr * g
        # Iterate explosion = function unbounded below. We catch it HERE, before
        # _safe samples f at a gigantic point (overflow -> None -> misleading "bad
        # step"). A convergent run never reaches here (|x_next| stays small).
        if _has_diverged(x_next, diverge_limit):
            raise PycodemathError(
                f"minimize DIVERGES from x0={x0} for {expr}: the gradient-descent "
                f"iterate escaped to |x|>{diverge_limit:.3g}, and the function value "
                f"goes to -inf — the function is most likely unbounded below "
                f"(no minimum). Check the function or provide a starting point near "
                f"the expected minimum."
            )
        f_next = _safe(f, x_next)
        if f_next is None:
            break
        if f_next > fx:
            lr *= 0.5  # overshot — shorten the step (transient, we do not count
            if lr < 1e-15:  # it toward stagnation — this is an lr correction, not a stall)
                break
            continue
        # STAGNATION (R11): we accepted the step, but the function value did not
        # drop SIGNIFICANTLY (improvement below rounding noise, relative to |f|).
        # Since the residual is still > tol (otherwise we would have returned above
        # on abs(g) < tol) and f stands still, the run makes no progress. K such
        # steps in a row is a STALL — SEPARATE from divergence (the norm does not
        # explode) and from unboundedness: the classic is x^4 with too large an lr,
        # falling into a period-2 cycle (±x0, f at a constant). The RELATIVE-progress
        # test is crucial for safety: convergence to a DISTANT minimum makes >> eps
        # of relative improvement at every step (despite a small step relative to
        # ‖x‖), so the streak resets and the results of convergent runs stay
        # bit-identical.
        if _value_progressed(fx, f_next):
            stag = 0
        else:
            stag += 1
            if stag >= _STAG_PATIENCE:
                raise PycodemathError(
                    f"minimize STALLED for {expr} at x0={x0}: over {stag} "
                    f"consecutive steps the function value did not drop significantly "
                    f"(f≈{f_next:.6g}), and |g|={abs(g):.3g} ≥ tol — this is STAGNATION, "
                    f"not divergence and not unboundedness: the run stalled (cycle / "
                    f"oscillation with too large a step). Reduce lr, reduce tol "
                    f"or provide a better starting point."
                )
        if abs(x_next - x) < tol:
            return x_next
        x, fx = x_next, f_next

    raise PycodemathError(
        f"minimize does not converge for {expr} at x0={x0} "
        f"(function unbounded below or bad step?)"
    )


# --- multidimensional numerics (module 5) -------------------------------
def _check_vars(exprs: "Sequence[Expr]", vars: "Sequence[str]", what: str) -> None:
    """Validate that the expressions depend solely on the given variables.

    The same contract as ``_scalar`` in 1D: an unsubstituted symbol would break
    ``evalf`` only in the middle of an iteration — here we give a readable error
    right away.
    """
    if not vars:
        raise PycodemathError(f"{what} requires at least one variable")
    if len(set(vars)) != len(vars):
        raise PycodemathError(f"{what}: the variable list contains duplicates")
    needed: set[str] = set()
    for e in exprs:
        needed |= set(e.symbol_names())
    extra = sorted(needed - set(vars))
    if extra:
        raise PycodemathError(
            f"{what}: the expression also depends on {', '.join(extra)} — "
            f"provide all variables"
        )


def _vector_fn(
    entries: "Sequence[Expr]", vars: "Sequence[str]"
) -> "Callable[[np.ndarray], np.ndarray | None]":
    """Compile a list of expressions into a function ``x -> vector`` (once, before the loop).

    The multidimensional counterpart of ``_scalar`` + ``_safe`` in one: the
    returned function samples all entries at the point ``x`` and gives ``None``
    when any is outside the domain (exception, complex result, NaN/inf) —
    iterations treat ``None`` as a "not possible here" signal.
    """
    fns = [e.compiled(vars) for e in entries]

    def sample(x: np.ndarray) -> "np.ndarray | None":
        out = np.empty(len(fns), dtype=float)
        for i, fn in enumerate(fns):
            try:
                y = fn(*x)
            except _DOMAIN_ERRORS:
                return None
            if isinstance(y, complex) or not math.isfinite(y):
                return None
            out[i] = y
        return out

    return sample


def _point(x0: "Sequence[float]", n: int, what: str) -> np.ndarray:
    """Validate and cast the starting point to a ``float`` vector of length ``n``."""
    x = np.asarray([float(v) for v in x0], dtype=float)
    if x.shape != (n,):
        raise PycodemathError(
            f"{what}: the starting point has {x.size} coordinates, "
            f"but there are {n} variables"
        )
    return x


def gradient(expr: Expr, vars: "Sequence[str]") -> Matrix:
    """Gradient ``∇f`` — a column of partial derivatives (symbolically).

    The derivatives are computed by the existing engine (``Expr.diff``); the
    result is an IR ``Matrix`` (a column vector, consistent with module 3), ready
    for numerical evaluation.
    """
    e = Expr(expr)
    vars = list(vars)
    if not vars:
        raise PycodemathError("gradient requires at least one variable")
    return Matrix([[e.diff(v).sy] for v in vars])


def jacobian(exprs: "Sequence[Expr]", vars: "Sequence[str]") -> Matrix:
    """Jacobian of the system — a row per function, a column per variable (symbolically)."""
    fs = [Expr(e) for e in exprs]
    vars = list(vars)
    if not fs or not vars:
        raise PycodemathError("jacobian requires at least one function and variable")
    return Matrix([[f.diff(v).sy for v in vars] for f in fs])


def hessian(expr: Expr, vars: "Sequence[str]") -> Matrix:
    """Hessian ``∇²f`` — the matrix of second partial derivatives (symbolically)."""
    e = Expr(expr)
    vars = list(vars)
    if not vars:
        raise PycodemathError("hessian requires at least one variable")
    return Matrix([[e.diff(vi).diff(vj).sy for vj in vars] for vi in vars])


def root_find_nd(
    exprs: "Sequence[Expr]",
    vars: "Sequence[str]",
    x0: "Sequence[float]",
    tol: float = 1e-10,
    max_iter: int = 100,
) -> list[float]:
    """Solve the nonlinear system ``F(x) = 0`` with multidimensional Newton.

    The Jacobian is computed SYMBOLICALLY (``jacobian``); at each iteration we
    solve the linear system ``J·Δ = -F`` via ``np.linalg.solve`` (the same
    mechanics as the numerical path ``linalg.solve_system``). A singular Jacobian
    or no convergence -> ``PycodemathError``.
    """
    fs = [Expr(e) for e in exprs]
    vars = list(vars)
    _check_vars(fs, vars, "root_find_nd")
    if len(fs) != len(vars):
        raise PycodemathError(
            f"root_find_nd requires a square system: "
            f"{len(fs)} equations, {len(vars)} variables"
        )
    J = [f.diff(v) for f in fs for v in vars]  # flattened, row by row
    n = len(vars)
    F_fn = _vector_fn(fs, vars)  # compile once — iterate without SymPy overhead
    J_fn = _vector_fn(J, vars)

    x = _point(x0, n, "root_find_nd")
    # DIVERGENCE threshold. Newton with a bad start can escape to infinity: the
    # iterate grows exponentially, spews RuntimeWarning "overflow", eventually
    # becomes inf/NaN — and the method ground through max_iter steps in vain and
    # ended with the generic "no solution nearby", confusing divergence with an
    # actual absence of a root. We detect the norm explosion early (before it
    # reaches overflow) and give an unambiguous error with a hint about the
    # starting point. The threshold is generous (scaled by ‖x0‖), so a convergent
    # run to a large but finite root has time to return a result (the residual
    # drops to zero) before the norm exceeds it.
    diverge_limit = _diverge_limit_for(x)
    for _ in range(max_iter):
        # Divergence takes priority over any sampling: a non-finite iterate is
        # hard proof that Newton escaped — do not sample at NaN/inf, just report
        # divergence right away (not the misleading "outside the domain").
        if not np.all(np.isfinite(x)):
            raise PycodemathError(
                f"root_find_nd DIVERGES from x0={list(x0)}: the Newton iterate "
                f"escaped to infinity (inf/NaN). Provide a starting point "
                f"closer to the expected root."
            )
        F = F_fn(x)
        if F is None:
            raise PycodemathError(
                f"cannot sample the system at x={x.tolist()} "
                f"(outside the domain or complex result)"
            )
        if float(np.max(np.abs(F))) < tol:
            return [float(v) for v in x]
        Jx = J_fn(x)
        if Jx is None:
            raise PycodemathError(
                f"cannot sample the Jacobian at x={x.tolist()} "
                f"(outside the domain or complex result)"
            )
        try:
            delta = np.linalg.solve(Jx.reshape(n, n), -F)
        except np.linalg.LinAlgError as exc:
            raise PycodemathError(
                f"singular Jacobian at x={x.tolist()} — Newton cannot proceed "
                f"(try a different starting point)"
            ) from exc
        x = x + delta
        # A norm explosion (and, via the shared R9 predicate, also an escape to
        # inf/NaN) = divergence. We catch it HERE, before the next turn samples the
        # system at a gigantic point and triggers overflow. The upper finiteness
        # guard (before sampling) STAYS — it catches a non-finite START with a
        # separate message; here delta is solve() of finite F/J, so it really
        # closes only the norm explosion (behavior bit-identical).
        if _has_diverged(x, diverge_limit):
            raise PycodemathError(
                f"root_find_nd DIVERGES from x0={list(x0)}: the iterate norm "
                f"exceeded {diverge_limit:.3g} (‖x‖={float(np.linalg.norm(x)):.3g}) "
                f"— Newton is escaping the solution. Provide a starting point closer "
                f"to the expected root."
            )

    raise PycodemathError(
        f"root_find_nd does not converge in {max_iter} iterations for the system "
        f"{[str(f) for f in fs]} at x0={list(x0)} — the iterate oscillates without "
        f"dropping below the tolerance. Try a starting point closer to the expected "
        f"root (or check whether a real solution exists at all)."
    )


# --- 2nd-order minimization (module 15): Newton / BFGS + Armijo ----------
#: available minimization methods; "gd" = historical gradient descent
_MIN_METHODS = ("gd", "newton", "bfgs")


def _check_method(method: str, what: str) -> None:
    """Validate the minimization method name — an unknown one is a readable error."""
    if method not in _MIN_METHODS:
        raise PycodemathError(
            f"{what}: unknown method {method!r} — available: "
            + ", ".join(_MIN_METHODS)
        )


def _descent_min(
    f: Expr,
    vars: "Sequence[str]",
    x0: "Sequence[float]",
    tol: float,
    max_iter: int,
    method: str,
    what: str,
) -> list[float]:
    """2nd-order directional minimization: Newton or BFGS + backtracking Armijo.

    Descent direction:
      * ``newton`` — solves ``H·d = -∇f`` with the Hessian computed
        SYMBOLICALLY (``Expr.diff``, compiled ONCE before the loop — the
        ``root_find_nd`` pattern); a singular or non-positive-definite Hessian
        (the direction does not descend: ``∇fᵀd ≥ 0``) -> fallback to ``-∇f``.
      * ``bfgs`` — maintains an approximation of the INVERSE Hessian (starting
        from the identity, a rank-2 update in inverse form — a double
        Sherman-Morrison formula) with the curvature condition
        ``sᵀy > 0``; a degenerate direction -> restart from the identity.

    The step length is chosen by a BACKTRACKING Armijo line search
    (``f(x+αd) ≤ f(x) + c·α·∇fᵀd``, c=1e-4, shrink 0.5) instead of a fixed
    ``lr`` — the full Newton/BFGS step passes as soon as it gives a descent.
    No convergence / unbounded function -> ``PycodemathError``.
    """
    n = len(vars)
    f_fn = _vector_fn([f], vars)  # compile once — iterate without SymPy overhead
    g_fn = _vector_fn([f.diff(v) for v in vars], vars)
    H_fn = None
    if method == "newton":
        H_fn = _vector_fn([f.diff(vi).diff(vj) for vi in vars for vj in vars], vars)

    x = _point(x0, n, what)
    f_vec = f_fn(x)
    if f_vec is None:
        raise PycodemathError(
            f"cannot sample the function at x={x.tolist()} "
            f"(outside the domain or complex result)"
        )
    fx = float(f_vec[0])
    g = g_fn(x)
    H_inv = np.eye(n)  # BFGS: current approximation of the inverse Hessian

    # DIVERGENCE threshold — consistent with ``root_find_nd``. A function unbounded
    # below has no minimum: the backtracking Armijo will ALWAYS find a descent, so
    # the iterate marches to infinity and the function value goes to -inf (along the
    # way RuntimeWarning "overflow", then inf/NaN). Instead of grinding through
    # max_iter steps and ending with the generic "does not converge", we detect the
    # iterate's norm explosion early and give an unambiguous error with a hint about
    # the start — exactly like Newton in ``root_find_nd``. The threshold is generous
    # (scaled by ‖x0‖), so a convergent run to a distant but finite minimum has time
    # to return a result (the gradient drops below the tolerance) before the norm
    # exceeds it.
    diverge_limit = _diverge_limit_for(x)

    for _ in range(max_iter):
        if g is None:
            break  # gradient outside the domain — cannot proceed
        if float(np.max(np.abs(g))) < tol:
            return [float(v) for v in x]

        d = -g  # shared fallback: the gradient direction
        if method == "newton":
            Hx = H_fn(x) if H_fn is not None else None
            if Hx is not None:
                try:
                    cand = np.linalg.solve(Hx.reshape(n, n), -g)
                    # ∇fᵀd < 0 guarantees a DESCENT direction — a non-positive-
                    # definite Hessian does not give one, we keep -∇f
                    if np.all(np.isfinite(cand)) and float(g @ cand) < 0.0:
                        d = cand
                except np.linalg.LinAlgError:
                    pass  # singular Hessian -> the gradient direction
        else:  # bfgs
            cand = -H_inv @ g
            if float(g @ cand) < 0.0:
                d = cand
            else:  # degenerate approximation — restart from the identity
                H_inv = np.eye(n)

        # backtracking Armijo: full step, then halves, until we descend
        gTd = float(g @ d)
        alpha = 1.0
        x_new = None
        f_new = fx
        while alpha >= 1e-15:
            cand_x = x + alpha * d
            cand_f = f_fn(cand_x)
            if cand_f is not None and float(cand_f[0]) <= fx + 1e-4 * alpha * gTd:
                x_new, f_new = cand_x, float(cand_f[0])
                break
            alpha *= 0.5
        if x_new is None:
            break  # the line search found no descent — we give up readably

        # An iterate norm explosion = function unbounded below (no minimum): the
        # line search still descends, so x escapes to infinity and f -> -inf. We
        # catch it HERE, before the next turn samples the gradient/Hessian at a
        # gigantic point and triggers overflow — consistent with ``root_find_nd``.
        xn_norm = float(np.linalg.norm(x_new))
        if _has_diverged(x_new, diverge_limit):
            raise PycodemathError(
                f"{what} DIVERGES from x0={list(x0)} with method {method!r}: the "
                f"iterate norm exceeded {diverge_limit:.3g} (‖x‖={xn_norm:.3g}), "
                f"and the function value goes to -inf — the function is most likely "
                f"unbounded below (no minimum). Check the function or provide a "
                f"starting point near the expected minimum."
            )

        g_new = g_fn(x_new)
        if method == "bfgs" and g_new is not None:
            s = x_new - x
            yv = g_new - g
            sy = float(yv @ s)
            # curvature condition sᵀy > 0 — without it the update would spoil the
            # positive-definiteness of the approximation (we skip the step, not break it)
            if sy > 1e-12 * float(np.linalg.norm(s) * np.linalg.norm(yv)):
                rho = 1.0 / sy
                V = np.eye(n) - rho * np.outer(s, yv)
                H_inv = V @ H_inv @ V.T + rho * np.outer(s, s)

        if float(np.max(np.abs(x_new - x))) < tol:
            return [float(v) for v in x_new]
        x, fx, g = x_new, f_new, g_new

    raise PycodemathError(
        f"{what} does not converge for {f} at x0={list(x0)} with method {method!r} "
        f"(function unbounded below or bad starting point?)"
    )


def minimize_nd(
    expr: Expr,
    vars: "Sequence[str]",
    x0: "Sequence[float]",
    lr: float = 0.1,
    tol: float = 1e-9,
    max_iter: int = 10000,
    method: str = "gd",
) -> list[float]:
    """Find the argument of the minimum of a function of several variables.

    By default (``method="gd"``) gradient descent — a SYMBOLIC gradient
    (``Expr.diff`` for each variable), numerical iteration, the step ``lr``
    halved on overshoot (historical behavior, results bit-identical to versions
    before module 15). ``method="newton"`` / ``"bfgs"`` are 2nd-order methods
    with an Armijo line search (``_descent_min``) — Newton with a symbolic
    Hessian converges on smooth problems in a few iterations, BFGS copes where
    gradient descent chokes (e.g. the Rosenbrock valley); the ``lr`` parameter
    does not apply to them. Returns the point ``x`` minimizing ``expr``.

    Function unbounded below / no convergence / unknown method ->
    ``PycodemathError``.
    """
    _check_method(method, "minimize_nd")
    f = Expr(expr)
    vars = list(vars)
    _check_vars([f], vars, "minimize_nd")
    if method != "gd":
        return _descent_min(f, vars, x0, tol, int(max_iter), method, "minimize_nd")
    f_fn = _vector_fn([f], vars)  # compile once — iterate without SymPy overhead
    g_fn = _vector_fn([f.diff(v) for v in vars], vars)

    x = _point(x0, len(vars), "minimize_nd")
    fx_vec = f_fn(x)
    if fx_vec is None:
        raise PycodemathError(
            f"cannot sample the function at x={x.tolist()} "
            f"(outside the domain or complex result)"
        )
    fx = float(fx_vec[0])
    # DIVERGENCE threshold — as in 1D minimize (R7) and _descent_min (R5): closing
    # the detection family for multidimensional gradient descent. A function
    # unbounded below has no minimum, gd descends at every step, the iterate escapes
    # to infinity. The check fires only on a norm explosion — a convergent run never
    # touches it, so results are bit-identical to the behavior before R7.
    diverge_limit = _diverge_limit_for(x)
    stag = 0  # consecutive accepted steps without a significant (relative) drop in f (R11)
    for _ in range(max_iter):
        g = g_fn(x)
        if g is None:
            break
        if float(np.max(np.abs(g))) < tol:
            return [float(v) for v in x]
        x_next = x - lr * g
        # A norm explosion = function unbounded below. We catch it HERE, before f_fn
        # samples at a gigantic point (overflow). A convergent run does not reach it.
        xn_norm = float(np.linalg.norm(x_next))
        if _has_diverged(x_next, diverge_limit):
            raise PycodemathError(
                f"minimize_nd DIVERGES from x0={list(x0)}: the gradient-descent "
                f"iterate norm exceeded {diverge_limit:.3g} (‖x‖={xn_norm:.3g}), "
                f"and the function value goes to -inf — the function is most likely "
                f"unbounded below (no minimum). Check the function or provide a "
                f"starting point near the expected minimum."
            )
        f_next_vec = f_fn(x_next)
        if f_next_vec is None:
            break
        f_next = float(f_next_vec[0])
        if f_next > fx:
            lr *= 0.5  # overshot — shorten the step (lr correction, not stagnation)
            if lr < 1e-15:
                break
            continue
        # STAGNATION (R11): the accepted step did not cut f significantly
        # (improvement below rounding noise relative to |f|), and the residual is
        # still > tol — the run makes no progress. K such steps is a STALL, separate
        # from divergence and unboundedness (e.g. x^4+y^4 with too large an lr falls
        # into a period-2 cycle). The RELATIVE-progress test protects convergence to
        # a distant minimum: it makes >> eps of relative improvement per step, so the
        # streak resets (bit-identical).
        if _value_progressed(fx, f_next):
            stag = 0
        else:
            stag += 1
            if stag >= _STAG_PATIENCE:
                raise PycodemathError(
                    f"minimize_nd STALLED for {f} at x0={list(x0)}: over {stag} "
                    f"consecutive steps the function value did not drop significantly "
                    f"(f≈{f_next:.6g}), and ‖g‖∞={float(np.max(np.abs(g))):.3g} ≥ tol "
                    f"— this is STAGNATION, not divergence and not unboundedness: the "
                    f"run stalled (cycle / oscillation with too large a step). Reduce "
                    f"lr, reduce tol or provide a better starting point."
                )
        if float(np.max(np.abs(x_next - x))) < tol:
            return [float(v) for v in x_next]
        x, fx = x_next, f_next

    raise PycodemathError(
        f"minimize_nd does not converge for {f} at x0={list(x0)} "
        f"(function unbounded below or bad step?)"
    )
