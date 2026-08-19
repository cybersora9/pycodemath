"""Numerical engine — where symbolic methods fall short.

Operates on IR (``Expr``). When an expression has no closed-form root,
integral or minimum — or when we simply want a number — we drop down to
numerical methods. Derivatives are computed SYMBOLICALLY (``Expr.diff``), and
we iterate through COMPILED functions ``Expr.compiled`` (lambdify): compile
once before the loop, iterate without SymPy overhead. ``Expr.evalf`` remains the
contract for a single evaluation in IR — here, inside loops, it would be orders
of magnitude too slow.

1D methods:
  root_find     — root f(x)=0: Newton with a fallback to bisection.
  integrate_num — definite integral: composite Simpson quadrature on samples;
                  with ``tol`` a locally adaptive refinement of that same rule.
  minimize      — 1D minimum: gradient descent by default; method="newton"
                  or "bfgs" gives 2nd-order methods with a backtracking Armijo
                  line search.

Multidimensional methods:
  gradient      — vector of partial derivatives (symbolically, IR Matrix).
  jacobian      — matrix of derivatives of a system of functions (IR Matrix).
  hessian       — matrix of second derivatives (IR Matrix).
  root_find_nd  — system of nonlinear equations: multidimensional Newton (J·Δ = -F).
  minimize_nd   — minimum of a function of several variables: gradient descent
                  by default; method="newton" (Hessian symbolically,
                  H·d = -∇f) or "bfgs" (approximation of the inverse Hessian)
                  — both with a backtracking Armijo line search instead of a fixed lr.

Every failure raises a ``PycodemathError`` subclass with a readable message
(consistent with ``evalf`` and the parser): a bad domain or a singular Jacobian
gives ``DomainError``, an escaping iterate ``DivergenceError``, a stalled
gradient descent ``StagnationError``, an exhausted iteration count
``NonConvergenceError``.

Every SUCCESS may be asked for its evidence. The four iterative solvers
(``root_find``, ``root_find_nd``, ``minimize``, ``minimize_nd``) take a
keyword-only ``full_result=True`` and then return a ``SolveResult`` — the same
value plus the iteration count, the residual at that point and the status —
instead of a bare number. The default (``full_result=False``) return type, value
and behaviour are untouched, down to the last bit.

Every FAILURE may be asked the same (module 4). Under ``full_result=True`` a
diverging / stalled / non-convergent solver RETURNS a ``SolveResult`` with
``converged=False`` and the status matching the exception it would otherwise raise
(``"diverged"`` / ``"stagnated"`` / ``"not_converged"``), so an agent can inspect
how far the run got instead of parsing an English message. ``value`` is then the
LAST ITERATE — the point the solver gave up at, NOT a solution; ``converged`` is
the only field to branch on. The default call raises exactly as before, with the
same message. INPUT refusals (``DomainError``) are NOT an iteration outcome and
raise either way — see ``_failure_residual`` for the one rule covering a residual
that cannot be sampled at a failed run's last iterate.

The INTEGRAL gets both of those, in its own vocabulary (module 5). It was the last
numerical answer here that could not be asked how good it was: a bare ``float`` from
composite Simpson on a fixed ``n``, with no accuracy claim of any kind. Now
``integrate_num`` can report an ERROR ESTIMATE (``full_result=True`` →
``QuadratureResult``) and, given a ``tol``, REACH a requested accuracy by refining
locally where the integrand is difficult — or refuse in the Module-2 vocabulary when
its evaluation budget runs out (``BudgetExhaustedError``) or when the float grid
cannot resolve a panel any further (``StagnationError``). The default call — no
``tol``, no ``full_result`` — is the same rule on the same grid, down to the last
bit. See ``integrate_num`` for what the estimate assumes and WHEN IT LIES.

Module 8 made ``converged`` MEAN what all of the above says it means. Until then
the four solvers ended on a TOLERANCE test and returned whatever fired it, which
is a WEAKER claim than "this value answers the question you asked" — and the two
came apart in both directions, measured:

  * ``‖∇f‖ < tol`` holds at EVERY stationary point, so ``minimize(-x^2, x0=0)``
    reported a maximum as a converged minimum (a saddle in nd likewise). The
    gradient exit is now CORROBORATED: when it fires, the solver probes ``f``
    around the point and refuses only on evidence — a nearby point that is
    strictly lower, which is the definition of "not a minimum" rather than a
    verdict read off a curvature sign. See ``_lower_value_nearby``. The outcome
    has its own status and its own exception (``not_a_minimum`` /
    ``NotAMinimumError``), because it is the one failure where the run had NO
    trouble: it arrived, at the wrong kind of point, and more iterations can never
    help.
  * ``|f(x)| < tol`` can come true because ``f`` DECAYED rather than because ``x``
    is a root, so ``root_find(exp(x))`` reported ``x=-24`` as a root of a function
    that has none. The residual exit is now corroborated by the Newton correction
    — the first-order distance still to travel. See ``_root_has_settled``.
  * and the reverse: a run that is converging, only slowly, was reported as a
    stall. The gd stagnation detector judged liveness by progress in ``f`` alone,
    and ``f`` reaches the rounding floor while the gradient is still an order of
    magnitude above ``tol``; ``minimize(sin(x), x0=0)`` came back
    ``converged=False`` at the minimum, right to 7 significant figures. The streak
    now also resets on progress in the RESIDUAL, which costs no samples and leaves
    the genuine cycle (``x^4`` at ``lr=1``, ``|g|`` pinned at 32) still stalling.

Both certificates run ONCE, on the exit path, on functions the solver already
compiled — never inside an iteration — and both may only REFUSE, on evidence they
actually computed. Where the evidence cannot be computed the old acceptance
stands: a verdict that cannot be computed honestly is not invented.

A note on the "(module N)" markers below: they refer to the **v0.3 series** — the
run of modules that built the exception hierarchy (2), the structured solver
result (3), its failure half (4), quadrature's own result (5), the MCP payload
(6), the installed caller's view (7) and the meaning of ``converged`` (8). This
file used to carry markers from the older, unrelated build order in the same
words; those are gone, so a single scheme is left.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Sequence, overload

import numpy as np

from ..core.errors import (
    BudgetExhaustedError,
    DivergenceError,
    DomainError,
    NonConvergenceError,
    NotAMinimumError,
    StagnationError,
)
from ..core.ir import Expr, Matrix
from ..core.result import QuadratureResult, SolveResult, SolveStatus


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
        raise DomainError(
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
        raise DomainError(
            f"cannot sample the function at x={x:g} "
            f"(outside the domain or complex result)"
        )
    return y


# --- shared RESIDUAL layer for the structured result (module 3) ---------
# What the ``SolveResult.residual`` means — ONE quantity per solver FAMILY, never
# a new definition per solver:
#   * root finding (``root_find``, ``root_find_nd``) — ``|f(x)|``, i.e. ``‖F(x)‖∞``
#     for a system: how far the returned point still is from zeroing the function.
#   * minimization (``minimize``, ``minimize_nd``, gd and 2nd-order alike) —
#     ``‖∇f(x)‖∞``: how far the returned point still is from a stationary point.
# In both cases it is measured AT THE RETURNED POINT — not the smallest value seen
# somewhere along the way — and read off the COMPILED function the solver already
# holds (``_scalar`` / ``_vector_fn``), never a fresh SymPy call. Where the exit
# test already computed the quantity we reuse that number; where it did not (the
# gd solvers can exit on a vanishing STEP, with the gradient last sampled one
# point back) we take exactly ONE extra sample, on the exit path, and only when
# the caller asked for ``full_result``. Nothing here runs inside an iteration.


def _inf_norm(v: np.ndarray) -> float:
    """``‖v‖∞`` — the residual norm of the nd solvers (system value or gradient)."""
    return float(np.max(np.abs(v)))


def _need_residual(residual: "float | None", what: str, point: object) -> float:
    """Refuse readably when a CONVERGED run cannot report its residual.

    ``None`` means the function/gradient could not be sampled at the returned point
    (outside the domain or a complex result) — rare, but possible when a derivative
    has its own singularity exactly there. We neither invent a NaN nor quietly
    report a value measured somewhere else: the house rule is a readable refusal,
    and the answer itself is still available from the default call.
    """
    if residual is None:
        raise DomainError(
            f"{what} converged at {point}, but the residual cannot be sampled "
            f"there (outside the domain or complex result) — call without "
            f"full_result=True to get the value alone"
        )
    return residual


def _failure_residual(residual: "float | None") -> float:
    """The residual of a FAILED run — with ONE rule for "not sampleable" (module 4).

    A failed run reports the same per-family quantity as a successful one, measured
    AT THE LAST ITERATE. But that point is often exactly where the function cannot
    be evaluated: a diverged iterate is inf/NaN or so large that ``f`` overflows,
    and a run may give up precisely because it stepped outside the domain. Three
    ways out were available; here is the one chosen, and why the other two lose:

      * REFUSING the whole call (the ``_need_residual`` treatment on the success
        path) would throw away the answer the caller came for — the iteration count
        and the last iterate — over the one number that could not be taken. On the
        success path a refusal is right (a "converged" result whose residual is
        unknown proves nothing); on the failure path the result is the diagnosis.
      * ``nan`` marks "no value" honestly, but every comparison against it is
        ``False``, so ``residual > tol`` would answer "the point is fine" for a
        point that is nowhere near solved — a silent trap in the one direction a
        caller is most likely to test.

    The rule: ``math.inf``. The field means "how far the returned point still is
    from solved", and a point where the function cannot even be evaluated is not
    near-solved to ANY degree, so every ``residual > tol`` test answers ``True`` —
    the safe direction. And it stays unambiguous: a MEASURED residual is always
    finite by construction, because the sampling layer (``_safe`` / ``_vector_fn``)
    rejects inf/NaN and returns ``None``. So ``math.isinf(r.residual)`` reads
    exactly as "not measurable at the point the solver gave up at", never as a
    measurement that happened to come out huge.
    """
    return math.inf if residual is None else residual


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


# --- module 8: an exit test is not a verdict -----------------------------
# Every solver here ends on a TOLERANCE test — ``|f(x)| < tol`` for a root,
# ``‖∇f(x)‖∞ < tol`` for a minimum — and until module 8 that test WAS the verdict:
# whatever fired it came back with ``converged=True``. Measured on the module-7
# HEAD, the test and the verdict came apart in two directions at once:
#
#   * ``root_find(exp(x), x0=0)`` → ``value=-24.0, residual=3.775e-11,
#     converged=True``. ``exp(x) = 0`` has NO root; the iterate walked left one unit
#     per Newton step until the function had DECAYED below ``tol``. A small residual
#     was being read as an existence proof.
#   * ``minimize(-x^2, x0=0)`` → ``value=0.0, converged=True``. ``∇f = 0`` holds at
#     EVERY stationary point — a maximum and a saddle satisfy the exit test exactly
#     as well as a minimum does. (``minimize_nd(x^2-y^2, [0,0])`` the same, and
#     ``x^3`` at 0, where the curvature is zero and the point is an inflection.)
#
# The two certificates below close those, and they share one discipline: the
# certificate may only REFUSE, on evidence it actually computed, and it never runs
# inside an iteration — one call, on the exit path, on the function the solver has
# already compiled. Where the evidence cannot be computed the old acceptance stands:
# a verdict that cannot be computed honestly must not be invented.

#: The ROOT certificate's threshold. ``sqrt(eps)`` ≈ 1.49e-8 — the half-precision
#: scale, and the floor of what any root finder can resolve in ``x``: at a distance
#: ``sqrt(eps)`` from a simple root, ``f`` is already down at the rounding noise of
#: its own evaluation, so no method can tell that point from the root itself.
_ROOT_SETTLE: float = math.sqrt(_EPS)


def _straddle(a: float, b: float) -> "bool | None":
    """Do ``a`` and ``b`` bracket a sign change? ``True`` strict, ``None`` touching zero.

    Compares SIGNS, never the product ``a·b``. The product is the obvious spelling
    and it is wrong in exactly the situation module 8 is about: two tiny values of
    the SAME sign multiply to zero (``1e-200 · 1e-201`` underflows to ``0.0``), and
    ``a·b <= 0`` then reports a sign change that is not there. That is how
    ``root_find(exp(x))`` kept producing a root after the Newton half of the fix —
    bisection was handed a bracket manufactured by underflow, out on the tail where
    every sample is denormal. (The mirror hazard, ``a·b`` overflowing to ``inf`` for
    two huge values, is closed by the same rewrite.)

    ``True`` — strictly opposite signs: the intermediate-value theorem applies and a
    root provably exists between them. ``None`` — one of them is exactly ``0.0``,
    which may be a root or may be underflow, and the caller decides with
    ``_isolated_zero``. ``False`` — same sign, nothing here.
    """
    if a == 0.0 or b == 0.0:
        return None
    return (a > 0.0) != (b > 0.0)


def _isolated_zero(f: Callable[[float], float], x: float) -> bool:
    """Is ``f(x) == 0.0`` a ROOT, or the float grid having underflowed to zero?

    An exact zero is normally the strongest evidence there is — nothing beats
    ``f(x) = 0`` — and the module-8 root certificate lets it through without asking
    the Newton model anything, which matters because ``f'`` vanishes there too at a
    multiple root (``x^2`` at 0) and the model would have nothing to say.

    But a DECAYING function manufactures exact zeros: ``exp(-746)`` is not small, it
    is ``0.0``, because that is where float64 runs out of exponent. Every point
    beyond is ``0.0`` as well. That is what tells the two apart, and it is the whole
    test: a genuine zero is ISOLATED — one step away, on the half-precision scale,
    ``f`` is measurably nonzero again — while an underflowed one sits on a plateau of
    zeros stretching to infinity. Two samples of the already-compiled ``f``, taken
    only when an exact zero actually shows up.

    (This closes the same defect from the OTHER side. With Newton no longer stopping
    at ``x=-24``, ``root_find(exp(x))`` marched on until ``exp`` underflowed, and
    ``_find_bracket`` read ``f(prev)·f(next) <= 0`` — true when one factor is exactly
    zero — as a sign change. It then "bisected" a root out of a stretch of the axis
    where the function is merely too small to represent.)
    """
    step = _ROOT_SETTLE * max(1.0, abs(x))
    return any(
        (y := _safe(f, x + s * step)) is not None and y != 0.0 for s in (1.0, -1.0)
    )


def _root_has_settled(x: float, correction: float) -> bool:
    """Has the root iteration SETTLED at ``x`` — is the Newton correction negligible?

    ``correction`` is ``f(x)/f'(x)``: the first-order estimate of the distance from
    ``x`` TO THE ROOT, in ``x``-units. It is the number that separates the two ways
    ``|f(x)| < tol`` can come true (measured at the exit point of every row of the
    module-8 baseline table):

      * ``x`` really is a root — ``x^2-2`` (correction 1.6e-12), the Kepler equation
        (7.1e-14), ``cos(x)-x`` (0.0), ``sin(x)`` at π (4.1e-10). The model says
        "the root is right here", and it is.
      * ``f`` merely DECAYED — ``exp(x)`` at -24 (correction 1.0), ``exp(-x^2)`` at
        19.75 (2.5e-2), ``x*exp(-x^2)`` at 5.05 (1.0e-1). The model says "the root
        is a whole unit further on", and every step confirms it: after moving one
        unit the root is STILL one unit away. That sequence does not converge, it
        MARCHES, and ``|f|`` falling below ``tol`` along the way proves nothing.

    Seven orders of magnitude separate the two groups (4.1e-10 against 1.0e-2) and
    the threshold sits inside the gap with ~36x of margin below it and ~10^6 above.
    Returns ``True`` when ``|correction| ≤ sqrt(eps)·max(1, |x|)`` — relative for a
    large iterate, absolute near zero, the same convention as ``_diverge_limit``
    and ``_has_stagnated``.

    NOT settled does not mean "no root": it means THIS point is not yet one. The
    iteration simply carries on, which is why the fix improves three answers instead
    of refusing them — a double root (``x^2`` from 1.0) used to stop at 7.6e-6 and
    now runs on to 1.5e-8, and ``exp(x)-1e-11`` used to stop 2.4 short of its real
    root and now reaches it. The run only fails if nothing settles AND bisection
    finds no sign change — and then there really was nothing there.
    """
    return abs(correction) <= _ROOT_SETTLE * max(1.0, abs(x))


#: The MINIMUM certificate's probe distances, relative to ``_probe_base``, tried
#: SMALLEST FIRST. No single distance works: a maximum with zero curvature (``-x^4``
#: at 0) is invisible at 1e-6 — the drop, 1e-24, is far under the rounding noise of
#: ``f`` — while a probe wide enough to see it would, on a rippled function whose
#: minima sit 6e-3 apart, walk into the NEXT minimum and refuse a correct answer.
#: The escalation rule below is what makes both work at once.
_PROBE_SCALES: tuple[float, ...] = (1e-6, 1e-4, 1e-2)


def _probe_base(x: np.ndarray) -> float:
    """Distance unit for the minimum certificate: ``max(1, ‖x‖·sqrt(eps))``.

    Absolute (the raw scale) for an ordinary point, so the probe stays LOCAL — a
    local minimum is a statement about a neighbourhood, and a probe scaled to ‖x‖
    would reach 1e6 away from a minimum at 1e8 and start reporting on a different
    part of the function entirely. Widened only when ‖x‖ is so large that the raw
    distance would be lost in the gaps between representable floats.
    """
    return max(1.0, float(np.linalg.norm(x)) * _ROOT_SETTLE)


def _lower_value_nearby(
    f_at: "Callable[[np.ndarray], float | None]",
    g_at: "Callable[[np.ndarray], np.ndarray | None] | None",
    x: np.ndarray,
    fx: float,
    residual: float,
) -> "tuple[np.ndarray, float] | None":
    """Is the converged point NOT a minimum? Return the evidence, or ``None``.

    THE VERDICT IS THE DEFINITION, NOT A CURVATURE SIGN. The certificate returns a
    concrete nearby point whose value is strictly lower — which is exactly what
    "not a local minimum" means — or nothing at all. Curvature never decides
    anything here; it only says WHERE TO LOOK (see the directions below). That is
    what makes the boundary cases honest rather than invented:

      * ZERO curvature is not a verdict, and both answers live there: ``x^4`` at 0
        (``f''=0``, a genuine minimum) and ``-x^4`` at 0 (``f''=0``, a maximum) are
        indistinguishable to any second-order test, and ``x^3`` at 0 (``f''=0``, an
        inflection) too. The probe separates all three, because it asks the question
        that actually distinguishes them.
      * A SINGULAR Hessian is not an INDEFINITE one, and neither gets a verdict of
        its own: a singular Hessian only means the curvature direction is
        uninformative, so the coordinate probes carry the case alone.
      * A stationary point is only stationary TO A TOLERANCE. A point with residual
        ``r`` has a downhill side by construction, and reporting THAT as "not a
        minimum" would refuse every gradient-descent answer ever returned. So a drop
        counts only when it is bigger than the residual itself explains over the
        distance walked (``residual·‖Δ‖₁``, the bound on the linear term) plus the
        rounding noise of ``f``.

    DIRECTIONS. The ``2n`` coordinate half-axes, plus — in more than one dimension —
    the eigenvector of the SMALLEST curvature, both ways. In 1D the half-axes are
    every direction there is, so nothing else is needed. In nd they are not: ``x*y``
    at the origin is a saddle whose descent runs diagonally, and its value along
    both axes is exactly 0.0, so axis probes see a plateau and report nothing. The
    curvature direction finds it, and finds the general case too (``(x+3y)^2 -
    1e-4·(3x-y)^2``, whose descent direction is neither an axis nor a diagonal).

    THE HESSIAN IS NUMERICAL, AND THAT IS THE DECISION THE ND CASE TURNS ON. It is
    a central difference of the gradient the solver ALREADY compiled — ``2n`` extra
    samples, no new symbolic work and no second ``lambdify``. The symbolic Hessian
    exists in this module (``hessian``) and was the alternative; it was rejected on
    cost, because it would put ``n²`` symbolic derivatives and a compile on the
    SUCCESS path of every ``minimize_nd`` call, and this module's whole point is
    that evidence must not cost an answer. Accuracy is not the trade it looks like:
    a finite-difference Hessian is good to ~1e-8 relative, and nothing here reads
    its VALUES — only the direction of its smallest eigenvector, and only as a hint
    about where to sample the true ``f``. If the gradient cannot be sampled around
    ``x``, or the eigendecomposition fails, that hint is simply dropped.

    ESCALATION. For each direction, the distances are tried smallest first and the
    walk stops at the first CONCLUSIVE one — a distance at which ``f`` moved further
    than the noise and the residual can account for. So the probe stays as local as
    the arithmetic permits: near a well-curved minimum, 1e-6 already answers (and a
    rippled function's neighbouring minimum, 6e-3 away, is never reached), while a
    flat maximum keeps widening because every narrow probe is genuinely
    inconclusive, until 1e-2 shows the drop.

    WHAT IT DOES NOT PROMISE. Finding no lower value is not a PROOF of minimality —
    a probe is finite and a function can hide a descent between two samples (the
    known gap: a direction that is flat to second order AND off-axis, e.g. a saddle
    of the form ``x^3·y^3``). It is the same epistemic status the gradient test
    always had, and one strictly stronger: everything the gradient test accepted
    before, this accepts too, minus what it can positively refute.
    """
    n = int(x.size)
    # noise floor of f itself: below this a difference carries no information (the
    # _STAG_SCALE·eps convention of _value_progressed, applied to |f| here)
    floor = _STAG_SCALE * _EPS * max(abs(fx), 1.0)
    base = _probe_base(x)

    directions = [np.eye(n)[i] for i in range(n)]
    if n > 1 and g_at is not None:
        curvature_dir = _least_curvature_direction(g_at, x, base)
        if curvature_dir is not None:
            directions.append(curvature_dir)

    for d in directions:
        for sign in (1.0, -1.0):
            for scale in _PROBE_SCALES:
                step = (sign * scale * base) * d
                probe = x + step
                f_probe = f_at(probe)
                if f_probe is None:
                    continue  # outside the domain — this probe says nothing
                allowance = residual * float(np.sum(np.abs(step))) + floor
                if f_probe < fx - allowance:
                    return probe, f_probe
                if f_probe > fx + allowance:
                    break  # conclusive ASCENT here — no reason to widen further
    return None


def _least_curvature_direction(
    g_at: "Callable[[np.ndarray], np.ndarray | None]",
    x: np.ndarray,
    base: float,
) -> "np.ndarray | None":
    """Eigenvector of the smallest curvature at ``x``, or ``None`` when unavailable.

    The Hessian is a central difference of the COMPILED gradient (``2n`` samples,
    step ``sqrt(eps)·base``), symmetrized — a finite-difference Hessian is not
    exactly symmetric, and ``eigh`` requires it to be. Returns ``None`` (never an
    invented direction) when the gradient cannot be sampled on both sides or the
    eigendecomposition fails; the coordinate probes then carry the case alone.
    """
    n = int(x.size)
    h = _ROOT_SETTLE * base
    columns = []
    for i in range(n):
        step = np.eye(n)[i] * h
        g_plus, g_minus = g_at(x + step), g_at(x - step)
        if g_plus is None or g_minus is None:
            return None
        columns.append((g_plus - g_minus) / (2.0 * h))
    H = np.array(columns).T
    if not bool(np.all(np.isfinite(H))):
        return None
    try:
        _, vectors = np.linalg.eigh(0.5 * (H + H.T))
    except np.linalg.LinAlgError:  # pragma: no cover - eigh on a 2x2 practically cannot fail
        return None
    return np.asarray(vectors[:, 0], dtype=float)  # eigh sorts ascending


def _not_a_minimum_error(
    what: str,
    f: object,
    x0: object,
    point: object,
    fx: float,
    probe: np.ndarray,
    f_probe: float,
) -> NotAMinimumError:
    """The message for a refused stationary point — evidence first, then the remedy."""
    where = probe.tolist() if probe.size > 1 else float(probe[0])
    fx = fx or 0.0  # a stationary point of -x^2 has fx == -0.0, and "f=-0" reads
    f_probe = f_probe or 0.0  # like a typo rather than like a value
    return NotAMinimumError(
        f"{what} reached a STATIONARY point at {point} for {f} (from x0={x0}), but "
        f"it is NOT a minimum: f={fx:.6g} there, while f={f_probe:.6g} at {where} "
        f"right next to it — a maximum, a saddle or an inflection. A vanishing "
        f"gradient holds at ALL of those, so it cannot tell them apart; this run "
        f"checked. More iterations cannot help (the step at a stationary point is "
        f"zero) — start BESIDE this point instead, or minimize -({f}) if what you "
        f"wanted was the maximum."
    )


# --- roots -------------------------------------------------------------
@overload
def root_find(
    expr: Expr,
    var: str,
    x0: float,
    tol: float = ...,
    max_iter: int = ...,
    *,
    full_result: Literal[False] = ...,
) -> float: ...


@overload
def root_find(
    expr: Expr,
    var: str,
    x0: float,
    tol: float = ...,
    max_iter: int = ...,
    *,
    full_result: Literal[True],
) -> SolveResult[float]: ...


def root_find(
    expr: Expr,
    var: str,
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100,
    *,
    full_result: bool = False,
) -> "float | SolveResult[float]":
    """Find a root ``f(x) = 0`` starting from ``x0``.

    First Newton (derivative computed SYMBOLICALLY via ``Expr.diff``, evaluated
    numerically). When Newton fails to deliver — the derivative vanishes, the
    step runs outside the domain, or there is no convergence within ``max_iter``
    — we drop down to bisection over an interval with a sign change found around
    ``x0``.

    ``full_result=True`` (keyword-only) returns a ``SolveResult[float]`` instead of
    the bare root: the same ``float`` in ``.value``, plus ``.iterations`` (Newton
    steps that ran, PLUS the bisection steps when the fallback delivered the root —
    the bracket search is not an iteration of the method and is not counted) and
    ``.residual`` = ``|f(root)|``. The default call is unchanged.

    No real root nearby -> ``PycodemathError``. Under ``full_result=True`` that
    outcome is RETURNED instead (module 4): ``converged=False`` with
    ``status="diverged"`` when Newton escaped to infinity and no sign change was in
    reach of bisection, otherwise ``"not_converged"``. ``.value`` is then the LAST
    ITERATE — the point the method gave up at, NOT a root, and ``.converged`` is the
    only field to branch on; ``.iterations`` counts the Newton steps that ran and
    ``.residual`` is ``|f(x)|`` there (``math.inf`` when that point cannot be
    sampled — see ``_failure_residual``). An input refusal (``DomainError`` — an
    expression depending on more than ``var``) raises either way.
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
    #: first point where ``|f| < tol`` fired but the iterate had NOT settled — kept
    #: so a refusal can name the decay instead of the misleading "no root nearby?"
    decayed: "tuple[float, float, float] | None" = None
    x = float(x0)
    it = 0  # Newton iterations that actually ran (module 3: reported as .iterations)
    for it in range(1, max_iter + 1):
        fx = _safe(f, x)
        if fx is None:
            break
        dfx = _safe(df, x)
        # Module 8: ``|f(x)| < tol`` alone is not a root. The Newton correction —
        # the first-order distance from here TO the root — has to be negligible too,
        # or the residual is small merely because ``f`` DECAYED on the way (see
        # ``_root_has_settled``). The sample of ``df`` is the one Newton needs
        # anyway, taken one line earlier than before: no extra work per iteration.
        if abs(fx) < tol:
            if fx == 0.0 and _isolated_zero(f, x):
                # an EXACT, isolated zero is a root by inspection; no model is
                # needed, and asking for one would refuse it whenever f' vanishes
                # there too (a multiple root). A zero on a plateau of zeros is
                # underflow, not a root — see ``_isolated_zero``.
                return SolveResult.success(x, it, 0.0) if full_result else x
            if dfx is None or dfx == 0.0:
                # the correction cannot be computed, so nothing can be refuted —
                # today's acceptance stands rather than an invented verdict
                return SolveResult.success(x, it, abs(fx)) if full_result else x
            if _root_has_settled(x, fx / dfx):
                # the exit test already measured the residual — reuse that number
                return SolveResult.success(x, it, abs(fx)) if full_result else x
            if decayed is None:
                # remembered only to make the refusal say WHY (nothing reads it
                # unless the whole run, bisection included, comes up empty)
                decayed = (x, fx, fx / dfx)
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
    found = _bisect(f, float(x0), tol)
    if found is not None:
        root, bisect_it = found
        if not full_result:
            return root
        # The iteration count spans BOTH phases: an agent asking "how much work was
        # this root" must not be told only about the half that worked.
        fr = _safe(f, root)
        return SolveResult.success(
            root,
            it + bisect_it,
            _need_residual(None if fr is None else abs(fr), "root_find", root),
        )
    if full_result:
        # FAILURE form (module 4). Both outcomes of this tail share one exit: the
        # last iterate is ``x`` (where Newton stopped — an escaped point when
        # ``diverged``, otherwise wherever it gave up), the count is the Newton
        # steps that ran (the fallback delivered no root, so it has no work to
        # attribute to an answer), and the residual is ONE sample of the
        # already-compiled ``f`` there — inf when that point is not sampleable.
        fx_end = _safe(f, x)
        return SolveResult.failure(
            x,
            it,
            _failure_residual(None if fx_end is None else abs(fx_end)),
            "diverged" if diverged else "not_converged",
        )
    if diverged:
        raise DivergenceError(
            f"root_find DIVERGES from x0={x0} for {expr}: the Newton step escaped "
            f"to |x|>{diverge_limit:.3g} and there is no sign change within reach of "
            f"bisection — this is divergence, not proof of no root. Provide a starting "
            f"point closer to the expected root."
        )
    if decayed is not None:
        # Module 8: the run DID drive |f| under tol — and that is precisely why the
        # old generic message would have been a lie in the reader's favour. Name the
        # decay, so nobody reads a tiny ``residual`` on this failure as near-success.
        xd, fd, correction = decayed
        raise NonConvergenceError(
            f"root_find finds no root of {expr} near x0={x0}: |f| fell below tol at "
            f"x={xd:g} (f={fd:.3g}), but the Newton correction there was still "
            f"{abs(correction):.3g} — the function is DECAYING "
            f"towards zero, not crossing it, and no sign change was found within "
            f"reach of bisection. A small |f| is not a root. Provide a starting point "
            f"nearer an actual sign change, or check whether a real root exists."
        )
    raise NonConvergenceError(
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

    A STRICT sign change (``prev_f·fk < 0``) is the intermediate-value theorem doing
    its work and needs no further evidence. The boundary case — one of the two
    samples exactly ``0.0`` — is where module 8 tightened this: that is a bracket
    only when the zero is ISOLATED (``_isolated_zero``). A decaying function such as
    ``exp`` eventually underflows to a whole half-line of exact zeros, and reading
    ``prev_f·fk <= 0`` there as a sign change let bisection "find" a root of a
    function that has none.
    """
    f0 = _safe(f, x0)
    for sign in (1.0, -1.0):
        prev_x, prev_f = x0, f0
        for k in range(1, max_steps + 1):
            xk = x0 + sign * k * step
            fk = _safe(f, xk)
            if prev_f is not None and fk is not None:
                crossed = _straddle(prev_f, fk)
                if crossed or (
                    crossed is None
                    and _isolated_zero(f, xk if fk == 0.0 else prev_x)
                ):
                    return (prev_x, xk) if prev_x <= xk else (xk, prev_x)
            prev_x, prev_f = xk, fk
    return None


def _bisect(
    f: Callable[[float], float],
    x0: float,
    tol: float,
    max_iter: int = 200,
) -> "tuple[float, int] | None":
    """Bisection over an interval found around ``x0`` (fallback for Newton).

    Returns ``(root, iterations)`` — the halvings that actually ran, so
    ``root_find`` can report the work of BOTH phases in its ``SolveResult``
    (module 3; the returned root is bit-identical to the former plain-``float``
    version). ``None`` when there is no sign-change interval — the signal of no
    real root in range.
    """
    bracket = _find_bracket(f, x0)
    if bracket is None:
        return None
    lo, hi = bracket
    flo = _safe(f, lo)
    if flo is None:
        return None

    it = 0
    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        fm = _safe(f, mid)
        if fm is None:
            return None
        if abs(fm) < tol or 0.5 * (hi - lo) < tol:
            return mid, it
        if _straddle(flo, fm) is not False:
            # sign comparison, not the product ``flo·fm`` — see ``_straddle`` for the
            # underflow that made the product lie. ``None`` (fm exactly zero) keeps
            # the former ``<= 0`` behaviour: the root is at or left of ``mid``.
            hi = mid
        else:
            lo, flo = mid, fm
    return mid, it


# --- integration -------------------------------------------------------
# --- the ANALYSIS module 5 rests on: how wrong is a FIXED n, really? ----
# Measured before a line of the mechanism was written (the R11/R13 pattern), against
# integrands whose exact integral is known in closed form. ``n=100`` was a number out
# of thin air, and this is what it buys — TRUE relative error of today's answer:
#
#   integrand                        [a, b]     true rel err   n needed for 1e-10 rel
#   2x                               [0, 1]        2.2e-16     100 (Simpson is exact)
#   sin x                            [0, π]        5.4e-09     400
#   e^x                              [0, 1]        5.6e-11     100
#   sin 50x  (fast oscillation)      [0, 1]        3.6e-04     6 400
#   1/(1+10⁴x²)  (sharp peak)        [-1, 1]       5.5e-02     1 600
#   e^(-10⁴x²)   (narrow gaussian)   [-1, 1]       1.9e-01     800
#   √x  (endpoint-singular deriv.)   [0, 1]        1.2e-04     1 638 400
#
# So: for a peak narrower than the grid the answer is wrong by PERCENT (19% for the
# gaussian) and nothing said so; and for √x the fourth derivative is unbounded at 0,
# Simpson's order collapses from 4 to ~1.5, and a GLOBAL grid needs over a million
# subintervals for an accuracy an agent may reasonably ask for. A fixed grid therefore
# cannot promise a requested accuracy at any sane cost — point 4 of the module applies,
# and the adaptive variant below exists. Which strategy: measured too, at tol=1e-10
# absolute, counting real evaluations of the compiled function —
#
#   integrand              global halving (double n)    local subdivision (this module)
#   √x on [0, 1]                    262 145 evals                     1 005 evals
#   sharp peak on [-1, 1]             4 097 evals                     1 865 evals
#   sin 50x on [0, 1]                 1 025 evals                     5 757 evals
#
# LOCAL subdivision wins where it decides usability (260× on √x: the difficulty is in
# one corner of the interval, and only local refinement can put the samples there) and
# loses ~5× on a uniform oscillation, where the difficulty is everywhere and a global
# grid is already optimal. A bounded loss on one side against a 260× win on the other
# — and the loss is in the SAFE direction (more samples, not a wrong answer).
#: Default evaluation budget of the adaptive path. Generous: the measurements above fit
#: in a few thousand samples, and the worst case tested (sin 50x on [0, π] to 1e-12)
#: took 78 801. A caller who needs more raises it explicitly — running out is a
#: reportable outcome, never a silently truncated answer.
_QUAD_MAX_EVALS: int = 100_000

#: A panel cannot be subdivided below this many float steps (``math.ulp``) of its own
#: endpoints: a Simpson panel needs FIVE distinct nodes, so under ~4 steps the quarter
#: points collapse onto the endpoints, ``S_coarse`` and ``S_fine`` become the same
#: number and the error estimate would read 0.0 for a panel nothing had resolved. 8
#: keeps a margin. Measured consequence: integrating over ``[1e15, 1e15+1]``, where one
#: float step is 0.125, the whole grid is below the limit — with this guard the run
#: reports it in 201 evaluations instead of burning 100 000 on panels that hold no
#: distinct floats.
_QUAD_MIN_ULPS: float = 8.0

#: Richardson factor for Simpson: halving ``h`` divides the error by ``2⁴ = 16``, so
#: ``I_2n - I_n`` covers ``15/16`` of the error of ``I_n`` and ``1/16`` of the error of
#: ``I_2n``. Hence ``err(I_n) ≈ 16/15·|I_2n - I_n|`` (the estimate of the value we
#: RETURN on the fixed grid) and ``err(I_2n) ≈ |I_2n - I_n|/15`` (the estimate of the
#: refined value the adaptive path returns). One classical result, two scopes.
_QUAD_RICHARDSON: float = 15.0


def _simpson_grid(
    f: Callable[[float], float], a: float, b: float, n: int
) -> "tuple[float, list[float], float]":
    """Composite Simpson on ``n`` subintervals: ``(value, samples, h)``.

    The value is accumulated in EXACTLY the order ``integrate_num`` used before module
    5 — the endpoints first, then the interior weighted 4/2 upward — so lifting the
    loop into a helper cannot move a bit of the default answer. The samples come back
    because everything else in this section (the error estimate, the adaptive seed)
    is built on them instead of sampling the same points again.
    """
    h = (b - a) / n
    ys = [0.0] * (n + 1)
    ys[0] = _sample(f, a)
    ys[n] = _sample(f, b)
    total = ys[0] + ys[n]
    for i in range(1, n):
        ys[i] = _sample(f, a + i * h)
        total += (4.0 if i % 2 else 2.0) * ys[i]
    return h / 3.0 * total, ys, h


def _simpson_from(ys: "Sequence[float]", h: float) -> float:
    """Composite Simpson from samples already taken (an even number of subintervals)."""
    total = ys[0] + ys[-1]
    for i in range(1, len(ys) - 1):
        total += (4.0 if i % 2 else 2.0) * ys[i]
    return h / 3.0 * total


def _halved_grid_total(
    f: Callable[[float], float], ys: "list[float]", a: float, h: float
) -> "tuple[float | None, float]":
    """``I_2n`` from the ``n``-grid samples plus the midpoints between them.

    ONE extra pass over the compiled function (``n`` new samples, the existing ``n+1``
    reused), so the error estimate costs a constant number of passes and not a single
    SymPy call. Returns ``(I_2n, 0.0)``, or ``(None, x)`` naming the midpoint that
    could not be sampled: the VALUE was computable there and only its estimate is not,
    so this is the one place quadrature reports a bad sample as "the estimate is
    unavailable" instead of refusing outright — the caller words it.
    """
    n = len(ys) - 1
    fine = [0.0] * (2 * n + 1)
    for i, y in enumerate(ys):
        fine[2 * i] = y
    for i in range(n):
        x = a + (i + 0.5) * h
        mid = _safe(f, x)
        if mid is None:
            return None, x
        fine[2 * i + 1] = mid
    return _simpson_from(fine, 0.5 * h), 0.0


def _panel_too_narrow(lo: float, hi: float) -> bool:
    """Is this panel below the resolution of the float grid at its own location?

    Absolute widths mean nothing here — near ``1e15`` one float step is ``0.125``,
    near zero it is ``5e-324``. The limit is therefore expressed in steps OF THE
    ENDPOINTS (``_QUAD_MIN_ULPS``), which is what actually decides whether the five
    nodes of a Simpson panel are distinct numbers.
    """
    return abs(hi - lo) <= _QUAD_MIN_ULPS * max(math.ulp(lo), math.ulp(hi))


def _adaptive_quad(
    f: Callable[[float], float],
    a: float,
    ys: "list[float]",
    h: float,
    tol: float,
    max_evals: int,
    evaluations: int,
) -> "tuple[float, float, int, int, SolveStatus, int, float]":
    """Locally adaptive Simpson, seeded from the fixed-``n`` grid already sampled.

    The seed matters and is not decoration: the panels of the fixed rule ARE the
    starting panels here, so the adaptive run begins from exactly the answer the
    default call gives and refines it. Starting instead from the whole interval as one
    panel is how adaptive quadrature gets fooled — on ``cos(16πx)`` a single panel's
    coarse and fine rules agree to the last bit (every node lands on a maximum), the
    criterion is satisfied at once and the run returns a 100%-wrong integral. Seeding
    from ``n`` panels does not make that impossible (nothing does — see
    ``integrate_num`` on when the estimate lies), it only makes it need an integrand
    aliased to a much finer grid.

    Each panel is compared against its own halving: ``|S_fine - S_coarse|/15``
    estimates the error of ``S_fine``, the panel keeps its share of ``tol``, and a
    split hands each half HALF the share — so the shares of any complete set of
    panels sum to ``tol``. An accepted panel contributes the Richardson-corrected
    ``S_fine + (S_fine - S_coarse)/15``. Two new samples per visit, the other three
    inherited from the parent.

    Returns ``(value, error_estimate, evaluations, refinements, status, unresolved,
    where)``: ``unresolved`` counts panels handed back without meeting their share
    and ``where`` locates the narrowest of them (for the message).
    """
    n = len(ys) - 1
    panels = n // 2
    # (lo, hi, f(lo), f(mid), f(hi), share of tol)
    stack: "list[tuple[float, float, float, float, float, float]]" = [
        (a + 2 * p * h, a + 2 * (p + 1) * h, ys[2 * p], ys[2 * p + 1], ys[2 * p + 2], tol / panels)
        for p in range(panels)
    ]
    total = 0.0
    estimate = 0.0
    refinements = 0
    floored = 0
    where = a
    while stack:
        lo, hi, y0, y1, y2, share = stack.pop()
        half = 0.5 * (hi - lo)  # signed: b < a integrates backwards, as it always did
        q1 = _sample(f, lo + 0.5 * half)
        q2 = _sample(f, lo + 1.5 * half)
        evaluations += 2
        coarse = half / 3.0 * (y0 + 4.0 * y1 + y2)
        fine = half / 6.0 * (y0 + 4.0 * q1 + 2.0 * y1 + 4.0 * q2 + y2)
        local = abs(fine - coarse) / _QUAD_RICHARDSON
        if local <= share:
            total += fine + (fine - coarse) / _QUAD_RICHARDSON
            estimate += local
            continue
        if _panel_too_narrow(lo, hi):
            # The float grid cannot resolve this panel further. We bank the best value
            # we have WITH its estimate (nothing here is unmeasurable) and let the
            # verdict below decide whether the total still fits inside tol.
            total += fine + (fine - coarse) / _QUAD_RICHARDSON
            estimate += local
            floored += 1
            where = lo
            continue
        if evaluations >= max_evals:
            # Budget gone. Everything still on the stack is banked at its COARSE value
            # — an answer of some sort — but its error is not measured at all, so the
            # estimate becomes math.inf by the same rule module 4 set for a residual
            # that cannot be taken: unmeasurable, and every `estimate <= tol` test
            # answers False, which is the safe direction.
            total += fine + (fine - coarse) / _QUAD_RICHARDSON
            for slo, shi, s0, s1, s2, _share in stack:
                total += (shi - slo) / 6.0 * (s0 + 4.0 * s1 + s2)
            return (
                total,
                math.inf,
                evaluations,
                refinements,
                "not_converged",
                floored + 1 + len(stack),
                lo,
            )
        refinements += 1
        mid = 0.5 * (lo + hi)
        stack.append((lo, mid, y0, q1, y1, 0.5 * share))
        stack.append((mid, hi, y1, q2, y2, 0.5 * share))
    # VERDICT. Every panel met its share -> the accuracy asked for was delivered, and
    # that half of the test needs no comparison of accumulated floats. A panel stopped
    # by the float grid can still be harmless: what it left behind is measured, so if
    # the TOTAL estimate fits inside tol the run converged anyway (that is the common
    # case for √x, where one panel touches the singularity and the rest is exact).
    if floored == 0 or estimate <= tol:
        return total, estimate, evaluations, refinements, "converged", 0, where
    return total, estimate, evaluations, refinements, "stagnated", floored, where


@overload
def integrate_num(
    expr: Expr,
    var: str,
    a: float,
    b: float,
    n: int = ...,
    *,
    tol: "float | None" = ...,
    max_evals: int = ...,
    full_result: Literal[False] = ...,
) -> float: ...


@overload
def integrate_num(
    expr: Expr,
    var: str,
    a: float,
    b: float,
    n: int = ...,
    *,
    tol: "float | None" = ...,
    max_evals: int = ...,
    full_result: Literal[True],
) -> QuadratureResult: ...


def integrate_num(
    expr: Expr,
    var: str,
    a: float,
    b: float,
    n: int = 100,
    *,
    tol: "float | None" = None,
    max_evals: int = _QUAD_MAX_EVALS,
    full_result: bool = False,
) -> "float | QuadratureResult":
    """Definite integral ``∫_a^b f(var) d(var)`` by composite Simpson quadrature.

    A numerical alternative to ``symbolic.integrate`` when the integral has no
    closed form. Simpson is exact for polynomials of degree ≤ 3 regardless of
    ``n``; for the rest ``n`` controls accuracy (even — we round up). A point
    outside the domain -> ``PycodemathError``.

    ``tol`` (keyword-only, module 5) turns the fixed rule into a LOCALLY ADAPTIVE one:
    the panels of the ``n``-grid are refined where the integrand is difficult until the
    estimated ABSOLUTE error fits inside ``tol``. Same rule, same seed, samples spent
    where they buy accuracy — measured: ``√x`` on ``[0, 1]`` to ``1e-10`` costs 1 005
    evaluations this way against 262 145 for a global grid, and a fixed ``n`` would
    need over a million subintervals. Without ``tol`` NOTHING changes: the same grid,
    the same arithmetic in the same order, the same ``float``, bit for bit.

    ``full_result=True`` (keyword-only) returns a ``QuadratureResult`` instead of the
    bare number: the same ``float`` in ``.value``, plus ``.error_estimate``,
    ``.evaluations``, ``.refinements``, ``.converged`` and ``.status``. On the fixed
    path it costs ONE extra pass over the same compiled function (the ``n`` midpoints
    of the grid, never a SymPy call per sample); the value is untouched.

    THE ERROR ESTIMATE — what it is, what it assumes, and WHEN IT LIES.
    It is the classical Richardson comparison of a grid with its halving: with the
    error of Simpson behaving as ``C·h⁴``, halving ``h`` divides it by 16, so
    ``err(I_n) ≈ 16/15·|I_2n - I_n|`` (fixed path, one extra pass) and
    ``err(I_fine) ≈ |I_fine - I_coarse|/15`` per panel (adaptive path). It ASSUMES the
    integrand is smooth enough on the interval for that ``C·h⁴`` law to hold — a
    bounded fourth derivative, and a grid fine enough to see the integrand's features.
    Where the assumption fails the estimate is not merely loose, it is OPTIMISTIC:

      * ``√x`` on ``[0, 1]``: the fourth derivative blows up at 0, the order drops to
        ~1.5 and the estimate comes out ~1.5× too small at ``n=100`` (5.6e-05 against
        a true 8.1e-05) — a mild lie, but a systematic one.
      * ``cos(16πx)`` on ``[0, 1]`` with ``n=4``: every node of the ``n``-grid AND of
        its halving lands on a maximum, both grids return exactly ``1.0``, so the
        estimate is exactly ``0.0`` — while the true integral is ``0`` and the answer
        is off by 100%. The test suite pins this case: an estimate of zero is a
        statement about two grids agreeing, NEVER a proof of accuracy.

    So: read ``error_estimate`` as evidence, not as a bound. It is honest on smooth
    integrands (it matched the true error to ~1× on ``sin``, ``exp`` and ``sin 50x``),
    and a large value is always a real warning; a small one is a warning's absence, not
    a guarantee. ``tol`` inherits exactly this status — it is a target measured BY the
    estimate, so an aliased integrand can satisfy it while being wrong.

    FAILURES, and where they are returnable. Without ``tol`` there is no accuracy to
    miss and the only failure is a refusal (below). With ``tol`` two outcomes are
    possible, both raising on the default path and RETURNED under ``full_result=True``
    (module 4's contract) with ``converged=False``:

      * the evaluation budget ``max_evals`` runs out -> ``BudgetExhaustedError``,
        ``status="not_converged"``, ``error_estimate=math.inf`` (part of the interval
        was never refined, so the error is not measured — the same "unmeasurable is
        inf" rule the solvers use for a residual). This is the one failure in the
        library that is purely ECONOMIC: nothing about the integrand is wrong, the
        refinement was still working, and a larger ``max_evals`` finishes the job. That
        is why it is the class that says an explicit budget ran out, and not the class
        the iterative solvers raise when their own ``max_iter`` cap ends a run with
        nothing to show.
      * a panel reaches the resolution of the float grid while its share of ``tol`` is
        still unmet, and what it left behind does not fit inside ``tol`` ->
        ``StagnationError``, ``status="stagnated"``, with a FINITE estimate (every
        panel was measured). Refinement stopped making progress and no larger budget
        will change that: the tolerance is below what double precision can deliver
        there. ``"diverged"`` never occurs — an integrand escaping to infinity is
        refused at the sample instead.

    ``.value`` on a failure is the best composite the run reached — a rough number, NOT
    accurate to ``tol``; ``.converged`` is the only field to branch on.

    REFUSALS stay refusals, with or without ``full_result`` (module 4's boundary — an
    input error is not an outcome of the run): ``n <= 0``, a non-positive ``tol`` or
    ``max_evals``, an expression depending on more than ``var``, and a sample outside
    the function's domain. Quadrature refuses at a bad sample rather than changing
    strategy — including a sample the ESTIMATE needs (a midpoint of the grid) that the
    value itself did not, and there the message says so, because the answer is still
    available from the call without ``full_result``.
    """
    if n <= 0:
        raise DomainError("the number of subintervals n must be positive")
    if n % 2:
        n += 1  # Simpson requires an even number of subintervals
    if tol is not None and not tol > 0.0:
        raise DomainError(
            f"integrate_num: the tolerance tol must be positive, got {tol!r} — "
            f"omit tol for the fixed rule on n subintervals"
        )
    if max_evals <= 0:
        raise DomainError(
            f"integrate_num: the evaluation budget max_evals must be positive, "
            f"got {max_evals!r}"
        )

    a, b = float(a), float(b)
    if a == b:
        # An empty interval is exact, needs no sample and refines nothing.
        return QuadratureResult.success(0.0, 0.0, 0, 0) if full_result else 0.0

    f = _scalar(expr, var)
    value, ys, h = _simpson_grid(f, a, b, n)
    if tol is None:
        if not full_result:
            return value  # the default path, byte for byte the pre-module-5 code
        i_2n, bad_x = _halved_grid_total(f, ys, a, h)
        if i_2n is None:
            raise DomainError(
                f"integrate_num integrated {expr} over [{a:g}, {b:g}], but the error "
                f"estimate needs one sample BETWEEN the grid points and the function "
                f"cannot be sampled at x={bad_x:g} (outside the domain or complex "
                f"result) — call without full_result=True to get the value alone, "
                f"or change n"
            )
        estimate = (_QUAD_RICHARDSON + 1.0) / _QUAD_RICHARDSON * abs(i_2n - value)
        return QuadratureResult.success(value, estimate, 2 * n + 1, 0)

    total, estimate, evaluations, refinements, status, unresolved, where = _adaptive_quad(
        f, a, ys, h, tol, max_evals, n + 1
    )
    if status == "converged":
        return (
            QuadratureResult.success(total, estimate, evaluations, refinements)
            if full_result
            else total
        )
    if full_result:
        return QuadratureResult.failure(total, estimate, evaluations, refinements, status)
    if status == "not_converged":
        raise BudgetExhaustedError(
            f"integrate_num exhausted its budget of {max_evals} function evaluations "
            f"for {expr} over [{a:g}, {b:g}]: {unresolved} panel(s) are still "
            f"unrefined, so tol={tol:g} is not certified and the remaining error is "
            f"not measured at all (the value reached is {total:.12g}). Nothing is wrong "
            f"with the integrand — raise max_evals, loosen tol, or split the interval "
            f"where the integrand peaks."
        )
    raise StagnationError(
        # The endpoints go in with full precision here, not the ``:g`` the other
        # messages use: this failure is ABOUT the resolution of the float grid, and
        # ``[1e+15, 1e+15]`` for an interval of width 1 would hide exactly the fact
        # being reported.
        f"integrate_num STALLED for {expr} over [{a!r}, {b!r}]: {unresolved} panel(s) "
        f"hit the resolution limit of double precision near x={where:g} (a panel under "
        f"{_QUAD_MIN_ULPS:g} float steps wide there has no distinct nodes left to "
        f"subdivide), and the error estimate stopped at {estimate:.3g} ≥ tol={tol:g} — "
        f"this is the limit of the grid, not divergence: a bigger max_evals will not "
        f"help. Loosen tol, or move the interval away from x={where:g}."
    )


# --- minimization ------------------------------------------------------
@overload
def minimize(
    expr: Expr,
    var: str,
    x0: float,
    lr: float = ...,
    tol: float = ...,
    max_iter: int = ...,
    method: str = ...,
    *,
    full_result: Literal[False] = ...,
) -> float: ...


@overload
def minimize(
    expr: Expr,
    var: str,
    x0: float,
    lr: float = ...,
    tol: float = ...,
    max_iter: int = ...,
    method: str = ...,
    *,
    full_result: Literal[True],
) -> SolveResult[float]: ...


def minimize(
    expr: Expr,
    var: str,
    x0: float,
    lr: float = 0.1,
    tol: float = 1e-9,
    max_iter: int = 10000,
    method: str = "gd",
    *,
    full_result: bool = False,
) -> "float | SolveResult[float]":
    """Find the argument of the 1D minimum.

    By default (``method="gd"``) gradient descent: the gradient is the SYMBOLIC
    derivative (``Expr.diff``), the step ``lr`` is halved on overshoot (historical
    behavior, results bit-identical to versions before module 15).
    ``method="newton"`` / ``"bfgs"`` are 2nd-order methods with an Armijo line
    search (the shared ``_descent_min`` core with ``minimize_nd``) — Newton hits
    the minimum of a quadratic in one step; ``lr`` does not apply to them. Returns
    the ``x`` minimizing ``expr``.

    ``full_result=True`` (keyword-only) returns a ``SolveResult[float]`` instead of
    the bare argument: the same ``float`` in ``.value``, plus ``.iterations`` and
    ``.residual`` = ``|f'(x)|`` at the returned point (the 1D case of the
    ``‖∇f‖∞`` convention). Works for every ``method``. The default call is
    unchanged.

    Function unbounded below / no convergence / unknown method ->
    ``PycodemathError``. Under ``full_result=True`` the ITERATION outcomes are
    RETURNED instead (module 4): ``converged=False`` with ``status="diverged"``
    (the iterate escaped to infinity), ``"stagnated"`` (the value stopped dropping
    while still above ``tol``) or ``"not_converged"`` (the budget ran out / the
    step could not be taken) — for every ``method``. ``.value`` is then the LAST
    ITERATE, the point the method gave up at and NOT a minimum; ``.converged`` is
    the only field to branch on, and ``.residual`` is ``|f'(x)|`` there (``math.inf``
    when it cannot be sampled — see ``_failure_residual``). An input refusal
    (``DomainError`` — an unknown ``method``, an expression depending on more than
    ``var``) raises either way: it is not an outcome of any iteration.
    """
    _check_method(method, "minimize")
    if method != "gd":
        e = Expr(expr)
        _check_vars([e], [var], "minimize")
        point, iters, res, status = _descent_min(
            e, [var], [x0], tol, int(max_iter), method, "minimize",
            full_result=full_result,
        )
        if full_result:
            if status != "converged":
                # ``_descent_min`` returned its failure instead of raising (module
                # 4); it reports the residual as ``float | None`` and the caller
                # applies the rule for its own path — a refusal on success, inf here.
                return SolveResult.failure(
                    point[0], iters, _failure_residual(res), status
                )
            return SolveResult.success(
                point[0], iters, _need_residual(res, "minimize", point[0])
            )
        return point[0]
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
    prev_res: "float | None" = None  # |g| one iterate back (module 8, see below)
    it = 0  # gd iterations that actually ran (module 3: reported as .iterations)

    def certified(point: float, iters: int, res: float, fp: float) -> "float | SolveResult[float]":
        """Hand back the answer — or refuse it (module 8). One rule for both exits."""
        arr = np.array([point], dtype=float)
        lower = _lower_value_nearby(lambda p: _safe(f, float(p[0])), None, arr, fp, res)
        if lower is None:
            return SolveResult.success(point, iters, res) if full_result else point
        if full_result:
            return SolveResult.failure(point, iters, res, "not_a_minimum")
        probe, f_probe = lower
        raise _not_a_minimum_error("minimize", expr, x0, point, fp, probe, f_probe)

    for it in range(1, max_iter + 1):
        g = _safe(grad, x)
        if g is None:
            break
        res = abs(g)
        if res < tol:
            # gradient exit: the test already measured the residual
            return certified(x, it, res, fx)
        # STAGNATION, half one (module 8): a run that is CONVERGING, only slowly, is
        # not stalled. The streak resets on progress in the RESIDUAL — the quantity
        # the exit test actually reads — and not only on progress in ``f``. Near a
        # minimum ``f - f*`` goes as the SQUARE of the distance, so it hits the
        # rounding noise of float64 while ``|g|`` is still around 1e-8 and falling
        # geometrically; judging liveness by ``f`` alone declared ``sin(x)`` from 0.0
        # stalled at x=-1.5707962883640527 — the minimum, right to 7 significant
        # figures. The cycle the detector was BUILT for is untouched: ``x^4`` at
        # lr=1.0 bounces between ±2 with ``|g|`` pinned at 32.0, so nothing here
        # resets and it still stalls. This costs no samples: ``g`` is the gradient
        # this iteration already needed.
        if prev_res is not None and _value_progressed(prev_res, res):
            stag = 0
        prev_res = res
        x_next = x - lr * g
        # Iterate explosion = function unbounded below. We catch it HERE, before
        # _safe samples f at a gigantic point (overflow -> None -> misleading "bad
        # step"). A convergent run never reaches here (|x_next| stays small).
        if _has_diverged(x_next, diverge_limit):
            if full_result:
                # FAILURE form (module 4): the last iterate is the escaped
                # ``x_next``. The gradient was last sampled one point back, so the
                # residual AT the returned point needs its own sample — one, on the
                # exit path, from the already-compiled derivative (the same
                # discipline as the STEP exit below); inf when it overflows there.
                g_end = _safe(grad, x_next)
                return SolveResult.failure(
                    x_next,
                    it,
                    _failure_residual(None if g_end is None else abs(g_end)),
                    "diverged",
                )
            raise DivergenceError(
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
                if full_result:
                    # FAILURE form (module 4): the stall is declared on the step we
                    # just accepted, so the last iterate is ``x_next`` — and ``g``
                    # is the gradient one point back, hence one fresh sample.
                    g_end = _safe(grad, x_next)
                    return SolveResult.failure(
                        x_next,
                        it,
                        _failure_residual(None if g_end is None else abs(g_end)),
                        "stagnated",
                    )
                raise StagnationError(
                    f"minimize STALLED for {expr} at x0={x0}: over {stag} "
                    f"consecutive steps the function value did not drop significantly "
                    f"(f≈{f_next:.6g}), and |g|={abs(g):.3g} ≥ tol — this is STAGNATION, "
                    f"not divergence and not unboundedness: the run stalled (cycle / "
                    f"oscillation with too large a step). Reduce lr, reduce tol "
                    f"or provide a better starting point."
                )
        if abs(x_next - x) < tol:
            # STEP exit: the gradient was last sampled one point BACK, so the
            # residual at the point we are returning needs its own sample — one,
            # on the exit path, from the already-compiled derivative.
            g_end = _safe(grad, x_next)
            if g_end is None and not full_result:
                # unchanged from before module 8: without a residual the certificate
                # has no allowance to measure a drop against, so nothing is refuted
                # and the default call still gets its value (``full_result`` refuses
                # readably one line below, as it always has)
                return x_next
            return certified(
                x_next,
                it,
                _need_residual(
                    None if g_end is None else abs(g_end), "minimize", x_next
                ),
                f_next,
            )
        x, fx = x_next, f_next

    if full_result:
        # FAILURE form (module 4): every exit that lands here — an exhausted budget,
        # a gradient/value outside the domain, an ``lr`` shrunk to nothing — gave up
        # at the last ACCEPTED iterate ``x``. One sample of the derivative there
        # (``g`` above may belong to an earlier point after the last accepted step).
        g_end = _safe(grad, x)
        return SolveResult.failure(
            x,
            it,
            _failure_residual(None if g_end is None else abs(g_end)),
            "not_converged",
        )
    raise NonConvergenceError(
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
        raise DomainError(f"{what} requires at least one variable")
    if len(set(vars)) != len(vars):
        raise DomainError(f"{what}: the variable list contains duplicates")
    needed: set[str] = set()
    for e in exprs:
        needed |= set(e.symbol_names())
    extra = sorted(needed - set(vars))
    if extra:
        raise DomainError(
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
        raise DomainError(
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
        raise DomainError("gradient requires at least one variable")
    return Matrix([[e.diff(v).sy] for v in vars])


def jacobian(exprs: "Sequence[Expr]", vars: "Sequence[str]") -> Matrix:
    """Jacobian of the system — a row per function, a column per variable (symbolically)."""
    fs = [Expr(e) for e in exprs]
    vars = list(vars)
    if not fs or not vars:
        raise DomainError("jacobian requires at least one function and variable")
    return Matrix([[f.diff(v).sy for v in vars] for f in fs])


def hessian(expr: Expr, vars: "Sequence[str]") -> Matrix:
    """Hessian ``∇²f`` — the matrix of second partial derivatives (symbolically)."""
    e = Expr(expr)
    vars = list(vars)
    if not vars:
        raise DomainError("hessian requires at least one variable")
    return Matrix([[e.diff(vi).diff(vj).sy for vj in vars] for vi in vars])


@overload
def root_find_nd(
    exprs: "Sequence[Expr]",
    vars: "Sequence[str]",
    x0: "Sequence[float]",
    tol: float = ...,
    max_iter: int = ...,
    *,
    full_result: Literal[False] = ...,
) -> list[float]: ...


@overload
def root_find_nd(
    exprs: "Sequence[Expr]",
    vars: "Sequence[str]",
    x0: "Sequence[float]",
    tol: float = ...,
    max_iter: int = ...,
    *,
    full_result: Literal[True],
) -> SolveResult[list[float]]: ...


def root_find_nd(
    exprs: "Sequence[Expr]",
    vars: "Sequence[str]",
    x0: "Sequence[float]",
    tol: float = 1e-10,
    max_iter: int = 100,
    *,
    full_result: bool = False,
) -> "list[float] | SolveResult[list[float]]":
    """Solve the nonlinear system ``F(x) = 0`` with multidimensional Newton.

    The Jacobian is computed SYMBOLICALLY (``jacobian``); at each iteration we
    solve the linear system ``J·Δ = -F`` via ``np.linalg.solve`` (the same
    mechanics as the numerical path ``linalg.solve_system``). A singular Jacobian
    or no convergence -> ``PycodemathError``.

    ``full_result=True`` (keyword-only) returns a ``SolveResult[list[float]]``
    instead of the bare solution: the same ``list[float]`` in ``.value``, plus
    ``.iterations`` and ``.residual`` = ``‖F(x)‖∞`` at the returned point (the
    quantity the convergence test itself uses). The default call is unchanged.

    Under ``full_result=True`` the ITERATION outcomes are RETURNED instead of raised
    (module 4): ``converged=False`` with ``status="diverged"`` (the iterate escaped
    to infinity) or ``"not_converged"`` (``max_iter`` exhausted while oscillating).
    ``.value`` is then the LAST ITERATE — the point Newton gave up at, NOT a
    solution (its coordinates may even be inf/NaN when that is what the iterate
    became), and ``.converged`` is the only field to branch on. An INPUT refusal
    still raises ``DomainError`` either way: a non-square system, an expression
    depending on more than ``vars``, a starting point of the wrong length, a
    singular Jacobian, or a point where the system/Jacobian cannot be sampled.
    """
    fs = [Expr(e) for e in exprs]
    vars = list(vars)
    _check_vars(fs, vars, "root_find_nd")
    if len(fs) != len(vars):
        raise DomainError(
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
    #: first point where ``‖F‖∞ < tol`` fired without the iterate settling (module 8)
    decayed: "tuple[list[float], float, float] | None" = None
    it = 0  # Newton iterations that actually ran (module 3: reported as .iterations)
    for it in range(1, max_iter + 1):
        # Divergence takes priority over any sampling: a non-finite iterate is
        # hard proof that Newton escaped — do not sample at NaN/inf, just report
        # divergence right away (not the misleading "outside the domain").
        if not np.all(np.isfinite(x)):
            if full_result:
                # FAILURE form (module 4). We do NOT sample here — the raise path
                # refuses to for the same reason (a non-finite iterate is hard proof
                # of an escape, and sampling at NaN/inf would only produce a
                # misleading "outside the domain"), so the residual falls to the
                # documented unsampleable rule and ``.value`` carries the
                # non-finite iterate itself, honestly.
                return SolveResult.failure(
                    [float(v) for v in x], it, _failure_residual(None), "diverged"
                )
            raise DivergenceError(
                f"root_find_nd DIVERGES from x0={list(x0)}: the Newton iterate "
                f"escaped to infinity (inf/NaN). Provide a starting point "
                f"closer to the expected root."
            )
        F = F_fn(x)
        if F is None:
            raise DomainError(
                f"cannot sample the system at x={x.tolist()} "
                f"(outside the domain or complex result)"
            )
        res = _inf_norm(F)
        settled_exit = res < tol
        if settled_exit and not bool(np.any(F)):
            # an EXACT zero of the whole system is a solution by inspection — and
            # asking the Newton model about it would need a J that may be singular
            # exactly there (the nd twin of the ``fx == 0.0`` case in ``root_find``)
            sol = [float(v) for v in x]
            return SolveResult.success(sol, it, res) if full_result else sol
        Jx = J_fn(x)
        if Jx is None:
            if settled_exit:
                # cannot compute the correction, so nothing can be refuted: today's
                # acceptance stands rather than a DomainError raised at a point the
                # solver was about to hand back as the answer
                sol = [float(v) for v in x]
                return SolveResult.success(sol, it, res) if full_result else sol
            raise DomainError(
                f"cannot sample the Jacobian at x={x.tolist()} "
                f"(outside the domain or complex result)"
            )
        try:
            delta = np.linalg.solve(Jx.reshape(n, n), -F)
        except np.linalg.LinAlgError as exc:
            if settled_exit:
                sol = [float(v) for v in x]  # singular J — same rule as above
                return SolveResult.success(sol, it, res) if full_result else sol
            raise DomainError(
                f"singular Jacobian at x={x.tolist()} — Newton cannot proceed "
                f"(try a different starting point)"
            ) from exc
        if settled_exit:
            # Module 8, the nd twin of the ``_root_has_settled`` gate in
            # ``root_find``: ``‖F‖∞ < tol`` is a root only when the Newton correction
            # ‖J⁻¹F‖ is negligible too. ``[exp(x), y-1]`` from the origin walks one
            # unit per step in x for ever, and its residual falls under ``tol`` on
            # the way down the tail — with a correction that stays exactly 1.0.
            if _root_has_settled(float(np.linalg.norm(x)), float(np.linalg.norm(delta))):
                sol = [float(v) for v in x]
                return SolveResult.success(sol, it, res) if full_result else sol
            if decayed is None:
                decayed = ([float(v) for v in x], res, float(np.linalg.norm(delta)))
        x = x + delta
        # A norm explosion (and, via the shared R9 predicate, also an escape to
        # inf/NaN) = divergence. We catch it HERE, before the next turn samples the
        # system at a gigantic point and triggers overflow. The upper finiteness
        # guard (before sampling) STAYS — it catches a non-finite START with a
        # separate message; here delta is solve() of finite F/J, so it really
        # closes only the norm explosion (behavior bit-identical).
        if _has_diverged(x, diverge_limit):
            if full_result:
                # FAILURE form (module 4): the last iterate is the exploded ``x``.
                # ONE sample of the already-compiled system there — ``_vector_fn``
                # answers ``None`` on overflow, which is exactly the unsampleable
                # rule; nothing runs inside the iteration.
                F_end = F_fn(x)
                return SolveResult.failure(
                    [float(v) for v in x],
                    it,
                    _failure_residual(None if F_end is None else _inf_norm(F_end)),
                    "diverged",
                )
            raise DivergenceError(
                f"root_find_nd DIVERGES from x0={list(x0)}: the iterate norm "
                f"exceeded {diverge_limit:.3g} (‖x‖={float(np.linalg.norm(x)):.3g}) "
                f"— Newton is escaping the solution. Provide a starting point closer "
                f"to the expected root."
            )

    if full_result:
        # FAILURE form (module 4): the budget ran out at ``x`` — one step past the
        # residual the loop last measured, so we sample the system once, here.
        F_end = F_fn(x)
        return SolveResult.failure(
            [float(v) for v in x],
            it,
            _failure_residual(None if F_end is None else _inf_norm(F_end)),
            "not_converged",
        )
    if decayed is not None:
        # Module 8, as in ``root_find``: the residual DID fall under tol on the way,
        # so the generic "oscillates without dropping below the tolerance" would be
        # false — it dropped, and stayed meaningless.
        pd, resd, corr = decayed
        raise NonConvergenceError(
            f"root_find_nd finds no solution of {[str(f) for f in fs]} near "
            f"x0={list(x0)}: ‖F‖∞ fell below tol at x={pd} (‖F‖∞={resd:.3g}), but the "
            f"Newton correction there was still ‖Δ‖={corr:.3g} — the system is "
            f"DECAYING towards zero, not reaching it. A small residual is not a "
            f"solution. Provide a starting point closer to an expected solution, or "
            f"check whether a real one exists at all."
        )
    raise NonConvergenceError(
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
        raise DomainError(
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
    *,
    full_result: bool = False,
) -> "tuple[list[float], int, float | None, SolveStatus]":
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

    Returns ``(point, iterations, residual, status)`` — the private counterpart of
    the public ``SolveResult`` (module 3): ``point`` is bit-identical to the former
    plain ``list[float]`` return, ``residual`` is ``‖∇f‖∞`` at that point, or
    ``None`` when the gradient is not sampleable there (the callers turn that into
    a readable refusal via ``_need_residual``, and only when the caller actually
    asked for ``full_result``).

    ``status`` (module 4) is ``"converged"`` on every success. With
    ``full_result=True`` the two ITERATION failures come back the same way instead
    of raising — ``"diverged"`` / ``"not_converged"`` with the last iterate as
    ``point`` — and the caller wraps them into the public failure form. With
    ``full_result=False`` (the default, and the whole default path of
    ``minimize``/``minimize_nd``) they still raise, message untouched, so ``status``
    is then always ``"converged"``. A ``DomainError`` (an unsampleable STARTING
    point, a bad ``x0`` length) raises either way — it is an input refusal, not an
    outcome of an iteration.
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
        raise DomainError(
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

    def _f_scalar(p: np.ndarray) -> "float | None":
        v = f_fn(p)
        return None if v is None else float(v[0])

    def certified(
        point: np.ndarray, iters: int, res: "float | None", fp: float
    ) -> "tuple[list[float], int, float | None, SolveStatus]":
        """Module 8: a stationary point is only an answer once nothing refutes it.

        Shared by both 2nd-order success exits, and by ``minimize`` and
        ``minimize_nd`` through them — Newton and BFGS reach a maximum or a saddle
        exactly as gd does when the START is one (the line search never gets to run:
        the gradient test fires on iteration 1). With ``res`` unknown the point comes
        back uncertified, because the drop-versus-allowance comparison has no
        allowance to use — the callers then refuse it readably via
        ``_need_residual``, which is what they did before this module too.
        """
        sol = [float(v) for v in point]
        if res is None:
            return sol, iters, None, "converged"
        lower = _lower_value_nearby(_f_scalar, g_fn, point, fp, res)
        if lower is None:
            return sol, iters, res, "converged"
        if full_result:
            return sol, iters, res, "not_a_minimum"
        probe, f_probe = lower
        raise _not_a_minimum_error(what, f, list(x0), sol, fp, probe, f_probe)

    it = 0  # iterations that actually ran (module 3: reported as .iterations)
    for it in range(1, max_iter + 1):
        if g is None:
            break  # gradient outside the domain — cannot proceed
        res = _inf_norm(g)
        if res < tol:
            # gradient exit: the test already measured the residual
            return certified(x, it, res, fx)

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
            if full_result:
                # FAILURE form (module 4): the last iterate is the escaped
                # ``x_new``; ``g`` belongs to ``x``, so the residual AT that point
                # takes one sample of the already-compiled gradient. ``None`` is
                # passed up honestly — the caller applies the inf rule.
                g_end = g_fn(x_new)
                return (
                    [float(v) for v in x_new],
                    it,
                    None if g_end is None else _inf_norm(g_end),
                    "diverged",
                )
            raise DivergenceError(
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
            # STEP exit: the gradient at the returned point is the ``g_new`` we
            # already sampled above — no extra work, and ``None`` there is passed
            # up honestly instead of being papered over with the previous point's.
            return certified(
                x_new, it, None if g_new is None else _inf_norm(g_new), f_new
            )
        x, fx, g = x_new, f_new, g_new

    if full_result:
        # FAILURE form (module 4): whatever brought us here — an exhausted budget, a
        # gradient outside the domain, a line search that found no descent — the run
        # gave up at ``x``, and ``g`` IS the gradient at ``x`` (sampled at the top of
        # the loop, replaced together with ``x`` at the bottom), so no extra sample
        # is needed anywhere on this path.
        return (
            [float(v) for v in x],
            it,
            None if g is None else _inf_norm(g),
            "not_converged",
        )
    raise NonConvergenceError(
        f"{what} does not converge for {f} at x0={list(x0)} with method {method!r} "
        f"(function unbounded below or bad starting point?)"
    )


@overload
def minimize_nd(
    expr: Expr,
    vars: "Sequence[str]",
    x0: "Sequence[float]",
    lr: float = ...,
    tol: float = ...,
    max_iter: int = ...,
    method: str = ...,
    *,
    full_result: Literal[False] = ...,
) -> list[float]: ...


@overload
def minimize_nd(
    expr: Expr,
    vars: "Sequence[str]",
    x0: "Sequence[float]",
    lr: float = ...,
    tol: float = ...,
    max_iter: int = ...,
    method: str = ...,
    *,
    full_result: Literal[True],
) -> SolveResult[list[float]]: ...


def minimize_nd(
    expr: Expr,
    vars: "Sequence[str]",
    x0: "Sequence[float]",
    lr: float = 0.1,
    tol: float = 1e-9,
    max_iter: int = 10000,
    method: str = "gd",
    *,
    full_result: bool = False,
) -> "list[float] | SolveResult[list[float]]":
    """Find the argument of the minimum of a function of several variables.

    By default (``method="gd"``) gradient descent — a SYMBOLIC gradient
    (``Expr.diff`` for each variable), numerical iteration, the step ``lr``
    halved on overshoot (historical behavior, results bit-identical to versions
    before module 15). ``method="newton"`` / ``"bfgs"`` are 2nd-order methods
    with an Armijo line search (``_descent_min``) — Newton with a symbolic
    Hessian converges on smooth problems in a few iterations, BFGS copes where
    gradient descent chokes (e.g. the Rosenbrock valley); the ``lr`` parameter
    does not apply to them. Returns the point ``x`` minimizing ``expr``.

    ``full_result=True`` (keyword-only) returns a ``SolveResult[list[float]]``
    instead of the bare point: the same ``list[float]`` in ``.value``, plus
    ``.iterations`` and ``.residual`` = ``‖∇f(x)‖∞`` at the returned point. Works
    for every ``method``. The default call is unchanged.

    Function unbounded below / no convergence / unknown method ->
    ``PycodemathError``. Under ``full_result=True`` the ITERATION outcomes are
    RETURNED instead (module 4): ``converged=False`` with ``status="diverged"``,
    ``"stagnated"`` or ``"not_converged"``, for every ``method``. ``.value`` is then
    the LAST ITERATE — the point the method gave up at, NOT a minimum — and
    ``.converged`` is the only field to branch on; ``.residual`` is ``‖∇f(x)‖∞``
    there, or ``math.inf`` when it cannot be sampled (see ``_failure_residual``). An
    input refusal (``DomainError`` — an unknown ``method``, extra symbols, a bad
    ``x0`` length, a starting point outside the domain) raises either way.
    """
    _check_method(method, "minimize_nd")
    f = Expr(expr)
    vars = list(vars)
    _check_vars([f], vars, "minimize_nd")
    if method != "gd":
        point, iters, res, status = _descent_min(
            f, vars, x0, tol, int(max_iter), method, "minimize_nd",
            full_result=full_result,
        )
        if full_result:
            if status != "converged":
                # ``_descent_min`` returned its failure instead of raising (module 4)
                return SolveResult.failure(point, iters, _failure_residual(res), status)
            return SolveResult.success(
                point, iters, _need_residual(res, "minimize_nd", point)
            )
        return point
    f_fn = _vector_fn([f], vars)  # compile once — iterate without SymPy overhead
    g_fn = _vector_fn([f.diff(v) for v in vars], vars)

    x = _point(x0, len(vars), "minimize_nd")
    fx_vec = f_fn(x)
    if fx_vec is None:
        raise DomainError(
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
    prev_res: "float | None" = None  # ‖g‖∞ one iterate back (module 8)
    it = 0  # gd iterations that actually ran (module 3: reported as .iterations)

    def _f_scalar(p: np.ndarray) -> "float | None":
        v = f_fn(p)
        return None if v is None else float(v[0])

    def certified(
        point: np.ndarray, iters: int, res: float, fp: float
    ) -> "list[float] | SolveResult[list[float]]":
        """Hand back the answer — or refuse it (module 8). One rule for both exits."""
        sol = [float(v) for v in point]
        lower = _lower_value_nearby(_f_scalar, g_fn, point, fp, res)
        if lower is None:
            return SolveResult.success(sol, iters, res) if full_result else sol
        if full_result:
            return SolveResult.failure(sol, iters, res, "not_a_minimum")
        probe, f_probe = lower
        raise _not_a_minimum_error("minimize_nd", f, list(x0), sol, fp, probe, f_probe)

    for it in range(1, max_iter + 1):
        g = g_fn(x)
        if g is None:
            break
        res = _inf_norm(g)
        if res < tol:
            # gradient exit: the test already measured the residual
            return certified(x, it, res, fx)
        # STAGNATION, half one (module 8): the streak resets on progress in the
        # RESIDUAL as well as in ``f`` — see the twin comment in ``minimize`` for the
        # measurement that forced it (``f`` hits the rounding floor while ``‖g‖`` is
        # still falling geometrically) and for why the ``x^4+y^4`` cycle at lr=1.0,
        # whose ‖g‖∞ is pinned at 32.0, still stalls.
        if prev_res is not None and _value_progressed(prev_res, res):
            stag = 0
        prev_res = res
        x_next = x - lr * g
        # A norm explosion = function unbounded below. We catch it HERE, before f_fn
        # samples at a gigantic point (overflow). A convergent run does not reach it.
        xn_norm = float(np.linalg.norm(x_next))
        if _has_diverged(x_next, diverge_limit):
            if full_result:
                # FAILURE form (module 4): the last iterate is the escaped
                # ``x_next``; the gradient was sampled one point back, so one fresh
                # sample here (inf when it overflows there).
                g_end = g_fn(x_next)
                return SolveResult.failure(
                    [float(v) for v in x_next],
                    it,
                    _failure_residual(None if g_end is None else _inf_norm(g_end)),
                    "diverged",
                )
            raise DivergenceError(
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
                if full_result:
                    # FAILURE form (module 4): the stall is declared on the step just
                    # accepted, so the last iterate is ``x_next`` — one fresh sample
                    # of the gradient there (``g`` belongs to the previous point).
                    g_end = g_fn(x_next)
                    return SolveResult.failure(
                        [float(v) for v in x_next],
                        it,
                        _failure_residual(None if g_end is None else _inf_norm(g_end)),
                        "stagnated",
                    )
                raise StagnationError(
                    f"minimize_nd STALLED for {f} at x0={list(x0)}: over {stag} "
                    f"consecutive steps the function value did not drop significantly "
                    f"(f≈{f_next:.6g}), and ‖g‖∞={float(np.max(np.abs(g))):.3g} ≥ tol "
                    f"— this is STAGNATION, not divergence and not unboundedness: the "
                    f"run stalled (cycle / oscillation with too large a step). Reduce "
                    f"lr, reduce tol or provide a better starting point."
                )
        if float(np.max(np.abs(x_next - x))) < tol:
            # STEP exit: the gradient was last sampled one point BACK — one extra
            # sample of the already-compiled gradient, on the exit path only.
            g_end = g_fn(x_next)
            if g_end is None and not full_result:
                return [float(v) for v in x_next]  # as in ``minimize``: no residual,
            return certified(  # no allowance, nothing refuted, value unchanged
                x_next,
                it,
                _need_residual(
                    None if g_end is None else _inf_norm(g_end),
                    "minimize_nd",
                    [float(v) for v in x_next],
                ),
                f_next,
            )
        x, fx = x_next, f_next

    if full_result:
        # FAILURE form (module 4): the run gave up at the last ACCEPTED iterate ``x``
        # (exhausted budget, gradient/value outside the domain, ``lr`` shrunk to
        # nothing) — one sample of the gradient there.
        g_end = g_fn(x)
        return SolveResult.failure(
            [float(v) for v in x],
            it,
            _failure_residual(None if g_end is None else _inf_norm(g_end)),
            "not_converged",
        )
    raise NonConvergenceError(
        f"minimize_nd does not converge for {f} at x0={list(x0)} "
        f"(function unbounded below or bad step?)"
    )
