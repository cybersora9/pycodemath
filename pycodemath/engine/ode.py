"""Ordinary differential equations (module 6) — symbolically or via RK4.

Operates on IR (``Expr``). The equation is written via the RIGHT-hand side
``y'(t) = f(t, y)``, in which the unknown function appears as an ordinary
symbol (``y``) — only ``dsolve`` substitutes ``sympy.Function`` for it.
When SymPy does not find a closed form (or finds only an implicit one),
we get a ``NoClosedFormError`` with a hint to drop down to numerics and
``route="ode"`` — the numerical command that answers the same question.

Numerics: the classic fixed-step RK4. As throughout the engine, the
right-hand side is compiled ONCE before the loop (``Expr.compiled`` via
``_vector_fn``), iterations run without SymPy overhead; ``Expr.evalf`` would
be orders of magnitude too slow here.

Functions:
  dsolve               — symbolic solution of an ODE (sympy.dsolve, IR Expr).
  solve_ode_num        — initial value problem y' = f(t, y) by RK4 (fixed step).
  solve_ode_adaptive   — the same, but the step is chosen automatically
                         (Dormand-Prince 5(4), tolerance rtol/atol) — fewer
                         steps on a smooth problem, denser sampling near a
                         singularity.
  solve_ode_dense      — as above + DENSE OUTPUT: returns a callable object
                         sol(t) that reads y(t) at ANY point
                         (4th-order DOPRI5 interpolant), not only at nodes.
  solve_ode_events     — like solve_ode_dense + EVENT DETECTION: the moments
                         at which g(t, y) = 0, located by bisection on the
                         interpolant (without re-integrating);
                         terminal=True STOPS integration at the first
                         event (nodes end at t_event).
  solve_ode_stiff      — IMPLICIT solver (BDF2 + Newton per step, symbolic
                         Jacobian) for STIFF equations, where explicit methods
                         are throttled by stability, not accuracy.
  solve_ode_system_stiff — vector counterpart of solve_ode_stiff (module 16):
                         BDF2 for stiff SYSTEMS, multidimensional Newton
                         per step (matrix I − coef·J, J = ∂F/∂y
                         symbolically, np.linalg.solve — the
                         root_find_nd pattern).
  solve_ode_stiff_adaptive / solve_ode_system_stiff_adaptive — BDF with a
                         VARIABLE step (module 17): variable-step BDF2
                         (coefficients depending on the step ratio ω),
                         local error estimate from the third divided
                         difference, step controller as in DOPRI5 — few
                         steps where an explicit method is throttled by
                         stability.
  solve_ode_system_num — system y' = F(t, y) (vectorized, RK4).
  solve_ode_system_dense / solve_ode_system_events — vector counterparts of
                         dense/events for systems (per-coordinate interpolant,
                         event g(t, y1..yn) = 0, also terminal).

Every failure raises a ``PycodemathError`` subclass with a readable message
(consistent with ``numerics``): a bad domain or a singular step matrix gives
``DomainError``, a NaN/inf solution ``DivergenceError``, a step shrinking below
``hmin`` ``StagnationError``, an exhausted step count ``NonConvergenceError``;
"no closed form" from ``dsolve`` is ``NoClosedFormError`` (module 10 — it used to
be a bare ``PycodemathError``, which said nothing an agent could branch on).

All numerical solvers also accept a DECREASING interval (t1 < t0)
— BACKWARD integration, step h < 0 (module 13). Beyond scope: PDEs.
"""

from __future__ import annotations

from typing import Callable, Sequence, cast

import numpy as np
import sympy as sp

from ..core.errors import (
    DivergenceError,
    DomainError,
    NoClosedFormError,
    NonConvergenceError,
    StagnationError,
)
from ..core.ir import Expr
from .numerics import _check_vars, _point, _vector_fn


def dsolve(expr: Expr, func: str = "y", var: str = "t") -> list[Expr]:
    """Solve the ODE ``func'(var) = expr`` symbolically (sympy.dsolve).

    ``expr`` is the IR ``Expr`` of the right-hand side, with the unknown as an
    ordinary symbol ``func``. Additional symbols (parameters, e.g. ``a`` in
    ``a*y``) are allowed — this is the symbolic path. Returns a list of ``Expr``
    — the right-hand sides of EXPLICIT solutions (with constants ``C1``… as
    symbols), consistent with ``symbolic.solve``.

    No closed form or an implicit solution -> ``NoClosedFormError`` with a hint to
    use ``solve_ode_num`` and ``route="ode"`` (module 10). Both refusals below are
    that class rather than ``UnsupportedFormError``, and the reason is how
    ``sp.dsolve`` works: it CLASSIFIES the equation first and raises
    ``NotImplementedError`` only once its classification has found nothing that
    fits. That is a search that came back empty, not a refusal to start one — the
    line ``UnsupportedFormError`` draws in ``engine.symbolic``.
    """
    rhs = Expr(expr)
    if func == var:
        raise DomainError(
            "dsolve: function name and independent variable name must differ"
        )
    t = sp.Symbol(var)
    Y = sp.Function(func)
    equation = sp.Eq(Y(t).diff(t), rhs.sy.subs(sp.Symbol(func), Y(t)))

    def refuse() -> NoClosedFormError:
        return NoClosedFormError(
            f"dsolve: cannot find a closed-form solution for "
            f"{func}'({var}) = {rhs} — use solve_ode_num",
            route="ode",
        )

    try:
        raw = sp.dsolve(equation, Y(t))
    except (NotImplementedError, ValueError, TypeError) as exc:
        raise refuse() from exc

    solutions = raw if isinstance(raw, list) else [raw]
    out: list[Expr] = []
    for sol in solutions:
        # explicitness: y(t) = <expression without y(t)> — an implicit solution
        # does not fit the IR Expr contract (an algebraic expression)
        if sol.lhs != Y(t) or sol.rhs.has(sp.core.function.AppliedUndef):
            raise refuse()
        out.append(Expr(sol.rhs))
    return out


# --- numerical core (shared by scalar and system) ------------------------
def _span(t_span: "Sequence[float]", what: str) -> tuple[float, float]:
    """Validate the integration interval; return ``(t0, t1)`` (requires ``t1 ≠ t0``).

    A DECREASING interval (``t1 < t0``) is allowed — the numerical solvers then
    integrate BACKWARD (step ``h < 0``, module 13).
    """
    try:
        t0, t1 = (float(v) for v in t_span)
    except (TypeError, ValueError) as exc:
        raise DomainError(f"{what}: t_span must be a pair of numbers (t0, t1)") from exc
    if t1 == t0:
        raise DomainError(
            f"{what}: interval cannot have zero length "
            f"(given {t0:g} .. {t1:g})"
        )
    return t0, t1


def _tols(rtol: float, atol: float, what: str) -> tuple[float, float]:
    """Validate the adaptive solver tolerances; return ``(rtol, atol)``."""
    if not (rtol > 0.0 and atol > 0.0):
        raise DomainError(f"{what}: rtol and atol must be positive")
    return float(rtol), float(atol)


def _system_inputs(
    rhs: "Sequence[Expr]",
    var: str,
    funcs: "Sequence[str]",
    y0: "Sequence[float]",
    what: str,
) -> tuple[list[Expr], list[str], np.ndarray]:
    """Validate the system input ``y' = F(var, y)``; return ``(fs, funcs, y)``.

    The shared gate of all system solvers: a square system (as many equations
    as functions), expressions depending at most on ``var`` and ``funcs``,
    a start point of the correct length.
    """
    fs = [Expr(e) for e in rhs]
    funcs = list(funcs)
    if not fs or len(fs) != len(funcs):
        raise DomainError(
            f"{what} requires a square system: "
            f"{len(fs)} equations, {len(funcs)} functions"
        )
    _check_vars(fs, [var, *funcs], what)
    y = _point(y0, len(funcs), what)
    return fs, funcs, y


def _grid(t_span: "Sequence[float]", n: int, what: str) -> tuple[float, float, float]:
    """Validate the interval and the number of steps; return ``(t0, t1, h)``."""
    if int(n) < 1:
        raise DomainError(f"{what}: number of steps n must be positive")
    t0, t1 = _span(t_span, what)
    return t0, t1, (t1 - t0) / int(n)


def _rk4(
    sample: "Callable[[np.ndarray], np.ndarray | None]",
    t0: float,
    t1: float,
    n: int,
    y: np.ndarray,
    what: str,
) -> tuple[list[float], list[list[float]]]:
    """Classic fixed-step RK4 over a COMPILED right-hand side.

    ``sample`` is the closure from ``_vector_fn`` (compilation before the loop),
    called on the vector ``[t, *y]``. Returns ``(ts, rows)``: ``n + 1`` nodes
    including ``t0``, ``rows[k]`` = state at time ``ts[k]``.
    """
    h = (t1 - t0) / n
    ts = [t0]
    rows = [[float(v) for v in y]]
    t = t0
    for i in range(1, n + 1):
        k1 = sample(np.concatenate(([t], y)))
        k2 = None if k1 is None else sample(np.concatenate(([t + h / 2], y + h / 2 * k1)))
        k3 = None if k2 is None else sample(np.concatenate(([t + h / 2], y + h / 2 * k2)))
        k4 = None if k3 is None else sample(np.concatenate(([t + h], y + h * k3)))
        if k1 is None or k2 is None or k3 is None or k4 is None:
            raise DomainError(
                f"{what}: cannot sample the right-hand side at t={t:g} "
                f"(outside the domain, complex result, or divergence)"
            )
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t0 + i * h  # node from multiplication, not summation — no float drift
        if not np.all(np.isfinite(y)):
            raise DivergenceError(
                f"{what}: solution diverges at t={t:g} (NaN/inf) — "
                f"narrow the interval or increase n"
            )
        ts.append(t)
        rows.append([float(v) for v in y])
    return ts, rows


def solve_ode_num(
    rhs: Expr,
    var: str,
    y0: float,
    t_span: "Sequence[float]",
    n: int = 100,
    func: str = "y",
) -> tuple[list[float], list[float]]:
    """Initial value problem ``y'(var) = f(var, y)``, ``y(t0) = y0`` — RK4.

    ``rhs`` depends at most on ``var`` (time) and ``func`` (the unknown).
    Fixed step ``h = (t1 - t0)/n``; returns samples ``(ts, ys)`` — lists
    of length ``n + 1`` with nodes including ``t0``.
    """
    r = Expr(rhs)
    _check_vars([r], [var, func], "solve_ode_num")  # duplicates also catch func == var
    t0, t1, _ = _grid(t_span, n, "solve_ode_num")
    sample = _vector_fn([r], [var, func])  # compiled once — before the loop
    ts, rows = _rk4(
        sample, t0, t1, int(n), np.array([float(y0)], dtype=float), "solve_ode_num"
    )
    return ts, [row[0] for row in rows]


# --- adaptive step (Dormand-Prince 5(4), "ode45") ------------------------
# Explicit Runge-Kutta pair 5(4): one set of right-hand-side evaluations gives
# a 5th-order solution AND a built-in 4th-order estimate — their difference
# drives the step selection. Coefficients after Dormand and Prince (1980), the
# same as in MATLAB's ode45 / scipy RK45.
_DP_C = (1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)  # nodes c2..c7 (c1 = 0)
_DP_A: tuple[tuple[float, ...], ...] = (
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_DP_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_DP_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)
#: difference b5 - b4 — multiplied by h*k gives the local error estimate (order 4)
_DP_E = tuple(b5 - b4 for b5, b4 in zip(_DP_B5, _DP_B4))

#: DOPRI5 DENSE OUTPUT coefficients (continuous 4th-order extension, the same
#: as scipy RK45): for θ = (t - t_n)/h in [0,1] the interpolant is
#: y(t_n + θh) = y_n + h · Σ_i k_i · Σ_j P[i,j] θ^(j+1). At the nodes θ=0 -> y_n,
#: θ=1 -> y_{n+1} (Σ_j P[i,j] = b5_i), so it stitches the samples smoothly.
_DP_P = np.array(
    [
        [1.0, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
        [0.0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
        [0.0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
        [0.0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0.0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
    ]
)


def _dp_step(
    sample: "Callable[[np.ndarray], np.ndarray | None]",
    t: float,
    y: np.ndarray,
    h: float,
    k1: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None":
    """One DOPRI5 step; return ``(y5, err, ks)`` or ``None`` (outside the domain).

    ``k1`` (= ``f(t, y)``) is passed in from outside — thanks to FSAL the next
    step reuses the last stage of the previous one instead of recomputing it.
    ``y5`` is the 5th-order state, ``err`` = ``h·Σ(b5-b4)·k`` — the error
    estimate; ``ks`` (7 stages) come back because they are the basis of the
    dense output.
    """
    ks: list[np.ndarray] = [k1]
    for ci, ai in zip(_DP_C, _DP_A):  # stages 2..7
        yi = y + h * sum(ai[j] * ks[j] for j in range(len(ai)))
        ki = sample(np.concatenate(([t + ci * h], yi)))
        if ki is None:
            return None
        ks.append(ki)
    y5 = y + h * sum(_DP_B5[i] * ks[i] for i in range(7))
    err = h * sum(_DP_E[i] * ks[i] for i in range(7))
    return y5, cast(np.ndarray, err), ks


def _dopri45(
    sample: "Callable[[np.ndarray], np.ndarray | None]",
    t0: float,
    t1: float,
    y: np.ndarray,
    rtol: float,
    atol: float,
    max_steps: int,
    what: str,
    segments: "list | None" = None,
    stop: "Callable | None" = None,
) -> tuple[list[float], list[list[float]]]:
    """Variable-step integration with the DOPRI5 pair over a COMPILED right-hand side.

    A step is accepted when the normalized error norm (weight ``atol + rtol·|y|``)
    does not exceed 1; then it is enlarged, and on rejection it is reduced and the
    stage is repeated. The first stage of the next step reuses the last one of the
    previous step (FSAL) — one fewer right-hand-side evaluation per step. Near a
    singularity the step shrinks until below the ``hmin`` limit -> ``PycodemathError``
    (integration REFUSES to jump, it does not return garbage). Returns ``(ts, rows)``
    at the chosen nodes, including ``t0`` and ``t1``. A DECREASING interval
    (``t1 < t0``) integrates BACKWARD — step ``h < 0``, decreasing nodes (module 13).

    When ``segments`` (a list) is given — on each accepted step it appends a tuple
    ``(t_old, h, y_old, ks)`` for reconstructing the dense output (interpolation).

    When ``stop`` is given — after each ACCEPTED step it calls
    ``stop(seg, t_new)``; a result ``(t_stop, y_stop)`` STOPS integration:
    the last node is replaced by the stop point (a terminal event
    — nodes end at ``t_stop``, not at ``t1``), and ``None`` continues.
    """
    hmin = 1e-12 * max(1.0, abs(t0), abs(t1))
    sgn = 1.0 if t1 > t0 else -1.0  # integration direction (backward: h < 0)
    h = (t1 - t0) / 100.0  # modest starting step (carries the sign) — the controller spins it up
    t = t0
    ts = [t0]
    rows = [[float(v) for v in y]]
    steps = 0
    k1 = sample(np.concatenate(([t], y)))  # first stage (FSAL reuses the next ones)
    if k1 is None:
        raise DomainError(
            f"{what}: cannot sample the right-hand side at the start point "
            f"t={t:g} (outside the domain or complex result)"
        )
    while (t1 - t) * sgn > 0:
        if steps >= max_steps:
            raise NonConvergenceError(
                f"{what}: exceeded {max_steps} steps at t={t:g} — "
                f"the solution is probably singular (narrow the interval)"
            )
        last = (t + h - t1) * sgn >= 0
        if last:
            h = t1 - t  # the last step lands exactly on t1
        stepped = _dp_step(sample, t, y, h, k1)
        if stepped is None:  # right-hand side outside the domain — try shorter
            h *= 0.5  # k1 still valid: t, y unchanged (sign preserved)
            if abs(h) < hmin:
                raise DomainError(
                    f"{what}: cannot sample the right-hand side at t={t:g} "
                    f"(singularity, complex result, or outside the domain)"
                )
            continue
        y5, err, ks = stepped
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        e = float(np.sqrt(np.mean((err / sc) ** 2)))  # RMS of the normalized error
        if e <= 1.0:  # --- step accepted ---
            seg = (t, h, y, ks)  # before the update: t_old, y_old
            if segments is not None:
                segments.append(seg)
            # explicit t1 on the last step: t + (t1 - t) need not, in IEEE, hit
            # t1 exactly, and a remainder of order ulp would make a duplicated mini-step
            t = t1 if last else t + h
            y = y5
            k1 = ks[6]  # FSAL: ks[6] = f(t+h, y5) = k1 of the next step
            ts.append(t)
            rows.append([float(v) for v in y])
            steps += 1
            if stop is not None:
                hit = stop(seg, t)
                if hit is not None:  # terminal event inside the step
                    t_stop, y_stop = hit
                    ts[-1] = float(t_stop)  # nodes end AT the event
                    rows[-1] = [float(v) for v in y_stop]
                    break
            # safety factor 0.9, exponent 1/5 (order-4 estimate); growth <= 5x
            grow = 5.0 if e == 0.0 else 0.9 * e ** -0.2
            h *= min(5.0, grow)
        else:  # --- step rejected: reduce and repeat ---
            h *= max(0.2, 0.9 * e ** -0.2)  # k1 unchanged; positive multiplier — sign preserved
            if abs(h) < hmin:
                raise StagnationError(
                    f"{what}: step shrank below the limit at t={t:g} — "
                    f"singularity (integration refuses to jump over it)"
                )
    return ts, rows


def solve_ode_adaptive(
    rhs: Expr,
    var: str,
    y0: float,
    t_span: "Sequence[float]",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
    func: str = "y",
) -> tuple[list[float], list[float]]:
    """Initial value problem ``y'(var) = f(var, y)`` by the Dormand-Prince 5(4) method.

    A variant of ``solve_ode_num`` with a VARIABLE step: the RK 5(4) pair
    estimates the local error at each step and chooses its length so as to
    stay within the tolerance ``atol + rtol·|y|``. On a smooth problem it
    reaches the requested accuracy with FEWER steps than fixed-step RK4;
    near a singularity the step densifies, and when it drops below the limit —
    ``PycodemathError`` (instead of a silent jump across a pole).

    ``rhs`` depends at most on ``var`` and ``func``. Returns samples
    ``(ts, ys)`` at adaptively chosen nodes, including ``t0`` and ``t1``
    (the number of nodes depends on the course of the solution).
    """
    r = Expr(rhs)
    _check_vars([r], [var, func], "solve_ode_adaptive")  # duplicates catch func == var
    t0, t1 = _span(t_span, "solve_ode_adaptive")
    rtol, atol = _tols(rtol, atol, "solve_ode_adaptive")
    sample = _vector_fn([r], [var, func])  # compiled once — before the loop
    ts, rows = _dopri45(
        sample, t0, t1, np.array([float(y0)], dtype=float),
        rtol, atol, int(max_steps), "solve_ode_adaptive",
    )
    return ts, [row[0] for row in rows]


def _seg_eval(seg: tuple, tq: float) -> np.ndarray:
    """Read the state ``y(tq)`` from the interpolant of ONE step ``(t_old, h, y_old, ks)``.

    The 4th-order continuous extension of DOPRI5: for ``θ = (tq - t_old)/h``
    ``y = y_old + h · Σ_i k_i · (P[i] · [θ, θ², θ³, θ⁴])``. At ``θ=0`` it returns
    ``y_old``, at ``θ=1`` exactly ``y_new`` (Σ_j P[i,j] = b5_i) — the shared
    core of dense output and terminal event localization.
    """
    t_old, h, y_old, ks = seg
    theta = (tq - t_old) / h
    powers = np.array([theta, theta**2, theta**3, theta**4])
    acc = sum(ks[i] * float(_DP_P[i] @ powers) for i in range(7))
    return y_old + h * acc


class DenseSolution:
    """DOPRI5 dense output — callable ``sol(t)`` reading ``y`` at EVERY ``t``.

    ``solve_ode_adaptive`` gives ``y`` only at the nodes; here each accepted
    step carries its own 4th-order interpolant (the continuous extension of
    DOPRI5 on the stages ``k``), so the solution is read at any point of the
    interval without re-integrating. The sample nodes remain available as
    ``.ts``/``.ys``.

    ``sol(t)`` accepts a number (returns ``float``) or a sequence/array
    (returns ``np.ndarray``); ``t`` outside ``[t0, t1]`` -> ``PycodemathError``.
    """

    def __init__(self, ts: list[float], rows: list[list[float]], segments: list):
        self.ts = ts
        self.ys: list = [row[0] for row in rows]  # scalar y at nodes (system subclasses: state lists)
        self.t0 = ts[0]
        self.t1 = ts[-1]
        self._segments = segments  # (t_old, h, y_old, ks) per step
        # integration direction: for a BACKWARD run (t1 < t0) the nodes decrease;
        # multiplying by the sign brings the step t_old values to increasing order,
        # so searchsorted works in both directions
        self._sgn = 1.0 if self.t1 >= self.t0 else -1.0
        self._starts = self._sgn * np.array([seg[0] for seg in segments])

    def _eval_vec(self, tq: float) -> np.ndarray:
        """Read the FULL state vector at ``tq`` (shared core of scalar and system)."""
        lo, hi = sorted((self.t0, self.t1))
        # boundary tolerance: allow a small float overshoot at the ends
        eps = 1e-9 * max(1.0, abs(lo), abs(hi))
        if tq < lo - eps or tq > hi + eps:
            raise DomainError(
                f"DenseSolution: t={tq:g} outside the interval "
                f"[{self.t0:g}, {self.t1:g}]"
            )
        # step containing tq: the last t_old <= tq ALONG the integration direction
        idx = int(np.searchsorted(self._starts, self._sgn * tq, side="right") - 1)
        idx = min(max(idx, 0), len(self._segments) - 1)
        return _seg_eval(self._segments[idx], tq)

    def _eval_one(self, tq: float) -> float:
        return float(self._eval_vec(tq)[0])

    def __call__(self, t):
        """Read ``y(t)`` — a scalar for a number, ``np.ndarray`` for a sequence."""
        if np.ndim(t) == 0:
            return self._eval_one(float(t))
        return np.array([self._eval_one(float(v)) for v in t])


def solve_ode_dense(
    rhs: Expr,
    var: str,
    y0: float,
    t_span: "Sequence[float]",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
    func: str = "y",
) -> DenseSolution:
    """Like ``solve_ode_adaptive``, but returns DENSE OUTPUT ``DenseSolution``.

    The engine and step control are identical (the DOPRI5 pair, tolerance
    ``atol + rtol·|y|``) — the nodes ``.ts``/``.ys`` are the same as in
    ``solve_ode_adaptive``. The difference: the object is CALLABLE and reads
    ``y(t)`` at any point of ``[t0, t1]`` via a 4th-order interpolant (not only
    at nodes). Singularity / bad input -> ``PycodemathError`` as above.
    """
    r = Expr(rhs)
    _check_vars([r], [var, func], "solve_ode_dense")  # duplicates catch func == var
    t0, t1 = _span(t_span, "solve_ode_dense")
    rtol, atol = _tols(rtol, atol, "solve_ode_dense")
    sample = _vector_fn([r], [var, func])  # compiled once — before the loop
    segments: list = []
    ts, rows = _dopri45(
        sample, t0, t1, np.array([float(y0)], dtype=float),
        rtol, atol, int(max_steps), "solve_ode_dense", segments,
    )
    return DenseSolution(ts, rows, segments)


# --- event detection (module 9) ------------------------------------------
def _bisect_event(
    G: "Callable[[float], float]",
    a: float,
    ga: float,
    b: float,
    gb: float,
    xtol: float = 1e-13,
    max_iter: int = 100,
) -> float:
    """Locate a zero of ``G`` in ``[a, b]`` (known sign-changing ``ga``, ``gb``).

    Bisection on the SCALAR function ``G(t) = g(t, sol(t))`` — sol(t) is a ready
    interpolant, so localization does NOT require re-integrating. Returns the
    ``t`` with the zero to tolerance ``xtol`` (relative on the scale of ``|t|``).
    The order of the ends is arbitrary (``a > b`` for backward integration) —
    the width is computed via ``abs``.
    """
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        gm = G(m)
        if gm == 0.0 or 0.5 * abs(b - a) <= xtol * max(1.0, abs(m)):
            return m
        if (ga < 0.0) != (gm < 0.0):  # sign change in [a, m]
            b, gb = m, gm
        else:
            a, ga = m, gm
    return 0.5 * (a + b)


def _find_events(
    sol: "DenseSolution",
    g_fn: "Callable[[np.ndarray], np.ndarray | None]",
    direction: int,
    what: str,
) -> list[tuple[float, np.ndarray]]:
    """Scan the nodes of ``sol`` and find zeros of ``g(t, y)`` (sign changes).

    On each segment [t_k, t_{k+1}] we check the sign of ``G`` at the nodes; at a
    strict sign change we locate the zero by bisection on the interpolant.
    ``direction``: +1 only rising (−→+), −1 only falling (+→−),
    0 any — the transition direction is measured ALONG the integration run
    (for backward integration the nodes go down the t axis). Returns a list of
    ``(t_event, state_vector)`` in time order — the caller flattens the state to
    a scalar or a list (system).
    """
    def G(t: float) -> float:
        vec = sol._eval_vec(t)
        val = g_fn(np.concatenate(([float(t)], vec)))
        if val is None:
            raise DomainError(
                f"{what}: event function outside the domain at t={t:g} "
                f"(complex result or NaN/inf)"
            )
        return float(val[0])

    events: list[tuple[float, np.ndarray]] = []
    ts = sol.ts
    g_prev = G(ts[0])
    for k in range(1, len(ts)):
        t_a, t_b = ts[k - 1], ts[k]
        g_b = G(t_b)
        if g_prev < 0.0 < g_b or g_prev > 0.0 > g_b:  # strict sign change
            cross = 1 if g_b > g_prev else -1  # rising / falling
            if direction == 0 or direction == cross:
                t_ev = _bisect_event(G, t_a, g_prev, t_b, g_b)
                events.append((t_ev, sol._eval_vec(t_ev)))
        g_prev = g_b
    return events


class EventSolution(DenseSolution):
    """``DenseSolution`` enriched with detected events ``g(t, y) = 0``.

    Inherits the full dense-output interface (still CALLABLE ``sol(t)``,
    ``.ts``/``.ys``) and adds a list ``.events`` = ``[(t, y), …]`` at the
    moments the event function crosses zero, plus separated ``.event_times``
    and ``.event_values`` for convenient use in NumPy.
    """

    def __init__(
        self,
        ts: list[float],
        rows: list[list[float]],
        segments: list,
        events: list[tuple[float, float]],
    ):
        super().__init__(ts, rows, segments)
        self.events = events
        self.event_times = [t for t, _ in events]
        self.event_values = [y for _, y in events]


def _terminal_stop(
    g_fn: "Callable[[np.ndarray], np.ndarray | None]",
    direction: int,
    what: str,
) -> Callable:
    """Build the ``stop`` callback for ``_dopri45`` — a TERMINAL event.

    After each accepted step it computes ``G`` at its ends (``G(t) =
    g(t, y(t))`` on the interpolant of THIS step); at a sign change consistent
    with ``direction`` it locates the zero by bisection and returns ``(t_ev, y_ev)`` —
    the stop signal. A hit is stored in ``stop.hit``. The sign from the previous
    step is carried by the closure — G computed ONCE per node, as in ``_find_events``.
    """
    state: dict = {"g_prev": None}

    def stop(seg: tuple, t_new: float) -> "tuple[float, np.ndarray] | None":
        def G(tq: float) -> float:
            val = g_fn(np.concatenate(([tq], _seg_eval(seg, tq))))
            if val is None:
                raise DomainError(
                    f"{what}: event function outside the domain at t={tq:g} "
                    f"(complex result or NaN/inf)"
                )
            return float(val[0])

        t_old = seg[0]
        if state["g_prev"] is None:  # first step: sign at the start point
            state["g_prev"] = G(t_old)
        g_a = state["g_prev"]
        g_b = G(t_new)
        state["g_prev"] = g_b
        if g_a < 0.0 < g_b or g_a > 0.0 > g_b:  # strict sign change
            cross = 1 if g_b > g_a else -1
            if direction == 0 or direction == cross:
                t_ev = _bisect_event(G, t_old, g_a, t_new, g_b)
                y_ev = _seg_eval(seg, t_ev)
                stop.hit = (t_ev, y_ev)  # type: ignore[attr-defined]  # result channel: attribute on the closure
                return t_ev, y_ev
        return None

    stop.hit = None  # type: ignore[attr-defined]  # result channel: attribute on the closure
    return stop


def solve_ode_events(
    rhs: Expr,
    var: str,
    y0: float,
    t_span: "Sequence[float]",
    event: Expr,
    direction: int = 0,
    terminal: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
    func: str = "y",
) -> EventSolution:
    """Integrate ``y' = f(var, y)`` and detect zeros of the event function ``g(var, y)``.

    The engine and step control are as in ``solve_ode_dense`` (the DOPRI5 pair,
    tolerance ``atol + rtol·|y|``); after integration we scan the nodes and where
    ``g`` changes sign we locate the zero by BISECTION on the interpolant —
    without re-integrating. ``event`` is an IR ``Expr`` depending at most on
    ``var`` and ``func``. ``direction``: +1 only rising transitions,
    −1 only falling, 0 (default) any.

    ``terminal=True`` STOPS integration at the FIRST event (consistent
    with ``direction``): events are checked ON THE FLY after each step,
    the nodes and ``sol(t)`` end exactly at ``t_event`` (~rtol), not
    at ``t1`` — so a singularity lying BEYOND the event does not interrupt integration.
    ``.events`` then has at most one entry; no transition -> the full
    interval and an empty list (as before).

    Returns ``EventSolution`` — still callable ``sol(t)`` (dense output),
    with a list ``.events`` = ``[(t, y), …]``. Singularity / bad input ->
    ``PycodemathError``.
    """
    r = Expr(rhs)
    g = Expr(event)
    _check_vars([r], [var, func], "solve_ode_events")  # duplicates catch func == var
    _check_vars([g], [var, func], "solve_ode_events (event function)")
    if direction not in (-1, 0, 1):
        raise DomainError("solve_ode_events: direction must be -1, 0, or +1")
    t0, t1 = _span(t_span, "solve_ode_events")
    rtol, atol = _tols(rtol, atol, "solve_ode_events")
    sample = _vector_fn([r], [var, func])  # compiled once — before the loop
    g_fn = _vector_fn([g], [var, func])  # event function — compiled once
    segments: list = []

    if terminal:  # event checked DURING integration (stop on hit)
        stop = _terminal_stop(g_fn, int(direction), "solve_ode_events")
        ts, rows = _dopri45(
            sample, t0, t1, np.array([float(y0)], dtype=float),
            rtol, atol, int(max_steps), "solve_ode_events",
            segments, stop,
        )
        events = []
        if stop.hit is not None:  # type: ignore[attr-defined]
            t_ev, y_ev = stop.hit  # type: ignore[attr-defined]
            events = [(float(t_ev), float(y_ev[0]))]
        return EventSolution(ts, rows, segments, events)

    ts, rows = _dopri45(
        sample, t0, t1, np.array([float(y0)], dtype=float),
        rtol, atol, int(max_steps), "solve_ode_events", segments,
    )
    sol = DenseSolution(ts, rows, segments)  # interpolant for locating zeros
    raw = _find_events(sol, g_fn, int(direction), "solve_ode_events")
    events = [(t, float(vec[0])) for t, vec in raw]  # scalar y of the event
    return EventSolution(ts, rows, segments, events)


def solve_ode_system_num(
    rhs: "Sequence[Expr]",
    var: str,
    funcs: "Sequence[str]",
    y0: "Sequence[float]",
    t_span: "Sequence[float]",
    n: int = 100,
) -> tuple[list[float], list[list[float]]]:
    """System ``y' = F(var, y)`` vectorized — fixed-step RK4.

    ``rhs[i]`` is the right-hand side of the equation for ``funcs[i]``; each
    expression depends at most on ``var`` and names from ``funcs``. Returns
    ``(ts, rows)``, where ``rows[k][i]`` = value of ``funcs[i]`` at time
    ``ts[k]`` (state-row per sample — a natural ``zip(ts, rows)``).
    """
    fs, funcs, y = _system_inputs(rhs, var, funcs, y0, "solve_ode_system_num")
    t0, t1, _ = _grid(t_span, n, "solve_ode_system_num")
    sample = _vector_fn(fs, [var, *funcs])
    return _rk4(sample, t0, t1, int(n), y, "solve_ode_system_num")


# --- stiff methods (module 12) --------------------------------------------
def solve_ode_stiff(
    rhs: Expr,
    var: str,
    y0: float,
    t_span: "Sequence[float]",
    n: int = 100,
    func: str = "y",
) -> tuple[list[float], list[float]]:
    """Initial value problem ``y' = f(var, y)`` by an IMPLICIT method (BDF2).

    For STIFF equations — where the damping time scale is orders of magnitude
    shorter than the solution scale (e.g. ``y' = -1000·(y - cos(t))``) — explicit
    methods (RK4, DOPRI5) are limited by STABILITY, not accuracy: the step must
    be tiny, because a larger one explodes. Implicit methods are A-stable:
    the step is matched to the solution, not to the stiffness.

    BDF2 (multistep, order 2): ``y_{k+1} = (4·y_k - y_{k-1})/3
    + (2h/3)·f(t_{k+1}, y_{k+1})``; the implicit Euler starts the first step.
    The step equation is solved by NEWTON with the derivative ``∂f/∂y`` computed
    SYMBOLICALLY (``Expr.diff``, the ``root_find_nd`` pattern) and compiled
    ONCE before the loop. Fixed step ``h = (t1-t0)/n``; returns ``(ts, ys)``
    like ``solve_ode_num``. Divergence / Newton non-convergence ->
    ``PycodemathError``.
    """
    r = Expr(rhs)
    _check_vars([r], [var, func], "solve_ode_stiff")  # duplicates catch func == var
    t0, t1, h = _grid(t_span, n, "solve_ode_stiff")
    n = int(n)
    f_fn = _vector_fn([r], [var, func])  # compiled once — before the loop
    df_fn = _vector_fn([r.diff(func)], [var, func])  # ∂f/∂y symbolically

    def _sample1(fn, t: float, u: float) -> "float | None":
        val = fn(np.array([t, u]))
        return None if val is None else float(val[0])

    def _implicit_solve(t_new: float, const: float, coef: float, u0: float) -> float:
        """Solve ``u = const + coef·f(t_new, u)`` by Newton (start ``u0``)."""
        u = u0
        for _ in range(50):
            fu = _sample1(f_fn, t_new, u)
            if fu is None:
                raise DomainError(
                    f"solve_ode_stiff: cannot sample the right-hand side at "
                    f"t={t_new:g} (outside the domain or complex result)"
                )
            res = u - coef * fu - const
            if abs(res) <= 1e-12 * max(1.0, abs(u)):
                return u
            dfu = _sample1(df_fn, t_new, u)
            slope = None if dfu is None else 1.0 - coef * dfu
            if slope is None or slope == 0.0:
                raise DomainError(
                    f"solve_ode_stiff: Newton on the implicit step cannot proceed at "
                    f"t={t_new:g} (derivative outside the domain or singular)"
                )
            u = u - res / slope
            if not np.isfinite(u):
                raise DivergenceError(
                    f"solve_ode_stiff: solution diverges at "
                    f"t={t_new:g} (NaN/inf) — narrow the interval or increase n"
                )
        raise NonConvergenceError(
            f"solve_ode_stiff: Newton does not converge at t={t_new:g} — increase n"
        )

    ts = [t0]
    ys = [float(y0)]
    # start-up: BDF2 needs two points — the first step is implicit
    # Euler (u = y_k + h·f(t_new, u)); Newton starts from the previous value
    y1 = _implicit_solve(t0 + h, ys[0], h, ys[0])
    ts.append(t0 + h)
    ys.append(y1)
    for i in range(2, n + 1):
        t_new = t0 + i * h
        const = (4.0 * ys[-1] - ys[-2]) / 3.0
        u = _implicit_solve(t_new, const, 2.0 * h / 3.0, ys[-1])
        ts.append(t_new)
        ys.append(u)
    return ts, ys


# --- stiff systems (module 16) ---------------------------------------------
def solve_ode_system_stiff(
    rhs: "Sequence[Expr]",
    var: str,
    funcs: "Sequence[str]",
    y0: "Sequence[float]",
    t_span: "Sequence[float]",
    n: int = 100,
) -> tuple[list[float], list[list[float]]]:
    """STIFF system ``y' = F(var, y)`` by an IMPLICIT method (BDF2) — vectorized.

    The vector counterpart of ``solve_ode_stiff``: the same BDF2 with an
    implicit Euler start-up, but the step equation ``u = const + coef·F(t_new, u)``
    is solved by a MULTIDIMENSIONAL NEWTON (the ``root_find_nd`` pattern): residual
    ``R(u) = u − coef·F(t_new, u) − const``, step matrix ``I − coef·J``,
    where ``J = ∂F/∂y`` computed SYMBOLICALLY (``Expr.diff``, flattened
    row by row) and compiled ONCE before the loop; the linear system via
    ``np.linalg.solve``. Iteration start: the previous state ``y_k`` (as in the
    scalar version — an explicit predictor for stiff problems flies off into space).

    ``rhs[i]`` is the right-hand side of the equation for ``funcs[i]``; the
    expressions depend at most on ``var`` and ``funcs``. Fixed step ``h = (t1-t0)/n``;
    a DECREASING interval (t1 < t0) integrates BACKWARD (module 13). Returns
    ``(ts, rows)`` like ``solve_ode_system_num``. Singular step matrix /
    Newton non-convergence / divergence -> ``PycodemathError``.
    """
    fs, funcs, y = _system_inputs(rhs, var, funcs, y0, "solve_ode_system_stiff")
    t0, t1, h = _grid(t_span, n, "solve_ode_system_stiff")
    n = int(n)
    m = len(funcs)
    F_fn = _vector_fn(fs, [var, *funcs])  # compiled once — before the loop
    # Jacobian ∂F/∂y symbolically, flattened row by row (root_find_nd)
    J_fn = _vector_fn([f.diff(g) for f in fs for g in funcs], [var, *funcs])
    eye = np.eye(m)

    def _implicit_solve_nd(
        t_new: float, const: np.ndarray, coef: float, u0: np.ndarray
    ) -> np.ndarray:
        """Solve ``u = const + coef·F(t_new, u)`` by nd Newton (start ``u0``)."""
        u = u0
        for _ in range(50):
            Fu = F_fn(np.concatenate(([t_new], u)))
            if Fu is None:
                raise DomainError(
                    f"solve_ode_system_stiff: cannot sample the right-hand side "
                    f"at t={t_new:g} (outside the domain or complex result)"
                )
            res = u - coef * Fu - const
            if float(np.max(np.abs(res))) <= 1e-12 * max(
                1.0, float(np.max(np.abs(u)))
            ):
                return u
            Ju = J_fn(np.concatenate(([t_new], u)))
            if Ju is None:
                raise DomainError(
                    f"solve_ode_system_stiff: cannot sample the Jacobian "
                    f"at t={t_new:g} (outside the domain or complex result)"
                )
            try:
                delta = np.linalg.solve(eye - coef * Ju.reshape(m, m), -res)
            except np.linalg.LinAlgError as exc:
                raise DomainError(
                    f"solve_ode_system_stiff: implicit step matrix is singular "
                    f"at t={t_new:g} — Newton cannot proceed (increase n)"
                ) from exc
            u = u + delta
            if not np.all(np.isfinite(u)):
                raise DivergenceError(
                    f"solve_ode_system_stiff: solution diverges at "
                    f"t={t_new:g} (NaN/inf) — narrow the interval or increase n"
                )
        raise NonConvergenceError(
            f"solve_ode_system_stiff: Newton does not converge at t={t_new:g} — increase n"
        )

    ts = [t0]
    rows = [[float(v) for v in y]]
    # start-up: implicit Euler (u = y_0 + h·F(t_1, u)) — as in the scalar version
    y1 = _implicit_solve_nd(t0 + h, y, h, y)
    ts.append(t0 + h)
    rows.append([float(v) for v in y1])
    prev, curr = y, y1
    for i in range(2, n + 1):
        t_new = t0 + i * h
        const = (4.0 * curr - prev) / 3.0
        u = _implicit_solve_nd(t_new, const, 2.0 * h / 3.0, curr)
        ts.append(t_new)
        rows.append([float(v) for v in u])
        prev, curr = curr, u
    return ts, rows


# --- adaptive step for BDF (module 17) -------------------------------------
# Variable-step BDF2: the derivative of the quadratic interpolant through
# (t_{n-1}, y_{n-1}), (t_n, y_n), (t_{n+1}, u) set equal to f(t_{n+1}, u).
# For the step ratio ω = h_new/h_old the step equation has the form
#   u = a1·y_n + a2·y_{n-1} + coef·f(t_{n+1}, u),
#   a1 = (1+ω)²/(1+2ω),  a2 = −ω²/(1+2ω),  coef = h_new·(1+ω)/(1+2ω)
# (ω = 1 reproduces the fixed-step 4/3, −1/3, 2h/3 from module 12). The local
# error (LTE, derived from the interpolation remainder):
#   LTE = (1+ω)²/(ω(1+2ω)) · h_new³ · y'''/6 + O(h⁴),
# and y''' is estimated by the THIRD DIVIDED DIFFERENCE (y''' ≈ 6·f[t_{n-2},…,t_{n+1}])
# from the three last nodes and the fresh u — the factor 6 cancels.
# Zero-stability of variable-step BDF2 requires ω < 1+√2 — step growth
# is limited to 2x per step.


def _dd3(ts4: "Sequence[float]", ys4: "Sequence[np.ndarray]") -> np.ndarray:
    """Third divided difference ``f[t0,t1,t2,t3]`` (vectorized per coordinate)."""
    d1 = [(ys4[i + 1] - ys4[i]) / (ts4[i + 1] - ts4[i]) for i in range(3)]
    d2 = [(d1[i + 1] - d1[i]) / (ts4[i + 2] - ts4[i]) for i in range(2)]
    return (d2[1] - d2[0]) / (ts4[3] - ts4[0])


def _bdf_adaptive(
    F_fn: "Callable[[np.ndarray], np.ndarray | None]",
    J_fn: "Callable[[np.ndarray], np.ndarray | None]",
    m: int,
    t0: float,
    t1: float,
    y: np.ndarray,
    rtol: float,
    atol: float,
    max_steps: int,
    what: str,
) -> tuple[list[float], list[list[float]]]:
    """Core of the adaptive BDF over a COMPILED right-hand side (scalar and system).

    Step controller as in ``_dopri45``: weight ``atol + rtol·|y|``, RMS of the
    normalized error, safety factor 0.9, rejection and repeat when
    ``e > 1``; exponent matched to the estimate order (1/3 for order-3 LTE,
    1/2 for order-2 start-up estimates). Start-up: the first step is BDF1
    with HALVING (error from comparing the full step and two halves),
    the second — variable-step BDF2 with an estimate from the BDF1/BDF2
    corrector pair; from the third the full estimate from the divided
    difference. Newton non-convergence / singular matrix / outside the domain
    do NOT topple integration — the step is REJECTED and halved; only
    ``|h| < hmin`` (singularity) or ``max_steps`` gives ``PycodemathError``.
    A decreasing interval integrates BACKWARD (h < 0, ω > 0 — module 13 for free).
    """
    hmin = 1e-12 * max(1.0, abs(t0), abs(t1))
    sgn = 1.0 if t1 > t0 else -1.0
    h = (t1 - t0) / 100.0  # modest starting step — the controller matches it
    eye = np.eye(m)

    def newton(t_new: float, const: np.ndarray, coef: float, u0: np.ndarray):
        """Solve ``u = const + coef·F(t_new, u)``; ``None`` = step to reject."""
        u = u0
        for _ in range(50):
            Fu = F_fn(np.concatenate(([t_new], u)))
            if Fu is None:
                return None
            res = u - coef * Fu - const
            if float(np.max(np.abs(res))) <= 1e-12 * max(
                1.0, float(np.max(np.abs(u)))
            ):
                return u
            Ju = J_fn(np.concatenate(([t_new], u)))
            if Ju is None:
                return None
            try:
                delta = np.linalg.solve(eye - coef * Ju.reshape(m, m), -res)
            except np.linalg.LinAlgError:
                return None  # a smaller step brings the matrix closer to I
            u = u + delta
            if not np.all(np.isfinite(u)):
                return None
        return None  # no convergence — the controller will try shorter

    ts = [t0]
    rows = [[float(v) for v in y]]
    hist_t: list[float] = [t0]  # last ≤ 3 accepted nodes
    hist_y: list[np.ndarray] = [y]
    t = t0
    steps = 0
    while (t1 - t) * sgn > 0:
        if steps >= max_steps:
            raise NonConvergenceError(
                f"{what}: exceeded {max_steps} steps at t={t:g} — "
                f"the solution is probably singular (narrow the interval)"
            )
        last = (t + h - t1) * sgn >= 0
        if last:
            h = t1 - t  # the last step lands exactly on t1
        t_new = t + h
        u = err = None
        expo = 0.5
        if len(hist_t) == 1:
            # start-up 1: BDF1 full step vs two halves — the difference of
            # order-1 solutions estimates the O(h²) error; we carry the more accurate u
            u_full = newton(t_new, y, h, y)
            y_half = None if u_full is None else newton(t + h / 2, y, h / 2, y)
            u = None if y_half is None else newton(t_new, y_half, h / 2, y_half)
            if u is not None:
                err = u - u_full
        else:
            h_old = hist_t[-1] - hist_t[-2]
            w = h / h_old  # step ratio ω > 0 (both carry the run's sign)
            a1 = (1.0 + w) ** 2 / (1.0 + 2.0 * w)
            a2 = -(w * w) / (1.0 + 2.0 * w)
            coef = h * (1.0 + w) / (1.0 + 2.0 * w)
            u = newton(t_new, a1 * hist_y[-1] + a2 * hist_y[-2], coef, hist_y[-1])
            if u is not None:
                if len(hist_t) >= 3:
                    # full estimate: LTE = (1+ω)²/(ω(1+2ω))·h³·(divided difference)
                    dd = _dd3([*hist_t[-3:], t_new], [*hist_y[-3:], u])
                    err = ((1.0 + w) ** 2 / (w * (1.0 + 2.0 * w))) * h**3 * dd
                    expo = 1.0 / 3.0
                else:
                    # start-up 2: BDF1/BDF2 corrector pair — the difference estimates
                    # the O(h²) error of the lower-order corrector (conservatively)
                    u1 = newton(t_new, hist_y[-1], h, hist_y[-1])
                    if u1 is None:
                        u = None
                    else:
                        err = u - u1
        if u is None:  # Newton/domain did not deliver — try shorter
            h *= 0.5
            if abs(h) < hmin:
                raise StagnationError(
                    f"{what}: step shrank below the limit at t={t:g} — "
                    f"singularity (integration refuses to jump over it)"
                )
            continue
        sc = atol + rtol * np.maximum(np.abs(hist_y[-1]), np.abs(u))
        e = float(np.sqrt(np.mean((err / sc) ** 2)))  # RMS of the normalized error
        if e <= 1.0:  # --- step accepted ---
            t = t1 if last else t_new
            y = u
            ts.append(t)
            rows.append([float(v) for v in y])
            hist_t.append(t)
            hist_y.append(y)
            if len(hist_t) > 3:
                hist_t.pop(0)
                hist_y.pop(0)
            steps += 1
            # safety factor 0.9; growth ≤ 2x (zero-stability of BDF2: ω < 1+√2)
            grow = 2.0 if e == 0.0 else min(2.0, 0.9 * e**-expo)
            h *= grow
        else:  # --- step rejected: reduce and repeat ---
            h *= max(0.2, 0.9 * e**-expo)
            if abs(h) < hmin:
                raise StagnationError(
                    f"{what}: step shrank below the limit at t={t:g} — "
                    f"singularity (integration refuses to jump over it)"
                )
    return ts, rows


def solve_ode_stiff_adaptive(
    rhs: Expr,
    var: str,
    y0: float,
    t_span: "Sequence[float]",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
    func: str = "y",
) -> tuple[list[float], list[float]]:
    """Stiff problem ``y' = f(var, y)`` — BDF2 with a VARIABLE step.

    Combines the merits of ``solve_ode_stiff`` (an A-stable implicit method — the
    step is limited by accuracy, not stiffness) and ``solve_ode_adaptive`` (the step
    is chosen by the local error estimate to the tolerance ``atol + rtol·|y|``):
    on a stiff problem it reaches the tolerance with an order of magnitude fewer
    steps than DOPRI5, which stability forces into tiny steps. Details of the
    scheme (variable-step BDF2, estimate from the divided difference, start-up)
    — see ``_bdf_adaptive``.

    ``rhs`` depends at most on ``var`` and ``func``; ``∂f/∂y`` computed
    SYMBOLICALLY and compiled ONCE. Returns ``(ts, ys)`` at adaptively chosen
    nodes. Singularity / bad input -> ``PycodemathError``.
    """
    r = Expr(rhs)
    _check_vars([r], [var, func], "solve_ode_stiff_adaptive")
    t0, t1 = _span(t_span, "solve_ode_stiff_adaptive")
    rtol, atol = _tols(rtol, atol, "solve_ode_stiff_adaptive")
    F_fn = _vector_fn([r], [var, func])  # compiled once — before the loop
    J_fn = _vector_fn([r.diff(func)], [var, func])
    ts, rows = _bdf_adaptive(
        F_fn, J_fn, 1, t0, t1, np.array([float(y0)], dtype=float),
        rtol, atol, int(max_steps), "solve_ode_stiff_adaptive",
    )
    return ts, [row[0] for row in rows]


def solve_ode_system_stiff_adaptive(
    rhs: "Sequence[Expr]",
    var: str,
    funcs: "Sequence[str]",
    y0: "Sequence[float]",
    t_span: "Sequence[float]",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
) -> tuple[list[float], list[list[float]]]:
    """STIFF system ``y' = F(var, y)`` — BDF2 with a VARIABLE step (vectorized).

    The vector counterpart of ``solve_ode_stiff_adaptive`` — the core
    ``_bdf_adaptive`` is vectorized, Newton per step solves the system
    ``(I − coef·J)·Δ = −R`` with the Jacobian ``∂F/∂y`` computed SYMBOLICALLY
    (the module-16 pattern). This is the right tool for classics of the
    Van der Pol kind with μ=1000, where a fixed step makes no economic sense
    (relaxation forces a tiny step only in places). Returns
    ``(ts, rows)`` like ``solve_ode_system_num``.
    """
    fs, funcs, y = _system_inputs(rhs, var, funcs, y0, "solve_ode_system_stiff_adaptive")
    t0, t1 = _span(t_span, "solve_ode_system_stiff_adaptive")
    rtol, atol = _tols(rtol, atol, "solve_ode_system_stiff_adaptive")
    F_fn = _vector_fn(fs, [var, *funcs])  # compiled once — before the loop
    J_fn = _vector_fn([f.diff(g) for f in fs for g in funcs], [var, *funcs])
    return _bdf_adaptive(
        F_fn, J_fn, len(funcs), t0, t1, y, rtol, atol, int(max_steps),
        "solve_ode_system_stiff_adaptive",
    )


# --- systems for dense/events (module 11) ---------------------------------
class DenseSystemSolution(DenseSolution):
    """DOPRI5 dense output for a SYSTEM — ``sol(t)`` returns the state VECTOR.

    The core (segments, per-coordinate interpolant, interval bounds) is shared
    with the scalar ``DenseSolution`` — only the read shape differs:
    ``sol(t)`` for a number gives an ``np.ndarray`` of length ``len(funcs)``
    (state in ``funcs`` order), for a sequence — an array ``(m, n)``
    (state-row per point). Nodes: ``.ts`` and ``.ys`` = list of states
    (``ys[k][i]`` = value of ``funcs[i]`` at ``ts[k]``, as in
    ``solve_ode_system_num``).
    """

    def __init__(
        self, ts: list[float], rows: list[list[float]], segments: list,
        funcs: "Sequence[str]",
    ):
        super().__init__(ts, rows, segments)
        self.funcs = list(funcs)
        self.ys = [list(row) for row in rows]  # state (list) at each node

    def __call__(self, t):
        """Read the state — a vector for a number, an array ``(m, n)`` for a sequence."""
        if np.ndim(t) == 0:
            return self._eval_vec(float(t))
        return np.array([self._eval_vec(float(v)) for v in t])


class EventSystemSolution(DenseSystemSolution):
    """``DenseSystemSolution`` + events ``g(t, y1..yn) = 0`` (like ``EventSolution``).

    ``.events`` = ``[(t, state), …]`` — the state is a list in ``funcs`` order;
    ``.event_times`` / ``.event_values`` separated as in the scalar version.
    """

    def __init__(
        self, ts: list[float], rows: list[list[float]], segments: list,
        funcs: "Sequence[str]", events: "list[tuple[float, list[float]]]",
    ):
        super().__init__(ts, rows, segments, funcs)
        self.events = events
        self.event_times = [t for t, _ in events]
        self.event_values = [y for _, y in events]


def solve_ode_system_dense(
    rhs: "Sequence[Expr]",
    var: str,
    funcs: "Sequence[str]",
    y0: "Sequence[float]",
    t_span: "Sequence[float]",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
) -> DenseSystemSolution:
    """System ``y' = F(var, y)`` with DENSE OUTPUT — vectorized ``solve_ode_dense``.

    The same DOPRI5 engine (the ``_dopri45`` core is vectorized since module 6);
    the 4th-order interpolant is computed per coordinate, so ``sol(t)`` reads the
    FULL state at any point of ``[t0, t1]``. ``rhs[i]`` is the right-hand side of
    the equation for ``funcs[i]``; the expressions depend at most on ``var``
    and ``funcs``. Singularity / bad input -> ``PycodemathError``.
    """
    fs, funcs, y = _system_inputs(rhs, var, funcs, y0, "solve_ode_system_dense")
    t0, t1 = _span(t_span, "solve_ode_system_dense")
    rtol, atol = _tols(rtol, atol, "solve_ode_system_dense")
    sample = _vector_fn(fs, [var, *funcs])  # compiled once — before the loop
    segments: list = []
    ts, rows = _dopri45(
        sample, t0, t1, y, rtol, atol, int(max_steps),
        "solve_ode_system_dense", segments,
    )
    return DenseSystemSolution(ts, rows, segments, funcs)


def solve_ode_system_events(
    rhs: "Sequence[Expr]",
    var: str,
    funcs: "Sequence[str]",
    y0: "Sequence[float]",
    t_span: "Sequence[float]",
    event: Expr,
    direction: int = 0,
    terminal: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
) -> EventSystemSolution:
    """System ``y' = F(var, y)`` + events ``g(var, y1..yn) = 0``.

    Vectorized ``solve_ode_events``: the event function is ONE scalar
    depending on ``var`` and any of ``funcs`` (e.g. ``x`` in an oscillator —
    an event on a coordinate crossing zero). ``direction`` filters the
    transition direction; ``terminal=True`` stops integration at the first
    event (nodes and ``sol(t)`` end at ``t_event``) — the mechanics are
    identical to the scalar case (the same ``_terminal_stop`` and ``_find_events``).

    Returns ``EventSystemSolution`` (callable, ``.events`` with the state as a list).
    """
    fs, funcs, y = _system_inputs(rhs, var, funcs, y0, "solve_ode_system_events")
    g = Expr(event)
    _check_vars([g], [var, *funcs], "solve_ode_system_events (event function)")
    if direction not in (-1, 0, 1):
        raise DomainError(
            "solve_ode_system_events: direction must be -1, 0, or +1"
        )
    t0, t1 = _span(t_span, "solve_ode_system_events")
    rtol, atol = _tols(rtol, atol, "solve_ode_system_events")
    sample = _vector_fn(fs, [var, *funcs])  # compiled once — before the loop
    g_fn = _vector_fn([g], [var, *funcs])  # event function — compiled once
    segments: list = []

    if terminal:  # event checked DURING integration (stop on hit)
        stop = _terminal_stop(g_fn, int(direction), "solve_ode_system_events")
        ts, rows = _dopri45(
            sample, t0, t1, y, rtol, atol, int(max_steps),
            "solve_ode_system_events", segments, stop,
        )
        events: list[tuple[float, list[float]]] = []
        if stop.hit is not None:  # type: ignore[attr-defined]
            t_ev, y_ev = stop.hit  # type: ignore[attr-defined]
            events = [(float(t_ev), [float(v) for v in y_ev])]
        return EventSystemSolution(ts, rows, segments, funcs, events)

    ts, rows = _dopri45(
        sample, t0, t1, y, rtol, atol, int(max_steps),
        "solve_ode_system_events", segments,
    )
    sol = DenseSystemSolution(ts, rows, segments, funcs)
    raw = _find_events(sol, g_fn, int(direction), "solve_ode_system_events")
    events = [(t, [float(v) for v in vec]) for t, vec in raw]
    return EventSystemSolution(ts, rows, segments, funcs, events)
