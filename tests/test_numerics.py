"""Numerics tests: 1D (module 4) and multidimensional (module 5).

Split of tests/test_pycodemath.py into files per area (step 0 of the
development session) — test content moved without changes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycodemath import M, parse
from pycodemath.core.errors import PycodemathError
from pycodemath.engine import numerics


# --- module 4: numerics -------------------------------------------------
def test_root_find_converges_to_sqrt2():
    root = numerics.root_find(parse("x^2 - 2"), "x", 1.0)
    assert root == pytest.approx(math.sqrt(2))
    # the root actually zeroes the function
    assert parse("x^2 - 2").evalf(x=root) == pytest.approx(0.0, abs=1e-9)


def test_root_find_bisection_fallback():
    # start at x0=0 zeroes the derivative (2x=0) — Newton fails, bisection saves it
    root = numerics.root_find(parse("x^2 - 2"), "x", 0.0)
    assert abs(root) == pytest.approx(math.sqrt(2))


def test_root_find_diverges_raises():
    # x^2 + 1 has no real root -> clear error
    with pytest.raises(PycodemathError):
        numerics.root_find(parse("x^2 + 1"), "x", 1.0)


def test_root_find_divergence_distinguished_from_no_root():
    # atan(x)+5 ∈ (3.43, 6.57) — no real root, but Newton at
    # x0=2 EXPLODES (step grows ~x^2: -28 -> -2.8e3 -> -2.8e7 -> -2.7e15).
    # The message must distinguish DIVERGENCE from the misleading "no root",
    # analogously to the detection in root_find_nd (R4).
    with pytest.raises(PycodemathError, match="DIVERGES|start"):
        numerics.root_find(parse("atan(x)+5"), "x", 2.0)


def test_root_find_bisection_still_saves_diverging_newton():
    # Newton with atan(x) at x0=2 diverges, BUT the root (x=0) exists and has
    # a sign change in range — bisection as a fallback must return it, even though
    # the Newton path blew up. Divergence must NOT win over a real root.
    root = numerics.root_find(parse("atan(x)"), "x", 2.0)
    assert root == pytest.approx(0.0, abs=1e-8)


def test_integrate_num_linear():
    val = numerics.integrate_num(parse("2*x"), "x", 0, 1)
    assert val == pytest.approx(1.0)


def test_integrate_num_matches_symbolic():
    # ∫_0^pi sin(x) dx = 2 — no need for a closed form on the user's side
    val = numerics.integrate_num(parse("sin(x)"), "x", 0.0, math.pi)
    assert val == pytest.approx(2.0, abs=1e-6)


def test_minimize_quadratic():
    xmin = numerics.minimize(parse("(x-3)^2"), "x", 0.0)
    assert xmin == pytest.approx(3.0, abs=1e-4)


def test_minimize_diverges_raises():
    # linear function unbounded below -> no minimum -> error
    with pytest.raises(PycodemathError):
        numerics.minimize(parse("x"), "x", 0.0)


# --- module 5: multidimensional numerics ------------------------------------
def test_gradient_matches_hand_computed():
    # ∇(x²y + y³) = [2xy, x² + 3y²]
    g = numerics.gradient(parse("x^2*y + y^3"), ["x", "y"])
    assert g.equivalent(M("[[2*x*y], [x**2 + 3*y**2]]"))


def test_jacobian_matches_hand_computed():
    # J(x²y, x+y) = [[2xy, x²], [1, 1]]
    J = numerics.jacobian([parse("x^2*y"), parse("x + y")], ["x", "y"])
    assert J.equivalent(M("[[2*x*y, x**2], [1, 1]]"))


def test_hessian_matches_hand_computed():
    # ∇²(x²y) = [[2y, 2x], [2x, 0]] — symmetric, as it should be
    H = numerics.hessian(parse("x^2*y"), ["x", "y"])
    assert H.equivalent(M("[[2*y, 2*x], [2*x, 0]]"))


def test_root_find_nd_circle_line_intersection():
    # intersection of the circle x²+y²=4 with the line y=x -> (√2, √2) starting at (1,1)
    sol = numerics.root_find_nd(
        [parse("x^2 + y^2 - 4"), parse("x - y")], ["x", "y"], [1.0, 1.0]
    )
    assert np.allclose(sol, [math.sqrt(2), math.sqrt(2)])


def test_root_find_nd_no_real_solution_raises():
    # x²+y²+1 = 0 has no real solution -> clear error
    with pytest.raises(PycodemathError):
        numerics.root_find_nd(
            [parse("x^2 + y^2 + 1"), parse("x - y")], ["x", "y"], [1.0, 1.0]
        )


def test_root_find_nd_divergence_raises_with_hint():
    # atan(x) has a root at 0, but Newton, starting from too large |x|, diverges
    # to infinity instead of converging. The engine must detect divergence
    # (rather than return NaNs or grind through all iterations with overflow) and
    # suggest changing the starting point.
    with pytest.raises(PycodemathError, match="DIVERGES|start"):
        numerics.root_find_nd([parse("atan(x)"), parse("y")], ["x", "y"], [2.0, 0.0])


def test_root_find_nd_large_real_root_still_converges():
    # Regression against a false divergence alarm: a root with a large (but
    # finite) norm must be computed — the divergence threshold is generous.
    sol = numerics.root_find_nd(
        [parse("x - 100000000000"), parse("y - 5")], ["x", "y"], [0.0, 0.0]
    )
    assert np.allclose(sol, [1e11, 5.0])


def test_root_find_nd_rejects_non_square_and_extra_symbols():
    with pytest.raises(PycodemathError):
        numerics.root_find_nd([parse("x + y")], ["x", "y"], [0.0, 0.0])
    with pytest.raises(PycodemathError):
        numerics.root_find_nd(
            [parse("x + a"), parse("x - y")], ["x", "y"], [0.0, 0.0]
        )


def test_minimize_nd_paraboloid():
    # the minimum of (x-1)² + (y+2)² is at (1, -2)
    sol = numerics.minimize_nd(parse("(x-1)^2 + (y+2)^2"), ["x", "y"], [0.0, 0.0])
    assert np.allclose(sol, [1.0, -2.0], atol=1e-4)


def test_minimize_nd_diverges_raises():
    # linear function unbounded below -> no minimum -> error
    with pytest.raises(PycodemathError):
        numerics.minimize_nd(parse("x + y"), ["x", "y"], [0.0, 0.0])


# --- module 15: better optimization (Newton / BFGS + Armijo) ---------------
# Rosenbrock — narrow curved valley, classic start (-1.2, 1);
# global minimum at (1, 1)
_ROSEN = "(1-x)^2 + 100*(y-x^2)^2"


def test_minimize_newton_quadratic_in_two_iterations():
    # quadratic: a full Newton step hits the minimum EXACTLY; measured:
    # newton and bfgs converge at max_iter=2 (gd needs 92 iterations)
    for method in ("newton", "bfgs"):
        x = numerics.minimize(parse("(x-3)^2"), "x", 0.0, method=method, max_iter=5)
        assert x == pytest.approx(3.0, abs=1e-12)


def test_minimize_nd_newton_paraboloid_in_two_iterations():
    # nd paraboloid: the same — measured minimum max_iter = 2 (gd: 90)
    for method in ("newton", "bfgs"):
        sol = numerics.minimize_nd(
            parse("(x-1)^2 + (y+2)^2"), ["x", "y"], [0.0, 0.0],
            method=method, max_iter=5,
        )
        assert np.allclose(sol, [1.0, -2.0], atol=1e-12)


def test_minimize_nd_newton_rosenbrock():
    # measured: Newton with a symbolic Hessian converges in 22 iterations,
    # error 1.2e-10 — budget 60 with margin
    sol = numerics.minimize_nd(
        parse(_ROSEN), ["x", "y"], [-1.2, 1.0], method="newton", max_iter=60
    )
    assert np.allclose(sol, [1.0, 1.0], atol=1e-8)


def test_minimize_nd_bfgs_rosenbrock_where_gd_chokes():
    # CONTRAST of methods (like RK4 vs BDF2 in module 12): on the Rosenbrock valley
    # gradient descent chokes and refuses at the default budget of 10000
    # iterations, while BFGS converges in 36 (measured; error 1.0e-12) — budget 100
    sol = numerics.minimize_nd(
        parse(_ROSEN), ["x", "y"], [-1.2, 1.0], method="bfgs", max_iter=100
    )
    assert np.allclose(sol, [1.0, 1.0], atol=1e-8)
    with pytest.raises(PycodemathError):
        numerics.minimize_nd(parse(_ROSEN), ["x", "y"], [-1.2, 1.0])  # gd


def test_minimize_second_order_unbounded_refuses():
    # function unbounded below: newton (singular Hessian -> gradient
    # direction) and bfgs descend to infinity — refusal after the budget
    for method in ("newton", "bfgs"):
        with pytest.raises(PycodemathError):
            numerics.minimize(parse("x"), "x", 0.0, method=method, max_iter=200)
        with pytest.raises(PycodemathError):
            numerics.minimize_nd(
                parse("x + y"), ["x", "y"], [0.0, 0.0], method=method, max_iter=200
            )


def test_minimize_nd_second_order_divergence_raises_with_hint():
    # CONCAVE function unbounded below: the Armijo line search still descends,
    # so newton/bfgs TRIPLE the iterate each step — the norm grows exponentially and escapes
    # to infinity, while f -> -inf. The engine must detect divergence early
    # (like Newton in root_find_nd), rather than grind 10000 iterations with overflow or
    # return NaNs. The message indicates that the function has no minimum + start.
    for method in ("newton", "bfgs"):
        with pytest.raises(PycodemathError, match="DIVERGES|unbounded|start"):
            numerics.minimize_nd(
                parse("-(x^2) - y^2"), ["x", "y"], [1.0, 1.0], method=method
            )


def test_minimize_nd_far_minimum_still_converges():
    # Regression against a false divergence alarm: a minimum with a large (but finite)
    # norm must be computed — the divergence threshold is generous (scaled by ‖x0‖).
    for method in ("newton", "bfgs"):
        sol = numerics.minimize_nd(
            parse("(x - 100000000)^2 + (y - 5)^2"), ["x", "y"], [0.0, 0.0],
            method=method, max_iter=200,
        )
        assert np.allclose(sol, [1e8, 5.0], atol=1e-2)


def test_minimize_validates_method_and_keeps_gd_default():
    # unknown method -> clear error (1D and nd)
    with pytest.raises(PycodemathError):
        numerics.minimize(parse("x^2"), "x", 0.0, method="sgd")
    with pytest.raises(PycodemathError):
        numerics.minimize_nd(parse("x^2 + y^2"), ["x", "y"], [1.0, 1.0], method="lbfgs")
    # the default gd path is BIT-IDENTICAL to the behavior from before module 15
    # (measured anchor: 2.999999996357496)
    x = numerics.minimize(parse("(x-3)^2"), "x", 0.0)
    assert x == pytest.approx(2.999999996357496, abs=1e-15)


def test_minimize_gd_divergence_raises_with_hint():
    # R7: closing the detection family for the HISTORICAL gradient descent
    # (default method). f = -x^2 is CONCAVE, unbounded below: gd moves away
    # from 0 at every step (x_next = 1.2·x), the iterate explodes. The engine must
    # detect DIVERGENCE, rather than grind max_iter and end with a generic "does not converge".
    with pytest.raises(PycodemathError, match="DIVERGES|unbounded|start"):
        numerics.minimize(parse("-x^2"), "x", 1.0)


def test_minimize_nd_gd_divergence_raises_with_hint():
    # R7: the same for multidimensional gd (default method). f = -(x²+y²)
    # unbounded below -> the gd iterate escapes to infinity.
    with pytest.raises(PycodemathError, match="DIVERGES|unbounded|start"):
        numerics.minimize_nd(parse("-(x^2 + y^2)"), ["x", "y"], [1.0, 1.0])


def test_minimize_gd_far_minimum_still_converges():
    # Regression against a false divergence alarm in gd: a minimum with a large (but
    # finite) norm must be computed — the check fires only on an explosion.
    x = numerics.minimize(parse("(x - 100000000)^2"), "x", 0.0, max_iter=100000)
    assert x == pytest.approx(1e8, rel=1e-6)


# --- R8: consistency audit of the divergence detection family ------------------
def test_diverge_limit_helper_is_bit_identical_to_old_formula():
    # R8 lifts the threshold 1e12·max(1,‖x0‖) into a single helper. Contract: the helper
    # returns EXACTLY the same number as the old literal scattered across the solvers —
    # no value drift, only centralization of the formula.
    assert numerics._DIVERGE_SCALE == 1e12
    for norm in (0.0, 0.5, 1.0, 5.0, 1e8, 3.7e11):
        assert numerics._diverge_limit(norm) == 1e12 * max(1.0, norm)


def test_all_solvers_share_one_divergence_threshold():
    # R8 anti-drift: the whole family (root_find, root_find_nd, _descent_min,
    # minimize/minimize_nd in the gd variant) MUST draw the threshold from one source.
    # If someone in the future hardcoded 1e12 in some solver,
    # the family could drift apart — we guard this mechanically at the source.
    import inspect

    src = inspect.getsource(numerics)
    # The scale 1e12 multiplied by the threshold must NOT appear anywhere in the
    # solvers' code anymore — the old literal ``1e12 * max(...)`` now lives exclusively inside
    # _diverge_limit. Zero occurrences = nobody hardcoded it back.
    assert "1e12 * max" not in src, (
        "the hardcoded threshold '1e12 * max(...)' returned to a solver — use _diverge_limit()"
    )
    # R10: each of the five solvers computes diverge_limit through ONE helper
    # that selects the norm for the dimension — _diverge_limit_for.
    assert src.count("diverge_limit = _diverge_limit_for(") == 5
    # And the raw _diverge_limit( (manual norm selection) no longer lives in any
    # solver — exclusively inside _diverge_limit_for (which itself chooses abs vs
    # ‖·‖). Zero open-coded calls = nobody entered the wrong norm for the dimensionality.
    assert "diverge_limit = _diverge_limit(" not in src, (
        "a solver computes diverge_limit with the raw _diverge_limit() using a manually "
        "chosen norm — use _diverge_limit_for(), which picks abs/‖·‖ from the dimension of x0"
    )


# --- R9: closing the centralization — one divergence PREDICATE ----------
def test_has_diverged_scalar_and_vector():
    # The shared predicate MUST work consistently for a scalar (1D) and a vector (nd) and
    # return a clean Python bool. Scalar: abs + math.isfinite; vector:
    # np.all(np.isfinite) + np.linalg.norm — two old forms in one place.
    lim = numerics._diverge_limit(1.0)  # == 1e12
    # scalar
    assert numerics._has_diverged(float("inf"), lim) is True
    assert numerics._has_diverged(float("nan"), lim) is True
    assert numerics._has_diverged(1e13, lim) is True
    assert numerics._has_diverged(-1e13, lim) is True
    assert numerics._has_diverged(1.0, lim) is False
    assert numerics._has_diverged(lim, lim) is False  # the threshold is strict (>)
    # vector
    assert numerics._has_diverged(np.array([1.0, float("inf")]), lim) is True
    assert numerics._has_diverged(np.array([float("nan"), 0.0]), lim) is True
    assert numerics._has_diverged(np.array([1e13, 0.0]), lim) is True
    assert numerics._has_diverged(np.array([1.0, 2.0]), lim) is False


def test_all_solvers_share_one_divergence_predicate():
    # R9 anti-drift (closing R8): R8 unified the THRESHOLD (_diverge_limit), R9
    # unifies the PREDICATE (not isfinite / norm > limit) into one helper
    # _has_diverged. The whole family (root_find, _descent_min, root_find_nd,
    # minimize_nd newton/bfgs, minimize_nd gd) MUST share it — otherwise the detection
    # could drift apart in the details (1D abs vs nd norm). We guard this
    # mechanically: 1 definition + 5 solvers = 6 occurrences.
    import inspect

    src = inspect.getsource(numerics)
    assert src.count("_has_diverged(") == 6, (
        "every solver MUST share one divergence predicate _has_diverged() "
        "(1 definition + 5 solvers)"
    )


# --- R11: STAGNATION (the reverse side of divergence) ----------------------
def test_has_stagnated_scalar_and_vector():
    # Stagnation primitive, twin of _has_diverged: True when the step vanishes at
    # machine precision (‖x_next − x‖ ≤ _STAG_SCALE·eps·max(1, ‖x‖)). Scalar and
    # vector, clean Python bool. The max(1, ·) term gives a RELATIVE threshold.
    assert numerics._STAG_SCALE == 4.0
    eps = numerics._EPS
    # scalar: step exactly zero — certainly stagnation
    assert numerics._has_stagnated(1.0, 1.0) is True
    # threshold edge at x=1 (threshold = 4·eps): 2 ulps ≤ 4·eps -> stagnation,
    # 6 ulps > 4·eps -> no longer
    assert numerics._has_stagnated(1.0 + 2 * eps, 1.0) is True
    assert numerics._has_stagnated(1.0 + 6 * eps, 1.0) is False
    # a large step (1.0) is NOT stagnation
    assert numerics._has_stagnated(2.0, 1.0) is False
    # RELATIVE scaling: THE SAME absolute step (1e-8) is stagnation at
    # ‖x‖=1e12 (threshold ~1.8e-4, and 1e-8 is already lost in the float representation), but NOT
    # at ‖x‖~1 (threshold ~4e-16, a step of 1e-8 is real movement).
    assert numerics._has_stagnated(1.0 + 1e-8, 1.0) is False
    assert numerics._has_stagnated(1e12 + 1e-8, 1e12) is True
    # vector: norm of the difference vs norm of the base
    v = np.array([1.0, 1.0])
    assert numerics._has_stagnated(v, v) is True
    assert numerics._has_stagnated(v + np.array([1.0, 0.0]), v) is False
    # returns a clean bool, not numpy.bool_
    assert type(numerics._has_stagnated(v, v)) is bool


def test_minimize_gd_stagnation_distinguished_from_unbounded():
    # x^4 IS bounded below (min at 0), but gradient descent with too
    # large an lr falls into a period-2 cycle (±x0, f constant) and makes no progress.
    # Previously it ended with the MISLEADING "unbounded below"; R11 must name it
    # STAGNATION (STALL), and NOT divergence or unboundedness.
    with pytest.raises(numerics.PycodemathError, match="STALLED|STAGNAT") as exc:
        numerics.minimize(parse("x^4"), "x", 2.0, lr=1.0)
    msg = str(exc.value)
    assert "DIVERGES" not in msg  # this is NOT divergence
    assert "unbounded below" not in msg  # and NOT unboundedness — the old falsehood


def test_minimize_nd_gd_stagnation_raises_utkn():
    # Multidimensional equivalent: x^4 + y^4 with too large an lr also falls into a cycle.
    with pytest.raises(numerics.PycodemathError, match="STALLED|STAGNAT") as exc:
        numerics.minimize_nd(parse("x^4 + y^4"), ["x", "y"], [2.0, 2.0], lr=1.0)
    assert "DIVERGES" not in str(exc.value)


def test_minimize_gd_overshoot_lr_not_falsely_stagnant():
    # Regression against a FALSE stagnation alarm: too large an lr, which gives a series
    # of overshoots, but after shortening the step it CONVERGES — lr halvings are a correction, not
    # a stall, so they must return the minimum, not a STALL.
    x = numerics.minimize(parse("(x-3)^2"), "x", 0.0, lr=100.0)
    assert x == pytest.approx(3.0, abs=1e-4)


def test_minimize_nd_gd_far_minimum_not_falsely_stagnant():
    # Regression: convergence to a FAR minimum has a step small relative to ‖x‖ (i.e.
    # "step-wise stagnating"), but makes real RELATIVE progress in f per step — it must not
    # be considered a stall. The detection rests on lack of f progress, not on the step.
    sol = numerics.minimize_nd(
        parse("(x-100000000)^2 + (y-5)^2"), ["x", "y"], [0.0, 0.0], max_iter=200000
    )
    assert np.allclose(sol, [1e8, 5.0], rtol=1e-6)


# --- R12: centralization of the VALUE PROGRESS predicate (mirror of R8–R10) ------
def test_value_progressed_predicate():
    # The predicate the gd solvers actually use to detect stagnation —
    # RELATIVE drop in f: fx − f_next > _STAG_SCALE·eps·|fx|. Scalar, clean bool.
    # (This is NOT the step-wise _has_stagnated — a far minimum has a small step, but real
    # f progress, so this predicate says True, while _has_stagnated would give a false alarm.)
    eps = numerics._EPS
    # large drop in f -> progress
    assert numerics._value_progressed(1.0, 0.0) is True
    # f stands still -> no progress
    assert numerics._value_progressed(1.0, 1.0) is False
    # threshold edge at |fx|=1 (threshold = 4·eps): 2 ulps ≤ threshold -> no progress,
    # 6 ulps > threshold -> progress
    assert numerics._value_progressed(1.0, 1.0 - 2 * eps) is False
    assert numerics._value_progressed(1.0, 1.0 - 6 * eps) is True
    # RELATIVE scaling to |fx|: at fx=1e12 the threshold ~8.9e-4, a drop of 1.0 is
    # real progress, while a microscopic drop is already lost in the float representation
    assert numerics._value_progressed(1e12, 1e12 - 1.0) is True
    assert numerics._value_progressed(1e12, 1e12 - 1e-6) is False
    # clean Python bool, not numpy.bool_
    assert type(numerics._value_progressed(1.0, 0.0)) is bool


def test_gd_solvers_share_one_value_progress_predicate():
    # R12 anti-drift (mirror of R9/R10 for divergence): the VALUE PROGRESS predicate was
    # written inline TWIN-LIKE in minimize and minimize_nd (gd). Lifted into one
    # _value_progressed — both solvers MUST share it, otherwise the stagnation threshold could
    # drift apart between 1D and nd. Mechanically: 1 definition + 2 solvers = 3 occurrences,
    # and the raw form of the threshold must no longer hang anywhere outside the helper's definition.
    import inspect

    src = inspect.getsource(numerics)
    assert src.count("_value_progressed(") == 3, (
        "minimize and minimize_nd (gd) MUST share one value-progress predicate "
        "_value_progressed() (1 definition + 2 solvers)"
    )
    assert "fx - f_next > _STAG_SCALE * _EPS * abs(fx)" not in src, (
        "the raw stagnation threshold must not be inlined in a solver — use "
        "_value_progressed(fx, f_next)"
    )


# --- R13: FINDING — the stagnation family ENDS at gd ----------------
def test_root_find_newton_cycle_is_rescued_by_bisection_not_stagnation():
    # R13 FINDING (root_find): a Newton cycle EXISTS — x^3-2x+2 from x0=0 jumps
    # 0→1→0→1 infinitely, neither converging nor diverging. In gd such a cycle
    # gave the MISLEADING "unbounded below" and required stagnation detection (R11). Here
    # it does NOT reach any message: bisection is the fallback and on a real
    # sign change it saves the run. Therefore stagnation is not wired in here — there is no
    # misleading diagnosis to fix. This test guards that the fallback still saves.
    root = numerics.root_find(parse("x^3 - 2*x + 2"), "x", 0.0)
    assert root == pytest.approx(-1.7692923542, abs=1e-6)
    assert abs(float(numerics._scalar(parse("x^3 - 2*x + 2"), "x")(root))) < 1e-8

    # Check that it really is a CYCLE (and not Newton convergence): bare iteration.
    f = numerics._scalar(parse("x^3 - 2*x + 2"), "x")
    df = numerics._scalar(parse("x^3 - 2*x + 2").diff("x"), "x")
    x, seq = 0.0, []
    for _ in range(6):
        x = x - f(x) / df(x)
        seq.append(x)
    assert seq == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_root_find_no_real_root_message_already_true_without_stagnation():
    # R13 FINDING (root_find, second branch): when there is NO real root,
    # today's message is simply TRUE — there is no falsehood here that
    # stagnation would straighten out. And it still must not be confused with DIVERGENCE.
    with pytest.raises(numerics.PycodemathError, match="no real root") as exc:
        numerics.root_find(parse("x^2 + 1"), "x", 1.0)
    assert "DIVERGES" not in str(exc.value)
    assert "STALLED" not in str(exc.value)


@pytest.mark.parametrize("method", ["newton", "bfgs"])
def test_descent_min_armijo_forbids_the_gd_stagnation_cycle(method):
    # R13 FINDING (_descent_min): the R11 bug had EXACTLY one cause — a fixed
    # lr. x^4 at lr=1 falls into a period-2 cycle and STAGNATES (see the test above). The same
    # functions under newton/bfgs have no way to stagnate: there is no lr, and the backtracking Armijo
    # accepts a step ONLY when f actually drops, so f is strictly monotonic and
    # the "f stands still" cycle is structurally impossible. They must simply CONVERGE.
    sol = numerics.minimize_nd(parse("x^4 + y^4"), ["x", "y"], [2.0, 2.0], method=method)
    assert np.allclose(sol, [0.0, 0.0], atol=1e-3)
    # and the 1D equivalent of exactly what in gd threw a STALL at lr=1
    sol1 = numerics.minimize_nd(parse("x^4"), ["x"], [2.0], method=method)
    assert sol1[0] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("method", ["newton", "bfgs"])
def test_descent_min_unbounded_message_is_true_not_a_stagnation_misreport(method):
    # R13 FINDING: the only _descent_min run that grinds through max_iter is a function with a
    # CONSTANT gradient (-x-y): the iterate marches LINEARLY, so it rightly does not fire
    # the divergence threshold (norm ~1e2 vs threshold 1e12), and f drops by a constant each step.
    # The generic message "unbounded below" is TRUE there — this is not
    # a hushed-up stagnation, so there is nothing to fix here with stall detection.
    with pytest.raises(numerics.PycodemathError, match="does not converge") as exc:
        numerics.minimize_nd(parse("-x - y"), ["x", "y"], [0.0, 0.0],
                             method=method, max_iter=200)
    msg = str(exc.value)
    assert "unbounded" in msg      # and this is true: -x-y has NO minimum
    assert "STALLED" not in msg and "STAGNAT" not in msg
