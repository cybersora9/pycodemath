"""ODE package tests: dsolve, RK4, adaptive/dense/events, stiff, backward.

Split of tests/test_pycodemath.py into per-area files (step 0 of the
development session) — test content moved unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycodemath import E, parse
from pycodemath.core.errors import PycodemathError
from pycodemath.engine import ode


# --- module 6: ordinary differential equations --------------------------
def test_dsolve_exponential_growth():
    sols = ode.dsolve(parse("y"), "y", "t")
    assert len(sols) == 1
    assert sols[0].equivalent(E("C1*exp(t)"))


def test_dsolve_parametric_ode():
    # parameters (a) are allowed on the symbolic path
    sols = ode.dsolve(parse("a*y"), "y", "t")
    assert sols[0].equivalent(E("C1*exp(a*t)"))


def test_dsolve_no_closed_form_suggests_numeric():
    # y' = y^y has only an implicit solution — dsolve redirects to numerics
    with pytest.raises(PycodemathError) as ei:
        ode.dsolve(parse("y^y"), "y", "t")
    assert "solve_ode_num" in str(ei.value)


def test_dsolve_rejects_same_names():
    with pytest.raises(PycodemathError):
        ode.dsolve(parse("t"), "t", "t")


def test_solve_ode_num_matches_exp():
    ts, ys = ode.solve_ode_num(parse("y"), "t", 1.0, (0.0, 1.0), 100)
    assert len(ts) == 101 == len(ys)
    assert ts[-1] == pytest.approx(1.0)
    # RK4 global error at n=100 is ~2e-10 — 1e-8 with a large margin
    assert ys[-1] == pytest.approx(math.e, abs=1e-8)
    assert ys[50] == pytest.approx(math.exp(0.5), abs=1e-8)


def test_solve_ode_num_polynomial_rhs_exact():
    # RK4 integrates polynomials of degree <= 3 exactly: y' = 2t -> y = t^2
    ts, ys = ode.solve_ode_num(parse("2*t"), "t", 0.0, (0.0, 1.0), 100)
    assert ys[-1] == pytest.approx(1.0)


def test_solve_ode_num_divergence_raises():
    # y' = y^2, y(0)=1 blows up at t=1 — interval [0,2] must raise an error
    with pytest.raises(PycodemathError):
        ode.solve_ode_num(parse("y^2"), "t", 1.0, (0.0, 2.0), 100)


def test_solve_ode_num_validates_input():
    with pytest.raises(PycodemathError):  # unsubstituted parameter a
        ode.solve_ode_num(parse("a*y"), "t", 1.0, (0.0, 1.0), 10)
    with pytest.raises(PycodemathError):  # n must be positive
        ode.solve_ode_num(parse("y"), "t", 1.0, (0.0, 1.0), 0)
    with pytest.raises(PycodemathError):  # zero-length interval
        ode.solve_ode_num(parse("y"), "t", 1.0, (1.0, 1.0), 10)


def test_solve_ode_system_num_oscillator_conserves_energy():
    # x'' = -x as a system: x' = v, v' = -x; E = (x^2 + v^2)/2 = const
    ts, rows = ode.solve_ode_system_num(
        [parse("v"), parse("-x")], "t", ["x", "v"], [1.0, 0.0],
        (0.0, 2 * math.pi), 1000,
    )
    x, v = rows[-1]
    assert 0.5 * (x * x + v * v) == pytest.approx(0.5, abs=1e-6)
    # after a full period we return to the initial state
    assert np.allclose([x, v], [1.0, 0.0], atol=1e-6)


def test_solve_ode_system_num_validation():
    with pytest.raises(PycodemathError):  # 2 equations, 1 function
        ode.solve_ode_system_num(
            [parse("v"), parse("-x")], "t", ["x"], [1.0], (0.0, 1.0), 10
        )
    with pytest.raises(PycodemathError):  # y0 of wrong length
        ode.solve_ode_system_num(
            [parse("v"), parse("-x")], "t", ["x", "v"], [1.0], (0.0, 1.0), 10
        )


# --- module 7: adaptive ODE solver (Dormand-Prince 5(4)) ----------------
def test_solve_ode_adaptive_hits_tolerance():
    # y' = y, y(0)=1 -> exp(t); step chosen automatically
    ts, ys = ode.solve_ode_adaptive(parse("y"), "t", 1.0, (0.0, 1.0), rtol=1e-8)
    assert ts[0] == 0.0 and ts[-1] == pytest.approx(1.0)  # nodes from t0 to t1
    assert len(ts) == len(ys)
    assert ys[0] == 1.0
    assert ys[-1] == pytest.approx(math.e, abs=1e-6)
    # nodes strictly increase (positive step across the whole interval)
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_rk4_convergence_order_four():
    # RK4 global error is O(h^4): doubling n cuts the error ~16x (2^4)
    def err(n):
        _, ys = ode.solve_ode_num(parse("y"), "t", 1.0, (0.0, 1.0), n)
        return abs(ys[-1] - math.e)

    ratio = err(20) / err(40)
    assert 14.0 < ratio < 17.0  # measured ~15.7, tends toward 16


def test_solve_ode_adaptive_fewer_steps_than_rk4():
    # smooth problem y' = y*cos(t) -> exp(sin(t)); adaptation hits the tolerance
    # with fewer steps than RK4 (for the same budget RK4 is worse)
    rhs, span = parse("y*cos(t)"), (0.0, 5.0)
    exact = math.exp(math.sin(5.0))
    ts, ys = ode.solve_ode_adaptive(rhs, "t", 1.0, span, rtol=1e-6, atol=1e-9)
    n_adapt = len(ts) - 1
    err_adapt = abs(ys[-1] - exact)
    assert err_adapt < 1e-5  # hit the requested tolerance
    # RK4 with the same number of steps is clearly less accurate (order 4 vs 5,
    # no densification) — so for a given tolerance it needs MORE steps
    _, ys_rk4 = ode.solve_ode_num(rhs, "t", 1.0, span, n_adapt)
    assert abs(ys_rk4[-1] - exact) > err_adapt


def test_solve_ode_adaptive_singularity_refuses():
    # y' = y^2, y(0)=1 -> y = 1/(1-t), pole at t=1
    # BEFORE the pole: adaptation reproduces the solution exactly
    ts, ys = ode.solve_ode_adaptive(parse("y^2"), "t", 1.0, (0.0, 0.9), rtol=1e-8)
    assert ys[-1] == pytest.approx(1.0 / (1.0 - 0.9), rel=1e-6)
    # THROUGH the pole: the step shrinks at the singularity until integration REFUSES
    # (instead of silently jumping over t=1 and returning garbage)
    with pytest.raises(PycodemathError):
        ode.solve_ode_adaptive(parse("y^2"), "t", 1.0, (0.0, 2.0), rtol=1e-8)


def test_solve_ode_adaptive_validates_input():
    with pytest.raises(PycodemathError):  # unsubstituted parameter a
        ode.solve_ode_adaptive(parse("a*y"), "t", 1.0, (0.0, 1.0))
    with pytest.raises(PycodemathError):  # zero-length interval
        ode.solve_ode_adaptive(parse("y"), "t", 1.0, (1.0, 1.0))
    with pytest.raises(PycodemathError):  # tolerances must be positive
        ode.solve_ode_adaptive(parse("y"), "t", 1.0, (0.0, 1.0), rtol=-1.0)


# --- module 8: dense output DOPRI5 (dense output) ------------------------
def test_solve_ode_dense_interpolates_between_nodes():
    # y' = y -> exp(t); the interpolant must hit points BETWEEN nodes, not only at them
    rtol = 1e-8
    sol = ode.solve_ode_dense(parse("y"), "t", 1.0, (0.0, 1.0), rtol=rtol)
    assert isinstance(sol, ode.DenseSolution)
    assert sol.t0 == 0.0 and sol.t1 == pytest.approx(1.0)
    # points in the MIDDLE of each step (not nodes) — error consistent with rtol
    for a, b in zip(sol.ts, sol.ts[1:]):
        tm = 0.5 * (a + b)
        assert sol(tm) == pytest.approx(math.exp(tm), abs=1e-6)
    # a dense grid independent of the nodes also hits the exact solution
    grid = np.linspace(0.0, 1.0, 101)
    assert np.allclose(sol(grid), np.exp(grid), atol=1e-6)


def test_solve_ode_dense_nodes_match_adaptive():
    # dense output shares the engine with solve_ode_adaptive -> the same nodes
    rhs, span = parse("y*cos(t)"), (0.0, 5.0)
    sol = ode.solve_ode_dense(rhs, "t", 1.0, span, rtol=1e-7)
    ts, ys = ode.solve_ode_adaptive(rhs, "t", 1.0, span, rtol=1e-7)
    assert np.allclose(sol.ts, ts) and np.allclose(sol.ys, ys)
    # at the nodes the interpolant reproduces the samples EXACTLY (theta=0 -> y_old, theta=1 -> y_new)
    for tk, yk in zip(sol.ts, sol.ys):
        assert sol(tk) == pytest.approx(yk, abs=1e-12)


def test_solve_ode_dense_scalar_and_array_and_range():
    sol = ode.solve_ode_dense(parse("y"), "t", 1.0, (0.0, 2.0), rtol=1e-8)
    assert isinstance(sol(1.0), float)  # scalar -> float
    out = sol(np.array([0.5, 1.0, 1.5]))  # sequence -> ndarray
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    assert np.allclose(out, np.exp([0.5, 1.0, 1.5]), atol=1e-6)
    with pytest.raises(PycodemathError):  # outside [t0, t1]
        sol(3.0)


def test_solve_ode_dense_validates_input():
    with pytest.raises(PycodemathError):  # unsubstituted parameter a
        ode.solve_ode_dense(parse("a*y"), "t", 1.0, (0.0, 1.0))
    with pytest.raises(PycodemathError):  # zero-length interval
        ode.solve_ode_dense(parse("y"), "t", 1.0, (1.0, 1.0))


# --- module 9: event detection (event detection) -----------------------
# y' = cos(t), y(0.5)=sin(0.5) -> y = sin(t); zeros of y at π, 2π, 3π
_EV_Y0 = math.sin(0.5)


def test_solve_ode_events_finds_analytic_roots():
    sol = ode.solve_ode_events(
        parse("cos(t)"), "t", _EV_Y0, (0.5, 10.0), parse("y"), rtol=1e-8
    )
    assert isinstance(sol, ode.EventSolution)
    assert callable(sol) and sol(1.0) == pytest.approx(math.sin(1.0), abs=1e-6)
    # three zeros of sin(t) in the interval: π, 2π, 3π — with tolerance matching rtol
    roots = [math.pi, 2 * math.pi, 3 * math.pi]
    assert len(sol.events) == 3
    assert np.allclose(sol.event_times, roots, atol=1e-6)
    # at the event the event function (here y) is ~zero
    assert all(abs(y) < 1e-6 for y in sol.event_values)


def test_solve_ode_events_direction_filters():
    common = dict(rtol=1e-8)
    # rising (−→+): only 2π; falling (+→−): π and 3π
    up = ode.solve_ode_events(
        parse("cos(t)"), "t", _EV_Y0, (0.5, 10.0), parse("y"), direction=1, **common
    )
    down = ode.solve_ode_events(
        parse("cos(t)"), "t", _EV_Y0, (0.5, 10.0), parse("y"), direction=-1, **common
    )
    assert np.allclose(up.event_times, [2 * math.pi], atol=1e-6)
    assert np.allclose(down.event_times, [math.pi, 3 * math.pi], atol=1e-6)


def test_solve_ode_events_none_when_no_crossing():
    # g = y + 2 = sin(t) + 2 never reaches zero -> no events
    sol = ode.solve_ode_events(
        parse("cos(t)"), "t", _EV_Y0, (0.5, 10.0), parse("y+2"), rtol=1e-8
    )
    assert sol.events == []
    # the object is still a functioning dense output
    assert sol(math.pi) == pytest.approx(math.sin(math.pi), abs=1e-6)


def test_solve_ode_events_validates_input():
    with pytest.raises(PycodemathError):  # direction outside {-1,0,1}
        ode.solve_ode_events(
            parse("cos(t)"), "t", 0.0, (0.0, 1.0), parse("y"), direction=2
        )
    with pytest.raises(PycodemathError):  # event function with a parameter
        ode.solve_ode_events(parse("cos(t)"), "t", 0.0, (0.0, 1.0), parse("a*y"))


# --- module 10: terminal events (terminal=True) -----------------------
def test_solve_ode_events_terminal_stops_at_event():
    # reference from the prompt: y' = -1, y(0)=1 -> y = 1 - t, zero at t=1 (not at t1=3)
    sol = ode.solve_ode_events(
        parse("-1"), "t", 1.0, (0.0, 3.0), parse("y"), terminal=True, rtol=1e-8
    )
    # measured: |t_ev - 1| ~ 2.4e-14, |y_ev| ~ 2.4e-14 (interpolant exact
    # for a linear problem) — tolerances with a huge margin
    assert len(sol.events) == 1
    assert sol.event_times[0] == pytest.approx(1.0, abs=1e-10)
    assert abs(sol.event_values[0]) < 1e-10
    # nodes and dense output end EXACTLY at the event, not at t1
    assert sol.ts[-1] == sol.event_times[0]
    assert sol.t1 == sol.event_times[0]
    # interpolant consistent on the shortened interval...
    assert sol(0.5) == pytest.approx(0.5, abs=1e-9)
    with pytest.raises(PycodemathError):  # ...and beyond the event there is nothing
        sol(2.0)


def test_terminal_event_respects_direction():
    # y = sin(t) on [0.5, 10]: the first rising zero is 2π, falling one is π
    y0 = math.sin(0.5)
    up = ode.solve_ode_events(
        parse("cos(t)"), "t", y0, (0.5, 10.0), parse("y"),
        direction=1, terminal=True, rtol=1e-8,
    )
    down = ode.solve_ode_events(
        parse("cos(t)"), "t", y0, (0.5, 10.0), parse("y"),
        direction=-1, terminal=True, rtol=1e-8,
    )
    # measured: |t-2π| ~ 6.3e-9, |t-π| ~ 2.7e-10 (~rtol)
    assert up.event_times == pytest.approx([2 * math.pi], abs=1e-6)
    assert down.event_times == pytest.approx([math.pi], abs=1e-6)


def test_terminal_event_stops_before_singularity():
    # y' = y², y(0)=1 -> y = 1/(1-t), pole at t=1; event y=10 at t=0.9.
    # The event is checked DURING integration — stop BEFORE the pole,
    # even though t1=2 lies far beyond it (measured: |t_ev-0.9| ~ 3.1e-10)
    sol = ode.solve_ode_events(
        parse("y^2"), "t", 1.0, (0.0, 2.0), parse("y-10"), terminal=True, rtol=1e-8
    )
    assert sol.event_times[0] == pytest.approx(0.9, abs=1e-7)
    assert sol.event_values[0] == pytest.approx(10.0, abs=1e-6)
    # without terminal the same interval MUST refuse (pole in the middle)
    with pytest.raises(PycodemathError):
        ode.solve_ode_events(
            parse("y^2"), "t", 1.0, (0.0, 2.0), parse("y-10"), rtol=1e-8
        )


def test_terminal_no_crossing_runs_full_span():
    # g = y + 2 never reaches zero -> full interval, empty list (like module 9)
    y0 = math.sin(0.5)
    sol = ode.solve_ode_events(
        parse("cos(t)"), "t", y0, (0.5, 10.0), parse("y+2"), terminal=True, rtol=1e-8
    )
    assert sol.events == []
    assert sol.t1 == pytest.approx(10.0)
    # without terminal, module 9 behavior is unchanged (same code — 3 events)
    ref = ode.solve_ode_events(
        parse("cos(t)"), "t", y0, (0.5, 10.0), parse("y"), rtol=1e-8
    )
    assert len(ref.events) == 3 and ref.t1 == pytest.approx(10.0)


# --- module 11: systems for dense/events ------------------------------------
# oscillator x' = v, v' = -x, start at t0=0.5: x = sin(t), v = cos(t)
_OSC_RHS = ["v", "-x"]
_OSC_Y0 = [math.sin(0.5), math.cos(0.5)]


def _osc():
    return [parse(e) for e in _OSC_RHS]


def test_solve_ode_system_dense_oscillator():
    sol = ode.solve_ode_system_dense(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), rtol=1e-8
    )
    assert isinstance(sol, ode.DenseSystemSolution)
    # scalar -> state vector (x, v); measured: max grid error 1.4e-8
    mid = sol(1.234)
    assert isinstance(mid, np.ndarray) and mid.shape == (2,)
    assert np.allclose(mid, [math.sin(1.234), math.cos(1.234)], atol=1e-6)
    # sequence -> array (m, n); accuracy between nodes ~rtol
    grid = np.linspace(0.5, 10.0, 197)
    got = sol(grid)
    assert got.shape == (197, 2)
    assert np.allclose(got, np.stack([np.sin(grid), np.cos(grid)], axis=1), atol=1e-6)
    # energy (x²+v²)/2 = 0.5 conserved also BETWEEN nodes (measured 1.5e-8)
    energy = 0.5 * (got[:, 0] ** 2 + got[:, 1] ** 2)
    assert np.allclose(energy, 0.5, atol=1e-6)
    # at the nodes the interpolant reproduces the samples EXACTLY (measured: 0.0)
    for tk, yk in zip(sol.ts, sol.ys):
        assert np.allclose(sol(tk), yk, atol=1e-12)


def test_solve_ode_system_events_oscillator_hits_pi():
    # event x = 0 at multiples of π; event state (x≈0, v=±1)
    sol = ode.solve_ode_system_events(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), parse("x"), rtol=1e-8
    )
    assert isinstance(sol, ode.EventSystemSolution)
    roots = [math.pi, 2 * math.pi, 3 * math.pi]
    # measured errors in t: ≤ 2.9e-9 (~rtol)
    assert np.allclose(sol.event_times, roots, atol=1e-6)
    # state at the event: x ~ 1e-13, v = cos(kπ) = -1, +1, -1 (error ≤ 1.4e-8)
    for state, v_exact in zip(sol.event_values, [-1.0, 1.0, -1.0]):
        assert abs(state[0]) < 1e-9
        assert state[1] == pytest.approx(v_exact, abs=1e-6)
    # event on the SECOND coordinate: v = cos(t) = 0 at π/2 + kπ
    on_v = ode.solve_ode_system_events(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), parse("v"), rtol=1e-8
    )
    v_roots = [math.pi / 2, 3 * math.pi / 2, 5 * math.pi / 2]
    assert np.allclose(on_v.event_times, v_roots, atol=1e-6)


def test_solve_ode_system_events_direction_and_terminal():
    # dir=+1: x rises through zero only at 2π
    up = ode.solve_ode_system_events(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), parse("x"),
        direction=1, rtol=1e-8,
    )
    assert np.allclose(up.event_times, [2 * math.pi], atol=1e-6)
    # terminal: stop at the FIRST event (π) — nodes end at the event
    stopped = ode.solve_ode_system_events(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), parse("x"),
        terminal=True, rtol=1e-8,
    )
    assert stopped.t1 == pytest.approx(math.pi, abs=1e-6)
    assert stopped.ts[-1] == stopped.event_times[0]
    assert stopped.event_values[0][1] == pytest.approx(-1.0, abs=1e-6)
    with pytest.raises(PycodemathError):  # beyond the event there is nothing
        stopped(5.0)


def test_solve_ode_system_dense_events_validate_input():
    with pytest.raises(PycodemathError):  # non-square system
        ode.solve_ode_system_dense(
            [parse("v")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0)
        )
    with pytest.raises(PycodemathError):  # y0 of wrong length
        ode.solve_ode_system_dense(_osc(), "t", ["x", "v"], [1.0], (0.0, 1.0))
    with pytest.raises(PycodemathError):  # event function with a parameter
        ode.solve_ode_system_events(
            _osc(), "t", ["x", "v"], _OSC_Y0, (0.0, 1.0), parse("a*x")
        )
    with pytest.raises(PycodemathError):  # tolerances must be positive
        ode.solve_ode_system_dense(
            _osc(), "t", ["x", "v"], _OSC_Y0, (0.0, 1.0), rtol=-1.0
        )
    with pytest.raises(PycodemathError):  # direction outside {-1,0,1}
        ode.solve_ode_system_events(
            _osc(), "t", ["x", "v"], _OSC_Y0, (0.0, 1.0), parse("x"), direction=5
        )


def test_scalar_dense_events_unchanged_after_system_refactor():
    # scalar API unchanged after the generalization (_eval_vec underneath):
    # sol(t) is still float, events are still (float, float)
    sol = ode.solve_ode_dense(parse("y"), "t", 1.0, (0.0, 1.0), rtol=1e-8)
    assert isinstance(sol(0.5), float)
    ev = ode.solve_ode_events(
        parse("cos(t)"), "t", math.sin(0.5), (0.5, 10.0), parse("y"), rtol=1e-8
    )
    assert all(isinstance(t, float) and isinstance(y, float) for t, y in ev.events)
    assert np.allclose(ev.event_times, [math.pi, 2 * math.pi, 3 * math.pi], atol=1e-6)


# --- module 12: stiff methods (BDF2, implicit) ----------------------------
# y' = -1000·(y - cos(t)), y(0)=0 — stiff: damping scale (1/1000) orders of
# magnitude shorter than the solution scale. Exact solution:
# y = A·cos t + B·sin t - A·e^(-1000t), A = 10⁶/(10⁶+1), B = 10³/(10⁶+1)
_STIFF_RHS = "-1000*(y - cos(t))"
_STIFF_A = 1_000_000 / 1_000_001
_STIFF_B = 1_000 / 1_000_001


def _stiff_exact(t: float) -> float:
    return (
        _STIFF_A * math.cos(t) + _STIFF_B * math.sin(t)
        - _STIFF_A * math.exp(-1000.0 * t)
    )


def test_solve_ode_stiff_matches_exact():
    # BDF2 with h=0.01 (h·λ = -10 — far beyond the stability limit of explicit methods)
    ts, ys = ode.solve_ode_stiff(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), 100)
    assert len(ts) == 101 == len(ys)
    assert ts[-1] == pytest.approx(1.0)
    # measured: error 2.8e-8 — tolerance with a large margin
    assert ys[-1] == pytest.approx(_stiff_exact(1.0), abs=1e-6)


def test_stiff_beats_explicit_at_same_budget():
    # same step budget: BDF2 hits the solution, explicit RK4 is
    # useless — instability either explodes (PycodemathError) or
    # returns astronomical garbage (measured: amplification ~291^100 ≈ 1e246)
    _, ys = ode.solve_ode_stiff(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), 100)
    err_bdf2 = abs(ys[-1] - _stiff_exact(1.0))
    assert err_bdf2 < 1e-6
    try:
        _, ys_rk4 = ode.solve_ode_num(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), 100)
        assert abs(ys_rk4[-1] - _stiff_exact(1.0)) > 1e50  # finite garbage
    except PycodemathError:
        pass  # or an honest refusal on overflow to inf


def test_stiff_convergence_order_two():
    # BDF2 is order 2: doubling n cuts the error ~4x (measured 3.99, 3.995)
    def err(n):
        _, ys = ode.solve_ode_stiff(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), n)
        return abs(ys[-1] - _stiff_exact(1.0))

    ratio = err(100) / err(200)
    assert 3.5 < ratio < 4.5


def test_stiff_nonlinear_agrees_with_smooth_solution():
    # non-stiff nonlinear y' = y·cos(t) -> exp(sin(t)) — Newton on the implicit
    # step iterates (nonlinear problem), result agrees with the exact one
    # (measured: error 2.3e-5 at n=400, O(h²))
    ts, ys = ode.solve_ode_stiff(parse("y*cos(t)"), "t", 1.0, (0.0, 5.0), 400)
    assert ys[-1] == pytest.approx(math.exp(math.sin(5.0)), abs=1e-4)


def test_solve_ode_stiff_validates_and_refuses():
    with pytest.raises(PycodemathError):  # unsubstituted parameter a
        ode.solve_ode_stiff(parse("a*y"), "t", 1.0, (0.0, 1.0), 10)
    with pytest.raises(PycodemathError):  # n must be positive
        ode.solve_ode_stiff(parse("y"), "t", 1.0, (0.0, 1.0), 0)
    with pytest.raises(PycodemathError):  # zero-length interval
        ode.solve_ode_stiff(parse("y"), "t", 1.0, (1.0, 1.0), 10)
    with pytest.raises(PycodemathError):  # y' = y², pole at t=1 — refusal
        ode.solve_ode_stiff(parse("y^2"), "t", 1.0, (0.0, 2.0), 100)


# --- module 16: stiff systems (solve_ode_system_stiff) --------------------
# linear stiff system x' = A·x, A = [[-500.5, 499.5],[499.5, -500.5]]:
# eigenvalues -1 (vector [1,1]) and -1000 (vector [1,-1]);
# y(0) = [2,0] = [1,1] + [1,-1]  ->  y(t) = e^-t·[1,1] + e^-1000t·[1,-1]
_LIN_STIFF = ["-500.5*x + 499.5*y", "499.5*x - 500.5*y"]


def _lin_stiff():
    return [parse(e) for e in _LIN_STIFF]


def _lin_stiff_exact(t: float) -> list[float]:
    s, f = math.exp(-t), math.exp(-1000.0 * t)
    return [s + f, s - f]


def _lin_stiff_err(n: int) -> float:
    _, rows = ode.solve_ode_system_stiff(
        _lin_stiff(), "t", ["x", "y"], [2.0, 0.0], (0.0, 1.0), n
    )
    ex = _lin_stiff_exact(1.0)
    return max(abs(rows[-1][0] - ex[0]), abs(rows[-1][1] - ex[1]))


def test_solve_ode_system_stiff_linear_matches_exact():
    # BDF2 with h=0.01 (h·λ = -10 for the fast mode) hits the exact
    # solution; measured: error 1.54e-5 — tolerance with a margin
    ts, rows = ode.solve_ode_system_stiff(
        _lin_stiff(), "t", ["x", "y"], [2.0, 0.0], (0.0, 1.0), 100
    )
    assert len(ts) == 101 == len(rows)
    assert ts[-1] == pytest.approx(1.0)
    assert np.allclose(rows[-1], _lin_stiff_exact(1.0), atol=1e-4)


def test_system_stiff_beats_explicit_at_same_budget():
    # contrast like in module 12: BDF2 hits, explicit RK4 with the same budget
    # is useless (measured: finite garbage 2.45e246 — amplification
    # ~291^100 fits in float64, the isfinite guard has nothing to catch)
    assert _lin_stiff_err(100) < 1e-4
    try:
        _, rows = ode.solve_ode_system_num(
            _lin_stiff(), "t", ["x", "y"], [2.0, 0.0], (0.0, 1.0), 100
        )
        assert abs(rows[-1][0]) > 1e50  # finite garbage
    except PycodemathError:
        pass  # or an honest refusal on overflow to inf


def test_system_stiff_convergence_order_two():
    # BDF2 is order 2 also in the vector case: doubling n cuts the error ~4x
    # (measured: 4.013 and 4.006)
    ratio = _lin_stiff_err(100) / _lin_stiff_err(200)
    assert 3.5 < ratio < 4.5


def test_system_stiff_oscillator_correctness():
    # non-stiff oscillator: correctness of the Newton nd on a 2x2 matrix
    # (measured: error 5.1e-4 after a full period at n=400, O(h²))
    _, rows = ode.solve_ode_system_stiff(
        [parse("v"), parse("-x")], "t", ["x", "v"], [0.0, 1.0],
        (0.0, 2 * math.pi), 400,
    )
    assert np.allclose(rows[-1], [0.0, 1.0], atol=5e-3)


def test_system_stiff_nonlinear_van_der_pol():
    # Van der Pol μ=2 — nonlinear coupled system: Newton nd ITERATES
    # (Jacobian depends on the state); reference = dense rtol 1e-10
    # (measured: error 3.9e-4 at n=2000)
    mu = 2.0
    vdp = [parse("v"), parse(f"{mu}*(1 - x^2)*v - x")]
    ref = ode.solve_ode_system_dense(
        [parse("v"), parse(f"{mu}*(1 - x^2)*v - x")], "t", ["x", "v"],
        [2.0, 0.0], (0.0, 10.0), rtol=1e-10, atol=1e-12,
    )
    _, rows = ode.solve_ode_system_stiff(
        vdp, "t", ["x", "v"], [2.0, 0.0], (0.0, 10.0), 2000
    )
    assert np.allclose(rows[-1], ref(10.0), atol=5e-3)


def test_system_stiff_backward_roundtrip():
    # backward (t1 < t0) works for free after module 13 — on a MILD system
    # (oscillator; measured: 3.8e-4 there and back, O(h²)).
    # Note: backward on a STIFF system is an ill-posed task — the fast
    # mode e^-1000t backward grows like e^+1000t and amplifies rounding error
    # (measured: garbage ~1.9e9); this is a feature of the task, not the solver.
    osc = [parse("v"), parse("-x")]
    _, rf = ode.solve_ode_system_stiff(
        osc, "t", ["x", "v"], [0.0, 1.0], (0.0, 2 * math.pi), 400
    )
    _, rb = ode.solve_ode_system_stiff(
        [parse("v"), parse("-x")], "t", ["x", "v"], rf[-1], (2 * math.pi, 0.0), 400
    )
    assert np.allclose(rb[-1], [0.0, 1.0], atol=5e-3)


def test_system_stiff_scalar_consistency_and_validation():
    # a 1-equation system computes THE SAME as the scalar solve_ode_stiff
    # (measured: max|Δy| = 0.0 — same BDF2, Newton nd 1x1 == scalar)
    rhs_scalar = parse("-1000*(y - cos(t))")
    _, ys = ode.solve_ode_stiff(rhs_scalar, "t", 0.0, (0.0, 1.0), 100)
    _, rows = ode.solve_ode_system_stiff(
        [parse("-1000*(y - cos(t))")], "t", ["y"], [0.0], (0.0, 1.0), 100
    )
    assert np.allclose([r[0] for r in rows], ys, atol=1e-12)
    # validations and refusals — everything PycodemathError
    with pytest.raises(PycodemathError):  # non-square system
        ode.solve_ode_system_stiff([parse("v")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0))
    with pytest.raises(PycodemathError):  # y0 of wrong length
        ode.solve_ode_system_stiff(
            [parse("v"), parse("-x")], "t", ["x", "v"], [1.0], (0.0, 1.0)
        )
    with pytest.raises(PycodemathError):  # unsubstituted parameter a
        ode.solve_ode_system_stiff(
            [parse("a*x"), parse("-x")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0)
        )
    with pytest.raises(PycodemathError):  # n must be positive
        ode.solve_ode_system_stiff(
            [parse("v"), parse("-x")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0), 0
        )
    with pytest.raises(PycodemathError):  # zero-length interval
        ode.solve_ode_system_stiff(
            [parse("v"), parse("-x")], "t", ["x", "v"], [1.0, 0.0], (1.0, 1.0)
        )
    with pytest.raises(PycodemathError):  # y' = y², pole at t=1 — refusal
        ode.solve_ode_system_stiff([parse("y^2")], "t", ["y"], [1.0], (0.0, 2.0), 100)


# --- module 17: adaptive step for BDF ------------------------------------
def test_stiff_adaptive_matches_exact_and_beats_dopri5_on_anchor():
    # reference problem m12: y' = -1000(y - cos t), y(0)=0, rtol=1e-6.
    # ANCHOR from m12: DOPRI5 needs 378 steps (STABILITY limit).
    # Measured: adaptive BDF 313 steps, error 2.3e-8 — fewer than DOPRI5,
    # but modestly: at y(0)=0 BOTH solvers must resolve the transient
    # e^-1000t to tolerance (see the slow-manifold test below).
    rhs = parse(_STIFF_RHS)
    ts_b, ys_b = ode.solve_ode_stiff_adaptive(rhs, "t", 0.0, (0.0, 1.0), rtol=1e-6)
    assert ys_b[-1] == pytest.approx(_stiff_exact(1.0), abs=1e-6)
    assert ts_b[0] == 0.0 and ts_b[-1] == pytest.approx(1.0)
    ts_d, _ = ode.solve_ode_adaptive(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), rtol=1e-6)
    assert len(ts_b) < len(ts_d)  # measured: 313 < 378


def test_stiff_adaptive_slow_manifold_far_fewer_steps():
    # start ON the slow manifold (y(0) = A -> zero transient): stiffness
    # in its pure form — DOPRI5 still holds the stability limit (measured
    # 357 steps), adaptive BDF is limited only by ACCURACY (measured 58,
    # error 2.8e-8) — SUBSTANTIALLY fewer steps (>4x)
    rhs = parse(_STIFF_RHS)
    slow_exact = _STIFF_A * math.cos(1.0) + _STIFF_B * math.sin(1.0)
    ts_b, ys_b = ode.solve_ode_stiff_adaptive(
        rhs, "t", _STIFF_A, (0.0, 1.0), rtol=1e-6
    )
    assert ys_b[-1] == pytest.approx(slow_exact, abs=1e-6)
    ts_d, _ = ode.solve_ode_adaptive(
        parse(_STIFF_RHS), "t", _STIFF_A, (0.0, 1.0), rtol=1e-6
    )
    assert len(ts_b) - 1 < (len(ts_d) - 1) / 4  # measured: 58 vs 357


def test_stiff_adaptive_tolerance_scaling():
    # error control works: tightening rtol cuts the error (measured:
    # 8.9e-7 / 2.3e-8 / 2.4e-9 for rtol 1e-4 / 1e-6 / 1e-8 — ratio 375x)
    def err(rtol):
        _, ys = ode.solve_ode_stiff_adaptive(
            parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), rtol=rtol
        )
        return abs(ys[-1] - _stiff_exact(1.0))

    e4, e6, e8 = err(1e-4), err(1e-6), err(1e-8)
    assert e4 > e6 > e8
    assert e8 < e4 / 50.0


def test_variable_step_bdf2_formula_order_two():
    # VERIFICATION OF THE DERIVATION of variable-step BDF2 (coefficients dependent
    # on omega = h_k/h_{k-1}): on a stiff alternating grid h, 1.5h the formula
    # must preserve order 2 — halving the base cuts the error ~4x (measured:
    # 3.937 and 3.955 on y' = y·cos t vs exp(sin t))
    def solve_pattern(n_base):
        f = lambda t, y: y * math.cos(t)
        df = lambda t, y: math.cos(t)
        h_base = 5.0 / n_base
        ts = [0.0]
        k = 0
        while ts[-1] < 5.0 - 1e-12:
            h = h_base if k % 2 == 0 else 1.5 * h_base
            ts.append(min(ts[-1] + h, 5.0))
            k += 1

        def newton(t_new, const, coef, u0):
            u = u0
            for _ in range(50):
                res = u - coef * f(t_new, u) - const
                if abs(res) <= 1e-14 * max(1.0, abs(u)):
                    return u
                u = u - res / (1.0 - coef * df(t_new, u))
            return u

        ys = [1.0]
        ys.append(newton(ts[1], ys[0], ts[1] - ts[0], ys[0]))  # BDF1 startup
        for i in range(2, len(ts)):
            h_new, h_old = ts[i] - ts[i - 1], ts[i - 1] - ts[i - 2]
            w = h_new / h_old
            a1 = (1 + w) ** 2 / (1 + 2 * w)
            a2 = -(w * w) / (1 + 2 * w)
            coef = h_new * (1 + w) / (1 + 2 * w)
            ys.append(newton(ts[i], a1 * ys[-1] + a2 * ys[-2], coef, ys[-1]))
        return abs(ys[-1] - math.exp(math.sin(5.0)))

    ratio = solve_pattern(100) / solve_pattern(200)
    assert 3.5 < ratio < 4.5


def test_stiff_adaptive_smooth_and_backward():
    # smooth non-stiff: correctness (measured: 1.7e-6 at rtol 1e-8 —
    # order 2 + local error control accumulate more than DOPRI5)
    ts, ys = ode.solve_ode_stiff_adaptive(
        parse("y*cos(t)"), "t", 1.0, (0.0, 5.0), rtol=1e-8
    )
    assert ys[-1] == pytest.approx(math.exp(math.sin(5.0)), abs=1e-5)
    # backward (module 13 for free): roundtrip (measured: 1.1e-7)
    tb, yb = ode.solve_ode_stiff_adaptive(
        parse("y*cos(t)"), "t", ys[-1], (5.0, 0.0), rtol=1e-8
    )
    assert yb[-1] == pytest.approx(1.0, abs=1e-5)
    assert tb[-1] == 0.0  # lands exactly on t1


def test_stiff_adaptive_singularity_refuses():
    # y' = y², pole at t=1: through the pole — refusal; before the pole — hits
    # (measured: |y(0.9) - 10| = 5.8e-4 at rtol 1e-8 — the y'=y² growth
    # amplifies local errors, order 2)
    with pytest.raises(PycodemathError):
        ode.solve_ode_stiff_adaptive(parse("y^2"), "t", 1.0, (0.0, 2.0), rtol=1e-8)
    _, ys = ode.solve_ode_stiff_adaptive(parse("y^2"), "t", 1.0, (0.0, 0.9), rtol=1e-8)
    assert ys[-1] == pytest.approx(10.0, abs=5e-3)


def test_system_stiff_adaptive_linear_and_validation():
    # linear stiff system from m16 (λ=-1/-1000), exact solution
    # (measured: 313 steps, error 2.6e-5 at rtol 1e-6)
    ts, rows = ode.solve_ode_system_stiff_adaptive(
        _lin_stiff(), "t", ["x", "y"], [2.0, 0.0], (0.0, 1.0), rtol=1e-6
    )
    assert np.allclose(rows[-1], _lin_stiff_exact(1.0), atol=1e-4)
    assert len(ts) - 1 < 350
    # validations — everything PycodemathError
    with pytest.raises(PycodemathError):  # unsubstituted parameter a
        ode.solve_ode_stiff_adaptive(parse("a*y"), "t", 1.0, (0.0, 1.0))
    with pytest.raises(PycodemathError):  # zero-length interval
        ode.solve_ode_stiff_adaptive(parse("y"), "t", 1.0, (1.0, 1.0))
    with pytest.raises(PycodemathError):  # tolerances must be positive
        ode.solve_ode_stiff_adaptive(parse("y"), "t", 1.0, (0.0, 1.0), rtol=-1.0)
    with pytest.raises(PycodemathError):  # step budget exhausted
        ode.solve_ode_stiff_adaptive(
            parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), max_steps=3
        )
    with pytest.raises(PycodemathError):  # non-square system
        ode.solve_ode_system_stiff_adaptive(
            [parse("v")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0)
        )


def test_system_stiff_adaptive_van_der_pol_mu1000():
    # CLASSIC: Van der Pol μ=1000 — relaxation oscillation, stiffness
    # varying in time; a fixed step makes no economic sense here, an explicit
    # method would need h~1e-3 (≥2e6 steps on [0,2000]).
    # Measured: 5332 steps, max|x| = 2.000089, x(2000) = 1.706 (~1.1 s)
    mu = 1000.0
    vdp = [parse("v"), parse(f"{mu}*(1 - x^2)*v - x")]
    ts, rows = ode.solve_ode_system_stiff_adaptive(
        vdp, "t", ["x", "v"], [2.0, 0.0], (0.0, 2000.0), rtol=1e-6, atol=1e-9
    )
    assert ts[-1] == pytest.approx(2000.0)
    assert len(ts) - 1 < 20000  # orders of magnitude fewer than explicit stability would force
    xs = [abs(r[0]) for r in rows]
    assert 1.95 < max(xs) < 2.1  # limit cycle amplitude ~2
    assert all(np.isfinite(r).all() for r in np.array(rows))


# --- module 13: backward integration (t1 < t0) ------------------------------
def test_backward_rk4_roundtrip_returns_to_initial():
    # forward, then BACKWARD from the end -> return to the initial condition
    rhs = parse("y*cos(t)")
    _, ys_f = ode.solve_ode_num(rhs, "t", 1.0, (0.0, 5.0), 400)
    ts_b, ys_b = ode.solve_ode_num(rhs, "t", ys_f[-1], (5.0, 0.0), 400)
    # measured: |y0' - 1| = 2.5e-12
    assert ys_b[-1] == pytest.approx(1.0, abs=1e-9)
    # nodes strictly decrease and land exactly on t1
    assert all(b < a for a, b in zip(ts_b, ts_b[1:]))
    assert ts_b[-1] == 0.0


def test_backward_adaptive_and_dense():
    rhs = parse("y*cos(t)")
    exact5 = math.exp(math.sin(5.0))
    # adaptive roundtrip (measured: 1.1e-8 at rtol 1e-8)
    _, ya = ode.solve_ode_adaptive(rhs, "t", 1.0, (0.0, 5.0), rtol=1e-8)
    tb, yb = ode.solve_ode_adaptive(rhs, "t", ya[-1], (5.0, 0.0), rtol=1e-8)
    assert yb[-1] == pytest.approx(1.0, abs=1e-6)
    assert tb[-1] == 0.0
    # dense output backward: read BETWEEN nodes vs exp(sin t)
    sol = ode.solve_ode_dense(rhs, "t", exact5, (5.0, 0.0), rtol=1e-8)
    grid = np.linspace(0.0, 5.0, 173)
    assert np.allclose(sol(grid), np.exp(np.sin(grid)), atol=1e-6)
    # nodes reproduced exactly; the domain is [0, 5] despite t0=5 > t1=0
    for tk, yk in zip(sol.ts, sol.ys):
        assert sol(tk) == pytest.approx(yk, abs=1e-12)
    with pytest.raises(PycodemathError):
        sol(6.0)


def test_backward_events_and_terminal():
    # y' = cos(t) backward from t=10 (y=sin) — zeros passed in the order 3pi, 2pi, pi
    ev = ode.solve_ode_events(
        parse("cos(t)"), "t", math.sin(10.0), (10.0, 0.5), parse("y"), rtol=1e-8
    )
    roots = [3 * math.pi, 2 * math.pi, math.pi]  # order of TRAVERSAL (downward)
    assert np.allclose(ev.event_times, roots, atol=1e-6)
    # terminal backward: the first event from t=10 going down is 3pi
    st = ode.solve_ode_events(
        parse("cos(t)"), "t", math.sin(10.0), (10.0, 0.5), parse("y"),
        terminal=True, rtol=1e-8,
    )
    assert st.t1 == pytest.approx(3 * math.pi, abs=1e-6)


def test_backward_stiff_roundtrip():
    # BDF2 forward and backward (mild problem): return with error O(h²)
    rhs = parse("y*cos(t)")
    _, ys_f = ode.solve_ode_stiff(rhs, "t", 1.0, (0.0, 5.0), 400)
    _, ys_b = ode.solve_ode_stiff(rhs, "t", ys_f[-1], (5.0, 0.0), 400)
    # measured: 2.4e-4 (h=0.0125, O(h²) there and back)
    assert ys_b[-1] == pytest.approx(1.0, abs=1e-3)


def test_backward_system_dense_and_terminal():
    # oscillator backward from t=10: dense hits sin/cos, terminal x=0 at 3pi
    osc = [parse("v"), parse("-x")]
    y10 = [math.sin(10.0), math.cos(10.0)]
    sol = ode.solve_ode_system_dense(
        osc, "t", ["x", "v"], y10, (10.0, 0.5), rtol=1e-8
    )
    grid = np.linspace(0.5, 10.0, 131)
    assert np.allclose(
        sol(grid), np.stack([np.sin(grid), np.cos(grid)], axis=1), atol=1e-6
    )
    st = ode.solve_ode_system_events(
        osc, "t", ["x", "v"], y10, (10.0, 0.5), parse("x"),
        terminal=True, rtol=1e-8,
    )
    assert st.event_times[0] == pytest.approx(3 * math.pi, abs=1e-6)


def test_backward_singularity_refuses():
    # y' = y², y = 1/(1-t): start at t=2 (y=-1), backward to 0 — pole at t=1
    # on the way; backward integration also REFUSES to jump over the singularity
    with pytest.raises(PycodemathError):
        ode.solve_ode_adaptive(parse("y^2"), "t", -1.0, (2.0, 0.0), rtol=1e-8)


# --- audit before extension: tolerance validation ----------------------------
def test_dense_and_events_validate_tolerances():
    with pytest.raises(PycodemathError):
        ode.solve_ode_dense(parse("y"), "t", 1.0, (0.0, 1.0), rtol=-1.0)
    with pytest.raises(PycodemathError):
        ode.solve_ode_events(parse("y"), "t", 1.0, (0.0, 1.0), parse("y"), atol=0.0)
