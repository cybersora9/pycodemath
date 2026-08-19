"""Testy pakietu ODE: dsolve, RK4, adaptive/dense/events, stiff, wsteczne.

Rozbicie tests/test_pycodemath.py na pliki per obszar (krok 0 sesji
rozwojowej) — treść testów przeniesiona bez zmian.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycodemath import E, parse
from pycodemath.core.errors import PycodemathError
from pycodemath.engine import ode


# --- moduł 6: równania różniczkowe zwyczajne -----------------------------
def test_dsolve_exponential_growth():
    sols = ode.dsolve(parse("y"), "y", "t")
    assert len(sols) == 1
    assert sols[0].equivalent(E("C1*exp(t)"))


def test_dsolve_parametric_ode():
    # parametry (a) są dozwolone na ścieżce symbolicznej
    sols = ode.dsolve(parse("a*y"), "y", "t")
    assert sols[0].equivalent(E("C1*exp(a*t)"))


def test_dsolve_no_closed_form_suggests_numeric():
    # y' = y^y ma tylko rozwiązanie uwikłane — dsolve odsyła do numeryki
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
    # błąd globalny RK4 przy n=100 to ~2e-10 — 1e-8 z dużym zapasem
    assert ys[-1] == pytest.approx(math.e, abs=1e-8)
    assert ys[50] == pytest.approx(math.exp(0.5), abs=1e-8)


def test_solve_ode_num_polynomial_rhs_exact():
    # RK4 całkuje wielomiany stopnia <= 3 dokładnie: y' = 2t -> y = t^2
    ts, ys = ode.solve_ode_num(parse("2*t"), "t", 0.0, (0.0, 1.0), 100)
    assert ys[-1] == pytest.approx(1.0)


def test_solve_ode_num_divergence_raises():
    # y' = y^2, y(0)=1 wybucha w t=1 — przedział [0,2] musi zgłosić błąd
    with pytest.raises(PycodemathError):
        ode.solve_ode_num(parse("y^2"), "t", 1.0, (0.0, 2.0), 100)


def test_solve_ode_num_validates_input():
    with pytest.raises(PycodemathError):  # niepodstawiony parametr a
        ode.solve_ode_num(parse("a*y"), "t", 1.0, (0.0, 1.0), 10)
    with pytest.raises(PycodemathError):  # n musi być dodatnie
        ode.solve_ode_num(parse("y"), "t", 1.0, (0.0, 1.0), 0)
    with pytest.raises(PycodemathError):  # przedział zerowej długości
        ode.solve_ode_num(parse("y"), "t", 1.0, (1.0, 1.0), 10)


def test_solve_ode_system_num_oscillator_conserves_energy():
    # x'' = -x jako układ: x' = v, v' = -x; E = (x^2 + v^2)/2 = const
    ts, rows = ode.solve_ode_system_num(
        [parse("v"), parse("-x")], "t", ["x", "v"], [1.0, 0.0],
        (0.0, 2 * math.pi), 1000,
    )
    x, v = rows[-1]
    assert 0.5 * (x * x + v * v) == pytest.approx(0.5, abs=1e-6)
    # po pełnym okresie wracamy do stanu początkowego
    assert np.allclose([x, v], [1.0, 0.0], atol=1e-6)


def test_solve_ode_system_num_validation():
    with pytest.raises(PycodemathError):  # 2 równania, 1 funkcja
        ode.solve_ode_system_num(
            [parse("v"), parse("-x")], "t", ["x"], [1.0], (0.0, 1.0), 10
        )
    with pytest.raises(PycodemathError):  # y0 złej długości
        ode.solve_ode_system_num(
            [parse("v"), parse("-x")], "t", ["x", "v"], [1.0], (0.0, 1.0), 10
        )


# --- moduł 7: adaptacyjny solver ODE (Dormand-Prince 5(4)) ---------------
def test_solve_ode_adaptive_hits_tolerance():
    # y' = y, y(0)=1 -> exp(t); krok dobierany automatycznie
    ts, ys = ode.solve_ode_adaptive(parse("y"), "t", 1.0, (0.0, 1.0), rtol=1e-8)
    assert ts[0] == 0.0 and ts[-1] == pytest.approx(1.0)  # węzły od t0 do t1
    assert len(ts) == len(ys)
    assert ys[0] == 1.0
    assert ys[-1] == pytest.approx(math.e, abs=1e-6)
    # węzły ściśle rosną (krok dodatni na całym przedziale)
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_rk4_convergence_order_four():
    # globalny błąd RK4 to O(h^4): podwojenie n tnie błąd ~16x (2^4)
    def err(n):
        _, ys = ode.solve_ode_num(parse("y"), "t", 1.0, (0.0, 1.0), n)
        return abs(ys[-1] - math.e)

    ratio = err(20) / err(40)
    assert 14.0 < ratio < 17.0  # zmierzone ~15.7, dąży do 16


def test_solve_ode_adaptive_fewer_steps_than_rk4():
    # gładki problem y' = y*cos(t) -> exp(sin(t)); adaptacja trafia w tolerancję
    # mniejszą liczbą kroków niż RK4 (dla tego samego budżetu RK4 jest gorszy)
    rhs, span = parse("y*cos(t)"), (0.0, 5.0)
    exact = math.exp(math.sin(5.0))
    ts, ys = ode.solve_ode_adaptive(rhs, "t", 1.0, span, rtol=1e-6, atol=1e-9)
    n_adapt = len(ts) - 1
    err_adapt = abs(ys[-1] - exact)
    assert err_adapt < 1e-5  # trafił w zadaną tolerancję
    # RK4 z tą samą liczbą kroków jest wyraźnie mniej dokładny (rząd 4 vs 5,
    # brak zagęszczania) — więc na daną tolerancję potrzebuje WIĘCEJ kroków
    _, ys_rk4 = ode.solve_ode_num(rhs, "t", 1.0, span, n_adapt)
    assert abs(ys_rk4[-1] - exact) > err_adapt


def test_solve_ode_adaptive_singularity_refuses():
    # y' = y^2, y(0)=1 -> y = 1/(1-t), biegun w t=1
    # PRZED biegunem: adaptacja dokładnie odwzorowuje rozwiązanie
    ts, ys = ode.solve_ode_adaptive(parse("y^2"), "t", 1.0, (0.0, 0.9), rtol=1e-8)
    assert ys[-1] == pytest.approx(1.0 / (1.0 - 0.9), rel=1e-6)
    # PRZEZ biegun: krok kurczy się przy osobliwości, aż całkowanie ODMAWIA
    # (zamiast cicho przeskoczyć przez t=1 i zwrócić śmieci)
    with pytest.raises(PycodemathError):
        ode.solve_ode_adaptive(parse("y^2"), "t", 1.0, (0.0, 2.0), rtol=1e-8)


def test_solve_ode_adaptive_validates_input():
    with pytest.raises(PycodemathError):  # niepodstawiony parametr a
        ode.solve_ode_adaptive(parse("a*y"), "t", 1.0, (0.0, 1.0))
    with pytest.raises(PycodemathError):  # przedział zerowej długości
        ode.solve_ode_adaptive(parse("y"), "t", 1.0, (1.0, 1.0))
    with pytest.raises(PycodemathError):  # tolerancje muszą być dodatnie
        ode.solve_ode_adaptive(parse("y"), "t", 1.0, (0.0, 1.0), rtol=-1.0)


# --- moduł 8: gęste wyjście DOPRI5 (dense output) ------------------------
def test_solve_ode_dense_interpolates_between_nodes():
    # y' = y -> exp(t); interpolant musi trafiać MIĘDZY węzłami, nie tylko w nich
    rtol = 1e-8
    sol = ode.solve_ode_dense(parse("y"), "t", 1.0, (0.0, 1.0), rtol=rtol)
    assert isinstance(sol, ode.DenseSolution)
    assert sol.t0 == 0.0 and sol.t1 == pytest.approx(1.0)
    # punkty w ŚRODKU każdego kroku (nie węzły) — błąd zgodny z rtol
    for a, b in zip(sol.ts, sol.ts[1:]):
        tm = 0.5 * (a + b)
        assert sol(tm) == pytest.approx(math.exp(tm), abs=1e-6)
    # gęsta siatka niezależna od węzłów też trafia w rozwiązanie dokładne
    grid = np.linspace(0.0, 1.0, 101)
    assert np.allclose(sol(grid), np.exp(grid), atol=1e-6)


def test_solve_ode_dense_nodes_match_adaptive():
    # gęste wyjście dzieli silnik z solve_ode_adaptive -> te same węzły
    rhs, span = parse("y*cos(t)"), (0.0, 5.0)
    sol = ode.solve_ode_dense(rhs, "t", 1.0, span, rtol=1e-7)
    ts, ys = ode.solve_ode_adaptive(rhs, "t", 1.0, span, rtol=1e-7)
    assert np.allclose(sol.ts, ts) and np.allclose(sol.ys, ys)
    # w węzłach interpolant odtwarza próbki DOKŁADNIE (θ=0 -> y_old, θ=1 -> y_new)
    for tk, yk in zip(sol.ts, sol.ys):
        assert sol(tk) == pytest.approx(yk, abs=1e-12)


def test_solve_ode_dense_scalar_and_array_and_range():
    sol = ode.solve_ode_dense(parse("y"), "t", 1.0, (0.0, 2.0), rtol=1e-8)
    assert isinstance(sol(1.0), float)  # skalar -> float
    out = sol(np.array([0.5, 1.0, 1.5]))  # sekwencja -> ndarray
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    assert np.allclose(out, np.exp([0.5, 1.0, 1.5]), atol=1e-6)
    with pytest.raises(PycodemathError):  # poza [t0, t1]
        sol(3.0)


def test_solve_ode_dense_validates_input():
    with pytest.raises(PycodemathError):  # niepodstawiony parametr a
        ode.solve_ode_dense(parse("a*y"), "t", 1.0, (0.0, 1.0))
    with pytest.raises(PycodemathError):  # przedział zerowej długości
        ode.solve_ode_dense(parse("y"), "t", 1.0, (1.0, 1.0))


# --- moduł 9: wykrywanie zdarzeń (event detection) -----------------------
# y' = cos(t), y(0.5)=sin(0.5) -> y = sin(t); zera y w π, 2π, 3π
_EV_Y0 = math.sin(0.5)


def test_solve_ode_events_finds_analytic_roots():
    sol = ode.solve_ode_events(
        parse("cos(t)"), "t", _EV_Y0, (0.5, 10.0), parse("y"), rtol=1e-8
    )
    assert isinstance(sol, ode.EventSolution)
    assert callable(sol) and sol(1.0) == pytest.approx(math.sin(1.0), abs=1e-6)
    # trzy zera sin(t) w przedziale: π, 2π, 3π — z tolerancją zgodną z rtol
    roots = [math.pi, 2 * math.pi, 3 * math.pi]
    assert len(sol.events) == 3
    assert np.allclose(sol.event_times, roots, atol=1e-6)
    # w zdarzeniu funkcja zdarzenia (tu y) jest ~zerowa
    assert all(abs(y) < 1e-6 for y in sol.event_values)


def test_solve_ode_events_direction_filters():
    common = dict(rtol=1e-8)
    # narastające (−→+): tylko 2π; opadające (+→−): π i 3π
    up = ode.solve_ode_events(
        parse("cos(t)"), "t", _EV_Y0, (0.5, 10.0), parse("y"), direction=1, **common
    )
    down = ode.solve_ode_events(
        parse("cos(t)"), "t", _EV_Y0, (0.5, 10.0), parse("y"), direction=-1, **common
    )
    assert np.allclose(up.event_times, [2 * math.pi], atol=1e-6)
    assert np.allclose(down.event_times, [math.pi, 3 * math.pi], atol=1e-6)


def test_solve_ode_events_none_when_no_crossing():
    # g = y + 2 = sin(t) + 2 nigdy nie osiąga zera -> brak zdarzeń
    sol = ode.solve_ode_events(
        parse("cos(t)"), "t", _EV_Y0, (0.5, 10.0), parse("y+2"), rtol=1e-8
    )
    assert sol.events == []
    # obiekt nadal jest sprawnym gęstym wyjściem
    assert sol(math.pi) == pytest.approx(math.sin(math.pi), abs=1e-6)


def test_solve_ode_events_validates_input():
    with pytest.raises(PycodemathError):  # direction spoza {-1,0,1}
        ode.solve_ode_events(
            parse("cos(t)"), "t", 0.0, (0.0, 1.0), parse("y"), direction=2
        )
    with pytest.raises(PycodemathError):  # funkcja zdarzenia z parametrem
        ode.solve_ode_events(parse("cos(t)"), "t", 0.0, (0.0, 1.0), parse("a*y"))


# --- moduł 10: zdarzenia terminalne (terminal=True) -----------------------
def test_solve_ode_events_terminal_stops_at_event():
    # wzorcowy z promptu: y' = -1, y(0)=1 -> y = 1 - t, zero w t=1 (nie w t1=3)
    sol = ode.solve_ode_events(
        parse("-1"), "t", 1.0, (0.0, 3.0), parse("y"), terminal=True, rtol=1e-8
    )
    # zmierzone: |t_ev - 1| ~ 2.4e-14, |y_ev| ~ 2.4e-14 (interpolant dokładny
    # dla problemu liniowego) — tolerancje z ogromnym zapasem
    assert len(sol.events) == 1
    assert sol.event_times[0] == pytest.approx(1.0, abs=1e-10)
    assert abs(sol.event_values[0]) < 1e-10
    # węzły i gęste wyjście kończą się DOKŁADNIE na zdarzeniu, nie na t1
    assert sol.ts[-1] == sol.event_times[0]
    assert sol.t1 == sol.event_times[0]
    # interpolant spójny na skróconym przedziale...
    assert sol(0.5) == pytest.approx(0.5, abs=1e-9)
    with pytest.raises(PycodemathError):  # ...a za zdarzeniem już nie ma nic
        sol(2.0)


def test_terminal_event_respects_direction():
    # y = sin(t) na [0.5, 10]: pierwsze narastające zero to 2π, opadające π
    y0 = math.sin(0.5)
    up = ode.solve_ode_events(
        parse("cos(t)"), "t", y0, (0.5, 10.0), parse("y"),
        direction=1, terminal=True, rtol=1e-8,
    )
    down = ode.solve_ode_events(
        parse("cos(t)"), "t", y0, (0.5, 10.0), parse("y"),
        direction=-1, terminal=True, rtol=1e-8,
    )
    # zmierzone: |t-2π| ~ 6.3e-9, |t-π| ~ 2.7e-10 (~rtol)
    assert up.event_times == pytest.approx([2 * math.pi], abs=1e-6)
    assert down.event_times == pytest.approx([math.pi], abs=1e-6)


def test_terminal_event_stops_before_singularity():
    # y' = y², y(0)=1 -> y = 1/(1-t), biegun w t=1; zdarzenie y=10 w t=0.9.
    # Zdarzenie sprawdzane W TRAKCIE całkowania — stop PRZED biegunem,
    # choć t1=2 leży daleko za nim (zmierzone: |t_ev-0.9| ~ 3.1e-10)
    sol = ode.solve_ode_events(
        parse("y^2"), "t", 1.0, (0.0, 2.0), parse("y-10"), terminal=True, rtol=1e-8
    )
    assert sol.event_times[0] == pytest.approx(0.9, abs=1e-7)
    assert sol.event_values[0] == pytest.approx(10.0, abs=1e-6)
    # bez terminal ten sam przedział MUSI odmówić (biegun w środku)
    with pytest.raises(PycodemathError):
        ode.solve_ode_events(
            parse("y^2"), "t", 1.0, (0.0, 2.0), parse("y-10"), rtol=1e-8
        )


def test_terminal_no_crossing_runs_full_span():
    # g = y + 2 nigdy nie zeruje -> pełny przedział, pusta lista (jak moduł 9)
    y0 = math.sin(0.5)
    sol = ode.solve_ode_events(
        parse("cos(t)"), "t", y0, (0.5, 10.0), parse("y+2"), terminal=True, rtol=1e-8
    )
    assert sol.events == []
    assert sol.t1 == pytest.approx(10.0)
    # bez terminal zachowanie modułu 9 bez zmian (ten sam kod — 3 zdarzenia)
    ref = ode.solve_ode_events(
        parse("cos(t)"), "t", y0, (0.5, 10.0), parse("y"), rtol=1e-8
    )
    assert len(ref.events) == 3 and ref.t1 == pytest.approx(10.0)


# --- moduł 11: układy dla dense/events ------------------------------------
# oscylator x' = v, v' = -x, start w t0=0.5: x = sin(t), v = cos(t)
_OSC_RHS = ["v", "-x"]
_OSC_Y0 = [math.sin(0.5), math.cos(0.5)]


def _osc():
    return [parse(e) for e in _OSC_RHS]


def test_solve_ode_system_dense_oscillator():
    sol = ode.solve_ode_system_dense(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), rtol=1e-8
    )
    assert isinstance(sol, ode.DenseSystemSolution)
    # skalar -> wektor stanu (x, v); zmierzone: max błąd siatki 1.4e-8
    mid = sol(1.234)
    assert isinstance(mid, np.ndarray) and mid.shape == (2,)
    assert np.allclose(mid, [math.sin(1.234), math.cos(1.234)], atol=1e-6)
    # sekwencja -> tablica (m, n); dokładność między węzłami ~rtol
    grid = np.linspace(0.5, 10.0, 197)
    got = sol(grid)
    assert got.shape == (197, 2)
    assert np.allclose(got, np.stack([np.sin(grid), np.cos(grid)], axis=1), atol=1e-6)
    # energia (x²+v²)/2 = 0.5 zachowana też MIĘDZY węzłami (zmierzone 1.5e-8)
    energy = 0.5 * (got[:, 0] ** 2 + got[:, 1] ** 2)
    assert np.allclose(energy, 0.5, atol=1e-6)
    # w węzłach interpolant odtwarza próbki DOKŁADNIE (zmierzone: 0.0)
    for tk, yk in zip(sol.ts, sol.ys):
        assert np.allclose(sol(tk), yk, atol=1e-12)


def test_solve_ode_system_events_oscillator_hits_pi():
    # zdarzenie x = 0 w wielokrotnościach π; stan zdarzenia (x≈0, v=±1)
    sol = ode.solve_ode_system_events(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), parse("x"), rtol=1e-8
    )
    assert isinstance(sol, ode.EventSystemSolution)
    roots = [math.pi, 2 * math.pi, 3 * math.pi]
    # zmierzone błędy t: ≤ 2.9e-9 (~rtol)
    assert np.allclose(sol.event_times, roots, atol=1e-6)
    # stan w zdarzeniu: x ~ 1e-13, v = cos(kπ) = -1, +1, -1 (błąd ≤ 1.4e-8)
    for state, v_exact in zip(sol.event_values, [-1.0, 1.0, -1.0]):
        assert abs(state[0]) < 1e-9
        assert state[1] == pytest.approx(v_exact, abs=1e-6)
    # zdarzenie na DRUGIEJ współrzędnej: v = cos(t) = 0 w π/2 + kπ
    on_v = ode.solve_ode_system_events(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), parse("v"), rtol=1e-8
    )
    v_roots = [math.pi / 2, 3 * math.pi / 2, 5 * math.pi / 2]
    assert np.allclose(on_v.event_times, v_roots, atol=1e-6)


def test_solve_ode_system_events_direction_and_terminal():
    # dir=+1: x rośnie przez zero dopiero w 2π
    up = ode.solve_ode_system_events(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), parse("x"),
        direction=1, rtol=1e-8,
    )
    assert np.allclose(up.event_times, [2 * math.pi], atol=1e-6)
    # terminal: stop na PIERWSZYM zdarzeniu (π) — węzły kończą się na zdarzeniu
    stopped = ode.solve_ode_system_events(
        _osc(), "t", ["x", "v"], _OSC_Y0, (0.5, 10.0), parse("x"),
        terminal=True, rtol=1e-8,
    )
    assert stopped.t1 == pytest.approx(math.pi, abs=1e-6)
    assert stopped.ts[-1] == stopped.event_times[0]
    assert stopped.event_values[0][1] == pytest.approx(-1.0, abs=1e-6)
    with pytest.raises(PycodemathError):  # za zdarzeniem nie ma nic
        stopped(5.0)


def test_solve_ode_system_dense_events_validate_input():
    with pytest.raises(PycodemathError):  # układ niekwadratowy
        ode.solve_ode_system_dense(
            [parse("v")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0)
        )
    with pytest.raises(PycodemathError):  # y0 złej długości
        ode.solve_ode_system_dense(_osc(), "t", ["x", "v"], [1.0], (0.0, 1.0))
    with pytest.raises(PycodemathError):  # funkcja zdarzenia z parametrem
        ode.solve_ode_system_events(
            _osc(), "t", ["x", "v"], _OSC_Y0, (0.0, 1.0), parse("a*x")
        )
    with pytest.raises(PycodemathError):  # tolerancje muszą być dodatnie
        ode.solve_ode_system_dense(
            _osc(), "t", ["x", "v"], _OSC_Y0, (0.0, 1.0), rtol=-1.0
        )
    with pytest.raises(PycodemathError):  # direction spoza {-1,0,1}
        ode.solve_ode_system_events(
            _osc(), "t", ["x", "v"], _OSC_Y0, (0.0, 1.0), parse("x"), direction=5
        )


def test_scalar_dense_events_unchanged_after_system_refactor():
    # skalarne API bez zmian po uogólnieniu (_eval_vec pod spodem):
    # sol(t) to nadal float, zdarzenia to nadal (float, float)
    sol = ode.solve_ode_dense(parse("y"), "t", 1.0, (0.0, 1.0), rtol=1e-8)
    assert isinstance(sol(0.5), float)
    ev = ode.solve_ode_events(
        parse("cos(t)"), "t", math.sin(0.5), (0.5, 10.0), parse("y"), rtol=1e-8
    )
    assert all(isinstance(t, float) and isinstance(y, float) for t, y in ev.events)
    assert np.allclose(ev.event_times, [math.pi, 2 * math.pi, 3 * math.pi], atol=1e-6)


# --- moduł 12: metody sztywne (BDF2, niejawna) ----------------------------
# y' = -1000·(y - cos(t)), y(0)=0 — sztywne: skala tłumienia (1/1000) o rzędy
# krótsza niż skala rozwiązania. Rozwiązanie dokładne:
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
    # BDF2 z h=0.01 (h·λ = -10 — daleko za granicą stabilności metod jawnych)
    ts, ys = ode.solve_ode_stiff(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), 100)
    assert len(ts) == 101 == len(ys)
    assert ts[-1] == pytest.approx(1.0)
    # zmierzone: błąd 2.8e-8 — tolerancja z dużym zapasem
    assert ys[-1] == pytest.approx(_stiff_exact(1.0), abs=1e-6)


def test_stiff_beats_explicit_at_same_budget():
    # ten sam budżet kroków: BDF2 trafia w rozwiązanie, jawny RK4 jest
    # bezużyteczny — niestabilność albo eksploduje (PycodemathError), albo
    # zwraca astronomiczne śmieci (zmierzone: wzmocnienie ~291^100 ≈ 1e246)
    _, ys = ode.solve_ode_stiff(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), 100)
    err_bdf2 = abs(ys[-1] - _stiff_exact(1.0))
    assert err_bdf2 < 1e-6
    try:
        _, ys_rk4 = ode.solve_ode_num(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), 100)
        assert abs(ys_rk4[-1] - _stiff_exact(1.0)) > 1e50  # skończone śmieci
    except PycodemathError:
        pass  # albo uczciwa odmowa przy przepełnieniu do inf


def test_stiff_convergence_order_two():
    # BDF2 jest rzędu 2: podwojenie n tnie błąd ~4x (zmierzone 3.99, 3.995)
    def err(n):
        _, ys = ode.solve_ode_stiff(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), n)
        return abs(ys[-1] - _stiff_exact(1.0))

    ratio = err(100) / err(200)
    assert 3.5 < ratio < 4.5


def test_stiff_nonlinear_agrees_with_smooth_solution():
    # niesztywne nieliniowe y' = y·cos(t) -> exp(sin(t)) — Newton na kroku
    # niejawnym iteruje (problem nieliniowy), wynik zgodny z dokładnym
    # (zmierzone: błąd 2.3e-5 przy n=400, O(h²))
    ts, ys = ode.solve_ode_stiff(parse("y*cos(t)"), "t", 1.0, (0.0, 5.0), 400)
    assert ys[-1] == pytest.approx(math.exp(math.sin(5.0)), abs=1e-4)


def test_solve_ode_stiff_validates_and_refuses():
    with pytest.raises(PycodemathError):  # niepodstawiony parametr a
        ode.solve_ode_stiff(parse("a*y"), "t", 1.0, (0.0, 1.0), 10)
    with pytest.raises(PycodemathError):  # n musi być dodatnie
        ode.solve_ode_stiff(parse("y"), "t", 1.0, (0.0, 1.0), 0)
    with pytest.raises(PycodemathError):  # przedział zerowej długości
        ode.solve_ode_stiff(parse("y"), "t", 1.0, (1.0, 1.0), 10)
    with pytest.raises(PycodemathError):  # y' = y², biegun w t=1 — odmowa
        ode.solve_ode_stiff(parse("y^2"), "t", 1.0, (0.0, 2.0), 100)


# --- moduł 16: układy sztywne (solve_ode_system_stiff) --------------------
# liniowy układ sztywny x' = A·x, A = [[-500.5, 499.5],[499.5, -500.5]]:
# wartości własne -1 (wektor [1,1]) i -1000 (wektor [1,-1]);
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
    # BDF2 z h=0.01 (h·λ = -10 dla szybkiego modu) trafia w rozwiązanie
    # dokładne; zmierzone: błąd 1.54e-5 — tolerancja z zapasem
    ts, rows = ode.solve_ode_system_stiff(
        _lin_stiff(), "t", ["x", "y"], [2.0, 0.0], (0.0, 1.0), 100
    )
    assert len(ts) == 101 == len(rows)
    assert ts[-1] == pytest.approx(1.0)
    assert np.allclose(rows[-1], _lin_stiff_exact(1.0), atol=1e-4)


def test_system_stiff_beats_explicit_at_same_budget():
    # kontrast jak w module 12: BDF2 trafia, jawny RK4 z tym samym budżetem
    # jest bezużyteczny (zmierzone: skończone śmieci 2.45e246 — wzmocnienie
    # ~291^100 mieści się w float64, guard isfinite nie ma czego łapać)
    assert _lin_stiff_err(100) < 1e-4
    try:
        _, rows = ode.solve_ode_system_num(
            _lin_stiff(), "t", ["x", "y"], [2.0, 0.0], (0.0, 1.0), 100
        )
        assert abs(rows[-1][0]) > 1e50  # skończone śmieci
    except PycodemathError:
        pass  # albo uczciwa odmowa przy przepełnieniu do inf


def test_system_stiff_convergence_order_two():
    # BDF2 jest rzędu 2 także wektorowo: podwojenie n tnie błąd ~4x
    # (zmierzone: 4.013 i 4.006)
    ratio = _lin_stiff_err(100) / _lin_stiff_err(200)
    assert 3.5 < ratio < 4.5


def test_system_stiff_oscillator_correctness():
    # niesztywny oscylator: poprawność Newtona nd na macierzy 2x2
    # (zmierzone: błąd 5.1e-4 po pełnym okresie przy n=400, O(h²))
    _, rows = ode.solve_ode_system_stiff(
        [parse("v"), parse("-x")], "t", ["x", "v"], [0.0, 1.0],
        (0.0, 2 * math.pi), 400,
    )
    assert np.allclose(rows[-1], [0.0, 1.0], atol=5e-3)


def test_system_stiff_nonlinear_van_der_pol():
    # Van der Pol μ=2 — nieliniowy sprzężony układ: Newton nd ITERUJE
    # (jakobian zależy od stanu); referencja = dense rtol 1e-10
    # (zmierzone: błąd 3.9e-4 przy n=2000)
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
    # wsteczne (t1 < t0) działa za darmo po module 13 — na ŁAGODNYM układzie
    # (oscylator; zmierzone: 3.8e-4 tam i z powrotem, O(h²)).
    # Uwaga: wstecz na układzie SZTYWNYM to zadanie źle postawione — szybki
    # mod e^-1000t wstecz rośnie jak e^+1000t i wzmacnia błąd zaokrągleń
    # (zmierzone: śmieci ~1.9e9); to cecha zadania, nie solvera.
    osc = [parse("v"), parse("-x")]
    _, rf = ode.solve_ode_system_stiff(
        osc, "t", ["x", "v"], [0.0, 1.0], (0.0, 2 * math.pi), 400
    )
    _, rb = ode.solve_ode_system_stiff(
        [parse("v"), parse("-x")], "t", ["x", "v"], rf[-1], (2 * math.pi, 0.0), 400
    )
    assert np.allclose(rb[-1], [0.0, 1.0], atol=5e-3)


def test_system_stiff_scalar_consistency_and_validation():
    # układ 1-równaniowy liczy TO SAMO co skalarny solve_ode_stiff
    # (zmierzone: max|Δy| = 0.0 — ta sama BDF2, Newton nd 1x1 == skalarny)
    rhs_scalar = parse("-1000*(y - cos(t))")
    _, ys = ode.solve_ode_stiff(rhs_scalar, "t", 0.0, (0.0, 1.0), 100)
    _, rows = ode.solve_ode_system_stiff(
        [parse("-1000*(y - cos(t))")], "t", ["y"], [0.0], (0.0, 1.0), 100
    )
    assert np.allclose([r[0] for r in rows], ys, atol=1e-12)
    # walidacje i odmowy — wszystko PycodemathError
    with pytest.raises(PycodemathError):  # układ niekwadratowy
        ode.solve_ode_system_stiff([parse("v")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0))
    with pytest.raises(PycodemathError):  # y0 złej długości
        ode.solve_ode_system_stiff(
            [parse("v"), parse("-x")], "t", ["x", "v"], [1.0], (0.0, 1.0)
        )
    with pytest.raises(PycodemathError):  # niepodstawiony parametr a
        ode.solve_ode_system_stiff(
            [parse("a*x"), parse("-x")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0)
        )
    with pytest.raises(PycodemathError):  # n musi być dodatnie
        ode.solve_ode_system_stiff(
            [parse("v"), parse("-x")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0), 0
        )
    with pytest.raises(PycodemathError):  # przedział zerowej długości
        ode.solve_ode_system_stiff(
            [parse("v"), parse("-x")], "t", ["x", "v"], [1.0, 0.0], (1.0, 1.0)
        )
    with pytest.raises(PycodemathError):  # y' = y², biegun w t=1 — odmowa
        ode.solve_ode_system_stiff([parse("y^2")], "t", ["y"], [1.0], (0.0, 2.0), 100)


# --- moduł 17: adaptacyjny krok dla BDF ------------------------------------
def test_stiff_adaptive_matches_exact_and_beats_dopri5_on_anchor():
    # problem wzorcowy m12: y' = -1000(y - cos t), y(0)=0, rtol=1e-6.
    # KOTWICA z m12: DOPRI5 potrzebuje 378 kroków (limit STABILNOŚCI).
    # Zmierzone: BDF adaptacyjna 313 kroków, błąd 2.3e-8 — mniej niż DOPRI5,
    # ale skromnie: przy y(0)=0 OBA solvery muszą rozwiązać transjent
    # e^-1000t do tolerancji (patrz test wolnej rozmaitości niżej).
    rhs = parse(_STIFF_RHS)
    ts_b, ys_b = ode.solve_ode_stiff_adaptive(rhs, "t", 0.0, (0.0, 1.0), rtol=1e-6)
    assert ys_b[-1] == pytest.approx(_stiff_exact(1.0), abs=1e-6)
    assert ts_b[0] == 0.0 and ts_b[-1] == pytest.approx(1.0)
    ts_d, _ = ode.solve_ode_adaptive(parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), rtol=1e-6)
    assert len(ts_b) < len(ts_d)  # zmierzone: 313 < 378


def test_stiff_adaptive_slow_manifold_far_fewer_steps():
    # start NA wolnej rozmaitości (y(0) = A -> zero transjentu): sztywność
    # w czystej postaci — DOPRI5 dalej trzyma limit stabilności (zmierzone
    # 357 kroków), BDF adaptacyjną ogranicza tylko DOKŁADNOŚĆ (zmierzone 58,
    # błąd 2.8e-8) — ISTOTNIE mniej kroków (>4x)
    rhs = parse(_STIFF_RHS)
    slow_exact = _STIFF_A * math.cos(1.0) + _STIFF_B * math.sin(1.0)
    ts_b, ys_b = ode.solve_ode_stiff_adaptive(
        rhs, "t", _STIFF_A, (0.0, 1.0), rtol=1e-6
    )
    assert ys_b[-1] == pytest.approx(slow_exact, abs=1e-6)
    ts_d, _ = ode.solve_ode_adaptive(
        parse(_STIFF_RHS), "t", _STIFF_A, (0.0, 1.0), rtol=1e-6
    )
    assert len(ts_b) - 1 < (len(ts_d) - 1) / 4  # zmierzone: 58 vs 357


def test_stiff_adaptive_tolerance_scaling():
    # kontrola błędu działa: zaostrzanie rtol tnie błąd (zmierzone:
    # 8.9e-7 / 2.3e-8 / 2.4e-9 dla rtol 1e-4 / 1e-6 / 1e-8 — stosunek 375x)
    def err(rtol):
        _, ys = ode.solve_ode_stiff_adaptive(
            parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), rtol=rtol
        )
        return abs(ys[-1] - _stiff_exact(1.0))

    e4, e6, e8 = err(1e-4), err(1e-6), err(1e-8)
    assert e4 > e6 > e8
    assert e8 < e4 / 50.0


def test_variable_step_bdf2_formula_order_two():
    # WERYFIKACJA WYPROWADZENIA zmiennokrokowej BDF2 (współczynniki zależne
    # od ω = h_k/h_{k-1}): na sztywnej siatce naprzemiennej h, 1.5h formuła
    # musi zachować rząd 2 — połówkowanie bazy tnie błąd ~4x (zmierzone:
    # 3.937 i 3.955 na y' = y·cos t vs exp(sin t))
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
        ys.append(newton(ts[1], ys[0], ts[1] - ts[0], ys[0]))  # rozbieg BDF1
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
    # gładki niesztywny: poprawność (zmierzone: 1.7e-6 przy rtol 1e-8 —
    # rząd 2 + lokalna kontrola błędu akumulują więcej niż DOPRI5)
    ts, ys = ode.solve_ode_stiff_adaptive(
        parse("y*cos(t)"), "t", 1.0, (0.0, 5.0), rtol=1e-8
    )
    assert ys[-1] == pytest.approx(math.exp(math.sin(5.0)), abs=1e-5)
    # wstecz (moduł 13 za darmo): roundtrip (zmierzone: 1.1e-7)
    tb, yb = ode.solve_ode_stiff_adaptive(
        parse("y*cos(t)"), "t", ys[-1], (5.0, 0.0), rtol=1e-8
    )
    assert yb[-1] == pytest.approx(1.0, abs=1e-5)
    assert tb[-1] == 0.0  # ląduje dokładnie na t1


def test_stiff_adaptive_singularity_refuses():
    # y' = y², biegun w t=1: przez biegun — odmowa; przed biegunem — trafia
    # (zmierzone: |y(0.9) - 10| = 5.8e-4 przy rtol 1e-8 — wzrost y'=y²
    # wzmacnia błędy lokalne, rząd 2)
    with pytest.raises(PycodemathError):
        ode.solve_ode_stiff_adaptive(parse("y^2"), "t", 1.0, (0.0, 2.0), rtol=1e-8)
    _, ys = ode.solve_ode_stiff_adaptive(parse("y^2"), "t", 1.0, (0.0, 0.9), rtol=1e-8)
    assert ys[-1] == pytest.approx(10.0, abs=5e-3)


def test_system_stiff_adaptive_linear_and_validation():
    # układ liniowy sztywny z m16 (λ=-1/-1000), rozwiązanie dokładne
    # (zmierzone: 313 kroków, błąd 2.6e-5 przy rtol 1e-6)
    ts, rows = ode.solve_ode_system_stiff_adaptive(
        _lin_stiff(), "t", ["x", "y"], [2.0, 0.0], (0.0, 1.0), rtol=1e-6
    )
    assert np.allclose(rows[-1], _lin_stiff_exact(1.0), atol=1e-4)
    assert len(ts) - 1 < 350
    # walidacje — wszystko PycodemathError
    with pytest.raises(PycodemathError):  # niepodstawiony parametr a
        ode.solve_ode_stiff_adaptive(parse("a*y"), "t", 1.0, (0.0, 1.0))
    with pytest.raises(PycodemathError):  # przedział zerowej długości
        ode.solve_ode_stiff_adaptive(parse("y"), "t", 1.0, (1.0, 1.0))
    with pytest.raises(PycodemathError):  # tolerancje muszą być dodatnie
        ode.solve_ode_stiff_adaptive(parse("y"), "t", 1.0, (0.0, 1.0), rtol=-1.0)
    with pytest.raises(PycodemathError):  # budżet kroków wyczerpany
        ode.solve_ode_stiff_adaptive(
            parse(_STIFF_RHS), "t", 0.0, (0.0, 1.0), max_steps=3
        )
    with pytest.raises(PycodemathError):  # układ niekwadratowy
        ode.solve_ode_system_stiff_adaptive(
            [parse("v")], "t", ["x", "v"], [1.0, 0.0], (0.0, 1.0)
        )


def test_system_stiff_adaptive_van_der_pol_mu1000():
    # KLASYK: Van der Pol μ=1000 — oscylacja relaksacyjna, sztywność
    # zmienna w czasie; stały krok jest tu ekonomicznie bez sensu, jawna
    # metoda potrzebowałaby h~1e-3 (≥2e6 kroków na [0,2000]).
    # Zmierzone: 5332 kroki, max|x| = 2.000089, x(2000) = 1.706 (~1.1 s)
    mu = 1000.0
    vdp = [parse("v"), parse(f"{mu}*(1 - x^2)*v - x")]
    ts, rows = ode.solve_ode_system_stiff_adaptive(
        vdp, "t", ["x", "v"], [2.0, 0.0], (0.0, 2000.0), rtol=1e-6, atol=1e-9
    )
    assert ts[-1] == pytest.approx(2000.0)
    assert len(ts) - 1 < 20000  # o rzędy mniej niż wymusiłaby jawna stabilność
    xs = [abs(r[0]) for r in rows]
    assert 1.95 < max(xs) < 2.1  # amplituda cyklu granicznego ~2
    assert all(np.isfinite(r).all() for r in np.array(rows))


# --- moduł 13: całkowanie wsteczne (t1 < t0) ------------------------------
def test_backward_rk4_roundtrip_returns_to_initial():
    # w przód, potem WSTECZ z końca -> powrót do warunku początkowego
    rhs = parse("y*cos(t)")
    _, ys_f = ode.solve_ode_num(rhs, "t", 1.0, (0.0, 5.0), 400)
    ts_b, ys_b = ode.solve_ode_num(rhs, "t", ys_f[-1], (5.0, 0.0), 400)
    # zmierzone: |y0' - 1| = 2.5e-12
    assert ys_b[-1] == pytest.approx(1.0, abs=1e-9)
    # węzły ściśle maleją i lądują dokładnie na t1
    assert all(b < a for a, b in zip(ts_b, ts_b[1:]))
    assert ts_b[-1] == 0.0


def test_backward_adaptive_and_dense():
    rhs = parse("y*cos(t)")
    exact5 = math.exp(math.sin(5.0))
    # adaptacyjny roundtrip (zmierzone: 1.1e-8 przy rtol 1e-8)
    _, ya = ode.solve_ode_adaptive(rhs, "t", 1.0, (0.0, 5.0), rtol=1e-8)
    tb, yb = ode.solve_ode_adaptive(rhs, "t", ya[-1], (5.0, 0.0), rtol=1e-8)
    assert yb[-1] == pytest.approx(1.0, abs=1e-6)
    assert tb[-1] == 0.0
    # gęste wyjście wstecz: odczyt MIĘDZY węzłami vs exp(sin t)
    sol = ode.solve_ode_dense(rhs, "t", exact5, (5.0, 0.0), rtol=1e-8)
    grid = np.linspace(0.0, 5.0, 173)
    assert np.allclose(sol(grid), np.exp(np.sin(grid)), atol=1e-6)
    # węzły odtwarzane dokładnie; dziedzina to [0, 5] mimo t0=5 > t1=0
    for tk, yk in zip(sol.ts, sol.ys):
        assert sol(tk) == pytest.approx(yk, abs=1e-12)
    with pytest.raises(PycodemathError):
        sol(6.0)


def test_backward_events_and_terminal():
    # y' = cos(t) wstecz z t=10 (y=sin) — zera mijane w kolejności 3π, 2π, π
    ev = ode.solve_ode_events(
        parse("cos(t)"), "t", math.sin(10.0), (10.0, 0.5), parse("y"), rtol=1e-8
    )
    roots = [3 * math.pi, 2 * math.pi, math.pi]  # kolejność PRZEBIEGU (w dół)
    assert np.allclose(ev.event_times, roots, atol=1e-6)
    # terminal wstecz: pierwsze zdarzenie od t=10 idąc w dół to 3π
    st = ode.solve_ode_events(
        parse("cos(t)"), "t", math.sin(10.0), (10.0, 0.5), parse("y"),
        terminal=True, rtol=1e-8,
    )
    assert st.t1 == pytest.approx(3 * math.pi, abs=1e-6)


def test_backward_stiff_roundtrip():
    # BDF2 w przód i wstecz (łagodny problem): powrót z błędem O(h²)
    rhs = parse("y*cos(t)")
    _, ys_f = ode.solve_ode_stiff(rhs, "t", 1.0, (0.0, 5.0), 400)
    _, ys_b = ode.solve_ode_stiff(rhs, "t", ys_f[-1], (5.0, 0.0), 400)
    # zmierzone: 2.4e-4 (h=0.0125, O(h²) tam i z powrotem)
    assert ys_b[-1] == pytest.approx(1.0, abs=1e-3)


def test_backward_system_dense_and_terminal():
    # oscylator wstecz z t=10: dense trafia sin/cos, terminal x=0 w 3π
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
    # y' = y², y = 1/(1-t): start w t=2 (y=-1), wstecz do 0 — biegun w t=1
    # po drodze; całkowanie wsteczne też ODMAWIA przeskoku przez osobliwość
    with pytest.raises(PycodemathError):
        ode.solve_ode_adaptive(parse("y^2"), "t", -1.0, (2.0, 0.0), rtol=1e-8)


# --- audyt przed dobudową: walidacja tolerancji ----------------------------
def test_dense_and_events_validate_tolerances():
    with pytest.raises(PycodemathError):
        ode.solve_ode_dense(parse("y"), "t", 1.0, (0.0, 1.0), rtol=-1.0)
    with pytest.raises(PycodemathError):
        ode.solve_ode_events(parse("y"), "t", 1.0, (0.0, 1.0), parse("y"), atol=0.0)
