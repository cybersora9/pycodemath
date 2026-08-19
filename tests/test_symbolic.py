"""Testy rundy 2 (Blok A): przecieki kontraktu w silniku symbolicznym.

Kontrakt: brzegi zawsze rzucają PycodemathError, nigdy surowy wyjątek
SymPy; w wyniku zero śmieci (nieobliczonych Integral/Sum).
Oczekiwania zweryfikowane na gołym sympy (python -c), nie przez pycodemath:
- sp.solve(sin(x)-x, x) -> NotImplementedError ("multiple generators")
- sp.integrate(x**x, x) -> nieobliczona Integral(x**x, x)
"""

from __future__ import annotations

import pytest

from pycodemath import parse
from pycodemath.core.errors import PycodemathError
from pycodemath.engine import symbolic
from pycodemath.engine.symbolic import _point


# --- solve: przeciek surowego wyjątku sympy --------------------------------
def test_solve_unsolvable_raises_pycodemath():
    # goły sympy rzuca tu NotImplementedError — kontrakt wymaga PycodemathError
    with pytest.raises(PycodemathError, match="solve"):
        symbolic.solve(parse("sin(x) - x"), "x")


def test_solve_happy_path_no_regression():
    roots = symbolic.solve(parse("x^2 - 9"), "x")
    vals = sorted(int(str(r)) for r in roots)
    assert vals == [-3, 3]


# --- solve: filtr "tylko rzeczywiste" (runda 3) ----------------------------
def test_solve_real_filters_complex_roots():
    # x^2 + 1 = 0 ma tylko pierwiastki zespolone (±i); real=True → puste
    assert symbolic.solve(parse("x^2 + 1"), "x", real=True) == []
    # a domyślnie (real=False) zespolone przechodzą — brak regresji
    assert len(symbolic.solve(parse("x^2 + 1"), "x")) == 2


def test_solve_real_keeps_real_roots():
    # filtr nie tyka pierwiastków rzeczywistych
    roots = symbolic.solve(parse("x^2 - 9"), "x", real=True)
    assert sorted(int(str(r)) for r in roots) == [-3, 3]


def test_solve_real_mixed_roots_keeps_only_real():
    # x^3 - x = 0 → {-1, 0, 1}, wszystkie rzeczywiste; real=True nic nie ucina
    roots = symbolic.solve(parse("x^3 - x"), "x", real=True)
    assert sorted(int(str(r)) for r in roots) == [-1, 0, 1]


# --- integrate: nieobliczona Integral to śmieć w wyniku ---------------------
def test_integrate_no_closed_form_raises():
    # goły sympy zwraca Integral(x**x, x) — u nas czytelna odmowa
    with pytest.raises(PycodemathError, match="closed form"):
        symbolic.integrate(parse("x^x"), "x")


def test_integrate_happy_path_no_regression():
    # ∫ 2x dx = x^2 (forma zamknięta z tablic, nie z pycodemath)
    assert symbolic.integrate(parse("2*x"), "x").equivalent(parse("x^2"))


def test_integrate_via_expr_api_guarded():
    # guard siedzi u źródła (Expr.integrate), więc łapie też bezpośrednie API
    with pytest.raises(PycodemathError, match="closed form"):
        parse("x^x").integrate("x")


# --- _point: bool nie może przejść jako int ---------------------------------
def test_point_rejects_bool():
    with pytest.raises(PycodemathError, match="point/bound"):
        _point(True)
    with pytest.raises(PycodemathError, match="point/bound"):
        _point(False)


def test_limit_rejects_bool_point():
    with pytest.raises(PycodemathError):
        symbolic.limit(parse("1/x"), "x", True)


def test_point_still_accepts_numbers():
    # 0 i 0.5 przechodzą jak dotąd (regresja guardu boola)
    assert str(_point(0)) == "0"
    assert float(_point(0.5)) == 0.5


# --- moduł 10: dwa miejsca OBRONNE, osiągnięte przez podmianę wywołania ----
def test_an_unevaluated_limit_is_no_closed_form_not_a_divergence(monkeypatch):
    # Jedna instrukcja `raise` w module 9 łączyła DWA różne fakty: „SymPy oddał
    # nieobliczony Limit" (silnik nie umiał policzyć) i „wyszło zoo/nan" (granica
    # NIE ISTNIEJE). Moduł 10 je rozdzielił, bo lekarstwa są różne.
    #
    # Gałąź `has(Limit)` jest OBRONNA i mówimy to wprost: dziewięć sond na
    # sp.limit (zeta w 1, gamma w −1, nan, polygamma, floor, frac, besselj, Ei,
    # digamma) dało wyłącznie zoo/nan albo czystą odpowiedź, ani razu nieobliczonego
    # Limita. Skoro nie da się go wywołać wejściem, sprawdzamy KLASYFIKACJĘ tego, co
    # SymPy oddaje — podmieniając samo sp.limit. Testujemy naszą decyzję, nie SymPy.
    import sympy as sp

    from pycodemath.core.errors import DivergenceError, NoClosedFormError

    x = sp.Symbol("x")
    monkeypatch.setattr(sp, "limit", lambda *a, **k: sp.Limit(sp.sin(x) / x, x, 0))
    with pytest.raises(NoClosedFormError) as exc:
        symbolic.limit(parse("sin(x)/x"), "x", 0)
    assert not isinstance(exc.value, DivergenceError)
    assert exc.value.route is None  # gramatyka nie ma granicy numerycznej


def test_zoo_and_nan_limits_are_divergence_like_their_neighbour_summation():
    # ŹRÓDŁO NIEZALEŻNE: zeta ma w 1 BIEGUN, więc sp.limit oddaje zoo (nieskończoność
    # zespolona) — wielkość ucieka do nieskończoności, czyli DOKŁADNIE to, co
    # DivergenceError opisuje. Rozstrzygnął to pomiar spójności: `summation` w tym
    # samym pliku testuje ten sam predykat (`has(oo, -oo, zoo, nan)`) i nazywa go
    # DivergenceError, a `limit` nazywał go gołym PycodemathError. Ten sam fakt pod
    # dwiema nazwami to dryf, a nie rozróżnienie.
    # (E, nie parse: `zeta` nie jest na białej liście parsera i stałoby się tam
    # zwykłym symbolem — a mierzymy klasyfikację granicy, nie zasięg parsera.)
    from pycodemath import E
    from pycodemath.core.errors import DivergenceError

    for source, point in (("zeta(x)", 1), ("polygamma(0,x)", 0), ("nan", 0)):
        with pytest.raises(DivergenceError):
            symbolic.limit(E(source), "x", point)


def test_a_system_shaped_solution_is_an_input_refusal_with_a_route(monkeypatch):
    # Drugie miejsce obronne: `sp.solve(expr, JEDEN_symbol)` nie zwróciło krotki w
    # żadnej z dziewięciu sond (x²−4, x*y−1, Eq(x,y), okrąg, exp(x)−y, |x|−1, x⁴−1,
    # √x−2, sin(x)−½). Klasyfikację sprawdzamy więc podmieniając sp.solve.
    #
    # DomainError, a nie „silnik nie umiał": matematycznie nic nie zawiodło —
    # jednozmiennowy solver dostał pytanie, którego odpowiedzią jest rozwiązanie
    # UKŁADU. To ten sam rodzaj odmowy co niekwadratowa macierz. Trasa nazywa
    # funkcję, która TO SAMO pytanie przyjmuje: solve_nd.
    import sympy as sp

    from pycodemath.core.errors import DomainError

    monkeypatch.setattr(sp, "solve", lambda *a, **k: [(sp.Integer(1), sp.Integer(2))])
    with pytest.raises(DomainError) as exc:
        symbolic.solve(parse("x^2 - 4"), "x")
    assert exc.value.route == "solve_nd"
    assert "solve_nd" in str(exc.value)
