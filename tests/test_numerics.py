"""Testy numeryki: 1D (moduł 4) i wielowymiarowej (moduł 5).

Rozbicie tests/test_pycodemath.py na pliki per obszar (krok 0 sesji
rozwojowej) — treść testów przeniesiona bez zmian.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycodemath import M, parse
from pycodemath.core.errors import (
    DivergenceError,
    DomainError,
    NonConvergenceError,
    NotAMinimumError,
    StagnationError,
)
from pycodemath.engine import numerics


# --- moduł 4: numeryka -------------------------------------------------
def test_root_find_converges_to_sqrt2():
    root = numerics.root_find(parse("x^2 - 2"), "x", 1.0)
    assert root == pytest.approx(math.sqrt(2))
    # pierwiastek faktycznie zeruje funkcję
    assert parse("x^2 - 2").evalf(x=root) == pytest.approx(0.0, abs=1e-9)


def test_root_find_bisection_fallback():
    # start w x0=0 zeruje pochodną (2x=0) — Newton pada, ratuje bisekcja
    root = numerics.root_find(parse("x^2 - 2"), "x", 0.0)
    assert abs(root) == pytest.approx(math.sqrt(2))


def test_root_find_diverges_raises():
    # x^2 + 1 nie ma pierwiastka rzeczywistego -> czytelny błąd
    with pytest.raises(NonConvergenceError):
        numerics.root_find(parse("x^2 + 1"), "x", 1.0)


def test_root_find_divergence_distinguished_from_no_root():
    # atan(x)+5 ∈ (3.43, 6.57) — brak pierwiastka rzeczywistego, ale Newton przy
    # x0=2 EKSPLODUJE (krok rośnie ~x^2: -28 -> -2.8e3 -> -2.8e7 -> -2.7e15).
    # Komunikat musi rozróżnić ROZBIEGANIE od mylącego „brak pierwiastka",
    # analogicznie do detekcji w root_find_nd (R4).
    with pytest.raises(DivergenceError, match="DIVERGES|start"):
        numerics.root_find(parse("atan(x)+5"), "x", 2.0)


def test_root_find_bisection_still_saves_diverging_newton():
    # Newton z atan(x) przy x0=2 rozbiega, ALE pierwiastek (x=0) istnieje i ma
    # zmianę znaku w zasięgu — bisekcja jako fallback musi go zwrócić, mimo że
    # ścieżka Newtona wybuchła. Rozbieganie NIE może wygrać z realnym pierwiastkiem.
    root = numerics.root_find(parse("atan(x)"), "x", 2.0)
    assert root == pytest.approx(0.0, abs=1e-8)


def test_integrate_num_linear():
    val = numerics.integrate_num(parse("2*x"), "x", 0, 1)
    assert val == pytest.approx(1.0)


def test_integrate_num_matches_symbolic():
    # ∫_0^pi sin(x) dx = 2 — brak potrzeby postaci zamkniętej po stronie użytkownika
    val = numerics.integrate_num(parse("sin(x)"), "x", 0.0, math.pi)
    assert val == pytest.approx(2.0, abs=1e-6)


def test_minimize_quadratic():
    xmin = numerics.minimize(parse("(x-3)^2"), "x", 0.0)
    assert xmin == pytest.approx(3.0, abs=1e-4)


def test_minimize_diverges_raises():
    # funkcja liniowa nieograniczona z dołu -> brak minimum -> błąd
    with pytest.raises(NonConvergenceError):
        numerics.minimize(parse("x"), "x", 0.0)


# --- moduł 5: numeryka wielowymiarowa ------------------------------------
def test_gradient_matches_hand_computed():
    # ∇(x²y + y³) = [2xy, x² + 3y²]
    g = numerics.gradient(parse("x^2*y + y^3"), ["x", "y"])
    assert g.equivalent(M("[[2*x*y], [x**2 + 3*y**2]]"))


def test_jacobian_matches_hand_computed():
    # J(x²y, x+y) = [[2xy, x²], [1, 1]]
    J = numerics.jacobian([parse("x^2*y"), parse("x + y")], ["x", "y"])
    assert J.equivalent(M("[[2*x*y, x**2], [1, 1]]"))


def test_hessian_matches_hand_computed():
    # ∇²(x²y) = [[2y, 2x], [2x, 0]] — symetryczny, jak przystało
    H = numerics.hessian(parse("x^2*y"), ["x", "y"])
    assert H.equivalent(M("[[2*y, 2*x], [2*x, 0]]"))


def test_root_find_nd_circle_line_intersection():
    # przecięcie okręgu x²+y²=4 z prostą y=x -> (√2, √2) przy starcie (1,1)
    sol = numerics.root_find_nd(
        [parse("x^2 + y^2 - 4"), parse("x - y")], ["x", "y"], [1.0, 1.0]
    )
    assert np.allclose(sol, [math.sqrt(2), math.sqrt(2)])


def test_root_find_nd_no_real_solution_raises():
    # x²+y²+1 = 0 nie ma rozwiązania rzeczywistego -> czytelny błąd
    with pytest.raises(NonConvergenceError):
        numerics.root_find_nd(
            [parse("x^2 + y^2 + 1"), parse("x - y")], ["x", "y"], [1.0, 1.0]
        )


def test_root_find_nd_divergence_raises_with_hint():
    # atan(x) ma pierwiastek w 0, ale Newton startując z |x| za dużego rozbiega
    # się do nieskończoności zamiast zbiegać. Silnik ma wykryć rozbieganie
    # (a nie zwrócić NaN-y ani mielić wszystkich iteracji z overflow) i
    # podpowiedzieć zmianę punktu startowego.
    with pytest.raises(DivergenceError, match="DIVERGES|start"):
        numerics.root_find_nd([parse("atan(x)"), parse("y")], ["x", "y"], [2.0, 0.0])


def test_root_find_nd_large_real_root_still_converges():
    # Regresja na fałszywy alarm rozbiegania: pierwiastek o dużej (ale
    # skończonej) normie musi się policzyć — próg rozbiegania jest hojny.
    sol = numerics.root_find_nd(
        [parse("x - 100000000000"), parse("y - 5")], ["x", "y"], [0.0, 0.0]
    )
    assert np.allclose(sol, [1e11, 5.0])


def test_root_find_nd_rejects_non_square_and_extra_symbols():
    with pytest.raises(DomainError):
        numerics.root_find_nd([parse("x + y")], ["x", "y"], [0.0, 0.0])
    with pytest.raises(DomainError):
        numerics.root_find_nd(
            [parse("x + a"), parse("x - y")], ["x", "y"], [0.0, 0.0]
        )


def test_minimize_nd_paraboloid():
    # minimum (x-1)² + (y+2)² jest w (1, -2)
    sol = numerics.minimize_nd(parse("(x-1)^2 + (y+2)^2"), ["x", "y"], [0.0, 0.0])
    assert np.allclose(sol, [1.0, -2.0], atol=1e-4)


def test_minimize_nd_diverges_raises():
    # funkcja liniowa nieograniczona z dołu -> brak minimum -> błąd
    with pytest.raises(NonConvergenceError):
        numerics.minimize_nd(parse("x + y"), ["x", "y"], [0.0, 0.0])


# --- moduł 15: lepsza optymalizacja (Newton / BFGS + Armijo) ---------------
# Rosenbrock — wąska zakrzywiona dolina, klasyczny start (-1.2, 1);
# minimum globalne w (1, 1)
_ROSEN = "(1-x)^2 + 100*(y-x^2)^2"


def test_minimize_newton_quadratic_in_two_iterations():
    # kwadratowa: pełny krok Newtona trafia w minimum DOKŁADNIE; zmierzone:
    # newton i bfgs zbiegają przy max_iter=2 (gd potrzebuje 92 iteracji)
    for method in ("newton", "bfgs"):
        x = numerics.minimize(parse("(x-3)^2"), "x", 0.0, method=method, max_iter=5)
        assert x == pytest.approx(3.0, abs=1e-12)


def test_minimize_nd_newton_paraboloid_in_two_iterations():
    # paraboloida nd: to samo — zmierzone minimum max_iter = 2 (gd: 90)
    for method in ("newton", "bfgs"):
        sol = numerics.minimize_nd(
            parse("(x-1)^2 + (y+2)^2"), ["x", "y"], [0.0, 0.0],
            method=method, max_iter=5,
        )
        assert np.allclose(sol, [1.0, -2.0], atol=1e-12)


def test_minimize_nd_newton_rosenbrock():
    # zmierzone: Newton z hessianem symbolicznym zbiega w 22 iteracjach,
    # błąd 1.2e-10 — budżet 60 z zapasem
    sol = numerics.minimize_nd(
        parse(_ROSEN), ["x", "y"], [-1.2, 1.0], method="newton", max_iter=60
    )
    assert np.allclose(sol, [1.0, 1.0], atol=1e-8)


def test_minimize_nd_bfgs_rosenbrock_where_gd_chokes():
    # KONTRAST metod (jak RK4 vs BDF2 w module 12): na dolinie Rosenbrocka
    # spadek gradientowy dławi się i odmawia przy domyślnym budżecie 10000
    # iteracji, a BFGS zbiega w 36 (zmierzone; błąd 1.0e-12) — budżet 100
    sol = numerics.minimize_nd(
        parse(_ROSEN), ["x", "y"], [-1.2, 1.0], method="bfgs", max_iter=100
    )
    assert np.allclose(sol, [1.0, 1.0], atol=1e-8)
    with pytest.raises(NonConvergenceError):
        numerics.minimize_nd(parse(_ROSEN), ["x", "y"], [-1.2, 1.0])  # gd


def test_minimize_second_order_unbounded_refuses():
    # funkcja nieograniczona z dołu: newton (hessian osobliwy -> kierunek
    # gradientu) i bfgs schodzą w nieskończoność — po budżecie odmowa
    for method in ("newton", "bfgs"):
        with pytest.raises(NonConvergenceError):
            numerics.minimize(parse("x"), "x", 0.0, method=method, max_iter=200)
        with pytest.raises(NonConvergenceError):
            numerics.minimize_nd(
                parse("x + y"), ["x", "y"], [0.0, 0.0], method=method, max_iter=200
            )


def test_minimize_nd_second_order_divergence_raises_with_hint():
    # Funkcja WKLĘSŁA nieograniczona z dołu: line search Armijo wciąż schodzi,
    # więc newton/bfgs TROJĄ iterat co krok — norma rośnie wykładniczo i ucieka
    # do nieskończoności, a f -> -inf. Silnik ma wykryć rozbieganie wcześnie
    # (jak Newton w root_find_nd), a nie mielić 10000 iteracji z overflow ani
    # zwracać NaN-ów. Komunikat wskazuje, że funkcja nie ma minimum + start.
    for method in ("newton", "bfgs"):
        with pytest.raises(DivergenceError, match="DIVERGES|unbounded|start"):
            numerics.minimize_nd(
                parse("-(x^2) - y^2"), ["x", "y"], [1.0, 1.0], method=method
            )


def test_minimize_nd_far_minimum_still_converges():
    # Regresja na fałszywy alarm rozbiegania: minimum o dużej (ale skończonej)
    # normie musi się policzyć — próg rozbiegania jest hojny (skalowany ‖x0‖).
    for method in ("newton", "bfgs"):
        sol = numerics.minimize_nd(
            parse("(x - 100000000)^2 + (y - 5)^2"), ["x", "y"], [0.0, 0.0],
            method=method, max_iter=200,
        )
        assert np.allclose(sol, [1e8, 5.0], atol=1e-2)


def test_minimize_validates_method_and_keeps_gd_default():
    # nieznana metoda -> czytelny błąd (1D i nd)
    with pytest.raises(DomainError):
        numerics.minimize(parse("x^2"), "x", 0.0, method="sgd")
    with pytest.raises(DomainError):
        numerics.minimize_nd(parse("x^2 + y^2"), ["x", "y"], [1.0, 1.0], method="lbfgs")
    # domyślna ścieżka gd BIT-IDENTYCZNA z zachowaniem sprzed modułu 15
    # (zmierzona kotwica: 2.999999996357496)
    x = numerics.minimize(parse("(x-3)^2"), "x", 0.0)
    assert x == pytest.approx(2.999999996357496, abs=1e-15)


def test_minimize_gd_divergence_raises_with_hint():
    # R7: domknięcie rodziny detekcji dla HISTORYCZNEGO spadku gradientowego
    # (domyślna metoda). f = -x^2 jest WKLĘSŁA, nieograniczona z dołu: gd oddala
    # się od 0 w każdym kroku (x_next = 1.2·x), iterat eksploduje. Silnik ma
    # wykryć ROZBIEGANIE, a nie mielić max_iter i kończyć generycznym „nie zbiega".
    with pytest.raises(DivergenceError, match="DIVERGES|unbounded|start"):
        numerics.minimize(parse("-x^2"), "x", 1.0)


def test_minimize_nd_gd_divergence_raises_with_hint():
    # R7: to samo dla wielowymiarowego gd (domyślna metoda). f = -(x²+y²)
    # nieograniczona z dołu -> iterat gd ucieka w nieskończoność.
    with pytest.raises(DivergenceError, match="DIVERGES|unbounded|start"):
        numerics.minimize_nd(parse("-(x^2 + y^2)"), ["x", "y"], [1.0, 1.0])


def test_minimize_gd_far_minimum_still_converges():
    # Regresja na fałszywy alarm rozbiegania w gd: minimum o dużej (ale
    # skończonej) normie musi się policzyć — check odpala się tylko na eksplozji.
    x = numerics.minimize(parse("(x - 100000000)^2"), "x", 0.0, max_iter=100000)
    assert x == pytest.approx(1e8, rel=1e-6)


# --- R8: audyt spójności rodziny detekcji rozbiegania ------------------
def test_diverge_limit_helper_is_bit_identical_to_old_formula():
    # R8 wynosi próg 1e12·max(1,‖x0‖) do jednego helpera. Kontrakt: helper
    # zwraca DOKŁADNIE tę samą liczbę co dawny literał rozsiany po solwerach —
    # żadnego dryfu wartości, tylko centralizacja wzoru.
    assert numerics._DIVERGE_SCALE == 1e12
    for norm in (0.0, 0.5, 1.0, 5.0, 1e8, 3.7e11):
        assert numerics._diverge_limit(norm) == 1e12 * max(1.0, norm)


def test_all_solvers_share_one_divergence_threshold():
    # R8 anti-dryf: cała rodzina (root_find, root_find_nd, _descent_min,
    # minimize/minimize_nd w wariancie gd) MUSI czerpać próg z jednego źródła.
    # Gdyby ktoś w przyszłości wpisał 1e12 na sztywno w którymś solwerze,
    # rodzina mogłaby się rozejść — pilnujemy tego mechanicznie na źródle.
    import inspect

    src = inspect.getsource(numerics)
    # Skala 1e12 mnożona przez próg NIE może już pojawiać się nigdzie w kodzie
    # solwerów — dawny literał ``1e12 * max(...)`` żyje teraz wyłącznie wewnątrz
    # _diverge_limit. Zero wystąpień = nikt go nie wpisał z powrotem na sztywno.
    assert "1e12 * max" not in src, (
        "hardcodowany próg '1e12 * max(...)' wrócił do solwera — użyj _diverge_limit()"
    )
    # R10: każdy z pięciu solwerów liczy diverge_limit przez JEDEN helper
    # doboru normy do wymiaru — _diverge_limit_for.
    assert src.count("diverge_limit = _diverge_limit_for(") == 5
    # A surowy _diverge_limit( (dobór normy ręcznie) nie żyje już w żadnym
    # solwerze — wyłącznie wewnątrz _diverge_limit_for (który sam wybiera abs vs
    # ‖·‖). Zero open-coded wywołań = nikt nie wpisał złej normy dla wymiarowości.
    assert "diverge_limit = _diverge_limit(" not in src, (
        "solwer liczy diverge_limit surowym _diverge_limit() z ręcznie dobraną "
        "normą — użyj _diverge_limit_for(), który dobiera abs/‖·‖ z wymiaru x0"
    )


# --- R9: domknięcie centralizacji — jeden PREDYKAT rozbiegania ----------
def test_has_diverged_scalar_and_vector():
    # Wspólny predykat MUSI działać spójnie dla skalara (1D) i wektora (nd) i
    # zwracać czysty pythonowy bool. Skalar: abs + math.isfinite; wektor:
    # np.all(np.isfinite) + np.linalg.norm — dwie dawne formy w jednym miejscu.
    lim = numerics._diverge_limit(1.0)  # == 1e12
    # skalar
    assert numerics._has_diverged(float("inf"), lim) is True
    assert numerics._has_diverged(float("nan"), lim) is True
    assert numerics._has_diverged(1e13, lim) is True
    assert numerics._has_diverged(-1e13, lim) is True
    assert numerics._has_diverged(1.0, lim) is False
    assert numerics._has_diverged(lim, lim) is False  # próg jest ostry (>)
    # wektor
    assert numerics._has_diverged(np.array([1.0, float("inf")]), lim) is True
    assert numerics._has_diverged(np.array([float("nan"), 0.0]), lim) is True
    assert numerics._has_diverged(np.array([1e13, 0.0]), lim) is True
    assert numerics._has_diverged(np.array([1.0, 2.0]), lim) is False


def test_all_solvers_share_one_divergence_predicate():
    # R9 anti-dryf (domknięcie R8): R8 ujednolicił PRÓG (_diverge_limit), R9
    # ujednolica PREDYKAT (not isfinite / norma > limit) do jednego helpera
    # _has_diverged. Cała rodzina (root_find, _descent_min, root_find_nd,
    # minimize_nd newton/bfgs, minimize_nd gd) MUSI go dzielić — inaczej detekcja
    # mogłaby się rozejść w szczegółach (1D abs vs nd norm). Pilnujemy tego
    # mechanicznie: 1 definicja + 5 solwerów = 6 wystąpień.
    import inspect

    src = inspect.getsource(numerics)
    assert src.count("_has_diverged(") == 6, (
        "każdy solwer MUSI dzielić jeden predykat rozbiegania _has_diverged() "
        "(1 definicja + 5 solwerów)"
    )


# --- R11: STAGNACJA (odwrotna strona rozbiegania) ----------------------
def test_has_stagnated_scalar_and_vector():
    # Prymityw stagnacji, bliźniak _has_diverged: True gdy krok znika przy
    # maszynowej precyzji (‖x_next − x‖ ≤ _STAG_SCALE·eps·max(1, ‖x‖)). Skalar i
    # wektor, czysty pythonowy bool. Człon max(1, ·) daje próg WZGLĘDNY.
    assert numerics._STAG_SCALE == 4.0
    eps = numerics._EPS
    # skalar: krok dokładnie zerowy — na pewno stagnacja
    assert numerics._has_stagnated(1.0, 1.0) is True
    # brzeg progu przy x=1 (próg = 4·eps): 2 ulpy ≤ 4·eps -> stagnacja,
    # 6 ulpów > 4·eps -> już nie
    assert numerics._has_stagnated(1.0 + 2 * eps, 1.0) is True
    assert numerics._has_stagnated(1.0 + 6 * eps, 1.0) is False
    # duży krok (1.0) NIE jest stagnacją
    assert numerics._has_stagnated(2.0, 1.0) is False
    # skalowanie WZGLĘDNE: TEN SAM absolutny krok (1e-8) jest stagnacją przy
    # ‖x‖=1e12 (próg ~1.8e-4, a 1e-8 ginie już w reprezentacji floata), ale NIE
    # przy ‖x‖~1 (próg ~4e-16, krok 1e-8 to realny ruch).
    assert numerics._has_stagnated(1.0 + 1e-8, 1.0) is False
    assert numerics._has_stagnated(1e12 + 1e-8, 1e12) is True
    # wektor: norma różnicy vs norma bazy
    v = np.array([1.0, 1.0])
    assert numerics._has_stagnated(v, v) is True
    assert numerics._has_stagnated(v + np.array([1.0, 0.0]), v) is False
    # zwraca czysty bool, nie numpy.bool_
    assert type(numerics._has_stagnated(v, v)) is bool


def test_minimize_gd_stagnation_distinguished_from_unbounded():
    # x^4 JEST ograniczona z dołu (min w 0), ale spadek gradientowy przy zbyt
    # dużym lr wpada w cykl okresu 2 (±x0, f na stałej) i nie robi postępu.
    # Dawniej kończyło się MYLĄCYM „nieograniczona z dołu"; R11 musi to nazwać
    # STAGNACJĄ (UTKNIĘCIEM), a NIE rozbieganiem ani nieograniczonością.
    with pytest.raises(StagnationError, match="STALLED|STAGNAT") as exc:
        numerics.minimize(parse("x^4"), "x", 2.0, lr=1.0)
    msg = str(exc.value)
    assert "DIVERGES" not in msg  # to NIE rozbieganie
    assert "unbounded below" not in msg  # i NIE nieograniczoność — dawny fałsz


def test_minimize_nd_gd_stagnation_raises_utkn():
    # Wielowymiarowy odpowiednik: x^4 + y^4 przy zbyt dużym lr też wpada w cykl.
    with pytest.raises(StagnationError, match="STALLED|STAGNAT") as exc:
        numerics.minimize_nd(parse("x^4 + y^4"), ["x", "y"], [2.0, 2.0], lr=1.0)
    assert "DIVERGES" not in str(exc.value)


def test_minimize_gd_overshoot_lr_not_falsely_stagnant():
    # Regresja na FAŁSZYWY alarm stagnacji: zbyt duży lr, który daje serię
    # przestrzeleń, ale po skróceniu kroku ZBIEGA — halvingi lr to korekta, nie
    # utknięcie, więc muszą zwrócić minimum, nie UTKNIĘCIE.
    x = numerics.minimize(parse("(x-3)^2"), "x", 0.0, lr=100.0)
    assert x == pytest.approx(3.0, abs=1e-4)


def test_minimize_nd_gd_far_minimum_not_falsely_stagnant():
    # Regresja: zbieżność do DALEKIEGO minimum ma krok mały względem ‖x‖ (czyli
    # „stagnujący krokowo"), ale robi realny WZGLĘDNY postęp f na krok — nie może
    # zostać uznana za utknięcie. Detekcja stoi na braku postępu f, nie na kroku.
    sol = numerics.minimize_nd(
        parse("(x-100000000)^2 + (y-5)^2"), ["x", "y"], [0.0, 0.0], max_iter=200000
    )
    assert np.allclose(sol, [1e8, 5.0], rtol=1e-6)


# --- R12: centralizacja predykatu POSTĘPU WARTOŚCI (lustro R8–R10) ------
def test_value_progressed_predicate():
    # Predykat, którego solwery gd realnie używają do wykrywania stagnacji —
    # WZGLĘDNY spadek f: fx − f_next > _STAG_SCALE·eps·|fx|. Skalarny, czysty bool.
    # (To NIE krokowy _has_stagnated — daleki minimum ma mały krok, ale realny
    # postęp f, więc ten predykat mówi True, a _has_stagnated dałby fałszywy alarm.)
    eps = numerics._EPS
    # duży spadek f -> postęp
    assert numerics._value_progressed(1.0, 0.0) is True
    # f stoi -> brak postępu
    assert numerics._value_progressed(1.0, 1.0) is False
    # brzeg progu przy |fx|=1 (próg = 4·eps): 2 ulpy ≤ próg -> brak postępu,
    # 6 ulpów > próg -> postęp
    assert numerics._value_progressed(1.0, 1.0 - 2 * eps) is False
    assert numerics._value_progressed(1.0, 1.0 - 6 * eps) is True
    # skalowanie WZGLĘDNE do |fx|: przy fx=1e12 próg ~8.9e-4, spadek o 1.0 to
    # realny postęp, a mikroskopijny spadek ginie już w reprezentacji floata
    assert numerics._value_progressed(1e12, 1e12 - 1.0) is True
    assert numerics._value_progressed(1e12, 1e12 - 1e-6) is False
    # czysty pythonowy bool, nie numpy.bool_
    assert type(numerics._value_progressed(1.0, 0.0)) is bool


def test_gd_solvers_share_one_value_progress_predicate():
    # R12 anti-dryf (lustro R9/R10 dla rozbiegania): predykat POSTĘPU WARTOŚCI był
    # wpisany inline BLIŹNIACZO w minimize i minimize_nd (gd). Wyniesiony do jednego
    # _value_progressed — oba solwery MUSZĄ go dzielić, inaczej próg stagnacji mógłby
    # się rozejść między 1D a nd. Mechanicznie: 1 definicja + 2 solwery = 3 wystąpienia,
    # a surowa forma progu nie może już nigdzie wisieć poza definicją helpera.
    import inspect

    # ŚWIADOME PRZEPISANIE LICZNIKA (moduł 8, przypadek (i)): 3 → 5. Granica z R12
    # pilnowała, że predykat POSTĘPU ma jedną definicję i dwóch użytkowników — i to
    # nadal pilnuje. Zmieniło się to, ILE RZECZY każdy solver gd uznaje za postęp:
    # moduł 8 dołożył postęp REZYDUUM obok postępu WARTOŚCI, bo sama wartość f
    # siada na szumie zaokrągleń, gdy |g| jest jeszcze ~1e-8 i wciąż spada (sin(x)
    # z x0=0 dostawał UTKNIĘCIE w minimum, z dokładnością do 7 cyfr znaczących).
    # Kluczowe: to NADAL ten sam predykat, wołany o inną wielkość — gdyby próg
    # rezydualny został wpisany inline, 1D i nd mogłyby się rozejść dokładnie tak,
    # jak broniło przed tym R12. Stąd 1 definicja + 2 solwery × 2 wielkości = 5.
    src = inspect.getsource(numerics)
    assert src.count("_value_progressed(") == 5, (
        "minimize i minimize_nd (gd) MUSZĄ dzielić jeden predykat postępu "
        "_value_progressed() (1 definicja + 2 solwery × {wartość f, rezyduum})"
    )
    assert "fx - f_next > _STAG_SCALE * _EPS * abs(fx)" not in src, (
        "surowy próg stagnacji nie może być inline w solwerze — użyj "
        "_value_progressed(fx, f_next)"
    )


# --- R13: USTALENIE — rodzina stagnacji KOŃCZY się na gd ----------------
def test_root_find_newton_cycle_is_rescued_by_bisection_not_stagnation():
    # USTALENIE R13 (root_find): cykl Newtona ISTNIEJE — x^3-2x+2 z x0=0 skacze
    # 0→1→0→1 w nieskończoność, nie zbiegając i nie rozbiegając się. W gd taki cykl
    # dawał MYLĄCE „nieograniczona z dołu" i wymagał detekcji stagnacji (R11). Tutaj
    # NIE dociera do żadnego komunikatu: bisekcja jest fallbackiem i przy realnej
    # zmianie znaku ratuje bieg. Dlatego stagnacja nie jest tu wpinana — nie ma
    # mylącej diagnozy do naprawienia. Ten test pilnuje, że fallback nadal ratuje.
    root = numerics.root_find(parse("x^3 - 2*x + 2"), "x", 0.0)
    assert root == pytest.approx(-1.7692923542, abs=1e-6)
    assert abs(float(numerics._scalar(parse("x^3 - 2*x + 2"), "x")(root))) < 1e-8

    # Kontrola, że to naprawdę CYKL (a nie zbieżność Newtona): iteracja goła.
    f = numerics._scalar(parse("x^3 - 2*x + 2"), "x")
    df = numerics._scalar(parse("x^3 - 2*x + 2").diff("x"), "x")
    x, seq = 0.0, []
    for _ in range(6):
        x = x - f(x) / df(x)
        seq.append(x)
    assert seq == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_root_find_no_real_root_message_already_true_without_stagnation():
    # USTALENIE R13 (root_find, druga gałąź): gdy pierwiastka rzeczywistego NIE ma,
    # dzisiejszy komunikat jest po prostu PRAWDZIWY — nie ma tu fałszu, który
    # stagnacja miałaby prostować. I nadal nie wolno go pomylić z ROZBIEGANIEM.
    with pytest.raises(NonConvergenceError, match="no real root") as exc:
        numerics.root_find(parse("x^2 + 1"), "x", 1.0)
    assert "DIVERGES" not in str(exc.value)
    assert "STALLED" not in str(exc.value)


@pytest.mark.parametrize("method", ["newton", "bfgs"])
def test_descent_min_armijo_forbids_the_gd_stagnation_cycle(method):
    # USTALENIE R13 (_descent_min): bug R11 miał DOKŁADNIE jedną przyczynę — stały
    # lr. x^4 przy lr=1 wpada w cykl okresu 2 i STAGNUJE (patrz test wyżej). Te same
    # funkcje pod newton/bfgs nie mają jak stagnować: nie ma lr, a nawrotowy Armijo
    # przyjmuje krok TYLKO gdy f realnie spada, więc f jest ściśle monotoniczna i
    # cykl „f stoi" jest strukturalnie niemożliwy. Muszą po prostu ZBIEC.
    sol = numerics.minimize_nd(parse("x^4 + y^4"), ["x", "y"], [2.0, 2.0], method=method)
    assert np.allclose(sol, [0.0, 0.0], atol=1e-3)
    # a 1D odpowiednik dokładnie tego, co w gd wywalało UTKNIĘCIE przy lr=1
    sol1 = numerics.minimize_nd(parse("x^4"), ["x"], [2.0], method=method)
    assert sol1[0] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("method", ["newton", "bfgs"])
def test_descent_min_unbounded_message_is_true_not_a_stagnation_misreport(method):
    # USTALENIE R13: jedyny bieg _descent_min, który MIELI max_iter, to funkcja o
    # STAŁYM gradiencie (-x-y): iterat maszeruje LINIOWO, więc słusznie nie odpala
    # progu rozbiegania (norma ~1e2 przy progu 1e12), a f spada o stałą co krok.
    # Generyczny komunikat „nieograniczona z dołu" jest tam PRAWDĄ — to nie jest
    # przemilczana stagnacja, więc nie ma tu czego naprawiać detekcją utknięcia.
    with pytest.raises(NonConvergenceError, match="does not converge") as exc:
        numerics.minimize_nd(parse("-x - y"), ["x", "y"], [0.0, 0.0],
                             method=method, max_iter=200)
    msg = str(exc.value)
    assert "unbounded" in msg      # i to jest prawda: -x-y NIE ma minimum
    assert "STALLED" not in msg and "STAGNAT" not in msg


# =========================================================================
# MODUŁ 8 — converged=True ZNACZY „to jest ODPOWIEDŹ NA PYTANIE, KTÓRE ZADANO"
# =========================================================================
# Seria v0.3 powtarzała w każdym docstringu i w obu README: „converged jest
# JEDYNYM polem, po którym wolno się rozgałęzić". Zmierzone na HEAD modułu 7 —
# nieprawda w trzech kierunkach naraz:
#   (A) minimize(-x^2, x0=0) -> converged=True. Test gradientu jest spełniony w
#       KAŻDYM punkcie stacjonarnym: maksimum i siodło zdają go tak samo dobrze
#       jak minimum.
#   (B) minimize(sin(x), x0=0) -> converged=False, status='stagnated', przy
#       wartości poprawnej do 7 cyfr znaczących. Detektor utknięcia patrzył na
#       postęp f, a f siada na szumie zaokrągleń, zanim |g| dojdzie do tol.
#   (C) root_find(exp(x), x0=0) -> converged=True, value=-24. exp(x)=0 nie ma
#       pierwiastka; iterat po prostu ZSZEDŁ po ogonie, aż |f| spadło pod tol.
# Źródło niezależne, jak w całej serii: prawda każdego wiersza jest MATEMATYCZNA
# (f''<0 w 0 dla -x², brak miejsca zerowego exp, −π/2 jako minimum sinusa), a nie
# „to, co pycodemath dziś zwraca".


# --- (A) certyfikat minimum ----------------------------------------------
@pytest.mark.parametrize("method", ["gd", "newton", "bfgs"])
@pytest.mark.parametrize(
    "src, why",
    [
        ("-x^2", "MAKSIMUM (f''=-2)"),
        ("x^4-4*x^2", "MAKSIMUM lokalne w 0 (f''=-8), prawdziwe minima w ±sqrt(2)"),
        ("cos(x)", "MAKSIMUM (f''=-1)"),
        ("x^3", "PRZEGIĘCIE (f''=0 — drugi rząd NIE MA tu werdyktu)"),
        ("-x^4", "MAKSIMUM z ZEROWĄ krzywizną (f''=0 — jw.)"),
    ],
)
def test_module8_stationary_point_that_is_not_a_minimum_is_refused(src, why, method):
    # Start JEST punktem stacjonarnym, więc żadna metoda nie ma dokąd pójść:
    # krok gd to lr·0, a line search w newton/bfgs nie zdąży ruszyć — test
    # gradientu strzela w iteracji 1. Dawniej wszystkie trzy mówiły converged.
    with pytest.raises(NotAMinimumError, match="STATIONARY"):
        numerics.minimize(parse(src), "x", 0.0, method=method)
    r = numerics.minimize(parse(src), "x", 0.0, method=method, full_result=True)
    assert r.converged is False
    assert r.status == "not_a_minimum"
    assert r.value == 0.0  # wartość to nadal punkt, w którym bieg stanął
    # ...i to NIE jest przemianowane rozbieganie ani brak zbieżności: bieg DOSZEDŁ
    assert r.residual == 0.0


@pytest.mark.parametrize("method", ["gd", "newton", "bfgs"])
@pytest.mark.parametrize(
    "src, x0, why",
    [
        ("x^2 - y^2", [0.0, 0.0], "SIODŁO — spadek wzdłuż osi y"),
        ("-x^2 - y^2", [0.0, 0.0], "MAKSIMUM"),
        ("x^3 + y^2", [0.0, 0.0], "PRZEGIĘCIE w x, minimum w y"),
        ("x*y", [0.0, 0.0], "SIODŁO DIAGONALNE — wzdłuż OBU osi f jest 0.0"),
        ("(x+3*y)^2 - 1e-4*(3*x-y)^2", [0.0, 0.0], "siodło ani osiowe, ani po przekątnej"),
    ],
)
def test_module8_nd_stationary_point_that_is_not_a_minimum_is_refused(src, x0, why, method):
    # Dwa ostatnie wiersze to POWÓD, dla którego sonda ma kierunek z krzywizny, a
    # nie tylko osie: w x*y wzdłuż obu osi f jest dokładnie 0.0 (sonda osiowa widzi
    # płaskowyż i nie ma czego zgłosić), a w ostatnim kierunek spadku nie jest ani
    # osią, ani przekątną. Wektor własny NAJMNIEJSZEJ krzywizny trafia w oba.
    with pytest.raises(NotAMinimumError, match="STATIONARY"):
        numerics.minimize_nd(parse(src), ["x", "y"], x0, method=method)
    r = numerics.minimize_nd(parse(src), ["x", "y"], x0, method=method, full_result=True)
    assert r.converged is False and r.status == "not_a_minimum"


@pytest.mark.parametrize("method", ["gd", "newton", "bfgs"])
def test_module8_flat_minimum_is_still_a_minimum(method):
    # GRANICA, KTÓRA ROZSTRZYGA PROJEKT: x^4 ma w 0 MINIMUM, a -x^4 MAKSIMUM, i
    # OBA mają f''=0. Żaden test drugiego rzędu ich nie rozróżni — dlatego werdykt
    # nie pochodzi z krzywizny, tylko z DEFINICJI (czy obok leży niższa wartość).
    # x^4 musi przejść, -x^4 musi paść; jedno bez drugiego to nie certyfikat,
    # tylko próg dobrany pod przykład.
    assert numerics.minimize(parse("x^4"), "x", 0.0, method=method) == 0.0
    with pytest.raises(NotAMinimumError):
        numerics.minimize(parse("-x^4"), "x", 0.0, method=method)
    sol = numerics.minimize_nd(parse("x^4 + y^4"), ["x", "y"], [0.0, 0.0], method=method)
    assert sol == [0.0, 0.0]


@pytest.mark.parametrize("method", ["gd", "newton", "bfgs"])
def test_module8_certificate_does_not_refuse_genuine_minima(method):
    # Regresja na FAŁSZYWE ODMOWY — najdroższy możliwy błąd tego modułu.
    # Wszystkie cztery kształty, na których sonda mogłaby się wywrócić:
    # (1) minimum w x0, (2) minimum ODLEGŁE (‖x‖=1e8: sonda skalowana do ‖x‖
    # sondowałaby 1e6 stąd i opisywała zupełnie inny kawałek funkcji),
    # (3) dolina Rosenbrocka (skrajnie różne krzywizny w dwóch kierunkach),
    # (4) funkcja STAŁA (każdy punkt JEST minimum, niewłaściwym, i nic tego nie
    # obala — sonda nie ma prawa wymyślić odmowy z braku informacji).
    assert numerics.minimize(parse("(x-3)^2"), "x", 3.0, method=method) == 3.0
    assert numerics.minimize(parse("x^2"), "x", 0.0, method=method) == 0.0
    assert numerics.minimize(parse("5"), "x", 1.0, method=method) == 1.0
    sol = numerics.minimize_nd(
        parse("(x-100000000)^2 + (y-5)^2"), ["x", "y"], [0.0, 0.0],
        method=method, max_iter=200000,
    )
    assert np.allclose(sol, [1e8, 5.0], rtol=1e-6)
    rosen = numerics.minimize_nd(
        parse(_ROSEN), ["x", "y"], [-1.2, 1.0], method="bfgs", max_iter=100
    )
    assert np.allclose(rosen, [1.0, 1.0], atol=1e-8)


@pytest.mark.parametrize("x0", [-0.002, 0.003, 0.01])
def test_module8_probe_stays_local_on_a_rippled_function(x0):
    # x² + 1e-3·sin(1000x) ma minima lokalne co ~6.28e-3. Sonda, która rozszerza
    # się do 1e-2 ZAWSZE, wpadłaby do SĄSIEDNIEGO (niższego) minimum i odmówiła
    # poprawnej odpowiedzi. Dlatego eskalacja zatrzymuje się na pierwszej skali
    # ROZSTRZYGAJĄCEJ: przy dobrze zakrzywionym minimum już 1e-6 wychodzi ponad
    # szum, więc 1e-2 nigdy nie jest sondowane.
    src = "x^2 + 1e-3*sin(1000*x)"
    x = numerics.minimize(parse(src), "x", x0)
    # to NAPRAWDĘ jest lokalne minimum: gradient znika, a f rośnie w obie strony
    assert abs(float(numerics._scalar(parse(src).diff("x"), "x")(x))) < 1e-6
    f = numerics._scalar(parse(src), "x")
    assert f(x - 1e-4) > f(x) and f(x + 1e-4) > f(x)


def test_module8_start_beside_the_maximum_still_reaches_the_minimum():
    # Obrona TEGO, CO DZIAŁAŁO: certyfikat dotyka wyłącznie wyjścia, więc bieg,
    # który startuje OBOK punktu stacjonarnego, ma być bit w bit taki jak przedtem.
    assert numerics.minimize(parse("x^4-4*x^2"), "x", 0.05) == 1.4142135626386807
    assert numerics.minimize(parse("x^4-4*x^2"), "x", -0.05) == -1.4142135626386807


def test_module8_unbounded_is_still_divergence_not_a_refused_minimum():
    # Rozróżnienie, którego nie wolno zgubić: -x² z x0=1 UCIEKA (nie ma minimum i
    # nie ma punktu stacjonarnego na drodze), a -x² z x0=0 STOI na maksimum.
    # To dwa różne wyniki i dwie różne klasy — mylenie ich cofnęłoby moduł 2.
    with pytest.raises(DivergenceError):
        numerics.minimize(parse("-x^2"), "x", 1.0)
    with pytest.raises(NotAMinimumError):
        numerics.minimize(parse("-x^2"), "x", 0.0)
    with pytest.raises(DivergenceError):
        numerics.minimize_nd(parse("x^2 - y^2"), ["x", "y"], [0.1, 0.1])
    with pytest.raises(NotAMinimumError):
        numerics.minimize_nd(parse("x^2 - y^2"), ["x", "y"], [0.0, 0.0])


def test_module8_certificate_has_one_definition_and_three_call_sites():
    # Antydryf w stylu R9/R10/R12: certyfikat MUSI być jeden. Trzy solwery
    # minimalizujące (minimize gd, minimize_nd gd, _descent_min dla newton/bfgs)
    # wołają ten sam _lower_value_nearby — gdyby któryś dostał własną sondę, próg
    # „istotnego spadku" mógłby się rozejść i converged znaczyłoby co innego
    # zależnie od metody. 1 definicja + 3 solwery = 4 wystąpienia.
    import inspect

    src = inspect.getsource(numerics)
    assert src.count("_lower_value_nearby(") == 4
    # ...a każdy z trzech solwerów ma DOKŁADNIE jedno domknięcie `certified`,
    # przez które przechodzą OBA jego wyjścia sukcesu (gradientowe i krokowe)
    assert src.count("def certified(") == 3


# --- (B) utknięcie NIE MOŻE wyprzedzać tolerancji -------------------------
def test_module8_slow_convergence_is_not_stagnation():
    # DEFEKT (B) wprost: sin(x) z x0=0 przy domyślnych parametrach. gd zbiega tu
    # geometrycznie (czynnik 1-lr·f'' = 0.9 na krok), więc |g| SPADA co krok — a
    # f-f* idzie jak KWADRAT odległości, czyli siada na szumie zaokrągleń float64
    # przy |g| ~ 1e-8, jeszcze rząd nad tol=1e-9. Detektor patrzący tylko na f
    # ogłaszał UTKNIĘCIE w punkcie poprawnym do 7 cyfr znaczących.
    r = numerics.minimize(parse("sin(x)"), "x", 0.0, full_result=True)
    assert r.converged is True and r.status == "converged"
    assert r.value == pytest.approx(-math.pi / 2, abs=1e-7)
    # ta sama historia z drugiej strony i na innej funkcji
    assert numerics.minimize(parse("sin(x)"), "x", 1.0) == pytest.approx(
        -math.pi / 2, abs=1e-7
    )
    assert numerics.minimize(parse("exp(x)+exp(-x)"), "x", 1.0) == pytest.approx(
        0.0, abs=1e-7
    )


def test_module8_the_cycle_the_detector_was_built_for_still_stalls():
    # DRUGA POŁOWA (B), bez której pierwsza byłaby rozbrojeniem detektora: x^4 przy
    # lr=1 to PRAWDZIWY cykl okresu 2 (±2), w którym ani f, ani |g| się nie ruszają
    # (|g| stoi na 32.0). Nowa reguła resetuje serię na postępie REZYDUUM — a tu
    # rezyduum jest stałe, więc seria dobija do 8 i utknięcie zostaje zgłoszone.
    with pytest.raises(StagnationError, match="STALLED"):
        numerics.minimize(parse("x^4"), "x", 2.0, lr=1.0)
    r = numerics.minimize(parse("x^4"), "x", 2.0, lr=1.0, full_result=True)
    assert r.status == "stagnated" and r.residual == 32.0
    with pytest.raises(StagnationError, match="STALLED"):
        numerics.minimize_nd(parse("x^4 + y^4"), ["x", "y"], [2.0, 2.0], lr=1.0)


def test_module8_residual_progress_resets_the_stagnation_streak():
    # Mechanizm wprost, na predykacie: to TEN SAM _value_progressed, wołany o
    # rezyduum zamiast o wartość f. Bieg sinusa robi ~10% względnego postępu |g|
    # na krok (czynnik 0.9) — miliardy razy powyżej progu 4·eps — więc seria nigdy
    # nie dociąga do 8. Cykl x^4 ma |g| stałe, więc postępu nie ma wcale.
    assert numerics._value_progressed(1.0, 0.9) is True     # sinus: 10% na krok
    assert numerics._value_progressed(32.0, 32.0) is False  # cykl x^4: |g| stoi


def test_module8_converging_runs_keep_their_iteration_counts():
    # Reguła bezpieczeństwa (B): dodanie RESETU serii może wyłącznie PRZEDŁUŻYĆ
    # bieg, który dawniej padał — nigdy nie skrócić ani nie zmienić biegu, który
    # zbiegał. Kotwice zapinowane przed modułem 8 muszą stać co do iteracji.
    assert numerics.minimize(
        parse("(x-3)^2"), "x", 0.0, full_result=True
    ).iterations == 92
    assert numerics.minimize_nd(
        parse("(x-1)^2 + (y+2)^2"), ["x", "y"], [0.0, 0.0], full_result=True
    ).iterations == 90
    assert numerics.minimize(parse("(x-3)^2"), "x", 0.0) == 2.999999996357496


# --- (C) małe residuum NIE jest dowodem istnienia -------------------------
@pytest.mark.parametrize("x0", [0.0, 5.0, -5.0])
def test_module8_decaying_function_is_not_a_root(x0):
    # DEFEKT (C) wprost. exp(x)=0 nie ma pierwiastka; krok Newtona dla exp to
    # DOKŁADNIE -1 w każdym punkcie, więc iterat maszeruje w lewo, a |f| spada pod
    # tol po drodze. Dawniej: converged=True, value=-24.0, z każdego z tych startów.
    with pytest.raises(NonConvergenceError, match="DECAYING"):
        numerics.root_find(parse("exp(x)"), "x", x0)
    r = numerics.root_find(parse("exp(x)"), "x", x0, full_result=True)
    assert r.converged is False and r.status == "not_converged"
    # I OTO DLACZEGO converged jest jedynym polem do rozgałęziania: residuum na
    # tej porażce jest MIKROSKOPIJNE (~1e-44) — mniejsze niż na niejednym sukcesie.
    assert r.residual < 1e-40


def test_module8_decay_in_nd_is_not_a_solution():
    with pytest.raises(NonConvergenceError, match="DECAYING"):
        numerics.root_find_nd(
            [parse("exp(x)"), parse("y - 1")], ["x", "y"], [0.0, 0.0]
        )
    r = numerics.root_find_nd(
        [parse("exp(x)"), parse("y - 1")], ["x", "y"], [0.0, 0.0], full_result=True
    )
    assert r.converged is False and r.residual < 1e-40


def test_module8_roots_that_exist_are_found_bit_for_bit():
    # Druga połowa (C): certyfikat ma odmawiać WYŁĄCZNIE tam, gdzie nie ma
    # pierwiastka. Te liczby są zmierzone PRZED modułem 8 i muszą zostać co do
    # bitu — przesunięcie poprawnej odpowiedzi o jeden ulp to regresja, nie fix.
    assert numerics.root_find(parse("x^2-2"), "x", 1.0) == 1.4142135623746899
    assert numerics.root_find(parse("x - 0.6*sin(x) - 0.8"), "x", 1.0) == 1.39024713355494
    assert numerics.root_find(
        parse("x^2-2"), "x", 1.0, full_result=True
    ).iterations == 5
    sol = numerics.root_find_nd(
        [parse("x^2+y^2-4"), parse("x*y-1")], ["x", "y"], [0.5, 2.0]
    )
    assert sol == [0.5176380902042443, 1.9318516525789344]


def test_module8_a_tiny_root_is_still_a_root():
    # PRZYPADEK, KTÓRY ODDZIELA DYSKRYMINATOR OD PROGU NA REZYDUUM: exp(x)-1e-11
    # MA pierwiastek (ln 1e-11 = -25.328...), i to taki, w którym |f| jest z natury
    # mikroskopijne. Kryterium oparte na „residuum za małe = podejrzane" odrzuciłoby
    # go razem z exp(x). Kryterium oparte na KROKU nie tylko go przyjmuje — poprawia
    # odpowiedź, bo dawniej bieg stawał 2.4 PRZED nim (w -22.944), z |f| pod tol.
    # Dokładność pinujemy KONTRAKTEM, nie magiczną stałą: kryterium zatrzymania
    # mówi „korekta Newtona ≤ sqrt(eps)·max(1,|x|)", czyli DOKŁADNIE tyle wynosi
    # gwarantowany promień wokół prawdziwego pierwiastka. Zmierzone: błąd 1.5e-8
    # przy progu 3.8e-7 — mieści się z zapasem 25×, i tak ma być z definicji.
    for src, truth in (("exp(x) - 1e-11", 1e-11), ("exp(x) - 1e-30", 1e-30)):
        r = numerics.root_find(parse(src), "x", 0.0, full_result=True)
        exact = math.log(truth)
        assert r.converged is True
        assert abs(r.value - exact) <= numerics._ROOT_SETTLE * max(1.0, abs(exact))


def test_module8_multiple_roots_are_found_more_precisely_not_refused():
    # Pierwiastek WIELOKROTNY nie ma zmiany znaku (x² nigdzie nie schodzi pod zero),
    # więc kryterium „musi być zmiana znaku" odrzuciłoby prawdziwy pierwiastek.
    # Kryterium kroku go przyjmuje — po prostu każe iteracji dojść do końca:
    # zmierzone 7.63e-06 (18 iteracji) -> 2.98e-08 (26), |f| 5.8e-11 -> 8.9e-16.
    r = numerics.root_find(parse("x^2"), "x", 1.0, full_result=True)
    assert r.converged is True and abs(r.value) < 1e-7 and r.residual < 1e-14
    r2 = numerics.root_find(parse("(x-1)^2"), "x", 0.0, full_result=True)
    assert r2.converged is True and abs(r2.value - 1.0) < 1e-7


def test_module8_root_settle_threshold_separates_the_two_families():
    # Predykat wprost, na liczbach ZMIERZONYCH w punktach wyjścia całej tabeli:
    # pierwiastki prawdziwe mają korektę <= 4.1e-10, ogony >= 1.0e-2 — siedem rzędów
    # przerwy, a próg sqrt(eps)=1.49e-8 leży w środku.
    assert numerics._root_has_settled(1.41421356237, 1.595e-12) is True   # x²-2
    assert numerics._root_has_settled(3.141592654, 4.102e-10) is True     # sin w π
    assert numerics._root_has_settled(-24.0, 1.0) is False                # exp
    assert numerics._root_has_settled(19.75, 2.532e-02) is False          # exp(-x²)
    # próg jest WZGLĘDNY dla dużego iteratu (max(1,|x|)) — jak w _diverge_limit
    assert numerics._root_has_settled(1e8, 1.0) is True
    assert numerics._root_has_settled(1.0, 1.0) is False


def test_module8_sign_change_is_tested_by_sign_not_by_the_product():
    # PUŁAPKA ZMIERZONA NA ŻYWO: po naprawie połowy newtonowskiej exp(x) maszerował
    # dalej, aż DWIE malutkie wartości TEGO SAMEGO znaku dały iloczyn 0.0 przez
    # niedomiar — i dawny test `prev_f*fk <= 0` zobaczył „zmianę znaku" tam, gdzie
    # jej nie ma. Bisekcja wyprodukowała wtedy pierwiastek exp(x) w -372.75.
    assert 1e-200 * 1e-201 == 0.0                       # POWÓD, nie hipoteza
    assert numerics._straddle(1e-200, 1e-201) is False  # ten sam znak — nic tu nie ma
    assert numerics._straddle(1.0, -1.0) is True
    assert numerics._straddle(-1.0, 1.0) is True
    assert numerics._straddle(1e308, 1e308) is False    # lustro: iloczyn -> inf
    assert numerics._straddle(0.0, 1.0) is None         # dotknięcie zera — decyduje
    assert numerics._straddle(1.0, 0.0) is None         # _isolated_zero


def test_module8_an_underflowed_zero_is_not_a_root():
    # Drugie ostrze tej samej pułapki: exp(-746) to w float64 DOKŁADNIE 0.0, i tak
    # jest już na całej półprostej w lewo. Zero na płaskowyżu zer to koniec skali
    # wykładnika, nie miejsce zerowe; zero IZOLOWANE (x² w 0) to pierwiastek.
    assert math.exp(-746) == 0.0                        # POWÓD, nie hipoteza
    f_exp = numerics._scalar(parse("exp(x)"), "x")
    assert numerics._isolated_zero(f_exp, -746.0) is False
    f_sq = numerics._scalar(parse("x^2"), "x")
    assert numerics._isolated_zero(f_sq, 0.0) is True


def test_module8_an_exact_zero_is_still_accepted_without_a_model():
    # Regresja na nadgorliwość: gdy f(x) jest DOKŁADNIE 0, żaden model nie jest
    # potrzebny (i nie byłby dostępny — w pierwiastku wielokrotnym f' też znika).
    root = numerics.root_find(parse("cos(x)-x"), "x", 0.0, full_result=True)
    assert root.converged is True and root.residual == 0.0
    # bisekcja po REALNEJ zmianie znaku zostaje nietknięta (cykl Newtona x³-2x+2)
    assert numerics.root_find(parse("x^3 - 2*x + 2"), "x", 0.0) == pytest.approx(
        -1.7692923542, abs=1e-6
    )


def test_module8_functions_with_no_root_and_no_decay_are_unchanged():
    # OBRONA TEGO, CO JUŻ DZIAŁAŁO: tam, gdzie funkcja NIE opada do zera, dawny
    # strażnik był poprawny i musi mówić dokładnie to samo, co przedtem.
    for src in ("x^2+1", "exp(x)+1", "cosh(x)"):
        r = numerics.root_find(parse(src), "x", 0.0, full_result=True)
        assert r.converged is False and r.status == "not_converged"
        assert r.residual == 1.0
    with pytest.raises(NonConvergenceError, match="no real root nearby"):
        numerics.root_find(parse("x^2 + 1"), "x", 1.0)  # dawny komunikat, bez zmian
