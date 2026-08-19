"""Testy kwadratury — oszacowanie błędu i QuadratureResult (moduł 5 serii v0.3).

Kontrakt modułu 5:
* ścieżka DOMYŚLNA (bez `tol`, bez `full_result`) jest NIETKNIĘTA — ta sama reguła na
  tej samej siatce, ta sama arytmetyka w tej samej kolejności, bit w bit;
* `full_result=True` dokłada OSZACOWANIE BŁĘDU (Richardson: `16/15·|I_2n − I_n|`) —
  jeden dodatkowy przebieg po SKOMPILOWANEJ funkcji, zero wywołań SymPy na próbkę;
* `tol` zamienia stałą siatkę w LOKALNIE ADAPTACYJNĄ, która realnie dochodzi do
  żądanej dokładności — albo odmawia słownikiem modułu 2 (`BudgetExhaustedError`,
  `StagnationError`);
* odmowy WEJŚCIA (`DomainError`) rzucają w OBIE strony, jak w module 4.

Źródło niezależne: wartości domyślne kotwiczymy na liczbach zmierzonych na kodzie
PRZED modułem 5 (i na kotwicach z tests/test_numerics.py), a dokładność liczymy wobec
całek o ZNANEJ postaci zamkniętej — nie wobec tego, co silnik sam o sobie mówi.
Oszacowanie przeliczamy DRUGĄ DROGĄ: z dwóch niezależnych wywołań `integrate_num`
(n i 2n), a nie z pola, które silnik raportuje.

NAJWAŻNIEJSZY test w tym pliku to `test_the_error_estimate_can_lie_...`: pinuje
przypadek, w którym oszacowanie mówi 0.0, a prawdziwy błąd wynosi 100%. Oszacowanie
NIE jest ograniczeniem górnym i nikt nie ma prawa go tak czytać.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
import pathlib

import pytest

from pycodemath import SOLVE_STATUSES, QuadratureResult, SolveResult, parse
from pycodemath.core.errors import (
    BudgetExhaustedError,
    DomainError,
    StagnationError,
)
from pycodemath.engine import numerics

# --- całki o ZNANEJ postaci zamkniętej — jedyne źródło prawdy o dokładności -----
#: (id, wyrażenie, a, b, wartość dokładna)
_SMOOTH = [
    ("2x on [0,1]", "2*x", 0.0, 1.0, 1.0),
    ("sin on [0,pi]", "sin(x)", 0.0, math.pi, 2.0),
    ("exp on [0,1]", "exp(x)", 0.0, 1.0, math.e - 1.0),
]
#: Integrandy, na których stałe n=100 jest ZŁE — zmierzone w analizie modułu 5.
_HARD = [
    # ostry pik: 5.5% błędu przy n=100, stała siatka potrzebuje n=1600 na 1e-10
    ("sharp peak", "1/(1+10000*x^2)", -1.0, 1.0, 2.0 * math.atan(100.0) / 100.0),
    # wąski gauss: 19% błędu przy n=100
    ("narrow gauss", "exp(-10000*x^2)", -1.0, 1.0, math.sqrt(math.pi) / 100.0),
    # pochodna osobliwa na brzegu: rząd Simpsona spada z 4 do ~1.5, stała siatka
    # potrzebuje 1 638 400 podprzedziałów na 1e-10 względnie
    ("sqrt endpoint", "sqrt(x)", 0.0, 1.0, 2.0 / 3.0),
    # szybka oscylacja
    ("sin 50x", "sin(50*x)", 0.0, 1.0, (1.0 - math.cos(50.0)) / 50.0),
]


def _integrate(text, a, b, n=100, **kw):
    return numerics.integrate_num(parse(text), "x", a, b, n, **kw)


# =========================================================================
# sam typ wyniku — i DECYZJA, że jest osobny
# =========================================================================
def test_quadrature_result_is_frozen_and_slotted():
    # ten sam reżim co SolveResult: wynik biegu to ZAPIS tego, co się stało
    r = QuadratureResult.success(1.0, 1e-9, 201, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.value = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.error_estimate = 0.0  # type: ignore[misc]
    assert not hasattr(r, "__dict__")
    assert QuadratureResult.__slots__ == (
        "value",
        "error_estimate",
        "evaluations",
        "refinements",
        "converged",
        "status",
    )


def test_quadrature_gets_its_OWN_type_and_the_two_contracts_do_not_leak():
    # DECYZJA modułu 5, zapinowana mechanicznie: kwadratura NIE reużywa SolveResult,
    # bo musiałaby przeciążyć „residual" (oszacowanie błędu nie jest residuum —
    # nic tu nie jest zerowane, a residuum jest ZMIERZONE, nie WYWNIOSKOWANE) i
    # „iterations" (kwadratura nie iteruje do punktu — jej pracą są PRÓBKI).
    # Test pilnuje, żeby żadne z tych pól nie wyciekło do drugiego typu.
    quad = {f.name for f in dataclasses.fields(QuadratureResult)}
    solve = {f.name for f in dataclasses.fields(SolveResult)}
    assert "residual" not in quad and "iterations" not in quad
    assert "error_estimate" not in solve and "evaluations" not in solve
    assert not issubclass(QuadratureResult, SolveResult)
    # a to, co WSPÓLNE, jest wspólne świadomie: jedna flaga do rozgałęziania i JEDEN
    # słownik statusów dla całego pakietu
    assert {"converged", "status"} <= quad & solve
    r = _integrate("sin(x)", 0.0, math.pi, full_result=True)
    assert isinstance(r, QuadratureResult) and not isinstance(r, SolveResult)
    assert r.status in SOLVE_STATUSES


def test_constructors_pair_the_flag_with_its_status():
    ok = QuadratureResult.success(2.0, 1e-8, 201, 0)
    assert (ok.converged, ok.status) == (True, "converged")
    bad = QuadratureResult.failure(2.0, math.inf, 2001, 477, "not_converged")
    assert (bad.converged, bad.status) == (False, "not_converged")


def test_failure_constructor_refuses_a_non_failure_status():
    # ta sama walidacja RUNTIME co w SolveResult.failure i z tej samej listy —
    # żaden z dwóch typów nie może wymyślić statusu spoza SOLVE_STATUSES
    with pytest.raises(ValueError, match="failure status"):
        QuadratureResult.failure(1.0, 1.0, 1, 0, "converged")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="failure status"):
        QuadratureResult.failure(1.0, 1.0, 1, 0, "almost")  # type: ignore[arg-type]
    assert {"success", "failure"} <= set(vars(QuadratureResult))


def test_public_api_exports_the_quadrature_result_type():
    import pycodemath

    assert pycodemath.QuadratureResult is QuadratureResult
    assert "QuadratureResult" in pycodemath.__all__


# =========================================================================
# (a) ścieżka DOMYŚLNA bit w bit
# =========================================================================
#: Wartości zmierzone na kodzie PRZED modułem 5 (`git stash` + odczyt repr) — pinowane
#: przez `==`, nie `approx`: obietnicą modułu jest brak zmiany ANI JEDNEGO bitu.
#: Pokryte: n domyślne, n=2, n ODD (zaokrąglane w górę), granice odwrócone (b < a),
#: aliasing (cos(16 pi x) przy n=4) i integrand z log.
_PINNED = [
    ("2*x", 0, 1, 100, 1.0000000000000002),
    ("sin(x)", 0.0, math.pi, 100, 2.0000000108245044),
    ("exp(x)", 0.0, 1.0, 100, 1.7182818285545038),
    ("1/(1+10000*x^2)", -1.0, 1.0, 100, 0.02950331515899545),
    ("sqrt(x)", 0.0, 1.0, 100, 0.666585482066724),
    ("sin(50*x)", 0.0, 1.0, 100, 0.0007009301574988847),
    ("exp(-10000*x^2)", -1.0, 1.0, 100, 0.014310170408337155),
    ("cos(16*pi*x)", 0.0, 1.0, 4, 1.0),
    ("x^3", 0.0, 2.0, 2, 4.0),
    ("2*x", 1.0, 0.0, 100, -0.9999999999999999),
    ("sin(x)", 0.0, math.pi, 7, 2.000269169948388),
    ("sin(x)", 0.0, math.pi, 6, 2.0008631896735363),
    ("log(x)", 1.0, 2.0, 10, 0.38629340380480576),
]


@pytest.mark.parametrize("text,a,b,n,pinned", _PINNED, ids=[c[0] + f"/n={c[3]}" for c in _PINNED])
def test_default_path_is_bit_identical_to_the_pre_module5_value(text, a, b, n, pinned):
    got = _integrate(text, a, b, n)
    assert type(got) is float
    assert got == pinned, f"DRYF: {got!r} != {pinned!r}"


def test_default_path_still_matches_the_anchors_from_test_numerics():
    # te same dwie kotwice, co w tests/test_numerics.py — przeniesione, nie przeliczone
    assert _integrate("2*x", 0, 1) == pytest.approx(1.0)
    assert _integrate("sin(x)", 0.0, math.pi) == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("text,a,b,n,pinned", _PINNED, ids=[c[0] + f"/n={c[3]}" for c in _PINNED])
def test_full_result_value_equals_the_default_return_exactly(text, a, b, n, pinned):
    # sedno opt-in: włączenie full_result nie może przesunąć ani jednego bitu wyniku
    try:
        r = _integrate(text, a, b, n, full_result=True)
    except DomainError:
        pytest.skip("oszacowanie odmawia na tym integrandzie — pokryte osobnym testem")
    assert type(r.value) is float
    assert r.value == pinned


def test_new_parameters_are_keyword_only_so_no_positional_caller_breaks():
    # cli/repl.py woła integrate_num POZYCYJNIE (expr, var, a, b) — a testy CLI
    # przechodzą nietknięte. Tutaj pilnujemy, że nie da się wstrzyknąć tol/max_evals/
    # full_result pozycyjnie, więc żaden przyszły wołający ich nie pomiesza z n.
    assert _integrate("2*x", 0, 1, 100) == 1.0000000000000002
    with pytest.raises(TypeError):
        numerics.integrate_num(parse("2*x"), "x", 0, 1, 100, 1e-8)  # type: ignore[misc]
    sig = inspect.signature(numerics.integrate_num)
    for name in ("tol", "max_evals", "full_result"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_empty_interval_is_exact_and_costs_nothing():
    # a == b: dawniej goły 0.0 (BEZ kompilacji wyrażenia — dlatego integrand z drugą
    # zmienną NIE jest tu odrzucany; ta kolejność zostaje nietknięta)
    assert _integrate("x*y", 2.0, 2.0) == 0.0
    r = _integrate("x*y", 2.0, 2.0, full_result=True)
    assert (r.value, r.error_estimate, r.evaluations, r.refinements) == (0.0, 0.0, 0, 0)
    assert r.converged is True and r.status == "converged"


# =========================================================================
# (b) OSZACOWANIE BŁĘDU — co potrafi i KIEDY KŁAMIE
# =========================================================================
def test_fixed_path_estimate_costs_exactly_one_extra_pass_over_the_compiled_function():
    # Kontrakt kosztu: wartość to n+1 próbek (jak dotąd), oszacowanie dokłada
    # DOKŁADNIE n (środki siatki) — jeden przebieg, ani jednej próbki więcej.
    # Liczymy realne wywołania warstwy próbkującej, nie wierzymy raportowanemu polu.
    calls: list[float] = []
    real_sample = numerics._sample

    def counting(f, x):
        calls.append(x)
        return real_sample(f, x)

    numerics._sample = counting  # type: ignore[assignment]
    try:
        _integrate("sin(x)", 0.0, math.pi, 100)
        assert len(calls) == 101, "ścieżka domyślna nie może próbkować więcej niż dawniej"
        calls.clear()
        r = _integrate("sin(x)", 0.0, math.pi, 100, full_result=True)
    finally:
        numerics._sample = real_sample  # type: ignore[assignment]
    # środki siatki idą przez _safe (mogą wypaść poza dziedzinę), więc licznik _sample
    # pokazuje samą wartość; raportowany koszt obejmuje oba przebiegi
    assert len(calls) == 101
    assert r.evaluations == 201 == 2 * 100 + 1
    assert r.refinements == 0, "stała siatka nic nie dzieli"


@pytest.mark.parametrize("cid,text,a,b,exact", _SMOOTH, ids=[c[0] for c in _SMOOTH])
def test_estimate_recomputed_a_second_way_matches_the_reported_field(cid, text, a, b, exact):
    # DRUGA DROGA: 16/15·|I_2n − I_n| z dwóch NIEZALEŻNYCH wywołań integrate_num,
    # a nie z wnętrza silnika. Ta sama liczba musi wyjść z obu stron.
    r = _integrate(text, a, b, 100, full_result=True)
    i_n = _integrate(text, a, b, 100)
    i_2n = _integrate(text, a, b, 200)
    assert r.error_estimate == pytest.approx(16.0 / 15.0 * abs(i_2n - i_n), rel=1e-12)


@pytest.mark.parametrize(
    "cid,text,a,b,exact",
    _SMOOTH[1:] + [_HARD[3]],
    ids=[c[0] for c in _SMOOTH[1:] + [_HARD[3]]],
)
def test_estimate_tracks_the_true_error_on_a_smooth_integrand(cid, text, a, b, exact):
    # Tam, gdzie ZAŁOŻENIE (ograniczona czwarta pochodna) jest spełnione, oszacowanie
    # trafia w prawdziwy błąd z dokładnością do czynnika ~2 — zmierzone: sin 1.00,
    # exp 1.00, sin 50x 1.00. To jest cała wartość tego pola.
    r = _integrate(text, a, b, 100, full_result=True)
    true_error = abs(r.value - exact)
    assert true_error > 0.0
    assert 0.5 <= r.error_estimate / true_error <= 2.0, (r.error_estimate, true_error)


def test_estimate_is_large_where_the_fixed_grid_is_badly_wrong():
    # Odwrotna strona: przy n=100 ostry pik jest zły o 5.5%, a wąski gauss o 19% —
    # i oszacowanie MÓWI o tym (rząd wielkości ten sam). Dziś agent nie miał
    # z czego tego wyczytać.
    for cid, text, a, b, exact in _HARD[:2]:
        r = _integrate(text, a, b, 100, full_result=True)
        true_error = abs(r.value - exact)
        assert true_error / abs(exact) > 0.05, (cid, true_error)
        assert r.error_estimate > 1e-4, (cid, r.error_estimate)
        assert 0.1 <= r.error_estimate / true_error <= 10.0, (cid, r.error_estimate, true_error)


def test_the_error_estimate_can_lie_so_it_is_never_a_bound():
    # NAJWAŻNIEJSZY test pliku. cos(16 pi x) na [0,1]: całka dokładna = 0.
    # Przy n=4 KAŻDY węzeł siatki n ORAZ jej połowienia trafia w maksimum cosinusa,
    # więc obie siatki zwracają dokładnie 1.0, różnica jest ZEREM i oszacowanie
    # wychodzi 0.0 — przy prawdziwym błędzie 100%. Założenie Richardsona (siatka
    # widzi kształt funkcji) jest tu złamane, a oszacowanie tego NIE WIE.
    r = _integrate("cos(16*pi*x)", 0.0, 1.0, 4, full_result=True)
    assert r.value == 1.0
    assert r.error_estimate == 0.0
    assert abs(r.value - 0.0) == 1.0  # prawdziwy błąd: 100% wartości
    assert r.converged is True  # bo o żadną dokładność nie proszono
    # ta sama pułapka domyka się na adaptacyjnej ścieżce: kryterium jest MIERZONE
    # tym samym oszacowaniem, więc „osiągnięta tolerancja" dziedziczy jego status
    adaptive = _integrate("cos(16*pi*x)", 0.0, 1.0, 4, tol=1e-12, full_result=True)
    assert adaptive.converged is True and adaptive.refinements == 0
    assert abs(adaptive.value - 0.0) == 1.0
    # i lekarstwo, które NIE jest gwarancją, tylko rozdzielczością: gęstszy zasiew
    # (domyślne n=100) widzi tę oscylację i wynik jest poprawny
    dense = _integrate("cos(16*pi*x)", 0.0, 1.0, 100, tol=1e-12, full_result=True)
    assert abs(dense.value) < 1e-10


def test_converged_on_the_fixed_path_is_not_a_quality_claim():
    # Granica znaczenia pola: bez `tol` nie zamawiano dokładności, więc nic nie mogło
    # jej nie dowieźć — `converged=True` mówi „reguła policzyła", NIE „wynik jest
    # dobry". Jakość mieszka w error_estimate i tylko tam.
    r = _integrate("exp(-10000*x^2)", -1.0, 1.0, 100, full_result=True)
    assert r.converged is True and r.status == "converged"
    assert abs(r.value - math.sqrt(math.pi) / 100.0) / (math.sqrt(math.pi) / 100.0) > 0.15
    assert r.error_estimate > 1e-3


# =========================================================================
# (c) ŚCIEŻKA ADAPTACYJNA — czy realnie dochodzi do żądanej dokładności
# =========================================================================
@pytest.mark.parametrize("cid,text,a,b,exact", _HARD, ids=[c[0] for c in _HARD])
@pytest.mark.parametrize("tol", [1e-8, 1e-10])
def test_adaptive_actually_reaches_the_requested_tolerance(cid, text, a, b, exact, tol):
    # Sedno punktu 4: tolerancja jest DOWIEZIONA, mierzone wobec postaci zamkniętej,
    # a nie wobec własnego oszacowania silnika.
    r = _integrate(text, a, b, tol=tol, full_result=True)
    assert r.converged is True and r.status == "converged"
    assert abs(r.value - exact) <= tol, (cid, abs(r.value - exact))
    assert r.error_estimate <= tol
    assert r.refinements > 0, "trudny integrand MUSI wymagać podziałów"
    assert r.evaluations < 20_000, (cid, r.evaluations)
    # goły zwrot == .value, bit w bit (opt-in dotyczy KSZTAŁTU, nie liczby)
    assert _integrate(text, a, b, tol=tol) == r.value


def test_adaptive_beats_a_fixed_grid_a_hundredfold_where_the_grid_fails():
    # Zmierzony powód istnienia tej ścieżki. sqrt(x) na [0,1]: pochodna osobliwa w 0,
    # więc GLOBALNA siatka musi zagęścić WSZĘDZIE, żeby naprawić JEDEN róg.
    exact = 2.0 / 3.0
    coarse = abs(_integrate("sqrt(x)", 0.0, 1.0, 100) - exact)
    dense = abs(_integrate("sqrt(x)", 0.0, 1.0, 10_000) - exact)
    r = _integrate("sqrt(x)", 0.0, 1.0, tol=1e-10, full_result=True)
    assert coarse > 1e-5  # zmierzone: 8.1e-05 przy 101 próbkach
    assert dense > 1e-9  # zmierzone: ~8e-08 przy 10 001 próbkach — nadal za mało
    # ...a lokalna subdywizja robi to lepiej o rzędy wielkości, przy ~1000 próbkach
    assert abs(r.value - exact) < 1e-12
    assert r.evaluations < dense_budget(), r.evaluations


def dense_budget() -> int:
    """Budżet próbek gęstej stałej siatki z testu powyżej (10 000 podprzedziałów)."""
    return 10_001


def test_adaptive_leaves_a_smooth_integrand_alone():
    # Ekonomia: gdzie stała siatka już wystarcza, adaptacja nie dzieli nic i kosztuje
    # tyle, ile zasiew (n+1 próbek siatki + 2 na panel).
    r = _integrate("exp(x)", 0.0, 1.0, tol=1e-8, full_result=True)
    assert r.converged is True
    assert r.refinements == 0
    assert r.evaluations == 201
    assert abs(r.value - (math.e - 1.0)) < 1e-8


def test_adaptive_seed_is_the_fixed_rule_so_n_still_means_resolution():
    # n na ścieżce adaptacyjnej NIE jest ignorowane: to gęstość ZASIEWU, czyli
    # rozdzielczość, z jaką kryterium w ogóle widzi integrand (patrz test o kłamstwie).
    coarse = _integrate("sqrt(x)", 0.0, 1.0, 2, tol=1e-8, full_result=True)
    dense = _integrate("sqrt(x)", 0.0, 1.0, 100, tol=1e-8, full_result=True)
    assert coarse.converged and dense.converged
    assert coarse.evaluations < dense.evaluations  # mniej zasiewu = mniej próbek
    assert abs(coarse.value - 2.0 / 3.0) <= 1e-8
    assert abs(dense.value - 2.0 / 3.0) <= 1e-8


# =========================================================================
# (d) PORAŻKI: rzucają domyślnie, są ZWRACALNE przy full_result (kontrakt modułu 4)
# =========================================================================
def test_budget_exhaustion_raises_the_budget_class_and_is_returnable():
    # sin(50x) z tol=1e-12 potrzebuje ZMIERZONYCH 18 237 ewaluacji; z budżetem 2000
    # trudność jest wszędzie i podziały nie mają z czego się dokończyć.
    run = dict(tol=1e-12, max_evals=2000)
    with pytest.raises(BudgetExhaustedError) as exc:
        _integrate("sin(50*x)", 0.0, 1.0, **run)
    msg = str(exc.value)
    assert "exhausted its budget of 2000 function evaluations" in msg
    assert "panel(s) are still unrefined" in msg
    assert "tol=1e-12" in msg and "raise max_evals" in msg

    r = _integrate("sin(50*x)", 0.0, 1.0, **run, full_result=True)
    assert r.converged is False and r.status == "not_converged"
    # REGUŁA: część przedziału nie została w ogóle wyrafinowana, więc błąd jest
    # NIEMIERZALNY — to samo `inf` i ten sam powód co w module 4 dla residuum
    assert r.error_estimate == math.inf
    assert math.isfinite(r.value)  # wartość jest: zgrubna, ale nie NaN
    assert r.evaluations >= 2000 and r.refinements > 0


def test_budget_class_is_deliberately_not_the_solver_nonconvergence_class():
    # DECYZJA modułu 5: cap `max_iter` solwera to DIAGNOZA o problemie (bieg nie ma
    # nic do pokazania) → NonConvergenceError. Tu integrand jest w porządku, wynik i
    # trwające rafinowanie ISTNIEJĄ, a skończyło się wyłącznie pozwolenie wołającego
    # → to BUDŻET. Dowód, że to nie kosmetyka, a różnica zachowania: TEN SAM bieg,
    # ta sama tolerancja, tylko większy budżet — i DOCHODZI (zmierzone: 2005 ewaluacji
    # to porażka, 18 237 to sukces, więc granica leży w budżecie, nie w integrandzie).
    exact = (1.0 - math.cos(50.0)) / 50.0
    with pytest.raises(BudgetExhaustedError):
        _integrate("sin(50*x)", 0.0, 1.0, tol=1e-12, max_evals=2000)
    ok = _integrate("sin(50*x)", 0.0, 1.0, tol=1e-12, max_evals=20_000, full_result=True)
    assert ok.converged is True and ok.status == "converged"
    assert ok.evaluations == 18_237  # deterministyczne: ten algorytm, ta siatka
    assert abs(ok.value - exact) <= 1e-12
    # ...a StagnationError z drugiej porażki NIE ma tej własności: tam większy budżet
    # nic nie da i komunikat to wprost mówi (patrz test poniżej)
    assert issubclass(BudgetExhaustedError, Exception)


def test_stagnation_when_the_float_grid_cannot_resolve_a_panel():
    # Przy x≈1e15 jeden krok float to 0.125, więc panel szerokości 0.02 nie ma już
    # ROZRÓŻNIALNYCH węzłów. Bez tego strażnika S_coarse == S_fine, oszacowanie
    # wyszłoby 0.0 i bieg zameldowałby zbieżność czegoś, czego nie policzył
    # (zmierzone: 100 000 ewaluacji spalonych na panelach bez odrębnych floatów).
    run = dict(a=1e15, b=1e15 + 1.0, tol=1e-30)
    with pytest.raises(StagnationError) as exc:
        _integrate("sin(x)", run["a"], run["b"], tol=run["tol"])
    msg = str(exc.value)
    assert "STALLED" in msg and "resolution limit of double precision" in msg
    assert "a bigger max_evals will not help" in msg

    r = _integrate("sin(x)", run["a"], run["b"], tol=run["tol"], full_result=True)
    assert r.converged is False and r.status == "stagnated"
    # w PRZECIWIEŃSTWIE do wyczerpanego budżetu: każdy panel został ZMIERZONY, więc
    # oszacowanie jest skończone — `inf` znaczy „niemierzalne" i tylko to
    assert math.isfinite(r.error_estimate) and r.error_estimate > r_tol_of(run)
    assert r.evaluations < 1000, "strażnik ma kończyć bieg tanio, nie mielić budżetu"


def r_tol_of(run: dict) -> float:
    """Tolerancja żądana w danym biegu — żeby test porównywał z liczbą z wywołania."""
    return float(run["tol"])


def test_a_floored_panel_alone_is_not_a_failure():
    # Odwrotna strona tego samego strażnika: panel dobity do rozdzielczości siatki
    # bywa NIESZKODLIWY. sqrt(x) z tol=1e-10 zostawia jeden taki panel przy zerze, a
    # łączne oszacowanie nadal mieści się w tol → bieg JEST zbieżny. Werdykt bierze
    # się z CAŁOŚCI, nie z pojedynczego panelu (inaczej byłby fałszywy alarm na
    # sztandarowym przypadku modułu).
    r = _integrate("sqrt(x)", 0.0, 1.0, tol=1e-10, full_result=True)
    assert r.converged is True and r.status == "converged"
    assert r.error_estimate <= 1e-10
    assert abs(r.value - 2.0 / 3.0) < 1e-12


def test_quadrature_never_reports_diverged():
    # USTALENIE (nie przeoczenie): integrand uciekający do nieskończoności jest
    # ODRZUCANY na próbce (DomainError), a nie całkowany do rozbieżnej sumy — więc
    # kwadratura nie ma wyniku, który to słowo opisuje.
    with pytest.raises(DomainError):
        _integrate("1/x", -1.0, 1.0, full_result=True)
    with pytest.raises(DomainError):
        _integrate("1/x", -1.0, 1.0, tol=1e-8)


# =========================================================================
# (e) ODMOWY WEJŚCIA — rzucają w OBIE strony (granica modułu 4)
# =========================================================================
_REFUSALS = [
    ("n=0", lambda **kw: _integrate("x", 0, 1, 0, **kw)),
    ("n<0", lambda **kw: _integrate("x", 0, 1, -4, **kw)),
    ("tol=0", lambda **kw: _integrate("x", 0, 1, tol=0.0, **kw)),
    ("tol<0", lambda **kw: _integrate("x", 0, 1, tol=-1e-6, **kw)),
    ("tol=nan", lambda **kw: _integrate("x", 0, 1, tol=float("nan"), **kw)),
    ("max_evals=0", lambda **kw: _integrate("x", 0, 1, tol=1e-8, max_evals=0, **kw)),
    ("max_evals<0", lambda **kw: _integrate("x", 0, 1, tol=1e-8, max_evals=-5, **kw)),
    ("extra symbol", lambda **kw: _integrate("x*y", 0, 1, **kw)),
    ("endpoint outside domain", lambda **kw: _integrate("sqrt(x)", -1, 1, **kw)),
    ("interior outside domain", lambda **kw: _integrate("log(x)", 0, 1, **kw)),
    ("adaptive sample outside", lambda **kw: _integrate("sqrt(cos(8*pi*x))", 0, 1, 4, tol=1e-8, **kw)),
]


@pytest.mark.parametrize("run", [c[1] for c in _REFUSALS], ids=[c[0] for c in _REFUSALS])
def test_input_refusals_raise_with_and_without_full_result(run):
    # Granica z modułu 4: ZWRACALNE są wyniki BIEGU, nie odmowy wejścia. Zła liczba
    # podprzedziałów czy próbka poza dziedziną nie mają statusu w SOLVE_STATUSES i
    # nie mają „najlepszej wartości, jaką bieg osiągnął" — bieg się nie zaczął.
    with pytest.raises(DomainError):
        run()
    with pytest.raises(DomainError):
        run(full_result=True)


def test_n_and_tol_refusal_messages_are_unchanged_and_readable():
    # komunikat o n jest ten SAM, co przed modułem 5 (bajt w bajt) — nowe parametry
    # dokładają własne, nazywając wartość i mówiąc, co zrobić
    with pytest.raises(DomainError, match="^the number of subintervals n must be positive$"):
        _integrate("x", 0, 1, 0)
    with pytest.raises(DomainError, match="the tolerance tol must be positive, got 0.0"):
        _integrate("x", 0, 1, tol=0.0)
    with pytest.raises(DomainError, match="the evaluation budget max_evals must be positive"):
        _integrate("x", 0, 1, tol=1e-8, max_evals=0)


def test_the_estimate_may_need_a_sample_the_value_did_not_and_says_so():
    # sqrt(cos(8 pi x)) przy n=4: WĘZŁY siatki trafiają w cos=1 (wartość policzalna),
    # ale ŚRODKI trafiają w cos=-1 → pierwiastek z liczby ujemnej. Wartość istnieje,
    # oszacowania nie da się wziąć. Leczenie jak w module 3 dla residuum: czytelna
    # odmowa, która mówi, że odpowiedź jest dostępna bez full_result — a NIE cichy
    # NaN i nie utrata wyniku.
    assert _integrate("sqrt(cos(8*pi*x))", 0.0, 1.0, 4) == 1.0
    with pytest.raises(DomainError) as exc:
        _integrate("sqrt(cos(8*pi*x))", 0.0, 1.0, 4, full_result=True)
    msg = str(exc.value)
    assert "the error estimate needs one sample BETWEEN the grid points" in msg
    assert "x=0.125" in msg  # nazywa punkt, który odmówił
    assert "call without full_result=True" in msg


# =========================================================================
# (f) ANTYDRYF — statusy i konstruktory czytane ze ŹRÓDŁA
# =========================================================================
def _quadrature_statuses_in_numerics() -> "set[str]":
    """Statusy, które kwadratura potrafi ODDAĆ — z AST, nie z grepa.

    Bliźniak helpera z tests/test_result.py: literał w komentarzu albo docstringu nie
    liczy się jako producent. Czytamy 5. argument każdego wywołania
    ``QuadratureResult.failure(...)`` ORAZ literały statusów zwracane przez
    ``_adaptive_quad`` (to on ustala werdykt, a ``integrate_num`` go tylko przekazuje).
    """
    tree = ast.parse(pathlib.Path(numerics.__file__).read_text(encoding="utf-8"))
    produced: set[str] = set()

    def literals(node: ast.AST) -> "set[str]":
        return {
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "failure"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "QuadratureResult"
        ):
            assert len(node.args) == 5, (
                "QuadratureResult.failure(value, error_estimate, evaluations, "
                "refinements, status)"
            )
            produced |= literals(node.args[4])
        if isinstance(node, ast.FunctionDef) and node.name == "_adaptive_quad":
            for ret in ast.walk(node):
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple):
                    assert len(ret.value.elts) == 7, "_adaptive_quad zwraca 7-krotkę"
                    produced |= literals(ret.value.elts[4])
    return produced


def test_every_status_quadrature_can_produce_is_in_the_declared_vocabulary():
    # Ta sama gwarancja co w module 4, dla nowego typu: kwadratura mówi słownikiem
    # SOLVE_STATUSES, nie wymyśla drugiego (np. "budget_exhausted"), i produkuje
    # DOKŁADNIE te dwie porażki, które umie mieć — bez "diverged", którego nie ma
    # z czego wyprodukować (ucieczka do nieskończoności to odmowa na próbce).
    produced = _quadrature_statuses_in_numerics()
    assert produced <= set(SOLVE_STATUSES), produced - set(SOLVE_STATUSES)
    assert produced == {"converged", "stagnated", "not_converged"}
    assert "diverged" not in produced

    src = inspect.getsource(numerics)
    assert "QuadratureResult(" not in src, (
        "obie formy wyniku MUSZĄ powstawać przez success()/failure() — surowy "
        "konstruktor pozwoliłby rozjechać parę (converged, status)"
    )


def test_quadrature_failures_map_to_exactly_one_exception_class_each():
    # Para klasa ↔ status jest JEDNA i ta sama z obu stron: co ścieżka domyślna
    # RZUCA, to full_result ZWRACA. Status wnioskujemy z klasy, nie z tabeli.
    pairs = [
        (BudgetExhaustedError, "not_converged", dict(tol=1e-12, max_evals=2000),
         ("sin(1000*x)", 0.0, 1.0)),
        (StagnationError, "stagnated", dict(tol=1e-30),
         ("sin(x)", 1e15, 1e15 + 1.0)),
    ]
    for cls, status, kw, (text, a, b) in pairs:
        with pytest.raises(cls):
            _integrate(text, a, b, **kw)
        r = _integrate(text, a, b, **kw, full_result=True)
        assert r.status == status and r.converged is False, (cls, r)
