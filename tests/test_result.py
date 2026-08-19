"""Testy strukturalnego wyniku solwera — SolveResult (moduły 3 i 4 serii v0.3).

Kontrakt modułu 3 (SUKCES): ścieżka domyślna NIETKNIĘTA (ten sam typ i ta sama
liczba, bit w bit), a ``full_result=True`` dokłada dowody biegu: liczbę iteracji,
residuum w ZWRÓCONYM punkcie, flagę zbieżności i status.

Kontrakt modułu 4 (PORAŻKA): ścieżka domyślna nadal RZUCA wyjątek modułu 2, z
komunikatem bajt w bajt tym samym, a przy ``full_result=True`` ten sam bieg ZWRACA
``SolveResult`` z ``converged=False`` i statusem odpowiadającym klasie wyjątku.
Odmowy WEJŚCIA (``DomainError``) rzucają w obie strony — to nie wynik iteracji.

Źródło niezależne: wartości domyślne kotwiczymy na liczbach JUŻ zapinowanych w
tests/test_numerics.py (a nie na świeżym przebiegu — inaczej test przyklepałby
dowolny dryf), a residuum przeliczamy z ORYGINALNEGO wyrażenia przez ``evalf``
(ścieżka mpmath w IR), a NIE z tego, co solwer raportuje ani z jego skompilowanej
funkcji. Dwie niezależne drogi liczenia tej samej liczby.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import math
import pathlib

import numpy as np
import pytest

from pycodemath import SOLVE_STATUSES, SolveResult, parse
from pycodemath.core import errors
from pycodemath.core.errors import (
    DivergenceError,
    DomainError,
    NonConvergenceError,
    NotAMinimumError,
    StagnationError,
)
from pycodemath.engine import numerics

_PARABOLOID = "(x-1)^2 + (y+2)^2"
_CIRCLE_LINE = ("x^2 + y^2 - 4", "x - y")
_ROSEN = "(1-x)^2 + 100*(y-x^2)^2"


# --- sama klasa ----------------------------------------------------------
def test_solve_result_is_frozen_and_slotted():
    # zamrożony: wynik biegu jest ZAPISEM tego, co się stało — nie wolno go po
    # fakcie „poprawić" (agent, który dostaje residuum, musi wierzyć, że pochodzi
    # z solwera). Slots: brak __dict__, tani w przekazywaniu.
    r: SolveResult[float] = SolveResult.success(1.5, 3, 1e-12)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.value = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.iterations = 99  # type: ignore[misc]
    assert not hasattr(r, "__dict__")
    assert SolveResult.__slots__ == (
        "value",
        "iterations",
        "residual",
        "converged",
        "status",
    )


def test_success_constructor_pairs_converged_with_its_status():
    # jeden konstruktor ścieżki sukcesu = para (converged, status) nie może się
    # rozejść między solwerami; wynik converged=True ze statusem mówiącym coś
    # innego byłby sprzecznością, na której agent rozgałęzia się w ciemno
    r = SolveResult.success([1.0, 2.0], 7, 0.5)
    assert r.converged is True
    assert r.status == "converged"
    assert r.value == [1.0, 2.0]
    assert r.iterations == 7
    assert r.residual == 0.5


def test_public_api_exports_the_result_type():
    import pycodemath

    assert pycodemath.SolveResult is SolveResult
    for name in ("SolveResult", "SolveStatus", "SOLVE_STATUSES"):
        assert name in pycodemath.__all__, name


def test_status_vocabulary_mirrors_the_module2_outcomes():
    # słownik statusów jest DOMKNIĘTY na tych samych porażkach, które moduł 2
    # wydzielił jako klasy wyjątków — moduł 4 ma czym zwrócić porażkę, nie
    # wymyślając nowego nazewnictwa.
    #
    # ŚWIADOME PRZEPISANIE (moduł 8, przypadek (i)): dochodzi PIĄTY status,
    # "not_a_minimum", RAZEM ze swoją klasą NotAMinimumError — i to jest cała
    # treść tej granicy. Pilnuje ona nie DŁUGOŚCI listy, tylko odwzorowania 1:1
    # status ↔ wyjątek; rozszerzenie o samą nazwę statusu (bez klasy) albo o samą
    # klasę (bez statusu) rozjechałoby dokładnie to, czego moduły 2–4 broniły.
    # DLACZEGO nowy, a nie zmapowany na istniejący: to jedyny wynik, w którym bieg
    # NIE miał kłopotu — DOSZEDŁ, tylko do punktu stacjonarnego, który okazał się
    # maksimum/siodłem/przegięciem. Pozostałe trzy mówią, że iterat nigdzie nie
    # dotarł (uciekł / skończył budżet) albo że stanął, będąc wciąż OD celu.
    # Różnica jest praktyczna, nie kosmetyczna: agent czytający "not_converged"
    # podniesie max_iter, a tu więcej iteracji NIE POMOŻE NIGDY — krok gd w punkcie
    # stacjonarnym jest zerowy. Pomaga wyłącznie inny punkt startowy.
    assert SOLVE_STATUSES == (
        "converged",
        "diverged",
        "stagnated",
        "not_converged",
        "not_a_minimum",
    )
    # Odwzorowanie 1:1, sprawdzone WPROST (nie arytmetyką na długościach): każdy
    # status porażki ma swoją klasę i odwrotnie. Poza parowaniem świadomie zostają:
    # ParseError i DomainError (odmowy WEJŚCIA — nie wyniki iteracji, więc nie mają
    # czego raportować), BudgetExhaustedError (należy do kwadratury, która na tej
    # samej liście statusów mapuje go na "not_converged" — patrz
    # tests/test_quadrature.py) oraz, od modułu 9, TimeBudgetError.
    #
    # PRZEPISANY W MODULE 9 — przypadek (i): ten wiersz pinował granicę, którą
    # moduł 9 świadomie przesuwa, dokładając ÓSMĄ klasę. Ale przesuwa ją w drugą
    # stronę niż moduł 8: tam nowa klasa DOSTAŁA status, tu świadomie go NIE
    # dostaje, i to jest teza, którą ten test pinuje.
    #
    # DLACZEGO BEZ STATUSU. SOLVE_STATUSES to słownik WYNIKÓW ITERACJI i każde jego
    # słowo opisuje, gdzie wylądował iterat: uciekł, stanął, skończył budżet, doszedł
    # nie tam. Bieg SYMBOLICZNY nie ma iteratu — nie ma ostatniego punktu, nie ma
    # residuum, nie ma SolveResult, w którym status miałby zamieszkać. Dopisanie
    # słowa dałoby `SolveResult` wartość, której ŻADEN solver nie potrafi
    # wyprodukować i której żaden wołający nie odczyta z wyniku; sprawdza to wprost
    # test_every_status_a_solver_can_produce_is_in_the_declared_vocabulary.
    # Reguła modułu 8 („status i wyjątek chodzą parami, 1:1") NIE jest tu złamana:
    # jest spełniona z drugiej strony — nie ma statusu, więc nie ma pary, a lista
    # `unpaired` niżej jest tym, co pilnuje, że brak pary jest DECYZJĄ, a nie
    # przeoczeniem. Na drucie MCP TimeBudgetError jedzie drogą odmowy WEJŚCIA
    # (`error.type`, `solve`/`quadrature` = null) — dokładnie tak, jak moduł 6
    # zapowiedział dla silników bez wyniku strukturalnego.
    #
    # PRZEPISANY PONOWNIE W MODULE 10 — przypadek (i), i tym razem rozstrzyga
    # WPROST pytanie, które moduł 9 zostawił otwarte („czy silnik symboliczny
    # potrzebuje wyniku strukturalnego — to robota następnego modułu").
    # ODPOWIEDŹ: NIE, a rozumowanie modułu 9 STOI, nie zostaje obalone.
    #
    # Moduł 10 zmierzył, CO ISTNIEJE w chwili odmowy symbolicznej i mogłoby zapełnić
    # pole wyniku: nie ma iteratu, nie ma residuum, nie ma licznika ewaluacji, nie ma
    # częściowej wartości. `SolveResult` ma pięć pól i CZTERY z nich nie mają tu
    # sensu; `SymbolicResult` niósłby `status` i nic więcej, czyli `error.type` w
    # przebraniu dataclassy. Zamiast tego moduł 10 rozbił klasy (NoClosedFormError,
    # UnsupportedFormError) i dołożył JEDNĄ daną, której z klasy ani z komunikatu nie
    # dało się wyczytać: `route`. Obie nowe klasy są więc BEZ STATUSU z dokładnie
    # tego samego powodu co TimeBudgetError — i dlatego dopisują się do tej listy, a
    # nie do SOLVE_STATUSES.
    assert set(_STATUS_OF_CLASS.values()) == set(SOLVE_STATUSES) - {"converged"}
    unpaired = {
        c
        for c in vars(errors).values()
        if isinstance(c, type)
        and issubclass(c, errors.PycodemathError)
        and c is not errors.PycodemathError
        and c not in _STATUS_OF_CLASS
    }
    assert unpaired == {
        errors.ParseError,
        errors.DomainError,
        errors.BudgetExhaustedError,
        errors.TimeBudgetError,
        errors.NoClosedFormError,
        errors.UnsupportedFormError,
    }


# --- (a) ścieżka domyślna BIT-IDENTYCZNA z kotwicami z test_numerics ------
def test_default_return_is_bit_identical_to_the_pinned_anchors():
    # Kotwice przeniesione z tests/test_numerics.py (nie ze świeżego przebiegu):
    # root_find x²−2 z x0=1 -> √2; minimize (x−3)² gd -> zapinowana DOKŁADNA
    # liczba 2.999999996357496; root_find_nd okrąg∩prosta -> (√2, √2);
    # minimize_nd paraboloida -> (1, −2). Typy też muszą zostać te same.
    root = numerics.root_find(parse("x^2 - 2"), "x", 1.0)
    assert type(root) is float
    assert root == pytest.approx(math.sqrt(2))

    xmin = numerics.minimize(parse("(x-3)^2"), "x", 0.0)
    assert type(xmin) is float
    assert xmin == pytest.approx(2.999999996357496, abs=1e-15)

    sol = numerics.root_find_nd(
        [parse(_CIRCLE_LINE[0]), parse(_CIRCLE_LINE[1])], ["x", "y"], [1.0, 1.0]
    )
    assert type(sol) is list
    assert np.allclose(sol, [math.sqrt(2), math.sqrt(2)])

    sol_nd = numerics.minimize_nd(parse(_PARABOLOID), ["x", "y"], [0.0, 0.0])
    assert type(sol_nd) is list
    assert np.allclose(sol_nd, [1.0, -2.0], atol=1e-4)


# --- (b) .value == wynik domyślny DOKŁADNIE (nie approx) ------------------
@pytest.mark.parametrize("method", ["gd", "newton", "bfgs"])
def test_full_result_value_equals_the_default_return_exactly(method):
    # sedno opt-in: włączenie full_result nie może przesunąć ANI JEDNEGO bitu
    # wyniku — inaczej „opcjonalna głębia" byłaby zmianą zachowania
    kwargs = {"method": method} if method == "gd" else {"method": method, "max_iter": 5}
    plain = numerics.minimize(parse("(x-3)^2"), "x", 0.0, **kwargs)
    rich = numerics.minimize(parse("(x-3)^2"), "x", 0.0, full_result=True, **kwargs)
    assert isinstance(rich, SolveResult)
    assert rich.value == plain  # ==, nie approx
    assert type(rich.value) is type(plain)

    plain_nd = numerics.minimize_nd(parse(_PARABOLOID), ["x", "y"], [0.0, 0.0], **kwargs)
    rich_nd = numerics.minimize_nd(
        parse(_PARABOLOID), ["x", "y"], [0.0, 0.0], full_result=True, **kwargs
    )
    assert rich_nd.value == plain_nd  # lista float, porównanie DOKŁADNE
    assert type(rich_nd.value) is type(plain_nd)


def test_full_result_value_equals_the_default_return_exactly_for_roots():
    for x0 in (1.0, 0.0):  # 1.0 = Newton, 0.0 = pochodna zeruje się -> bisekcja
        plain = numerics.root_find(parse("x^2 - 2"), "x", x0)
        rich = numerics.root_find(parse("x^2 - 2"), "x", x0, full_result=True)
        assert rich.value == plain
        assert type(rich.value) is type(plain)

    exprs = [parse(_CIRCLE_LINE[0]), parse(_CIRCLE_LINE[1])]
    plain_nd = numerics.root_find_nd(exprs, ["x", "y"], [1.0, 1.0])
    rich_nd = numerics.root_find_nd(exprs, ["x", "y"], [1.0, 1.0], full_result=True)
    assert rich_nd.value == plain_nd
    assert type(rich_nd.value) is type(plain_nd)


def test_full_result_is_keyword_only_so_no_positional_caller_breaks():
    # parametr wstawiony za ``*``: żaden istniejący wołający pozycyjnie nie może
    # przypadkiem trafić w full_result (ani go tu wcisnąć)
    with pytest.raises(TypeError):
        numerics.root_find(parse("x^2 - 2"), "x", 1.0, 1e-10, 100, True)  # type: ignore[misc]
    for fn in (
        numerics.root_find,
        numerics.root_find_nd,
        numerics.minimize,
        numerics.minimize_nd,
    ):
        param = inspect.signature(fn).parameters["full_result"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
        assert param.default is False, fn.__name__


# --- (c) residuum przeliczone NIEZALEŻNIE z oryginalnego wyrażenia --------
def test_root_find_residual_is_the_function_value_at_the_returned_root():
    # residuum root_find = |f(x)| W ZWRÓCONYM punkcie. Liczymy je drugą drogą:
    # evalf na ORYGINALNYM wyrażeniu (mpmath w IR), nie skompilowaną funkcją
    # solwera. Drobna różnica ostatnich cyfr jest oczekiwana (dwie różne ścieżki
    # ewaluacji przy kasowaniu się składników), sam rząd wielkości — nie.
    expr = parse("x^2 - 2")
    r = numerics.root_find(expr, "x", 1.0, full_result=True)
    independent = abs(expr.evalf(x=r.value))
    assert r.residual == pytest.approx(independent, rel=1e-6, abs=1e-15)
    assert r.residual < 1e-10  # i faktycznie zeruje funkcję (tol biegu)


def test_root_find_nd_residual_is_the_inf_norm_of_the_system_at_the_solution():
    exprs = [parse(_CIRCLE_LINE[0]), parse(_CIRCLE_LINE[1])]
    r = numerics.root_find_nd(exprs, ["x", "y"], [1.0, 1.0], full_result=True)
    independent = max(
        abs(e.evalf(x=r.value[0], y=r.value[1])) for e in exprs
    )  # ‖F(x)‖∞
    assert r.residual == pytest.approx(independent, rel=1e-6, abs=1e-15)
    assert r.residual < 1e-10


@pytest.mark.parametrize("method", ["gd", "newton", "bfgs"])
def test_minimize_residual_is_the_derivative_at_the_returned_point(method):
    # residuum minimize = |f'(x)| (jednowymiarowy przypadek konwencji ‖∇f‖∞).
    # Niezależnie: pochodna SYMBOLICZNA oryginalnego wyrażenia + evalf.
    expr = parse("(x-3)^2")
    kwargs = {"method": method} if method == "gd" else {"method": method, "max_iter": 5}
    r = numerics.minimize(expr, "x", 0.0, full_result=True, **kwargs)
    independent = abs(expr.diff("x").evalf(x=r.value))
    assert r.residual == pytest.approx(independent, rel=1e-6, abs=1e-15)


@pytest.mark.parametrize("method", ["gd", "newton", "bfgs"])
def test_minimize_nd_residual_is_the_gradient_inf_norm_at_the_returned_point(method):
    expr = parse(_PARABOLOID)
    kwargs = {"method": method} if method == "gd" else {"method": method, "max_iter": 5}
    r = numerics.minimize_nd(expr, ["x", "y"], [0.0, 0.0], full_result=True, **kwargs)
    grad = [expr.diff(v) for v in ("x", "y")]
    independent = max(abs(g.evalf(x=r.value[0], y=r.value[1])) for g in grad)
    assert r.residual == pytest.approx(independent, rel=1e-6, abs=1e-15)


def test_residual_is_measured_at_the_returned_point_not_the_best_seen():
    # PO CO ten moduł istnieje: bieg, który „ledwo się wcisnął". Przy lr=1e-3 i
    # tol=1e-4 gd wychodzi po ZNIKAJĄCYM KROKU (lr·|g| < tol), więc gradient w
    # zwróconym punkcie jest ~1e-1 — o trzy rzędy WIĘKSZY od tol. Uczciwe
    # residuum musi to pokazać (a nie zaraportować najmniejszej liczby z biegu),
    # bo tylko po nim agent odróżni solidny wynik od wciśniętego na styk.
    expr = parse("(x-3)^2")
    r = numerics.minimize(expr, "x", 0.0, lr=1e-3, tol=1e-4, full_result=True)
    assert r.converged is True
    assert r.residual > 1e-4  # > tol: zbiegło po kroku, nie po gradiencie
    assert r.residual == pytest.approx(abs(expr.diff("x").evalf(x=r.value)), rel=1e-9)
    # a wynik domyślny nadal ten sam co bez full_result
    assert numerics.minimize(expr, "x", 0.0, lr=1e-3, tol=1e-4) == r.value


# --- (c) iteracje uczciwe: > 0 i ograniczone przez max_iter ---------------
def test_iterations_are_positive_and_bounded_by_max_iter():
    budget = 50
    checks = (
        numerics.root_find(parse("x^2 - 2"), "x", 1.0, max_iter=budget, full_result=True),
        numerics.root_find_nd(
            [parse(_CIRCLE_LINE[0]), parse(_CIRCLE_LINE[1])],
            ["x", "y"],
            [1.0, 1.0],
            max_iter=budget,
            full_result=True,
        ),
        numerics.minimize(
            parse("(x-3)^2"), "x", 0.0, method="newton", max_iter=budget, full_result=True
        ),
        numerics.minimize_nd(
            parse(_PARABOLOID),
            ["x", "y"],
            [0.0, 0.0],
            method="bfgs",
            max_iter=budget,
            full_result=True,
        ),
    )
    for r in checks:
        assert 0 < r.iterations <= budget, r
        assert r.converged is True
        assert r.status == "converged"


def test_iteration_count_matches_the_counts_measured_in_the_suite():
    # Źródło niezależne dla LICZNIKA: komentarze w test_numerics.py zapisują
    # zmierzone budżety — gd na (x−3)² potrzebuje 92 iteracji, a paraboloida gd
    # 90; newton/bfgs zbiegają przy max_iter=2. .iterations musi podać dokładnie
    # te liczby, inaczej „ile pracy to kosztowało" jest fikcją.
    assert numerics.minimize(
        parse("(x-3)^2"), "x", 0.0, full_result=True
    ).iterations == 92
    assert numerics.minimize_nd(
        parse(_PARABOLOID), ["x", "y"], [0.0, 0.0], full_result=True
    ).iterations == 90
    for method in ("newton", "bfgs"):
        assert numerics.minimize(
            parse("(x-3)^2"), "x", 0.0, method=method, max_iter=5, full_result=True
        ).iterations == 2


def test_root_find_iterations_span_newton_and_the_bisection_fallback():
    # x0=0 zeruje pochodną: Newton pada po PIERWSZEJ iteracji, wynik daje
    # bisekcja. Licznik musi objąć OBIE fazy — inaczej raport mówiłby o 1
    # iteracji za robotę, która wymagała kilkudziesięciu połowień.
    r = numerics.root_find(parse("x^2 - 2"), "x", 0.0, full_result=True)
    assert abs(r.value) == pytest.approx(math.sqrt(2))
    assert r.iterations > 1
    assert r.residual < 1e-10


# =========================================================================
# MODUŁ 4 — PORAŻKI OPCJONALNIE ZWRACALNE
# =========================================================================
# Ten blok jest ŚWIADOMYM przepisaniem antydryfu modułu 3
# (`test_failures_still_raise_and_no_solver_produces_a_failure_status_yet`),
# który pilnował, że ŻADEN solver nie produkuje statusu porażki. Dokładnie po to
# tam był: granica „porażki tylko rzucają" nie mogła zniknąć po cichu, tylko
# decyzją modułu, który ją przenosi. Nowy kontrakt: ścieżka DOMYŚLNA rzuca jak
# dawniej (klasa I komunikat), a `full_result=True` zwraca ten sam wynik jako dane.

#: Reproducery WSZYSTKICH ośmiu miejsc rzucania w czterech solwerach, 1D i nd.
#: Kolumny: (id, wywołanie przyjmujące **kw, klasa wyjątku ścieżki domyślnej,
#: status ścieżki full_result, KOMUNIKAT bajt w bajt zmierzony PRZED modułem 4).
#: Matematyka każdego reproducera jest już ustalona w test_errors.py/test_numerics.py
#: (−x² wklęsła → ucieczka; x⁴ z lr=1 → cykl okresu 2; x²+1 → brak pierwiastka
#: rzeczywistego; atan(x)+5 → eksplozja kroku Newtona; Rosenbrock gd → dławi się
#: budżetem), więc oczekiwany status wynika z MATEMATYKI, nie ze zgadywania.
_FAILURES = [
    (
        "root_find/1D/diverged",
        lambda **kw: numerics.root_find(parse("atan(x)+5"), "x", 2.0, **kw),
        DivergenceError,
        "diverged",
        "root_find DIVERGES from x0=2.0 for atan(x) + 5: the Newton step escaped "
        "to |x|>2e+12 and there is no sign change within reach of bisection — this "
        "is divergence, not proof of no root. Provide a starting point closer to "
        "the expected root.",
    ),
    (
        "root_find/1D/not_converged",
        lambda **kw: numerics.root_find(parse("x^2 + 1"), "x", 1.0, **kw),
        NonConvergenceError,
        "not_converged",
        "root_find does not converge for x**2 + 1 at x0=1.0 (no real root nearby?)",
    ),
    (
        "minimize/1D/gd/diverged",
        lambda **kw: numerics.minimize(parse("-x^2"), "x", 1.0, **kw),
        DivergenceError,
        "diverged",
        "minimize DIVERGES from x0=1.0 for -x**2: the gradient-descent iterate "
        "escaped to |x|>1e+12, and the function value goes to -inf — the function "
        "is most likely unbounded below (no minimum). Check the function or provide "
        "a starting point near the expected minimum.",
    ),
    (
        "minimize/1D/gd/stagnated",
        lambda **kw: numerics.minimize(parse("x^4"), "x", 2.0, lr=1.0, **kw),
        StagnationError,
        "stagnated",
        "minimize STALLED for x**4 at x0=2.0: over 8 consecutive steps the function "
        "value did not drop significantly (f≈16), and |g|=32 ≥ tol — this is "
        "STAGNATION, not divergence and not unboundedness: the run stalled (cycle / "
        "oscillation with too large a step). Reduce lr, reduce tol or provide a "
        "better starting point.",
    ),
    (
        "minimize/1D/gd/not_converged",
        lambda **kw: numerics.minimize(parse("x"), "x", 0.0, **kw),
        NonConvergenceError,
        "not_converged",
        "minimize does not converge for x at x0=0.0 "
        "(function unbounded below or bad step?)",
    ),
    (
        "minimize/1D/newton/diverged",  # miejsce w _descent_min
        lambda **kw: numerics.minimize(parse("-x^2"), "x", 1.0, method="newton", **kw),
        DivergenceError,
        "diverged",
        "minimize DIVERGES from x0=[1.0] with method 'newton': the iterate norm "
        "exceeded 1e+12 (‖x‖=2.54e+12), and the function value goes to -inf — the "
        "function is most likely unbounded below (no minimum). Check the function "
        "or provide a starting point near the expected minimum.",
    ),
    (
        "minimize/1D/bfgs/not_converged",  # miejsce w _descent_min
        lambda **kw: numerics.minimize(
            parse("x"), "x", 0.0, method="bfgs", max_iter=200, **kw
        ),
        NonConvergenceError,
        "not_converged",
        "minimize does not converge for x at x0=[0.0] with method 'bfgs' "
        "(function unbounded below or bad starting point?)",
    ),
    (
        "root_find_nd/diverged/norm",
        lambda **kw: numerics.root_find_nd(
            [parse("atan(x)"), parse("y")], ["x", "y"], [2.0, 0.0], **kw
        ),
        DivergenceError,
        "diverged",
        "root_find_nd DIVERGES from x0=[2.0, 0.0]: the iterate norm exceeded 2e+12 "
        "(‖x‖=8.59e+20) — Newton is escaping the solution. Provide a starting point "
        "closer to the expected root.",
    ),
    (
        "root_find_nd/diverged/nonfinite-start",  # osobne miejsce: strażnik na wejściu pętli
        lambda **kw: numerics.root_find_nd(
            [parse("x"), parse("y")], ["x", "y"], [float("inf"), 0.0], **kw
        ),
        DivergenceError,
        "diverged",
        "root_find_nd DIVERGES from x0=[inf, 0.0]: the Newton iterate escaped to "
        "infinity (inf/NaN). Provide a starting point closer to the expected root.",
    ),
    (
        "root_find_nd/not_converged",
        lambda **kw: numerics.root_find_nd(
            [parse("x^2 + y^2 + 1"), parse("x - y")], ["x", "y"], [1.0, 1.0], **kw
        ),
        NonConvergenceError,
        "not_converged",
        "root_find_nd does not converge in 100 iterations for the system "
        "['x**2 + y**2 + 1', 'x - y'] at x0=[1.0, 1.0] — the iterate oscillates "
        "without dropping below the tolerance. Try a starting point closer to the "
        "expected root (or check whether a real solution exists at all).",
    ),
    (
        "minimize_nd/gd/diverged",
        lambda **kw: numerics.minimize_nd(
            parse("-(x^2 + y^2)"), ["x", "y"], [1.0, 1.0], **kw
        ),
        DivergenceError,
        "diverged",
        "minimize_nd DIVERGES from x0=[1.0, 1.0]: the gradient-descent iterate norm "
        "exceeded 1.41e+12 (‖x‖=1.53e+12), and the function value goes to -inf — the "
        "function is most likely unbounded below (no minimum). Check the function or "
        "provide a starting point near the expected minimum.",
    ),
    (
        "minimize_nd/gd/stagnated",
        lambda **kw: numerics.minimize_nd(
            parse("x^4 + y^4"), ["x", "y"], [2.0, 2.0], lr=1.0, **kw
        ),
        StagnationError,
        "stagnated",
        "minimize_nd STALLED for x**4 + y**4 at x0=[2.0, 2.0]: over 8 consecutive "
        "steps the function value did not drop significantly (f≈32), and ‖g‖∞=32 ≥ "
        "tol — this is STAGNATION, not divergence and not unboundedness: the run "
        "stalled (cycle / oscillation with too large a step). Reduce lr, reduce tol "
        "or provide a better starting point.",
    ),
    (
        "minimize_nd/gd/not_converged",
        lambda **kw: numerics.minimize_nd(parse(_ROSEN), ["x", "y"], [-1.2, 1.0], **kw),
        NonConvergenceError,
        "not_converged",
        "minimize_nd does not converge for (1 - x)**2 + 100*(-x**2 + y)**2 at "
        "x0=[-1.2, 1.0] (function unbounded below or bad step?)",
    ),
    (
        "minimize_nd/newton/diverged",  # miejsce w _descent_min, ścieżka nd
        lambda **kw: numerics.minimize_nd(
            parse("-(x^2) - y^2"), ["x", "y"], [1.0, 1.0], method="newton", **kw
        ),
        DivergenceError,
        "diverged",
        "minimize_nd DIVERGES from x0=[1.0, 1.0] with method 'newton': the iterate "
        "norm exceeded 1.41e+12 (‖x‖=3.59e+12), and the function value goes to -inf "
        "— the function is most likely unbounded below (no minimum). Check the "
        "function or provide a starting point near the expected minimum.",
    ),
    (
        "minimize_nd/newton/not_converged",  # miejsce w _descent_min, ścieżka nd
        lambda **kw: numerics.minimize_nd(
            parse("x + y"), ["x", "y"], [0.0, 0.0], method="newton", max_iter=200, **kw
        ),
        NonConvergenceError,
        "not_converged",
        "minimize_nd does not converge for x + y at x0=[0.0, 0.0] with method "
        "'newton' (function unbounded below or bad starting point?)",
    ),
    # --- moduł 8: PIĄTY wynik, obecny w obu ścieżkach jak każdy inny -------
    # Nie „nowy rodzaj porażki obok", tylko ten sam kontrakt modułu 4 zastosowany
    # do wyniku, którego wtedy nie było: ścieżka domyślna rzuca klasę I komunikat,
    # full_result oddaje ten sam bieg jako dane. Reproducery są matematyczne: -x²
    # ma w 0 MAKSIMUM (f''=-2), x³ ma PRZEGIĘCIE (f''=0 — drugiego rzędu nie da
    # się o to zapytać, więc sonda pyta o definicję), x²-y² SIODŁO, a x*y siodło
    # DIAGONALNE (wzdłuż obu osi f jest dokładnie 0.0 — sonda osiowa widzi
    # płaskowyż, więc kierunek bierze się z krzywizny).
    (
        "minimize/1D/gd/not_a_minimum",
        lambda **kw: numerics.minimize(parse("-x^2"), "x", 0.0, **kw),
        NotAMinimumError,
        "not_a_minimum",
        "minimize reached a STATIONARY point at 0.0 for -x**2 (from x0=0.0), but it "
        "is NOT a minimum: f=0 there, while f=-1e-12 at 1e-06 right next to it — a "
        "maximum, a saddle or an inflection. A vanishing gradient holds at ALL of "
        "those, so it cannot tell them apart; this run checked. More iterations "
        "cannot help (the step at a stationary point is zero) — start BESIDE this "
        "point instead, or minimize -(-x**2) if what you wanted was the maximum.",
    ),
    (
        "minimize/1D/bfgs/not_a_minimum",  # miejsce w _descent_min
        lambda **kw: numerics.minimize(parse("x^3"), "x", 0.0, method="bfgs", **kw),
        NotAMinimumError,
        "not_a_minimum",
        "minimize reached a STATIONARY point at [0.0] for x**3 (from x0=[0.0]), but "
        "it is NOT a minimum: f=0 there, while f=-1e-12 at -0.0001 right next to it "
        "— a maximum, a saddle or an inflection. A vanishing gradient holds at ALL "
        "of those, so it cannot tell them apart; this run checked. More iterations "
        "cannot help (the step at a stationary point is zero) — start BESIDE this "
        "point instead, or minimize -(x**3) if what you wanted was the maximum.",
    ),
    (
        "minimize_nd/gd/not_a_minimum",
        lambda **kw: numerics.minimize_nd(parse("x^2-y^2"), ["x", "y"], [0.0, 0.0], **kw),
        NotAMinimumError,
        "not_a_minimum",
        "minimize_nd reached a STATIONARY point at [0.0, 0.0] for x**2 - y**2 (from "
        "x0=[0.0, 0.0]), but it is NOT a minimum: f=0 there, while f=-1e-12 at "
        "[0.0, 1e-06] right next to it — a maximum, a saddle or an inflection. A "
        "vanishing gradient holds at ALL of those, so it cannot tell them apart; "
        "this run checked. More iterations cannot help (the step at a stationary "
        "point is zero) — start BESIDE this point instead, or minimize "
        "-(x**2 - y**2) if what you wanted was the maximum.",
    ),
    (
        "minimize_nd/newton/not_a_minimum",  # siodło DIAGONALNE, kierunek z krzywizny
        lambda **kw: numerics.minimize_nd(
            parse("x*y"), ["x", "y"], [0.0, 0.0], method="newton", **kw
        ),
        NotAMinimumError,
        "not_a_minimum",
        "minimize_nd reached a STATIONARY point at [0.0, 0.0] for x*y (from "
        "x0=[0.0, 0.0]), but it is NOT a minimum: f=0 there, while f=-5e-13 at "
        "[-7.071067811865475e-07, 7.071067811865475e-07] right next to it — a "
        "maximum, a saddle or an inflection. A vanishing gradient holds at ALL of "
        "those, so it cannot tell them apart; this run checked. More iterations "
        "cannot help (the step at a stationary point is zero) — start BESIDE this "
        "point instead, or minimize -(x*y) if what you wanted was the maximum.",
    ),
]

_FAILURE_IDS = [case[0] for case in _FAILURES]

#: Para klasa wyjątku ↔ status. Jedno źródło dla testu, który sprawdza, że OBIE
#: ścieżki mówią o tym samym biegu to samo — bez zaglądania do kolumny „status".
_STATUS_OF_CLASS = {
    DivergenceError: "diverged",
    StagnationError: "stagnated",
    NonConvergenceError: "not_converged",
    NotAMinimumError: "not_a_minimum",  # moduł 8
}


@pytest.mark.parametrize("case", _FAILURES, ids=_FAILURE_IDS)
def test_default_path_still_raises_the_same_class_and_the_same_message(case):
    # (a) Sedno obietnicy modułu 4: NIC się nie zmienia dla dzisiejszego wołającego.
    # Komunikaty są zapinowane BAJT W BAJT — snapshoty zmierzone na kodzie PRZED
    # tym modułem. Ten test jest celowo kruchy: przeformatowanie komunikatu ma go
    # zapalić, bo agenty (i README) czytają dziś te teksty.
    _, run, exc_cls, _, message = case
    with pytest.raises(exc_cls) as exc:
        run()
    assert str(exc.value) == message


@pytest.mark.parametrize("case", _FAILURES, ids=_FAILURE_IDS)
def test_full_result_returns_the_failure_form_instead_of_raising(case):
    # (b) Ta sama porażka, ta sama matematyka — tylko zamiast wyjątku DANE.
    _, run, _, status, _ = case
    r = run(full_result=True)
    assert isinstance(r, SolveResult)
    assert r.converged is False
    assert r.status == status
    assert r.status in SOLVE_STATUSES  # słownik z modułu 3 domyka się na tym
    assert r.status != "converged"


@pytest.mark.parametrize("case", _FAILURES, ids=_FAILURE_IDS)
def test_both_paths_tell_the_same_story_about_the_same_run(case):
    # Antydryf pary (klasa wyjątku ↔ status): status NIE jest tu brany z tabeli,
    # tylko WYWNIOSKOWANY z klasy, którą rzuciła ścieżka domyślna. Gdyby kiedyś
    # któreś miejsce zwracało inny status, niż rzuca, ten test to złapie — a
    # właśnie na tej równoważności opiera się cała wartość modułu 4 dla agenta.
    _, run, _, _, _ = case
    with pytest.raises(Exception) as exc:  # klasę bierzemy z biegu, nie z tabeli
        run()
    assert _STATUS_OF_CLASS[type(exc.value)] == run(full_result=True).status


def test_failure_value_is_the_last_iterate_with_an_independently_recomputed_residual():
    # (c) UCZCIWOŚĆ porażki: value to punkt, w którym solver się poddał, a residuum
    # to ta sama wielkość co na sukcesie, zmierzona W TYM punkcie. Liczymy ją DRUGĄ
    # DROGĄ — evalf na ORYGINALNYM wyrażeniu (mpmath w IR), nie skompilowaną
    # funkcją solwera.
    # 1D root_find: x²+1 — Newton z x0=1 wpada w x=0 (pochodna zeruje się), więc
    # ostatni iterat to 0.0, a |f(0)| = 1 — daleko od zera, i to musi być widoczne.
    expr = parse("x^2 + 1")
    r = numerics.root_find(expr, "x", 1.0, full_result=True)
    assert r.value == 0.0
    assert r.residual == pytest.approx(abs(expr.evalf(x=r.value)), rel=1e-9)
    assert r.residual > 1e-10  # rezolutnie NIE zeruje funkcji

    # 1D minimize (stagnacja): cykl ±2 na x⁴ — ostatni iterat 2.0, |f'(2)| = 32
    expr = parse("x^4")
    r = numerics.minimize(expr, "x", 2.0, lr=1.0, full_result=True)
    assert r.value == 2.0
    assert r.residual == pytest.approx(abs(expr.diff("x").evalf(x=r.value)), rel=1e-9)

    # nd root_find_nd: układ bez rozwiązania rzeczywistego — ‖F(x)‖∞ w ostatnim
    # iteracie, przeliczone z obu wyrażeń niezależnie
    exprs = [parse("x^2 + y^2 + 1"), parse("x - y")]
    r = numerics.root_find_nd(exprs, ["x", "y"], [1.0, 1.0], full_result=True)
    independent = max(abs(e.evalf(x=r.value[0], y=r.value[1])) for e in exprs)
    assert r.residual == pytest.approx(independent, rel=1e-9)
    assert r.residual > 1e-10

    # nd minimize_nd (gd dławi się na dolinie Rosenbrocka): ‖∇f‖∞ w ostatnim
    # iteracie, z gradientu SYMBOLICZNEGO oryginalnego wyrażenia
    expr = parse(_ROSEN)
    r = numerics.minimize_nd(expr, ["x", "y"], [-1.2, 1.0], full_result=True)
    grad = [expr.diff(v) for v in ("x", "y")]
    independent = max(abs(g.evalf(x=r.value[0], y=r.value[1])) for g in grad)
    assert r.residual == pytest.approx(independent, rel=1e-9)
    assert r.residual > 1e-9  # powyżej tol — bieg NIE dowiózł minimum


def test_failure_value_type_matches_the_family_not_the_outcome():
    # typ .value zależy od solwera (1D → float, nd → list[float]), nie od tego,
    # czy bieg się udał: wołający po `full_result=True` obsługuje JEDEN kształt
    r1 = numerics.minimize(parse("-x^2"), "x", 1.0, full_result=True)
    assert type(r1.value) is float
    rnd = numerics.minimize_nd(parse("-(x^2 + y^2)"), ["x", "y"], [1.0, 1.0], full_result=True)
    assert type(rnd.value) is list
    assert all(type(v) is float for v in rnd.value)


def test_failure_iterations_are_the_honest_count_not_a_placeholder():
    # (c) LICZNIK na porażce: tyle iteracji, ile REALNIE zeszło — nigdy 0 „bo nie
    # wyszło" i nigdy więcej niż budżet. Liczby zmierzone: x²+1 pada po 2 krokach
    # Newtona (drugi trafia w zerową pochodną), a bieg bez zbieżności wyczerpuje
    # DOKŁADNIE max_iter.
    assert numerics.root_find(parse("x^2 + 1"), "x", 1.0, full_result=True).iterations == 2
    assert (
        numerics.minimize(parse("x"), "x", 0.0, full_result=True).iterations == 10000
    )  # domyślny max_iter gd
    for budget in (7, 25):
        assert (
            numerics.root_find_nd(
                [parse("x^2 + y^2 + 1"), parse("x - y")],
                ["x", "y"],
                [1.0, 1.0],
                max_iter=budget,
                full_result=True,
            ).iterations
            == budget
        )
        assert (
            numerics.minimize_nd(
                parse(_ROSEN), ["x", "y"], [-1.2, 1.0], max_iter=budget, full_result=True
            ).iterations
            == budget
        )
        assert (
            numerics.minimize(
                parse("x"), "x", 0.0, method="newton", max_iter=budget, full_result=True
            ).iterations
            == budget
        )


def test_unsampleable_failure_residual_is_infinity_by_the_documented_rule():
    # (c) REGUŁA dla residuum, którego NIE DA SIĘ zmierzyć w ostatnim iteracie:
    # math.inf. Nie NaN (każde porównanie z nim jest False, więc `residual > tol`
    # odpowiadałoby „punkt jest w porządku" dla punktu, który nie jest nigdzie
    # blisko rozwiązania) i nie odmowa całego wywołania (to wyrzuciłoby diagnozę,
    # po którą wołający tu przyszedł).
    # 1D: sqrt(x)+1 nie ma pierwiastka rzeczywistego, a krok Newtona wyprowadza
    # iterat do x=-3, gdzie funkcja jest ZESPOLONA — nie ma czego próbkować.
    expr = parse("sqrt(x)+1")
    r = numerics.root_find(expr, "x", 1.0, full_result=True)
    assert r.status == "not_converged" and r.converged is False
    assert r.value == -3.0
    assert math.isinf(r.residual)
    # źródło niezależne: DRUGA droga ewaluacji (evalf/mpmath) też odmawia w tym
    # punkcie — czyli „niemierzalne" to fakt o punkcie, nie kaprys solwera
    with pytest.raises(DomainError):
        expr.evalf(x=r.value)

    # nd: iterat, który stał się nieskończony — tam świadomie NIE próbkujemy
    # (ścieżka rzucająca też nie), więc reguła zwraca inf
    r = numerics.root_find_nd(
        [parse("x"), parse("y")], ["x", "y"], [float("inf"), 0.0], full_result=True
    )
    assert r.status == "diverged"
    assert math.isinf(r.residual)
    assert math.isinf(r.value[0])  # ostatni iterat oddany UCZCIWIE, z inf w środku

    # inf jest JEDNOZNACZNE: residuum ZMIERZONE jest zawsze skończone, bo warstwa
    # próbkująca odrzuca inf/NaN — więc isinf czyta się jako „niemierzalne”, nigdy
    # jako „zmierzone i wyszło ogromne”
    for _, run, _, _, _ in _FAILURES:
        res = run(full_result=True).residual
        assert res >= 0.0
        assert math.isinf(res) or math.isfinite(res)


def test_only_converged_may_be_branched_on_because_a_failed_run_can_look_close():
    # PO CO `converged` jest jedynym polem do rozgałęziania: atan(x)+5 jest
    # OGRANICZONA, więc uciekający iterat ma residuum ~3.4 — MAŁE i skończone,
    # nieodróżnialne po samej liczbie od biegu, który po prostu jeszcze nie dobił.
    # Kto rozgałęzia się po residuum, weźmie rozbiegnięcie za prawie-rozwiązanie.
    r = numerics.root_find(parse("atan(x)+5"), "x", 2.0, full_result=True)
    assert r.converged is False and r.status == "diverged"
    assert r.residual < 4.0  # „mała" liczba przy KATASTROFALNEJ porażce
    assert abs(r.value) > 1e12  # a ostatni iterat jest astronomiczny


#: Odmowy WEJŚCIA: zła lista zmiennych, układ niekwadratowy, osobliwy jakobian,
#: nieznana metoda, punkt poza dziedziną. To NIE wyniki iteracji — nie mają statusu
#: i nie mają „ostatniego iteratu" do oddania, więc rzucają w OBIE strony.
_DOMAIN_REFUSALS = [
    ("root_find/extra-symbol", lambda **kw: numerics.root_find(parse("x + a"), "x", 1.0, **kw)),
    (
        "minimize/unknown-method",
        lambda **kw: numerics.minimize(parse("x^2"), "x", 0.0, method="sgd", **kw),
    ),
    (
        "minimize_nd/unknown-method",
        lambda **kw: numerics.minimize_nd(
            parse("x^2 + y^2"), ["x", "y"], [1.0, 1.0], method="lbfgs", **kw
        ),
    ),
    (
        "root_find_nd/non-square",
        lambda **kw: numerics.root_find_nd([parse("x + y")], ["x", "y"], [0.0, 0.0], **kw),
    ),
    (
        "root_find_nd/extra-symbol",
        lambda **kw: numerics.root_find_nd(
            [parse("x + a"), parse("x - y")], ["x", "y"], [0.0, 0.0], **kw
        ),
    ),
    (
        "root_find_nd/singular-jacobian",
        lambda **kw: numerics.root_find_nd(
            [parse("x^2 - 1"), parse("y^2 - 1")], ["x", "y"], [0.0, 0.0], **kw
        ),
    ),
    (
        "minimize_nd/bad-x0-length",
        lambda **kw: numerics.minimize_nd(parse("x^2 + y^2"), ["x", "y"], [1.0], **kw),
    ),
    (
        "minimize_nd/gd/start-outside-domain",
        lambda **kw: numerics.minimize_nd(
            parse("log(x) + y^2"), ["x", "y"], [-1.0, 0.0], **kw
        ),
    ),
    (
        "minimize_nd/newton/start-outside-domain",
        lambda **kw: numerics.minimize_nd(
            parse("log(x) + y^2"), ["x", "y"], [-1.0, 0.0], method="newton", **kw
        ),
    ),
]


@pytest.mark.parametrize(
    "run", [c[1] for c in _DOMAIN_REFUSALS], ids=[c[0] for c in _DOMAIN_REFUSALS]
)
def test_domain_refusals_raise_with_and_without_full_result(run):
    # (d) Granica modułu 4: zwracalne są WYNIKI ITERACJI, nie odmowy wejścia.
    # DomainError nie ma statusu w SOLVE_STATUSES i nie da się go zwrócić bez
    # kłamstwa („oto ostatni iterat" dla biegu, który się nie zaczął).
    with pytest.raises(DomainError):
        run()
    with pytest.raises(DomainError):
        run(full_result=True)


# --- konstruktor formy porażki -------------------------------------------
def test_failure_constructor_pairs_converged_false_with_its_status():
    # bliźniak testu success(): para (converged, status) powstaje w JEDNYM miejscu
    r = SolveResult.failure(2.0, 11, 32.0, "stagnated")
    assert r.converged is False
    assert r.status == "stagnated"
    assert (r.value, r.iterations, r.residual) == (2.0, 11, 32.0)


def test_failure_constructor_refuses_a_non_failure_status():
    # forma porażki NIE może przyjąć "converged" (byłaby sprzeczna sama w sobie)
    # ani statusu spoza słownika — to walidacja RUNTIME, czyli mechaniczna
    # gwarancja, że każdy status wyprodukowany przez solver należy do SOLVE_STATUSES
    with pytest.raises(ValueError, match="failure status"):
        SolveResult.failure(1.0, 1, 1.0, "converged")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="failure status"):
        SolveResult.failure(1.0, 1, 1.0, "nearly")  # type: ignore[arg-type]
    # i nadal nie ma trzeciej drogi: klasa zna dwa konstruktory formy
    assert {"success", "failure"} <= set(vars(SolveResult))


# --- antydryf: statusy produkowane przez solwery --------------------------
def _statuses_produced_in_numerics() -> "tuple[set[str], set[str]]":
    """Statusy, które silnik potrafi ODDAĆ — czytane ze ŹRÓDŁA, nie z przebiegu.

    Zwraca (publiczne, wewnętrzne): pierwsze to argument ``status`` każdego
    wywołania ``SolveResult.failure(...)``, drugie to czwarty element każdej krotki
    zwracanej przez prywatny ``_descent_min``. AST, a nie grep — dzięki temu
    literał w komentarzu czy docstringu nie liczy się jako producent, a wyrażenie
    warunkowe (``"diverged" if diverged else "not_converged"``) liczy się jako dwa.
    """
    tree = ast.parse(pathlib.Path(numerics.__file__).read_text(encoding="utf-8"))
    public: set[str] = set()
    internal: set[str] = set()

    def literals(node: ast.AST) -> set[str]:
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
            and node.func.value.id == "SolveResult"
        ):
            assert len(node.args) == 4, "SolveResult.failure(value, iterations, residual, status)"
            public |= literals(node.args[3])
        if isinstance(node, ast.FunctionDef) and node.name == "_descent_min":
            for ret in ast.walk(node):
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple):
                    assert len(ret.value.elts) == 4, "_descent_min zwraca 4-krotkę"
                    internal |= literals(ret.value.elts[3])
    return public, internal


def test_every_status_a_solver_can_produce_is_in_the_declared_vocabulary():
    # Antydryf, który ZASTĄPIŁ granicę modułu 3. Tam pilnowaliśmy, że statusów
    # porażki NIKT nie produkuje; tutaj — że każdy, kto je produkuje, mówi
    # słownikiem z SOLVE_STATUSES i że wszystkie trzy MAJĄ producenta (inaczej
    # moduł 4 byłby zrobiony w połowie, a pole znów opisywałoby coś nieosiągalnego).
    public, internal = _statuses_produced_in_numerics()
    assert public <= set(SOLVE_STATUSES), public - set(SOLVE_STATUSES)
    assert internal <= set(SOLVE_STATUSES), internal - set(SOLVE_STATUSES)
    # ŚWIADOME PRZEPISANIE (moduł 8, przypadek (i)): dochodzi "not_a_minimum" po
    # OBU stronach. Granica trzyma dokładnie to samo co wcześniej — każdy status ma
    # producenta, żaden producent nie mówi spoza słownika — a rozszerzenie jest
    # celem modułu, nie dryfem: bez tego statusu maksimum i siodło wracały jako
    # converged=True (patrz test_module8_* niżej).
    assert public == {"diverged", "stagnated", "not_converged", "not_a_minimum"}
    # _descent_min (newton/bfgs) świadomie NIE produkuje "stagnated": USTALENIE R13
    # mówi, że nawrotowy Armijo strukturalnie wyklucza cykl „f stoi" (nie ma lr do
    # przesadzenia), więc stagnacja jest wpięta WYŁĄCZNIE w gd. "not_a_minimum"
    # produkuje JEDNAK — bo punkt stacjonarny nie jest wadą kroku, tylko własnością
    # punktu: newton i bfgs trafiają na maksimum dokładnie tak samo jak gd, gdy jest
    # nim START (test gradientu strzela w iteracji 1, zanim line search zdąży ruszyć).
    assert internal == {"converged", "diverged", "not_converged", "not_a_minimum"}

    src = inspect.getsource(numerics)
    assert "SolveResult(" not in src, (
        "obie formy wyniku MUSZĄ powstawać przez SolveResult.success()/.failure() — "
        "surowy konstruktor pozwoliłby rozjechać parę (converged, status)"
    )
