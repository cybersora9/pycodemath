"""Testy typowanej hierarchii wyjątków (moduł 2 serii v0.3).

Kontrakt: KAŻDA porażka to nadal ``PycodemathError`` — podklasa mówi tylko, JAKIEGO
RODZAJU była porażka, żeby agent rozgałęział się na KLASIE, a nie na dopasowaniu
angielskiego komunikatu.

Źródło niezależne: oczekiwana klasa wynika z MATEMATYKI reprodukera (czy iterat
faktycznie ucieka do nieskończoności, czy stoi w miejscu, czy po prostu nie ma
dokąd zbiec), a nie z tego, co pycodemath dziś zwraca. Każdy test nazywa powód.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from pycodemath import M, parse
from pycodemath.core import errors
from pycodemath.core.errors import (
    NUMERIC_ROUTES,
    ROUTES,
    BudgetExhaustedError,
    DivergenceError,
    DomainError,
    IsolationError,
    NoClosedFormError,
    NonConvergenceError,
    NotAMinimumError,
    ParseError,
    PycodemathError,
    StagnationError,
    TimeBudgetError,
    UnsupportedFormError,
)
from pycodemath.engine import linalg, numerics, symbolic

# PRZEPISANY W MODULE 9, PONOWNIE W MODULE 10 (przypadek (i): pinował granicę,
# którą ten moduł świadomie przesuwa). Lista rosła 6 → 7 (moduł 8), 7 → 8 (moduł 9)
# i rośnie 8 → 10 tutaj. Nie jest rozluźniona: `test_hierarchy_list_is_complete`
# niżej porównuje ją Z RÓWNOŚCIĄ do wszystkiego, co dziedziczy po PycodemathError w
# errors.py, więc dopisanie klasy do pakietu BEZ dopisania jej tutaj dalej wywraca
# zestaw — a każdy test strukturalny wyżej (rodzeństwo, łapanie bazowym except,
# eksport z pakietu) automatycznie obejmuje nową klasę.
_SUBCLASSES = (
    ParseError,
    DomainError,
    DivergenceError,
    StagnationError,
    NonConvergenceError,
    NotAMinimumError,  # moduł 8
    BudgetExhaustedError,
    TimeBudgetError,  # moduł 9
    NoClosedFormError,  # moduł 10
    UnsupportedFormError,  # moduł 10
    IsolationError,  # moduł B (backstop podprocesem)
)


# --- struktura hierarchii ------------------------------------------------
def test_every_subclass_is_a_pycodemath_error():
    # gwarancja wstecznej zgodności: istniejące `except PycodemathError` nadal łapie
    for cls in _SUBCLASSES:
        assert issubclass(cls, PycodemathError)
        assert cls is not PycodemathError


def test_every_subclass_is_caught_by_the_base_except():
    # nie sama relacja klas — REALNE złapanie instancji przez bazowy except
    for cls in _SUBCLASSES:
        try:
            raise cls("probe")
        except PycodemathError as exc:
            assert type(exc) is cls
        else:  # pragma: no cover - raise zawsze wychodzi wyżej
            pytest.fail(f"{cls.__name__} nie został złapany przez PycodemathError")


def test_subclasses_are_siblings_not_a_chain():
    # żadna podklasa nie dziedziczy po innej — inaczej `except DomainError` łapałby
    # np. rozbieganie i rozgałęzianie na klasie dawałoby fałszywe trafienia
    for cls in _SUBCLASSES:
        for other in _SUBCLASSES:
            if cls is not other:
                assert not issubclass(cls, other)


def test_public_api_exports_the_whole_hierarchy():
    import pycodemath

    for cls in (PycodemathError, *_SUBCLASSES):
        assert getattr(pycodemath, cls.__name__) is cls
        assert cls.__name__ in pycodemath.__all__
        assert cls.__name__ in errors.__all__


# --- rozbieganie vs stagnacja vs brak zbieżności -------------------------
def test_divergence_is_not_stagnation():
    # f = -x² jest WKLĘSŁA i nieograniczona z dołu: krok gd oddala się od 0 w każdej
    # iteracji, więc |x| rośnie geometrycznie i UCIEKA do nieskończoności.
    # To rozbieganie — nie utknięcie (iterat się rusza, i to coraz szybciej).
    with pytest.raises(DivergenceError) as exc:
        numerics.minimize(parse("-x^2"), "x", 1.0)
    assert not isinstance(exc.value, StagnationError)
    assert not isinstance(exc.value, NonConvergenceError)


def test_stagnation_is_not_divergence():
    # f = x⁴ JEST ograniczona z dołu (minimum w 0), więc nic nie może uciec do
    # nieskończoności. Przy lr=1 gd wpada w cykl okresu 2 (±2) i f stoi:
    # utknięcie, nie rozbieganie.
    with pytest.raises(StagnationError) as exc:
        numerics.minimize(parse("x^4"), "x", 2.0, lr=1.0)
    assert not isinstance(exc.value, DivergenceError)
    assert not isinstance(exc.value, NonConvergenceError)


def test_nonconvergence_is_neither_divergence_nor_stagnation():
    # x² + 1 nie ma pierwiastka rzeczywistego, ale Newton z x0=1 ani nie eksploduje
    # (iterat zostaje w okolicy zera), ani nie zamiera — po prostu nie ma dokąd
    # zbiec. To trzeci, osobny przypadek.
    with pytest.raises(NonConvergenceError) as exc:
        numerics.root_find(parse("x^2 + 1"), "x", 1.0)
    assert not isinstance(exc.value, DivergenceError)
    assert not isinstance(exc.value, StagnationError)


def test_divergent_series_is_a_divergence():
    # suma k dla k=1..oo rozbiega do +oo (kryterium konieczne zbieżności: wyraz
    # ogólny nie dąży do 0) — ta sama kategoria co uciekający iterat
    with pytest.raises(DivergenceError):
        symbolic.summation(parse("k"), "k", 1, "oo")


def test_oscillating_limit_is_a_nonconvergence():
    # sin(x) przy x -> oo nie ma granicy: wartości krążą po [-1, 1], nie uciekają
    # do nieskończoności i nie zamierają — brak zbieżności, nie rozbieganie
    with pytest.raises(NonConvergenceError) as exc:
        symbolic.limit(parse("sin(x)"), "x", "oo")
    assert not isinstance(exc.value, DivergenceError)


# --- odmowy wejścia ------------------------------------------------------
def test_parse_refusal_is_a_parse_error():
    # literał tekstowy i dostęp przez kropkę to nie notacja matematyczna —
    # odmowa strażnika tokenów zapada ZANIM cokolwiek policzymy
    with pytest.raises(ParseError):
        parse("'os'")
    with pytest.raises(ParseError):
        parse("x.__class__")


def test_shape_mismatch_is_a_domain_error():
    # wyznacznik jest zdefiniowany WYŁĄCZNIE dla macierzy kwadratowej —
    # 2×3 jest poza dziedziną operacji
    with pytest.raises(DomainError):
        linalg.det(M("[[1,2,3],[4,5,6]]"))


def test_value_outside_the_real_domain_is_a_domain_error():
    # log jest nieokreślony na ℝ dla x ≤ 0; wynik zespolony nie jest liczbą
    # rzeczywistą, którą evalf ma prawo zwrócić
    with pytest.raises(DomainError):
        parse("log(x)").evalf(x=-1.0)


def test_parse_error_and_domain_error_do_not_overlap():
    # odmowa parsera i odmowa dziedziny to dwie różne decyzje, na dwóch różnych
    # etapach — agent musi móc je rozróżnić
    with pytest.raises(ParseError) as parse_exc:
        parse("'os'")
    with pytest.raises(DomainError) as domain_exc:
        linalg.det(M("[[1,2,3],[4,5,6]]"))
    assert not isinstance(parse_exc.value, DomainError)
    assert not isinstance(domain_exc.value, ParseError)


# --- antydryf: gdzie wolno zostać generycznemu PycodemathError ------------
_TYPED_MODULES = (
    "engine/numerics.py",
    "engine/linalg.py",
    "frontend/parser.py",
    "codegen/emit.py",
    "cli/repl.py",
)


def test_fully_typed_modules_have_no_generic_raise_left():
    # w tych modułach KAŻDA porażka ma konkretną kategorię; generyczny raise
    # oznaczałby nieotagowane miejsce, którego agent nie rozróżni
    root = pathlib.Path(numerics.__file__).parent.parent
    for rel in _TYPED_MODULES:
        src = (root / rel).read_text(encoding="utf-8")
        assert "raise PycodemathError(" not in src, rel


def test_no_generic_raise_is_left_anywhere_in_the_package():
    # ŚWIADOME PRZEPISANIE (moduł 10, przypadek (i)). Dawna nazwa:
    # `test_generic_raises_live_only_where_the_engine_cannot_answer`. Pinowała, że
    # 10 generycznych `raise PycodemathError(` ZOSTAJE — bo „silnik symboliczny nie
    # potrafi podać odpowiedzi" uchodziło za jedną kategorię.
    #
    # POMIAR MODUŁU 10 POKAZAŁ, ŻE TO NIE JEST JEDNA KATEGORIA. Wszystkie 10 miejsc
    # leżało na powierzchni symbolicznej (ze 118 `raise` w pakiecie) i niosło CZTERY
    # różne wyniki matematyczne o CZTERECH różnych lekarstwach: „silnik szukał i nie
    # znalazł" (całka, suma, ODE), „silnik nie ma metody na ten KSZTAŁT" (solve z
    # mieszanymi generatorami, PoleError w limit/series), „granica nie istnieje"
    # (zoo/nan) i „wywołałeś nie tę funkcję" (kształt układu w solve). Agent łapiący
    # `PycodemathError` nie odróżniał „przestań, nie ma czego szukać" od „spytaj
    # numerycznie" — czyli musiał czytać angielski, przed czym broni moduł 2.
    #
    # Test NIE JEST rozluźniony, tylko ZAOSTRZONY dwa razy: sprawdza CAŁY pakiet
    # (zamiast trzech wskazanych plików), wymaga ZERA (zamiast liczby dozwolonych
    # wyjątków od reguły) i liczy przez AST, a nie przez `in` na tekście — bo
    # `errors.py` OPISUJE tę zmianę w docstringu i dopasowanie tekstowe uznało jego
    # własną dokumentację za łamanie reguły.
    root = pathlib.Path(numerics.__file__).parent.parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", None) == "PycodemathError"
            ):
                offenders.append(f"{path.relative_to(root).as_posix()}:{node.lineno}")
    assert offenders == []


# --- moduł 10: TRASA — co zrobić dalej, jako DANE ------------------------
def test_every_route_names_a_command_the_repl_really_dispatches():
    # MECHANICZNIE, nie obietnicą. Trasa jest tym elementem payloadu, który jest
    # najgroźniejszy, gdy się myli: agent pójdzie w nią bez pytania. Jedyne, co da
    # się sprawdzić bez wróżenia, to że nazwa JEST komendą, którą można wywołać —
    # więc to sprawdzamy, przy KAŻDEJ wartości słownika, i przy tych z NUMERIC_ROUTES.
    from pycodemath.cli import repl

    assert set(ROUTES) <= set(repl._COMMANDS)
    assert set(NUMERIC_ROUTES.values()) <= set(ROUTES)


def test_route_defaults_to_none_on_every_class_in_the_hierarchy():
    # `route` siedzi na KLASIE BAZOWEJ, więc każda porażka MOŻE ją nieść. Domyślnie
    # nie niesie żadnej — brak trasy to informacja („nie ma czego spróbować"), a nie
    # dziura, i nie wolno jej udawać wartością.
    for cls in (PycodemathError, *_SUBCLASSES):
        assert cls("probe").route is None


def test_no_closed_form_is_not_unsupported_form():
    # ŹRÓDŁO NIEZALEŻNE, nie „co dziś zwraca pycodemath":
    #  * exp(sin(x)) NIE MA elementarnej funkcji pierwotnej — sympy.integrate
    #    PRZECHODZI swój algorytm i zwraca nieobliczoną całkę. Silnik SZUKAŁ.
    #  * exp(x)+x^5-3 miesza x z exp(x); solver algebraiczny sympy odmawia WYJĄTKIEM
    #    („multiple generators"), zanim czegokolwiek poszuka. To brak METODY.
    # Lekarstwa są różne, więc klasy muszą być różne.
    with pytest.raises(NoClosedFormError) as no_form:
        symbolic.integrate(parse("exp(sin(x))"), "x")
    with pytest.raises(UnsupportedFormError) as no_method:
        symbolic.solve(parse("exp(x)+x^5-3"), "x")
    assert not isinstance(no_form.value, UnsupportedFormError)
    assert not isinstance(no_method.value, NoClosedFormError)
    assert isinstance(no_form.value, PycodemathError)  # stary `except` nadal łapie
    assert isinstance(no_method.value, PycodemathError)


def test_the_sympy_vocabulary_does_not_reach_the_message():
    # Punkt 4 zamówienia modułu 10. „multiple generators [x, exp(x)]" to wewnętrzne
    # słownictwo SymPy — a moduł 2 istnieje po to, żeby agent nie musiał czytać
    # komunikatu. Oryginał zostaje osiągalny przez __cause__ (dla człowieka przy
    # debugowaniu), ale NIE w tekście, który dostaje wołający.
    with pytest.raises(UnsupportedFormError) as exc:
        symbolic.solve(parse("exp(x)+x^5-3"), "x")
    msg = str(exc.value)
    assert "multiple generators" not in msg
    assert "No algorithms are implemented" not in msg
    assert "root" in msg  # mówi, co zrobić dalej
    assert exc.value.route == "root"
    # ...a dowód dla człowieka NIE zniknął
    assert "multiple generators" in str(exc.value.__cause__)


def test_the_route_distinguishes_refusals_that_share_a_class():
    # SEDNO: dwa razy ta sama klasa, dwa RÓŻNE lekarstwa. Całka bez formy zamkniętej
    # ma numeryczny odpowiednik pytania (nintegrate); nieskończona suma go NIE MA w
    # tej gramatyce. Bez trasy agent nie odróżnia „spróbuj inaczej" od „nie ma czego
    # spróbować" — a to jedyna różnica, która zmienia jego następny ruch.
    with pytest.raises(NoClosedFormError) as integral:
        symbolic.integrate(parse("exp(sin(x))"), "x")
    with pytest.raises(NoClosedFormError) as infinite_sum:
        symbolic.summation(parse("1/(k^2+k+1)"), "k", 1, "oo")
    assert integral.value.route == "nintegrate"
    assert infinite_sum.value.route is None


def test_budget_exhausted_error_is_raised_only_by_the_quadrature_budget():
    # ŚWIADOME PRZEPISANIE granicy z modułu 2 (dawne
    # `test_budget_exhausted_error_has_no_raise_site_yet`, które pilnowało, że NIC tej
    # klasy nie rzuca). Ta granica po to tam była: „budżet" nie mógł pojawić się po
    # cichu, tylko decyzją modułu, który go wprowadza. Moduł 5 go wprowadza —
    # `integrate_num(..., tol=...)` dostaje `max_evals`, czyli JAWNY budżet ewaluacji
    # ustawiony przez wołającego.
    #
    # DLACZEGO tu, a nie NonConvergenceError (tak jak cap max_iter w solwerach):
    # tam cap jest DIAGNOZĄ — bieg nie ma żadnej wartości do pokazania, a podejrzany
    # jest sam problem. Tu z integrandem nie ma nic złego: wartość i trwające
    # rafinowanie ISTNIEJĄ, oszacowanie wciąż spadało, a skończyło się WYŁĄCZNIE
    # pozwolenie wołającego — ten sam bieg z większym max_evals dochodzi do tej samej
    # tolerancji (zapinowane w tests/test_quadrature.py).
    root = pathlib.Path(numerics.__file__).parent.parent
    hits = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8").count(
            "raise BudgetExhaustedError("
        )
        for path in root.rglob("*.py")
        if "raise BudgetExhaustedError(" in path.read_text(encoding="utf-8")
    }
    assert hits == {"engine/numerics.py": 1}


# --- moduł 8: „DOSZEDŁ, ale nie tam" jako osobny rodzaj porażki -----------
def test_hierarchy_list_is_complete():
    # Antydryf listy: _SUBCLASSES to źródło dla wszystkich testów wyżej, więc
    # klasa dopisana do errors.py, a pominięta tutaj, byłaby NIEPRZETESTOWANA —
    # dokładnie tak moduł 8 mógłby dołożyć siódmą klasę po cichu. (Moduł 10:
    # ten mechanizm zadziałał — dwie nowe klasy wywróciły go, zanim zdążyły ujść.)
    declared = {
        c
        for c in vars(errors).values()
        if isinstance(c, type)
        and issubclass(c, PycodemathError)
        and c is not PycodemathError
    }
    assert declared == set(_SUBCLASSES)


def test_arriving_at_a_maximum_is_not_divergence_stagnation_or_nonconvergence():
    # Źródło niezależne: -x² ma w 0 f'=0 i f''=-2, czyli jest tam MAKSIMUM. To nie
    # jest żadna z trzech dotychczasowych porażek i dlatego dostało własną klasę:
    #  * nic nie uciekło (iterat stoi w 0),
    #  * nic nie utknęło „będąc od celu" — test gradientu jest SPEŁNIONY,
    #  * budżet się nie skończył (bieg trwał jedną iterację).
    # Bieg DOSZEDŁ; złe jest miejsce, do którego doszedł.
    with pytest.raises(NotAMinimumError) as exc:
        numerics.minimize(parse("-x^2"), "x", 0.0)
    assert not isinstance(exc.value, DivergenceError)
    assert not isinstance(exc.value, StagnationError)
    assert not isinstance(exc.value, NonConvergenceError)
    assert isinstance(exc.value, PycodemathError)  # stary `except` nadal łapie


def test_not_a_minimum_message_carries_the_evidence_and_the_remedy():
    # Zasada domu: komunikat nazywa WARTOŚĆ, która obala, i mówi, co zrobić dalej.
    # Tu „co dalej" jest nietypowe i dlatego musi być wprost: więcej iteracji NIE
    # POMOŻE NIGDY (krok w punkcie stacjonarnym jest zerowy) — trzeba zmienić start.
    with pytest.raises(NotAMinimumError) as exc:
        numerics.minimize(parse("-x^2"), "x", 0.0)
    msg = str(exc.value)
    assert "STATIONARY" in msg
    assert "1e-06" in msg          # KONKRETNY punkt obok, który jest niżej
    assert "More iterations cannot help" in msg
    assert "start BESIDE this point" in msg
