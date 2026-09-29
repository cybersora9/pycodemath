"""Moduł 9: wywołanie symboliczne NIE MOŻE wisieć w nieskończoność.

POMIAR, KTÓRY TEN PLIK ISTNIEJE, ŻEBY POWTÓRZYĆ. Przed modułem 9:

    integrate 1/(x^5+x+1) dx        NIE skończyło w 500 s
    integrate log(x)/(x^5+x+1) dx   nie skończyło
    series exp(sin(exp(x))) n 14    nie skończyło

Goła SymPy też ich nie kończy — koszt jest ODZIEDZICZONY, nie spowodowany tutaj —
i właśnie dlatego to problem pycodemath do OBSŁUŻENIA, a nie SymPy do naprawienia.
Moduł 2 otypował każdą porażkę, która WRACA. Ta nie wracała: przez MCP to nie jest
zła odpowiedź, tylko wywołanie narzędzia, które wisi na zawsze, bez wyniku, bez
wyjątku i bez payloadu, po którym agent mógłby się rozgałęzić.

--- JAK TEN PLIK UNIKA MIGOTANIA NA OBCIĄŻONEJ MASZYNIE ------------------------

Test zależny od zegara ściennego jest domyślnie kruchy, więc obowiązują tu trzy
reguły i żaden asert ich nie łamie:

1. **Nigdy nie asertujemy GÓRNEJ granicy czasu blisko budżetu.** Obciążony
   runner CI dokłada opóźnienie w jedną stronę — czas może tylko UROSNĄĆ. Górna
   granica pojawia się wyłącznie jako WIELOKROTNOŚĆ budżetu (30 s zapasu przy
   budżecie 1 s), i nie po to, żeby zmierzyć zwłokę, tylko żeby udowodnić, że
   strażnik PRZERWAŁ bieg, zamiast doczekać końca SymPy (której koniec nie
   nadchodzi — więc bez przerwania test nie skończyłby się nigdy).
2. **Dolne granice są bezpieczne i to ich używamy** (`spent >= budżet`): pod
   obciążeniem robią się tylko mocniejsze.
3. **Reproducer nigdy się nie kończy.** Żadne obciążenie nie zamieni tego testu w
   fałszywą porażkę, bo nie ma świata, w którym `sp.integrate` tu wraca.

--- DLACZEGO KOMUNIKAT NIE ZAWIERA ZUŻYTEGO CZASU -------------------------------

Bo wtedy nie dałoby się go zapinować. Moduł 8 porównywał snapshot 150 obserwacji
`(klasa, komunikat)` CO DO BAJTU; liczba z zegara zabrałaby tę własność każdemu
komunikatowi, który by ją niósł. Komunikat cytuje BUDŻET (liczbę, którą wybrał
WOŁAJĄCY), a zużyty czas jedzie na wyjątku jako `.spent` — dowód jest, a test
nigdy go nie czyta jako tekstu.
"""

from __future__ import annotations

import math
import threading
import time

import pytest

from pycodemath import DEFAULT_TIME_BUDGET, E, TimeBudgetError, series, time_budget
from pycodemath.core import budget as budget_mod
from pycodemath.core.errors import DomainError, PycodemathError
from pycodemath.cli import mcp_server, repl
from pycodemath.engine import linalg, symbolic
from pycodemath.frontend.parser import parse

#: Budżet używany w testach, które muszą naprawdę odpalić strażnika. Krótki, bo
#: każdy taki test PŁACI go zegarem; reproducer i tak nie skończy w żadnym.
_SHORT = 1.0

#: Wywołania, które NIE KOŃCZĄ SIĘ NIGDY (zmierzone: reproducer > 500 s). Każde
#: idzie przez inne wejście silnika, bo strażnik nie może być łatką na `integrate`.
_HANGS = (
    ("integrate 1/(x^5+x+1) dx", "integrate"),
    ("integrate log(x)/(x^5+x+1) dx", "integrate"),
    ("series exp(sin(exp(x))) for x n 14", "series"),
)

#: Ścieżka szybka: przykłady z README i ich sąsiedzi. Zmierzone zimno, każdy
#: poniżej 0.51 s. Budżet, który tknie KTÓREKOLWIEK z nich, jest zepsuty.
_FAST = (
    "sin(x)^2 + cos(x)^2",
    "diff sin(x)*x dx",
    "integrate 2*x dx",
    "integrate x*exp(x) dx",
    "solve x^2-4 for x",
    "limit sin(x)/x for x to 0",
    "series exp(x) for x n 4",
    "sum k for k from 1 to n",
    "det [[1,2],[3,4]]",
    "inv [[4,7],[2,6]]",
    "eig [[2,0],[0,3]]",
    "solve_system [[2,1],[1,3]] = [3,5]",
    "code (sin(x)+cos(x))^2",
)


# --- (1) strażnik ODPALA na prawdziwym zawieszeniu -------------------------
@pytest.mark.parametrize("cmd,operation", _HANGS, ids=[c for c, _ in _HANGS])
def test_a_call_that_never_returns_is_refused_not_awaited(cmd: str, operation: str) -> None:
    # Sedno modułu. Bez strażnika ten test nie skończyłby się NIGDY — samo jego
    # zakończenie jest dowodem, że przerwanie działa na prawdziwym zawieszeniu, a
    # nie tylko na sztucznej pętli.
    started = time.monotonic()
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(_SHORT):
            repl.handle(cmd)
    elapsed = time.monotonic() - started

    assert exc.value.operation == operation
    assert exc.value.budget == _SHORT
    assert exc.value.spent >= _SHORT * 0.9  # granica DOLNA — patrz docstring
    # Górna granica wyłącznie jako dowód PRZERWANIA, z 30x zapasem (reguła 1).
    assert elapsed < _SHORT + 30.0


def test_the_refusal_is_a_pycodemath_error_like_every_other_failure() -> None:
    # Wsteczna zgodność modułu 2: `except PycodemathError` zostaje pełnym łapaniem.
    with pytest.raises(PycodemathError):
        with time_budget(_SHORT):
            repl.handle(_HANGS[0][0])


def test_the_injected_interrupt_never_reaches_the_caller() -> None:
    # _Deadline jest BaseException (musi być — patrz test niżej), więc gdyby
    # wyciekł, minąłby KAŻDY `except Exception` w kodzie wołającego i ubił proces.
    # Tu pilnujemy, że z pakietu wychodzi wyłącznie klasa publiczna.
    with time_budget(_SHORT):
        try:
            repl.handle(_HANGS[0][0])
        except TimeBudgetError:
            pass
        except BaseException as exc:  # noqa: BLE001
            pytest.fail(f"z pakietu wyszło {type(exc).__name__}, nie TimeBudgetError")


def test_the_interrupt_is_a_baseexception_so_broad_handlers_cannot_eat_it() -> None:
    # To nie jest szczegół implementacyjny, tylko powód, dla którego strażnik w
    # ogóle działa: SymPy jest pełna `except Exception`, a linalg._FALLBACK_ERRORS
    # łapie ValueError/TypeError/NotImplementedError, żeby zejść na NumPy. Gdyby
    # przerwanie było Exception, przeterminowany bieg symboliczny zamieniłby się w
    # cichy bieg numeryczny zamiast w odmowę.
    assert issubclass(budget_mod._Deadline, BaseException)
    assert not issubclass(budget_mod._Deadline, Exception)


def test_the_guard_fires_off_the_main_thread() -> None:
    # Serwer MCP nie jest właścicielem wątku głównego — to jedna z dwóch rzeczy,
    # które wykluczyły SIGALRM (drugą jest brak SIGALRM na Windows).
    box: dict[str, object] = {}

    def body() -> None:
        try:
            with time_budget(_SHORT):
                repl.handle(_HANGS[0][0])
            box["out"] = "FINISHED"
        except TimeBudgetError as exc:
            box["out"] = type(exc).__name__

    worker = threading.Thread(target=body)
    worker.start()
    worker.join(timeout=_SHORT + 30.0)
    assert not worker.is_alive(), "wątek roboczy nie został przerwany"
    assert box["out"] == "TimeBudgetError"


# --- (2) ścieżka szybka SIĘ NIE RUSZYŁA ------------------------------------
@pytest.mark.parametrize("cmd", _FAST)
def test_the_fast_path_answers_under_the_default_budget(cmd: str) -> None:
    # Domyślny budżet (120 s) jest 235x powyżej najwolniejszego z tych wywołań
    # (0.51 s zimno). Test nie mierzy czasu — sprawdza, że ODPOWIEDŹ jest.
    out = repl.handle(cmd)
    assert out and "gave up" not in out


def test_a_correct_answer_is_never_replaced_by_a_budget_refusal() -> None:
    # Granica, o którą chodzi: budżet HOJNY wobec pracy, która się kończy, musi
    # oddać ODPOWIEDŹ, nie odmowę — nawet gdy strażnik był uzbrojony.
    with time_budget(60.0):
        assert repl.handle("integrate 2*x dx") == "x**2"
        assert repl.handle("solve x^2-4 for x") == "-2, 2"


def test_no_closed_form_stays_no_closed_form_and_does_not_become_a_budget() -> None:
    # NAJWAŻNIEJSZE ROZRÓŻNIENIE MODUŁU (dyscyplina modułu 8 przeniesiona na czas):
    # „nie ma odpowiedzi" i „nie dałem jej dość czasu" to NIE ten sam wynik.
    # `integrate exp(sin(x)) dx` naprawdę nie ma formy zamkniętej i mówi to w
    # 0.55 s — z hojnym budżetem musi dalej mówić TO, a nie o naszej cierpliwości.
    with time_budget(60.0):
        with pytest.raises(PycodemathError) as exc:
            repl.handle("integrate exp(sin(x)) dx")
    assert not isinstance(exc.value, TimeBudgetError)
    assert "no closed form" in str(exc.value)


# --- (3) komunikat jest DETERMINISTYCZNY ------------------------------------
def test_the_message_quotes_the_budget_and_never_the_time_spent() -> None:
    # Zapinowane CO DO BAJTU — możliwe tylko dlatego, że w tekście nie ma ani
    # jednej liczby pochodzącej z zegara (patrz docstring modułu).
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(1.5):
            symbolic.integrate(parse("1/(x^5+x+1)"), "x")
    assert str(exc.value) == (
        "integrate: gave up on 1/(x**5 + x + 1) after the 1.5s time budget — "
        "SymPy was still working and there is no partial result. This is NOT "
        "'no closed form': an answer may well exist. Raise the budget "
        "(pycodemath.time_budget) or compute numerically."
    )
    # ...a dowód kosztu istnieje, tylko nie w tekście
    assert exc.value.spent >= 1.5 * 0.9
    # Windows monotonic ticks every 15.625 ms and 1.5 s is exactly 96 ticks, so
    # `spent` can come out as 1.5 — the budget, which the message DOES quote.
    # The byte-exact pin above already proves no clock number is in the text.
    if exc.value.spent != 1.5:
        assert f"{exc.value.spent}" not in str(exc.value)


def test_the_message_names_the_offending_expression_but_stays_bounded() -> None:
    # Zasada domu: komunikat nazywa WARTOŚĆ, na której poległ. Ale całkowany
    # potwór nie może zamienić komunikatu w stronę tekstu.
    long_expr = "1/(x^5+x+1) + " + " + ".join(f"sin({k}*x)" for k in range(1, 30))
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(_SHORT):
            symbolic.integrate(parse(long_expr), "x")
    quoted = str(exc.value).split(" after the ")[0]
    assert quoted.startswith("integrate: gave up on ")
    assert len(quoted) <= len("integrate: gave up on ") + budget_mod._EXPR_IN_MESSAGE
    assert quoted.endswith("…")


# --- (4) wołający MOŻE powiedzieć z góry, ile czeka -------------------------
def test_an_explicit_budget_replaces_the_default() -> None:
    assert DEFAULT_TIME_BUDGET == 120.0
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(_SHORT):
            repl.handle(_HANGS[0][0])
    assert exc.value.budget == _SHORT != DEFAULT_TIME_BUDGET


def test_the_default_applies_when_the_caller_says_nothing(monkeypatch) -> None:
    # Domyślnego 120 s nie da się odczekać w zestawie testów, więc pinujemy
    # MECHANIZM (bez budżetu jawnego bierze się DEFAULT_TIME_BUDGET), podmieniając
    # samą stałą. Bez tego testu domyślna ścieżka — jedyna, którą idzie agent —
    # byłaby nieprzetestowana.
    monkeypatch.setattr(budget_mod, "DEFAULT_TIME_BUDGET", _SHORT)
    with pytest.raises(TimeBudgetError) as exc:
        repl.handle(_HANGS[0][0])
    assert exc.value.budget == _SHORT


def test_infinity_really_means_unlimited_and_not_quietly_the_default(monkeypatch) -> None:
    # REGRESJA ZE ZMIERZONEGO BŁĘDU, nie test hipotetyczny. Pierwsza wersja przy
    # `inf` nie uzbrajała NICZEGO — i wtedy wewnętrzne wejście silnika nie
    # znajdowało żadnego budżetu w wątku, więc uzbrajało DOMYŚLNY. Zmierzone:
    # `with time_budget(inf): series exp(sin(exp(x))) n 10` zostało odrzucone
    # dokładnie po 120 s. „Bez budżetu", które po cichu znaczy „budżet domyślny",
    # jest gorsze niż brak wyjścia awaryjnego.
    #
    # Pinujemy to PRACĄ SZTUCZNĄ, nie zawieszeniem SymPy: pętla, która trwa
    # zauważalnie dłużej niż (podmieniony) domyślny budżet i sama się kończy — więc
    # test nigdy nie zostawia po sobie wątku mielącego całkę w tle.
    monkeypatch.setattr(budget_mod, "DEFAULT_TIME_BUDGET", 0.2)

    def slow_but_finite() -> str:
        stop = time.monotonic() + 1.0
        while time.monotonic() < stop:
            pass
        return "done"

    with time_budget(math.inf):
        assert budget_mod._ARMED[threading.get_ident()][0] == math.inf
        assert budget_mod.guard("probe", "x", slow_but_finite) == "done"
        # ...i realne wejście silnika w środku też nie dostaje 0.2 s
        assert repl.handle("integrate 2*x dx") == "x**2"
    assert threading.get_ident() not in budget_mod._ARMED


def test_infinity_leaves_no_armed_state_behind() -> None:
    before = dict(budget_mod._ARMED)
    with time_budget(math.inf):
        assert repl.handle("integrate 2*x dx") == "x**2"
    assert dict(budget_mod._ARMED) == before


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), "5", None, True])
def test_a_budget_that_is_not_a_positive_number_is_refused_at_the_input(bad) -> None:
    # Budżet, który wygasł zanim cokolwiek ruszyło, to pomyłka WOŁAJĄCEGO, nie
    # wynik matematyczny — czyli DomainError, tak jak każda inna odmowa wejścia.
    with pytest.raises(DomainError):
        time_budget(bad)


# --- (5) zagnieżdżanie: ciaśniejszy wygrywa, w obie strony ------------------
def test_a_nested_budget_cannot_extend_the_one_its_caller_chose() -> None:
    started = time.monotonic()
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(_SHORT):
            with time_budget(600.0):  # próba przedłużenia — musi zostać zignorowana
                repl.handle(_HANGS[0][0])
    assert exc.value.budget == _SHORT
    assert time.monotonic() - started < _SHORT + 30.0


def test_a_nested_budget_that_wants_less_time_gets_less() -> None:
    started = time.monotonic()
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(600.0):
            with time_budget(_SHORT):
                repl.handle(_HANGS[0][0])
    assert exc.value.budget == _SHORT
    assert time.monotonic() - started < _SHORT + 30.0


def test_nesting_leaves_no_armed_state_behind() -> None:
    tid = threading.get_ident()
    with time_budget(30.0):
        with time_budget(20.0):
            assert tid in budget_mod._ARMED
            assert budget_mod._ARMED[tid][1] == 2  # głębokość
        assert budget_mod._ARMED[tid][1] == 1
    assert tid not in budget_mod._ARMED


# --- (6) strażnik nie psuje niczego, czego nie przerwał --------------------
def test_nothing_escapes_when_the_work_finishes_at_the_deadline() -> None:
    # REGRESJA ZE ZMIERZONEGO BŁĘDU, i to najgroźniejszego w całym module.
    # Iniekcja asynchroniczna ląduje na DOWOLNEJ granicy bajtkodu — także wewnątrz
    # `__exit__`. Pierwsza wersja pozwalała na to: przerwane sprzątanie nie usuwało
    # wpisu z tabeli, więc pies stróżujący wstrzykiwał w wątek, który dawno wyszedł
    # ze strażnika, co 50 ms, w nieskończoność. Jedno nieudane sprzątanie zatruwało
    # KAŻDE następne wywołanie w tym wątku. Zmierzone na 40 000 kolidujących
    # przebiegów: 103 ucieczki, KASKADĄ (podpis wycieku, nie pojedynczego wyścigu).
    # Po naprawie (`__exit__` ponawia sprzątanie i sam tłumaczy `_Deadline`):
    # 0 ucieczek, tabela pusta. `_Deadline` jest BaseException, więc ucieczka
    # przeszłaby przez każdy `except Exception` w kodzie wołającego.
    escaped: list[str] = []
    for i in range(2000):
        try:
            with time_budget(0.00002 + (i % 40) * 0.0000025):
                symbolic.diff(parse("sin(x)*x"), "x")
        except TimeBudgetError:
            pass  # uczciwa odmowa: praca naprawdę była wolniejsza niż budżet
        except BaseException as exc:  # noqa: BLE001
            escaped.append(type(exc).__name__)
    assert escaped == []
    # ...i nic nie zostało uzbrojone, bo to WYCIEK, nie ucieczka, robi kaskadę
    assert threading.get_ident() not in budget_mod._ARMED
    # a teraz: czy coś zostało „w locie" i uderzy w kod PO strażniku?
    for _ in range(20000):
        pass
    assert repl.handle("integrate 2*x dx") == "x**2"


def test_a_caller_running_their_own_code_in_the_block_gets_the_public_error() -> None:
    # DRUGA POŁOWA TEJ SAMEJ DZIURY. `time_budget` jest PUBLICZNY, więc w środku
    # może być dowolny kod wołającego — bez żadnego `guard` na stosie. Zanim
    # `__exit__` zaczął sam tłumaczyć, taki wołający dostawał gołego `_Deadline`.
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(0.3):
            stop = time.monotonic() + 30.0
            while time.monotonic() < stop:  # zwykła pętla, zero SymPy
                pass
    assert exc.value.budget == 0.3
    assert threading.get_ident() not in budget_mod._ARMED


def test_sympy_still_gives_right_answers_after_being_interrupted() -> None:
    # Przerwanie asynchroniczne trafia w losowe miejsce w SymPy. Gdyby zostawiało
    # zatruty cache globalny, moduł „naprawiłby" zawieszenie, psując wszystko inne.
    for _ in range(3):
        with pytest.raises(TimeBudgetError):
            with time_budget(0.3):
                repl.handle(_HANGS[0][0])
    assert repl.handle("integrate 2*x dx") == "x**2"
    assert repl.handle("diff sin(x)*x dx") == "x*cos(x) + sin(x)"
    assert repl.handle("solve x^2-4 for x") == "-2, 2"
    assert repl.handle("limit sin(x)/x for x to 0") == "1"
    assert repl.handle("sin(x)^2 + cos(x)^2") == "1"


# --- (7) KOMPLETNOŚĆ: żadne wejście symboliczne nie jest niepilnowane -------
def _public_functions(module) -> dict:
    return {
        name: obj
        for name, obj in vars(module).items()
        if callable(obj)
        and not name.startswith("_")
        and getattr(obj, "__module__", "") == module.__name__
    }


@pytest.mark.parametrize("module", [symbolic, linalg], ids=["symbolic", "linalg"])
def test_every_public_entry_point_of_the_engine_is_under_the_budget(module) -> None:
    # Antydryf MECHANICZNY, nie obietnica: funkcja dopisana do silnika bez
    # dekoratora wywraca ten test. Dokładnie tak samo moduł 8 pilnował
    # kompletności listy klas wyjątków, zamiast ufać, że ktoś pamiętał.
    unguarded = [
        name
        for name, fn in _public_functions(module).items()
        if not hasattr(fn, "__pcm_time_budget__")
    ]
    assert unguarded == []


def test_the_ir_methods_that_reach_unbounded_sympy_are_guarded_too() -> None:
    # E("1/(x^5+x+1)").integrate("x") NIE przechodzi przez engine.symbolic, a
    # codegen.pipeline.generate woła Expr.simplify, nie funkcję silnika. Bez tego
    # dziura byłaby dokładnie w Pythonowym API, które moduł 7 wystawił na zewnątrz.
    from pycodemath.core.ir import Expr, Matrix

    for owner, name in (
        (Expr, "simplify"), (Expr, "expand"), (Expr, "integrate"),
        (Expr, "equivalent"), (Matrix, "equivalent"),
    ):
        assert hasattr(getattr(owner, name), "__pcm_time_budget__"), f"{owner.__name__}.{name}"
    # ...a ścieżka numeryczna świadomie NIE, bo strażnik siedziałby w pętli, z
    # której reguła jednego IR wypycha SymPy.
    for name in ("diff", "subs", "evalf", "compiled"):
        assert not hasattr(getattr(Expr, name), "__pcm_time_budget__"), name


def test_the_python_api_is_bounded_without_going_through_the_engine() -> None:
    from pycodemath.core.ir import E

    with pytest.raises(TimeBudgetError):
        with time_budget(_SHORT):
            E("1/(x^5+x+1)").integrate("x")


# --- (8) strażnik nie kosztuje nic, dopóki nie jest potrzebny ---------------
def test_the_watchdog_is_started_lazily_and_there_is_only_ever_one() -> None:
    # Import biblioteki nie ma prawa zakładać wątku; proces, który woła tylko
    # `diff`, nie powinien nieść nic, o co nie prosił. Ale po pierwszym uzbrojeniu
    # ma być DOKŁADNIE JEDEN wątek na proces — inaczej „dzielony" strażnik byłby
    # tylko inaczej nazwanym wątkiem na wywołanie (zmierzone: 217 us startu).
    for _ in range(50):
        with time_budget(30.0):
            pass
    named = [t for t in threading.enumerate() if t.name == "pycodemath-time-budget"]
    assert len(named) == 1
    assert named[0].daemon  # nie może trzymać procesu przy życiu


def test_an_idle_watchdog_holds_no_armed_state() -> None:
    assert budget_mod._ARMED == {}


# --- (9) powierzchnie: REPL i drut MCP --------------------------------------
def test_the_repl_reports_the_budget_refusal_as_an_error_line() -> None:
    # Jednorazowe wołanie (`python -m pycodemath "..."`) drukuje `error: ...` na
    # stderr; interaktywna sesja to samo, bez zabijania sesji.
    with time_budget(_SHORT):
        with pytest.raises(TimeBudgetError):
            repl.handle(_HANGS[0][0])


def test_the_mcp_payload_says_the_budget_was_what_stopped_the_run() -> None:
    # Punkt 5 zamówienia: agent musi to zobaczyć Z SAMEGO PAYLOADU. Widzi —
    # `error.type` niesie NAZWĘ KLASY, czyli tę część hierarchii modułu 2, która
    # jest czytelna dla maszyny. I jedzie drogą odmowy WEJŚCIA (`solve` i
    # `quadrature` = null), bo bieg symboliczny nie ma wyniku strukturalnego —
    # dokładnie tak, jak moduł 6 zapowiedział dla silników, które go nie mają.
    with time_budget(_SHORT):
        payload = mcp_server.math_eval(_HANGS[0][0])
    assert payload["error"] is not None
    assert payload["error"]["type"] == "TimeBudgetError"
    assert payload["solve"] is None
    assert payload["quadrature"] is None
    assert payload["text"].startswith("error: integrate: gave up on")


def test_the_mcp_tool_description_warns_before_the_agent_calls() -> None:
    # Opis narzędzia to JEDYNA rzecz, którą agent czyta PRZED wołaniem (zasada z
    # modułu 6 i 8). Musi nieść nazwę klasy, po której się rozgałęzi, i to, czego
    # NIE wolno z niej wyczytać.
    doc = mcp_server.math_eval.__doc__ or ""
    assert "TimeBudgetError" in doc
    # PRZEPISANY W MODULE 10 — przypadek (i). Pinował FRAZĘ „no closed form", bo w
    # module 9 to była jedyna forma, w jakiej tamta odmowa istniała: proza w
    # komunikacie bez własnej klasy. Moduł 10 dał jej klasę, więc opis rozróżnia je
    # teraz po NAZWIE KLASY, którą agent odczyta z `error.type` — czyli dokładnie to,
    # czego ten test pilnował, tylko silniej. Rozróżnienie NIE ZNIKA: pinujemy obie
    # nazwy plus zdanie o tym, czego z żadnej z nich nie wolno wyczytać.
    assert "NoClosedFormError" in doc
    assert "UnsupportedFormError" in doc
    assert "says nothing about whether an answer EXISTS" in doc


# --- moduł 10: budżet czasu też mówi, DOKĄD pójść -------------------------
def test_the_time_budget_refusal_carries_the_route_of_the_operation() -> None:
    # Komunikat modułu 9 kończy się słowami „or compute numerically" i NIGDY nie
    # mówił, CZYM. Moduł 10 wyprowadza trasę z OPERACJI, więc ta sama odmowa niesie
    # teraz nazwę wywołania — jako dana, nie jako zdanie.
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(_SHORT):
            symbolic.integrate(E("1/(x^5+x+1)"), "x")
    assert exc.value.operation == "integrate"
    assert exc.value.route == "nintegrate"


def test_an_operation_with_no_numerical_counterpart_gets_no_route() -> None:
    # I to jest ta połowa, której nie wolno zgadywać. `simplify` nie ma
    # numerycznego odpowiednika w gramatyce, więc trasy NIE MA — wymyślona byłaby
    # dokładnie tym rodzajem pomyłki, który jest gorszy od milczenia, bo agent
    # pójdzie w nią bez pytania.
    from pycodemath.core.errors import NUMERIC_ROUTES

    assert "simplify" not in NUMERIC_ROUTES
    assert "series" not in NUMERIC_ROUTES
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(_SHORT):
            series(E("exp(sin(exp(x)))"), "x", 0, 14)
    assert exc.value.operation == "series"
    assert exc.value.route is None


def test_the_mcp_wire_carries_the_time_budget_route_too() -> None:
    # Ta sama dana MUSI przejść przez granicę procesu — inaczej byłaby widoczna
    # tylko dla wołającego z Pythona, a agent MCP (czyli powierzchnia, dla której
    # cała ta seria istnieje) dalej czytałby prozę.
    with time_budget(_SHORT):
        payload = mcp_server.math_eval("integrate 1/(x^5+x+1) dx")
    assert payload["error"]["type"] == "TimeBudgetError"
    assert payload["error"]["route"] == "nintegrate"


# --- (10) znalezisko A: wyciek _Deadline i zatrucie wątku ------------------
def _c_level_past(budget: float) -> None:
    # JEDNO wywołanie C, które trzyma GIL: 7**(10**6) to ~0,2 s na Windows /
    # 3.12.10. Strażnik nie dostaje GIL-a, dopóki ono nie wróci — pierwsze
    # oddanie GIL-a to wejście do __exit__, i tam dawniej lądował _Deadline.
    with time_budget(budget):
        _ = 7 ** (10**6)


def test_a_c_level_call_past_the_budget_gets_the_public_error_every_time() -> None:
    # Na SERII, nie na jednym przypadku: przed naprawą 10/10 przebiegów oddało
    # wołającemu goły _Deadline, a slot zostawał z rosnącą głębokością (1, 2, 3…)
    # i zatruwał każdy następny blok na tym wątku.
    for _ in range(5):
        with pytest.raises(TimeBudgetError):
            _c_level_past(0.02)
        assert threading.get_ident() not in budget_mod._ARMED
    # i wątek dalej pracuje — zwykłe wywołanie po serii przepaleń
    assert str(parse("sin(x)*x").diff("x")) == "x*cos(x) + sin(x)"


def test_a_tighter_nested_budget_that_fires_does_not_poison_the_outer() -> None:
    # Znalezisko V1 (25.09): ciaśniejszy wewnętrzny budżet, który odpalił,
    # nadpisywał termin zewnętrznego i zostawiał „fired" — zmierzone:
    # time_budget(10) odmówił po 0,17 s. Po naprawie zewnętrzny termin wraca.
    tid = threading.get_ident()
    for _ in range(3):
        with time_budget(30.0):
            outer = budget_mod._ARMED[tid][0]
            with pytest.raises(TimeBudgetError):
                with time_budget(_SHORT):
                    symbolic.integrate(E("1/(x^5+x+1)"), "x")
            assert budget_mod._ARMED[tid][0] == outer  # termin przywrócony
            assert budget_mod._ARMED[tid][2] is False  # nie „odpalony"
            spin_until = time.monotonic() + 0.3  # czysty Python: tu wstrzyknięcie by trafiło
            while time.monotonic() < spin_until:
                pass
        assert tid not in budget_mod._ARMED


def test_a_real_error_leaving_a_block_whose_deadline_passed_is_kept() -> None:
    # Prawdziwy wyjątek, który już wychodzi z bloku, wygrywa z odmową bloku —
    # także gdy termin zdążył minąć (dawniej przerwanie w sprzątaniu podmieniało go).
    with pytest.raises(ZeroDivisionError):
        with time_budget(0.02):
            _ = 7 ** (10**6)
            1 / 0
    assert threading.get_ident() not in budget_mod._ARMED
