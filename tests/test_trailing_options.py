"""Moduł 11: gramatyka REPL-a/MCP dociera do knobów, które API Pythona miało od zawsze.

POMIAR, KTÓRY TEN PLIK ISTNIEJE, ŻEBY POWTÓRZYĆ. `tol`/`max_iter` na
`minimize`/`minimize_nd`, `tol` na `integrate_num` i `time_budget` na silniku
symbolicznym istniały w API Pythona od modułów 5 i 9 — ale gramatyka komend
REPL-a i MCP (`min ... at 0`, `nintegrate ... from a to b`, `integrate ... dx`)
nie miała jak ich poprosić. Moduły 5-10 odraczały to PIĘĆ razy (dev-journal,
"NIE ROBIONE" każdego wpisu). Moduł 11 domyka lukę na DOWODZIE, nie na
symetrii: `tol`/`max_iter`/`budget` wchodzą, bo zmierzone RATUJĄ konkretne
wywołania (niżej); `lr`, `max_evals` i `tol`/`max_iter` na `root`/`solve_nd`
zostają wyłącznie w API Pythona, bo zmierzone tego nie robią.

Metoda z modułów 8-10: najpierw ZMIERZ prawdziwą liczbę (patrz sondy w opisie
commitu), potem asertuj ją — z marginesem tam, gdzie liczba może się różnić o
ostatni ulp między maszynami, dokładnie tam, gdzie nie może.
"""

from __future__ import annotations

import time

import pytest

from pycodemath.cli import mcp_server, repl
from pycodemath.core.errors import (
    DomainError,
    NonConvergenceError,
    ParseError,
    PycodemathError,
    TimeBudgetError,
)
from pycodemath.engine import numerics
from pycodemath.frontend.parser import parse

# --- ile komenda z modułu 11 miała dostać opcji — zapisane raz, pilnowane ----

#: Komendy, których wzorzec ROZPOZNAJE `_OPTS` (budget/tol/max_iter na końcu
#: linii) — dokładnie te dziewięć, i żadna inna. Reszta gramatyki (`diff`,
#: `grad`, `root`, `solve_nd`, macierze, `code`, cały ODE poza `dsolve`) nie
#: dostała tej gałęzi, bo pomiar (README, sekcja „Co zostało poza") pokazał,
#: że nic by tam nie zyskała albo by SZKODZIŁA (root/solve_nd).
_COMMANDS_WITH_OPTIONS = frozenset(
    {"integrate", "solve", "limit", "series", "sum", "dsolve", "min", "min_nd", "nintegrate"}
)

#: Klucze, które KAŻDA z tych komend deklaruje w swoim tekście `Usage:` —
#: mechaniczne pilnowanie, że dokumentacja nie mija się z gramatyką (moduł 7's
#: metoda zastosowana tu do samego repl.py, nie tylko do README).
_EXPECTED_OPTION_KEYS = {
    "integrate": {"budget"},
    "solve": {"budget"},
    "limit": {"budget"},
    "series": {"budget"},
    "sum": {"budget"},
    "dsolve": {"budget"},
    "min": {"tol", "max_iter"},
    "min_nd": {"tol", "max_iter"},
    "nintegrate": {"tol"},
}


def test_exactly_the_measured_commands_carry_trailing_options():
    with_opts = {
        token
        for token, (pattern, _handler, _usage) in repl._COMMANDS.items()
        if repl._OPTS in pattern.pattern
    }
    assert with_opts == _COMMANDS_WITH_OPTIONS


def test_the_usage_string_of_every_option_bearing_command_names_its_keys():
    for token, keys in _EXPECTED_OPTION_KEYS.items():
        usage = repl._COMMANDS[token][2]
        for key in keys:
            assert f"{key} <" in usage, (token, key, usage)


def test_the_option_whitelists_match_the_measured_decision():
    # `lr` jest w numerics.minimize od zawsze, a NIE jest tu — bo zmierzone
    # (README) tylko zamienia jedną niezbieżność na inną. `max_evals` nie jest
    # w _NINTEGRATE_OPTS z tego samego powodu (niewyczerpane na żadnej sondzie).
    assert set(repl._MIN_OPTS) == {"tol", "max_iter"}
    assert set(repl._NINTEGRATE_OPTS) == {"tol"}
    assert set(repl._BUDGET_OPTS) == {"budget"}


# --- ratunek min x^4: dokładnie te liczby, zmierzone ------------------------


def test_min_x4_needs_a_rescue_the_default_iteration_budget_cannot_give():
    """`min x^4 for x at 1` pod domyślnym (tol=1e-9, max_iter=10000) NIE zbiega —
    zmierzone: potrzeba 678 594 iteracji do tol=1e-9, więc gramatyka bez opcji
    rzuca. Dwa RÓŻNE ratunki, i to jest teza: WIĘKSZY max_iter (ten sam cel,
    więcej czasu) i LUŹNIEJSZY tol (inny cel, mniej czasu) — obie ścieżki
    działają, bo mierzą różne rzeczy."""
    with pytest.raises(NonConvergenceError):
        repl.handle("min x^4 for x at 1")

    rescued_by_iterations = float(repl.handle("min x^4 for x at 1 max_iter 700000"))
    assert abs(rescued_by_iterations) < 0.01

    rescued_by_looser_tol = float(repl.handle("min x^4 for x at 1 tol 1e-5"))
    assert repl.handle("min x^4 for x at 1 tol 1e-5") == "0.029229144526165384"
    assert abs(rescued_by_looser_tol) < 0.1


# --- parytet: REPL z opcją == API Pythona z tym samym kwargiem, bit w bit ---


def test_min_tol_through_the_repl_is_bit_identical_to_the_python_api():
    via_repl = float(repl.handle("min x^4 for x at 1 tol 1e-5"))
    via_api = numerics.minimize(parse("x^4"), "x", 1.0, tol=1e-5)
    assert via_repl == via_api


def test_min_nd_tol_and_max_iter_through_the_repl_are_bit_identical_to_the_python_api():
    via_repl = repl.handle("min_nd (x-1)^2+(y+2)^2 for x,y at 0,0 tol 1e-3 max_iter 50")
    via_api = numerics.minimize_nd(
        parse("(x-1)^2+(y+2)^2"), ["x", "y"], [0.0, 0.0], tol=1e-3, max_iter=50
    )
    assert via_repl == f"x = {via_api[0]!r}, y = {via_api[1]!r}"


def test_nintegrate_tol_through_the_repl_is_bit_identical_to_the_python_api():
    via_repl = float(repl.handle("nintegrate 2*x dx from 0 to 1 tol 1e-8"))
    via_api = numerics.integrate_num(parse("2*x"), "x", 0.0, 1.0, tol=1e-8)
    assert via_repl == via_api


# --- nintegrate tol: mierzy error_estimate, i UMIE nim wycelować -------------


def test_nintegrate_tol_aims_the_error_estimate_the_fixed_grid_cannot():
    """`√x` na `[0, 1]`: pochodna eksploduje w 0, więc stała siatka (bez `tol`)
    myli się o ~8.1e-5 (zmierzone w numerics.integrate_num). `tol 1e-10` zbija
    prawdziwy błąd o ponad 3 000×."""
    true_value = 2.0 / 3.0

    fixed = float(repl.handle("nintegrate sqrt(x) dx from 0 to 1"))
    assert repl.handle("nintegrate sqrt(x) dx from 0 to 1") == "0.666585482066724"
    fixed_error = abs(fixed - true_value)

    aimed = float(repl.handle("nintegrate sqrt(x) dx from 0 to 1 tol 1e-10"))
    assert repl.handle("nintegrate sqrt(x) dx from 0 to 1 tol 1e-10") == "0.6666666666666469"
    aimed_error = abs(aimed - true_value)

    assert aimed_error < fixed_error / 1000


# --- budget: zastępuje domyślne 120 s TYLKO na to wywołanie ------------------
#
# `integrate 1/(x^8+x+1) dx` zmierzone na tej maszynie: ~2.4 s bez budżetu —
# dość wolne, żeby udowodnić że mały budżet PRZERYWA je (a nie czeka), i dość
# szybkie, żeby duży budżet skończył test w rozsądnym czasie (moduł 9 uczy: nie
# pożyczaj prawdziwie wiszącego reproducenta do testu, który MUSI się skończyć —
# tu kończy się zawsze, więc to bezpieczne).
_MODERATE_INTEGRAL = "integrate 1/(x^8+x+1) dx"

#: Budżet użyty, gdy test MUSI naprawdę odpalić strażnika — 1.0 s, ta sama
#: skala co `_SHORT` w test_budget.py, z tego samego powodu: kiedy przerwanie
#: trafia w zagnieżdżony (odziedziczony) strażnik zamiast w zewnętrzny, jego
#: `.expired()` liczy `.spent` od WŁASNEGO wejścia, nie od początku bloku —
#: zmierzony niedomiar rzędu ~0,08 s. Przy budżecie 0,3 s to złamało nawet
#: margines 0,9× (test_budget.py's własna reguła); przy 1,0 s ten sam niedomiar
#: mieści się w nim wygodnie.
_SHORT = 1.0


def test_a_small_budget_refuses_what_the_default_would_have_answered():
    t0 = time.monotonic()
    with pytest.raises(TimeBudgetError) as excinfo:
        repl.handle(f"{_MODERATE_INTEGRAL} budget {_SHORT}")
    spent = time.monotonic() - t0

    # dolna granica jest bezpieczna pod obciążeniem (moduł 9's reguła —
    # test_budget.py's margines 0,9×, patrz `_SHORT` wyżej); górna jest hojną
    # wielokrotnością, nie pomiarem zwłoki.
    assert spent >= _SHORT * 0.9
    assert spent < _SHORT + 30.0

    exc = excinfo.value
    assert exc.budget == _SHORT
    assert exc.spent >= _SHORT * 0.9
    assert exc.route == "nintegrate"


def test_a_generous_budget_answers_the_same_call_a_short_one_refused():
    out = repl.handle(f"{_MODERATE_INTEGRAL} budget 10")
    assert out and "gave up" not in out


def test_budget_is_scoped_to_one_call_the_next_one_gets_the_default_back():
    # Krótki budżet na JEDNO wywołanie odmawia; zaraz potem, BEZ budżetu, ta
    # sama komenda korzysta z domyślnych 120 s i wraca odpowiedzią — udowadnia,
    # że `budget` nie zostawia śladu w wątku (moduł 9's re-entrancy, przetestowane
    # tu przez gramatykę, nie przez API).
    with pytest.raises(TimeBudgetError):
        repl.handle(f"{_MODERATE_INTEGRAL} budget {_SHORT}")
    out = repl.handle(_MODERATE_INTEGRAL)
    assert out and "gave up" not in out


# --- gramatyka opcji: kolejność, powtórzenia, złe klucze, złe wartości ------


def test_trailing_options_are_order_independent():
    a = repl.handle("min x^2 for x at 1 tol 1e-3 max_iter 50")
    b = repl.handle("min x^2 for x at 1 max_iter 50 tol 1e-3")
    assert a == b


def test_a_repeated_option_key_is_a_parse_error():
    with pytest.raises(ParseError, match="given twice"):
        repl.handle("min x^2 for x at 0 tol 1e-3 tol 1e-4")


def test_an_option_key_outside_this_commands_whitelist_is_a_parse_error():
    # `lr` istnieje w numerics.minimize, ale NIE w gramatyce (README: zmierzone
    # nie ratuje niczego, co `method newton|bfgs` już nie ratuje) — więc jest
    # nieznanym kluczem tutaj, dokładnie jak każdy inny literówka.
    with pytest.raises(ParseError, match="unknown option 'lr'"):
        repl.handle("min x^2 for x at 0 lr 0.5")


def test_an_unreadable_option_value_is_a_parse_error():
    with pytest.raises(ParseError, match="invalid number"):
        repl.handle("min x^2 for x at 0 tol abc")
    with pytest.raises(ParseError, match="invalid integer"):
        repl.handle("min x^2 for x at 0 max_iter 1.5")


@pytest.mark.parametrize(
    "command",
    [
        "min x^2 for x at 0 tol 0",
        "min x^2 for x at 0 tol -1",
        "min x^2 for x at 0 tol inf",  # audyt v0.3: niefinityczny tol certyfikował
        "min x^2 for x at 0 tol nan",  # nietknięty punkt startowy jako "zbieżny"
        "min x^2 for x at 0 max_iter 0",
        "integrate 2*x dx budget 0",
        "integrate 2*x dx budget -5",
        "integrate 2*x dx budget inf",  # niefinityczny budżet: dozwolony w API
        "integrate 2*x dx budget nan",  # (math.inf), NIEDOZWOLONY na tej powierzchni
    ],
)
def test_a_readable_value_outside_its_domain_is_a_domain_error(command):
    with pytest.raises(DomainError):
        repl.handle(command)


def test_tol_inf_does_not_certify_the_untouched_starting_guess_as_converged():
    # Audyt v0.3 (sesja weryfikacyjna Modulu 11): `_opt_tol`'s original domain
    # check (`not value > 0.0`) let `inf` through, because `inf > 0.0` is True.
    # `res < tol` on gradient descent's FIRST iteration is then trivially true
    # for any finite residual, so `min (x-3)^2 for x at 100 tol inf` came back
    # `SolveResult(value=100.0, residual=194.0, converged=True)` — the starting
    # guess, unmoved, certified as the answer. `_opt_budget` already guarded
    # this exact class of input (non-finite) for the same reason; `_opt_tol`
    # now does too.
    with pytest.raises(DomainError, match="finite"):
        repl.handle("min (x-3)^2 for x at 100 tol inf")
    payload = mcp_server.math_eval("min (x-3)^2 for x at 100 tol inf")
    assert payload["solve"] is None
    assert payload["error"]["type"] == "DomainError"


def test_a_dangling_option_key_falls_through_to_the_usage_message_not_an_exception():
    # `_OPTS` wymaga PEŁNYCH par `klucz wartość` — więc `... tol` bez wartości
    # nie dopasowuje się do wzorca w ogóle i dostaje ten sam komunikat Usage,
    # co każda inna źle sformowana linia od modułu 1.
    out = repl.handle("min x^2 for x at 0 tol")
    assert out.startswith("Usage: min ")


@pytest.mark.parametrize(
    "command",
    [
        "root x^2-2 for x at 1 tol 1e-5",
        "root x^2-2 for x at 1 budget 5",
        "solve_nd x^2+y^2-4; x-y for x,y at 1,1 max_iter 10",
        "diff sin(x)*x dx budget 5",
        "grad x^2*y for x,y budget 5",
    ],
)
def test_option_shaped_text_on_a_command_without_the_grammar_is_just_a_bad_line(command):
    # `root`, `solve_nd`, `diff`, `grad` w ogóle NIE MAJĄ `_OPTS` we wzorcu —
    # więc doklejony `klucz wartość` nie jest "nieznaną opcją" (ParseError),
    # tylko psuje CAŁY wzorzec, jak każdy zbędny token od modułu 1: pada
    # usage, nie wyjątek.
    out = repl.handle(command)
    assert out.startswith("Usage: ")


@pytest.mark.parametrize(
    "command",
    [
        "nintegrate 2*x dx from 0 to 1 max_iter 10",
        "nintegrate 2*x dx from 0 to 1 budget 5",
    ],
)
def test_nintegrate_has_opts_in_its_grammar_but_rejects_keys_outside_its_own_whitelist(command):
    # `nintegrate` MA `_OPTS` (jego whitelist to `tol`), więc `max_iter`/
    # `budget` na niej to inna gałąź niż powyżej: pattern DOPASOWUJE się i
    # `_options` odrzuca dopiero po stronie whitelisty — `ParseError`, tak jak
    # `lr` na `min`, nie usage.
    with pytest.raises(ParseError, match="unknown option"):
        repl.handle(command)


# --- drut MCP: opcje złe na wejściu I odmowa budżetu, obie jako DANE --------


_INPUT_REFUSALS_FROM_OPTIONS = [
    ("nieznany klucz", "min x^2 for x at 0 lr 0.5", "ParseError"),
    ("powtórzony klucz", "min x^2 for x at 0 tol 1e-3 tol 1e-4", "ParseError"),
    ("nieczytelna wartość", "min x^2 for x at 0 tol abc", "ParseError"),
    ("tol spoza dziedziny", "min x^2 for x at 0 tol 0", "DomainError"),
    ("budget spoza dziedziny", "integrate 2*x dx budget -1", "DomainError"),
    ("budget niefinityczny", "integrate 2*x dx budget inf", "DomainError"),
]


@pytest.mark.parametrize(
    "command,cls",
    [(c[1], c[2]) for c in _INPUT_REFUSALS_FROM_OPTIONS],
    ids=[c[0] for c in _INPUT_REFUSALS_FROM_OPTIONS],
)
def test_mcp_wire_carries_a_bad_option_as_an_ERROR_with_no_route(command, cls):
    # Ta sama reguła modułu 10, zastosowana do NOWEGO źródła odmowy wejścia:
    # linia źle sformowana przez opcję nie odbyła żadnego biegu, więc `route`
    # jest `None` — dokładnie jak dla błędnej macierzy albo nieznanej metody.
    payload = mcp_server.math_eval(command)
    assert payload["solve"] is None and payload["quadrature"] is None
    assert payload["error"]["type"] == cls
    assert payload["error"]["route"] is None
    assert payload["text"] == f"error: {payload['error']['message']}"


def test_mcp_wire_carries_a_budget_refusal_with_its_route():
    # W przeciwieństwie do powyższych: `budget` sam w sobie jest CZYTELNY i W
    # DZIEDZINIE — odmowa przychodzi z BIEGU (strażnik przerwał SymPy), więc
    # `route` NIE jest `None`, tak jak dla każdego innego `TimeBudgetError`.
    payload = mcp_server.math_eval(f"{_MODERATE_INTEGRAL} budget {_SHORT}")
    assert payload["error"]["type"] == "TimeBudgetError"
    assert payload["error"]["route"] == "nintegrate"


def test_mcp_wire_never_raises_on_a_bad_option_it_always_returns_data():
    # Zamówienie modułu 6: agent dostaje DANE, nigdy traceback. Sprawdzone tu
    # osobno, bo `_options`/`_opt_budget`/`_opt_tol` to nowy kod modułu 11 —
    # gdyby któryś rzucał czymś spoza `PycodemathError`, `math_eval`'s `except
    # Exception` by to i tak złapał, ale to WARTE osobnego pinu.
    for command in (c[1] for c in _INPUT_REFUSALS_FROM_OPTIONS):
        payload = mcp_server.math_eval(command)  # nie rzuca
        assert isinstance(payload, dict)
        assert payload["error"] is not None
