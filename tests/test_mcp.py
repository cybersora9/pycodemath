"""Testy powierzchni MCP (moduł 6): strukturalne wyniki DOCHODZĄ do modelu.

Moduły 3–5 zbudowały dowody (``SolveResult``, ``QuadratureResult``), ale nic, z
czym agent rozmawia, o nie nie prosiło — serwer MCP zwracał goły string, więc
każda gwarancja modułów 2–5 ginęła na granicy procesu. Ten plik pinuje kształt
odpowiedzi, granicę „porażka jako dane vs odmowa jako błąd" i to, że kanał
strukturalny NIGDY nie kosztuje odpowiedzi.

Konwencja serii: komentarze po polsku, kod i stringi pakietu po angielsku.
Testy dzielą się na dwie grupy:
  * PAYLOAD — wołają ``mcp_server.math_eval`` wprost, więc działają BEZ pakietu
    ``mcp`` (payload jest zwykłym dict-em, celowo);
  * WIRE — wołają serwer przez PRAWDZIWĄ sesję klienta MCP w pamięci, więc
    mierzą to, co naprawdę leci na drut (schema, walidacja, structuredContent).
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from pycodemath.cli import mcp_server, repl
from pycodemath.core.errors import DomainError
from pycodemath.core.result import SOLVE_STATUSES, QuadratureResult, SolveResult


# --- POMIAR WYJŚCIOWY: co model dostawał PRZED modułem 6 ---------------------
def test_the_old_shape_is_gone_and_prose_is_no_longer_the_only_channel():
    # PRZED: outputSchema = {"result": string}, a KAŻDA odpowiedź (sukces, porażka,
    # odmowa) była JEDNYM stringiem — agent miał do dyspozycji angielską prozę.
    # Zmierzone przed zmianą, verbatim:
    #   {"result": "1.4142135623746899"}
    #   {"result": "error: minimize does not converge for x**4 at x0=1.0 (...)"}
    # Teraz ten sam string nadal JEST (pole ``text``), ale obok niego jadą dane.
    payload = mcp_server.math_eval("root x^2-2 for x at 1")
    assert set(payload) == {"text", "solve", "quadrature", "error"}
    assert payload["text"] == "1.4142135623746899"
    assert payload["solve"] is not None
    assert payload["solve"]["converged"] is True


# --- KSZTAŁT ODPOWIEDZI -----------------------------------------------------
#: Komendy pokrywające wszystkie trzy formy payloadu. Kolumny:
#: (id, komenda, które pole niesie dowód albo None gdy żadne).
_SHAPES = [
    ("root/sukces", "root x^2-2 for x at 1", "solve"),
    ("min/sukces", "min (x-3)^2 for x at 0 method newton", "solve"),
    ("solve_nd/sukces", "solve_nd x^2+y^2-4; x-y for x,y at 1,1", "solve"),
    ("min_nd/sukces", "min_nd (x-1)^2+(y+2)^2 for x,y at 0,0", "solve"),
    ("nintegrate/sukces", "nintegrate 2*x dx from 0 to 1", "quadrature"),
    ("root/porażka", "root x^2+1 for x at 1", "solve"),
    ("diff", "diff x^2 dx", None),
    ("det", "det [[1,2],[3,4]]", None),
    ("code", "code (sin(x)+cos(x))^2", None),
    ("ode", "ode y for y(t) from 0 to 1 at 1", None),
    ("simplify (bez komendy)", "sin(x)^2 + cos(x)^2", None),
    ("help", "help", None),
    ("usage", "diff x^2", None),
]


@pytest.mark.parametrize(
    "command,carrier", [(c[1], c[2]) for c in _SHAPES], ids=[c[0] for c in _SHAPES]
)
def test_every_response_has_the_same_four_fields(command, carrier):
    # JEDEN kształt na wszystko, co narzędzie umie odpowiedzieć — agent nie musi
    # zgadywać schematu per komenda (a przy 25 komendach nie miałby jak).
    payload = mcp_server.math_eval(command)
    assert set(payload) == {"text", "solve", "quadrature", "error"}
    assert isinstance(payload["text"], str) and payload["text"] != ""
    assert payload["error"] is None
    # dowód siedzi DOKŁADNIE w jednym polu — albo w żadnym
    for field in ("solve", "quadrature"):
        if field == carrier:
            assert payload[field] is not None
        else:
            assert payload[field] is None


def test_solve_and_quadrature_are_separate_fields_and_do_not_swap_vocabulary():
    # Moduł 5 dał kwadraturze WŁASNY typ, bo oszacowanie błędu nie jest residuum
    # (residuum jest MIERZONE, oszacowanie WYWNIOSKOWANE pod założeniem, które
    # integrand może złamać). Spłaszczenie obu w jeden luźny „result" cofnęłoby
    # dokładnie to rozróżnienie — tu pilnuje go NAZWA POLA.
    solved = mcp_server.math_eval("root x^2-2 for x at 1")["solve"]
    integrated = mcp_server.math_eval("nintegrate 2*x dx from 0 to 1")["quadrature"]
    assert set(solved) == {"value", "iterations", "residual", "converged", "status"}
    assert set(integrated) == {
        "value",
        "error_estimate",
        "evaluations",
        "refinements",
        "converged",
        "status",
    }
    # żadne słownictwo nie przeciekło w drugą stronę
    assert "error_estimate" not in solved and "evaluations" not in solved
    assert "residual" not in integrated and "iterations" not in integrated


def test_there_is_no_second_encoding_of_refusal_or_of_quality():
    # Świadomie NIE ma pola „ok": „czy odmówiono" mówi już ``error is None``,
    # a „czy wartość jest odpowiedzią" mówi ``converged``. Trzeci boolean byłby
    # drugim zapisem tego samego faktu — dryf, przed którym moduły 3–5 broniły
    # się jednym konstruktorem na formę. Ta sama dyscyplina: payload powstaje
    # WYŁĄCZNIE w ``_answer`` albo ``_refusal``.
    for _id, command, _carrier in _SHAPES:
        payload = mcp_server.math_eval(command)
        assert "ok" not in payload and "success" not in payload
    refused = mcp_server.math_eval("det [[1,2],[3]]")
    assert refused["error"] is not None
    assert refused["solve"] is None and refused["quadrature"] is None


# --- GRANICA: PORAŻKA JAKO DANE vs ODMOWA JAKO BŁĄD -------------------------
#: Porażki ITERACJI osiągalne przez gramatykę REPL-a (repro z test_result.py,
#: te same funkcje: atan(x)+5 → ucieczka Newtona; x²+1 → brak pierwiastka
#: rzeczywistego; −x² wklęsła → ucieczka; Rosenbrock gd → dławi się budżetem).
#: „stagnated" NIE jest tędy osiągalne — komenda ``min`` nie wystawia ``lr``,
#: a stagnacja gd wymaga lr=1 (patrz test niżej, który pinuje to jako USTALENIE).
_ITERATION_FAILURES = [
    ("root/diverged", "root atan(x)+5 for x at 2", "diverged"),
    ("root/not_converged", "root x^2+1 for x at 1", "not_converged"),
    ("min/diverged", "min -x^2 for x at 1", "diverged"),
    # moduł 8: PIĄTY wynik jedzie po tym samym drucie, bez zmiany kształtu — i
    # jest OSIĄGALNY z gramatyki komend (w odróżnieniu od "stagnated" niżej),
    # bo nie wymaga żadnej opcji, której komenda nie wystawia: wystarczy
    # wystartować NA punkcie stacjonarnym.
    ("min/not_a_minimum", "min -x^2 for x at 0", "not_a_minimum"),
    ("min_nd/not_a_minimum", "min_nd x*y for x,y at 0,0", "not_a_minimum"),
    ("min/not_converged", "min x for x at 0", "not_converged"),
    (
        "min_nd/not_converged",
        "min_nd (1-x)^2 + 100*(y-x^2)^2 for x,y at -1.2,1",
        "not_converged",
    ),
    (
        "solve_nd/not_converged",
        "solve_nd x^2+y^2+1; x-y for x,y at 1,1",
        "not_converged",
    ),
]


@pytest.mark.parametrize(
    "command,status",
    [(c[1], c[2]) for c in _ITERATION_FAILURES],
    ids=[c[0] for c in _ITERATION_FAILURES],
)
def test_iteration_failures_travel_as_DATA_not_as_an_error(command, status):
    # Moduły 4–5 uczyniły wynik iteracji ZWRACALNYM — tu to się opłaca. Bieg,
    # który się nie zbiegł, MA co zameldować: ostatni iterat, residuum, koszt.
    # Więc ``error`` jest puste, a rozgałęzia się po ``converged``.
    payload = mcp_server.math_eval(command)
    assert payload["error"] is None, "porażka iteracji to NIE odmowa wejścia"
    evidence = payload["solve"]
    assert evidence is not None
    assert evidence["converged"] is False
    assert evidence["status"] == status
    assert evidence["status"] in SOLVE_STATUSES
    assert evidence["iterations"] >= 1
    # ta sama komenda BEZ powierzchni MCP nadal RZUCA — ścieżka domyślna
    # niezmieniona (dowód, że to opcja wołającego, a nie zmiana silnika)
    with pytest.raises(Exception):
        repl.handle(command)


#: Odmowy WEJŚCIA osiągalne przez gramatykę REPL-a: klasa wyjątku i komenda.
_INPUT_REFUSALS = [
    ("ParseError/macierz", "det [[1,2],[3]]", "ParseError"),
    ("ParseError/liczba", "root x^2-2 for x at 1e", "ParseError"),
    ("DomainError/metoda", "min (x-3)^2 for x at 0 method sgd", "DomainError"),
    ("DomainError/układ", "solve_nd x^2+y^2-4 for x,y at 1,1", "DomainError"),
]


@pytest.mark.parametrize(
    "command,cls",
    [(c[1], c[2]) for c in _INPUT_REFUSALS],
    ids=[c[0] for c in _INPUT_REFUSALS],
)
def test_input_refusals_travel_as_an_ERROR_with_the_class_on_the_wire(command, cls):
    # Granica modułu 4, teraz WIDOCZNA na drucie: odmowa wejścia nie jest wynikiem
    # iteracji, więc nie ma statusu ani ostatniego iteratu — i nie udaje, że ma.
    # Klasa wyjątku jest agent-czytelną częścią hierarchii modułu 2; dlatego
    # jedzie jako ``error.type``, a nie tonie w prozie.
    #
    # PRZEPISANY W MODULE 10 — przypadek (i): porównanie CAŁEGO słownika `error`
    # pinowało jego kształt na dwóch kluczach, a moduł 10 dokłada trzeci (`route`).
    # Test NIE JEST rozluźniony — dalej porównuje CAŁY słownik, a nowy klucz dostaje
    # oczekiwaną wartość WPROST i jest nią `None`. To jest twierdzenie, nie
    # wypełniacz: odmowa WEJŚCIA nie ma trasy numerycznej, bo nie odbył się żaden
    # bieg, któremu można by zaproponować inną drogę — źle sparsowana macierz nie
    # stanie się poprawna od policzenia jej numerycznie.
    payload = mcp_server.math_eval(command)
    assert payload["solve"] is None and payload["quadrature"] is None
    assert payload["error"] == {
        "type": cls,
        "message": payload["error"]["message"],  # treść pinuje test_errors.py
        "route": None,
    }
    assert payload["error"]["message"] != ""
    assert payload["text"] == f"error: {payload['error']['message']}"


def test_the_same_command_produces_both_forms_so_the_boundary_is_visible():
    # Sedno punktu 3 zakresu: TA SAMA komenda, to samo narzędzie, dwie zupełnie
    # różne formy na drucie — bo różnica jest w MATEMATYCE, nie w składni.
    outcome = mcp_server.math_eval("min -x^2 for x at 1")  # wynik iteracji
    refusal = mcp_server.math_eval("min -x^2 for x at 1 method sgd")  # odmowa wejścia
    assert outcome["error"] is None and outcome["solve"] is not None
    assert refusal["error"] is not None and refusal["solve"] is None


def test_a_failure_outside_the_returnable_family_still_names_its_class():
    # UCZCIWE OGRANICZENIE: dane zamiast błędu istnieją TYLKO tam, gdzie moduły
    # 4–5 zbudowały formę zwracalną (cztery solwery + kwadratura). ODE nie ma
    # dziś typu wyniku, więc jej porażka jedzie jako błąd — ale z nazwą klasy
    # modułu 2, czyli i tak strukturalnie, a nie samą prozą. Dobudowa formy dla
    # ODE to praca w SILNIKU, poza zakresem tego modułu.
    # y' = y², y(0)=1 ucieka do nieskończoności w t=1 — Newton BDF2 nie zbiega,
    # czyli DOKŁADNIE wynik iteracji, który u solwera byłby dziś danymi.
    payload = mcp_server.math_eval("odestiff y^2 for y(t) from 0 to 2 at 1")
    assert payload["error"] is not None
    assert payload["error"]["type"] == "NonConvergenceError"
    assert payload["solve"] is None and payload["quadrature"] is None


# --- inf: „NIEMIERZALNE" przez JSON -----------------------------------------
def test_infinite_residual_becomes_null_because_JSON_has_no_infinity():
    # Moduł 4 wybrał ``math.inf`` na „residuum niemierzalne" (nie NaN: każde
    # porównanie z NaN jest False, więc `residual > tol` kłamałoby w najgorszą
    # stronę). JSON nie umie nieskończoności — ZMIERZONE: leci jako ``null`` i
    # WYWRACA walidację klienta na polu typu ``number``, zabijając całe
    # wywołanie. Tłumaczymy więc jawnie na ``null`` = „brak wartości", czyli
    # dokładnie to, co inf tam znaczyło.
    # sqrt(x)+1: krok Newtona wyprowadza iterat do x=-3, gdzie funkcja jest
    # zespolona — nie ma czego próbkować (repro z test_result.py).
    raw = mcp_server.repl.handle_full("root sqrt(x)+1 for x at 1").evidence
    assert isinstance(raw, SolveResult) and math.isinf(raw.residual)
    payload = mcp_server.math_eval("root sqrt(x)+1 for x at 1")
    assert payload["solve"]["residual"] is None
    assert payload["solve"]["value"] == -3.0
    # payload jest ŚCIŚLE serializowalny: żadnego Infinity/NaN na drucie
    json.loads(json.dumps(payload), parse_constant=_no_json_constants)


def _no_json_constants(name: str):
    raise AssertionError(f"na drucie pojawiła się nie-JSON-owa stała {name!r}")


@pytest.mark.parametrize(
    "command",
    [c[1] for c in _SHAPES] + [c[1] for c in _ITERATION_FAILURES],
    ids=[c[0] for c in _SHAPES] + [c[0] for c in _ITERATION_FAILURES],
)
def test_no_payload_ever_carries_a_non_json_number(command):
    # Reguła obowiązuje CAŁY payload, nie tylko residuum — łącznie z ostatnim
    # iteratem biegu rozbieżnego, który potrafi być inf/NaN.
    json.loads(json.dumps(mcp_server.math_eval(command)), parse_constant=_no_json_constants)


# --- KANAŁ STRUKTURALNY NIGDY NIE KOSZTUJE ODPOWIEDZI -----------------------
def test_asking_for_evidence_never_loses_an_answer_the_plain_call_would_give():
    # Dowody kosztują DODATKOWE próbki (residuum w punkcie zwróconym, siatka
    # środków dla oszacowania błędu), a te potrafią wypaść poza dziedzinę tam,
    # gdzie sama ODPOWIEDŹ była w porządku. Silnik wtedy odmawia (DomainError)
    # i mówi wprost: wartość jest dostępna bez full_result. Bierzemy to
    # dosłownie — spadamy na zwykły handler i meldujemy odpowiedź bez dowodu.
    # REPRO NATURALNY: węzły siatki n=100 na [0,1] trafiają w cos(200πx)=+1,
    # a ŚRODKI (potrzebne dopiero oszacowaniu) w cos=−1 → pierwiastek zespolony.
    command = "nintegrate sqrt(cos(200*pi*x)) dx from 0 to 1"
    assert repl.handle(command) == "1.0"  # zwykła droga: odpowiedź JEST
    with pytest.raises(DomainError):  # droga z dowodem: odmowa
        mcp_server.math_eval  # (kotwica czytelności; właściwe wywołanie niżej)
        repl._EVIDENCE["nintegrate"](
            repl._COMMANDS["nintegrate"][0].match(command)  # type: ignore[arg-type]
        )
    payload = mcp_server.math_eval(command)
    assert payload["text"] == "1.0"  # odpowiedź OCALONA
    assert payload["quadrature"] is None  # dowodu uczciwie NIE MA
    assert payload["error"] is None  # i to NIE jest odmowa


def test_the_fallback_is_wired_for_solvers_too(monkeypatch):
    # Ten sam strażnik na ścieżce solwerów. Naturalnego repro w gramatyce REPL-a
    # nie znaleziono (moduł 3 nazywa ten przypadek RZADKIM — trzeba pierwiastka
    # w punkcie, gdzie f nie da się spróbkować), więc wiring sprawdzamy
    # mechanicznie: gdy droga z dowodem odmawia, odpowiedź i tak wraca.
    def refuse(*_a, **_kw):
        raise DomainError("residual cannot be sampled there")

    monkeypatch.setattr(repl, "_ev_root", refuse)
    monkeypatch.setitem(repl._EVIDENCE, "root", refuse)
    payload = mcp_server.math_eval("root x^2-2 for x at 1")
    assert payload["text"] == "1.4142135623746899"
    assert payload["solve"] is None and payload["error"] is None


# --- ANTYDRYF: dwie drogi, JEDEN tekst --------------------------------------
#: Do testu równości tekstu bierzemy każdą komendę z ``_SHAPES`` PRÓCZ tej, którą
#: ``handle`` rzuca — bo niezmiennik jest warunkowy z założenia (patrz niżej).
_SAME_TEXT = [c for c in _SHAPES if c[0] != "root/porażka"]


@pytest.mark.parametrize(
    "command",
    [c[1] for c in _SAME_TEXT],
    ids=[c[0] for c in _SAME_TEXT],
)
def test_both_dispatchers_print_the_same_text(command):
    # ``handle`` (człowiek) i ``handle_full`` (agent) wołają silnik OSOBNO —
    # pierwszy bez ``full_result``, drugi z nim — więc tekst mógłby się rozjechać.
    # Nie może: formatowanie żyje w ``_fmt_scalar``/``_fmt_named`` i nigdzie
    # indziej. Niezmiennik: dla KAŻDEJ linii, którą ``handle`` odpowiada bez
    # rzucenia, ``handle_full`` daje bajt w bajt ten sam tekst.
    assert repl.handle_full(command).text == repl.handle(command)


def test_the_repl_is_unchanged_and_still_raises_where_it_raised():
    # Punkt 4 zakresu: REPL zostaje CZŁOWIEKOWI. Nie woła ``full_result``, bo
    # tam porażka iteracji jest ZWRACANA — REPL wypisywałby ostatni iterat tam,
    # gdzie dziś wypisuje błąd. To byłaby cicha zmiana zachowania na powierzchni,
    # po której drugiej stronie siedzi człowiek czytający liczbę.
    for _id, command, _status in _ITERATION_FAILURES:
        with pytest.raises(Exception):
            repl.handle(command)
        # a agent dostaje TEN SAM bieg jako dane
        assert mcp_server.math_eval(command)["solve"]["converged"] is False


def test_the_text_warns_when_the_value_is_not_an_answer():
    # Klient, który renderuje SAM tekst (bez structuredContent), nie może
    # przeczytać ostatniego iteratu jako odpowiedzi. Więc tekst to mówi — a
    # strukturalna połowa mówi to samo w ``converged``/``status``; obie
    # powstają razem, w jednym miejscu.
    payload = mcp_server.math_eval("root x^2+1 for x at 1")
    assert "NOT CONVERGED" in payload["text"]
    assert payload["solve"]["status"] in payload["text"]
    ok = mcp_server.math_eval("root x^2-2 for x at 1")
    assert "NOT CONVERGED" not in ok["text"]


def test_stagnated_is_unreachable_through_the_command_grammar_and_that_is_known():
    # USTALENIE, nie przeoczenie: stagnacja gd wymaga lr=1 (moduł R11 — cykl
    # okresu 2 na x⁴), a komendy ``min``/``min_nd`` nie wystawiają ``lr``.
    # Podobnie porażki KWADRATURY wymagają ``tol``, którego ``nintegrate`` nie
    # wystawia. Nie dokładamy tych opcji w tym module: to zmiana gramatyki
    # komend (powierzchnia człowieka), a moduł 6 ma DONIEŚĆ istniejące
    # odpowiedzi, nie poszerzać ich. Status żyje i jest osiągalny z biblioteki:
    from pycodemath.engine import numerics
    from pycodemath.frontend.parser import parse

    stalled = numerics.minimize(parse("x^4"), "x", 2.0, lr=1.0, full_result=True)
    assert stalled.status == "stagnated"
    # a gdyby kiedyś doszedł do powierzchni, payload już go uniesie bez zmian
    assert mcp_server._solve_evidence(stalled)["status"] == "stagnated"


def test_quadrature_evidence_carries_module_5_fields_including_the_estimate():
    payload = mcp_server.math_eval("nintegrate sin(x) dx from 0 to 3.141592653589793")
    quad = payload["quadrature"]
    assert quad["converged"] is True and quad["status"] == "converged"
    assert quad["refinements"] == 0  # ścieżka stałej siatki niczego nie dzieli
    assert quad["evaluations"] == 201  # 101 węzłów + 100 środków oszacowania
    # oszacowanie to OSZACOWANIE, nie granica — ale tu jest realne i małe
    assert 0.0 <= quad["error_estimate"] < 1e-7
    # zmierzone w module 5: błąd WZGLĘDNY sin po [0,π] przy n=100 to 5.4e-09,
    # czyli bezwzględny ~1.1e-08 przy całce równej 2
    assert abs(quad["value"] - 2.0) < 2e-8


def test_quadrature_converged_on_the_fixed_path_is_not_a_quality_claim():
    # Moduł 5, przeniesione na drut: bez ``tol`` nie zamówiono dokładności, więc
    # nic jej nie mogło nie dowieźć → ``converged=True``. To NIE jest twierdzenie
    # o jakości — wąski gauss wraca „zbieżny" przy 19% błędu. Jakość mieszka
    # WYŁĄCZNIE w ``error_estimate`` i tak jest napisane w opisie narzędzia.
    payload = mcp_server.math_eval("nintegrate exp(-10000*x^2) dx from -1 to 1")
    quad = payload["quadrature"]
    assert quad["converged"] is True
    true_value = math.sqrt(math.pi) / 100.0  # ∫ e^(-10⁴x²) ≈ √π/100
    assert abs(quad["value"] - true_value) / true_value > 0.1  # zły o >10%


# --- WIRE: to, co NAPRAWDĘ leci na drut -------------------------------------
def _call(command: str) -> tuple[bool | None, dict]:
    """Zawołaj narzędzie przez PRAWDZIWĄ sesję klienta MCP i zwróć (isError, struct)."""
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    async def go():
        assert mcp_server.server is not None
        async with connect(mcp_server.server._mcp_server) as client:
            res = await client.call_tool("math_eval", {"command": command})
            return res.isError, res.structuredContent

    return asyncio.run(go())


def test_the_wire_carries_the_structure_and_the_client_validates_it():
    # Sedno modułu: payload nie tylko powstaje — PRZECHODZI walidację schematem
    # po stronie klienta (``call_tool`` sam by rzucił) i dociera jako
    # ``structuredContent``. Moduł 1 zmierzył, że to JEST kanał, który dochodzi
    # do modelu — i że na kliencie Claude Code WYPIERA blok tekstowy, dlatego
    # czytelna odpowiedź jedzie POLEM ``text``, a nie osobnym blokiem.
    is_error, struct = _call("root x^2-2 for x at 1")
    assert is_error is False
    assert struct["solve"]["converged"] is True
    assert struct["solve"]["iterations"] >= 1
    assert struct["text"] == "1.4142135623746899"


def test_the_wire_carries_a_failure_as_data_and_a_refusal_as_error_field():
    is_error, failure = _call("root x^2+1 for x at 1")
    assert is_error is False  # porażka iteracji NIE jest błędem protokołu
    assert failure["solve"]["status"] == "not_converged"
    assert failure["error"] is None

    is_error, refusal = _call("det [[1,2],[3]]")
    assert is_error is False  # bo ``isError=true`` NIE niesie structuredContent
    assert refusal["error"]["type"] == "ParseError"


def test_the_wire_survives_an_unmeasurable_residual():
    # gdyby inf poleciał jako liczba, ``call_tool`` wywaliłby CAŁE wywołanie
    # („Invalid structured content ... None is not of type 'number'") — zmierzone
    _is_error, struct = _call("root sqrt(x)+1 for x at 1")
    assert struct["solve"]["residual"] is None
    assert struct["solve"]["status"] == "not_converged"


def test_the_declared_output_schema_names_every_field_an_agent_may_branch_on():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    async def go():
        assert mcp_server.server is not None
        async with connect(mcp_server.server._mcp_server) as client:
            return (await client.list_tools()).tools

    tools = asyncio.run(go())
    tool = next(t for t in tools if t.name == "math_eval")
    schema = tool.outputSchema
    assert schema is not None
    assert set(schema["required"]) == {"text", "solve", "quadrature", "error"}
    defs = schema["$defs"]
    assert set(defs["SolveEvidence"]["required"]) == {
        "value",
        "iterations",
        "residual",
        "converged",
        "status",
    }
    assert set(defs["QuadratureEvidence"]["required"]) == {
        "value",
        "error_estimate",
        "evaluations",
        "refinements",
        "converged",
        "status",
    }
    # PRZEPISANY W MODULE 10 — przypadek (i): pinował, że odmowa deklaruje DWA pola.
    # Moduł 10 dokłada trzecie, `route`, i musi ono być WYMAGANE jak wszystkie inne
    # (pomiar 3 z modułu 6: klucza nie da się pominąć — `NotRequired` i tak jedzie
    # jako null i wywraca walidację po stronie klienta, więc „nie dotyczy" pisze się
    # jako null przy ZAWSZE obecnym kluczu).
    assert set(defs["Refusal"]["required"]) == {"type", "message", "route"}
    # ...i musi dopuszczać null, bo brak trasy jest normalnym, informatywnym stanem
    assert {t.get("type") for t in defs["Refusal"]["properties"]["route"]["anyOf"]} == {
        "string",
        "null",
    }
    # pola „niemierzalne" MUSZĄ dopuszczać null — inaczej inf zabija wywołanie
    for field, holder in (("residual", "SolveEvidence"), ("error_estimate", "QuadratureEvidence")):
        types = {t.get("type") for t in defs[holder]["properties"][field]["anyOf"]}
        assert types == {"number", "null"}


def test_the_tool_description_tells_the_model_what_to_branch_on():
    # Opis narzędzia to jedyna instrukcja, jaką model dostaje ZANIM zawoła.
    # Musi nazwać pola i powiedzieć, że porażka nie jest błędem — inaczej model
    # dalej będzie czytał prozę, choć dane leżą obok.
    doc = mcp_server.math_eval.__doc__ or ""
    for field in ("text", "solve", "quadrature", "error", "converged", "status"):
        assert field in doc
    assert "estimate, never a bound" in doc


def test_the_payload_builders_work_without_the_mcp_package():
    # Kontrakt utrzymany z rundy 2 (Blok D): brak opcjonalnej zależności nie może
    # wywracać importu. Moduł 6 dokłada warunek MOCNIEJSZY — cały payload jest
    # zwykłym dict-em i powstaje bez ``mcp``, więc jest w pełni testowalny.
    import importlib
    import sys

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setitem(sys.modules, "mcp", None)
        monkey.setitem(sys.modules, "mcp.server.fastmcp", None)
        importlib.reload(mcp_server)
        assert mcp_server.server is None
        payload = mcp_server.math_eval("root x^2-2 for x at 1")
        assert payload["solve"]["converged"] is True
        assert payload["error"] is None
    finally:
        monkey.undo()
        importlib.reload(mcp_server)


def test_evidence_types_are_the_module_3_to_5_types_not_a_copy():
    # Payload jest TŁUMACZENIEM na drut, nie drugą definicją wyniku: pola biorą
    # się z instancji ``SolveResult``/``QuadratureResult``, więc dodanie pola tam
    # nie zostawi drutu w tyle po cichu.
    answer = repl.handle_full("nintegrate 2*x dx from 0 to 1")
    assert isinstance(answer.evidence, QuadratureResult)
    wire = mcp_server._quadrature_evidence(answer.evidence)
    assert set(wire) == {f.name for f in _fields(answer.evidence)}
    answer = repl.handle_full("root x^2-2 for x at 1")
    assert isinstance(answer.evidence, SolveResult)
    wire = mcp_server._solve_evidence(answer.evidence)
    assert set(wire) == {f.name for f in _fields(answer.evidence)}


def _fields(obj):
    import dataclasses

    return dataclasses.fields(obj)


def test_module8_the_caveat_distinguishes_not_a_minimum_from_not_converged():
    # „NOT CONVERGED" byłoby dla tego wyniku NIEPRAWDĄ: bieg zbiegł — do punktu
    # stacjonarnego, który okazał się maksimum. Klient renderujący sam tekst musi
    # zobaczyć różnicę, bo podpowiedź „daj więcej iteracji" jest tu bezużyteczna.
    payload = mcp_server.math_eval("min -x^2 for x at 0")
    assert payload["error"] is None                     # to WYNIK BIEGU, nie odmowa
    assert payload["solve"]["converged"] is False
    assert payload["solve"]["status"] == "not_a_minimum"
    assert payload["solve"]["value"] == 0.0             # punkt, w którym stanął
    assert "NOT A MINIMUM" in payload["text"]
    assert "NOT CONVERGED" not in payload["text"]
    # a prawdziwy brak zbieżności nadal mówi po staremu
    assert "NOT CONVERGED" in mcp_server.math_eval("min -x^2 for x at 1")["text"]


def test_module8_tool_description_warns_that_residual_is_not_a_substitute():
    # Opis narzędzia jest jedyną rzeczą, którą agent czyta PRZED wołaniem, więc
    # to w nim musi stać ostrzeżenie, którego nie da się wyczytać z payloadu:
    # residuum bywa mikroskopijne na PORAŻCE (exp(x): brak pierwiastka, funkcja
    # tylko opadła pod tol) i spore na SUKCESIE. I że "not_a_minimum" jest jedynym
    # statusem, którego nie wolno ponawiać z większym max_iter.
    # zwijamy białe znaki: pinujemy TREŚĆ ostrzeżenia, nie miejsce łamania wiersza
    doc = " ".join((mcp_server.math_eval.__doc__ or "").split())
    assert "not_a_minimum" in doc
    assert "1e-44" in doc and "DECAYED" in doc
    assert "Do NOT retry it with more iterations" in doc
    assert "Retry from a DIFFERENT starting point" in doc


# --- moduł 10: odmowa symboliczna mówi na DRUCIE, co zrobić dalej ---------
_SYMBOLIC_REFUSALS = [
    # (id, komenda, klasa, trasa) — trasa wyliczona NIEZALEŻNIE od kodu:
    # całka nieoznaczona bez formy zamkniętej ma numeryczny odpowiednik pytania
    # (nintegrate na przedziale), równanie przestępne ma go w root, ODE w ode.
    ("integrate/no-closed-form", "integrate exp(sin(x)) dx", "NoClosedFormError", "nintegrate"),
    ("solve/no-method", "solve exp(x)+x^5-3 for x", "UnsupportedFormError", "root"),
    ("dsolve/no-closed-form", "dsolve y^2+t^2 for y(t)", "NoClosedFormError", "ode"),
]


@pytest.mark.parametrize(
    "command,cls,route",
    [(c[1], c[2], c[3]) for c in _SYMBOLIC_REFUSALS],
    ids=[c[0] for c in _SYMBOLIC_REFUSALS],
)
def test_a_symbolic_refusal_carries_its_kind_and_its_route_on_the_wire(command, cls, route):
    # SEDNO MODUŁU 10 NA DRUCIE. Przed nim wszystkie trzy przychodziły jako
    # {"type": "PycodemathError"} + proza — a lekarstwa są trzy różne. Agent ma je
    # rozróżnić Z SAMEGO PAYLOADU: klasa mówi, JAKIEGO RODZAJU jest odmowa, trasa
    # mówi, CO ZAWOŁAĆ. Żadne z tego nie wymaga czytania angielskiego.
    payload = mcp_server.math_eval(command)
    assert payload["error"]["type"] == cls
    assert payload["error"]["route"] == route
    # ...i nic się nie zmienia w regule modułu 6: odmowa nie ma wyniku biegu
    assert payload["solve"] is None and payload["quadrature"] is None


def test_the_route_is_null_when_there_is_nothing_else_to_try():
    # Druga połowa tej samej informacji, i to ta trudniejsza. Ta sama klasa co
    # całka wyżej, ale suma po przedziale nieskończonym NIE MA numerycznego
    # odpowiednika w tej gramatyce — więc `route` = null znaczy „przestań ponawiać,
    # zmień pytanie". Bez tego pola agent nie odróżni tego od przypadku, w którym
    # ponowienie ma sens.
    payload = mcp_server.math_eval("sum 1/(k^2+k+1) for k from 1 to oo")
    assert payload["error"]["type"] == "NoClosedFormError"
    assert payload["error"]["route"] is None


def test_every_route_on_the_wire_is_a_command_this_server_dispatches():
    # Trasa jest najgroźniejszym polem payloadu, bo agent pójdzie w nią bez pytania.
    # Nie da się obiecać, że komenda ZADZIAŁA — da się mechanicznie sprawdzić, że
    # ISTNIEJE. Sprawdzamy to na REALNYCH payloadach, nie na samym słowniku.
    from pycodemath.cli import repl

    for _, command, _, _ in _SYMBOLIC_REFUSALS:
        route = mcp_server.math_eval(command)["error"]["route"]
        assert route in repl._COMMANDS


def test_module10_tool_description_names_the_route_and_what_it_does_not_promise():
    # Opis narzędzia to jedyna rzecz, którą agent czyta PRZED wołaniem. Musi
    # nazwać pole, obie nowe klasy — i to, czego trasa NIE obiecuje, bo agent,
    # który przeczyta ją jako gwarancję sukcesu, zrobi gorzej niż bez niej.
    doc = " ".join((mcp_server.math_eval.__doc__ or "").split())
    assert "error.route" in doc
    assert "NoClosedFormError" in doc and "UnsupportedFormError" in doc
    assert "A route is a fact about this API, not a promise about the answer" in doc
    assert "nintegrate`` needs BOUNDS" in doc
    assert "A null route is informative in its own right" in doc
