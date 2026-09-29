"""VERIFY V4: powierzchnia — `verify` / `certify` w REPL-u i narzędzie MCP `math_verify`.

V1-V3 zbudowały werdykty (`check_equal`, `certify_*`, `check_steps`), ale nic, z
czym rozmawia człowiek albo agent, o nie nie prosiło. Ten plik pinuje, że
warstwa powierzchni TYLKO je wystawia: ten sam werdykt co wywołanie wprost,
tekst co do bajtu wspólny dla REPL-a i MCP, porażka jako pole ``error`` (nie
wyjątek) i ten sam kształt payloadu co ``math_eval`` (każdy klucz zawsze obecny,
``null`` = „nie dotyczy").

POMIAR WYJŚCIOWY (przed V4, baza 0d8de82): ``repl.handle("verify sqrt(x^2) == x")``
nie było komendą — linia szła do domyślnego uproszczenia i parser zwracał
``ParseError``; narzędzie MCP było jedno (``math_eval``).

Konwencja serii: komentarze po polsku, kod i stringi pakietu po angielsku.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from pycodemath.cli import mcp_server, repl
from pycodemath.core.errors import (
    DomainError,
    NoClosedFormError,
    ParseError,
)
from pycodemath.core.result import QuadratureResult
from pycodemath.verify import (
    CERTIFIERS,
    VerdictStatus,
    certify,
    check_equal,
    check_steps,
)
from pycodemath.verify import equal as _equal

VERIFY_FIELDS = {"text", "answer", "verdict", "steps", "error"}
VERDICT_FIELDS = {"status", "method", "counterexample", "detail"}


# --- REPL: verify <a> == <b> --------------------------------------------------
@pytest.mark.parametrize(
    "a,b,status",
    [
        ("sqrt(x^2)", "x", VerdictStatus.REFUTED),
        ("log(x^2)", "2*log(x)", VerdictStatus.REFUTED),
        ("sin(x)^2 + cos(x)^2", "1", VerdictStatus.VERIFIED),
        ("(x^2-1)/(x-1)", "x+1", VerdictStatus.VERIFIED),
        ("0.1 + 0.2", "0.3", VerdictStatus.VERIFIED),
    ],
)
def test_verify_equal_is_check_equal_unchanged(a, b, status):
    # Powierzchnia NIE decyduje — werdykt ma być dokładnie tym, co zwraca V1
    # wywołany wprost (ten sam obiekt co do pola, bo dataclass jest frozen/eq).
    checked = repl.handle_checked(f"verify {a} == {b}")
    assert checked.verdict == check_equal(a, b)
    assert checked.verdict.status is status
    assert checked.steps is None and checked.answer is None


def test_verify_equal_text_names_the_word_the_method_and_the_point():
    assert repl.handle("verify sqrt(x^2) == x") == (
        "REFUTED (numeric-sampling) — at x = -1: left = 1.0, right = -1.0"
    )
    assert repl.handle("verify sin(x)^2 + cos(x)^2 == 1").startswith(
        "VERIFIED (symbolic) — "
    )


def test_verify_is_a_command_now_and_was_not_before():
    # przed V4 ta linia była wyrażeniem dla parsera (i odmową) — teraz jest komendą
    assert "verify" in repl._COMMANDS and "certify" in repl._COMMANDS
    help_text = repl.handle("help")
    assert "verify <a> == <b>" in help_text
    assert "verify steps" in help_text
    assert "certify <command>" in help_text


@pytest.mark.parametrize(
    "line",
    ["verify", "verify x", "verify a == b == c", "verify steps", "verify == 1"],
)
def test_a_line_that_does_not_fit_earns_the_usage_message(line):
    # „a == b == c" to łańcuch — derywacja, nie jedno twierdzenie; odczytanie go
    # jako jednej równości po cichu zgubiłoby krok. Stąd usage wskazujące steps.
    out = repl.handle(line)
    assert out.startswith("Usage: verify <a> == <b>")
    assert "verify steps" in out


def test_budget_is_matched_by_name_so_juxtaposed_names_stay_an_expression():
    # Parser czyta ``x y z`` jako iloczyn. Ogólny wzorzec opcji (``_OPTS``,
    # „klucz wartość" na końcu) wziąłby tu ``y z`` za opcję ``y`` = ``z`` i
    # odmówił ParseError-em. Dlatego verify rozpoznaje TYLKO ``budget``.
    assert repl.handle("verify x*y*z == x y z").startswith("VERIFIED")
    assert repl._OPTS not in repl._COMMANDS["verify"][0].pattern


def test_budget_reaches_check_equal_and_a_bad_budget_is_refused(monkeypatch):
    seen = {}
    real = repl.check_equal

    def spy(a, b, **kw):
        seen.update(kw)
        return real(a, b, **kw)

    monkeypatch.setattr(repl, "check_equal", spy)
    repl.handle("verify sin(x)^2 == 1 - cos(x)^2 budget 5")
    assert seen == {"budget": 5.0}
    with pytest.raises(DomainError):
        repl.handle("verify a == b budget 0")
    with pytest.raises(DomainError):
        repl.handle("verify a == b budget inf")
    with pytest.raises(ParseError):
        repl.handle("verify a == b budget x")


def test_running_out_of_budget_is_an_undecided_verdict_not_an_error():
    checked = repl.handle_checked("verify sin(x)^2 == 1 - cos(x)^2 budget 0.000001")
    assert checked.verdict is not None
    assert checked.verdict.status is VerdictStatus.UNDECIDED
    assert checked.verdict.method == "time-budget"


# --- REPL: verify steps ---------------------------------------------------------
_PLAN_EXAMPLE = "(x+1)^2 - (x-1)^2\n= x^2 + 2x + 1 - x^2 + 2x + 1\n= 4x + 2"


@pytest.mark.parametrize(
    "derivation",
    [
        _PLAN_EXAMPLE,
        "(x+1)^2 - (x-1)^2 = x^2 + 2x + 1 - x^2 + 2x + 1 = 4x + 2",
    ],
    ids=["multi-line", "one-line"],
)
def test_verify_steps_is_check_steps_unchanged(derivation):
    # przykład z PLAN_VERIFY §2 V3: krok 2 REFUTED, kontrprzykład x = 0 (0 vs 2)
    checked = repl.handle_checked(f"verify steps {derivation}")
    assert checked.steps == check_steps(derivation)
    assert checked.steps.first_error == 2
    assert checked.steps.counterexample == {"x": "0"}
    assert checked.verdict is None


def test_verify_steps_text_one_line_per_step_plus_the_derivation():
    out = repl.handle("verify steps " + _PLAN_EXAMPLE)
    assert out.splitlines() == [
        "step 2: REFUTED (numeric-sampling) — at x = 0: left = 0.0, right = 2.0",
        "step 3: VERIFIED (symbolic) — the two sides are identical after "
        "automatic simplification",
        "derivation: REFUTED — first wrong step: 2 (counterexample x = 0)",
    ]


def test_verify_steps_equations_carry_the_warning_line():
    out = repl.handle("verify steps sqrt(x) = x - 2 -> x = (x-2)^2")
    assert "  warning: step 2 gains the root x = 1" in out
    assert out.endswith("derivation: REFUTED — first wrong step: 2 (counterexample x = 1)")
    ok = repl.handle("verify steps 2x + 3 = 7 -> 2x = 4 -> x = 2")
    assert ok.endswith("derivation: VERIFIED — all 2 steps proved")


def test_verify_steps_undecided_summary_counts_the_open_steps():
    # granica z V3 (znalezisko 3): krok przestępny z ConditionSet zostaje UNDECIDED
    out = repl.handle("verify steps exp(x) = x + 2 -> exp(x) = x + 3")
    assert out.endswith("derivation: UNDECIDED — no step refuted, 1 of 1 not proved")


def test_verify_steps_refusals_are_check_steps_refusals():
    with pytest.raises(DomainError, match="inequality"):
        repl.handle("verify steps x < 2 -> x < 3")


def test_interactive_verify_steps_reads_until_an_empty_line():
    lines = iter(["(x+1)^2 - (x-1)^2", "= x^2 + 2x + 1 - x^2 + 2x + 1", "= 4x + 2", ""])
    prompts = []

    def read(prompt):
        prompts.append(prompt)
        return next(lines)

    command = repl._read_steps("verify steps", read)
    assert command == "verify steps\n" + _PLAN_EXAMPLE
    assert prompts == ["...> "] * 4
    # i trafia do check_steps tak samo jak forma one-shot
    assert repl.handle(command) == repl.handle("verify steps " + _PLAN_EXAMPLE)


def test_interactive_verify_steps_stops_at_end_of_input_and_leaves_other_lines():
    def eof(_prompt):
        raise EOFError

    assert repl._read_steps("verify steps", eof) == "verify steps"
    assert repl._read_steps("verify a == b", eof) == "verify a == b"


# --- REPL: certify <command> ----------------------------------------------------
#: Jedna poprawna odpowiedź silnika na każdy certyfikat z V2 (komenda REPL-a,
#: wywołanie ``certify`` wprost z tymi samymi argumentami i wynikiem silnika).
_CERTIFIED = [
    ("integrate x*cos(x) dx", "integrate", ("x*cos(x)", "x", "x*sin(x) + cos(x)"), {}),
    ("diff sin(x)*x dx", "diff", ("sin(x)*x", "x", "x*cos(x) + sin(x)"), {}),
    ("solve x^2 - 4 for x", "solve", ("x^2 - 4", "x", ["-2", "2"]), {"real": True}),
    ("limit sin(x)/x for x to 0", "limit", ("sin(x)/x", "x", "0", "1"), {}),
    ("dsolve y for y(t)", "dsolve", ("y", "y", "t", ["C1*exp(t)"]), {}),
]


@pytest.mark.parametrize(
    "command,op,args,kwargs", _CERTIFIED, ids=[c[1] for c in _CERTIFIED]
)
def test_certify_runs_the_command_and_certifies_its_answer(command, op, args, kwargs):
    checked = repl.handle_checked(f"certify {command}")
    # odpowiedź co do bajtu ta sama, co bez certyfikatu
    assert checked.answer == repl.handle(command)
    # certyfikat = V2 wywołany wprost na tej samej odpowiedzi
    assert checked.verdict == certify(op, *args, **kwargs)
    assert checked.verdict.status is VerdictStatus.VERIFIED
    assert checked.text == f"{checked.answer}\ncertificate: " + repl._verdict_line(
        checked.verdict
    )


def test_every_v2_certificate_is_reachable_from_the_grammar():
    assert set(repl._CERTIFY) == set(CERTIFIERS)


def test_certify_nintegrate_checks_the_runs_own_error_estimate(monkeypatch):
    seen = {}
    real = repl.certify_nintegrate

    def spy(expr, var, a, b, claim, **kw):
        seen["claim"] = claim
        return real(expr, var, a, b, claim, **kw)

    monkeypatch.setattr(repl, "certify_nintegrate", spy)
    checked = repl.handle_checked("certify nintegrate 2*x dx from 0 to 1")
    assert isinstance(seen["claim"], QuadratureResult)
    assert checked.answer == repl.handle("nintegrate 2*x dx from 0 to 1")
    assert checked.verdict.status is VerdictStatus.VERIFIED
    # opcja tol należy do komendy nintegrate i przechodzi (nie jest odrzucana)
    tight = repl.handle_checked("certify nintegrate 1/x dx from 1 to 2 tol 1e-12")
    assert tight.verdict.status is VerdictStatus.VERIFIED


def test_the_commands_budget_bounds_the_certificate_separately(monkeypatch):
    # Budżet komendy dostaje też certyfikat — ale OSOBNO, nie zagnieżdżony w
    # ``time_budget`` komendy: wygaśnięcie tam odmówiłoby całej linii i zgubiło
    # policzoną już odpowiedź. Spy sprawdza obie połowy.
    seen = {}
    real = repl.certify_integrate

    def spy(expr, var, result, **kw):
        seen["budget"] = kw.get("budget")
        seen["nested"] = _equal._caller_budget_active()
        return real(expr, var, result, **kw)

    monkeypatch.setattr(repl, "certify_integrate", spy)
    repl.handle("certify integrate 2*x dx budget 5")
    assert seen == {"budget": 5.0, "nested": False}
    repl.handle("certify integrate 2*x dx")
    assert seen["budget"] is None  # → domyślny budżet certyfikatu


def test_certify_refusals():
    with pytest.raises(DomainError, match="no certificate for 'det'"):
        repl.handle("certify det [[1,2],[3,4]]")
    # odmowa SILNIKA przechodzi bez zmian, z trasą (moduł 10)
    with pytest.raises(NoClosedFormError) as info:
        repl.handle("certify integrate exp(sin(x)) dx")
    assert info.value.route == "nintegrate"
    # składnia komendy wewnętrznej nie pasuje → jej usage, z prefiksem certify
    assert repl.handle("certify integrate x") == (
        "Usage: certify integrate <expression> d<variable> [budget <s>]"
        "   (e.g. certify integrate 2*x dx)"
    )
    assert repl.handle("certify").startswith("Usage: certify <command>")


# --- dwa dyspozytory, jeden tekst --------------------------------------------------
_LINES = [
    "verify sqrt(x^2) == x",
    "verify steps 2x + 3 = 7 -> 2x = 4 -> x = 2",
    "certify integrate 2*x dx",
    "verify x",
]


@pytest.mark.parametrize("line", _LINES)
def test_all_dispatchers_print_the_same_text(line):
    text = repl.handle(line)
    assert repl.handle_checked(line).text == text
    assert repl.handle_full(line).text == text  # math_eval też to umie — jako tekst


def test_handle_checked_refuses_a_line_that_does_not_check_anything():
    with pytest.raises(ParseError, match="not a verify/certify line"):
        repl.handle_checked("diff x^2 dx")


# --- MCP: payload -----------------------------------------------------------------
_PAYLOADS = [
    ("equal", "sqrt(x^2) == x", "verdict"),
    ("steps", "steps 2x + 3 = 7 -> 2x = 4 -> x = 2", "steps"),
    ("certify", "certify integrate 2*x dx", "verdict"),
    ("usage", "x", None),
]


@pytest.mark.parametrize(
    "command,carrier", [(p[1], p[2]) for p in _PAYLOADS], ids=[p[0] for p in _PAYLOADS]
)
def test_every_math_verify_response_has_the_same_five_fields(command, carrier):
    payload = mcp_server.math_verify(command)
    assert set(payload) == VERIFY_FIELDS
    assert payload["error"] is None
    assert isinstance(payload["text"], str) and payload["text"]
    for field in ("verdict", "steps"):
        assert (payload[field] is not None) == (field == carrier)
    assert (payload["answer"] is not None) == command.startswith("certify")
    # nic, czego JSON nie wyrazi (inf/nan) — liczników brak, kontrprzykład to tekst
    json.dumps(payload, allow_nan=False)
    assert "ok" not in payload and "success" not in payload


def test_the_verdict_on_the_wire_is_the_verdict_not_a_copy():
    payload = mcp_server.math_verify("sqrt(x^2) == x")
    v = check_equal("sqrt(x^2)", "x")
    assert payload["verdict"] == {
        "status": v.status.value,
        "method": v.method,
        "counterexample": v.counterexample,
        "detail": v.detail,
    }
    assert payload["text"] == repl.handle("verify sqrt(x^2) == x")


def test_the_steps_payload_carries_every_step_and_the_derivation():
    payload = mcp_server.math_verify("steps sqrt(x) = x - 2 -> x = (x-2)^2 -> x^2 - 5x + 4 = 0")
    steps = payload["steps"]
    assert steps["kind"] == "equations"
    assert steps["status"] == "refuted"
    assert steps["first_error"] == 2
    assert steps["counterexample"] == {"x": "1"}
    assert [c["number"] for c in steps["checks"]] == [2, 3]
    first, second = steps["checks"]
    assert first["effect"] == "gains-roots" and "squaring" in first["warning"]
    assert first["verdict"]["status"] == "refuted"
    assert second["verdict"]["status"] == "verified"
    assert second["effect"] is None and second["warning"] is None
    assert set(first["verdict"]) == VERDICT_FIELDS


def test_the_leading_verify_word_is_optional():
    assert mcp_server.math_verify("verify sqrt(x^2) == x") == mcp_server.math_verify(
        "sqrt(x^2) == x"
    )
    assert mcp_server.math_verify("  certify diff x^2 dx ")["answer"] == "2*x"


@pytest.mark.parametrize(
    "command,cls,route",
    [
        ("x^^2 == 1", "ParseError", None),
        ("steps x < 2 -> x < 3", "DomainError", None),
        ("certify det [[1]]", "DomainError", None),
        ("certify integrate exp(sin(x)) dx", "NoClosedFormError", "nintegrate"),
        ("sin(x) == cos(x) budget 0", "DomainError", None),
    ],
)
def test_a_refusal_is_the_error_field_never_an_exception(command, cls, route):
    payload = mcp_server.math_verify(command)
    assert set(payload) == VERIFY_FIELDS
    assert payload["error"] == {
        "type": cls,
        "message": payload["error"]["message"],
        "route": route,
    }
    assert payload["text"] == f"error: {payload['error']['message']}"
    assert payload["verdict"] is None and payload["steps"] is None
    assert payload["answer"] is None


def test_both_tools_build_the_refusal_object_the_same_way():
    exc = NoClosedFormError("integrate: no closed form", route="nintegrate")
    assert mcp_server._refusal(exc)["error"] == mcp_server._verify_refusal(exc)["error"]


def test_the_tool_description_names_the_fields_and_the_three_words():
    doc = mcp_server.math_verify.__doc__ or ""
    for word in (*VERIFY_FIELDS, "verified", "refuted", "undecided", "first_error",
                 "counterexample", "gains-roots", "time-budget"):
        assert word in doc, word
    # math_eval odsyła do drugiego narzędzia
    assert "math_verify" in (mcp_server.math_eval.__doc__ or "")


def test_the_new_types_keep_one_line_docstrings_because_they_travel():
    # komentarz modułu: docstringi TypedDict są serializowane do outputSchema
    for td in (
        mcp_server.VerdictEvidence,
        mcp_server.StepEvidence,
        mcp_server.StepsEvidence,
        mcp_server.VerifyResult,
    ):
        assert td.__doc__ and "\n" not in td.__doc__.strip()


# --- MCP: to, co NAPRAWDĘ leci na drut ----------------------------------------------
def _session(fn):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    async def go():
        assert mcp_server.server is not None
        async with connect(mcp_server.server._mcp_server) as client:
            return await fn(client)

    return asyncio.run(go())


def _call(command: str):
    async def call(client):
        res = await client.call_tool("math_verify", {"command": command})
        return res.isError, res.structuredContent

    return _session(call)


@pytest.mark.parametrize(
    "command",
    [
        "sqrt(x^2) == x",
        "pi == 3.14159",  # kontrprzykład {} — pusty obiekt musi przejść walidację
        "steps sqrt(x) = x - 2 -> x = (x-2)^2",
        "certify solve x^2 - 4 for x",
        "certify integrate exp(sin(x)) dx",
        "x",
    ],
)
def test_the_wire_carries_the_verdict_and_the_client_validates_it(command):
    is_error, struct = _call(command)
    assert is_error is False  # odmowa też: isError=true nie niesie struktury
    assert struct == mcp_server.math_verify(command)


def test_the_server_lists_both_tools_with_the_declared_schema():
    async def tools(client):
        return (await client.list_tools()).tools

    listed = {t.name: t for t in _session(tools)}
    assert {"math_eval", "math_verify"} <= set(listed)
    schema = listed["math_verify"].outputSchema
    assert schema is not None
    assert set(schema["required"]) == VERIFY_FIELDS
    defs = schema["$defs"]
    assert set(defs["VerdictEvidence"]["required"]) == VERDICT_FIELDS
    assert set(defs["StepsEvidence"]["required"]) == {
        "kind", "status", "first_error", "counterexample", "checks",
    }
    assert set(defs["StepEvidence"]["required"]) == {
        "number", "text", "verdict", "effect", "warning",
    }
    assert set(defs["Refusal"]["required"]) == {"type", "message", "route"}


def test_math_verify_works_without_the_mcp_package():
    import importlib
    import sys

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setitem(sys.modules, "mcp", None)
        monkey.setitem(sys.modules, "mcp.server.fastmcp", None)
        importlib.reload(mcp_server)
        assert mcp_server.server is None
        assert mcp_server.math_verify("sqrt(x^2) == x")["verdict"]["status"] == "refuted"
    finally:
        monkey.undo()
        importlib.reload(mcp_server)
