"""I1 — dziewięć modułów serii FULL jako JEDEN produkt.

Strażnicy (z promptu I1), napisani tak, żeby na bazie ``nextgen/plan-full`` (przed
podpięciem) były CZERWONE — wynik w ``dev-journal/full/I1.md``:

* ``test_code_pillar_end_to_end`` — plik z funkcją z błędem off-by-one ->
  ``verify code`` -> REFUTED z wywołaniem -> ``--emit-test`` daje test, który
  naprawdę pada w pytest -> certyfikat -> ``recheck`` odtwarza;
* ``test_every_public_feature_is_reachable_everywhere`` — dla każdego modułu z
  tabeli podpięcia: nazwa w ``__all__``, polecenie w ``help`` REPL, narzędzie albo
  trasa MCP, przykład w obu README.

Reszta pliku pinuje kontrakty warstwy integracji: każde nowe polecenie ma wpis w
``help`` i działający przykład w swoim usage; każdy ``kind`` certyfikatu przeżywa
wydanie -> JSON -> ``recheck`` i łapie podmienioną liczbę; MCP ma stały zestaw pól,
nie wykonuje cudzego kodu bez zgody, odmawia drogich drzew i budżetów ponad limit.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import pycodemath
import pycodemath.code
from pycodemath.cli import full, mcp_server, repl
from pycodemath.core.errors import DomainError, ParseError
from pycodemath.frontend.parser import parse_mode
from pycodemath.verify import certificate as C

_ROOT = Path(__file__).resolve().parent.parent
READMES = (_ROOT / "README.md", _ROOT / "README.pl.md")

OFF_BY_ONE = textwrap.dedent(
    """
    def sum_to(n: int) -> int:
        \"\"\"1 + 2 + ... + n — with the classic off-by-one: range(n) stops at n-1.\"\"\"
        total = 0
        for k in range(n):
            total += k
        return total
    """
)


# --- strażnik 1: filar kodu od pliku do certyfikatu ------------------------------
def test_code_pillar_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "sums.py"
    src.write_text(OFF_BY_ONE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    out = repl.handle("verify code sums.py:sum_to == n*(n+1)/2 --emit-test")
    first = out.splitlines()[0]
    assert first.startswith("REFUTED (execution)"), out
    assert "sum_to(n=1) returns 0" in first  # the call that was really executed

    # --emit-test: the printed pytest pins the counterexample and FAILS on this code
    test_src = out[out.index('"""Regression test'):]
    (tmp_path / "test_pinned.py").write_text(test_src, encoding="utf-8")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                          "test_pinned.py"], cwd=tmp_path, capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 1 and "1 failed" in run.stdout, run.stdout + run.stderr

    # the same check, written down, re-checks from the file alone
    cert_json = repl.handle("certificate verify code sums.py:sum_to == n*(n+1)/2")
    cert = C.Certificate.from_json(cert_json)
    assert cert.kind == "code_equiv"
    assert cert.verdict["status"] == "refuted" and cert.verdict["counterexample"] == {"n": "1"}
    (tmp_path / "c.json").write_text(cert_json, encoding="utf-8")
    assert repl.handle("recheck c.json").startswith("REPRODUCED — code_equiv")

    # the corrected function is VERIFIED by a proof, never by agreement on samples
    fixed = OFF_BY_ONE.replace("range(n)", "range(n + 1)")
    src.write_text(fixed, encoding="utf-8")
    ok = repl.handle("verify code sums.py:sum_to == n*(n+1)/2 where n: integer, n>=0")
    assert ok.startswith("VERIFIED"), ok


# --- strażnik 2: tabela podpięcia ------------------------------------------------
#: (moduł, nazwa publicznego API, (moduł z __all__), słowo polecenia w help REPL,
#:  narzędzie MCP albo linia math_eval, marker README EN, marker README PL)
SURFACES = [
    ("X1", "recheck", pycodemath, "recheck", "math_recheck", "math_recheck", "math_recheck"),
    ("K1", "parse", pycodemath, "implicit", "math_eval", "implicit on", "implicit on"),
    ("C1", "check_code", pycodemath, "verify code", "math_check_code", "math_check_code", "math_check_code"),
    ("C2", "diagnose", pycodemath, "stability", "math_stability", "math_stability", "math_stability"),
    ("C3", "analyze", pycodemath.code, "ranges", "math_ranges", "math_ranges", "math_ranges"),
    ("C4", "allocate", pycodemath, "money split", "money split 100 PLN into 3",
     "money split 100 PLN into 3", "money split 100 PLN into 3"),
    ("K3", "calc", pycodemath, "factor", "factor 2^64+1", "factor 2^64+1", "factor 2^64+1"),
    ("J1", "convert", pycodemath, "convert", "convert 5 km/h to m/s", "check_dimensions", "check_dimensions"),
    ("G1", "compare_rotation_expressions", pycodemath, "rotcheck", "rotcheck q1*q2 == R2@R1",
     "rotcheck", "rotcheck"),
    # I2 — wave 2
    ("K2", "identify", pycodemath, "identify", "evalf sqrt(2) 50",
     "identify 1.6449340668482264", "identify 1.6449340668482264"),
    ("K5", "definite_integral", pycodemath, "integrate <expr> d<var> from",
     "integrate 1/x^2 dx from -1 to 1", "integrate 1/x^2 dx from -1 to 1",
     "integrate 1/x^2 dx from -1 to 1"),
    ("C5", "count_iterations", pycodemath.code, "iterations", "math_iterations",
     "How many times does this loop run?", "Ile razy wykona się ta pętla?"),
    ("L1", "lint_source", pycodemath, "lint", "math_lint",
     "Lint the math in your repo", "Sprawdź matematykę w całym repo"),
]
_TOOL_NAMES = {t.__name__ for t in mcp_server.TOOLS}


@pytest.mark.parametrize("module,api,package,command,mcp,readme_en,readme_pl", SURFACES,
                         ids=[s[0] for s in SURFACES])
def test_every_public_feature_is_reachable_everywhere(module, api, package, command, mcp,
                                                      readme_en, readme_pl) -> None:
    # 1. API: the name is public and imports
    assert api in package.__all__, f"{module}: {api} not in {package.__name__}.__all__"
    assert getattr(package, api) is not None
    # 2. REPL/CLI: the command is in help
    assert command in repl.handle("help"), f"{module}: '{command}' missing from help"
    # 3. MCP: a tool of its own, or a math_eval line answered with a certificate
    if mcp in _TOOL_NAMES:
        assert (getattr(mcp_server, mcp).__doc__ or "").strip()
    else:
        payload = mcp_server.math_eval(mcp)
        assert payload["error"] is None, payload["error"]
        assert payload["certificate"] is not None and payload["result"] is not None
    # 4. README: both languages document it
    for path, marker in zip(READMES, (readme_en, readme_pl)):
        assert marker in path.read_text(encoding="utf-8"), f"{module}: {marker!r} not in {path.name}"


# --- re-eksporty ----------------------------------------------------------------------
@pytest.mark.parametrize("package", [pycodemath, pycodemath.code], ids=lambda p: p.__name__)
def test_every_name_in_all_imports(package) -> None:
    missing = [n for n in package.__all__ if getattr(package, n, None) is None]
    assert not missing
    assert len(set(package.__all__)) == len(package.__all__)


def test_the_full_names_load_lazily() -> None:
    # ``import pycodemath`` must not import the NumPy-heavy modules (measured: +80 ms)
    code = ("import sys, pycodemath; "
            "print(any(m.split('.')[:2] == ['pycodemath', 'code'] for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "False", out.stderr


# --- każde polecenie: wpis w help + działający przykład --------------------------------
_EXAMPLE = re.compile(r"\(e\.g\. (.+?)\)\s*$", re.MULTILINE)


def _example(usage: str) -> str:
    found = _EXAMPLE.findall(usage)
    assert found, f"usage without an example: {usage!r}"
    return found[0]


@pytest.mark.parametrize("word", sorted(full.COMMANDS))
def test_every_full_command_has_help_and_a_working_example(word: str) -> None:
    pattern, handler, usage = full.COMMANDS[word]
    assert re.search(rf"(^|[\s|]){re.escape(word)}([\s|]|$)", repl.handle("help"), re.M), word
    example = _example(usage)
    match = pattern.match(example)
    assert match is not None, f"{word}: its own usage example does not parse: {example!r}"
    outcome = handler(match)
    assert outcome.text and not outcome.text.startswith("Usage"), outcome.text
    assert outcome.result is not None
    # the certificate line closes every answer (rule 4 of ``full``)
    assert outcome.text.splitlines()[-1].startswith("certificate:"), outcome.text


def test_a_command_line_that_does_not_fit_gets_its_usage() -> None:
    assert repl.handle("money").startswith("Usage: money")
    assert repl.handle("money split 100 PLN").startswith("Usage: money round")
    assert repl.handle("ranges x").startswith("Usage: ranges")


def test_an_expression_that_opens_with_a_command_word_is_still_an_expression() -> None:
    assert repl.handle("mean + 1") == "mean + 1"
    assert repl.handle("power*2") == "2*power"


def test_unknown_command_suggests_the_new_ones() -> None:
    with pytest.raises(ParseError, match="did you mean 'factor'"):
        repl.handle("facter 7")
    with pytest.raises(ParseError, match="did you mean 'stability'"):
        repl.handle("stabilty x for x in [0, 1]")


def test_a_bare_conversion_reads_as_units_only_when_both_sides_are_units() -> None:
    assert repl.handle("5 km/h to m/s").startswith("25/18 m/s")
    with pytest.raises(ParseError):  # K1's refusal stands when it is not units
        repl.handle("5 apples to pears")


def test_ranges_names_its_dtype() -> None:
    with pytest.raises(ParseError, match="as float32"):
        repl.handle("ranges x*x for x in [0, 1]")


# --- certyfikaty: każdy nowy kind -----------------------------------------------------
def _issued() -> "list[tuple[str, C.Certificate, str, str]]":
    """(kind, certificate, a recorded number, what to change it to)."""
    out = []
    _, cert = C.certify_op("calc", "factorint", {"args": ["2^64+1"]})
    out.append(("calc", cert, "274177", "274179"))
    _, cert = C.certify_op("money", "allocate", {"total": "100", "parts": 3, "currency": "PLN"})
    out.append(("money", cert, "33.34", "33.35"))
    _, cert = C.certify_op("geometry", "orient2d", {"points": [["0", "0"], ["1", "0"], ["0", "1"]]})
    out.append(("geometry", cert, '\\"sign\\":1', '\\"sign\\":-1'))
    cert = C.issue("unit_conversion", "5 km/h", "m/s", "25/18")
    out.append(("unit_conversion", cert, "25/18", "25/17"))
    cert = C.issue("dimensions", "F = m*a", "F:N m:kg a:m/s^2")
    out.append(("dimensions", cert, "m/s^2", "m/s^3"))
    cert = C.issue("code_equiv", OFF_BY_ONE, "n*(n+1)/2", "sum_to", "")
    out.append(("code_equiv", cert, '"n": "1"', '"n": "2"'))
    from pycodemath.code import stability as S
    args = C.stability_args(S.diagnose("sqrt(x + 1) - sqrt(x)", {"x": (1e10, 1e15)}))
    assert args is not None
    cert = C.issue("stability", *args)
    out.append(("stability", cert, args[3][:6], args[3][:5] + "9"))
    from pycodemath.code import ranges as R
    spec = {"lo": [0, 2**31 - 1], "hi": [0, 2**31 - 1]}
    rep = R.analyze("(lo + hi) // 2", spec, "int32")
    cert = C.from_verdict("range", ["(lo + hi) // 2", C._jtext(spec), "int32"], C.range_verdict(rep))
    out.append(("range", cert, "2147483647", "2147483646"))
    return out


@pytest.fixture(scope="module")
def issued() -> "list[tuple[str, C.Certificate, str, str]]":
    return _issued()


_NEW_KINDS = ["calc", "money", "geometry", "unit_conversion", "dimensions", "code_equiv",
              "stability", "range"]


@pytest.mark.parametrize("kind", _NEW_KINDS)
def test_every_new_kind_round_trips_and_rechecks(issued, kind) -> None:
    cert = next(c for k, c, _, _ in issued if k == kind)
    text = cert.to_json()
    again = C.Certificate.from_json(text)
    assert again.to_json() == text and again.digest_ok
    r = C.recheck(text)
    assert r.ok, r.text()


@pytest.mark.parametrize("kind", _NEW_KINDS)
def test_every_new_kind_catches_one_changed_number(issued, kind) -> None:
    cert, old, new = next((c, o, n) for k, c, o, n in issued if k == kind)
    text = cert.to_json()
    assert old in text, (kind, old)
    tampered = text.replace(old, new, 1)
    r = C.recheck(tampered)
    assert not r.ok and not r.digest_ok
    # with the digest re-sealed (a forger's next step), the RE-RUN still catches it
    data = json.loads(tampered)
    data["digest"] = C.content_digest(data)
    r = C.recheck(data)
    assert r.digest_ok and not r.ok, (kind, r.text())


def test_a_tampered_result_is_refuted_by_recomputation() -> None:
    value, cert = C.certify_op("calc", "gcd", {"args": ["1071", "462"]})
    data = cert.to_dict()
    data["claim"]["args"]["result"] = data["claim"]["args"]["result"].replace('"value":21', '"value":7')
    data["digest"] = C.content_digest(data)
    r = C.recheck(data)
    assert r.verdict["status"] == "refuted" and r.verdict["method"] == "recomputation"


def test_the_parse_mode_is_part_of_the_certificate() -> None:
    # X1 review note 1: a certificate issued under ``implicit on`` used to re-check as
    # "claim parses differently" — the mode was not recorded
    with parse_mode(strict=False):
        cert = C.issue("identity", "2xy", "2*x*y")
    assert cert.claim["options"]["strict"] is False
    assert cert.verdict["status"] == "verified"
    r = C.recheck(cert)  # re-checked in the default (strict) context
    assert r.ok, r.text()
    strict = C.issue("identity", "xy", "x*y")
    assert strict.claim["options"]["strict"] is True
    assert strict.verdict["status"] == "refuted"


def test_a_costly_sympy_tree_is_refused_on_reading() -> None:
    # X1 review note 2: Pow(10, Pow(10, 999)) built fine and hung once evaluated
    cert = C.issue("identity", "x", "x")
    data = cert.to_dict()
    data["claim"]["args"]["right"] = {"sympy": "Pow(Integer(10), Pow(Integer(10), Integer(999)))"}
    with pytest.raises(DomainError, match="too costly"):
        C.recheck(data)
    deep = "Symbol('x')"
    for _ in range(150):
        deep = f"Add({deep}, Integer(1))"
    data["claim"]["args"]["right"] = {"sympy": deep}
    with pytest.raises(DomainError, match="nested deeper"):
        C.recheck(data)


# --- MCP ------------------------------------------------------------------------------
CHECK_FIELDS = {"text", "kind", "status", "result", "certificate", "error"}


def test_mcp_check_code_does_not_run_outside_code_by_default() -> None:
    p = mcp_server.math_check_code(OFF_BY_ONE, "n*(n+1)/2")
    assert set(p) == CHECK_FIELDS and p["error"] is None
    assert p["status"] == "undecided"  # a refutation would need running the code
    assert any(n["code"] == "not_executed" for n in p["result"]["notes"])
    assert p["result"]["executed_calls"] == 0
    # with consent it is executed (isolated worker) and refuted with the call
    p = mcp_server.math_check_code(OFF_BY_ONE, "n*(n+1)/2", execute=True, emit_test=True)
    assert p["status"] == "refuted" and p["result"]["counterexample"]["call"] == "sum_to(n=1)"
    assert "assert sum_to(n=1) == 1" in p["result"]["test_file"]
    # a proof needs no execution at all
    p = mcp_server.math_check_code(OFF_BY_ONE, "n*(n-1)/2", domain={"n": "integer, n>=0"})
    assert p["status"] == "verified" and p["result"]["executed_calls"] == 0


def test_mcp_check_code_never_executes_a_writer(tmp_path: Path) -> None:
    # numpy is on C1's safe list, and np.save writes files: with execute off, nothing runs
    target = tmp_path / "pwned.npy"
    src = textwrap.dedent(f"""
        import numpy as np
        def f(x):
            np.save({str(target)!r}, np.zeros(1))
            return x + 1
    """)
    p = mcp_server.math_check_code(src, "x + 2")
    assert p["error"] is None and p["status"] == "undecided"
    assert not target.exists()


def test_mcp_recheck_round_trip_and_refusals() -> None:
    p = mcp_server.math_eval("factor 2^64+1")
    text = json.dumps(p["certificate"])
    r = mcp_server.math_recheck(text)
    assert set(r) == CHECK_FIELDS and r["status"] == "verified" and r["result"]["ok"]
    bad = text.replace("274177", "274179")
    r = mcp_server.math_recheck(bad)
    assert r["status"] == "refuted" and not r["result"]["digest_ok"]
    r = mcp_server.math_recheck("x" * (mcp_server.MAX_CERTIFICATE_TEXT + 1))
    assert r["error"]["type"] == "DomainError"
    code = mcp_server.math_check_code(OFF_BY_ONE, "n*(n+1)/2", execute=True)
    r = mcp_server.math_recheck(json.dumps(code["certificate"]))
    assert r["error"] is not None and "execute=true" in r["error"]["message"]


def test_mcp_typed_tools_refuse_budgets_over_the_limit() -> None:
    for payload in (
        mcp_server.math_stability("x", {"x": [0, 1]}, budget=1000),
        mcp_server.math_ranges("x", {"x": [0, 1]}, budget=-1),
        mcp_server.math_check_code("def f(x):\n    return x\n", "x", budget=float("inf")),
    ):
        assert set(payload) == CHECK_FIELDS
        assert payload["error"]["type"] == "DomainError"


def test_mcp_stability_and_ranges_answer_with_witnesses() -> None:
    p = mcp_server.math_stability("sqrt(x + 1) - sqrt(x)", {"x": [1e10, 1e15]})
    assert p["status"] == "refuted" and p["certificate"]["verdict"]["counterexample"]
    p = mcp_server.math_stability("sqrt(x + 1) - sqrt(x)", {"x": [1e10, 1e15]}, improve=True)
    assert p["status"] == "verified" and p["kind"] == "identity"
    assert p["result"]["best"]["expression"] == "1 / (sqrt(x + 1) + sqrt(x))"
    p = mcp_server.math_ranges("(lo + hi) // 2", {"lo": [0, 2**31 - 1], "hi": [0, 2**31 - 1]}, "int32")
    assert p["status"] == "refuted"
    assert p["certificate"]["verdict"]["counterexample"] == {"hi": "2147483647", "lo": "2147483647"}


def test_the_numeric_lock_serializes_c2_and_c3() -> None:
    import threading

    assert isinstance(full.NUMERIC_LOCK, type(threading.RLock()))
    held = []

    def probe() -> None:
        held.append(full.NUMERIC_LOCK.acquire(blocking=False))

    with full.NUMERIC_LOCK:
        t = threading.Thread(target=probe)
        t.start()
        t.join()
    assert held == [False]


def test_the_tools_are_registered_with_their_output_schemas() -> None:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    async def go():
        assert mcp_server.server is not None
        async with connect(mcp_server.server._mcp_server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == _TOOL_NAMES
    for name in ("math_check_code", "math_stability", "math_ranges", "math_recheck"):
        assert set(tools[name].outputSchema["required"]) == CHECK_FIELDS


#: Total characters of the tool descriptions the client loads (I1 measurement, for
#: K6's profiles). Pinned with headroom so growth is a decision, not a drift.
MAX_DESCRIPTION_CHARS = 26_000


def test_the_tool_descriptions_stay_within_the_measured_budget() -> None:
    total = sum(len(t.__doc__ or "") for t in mcp_server.TOOLS)
    assert total <= MAX_DESCRIPTION_CHARS, total


# --- pakiet ---------------------------------------------------------------------------
def test_package_data_ships_the_module_data() -> None:
    import tomllib

    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pkg = data["tool"]["setuptools"]["package-data"]["pycodemath"]
    assert "code/data/*.json" in pkg and "code/data/*.jsonl" in pkg


# --- wywołanie funkcji, której parser nie zna, a kalkulator tak --------------------
@pytest.mark.parametrize("line,first", [
    ("gcd(12, 18)", "6"),
    ("mod(2^100, 7)", "2"),
    ("mean([1,2,3,4])", "5/2"),
    ("zeta(3)", "1.20205690315959"),
])
def test_a_call_the_parser_refuses_goes_to_the_calculator(line: str, first: str) -> None:
    out = repl.handle(line)
    assert out.splitlines()[0] == first
    assert out.splitlines()[-1].startswith("certificate: VERIFIED")
    payload = mcp_server.math_eval(line)
    assert payload["certificate"]["kind"] == "calc"


def test_the_call_route_never_moves_what_the_parser_reads() -> None:
    assert repl.handle("gamma(5)") == "24"  # SymPy's gamma, as before
    assert repl.handle("binomial(50, 25)") == "126410606437752"
    with pytest.raises(ParseError, match="unknown function 'foo'"):
        repl.handle("foo(3)")


# --- regresje z rundy adwersarza I1 ---------------------------------------------------
def test_adversary_factor_of_a_negative_number_is_not_a_false_refusal() -> None:
    # the certificate layer's own re-multiplication compared with |n| and refused -1
    assert repl.handle("factor -12").splitlines()[0] == "-1 * 2^2 * 3"
    assert repl.handle("factor -1").splitlines()[0] == "-1"


def test_adversary_a_bound_beyond_every_float_is_a_typed_refusal() -> None:
    # C3 formats such a bound with float() and died of a raw OverflowError
    with pytest.raises(DomainError, match="beyond every finite float"):
        repl.handle("ranges x*x for x in [0, 1e400] as float64")
    p = mcp_server.math_ranges("x*x", {"x": ["0", "1" + "0" * 400]}, "float64")
    assert p["error"]["type"] == "DomainError"


def test_adversary_crt_without_a_solution_says_so() -> None:
    assert repl.handle("crt 1,2 mod 2,4").splitlines()[0] == "no solution"
