"""K1 — strict parser ("never guesses"): the guards of the module.

Every case below was a SILENT wrong answer or a raw Python exception before K1
(``dev-journal/ANALIZA_ROZWOJU_30-10.md`` §A.2/§A.3): a multi-letter name split
into a product of letters, an unknown command word swallowed into the
expression, a parser hang. The contract pinned here: the user gets the right
answer or a typed ``PycodemathError`` with a hint — never a product of letters,
never a traceback, never a hang.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import warnings

import pytest
import sympy as sp

from pycodemath import parse
from pycodemath.cli import repl
from pycodemath.core.errors import ParseError, PycodemathError
from pycodemath.frontend.parser import parse_mode
from pycodemath.verify import check_equal


def _cli(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pycodemath", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={"PYTHONUTF8": "1", "PATH": "/usr/bin:/bin"},
    )


# A product of three or more single letters (``a*c*f*o*r*t``) is the fingerprint
# of a name taken apart letter by letter.
_LETTER_PRODUCT = re.compile(r"(?<![A-Za-z_])[a-z](?:\*\*\d+)?(?:\*[a-z](?:\*\*\d+)?){2,}(?![A-Za-z_(])")


@pytest.mark.parametrize(
    "line, correct",
    [
        ("diff speed*time dtime", "speed"),
        # I1: the three lines K1 could only refuse are commands now (K3 calculator,
        # J1 units) — the correct answer is the first line, the certificate follows
        ("factor 2^64+1", "274177 * 67280421310721"),
        ("isprime 2^61-1", "True"),
        ("5 km/h to m/s", "25/18 m/s ≈ 1.38888888888889 m/s"),
        # I2: K2 is wired — these have correct answers now (no letter product)
        ("latex x^2/2", "\\frac{x^{2}}{2}"),
        ("evalf sqrt(2) 50", "≈ 1.4142135623730950488016887242096980785696718753769"),
    ],
)
def test_no_silent_symbol_split(line: str, correct: "str | None") -> None:
    try:
        out = repl.handle(line)
    except ParseError as exc:
        # a refusal must say what to do instead
        assert re.search(r"did you mean|write |type 'help'|implicit", str(exc)), str(exc)
        return
    assert not _LETTER_PRODUCT.search(out), f"{line!r} -> {out!r}: a name split into letters"
    assert correct is not None, f"{line!r} -> {out!r}: no refusal, and no correct answer known"
    assert out.splitlines()[0] == correct


def test_variable_hidden_by_the_other_reading_is_refused() -> None:
    # strict reads ``xy`` as one name, so d/dx would silently be 0
    with pytest.raises(ParseError, match="implicit"):
        repl.handle("diff xy dx")
    # and with implicit multiplication on, ``time`` would be t*i*m*e
    with parse_mode(strict=False):
        with pytest.raises(ParseError, match="implicit"):
            repl.handle("diff speed*time dtime")
        assert repl.handle("diff xy dx") == "y"


def test_strict_is_the_default_and_implicit_is_opt_in() -> None:
    assert parse("speed*time").sy == sp.Symbol("speed") * sp.Symbol("time")
    assert parse("2x").sy == 2 * sp.Symbol("x")  # a number before a name stays a product
    with parse_mode(strict=False):
        x, y = sp.symbols("x y")
        assert parse("2xy").sy == 2 * x * y


def test_verifier_does_not_prove_a_split_name() -> None:
    v = check_equal("mass*c^2", "m*a*s*s*c^2")
    assert not v.verified


def test_unknown_command_suggests_the_close_one() -> None:
    with pytest.raises(ParseError, match="did you mean 'integrate'"):
        repl.handle("integrat x^2 dx")


# --- no raw Python exception reaches the user -------------------------------


def test_no_raw_python_exceptions_reach_user_solve_inequality() -> None:
    # I2: K5 answers it (it used to be a typed refusal, never "not iterable")
    out = repl.handle("solve x^2-4 > 0 for x")
    assert out.splitlines()[0] == "x ∈ (-oo, -2) ∪ (2, oo)"
    assert "not iterable" not in out


def test_no_raw_python_exceptions_reach_user_n_call() -> None:
    # I2: K2 answers N(pi, 50) — still without a single SymPy warning on the way
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any SymPy warning fails the test
        out = repl.handle("N(pi, 50)")
    assert out.splitlines()[0] == "≈ 3.1415926535897932384626433832795028841971693993751"
    done = _cli("N(pi, 50)")
    assert done.returncode == 0
    assert "Warning" not in done.stderr


def test_no_raw_python_exceptions_reach_user_nan_comparison() -> None:
    try:
        v = check_equal("cosh(t1 - log(0))", "1")
    except PycodemathError:
        return
    assert not v.verified


def test_no_raw_python_exceptions_reach_user_power_tower() -> None:
    # was a hang > 60 s; budget for a plain expression is the default, so the
    # bound here is the refusal itself being immediate (budget + 1 s with the
    # smallest budget the grammar accepts on a symbolic command: 1 s)
    t0 = time.perf_counter()
    try:
        done = _cli("sqrt(2)^10^20", timeout=20)
    except subprocess.TimeoutExpired:
        pytest.fail("sqrt(2)^10^20 hung the CLI")
    assert time.perf_counter() - t0 < 2.0 + 3.0  # process start + 1 s
    assert done.returncode == 1
    assert done.stderr.startswith("error: ") and "Traceback" not in done.stderr


@pytest.mark.parametrize(
    "text",
    ["sqrt(2)^10^20", "cbrt(3)^(10^20)", "(2^(1/3))^(10^20)", "root(5, 4)^(10^30)"],
)
def test_power_tower_on_an_irrational_base_is_refused_before_eval(text: str) -> None:
    code = f"from pycodemath import parse\ntry:\n    parse({text!r})\nexcept Exception as e:\n    print(type(e).__name__)"
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{text} hung the parser")
    assert done.stdout.strip() == "ParseError"


# --- the settings: REPL, one-shot, MCP ---------------------------------------


def test_repl_implicit_setting_round_trip() -> None:
    from pycodemath.frontend.parser import is_strict

    with parse_mode(strict=True):
        assert "off" in repl._implicit_setting("implicit")
        assert "on" in repl._implicit_setting("implicit on")
        assert not is_strict()
        assert repl.handle("2xy") == "2*x*y"
        assert "off" in repl._implicit_setting("implicit off")
        assert is_strict()
        assert repl.handle("2xy") == "2*xy"
        assert repl._implicit_setting("implicit maybe") == "Usage: implicit on|off"
        assert repl._implicit_setting("implicitly 2") is None


def test_one_shot_implicit_flag() -> None:
    assert _cli("--implicit", "diff xy dx").stdout.strip() == "y"
    done = _cli("diff xy dx")
    assert done.returncode == 1 and "implicit on" in done.stderr


def test_mcp_is_strict_even_when_the_session_is_not() -> None:
    from pycodemath.cli import mcp_server

    with parse_mode(strict=False):
        out = mcp_server.math_eval("diff speed*time dtime")
        assert out["text"] == "speed"
        refused = mcp_server.math_eval("implicit on")
        assert refused["error"]["type"] == "ParseError"


@pytest.mark.parametrize(
    "line",
    ["sin x", "x y", "2 x", "ab + c", "speed*time", "v0 t", "zeta (2)"],
)
def test_ordinary_notation_is_not_mistaken_for_a_command(line: str) -> None:
    # these either compute or keep their own, older refusal ("unknown function")
    try:
        repl.handle(line)
    except ParseError as exc:
        assert "unknown command" not in str(exc) and "side by side" not in str(exc)
