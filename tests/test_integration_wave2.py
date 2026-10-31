"""I2 — fala 2 (K2, L1, C5, K5) jako część jednego produktu.

Strażnik z promptu I2 (``test_wave2_end_to_end``) i kontrakty warstwy integracji
fali 2: polecenia, trasy ``math_eval``, narzędzia ``math_lint`` / ``math_iterations``,
nowe ``kind`` certyfikatów, re-eksporty, wpis pytest11, bezpieczne kodowanie wyjścia.
Wiersze tabeli podpięcia są w ``test_integration_full.SURFACES``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pycodemath.cli import full, mcp_server, repl
from pycodemath.verify import certificate as C

_ROOT = Path(__file__).resolve().parent.parent

LOOP_OFF_BY_ONE = textwrap.dedent(
    '''
    def sum_to(n: int) -> int:
        """Sum of 1..n.

        :math: n*(n+1)/2
        :domain n: integer, n>=0
        """
        total = 0
        for k in range(n):
            total += k
        return total
    '''
)


# --- strażnik: od pliku do certyfikatu przez dwa moduły fali 2 --------------------
def test_wave2_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "sums.py").write_text(LOOP_OFF_BY_ONE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # C5: the loop runs n times, the author claimed n + 1 -> REFUTED with an executed witness
    answer = repl.handle_full("iterations sums.py:sum_to == n + 1")
    outcome = answer.outcome
    assert outcome is not None and outcome.kind == "iterations"
    assert outcome.text.startswith("REFUTED"), outcome.text
    assert "at n=0 the loop ran 0 times; the claim gives 1" in outcome.text
    assert outcome.result is not None and outcome.result["witness"]["n"] == 0

    # L1: the same off-by-one, found by the linter against the :math: line
    out = subprocess.run([sys.executable, "-m", "pycodemath", "lint", "sums.py", "--no-cache"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=300,
                         env={**os.environ, "PYTHONUTF8": "1"})
    assert out.returncode == 1, out.stdout + out.stderr  # ruff's code: findings
    assert "PCM504" in out.stdout and "sum_to(n=1) returns 0" in out.stdout

    # the C5 answer, written down, re-checks from the file alone
    cert = outcome.certificate
    assert cert is not None and cert.verdict["status"] == "refuted"
    (tmp_path / "c.json").write_text(cert.to_json(), encoding="utf-8")
    assert repl.handle("recheck c.json").startswith("REPRODUCED — iterations")
    # ... and a changed witness number is caught
    data = json.loads(cert.to_json())
    data["verdict"]["counterexample"] = {"n": "5"}
    data["digest"] = C.content_digest(data)
    assert not C.recheck(data).ok


# --- certyfikaty: każdy nowy kind fali 2 -------------------------------------------------
def _issued_wave2() -> "list[tuple[str, C.Certificate, str, str]]":
    """(kind, certificate, a recorded number, what to change it to)."""
    out = []
    o = full.approx_outcome("sqrt(2)", 15)
    out.append(("approx", o.certificate, "1.41421356237310", "1.41421356237311"))
    o = repl.handle_full("identify zeta(2)").outcome
    out.append(("identify", o.certificate, "pi**2/6", "pi**2/7"))
    o = repl.handle_full("solve x^2-4 > 0 for x").outcome
    out.append(("inequality", o.certificate, "Interval.open(2, oo)", "Interval.open(3, oo)"))
    o = repl.handle_full("solve x + y = 3, x - y = 1 for x, y").outcome
    out.append(("system", o.certificate, '\\"y\\":\\"1\\"', '\\"y\\":\\"2\\"'))
    o = repl.handle_full("solve sin(x) = 1/2 all").outcome
    out.append(("periodic", o.certificate, "pi/6 + 2*pi*k", "pi/5 + 2*pi*k"))
    o = repl.handle_full("integrate exp(-x^2) dx from -oo to oo").outcome
    out.append(("definite_integral", o.certificate, '\\"value\\":\\"sqrt(pi)\\"', '\\"value\\":\\"pi\\"'))
    o = full.iterations_outcome(LOOP_OFF_BY_ONE, None, "n + 1", execute=True)
    out.append(("iterations", o.certificate, '"n": "0"', '"n": "1"'))
    o = full.complexity_outcome(LOOP_OFF_BY_ONE, None, "O(1)", execute=True)
    out.append(("complexity", o.certificate, '\\"limit\\":\\"oo\\"', '\\"limit\\":\\"1\\"'))
    o = full.recurrence_outcome("T(n) = 2*T(n/2) + n, T(1) = 1")
    out.append(("recurrence", o.certificate, "n*log(n)/log(2) + n", "n*log(n)/log(2) + 2*n"))
    o = full.lint_outcome("def gross(price: float) -> float:\n    return round(price * 1.23, 2)\n",
                          "shop.py", select=["PCM401"], execute=False)
    out.append(("lint", o.certificate, '\\"observed\\":\\"0.61\\"', '\\"observed\\":\\"0.62\\"'))
    return out


@pytest.fixture(scope="module")
def issued() -> "list[tuple[str, C.Certificate, str, str]]":
    return _issued_wave2()


WAVE2_KINDS = ["approx", "identify", "inequality", "system", "periodic", "definite_integral",
               "iterations", "complexity", "recurrence", "lint"]


@pytest.mark.parametrize("kind", WAVE2_KINDS)
def test_every_wave2_kind_round_trips_and_rechecks(issued, kind) -> None:
    cert = next(c for k, c, _, _ in issued if k == kind)
    assert cert is not None and cert.kind == kind
    text = cert.to_json()
    again = C.Certificate.from_json(text)
    assert again.to_json() == text and again.digest_ok
    r = C.recheck(text)
    assert r.ok, r.text()


@pytest.mark.parametrize("kind", WAVE2_KINDS)
def test_every_wave2_kind_catches_one_changed_number(issued, kind) -> None:
    cert, old, new = next((c, o, n) for k, c, o, n in issued if k == kind)
    text = cert.to_json()
    assert old in text, (kind, old, text[:600])
    tampered = text.replace(old, new, 1)
    assert not C.recheck(tampered).ok
    data = json.loads(tampered)
    data["digest"] = C.content_digest(data)  # a forger re-seals: the RE-RUN still catches it
    r = C.recheck(data)
    assert r.digest_ok and not r.ok, (kind, r.text())


def test_a_non_rigorous_approximation_is_never_verified() -> None:
    # review note 5: stable-doubling agrees, it does not prove
    o = full.approx_outcome("besselj(0, 1)", 15)
    assert o.result is not None and o.result["rigorous"] is False
    assert "NOT rigorous" in o.text
    assert o.status == "undecided"
    assert C.recheck(o.certificate).ok


def test_a_k5_refusal_carries_its_number() -> None:
    # review note 6: no closed form -> converges, value ≈ …, not a bare error
    out = repl.handle_full("integrate exp(sin(x)) dx from 0 to 1")
    assert out.text.startswith("no closed form; the integral converges (proved), value ≈ 1.63186960841805")
    assert out.outcome is not None and out.outcome.status == "undecided"
    assert out.outcome.result["approx"].startswith("1.63186960841805")
    p = mcp_server.math_eval("integrate exp(sin(x)) dx from 0 to 1")
    assert p["error"] is None and p["certificate"]["verdict"]["status"] == "undecided"


def test_the_principal_value_only_on_request() -> None:
    # review note 7
    plain = repl.handle("integrate 1/x dx from -1 to 1")
    assert plain.startswith("diverges") and "add 'pv'" in plain
    pv = repl.handle("integrate 1/x dx from -1 to 1 pv")
    assert pv.startswith("0  ≈ 0") and "Cauchy principal value — requested" in pv


def test_old_solve_and_integrate_lines_keep_their_meaning() -> None:
    assert repl.handle("solve x^2-4 for x") == "-2, 2"
    assert repl.handle("integrate 2*x dx") == "x**2"
    assert float(repl.handle("nintegrate 2*x dx from 0 to 1")) == pytest.approx(1.0)


def test_the_dual_answer_only_where_a_decimal_says_something() -> None:
    assert repl.handle("sqrt(2)/2") == "sqrt(2)/2  ≈ 0.707106781186548"
    assert repl.handle("2^100") == "1267650600228229401496703205376"  # an integer: no decimal
    assert repl.handle("x + 1") == "x + 1"  # free symbols: no decimal
    p = mcp_server.math_eval("pi")
    assert p["result"]["kind"] == "approx" and p["certificate"]["verdict"]["status"] == "verified"


# --- MCP fali 2 ------------------------------------------------------------------------
CHECK_FIELDS = {"text", "kind", "status", "result", "certificate", "error"}


def test_math_lint_does_not_execute_by_default(tmp_path: Path) -> None:
    target = tmp_path / "pwned.npy"
    src = LOOP_OFF_BY_ONE.replace("    total = 0\n",
                                  f"    import numpy as np\n    np.save({str(target)!r}, np.zeros(1))\n"
                                  "    total = 0\n")
    p = mcp_server.math_lint(src)
    assert set(p) == CHECK_FIELDS and p["error"] is None
    assert not target.exists()
    assert all(f["code"] != "PCM504" for f in p["result"]["findings"])
    p = mcp_server.math_lint(LOOP_OFF_BY_ONE, execute=True)
    assert p["status"] == "refuted" and p["result"]["findings"][0]["code"] == "PCM504"
    r = mcp_server.math_recheck(json.dumps(p["certificate"]))
    assert r["error"] is not None and "execute=true" in r["error"]["message"]
    assert mcp_server.math_recheck(json.dumps(p["certificate"]), execute=True)["status"] == "verified"


def test_math_lint_finds_float_money_with_a_witness() -> None:
    p = mcp_server.math_lint("def gross(price: float) -> float:\n    return round(price * 1.23, 2)\n")
    assert p["status"] == "refuted"
    codes = {f["code"] for f in p["result"]["findings"]}
    assert "PCM401" in codes
    w = next(f for f in p["result"]["findings"] if f["code"] == "PCM401")["witness"]
    assert w["inputs"] and w["observed"] != w["expected"]


def test_math_iterations_counts_without_running_by_default() -> None:
    p = mcp_server.math_iterations(LOOP_OFF_BY_ONE)
    assert set(p) == CHECK_FIELDS and p["error"] is None
    assert p["result"]["main"] == "n" and p["result"]["certificate"] is None
    assert p["status"] == "undecided"
    p = mcp_server.math_iterations(LOOP_OFF_BY_ONE, claim="n + 1", execute=True)
    assert p["status"] == "refuted" and p["certificate"]["verdict"]["counterexample"] == {"n": "0"}
    # C5 decides a complexity only once the count was confirmed by running it
    p = mcp_server.math_iterations(LOOP_OFF_BY_ONE, claim="O(n)")
    assert p["kind"] == "complexity" and p["status"] == "undecided"
    assert "not executed" in p["text"]
    p = mcp_server.math_iterations(LOOP_OFF_BY_ONE, claim="O(n)", execute=True)
    assert p["kind"] == "complexity" and p["status"] == "verified"


def test_math_eval_never_runs_a_file_an_agent_names(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "pwned.npy"
    (tmp_path / "f.py").write_text(
        LOOP_OFF_BY_ONE.replace("    total = 0\n", f"    import numpy as np\n    np.save({str(target)!r}, "
                                "np.zeros(1))\n    total = 0\n"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    p = mcp_server.math_eval("iterations f.py:sum_to == n + 1")
    assert p["error"] is None and p["certificate"]["verdict"]["status"] == "undecided"
    p = mcp_server.math_eval("lint f.py --no-cache")
    assert p["error"] is None
    assert not target.exists()


def test_the_wave2_tools_are_registered() -> None:
    pytest.importorskip("mcp")
    import asyncio

    from mcp.shared.memory import create_connected_server_and_client_session as connect

    async def go():
        assert mcp_server.server is not None
        async with connect(mcp_server.server._mcp_server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(go())}
    for name in ("math_lint", "math_iterations"):
        assert set(tools[name].outputSchema["required"]) == CHECK_FIELDS
        assert len(tools[name].description or "") <= 600


# --- lint: kody wyjścia, kodowanie konsoli, wtyczka pytest --------------------------------
def _run(args: "list[str]", cwd: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, timeout=300,
                          env={**os.environ, **env})


def test_lint_exit_codes_are_ruffs_and_only_for_lint(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("def gross(price: float) -> float:\n    return round(price * 1.23, 2)\n",
                                     encoding="utf-8")
    (tmp_path / "good.py").write_text("def f(n: int) -> int:\n    return n + 1\n", encoding="utf-8")
    assert _run(["-m", "pycodemath", "lint", "bad.py", "--no-cache"], tmp_path).returncode == 1
    assert _run(["-m", "pycodemath", "lint", "good.py", "--no-cache"], tmp_path).returncode == 0
    assert _run(["-m", "pycodemath", "lint", "missing.py"], tmp_path).returncode == 2
    assert _run(["-m", "pycodemath", "lint", "--bogus"], tmp_path).returncode == 2
    # every other command keeps 0 = answered (even REFUTED), 1 = refused
    assert _run(["-m", "pycodemath", "verify sqrt(x^2) == x"], tmp_path).returncode == 0
    assert _run(["-m", "pycodemath", "solve x^2 > 0 for"], tmp_path).returncode in (0, 1)


def test_lint_output_survives_a_cp437_console(tmp_path: Path) -> None:
    # review note 1: "—" and "≈" cannot be encoded in cp437; nothing may die of it
    (tmp_path / "bad.py").write_text("def gross(price: float) -> float:\n    return round(price * 1.23, 2)\n",
                                     encoding="utf-8")
    entry = "import sys; from pycodemath.cli.full import lint_entry; sys.argv = ['pycodemath-lint', 'bad.py', '--no-cache']; lint_entry()"
    for args in (["-c", entry], ["-m", "pycodemath", "lint", "bad.py", "--no-cache"]):
        done = _run(args, tmp_path, PYTHONIOENCODING="cp437", PYTHONUTF8="0")
        err = done.stderr.decode("utf-8", "replace")
        assert done.returncode == 1, err
        assert "Traceback" not in err and "UnicodeEncodeError" not in err
        assert b"PCM401" in done.stdout


def test_the_pytest_plugin_entry_point_is_inert_without_pcm(tmp_path: Path) -> None:
    # review note 3: once installed the plugin loads always; without --pcm it imports no engine
    (tmp_path / "test_a.py").write_text(
        "import sys\n\ndef test_a():\n"
        "    assert not any(m.startswith(('pycodemath.code.', 'pycodemath.lint.engine')) for m in sys.modules)\n",
        encoding="utf-8")
    done = _run(["-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "pycodemath.lint.pytest_plugin",
                 "test_a.py"], tmp_path)
    assert done.returncode == 0, done.stdout.decode() + done.stderr.decode()
    import tomllib

    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    eps = data["project"]["entry-points"]["pytest11"]
    # named by the module path: ``-p pycodemath.lint.pytest_plugin`` then finds it registered
    assert eps == {"pycodemath.lint.pytest_plugin": "pycodemath.lint.pytest_plugin"}
    assert data["project"]["scripts"]["pycodemath-lint"] == "pycodemath.cli.full:lint_entry"
    assert "lint/RULES.md" in data["tool"]["setuptools"]["package-data"]["pycodemath"]


def test_the_wave2_names_load_lazily() -> None:
    code = ("import sys, pycodemath; "
            "print(sorted(m for m in sys.modules if m.startswith(('pycodemath.code.', 'pycodemath.lint', "
            "'pycodemath.engine.precision', 'pycodemath.engine.solvesets', 'pycodemath.engine.definite', "
            "'pycodemath.engine.identify'))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
    import pycodemath
    import pycodemath.code

    assert pycodemath.LintConfig is pycodemath.lint.Config
    assert pycodemath.code.count_iterations is pycodemath.count_iterations


# --- regresje z rundy adwersarza I2 -------------------------------------------------
_PAYLOAD = "__import__('os').system('touch {path}')"


def test_adversary_c5_text_is_never_evaluated(tmp_path: Path) -> None:
    # C5 sympifies claims and recurrences without a restricted namespace: os.system RAN
    # over MCP (claim, complexity claim, recurrence, initial value, master). Guarded now.
    p = lambda tag: _PAYLOAD.format(path=tmp_path / tag)  # noqa: E731
    payloads = [
        mcp_server.math_iterations(LOOP_OFF_BY_ONE, claim=p("claim")),
        mcp_server.math_iterations(LOOP_OFF_BY_ONE, claim=f"O({p('cx')})"),
        mcp_server.math_eval(f"recurrence T(n) = T(n-1) + {p('rec')}"),
        mcp_server.math_eval(f"recurrence T(n) = T(n-1) + 1, T(0) = {p('init')}"),
        mcp_server.math_eval(f"master 2 2 {p('master')}"),
    ]
    for payload in payloads:
        assert payload["error"] is not None and payload["error"]["type"] == "ParseError", payload
    # a forged certificate goes through the same guard on re-check
    good = mcp_server.math_eval("recurrence T(n) = 2*T(n/2) + n, T(1) = 1")["certificate"]
    good["claim"]["args"]["recurrence"] = f"T(n) = T(n-1) + {p('forged')}"
    good["digest"] = C.content_digest(good)
    r = mcp_server.math_recheck(json.dumps(good))
    assert r["error"] is not None and r["error"]["type"] == "ParseError"
    assert list(tmp_path.iterdir()) == []


def test_adversary_c5_guard_lets_real_mathematics_through() -> None:
    for text in ("T(n) = T(n-1) + T(n-2), T(0) = 0, T(1) = 1", "T(n) = 4T(n/2) + n^2",
                 "T(n) = 2*T(n/2) + n/log(n)", "O(n log n)", "Θ(n^2)", "n*(n-1)/2",
                 "floor(log2(n)) + 1", "Omega(sqrt(n))"):
        assert C.c5_text_guard(text, "t") == text


def test_adversary_conflicting_initial_values_are_refused() -> None:
    # C5 silently kept the last of T(1) = 1, T(1) = 2
    from pycodemath.core.errors import ParseError

    with pytest.raises(ParseError, match="two different initial values at 1"):
        repl.handle("recurrence T(n) = 2*T(n/2) + n, T(1) = 1, T(1) = 2")


def test_adversary_lint_line_as_one_argument_keeps_ruffs_exit_codes(tmp_path: Path) -> None:
    # ``python -m pycodemath "lint nope.py"`` exited 0 on an error
    assert _run(["-m", "pycodemath", "lint nope.py"], tmp_path).returncode == 2
    assert _run(["-m", "pycodemath", "lint"], tmp_path).returncode == 2
    (tmp_path / "bin.py").write_bytes(bytes(range(256)))
    done = _run(["-m", "pycodemath", "iterations bin.py"], tmp_path)
    assert done.returncode == 1 and b"is not UTF-8 text" in done.stderr


def test_adversary_n_with_bad_digits_says_how_to_write_it() -> None:
    from pycodemath.core.errors import ParseError

    for line in ("N(pi, -5)", "N(pi, 10**9)", "N()"):
        with pytest.raises(ParseError, match=r"N\(<expression>, <digits>\)"):
            repl.handle(line)
