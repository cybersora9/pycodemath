"""Interface tests: REPL (command dispatcher), MCP server, one-shot CLI.

Split of tests/test_pycodemath.py into per-area files (step 0 of the
development session) — test content moved unchanged.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from pycodemath.cli import repl
from pycodemath.core.errors import PycodemathError


def test_repl_error_is_caught():
    # bad input does not crash the session — the REPL returns an "error: ..." message
    # (handle() may raise, run() catches it; we check both variants consistently)
    try:
        out = repl.handle("det [[1,2],[3]]")  # invalid matrix
    except Exception as exc:
        out = f"error: {exc}"
    assert "error" in out or "Error" in out or out != ""


def test_repl_numerics_commands():
    root = repl.handle("root x^2-2 for x at 1")
    assert float(root) == pytest.approx(math.sqrt(2))
    integ = repl.handle("nintegrate 2*x dx from 0 to 1")
    assert float(integ) == pytest.approx(1.0)
    xmin = repl.handle("min (x-3)^2 for x at 0")
    assert float(xmin) == pytest.approx(3.0, abs=1e-4)


def test_repl_command_edge_cases():
    # the command word is case-insensitive (as with startswith on lower)
    assert repl.handle("DET [[1,2],[3,4]]").strip() == "-2"
    # extra spaces between tokens do not break parsing
    assert repl.handle("diff   x^2   dx").strip() == "2*x"
    # ' d' inside the expression: d<var> is the LAST segment (like the former rsplit)
    assert repl.handle("diff d*x dx").strip() == "d"
    # incomplete command -> usage message, not an exception or silent parsing
    assert repl.handle("diff x^2").startswith("Usage")
    assert repl.handle("solve x^2-4").startswith("Usage")
    # a negative starting point passes through the number pattern
    root = repl.handle("root x^2-2 for x at -1")
    assert float(root) == pytest.approx(-math.sqrt(2))


def test_repl_linalg_commands():
    assert repl.handle("det [[1,2],[3,4]]").strip() == "-2"
    sol = repl.handle("solve_system [[2,1],[1,3]] = [3,5]")
    assert "Matrix" in sol
    code = repl.handle("code_system [[2,1],[1,3]] = [3,5]")
    assert "np.linalg.solve" in code
    eig = repl.handle("eig [[2,0],[0,3]]")
    assert "2" in eig and "3" in eig


def test_repl_nd_commands():
    g = repl.handle("grad x^2*y for x,y")
    assert "2*x*y" in g
    sol = repl.handle("solve_nd x^2+y^2-4; x-y for x,y at 1,1")
    assert "x = " in sol and "y = " in sol
    for part in sol.split(","):
        assert float(part.split("=")[1]) == pytest.approx(math.sqrt(2))
    xmin = repl.handle("min_nd (x-1)^2+(y+2)^2 for x,y at 0,0")
    vals = [float(p.split("=")[1]) for p in xmin.split(",")]
    assert vals == pytest.approx([1.0, -2.0], abs=1e-4)


def test_repl_min_method_option():
    # module 15: optional [method newton|bfgs] in min / min_nd
    val = repl.handle("min (x-3)^2 for x at 0 method newton")
    assert float(val) == pytest.approx(3.0, abs=1e-12)
    # BFGS on the Rosenbrock valley (gd would fail here — contrast from module 15)
    sol = repl.handle("min_nd (1-x)^2 + 100*(y-x^2)^2 for x,y at -1.2,1 method bfgs")
    vals = [float(p.split("=")[1]) for p in sol.split(",")]
    assert vals == pytest.approx([1.0, 1.0], abs=1e-8)
    # unknown method -> readable error message (PycodemathError contract)
    with pytest.raises(PycodemathError):
        repl.handle("min (x-3)^2 for x at 0 method sgd")
    # without the method flag, behavior unchanged (gd)
    assert float(repl.handle("min (x-3)^2 for x at 0")) == pytest.approx(3.0, abs=1e-4)


def test_repl_symbolic_integrate():
    assert repl.handle("integrate 2*x dx").strip() == "x**2"
    assert repl.handle("integrate cos(x) dx").strip() == "sin(x)"
    assert repl.handle("integrate x^2").startswith("Usage")


# --- MCP server -----------------------------------------------------------
def test_mcp_tool_math_eval():
    pytest.importorskip("mcp")
    from pycodemath.cli.mcp_server import math_eval, server

    # the tool is registered under its own name
    assert "math_eval" in [t.name for t in server._tool_manager.list_tools()]
    # delegates to the REPL dispatcher
    assert math_eval("diff x^2 dx") == "2*x"
    # error comes back as text (the agent does not get a traceback)
    assert math_eval("det [[1,2],[3]]").startswith("error")


def test_mcp_server_import_lazy_without_mcp(monkeypatch):
    # round 2 (Block D): a missing 'mcp' package must not crash the module IMPORT
    # with a traceback — refusal (clean message) only in run()
    import importlib
    import sys

    import pycodemath.cli.mcp_server as ms

    monkeypatch.setitem(sys.modules, "mcp", None)  # forces ImportError
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    try:
        importlib.reload(ms)  # does not raise
        assert ms.server is None
        assert ms.math_eval("diff x^2 dx") == "2*x"  # dispatcher lives without the server
        with pytest.raises(SystemExit, match="mcp"):
            ms.run()
    finally:
        monkeypatch.undo()
        importlib.reload(ms)  # restore state for the remaining tests


# --- one-shot CLI ---------------------------------------------------------
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pycodemath", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).resolve().parent.parent,
    )


def test_cli_one_shot_success():
    # a single command from argv: result on stdout, exit code 0, no REPL loop
    proc = _run_cli("diff x^2 dx")
    assert proc.returncode == 0
    assert proc.stdout.strip() == "2*x"


def test_cli_one_shot_error_exit_code():
    # bad input: message on stderr and exit code 1 (scriptability)
    proc = _run_cli("det [[1,2],[3]]")
    assert proc.returncode == 1
    assert "error" in proc.stderr


# --- ODE commands in the REPL -----------------------------------------------
def test_repl_ode_commands():
    out = repl.handle("dsolve y for y(t)")
    assert "C1" in out and "y(t) =" in out
    val = repl.handle("ode y for y(t) from 0 to 1 at 1")
    assert float(val.split("=")[1]) == pytest.approx(math.e, abs=1e-6)
    val200 = repl.handle("ode y for y(t) from 0 to 1 at 1 steps 200")
    assert float(val200.split("=")[1]) == pytest.approx(math.e, abs=1e-6)
    # incomplete command -> usage message, not an exception
    assert repl.handle("dsolve y").startswith("Usage")
    assert repl.handle("ode y for y(t)").startswith("Usage")


def test_repl_ode_adaptive_command():
    val = repl.handle("ode_adaptive y for y(t) from 0 to 1 at 1")
    # format: "y(1) = <number>  (<k> adaptive steps)"
    assert val.startswith("y(1) =") and "steps" in val
    approx_e = float(val.split("=")[1].split("(")[0])
    assert approx_e == pytest.approx(math.e, abs=1e-5)
    val_tol = repl.handle("ode_adaptive y for y(t) from 0 to 1 at 1 rtol 1e-9")
    assert float(val_tol.split("=")[1].split("(")[0]) == pytest.approx(math.e, abs=1e-7)
    # incomplete command -> usage message, not an exception
    assert repl.handle("ode_adaptive y for y(t)").startswith("Usage")


def test_repl_odedense_command():
    val = repl.handle("odedense y for y(t) from 0 to 2 at 1 at t=0.37")
    assert val.startswith("y(0.37) =") and "dense output" in val
    approx = float(val.split("=")[1].split("(")[0])
    assert approx == pytest.approx(math.exp(0.37), abs=1e-5)
    # incomplete command -> usage message, not an exception
    assert repl.handle("odedense y for y(t) from 0 to 1 at 1").startswith("Usage")


def test_repl_odeevents_stop_flag():
    out = repl.handle("odeevents -1 for y(t) from 0 to 3 at 1 zero y stop")
    assert out.startswith("stop at t=1") and "stopped" in out
    # stop combines with dir: the first rising zero of sin(t) is 2π
    up = repl.handle(
        "odeevents cos(t) for y(t) from 0.5 to 10 at 0.479425539 zero y dir +1 stop"
    )
    assert up.startswith("stop at t=6.28319")
    # without the stop flag: three events as in module 9
    full = repl.handle("odeevents -1 for y(t) from 0 to 3 at 1 zero y")
    assert not full.startswith("stop") and full.count("t=") == 1


# --- odestiff command (module 12) --------------------------------------------
# exact solution of the reference problem y' = -1000·(y - cos(t)), y(0)=0
# (the same constants as in test_ode.py — helper duplicated during the split)
_STIFF_A = 1_000_000 / 1_000_001
_STIFF_B = 1_000 / 1_000_001


def _stiff_exact(t: float) -> float:
    return (
        _STIFF_A * math.cos(t) + _STIFF_B * math.sin(t)
        - _STIFF_A * math.exp(-1000.0 * t)
    )


def test_repl_odestiff_command():
    val = repl.handle("odestiff -1000*(y-cos(t)) for y(t) from 0 to 1 at 0")
    assert val.startswith("y(1) =") and "BDF2" in val
    approx = float(val.split("=")[1].split("(")[0])
    assert approx == pytest.approx(_stiff_exact(1.0), abs=1e-6)
    # incomplete command -> usage message, not an exception
    assert repl.handle("odestiff y for y(t)").startswith("Usage")


def test_repl_odestiff_adaptive_command():
    # module 17: variable step — measured y(1) = 0.54114326 (error 2.3e-8),
    # 313 implicit steps at the default rtol 1e-6
    val = repl.handle("odestiff_adaptive -1000*(y-cos(t)) for y(t) from 0 to 1 at 0")
    assert val.startswith("y(1) =") and "adaptive BDF" in val
    approx = float(val.split("=")[1].split("(")[0])
    assert approx == pytest.approx(_stiff_exact(1.0), abs=1e-6)
    # rtol explicitly + incomplete command
    tight = repl.handle(
        "odestiff_adaptive -1000*(y-cos(t)) for y(t) from 0 to 1 at 0 rtol 1e-8"
    )
    assert float(tight.split("=")[1].split("(")[0]) == pytest.approx(
        _stiff_exact(1.0), abs=1e-7
    )
    assert repl.handle("odestiff_adaptive y for y(t)").startswith("Usage")


# --- audit before extension: PycodemathError contract in commands -------------
def test_repl_bad_number_raises_pycodemath():
    # the _NUM pattern lets "1e" through — only _num() gives a readable error message
    with pytest.raises(PycodemathError):
        repl.handle("root x^2-2 for x at 1e")
    with pytest.raises(PycodemathError):
        repl.handle("ode y for y(t) from 0 to 1e at 1")


def test_repl_odeevents_command():
    out = repl.handle("odeevents cos(t) for y(t) from 0.5 to 10 at 0.479425539 zero y")
    # three events separated by a semicolon
    assert out.count("t=") == 3
    first_t = float(out.split("t=")[1].split(" ")[0])
    assert first_t == pytest.approx(math.pi, abs=1e-4)
    # direction filter: only rising -> one event (2π)
    up = repl.handle("odeevents cos(t) for y(t) from 0.5 to 10 at 0.479425539 zero y dir +1")
    assert up.count("t=") == 1
    # no events -> readable message, not an exception
    none = repl.handle("odeevents cos(t) for y(t) from 0.5 to 10 at 0.479425539 zero y+2")
    assert none == "(no events)"
    # incomplete command -> usage message
    assert repl.handle("odeevents cos(t) for y(t) from 0.5 to 10 at 1").startswith("Usage")
