"""Testy interfejsów: REPL (dispatcher komend), serwer MCP, one-shot CLI.

Rozbicie tests/test_pycodemath.py na pliki per obszar (krok 0 sesji
rozwojowej) — treść testów przeniesiona bez zmian.
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
    # błędne wejście nie wywraca sesji — REPL zwraca komunikat "błąd: ..."
    # (handle() może rzucić, run() go łapie; sprawdzamy oba warianty spójnie)
    try:
        out = repl.handle("det [[1,2],[3]]")  # niepoprawna macierz
    except Exception as exc:
        out = f"błąd: {exc}"
    assert "error" in out or "Error" in out or out != ""


def test_repl_numerics_commands():
    root = repl.handle("root x^2-2 for x at 1")
    assert float(root) == pytest.approx(math.sqrt(2))
    integ = repl.handle("nintegrate 2*x dx from 0 to 1")
    assert float(integ) == pytest.approx(1.0)
    xmin = repl.handle("min (x-3)^2 for x at 0")
    assert float(xmin) == pytest.approx(3.0, abs=1e-4)


def test_repl_command_edge_cases():
    # słowo komendy jest niewrażliwe na wielkość liter (jak przy startswith na lower)
    assert repl.handle("DET [[1,2],[3,4]]").strip() == "-2"
    # nadmiarowe spacje między tokenami nie psują parsowania
    assert repl.handle("diff   x^2   dx").strip() == "2*x"
    # ' d' wewnątrz wyrażenia: d<var> to OSTATNI segment (jak dawny rsplit)
    assert repl.handle("diff d*x dx").strip() == "d"
    # niekompletna komenda -> komunikat użycia, nie wyjątek ani ciche parsowanie
    assert repl.handle("diff x^2").startswith("Usage")
    assert repl.handle("solve x^2-4").startswith("Usage")
    # ujemny punkt startowy przechodzi przez wzorzec liczby
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
    # moduł 15: opcjonalny [method newton|bfgs] w min / min_nd
    val = repl.handle("min (x-3)^2 for x at 0 method newton")
    assert float(val) == pytest.approx(3.0, abs=1e-12)
    # BFGS na dolinie Rosenbrocka (gd by tu odmówił — kontrast z modułu 15)
    sol = repl.handle("min_nd (1-x)^2 + 100*(y-x^2)^2 for x,y at -1.2,1 method bfgs")
    vals = [float(p.split("=")[1]) for p in sol.split(",")]
    assert vals == pytest.approx([1.0, 1.0], abs=1e-8)
    # nieznana metoda -> czytelny polski błąd (kontrakt PycodemathError)
    with pytest.raises(PycodemathError):
        repl.handle("min (x-3)^2 for x at 0 method sgd")
    # bez flagi method zachowanie bez zmian (gd)
    assert float(repl.handle("min (x-3)^2 for x at 0")) == pytest.approx(3.0, abs=1e-4)


def test_repl_symbolic_integrate():
    assert repl.handle("integrate 2*x dx").strip() == "x**2"
    assert repl.handle("integrate cos(x) dx").strip() == "sin(x)"
    assert repl.handle("integrate x^2").startswith("Usage")


# --- serwer MCP -----------------------------------------------------------
def test_mcp_tool_math_eval():
    pytest.importorskip("mcp")
    from pycodemath.cli.mcp_server import math_eval, server

    # narzędzie zarejestrowane pod swoją nazwą
    assert "math_eval" in [t.name for t in server._tool_manager.list_tools()]
    # MODUŁ 6 ŚWIADOMIE ZMIENIŁ TEN KONTRAKT: narzędzie zwraca ustrukturyzowany
    # payload, a nie goły string — o to w tym module chodziło (dowody modułów 3–5
    # ginęły na granicy procesu). Czytelna odpowiedź NIE znika: jest polem
    # ``text``, bo moduł 1 zmierzył, że structuredContent wypiera blok tekstowy.
    # Kształt payloadu i granica „porażka vs odmowa" mają własny plik: test_mcp.py.
    assert math_eval("diff x^2 dx")["text"] == "2*x"
    # błąd wraca jako DANE (agent nie dostaje tracebacku ani samej prozy)
    assert math_eval("det [[1,2],[3]]")["error"]["type"] == "ParseError"


def test_mcp_server_import_lazy_without_mcp(monkeypatch):
    # runda 2 (Blok D): brak pakietu 'mcp' nie może wywracać IMPORTU modułu
    # tracebackiem — odmowa (czysty komunikat) dopiero w run()
    import importlib
    import sys

    import pycodemath.cli.mcp_server as ms

    monkeypatch.setitem(sys.modules, "mcp", None)  # wymusza ImportError
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    try:
        importlib.reload(ms)  # nie rzuca
        assert ms.server is None
        # dispatcher żyje bez serwera — a od modułu 6 także CAŁY payload
        # (zwykły dict, żadnej zależności od pydantic/mcp)
        assert ms.math_eval("diff x^2 dx")["text"] == "2*x"
        with pytest.raises(SystemExit, match="mcp"):
            ms.run()
    finally:
        monkeypatch.undo()
        importlib.reload(ms)  # przywróć stan dla pozostałych testów


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
    # jedna komenda z argv: wynik na stdout, kod wyjścia 0, bez pętli REPL
    proc = _run_cli("diff x^2 dx")
    assert proc.returncode == 0
    assert proc.stdout.strip() == "2*x"


def test_cli_one_shot_error_exit_code():
    # błędne wejście: komunikat na stderr i kod wyjścia 1 (skryptowalność)
    proc = _run_cli("det [[1,2],[3]]")
    assert proc.returncode == 1
    assert "error" in proc.stderr


# --- komendy ODE w REPL -----------------------------------------------------
def test_repl_ode_commands():
    out = repl.handle("dsolve y for y(t)")
    assert "C1" in out and "y(t) =" in out
    val = repl.handle("ode y for y(t) from 0 to 1 at 1")
    assert float(val.split("=")[1]) == pytest.approx(math.e, abs=1e-6)
    val200 = repl.handle("ode y for y(t) from 0 to 1 at 1 steps 200")
    assert float(val200.split("=")[1]) == pytest.approx(math.e, abs=1e-6)
    # niekompletna komenda -> komunikat użycia, nie wyjątek
    assert repl.handle("dsolve y").startswith("Usage")
    assert repl.handle("ode y for y(t)").startswith("Usage")


def test_repl_ode_adaptive_command():
    val = repl.handle("ode_adaptive y for y(t) from 0 to 1 at 1")
    # format: "y(1) = <liczba>  (<k> kroków adaptacyjnych)"
    assert val.startswith("y(1) =") and "steps" in val
    approx_e = float(val.split("=")[1].split("(")[0])
    assert approx_e == pytest.approx(math.e, abs=1e-5)
    val_tol = repl.handle("ode_adaptive y for y(t) from 0 to 1 at 1 rtol 1e-9")
    assert float(val_tol.split("=")[1].split("(")[0]) == pytest.approx(math.e, abs=1e-7)
    # niekompletna komenda -> komunikat użycia, nie wyjątek
    assert repl.handle("ode_adaptive y for y(t)").startswith("Usage")


def test_repl_odedense_command():
    val = repl.handle("odedense y for y(t) from 0 to 2 at 1 at t=0.37")
    assert val.startswith("y(0.37) =") and "dense output" in val
    approx = float(val.split("=")[1].split("(")[0])
    assert approx == pytest.approx(math.exp(0.37), abs=1e-5)
    # niekompletna komenda -> komunikat użycia, nie wyjątek
    assert repl.handle("odedense y for y(t) from 0 to 1 at 1").startswith("Usage")


def test_repl_odeevents_stop_flag():
    out = repl.handle("odeevents -1 for y(t) from 0 to 3 at 1 zero y stop")
    assert out.startswith("stop at t=1") and "stopped" in out
    # stop łączy się z dir: pierwsze narastające zero sin(t) to 2π
    up = repl.handle(
        "odeevents cos(t) for y(t) from 0.5 to 10 at 0.479425539 zero y dir +1 stop"
    )
    assert up.startswith("stop at t=6.28319")
    # bez flagi stop: trzy zdarzenia jak w module 9
    full = repl.handle("odeevents -1 for y(t) from 0 to 3 at 1 zero y")
    assert not full.startswith("stop") and full.count("t=") == 1


# --- komenda odestiff (moduł 12) --------------------------------------------
# rozwiązanie dokładne problemu wzorcowego y' = -1000·(y - cos(t)), y(0)=0
# (te same stałe co w test_ode.py — pomocnik zduplikowany przy rozbiciu)
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
    # niekompletna komenda -> komunikat użycia, nie wyjątek
    assert repl.handle("odestiff y for y(t)").startswith("Usage")


def test_repl_odestiff_adaptive_command():
    # moduł 17: zmienny krok — zmierzone y(1) = 0.54114326 (błąd 2.3e-8),
    # 313 kroków niejawnych przy domyślnym rtol 1e-6
    val = repl.handle("odestiff_adaptive -1000*(y-cos(t)) for y(t) from 0 to 1 at 0")
    assert val.startswith("y(1) =") and "adaptive BDF" in val
    approx = float(val.split("=")[1].split("(")[0])
    assert approx == pytest.approx(_stiff_exact(1.0), abs=1e-6)
    # rtol jawnie + niekompletna komenda
    tight = repl.handle(
        "odestiff_adaptive -1000*(y-cos(t)) for y(t) from 0 to 1 at 0 rtol 1e-8"
    )
    assert float(tight.split("=")[1].split("(")[0]) == pytest.approx(
        _stiff_exact(1.0), abs=1e-7
    )
    assert repl.handle("odestiff_adaptive y for y(t)").startswith("Usage")


# --- audyt przed dobudową: kontrakt PycodemathError w komendach -------------
def test_repl_bad_number_raises_pycodemath():
    # wzorzec _NUM przepuszcza "1e" — dopiero _num() daje czytelny polski błąd
    with pytest.raises(PycodemathError):
        repl.handle("root x^2-2 for x at 1e")
    with pytest.raises(PycodemathError):
        repl.handle("ode y for y(t) from 0 to 1e at 1")


def test_repl_odeevents_command():
    out = repl.handle("odeevents cos(t) for y(t) from 0.5 to 10 at 0.479425539 zero y")
    # trzy zdarzenia rozdzielone średnikiem
    assert out.count("t=") == 3
    first_t = float(out.split("t=")[1].split(" ")[0])
    assert first_t == pytest.approx(math.pi, abs=1e-4)
    # filtr kierunku: tylko narastające -> jedno zdarzenie (2π)
    up = repl.handle("odeevents cos(t) for y(t) from 0.5 to 10 at 0.479425539 zero y dir +1")
    assert up.count("t=") == 1
    # brak zdarzeń -> czytelny komunikat, nie wyjątek
    none = repl.handle("odeevents cos(t) for y(t) from 0.5 to 10 at 0.479425539 zero y+2")
    assert none == "(no events)"
    # niekompletna komenda -> komunikat użycia
    assert repl.handle("odeevents cos(t) for y(t) from 0.5 to 10 at 1").startswith("Usage")
