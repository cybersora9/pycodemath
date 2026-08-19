"""Testy generatora kodu: emit/pipeline, codegen ODE.

Rozbicie tests/test_pycodemath.py na pliki per obszar (krok 0 sesji
rozwojowej) — treść testów przeniesiona bez zmian.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pycodemath import (
    E,
    M,
    V,
    generate,
    generate_ode,
    generate_ode_adaptive,
    generate_ode_dense,
    generate_system,
    parse,
)
from pycodemath.codegen.emit import emit_ode
from pycodemath.core.errors import PycodemathError
from pycodemath.engine import ode


def test_codegen_matches_eval():
    expr = parse("(sin(x)+cos(x))**2 + x**3")
    art = generate(expr)
    x = np.linspace(0.1, 5.0, 50)
    expected = (np.sin(x) + np.cos(x)) ** 2 + x**3
    assert np.allclose(art(x), expected)


def test_cse_and_naive_agree():
    expr = parse("(sin(x)+cos(x))**2 + (sin(x)+cos(x))**3")
    cse = generate(expr, simplify=False, use_cse=True)
    naive = generate(expr, simplify=False, use_cse=False)
    x = np.linspace(0.1, 5.0, 50)
    assert np.allclose(cse(x), naive(x))
    # CSE powinien wprowadzić zmienną pomocniczą _c
    assert "_c0" in cse.source


def test_multivariate_codegen():
    art = generate(parse("x*y + x**2"))
    assert set(art.code.arg_names) == {"x", "y"}
    assert art.func(2.0, 3.0) == 2.0 * 3.0 + 2.0**2


def test_codegen_system_numeric_matches_numpy():
    A = M("[[2,1],[1,3]]")
    b = V("[3,5]")
    art = generate_system(A, b)
    assert "np.linalg.solve" in art.source
    expected = np.linalg.solve(
        np.array([[2, 1], [1, 3]], dtype=float), np.array([3, 5], dtype=float)
    )
    assert np.allclose(art(), expected)


def test_codegen_system_parametric_and_cse():
    A = M("[[a,0],[0,1]]")
    b = V("[1,1]")
    art = generate_system(A, b)
    assert set(art.code.arg_names) == {"a"}
    assert np.allclose(art(a=2.0), [0.5, 1.0])


# --- moduł 6: codegen ODE (RK4) -------------------------------------------
def test_emit_ode_source_structure():
    # powtórzone podwyrażenie wymusza zmienną CSE _c0
    g = emit_ode(parse("(sin(t)+cos(t))^2*y + (sin(t)+cos(t))^3*y"), "t", "y")
    assert "import numpy as np" in g.source
    assert "def solve_ode(y0, t0, t1, n):" in g.source
    assert "_c0" in g.source
    assert "k4" in g.source
    assert "return ts, ys" in g.source
    assert g.arg_names == ["y0", "t0", "t1", "n"]
    with pytest.raises(PycodemathError):  # parametr a bez wartości
        emit_ode(parse("a*y"), "t", "y")


def test_generate_ode_matches_solve_ode_num():
    art = generate_ode(parse("y*cos(t)"), "t", "y")
    ts_g, ys_g = art(1.0, 0.0, 1.0, 100)
    ts, ys = ode.solve_ode_num(parse("y*cos(t)"), "t", 1.0, (0.0, 1.0), 100)
    assert np.allclose(ts_g, ts)
    assert np.allclose(ys_g, ys)
    # rozwiązanie dokładne: y = exp(sin(t))
    assert ys_g[-1] == pytest.approx(math.exp(math.sin(1.0)), abs=1e-6)


# --- moduł 14: codegen adaptacyjnego solvera (DOPRI5) ---------------------
def test_emit_ode_adaptive_source_structure():
    from pycodemath.codegen.emit import emit_ode_adaptive

    # powtórzone podwyrażenie wymusza zmienną CSE _c0
    g = emit_ode_adaptive(
        parse("(sin(t)+cos(t))^2*y + (sin(t)+cos(t))^3*y"), "t", "y"
    )
    assert "import numpy as np" in g.source
    assert "def solve_ode_adaptive(y0, t0, t1, rtol=1e-6" in g.source
    assert "_c0" in g.source
    assert "k1 = ks[6]" in g.source  # FSAL jak w silniku
    assert "return ts, ys" in g.source
    assert g.arg_names == ["y0", "t0", "t1", "rtol", "atol", "max_steps"]
    with pytest.raises(PycodemathError):  # parametr a bez wartości
        emit_ode_adaptive(parse("a*y"), "t", "y")
    with pytest.raises(PycodemathError):  # nazwa zacienia alias NumPy
        emit_ode_adaptive(parse("y"), "t", "np")


def test_generate_ode_adaptive_parity_with_engine():
    # te same współczynniki i sterowanie krokiem co silnik — zmierzone:
    # identyczna liczba węzłów i max|Δt| = max|Δy| = 0.0 (parytet co do bitu)
    rhs = parse("y*cos(t)")
    art = generate_ode_adaptive(rhs, "t", "y")
    ts_g, ys_g = art(1.0, 0.0, 5.0, 1e-8, 1e-9)
    ts_e, ys_e = ode.solve_ode_adaptive(rhs, "t", 1.0, (0.0, 5.0), rtol=1e-8, atol=1e-9)
    assert len(ts_g) == len(ts_e)
    assert np.allclose(ts_g, ts_e) and np.allclose(ys_g, ys_e)
    assert ys_g[-1] == pytest.approx(math.exp(math.sin(5.0)), abs=1e-6)
    # domyślne tolerancje też zgodne
    _, yd = art(1.0, 0.0, 1.0)
    _, ye = ode.solve_ode_adaptive(rhs, "t", 1.0, (0.0, 1.0))
    assert yd[-1] == pytest.approx(ye[-1], abs=1e-12)


def test_generate_ode_adaptive_backward_parity():
    # wygenerowany kod dziedziczy całkowanie WSTECZNE po silniku (moduł 13)
    rhs = parse("y*cos(t)")
    art = generate_ode_adaptive(rhs, "t", "y")
    y5 = math.exp(math.sin(5.0))
    ts_g, ys_g = art(y5, 5.0, 0.0, 1e-8, 1e-9)
    ts_e, ys_e = ode.solve_ode_adaptive(rhs, "t", y5, (5.0, 0.0), rtol=1e-8, atol=1e-9)
    assert len(ts_g) == len(ts_e)
    assert np.allclose(ys_g, ys_e)
    # roundtrip: powrót do warunku początkowego (zmierzone 7.6e-9)
    assert ys_g[-1] == pytest.approx(1.0, abs=1e-6)


def test_generate_ode_adaptive_singularity_refuses():
    # y' = y², biegun w t=1: wygenerowany kod ODMAWIA czystym ValueError
    art = generate_ode_adaptive(parse("y^2"), "t", "y")
    with pytest.raises(ValueError, match="shrank|sample"):
        art(1.0, 0.0, 2.0)
    # przed biegunem liczy dokładnie (zmierzone: |y(0.9) - 10| = 7.9e-8)
    _, ys = art(1.0, 0.0, 0.9, 1e-8)
    assert ys[-1] == pytest.approx(10.0, abs=1e-5)


# --- moduł 18: codegen gęstego wyjścia (DOPRI5 + interpolant) --------------
def test_emit_ode_dense_source_structure():
    from pycodemath.codegen.emit import emit_ode_dense

    # powtórzone podwyrażenie wymusza zmienną CSE _c0
    g = emit_ode_dense(
        parse("(sin(t)+cos(t))^2*y + (sin(t)+cos(t))^3*y"), "t", "y"
    )
    assert "import numpy as np" in g.source
    assert "def solve_ode_dense(y0, t0, t1, rtol=1e-6" in g.source
    assert "_c0" in g.source
    assert "k1 = ks[6]" in g.source  # FSAL jak w silniku
    assert "_P = np.array" in g.source  # współczynniki interpolanta z silnika
    assert "return _sol" in g.source  # zwraca funkcję-interpolant
    assert g.arg_names == ["y0", "t0", "t1", "rtol", "atol", "max_steps"]
    with pytest.raises(PycodemathError):  # parametr a bez wartości
        emit_ode_dense(parse("a*y"), "t", "y")
    with pytest.raises(PycodemathError):  # nazwa zacienia alias NumPy
        emit_ode_dense(parse("y"), "t", "np")


def test_generate_ode_dense_parity_with_engine():
    # zmierzone: identyczne węzły (max|Δt| = max|Δy| = 0.0) ORAZ identyczny
    # interpolant na siatce 173 punktów (max|Δ| = 0.0) — parytet co do bitu
    rhs = parse("y*cos(t)")
    art = generate_ode_dense(rhs, "t", "y")
    sol_g = art(1.0, 0.0, 5.0, 1e-8, 1e-9)
    sol_e = ode.solve_ode_dense(rhs, "t", 1.0, (0.0, 5.0), rtol=1e-8, atol=1e-9)
    assert len(sol_g.ts) == len(sol_e.ts)
    assert np.allclose(sol_g.ts, sol_e.ts) and np.allclose(sol_g.ys, sol_e.ys)
    grid = np.linspace(0.0, 5.0, 173)
    assert np.allclose(sol_g(grid), sol_e(grid), atol=1e-14)
    # trafia w rozwiązanie dokładne między węzłami (zmierzone: 8.4e-8)
    assert np.allclose(sol_g(grid), np.exp(np.sin(grid)), atol=1e-6)
    # skalar -> float (jak DenseSolution silnika)
    assert isinstance(sol_g(1.234), float)


def test_generate_ode_dense_backward_and_refusals():
    # wstecz: parytet interpolanta odziedziczony (zmierzone: max|Δ| = 0.0)
    rhs = parse("y*cos(t)")
    art = generate_ode_dense(rhs, "t", "y")
    y5 = math.exp(math.sin(5.0))
    sol_g = art(y5, 5.0, 0.0, 1e-8, 1e-9)
    sol_e = ode.solve_ode_dense(rhs, "t", y5, (5.0, 0.0), rtol=1e-8, atol=1e-9)
    grid = np.linspace(0.0, 5.0, 91)
    assert len(sol_g.ts) == len(sol_e.ts)
    assert np.allclose(sol_g(grid), sol_e(grid), atol=1e-14)
    with pytest.raises(ValueError, match="outside the interval"):  # poza [t0, t1]
        sol_g(6.0)
    # osobliwość: przez biegun odmowa; przed biegunem dokładnie
    art2 = generate_ode_dense(parse("y^2"), "t", "y")
    with pytest.raises(ValueError, match="shrank|sample"):
        art2(1.0, 0.0, 2.0)
    s09 = art2(1.0, 0.0, 0.9, 1e-8)
    assert s09(0.9) == pytest.approx(10.0, abs=1e-5)  # zmierzone: 7.9e-8


# --- audyt przed dobudową: guardy codegen ----------------------------------
def test_emit_reserved_name_np_rejected():
    # symbol 'np' zacieniłby alias NumPy w wygenerowanym kodzie
    with pytest.raises(PycodemathError):
        generate(E("np**2 + sin(np)"))
    with pytest.raises(PycodemathError):
        emit_ode(parse("y"), "t", "np")
    with pytest.raises(PycodemathError):
        generate_system(M("[[np,0],[0,1]]"), V("[1,1]"))


def test_generate_ode_divergence_raises_clean_error():
    # y' = y^2, y(0)=1 wybucha w t=1 — wygenerowany kod odmawia jak silnik
    # (czysty ValueError z komunikatem, nie surowy OverflowError ani inf/NaN)
    art = generate_ode(parse("y^2"), "t", "y")
    with pytest.raises(ValueError, match="diverges|step"):
        art(1.0, 0.0, 2.0, 100)


# --- runda 2 (Blok B): parytet wyjątków dziedziny + brzegi loadera ----------
def test_emitted_domain_errors_match_engine():
    # antydryf: krotka except w EMITOWANYM kodzie musi być dokładnie tym,
    # co łapie silnik (numerics._DOMAIN_ERRORS) — w tym TypeError
    from pycodemath.codegen.emit import emit_ode_adaptive, emit_ode_dense
    from pycodemath.engine.numerics import _DOMAIN_ERRORS

    expected = "except (" + ", ".join(e.__name__ for e in _DOMAIN_ERRORS) + ")"
    for g in (
        emit_ode(parse("y"), "t", "y"),
        emit_ode_adaptive(parse("y"), "t", "y"),
        emit_ode_dense(parse("y"), "t", "y"),
    ):
        assert expected in g.source
        assert "TypeError" in g.source


def test_generated_adaptive_typeerror_is_clean_refusal():
    # y0=None → np.isfinite(None) rzuca TypeError WEWNĄTRZ emitowanego kodu;
    # kontrakt: czysty ValueError z komunikatem, nie surowy TypeError
    art = generate_ode_adaptive(parse("y"), "t", "y")
    with pytest.raises(ValueError, match="sample"):
        art(None, 0.0, 1.0)


def test_emit_solver_empty_matrix_no_forced_float():
    # all([]) == True — pusta macierz nie może wymusić dtype=float
    from pycodemath.codegen.emit import emit_solver

    g = emit_solver(M([]), V([]))
    assert "dtype=float" not in g.source


# --- runda fix: brzegi codegen ODE ------------------------------------------
def test_emit_ode_var_equals_func_is_pycodemath_error():
    # def _rhs(y, y): to zduplikowany argument = SyntaxError w artefakcie.
    # Silnik już to blokuje; codegen musi tak samo (kontrakt: PycodemathError,
    # nie surowy SyntaxError przy uruchomieniu skopiowanego kodu).
    from pycodemath.codegen.emit import emit_ode, emit_ode_adaptive, emit_ode_dense

    for emitter in (emit_ode, emit_ode_adaptive, emit_ode_dense):
        with pytest.raises(PycodemathError, match="differ"):
            emitter(parse("y"), var="y", func="y")
        # zdrowa para (var != func) nadal produkuje kompilowalne źródło
        src = emitter(parse("y"), var="t", func="y").source
        compile(src, "<ode>", "exec")  # brak SyntaxError


