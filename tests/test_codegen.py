"""Code generator tests: emit/pipeline, AES-GCM protection, ODE codegen.

Split of tests/test_pycodemath.py into per-area files (step 0 of the
development session) — test content moved without changes.
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
    # CSE should introduce the helper variable _c
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


# --- module 6: ODE codegen (RK4) -------------------------------------------
def test_emit_ode_source_structure():
    # repeated subexpression forces the CSE variable _c0
    g = emit_ode(parse("(sin(t)+cos(t))^2*y + (sin(t)+cos(t))^3*y"), "t", "y")
    assert "import numpy as np" in g.source
    assert "def solve_ode(y0, t0, t1, n):" in g.source
    assert "_c0" in g.source
    assert "k4" in g.source
    assert "return ts, ys" in g.source
    assert g.arg_names == ["y0", "t0", "t1", "n"]
    with pytest.raises(PycodemathError):  # parameter a without a value
        emit_ode(parse("a*y"), "t", "y")


def test_generate_ode_matches_solve_ode_num():
    art = generate_ode(parse("y*cos(t)"), "t", "y")
    ts_g, ys_g = art(1.0, 0.0, 1.0, 100)
    ts, ys = ode.solve_ode_num(parse("y*cos(t)"), "t", 1.0, (0.0, 1.0), 100)
    assert np.allclose(ts_g, ts)
    assert np.allclose(ys_g, ys)
    # exact solution: y = exp(sin(t))
    assert ys_g[-1] == pytest.approx(math.exp(math.sin(1.0)), abs=1e-6)


# --- module 14: adaptive solver codegen (DOPRI5) ---------------------
def test_emit_ode_adaptive_source_structure():
    from pycodemath.codegen.emit import emit_ode_adaptive

    # repeated subexpression forces the CSE variable _c0
    g = emit_ode_adaptive(
        parse("(sin(t)+cos(t))^2*y + (sin(t)+cos(t))^3*y"), "t", "y"
    )
    assert "import numpy as np" in g.source
    assert "def solve_ode_adaptive(y0, t0, t1, rtol=1e-6" in g.source
    assert "_c0" in g.source
    assert "k1 = ks[6]" in g.source  # FSAL as in the engine
    assert "return ts, ys" in g.source
    assert g.arg_names == ["y0", "t0", "t1", "rtol", "atol", "max_steps"]
    with pytest.raises(PycodemathError):  # parameter a without a value
        emit_ode_adaptive(parse("a*y"), "t", "y")
    with pytest.raises(PycodemathError):  # name shadows the NumPy alias
        emit_ode_adaptive(parse("y"), "t", "np")


def test_generate_ode_adaptive_parity_with_engine():
    # same coefficients and step control as the engine — measured:
    # identical number of nodes and max|Δt| = max|Δy| = 0.0 (bit-for-bit parity)
    rhs = parse("y*cos(t)")
    art = generate_ode_adaptive(rhs, "t", "y")
    ts_g, ys_g = art(1.0, 0.0, 5.0, 1e-8, 1e-9)
    ts_e, ys_e = ode.solve_ode_adaptive(rhs, "t", 1.0, (0.0, 5.0), rtol=1e-8, atol=1e-9)
    assert len(ts_g) == len(ts_e)
    assert np.allclose(ts_g, ts_e) and np.allclose(ys_g, ys_e)
    assert ys_g[-1] == pytest.approx(math.exp(math.sin(5.0)), abs=1e-6)
    # default tolerances agree too
    _, yd = art(1.0, 0.0, 1.0)
    _, ye = ode.solve_ode_adaptive(rhs, "t", 1.0, (0.0, 1.0))
    assert yd[-1] == pytest.approx(ye[-1], abs=1e-12)


def test_generate_ode_adaptive_backward_parity():
    # the generated code inherits BACKWARD integration from the engine (module 13)
    rhs = parse("y*cos(t)")
    art = generate_ode_adaptive(rhs, "t", "y")
    y5 = math.exp(math.sin(5.0))
    ts_g, ys_g = art(y5, 5.0, 0.0, 1e-8, 1e-9)
    ts_e, ys_e = ode.solve_ode_adaptive(rhs, "t", y5, (5.0, 0.0), rtol=1e-8, atol=1e-9)
    assert len(ts_g) == len(ts_e)
    assert np.allclose(ys_g, ys_e)
    # roundtrip: return to the initial condition (measured 7.6e-9)
    assert ys_g[-1] == pytest.approx(1.0, abs=1e-6)


def test_generate_ode_adaptive_singularity_refuses():
    # y' = y², pole at t=1: the generated code REFUSES with a clean ValueError
    art = generate_ode_adaptive(parse("y^2"), "t", "y")
    with pytest.raises(ValueError, match="shrank|sample"):
        art(1.0, 0.0, 2.0)
    # before the pole it computes accurately (measured: |y(0.9) - 10| = 7.9e-8)
    _, ys = art(1.0, 0.0, 0.9, 1e-8)
    assert ys[-1] == pytest.approx(10.0, abs=1e-5)


# --- module 18: dense-output codegen (DOPRI5 + interpolant) --------------
def test_emit_ode_dense_source_structure():
    from pycodemath.codegen.emit import emit_ode_dense

    # repeated subexpression forces the CSE variable _c0
    g = emit_ode_dense(
        parse("(sin(t)+cos(t))^2*y + (sin(t)+cos(t))^3*y"), "t", "y"
    )
    assert "import numpy as np" in g.source
    assert "def solve_ode_dense(y0, t0, t1, rtol=1e-6" in g.source
    assert "_c0" in g.source
    assert "k1 = ks[6]" in g.source  # FSAL as in the engine
    assert "_P = np.array" in g.source  # interpolant coefficients from the engine
    assert "return _sol" in g.source  # returns the interpolant function
    assert g.arg_names == ["y0", "t0", "t1", "rtol", "atol", "max_steps"]
    with pytest.raises(PycodemathError):  # parameter a without a value
        emit_ode_dense(parse("a*y"), "t", "y")
    with pytest.raises(PycodemathError):  # name shadows the NumPy alias
        emit_ode_dense(parse("y"), "t", "np")


def test_generate_ode_dense_parity_with_engine():
    # measured: identical nodes (max|Δt| = max|Δy| = 0.0) AND identical
    # interpolant on a grid of 173 points (max|Δ| = 0.0) — bit-for-bit parity
    rhs = parse("y*cos(t)")
    art = generate_ode_dense(rhs, "t", "y")
    sol_g = art(1.0, 0.0, 5.0, 1e-8, 1e-9)
    sol_e = ode.solve_ode_dense(rhs, "t", 1.0, (0.0, 5.0), rtol=1e-8, atol=1e-9)
    assert len(sol_g.ts) == len(sol_e.ts)
    assert np.allclose(sol_g.ts, sol_e.ts) and np.allclose(sol_g.ys, sol_e.ys)
    grid = np.linspace(0.0, 5.0, 173)
    assert np.allclose(sol_g(grid), sol_e(grid), atol=1e-14)
    # hits the exact solution between nodes (measured: 8.4e-8)
    assert np.allclose(sol_g(grid), np.exp(np.sin(grid)), atol=1e-6)
    # scalar -> float (like the engine's DenseSolution)
    assert isinstance(sol_g(1.234), float)


def test_generate_ode_dense_backward_and_refusals():
    # backward: interpolant parity inherited (measured: max|Δ| = 0.0)
    rhs = parse("y*cos(t)")
    art = generate_ode_dense(rhs, "t", "y")
    y5 = math.exp(math.sin(5.0))
    sol_g = art(y5, 5.0, 0.0, 1e-8, 1e-9)
    sol_e = ode.solve_ode_dense(rhs, "t", y5, (5.0, 0.0), rtol=1e-8, atol=1e-9)
    grid = np.linspace(0.0, 5.0, 91)
    assert len(sol_g.ts) == len(sol_e.ts)
    assert np.allclose(sol_g(grid), sol_e(grid), atol=1e-14)
    with pytest.raises(ValueError, match="outside the interval"):  # outside [t0, t1]
        sol_g(6.0)
    # singularity: refusal through the pole; accurate before the pole
    art2 = generate_ode_dense(parse("y^2"), "t", "y")
    with pytest.raises(ValueError, match="shrank|sample"):
        art2(1.0, 0.0, 2.0)
    s09 = art2(1.0, 0.0, 0.9, 1e-8)
    assert s09(0.9) == pytest.approx(10.0, abs=1e-5)  # measured: 7.9e-8


# --- pre-extension audit: codegen guards ----------------------------------
def test_emit_reserved_name_np_rejected():
    # the symbol 'np' would shadow the NumPy alias in the generated code
    with pytest.raises(PycodemathError):
        generate(E("np**2 + sin(np)"))
    with pytest.raises(PycodemathError):
        emit_ode(parse("y"), "t", "np")
    with pytest.raises(PycodemathError):
        generate_system(M("[[np,0],[0,1]]"), V("[1,1]"))


def test_generate_ode_divergence_raises_clean_error():
    # y' = y^2, y(0)=1 blows up at t=1 — the generated code refuses like the engine
    # (clean ValueError with a message, not a raw OverflowError or inf/NaN)
    art = generate_ode(parse("y^2"), "t", "y")
    with pytest.raises(ValueError, match="diverges|step"):
        art(1.0, 0.0, 2.0, 100)


# --- round 2 (Block B): domain-exception parity + loader edges ----------
def test_emitted_domain_errors_match_engine():
    # anti-drift: the except tuple in the EMITTED code must be exactly what
    # the engine catches (numerics._DOMAIN_ERRORS) — including TypeError
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
    # y0=None → np.isfinite(None) raises TypeError INSIDE the emitted code;
    # contract: clean ValueError with a message, not a raw TypeError
    art = generate_ode_adaptive(parse("y"), "t", "y")
    with pytest.raises(ValueError, match="sample"):
        art(None, 0.0, 1.0)


def test_emit_solver_empty_matrix_no_forced_float():
    # all([]) == True — an empty matrix must not force dtype=float
    from pycodemath.codegen.emit import emit_solver

    g = emit_solver(M([]), V([]))
    assert "dtype=float" not in g.source


# --- fix round: ODE codegen edges + raw cause of license refusal -------
def test_emit_ode_var_equals_func_is_pycodemath_error():
    # def _rhs(y, y): is a duplicate argument = SyntaxError in the artifact.
    # The engine already blocks this; codegen must do the same (contract:
    # PycodemathError, not a raw SyntaxError when running the copied code).
    from pycodemath.codegen.emit import emit_ode, emit_ode_adaptive, emit_ode_dense

    for emitter in (emit_ode, emit_ode_adaptive, emit_ode_dense):
        with pytest.raises(PycodemathError, match="differ"):
            emitter(parse("y"), var="y", func="y")
        # a healthy pair (var != func) still produces compilable source
        src = emitter(parse("y"), var="t", func="y").source
        compile(src, "<ode>", "exec")  # no SyntaxError


