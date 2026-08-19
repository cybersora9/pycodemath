"""Testy rdzenia: IR (Expr/Matrix), parser, silnik symboliczny, algebra liniowa.

Rozbicie tests/test_pycodemath.py na pliki per obszar (krok 0 sesji
rozwojowej) — treść testów przeniesiona bez zmian.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycodemath import E, M, V, parse
from pycodemath.cli import repl
from pycodemath.core.errors import (
    DivergenceError,
    ParseError,
    PycodemathError,
)
from pycodemath.core.ir import Matrix
from pycodemath.engine import linalg, ode, symbolic


def test_parser_roundtrip():
    src = "sin(x)*x + 2*x**3"
    e = parse(src)
    # round-trip: ponowne sparsowanie tekstowej postaci daje równoważne wyrażenie
    assert parse(str(e)).equivalent(e)


def test_parser_caret_is_power():
    assert parse("2^3").equivalent(parse("8"))


def test_diff():
    # d/dx [x^2] = 2x
    assert symbolic.diff(parse("x^2"), "x").equivalent(parse("2*x"))


def test_diff_abs_reduces_to_sign():
    # A bare (assumption-free) sympy Symbol is not known to be real, so
    # Abs(y).diff(y) does not reduce to sign(y) by default — it stays an
    # unevaluated Derivative(re(y), y): silently wrong from `diff` itself,
    # and (via numerics.gradient/jacobian, which differentiate the same way
    # for min/min_nd/solve_nd/odestiff) a raw sympy
    # PrintMethodNotImplementedError instead of a PycodemathError from any
    # command that differentiates an Abs()-containing expression internally.
    # d/dy |y| = sign(y) for real y.
    assert symbolic.diff(parse("Abs(y)"), "y").equivalent(parse("sign(y)"))


def test_diff_abs_does_not_crash_stiff_jacobian():
    # The concrete failure this guarded against: odestiff's Jacobian
    # differentiates the right-hand side internally, so any Abs()-based
    # dynamics (e.g. quadratic drag on a falling body) used to crash with a
    # raw sympy PrintMethodNotImplementedError instead of computing.
    ts, ys = ode.solve_ode_stiff(
        parse("-9.81-0.01*y*Abs(y)"), "t", 0.0, (0.0, 3.0), 3000, func="y"
    )
    # Cross-checked independently against scipy.integrate.solve_ivp on the
    # same dv/dt = -g - k*v*|v| at t=3s: -23.02247607563...
    assert ys[-1] == pytest.approx(-23.02247607563401, rel=1e-6)


def test_solve():
    roots = symbolic.solve(parse("x^2 - 4"), "x")
    vals = sorted(int(str(r)) for r in roots)
    assert vals == [-2, 2]


def test_simplify():
    assert symbolic.simplify(parse("sin(x)^2 + cos(x)^2")).equivalent(parse("1"))


def test_eq_is_structural_and_hash_consistent():
    # Wyrażenia równoważne matematycznie, ale o różnej strukturze:
    a = parse("(x+1)**2")
    b = parse("x**2 + 2*x + 1")
    assert a.equivalent(b)          # równość matematyczna
    assert a != b                   # ale NIE strukturalna (== jest strukturalne)
    # Kontrakt hash/eq: równe strukturalnie -> równy hash -> działa w set/dict.
    c = parse("x**2 + 2*x + 1")
    assert b == c
    assert hash(b) == hash(c)
    assert len({b, c}) == 1


def test_to_source_roundtrip_equivalent():
    e = parse("sin(x)*x + 2*x**3")
    # str()/to_source() odtwarza się parserem do wyrażenia równoważnego
    assert parse(e.to_source()).equivalent(e)


# --- moduł P2: limit / series / summation --------------------------------
def test_limit_basic():
    # klasyk analizy: lim sin(x)/x przy x->0 = 1 (forma zamknięta)
    assert symbolic.limit(parse("sin(x)/x"), "x", 0).equivalent(parse("1"))


def test_limit_one_sided():
    # 1/x przy 0+ i 0- — granice jednostronne różnych znaków
    assert str(symbolic.limit(parse("1/x"), "x", 0, dir="+")) == "oo"
    assert str(symbolic.limit(parse("1/x"), "x", 0, dir="-")) == "-oo"


def test_limit_at_infinity():
    # definicja liczby e: lim (1+1/n)^n przy n->oo
    assert symbolic.limit(parse("(1+1/n)^n"), "n", "oo").equivalent(parse("E"))


def test_limit_nonexistent_oscillation():
    # sin(x) przy oo oscyluje — granica nie istnieje, ma być odmowa
    with pytest.raises(PycodemathError):
        symbolic.limit(parse("sin(x)"), "x", "oo")


def test_limit_bad_dir():
    with pytest.raises(PycodemathError):
        symbolic.limit(parse("1/x"), "x", 0, dir="+-")


def test_series_exp():
    # szereg Maclaurina exp: 1 + x + x^2/2 + x^3/6 (n=4 obcina na x^3)
    s = symbolic.series(parse("exp(x)"), "x", at=0, n=4)
    assert s.equivalent(parse("1 + x + x^2/2 + x^3/6"))


def test_series_geometric():
    # 1/(1-x) = 1 + x + x^2 + x^3 + ... (szereg geometryczny)
    s = symbolic.series(parse("1/(1-x)"), "x", n=4)
    assert s.equivalent(parse("1 + x + x^2 + x^3"))


def test_series_bad_n():
    with pytest.raises(PycodemathError):
        symbolic.series(parse("exp(x)"), "x", n=0)


def test_summation_symbolic_bound():
    # wzór Gaussa: sum k, k=1..n = n(n+1)/2
    s = symbolic.summation(parse("k"), "k", 1, "n")
    assert s.equivalent(parse("n*(n+1)/2"))


def test_summation_finite():
    # n(n+1)(2n+1)/6 dla n=10 daje 385
    assert symbolic.summation(parse("k^2"), "k", 1, 10).equivalent(parse("385"))


def test_summation_infinite_convergent():
    # problem bazylejski: sum 1/k^2, k=1..oo = pi^2/6
    s = symbolic.summation(parse("1/k^2"), "k", 1, "oo")
    assert s.equivalent(parse("pi^2/6"))


def test_summation_divergent_raises():
    with pytest.raises(DivergenceError, match="divergent"):
        symbolic.summation(parse("k"), "k", 1, "oo")


def test_summation_no_closed_form_raises():
    # suma ZBIEŻNA (porównanie z 1/k^3), ale bez formy zamkniętej —
    # komunikat musi mówić o braku formy, nie o rozbieżności
    with pytest.raises(PycodemathError, match="closed form"):
        symbolic.summation(parse("1/(k^3+k+1)"), "k", 1, "oo")


def test_repl_limit_series_sum():
    assert repl.handle("limit sin(x)/x for x to 0") == "1"
    assert repl.handle("limit 1/x for x to 0 dir -") == "-oo"
    assert repl.handle("sum k for k from 1 to 10") == "55"
    out = repl.handle("series exp(x) for x n 4")
    assert parse(out).equivalent(parse("1 + x + x^2/2 + x^3/6"))
    # zła składnia -> komunikat użycia, nie crash
    assert "Usage" in repl.handle("limit sin(x)/x")


# --- moduł 3: algebra liniowa ------------------------------------------
def test_matrix_ir_and_equality():
    A = M("[[1,2],[3,4]]")
    assert A.shape == (2, 2)
    # == jest STRUKTURALNE (spójne z hash), jak dla Expr
    assert A == M("[[1,2],[3,4]]")
    assert A != M("[[1,2],[3,5]]")
    # round-trip literału przez IR
    assert M(A.to_source()) == A


def test_transpose_and_multiply():
    A = M("[[1,2],[3,4]]")
    assert linalg.transpose(A) == M("[[1,3],[2,4]]")
    identity = M("[[1,0],[0,1]]")
    assert linalg.multiply(A, identity).equivalent(A)


def test_determinant_and_inverse():
    A = M("[[4,7],[2,6]]")
    assert linalg.det(A) == E("10")
    inv = linalg.inv(A)
    # A · A⁻¹ = I — równość MATEMATYCZNA iloczynu (nie strukturalna)
    assert linalg.multiply(A, inv).equivalent(M("[[1,0],[0,1]]"))


def test_inverse_numeric_fallback():
    A = M("[[4,7],[2,6]]")
    inv = linalg.inv(A, numeric=True)
    prod = np.array(linalg.multiply(A, inv).sy.evalf().tolist(), dtype=float)
    assert np.allclose(prod, np.eye(2))


def test_solve_system_symbolic():
    A = M("[[2,1],[1,3]]")
    b = V("[3,5]")
    x = linalg.solve_system(A, b)
    # rozwiązanie musi spełniać A x = b
    assert linalg.multiply(A, x).equivalent(b)


def test_solve_system_parametric():
    # układ z parametrem: [[a,0],[0,1]] x = [1,1]  ->  x = [1/a, 1]
    A = M("[[a,0],[0,1]]")
    b = V("[1,1]")
    x = linalg.solve_system(A, b)
    assert x == V("[1/a, 1]")


def test_eigenvalues_symbolic_and_numeric():
    A = M("[[2,0],[0,3]]")
    sym = sorted(int(str(v)) for v in linalg.eigenvalues(A))
    assert sym == [2, 3]
    num = sorted(round(float(str(v))) for v in linalg.eigenvalues(A, numeric=True))
    assert num == [2, 3]


def test_eigenvectors_satisfy_definition():
    A = M("[[2,0],[0,3]]")
    pairs = linalg.eigenvectors(A)
    assert len(pairs) == 2
    for val, vec in pairs:
        # A v = lambda v — równość matematyczna
        assert linalg.multiply(A, vec).equivalent(Matrix(val.sy * vec.sy))


def test_matrix_eq_structural_and_hashable():
    # == strukturalne; .equivalent() łapie równoważność matematyczną
    a = M("[[1, 2], [0, 1]]")
    b = M("[[1, 1+1], [0, 1]]")  # równoważna, ale inny zapis wpisu
    assert a.equivalent(b)
    assert a == M("[[1,2],[0,1]]")  # identyczny zapis -> strukturalnie równe
    # hashowalna i spójna z ==: równe strukturalnie -> jeden element w set
    assert len({a, M("[[1,2],[0,1]]")}) == 1


def test_numeric_flag_rejects_symbolic():
    A = M("[[a,0],[0,1]]")
    with pytest.raises(PycodemathError):
        linalg.eigenvalues(A, numeric=True)
    with pytest.raises(PycodemathError):
        linalg.inv(A, numeric=True)


def test_solve_least_squares_overdetermined():
    # układ nadokreślony 3x2 — rozwiązanie w sensie najmniejszych kwadratów
    A = M("[[1,1],[2,1],[1,2]]")
    b = V("[3,3,4]")
    x = np.array(linalg.solve_system(A, b).sy.evalf().tolist(), dtype=float).reshape(-1)
    expected = np.linalg.lstsq(
        np.array([[1, 1], [2, 1], [1, 2]], dtype=float),
        np.array([3, 3, 4], dtype=float),
        rcond=None,
    )[0]
    assert np.allclose(x, expected)


# --- audyt przed dobudową: kontrakt PycodemathError ------------------------
def test_matrix_bad_literal_raises_pycodemath():
    # wiersze różnej długości: literał tekstowy i lista — oba jako PycodemathError
    with pytest.raises(ParseError):
        Matrix("[[1,2],[3]]")
    with pytest.raises(ParseError):
        Matrix([[1, 2], [3]])
    with pytest.raises(PycodemathError):
        repl.handle("det [[1,2],[3]]")


# --- runda 2 (Blok D): subs przez strzeżony parser --------------------------
def test_subs_string_via_whitelisted_parser():
    # string-wartość rozwiązuje funkcję z whitelisty parsera
    e = parse("x + 1")
    assert e.subs({"x": "sin(y)"}).equivalent(parse("sin(y) + 1"))


def test_subs_malicious_string_denied():
    # literał tekstowy w wartości → odmowa strażnika tokenów (jak w parserze),
    # a nie druga powierzchnia sympify z pełnym namespace
    with pytest.raises(ParseError):
        parse("x").subs({"x": "__import__('os')"})


def test_subs_numbers_unchanged():
    # liczby idą jak dotąd (regresja: subs nie zepsuł się dla int/float)
    assert parse("x^2").subs({"x": 3}).equivalent(parse("9"))
    assert float(str(parse("x").subs({"x": 2.5}))) == 2.5


def test_eigenvectors_numeric_fallback():
    A = M("[[2,0],[0,3]]")
    pairs = linalg.eigenvectors(A, numeric=True)
    assert len(pairs) == 2
    An = np.array([[2, 0], [0, 3]], dtype=float)
    for val, vec in pairs:
        lam = float(val.sy)
        v = np.array(vec.sy.tolist(), dtype=float).reshape(-1)
        assert np.allclose(An @ v, lam * v)
