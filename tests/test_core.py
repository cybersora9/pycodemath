"""Core tests: IR (Expr/Matrix), parser, symbolic engine, linear algebra.

Split of tests/test_pycodemath.py into files per area (step 0 of the
development session) — test content moved without changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycodemath import E, M, V, parse
from pycodemath.cli import repl
from pycodemath.core.errors import PycodemathError
from pycodemath.core.ir import Matrix
from pycodemath.engine import linalg, symbolic


def test_parser_roundtrip():
    src = "sin(x)*x + 2*x**3"
    e = parse(src)
    # round-trip: re-parsing the textual form yields an equivalent expression
    assert parse(str(e)).equivalent(e)


def test_parser_caret_is_power():
    assert parse("2^3").equivalent(parse("8"))


def test_diff():
    # d/dx [x^2] = 2x
    assert symbolic.diff(parse("x^2"), "x").equivalent(parse("2*x"))


def test_solve():
    roots = symbolic.solve(parse("x^2 - 4"), "x")
    vals = sorted(int(str(r)) for r in roots)
    assert vals == [-2, 2]


def test_simplify():
    assert symbolic.simplify(parse("sin(x)^2 + cos(x)^2")).equivalent(parse("1"))


def test_eq_is_structural_and_hash_consistent():
    # Expressions that are mathematically equivalent but structurally different:
    a = parse("(x+1)**2")
    b = parse("x**2 + 2*x + 1")
    assert a.equivalent(b)          # mathematical equality
    assert a != b                   # but NOT structural (== is structural)
    # hash/eq contract: structurally equal -> equal hash -> works in set/dict.
    c = parse("x**2 + 2*x + 1")
    assert b == c
    assert hash(b) == hash(c)
    assert len({b, c}) == 1


def test_to_source_roundtrip_equivalent():
    e = parse("sin(x)*x + 2*x**3")
    # str()/to_source() reconstructs via the parser into an equivalent expression
    assert parse(e.to_source()).equivalent(e)


# --- module P2: limit / series / summation --------------------------------
def test_limit_basic():
    # calculus classic: lim sin(x)/x as x->0 = 1 (closed form)
    assert symbolic.limit(parse("sin(x)/x"), "x", 0).equivalent(parse("1"))


def test_limit_one_sided():
    # 1/x at 0+ and 0- — one-sided limits of different signs
    assert str(symbolic.limit(parse("1/x"), "x", 0, dir="+")) == "oo"
    assert str(symbolic.limit(parse("1/x"), "x", 0, dir="-")) == "-oo"


def test_limit_at_infinity():
    # definition of the number e: lim (1+1/n)^n as n->oo
    assert symbolic.limit(parse("(1+1/n)^n"), "n", "oo").equivalent(parse("E"))


def test_limit_nonexistent_oscillation():
    # sin(x) at oo oscillates — the limit does not exist, so it must be refused
    with pytest.raises(PycodemathError):
        symbolic.limit(parse("sin(x)"), "x", "oo")


def test_limit_bad_dir():
    with pytest.raises(PycodemathError):
        symbolic.limit(parse("1/x"), "x", 0, dir="+-")


def test_series_exp():
    # Maclaurin series of exp: 1 + x + x^2/2 + x^3/6 (n=4 truncates at x^3)
    s = symbolic.series(parse("exp(x)"), "x", at=0, n=4)
    assert s.equivalent(parse("1 + x + x^2/2 + x^3/6"))


def test_series_geometric():
    # 1/(1-x) = 1 + x + x^2 + x^3 + ... (geometric series)
    s = symbolic.series(parse("1/(1-x)"), "x", n=4)
    assert s.equivalent(parse("1 + x + x^2 + x^3"))


def test_series_bad_n():
    with pytest.raises(PycodemathError):
        symbolic.series(parse("exp(x)"), "x", n=0)


def test_summation_symbolic_bound():
    # Gauss formula: sum k, k=1..n = n(n+1)/2
    s = symbolic.summation(parse("k"), "k", 1, "n")
    assert s.equivalent(parse("n*(n+1)/2"))


def test_summation_finite():
    # n(n+1)(2n+1)/6 for n=10 gives 385
    assert symbolic.summation(parse("k^2"), "k", 1, 10).equivalent(parse("385"))


def test_summation_infinite_convergent():
    # Basel problem: sum 1/k^2, k=1..oo = pi^2/6
    s = symbolic.summation(parse("1/k^2"), "k", 1, "oo")
    assert s.equivalent(parse("pi^2/6"))


def test_summation_divergent_raises():
    with pytest.raises(PycodemathError, match="divergent"):
        symbolic.summation(parse("k"), "k", 1, "oo")


def test_summation_no_closed_form_raises():
    # CONVERGENT sum (comparison with 1/k^3), but with no closed form —
    # the message must mention the missing form, not divergence
    with pytest.raises(PycodemathError, match="closed form"):
        symbolic.summation(parse("1/(k^3+k+1)"), "k", 1, "oo")


def test_repl_limit_series_sum():
    assert repl.handle("limit sin(x)/x for x to 0") == "1"
    assert repl.handle("limit 1/x for x to 0 dir -") == "-oo"
    assert repl.handle("sum k for k from 1 to 10") == "55"
    out = repl.handle("series exp(x) for x n 4")
    assert parse(out).equivalent(parse("1 + x + x^2/2 + x^3/6"))
    # bad syntax -> usage message, not a crash
    assert "Usage" in repl.handle("limit sin(x)/x")


# --- module 3: linear algebra ------------------------------------------
def test_matrix_ir_and_equality():
    A = M("[[1,2],[3,4]]")
    assert A.shape == (2, 2)
    # == is STRUCTURAL (consistent with hash), as for Expr
    assert A == M("[[1,2],[3,4]]")
    assert A != M("[[1,2],[3,5]]")
    # round-trip of the literal through IR
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
    # A · A⁻¹ = I — MATHEMATICAL equality of the product (not structural)
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
    # the solution must satisfy A x = b
    assert linalg.multiply(A, x).equivalent(b)


def test_solve_system_parametric():
    # system with a parameter: [[a,0],[0,1]] x = [1,1]  ->  x = [1/a, 1]
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
        # A v = lambda v — mathematical equality
        assert linalg.multiply(A, vec).equivalent(Matrix(val.sy * vec.sy))


def test_matrix_eq_structural_and_hashable():
    # == structural; .equivalent() catches mathematical equivalence
    a = M("[[1, 2], [0, 1]]")
    b = M("[[1, 1+1], [0, 1]]")  # equivalent, but a different entry notation
    assert a.equivalent(b)
    assert a == M("[[1,2],[0,1]]")  # identical notation -> structurally equal
    # hashable and consistent with ==: structurally equal -> one element in a set
    assert len({a, M("[[1,2],[0,1]]")}) == 1


def test_numeric_flag_rejects_symbolic():
    A = M("[[a,0],[0,1]]")
    with pytest.raises(PycodemathError):
        linalg.eigenvalues(A, numeric=True)
    with pytest.raises(PycodemathError):
        linalg.inv(A, numeric=True)


def test_solve_least_squares_overdetermined():
    # overdetermined 3x2 system — least-squares solution
    A = M("[[1,1],[2,1],[1,2]]")
    b = V("[3,3,4]")
    x = np.array(linalg.solve_system(A, b).sy.evalf().tolist(), dtype=float).reshape(-1)
    expected = np.linalg.lstsq(
        np.array([[1, 1], [2, 1], [1, 2]], dtype=float),
        np.array([3, 3, 4], dtype=float),
        rcond=None,
    )[0]
    assert np.allclose(x, expected)


# --- audit before extension: PycodemathError contract ------------------------
def test_matrix_bad_literal_raises_pycodemath():
    # rows of different lengths: text literal and list — both as PycodemathError
    with pytest.raises(PycodemathError):
        Matrix("[[1,2],[3]]")
    with pytest.raises(PycodemathError):
        Matrix([[1, 2], [3]])
    with pytest.raises(PycodemathError):
        repl.handle("det [[1,2],[3]]")


# --- round 2 (Block D): subs via the guarded parser --------------------------
def test_subs_string_via_whitelisted_parser():
    # a string value resolves a function from the parser's whitelist
    e = parse("x + 1")
    assert e.subs({"x": "sin(y)"}).equivalent(parse("sin(y) + 1"))


def test_subs_malicious_string_denied():
    # a text literal in the value → refusal by the token guard (as in the parser),
    # and not a second sympify surface with the full namespace
    with pytest.raises(PycodemathError):
        parse("x").subs({"x": "__import__('os')"})


def test_subs_numbers_unchanged():
    # numbers go through as before (regression: subs did not break for int/float)
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
