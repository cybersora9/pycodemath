"""Tests for the secure parser (MODULE P1): sympify whitelist.

The parser must resolve ONLY a whitelist of mathematical functions; Python
builtins (__import__/eval/open), string literals, and attribute access must
be unreachable from the input text. Expected values are built from bare
sympy (independently of pycodemath) — no tautologies.
"""

from __future__ import annotations

import sympy as sp
import pytest

from pycodemath import parse
from pycodemath.core.errors import PycodemathError

x = sp.Symbol("x")


# --- ordinary math is NOT affected ---------------------------------------

def test_whitelist_math_still_works():
    # expectation from bare sympy: sin(x)*x + 2^3 = x*sin(x) + 8
    assert parse("sin(x)*x+2^3").sy == sp.sin(x) * x + 8


def test_implicit_multiplication_still_works():
    assert parse("2x").sy == 2 * x


def test_roundtrip_preserved():
    e = parse("sin(x)*x + 2*x**3")
    assert parse(str(e)).equivalent(e)


def test_whitelist_aliases_and_constants():
    # aliases for informal notation (previously captured by Python builtins)
    assert parse("abs(x)").sy == sp.Abs(x)
    assert parse("ln(x)").sy == sp.log(x)
    # constants
    assert parse("pi").sy == sp.pi
    assert parse("E").sy == sp.E
    assert parse("oo").sy == sp.oo


def test_factorial_notation():
    # 5! = 120 (closed form, computed by hand)
    assert parse("5!").sy == sp.Integer(120)


def test_decimal_point_is_not_attribute_access():
    # a decimal point is a NUMBER, not the OP "." — it does not trip the guard
    assert parse("1.5 + .3x").sy == sp.Float("0.3") * x + sp.Float("1.5")


# --- Python builtins are unresolvable -------------------------------------

def test_import_with_string_arg_is_rejected():
    with pytest.raises(PycodemathError):
        parse("__import__('os')")


def test_import_with_symbol_arg_is_not_python_call():
    # __import__ does NOT resolve to a builtin — it stays an ordinary symbol
    # in a product (implicit multiplication), zero Python code execution.
    e = parse("__import__(x)")
    assert e.sy == sp.Symbol("__import__") * x
    assert isinstance(e.sy, sp.Expr)


def test_open_and_eval_are_rejected():
    with pytest.raises(PycodemathError):
        parse("open('f')")
    with pytest.raises(PycodemathError):
        parse("eval('1+1')")


def test_string_literal_is_rejected():
    # a string must NOT slip out of the parser (it would reach sympify with the
    # full namespace in Expr.__init__) — rejected at the token level
    with pytest.raises(PycodemathError):
        parse("'sin(3)'")


def test_attribute_access_is_rejected():
    with pytest.raises(PycodemathError):
        parse("x.__class__")
    # the classic eval escape path — cut off at the first dot
    with pytest.raises(PycodemathError):
        parse("x.__class__.__mro__[1].__subclasses__()")


# --- unknown names: symbols, not code -------------------------------------

def test_unknown_single_letter_function_is_multiplication():
    # documented behavior of implicit_multiplication_application:
    # f(x) reads as f*x (f = symbol), without any Python eval
    assert parse("f(x)").sy == sp.Symbol("f") * x


def test_unknown_multiletter_name_splits_into_symbols():
    # split_symbols (the price of 2x notation): foo(x) -> f*o**2*x; still pure
    # math on symbols, zero Python code
    f, o = sp.Symbol("f"), sp.Symbol("o")
    assert parse("foo(x)").sy == f * o**2 * x


def test_underscore_name_stays_single_symbol():
    assert parse("x_1 + 2").sy == sp.Symbol("x_1") + 2


def test_syntax_error_still_readable():
    with pytest.raises(PycodemathError):
        parse("sin(")


# --- evaluation-cost guard (DoS, MODULE P-fix) ---------------------------
# The whitelist permits exponentiation and factorial/binomial — and on concrete
# numbers these build a gigantic int (reachable directly from the MCP server).
# The parser refuses BEFORE SymPy computes it: a readable PycodemathError, not
# an OOM/hang.

@pytest.mark.parametrize(
    "src",
    [
        "9**9**9",  # ~1.2 billion bits
        "9**9**9**9",
        "2^9**9**9",
        "factorial(10**8)",
        "factorial(100000000)",
        "gamma(10**8)",
        "binomial(10**9, 500000000)",
        "factorial(factorial(20))",  # nested factorial
        "sin(9**9**9)",  # a dangerous fragment hidden inside a function argument
        "10**500000",
    ],
)
def test_parser_rejects_evaluation_bombs(src):
    import time

    start = time.monotonic()
    with pytest.raises(PycodemathError):
        parse(src)
    # key point: the refusal is IMMEDIATE (we do not compute the bomb)
    assert time.monotonic() - start < 1.0


@pytest.mark.parametrize(
    "src, expected",
    [
        ("2^10", sp.Integer(1024)),
        ("factorial(10)", sp.Integer(3628800)),
        ("binomial(10, 3)", sp.Integer(120)),
        ("9**9", sp.Integer(387420489)),
        ("2^1000", sp.Integer(2) ** 1000),
    ],
)
def test_guard_lets_ordinary_powers_and_factorials_through(src, expected):
    # the guard must NOT reject reasonable math
    assert parse(src).sy == expected
