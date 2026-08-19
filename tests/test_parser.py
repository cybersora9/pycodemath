"""Testy bezpiecznego parsera (MODUŁ P1): whitelist sympify.

Parser ma rozwiązywać TYLKO whitelistę funkcji matematycznych; builtiny
Pythona (__import__/eval/open), literały tekstowe i dostęp atrybutowy mają
być nieosiągalne z tekstu wejściowego. Oczekiwane wartości budowane z gołego
sympy (niezależnie od pycodemath) — bez tautologii.
"""

from __future__ import annotations

import sympy as sp
import pytest

from pycodemath import parse
from pycodemath.core.errors import ParseError

x = sp.Symbol("x")


# --- zwykła matematyka NIE ucierpiała ------------------------------------

def test_whitelist_math_still_works():
    # oczekiwanie z gołego sympy: sin(x)*x + 2^3 = x*sin(x) + 8
    assert parse("sin(x)*x+2^3").sy == sp.sin(x) * x + 8


def test_implicit_multiplication_still_works():
    assert parse("2x").sy == 2 * x


def test_roundtrip_preserved():
    e = parse("sin(x)*x + 2*x**3")
    assert parse(str(e)).equivalent(e)


def test_whitelist_aliases_and_constants():
    # aliasy potocznej notacji (wcześniej łapały je builtiny Pythona)
    assert parse("abs(x)").sy == sp.Abs(x)
    assert parse("ln(x)").sy == sp.log(x)
    # stałe
    assert parse("pi").sy == sp.pi
    assert parse("E").sy == sp.E
    assert parse("oo").sy == sp.oo


def test_factorial_notation():
    # 5! = 120 (forma zamknięta, ręcznie)
    assert parse("5!").sy == sp.Integer(120)


def test_decimal_point_is_not_attribute_access():
    # kropka dziesiętna to NUMBER, nie OP "." — nie wpada w strażnika
    assert parse("1.5 + .3x").sy == sp.Float("0.3") * x + sp.Float("1.5")


# --- builtiny Pythona nierozwiązywalne ------------------------------------

def test_import_with_string_arg_is_rejected():
    with pytest.raises(ParseError):
        parse("__import__('os')")


def test_import_with_symbol_arg_is_not_python_call():
    # __import__ NIE rozwiązuje się do builtina — zostaje zwykłym symbolem
    # w iloczynie (niejawne mnożenie), zero wywołania kodu Pythona.
    e = parse("__import__(x)")
    assert e.sy == sp.Symbol("__import__") * x
    assert isinstance(e.sy, sp.Expr)


def test_open_and_eval_are_rejected():
    with pytest.raises(ParseError):
        parse("open('f')")
    with pytest.raises(ParseError):
        parse("eval('1+1')")


def test_string_literal_is_rejected():
    # łańcuch NIE może wypaść z parsera (wpadłby do sympify z pełnym
    # namespace w Expr.__init__) — odmowa na poziomie tokenów
    with pytest.raises(ParseError):
        parse("'sin(3)'")


def test_attribute_access_is_rejected():
    with pytest.raises(ParseError):
        parse("x.__class__")
    # klasyczna ścieżka ucieczki z eval — ucięta na pierwszej kropce
    with pytest.raises(ParseError):
        parse("x.__class__.__mro__[1].__subclasses__()")


# --- nieznane nazwy: symbole, nie kod -------------------------------------

def test_unknown_single_letter_function_is_multiplication():
    # udokumentowane zachowanie implicit_multiplication_application:
    # f(x) czyta się jako f*x (f = symbol), bez evalu Pythona
    assert parse("f(x)").sy == sp.Symbol("f") * x


def test_unknown_multiletter_name_splits_into_symbols():
    # split_symbols (cena notacji 2x): foo(x) -> f*o**2*x; nadal czysta
    # matematyka na symbolach, zero kodu Pythona
    f, o = sp.Symbol("f"), sp.Symbol("o")
    assert parse("foo(x)").sy == f * o**2 * x


def test_underscore_name_stays_single_symbol():
    assert parse("x_1 + 2").sy == sp.Symbol("x_1") + 2


def test_trailing_digit_name_stays_single_symbol():
    # letters+trailing digits (v0, x1, R1, ...) are an extremely common way to
    # write a subscript in physics/engineering formulas. Default split_symbols
    # breaks "q1" into Symbol('q') * <the digit '1'>, built via a bare
    # Number(...) call the whitelist's global_dict does not define — every
    # such name crashed with a leaked "NameError: name 'Number' is not
    # defined". (Bare sympy without the whitelist does not crash, but is
    # worse: it silently drops the digit, so "q1" reads as "q" and "q2" as
    # "2*q" — verified independently against sympy.parsing before this fix.)
    assert parse("q1").sy == sp.Symbol("q1")
    assert parse("q1*q2").sy == sp.Symbol("q1") * sp.Symbol("q2")
    assert parse("x1+x2").sy == sp.Symbol("x1") + sp.Symbol("x2")
    assert parse("v0").sy == sp.Symbol("v0")
    # multi-letter, no digits still splits into a product of letters —
    # unaffected by the fix (same case test_unknown_multiletter_name_splits_into_symbols covers)
    f, o = sp.Symbol("f"), sp.Symbol("o")
    assert parse("foo(x)").sy == f * o**2 * x


def test_syntax_error_still_readable():
    with pytest.raises(ParseError):
        parse("sin(")


# --- strażnik kosztu ewaluacji (DoS, MODUŁ P-fix) ------------------------
# Whitelist dopuszcza potęgowanie i silnię/dwumian — a te na konkretnych
# liczbach budują gigantyczny int (osiągalne wprost z serwera MCP). Parser
# odmawia ZANIM SymPy to policzy: czytelny PycodemathError, nie OOM/zawieszenie.

@pytest.mark.parametrize(
    "src",
    [
        "9**9**9",  # ~1.2 mld bitów
        "9**9**9**9",
        "2^9**9**9",
        "factorial(10**8)",
        "factorial(100000000)",
        "gamma(10**8)",
        "binomial(10**9, 500000000)",
        "factorial(factorial(20))",  # zagnieżdżona silnia
        "sin(9**9**9)",  # groźny fragment ukryty w argumencie funkcji
        "10**500000",
    ],
)
def test_parser_rejects_evaluation_bombs(src):
    import time

    start = time.monotonic()
    with pytest.raises(ParseError):
        parse(src)
    # kluczowe: odmowa jest NATYCHMIASTOWA (nie liczymy bomby)
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
    # strażnik NIE może odrzucać rozsądnej matematyki
    assert parse(src).sy == expected
