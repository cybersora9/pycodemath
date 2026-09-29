"""Testy bezpiecznego parsera (MODUŁ P1): whitelist sympify.

Parser ma rozwiązywać TYLKO whitelistę funkcji matematycznych; builtiny
Pythona (__import__/eval/open), literały tekstowe i dostęp atrybutowy mają
być nieosiągalne z tekstu wejściowego. Oczekiwane wartości budowane z gołego
sympy (niezależnie od pycodemath) — bez tautologii.
"""

from __future__ import annotations

import re

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
    # split_symbols (cena notacji 2x): foo -> f*o**2; nadal czysta
    # matematyka na symbolach, zero kodu Pythona
    f, o = sp.Symbol("f"), sp.Symbol("o")
    assert parse("foo").sy == f * o**2
    assert parse("xy*(x+1)").sy == sp.Symbol("y") * x * (x + 1)


# V9 (znalezisko V4): WYWOŁANIE nazwy spoza białej listy to odmowa, nie iloczyn.
# Czytane jako iloczyn dawało zeta(2) = 2*zeta, a check_equal("zeta(2)", "pi^2/6")
# REFUTED przy zeta = 0 — fałszywy dowód fałszu.
@pytest.mark.parametrize(
    "text, name",
    [
        ("zeta(2)", "zeta"),
        ("foo(x)", "foo"),
        ("besselj(0, x)", "besselj"),
        ("Ei(x)", "Ei"),
        ("xsin(x)", "xsin"),  # sklejone z nazwą funkcji: dawniej x*s*i*n*x
        ("zeta (2)", "zeta"),  # spacja nic nie zmienia — to te same tokeny
        ("log_2(8)", "log_2"),  # indeks dolny przy ZNANEJ funkcji: wywołanie
        ("2 + zeta(2)", "zeta"),
        # cena reguły: dwie litery przed nawiasem to też wywołanie — iloczyn
        # trzeba zapisać jawnie (xy*(x+1) albo x y(x+1))
        ("xy(x+1)", "xy"),
    ],
)
def test_calling_an_unknown_function_is_a_parse_error(text, name):
    with pytest.raises(ParseError, match=f"unknown function '{name}'"):
        parse(text)


def test_unknown_function_call_emits_no_sympy_warning():
    # besselj(0,x) dawał ParseError + SymPyDeprecationWarning (mnożenie krotki);
    # teraz odmowa pada na tokenach, zanim SymPy cokolwiek zbuduje
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ParseError, match="unknown function 'besselj'"):
            parse("besselj(0,x)")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("x(x+1)", x * (x + 1)),  # jedna litera: iloczyn, jak dotąd
        ("2x(x-1)", 2 * x * (x - 1)),
        ("x1(x+1)", sp.Symbol("x1") * (x + 1)),  # indeks cyfrą: symbol
        ("x_1(x+1)", sp.Symbol("x_1") * (x + 1)),  # indeks podkreśleniem: symbol
        ("pi(x+1)", sp.pi * (x + 1)),  # stała: iloczyn
        ("E(x)", sp.E * x),
        ("gamma(3)", sp.Integer(2)),  # znana funkcja: wywołanie
        ("sin (x)", sp.sin(x)),
    ],
)
def test_what_still_reads_as_a_product_or_a_known_call(text, expected):
    assert parse(text).sy == expected


# V9 (znalezisko V5): goła nazwa funkcji docierała do strażnika kosztu jako KLASA,
# iteracja po jej `args` (property) = goły TypeError. Tak samo konstruktory.
@pytest.mark.parametrize(
    "text",
    ["sin", "log", "Abs", "(sqrt)", "2*cos", "sin + 1", "abs", "gamma",
     "Symbol", "Integer", "Function"],
)
def test_a_bare_function_name_is_a_parse_error(text):
    with pytest.raises(ParseError, match="function name without arguments"):
        parse(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("sin x", sp.sin(x)),
        ("sin^2 x", sp.sin(x) ** 2),
        ("sin^2(x)", sp.sin(x) ** 2),
        ("2 sin x", 2 * sp.sin(x)),
        ("sqrt x", sp.sqrt(x)),
        ("(sin x)^2", sp.sin(x) ** 2),
        ("exp(x)exp(x)", sp.exp(2 * x)),
    ],
)
def test_function_application_without_parentheses_still_works(text, expected):
    assert parse(text).sy == expected


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
    assert parse("foo*x").sy == f * o**2 * x


# V3 znalezisko 2 (naprawione 29.09) + V9: nawias zamykający bez otwartego albo
# zamykający INNY rodzaj — ParseError w samym parserze. `[)` przechodziło przez
# liczenie głębokości i SymPy zdejmował pusty stos: goły IndexError (V9: 90 z
# 21124 tekstów korpusu).
@pytest.mark.parametrize(
    "text, message",
    [
        (")", "')' closes nothing"),
        ("x)", "')' closes nothing"),
        ("))", "')' closes nothing"),
        ("(x))", "')' closes nothing"),
        ("]", "']' closes nothing"),
        ("[)", "')' closes a '['"),
        ("5[x)", "')' closes a '['"),
        ("(x]", "']' closes a '('"),
    ],
)
def test_a_stray_or_mismatched_close_is_a_parse_error(text, message):
    with pytest.raises(ParseError, match=re.escape(message)):
        parse(text)


def test_matched_brackets_still_parse():
    assert parse("((x+1)*(x-1))").sy == (x + 1) * (x - 1)
    from pycodemath.frontend.parser import parse_matrix

    assert parse_matrix("[[1, (2)], [3, 4]]") == sp.Matrix([[1, 2], [3, 4]])
    with pytest.raises(ParseError, match="closes a"):
        parse_matrix("[[1, 2)]")


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
        # V9: ujemny wykładnik i wymierna podstawa też budują dokładną liczbę —
        # wisiały w parse (dokładny zapis 2^-1e10 od V8)
        "2^-10000000000",
        "2^(-10000000000)",
        "(1/2)^10000000000",
        "(3/2)^10000000",
        "1/2^10000000000",
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
        ("2^-10", sp.Rational(1, 1024)),
        ("(2/3)^5", sp.Rational(32, 243)),
        ("(-1)^10000000000", sp.Integer(1)),  # |podstawa| 1: nic nie rośnie
        ("(1/1)^10000000000", sp.Integer(1)),
        ("x^-10000000000", sp.Symbol("x") ** -10000000000),  # symbol: nic nie liczy
    ],
)
def test_guard_lets_ordinary_powers_and_factorials_through(src, expected):
    # strażnik NIE może odrzucać rozsądnej matematyki
    assert parse(src).sy == expected
