"""K2 — the double result: every closed-form value carries a number whose printed
digits are correct (``pycodemath.engine.precision``).

Guardians: ``test_every_closed_root_carries_a_number`` and
``test_guaranteed_digits_are_correct``. The reference values are mpmath's, at
far more digits than asked, computed without the code under test
(``dev-journal/full/K2/make_constants.py`` for the data file).
"""

from __future__ import annotations

import doctest
import json
import math
import warnings
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from importlib import resources

import mpmath
import pytest
import sympy as sp

from pycodemath.core.budget import time_budget
from pycodemath.core.errors import (
    DomainError, NonConvergenceError, ParseError, PycodemathError, TimeBudgetError,
)
from pycodemath.core.ir import Expr
from pycodemath.engine import precision
from pycodemath.engine.precision import (
    Approx, PrecisionUndecided, approx, dual, evalf_to, parse_precision_request, to_sympy,
)
from pycodemath.engine.symbolic import solve
from pycodemath.frontend.parser import parse


def _data() -> dict:
    text = resources.files("pycodemath.code.data").joinpath("K2_constants.json").read_text("utf-8")
    return json.loads(text)


def _rounded(ref: str, digits: int) -> Decimal:
    """``ref`` correctly rounded (half-even) to ``digits`` significant digits."""
    d = Decimal(ref)
    if d == 0:
        return d
    with localcontext() as ctx:
        ctx.prec = digits
        ctx.rounding = ROUND_HALF_EVEN
        return +d


def _same_digits(shown: str, ref: str, digits: int) -> bool:
    return Decimal(shown) == _rounded(ref, digits)


# --- guardian 1 -----------------------------------------------------------------------

ROOT_EQUATIONS = [
    "x**5 - x + 1",            # CRootOf, no radicals exist
    "x**3 - x - 1",            # Cardano, one real root
    "x**3 - 3*x + 1",          # casus irreducibilis: three real roots written with I
    "x**3 - 2",
    "x**4 - 10*x**2 + 1",      # sqrt(2) +- sqrt(3)
    "x**4 + 1",                # complex radicals
    "x**7 - x - 1",
    "x**3 + x + 1",
    "x**2 - x - 1",
    "exp(x) + x - 2",          # LambertW
    "x*exp(x) - 3",
    "x**6 - 2*x**3 - 1",
]


def _reference_roots(text: str) -> list[mpmath.mpc]:
    sy = parse(text).sy
    x = sp.Symbol("x")
    with mpmath.workdps(80):
        if sy.is_polynomial(x):
            coeffs = [int(c) for c in sp.Poly(sy, x).all_coeffs()]
            return [mpmath.mpc(r) for r in mpmath.polyroots(coeffs, maxsteps=400, extraprec=600)]
        f = sp.lambdify(x, sy, "mpmath")
        return [mpmath.mpc(mpmath.findroot(f, mpmath.mpf(0.5)))]


def test_every_closed_root_carries_a_number():
    """Every root SymPy returns in closed form gets a certified number: real roots
    without a spurious imaginary part, each component correctly rounded."""
    checked = 0
    for text in ROOT_EQUATIONS:
        refs = _reference_roots(text)
        for root in solve(parse(text), "x"):
            d = dual(root)
            assert d.approx is not None, (text, d.exact, d.reason)
            a = d.approx
            with mpmath.workdps(80):
                re_val = mpmath.mpf(a.real)
                im_val = mpmath.mpf(a.imag) if a.imag is not None else mpmath.mpf(0)
                ref = min(refs, key=lambda r: abs(r - mpmath.mpc(re_val, im_val)))
                re_ref = mpmath.nstr(ref.real, 60, strip_zeros=False)
                im_ref = mpmath.nstr(ref.imag, 60, strip_zeros=False)
            assert _same_digits(a.real, re_ref, 15), (text, a.value, re_ref)
            if abs(ref.imag) < mpmath.mpf(10) ** -50:
                assert a.imag is None, (text, d.exact, a.value)  # no "- 0.e-38*I"
            else:
                assert a.imag is not None and _same_digits(a.imag, im_ref, 15)
            assert a.rigorous, (text, a.method)
            checked += 1
    assert checked >= 40


# --- guardian 2 -----------------------------------------------------------------------

HARD = _data()["hard"]


@pytest.mark.parametrize("case", HARD, ids=[c["expr"][:40] for c in HARD])
def test_guaranteed_digits_are_correct(case):
    """The hard-for-float set: the digits shown are the reference's, correctly
    rounded — or the call refuses. Never a wrong digit."""
    try:
        a = approx(case["expr"], case["digits"])
    except PrecisionUndecided:
        pytest.fail(f"undecided on {case['expr']} (expected an answer on this set)")
    d = case["digits"]
    if case["reference"] == "0":
        assert a.real == "0", a
    else:
        assert _same_digits(a.real, case["reference"], d), (a.value, case["reference"])
    if case["reference_imag"] == "0":
        assert a.imag is None, a
    else:
        assert a.imag is not None and _same_digits(a.imag, case["reference_imag"], d), a
    assert a.guaranteed_digits == d


SUMS = [
    "pi + E", "zeta(3)*Catalan", "sqrt(2)**sqrt(3)", "log(3)/log(2)", "exp(-pi)",
    "gamma(1/3)*gamma(2/3)", "EulerGamma**2", "atan(2) - atan(1/3)", "asin(1/7)",
    "cosh(3)/sinh(2)", "erf(1/3)", "LambertW(5)", "zeta(7)", "tan(1)", "1/7",
    "10**50 + 1", "-sqrt(10**-60 + 1) + 1", "Catalan - 0.915965594177219",
]


@pytest.mark.parametrize("text", SUMS)
@pytest.mark.parametrize("digits", [1, 15, 50, 200])
def test_digits_match_mpmath_at_twice_the_precision(text, digits):
    a = approx(text, digits)
    sy = to_sympy(text)
    with mpmath.workdps(2 * digits + 400):
        ref = sp.lambdify([], sy, "mpmath")()
        ref_text = mpmath.nstr(ref, 2 * digits + 50, strip_zeros=False)
    assert _same_digits(a.real, ref_text, digits), (text, digits, a.value)


# --- shapes and behaviour --------------------------------------------------------------

def test_casus_irreducibilis_has_no_imaginary_noise():
    root = solve(parse("x**3 - 3*x + 1"), "x")[0]
    assert "I" in str(root.sy)  # SymPy writes it with I ...
    a = approx(root)
    assert a.imag is None and a.method == "algebraic"  # ... and it is real
    assert a.value == "0.347296355333861"


def test_cancellation_is_resolved_not_printed_as_noise():
    a = approx("exp(pi*sqrt(163)) - 640320**3 - 744")
    assert a.value == "-7.49927402801814e-13"
    assert approx("sin(10**30)", 20).value == "-0.090116901912138058030"
    assert approx("sin(1e30)").value == approx("sin(10**30)").value  # decimal as written


def test_exact_zero_is_proved_and_unprovable_zero_refused():
    a = approx("sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2)")
    assert a.value == "0" and a.rigorous
    with pytest.raises(PrecisionUndecided) as info:
        approx("asin(1/3) + acos(1/3) - pi/2")
    exc = info.value
    assert isinstance(exc, NonConvergenceError)
    assert exc.bound is not None and exc.bound.startswith("|value| < 1e-")


def test_methods_and_rigour():
    assert approx("1/3").method == "exact"
    assert approx("zeta(3)").method == "interval"
    assert approx("CRootOf(x**5 - x + 1, 0)").method == "interval"
    b = approx("besselj(0, 1)")
    assert b.method == "stable-doubling" and b.rigorous is False


def test_formatting():
    assert approx("1/2").value == "0.500000000000000"
    assert approx("2**100").value == "1.26765060022823e+30"
    assert approx("exp(-10**5)").value.endswith("e-43430")
    assert approx("12345.678", 8).value == "12345.678"
    assert approx("pi*10**-30").value == "3.14159265358979e-30"
    c = approx("(-8)**(1/3)")
    assert c.value == "1.00000000000000 + 1.73205080756888*I"
    assert approx("log(-1)").value == "3.14159265358979*I"
    assert approx("1 - 2*I").value == "1.00000000000000 - 2.00000000000000*I"
    assert str(approx("sqrt(2)", 5)) == "≈ 1.4142"
    assert approx("sqrt(2)", 5).text == "≈ 1.4142 (5 digits, interval)"


def test_complex_value_reads_back():
    a = approx("CRootOf(x**5 - x + 1, 1)", 30)
    back = complex(sp.N(to_sympy(a.value), 30))
    ref = complex(sp.N(sp.CRootOf(sp.Symbol("x") ** 5 - sp.Symbol("x") + 1, 1), 30))
    assert abs(back - ref) < 1e-14


def test_to_dict_is_json_and_kind_approx():
    a = approx("zeta(3)", 20)
    d = a.to_dict()
    assert d["kind"] == "approx" and d["exact"] == "zeta(3)"
    assert d["value"] == "1.2020569031595942854"
    assert json.loads(json.dumps(d)) == d
    du = dual("x + 1")
    assert du.approx is None and "free symbols" in (du.reason or "")
    assert json.loads(json.dumps(du.to_dict()))["kind"] == "approx"


def test_python_float_is_its_binary_value_text_is_the_decimal():
    assert approx(0.1, 30).value == "0.100000000000000005551115123126"
    assert approx("0.1", 30).value == "0.100000000000000000000000000000"


def test_inputs_and_refusals():
    assert approx(Expr(sp.sqrt(2))).value == approx("sqrt(2)").value
    assert approx(sp.Rational(1, 3)).method == "exact"
    assert approx(Fraction(1, 3)).value == "0.333333333333333"
    assert approx(Decimal("0.25")).value == "0.250000000000000"
    assert approx(3).value == "3.00000000000000"
    assert approx(1 + 2j).imag == "2.00000000000000"
    for bad in [float("nan"), float("inf"), "x + 1", "oo", "1/0", True, None, [1]]:
        with pytest.raises(PycodemathError):
            approx(bad)  # type: ignore[arg-type]
    for digits in [0, -1, 10_001, 1.5, True, "15"]:
        with pytest.raises(DomainError):
            approx("pi", digits)  # type: ignore[arg-type]
    with pytest.raises(ParseError):
        approx("")
    with pytest.raises(ParseError):
        approx("zeta")
    with pytest.raises(ParseError):
        approx("9**9**9**9")
    with pytest.raises(DomainError):
        approx("pi", budget=0)
    with pytest.raises(ParseError):
        approx("__import__('os')")


def test_budget_is_respected():
    with pytest.raises(TimeBudgetError):
        with time_budget(0.05):
            approx("asin(1/3) + acos(1/3) - pi/2")


def test_parse_precision_request():
    assert parse_precision_request("N(pi, 50)") == ("pi", 50)
    assert parse_precision_request("N(atan2(1, 2), 20)") == ("atan2(1, 2)", 20)
    assert parse_precision_request("evalf(sqrt(2), 30)") == ("sqrt(2)", 30)
    assert parse_precision_request("N(pi)") == ("pi", 15)
    assert parse_precision_request("evalf sqrt(2) 50") == ("sqrt(2)", 50)
    assert parse_precision_request("zeta(3) digits 40") == ("zeta(3)", 40)
    assert parse_precision_request("pi + 1") is None
    assert parse_precision_request("N(pi, x)") is None
    assert evalf_to("N(pi, 30)").value == "3.14159265358979323846264338328"
    assert evalf_to("pi", 3).value == "3.14"
    with pytest.raises(DomainError):
        evalf_to("pi + 1")


def test_no_warning_or_stderr_leak(capfd):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        evalf_to("N(pi, 50)")
        approx("zeta(3)", 50)
        approx("besselj(0, 1)", 30)
        approx("polylog(3, 1/3)")
        dual(solve(parse("x**3 - 3*x + 1"), "x")[0])
    out, err = capfd.readouterr()
    assert caught == [] and err == "" and out == ""


def test_no_global_mpmath_state_touched():
    before = (mpmath.mp.prec, mpmath.iv.prec)
    approx("zeta(3)", 500)
    approx("besselj(0, 1)", 100)
    assert (mpmath.mp.prec, mpmath.iv.prec) == before


def test_rigorous_zeta_matches_mpmath_far_out():
    for s in (3, 5, 9, 31):
        a = approx(f"zeta({s})", 100)
        with mpmath.workdps(300):
            ref = mpmath.nstr(mpmath.zeta(s), 200, strip_zeros=False)
        assert _same_digits(a.value, ref, 100), s


def test_rigorous_erf_and_lambertw():
    for x in ("1/3", "-2", "7", "70", "10**-20"):
        a = approx(f"erf({x})", 40)
        with mpmath.workdps(200):
            ref = mpmath.nstr(mpmath.erf(mpmath.mpmathify(sp.N(to_sympy(x), 200))), 100,
                              strip_zeros=False)
        assert _same_digits(a.value, ref, 40), x
    for x, k in (("1", 0), ("-1/4", -1), ("-1/4", 0), ("10**6", 0)):
        a = approx(f"LambertW({x}, {k})", 40)
        assert a.method == "interval"
        with mpmath.workdps(200):
            ref = mpmath.nstr(mpmath.lambertw(mpmath.mpmathify(sp.N(to_sympy(x), 200)), k),
                              100, strip_zeros=False)
        assert _same_digits(a.value, ref, 40), (x, k)


def test_huge_values_have_exponents_not_integers():
    a = approx("exp(exp(100))")
    assert a.value.startswith("2.76636181554699e+1167434441400288663279816738100")
    assert approx("exp(10**6)").value.endswith("e+434294")


def test_doctests():
    res = doctest.testmod(precision, optionflags=doctest.ELLIPSIS)
    assert res.failed == 0 and res.attempted > 5


def test_approx_dataclass_is_frozen():
    a = approx("pi")
    with pytest.raises(Exception):
        a.value = "3"  # type: ignore[misc]
    assert isinstance(a, Approx)


def test_digits_cap_large_request_is_fast_enough():
    a = approx("pi", 5000)
    assert a.value.startswith("3.14159265358979323846") and len(a.value) == 5001
    assert math.isfinite(len(a.value))
