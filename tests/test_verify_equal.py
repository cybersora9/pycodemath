"""``pycodemath.verify.check_equal`` and the ``Verdict`` type (VERIFY V1).

Every REFUTED case below is checked twice: once for the verdict, and once by an
INDEPENDENT evaluation of the reported counterexample (plain SymPy, 50 digits,
exact substitution) — a counterexample the verifier believes but SymPy does not
reproduce would be a verifier bug, and the verdict alone could not show it.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import sympy as sp

from pycodemath import (
    DomainError,
    Expr,
    ParseError,
    PycodemathError,
    TimeBudgetError,
    parse,
    time_budget,
)
from pycodemath.verify import Verdict, VerdictStatus, check_equal

VERIFIED, REFUTED, UNDECIDED = (
    VerdictStatus.VERIFIED,
    VerdictStatus.REFUTED,
    VerdictStatus.UNDECIDED,
)


def _really_differ(a: str, b: str, point: dict[str, str]) -> bool:
    """Independent of the verifier: do the sides differ at ``point``?"""
    subs = {sp.Symbol(k): parse(v).sy for k, v in point.items()}
    left = parse(a).sy.xreplace(subs).evalf(50)
    right = parse(b).sy.xreplace(subs).evalf(50)
    return bool(abs(sp.N(left - right, 50)) > sp.Float("1e-30"))


# --- the Verdict type -------------------------------------------------------
def test_verdict_is_frozen() -> None:
    v = Verdict(VERIFIED, "symbolic", None, "proved")
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.status = REFUTED  # type: ignore[misc]


@pytest.mark.parametrize(
    "status, counterexample",
    [(REFUTED, None), (VERIFIED, {"x": "1"}), (UNDECIDED, {})],
)
def test_counterexample_is_set_exactly_when_refuted(status, counterexample) -> None:
    with pytest.raises(DomainError, match="counterexample"):
        Verdict(status, "numeric-sampling", counterexample, "")


def test_verdict_status_has_exactly_three_values() -> None:
    # "probably" is not an answer — adding a fourth status is a design change.
    assert {s.value for s in VerdictStatus} == {"verified", "refuted", "undecided"}


# --- VERIFIED: only from a symbolic proof -----------------------------------
@pytest.mark.parametrize(
    "a, b",
    [
        ("(x+1)^2", "x^2 + 2x + 1"),
        ("sin(x)^2 + cos(x)^2", "1"),
        ("sin(2x)", "2 sin(x) cos(x)"),
        ("(x+y)^5", "x^5 + 5x^4 y + 10x^3 y^2 + 10x^2 y^3 + 5x y^4 + y^5"),
        ("exp(x+y)", "exp(x) exp(y)"),
        ("tan(x)", "sin(x)/cos(x)"),
    ],
)
def test_true_identities_are_verified_symbolically(a: str, b: str) -> None:
    v = check_equal(a, b)
    assert v.status is VERIFIED
    assert v.method == "symbolic"
    assert v.counterexample is None
    assert v.verified and not v.refuted and not v.undecided


def test_decimals_mean_the_decimal_written() -> None:
    # Binary rounding must not decide a verdict about hand-written algebra.
    assert check_equal("0.1 + 0.2", "0.3").status is VERIFIED
    assert check_equal("0.5*x", "x/2").status is VERIFIED


def test_true_identity_whose_value_cancels_to_zero_is_never_refuted() -> None:
    # sin^2 + cos^2 - 1 evaluates to `-0.e-165` with 1 bit of precision at 7/3:
    # read as a number it is a nonzero where there is a zero.
    v = check_equal("sin(x)^2 + cos(x)^2 - 1", "0")
    assert v.status is VERIFIED


# --- REFUTED: a confirmed counterexample ------------------------------------
@pytest.mark.parametrize(
    "a, b",
    [
        ("sqrt(x^2)", "x"),                     # domain: differs for x < 0
        ("log(x^2)", "2 log(x)"),               # branch of log
        ("x^(2/3)", "(x^2)^(1/3)"),             # branch of a fractional power
        ("sqrt(x) sqrt(y)", "sqrt(x y)"),       # branch, two variables
        ("atan(x) + atan(1/x)", "pi/2"),        # true only for x > 0
        ("(x+1)^2 - (x-1)^2", "4x + 2"),        # a lost sign
        ("sin(3x)", "3 sin(x) - 4 sin(x)^2"),   # a wrong power
        ("(x+y)^5", "x^5 + 5x^4 y + 10x^3 y^2 + 10x^2 y^3 + 5x y^4 + y^4"),
        ("x - y", "y - x"),
        ("floor(x)", "x"),                      # agrees at every integer
        ("x", "x + x/10^6"),
    ],
)
def test_false_identities_are_refuted_with_a_real_counterexample(a: str, b: str) -> None:
    v = check_equal(a, b)
    assert v.status is REFUTED
    assert v.method == "numeric-sampling"
    assert v.counterexample is not None
    assert _really_differ(a, b, v.counterexample)


def test_sqrt_of_square_is_caught_at_a_negative_point() -> None:
    v = check_equal("sqrt(x^2)", "x")
    assert v.counterexample is not None
    assert parse(v.counterexample["x"]).sy < 0


def test_counterexample_is_the_readable_one() -> None:
    # The plan's own example: the lost sign is visible at x = 0 (0 != 2). That
    # point cancels to an EXACT zero, which evalf alone cannot certify — it used
    # to be skipped and x = 1 reported instead.
    v = check_equal("(x+1)^2 - (x-1)^2", "4x + 2")
    assert v.counterexample == {"x": "0"}
    assert v.detail == "at x = 0: left = 0.0, right = 2.0"


def test_counterexample_values_are_exact_parseable_text() -> None:
    v = check_equal("floor(x)", "x")
    assert v.counterexample is not None
    value = parse(v.counterexample["x"]).sy
    assert value.is_Rational and not value.is_Integer


def test_distinct_variables_never_share_a_first_point() -> None:
    # x - y vs y - x agrees wherever x == y; the first point must not be one.
    v = check_equal("x - y", "y - x")
    assert v.counterexample is not None
    assert v.counterexample["x"] != v.counterexample["y"]


def test_complex_value_is_shown_in_sympy_spelling() -> None:
    v = check_equal("log(x^2)", "2 log(x)")
    assert v.counterexample == {"x": "-1"}
    assert v.detail == "at x = -1: left = 0.0, right = 6.283185307*I"


def test_claim_without_variables_is_refuted_at_no_point() -> None:
    v = check_equal("pi", "3.14159")
    assert v.status is REFUTED
    assert v.counterexample == {}


def test_sampling_keeps_the_digits_it_computes() -> None:
    # V2 finding 3, fixed in V5: _evaluate converted the 30- and 60-digit values
    # to mpmath at the GLOBAL 15 digits, so a difference in the 19th digit
    # (4.6e-19, 1.5e-19 relative — over the 1e-20 tolerance) was invisible and
    # the verdict UNDECIDED. The detail then shows as many digits as it takes.
    v = check_equal("pi", "3.141592653589793238")
    assert v.status is REFUTED
    assert v.counterexample == {}
    assert v.detail == (
        "the two sides differ: left = 3.1415926535897932385, "
        "right = 3.141592653589793238"
    )
    assert sp.N(sp.pi - sp.Rational("3.141592653589793238"), 50) > 1e-19
    # the same difference at a point with a variable, and still under the
    # tolerance when it is 27 digits down
    assert check_equal("pi*x", "3.141592653589793238*x").counterexample == {"x": "1"}
    assert check_equal("pi", "3.14159265358979323846264338").status is UNDECIDED


# --- equal where both sides are defined -------------------------------------
def test_removable_singularity_is_verified_not_refuted() -> None:
    # (x^2-1)/(x-1) has no value at x = 1; x + 1 does. Equality is claimed where
    # both are defined, and the undefined point is skipped, not a counterexample.
    v = check_equal("(x^2 - 1)/(x - 1)", "x + 1")
    assert v.status is VERIFIED
    assert "1 not evaluable" in v.detail


# --- UNDECIDED: agreement is not proof --------------------------------------
def test_agreement_at_every_point_is_not_promoted_to_verified() -> None:
    # log(exp(x)) == x for every REAL x (all the samples), but not for complex x,
    # so no symbolic proof exists: the honest answer is UNDECIDED.
    v = check_equal("log(exp(x))", "x")
    assert v.status is UNDECIDED
    assert v.method == "numeric-sampling"
    assert v.counterexample is None
    assert v.detail == (
        "agree numerically at 32 of 32 sample points, without a symbolic proof"
    )


def test_difference_below_the_tolerance_is_undecided_never_verified() -> None:
    # 1e-25 relative is under the 1e-20 sampling tolerance, so sampling cannot see
    # it — and simplify does not reach 0, so it is not proved either.
    assert check_equal("x", "x + x/10^25").status is UNDECIDED


# --- determinism and arguments ----------------------------------------------
def test_same_call_same_verdict() -> None:
    assert check_equal("floor(x)", "x") == check_equal("floor(x)", "x")
    assert check_equal("floor(x)", "x", seed=7).status is REFUTED


def test_samples_bounds_the_evidence() -> None:
    v = check_equal("log(exp(x))", "x", samples=5)
    assert "at 5 of 5 sample points" in v.detail


def test_vars_may_be_a_string_or_a_list() -> None:
    assert check_equal("x - y", "y - x", vars="x y").status is REFUTED
    assert check_equal("x - y", "y - x", vars=["y", "x"]).status is REFUTED


def test_vars_missing_a_symbol_is_refused() -> None:
    with pytest.raises(DomainError, match="y not listed in vars"):
        check_equal("x + y", "y + x + 1", vars=["x"])


def test_accepts_expr_and_numbers() -> None:
    assert check_equal(Expr(sp.Integer(2)), 2).status is VERIFIED
    assert check_equal(parse("x^2"), "x*x").status is VERIFIED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget": 0},
        {"budget": -1},
        {"budget": math.nan},
        {"budget": True},
        {"budget": "5"},
        {"samples": 0},
        {"samples": 2.5},
        {"seed": 1.5},
        {"vars": [1]},
    ],
)
def test_bad_arguments_are_domain_errors(kwargs) -> None:
    with pytest.raises(DomainError):
        check_equal("x", "x + 1", **kwargs)


def test_bad_inputs_are_pycodemath_errors() -> None:
    with pytest.raises(ParseError):
        check_equal("x +* 2", "x")
    with pytest.raises(DomainError):
        check_equal([1, 2], "x")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "a, b",
    [
        ("1/0", "1"),
        # objects, not text: the parser reads `zoo` / `nan` as products of letters
        (Expr(sp.zoo), Expr(sp.oo)),
        (Expr(sp.nan), "0"),
        ("0/0", "0"),
        ("1/x", "1/x + 0*x"),
        ("x^x^x", "x^(x^2)"),
        ("exp(exp(exp(x)))", "exp(exp(exp(x))) + 1"),
        ("gamma(x)", "factorial(x - 1)"),
        ("sin(1/x)", "0"),
        ("log(0*x)", "x"),
        ("sqrt(-x^2 - 1)", "I sqrt(x^2 + 1)"),
    ],
)
def test_awkward_inputs_return_a_verdict(a: "str | Expr", b: "str | Expr") -> None:
    # The contract: a Verdict, or a PycodemathError — never a raw exception.
    try:
        v = check_equal(a, b, budget=20)
    except PycodemathError:
        return
    assert isinstance(v, Verdict)


# --- the time budget --------------------------------------------------------
def _slow_pair() -> tuple[Expr, Expr]:
    # simplify of this difference measured 0.87 s; sampling it, a few ms.
    s = sp.sin(sum(sp.symbols("x y z w"))) ** 3
    return Expr(s), Expr(sp.expand_trig(s))


def test_budget_running_out_in_simplify_is_undecided_not_an_exception() -> None:
    # few samples, so sampling finishes well inside the budget on a slow machine
    # too (32 samples took > 0.1 s on Windows/py3.12.10 and the budget ran out
    # in sampling instead) — the budget must then run out in simplify
    a, b = _slow_pair()
    v = check_equal(a, b, budget=0.1, samples=4)
    assert v.status is UNDECIDED
    assert v.method == "time-budget"
    assert "during symbolic simplification" in v.detail
    assert "agree numerically at 4 of 4" in v.detail


def test_budget_running_out_in_sampling_is_undecided_not_an_exception() -> None:
    # time.monotonic ticks every 15.6 ms on Windows: with the default 24 samples
    # sampling + a cached simplify could finish inside one tick and come back
    # VERIFIED (2/8 runs). 400 samples cannot fit in a tick (59 done at 0.001 s).
    v = check_equal("sin(x)^2 + cos(x)^2", "1", budget=0.001, samples=400)
    assert v.status is UNDECIDED
    assert v.method == "time-budget"
    assert "during numeric sampling" in v.detail


def test_does_not_poison_an_enclosing_time_budget() -> None:
    # A tighter time_budget nested in another poisons the outer once it fires
    # (measured: a 10 s block refused at 0.26 s). Inside a caller's budget
    # check_equal must therefore arm nothing — so the enclosing block survives
    # and keeps working after it.
    a, b = _slow_pair()
    with time_budget(60):
        check_equal(a, b, budget=0.1)
        assert check_equal("sqrt(x^2)", "x").status is REFUTED


def test_an_enclosing_budget_that_expires_refuses_the_enclosing_block() -> None:
    a, b = _slow_pair()
    with pytest.raises(TimeBudgetError):
        with time_budget(0.1):
            check_equal(a, b)


# --- V8: a decimal exponent is the decimal written -----------------------------
@pytest.mark.parametrize(
    "a, b",
    [
        ("2^0.5", "sqrt(2)"),  # V7 finding: was REFUTED
        ("sqrt(5^15)", "5^7.5"),  # V7 finding: was REFUTED
        ("10^2.5", "100*sqrt(10)"),
        ("2^-0.5", "1/sqrt(2)"),
        ("(1 + 0.5)^0.5", "sqrt(3/2)"),
        ("1.5e3*x", "1500*x"),
        ("0.1 + 0.2", "0.3"),
        ("x^0.5", "sqrt(x)"),
    ],
)
def test_a_decimal_exponent_is_the_decimal_written(a: str, b: str) -> None:
    assert check_equal(a, b).status is VERIFIED


@pytest.mark.parametrize(
    "a, b",
    [
        ("2^0.5000001", "sqrt(2)"),
        ("1.41421356237310", "sqrt(2)"),  # a written decimal is not sqrt(2)
    ],
)
def test_a_different_decimal_is_still_refuted(a: str, b: str) -> None:
    assert check_equal(a, b).status is REFUTED


@pytest.mark.parametrize(
    "text, exact",
    [
        ("2^0.5", "2^(1/2)"),
        ("1.5e3x + .5", "(1500)x + (1/2)"),
        ("3e-2", "(3/100)"),
        ("x1e5 + 7", "x1e5 + 7"),  # a name, not a number
        ("2.j + 0x1F", "2.j + 0x1F"),  # imaginary and hex literals as written
        ("(0.5", "(0.5"),  # unbalanced: left for the parser to refuse
    ],
)
def test_decimal_literals_are_spelled_as_fractions(text: str, exact: str) -> None:
    from pycodemath.verify.equal import _exact_decimals

    assert _exact_decimals(text) == exact


# --- V9: nothing hangs before the budget is armed ------------------------------
# V8 finding: check_equal("2^1e10", "1") never returned. The exact spelling
# 2^10000000000 is refused by the parser's cost guard, the text as written is a
# Float of ~10^(3*10^9), and nsimplify's mpmath.identify on it ran BEFORE any
# budget was armed. Run in a subprocess: a regression must fail the test, not
# hang the gate. The limit is the call's own time; the timeout kills a hang.
_HANG_PROBE = """
import sys, time
from pycodemath import PycodemathError
from pycodemath.verify import check_equal, check_steps, certify_integrate
calls = [
    lambda: check_equal("2^1e10", "1", budget=1),
    lambda: check_equal("2^-1e10", "0", budget=1),
    lambda: check_equal("10^-1e10", "0", budget=1),
    lambda: check_steps(["2^1e10", "1"], budget=1),
    lambda: certify_integrate("2^1e10", "x", "x", budget=1),
]
for call in calls:
    start = time.perf_counter()
    try:
        outcome = call().status.name
    except PycodemathError as exc:
        outcome = type(exc).__name__
    print(outcome, round(time.perf_counter() - start, 3))
"""


def test_a_huge_decimal_power_does_not_hang_before_the_budget() -> None:
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-c", _HANG_PROBE],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    lines = done.stdout.split()
    outcomes, seconds = lines[0::2], [float(s) for s in lines[1::2]]
    assert len(outcomes) == 5
    # refused like its exact spelling 2^10000000000 (the parser's cost guard)
    assert outcomes == ["ParseError"] * 5, done.stdout
    assert max(seconds) <= 5, done.stdout  # budget 1 s: "a hard limit of 5 s"


def test_the_refusal_names_the_number_and_the_line() -> None:
    with pytest.raises(ParseError, match=r"too costly to read as the decimal written"):
        check_equal("2^1e10", "1", budget=1)
    # an Expr carries the Float too: the same guard, not only for text
    with pytest.raises(ParseError, match=r"beyond 200000 bits"):
        check_equal(parse("2^1e10"), "1", budget=1)


@pytest.mark.parametrize(
    "a, b, status",
    [
        ("2^1e5", "2^100000", VERIFIED),  # within the line: read exactly
        ("x^1e10", "x", REFUTED),  # a symbolic power is not a built number
        ("exp(1e10)", "1", REFUTED),
        ("1e400", "10^400", VERIFIED),
    ],
)
def test_large_but_buildable_decimals_still_get_a_verdict(
    a: str, b: str, status: VerdictStatus
) -> None:
    assert check_equal(a, b, budget=10).status is status


# --- V9: the parser's refusals reach check_equal typed --------------------------
@pytest.mark.parametrize(
    "a, b, message",
    [
        ("sin", "x", "function name without arguments"),
        ("(sqrt)", "x", "function name without arguments"),
        (")", "x", "closes nothing"),
        ("x)", "x", "closes nothing"),
        ("))", "x", "closes nothing"),
        ("5[x)", "x", r"'\)' closes a '\['"),  # V9 corpus: raw IndexError
        ("(x]", "x", r"'\]' closes a '\('"),
        # V4 finding: was REFUTED at zeta = 0 (zeta(2) read as 2*zeta)
        ("zeta(2)", "pi^2/6", "unknown function 'zeta'"),
    ],
)
def test_parser_refusals_are_parse_errors(a: str, b: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        check_equal(a, b)
