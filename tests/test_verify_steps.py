"""``pycodemath.verify.check_steps`` — step-by-step derivation checking (VERIFY V3).

As in ``test_verify_equal``, every REFUTED step is checked twice: once for the
verdict, and once by an INDEPENDENT evaluation of the reported counterexample
(plain SymPy, 50 digits, exact substitution). For an equation step that means the
point really solves one of the two steps and really does not solve the other — a
counterexample only the verifier believes would be a verifier bug.
"""

from __future__ import annotations

import dataclasses

import pytest
import sympy as sp

from pycodemath import (
    DomainError,
    ParseError,
    PycodemathError,
    TimeBudgetError,
    parse,
    time_budget,
)
from pycodemath.verify import StepCheck, StepsResult, VerdictStatus, check_steps
from pycodemath.verify import steps as steps_module

VERIFIED, REFUTED, UNDECIDED = (
    VerdictStatus.VERIFIED,
    VerdictStatus.REFUTED,
    VerdictStatus.UNDECIDED,
)

#: The derivation from PLAN_VERIFY.md §2 V3: step 2 flips the sign of the last 1.
PLAN_EXAMPLE = """(x+1)^2 - (x-1)^2
= x^2 + 2x + 1 - x^2 + 2x + 1
= 4x + 2"""


def _subs(point: dict[str, str]) -> dict[sp.Symbol, sp.Expr]:
    return {sp.Symbol(k): parse(v).sy for k, v in point.items()}


def _value(text: str, point: dict[str, str]) -> sp.Expr:
    return sp.N(parse(text).sy.xreplace(_subs(point)), 50)


def _really_differ(a: str, b: str, point: dict[str, str]) -> bool:
    """Independent of the verifier: do two expressions differ at ``point``?"""
    return bool(abs(_value(a, point) - _value(b, point)) > sp.Float("1e-30"))


def _really_solves(equation: str, point: dict[str, str]) -> bool:
    """Independent of the verifier: does ``point`` solve ``left = right``
    (any alternative of an ``or``)?"""
    for alternative in equation.split(" or "):
        left, right = alternative.split("=")
        a, b = _value(left, point), _value(right, point)
        if any(v.has(sp.nan, sp.zoo, sp.oo, -sp.oo) for v in (a, b)):
            continue
        if abs(a - b) < sp.Float("1e-30"):
            return True
    return False


def _statuses(result: StepsResult) -> list[VerdictStatus]:
    return [c.verdict.status for c in result.checks]


# --- the plan's example, and expression chains --------------------------------
def test_plan_example_refutes_step_2_at_x_0() -> None:
    r = check_steps(PLAN_EXAMPLE)
    assert r.kind == "expressions"
    assert r.status is REFUTED
    assert r.first_error == 2
    assert r.counterexample == {"x": "0"}
    assert _statuses(r) == [REFUTED, VERIFIED]  # step 3 follows from step 2
    assert [c.number for c in r.checks] == [2, 3]
    assert r.failed is r.checks[0]
    assert r.failed.text == "x^2 + 2x + 1 - x^2 + 2x + 1"
    assert "at x = 0" in r.failed.verdict.detail
    # 0 vs 2, by plain SymPy
    assert _really_differ(r.steps[0], r.steps[1], r.counterexample)


def test_corrected_derivation_is_verified() -> None:
    r = check_steps(
        """(x+1)^2 - (x-1)^2
        = x^2 + 2x + 1 - x^2 + 2x - 1
        = 4x"""
    )
    assert r.status is VERIFIED
    assert r.first_error is None
    assert r.counterexample is None
    assert r.failed is None
    assert _statuses(r) == [VERIFIED, VERIFIED]


@pytest.mark.parametrize(
    "written",
    [
        "(x+1)^2 - (x-1)^2 = x^2 + 2x + 1 - x^2 + 2x + 1 = 4x + 2",
        ["(x+1)^2 - (x-1)^2", "= x^2 + 2x + 1 - x^2 + 2x + 1", "= 4x + 2"],
        ["(x+1)^2 - (x-1)^2", "x^2 + 2x + 1 - x^2 + 2x + 1", "4x + 2"],
        "(x+1)^2 - (x-1)^2\nx^2 + 2x + 1 - x^2 + 2x + 1\n4x + 2",
        "(x+1)^2 - (x-1)^2 = x^2 + 2x + 1 - x^2 + 2x + 1\n= 4x + 2",
    ],
)
def test_expression_chain_notations_agree(written) -> None:
    r = check_steps(written)
    assert r.kind == "expressions"
    assert r.steps == (
        "(x+1)^2 - (x-1)^2", "x^2 + 2x + 1 - x^2 + 2x + 1", "4x + 2",
    )
    assert (r.first_error, r.counterexample) == (2, {"x": "0"})


def test_a_single_line_with_one_equals_is_an_identity_claim() -> None:
    # One line is always an expression chain: ``x^2 = 4`` claims an identity.
    r = check_steps("x^2 = 4")
    assert r.kind == "expressions"
    assert r.status is REFUTED
    assert _really_differ("x^2", "4", r.counterexample)


def test_one_slip_is_reported_once_later_steps_judged_on_their_own() -> None:
    r = check_steps("(x+1)^2\n= x^2 + 2x\n= x(x + 2)\n= x^2 + 2x + 1")
    assert _statuses(r) == [REFUTED, VERIFIED, REFUTED]
    assert r.first_error == 2


def test_an_unproved_step_makes_the_chain_undecided_not_verified() -> None:
    r = check_steps("log(exp(x))\n= x\n= x + 0")
    assert _statuses(r) == [UNDECIDED, VERIFIED]
    assert r.status is UNDECIDED
    assert r.first_error is None


def test_expression_semantics_are_check_equals() -> None:
    # principal branch: the classic sqrt(x^2) = x slip is caught at x = -1 ...
    assert check_steps("sqrt(x^2)\n= x").counterexample == {"x": "-1"}
    # ... and a removable singularity is not an error
    assert check_steps("(x^2 - 1)/(x - 1)\n= x + 1").status is VERIFIED


# --- equation chains ------------------------------------------------------------
@pytest.mark.parametrize(
    "written",
    [
        "2x + 3 = 7 -> 2x = 4 -> x = 2",
        "2x + 3 = 7\n2x = 4\nx = 2",
        "2x + 3 = 7 => 2x = 4 ⇒ x = 2",
        "2x + 3 = 7 <=> 2x = 4 ⇔ x = 2",
        "2x + 3 == 7 → 2x == 4 → x == 2",
        ["2x + 3 = 7", "2x = 4", "x = 2"],
    ],
)
def test_plan_equation_chain_is_verified(written) -> None:
    r = check_steps(written)
    assert r.kind == "equations"
    assert r.status is VERIFIED
    assert all(c.verdict.method == "symbolic" for c in r.checks)
    assert all(c.effect is None and c.warning is None for c in r.checks)


def test_arithmetic_slip_in_an_equation_step() -> None:
    r = check_steps("2x + 3 = 7\n2x = 10\nx = 5")
    assert r.first_error == 2
    assert r.counterexample == {"x": "2"}
    assert _statuses(r) == [REFUTED, VERIFIED]
    # a plain slip both loses x = 2 and gains x = 5: not a one-sided effect
    assert r.failed is not None and r.failed.effect is None
    assert _really_solves("2x + 3 = 7", r.counterexample)
    assert not _really_solves("2x = 10", r.counterexample)


def test_squaring_both_sides_gains_a_root_and_says_so() -> None:
    steps = ["sqrt(x) = x - 2", "x = (x-2)^2", "x^2 - 5x + 4 = 0", "x = 1 or x = 4"]
    r = check_steps(steps)
    assert _statuses(r) == [REFUTED, VERIFIED, VERIFIED]
    failed = r.failed
    assert failed is not None
    assert failed.number == 2
    assert failed.effect == "gains-roots"
    assert failed.warning is not None and "squaring both sides" in failed.warning
    assert r.counterexample == {"x": "1"}
    assert _really_solves(steps[1], r.counterexample)
    assert not _really_solves(steps[0], r.counterexample)  # sqrt(1) = 1, not -1


def test_dividing_by_a_variable_loses_a_root_and_names_the_divisor() -> None:
    r = check_steps("x^2 = 2x\nx = 2")
    failed = r.failed
    assert failed is not None
    assert failed.effect == "loses-roots"
    assert r.counterexample == {"x": "0"}
    assert failed.warning == (
        "step 2 loses the root x = 0 — both sides were divided by x, which is 0 there"
    )
    assert _really_solves("x^2 = 2x", r.counterexample)
    assert not _really_solves("x = 2", r.counterexample)


def test_multiplying_by_a_variable_gains_a_root_and_names_the_factor() -> None:
    r = check_steps("x = 1\nx^2 = x")
    assert r.failed is not None and r.failed.effect == "gains-roots"
    assert r.counterexample == {"x": "0"}
    assert "multiplied by x" in (r.failed.warning or "")


def test_a_step_undefined_at_a_root_loses_it() -> None:
    r = check_steps("x - 1 = 0\n(x^2 - 2x + 1)/(x - 1) = 0")
    assert r.failed is not None and r.failed.effect == "loses-roots"
    assert r.counterexample == {"x": "1"}
    assert "undefined there" in (r.failed.warning or "")


def test_log_rewrite_is_refuted_although_solveset_misses_it() -> None:
    # solveset(log(x^2) - 2*log(x), x, Reals) is Reals — not on the principal
    # branch at x < 0. The candidates are substituted independently, so the lost
    # root x = -1 (0 vs 2*pi*I) is caught all the same.
    r = check_steps("log(x^2) = 0\n2 log(x) = 0")
    assert r.counterexample == {"x": "-1"}
    assert r.failed is not None and r.failed.effect == "loses-roots"
    # no cause is invented for a quotient nobody divided by
    assert r.failed.warning == "step 2 loses the root x = -1"
    assert _really_solves("log(x^2) = 0", r.counterexample)
    assert not _really_solves("2 log(x) = 0", r.counterexample)


def test_periodic_solutions_lose_a_member() -> None:
    r = check_steps("cos(x) = 1\nx = 0")
    assert r.failed is not None and r.failed.effect == "loses-roots"
    assert _really_solves("cos(x) = 1", r.counterexample)
    assert not _really_solves("x = 0", r.counterexample)


def test_disjunction_of_equations() -> None:
    r = check_steps("x^2 - 4 = 0\n(x-2)(x+2) = 0\nx = 2 or x = -2")
    assert r.status is VERIFIED
    lost = check_steps("x^2 = 4\nx = 2")
    assert lost.counterexample == {"x": "-2"}
    assert lost.failed is not None and lost.failed.effect == "loses-roots"


def test_interval_solution_sets() -> None:
    same = check_steps("sqrt(x^2) = x\nabs(x) = x")
    assert same.status is VERIFIED
    assert same.checks[0].verdict.method == "solution-sets"
    gained = check_steps("sqrt(x^2) = x\nx = x")
    assert gained.failed is not None and gained.failed.effect == "gains-roots"
    assert _really_solves("x = x", gained.counterexample)
    assert not _really_solves("sqrt(x^2) = x", gained.counterexample)


def test_finite_solution_sets_verified_without_proportionality() -> None:
    # (x-1)(x-2) = 0 vs x^3 - 3x^2 + 2x = x^3 - x^3 + ... : not proportional,
    # same real roots — decided by solving, not by cancelling
    r = check_steps("x^2 - 3x + 2 = 0\n(x^2 - 3x + 2)(x^2 + 1) = 0")
    assert r.status is VERIFIED
    assert r.checks[0].verdict.method == "solution-sets"
    assert "{1, 2}" in r.checks[0].verdict.detail


def test_several_variables() -> None:
    # proportional rearrangements are proved in any number of variables
    ok = check_steps("y = 2x + 3\ny - 3 = 2x\nx = (y - 3)/2")
    assert ok.status is VERIFIED
    # a wrong rearrangement is refuted at a full point
    bad = check_steps("y = 2x + 3\nx = (y + 3)/2")
    assert bad.first_error == 2
    assert set(bad.counterexample or {}) == {"x", "y"}
    assert _really_solves("y = 2x + 3", bad.counterexample)
    assert not _really_solves("x = (y + 3)/2", bad.counterexample)
    # a correct step with different poles is only sampled: UNDECIDED, not VERIFIED
    sampled = check_steps("x*y = 1\ny = 1/x")
    assert sampled.status is UNDECIDED
    assert "without a proof for all values" in sampled.checks[0].verdict.detail


def test_equations_without_variables() -> None:
    assert check_steps("2 = 2\n3 = 3").status is VERIFIED
    r = check_steps("2 = 2\n3 = 4")
    assert r.counterexample == {}
    assert r.failed is not None
    assert r.failed.verdict.detail == "step 1 is true and step 2 is false"
    assert r.failed.effect is None


def test_transcendental_step_without_roots_is_never_verified() -> None:
    # correct (x^2 + 1 has no real zero), but solveset returns a ConditionSet and
    # the factor stops proportionality: honestly UNDECIDED
    ok = check_steps("exp(x) = x + 2\nexp(x)*(x^2 + 1) = (x + 2)*(x^2 + 1)")
    assert ok.status is UNDECIDED
    # wrong, and its roots cannot be found in closed form: never VERIFIED
    bad = check_steps("exp(x) = x + 2\nexp(x) = x + 3")
    assert bad.status is not VERIFIED


# --- the result types --------------------------------------------------------
def test_result_types_are_frozen() -> None:
    r = check_steps(PLAN_EXAMPLE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.kind = "equations"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.checks[0].number = 7  # type: ignore[misc]
    assert isinstance(r.checks[0], StepCheck)


def test_same_call_same_result() -> None:
    assert check_steps(PLAN_EXAMPLE) == check_steps(PLAN_EXAMPLE)
    eq = "sqrt(x) = x - 2\nx = (x-2)^2"
    assert check_steps(eq) == check_steps(eq)


def test_vars_are_checked_across_every_step() -> None:
    assert check_steps(PLAN_EXAMPLE, vars="x").first_error == 2
    with pytest.raises(DomainError, match="check_steps: y not listed in vars"):
        check_steps("x + y\n= y + x", vars=["x"])
    with pytest.raises(DomainError, match="check_steps: y not listed in vars"):
        check_steps("x = y\nx - y = 0", vars="x")


# --- refusals ------------------------------------------------------------------
@pytest.mark.parametrize(
    "steps, match",
    [
        ("x + 1", "at least two steps"),
        ("", "at least two steps"),
        ("2x + 3 < 7\n2x < 4", "inequality"),
        ("x >= 1\nx - 1 >= 0", "inequality"),
        ("x != 1\nx - 1 != 0", "inequality"),
        ("(x+1)^2\n= x^2 + 2x + 1\nx^2", "starts with '='"),
        ("(x+1)^2 -> x^2 + 2x + 1", "arrow"),
        ("x = 1\nx = 1 = 1", "exactly one '='"),
        ("x = 1\nx + 1", "exactly one '='"),
        ("x^2\n= = x*x", "empty"),
        (42, "text or a list"),
        (["x", 1], "text or a list"),
    ],
)
def test_malformed_derivations_are_domain_errors(steps, match) -> None:
    with pytest.raises(DomainError, match=match):
        check_steps(steps)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget": 0},
        {"budget": -1},
        {"budget": float("nan")},
        {"budget": "1"},
        {"budget": True},
        {"samples": 0},
        {"samples": 2.5},
        {"samples": True},
        {"seed": 1.5},
        {"vars": [1]},
    ],
)
def test_bad_arguments_are_domain_errors(kwargs) -> None:
    with pytest.raises(DomainError, match="check_steps"):
        check_steps(PLAN_EXAMPLE, **kwargs)


def test_a_step_that_does_not_parse_is_refused_before_any_work() -> None:
    # the typo is in the LAST step: it is still refused up front, as a ParseError
    with pytest.raises(ParseError):
        check_steps("x^2 = 2x\nx = 2\nx = (")
    with pytest.raises(ParseError):
        check_steps("(x+1)^2\n= x^2 + 2x + 1\n= )")
    with pytest.raises(ParseError):
        check_steps("x = 1\nx) = 1")


@pytest.mark.parametrize(
    "steps",
    [
        "1/x = 0\n1 = 0",
        "x^x = 1\nx = 1",
        "tan(x) = 0\nsin(x) = 0",
        "floor(x) = 1\nx = 1",
        "0 = 0\nx = x",
        "zoo\n= 1",
        "sqrt(-1) = I\nI = I",
        "x^2 + 1 = 0\nx = I",
        "gamma(x) = 1\nx = 1 or x = 2",
    ],
)
def test_awkward_derivations_return_a_result(steps) -> None:
    try:
        r = check_steps(steps)
    except PycodemathError:
        return  # a typed refusal is an acceptable answer too
    assert isinstance(r, StepsResult)
    for c in r.checks:
        if c.verdict.refuted and r.kind == "equations" and c.verdict.counterexample:
            before, after = r.steps[c.number - 2], r.steps[c.number - 1]
            assert _really_solves(before, c.verdict.counterexample) != _really_solves(
                after, c.verdict.counterexample
            )


# --- the budget --------------------------------------------------------------
def _slow_equation_step() -> str:
    # _constant_ratio alone measured 0.13-0.38 s, solveset up to 6.5 s
    s = sp.sin(sum(sp.symbols("x y z w"))) ** 3
    return f"{s} = 0\n{sp.expand_trig(s)} = 1"


def test_budget_spent_leaves_later_steps_undecided_not_an_exception() -> None:
    r = check_steps("(x+1)^2\n= x^2 + 2x + 1\n= x^2 + 2x + 2\n= x^2", budget=1e-9)
    assert _statuses(r) == [UNDECIDED] * 3
    assert all(c.verdict.method == "time-budget" for c in r.checks)
    # the detail quotes the budget the caller chose, not what was left of it
    assert all("the 1e-09s time budget ran out" in c.verdict.detail for c in r.checks)
    assert r.first_error is None


def test_budget_running_out_inside_an_equation_step_is_undecided() -> None:
    r = check_steps(_slow_equation_step(), budget=0.05)
    (check,) = r.checks
    assert check.verdict.status is UNDECIDED
    assert check.verdict.method == "time-budget"
    assert check.verdict.detail == (
        "the 0.05s time budget ran out while step 2 was decided"
    )


def test_arms_no_budget_inside_a_callers_budget(monkeypatch) -> None:
    # A tighter time_budget nested in another poisons the outer when it fires
    # (V1: a 10 s block refused at 0.26 s). Inside a caller's budget check_steps
    # must arm nothing of its own — and the enclosing block keeps working.
    def armed(*args, **kwargs):  # pragma: no cover - reached only on regression
        raise AssertionError("check_steps armed its own time_budget")

    with time_budget(60):
        monkeypatch.setattr(steps_module, "time_budget", armed)
        assert check_steps("x^2 = 2x\nx = 2").first_error == 2
        assert check_steps(PLAN_EXAMPLE).first_error == 2
        monkeypatch.undo()
        assert check_steps("sqrt(x) = x - 2\nx = (x-2)^2").first_error == 2


def test_an_enclosing_budget_that_expires_refuses_the_enclosing_block() -> None:
    with pytest.raises(TimeBudgetError):
        with time_budget(0.05):
            check_steps(_slow_equation_step())


# --- V8: roots of a quartic located by nroots ---------------------------------
def test_squaring_into_a_ferrari_quartic_is_refuted_within_the_budget() -> None:
    # V6's five misses: each ran out the whole 30 s budget in simplify.
    result = check_steps(["sqrt(x) = x^2 - 6", "x = (x^2 - 6)^2"], budget=10)
    step = result.checks[0]
    assert step.verdict.status is REFUTED and step.verdict.method == "located-root"
    assert step.effect == "gains-roots"
    assert step.warning is not None and "squaring both sides" in step.warning
    assert step.verdict.counterexample == {"x": "2.13079247594210"}


def test_squaring_a_quartic_with_no_real_root_is_still_verified() -> None:
    result = check_steps(["sqrt(x) = x^2 + 6", "x = (x^2 + 6)^2"], budget=10)
    assert result.checks[0].verdict.status is VERIFIED


def test_a_lost_quartic_root_is_reported_exactly_when_it_is_rational() -> None:
    result = check_steps("x^4 - 5*x^2 + 4 = 0 -> x = 1 or x = -1 or x = 2")
    step = result.checks[0]
    assert step.verdict.status is REFUTED and step.effect == "loses-roots"
    assert step.verdict.counterexample == {"x": "-2"}


def test_a_lost_irrational_quartic_root_is_located() -> None:
    step = check_steps("x^4 = 2 -> x = 2^(1/4)").checks[0]
    assert step.verdict.status is REFUTED and step.effect == "loses-roots"
    assert step.verdict.counterexample == {"x": "-1.18920711500272"}


def test_a_complete_quartic_split_is_still_verified() -> None:
    assert check_steps("x^4 = 2 -> x = 2^(1/4) or x = -2^(1/4)").status is VERIFIED


def test_located_roots_match_the_sturm_count_or_nothing_is_located() -> None:
    x = sp.Symbol("x")
    read = steps_module._Equation.read
    assert steps_module._numeric_roots(read("sqrt(x) = x^2 - 6"), x) is None
    assert steps_module._numeric_roots(read("x^3 = 2"), x) is None  # degree 3
    assert steps_module._numeric_roots(read("x^4 + 1 = 0"), x) == []
    located = steps_module._numeric_roots(read("x = (x^2 - 6)^2"), x)
    assert located is not None and len(located) == 2
    for mid, lo, hi in located:  # type: ignore[misc]
        assert lo < mid < hi and hi - lo < sp.Rational(1, 10**29)
