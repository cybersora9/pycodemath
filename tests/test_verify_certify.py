"""``pycodemath.verify.certify`` — engine results checked by a route that did not
produce them (VERIFY V2).

As in ``test_verify_equal``, a REFUTED verdict is not taken on its own word: where
it carries a point, the point is re-checked here with plain SymPy at 50 digits, by
a computation the certificate did not make.
"""

from __future__ import annotations

import math
import sys

import pytest
import sympy as sp

from pycodemath import (
    DomainError,
    E,
    Expr,
    ParseError,
    PycodemathError,
    TimeBudgetError,
    numerics,
    ode,
    parse,
    symbolic,
    time_budget,
)
from pycodemath.verify import (
    CERTIFIERS,
    Verdict,
    VerdictStatus,
    certify,
    certify_diff,
    certify_dsolve,
    certify_integrate,
    certify_limit,
    certify_nintegrate,
    certify_solve,
    check_equal,
)

VERIFIED, REFUTED, UNDECIDED = (
    VerdictStatus.VERIFIED,
    VerdictStatus.REFUTED,
    VerdictStatus.UNDECIDED,
)


def _at(text: "str | Expr", point: "dict[str, str]") -> sp.Expr:
    """``text`` at ``point``, exact substitution, 50 digits — the independent check."""
    sy = text.sy if isinstance(text, Expr) else parse(text).sy
    return sy.xreplace({sp.Symbol(k): parse(v).sy for k, v in point.items()}).evalf(50)


def _derivative_at(text: "str | Expr", var: str, point: "dict[str, str]") -> sp.Expr:
    sy = text.sy if isinstance(text, Expr) else parse(text).sy
    return _at(Expr(sp.diff(sy, sp.Symbol(var))), point)


# --- engine results certify ------------------------------------------------------
@pytest.mark.parametrize(
    "operation, args",
    [
        ("integrate", ("2*x", "x")),
        ("integrate", ("x*sin(x)", "x")),
        ("integrate", ("1/x", "x")),
        ("integrate", ("exp(x)*cos(x)", "x")),
        ("diff", ("sin(x)*x", "x")),
        ("diff", ("Abs(x)", "x")),  # the v0.3 audit's wrong-derivative case
        ("diff", ("Max(x, x^2)", "x")),
        ("diff", ("exp(x^2)*log(x)", "x")),
        ("dsolve", ("y", "y", "t")),
        ("dsolve", ("y^2", "y", "t")),
        ("dsolve", ("t*y", "y", "t")),
    ],
)
def test_engine_results_are_verified(operation: str, args: tuple) -> None:
    engine = {"integrate": symbolic.integrate, "diff": symbolic.diff, "dsolve": ode.dsolve}
    result = engine[operation](E(args[0]), *args[1:])
    v = certify(operation, *args, result)
    assert v.status is VERIFIED, v
    assert v.method == "symbolic"


@pytest.mark.parametrize(
    "equation, real",
    [
        ("x^2 - 4", False),
        ("x^3 - x - 1", True),
        ("x^3 - x - 1", False),  # three Cardano radicals
        ("x^5 - x - 1", False),  # five CRootOf — simplify cannot reduce them
        ("x^4 - 5*x^2 + 4", True),
        ("(x^2 - 1)/(x - 1)", False),
        ("x^2 + 1", True),  # no real roots, and none listed
    ],
)
def test_engine_roots_of_rational_equations_are_verified(equation: str, real: bool) -> None:
    roots = symbolic.solve(E(equation), "x", real=real)
    v = certify_solve(equation, "x", roots, real=real, budget=60)
    assert v.status is VERIFIED, v
    assert "distinct" in v.detail


def test_engine_limits_are_verified() -> None:
    for expr, to, dir in [("sin(x)/x", "0", "+"), ("1/x", "0", "-"), ("(1+1/x)^x", "oo", "+")]:
        result = symbolic.limit(E(expr), "x", to, dir=dir)
        v = certify_limit(expr, "x", to, result, dir=dir)
        assert v.status is VERIFIED, (expr, v)
        assert v.method == "leading-term"


def test_engine_numerical_integral_is_verified() -> None:
    res = numerics.integrate_num(E("sin(x)"), "x", 0, math.pi, full_result=True)
    v = certify_nintegrate("sin(x)", "x", 0, math.pi, res)
    assert v.status is VERIFIED, v
    # The estimate holds — barely: 1.08245044e-8 against an estimate of 1.08245663e-8.
    assert "error estimate" in v.detail


# --- derivative claims: integrate, diff, dsolve -----------------------------------
def test_constant_of_integration_does_not_matter() -> None:
    assert certify_integrate("2*x", "x", "x^2 + 5").status is VERIFIED


def test_wrong_antiderivative_is_refuted_at_a_readable_point() -> None:
    v = certify_integrate("2*x", "x", "x^3")
    assert v.status is REFUTED
    assert v.method == "finite-differences"
    assert v.counterexample == {"x": "1"}
    assert _derivative_at("x^3", "x", v.counterexample) != _at("2*x", v.counterexample)


def test_wrong_antiderivative_with_a_parameter_names_both_values() -> None:
    v = certify_integrate("a*x", "x", "a*x^2")
    assert v.status is REFUTED
    assert set(v.counterexample or {}) == {"a", "x"}
    assert _derivative_at("a*x^2", "x", v.counterexample or {}) != _at("a*x", v.counterexample or {})


def test_wrong_derivative_is_refuted() -> None:
    v = certify_diff("x*sin(x)", "x", "cos(x)")
    assert v.status is REFUTED
    assert v.counterexample == {"x": "0"}
    assert _derivative_at("x*sin(x)", "x", v.counterexample) != _at("cos(x)", v.counterexample)


def test_a_wrong_engine_derivative_cannot_certify_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole point of finite differences: with ``Expr.diff`` broken (as the v0.3
    # audit found it, for Abs/sign/Min/Max), the broken answer must still be caught
    # and the right one must not be accused on the broken engine's word.
    monkeypatch.setattr(Expr, "diff", lambda self, var: Expr(sp.Integer(0)))
    wrong = certify_diff("x^2", "x", "0")
    assert wrong.status is REFUTED
    assert wrong.method == "finite-differences"
    right = certify_diff("x^2", "x", "2*x")
    assert right.status is UNDECIDED
    assert "do not confirm" in right.detail


def test_kinks_and_jumps_are_skipped_not_counted() -> None:
    # at x = 0 the one-sided quotients of Abs differ, at an integer floor jumps:
    # those points are neither agreement nor a counterexample
    v = certify_diff("Abs(x)", "x", "sign(x)")
    assert v.status is VERIFIED
    assert "15 of 16" in v.detail
    assert certify_diff("floor(x)", "x", "0").status is not REFUTED


def test_wrong_ode_solution_is_refuted_with_its_constant() -> None:
    v = certify_dsolve("y", "y", "t", ["C1*exp(2*t)"])
    assert v.status is REFUTED
    assert set(v.counterexample or {}) == {"C1", "t"}
    point = v.counterexample or {}
    assert _derivative_at("C1*exp(2*t)", "t", point) != _at("C1*exp(2*t)", point)


def test_dsolve_verdict_says_completeness_is_not_claimed() -> None:
    # y' = y^2 also has y = 0, which -1/(C1 + t) never is: the certificate
    # vouches for what was returned, and says what it does not vouch for
    v = certify_dsolve("y^2", "y", "t", ["-1/(C1 + t)"])
    assert v.status is VERIFIED
    assert "not certified" in v.detail


# --- solve -------------------------------------------------------------------------
def test_value_that_is_not_a_root_is_refuted() -> None:
    v = certify_solve("x^2 - 4", "x", ["2", "3"])
    assert v.status is REFUTED
    assert v.counterexample == {"x": "3"}
    assert _at("x^2 - 4", v.counterexample) != 0


def test_value_where_the_equation_is_undefined_is_not_a_root() -> None:
    v = certify_solve("(x^2 - 1)/(x - 1)", "x", ["1", "-1"])
    assert v.status is REFUTED
    assert v.counterexample == {"x": "1"}
    assert "undefined" in v.detail


def test_missing_polynomial_root_is_refuted_by_the_count() -> None:
    v = certify_solve("x^2 - 4", "x", ["2"])
    assert v.status is REFUTED
    assert v.method == "root-count"
    assert v.counterexample is not None
    assert float(parse(v.counterexample["x"]).sy) == pytest.approx(-2.0)


def test_complex_roots_count_unless_real_roots_were_asked_for() -> None:
    assert certify_solve("x^2 + 1", "x", [], real=True).status is VERIFIED
    v = certify_solve("x^2 + 1", "x", [])
    assert v.status is REFUTED
    missing = complex(parse((v.counterexample or {})["x"]).sy)
    assert abs(missing * missing + 1) < 1e-12


def test_missing_complex_crootof_is_located() -> None:
    roots = symbolic.solve(E("x^5 - x - 1"), "x")
    v = certify_solve("x^5 - x - 1", "x", roots[:4])
    assert v.status is REFUTED
    missing = complex(parse((v.counterexample or {})["x"]).sy)
    assert abs(missing**5 - missing - 1) < 1e-12


def test_non_real_value_is_refuted_when_real_roots_were_asked_for() -> None:
    v = certify_solve("x^2 + 1", "x", ["I"], real=True)
    assert v.status is REFUTED
    assert v.counterexample == {"x": "I"}


def test_periodic_equation_is_refuted_for_its_missing_roots() -> None:
    # sympy.solve returns [0, pi] for sin(x) = 0 — a fundamental set, not all roots;
    # the scan finds the one nearest the origin that is not listed
    roots = symbolic.solve(E("sin(x)"), "x", real=True)
    v = certify_solve("sin(x)", "x", roots, real=True)
    assert v.status is REFUTED
    assert v.method == "sign-scan"
    found = float(parse((v.counterexample or {})["x"]).sy)
    assert found == pytest.approx(-math.pi, abs=1e-12)


def test_transcendental_completeness_stays_undecided() -> None:
    v = certify_solve("exp(x) - 2", "x", symbolic.solve(E("exp(x) - 2"), "x"))
    assert v.status is UNDECIDED
    assert "completeness unconfirmed" in v.detail
    assert "1 of 1 listed roots proved" in v.detail


def test_parametric_completeness_is_not_attempted() -> None:
    v = certify_solve("x^2 - a", "x", symbolic.solve(E("x^2 - a"), "x"))
    assert v.status is UNDECIDED
    assert "parameters" in v.detail


def test_scan_does_not_mistake_a_pole_for_a_root() -> None:
    # tan(x) - 1 changes sign at every pole too, and only its genuine roots may be
    # reported; 1/sin(x) and floor(x) + 1/2 change sign all over and have none
    v = certify_solve("tan(x) - 1", "x", symbolic.solve(E("tan(x) - 1"), "x"))
    assert v.status is REFUTED  # pi/4 + k*pi, only pi/4 listed
    found = float(parse((v.counterexample or {})["x"]).sy)
    assert math.tan(found) == pytest.approx(1.0, abs=1e-9)
    assert certify_solve("1/sin(x)", "x", []).status is UNDECIDED
    assert certify_solve("floor(x) + 1/2", "x", []).status is UNDECIDED
    # a rational function is counted, not scanned: 1/x provably has no root
    assert certify_solve("1/x", "x", []).status is VERIFIED


# --- limit ---------------------------------------------------------------------------
def test_wrong_finite_limit_is_refuted_by_the_approach_sequence() -> None:
    v = certify_limit("sin(x)/x", "x", 0, "2")
    assert v.status is REFUTED
    assert v.method == "approach-sequence"
    assert abs(_at("sin(x)/x", v.counterexample or {}) - 2) > 0.9


def test_wrong_infinite_limits_are_refuted() -> None:
    assert certify_limit("1/x", "x", 0, "-oo").status is REFUTED
    assert certify_limit("1/x", "x", 0, "5").status is REFUTED
    assert certify_limit("sin(x)/x", "x", 0, "oo").status is REFUTED


def test_limit_with_a_parameter_is_verified() -> None:
    assert certify_limit("a*sin(x)/x", "x", 0, "a").status is VERIFIED


def test_limit_without_a_closed_leading_term_is_undecided() -> None:
    # x*log(x) -> 0: the sequence agrees, the leading term is x*log(x) itself
    v = certify_limit("x*log(x)", "x", 0, "0")
    assert v.status is UNDECIDED
    assert "1 of 1" in v.detail


def test_slow_limit_is_never_refuted_by_the_sequence() -> None:
    # 1/log(1/x) -> 0, so slowly that at x = 1e-30 it is still 0.0145: the sequence
    # neither settles nor diverges, and a wrong claim is left UNDECIDED, not accused
    assert certify_limit("1/log(1/x)", "x", 0, "1").status is UNDECIDED


# --- nintegrate ----------------------------------------------------------------------
def test_an_error_estimate_that_lies_is_refuted() -> None:
    # engine.numerics documents this case: every node of both grids lands on a
    # maximum of cos(16*pi*x), the estimate is exactly 0.0, the answer is off by 100%
    res = numerics.integrate_num(E("cos(16*pi*x)"), "x", 0, 1, n=4, full_result=True)
    assert res.error_estimate == 0.0 and res.value == 1.0
    v = certify_nintegrate("cos(16*pi*x)", "x", 0, 1, res)
    assert v.status is REFUTED
    assert v.method == "reference-quadrature"
    assert v.counterexample == {}


def test_an_optimistic_estimate_near_a_singular_derivative_is_refuted() -> None:
    # sqrt(x) on [0, 1]: estimate 5.6e-5 against a true error of 8.1e-5 (documented)
    res = numerics.integrate_num(E("sqrt(x)"), "x", 0, 1, full_result=True)
    assert certify_nintegrate("sqrt(x)", "x", 0, 1, res).status is REFUTED


def test_bare_float_is_held_to_rtol() -> None:
    assert certify_nintegrate("sin(x)", "x", 0, "pi", 2.0).status is VERIFIED
    assert certify_nintegrate("sin(x)", "x", 0, "pi", 2.1).status is REFUTED
    assert certify_nintegrate("sin(x)", "x", 0, "pi", 2.0 + 1e-6, rtol=1e-5).status is VERIFIED


def test_integral_that_cancels_to_zero_is_not_refuted_for_rounding() -> None:
    assert certify_nintegrate("sin(x)", "x", 0, 2 * math.pi, 1e-17).status is not REFUTED


def test_numeric_agreement_without_a_closed_form_is_undecided() -> None:
    res = numerics.integrate_num(E("exp(sin(x))"), "x", 0, 1, full_result=True)
    v = certify_nintegrate("exp(sin(x))", "x", 0, 1, res)
    assert v.status is UNDECIDED
    assert "no closed form" in v.detail


def test_a_failed_run_claims_nothing() -> None:
    res = numerics.integrate_num(
        E("sqrt(x)"), "x", 0, 1, tol=1e-14, max_evals=20, full_result=True
    )
    assert res.error_estimate == math.inf
    v = certify_nintegrate("sqrt(x)", "x", 0, 1, res)
    assert v.status is UNDECIDED
    assert v.method == "none"


# --- the entry point, and the contract --------------------------------------------------
def test_certify_dispatches_by_command_token() -> None:
    assert set(CERTIFIERS) == {"integrate", "diff", "solve", "limit", "dsolve", "nintegrate"}
    assert certify("integrate", "2*x", "x", "x^2").status is VERIFIED
    assert certify("solve", "x^2 - 4", "x", ["2", "-2"]).status is VERIFIED
    assert certify("limit", "sin(x)/x", "x", "0", "1", dir="+").status is VERIFIED


def test_every_verdict_is_a_verdict_and_repeatable() -> None:
    first = certify_solve("sin(x)", "x", ["0", "pi"])
    assert isinstance(first, Verdict)
    assert certify_solve("sin(x)", "x", ["0", "pi"]) == first


@pytest.mark.parametrize(
    "call",
    [
        lambda: certify("nope", "x"),
        lambda: certify("integrate", "x"),
        lambda: certify("integrate", "2*x", "x", "x^2", colour="red"),
        lambda: certify_integrate(object(), "x", "x"),  # type: ignore[arg-type]
        lambda: certify_integrate("x", "1x", "x"),
        lambda: certify_integrate("x", "x", "x", budget=0),
        lambda: certify_integrate("x", "x", "x", budget=math.nan),
        lambda: certify_integrate("x", "x", "x", budget=True),
        lambda: certify_integrate("[[1, 2]]", "x", "x"),
        lambda: certify_solve("x^2", "x", "0"),
        lambda: certify_solve("x^2", "x", ["0"], real="yes"),  # type: ignore[arg-type]
        lambda: certify_limit("1/x", "x", 0, "oo", dir="left"),
        lambda: certify_limit("1/x", "x", "a", "oo"),
        lambda: certify_limit("1/x", "x", 0, Expr(sp.nan)),
        lambda: certify_dsolve("y", "y", "y", ["1"]),
        lambda: certify_dsolve("y", "y", "t", []),
        lambda: certify_dsolve("y", "y", "t", ["y + t"]),
        lambda: certify_nintegrate("a*x", "x", 0, 1, 0.5),
        lambda: certify_nintegrate("x", "x", 0, "b", 0.5),
        lambda: certify_nintegrate("x", "x", 0, 1, "0.5"),  # type: ignore[arg-type]
        lambda: certify_nintegrate("x", "x", 0, 1, math.nan),
        lambda: certify_nintegrate("x", "x", 0, 1, 0.5, rtol=0),
    ],
)
def test_bad_input_is_a_domain_error(call) -> None:
    with pytest.raises(DomainError):
        call()


def test_text_that_does_not_parse_is_a_parse_error() -> None:
    with pytest.raises(ParseError):
        certify_integrate("2*", "x", "x^2")
    with pytest.raises(PycodemathError):
        certify_solve("x^2 - 4", "x", ["import os"])


# --- the time budget -----------------------------------------------------------------
def _slow_claim() -> tuple[Expr, Expr]:
    # d/dx sin(x+y+z+w)^3 against its expand_trig form: finite differences take
    # milliseconds, the symbolic proof ~2 s (measured, warm)
    s = sp.sin(sum(sp.symbols("x y z w")))
    return Expr(s**3), Expr(sp.expand_trig(3 * s**2 * sp.cos(sum(sp.symbols("x y z w")))))


def test_budget_running_out_is_undecided_not_an_exception() -> None:
    v = certify_integrate("2*x", "x", "x^2", budget=1e-6)
    assert v.status is UNDECIDED
    assert v.method == "time-budget"


def test_out_of_time_verdict_keeps_the_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    # the symbolic phase never returns; the finite differences it follows did finish
    def forever(*args: object, **kwargs: object) -> Verdict:
        while True:
            pass

    # ``pycodemath.verify.certify`` is the FUNCTION (the package re-exports it
    # under the module's name), so the module is reached through sys.modules
    monkeypatch.setattr(sys.modules["pycodemath.verify.certify"], "check_equal", forever)
    v = certify_diff("x^2", "x", "2*x", budget=1.5)
    assert v.status is UNDECIDED
    assert v.method == "time-budget"
    assert "during symbolic differentiation" in v.detail
    assert "finite differences agree at 16 of 16" in v.detail


def test_does_not_poison_an_enclosing_time_budget() -> None:
    # V1's finding: a tighter budget nested in another poisons the outer when it
    # fires. Inside a caller's budget a certificate arms nothing, so the enclosing
    # block survives it and keeps working.
    f, g = _slow_claim()
    with time_budget(60):
        certify_diff(f, "x", g, budget=0.1)
        assert check_equal("sqrt(x^2)", "x").status is REFUTED


def test_an_enclosing_budget_that_expires_refuses_the_enclosing_block() -> None:
    f, g = _slow_claim()
    with pytest.raises(TimeBudgetError):
        with time_budget(0.3):
            certify_diff(f, "x", g)


# --- V8: casus irreducibilis, quartics ------------------------------------------
def _casus_cubics(count: int, seed: int = 20260929) -> list[sp.Expr]:
    """Irreducible cubics with three real roots, drawn with a seed: integer roots
    at least 3 apart, the polynomial moved off them by 1 — so SymPy writes every
    real root with complex cube roots (casus irreducibilis)."""
    import random

    x = sp.Symbol("x")
    rng = random.Random(seed)
    found: list[sp.Expr] = []
    while len(found) < count:
        a, b, c = sorted(rng.sample(range(-9, 10), 3))
        if b - a < 3 or c - b < 3:
            continue
        p = sp.expand((x - a) * (x - b) * (x - c) + rng.choice([1, -1]))
        poly = sp.Poly(p, x)
        if poly.discriminant() > 0 and len(poly.factor_list()[1]) == 1:
            found.append(p)
    return found


@pytest.mark.parametrize("p", _casus_cubics(3), ids=str)
def test_casus_irreducibilis_cubics_are_verified(p: sp.Expr) -> None:
    roots = symbolic.solve(Expr(p), "x", real=True)
    assert len(roots) == 3 and all(r.sy.has(sp.I) for r in roots)
    v = certify_solve(Expr(p), "x", roots, real=True)
    assert v.status is VERIFIED, v.detail
    assert "exactly 3 distinct real roots" in v.detail


def test_a_casus_root_has_a_value_and_a_truly_complex_root_keeps_its_imaginary_part() -> None:
    from pycodemath.verify.equal import _evaluate

    casus = symbolic.solve(parse("x^3 - 8*x^2 - 3*x + 92"), "x")[0].sy
    raw = casus.evalf(30).as_real_imag()[1]
    assert raw._prec == 1  # the defect's precondition: a 1-bit imaginary part
    value = _evaluate(casus, {}, 30, exact=False)
    assert value is not None and value.imag == 0
    complex_root = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2
    value = _evaluate(complex_root, {}, 30, exact=False)
    assert value is not None and abs(value.imag - math.sqrt(3) / 2) < 1e-15


def test_a_truly_complex_root_is_still_refuted_when_real_roots_are_asked_for() -> None:
    roots = symbolic.solve(parse("x^3 - 2"), "x", real=False)
    v = certify_solve("x^3 - 2", "x", roots, real=True)
    assert v.status is REFUTED and "is not real" in v.detail


def test_a_low_precision_zero_with_no_magnitude_is_never_negligible() -> None:
    from pycodemath.verify.equal import _negligible

    assert not _negligible(sp.Float(0, 1), sp.Integer(5), 30)
    assert _negligible(sp.Float("1e-40", 1), sp.Integer(5), 30)
    assert not _negligible(sp.Float("1e-25", 1), sp.Integer(5), 30)


QUARTIC = "-2*x^4 - 10*x^3 - 2*x^2 + 42*x + 37"


def test_a_ferrari_quartic_out_of_budget_is_the_numeric_verdict_never_verified() -> None:
    roots = symbolic.solve(parse(QUARTIC), "x")
    v = certify_solve(QUARTIC, "x", roots, budget=1)
    assert v.status is UNDECIDED and v.method == "numeric-roots", v.detail
    assert "0 of 4 listed roots proved" in v.detail
    assert "4 of the 4 not proved by substitution agree" in v.detail
    assert "exactly 4 distinct roots" in v.detail
    assert "not a proof" in v.detail


def test_a_wrong_root_of_a_quartic_is_refuted_by_its_value() -> None:
    roots = symbolic.solve(parse(QUARTIC), "x")[:3] + [Expr(sp.Integer(2))]
    v = certify_solve(QUARTIC, "x", roots, budget=5)
    assert v.status is REFUTED and v.method == "numeric-substitution"
    assert v.counterexample == {"x": "2"}


def test_a_missing_root_of_a_quartic_is_still_found() -> None:
    roots = symbolic.solve(parse(QUARTIC), "x")[:3]
    v = certify_solve(QUARTIC, "x", roots, budget=5)
    assert v.status is REFUTED and v.method == "root-count"


@pytest.mark.parametrize(
    "p, roots",
    [
        ("x^4 - 5*x^2 + 4", ["1", "-1", "2", "-2"]),
        ("x^4 - 10*x^2 + 1", ["sqrt(2)+sqrt(3)", "sqrt(2)-sqrt(3)",
                              "-sqrt(2)+sqrt(3)", "-sqrt(2)-sqrt(3)"]),
        ("x^4 - 4*x^2 + 1", None),  # nested radicals: proved by minimal polynomial
    ],
)
def test_quartics_with_provable_roots_are_still_verified(p: str, roots) -> None:
    listed = roots if roots is not None else symbolic.solve(parse(p), "x")
    v = certify_solve(p, "x", listed, budget=30)
    assert v.status is VERIFIED, v.detail


def test_certify_reads_a_decimal_exponent_as_written() -> None:
    v = certify_solve("x^2 - 2", "x", ["2^0.5", "-2^0.5"])
    assert v.status is VERIFIED, v.detail
