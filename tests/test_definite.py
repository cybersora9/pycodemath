"""K5 — certified definite and improper integrals (``engine.definite``)."""

from __future__ import annotations

import pytest

from pycodemath.engine.definite import definite_integral


@pytest.mark.parametrize(
    "f, a, b, value",
    [
        ("1/x^2", "-1", "1", "oo"),          # the classic: FTC through the pole gives -2
        ("1/(x-1)^2", "0", "3", "oo"),
        ("1/cos(x)^2", "0", "2*pi", "oo"),  # sympy#11079 answered 0
        ("1/x", "-1", "1", None),            # +oo on one side, -oo on the other: no value
        ("1/(x^2-1)", "-2", "2", None),
    ],
)
def test_pole_inside_interval_is_never_integrated_through(f, a, b, value):
    """STRAŻNIK K5: a pole strictly inside [a, b] is found, and the answer is a
    PROVED divergence — never a finite number obtained by stepping over it."""
    res = definite_integral(f, "x", a, b)
    assert res.converges is False
    assert res.value == value
    assert res.singularities, "the pole must be reported"
    assert res.approx is None


import json  # noqa: E402

import sympy as sp  # noqa: E402

from pycodemath.core.budget import time_budget  # noqa: E402
from pycodemath.core.errors import DomainError, TimeBudgetError, UnsupportedFormError  # noqa: E402
from pycodemath.engine import definite as D  # noqa: E402
from pycodemath.engine.definite import IntegralNoClosedForm, IntegralUndecided  # noqa: E402


@pytest.mark.parametrize(
    "f, a, b, value",
    [
        ("exp(-x^2)", "-oo", "oo", "sqrt(pi)"),
        ("1/sqrt(x)", "0", "1", "2"),
        ("log(x)", "0", "1", "-1"),
        ("1/x^2", "1", "oo", "1"),
        ("sin(x)/x", "0", "oo", "pi/2"),       # Dirichlet's test + quadosc
        ("1/(exp(x)+1)", "0", "oo", "log(2)"),  # sympy#10445
        ("x^2*sqrt(1-x^2)", "0", "1", "pi/16"),  # sympy#20941, via the antiderivative
    ],
)
def test_convergent_values_are_cross_checked(f, a, b, value):
    r = definite_integral(f, "x", a, b)
    assert r.converges and r.value == value and r.proof == "cross-check"
    assert float(r.numeric["error"]) < 1e-10


def test_sympy_definite_answer_refuted_and_replaced():
    """sympy#24004: SymPy's definite integral is 0; the antiderivative with one-sided
    limits on each cut piece gives 2*pi, which the quadrature confirms."""
    r = definite_integral("sin(x)*atan2(sin(x), cos(x))", "x", "-pi", "pi")
    assert r.value == "2*pi" and r.sympy_value == "0"
    assert "REFUTED" in r.certificate


def test_disagreement_without_a_second_route_refuses(monkeypatch):
    monkeypatch.setattr(D, "_closed_form", lambda f, x, seq, route: sp.Integer(3) if route == "definite" else None)
    with pytest.raises(IntegralUndecided) as exc:
        definite_integral("x", "x", 0, 1)
    assert exc.value.sympy_value == "3" and exc.value.approx.startswith("0.5")
    assert exc.value.route == "nintegrate"


def test_divergence_directions_and_proof():
    r = definite_integral("1/x", "x", 0, 1)
    assert (r.converges, r.value, r.proof) == (False, "oo", "proof")
    assert r.behaviour[0]["verdict"] == "diverges"
    assert definite_integral("-1/x", "x", 0, 1).value == "-oo"
    assert definite_integral("exp(x)", "x", 0, "oo").value == "oo"


def test_oscillating_divergence_has_no_value():
    r = definite_integral("sin(x)", "x", 0, "oo")
    assert r.converges is False and r.value is None


def test_principal_value_only_on_request():
    plain = definite_integral("1/x", "x", -1, 1)
    assert plain.value is None and "principal_value=True" in plain.certificate
    pv = definite_integral("1/x", "x", -1, 1, principal_value=True)
    assert pv.value == "0" and pv.principal_value and pv.converges is False
    pv2 = definite_integral("1/(x-1/2)", "x", 0, 2, principal_value=True)
    assert sp.simplify(sp.sympify(pv2.value) - sp.log(3)) == 0
    # a double pole has no principal value either
    assert definite_integral("1/x^2", "x", -1, 1, principal_value=True).value == "oo"


def test_reversed_and_empty_bounds():
    assert definite_integral("x", "x", 1, 0).value == "-1/2"
    assert definite_integral("1/x", "x", 1, 0).value == "-oo"
    assert definite_integral("1/x", "x", 2, 2).value == "0"


def test_no_closed_form_keeps_the_number():
    with pytest.raises(IntegralNoClosedForm) as exc:
        definite_integral("x/(exp(x)-1)", "x", 0, "oo")
    assert exc.value.route == "nintegrate" and exc.value.approx.startswith("1.644934066848")


def test_integrand_not_real_is_a_domain_error():
    with pytest.raises(DomainError):
        definite_integral("log(x)", "x", -1, 1)
    with pytest.raises(DomainError):
        definite_integral("sqrt(x)", "x", -2, -1)


@pytest.mark.parametrize(
    "f, a, b",
    [("x*y", "0", "1"), ("gamma(x)", "1", "2"), ("x", "0", "a")],
)
def test_out_of_scope_integrals_are_refused(f, a, b):
    with pytest.raises(UnsupportedFormError):
        definite_integral(f, "x", a, b)


def test_time_budget_routes_to_nintegrate():
    with pytest.raises(TimeBudgetError) as exc:
        with time_budget(0.05):
            definite_integral("exp(-x^2)*log(x)/(x^2+1)", "x", 0, "oo")
    assert exc.value.route == "nintegrate"


def test_result_to_dict_is_json():
    d = definite_integral("1/(cos(x)+2)", "x", 0, "2*pi").to_dict()
    json.dumps(d)
    assert d["kind"] == "definite_integral" and d["value"] == "2*sqrt(3)*pi/3"
    assert d["verdict"] == "verified"


def test_oscillating_absolutely_convergent_tail_uses_quadosc():
    """sin(x)^2/x^2 on R: tanh-sinh alone came back with error 8e-4 (refused)."""
    r = definite_integral("sin(x)^2/x^2", "x", "-oo", "oo")
    assert r.value == "pi" and "quadosc" in r.numeric["method"]


def test_divergent_both_ways_on_the_line_has_no_value():
    r = definite_integral("x", "x", "-oo", "oo")
    assert r.converges is False and r.value is None
