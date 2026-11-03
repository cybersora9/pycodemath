"""0.5.1 — fixes from a hands-on session (49 CLI calls against 0.5.0, 2026-11-03).

Every test below except the regression block at the end was seen FAILING on the
0.5.0 code before its fix went in.
"""

from __future__ import annotations

import json
import math

import pytest
import sympy as sp

from pycodemath import parse
from pycodemath.cli import mcp_server, repl
from pycodemath.core.errors import NoClosedFormError, ParseError
from pycodemath.engine import ode, symbolic
from pycodemath.engine.definite import definite_integral
from pycodemath.verify import recheck


# --- 1. money loan: "for N years" is a duration paid monthly ------------------------
def test_loan_for_years_is_monthly_by_default():
    out = repl.handle("money loan 300000 PLN at 7.2% for 25 years")
    assert out.startswith("300 monthly instalments: payment 2158.77 PLN"), out
    assert "VERIFIED" in out


def test_loan_for_months_unchanged_and_named():
    out = repl.handle("money loan 300000 PLN at 7.2% for 300 months")
    assert out.startswith("300 monthly instalments: payment 2158.77 PLN"), out


@pytest.mark.parametrize("flag", ["yearly", "annual", "annually"])
def test_loan_yearly_flag_gives_annual_instalments(flag: str):
    out = repl.handle(f"money loan 300000 PLN at 7.2% for 25 years {flag}")
    assert out.startswith("25 yearly instalments: payment 26208.67 PLN"), out


def test_loan_yearly_with_schedule_rows():
    out = repl.handle("money loan 1000 PLN at 10% for 2 years yearly schedule")
    rows = [line for line in out.splitlines() if line.startswith("  ") and line.split()[0].isdigit()]
    assert len(rows) == 2


def test_loan_yearly_needs_years():
    with pytest.raises(ParseError, match="yearly"):
        repl.handle("money loan 1000 PLN at 10% for 12 months yearly")


def test_loan_mcp_matches_cli_and_its_certificate_rechecks():
    r = mcp_server.math_eval("money loan 300000 PLN at 7.2% for 25 years")
    assert r["text"].startswith("300 monthly instalments: payment 2158.77 PLN"), r["text"]
    inputs = json.loads(r["certificate"]["claim"]["args"]["inputs"])
    assert inputs["periods"] == 300 and inputs["per_year"] == 12
    assert recheck(r["certificate"]).ok
    # the tool description an agent reads says the same thing
    assert "for 25 years" in (mcp_server.math_eval.__doc__ or "")
    assert "yearly" in (mcp_server.math_eval.__doc__ or "")


def test_interest_names_the_compounding_period():
    out = repl.handle("money interest 1000 PLN at 5% for 10 years")
    assert out.startswith("1628.89 PLN rounded once"), out  # the value is unchanged
    assert "compounded yearly, 10 years" in out
    assert "compounded monthly, 12 months" in repl.handle("money interest 1000 PLN at 5% for 12 months")


# --- 2. dsolve's hint names the CLI command, not a Python function -------------------
def test_dsolve_hint_names_the_ode_command():
    with pytest.raises(NoClosedFormError) as ei:
        repl.handle("dsolve y^2 + t for y(t)")
    msg = str(ei.value)
    assert "solve_ode_num" not in msg
    assert "ode <expr> for y(t) from <t0> to <t1> at <y0>" in msg
    assert ei.value.route == "ode"


def test_dsolve_hint_uses_the_given_names():
    with pytest.raises(NoClosedFormError, match=r"for u\(s\) from"):
        ode.dsolve(parse("u^2 + s"), "u", "s")


# --- 3. sum of p(k)*binomial(n, k) ---------------------------------------------------
def _brute(expr: str, lo: int, n: int) -> sp.Integer:
    e, k, N = sp.sympify(expr.replace("^", "**")), sp.Symbol("k"), sp.Symbol("n")
    return sum((e.subs({N: n, k: i}) for i in range(lo, n + 1)), sp.Integer(0))


def test_sum_k3_binomial_closed_form():
    out = repl.handle("sum k^3*binomial(n,k) for k from 0 to n")
    n = sp.Symbol("n")
    assert sp.simplify(sp.sympify(out) - n**2 * (n + 3) * 2 ** (n - 3)) == 0


@pytest.mark.parametrize("expr, lo", [
    ("k^3*binomial(n,k)", 0),
    ("k^5*binomial(n,k)", 0),
    ("(2*k^4 - 3*k + 7)*binomial(n,k)", 0),
    ("(k^2 + n*k + 1)*binomial(n,k)", 1),
    ("k^2*binomial(n,k)", 2),
])
def test_sum_binomial_moment_matches_brute_force(expr: str, lo: int):
    closed = symbolic.summation(parse(expr), "k", lo, "n").sy
    n = sp.Symbol("n")
    for v in range(max(lo - 1, 0), 21):
        assert sp.nsimplify(closed.subs(n, v)) == _brute(expr, lo, v), (expr, v)


def test_sum_binomial_other_shapes_still_refused():
    # top of the binomial is not the upper bound: not this rule
    with pytest.raises(NoClosedFormError):
        symbolic.summation(parse("k^3*binomial(n,k)/(k+1)^2"), "k", 0, "n")


# --- 4. Bose–Einstein / Fermi–Dirac integrals ----------------------------------------
@pytest.mark.parametrize("f, value", [
    ("x^3/(exp(x)-1)", sp.pi**4 / 15),
    ("x/(exp(x)-1)", sp.pi**2 / 6),
    ("x^3/(exp(x)+1)", 7 * sp.pi**4 / 120),
    ("x^2/(exp(2*x)-1)", sp.zeta(3) / 4),
])
def test_bose_fermi_family(f: str, value: sp.Expr):
    r = definite_integral(f, "x", 0, "oo")
    assert r.converges and r.proof == "cross-check"
    assert sp.simplify(sp.sympify(r.value) - value) == 0, r.value
    assert "integral Γ(s)" in r.certificate or "Γ(s)ζ(s)" in r.certificate


def test_bose_einstein_on_the_cli():
    out = repl.handle("integrate x^3/(exp(x)-1) dx from 0 to oo")
    assert out.startswith("pi**4/15"), out
    assert "VERIFIED (cross-check)" in out


# --- 5. sum(...)/integrate(...) inside an expression point at the commands ----------
@pytest.mark.parametrize("line, command", [
    ("verify sum(k^3*binomial(n,k), (k,0,n)) == n^2*(n+3)*2^(n-3)", "sum <expr> for <k> from"),
    ("evalf integrate(x, (x,0,1)) 60", "integrate <expr> dx from <a> to <b>"),
    ("verify Sum(k, (k, 1, n)) == n*(n+1)/2", "sum <expr> for <k> from"),
    ("evalf Integral(x, (x, 0, 1)) 30", "integrate <expr> dx"),
])
def test_sum_integrate_calls_point_to_commands(line: str, command: str):
    with pytest.raises(ParseError) as ei:
        repl.handle(line)
    msg = str(ei.value)
    assert command in msg and "product" not in msg, msg


def test_other_unknown_functions_keep_the_product_hint():
    with pytest.raises(ParseError, match=r"write foo\*\(\.\.\.\) if a product is meant"):
        parse("foo(x)")


# --- regression: worked in the session, not pinned elsewhere at the CLI -----------
def test_session_regressions():
    assert repl.handle("limit (tan(sin(x))-sin(tan(x)))/x^7 for x to 0").strip() == "1/30"
    out = repl.handle("factor 2^128+1")
    assert out.startswith("59649589127497217 * 5704689200685129054721") and "VERIFIED" in out
    out = repl.handle("integrate exp(-x^2)*cos(3*x) dx from -oo to oo")
    assert out.startswith("sqrt(pi)*exp(-9/4)") and "VERIFIED" in out
    out = repl.handle("solve x^2+y^2=25, x*y=12 for x, y")
    assert "x = 3, y = 4" in out and "VERIFIED (groebner)" in out
    assert math.isclose(float(sp.N(sp.pi**4 / 15)), 6.493939402266829)
