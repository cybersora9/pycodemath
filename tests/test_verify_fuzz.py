"""Property-based fuzzing of the engine and of the verifier itself (VERIFY V5).

The oracles are the verifier's own contracts, checked against inputs nobody wrote
by hand:

* ``certify_integrate(f, x, integrate(f))`` is never REFUTED, and neither is
  ``certify_diff(f, x, diff(f))`` or ``certify_solve`` on a polynomial's roots —
  the engine's answer, checked by a route that did not produce it;
* ``check_equal(e, simplify(e))`` is never REFUTED — ``simplify`` is an identity;
* ``check_equal(e, e + eps*x)`` is ALWAYS REFUTED wherever ``e`` has a value at a
  sample point with ``x != 0`` — a verifier that misses a perturbation it can see
  is blind;
* ``check_steps`` on a chain built from valid rewrites never names a wrong step;
* ``parse(e.to_source())`` is equivalent to ``e`` or a refused unknown function
  — never a different expression (module C);
* every failure, on any text, is a ``PycodemathError``.

DETERMINISM. By default every test runs ``derandomize=True`` (the examples are a
function of the test, not of the clock or of a database) with a small
``max_examples``: measured, the file costs ~16 s and the whole gate grew from
57-59 s to 76-77 s — V5's allowance was 30 s. ``PYCODEMATH_FUZZ=deep`` switches to a randomised profile with 25x the
examples, for hunting; anything it finds becomes a regression test below, not a
flaky one here.

WHAT IS EXCLUDED, AND WHY. The generators avoid time-budget expiry on purpose: a
budget that fires is finding A (``_Deadline`` leaking from the budget gate, a
local, Windows-measured module), and a fuzz test that fired budgets would turn A
into flakiness in the whole gate. Known engine defects the fuzz reaches are
pinned as ``xfail(strict=True)`` regression tests, and the fuzz steps around them
with a predicate that names the defect — so when the engine is fixed, the strict
xfail turns red and the exclusion is removed with it. Since V9 none is left: the
contract properties run on every text the generator makes, with no exclusion.
"""

from __future__ import annotations

import os

import pytest
import sympy as sp
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pycodemath import parse
from pycodemath.core.errors import ParseError, PycodemathError
from pycodemath.core.ir import Expr
from pycodemath.engine.symbolic import diff, integrate, simplify, solve
from pycodemath.verify import (
    VerdictStatus,
    certify_diff,
    certify_integrate,
    certify_solve,
    check_equal,
    check_steps,
)
from pycodemath.verify.equal import (
    _DPS,
    _NICE,
    _recheck,
    _undefined_as_written,
    _value,
)

REFUTED = VerdictStatus.REFUTED

_DEEP = os.environ.get("PYCODEMATH_FUZZ", "").lower() == "deep"


def _fuzz(examples: int):
    """The settings every property here runs under — see the module docstring."""
    return settings(
        max_examples=examples * 25 if _DEEP else examples,
        derandomize=not _DEEP,
        database=None,
        deadline=None,
        print_blob=_DEEP,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )


# --- generators ------------------------------------------------------------

#: ``x`` is a third of the leaves: a tree with no ``x`` is useless to three of the
#: four properties (measured with a flat choice: 63% of the derivative property's
#: draws were rejected for having no ``x``).
_ATOMS = st.one_of(
    st.just("x"),
    st.sampled_from(["y", "x", "2*x", "x^2", "x + 1"]),
    st.sampled_from(["1", "2", "3", "-1", "1/2", "-2/3", "pi", "E", "sqrt(2)"]),
)
_UNARY = st.sampled_from(
    ["sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "atan", "asin", "sinh",
     "cosh", "tanh"]
)
_BINARY = st.sampled_from(["+", "-", "*", "/"])
_POWERS = st.sampled_from(["2", "3", "-1", "1/2", "-2"])


def _tree(children: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    return st.one_of(
        st.builds(lambda f, a: f"{f}({a})", _UNARY, children),
        st.builds(lambda a, op, b: f"({a}{op}{b})", children, _BINARY, children),
        st.builds(lambda a, p: f"({a})^({p})", children, _POWERS),
    )


#: Random elementary expressions in x (and sometimes y), up to 8 leaves.
expressions = st.recursive(_ATOMS, _tree, max_leaves=8)

_K = st.sampled_from(["1", "2", "3", "-1", "1/2", "-3"])
_N = st.sampled_from(["0", "1", "2", "3"])
_A = st.sampled_from(["1", "2", "-1", "3", "1/2"])
_C = st.sampled_from(["1", "2", "-1", "3", "-5", "1/3", "7/2"])

#: One term of an integrand SymPy integrates in well under a second — the
#: families an antiderivative table is made of, with random parameters.
_TERM = st.one_of(
    st.builds(lambda n: f"x^{n}", _N),
    st.builds(lambda k: f"sin({k}*x)", _K),
    st.builds(lambda k: f"cos({k}*x)", _K),
    st.builds(lambda k: f"exp({k}*x)", _K),
    st.builds(lambda n, k: f"x^{n}*exp({k}*x)", _N, _K),
    st.builds(lambda n, k: f"x^{n}*sin({k}*x)", _N, _K),
    st.builds(lambda k, m: f"sin({k}*x)*cos({m}*x)", _K, _K),
    st.builds(lambda k: f"sin({k}*x)^2", _K),
    st.builds(lambda a: f"1/(x+{a})", _A),
    st.builds(lambda a: f"1/(x^2+{a}^2)", _A),
    st.builds(lambda n: f"x^{n}*log(x)", _N),
    st.builds(lambda a: f"sqrt(x+{a})", _A),
    st.builds(lambda a: f"x/(x^2+{a})", _A),
    st.just("atan(x)"),
    st.just("1/(1+exp(x))"),
)
integrands = st.lists(st.tuples(_C, _TERM), min_size=1, max_size=3).map(
    lambda terms: " + ".join(f"({c})*{t}" for c, t in terms)
)


# --- the oracles -----------------------------------------------------------

@_fuzz(40)
@given(integrands)
def test_certify_never_refutes_the_engines_integral(f):
    F = integrate(parse(f), "x")
    verdict = certify_integrate(f, "x", F)
    assert verdict.status is not REFUTED, (f, F.sy, verdict.detail)


@_fuzz(40)
@given(expressions)
def test_certify_never_refutes_the_engines_derivative(f):
    e = parse(f)
    assume("x" in e.symbol_names())
    g = diff(e, "x")
    verdict = certify_diff(e, "x", g)
    assert verdict.status is not REFUTED, (f, g.sy, verdict.detail)


@_fuzz(60)
@given(expressions)
def test_simplify_is_never_refuted(e):
    source = parse(e)
    simplified = simplify(source)
    verdict = check_equal(source, simplified)
    assert verdict.status is not REFUTED, (e, simplified.sy, verdict.detail)


#: ``expressions`` plus ``0`` and ``oo``: a division by a zero subtree is what
#: makes ``zoo``/``nan`` (printed as names the parser read as letters before
#: module C), ``oo`` what makes ``AccumBounds`` (``sin(oo)``, off the whitelist).
#: No decimals: a Float computed from one prints to 15 digits, the documented
#: boundary pinned in test_core, not a round-trip this property can hold.
roundtrip_expressions = st.recursive(
    st.one_of(_ATOMS, st.sampled_from(["0", "oo"])), _tree, max_leaves=8
)


@_fuzz(100)
@given(roundtrip_expressions)
def test_to_source_reads_back_as_an_equivalent_expression(text):
    # Module C's contract (Expr.to_source): parse(e.to_source()) is equivalent to
    # e — never a different expression — or the parser refuses a function it does
    # not know. ``==`` is not promised: parse re-evaluates what it reads.
    e = parse(text)
    source = e.to_source()
    try:
        back = parse(source)
    except ParseError as exc:
        assert "unknown function" in str(exc), (text, source, str(exc))
        return
    if back == e:
        return
    # A name misread as letters shows up as a variable the original did not have
    # (zoo -> o**2*z); check_equal alone left 107 of 114 such cases UNDECIDED.
    assert back.sy.free_symbols <= e.sy.free_symbols, (text, source, back.sy)
    verdict = check_equal(e, back)
    assert verdict.status is not REFUTED, (text, source, back.sy, verdict.detail)


_EPS = st.sampled_from(["1", "1/1000", "10^-6", "10^-12", "-10^-15"])


def _visible_point(e: sp.Expr, eps: sp.Expr) -> "dict | None":
    """A readable sample point (``x != 0``) where ``e`` and ``e + eps*x`` both have
    trusted values and ``eps*x`` is not lost under the relative tolerance — the
    precondition under which ``check_equal`` MUST see the perturbation, because it
    samples that very point (``_NICE`` comes first, before any random point)."""
    x = sp.Symbol("x")
    others = sorted(e.free_symbols - {x}, key=lambda s: s.name)
    for i, v in enumerate(_NICE):
        if v == 0:
            continue
        point = {x: v}
        point.update({s: _NICE[(i + 3 * (j + 1)) % len(_NICE)] for j, s in enumerate(others)})
        value = _value(e, point, _DPS)
        if value is None or _value(e + eps * x, point, _DPS) is None:
            continue
        if abs(value) * sp.Float("1e-17") < abs(eps * v):
            return point
    return None


@_fuzz(60)
@given(expressions, _EPS)
def test_a_visible_perturbation_is_always_refuted(e, eps):
    source = parse(e).sy
    shift = parse(eps).sy
    x = sp.Symbol("x")
    perturbed = source + shift * x
    # nan swallows the shift: ``sin(log(0))`` is nan, and so is nan + x
    assume(x in perturbed.free_symbols)
    # check_equal gives the variables their values in name order; _visible_point
    # mirrors that with x first, which it is (x, y are the only names)
    assume(min(perturbed.free_symbols, key=lambda s: s.name) == x)
    assume(_visible_point(source, shift) is not None)
    verdict = check_equal(Expr(source), Expr(perturbed))
    assert verdict.status is REFUTED, (e, eps, verdict.detail)


# --- check_steps on chains that are right by construction -------------------

_SMALL = st.integers(-6, 6)
_NONZERO = _SMALL.filter(bool)


@st.composite
def expression_chains(draw) -> list[str]:
    """A product of linear factors, then SymPy's own valid rewrites of it."""
    x = sp.Symbol("x")
    factors = draw(st.lists(st.tuples(_NONZERO, _SMALL), min_size=1, max_size=3))
    start = sp.Mul(*(a * x + b for a, b in factors), evaluate=False)
    extra = draw(st.sampled_from([0, 1, sp.Rational(1, 2), -3]))
    shift = extra * x ** draw(st.integers(0, 2))
    chain = [start + shift]
    expanded = sp.expand(start) + shift
    chain.append(expanded)
    chain.append(sp.expand(expanded))
    chain.append(sp.factor(sp.expand(expanded)))
    if draw(st.booleans()):
        chain.append(sp.horner(sp.expand(expanded)))
    return [sp.sstr(c) for c in chain]


@st.composite
def equation_chains(draw) -> list[str]:
    """``a x + b = c x + d`` solved the way a student does, each step valid."""
    x = sp.Symbol("x")
    a, c = draw(_SMALL), draw(_SMALL)
    assume(a != c)
    b, d = draw(_SMALL), draw(_SMALL)
    k = draw(_NONZERO)
    m = draw(_SMALL)
    steps = [
        (a * x + b, c * x + d),
        (a * x + b + m, c * x + d + m),         # add m to both sides
        ((a - c) * x, d - b),                   # collect
        (k * (a - c) * x, k * (d - b)),         # multiply by k != 0
        (x, sp.Rational(d - b, a - c)),         # divide
    ]
    return [f"{sp.sstr(lhs)} = {sp.sstr(rhs)}" for lhs, rhs in steps]


@st.composite
def root_chains(draw) -> list[str]:
    """``(x - r)(x - s) = 0`` expanded, then split into its roots."""
    r, s = draw(_SMALL), draw(_SMALL)
    x = sp.Symbol("x")
    product = (x - r) * (x - s)
    return [
        f"{sp.sstr(sp.expand(product))} = 0",
        f"{sp.sstr(product)} = 0",
        f"x = {r} or x = {s}",
    ]


@_fuzz(25)
@given(st.one_of(expression_chains(), equation_chains(), root_chains()))
def test_a_correct_chain_never_has_a_wrong_step(chain):
    result = check_steps(chain)
    wrong = [c for c in result.checks if c.verdict.status is REFUTED]
    assert result.first_error is None and not wrong, (
        chain, [(c.number, c.verdict.detail) for c in wrong]
    )


# --- solve, on the shape where completeness is provable ---------------------

@st.composite
def polynomials(draw) -> sp.Expr:
    """A polynomial with integer roots, moved off them by a constant for about
    half the draws — so irrational and complex roots are in the mix. Shifted only
    up to degree 3: an irreducible quartic's Ferrari radicals take ``certify_solve``
    ~5 s per root to prove (V8; before it, the whole 120 s budget), too slow for
    20 draws, and the fuzz must not fire budgets."""
    x = sp.Symbol("x")
    roots = draw(st.lists(_SMALL, min_size=1, max_size=4))
    lead = draw(_NONZERO)
    shift = draw(st.sampled_from([0, 1, -2])) if len(roots) <= 3 else 0
    return sp.expand(lead * sp.Mul(*(x - r for r in roots)) + shift)


@_fuzz(20)
@given(polynomials(), st.booleans())
def test_certify_never_refutes_the_roots_of_a_polynomial(p, real):
    listed = solve(Expr(p), "x", real=real)
    verdict = certify_solve(Expr(p), "x", listed, real=real)
    assert verdict.status is not REFUTED, (p, [r.sy for r in listed], verdict.detail)


# --- the contract: every failure is a PycodemathError -----------------------

#: Fragments, not characters: ``sin``, ``oo`` and ``zoo`` whole, so the text
#: reaches the parser's name handling and SymPy's special values, not just its
#: tokenizer. Unbalanced brackets, dangling operators and ``=`` are all in reach;
#: ``[`` and ``]`` since V9 (a ``)`` closing a ``[`` leaked a raw IndexError the
#: alphabet without them could not reach).
_TOKENS = list("xy0123456789+-*/^()[]=., ") + [
    "sin", "log", "sqrt", "pi", "oo", "zoo", "nan", "E", "I", "Abs", "**",
]
_TEXT = st.lists(st.sampled_from(_TOKENS), max_size=12).map("".join)


def _contract(call, *texts: str) -> None:
    try:
        call()
    except PycodemathError:
        pass
    except Exception as exc:  # the contract under test
        raise AssertionError(f"{type(exc).__name__} escaped for {texts!r}: {exc}") from exc


@_fuzz(80)
@given(_TEXT, _TEXT)
def test_check_equal_fails_only_with_typed_errors(a, b):
    _contract(lambda: check_equal(a, b, budget=30), a, b)


@_fuzz(40)
@given(st.lists(_TEXT, min_size=2, max_size=3))
def test_check_steps_fails_only_with_typed_errors(steps):
    _contract(lambda: check_steps(steps, budget=30), *steps)


@_fuzz(40)
@given(_TEXT, _TEXT)
def test_certify_fails_only_with_typed_errors(f, F):
    _contract(lambda: certify_integrate(f, "x", F, budget=30), f, F)
    _contract(lambda: certify_diff(f, "x", F, budget=30), f, F)


# --- regressions: what the fuzz (and V2) found ------------------------------

def test_certify_diff_does_not_refute_where_the_claim_is_zero_over_zero():
    # V5 fuzz finding, fixed: the engine's derivative of sin(sqrt(x+1))^2 is 0/0 at
    # x = -1 as written; evalf(subs=) returned 3.05e-5 there with full precision
    # claimed, finite differences said 1, and certify_diff REFUTED a correct result.
    f = parse("(sin((x + 1)^(1/2)))^(2)")
    g = diff(f, "x")
    assert g.sy.xreplace({sp.Symbol("x"): -1}) is sp.nan
    verdict = certify_diff(f, "x", g)
    assert verdict.status is VerdictStatus.VERIFIED
    # and a claim really off by 1e-6 is still refuted, at a point that has a value
    wrong = certify_diff(f, "x", Expr(g.sy + sp.Rational(1, 10**6)))
    assert wrong.status is REFUTED and wrong.counterexample == {"x": "0"}


@pytest.mark.parametrize(
    "source, at",
    [("sin(x-2)/(x-2)", 2), ("sin(sqrt(x+1))/sqrt(x+1)", -1), ("(x^2-1)/(x-1)", 1)],
)
def test_a_zero_over_zero_point_has_no_value_at_the_recheck(source, at):
    # The mechanism: at 60 digits evalf(subs=) gives 2.8e-14 and 3.05e-5 for the
    # first two with 203 bits claimed (accepted before the fix), and 32768 with 1
    # bit for the third (refused already); exact substitution gives nan for all.
    sy = parse(source).sy
    point = {sp.Symbol("x"): sp.Integer(at)}
    assert _undefined_as_written(sy, point)
    assert _recheck(sy, point) is None


# V3 finding 2, fixed 29.09 in parser._refuse_stray_close: a raw IndexError on
# 3.12, a TokenError-backed ParseError on 3.11 — the strict xfail went XPASS on
# 3.11 in CI. Both versions now give the same ParseError.
def test_unbalanced_close_paren_is_a_parse_error():
    with pytest.raises(PycodemathError):
        check_equal(")", "x")


# V5 finding, fixed in V9 (parser._refuse_bare_function_names): a bare function
# name reached parser._check_cost as a function CLASS and iterating its `args`
# property was a raw TypeError. Was a strict xfail; the fuzz no longer excuses it.
@pytest.mark.parametrize("text", ["sin", "log", "Abs", "(sqrt)"])
def test_a_bare_function_name_is_a_parse_error(text):
    with pytest.raises(ParseError, match="function name without arguments"):
        check_equal(text, "x")


# V5 finding (verifier, not engine), fixed in V8: a real cubic root written with
# complex radicals (casus irreducibilis) evaluates to re + 0.e-37*I; the 1-bit
# imaginary part made equal._evaluate refuse the value, and completeness stayed
# unconfirmed although all 3 roots were proved. Now a part with no bits of its own
# is 0 when it is below 10^-dps of the other, full-precision part.
def test_certify_solve_proves_a_casus_irreducibilis_cubic():
    p = "x^3 - 8*x^2 - 3*x + 92"
    roots = solve(parse(p), "x")
    assert len(roots) == 3 and all(r.sy.has(sp.I) for r in roots)
    assert certify_solve(p, "x", roots).status is VerdictStatus.VERIFIED


# V5 finding (verifier), fixed in V8: a Ferrari root of an irreducible quartic,
# substituted, cancels to a 0 with no reliable digits and simplify does not reach
# 0 either, so certify_solve spent its whole budget (all 120 s of the default on
# 4 of 4 probed quartics) and ended UNDECIDED 'time-budget'. Now the roots are
# checked by value and against nroots first; the proofs (minimal polynomial, ~5 s
# per root) that do not fit in 1 s leave the numeric verdict, never VERIFIED.
def test_certify_solve_decides_an_irreducible_quartic_within_its_budget():
    p = "-2*x^4 - 10*x^3 - 2*x^2 + 42*x + 37"
    roots = solve(parse(p), "x")
    assert len(roots) == 4
    # 1 s is 64 Windows clock ticks — no millisecond-scale assertion here
    verdict = certify_solve(p, "x", roots, budget=1)
    assert verdict.method != "time-budget"
    assert verdict.status is not VerdictStatus.VERIFIED
