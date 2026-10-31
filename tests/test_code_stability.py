"""C2 — precision doctor: measurement, localisation, proven rewrites."""

from __future__ import annotations

import pytest

from pycodemath.code.stability import diagnose, improve


# --- guards (seen RED before the engine existed, see dev-journal/full/C2.md) ---


def test_rewrite_is_equivalent_and_more_accurate():
    imp = improve("sqrt(x + 1) - sqrt(x)", {"x": (1e10, 1e15)}, samples=128, seed=1)
    assert imp.improved
    best = imp.best
    assert best is not None
    assert best.proof.status == "verified"
    assert best.proof.method in ("check_equal", "symbolic")
    assert imp.before_mean_bits - best.mean_bits >= 20
    # the classic conjugate form, up to SymPy's choice of order
    assert "sqrt" in best.expression and "/" in best.expression and "-" not in best.expression


def test_stable_expression_gets_no_fake_fix():
    imp = improve("x*x + y*y", {"x": (1, 10), "y": (1, 10)}, samples=128, seed=1)
    assert not imp.improved
    assert imp.best is None
    assert "already accurate" in imp.message or "no rewrite" in imp.message
    rep = diagnose("x*x + y*y", {"x": (1, 10), "y": (1, 10)}, samples=128)
    assert rep.accurate


# --- the error measure ----------------------------------------------------------

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import mpmath
import numpy as np

from pycodemath.code import stability as st
from pycodemath.code.stability import (
    Node,
    StabilityDomainError,
    from_fpcore,
    parse_formula,
)
from pycodemath.core.errors import DomainError, ParseError, PycodemathError, TimeBudgetError

DATA = Path(st.__file__).parent / "data" / "C2_cases.json"


def test_rounding_is_correct_including_ties_subnormals_and_overflow():
    f64 = st._F64
    rng = np.random.default_rng(0)
    for _ in range(300):
        num = int(rng.integers(-(10**18), 10**18))
        den = int(rng.integers(1, 10**18))
        assert st._round_rational(num, den, f64) == float(Fraction(num, den))
    assert st._round_rational(2**53 + 1, 1, f64) == 2.0**53  # tie -> even
    assert st._round_rational(2**53 + 3, 1, f64) == 2.0**53 + 4
    assert st._round_rational(1, 2**1074, f64) == 5e-324
    assert st._round_rational(1, 2**1075, f64) == 0.0  # tie with 0 -> even (0)
    assert st._round_rational(3, 2**1076, f64) == 5e-324
    assert st._round_rational(2**1024, 1, f64) == math.inf
    assert st._round_rational(2**1024 - 2**970, 1, f64) == math.inf  # rounds up to 2**1024
    assert st._round_rational(2**1024 - 2**970 - 1, 1, f64) == 1.7976931348623157e308
    f32 = st._F32
    assert st._round_rational(1, 3, f32) == float(np.float32(1 / 3))
    assert st._round_rational(1, 2**149, f32) == float(np.float32(1.4e-45))
    # no double rounding through float64 on the way to float32
    x = Fraction(1) + Fraction(1, 2**24) + Fraction(1, 2**60)
    assert st._round_rational(x.numerator, x.denominator, f32) == 1.0 + 2.0**-23


def test_bits_of_error_counts_floats_between():
    f = st._F64
    one = np.array([1.0])
    assert st._bits_of_error(one, one, f)[0] == 0
    assert st._bits_of_error(np.nextafter(one, 2), one, f)[0] == 1.0
    assert st._bits_of_error(np.array([np.nan]), one, f)[0] == 64
    assert st._bits_of_error(np.array([-0.0]), np.array([0.0]), f)[0] == 0
    assert st._bits_of_error(np.array([np.inf]), np.array([1.7976931348623157e308]), f)[0] == 1.0
    big = st._bits_of_error(np.array([-1.0]), one, f)[0]
    assert 62 < big <= 64


def test_sampling_is_uniform_over_bit_patterns():
    xs = st._uniform_bits(np.random.default_rng(1), 1e-10, 1e15, 4000, st._F64)
    below_one = float(np.mean(xs < 1.0))
    assert 0.3 < below_one < 0.5  # 10 of 25 decades; uniform-by-value would give ~1e-15
    assert xs.min() >= 1e-10 and xs.max() <= 1e15


# --- reading formulas -------------------------------------------------------------


def test_written_order_is_kept_and_text_round_trips():
    assert parse_formula("(x + 1) - x") != parse_formula("x + (1 - x)")
    assert str(parse_formula("x - (y - z)")) == "x - (y - z)"
    assert str(parse_formula("2^x^2")) == "2**x**2"
    assert parse_formula("0.1").value == st.sp.Rational(1, 10)
    for case in json.loads(DATA.read_text())["cases"]:
        n = parse_formula(case["expr"])
        assert parse_formula(str(n)) == n, case["id"]


@pytest.mark.parametrize("bad", [
    "__import__('os')", "x.y", "foo(x)", "'a'", "lambda: 1", "sqrt", "x +", "[1, 2]",
    "_secret + 1", "sqrt(1, 2)", "x == 1", "True",
])
def test_parser_refuses_instead_of_guessing(bad):
    with pytest.raises(ParseError):
        parse_formula(bad)


def test_fpcore_reader():
    f, props = from_fpcore('(FPCore (x) :name "t" :pre (>= x 0) (let ([t (+ x 1)]) (- (sqrt t) (sqrt x))))')
    assert str(f) == "sqrt(x + 1) - sqrt(x)" and props["args"] == ["x"]
    g, _ = from_fpcore("(if (<= x 1/2) (sqr x) (cube (- x)))")
    assert str(g) == "x * x if x <= 0.5 else -x * -x * -x"
    h, _ = from_fpcore("(/ (+ (- b/2) (sqrt (- (sqr b/2) (* a c)))) a)")
    assert h.variables == ("a", "b_2", "c")
    with pytest.raises(ParseError):
        from_fpcore("(while (< i 3) x)")


# --- ground truth -------------------------------------------------------------------


def test_truth_is_the_correctly_rounded_exact_value():
    cases = [
        ("exp(x) - 1", {"x": 1e-300}, mpmath.mpf(1e-300)),
        ("x - sin(x)", {"x": 1e-5}, None),
        ("log(1 + x)", {"x": 3e-20}, None),
        ("atan(x) + cbrt(x)", {"x": 7.25}, None),
        ("cosh(x) - 1", {"x": 1e-8}, None),
    ]
    for text, point, _ in cases:
        n = parse_formula(text)
        t = st._truth(n, point, st._F64)
        assert not isinstance(t, str), text
        with mpmath.workprec(4000):
            sy = st.to_sympy(n)
            ref = st.sp.N(sy.subs({st.sp.Symbol(k, real=True): st.sp.Float(v, 400) for k, v in point.items()}), 600)
            assert t.value == float(mpmath.mpf(str(ref))), text


def test_points_without_a_real_value_are_skipped_not_guessed():
    with pytest.raises(StabilityDomainError):
        st.diagnose("sqrt(x)", {"x": (-10, -1)}, samples=16)
    r = st.diagnose("sqrt(x)", {"x": (-1, 1)}, samples=64)
    assert r.skipped["invalid"] > 0 and r.samples == 64
    assert r.mean_bits <= 0.5


# --- diagnose -------------------------------------------------------------------------


def test_diagnosis_names_the_culprit_and_shows_a_witness():
    r = st.diagnose("x * y + (1 - cos(z))", {"x": (1, 2), "y": (1, 2), "z": (1e-9, 1e-3)}, samples=128)
    assert r.culprits[0].expression == "1 - cos(z)"
    assert "cancellation" in r.culprits[0].cause
    assert "2*sin" in r.culprits[0].hint
    w = r.worst
    assert w is not None and float(w.computed) != float(mpmath.mpf(w.exact))
    assert len(r.regions) == 3 * 8
    json.dumps(r.to_dict())


def test_float32_measures_float32():
    r32 = st.diagnose("x * x + y * y", {"x": (1, 10), "y": (1, 10)}, dtype="float32", samples=128)
    r64 = st.diagnose("x * x + y * y", {"x": (1, 10), "y": (1, 10)}, samples=128)
    assert r32.dtype == "float32" and r32.mean_bits <= 1.0 and r64.mean_bits <= 1.0
    loss = st.diagnose("(x + 1) - x", {"x": (1e8, 1e9)}, dtype="float32", samples=64)
    assert loss.mean_bits > 20  # 1 is absorbed by x in float32 long before float64


@pytest.mark.parametrize("kwargs, exc", [
    ({"dtype": "float16"}, DomainError),
    ({"samples": 0}, DomainError),
    ({"samples": True}, DomainError),
    ({"budget": -1}, DomainError),
])
def test_bad_arguments_are_typed_errors(kwargs, exc):
    with pytest.raises(exc):
        st.diagnose("x", {"x": (0, 1)}, **kwargs)


def test_bad_ranges_are_domain_errors():
    for ranges in ({}, {"x": (2, 1)}, {"x": (0, 1), "y": (0, 1)}, {"x": (math.nan, 1)}, {"x": "ab"}):
        with pytest.raises(DomainError):
            st.diagnose("x + 1", ranges, samples=4)


def test_budget_is_respected():
    with pytest.raises(TimeBudgetError):
        st.improve("log(1 + x)", {"x": (-1e-3, 1e-3)}, samples=2000, budget=0.05)


# --- improve: the two proofs -----------------------------------------------------------


def test_a_wrong_but_accurate_looking_rewrite_is_never_returned(monkeypatch):
    from pycodemath.code import _rewrites

    wrong = parse_formula("1 / (2 * sqrt(x))")  # the leading term only: close, NOT equal
    real = _rewrites.local_rewrites

    def inject(n):
        out = real(n)
        if n == parse_formula("sqrt(x + 1) - sqrt(x)"):
            out = [(wrong, "injected")] + out
        return out

    monkeypatch.setattr(_rewrites, "local_rewrites", inject)
    imp = st.improve("sqrt(x + 1) - sqrt(x)", {"x": (1e8, 1e15)}, samples=96)
    for c in imp.candidates:
        assert "injected" not in c.rules
    assert imp.stats["refuted"] >= 1


def test_conditional_rewrite_states_its_domain():
    imp = st.improve("(1 + r)**n", {"r": (1e-10, 1e-6), "n": (1, 1000)}, samples=128)
    assert imp.improved
    assert "log1p" in imp.best.expression
    assert imp.best.proof.method != "check_equal"
    assert imp.best.conditions  # (1+r)^n = exp(n log1p r) needs 1 + r > 0


def test_taylor_branch_is_certified_and_flagged_as_approximation():
    imp = st.improve("x - sin(x)", {"x": (-0.01, 0.01)}, samples=128)
    assert imp.improved and imp.gain_bits > 15
    proofs = [imp.best.proof, *imp.best.proof.branches]
    assert any(p.method == "taylor-remainder" and not p.exact for p in proofs)
    assert imp.best.proof.exact is False


def test_certify_taylor_is_a_real_bound():
    from pycodemath.code import _rewrites as rw

    f = parse_formula("x - sin(x)")
    x = st.sp.Symbol("x", real=True)
    poly = x**3 / 6
    assert rw.certify_taylor(f, "x", poly, 5, 3, 1e-9, 2.0**-53)
    assert not rw.certify_taylor(f, "x", poly, 5, 3, 1e-2, 2.0**-53)  # x^5/120 term matters
    assert rw.certify_taylor(f, "x", poly - x**5 / 120, 7, 3, -1e-4, 2.0**-53)


def test_logsumexp_and_softmax_in_n_variables():
    lse = st.improve("log(exp(x1) + exp(x2) + exp(x3))", {f"x{i}": (-1000, 1000) for i in (1, 2, 3)}, samples=96)
    assert lse.improved and "max" in lse.best.expression
    assert lse.best.mean_bits < 1.0
    sm = st.improve("exp(x1) / (exp(x1) + exp(x2) + exp(x3))", {f"x{i}": (-1000, 1000) for i in (1, 2, 3)},
                    samples=96)
    assert sm.improved and "max" in sm.best.expression


def test_numpy_code_evaluates_exactly_what_was_measured():
    for text, ranges in [
        ("sqrt(x + 1) - sqrt(x)", {"x": (1e10, 1e15)}),
        ("x - sin(x)", {"x": (-0.01, 0.01)}),
        ("log(exp(a) + exp(b))", {"a": (-1000, 1000), "b": (-1000, 1000)}),
    ]:
        imp = st.improve(text, ranges, samples=64)
        best = imp.best
        assert best is not None
        scope: dict = {}
        exec(best.to_numpy("g"), scope)  # noqa: S102 - code we generated
        env = {v: st._uniform_bits(np.random.default_rng(5), *ranges[v], 500, st._F64) for v in best.variables}
        ours = st._eval_float(best.node, env, st._F64)
        theirs = scope["g"](*(env[v] for v in best.variables))
        assert np.array_equal(ours.view(np.int64), np.asarray(theirs, dtype=np.float64).view(np.int64)), text
        py: dict = {}
        exec(best.to_python("h"), py)  # noqa: S102
        i = 7
        val = py["h"](*(float(env[v][i]) for v in best.variables))
        assert st._bits_of_error(np.array([val]), ours[i:i + 1], st._F64)[0] <= 2


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_code_compiles_and_agrees(tmp_path):
    imp = st.improve("log(1 + x)", {"x": (-1e-3, 1e-3)}, samples=64)
    src = imp.best.to_c("f") + '\n#include <stdio.h>\nint main(void){printf("%.17g\\n", f(1e-10));return 0;}\n'
    (tmp_path / "f.c").write_text(src)
    subprocess.run(["cc", "-O2", "-o", str(tmp_path / "f"), str(tmp_path / "f.c"), "-lm"], check=True)
    out = subprocess.run([str(tmp_path / "f")], capture_output=True, text=True, check=True).stdout
    assert float(out) == math.log1p(1e-10)


def test_results_serialise_to_json():
    imp = st.improve("exp(x) - 1", {"x": (-1e-3, 1e-3)}, samples=64)
    d = imp.to_dict()
    assert d["kind"] == "stability_improvement" and d["improved"]
    assert d["best"]["proof"]["status"] == "verified"
    json.dumps(d)


def test_every_failure_is_a_pycodemath_error():
    for args in [("x +", {"x": (0, 1)}), ("x", {"y": (0, 1)}), (3, {"x": (0, 1)})]:
        with pytest.raises(PycodemathError):
            st.improve(*args, samples=8)


# --- regressions from the adversarial round (dev-journal/full/C2.md) ---------------------


def test_e_is_a_variable_not_eulers_number():
    assert parse_formula("a*b + c*d - e*f").variables == ("a", "b", "c", "d", "e", "f")
    assert parse_formula("E").value == st.sp.E


def test_real_cube_root_of_negative_numbers():
    r = st.diagnose("cbrt(x)", {"x": (-8, 8)}, samples=64)
    assert r.skipped["invalid"] == 0 and r.mean_bits < 1


def test_formula_without_variables():
    r = st.diagnose("0.1 + 0.2 - 0.3", {}, samples=8)
    assert r.mean_bits > 50  # the doubles 0.1 + 0.2 - 0.3 = 5.55e-17, the decimals give 0
    imp = st.improve("0.1 + 0.2 - 0.3", {}, samples=8)
    assert imp.improved and imp.best.expression == "0"


def test_sympy_evaluation_is_a_candidate():
    imp = st.improve("(x + 1) - x", {"x": (1e16, 1e17)}, samples=64)
    assert imp.best.expression == "1" and imp.best.proof.exact


def test_exact_identity_preferred_over_equal_taylor_branch():
    imp = st.improve("sqrt(x*x + 1) - 1", {"x": (1e-9, 1e-2)}, samples=128)
    assert imp.improved and imp.best.proof.exact


def test_divide_through_avoids_overflow_of_the_square():
    big = 1.7976931348623157e308
    imp = st.improve("x / (1 + x*x)", {"x": (-big, big)}, samples=128)
    assert imp.improved and imp.best.max_bits < 8


def test_domain_of_the_original_is_used_by_the_proof():
    big = 1.7976931348623157e308
    imp = st.improve("log(N + 1) - log(N)", {"N": (-big, big)}, samples=128)
    assert imp.improved and "log1p" in imp.best.expression
    assert any("where the original is real" in c for c in imp.best.conditions)


def test_atan_difference_needs_and_gets_its_side_condition():
    big = 1.7976931348623157e308
    imp = st.improve("atan(N + 1) - atan(N)", {"N": (-big, big)}, samples=128)
    assert imp.improved and imp.best.proof.method in ("tan-injective", "piecewise")
    o = parse_formula("atan(x) - atan(y)")
    c = parse_formula("atan((x - y) / (1 + x * y))")
    assert st._prove(o, c, {"x": (-10, 10), "y": (-10, 10)}) is None  # 1 + xy can be <= 0


def test_two_taylor_replacements_on_a_common_radius():
    imp = st.improve("(x - sin(x)) / (x - tan(x))", {"x": (-1, 1)}, samples=128)
    assert imp.improved and imp.best.mean_bits < 2
    assert not imp.best.proof.exact


def test_congruence_proof_for_a_scaled_root():
    o = parse_formula("sqrt(b * b - a * c)")
    c = parse_formula("fabs(b) * sqrt(1 - a / b * (c / b))")
    p = st._prove(o, c, {"a": (-1, 1), "b": (-1, 1), "c": (-1, 1)})
    assert p is not None and p.method == "square-injective"
    wrong = parse_formula("b * sqrt(1 - a / b * (c / b))")  # sign lost for b < 0
    assert st._prove(o, wrong, {"a": (-1, 1), "b": (-1, 1), "c": (-1, 1)}) is None


def test_case_split_refuses_a_wrong_log_sum_exp():
    o = parse_formula("log(exp(a) + exp(b))")
    c = parse_formula("max(a, b) + log(exp(a - max(a, b)) - exp(b - max(a, b)))")
    assert st._prove(o, c, {"a": (-9, 9), "b": (-9, 9)}) is None


def test_taylor_branch_never_leaves_its_proved_interval():
    """Regression: bit-uniform sampling of [-1, 1] puts almost no point above 1/16, so a
    branch certified on |x| <= 1/16 was once placed on all of [-1, -0.002]."""
    imp = st.improve("(x - sin(x)) / (x - tan(x))", {"x": (-1, 1)}, samples=128)
    best = imp.best
    assert best.proof.method == "piecewise"
    pieces, node = [], best.node
    while node.op == "if":
        pieces.append((float(node.args[0].args[1].value), node.args[1]))
        node = node.args[2]
    pieces.append((1.0, node))
    lower = -1.0
    for (upper, _), proof in zip(pieces, best.proof.branches):
        if proof.method == "taylor-remainder":
            a, _, _, _, b = proof.conditions[-1].split()
            assert float(a) <= np.nextafter(lower, np.inf) and upper <= float(b), (lower, upper, proof.conditions)
        lower = upper
    # and nothing catastrophic anywhere on a value grid (a series used far outside its
    # interval would be off by tens of bits)
    n = parse_formula("(x - sin(x)) / (x - tan(x))")
    xs = np.linspace(-1, 1, 400)
    exact = np.array([st._truth(n, {"x": float(x)}, st._F64).value for x in xs])
    bits = st._bits_of_error(st._eval_float(best.node, {"x": xs}, st._F64), exact, st._F64)
    base = st._bits_of_error(st._eval_float(n, {"x": xs}, st._F64), exact, st._F64)
    assert np.all(bits <= np.maximum(base, 3.0) + 8.0)


def test_docstring_examples_run():
    import doctest

    from pycodemath.code import _rewrites

    for mod in (st, _rewrites):
        result = doctest.testmod(mod, optionflags=doctest.ELLIPSIS)
        assert result.failed == 0, mod.__name__
