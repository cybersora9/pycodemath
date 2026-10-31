"""C1 — code <-> formula (``pycodemath.code.frontend``)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

OFF_BY_ONE = textwrap.dedent(
    """
    def sum_to(n: int) -> int:
        total = 0
        for i in range(1, n):
            total += i
        return total
    """
)


def test_off_by_one_loop_is_refuted_with_call():
    from pycodemath.code.frontend import check_code

    res = check_code(OFF_BY_ONE, "n*(n+1)/2", domain={"n": "integer, n>=0"})
    assert res.verdict == "refuted"
    ce = res.counterexample
    assert ce is not None
    assert ce.args == {"n": 1}
    assert ce.function_value == 0  # really executed: range(1, 1) is empty
    assert ce.formula_value == "1"
    assert ce.confirmed_by_execution


def test_emitted_test_fails_on_buggy_function(tmp_path):
    from pycodemath.code.frontend import check_code, emit_test

    (tmp_path / "buggy.py").write_text(OFF_BY_ONE)
    res = check_code(OFF_BY_ONE, "n*(n+1)/2", domain={"n": "integer, n>=0"})
    text = emit_test(res, module="buggy")
    (tmp_path / "test_pinned.py").write_text(text)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path / "test_pinned.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    assert "1 failed" in run.stdout


# --------------------------------------------------------------------------- #
# everything below was written after the guards above were seen red           #
# --------------------------------------------------------------------------- #

import json
import math
import pathlib
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from pycodemath.code import _ast_rules as rules
from pycodemath.code.frontend import (
    CodeCheck,
    CodeReadError,
    _definitions_only,
    _load,
    check_code,
    emit_test,
    to_symbolic,
)
from pycodemath.core.errors import DomainError, ParseError, UnsupportedFormError

DATA = pathlib.Path(__file__).resolve().parents[1] / "pycodemath" / "code" / "data"
NN = {"n": "integer, n>=0"}


# ---- the one mapping table: every entry against the real function ------------

_POINTS = {  # an argument tuple inside each function's domain
    1: (0.37,), 2: (0.37, 1.9),
}
_SPECIAL = {
    "acosh": (1.7,), "factorial": (5,), "comb": (7, 3), "perm": (7, 3), "isqrt": (17,),
    "arccosh": (1.7,), "log": (2.5,), "pow": (1.3, 2.2), "power": (1.3, 2.2),
    "round": (2.5,), "around": (2.5,), "rint": (3.5,), "atan2": (0.4, -1.2), "arctan2": (0.4, -1.2),
    "copysign": (2.0, -0.5), "hypot": (3.0, 4.0), "floor": (-2.5,), "ceil": (-2.5,), "trunc": (-2.7,),
    "maximum": (1.0, 2.0), "minimum": (1.0, 2.0), "cbrt": (-8.0,),
}


@pytest.mark.parametrize("table,module", [(rules.MATH_FUNCS, math), (rules.NUMPY_FUNCS, np)])
def test_every_mapped_function_matches_python(table, module):
    checked = 0
    for name, (build, arities) in table.items():
        if not hasattr(module, name):
            continue  # e.g. math.cbrt / exp2 only exist on newer Pythons
        args = _SPECIAL.get(name, _POINTS[arities[0]])
        want = float(getattr(module, name)(*args))
        got = float(sp.N(build([sp.nsimplify(a, rational=True) for a in args]), 30))
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-14), (name, got, want)
        checked += 1
    assert checked >= len(table) - 2


def test_constants_match():
    for name, value in rules.MATH_CONSTANTS.items():
        assert math.isclose(float(value), getattr(math, name))
    for name, value in rules.NUMPY_CONSTANTS.items():
        assert math.isclose(float(value), getattr(np, name))


def test_py_round_is_bankers_like_python():
    for k in range(-40, 41):
        x = Fraction(k, 4)
        assert int(rules.py_round(sp.Rational(x.numerator, x.denominator))) == round(x), x
    assert int(rules.py_trunc(sp.Rational(-7, 2))) == int(-3.5)


# ---- the reader -------------------------------------------------------------


def test_reader_semantics_floordiv_mod_int_round():
    sf = to_symbolic("def f(a: int, b: int):\n    return a // b + a % b\n")
    a, b = sf.symbols["a"], sf.symbols["b"]
    for x, y in [(7, 2), (-7, 2), (7, -2), (-7, -2)]:
        assert sf.expr.xreplace({a: x, b: y}) == x // y + x % y
    sf = to_symbolic("def f(x: float):\n    return int(x) + round(x)\n")
    x = sf.symbols["x"]
    for v in ("-2.5", "-0.5", "0.5", "1.5", "2.5", "2.7"):
        assert sf.expr.xreplace({x: sp.Rational(v)}) == int(float(v)) + round(float(v))


def test_decimal_literals_mean_the_decimal_written():
    assert to_symbolic("def f(x: float):\n    return 0.1 * x\n").expr == sp.Symbol("x", real=True) / 10


def test_xor_is_refused_and_flagged():
    with pytest.raises(CodeReadError) as e:
        to_symbolic("def f(n: int):\n    return n ^ 2\n")
    assert e.value.construct == "bitwise ^ (XOR)" and e.value.line == 2
    assert isinstance(e.value, UnsupportedFormError)
    r = check_code("def f(n: int):\n    return n ^ 2\n", "n^2", domain={"n": "integer"})
    assert r.refuted and r.has_note("suspicious_xor")
    assert r.counterexample.args == {"n": 0} and r.counterexample.function_value == 2


@pytest.mark.parametrize("src,construct,line", [
    ("def f(n: int):\n    while n:\n        n -= 1\n    return n\n", "while loop", 2),
    ("def f(n: int):\n    return f(n - 1)\n", "recursion", 1),
    ("def f(n: int):\n    xs = [1, 2]\n    return n\n", "list", 2),
    ("def f(x):\n    return x[0]\n", "subscript / indexing", 2),
    ("def f(x):\n    print(x)\n    return x\n", "expression statement (side effect: print(x))", 2),
    ("def f(n: int):\n    for i in range(n):\n        pass\n    return i\n", "loop variable read after its loop", 4),
    ("def f(x):\n    return x, x\n", "tuple", 2),
])
def test_unsupported_constructs_are_named_with_line(src, construct, line):
    with pytest.raises(CodeReadError) as e:
        to_symbolic(src)
    assert e.value.construct == construct
    assert e.value.line == line


def test_loops_close_to_sums_products_and_recurrences():
    n = sp.Symbol("n", integer=True, nonnegative=True)
    src = "def f(n: int):\n    s = 0\n    for i in range(1, n + 1):\n        s += i * i\n    return s\n"
    assert sp.simplify(to_symbolic(src, domain=NN).expr - n * (n + 1) * (2 * n + 1) / 6) == 0
    src = "def f(n: int):\n    p = 1\n    for i in range(1, n + 1):\n        p *= i\n    return p\n"
    e = to_symbolic(src, domain=NN).expr
    assert all(e.xreplace({n: k}) == math.factorial(k) for k in range(8))
    src = "def f(n: int):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
    e = to_symbolic(src, domain=NN).expr
    assert all(sp.nsimplify(sp.N(e.xreplace({n: k}), 50)) == sp.fibonacci(k) for k in range(12))


def test_small_constant_loop_is_unrolled_when_not_closable():
    src = "def f(x: float):\n    for _ in range(3):\n        x = x * x + 1 if x > 0 else -x\n    return x\n"
    e = to_symbolic(src).expr
    x = sp.Symbol("x", real=True)
    def py(v):
        for _ in range(3):
            v = v * v + 1 if v > 0 else -v
        return v
    for v in (-2, -1, 0, 1, 2):
        assert e.xreplace({x: v}) == py(v)


def test_generator_sum_inline_helper_and_module_constant():
    src = (
        "import math as m\nK = 3\n\ndef sq(t):\n    return t * t\n\n"
        "def f(n: int):\n    return sum(sq(i) for i in range(n) if i % 2 == 0) + K * m.pi\n"
    )
    e = to_symbolic(src).expr
    n = sp.Symbol("n", integer=True)
    for k in range(9):
        assert sp.simplify(e.xreplace({n: k}).doit() - (sum(i * i for i in range(k) if i % 2 == 0) + 3 * sp.pi)) == 0


def test_to_symbolic_never_executes(tmp_path):
    marker = tmp_path / "ran"
    src = f"open({str(marker)!r}, 'w').write('x')\n\ndef f(x: float):\n    return x + 1\n"
    assert to_symbolic(src).expr == sp.Symbol("x", real=True) + 1
    assert not marker.exists()
    # and check_code refuses to RUN a module that opens files
    r = check_code(src, "x + 2")
    assert r.undecided and r.has_note("not_executed")
    assert not marker.exists()


def test_definitions_only_drops_top_level_statements():
    src = "import math\nX = 2\nY = foo()\nprint('hi')\n\ndef f(x):\n    return x\n"
    out = _definitions_only(_load(src, None))
    assert "print" not in out and "foo" not in out and "X = 2" in out and "def f" in out


# ---- check_code ---------------------------------------------------------------


def test_counterexample_is_the_most_readable_call():
    r = check_code(OFF_BY_ONE, "n*(n+1)/2", domain=NN)
    assert r.counterexample.call == "sum_to(n=1)"
    assert "returns 0, the formula gives 1" in r.reason


def test_float_refutation_carries_tolerance_and_reading():
    r = check_code("def area(r: float):\n    return 3.14 * r * r\n", "pi*r^2")
    assert r.refuted and not r.counterexample.exact
    c = r.counterexample
    assert c.args == {"r": 1.0} and c.function_value == 3.14
    assert 0 < c.tolerance <= 1e-9


def test_rounding_alone_never_refutes():
    # mathematically equal: the float gap is rounding (C2's domain)
    r = check_code("def f(r: float):\n    return 3.141592653589793 * r * r\n", "pi*r^2")
    assert r.undecided
    src = "def f(n: int):\n    s = 0.0\n    for _ in range(n):\n        s += 0.1\n    return s\n"
    assert check_code(src, "n/10", domain=NN).verified


def test_domain_changes_the_verdict():
    src = "import math\ndef f(x: float):\n    return math.sqrt(x * x)\n"
    assert check_code(src, "x", domain={"x": "real, x>=0"}).verified
    r = check_code(src, "x")
    assert r.refuted and r.counterexample.args == {"x": -1.0}


def test_quasi_polynomial_certificate_for_integer_division():
    r = check_code("def t(n: int):\n    return n * (n + 1) // 2\n", "n*(n+1)/2", domain={"n": "integer"})
    assert r.verified and r.method == "quasi-polynomial"
    r = check_code("def t(n: int):\n    return n * (n - 1) // 3\n", "n*(n-1)/3", domain={"n": "integer"})
    assert r.refuted and r.counterexample.args == {"n": 2}


def test_fibonacci_loop_is_proved_by_exponential_polynomial_certificate():
    src = "def fib(n: int):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
    r = check_code(src, "fibonacci(n)", domain=NN)
    assert r.verified


def test_function_object_input():
    r = check_code(_obj_sum_to, "n*(n+1)/2", domain=NN)
    assert r.refuted and r.counterexample.call == "_obj_sum_to(n=1)"
    assert r.module == __name__


def _obj_sum_to(n: int) -> int:
    total = 0
    for i in range(1, n):
        total += i
    return total


def test_bad_inputs_are_typed_errors():
    with pytest.raises(ParseError):
        check_code("def f(:\n", "x")
    with pytest.raises(DomainError):
        check_code("x = 1\n", "x")
    with pytest.raises(DomainError):
        check_code("def f(x):\n    return x\n", "x + y")
    with pytest.raises(DomainError):
        check_code("def f(x):\n    return x\n", "x", domain={"z": "real"})
    with pytest.raises(DomainError):
        check_code("def f(x):\n    return x\n", "x", budget=0)


def test_result_is_json_and_kind_is_code_equiv():
    r = check_code(OFF_BY_ONE, "n*(n+1)/2", domain=NN)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["kind"] == "code_equiv" and d["verdict"] == "refuted"
    assert d["counterexample"]["call"] == "sum_to(n=1)"
    with pytest.raises(DomainError):
        CodeCheck("refuted", "x", "x", "f", (), "", None, None, (), {}, "")


def test_tiny_budget_is_undecided_not_an_exception():
    src = "def f(x: float):\n    return x\n"
    r = check_code(src, "x", budget=1e-6)
    assert r.verdict in ("undecided", "verified")


# ---- emit_test ----------------------------------------------------------------


def _run_pytest(tmp_path, name, text):
    (tmp_path / name).write_text(text)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path / name)],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("src,formula,domain", [
    ("def area(r: float):\n    return 3.14 * r * r\n", "pi*r^2", None),
    ("def half(n: int):\n    return n // 2\n", "n/2", {"n": "integer"}),
    ("def m(a: float, b: float):\n    return a + b / 2\n", "(a+b)/2", None),
])
def test_emitted_tests_are_red_on_buggy_and_green_on_fixed(tmp_path, src, formula, domain):
    r = check_code(src, formula, domain=domain)
    assert r.refuted
    text = emit_test(r)
    assert "pycodemath" in text and formula in text
    run = _run_pytest(tmp_path, "test_red.py", text)
    assert run.returncode == 1 and "1 failed" in run.stdout, run.stdout
    fixed = {"pi*r^2": "def area(r):\n    import math\n    return math.pi * r * r\n",
             "n/2": "def half(n):\n    return n / 2\n",
             "(a+b)/2": "def m(a, b):\n    return (a + b) / 2\n"}[formula]
    body = text.split("import pytest\n", 1)[1]
    head, test = body.split("\n\ndef test_", 1)
    run = _run_pytest(tmp_path, "test_green.py", "import pytest\n" + fixed + "\n\ndef test_" + test)
    assert run.returncode == 0, run.stdout


def test_emit_test_needs_a_counterexample():
    r = check_code("def f(x: float):\n    return 2 * x\n", "x + x")
    assert r.verified
    with pytest.raises(DomainError):
        emit_test(r)


# ---- the evaluation set and the adversarial round -----------------------------


def _pairs():
    return [json.loads(line) for line in (DATA / "C1_pairs.jsonl").read_text().splitlines()]


def test_evaluation_set_shape():
    pairs = _pairs()
    assert len(pairs) >= 60
    assert sum(p["expected"] == "verified" for p in pairs) >= 30
    assert sum(p["expected"] == "refuted" for p in pairs) >= 30
    held = sum(p["split"] == "held" for p in pairs)
    assert abs(held - len(pairs) / 3) <= 3


@pytest.mark.parametrize("pair", _pairs(), ids=lambda p: p["id"])
def test_evaluation_pair(pair):
    r = check_code(pair["source"], pair["formula"], name=pair["name"], domain=pair["domain"], budget=20)
    if pair["expected"] == "verified":
        assert not r.refuted, r.reason  # zero false alarms, the hard contract
    else:
        assert r.refuted, r.reason      # every bug in the set is caught with a call
        assert r.counterexample.confirmed_by_execution


def _adversarial():
    return json.loads((DATA / "C1_adversarial.json").read_text())


@pytest.mark.parametrize("case", _adversarial(), ids=lambda c: c["id"])
def test_adversarial_case(case):
    r = check_code(case["source"], case["formula"], domain=case["domain"], budget=10)
    assert r.verdict in case["allowed"], (r.verdict, r.reason)
