"""C5 — loops, recurrences, complexity (``pycodemath.code.loops`` / ``recurrence``).

Guards (seen RED before the modules existed, see dev-journal/full/C5.md):
``test_iteration_count_formula_matches_brute_count`` and
``test_master_theorem_refuses_outside_its_conditions``.

The evaluation set lives in ``pycodemath/code/data/C5_cases.jsonl``; only its
``dev`` split is used here. The ``held`` split is measured once, at the end, by
``dev-journal/full/C5/measure.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sympy as sp

from pycodemath.code.loops import (
    LoopShapeError,
    _canon_logs,
    check_complexity,
    check_iterations,
    count_iterations,
)
from pycodemath.code.recurrence import (
    MasterTheoremInapplicable,
    master_theorem,
    recurrence_from_code,
    solve_recurrence,
)
from pycodemath.verify.verdict import VerdictStatus

V, R, U = VerdictStatus.VERIFIED, VerdictStatus.REFUTED, VerdictStatus.UNDECIDED

CASES = [
    json.loads(line)
    for line in (Path(__file__).resolve().parents[1] / "pycodemath/code/data/C5_cases.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
DEV = [c for c in CASES if c["split"] == "dev"]


def _same(got: sp.Expr, expected: str) -> bool:
    """``got`` equals the expected text, symbols matched BY NAME (the module's
    symbols carry assumptions), logs in the module's exact vocabulary."""
    by_name = {s.name: s for s in got.free_symbols}
    e = sp.sympify(expected, locals={k: v for k, v in by_name.items()})
    e = _canon_logs(e)
    return sp.simplify(got - e) == 0


# --- guards ------------------------------------------------------------------------

TRIANGULAR = '''
def pairs(n: int) -> int:
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            c += 1
    return c
'''

STEPPED = '''
def stepped(n: int) -> int:
    c = 0
    for i in range(1, n, 3):
        for j in range(i, 2 * n, 2):
            c += 1
    return c
'''

DOUBLING = '''
def doubling(n: int) -> int:
    i = 1
    c = 0
    while i < n:
        i *= 2
        c += 1
    return c
'''


@pytest.mark.parametrize("src", [TRIANGULAR, STEPPED, DOUBLING])
def test_iteration_count_formula_matches_brute_count(src: str) -> None:
    """The formula is exact: equal to a real execution counter for every n tried."""
    r = count_iterations(src)
    assert r.certificate is not None and r.certificate.agree, r.to_dict()
    assert r.certificate.points >= 31  # n = 0..30 at least, plus negatives
    n = sp.Symbol("n", integer=True)
    sym = next(iter(r.formula.free_symbols))
    # an independent brute count, here in the test process
    ns: dict = {}
    exec(src.replace("c += 1", "c += 1"), ns)  # noqa: S102
    fn = [v for k, v in ns.items() if callable(v) and not k.startswith("__")][0]
    for k in range(-3, 41):
        assert int(r.formula.subs(sym, k)) == fn(k), (k, r.formula)
    del n


def test_triangular_count_is_the_textbook_formula() -> None:
    r = count_iterations(TRIANGULAR)
    n = next(iter(r.main.free_symbols))
    assert sp.simplify(r.main - n * (n - 1) / 2) == 0
    assert str(r.theta) == "n**2"


def test_master_theorem_refuses_outside_its_conditions() -> None:
    """Never a silent answer outside the theorem: gap cases, failed regularity,
    non-constant a, a < 1 and negative f are refused with the reason."""
    n = sp.Symbol("n", positive=True)
    for a, b, f, why in [
        (2, 2, n / sp.log(n), "gap"),                 # n/log n: between case 1 and 2
        (2, 2, n * sp.log(n) ** -2, "gap"),
        (1, 2, n * (2 - sp.cos(n)), "regularity"),    # case 3 without regularity
        (n, 2, n, "constant"),                        # a depends on n
        (sp.Rational(1, 2), 2, 1 / n, "a >= 1"),
        (64, 8, -n**2 * sp.log(n), "positive"),
        (2, 1, n, "b > 1"),
    ]:
        with pytest.raises(MasterTheoremInapplicable) as info:
            master_theorem(a, b, f)
        assert why in str(info.value), (a, b, f, str(info.value))
    # and inside the conditions it answers, with the case and the checks
    r = master_theorem(2, 2, n)
    assert r.case == 2 and str(r.theta) == "n*log(n)"
    r = master_theorem(3, 2, n)
    assert r.case == 1 and _same(r.theta, "n**(log(3)/log(2))")
    r = master_theorem(2, 2, n**2)
    assert r.case == 3 and "regularity" in " ".join(r.checks)


# --- the dev split of the evaluation set -------------------------------------------


def _run_case(c: dict):  # noqa: ANN202 — returns one of several result types
    op = c["op"]
    if op == "count":
        return count_iterations(c["code"])
    if op == "check_iterations":
        return check_iterations(c["code"], c["claim"])
    if op == "check_complexity":
        return check_complexity(c["code"], c["claim"])
    if op == "solve":
        return solve_recurrence(c["recurrence"])
    if op == "code_recurrence":
        return recurrence_from_code(c["code"])
    raise AssertionError(op)


@pytest.mark.parametrize("case", DEV, ids=[c["id"] for c in DEV])
def test_dev_case(case: dict) -> None:
    exp = case["expected"]
    if "error" in exp:
        with pytest.raises(LoopShapeError):
            _run_case(case)
        return
    r = _run_case(case)
    if "verdict" in exp:
        assert r.verdict.value == exp["verdict"], r.to_dict()
    if "formula" in exp:
        assert r.main is not None and _same(r.main, exp["formula"]), r.to_dict()
    if "theta" in exp:
        assert r.theta is not None and _same(r.theta, exp["theta"]), r.to_dict()
    if "exact" in exp:
        assert r.exact is exp["exact"]
    if "certified" in exp:
        assert (r.certificate is not None and r.certificate.agree) is exp["certified"], r.to_dict()
    assert json.dumps(r.to_dict())  # JSON-able, always


# --- regressions from the adversary round (dev-journal/full/C5/adversary.py) --------


def _f(body: str, sig: str = "n: int") -> str:
    lines = "\n".join("    " + ln for ln in body.strip("\n").splitlines())
    return f"def f({sig}) -> int:\n    c = 0\n{lines}\n    return c\n"


DOUBLING_SRC = _f("i = 1\nwhile i < n:\n    i *= 2\n    c += 1")


def test_claim_undefined_at_zero_is_not_a_witness() -> None:
    # ceil(log2(0)) is undefined: it must not become "the claim gives ceiling(log(0, 2))"
    r = check_iterations(DOUBLING_SRC, "ceiling(log(n, 2))")
    assert r.verdict is V, r.reason
    assert "n >= 1" in r.reason


def test_non_integer_claim_value_is_a_witness() -> None:
    r = check_iterations(TRIANGULAR, "n**2/2")
    assert r.verdict is R and r.witness is not None
    assert r.witness["counted"] != r.witness["claim"]


def test_upper_bound_survives_data_dependence() -> None:
    src = _f("for i in range(len(a)):\n    if a[i] == 1:\n        return i\n    c += 1", "a: list")
    r = count_iterations(src)
    assert r.exact is False and r.certificate is not None and r.certificate.agree
    assert r.certificate.relation == "<="


def test_constant_count_is_still_certified() -> None:
    r = count_iterations(_f("i = n\nwhile i < 4 * n:\n    i *= 2\n    c += 1"))
    assert r.main == 2 and r.certificate is not None and r.certificate.agree and r.certificate.points > 10


def test_theta_is_zero_when_the_loop_stops_running() -> None:
    r = count_iterations(_f("for i in range(0, n, -1):\n    c += 1"))
    assert r.theta == 0 and r.certificate is not None and r.certificate.agree


@pytest.mark.parametrize("src, what", [
    (_f("for x in a:\n    a.append(x)\n    c += 1", "a: list"), "append"),
    ("def range(n):\n    return [0]\n\n" + _f("for i in range(n):\n    c += 1"), "range"),
    (_f("for i in range(0, n, 0):\n    c += 1"), "step 0"),
    (_f("i = 0\nwhile i < n:\n    i *= 2\n    c += 1"), "never grows"),
    (_f("lo, hi = 0, n\nwhile lo <= hi:\n    lo += 1"), "no known value"),
])
def test_unreadable_shapes_are_refused_with_the_construct(src: str, what: str) -> None:
    with pytest.raises(LoopShapeError) as info:
        count_iterations(src, certify=False)
    assert what in str(info.value) and info.value.line is not None


def test_floor_log_is_exact_at_powers() -> None:
    from pycodemath.code.loops import ceil_log, floor_log

    assert floor_log(10**30, 10) == 30 and floor_log(10**30 - 1, 10) == 29
    assert ceil_log(2**60, 2) == 60 and ceil_log(2**60 + 1, 2) == 61
    assert floor_log(sp.Rational(1, 8), 2) == -3


def test_docstring_claim_is_read() -> None:
    src = TRIANGULAR.replace("    c = 0", '    ":iterations: n**2\\n:complexity: O(n^2)"\n    c = 0', 1)
    assert check_iterations(src).verdict is R
    assert check_complexity(src).verdict is V


def test_recurrence_results_are_json_and_pickle() -> None:
    import pickle

    r = solve_recurrence("T(n) = 2*T(n/2) + n, T(1) = 1")
    assert json.dumps(r.to_dict()) and r.exact_on_powers is not None
    n = sp.Symbol("n", integer=True)
    assert all(sp.simplify(r.exact_on_powers.subs(n, 2**k)) == (k + 1) * 2**k for k in range(8))
    try:
        master_theorem(2, 2, "n/log(n)")
    except MasterTheoremInapplicable as exc:
        back = pickle.loads(pickle.dumps(exc))
        assert back.reason == exc.reason and back.alternative == "akra_bazzi"


def test_akra_bazzi_with_irrational_p() -> None:
    r = solve_recurrence("T(n) = T(n/2) + T(n/3) + n")
    assert r.verdict is V and str(r.theta) == "n" and r.method == "Akra–Bazzi"


def test_rsolve_answer_is_checked_not_trusted() -> None:
    r = solve_recurrence("T(n) = T(n-1) + 1/n, T(1) = 1")
    assert r.verdict is V and any("substitution" in c and "residual 0" in c for c in r.checks)
    assert str(r.theta) == "log(n)"
