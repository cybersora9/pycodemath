"""K5 — inequalities, systems and periodic families (``engine.solvesets``)."""

from __future__ import annotations

import sympy as sp

from pycodemath.engine.solvesets import solve_inequality

X = sp.Symbol("x")


def _holds(relation: str, point: sp.Expr) -> bool:
    """The relation at ``point``, decided by the TEST (not the module): a value that
    is undefined or not real is not a solution."""
    from pycodemath.frontend.parser import parse

    for op in ("<=", ">=", "<", ">"):
        if op in relation:
            lhs, rhs = relation.split(op, 1)
            break
    d = (parse(lhs).sy - parse(rhs).sy).subs(sp.Symbol("x"), point)
    d = sp.simplify(d)
    if d.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or d.is_real is False:
        return False
    v = sp.N(d, 50)
    if not v.is_real:
        return False
    return {"<": v < 0, "<=": v <= 0, ">": v > 0, ">=": v >= 0}[op] and True


GUARD_CASES = [
    "x^2 - 3*x + 2 < 0",
    "x^2 - 3*x + 2 <= 0",
    "(x-1)/(x+2) >= 0",
    "abs(x-1) >= 2",
    "sqrt(x+3) > x+1",
    "log(x) > 1",
    "(x-2)^2*(x+1) > 0",
]


def test_inequality_answer_is_certified_on_both_sides_of_every_endpoint():
    """STRAŻNIK K5: every finite endpoint of the answer is checked EXACTLY (in or
    out), and the cells on BOTH sides of it carry a sample whose membership agrees
    with the relation evaluated independently here."""
    for rel in GUARD_CASES:
        res = solve_inequality(rel, "x")
        assert res.kind == "inequality"
        assert res.proof == "proof", rel
        ends = {sp.sympify(e.point) for e in res.endpoints}
        cells_lo = {sp.sympify(c.lo): c for c in res.cells}
        cells_hi = {sp.sympify(c.hi): c for c in res.cells}
        solution = sp.sympify(res.solution)
        boundary = [b for b in solution.boundary if b.is_finite] if solution != sp.S.EmptySet else []
        for b in boundary:
            assert b in ends, (rel, b)
            check = next(e for e in res.endpoints if sp.sympify(e.point) == b)
            assert check.exact is True
            assert check.member == bool(solution.contains(b)) == _holds(rel, b), (rel, b)
            assert b in cells_hi and b in cells_lo, (rel, b)  # both sides
            for cell in (cells_hi[b], cells_lo[b]):
                s = sp.sympify(cell.sample)
                assert cell.member == bool(solution.contains(s)) == _holds(rel, s), (rel, cell)


# --- inequalities: API, refusals, certificate ----------------------------------------
import json  # noqa: E402
import random  # noqa: E402

import pytest  # noqa: E402

from pycodemath.core.errors import (  # noqa: E402
    DomainError,
    ParseError,
    PycodemathError,
    UnsupportedFormError,
)
from pycodemath.engine.solvesets import (  # noqa: E402
    SolveSetUndecided,
    periodic_solutions,
    solve_system,
)


@pytest.mark.parametrize(
    "rel, pretty",
    [
        ("x^2 - 3*x + 2 < 0", "(1, 2)"),
        ("x^2 - 3*x + 2 <= 0", "[1, 2]"),
        ("x^2 + 1 < 0", "∅"),
        ("x^2 + 1 > 0", "ℝ"),
        ("(x-1)^2 <= 0", "{1}"),
        ("1/(x-1) < 1/(x+1)", "(-1, 1)"),
        ("abs(x) + abs(x-2) <= 4", "[-1, 3]"),
        ("x ≥ 2", "[2, oo)"),
    ],
)
def test_inequality_pretty_answers(rel, pretty):
    assert solve_inequality(rel).pretty == pretty


def test_domain_is_part_of_the_answer():
    r = solve_inequality("log(x) > 1")
    assert r.solution == "Interval.open(E, oo)"
    assert r.domain == "Interval.open(0, oo)"
    assert r.conditions == ("x > 0",)
    r = solve_inequality("(x-1)/(x+2) >= 0")
    assert "x + 2 != 0" in r.conditions
    assert not next(e for e in r.endpoints if e.point == "-2").in_domain


def test_methods_are_recorded_and_honest():
    assert solve_inequality("x^3 - x >= 0").method == "exact-roots"
    r = solve_inequality("exp(2*x) - 3*exp(x) + 2 > 0")
    assert r.method == "interval-arithmetic"
    assert {c.method for c in r.cells} <= {"exact-roots", "interval-arithmetic", "outside-domain"}


def test_every_sample_lies_strictly_inside_its_cell():
    r = solve_inequality("sqrt(x^2-4) >= x-1")
    for c in r.cells:
        lo, hi, q = sp.sympify(c.lo), sp.sympify(c.hi), sp.sympify(c.sample)
        assert bool(lo < q) and bool(q < hi)


def test_window_for_trigonometric_inequalities():
    r = solve_inequality("sin(x) >= 1/2", within=("0", "2*pi"))
    assert r.solution == "Interval(pi/6, 5*pi/6)"
    assert r.within == ("0", "2*pi")


def test_trigonometric_inequality_on_the_whole_line_is_refused_not_guessed():
    # raw SymPy answers Interval.open(0, pi) here, which misses (2*pi, 3*pi)
    assert sp.solveset(sp.sin(X) > 0, X, sp.S.Reals) == sp.Interval.open(0, sp.pi)
    with pytest.raises(SolveSetUndecided) as exc:
        solve_inequality("sin(x) > 0")
    assert exc.value.reason == "periodic-unbounded"


@pytest.mark.parametrize(
    "rel, err",
    [
        ("x + y < 1", UnsupportedFormError),
        ("gamma(x) > 1", UnsupportedFormError),
        ("1 < x < 2", ParseError),
        ("x^2 - 1", ParseError),
        ("x = 1", ParseError),
        ("< 3", ParseError),
        ("I*x > 0", DomainError),
    ],
)
def test_bad_inequalities_are_typed_refusals(rel, err):
    with pytest.raises(err):
        solve_inequality(rel)


def test_empty_or_bad_window_is_refused():
    with pytest.raises(DomainError):
        solve_inequality("x > 0", within=("2", "1"))
    with pytest.raises(DomainError):
        solve_inequality("x > 0", within=("1",))  # type: ignore[arg-type]


def test_relational_object_is_accepted():
    r = solve_inequality(sp.Symbol("x") ** 2 < 4)
    assert r.solution == "Interval.open(-2, 2)"


def test_to_dict_is_json_and_complete():
    d = solve_inequality("(x-1)/(x+2) >= 0").to_dict()
    json.dumps(d)
    assert d["kind"] == "inequality" and d["verdict"] == "verified" and d["proof"] == "proof"
    assert d["sympy"]["agrees"] is True
    assert {"point", "approx", "in_domain", "sign", "member", "exact"} <= set(d["endpoints"][0])


def test_double_root_is_a_closed_point_not_a_gap():
    r = solve_inequality("(x-2)^2*(x+1) >= 0")
    assert r.solution == "Interval(-1, oo)"
    two = next(e for e in r.endpoints if e.point == "2")
    assert two.sign == 0 and two.member


def test_crootof_endpoint_is_exact():
    r = solve_inequality("x^5 - x + 1 > 0")
    assert r.endpoints[0].point == "CRootOf(x**5 - x + 1, 0)"
    assert r.endpoints[0].sign == 0 and not r.endpoints[0].member
    assert r.endpoints[0].approx.startswith("-1.16730")


def _exact_holds(h: sp.Expr, op: str, p: sp.Expr) -> bool:
    v = h.subs(sp.Symbol("x"), p)
    if v.has(sp.zoo, sp.nan):
        return False
    return bool({"<": v < 0, "<=": v <= 0, ">": v > 0, ">=": v >= 0}[op])


def test_fuzz_random_rational_inequalities_never_wrong_and_never_raw():
    """60 random polynomial/rational inequalities (the 1000-case run is in
    dev-journal/full/K5/fuzz_inequalities.py): every answer agrees with direct exact
    evaluation at random rationals and around every endpoint; failures are typed."""
    rng = random.Random(5)
    xs = sp.Symbol("x")
    for _ in range(60):
        deg = rng.randint(1, 4)
        P = sum(rng.randint(-5, 5) * xs**k for k in range(deg)) + rng.choice([1, -1, 2]) * xs**deg
        h = P / (xs - rng.randint(-3, 3)) if rng.random() < 0.4 else P
        op = rng.choice(["<", "<=", ">", ">="])
        rel = f"{sp.sstr(h)} {op} 0"
        try:
            res = solve_inequality(rel)
        except PycodemathError:
            continue
        S = sp.sympify(res.solution)
        probes = [sp.Rational(rng.randint(-300, 300), rng.choice([1, 7, 11])) for _ in range(15)]
        for b in (S.boundary if S != sp.S.EmptySet else []):
            if b.is_finite:
                q = sp.nsimplify(sp.N(b, 40), rational=True)
                probes += [b, q - sp.Rational(1, 10**8), q + sp.Rational(1, 10**8)]
        for p in probes:
            assert bool(S.contains(p)) == _exact_holds(h, op, p), (rel, p, res.solution)


# --- systems ---------------------------------------------------------------------------
def test_linear_unique():
    r = solve_system(["x + y = 3", "x - y = 1"], ["x", "y"])
    assert (r.status, r.solutions, r.rank, r.method) == ("unique", ({"x": "2", "y": "1"},), 2, "linear")
    assert r.checks[0]["residual"] == "0"


def test_linear_infinite_is_parametrised_by_free_variables():
    r = solve_system(["x + y + z = 1", "x + y + z = 1", "x - z = 0"], ["x", "y", "z"])
    assert r.status == "infinite" and r.rank == 2 and len(r.parameters) == 1
    sol = {k: sp.sympify(v) for k, v in r.solutions[0].items()}
    for e in ("x + y + z - 1", "x - z"):
        assert sp.expand(sp.sympify(e).subs({sp.Symbol(k): v for k, v in sol.items()})) == 0


def test_linear_inconsistent_has_a_checkable_certificate():
    r = solve_system(["x + 2*y = 1", "2*x + 4*y = 3"], ["x", "y"])
    assert r.status == "none" and r.rank == 1 and r.rank_augmented == 2
    y = [sp.sympify(c) for c in r.proof["y"]]
    A = sp.Matrix([[1, 2], [2, 4]])
    b = sp.Matrix([1, 3])
    assert (sp.Matrix([y]) * A).is_zero_matrix
    assert (sp.Matrix([y]) * b)[0] != 0
    assert r.proof["combination"].startswith("0 = ")


def test_groebner_one_proves_no_solution():
    r = solve_system(["x*y = 1", "x = 0"])
    assert r.status == "none" and r.proof["groebner_basis"] == ["1"]


def test_polynomial_solutions_complete_by_radical_count():
    r = solve_system(["x^2 + y^2 = 1", "x = y"], ["x", "y"])
    assert r.status == "finite" and len(r.solutions) == 2 == r.proof["distinct_complex_solutions"]
    assert r.real == (True, True)


def test_multiplicity_is_not_mistaken_for_a_missing_solution():
    r = solve_system(["x^2 = 0", "y = 1"], ["x", "y"])
    assert r.solutions == ({"x": "0", "y": "1"},)


def test_complex_solutions_are_tagged():
    r = solve_system(["x^2 + y^2 = -1", "x - y = 0"], ["x", "y"])
    assert len(r.solutions) == 2 and r.real == (False, False)
    rr = solve_system(["x^2 + y^2 = -1", "x - y = 0"], ["x", "y"], real=True)
    assert rr.status == "none" and rr.solutions == ()


def test_sympy_solve_coming_back_empty_is_caught():
    """sp.solve returns [] for this system (SymPy 1.14); the radical count says 5."""
    xs, ys = sp.symbols("x y")
    assert sp.solve([xs**5 - xs + 1, ys - xs], [xs, ys], dict=True) == []
    r = solve_system(["x^5 - x + 1 = 0", "y = x"], ["x", "y"])
    assert len(r.solutions) == 5 and sum(bool(v) for v in r.real) == 1
    assert all(c["how"] == "exact" for c in r.checks)


def test_denominators_exclude_points():
    r = solve_system(["(x^2-1)/(x-1) = 0"])
    assert r.solutions == ({"x": "-1"},)
    r = solve_system("x/y = 2, x + y = 3")
    assert r.solutions == ({"x": "2", "y": "1"},)


@pytest.mark.parametrize(
    "eqs",
    [["exp(x) + y = 1", "x - y = 0"], ["x^2 - y^2 = 0"], ["sin(x) = y", "x + y = 1"]],
)
def test_out_of_reach_systems_route_to_solve_nd(eqs):
    with pytest.raises(UnsupportedFormError) as exc:
        solve_system(eqs)
    assert exc.value.route == "solve_nd"


def test_system_input_errors():
    with pytest.raises(DomainError):
        solve_system([])
    with pytest.raises(UnsupportedFormError):
        solve_system(["a*x = 1"], ["x"])
    with pytest.raises(ParseError):
        solve_system(["x < 1"])
    with pytest.raises(ParseError):
        solve_system(["x = 1 = 2"])
    with pytest.raises(DomainError):
        solve_system(["x = 1"], ["x", "x"])


def test_system_to_dict_is_json():
    d = solve_system(["x^3 + y = 2", "y = x"]).to_dict()
    json.dumps(d)
    assert d["kind"] == "system" and d["status"] == "finite" and len(d["solutions"]) == 3


# --- periodic families -----------------------------------------------------------------
def test_periodic_plan_example():
    r = periodic_solutions("sin(x) = 1/2")
    assert r.text == "x = pi/6 + 2*pi*k, x = 5*pi/6 + 2*pi*k, k ∈ ℤ"
    assert {c["k"] for c in r.checks} == {-2, -1, 0, 1, 2} and len(r.checks) == 10


@pytest.mark.parametrize(
    "eq, families",
    [
        ("sin(x) = 0", [("0", "pi")]),
        ("sin(2*x) = 0", [("0", "pi/2")]),
        ("cos(2*x) = cos(x)", [("0", "2*pi/3")]),
        ("tan(x) = 1", [("pi/4", "pi")]),
        ("sin(x)^2 = 1/4", [("pi/6", "pi"), ("5*pi/6", "pi")]),
    ],
)
def test_families_are_merged(eq, families):
    r = periodic_solutions(eq)
    assert [(f.base, f.period) for f in r.families] == families


def test_no_real_solution_is_proved_empty():
    r = periodic_solutions("cos(x) = 2")
    assert r.status == "empty" and r.families == () and r.cells


def test_tan_pole_is_outside_the_domain():
    r = periodic_solutions("tan(x) = 1")
    pole = next(e for e in r.endpoints if e.point == "pi/2")
    assert not pole.in_domain


def test_non_periodic_equation_is_refused_with_route():
    with pytest.raises(UnsupportedFormError) as exc:
        periodic_solutions("x*sin(x) = 0")
    assert exc.value.route == "root"


def test_periodic_to_dict_is_json():
    d = periodic_solutions("sin(x) + cos(x) = 1").to_dict()
    json.dumps(d)
    assert d["kind"] == "periodic" and len(d["families"]) == 2


# --- regressions from the adversarial round and the fuzz run ---------------------------
def test_finite_sets_print_as_readable_sympy():
    """'{1}' read back with sympify is a Python set; the answer must round-trip."""
    r = solve_inequality("(x-1)^2 <= 0")
    assert r.solution == "FiniteSet(1)"
    assert sp.sympify(r.solution) == sp.FiniteSet(1)


def test_identical_sides():
    assert solve_inequality("x > x").solution == "EmptySet"
    assert solve_inequality("x >= x").pretty == "ℝ"


def test_decimals_mean_the_decimal_written():
    assert solve_inequality("0.1*x + 0.2 > 0.3").solution == "Interval.open(1, oo)"
    assert solve_system(["0.5*x = 1"]).solutions == ({"x": "2"},)


def test_quartic_with_casus_irreducibilis_is_fast_and_exact():
    """Measured before the fix: 120 s time budget (SymPy's solveset building radicals);
    now the roots are CRootOf and the isolated roots are known zeros."""
    import time

    t = time.perf_counter()
    r = solve_inequality("(-x^4 + 5*x^3 - 4*x^2 - 2*x + 3)/(-5*x^5 - 5*x^4 - 2*x^3 + 5*x^2 + 5*x) < 0")
    assert time.perf_counter() - t < 30
    assert "CRootOf" in r.solution and r.approx.count("∪") == 2


def test_cot_from_sympy_shift_is_enclosable():
    """tan(pi/4 - u) is rewritten by SymPy as cot(u + pi/4) at a cell's end."""
    r = solve_inequality("tan(x) > 1", within=("0", "pi/2 - 1/10"))
    assert sp.sympify(r.solution) == sp.Interval.Lopen(sp.pi / 4, sp.pi / 2 - sp.Rational(1, 10))


def test_known_weak_spots_refuse_instead_of_guessing():
    with pytest.raises(SolveSetUndecided):
        solve_inequality("x^x > 1")  # SymPy's solveset returns a ConditionSet
