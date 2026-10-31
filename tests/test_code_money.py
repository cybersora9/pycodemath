"""C4 — money and rounding (``pycodemath.code.money``).

Guards (seen RED before the module existed, see dev-journal/full/C4.md):
``test_allocation_always_sums_exactly`` and ``test_per_line_vat_vs_total_vat_has_witness``.

The evaluation set lives in ``pycodemath/code/data/C4_cases.json``; only its
``dev`` split is used here. The ``held`` split is measured once, at the end, by
``dev-journal/full/C4/measure.py``.
"""

from __future__ import annotations

import json
import random
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pycodemath.code.money import (
    allocate,
    amortization_schedule,
    annuity_payment,
    check_rounding_strategy,
    compound_interest,
    currency_exponent,
    float_money_hazard,
    irr,
    money_grid,
    npv,
    round_money,
    vat_per_line_vs_total,
)
from pycodemath.verify.verdict import VerdictStatus

CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "pycodemath/code/data/C4_cases.json").read_text()
)["cases"]
DEV = [c for c in CASES if c["split"] == "dev"]


# --- guards -----------------------------------------------------------------


def _random_allocation(rng: random.Random) -> "tuple[str, list[int] | int, str]":
    currency = rng.choice(["PLN", "JPY", "KWD", "EUR", "CLF"])
    exp = currency_exponent(currency)
    units = rng.choice([0, 1, 2, 3, rng.randrange(10**6), rng.randrange(10**15)])
    units *= rng.choice([1, -1])
    total = str(Decimal(units).scaleb(-exp))
    if rng.random() < 0.3:
        return total, rng.randrange(1, 40), currency
    weights = [rng.choice([0, 1, 2, 3, 7, rng.randrange(10**9)]) for _ in range(rng.randrange(1, 12))]
    if not any(weights):
        weights[0] = 1
    return total, weights, currency


def test_allocation_always_sums_exactly() -> None:
    """10 000 seeded random allocations: shares sum to the total to the last minor
    unit, each share is within one unit of its exact quota, equal splits differ by
    at most one unit — for both methods."""
    rng = random.Random(20261030)
    for _ in range(10_000):
        total, weights, currency = _random_allocation(rng)
        for method in ("hamilton", "in_order"):
            a = allocate(total, weights, currency, method=method)
            assert sum(a.shares) == Decimal(total), (total, weights, currency, method)
            unit = Decimal(1).scaleb(-currency_exponent(currency))
            for s, q in zip(a.shares, a.quotas):
                # hamilton: strictly within one unit; in_order: up to one full unit
                # (an integer quota can get +1 — the quota violation it is known for)
                assert abs(Fraction(s) - q) < Fraction(unit) or (
                    method == "in_order" and abs(Fraction(s) - q) == Fraction(unit))
            if isinstance(weights, int):
                assert max(a.shares) - min(a.shares) <= unit
            if method == "hamilton":
                assert a.within_quota


@settings(max_examples=400, deadline=None)
@given(
    units=st.integers(min_value=-(10**20), max_value=10**20),
    weights=st.lists(st.integers(min_value=0, max_value=10**6), min_size=1, max_size=15).filter(any),
    exp=st.sampled_from(["JPY", "PLN", "KWD", "CLF"]),
)
def test_allocation_sums_exactly_hypothesis(units: int, weights: "list[int]", exp: str) -> None:
    total = str(Decimal(units).scaleb(-currency_exponent(exp)))
    a = allocate(total, weights, exp)
    assert sum(a.shares) == Decimal(total)
    assert a.within_quota and a.sums_exactly


def test_per_line_vat_vs_total_vat_has_witness() -> None:
    """Three 1-grosz lines at 23%: per-line VAT is 0.00, VAT on the total is 0.01.
    The divergence comes with the concrete lines, and every number is exact."""
    r = vat_per_line_vs_total(["0.01", "0.01", "0.01"], "0.23", "PLN", "HALF_UP")
    assert r.diverges
    assert r.per_line == Decimal("0.00") and r.on_total == Decimal("0.01")
    assert r.witness_lines == ("0.01", "0.01", "0.01")
    assert r.difference == Decimal("-0.01")
    # data that happens to agree still gets a witness for the same rate and line count
    r2 = vat_per_line_vs_total(["10.05", "10.05"], "0.23", "PLN", "HALF_UP")
    assert not r2.diverges and r2.always_equal is False and r2.witness_lines is not None
    w = [Decimal(x) for x in r2.witness_lines]
    per = sum(round_money(x * Decimal("0.23"), "PLN", "HALF_UP").rounded for x in w)
    tot = round_money(sum(w) * Decimal("0.23"), "PLN", "HALF_UP").rounded
    assert per != tot


# --- the dev split of the evaluation set --------------------------------------


def run_case(case: dict) -> dict:
    """Run one evaluation case through the module and return the observed values,
    keyed like the case's ``expect``. Shared with dev-journal/full/C4/measure.py."""
    i, kind = case["input"], case["kind"]
    if kind == "round":
        r = round_money(i["value"], i["currency"], i["mode"], step=i.get("step"))
        return {"rounded": str(r.rounded)}
    if kind == "allocate":
        a = allocate(i["total"], i.get("weights", i.get("parts")), i["currency"],
                     method=i.get("method", "hamilton"))
        return {"shares": [str(s) for s in a.shares], "within_quota": a.within_quota}
    if kind == "vat_lines":
        v = vat_per_line_vs_total(i["lines"], i["rate"], i["currency"], i["mode"])
        return {"per_line": str(v.per_line), "on_total": str(v.on_total), "diverges": v.diverges}
    if kind == "staged":
        steps = i["steps"]

        def staged(rnd, value):  # type: ignore[no-untyped-def]
            for s in steps:
                value = rnd(value, step=s)
            return value

        rep = check_rounding_strategy(staged, {"value": i["value"]}, "EUR", i["mode"], step=i["final_step"])
        return {"once": str(rep.once), "stepwise": str(rep.stepwise)}
    if kind == "compound":
        c = compound_interest(i["principal"], i["rate"], i["periods"], i["currency"], i["mode"],
                              periods_per_year=i["periods_per_year"])
        return {"once": str(c.once), "stepwise": str(c.stepwise)}
    if kind == "float":
        variables = None
        if "grid" in i:
            variables = {k: money_grid(*v) for k, v in i["grid"].items()}
        h = float_money_hazard(i["expr"], variables=variables, currency=i.get("currency"),
                               mode=i.get("mode", "HALF_EVEN"))
        return {"hazard": h.hazard, "float": h.float_value, "exact": h.exact_value,
                "hazards": h.hazards, "checked": h.checked, "mode_differs": h.mode_differs}
    if kind in ("annuity", "schedule"):
        args = (i["principal"], i["rate"], i["periods"], i["currency"], i["mode"])
        if kind == "annuity":
            return {"payment": str(annuity_payment(*args, periods_per_year=i["periods_per_year"]).payment)}
        s = amortization_schedule(*args, periods_per_year=i["periods_per_year"])
        return {"payment": str(s.regular_payment), "last_payment": str(s.last_payment),
                "total_interest": str(s.total_interest), "total_paid": str(s.total_paid)}
    if kind == "npv":
        return {"exact": str(npv(i["rate"], i["cashflows"]).exact)}
    if kind == "irr":
        r = irr(i["cashflows"])
        return {"rates": [str(x) for x in r.rates], "unique": r.unique}
    if kind == "currency":
        return {"exponent": currency_exponent(i["code"])}
    raise AssertionError(kind)


def matches(case: dict, got: dict) -> bool:
    for key, want in case["expect"].items():
        have = got.get(key)
        if key == "rates":
            if len(have) != len(want) or any(abs(Decimal(h) - Decimal(w)) > Decimal("1e-10")
                                             for h, w in zip(have, want)):
                return False
        elif have != want:
            return False
    return True


@pytest.mark.parametrize("case", DEV, ids=[c["id"] for c in DEV])
def test_dev_case(case: dict) -> None:
    got = run_case(case)
    assert matches(case, got), (case["expect"], got)


def test_eval_set_shape() -> None:
    assert len(CASES) >= 40
    held = sum(c["split"] == "held" for c in CASES)
    assert held * 3 >= len(CASES) - 3  # about one third held out


# --- regressions from the adversarial round (dev-journal/full/C4/adversary.py) ----

from pycodemath.code.money import MoneyInputError  # noqa: E402


@pytest.mark.parametrize(
    "call",
    [
        lambda: round_money(2.675, "PLN"),
        lambda: round_money(Decimal("NaN"), "PLN"),
        lambda: round_money("inf", "PLN"),
        lambda: round_money("1", "XAU"),
        lambda: round_money("1", "PLN", "ROUND_05UP"),
        lambda: round_money("1", "PLN", step="0.003"),
        lambda: allocate("0.005", 2, "PLN"),
        lambda: allocate("1", [0, 0], "PLN"),
        lambda: allocate("1", [0.5, 0.5], "PLN"),
        lambda: allocate("1", [-1, 2], "PLN"),
        lambda: float_money_hazard("__import__('os').system('true')"),
        lambda: float_money_hazard("9**9**9"),
        lambda: float_money_hazard("1/0"),
        lambda: annuity_payment("1000", "-1", 12, "PLN"),
        lambda: irr(["0", "0"]),
    ],
)
def test_refusals_are_typed(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(MoneyInputError):
        call()


def test_to_dict_is_json_and_never_e_notation() -> None:
    results = [
        round_money("0.000000014", 8, "HALF_UP"),
        allocate("100", 3, "PLN"),
        vat_per_line_vs_total(["0.01"] * 3, "0.23", "PLN"),
        compound_interest("100", "0.05", 3, "PLN"),
        float_money_hazard("round(2.675, 2)"),
        annuity_payment("1000", "0.1", 12, "PLN", periods_per_year=12),
        amortization_schedule("1000", "0.1", 12, "PLN", periods_per_year=12),
        npv("0.1", ["-100", "110"]),
        irr(["-100", "100"]),
    ]
    for r in results:
        text = json.dumps(r.to_dict())
        assert "E-" not in text and r.to_dict()["kind"]
    assert results[0].to_dict()["rounded"] == "0.00000001"
    assert results[-1].to_dict()["rate"] == "0.000000000000"


def test_no_false_alarm_on_correct_money_code() -> None:
    cents = money_grid("0.01", "20.00")
    assert not float_money_hazard("x * 3", variables={"x": cents}, currency="PLN").hazard
    assert not float_money_hazard("x * 12", variables={"x": cents}, currency="PLN").hazard
    # ...but a VAT multiplication rounded to the cent IS a hazard at exact ties (0.50 * 1.23 = 0.615)
    tie = float_money_hazard("x * 1.23", variables={"x": cents}, currency="PLN")
    assert tie.hazard and tie.witness == {"x": "0.50"}
    assert (tie.float_value, tie.exact_value, tie.float_raw) == ("0.61", "0.62", "0.615")
    assert not float_money_hazard("0.5 + 0.25 == 0.75").hazard
    r = float_money_hazard("x * 100", variables={"x": cents})
    assert not r.hazard and r.representation_only > 0 and r.verdict is VerdictStatus.VERIFIED


def test_float_hazard_cause_is_attributed() -> None:
    h = float_money_hazard("round(2.675, 2)")
    assert h.literal_errors == (("2.675", "2.67499999999999982236431605997495353221893310546875"),)
    m = float_money_hazard("round(1.005 * 100)", mode="HALF_UP")
    assert not m.hazard and m.mode_differs and m.exact_mode_value == "101"


def test_irr_multiplicity_and_uniqueness() -> None:
    r = irr(["-1", "3", "-3", "1"])  # (x - 1)^3: IRR 0 with multiplicity 3
    assert r.unique and r.multiplicities == (3,) and r.rate == 0
    two = irr(["-100", "230", "-132"])
    assert not two.unique and two.rate is None and [float(x) for x in two.rates] == [0.1, 0.2]
    for (lo, hi), x in zip(two.brackets, two.rates):
        assert lo <= Fraction(x) + Fraction(1, 10**12) and Fraction(x) - Fraction(1, 10**12) <= hi


def test_schedule_invariants_and_early_payoff() -> None:
    s = amortization_schedule("10", "0", 12, "JPY")
    assert s.periods_used == 10 and s.total_paid == 10
    e = amortization_schedule("1000", "0.1", 12, "PLN", periods_per_year=12, schedule="equal_principal")
    assert sum(r.principal for r in e.rows) == Decimal("1000.00")
    assert e.rows[0].principal == Decimal("83.34") and e.rows[-1].balance == 0


def test_alabama_paradox_is_documented_behaviour() -> None:
    """Hamilton's known price: a larger total can lower a share (module docstring)."""
    small = allocate("0.03", [1, 3, 3], "PLN").shares
    large = allocate("0.04", [1, 3, 3], "PLN").shares
    assert small[0] > large[0]


@pytest.mark.parametrize("expr", [
    "(((10**64)**64)**64)**64 > 1",
    "((10**64 * 2)**64)**64 > 1",
    "(round(10**64)**64)**64 > 1",
])
def test_nested_powers_are_refused_before_they_run(expr: str) -> None:
    """Integer powers stack (10**64**4 has 16.7 million digits) and CPython computes
    them in one C call that no time budget can interrupt — so a power whose base
    already contains a power is refused up front."""
    with pytest.raises(MoneyInputError, match=r"\*\*"):
        float_money_hazard(expr)


def test_single_power_still_works() -> None:
    assert float_money_hazard("round(1.05**2, 2) == 1.1").hazard is False
