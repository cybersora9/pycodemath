"""K2 — ``identify``: a closed form only when it cannot be a coincidence
(``pycodemath.engine.identify``).

Guardian: ``test_identify_never_false_positive`` — the constants of
``K2_constants.json`` at 60 digits are never recognised as something they are not,
and the random numbers (seed in the file) are never recognised at all.
"""

from __future__ import annotations

import doctest
import json
import warnings
from decimal import Decimal
from importlib import resources

import mpmath
import pytest
import sympy as sp

from pycodemath.core.errors import DomainError, ParseError, PycodemathError
from pycodemath.engine import identify as identify_module
from pycodemath.engine.identify import (
    IDENTIFY_MIN_DIGITS, IDENTIFY_MIN_MARGIN, Identification, default_basis, identify,
)
from pycodemath.engine.precision import approx, to_sympy
from pycodemath.verify.verdict import VerdictStatus


def _data() -> dict:
    text = resources.files("pycodemath.code.data").joinpath("K2_constants.json").read_text("utf-8")
    return json.loads(text)


DATA = _data()
CONSTANTS = DATA["constants"]
RANDOM = DATA["random"]


def _same_number(a: str, b: str, digits: int = 70) -> bool:
    va = approx(a, digits)
    vb = approx(b, digits)
    return va.value == vb.value


def _check_constant(case: dict) -> Identification:
    res = identify(case["value60"])
    if res.verified:
        assert res.value is not None
        assert _same_number(res.value, case["expr"]), (case["expr"], res.value)
    else:
        assert res.value is None
    return res


def _check_random(case: dict) -> Identification:
    res = identify(case["value60"])
    assert not res.verified, (case["value60"], res.value, res.detail)
    assert res.value is None
    return res


DEV_CONSTANTS = [c for c in CONSTANTS if c["split"] == "dev"]
DEV_RANDOM = [c for c in RANDOM if c["split"] == "dev"]


def test_identify_never_false_positive():
    """50 constants + 50 random numbers (the guardian): no false recognition."""
    assert len(CONSTANTS) >= 50 and len(RANDOM) >= 50
    for case in CONSTANTS[:50]:
        _check_constant(case)
    for case in RANDOM[:50]:
        _check_random(case)


@pytest.mark.parametrize("case", CONSTANTS[50:], ids=[c["expr"] for c in CONSTANTS[50:]])
def test_remaining_constants_never_false(case):
    _check_constant(case)


@pytest.mark.parametrize("case", RANDOM[50:], ids=[c["value60"][:12] for c in RANDOM[50:]])
def test_remaining_random_never_recognised(case):
    _check_random(case)


def test_double_precision_input_is_never_verified():
    """16-17 digits (a double) can never be VERIFIED — whatever the candidate."""
    for case in CONSTANTS[:20] + RANDOM[:20]:
        short = mpmath.nstr(mpmath.mpf(case["value60"]), 16)
        res = identify(short)
        assert res.status is VerdictStatus.UNDECIDED
        assert any("numerology" in w for w in res.warnings)


def test_recall_on_dev_constants_in_family():
    """Measured recall (dev split, families that should contain them): pinned
    below the measured value so a regression shows, not to flatter."""
    fam = [c for c in DEV_CONSTANTS if c["in_family"]]
    hits = sum(identify(c["value60"]).verified for c in fam)
    assert hits / len(fam) >= 0.85, (hits, len(fam))


def test_pi_squared_over_six_from_a_double():
    res = identify("1.6449340668482264")
    assert res.status is VerdictStatus.UNDECIDED and res.value is None
    top = res.candidates[0]
    assert top.expr == "pi**2/6" and top.consistent
    assert top.margin < IDENTIFY_MIN_MARGIN or res.digits < IDENTIFY_MIN_DIGITS
    assert "numerology" in " ".join(res.warnings)
    assert "pi**2/6" in res.detail


def test_pi_squared_over_six_from_sixty_digits():
    digits = mpmath.nstr(mpmath.zeta(2), 60) if False else None
    with mpmath.workdps(80):
        digits = mpmath.nstr(mpmath.pi ** 2 / 6, 60)
    res = identify(digits)
    assert res.verified and res.value == "pi**2/6"
    assert res.candidates[0].rigorous and res.candidates[0].margin >= IDENTIFY_MIN_MARGIN
    assert res.verdict.verified


def test_a_wrong_last_digit_is_not_recognised():
    with mpmath.workdps(80):
        text = mpmath.nstr(mpmath.pi, 60)
    wrong = text[:-1] + str((int(text[-1]) + 3) % 10)
    res = identify(wrong)
    assert not res.verified


def test_truncated_digits_are_accepted():
    with mpmath.workdps(100):
        full = mpmath.nstr(mpmath.e, 80)
    truncated = full[:62]  # 61 digits, cut not rounded
    res = identify(truncated)
    assert res.verified and res.value == "E"


def test_digits_parameter_limits_what_is_trusted():
    with mpmath.workdps(80):
        text = mpmath.nstr(mpmath.pi ** 2 / 6, 60)
    assert identify(text, digits=10).status is VerdictStatus.UNDECIDED
    assert identify(text, digits=10).digits == 10
    with pytest.raises(DomainError):
        identify("1.5", digits=10)
    with pytest.raises(DomainError):
        identify("1.5", digits=0)


def test_exact_input_is_proved_by_check_equal():
    res = identify("zeta(2)")
    assert res.verified and res.value == "pi**2/6" and res.method in ("check_equal", "exact")
    assert identify("1/3").value == "1/3"
    res = identify("atan(1)*4")
    assert res.verified and res.value == "pi"
    res = identify("gamma(1/3)")
    assert res.status is VerdictStatus.UNDECIDED


def test_custom_basis():
    with mpmath.workdps(80):
        text = mpmath.nstr(mpmath.log(3) * 2 + 1, 60)
    res = identify(text, basis=["log(3)"])
    assert res.verified and _same_number(res.value or "0", "1 + 2*log(3)")
    with pytest.raises(DomainError):
        identify("1.5", basis="pi")
    with pytest.raises(PycodemathError):
        identify("1.5", basis=["x + 1"])


def test_inputs():
    assert identify(1.6449340668482264).candidates[0].expr == "pi**2/6"
    assert identify(Decimal("0.5")).status is VerdictStatus.UNDECIDED
    assert identify(sp.Float("3.14159265358979", 15)).candidates[0].expr == "pi"
    for bad in [float("nan"), float("inf"), True, None, "x + 1", ""]:
        with pytest.raises(PycodemathError):
            identify(bad)  # type: ignore[arg-type]
    with pytest.raises(DomainError):
        identify("I + 1")


def test_negative_and_scaled_numbers():
    with mpmath.workdps(80):
        neg = mpmath.nstr(-mpmath.sqrt(2), 60)
        tiny = mpmath.nstr(mpmath.pi * mpmath.mpf(10) ** -30, 60)
        big = mpmath.nstr(mpmath.e * mpmath.mpf(10) ** 12, 60)
    assert identify(neg).value == "-sqrt(2)"
    assert _same_number(identify(tiny).value or "0", "pi/10**30")
    assert _same_number(identify(big).value or "0", "E*10**12")


def test_to_dict_kind_and_json():
    res = identify("1.6449340668482264")
    d = res.to_dict()
    assert d["kind"] == "identify" and d["status"] == "undecided"
    assert json.loads(json.dumps(d)) == d
    assert d["candidates"][0]["expr"] == "pi**2/6"
    assert "UNDECIDED" in str(res)


def test_default_basis_is_the_data_list():
    assert default_basis() == tuple(DATA["basis"])
    assert len(default_basis()) == 9


def test_no_warning_leak(capfd):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        identify("1.6449340668482264")
        identify("zeta(2)")
    out, err = capfd.readouterr()
    assert caught == [] and err == "" and out == ""


def test_doctests():
    res = doctest.testmod(identify_module)
    assert res.failed == 0 and res.attempted >= 3


def test_the_seed_and_licence_are_recorded():
    assert DATA["seed"] == 20261030
    assert "mpmath" in DATA["source"] and "OEIS" in DATA["source"]
    assert all(to_sympy(c["expr"]) is not None for c in CONSTANTS)
    with pytest.raises(ParseError):
        to_sympy("exp(10**20*log(1 + 10**-20))")  # would build r**(10**20) exactly
