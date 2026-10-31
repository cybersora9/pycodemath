"""K3 — the calculator layer: number theory, combinatorics, special functions.

Strażniki modułu (widziane CZERWONE przed implementacją, raport K3 §3):
``test_calculator_layer_certifies_itself``, ``test_carmichael_numbers_are_not_prime``
(i w ``test_stats.py``: ``test_ab_sample_size_matches_reference``).

Każdy wynik niesie certyfikat; tu sprawdzamy go DRUGI raz, z zewnątrz, innymi
narzędziami niż te, które wyprodukowały wynik (gołe ``pow``/``%``, SymPy jako
niezależna implementacja) — test, który ufa certyfikatowi, niczego nie dowodzi.
"""

from __future__ import annotations

import inspect
import json
import math

import pytest
import sympy as sp

from pycodemath.core.errors import DomainError, PycodemathError
from pycodemath.engine import calc
from pycodemath.engine.calc import CalcResult


def _json_roundtrip(result: CalcResult) -> dict:
    payload = result.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    return payload


# --- strażnik 1 -----------------------------------------------------------
def test_calculator_layer_certifies_itself() -> None:
    n = 2**64 + 1
    f = calc.factorint(n)
    assert f.value == {274177: 1, 67280421310721: 1}
    assert math.prod(p**e for p, e in f.value.items()) == n
    assert all(sp.isprime(p) for p in f.value)
    assert f.proof == "proof" and f.certificate and f.exact

    g = calc.gcd(1071, 462)
    assert g.value == 21
    s, t = g.witness["bezout"]
    assert 1071 * s + 462 * t == 21
    assert 1071 % 21 == 0 and 462 % 21 == 0

    inv = calc.modinv(17, 3120)
    assert inv.value == 2753 and 17 * 2753 % 3120 == 1

    x = calc.crt([2, 3, 2], [3, 5, 7])
    assert x.value == 23 and all(23 % m == r for r, m in [(2, 3), (3, 5), (2, 7)])

    assert calc.isprime(2**61 - 1).value is True
    assert calc.binomial(100, 50).value == sp.binomial(100, 50)
    assert calc.fibonacci(100).value == 354224848179261915075

    z = calc.zeta(2, dps=30)
    assert z.proof == "cross-check" and z.exact and z.value == "pi**2/6"
    assert abs(z.as_float() - math.pi**2 / 6) < 1e-15

    for result in (f, g, inv, x, z):
        payload = _json_roundtrip(result)
        assert payload["kind"] and payload["certificate"]


# --- strażnik 2 -----------------------------------------------------------
CARMICHAEL = [561, 1105, 1729, 2465, 2821, 6601, 8911, 41041, 825265, 321197185]
# A014233: the smallest strong pseudoprimes to the first m prime bases. Each one
# fools a Miller-Rabin test that uses exactly those bases.
STRONG_PSEUDOPRIMES = [
    2047,
    1373653,
    25326001,
    3215031751,
    2152302898747,
    3474749660383,
    341550071728321,
    3825123056546413051,
    318665857834031151167461,
    3317044064679887385961981,
]


@pytest.mark.parametrize("n", CARMICHAEL + STRONG_PSEUDOPRIMES)
def test_carmichael_numbers_are_not_prime(n: int) -> None:
    r = calc.isprime(n)
    assert r.value is False
    # a compositeness claim is ALWAYS a proof, and the test re-checks the witness
    assert r.proof == "proof"
    w = r.witness
    if "factor" in w:
        d = w["factor"]
        assert 1 < d < n and n % d == 0
    else:
        assert w["kind"] in ("miller-rabin", "lucas")
        if w["kind"] == "miller-rabin":
            a = w["base"]
            d, s = n - 1, 0
            while d % 2 == 0:
                d //= 2
                s += 1
            x = pow(a, d, n)
            assert x != 1 and all(pow(x, 2**r, n) != n - 1 for r in range(s))


def test_refusals_are_typed() -> None:
    with pytest.raises(DomainError):
        calc.factorint(0)
    with pytest.raises(DomainError):
        calc.modinv(6, 9)
    with pytest.raises(PycodemathError):
        calc.isprime(2.5)  # type: ignore[arg-type]


# --- zestaw oceny K3 (rozwojowy; wstrzymany mierzy dev-journal/full/K3/run_eval.py) ---
DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "pycodemath" / "code" / "data" / "K3_cases.jsonl"


def _load(split: str) -> list[dict]:
    with DATA.open() as fh:
        return [c for c in map(json.loads, fh) if c["split"] == split]


def _as_number(v: object) -> float:
    from fractions import Fraction

    import mpmath

    if isinstance(v, str) and "/" in v:
        return float(Fraction(v))
    return float(mpmath.mpf(v)) if isinstance(v, str) else float(v)  # type: ignore[arg-type]


def run_case(case: dict) -> tuple[bool, str]:
    """(passed, detail). A refusal counts as a miss, never as a pass."""
    from pycodemath.engine import stats

    module = calc if case["op"].startswith("calc.") else stats
    fn = getattr(module, case["op"].split(".", 1)[1])
    try:
        result = fn(*case["args"], **case["kwargs"])
    except PycodemathError as exc:
        return False, f"refused ({type(exc).__name__}): {exc}"
    payload = result.to_dict()
    if not payload["certificate"]:
        return False, "no certificate"
    value = payload["value"]
    if case["path"] is not None:
        value = value[case["path"]]
    elif payload["approx"] is not None and case["compare"] != "exact":
        value = payload["approx"]
    expected = case["expected"]
    if case["compare"] == "exact":
        return value == expected, f"got {str(value)[:80]}"
    kind, tol = case["compare"].split(":")
    got, want = _as_number(value), _as_number(expected)
    err = abs(got - want) if kind == "abs" else abs(got - want) / max(abs(want), 1e-300)
    return err <= float(tol), f"got {got!r}, want {want!r}, {kind} err {err:.2e}"


@pytest.mark.parametrize("case", _load("dev"), ids=lambda c: c["id"])
def test_dev_split_case(case: dict) -> None:
    ok, detail = run_case(case)
    assert ok, f"{case['id']}: {detail} (source: {case['source']})"


# --- regresje z rundy adwersarza (raport K3 §5) ----------------------------
def test_huge_integers_survive_messages_and_json() -> None:
    # CPython odmawia str() int-a > 4300 cyfr; certyfikat i to_dict tego nie mogą robić
    r = calc.isprime(2**11213 - 1)
    assert r.value is True and r.proof == "proof" and "≈" in r.certificate
    payload = calc.fibonacci(30000).to_dict()  # 6270 cyfr
    assert isinstance(payload["value"], str) and int(payload["value"][:5]) > 0
    json.dumps(payload)


def test_input_digits_raise_the_working_precision() -> None:
    # -1/e do 40 cyfr: przy 30 cyfrach roboczych sam argument byłby zaokrąglony
    r = calc.lambertw("-0.3678794411714423215955237701614608674458")
    assert abs(r.as_float() + 1) < 1e-12


def test_cancellation_next_to_a_zero_climbs_the_precision_ladder() -> None:
    import mpmath

    r = calc.besselj(0, "2.404825557695773")
    ctx = mpmath.MPContext()
    ctx.dps = 60
    want = ctx.besselj(0, ctx.mpf("2.404825557695773"))
    assert abs(r.as_float() - float(want)) <= 1e-13 * abs(float(want))


@pytest.mark.parametrize("x", ["0.001", "1e-300", "171.7"])
def test_gamma_certificate_holds_near_zero_and_beyond_float(x: str) -> None:
    import mpmath

    ctx = mpmath.MPContext()
    ctx.dps = 40
    got, want = ctx.mpf(calc.gamma(x).value), ctx.gamma(ctx.mpf(x))  # 171.7 is past float range
    assert abs(got / want - 1) < 1e-14


def test_erfc_far_tail_and_erfinv_edges() -> None:
    assert calc.erfc(30).value.endswith("e-393")
    assert calc.erfinv(1).value == "inf" and calc.erfinv(-1).value == "-inf"
    with pytest.raises(DomainError):
        calc.erfinv("1.5")


def test_closed_forms_are_exact_and_numbers_back_them() -> None:
    r = calc.polylog(2, 2)
    assert r.exact and "I" in r.value and r.approx is not None
    assert calc.polylog(2, 1).value == "pi**2/6"
    assert calc.gamma("0.5").value == "sqrt(pi)"


def test_mersenne_and_pocklington_proofs_beyond_psi13() -> None:
    assert calc.isprime(2**127 - 1).witness["kind"] == "lucas-lehmer"
    assert calc.isprime(2**4421 - 1).value is False
    r = calc.isprime(999999999999999999999999999989)  # the largest prime below 1e30
    assert r.proof == "proof" and r.witness["kind"] == "pocklington"
    assert calc.check_primality_certificate(r.witness["certificate"])
    forged = dict(r.witness["certificate"], a=1)
    assert not calc.check_primality_certificate(forged)


def test_probabilistic_primes_say_so() -> None:
    r = calc.isprime(2**256 + 297)
    assert r.value is True and r.proof == "probabilistic"
    assert any("BPSW" in n for n in r.notes)
    f = calc.factorint(3 * (2**256 + 297))
    assert f.proof == "probabilistic" and f.notes


def test_size_refusals_happen_before_the_uninterruptible_call() -> None:
    import time

    t = time.perf_counter()
    with pytest.raises(DomainError):
        calc.binomial(10**9, 10**8)
    with pytest.raises(DomainError):
        calc.fibonacci(10**10)
    assert time.perf_counter() - t < 1


def test_mod_names_its_convention() -> None:
    r = calc.mod(-7, 3)
    assert r.value == 2 and "trunc (C/Java/JS %) -1" in r.notes[0]
    assert calc.mod(-7, 3, convention="trunc").value == -1
    assert calc.mod(-7, 3, convention="euclid").value == 2


def test_crt_inconsistent_system_names_the_conflict() -> None:
    r = calc.crt([0, 1], [4, 6])
    assert r.value is None and r.witness["gcd"] == 2


def test_sqrt_mod_is_complete() -> None:
    for a, n in [(0, 1024), (4, 32), (9, 972), (1, 24), (3, 7)]:
        brute = [x for x in range(n) if (x * x - a) % n == 0]
        assert calc.sqrt_mod(a, n).value == brute


def test_every_public_entry_point_is_under_the_budget() -> None:
    from pycodemath.engine import stats

    for module in (calc, stats):
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.isfunction(obj) and name not in (
                "check_primality_certificate",
            ) and not name.isupper():
                inner = obj
                # the thin per-function wrappers delegate to the guarded ``special``
                if name in calc.SPECIAL_FUNCTIONS:
                    inner = calc.special
                assert hasattr(inner, "__pcm_time_budget__"), f"{module.__name__}.{name}"
