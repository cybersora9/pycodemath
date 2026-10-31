"""K3 — the calculator layer: descriptive statistics, distributions, A/B tests.

Strażnik modułu: ``test_ab_sample_size_matches_reference`` (widziany CZERWONY przed
implementacją, raport K3 §3).
"""

from __future__ import annotations

import pytest

from pycodemath.engine import stats


# Reference: Evan Miller's sample-size calculator, formula as published in its
# JavaScript (``num_subjects``; copy in gist mottalrd/7ddfd45d14bc7433dec2), with
# z-quantiles from SciPy 1.17.1. Values per group, rounded UP.
EVAN_MILLER = [
    # (baseline, absolute MDE, alpha, power, n per group, unrounded n)
    (0.20, 0.05, 0.05, 0.80, 1031, 1030.2192831104596),
    (0.10, 0.02, 0.05, 0.80, 3623, 3622.6163992757115),
    (0.05, 0.01, 0.05, 0.80, 7664, 7663.00027166803),
    (0.50, 0.05, 0.05, 0.80, 1568, 1567.4160061574562),
]


@pytest.mark.parametrize("p1, mde, alpha, power, n, raw", EVAN_MILLER)
def test_ab_sample_size_matches_reference(p1, mde, alpha, power, n, raw) -> None:
    r = stats.ab_sample_size(p1, mde=mde, alpha=alpha, power=power)
    assert r.value["per_group"] == n
    assert r.value["total"] == 2 * n
    assert abs(float(r.value["n_unrounded"]) - raw) < 1e-6 * raw
    # the warning is part of the RESULT, not of the documentation
    assert any("peek" in note for note in r.notes)


def test_ab_sample_size_matches_statsmodels_arcsine() -> None:
    # statsmodels 0.15.0: NormalIndPower().solve_power(proportion_effectsize(.12,.10),
    # alpha=.05, power=.8) == 3834.595739884031
    r = stats.ab_sample_size(0.10, p2=0.12, method="arcsine")
    assert r.value["per_group"] == 3835
    assert abs(float(r.value["n_unrounded"]) - 3834.595739884031) < 1e-6


# --- statystyka opisowa: dokładnie, z nazwanymi konwencjami ----------------
from fractions import Fraction  # noqa: E402

import math  # noqa: E402

from pycodemath.core.errors import DomainError  # noqa: E402


def _numacc4(as_str: bool) -> list:
    vals = ["10000000.2"] + [f"10000000.{d}" for _ in range(500) for d in ("1", "3")]
    return vals if as_str else [float(v) for v in vals]


def test_nist_numacc4_exact_and_the_one_pass_trap_is_named() -> None:
    exact = stats.std(_numacc4(True), kind="sample")
    assert exact.value == Fraction(1, 10) and exact.exact
    fl = stats.std(_numacc4(False), kind="sample")
    assert abs(fl.value - 0.1) < 1e-7 and not fl.exact
    assert any("one-pass" in n for n in fl.notes)  # the naive formula is off here
    assert fl.witness["welford_ulps"] < 2**20


def test_kind_is_required_and_refused_typed() -> None:
    with pytest.raises(DomainError, match="sample"):
        stats.variance([1, 2, 3])
    assert stats.variance([1, 2, 3, 4], kind="population").value == Fraction(5, 4)
    assert stats.variance([1, 2, 3, 4], kind="sample").value == Fraction(5, 3)


def test_float_results_are_correctly_rounded() -> None:
    xs = [0.1] * 10 + [1e-17]
    want = float(sum(map(Fraction, xs)) / len(xs))
    assert stats.mean(xs).value == want
    r = stats.std([1.0, 2.0, 4.0], kind="sample")
    v = (sum(Fraction(x) ** 2 for x in (1, 2, 4)) - Fraction(49, 3)) / 2
    lo = (Fraction(math.nextafter(r.value, 0)) + Fraction(r.value)) / 2
    hi = (Fraction(r.value) + Fraction(math.nextafter(r.value, 9))) / 2
    assert lo * lo <= v <= hi * hi


def test_overflowing_float_statistics_come_back_exact() -> None:
    r = stats.variance([1e308, -1e308, 1e308], kind="sample")
    assert r.exact and r.value == Fraction(1e308) ** 2 * Fraction(4, 3)
    assert stats.mean([1e308, 1e308]).value == 1e308


@pytest.mark.parametrize("t", range(1, 10))
def test_quantile_types_match_numpy(t: int) -> None:
    import numpy as np

    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    for q in ("0", "0.1", "0.25", "0.5", "0.9", "1"):
        r = stats.quantile(data, q, method=t)
        ref = np.quantile(np.array(data, dtype=float), float(q), method=stats.QUANTILE_METHODS[t])
        assert abs(float(r.value) - ref) < 1e-12


def test_spearman_and_pearson_refuse_constant_input() -> None:
    with pytest.raises(DomainError):
        stats.correlation([2, 2, 2], [1, 2, 3])
    assert stats.correlation([1, 2, 3], [3, 2, 1], method="spearman").value == -1


# --- rozkłady: siatka porównania z SciPy ------------------------------------
GRID = [
    ("normal", {}, ["-8", "-1.5", "0", "0.3", "2", "7"]),
    ("normal", {"mu": "3", "sigma": "0.5"}, ["1", "3", "4.2"]),
    ("t", {"df": 1}, ["-30", "-1", "0.5", "12"]),
    ("t", {"df": "4.5"}, ["-3", "0.1", "2.7"]),
    ("chi2", {"df": 3}, ["0.01", "1", "7.8", "40"]),
    ("chi2", {"df": "0.7"}, ["0.001", "0.5", "5"]),
    ("f", {"dfn": 2, "dfd": 9}, ["0.05", "1", "4.3"]),
    ("exponential", {"rate": "1.5"}, ["0.01", "1", "20"]),
]


@pytest.mark.parametrize("name, params, xs", GRID, ids=[f"{g[0]}-{i}" for i, g in enumerate(GRID)])
def test_distributions_match_scipy_on_a_grid(name, params, xs) -> None:
    st = pytest.importorskip("scipy.stats")
    frozen = {
        "normal": lambda p: st.norm(loc=float(p.get("mu", 0)), scale=float(p.get("sigma", 1))),
        "t": lambda p: st.t(float(p["df"])),
        "chi2": lambda p: st.chi2(float(p["df"])),
        "f": lambda p: st.f(float(p["dfn"]), float(p["dfd"])),
        "exponential": lambda p: st.expon(scale=1 / float(p["rate"])),
    }[name](params)
    for x in xs:
        for ours, theirs in (
            (stats.pdf(name, x, **params), frozen.pdf(float(x))),
            (stats.cdf(name, x, **params), frozen.cdf(float(x))),
            (stats.sf(name, x, **params), frozen.sf(float(x))),
        ):
            got = ours.as_float()
            assert abs(got - theirs) <= 1e-12 * abs(theirs) + 1e-300, (name, x, ours.kind, got, theirs)
        p = frozen.cdf(float(x))
        if 1e-300 < p < 1 - 1e-12:
            q = stats.ppf(name, Fraction(float(p)), **params).as_float()  # the exact binary p SciPy sees
            assert abs(q - frozen.ppf(p)) <= 1e-9 * max(1.0, abs(q)), (name, x, q)


def test_sf_keeps_the_digits_one_minus_cdf_loses() -> None:
    r = stats.sf("normal", 10)
    assert r.value.startswith("7.61985302416") and r.value.endswith("e-24")
    assert 1 - float(stats.cdf("normal", 10).as_float()) == 0.0


def test_binomial_is_exact_rational() -> None:
    assert stats.pmf("binomial", 3, n=10, p="1/2").value == Fraction(15, 128)
    c = stats.cdf("binomial", 4, n=20, p="0.3")
    assert isinstance(c.value, Fraction) and c.exact


# --- testy A/B ----------------------------------------------------------------
def test_ab_results_carry_the_peeking_and_multiplicity_warnings() -> None:
    for r in (
        stats.ab_power(0.10, 0.12, 3623),
        stats.wilson_interval(81, 263),
        stats.clopper_pearson(7, 20),
        stats.two_proportion_ztest(120, 1000, 150, 1000),
        stats.welch_ttest([1, 2, 3, 4], [2, 3, 5, 9]),
    ):
        assert stats.PEEKING_NOTE in r.notes and stats.MULTIPLE_COMPARISONS_NOTE in r.notes


def test_ab_power_is_the_inverse_of_sample_size() -> None:
    n = stats.ab_sample_size(0.10, p2=0.12).value["per_group"]
    assert float(stats.ab_power(0.10, 0.12, n).value) >= 0.8
    assert float(stats.ab_power(0.10, 0.12, n - 1).value) < 0.8


def test_ab_relative_mde() -> None:
    a = stats.ab_sample_size(0.03, mde=0.1, relative=True)
    b = stats.ab_sample_size(0.03, mde="0.003")
    assert a.value["per_group"] == b.value["per_group"] == 51487


def test_holm_and_bonferroni_match_statsmodels() -> None:
    mt = pytest.importorskip("statsmodels.stats.multitest")
    pv = [0.01, 0.04, 0.03, 0.005, 0.2]
    for m in ("holm", "bonferroni"):
        ours = stats.adjust_pvalues(pv, method=m).value
        ref = mt.multipletests(pv, method=m)[1]
        assert all(abs(a - b) < 1e-15 for a, b in zip(ours, ref))


def test_numpy_float_parameters_are_read_as_their_decimal() -> None:
    # regresja: np.float64 jest podklasą float, ale jej repr to "np.float64(...)"
    import numpy as np

    assert stats.ppf("normal", np.float64(0.975)).value == stats.ppf("normal", "0.975").value
