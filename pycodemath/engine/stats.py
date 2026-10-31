"""Calculator layer, statistics half (series FULL, module K3).

Descriptive statistics, seven distributions and the A/B-testing toolkit a developer
actually reaches for — every result a ``CalcResult`` (see ``engine.calc``) with an
independent certificate.

--- DESCRIPTIVE STATISTICS ARE EXACT -------------------------------------------------

A float is an exact dyadic rational, so the mean, variance, covariance and
correlation OF THE FLOATS YOU PASSED have exact rational values, and this module
computes them exactly (integer arithmetic over a common denominator — fast, not
``Fraction``-by-``Fraction``). For float input the value returned is that exact
statistic CORRECTLY ROUNDED to the nearest float, and for square roots (standard
deviation, correlation) the rounding is PROVEN by squaring the two neighbouring
rounding midpoints in exact arithmetic. For exact input (ints, ``Fraction``, decimal
strings such as ``"0.1"``) the value is the exact ``Fraction``.

The point is the classic trap. NIST StRD NumAcc4 — values near 1e7 that differ by
0.1 — defeats the one-pass formula ``E[x^2] - E[x]^2`` in double precision; the
result's ``notes`` say so, with the number the one-pass formula gives, whenever that
number is off. Welford's update runs as the float-native primary for float input,
and its distance from the exact value is reported in ulps in ``witness``.

Sample versus population variance is a REQUIRED keyword (``kind=``): it is the
single most common silent disagreement between tools (NumPy defaults to
population, pandas and R to sample), so no default is offered.

--- DISTRIBUTIONS ------------------------------------------------------------------

``pdf``/``pmf``, ``cdf``, ``sf`` and ``ppf`` for normal, t, chi2, F, exponential,
binomial and Poisson, evaluated in mpmath at ``dps`` digits on request. ``sf`` is a
separate function and not ``1 - cdf``: ``sf("normal", 10)`` is 7.6e-24, which
``1 - cdf`` rounds to 0 in double precision. Certificates: two precisions plus an
independent route (quadrature of the density for continuous laws, the incomplete
beta/gamma identity against the exact finite sum for discrete ones); ``ppf`` is
certified by bracketing — ``cdf(x - d) < p <= cdf(x + d)`` at the returned digits.

--- A/B TESTS ------------------------------------------------------------------------

Sample size (three named methods — Evan Miller's calculator, the pooled textbook
formula, Cohen's arcsine as in statsmodels/G*Power), power, Wilson and
Clopper-Pearson intervals, the two-proportion z test, Welch's t test and Holm /
Bonferroni adjustment. Every one of these results carries, in ``notes``, the two
warnings that matter more than any decimal: do not peek, and adjust for multiple
comparisons.

Parameters given as floats (``alpha=0.05``) are read by their shortest decimal
representation (1/20), which is what was typed; data given as floats are their
exact binary values, which is what a program holds. Both rules are stated in the
results' notes where they matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Iterable, Literal, Sequence

import mpmath

from ..core.budget import under_budget
from ..core.errors import DomainError
from .calc import CalcResult, CalcUndecided, Proof

__all__ = [
    "mean",
    "median",
    "quantile",
    "variance",
    "std",
    "covariance",
    "correlation",
    "DISTRIBUTIONS",
    "pdf",
    "pmf",
    "cdf",
    "sf",
    "ppf",
    "ab_sample_size",
    "ab_power",
    "wilson_interval",
    "clopper_pearson",
    "two_proportion_ztest",
    "welch_ttest",
    "adjust_pvalues",
    "PEEKING_NOTE",
    "MULTIPLE_COMPARISONS_NOTE",
]

Kind = Literal["sample", "population"]

PEEKING_NOTE = (
    "fixed-horizon test: fix the sample size BEFORE the experiment and look once — "
    "checking the p-value repeatedly and stopping at the first 'significant' look "
    "('peeking') inflates the false-positive rate far above alpha; use a sequential "
    "method if you must look early"
)
MULTIPLE_COMPARISONS_NOTE = (
    "several metrics, variants or segments multiply the false positives: adjust the "
    "p-values (adjust_pvalues(..., method='holm'); Holm is never less powerful than "
    "Bonferroni)"
)


# =============================================================================
# exact data
# =============================================================================
def _to_fraction(x: object, where: str) -> tuple[Fraction, bool]:
    """(exact value, was it a float)."""
    if isinstance(x, bool):
        raise DomainError(f"{where}: expected a number, got the boolean {x!r}")
    if isinstance(x, int):
        return Fraction(x), False
    if isinstance(x, Fraction):
        return x, False
    if isinstance(x, float):
        if not math.isfinite(x):
            raise DomainError(f"{where}: {x!r} is not a finite number")
        return Fraction(x), True
    if isinstance(x, str):
        try:
            return Fraction(x.strip()), False
        except (ValueError, ZeroDivisionError):
            raise DomainError(f"{where}: cannot read {x!r} as an exact number") from None
    try:
        import numpy as np

        if isinstance(x, np.integer):
            return Fraction(int(x)), False
        if isinstance(x, np.floating):
            return _to_fraction(float(x), where)
    except ImportError:  # pragma: no cover
        pass
    try:
        import sympy as sp

        if isinstance(x, sp.Rational):
            return Fraction(int(x.p), int(x.q)), False
    except ImportError:  # pragma: no cover
        pass
    if hasattr(x, "numerator") and hasattr(x, "denominator"):  # decimal.Decimal via as_integer_ratio
        try:
            return Fraction(x), False  # type: ignore[call-overload]
        except (TypeError, ValueError):
            pass
    raise DomainError(f"{where}: expected a number, got {x!r}")


@dataclass(frozen=True, slots=True)
class _Data:
    values: tuple[Fraction, ...]
    floats: bool  # at least one float in the input: output is a rounded float

    @property
    def n(self) -> int:
        return len(self.values)


def _data(xs: Iterable[object], name: str = "data", *, min_n: int = 1) -> _Data:
    if isinstance(xs, (str, bytes)):
        raise DomainError(f"{name}: expected a sequence of numbers, got a string")
    vals, any_float = [], False
    for i, x in enumerate(xs):
        v, f = _to_fraction(x, f"{name}[{i}]")
        vals.append(v)
        any_float |= f
    if len(vals) < min_n:
        raise DomainError(f"{name}: needs at least {min_n} value(s), got {len(vals)}")
    return _Data(tuple(vals), any_float)


def _scaled(values: Sequence[Fraction]) -> tuple[list[int], int]:
    """Integers A_i and a common denominator D with values[i] == A_i / D."""
    d = math.lcm(*(v.denominator for v in values)) if values else 1
    return [v.numerator * (d // v.denominator) for v in values], d


def _param(x: object, name: str) -> Fraction:
    """A PARAMETER (alpha, p, quantile level): floats by their shortest decimal repr."""
    if isinstance(x, float):
        if not math.isfinite(x):
            raise DomainError(f"{name}: {x!r} is not a finite number")
        return Fraction(repr(float(x)))  # float(): np.float64 reprs as "np.float64(...)"
    return _to_fraction(x, name)[0]


def _out(v: Fraction, floats: bool) -> "Fraction | float":
    # int / int true division is correctly rounded in CPython
    return float(v) if floats else v


def _fmt_q(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def _ulps(a: float, b: float) -> int:
    if a == b:
        return 0
    if not (math.isfinite(a) and math.isfinite(b)):  # pragma: no cover
        return 2**63
    import struct

    def key(x: float) -> int:
        i = struct.unpack("<q", struct.pack("<d", x))[0]
        return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)

    return abs(key(a) - key(b))


def _sqrt_exact_or_rounded(v: Fraction) -> "tuple[Fraction | None, float]":
    """(exact root if v is a rational square, correctly rounded float root).

    The rounding is PROVEN: with y the returned float and m-, m+ the midpoints to
    its neighbours, m-^2 <= v <= m+^2 holds in exact arithmetic.
    """
    if v < 0:
        raise CalcUndecided("sqrt of a negative exact value")  # pragma: no cover
    if v == 0:
        return Fraction(0), 0.0
    n, d = v.numerator, v.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    exact = Fraction(rn, rd) if rn * rn == n and rd * rd == d else None
    k = max(0, 120 - (n.bit_length() - d.bit_length()) // 2)
    s = math.isqrt((n << (2 * k)) // d)
    y = float(Fraction(s, 1 << k))
    for _ in range(4):
        lo = (Fraction(math.nextafter(y, 0.0)) + Fraction(y)) / 2
        hi = (Fraction(y) + Fraction(math.nextafter(y, math.inf))) / 2
        if lo * lo > v:
            y = math.nextafter(y, 0.0)
        elif hi * hi < v:
            y = math.nextafter(y, math.inf)
        else:
            return exact, y
    raise CalcUndecided("sqrt: correct rounding could not be certified")  # pragma: no cover


def _value_and_exact(v: Fraction, floats: bool) -> tuple[Any, bool]:
    """Correctly rounded float for float input — unless the exact value lies beyond
    the float range (the mean of [1e308, 1e308] does not, their variance does), in
    which case the exact rational is returned rather than inf or an OverflowError."""
    if not floats:
        return v, True
    try:
        return float(v), False
    except OverflowError:
        return v, True


_OVERFLOW_NOTE = "the value exceeds the float range, so it is returned as an exact rational"


def _overflow_notes(floats: bool, is_exact: bool) -> tuple[str, ...]:
    return (_OVERFLOW_NOTE,) if floats and is_exact else ()


# =============================================================================
# descriptive statistics
# =============================================================================
@under_budget("mean")
def mean(data: Iterable[object]) -> CalcResult:
    """Arithmetic mean — exact for exact input, correctly rounded for floats.

        >>> mean([1, 2, 3, 4]).value
        Fraction(5, 2)
        >>> mean([0.1, 0.2, 0.3]).value
        0.2
    """
    d = _data(data)
    ints, den = _scaled(d.values)
    exact = Fraction(sum(ints), den * d.n)
    value, is_exact = _value_and_exact(exact, d.floats)
    witness: dict = {"n": d.n}
    how = "exact rational sum over a common denominator"
    if d.floats and not is_exact:
        how += "; correctly rounded"
        try:
            fs = math.fsum(float(v) for v in d.values) / d.n
            witness["fsum_value"] = fs
            witness["fsum_ulps"] = _ulps(fs, value)
            how += f" (math.fsum/n is {witness['fsum_ulps']} ulp away)"
        except OverflowError:
            # the float-native route overflows where the exact one does not
            witness["fsum_value"] = "overflow"
            how += " (math.fsum overflows on this input: the sum exceeds the float range)"
    return CalcResult("mean", value, is_exact, how, "proof", witness, _overflow_notes(d.floats, is_exact))


def _hyndman_fan(x: Sequence[Fraction], p: Fraction, t: int) -> Fraction:
    """Sample quantile of type t (Hyndman & Fan 1996), exact; x sorted."""
    n = len(x)

    def at(k: int) -> Fraction:
        return x[min(max(k, 1), n) - 1]

    if t in (1, 2, 3):
        m = Fraction(-1, 2) if t == 3 else Fraction(0)
        h = n * p + m
        j = math.floor(h)
        g = h - j
        if t == 1:
            gamma = Fraction(0) if g == 0 else Fraction(1)
        elif t == 2:
            gamma = Fraction(1, 2) if g == 0 else Fraction(1)
        else:
            gamma = Fraction(0) if g == 0 and j % 2 == 0 else Fraction(1)
    else:
        alpha, beta = _HF_CONTINUOUS[t]
        h = n * p + alpha + p * (1 - alpha - beta)
        j = math.floor(h)
        gamma = h - j
    return (1 - gamma) * at(j) + gamma * at(j + 1)


_HF_CONTINUOUS = {
    4: (Fraction(0), Fraction(1)),
    5: (Fraction(1, 2), Fraction(1, 2)),
    6: (Fraction(0), Fraction(0)),
    7: (Fraction(1), Fraction(1)),
    8: (Fraction(1, 3), Fraction(1, 3)),
    9: (Fraction(3, 8), Fraction(3, 8)),
}
#: Hyndman-Fan type -> the NumPy ``method=`` name of the same definition.
QUANTILE_METHODS = {
    1: "inverted_cdf",
    2: "averaged_inverted_cdf",
    3: "closest_observation",
    4: "interpolated_inverted_cdf",
    5: "hazen",
    6: "weibull",
    7: "linear",
    8: "median_unbiased",
    9: "normal_unbiased",
}


def _method_type(method: "int | str") -> int:
    if isinstance(method, int) and not isinstance(method, bool) and method in QUANTILE_METHODS:
        return method
    for t, name in QUANTILE_METHODS.items():
        if method == name:
            return t
    raise DomainError(
        f"quantile: unknown method {method!r}; use a Hyndman-Fan type 1..9 or one of "
        + ", ".join(QUANTILE_METHODS.values())
    )


@under_budget("quantile")
def quantile(data: Iterable[object], q: object, *, method: "int | str" = 7) -> CalcResult:
    """Sample quantile at level q in [0, 1], definition NAMED (Hyndman-Fan type 1..9,
    default 7 = NumPy's ``linear`` = R's default = Excel's ``PERCENTILE.INC``).

    Exact arithmetic on the sorted data, cross-checked against ``numpy.quantile`` with
    the matching ``method`` (an independent implementation).

        >>> quantile(range(1, 11), 0.25).value
        Fraction(13, 4)
        >>> quantile(range(1, 11), 0.25, method=6).value
        Fraction(11, 4)
    """
    d = _data(data)
    t = _method_type(method)
    p = _param(q, "q")
    if not 0 <= p <= 1:
        raise DomainError(f"quantile: q must be in [0, 1], got {q!r}")
    xs = sorted(d.values)
    exact = _hyndman_fan(xs, p, t)
    value, is_exact = _value_and_exact(exact, d.floats)
    how = f"Hyndman-Fan type {t} ({QUANTILE_METHODS[t]}) in exact arithmetic"
    proof: Proof = "cross-check"
    notes: list[str] = []
    if isinstance(q, float) and Fraction(q) != p:
        notes.append(f"q={q!r} read as the decimal {_fmt_q(p)}, not as its binary float value")
    try:
        import numpy as np

        arr = np.array([float(v) for v in xs], dtype=float)
        if not np.all(np.isfinite(arr)):
            raise OverflowError
        ref = float(np.quantile(arr, float(p), method=QUANTILE_METHODS[t]))  # type: ignore[call-overload]
        mine = float(exact)
        tol = 8 * max(abs(mine), abs(ref), float(np.max(np.abs(arr)))) * 2.0**-52
        if abs(ref - mine) > tol:
            # a discontinuous type at a jump: NumPy sees the binary float of q
            alt = float(_hyndman_fan(xs, Fraction(float(p)), t))
            if abs(ref - alt) > tol:
                raise CalcUndecided(
                    f"quantile: exact value {mine!r} and numpy.quantile {ref!r} disagree"
                )
            notes.append(
                "q sits on a jump of this discontinuous definition: NumPy, which sees "
                f"the binary float of q, returns {ref!r}"
            )
        how += f"; numpy.quantile(method='{QUANTILE_METHODS[t]}') agrees"
    except (OverflowError, ImportError):
        proof = "proof" if t in (1, 3) else "cross-check"
        how += "; values exceed the float range, NumPy cross-check skipped"
    return CalcResult("quantile", value, is_exact, how, proof, {"type": t}, tuple(notes))


@under_budget("median")
def median(data: Iterable[object]) -> CalcResult:
    """Median (the mean of the two middle values for even n).

    Certificate by the DEFINITION: at most n/2 values lie strictly below it and at
    most n/2 strictly above — counted, not assumed from the sort.
    """
    d = _data(data)
    xs = sorted(d.values)
    m = _hyndman_fan(xs, Fraction(1, 2), 7)
    below = sum(1 for v in d.values if v < m)
    above = sum(1 for v in d.values if v > m)
    if 2 * below > d.n or 2 * above > d.n:
        raise CalcUndecided("median: counting check failed")  # pragma: no cover
    value, is_exact = _value_and_exact(m, d.floats)
    return CalcResult(
        "median", value, is_exact,
        f"{below} values below and {above} above, both <= n/2 = {Fraction(d.n, 2)}",
        "proof", {"below": below, "above": above},
    )


def _check_kind(kind: object, what: str) -> Kind:
    if kind not in ("sample", "population"):
        raise DomainError(
            f"{what}: kind must be 'sample' (divide by n-1) or 'population' (divide "
            f"by n) — it is required because tools disagree on the default "
            f"(NumPy: population; pandas, R, Excel STDEV: sample)"
        )
    return kind  # type: ignore[return-value]


def _exact_var(d: _Data, kind: Kind) -> Fraction:
    ints, den = _scaled(d.values)
    n, s1 = d.n, sum(ints)
    s2 = sum(a * a for a in ints)
    ss = Fraction(n * s2 - s1 * s1, n * den * den)  # sum of squared deviations
    return ss / (n - 1 if kind == "sample" else n)


def _welford(xs: Sequence[float], kind: Kind) -> float:
    m, s = 0.0, 0.0
    for i, x in enumerate(xs, 1):
        delta = x - m
        m += delta / i
        s += delta * (x - m)
    return s / (len(xs) - 1 if kind == "sample" else len(xs))


def _naive_var(xs: Sequence[float], kind: Kind) -> float:
    n = len(xs)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    return (s2 - s1 * s1 / n) / (n - 1 if kind == "sample" else n)


def _variance_result(name: str, data: Iterable[object], kind: object, root: bool) -> CalcResult:
    k = _check_kind(kind, name)
    d = _data(data, min_n=2 if k == "sample" else 1)
    var = _exact_var(d, k)
    witness: dict = {"n": d.n, "kind": k}
    notes: list[str] = []
    if root:
        try:
            exact_root, rounded = _sqrt_exact_or_rounded(var)
        except OverflowError:
            c = _ctx(30)
            text = c.nstr(c.sqrt(c.mpf(var.numerator) / var.denominator), 17)
            return CalcResult(
                name, text, False,
                f"sqrt of the exact {k} variance at 30 digits (beyond the float range)",
                "cross-check", witness, (_OVERFLOW_NOTE.replace("an exact rational", "a 17-digit decimal string"),),
            )
        if exact_root is not None and not d.floats:
            value: Any = exact_root
            is_exact = True
            how = f"exact: the {k} variance {_fmt_q(var)} is the square of a rational"
        else:
            value, is_exact = rounded, False
            how = (
                f"sqrt of the exact {k} variance, correctly rounded (both rounding "
                f"midpoints squared in exact arithmetic bracket it)"
            )
            if not d.floats:
                witness["variance"] = _fmt_q(var)
    else:
        value, is_exact = _value_and_exact(var, d.floats)
        how = f"exact rational {k} variance over a common denominator"
        if d.floats and not is_exact:
            how += ", correctly rounded"
        notes.extend(_overflow_notes(d.floats, is_exact))
    if d.floats and not is_exact:
        fl = [float(v) for v in d.values]
        try:
            w = _welford(fl, k)
            naive = _naive_var(fl, k)
            ref = float(value) if root else float(var)
            if root:
                w, naive = math.sqrt(max(w, 0.0)), math.sqrt(max(naive, 0.0))
        except OverflowError:
            w = naive = math.inf
            ref = float(value)
        witness["welford"] = w if math.isfinite(w) else "overflow"
        if math.isfinite(w):
            witness["welford_ulps"] = _ulps(w, ref)
        if ref != 0 and math.isfinite(naive) and abs(naive - ref) > 1e-6 * abs(ref) or not math.isfinite(naive):
            notes.append(
                f"the one-pass formula E[x^2]-E[x]^2 gives {naive!r} here in double "
                f"precision (catastrophic cancellation); the exact value is {ref!r}"
            )
    return CalcResult(name, value, is_exact, how, "proof", witness, tuple(notes))


@under_budget("variance")
def variance(data: Iterable[object], *, kind: "Kind | None" = None) -> CalcResult:
    """Variance; ``kind="sample"`` divides by n-1, ``"population"`` by n (required).

        >>> variance([10000001, 10000003, 10000002], kind="sample").value
        Fraction(1, 1)
    """
    return _variance_result("variance", data, kind, root=False)


@under_budget("std")
def std(data: Iterable[object], *, kind: "Kind | None" = None) -> CalcResult:
    """Standard deviation (``kind`` required); exact when the variance is a rational
    square, otherwise correctly rounded with a proof of the rounding."""
    return _variance_result("std", data, kind, root=True)


def _paired(x: Iterable[object], y: Iterable[object], min_n: int) -> tuple[_Data, _Data]:
    dx, dy = _data(x, "x", min_n=min_n), _data(y, "y", min_n=min_n)
    if dx.n != dy.n:
        raise DomainError(f"x and y must have the same length, got {dx.n} and {dy.n}")
    return dx, dy


def _co_sums(dx: _Data, dy: _Data) -> tuple[Fraction, Fraction, Fraction]:
    """(Sxx, Syy, Sxy): sums of centred squares/products, exact."""
    ax, ex = _scaled(dx.values)
    ay, ey = _scaled(dy.values)
    n = dx.n
    sx, sy = sum(ax), sum(ay)
    sxx = Fraction(n * sum(a * a for a in ax) - sx * sx, n * ex * ex)
    syy = Fraction(n * sum(b * b for b in ay) - sy * sy, n * ey * ey)
    sxy = Fraction(n * sum(a * b for a, b in zip(ax, ay)) - sx * sy, n * ex * ey)
    return sxx, syy, sxy


@under_budget("covariance")
def covariance(x: Iterable[object], y: Iterable[object], *, kind: "Kind | None" = None) -> CalcResult:
    """Covariance of paired data (``kind`` required), exact."""
    k = _check_kind(kind, "covariance")
    dx, dy = _paired(x, y, 2 if k == "sample" else 1)
    _sxx, _syy, sxy = _co_sums(dx, dy)
    cov = sxy / (dx.n - 1 if k == "sample" else dx.n)
    # independent formulation: the definition sum (x - mx)(y - my) in Fractions
    mx, my = sum(dx.values, Fraction(0)) / dx.n, sum(dy.values, Fraction(0)) / dy.n
    if sum((a - mx) * (b - my) for a, b in zip(dx.values, dy.values)) != sxy:
        raise CalcUndecided("covariance: the two exact formulations disagree")  # pragma: no cover
    value, is_exact = _value_and_exact(cov, dx.floats or dy.floats)
    return CalcResult(
        "covariance", value, is_exact,
        f"exact {k} covariance; the integer-sum formula equals the centred definition",
        "proof", {"n": dx.n, "kind": k}, _overflow_notes(dx.floats or dy.floats, is_exact),
    )


def _average_ranks(values: Sequence[Fraction]) -> list[Fraction]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [Fraction(0)] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = Fraction(i + j + 2, 2)  # mean of the 1-based positions i+1 .. j+1
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    return ranks


@under_budget("correlation")
def correlation(
    x: Iterable[object], y: Iterable[object], *, method: Literal["pearson", "spearman"] = "pearson"
) -> CalcResult:
    """Pearson (linear) or Spearman (rank, ties get average ranks) correlation.

    r^2 = Sxy^2 / (Sxx Syy) is computed EXACTLY; r is exact when that is a rational
    square, otherwise correctly rounded with a proof. Constant input is refused —
    r is 0/0 there, and returning NaN (NumPy) or 0 would both be guesses.
    """
    if method not in ("pearson", "spearman"):
        raise DomainError(f"correlation: method must be 'pearson' or 'spearman', got {method!r}")
    dx, dy = _paired(x, y, 2)
    floats = dx.floats or dy.floats
    if method == "spearman":
        dx = _Data(tuple(_average_ranks(dx.values)), False)
        dy = _Data(tuple(_average_ranks(dy.values)), False)
    sxx, syy, sxy = _co_sums(dx, dy)
    if sxx == 0 or syy == 0:
        which = "x" if sxx == 0 else "y"
        raise DomainError(f"correlation: {which} is constant, so r = 0/0 is undefined")
    r2 = sxy * sxy / (sxx * syy)
    exact_root, rounded = _sqrt_exact_or_rounded(r2)
    sign = -1 if sxy < 0 else 1
    exact_input = not floats or method == "spearman"
    if exact_root is not None and exact_input:
        value: Any = sign * exact_root
        is_exact = True
    else:
        value, is_exact = sign * rounded, False
    how = f"exact r^2 = {_fmt_q(r2)}" if len(_fmt_q(r2)) < 60 else "exact r^2"
    how += "; r " + ("exact" if is_exact else "correctly rounded (proven by midpoint squares)")
    if method == "spearman":
        ties = len(set(dx.values)) < dx.n or len(set(dy.values)) < dy.n
        if not ties:
            n = dx.n
            d2 = sum((a - b) ** 2 for a, b in zip(dx.values, dy.values))
            alt = 1 - Fraction(6) * d2 / (n * (n * n - 1))
            if alt * alt != r2 or (alt < 0) != (sxy < 0):
                raise CalcUndecided("spearman: the no-ties formula disagrees")  # pragma: no cover
            how += "; equals 1 - 6 sum d^2 / (n(n^2-1)) (no ties)"
        else:
            how += "; ties ranked by their average position"
    return CalcResult("correlation", value, is_exact, how, "proof", {"method": method, "r_squared": _fmt_q(r2)})


# =============================================================================
# distributions
# =============================================================================
def _ctx(dps: int) -> Any:
    c = mpmath.MPContext()
    c.dps = dps
    return c


def _mp(ctx: Any, x: object, name: str) -> Any:
    v = _param(x, name)
    return ctx.mpf(v.numerator) / v.denominator


@dataclass(frozen=True)
class _Dist:
    name: str
    params: tuple[str, ...]
    discrete: bool
    validate: Callable[[dict], None]
    support: Callable[[Any, dict], tuple[Any, Any]]
    density: Callable[[Any, Any, dict], Any]
    cdf: Callable[[Any, Any, dict], Any]
    sf: Callable[[Any, Any, dict], Any]


def _positive(*names: str) -> Callable[[dict], None]:
    def check(p: dict) -> None:
        for n in names:
            if not p[n] > 0:
                raise DomainError(f"{n} must be > 0, got {p[n]}")

    return check


def _bin_validate(p: dict) -> None:
    if p["n"] != int(p["n"]) or p["n"] < 0:
        raise DomainError(f"binomial: n must be a non-negative integer, got {p['n']}")
    if not 0 <= p["p"] <= 1:
        raise DomainError(f"binomial: p must be in [0, 1], got {p['p']}")


def _normal_pdf(c: Any, x: Any, p: dict) -> Any:
    z = (x - p["mu"]) / p["sigma"]
    return c.exp(-z * z / 2) / (p["sigma"] * c.sqrt(2 * c.pi))


def _t_cdf(c: Any, x: Any, p: dict) -> Any:
    v = p["df"]
    tail = c.betainc(v / 2, c.mpf(1) / 2, 0, v / (v + x * x), regularized=True) / 2
    return tail if x < 0 else 1 - tail


def _t_sf(c: Any, x: Any, p: dict) -> Any:
    return _t_cdf(c, -x, p)


def _f_cdf(c: Any, x: Any, p: dict) -> Any:
    if x <= 0:
        return c.zero
    a, b = p["dfn"], p["dfd"]
    return c.betainc(a / 2, b / 2, 0, a * x / (a * x + b), regularized=True)


def _f_sf(c: Any, x: Any, p: dict) -> Any:
    if x <= 0:
        return c.one
    a, b = p["dfn"], p["dfd"]
    return c.betainc(b / 2, a / 2, 0, b / (b + a * x), regularized=True)


def _f_pdf(c: Any, x: Any, p: dict) -> Any:
    if x <= 0:
        return c.zero
    a, b = p["dfn"], p["dfd"]
    return c.sqrt((a * x) ** a * b**b / (a * x + b) ** (a + b)) / (x * c.beta(a / 2, b / 2))


def _bin_pmf(c: Any, k: Any, p: dict) -> Any:
    n = int(p["n"])
    if k != c.floor(k) or k < 0 or k > n:
        return c.zero
    k = int(k)
    return c.binomial(n, k) * p["p"] ** k * (1 - p["p"]) ** (n - k)


def _bin_cdf(c: Any, x: Any, p: dict) -> Any:
    n, k = int(p["n"]), int(c.floor(x))
    if k < 0:
        return c.zero
    if k >= n:
        return c.one
    return c.betainc(n - k, k + 1, 0, 1 - p["p"], regularized=True)


def _bin_sf(c: Any, x: Any, p: dict) -> Any:
    n, k = int(p["n"]), int(c.floor(x))
    if k < 0:
        return c.one
    if k >= n:
        return c.zero
    return c.betainc(k + 1, n - k, 0, p["p"], regularized=True)


def _poi_pmf(c: Any, k: Any, p: dict) -> Any:
    if k != c.floor(k) or k < 0:
        return c.zero
    lam = p["lam"]
    return c.exp(-lam + k * c.log(lam) - c.loggamma(k + 1))


def _poi_cdf(c: Any, x: Any, p: dict) -> Any:
    k = c.floor(x)
    if k < 0:
        return c.zero
    return c.gammainc(k + 1, p["lam"], c.inf, regularized=True)


def _poi_sf(c: Any, x: Any, p: dict) -> Any:
    k = c.floor(x)
    if k < 0:
        return c.one
    return c.gammainc(k + 1, 0, p["lam"], regularized=True)


DISTRIBUTIONS: dict[str, _Dist] = {
    "normal": _Dist(
        "normal", ("mu", "sigma"), False, _positive("sigma"),
        lambda c, p: (c.ninf, c.inf),
        _normal_pdf,
        lambda c, x, p: c.erfc(-(x - p["mu"]) / (p["sigma"] * c.sqrt(2))) / 2,
        lambda c, x, p: c.erfc((x - p["mu"]) / (p["sigma"] * c.sqrt(2))) / 2,
    ),
    "t": _Dist(
        "t", ("df",), False, _positive("df"),
        lambda c, p: (c.ninf, c.inf),
        lambda c, x, p: c.gamma((p["df"] + 1) / 2) / (c.sqrt(p["df"] * c.pi) * c.gamma(p["df"] / 2))
        * (1 + x * x / p["df"]) ** (-(p["df"] + 1) / 2),
        _t_cdf, _t_sf,
    ),
    "chi2": _Dist(
        "chi2", ("df",), False, _positive("df"),
        lambda c, p: (c.zero, c.inf),
        lambda c, x, p: c.zero if x <= 0 else
        c.exp((p["df"] / 2 - 1) * c.log(x) - x / 2 - (p["df"] / 2) * c.log(2) - c.loggamma(p["df"] / 2)),
        lambda c, x, p: c.zero if x <= 0 else c.gammainc(p["df"] / 2, 0, x / 2, regularized=True),
        lambda c, x, p: c.one if x <= 0 else c.gammainc(p["df"] / 2, x / 2, c.inf, regularized=True),
    ),
    "f": _Dist(
        "f", ("dfn", "dfd"), False, _positive("dfn", "dfd"),
        lambda c, p: (c.zero, c.inf), _f_pdf, _f_cdf, _f_sf,
    ),
    "exponential": _Dist(
        "exponential", ("rate",), False, _positive("rate"),
        lambda c, p: (c.zero, c.inf),
        lambda c, x, p: c.zero if x < 0 else p["rate"] * c.exp(-p["rate"] * x),
        lambda c, x, p: c.zero if x <= 0 else -c.expm1(-p["rate"] * x),
        lambda c, x, p: c.one if x <= 0 else c.exp(-p["rate"] * x),
    ),
    "binomial": _Dist(
        "binomial", ("n", "p"), True, _bin_validate,
        lambda c, p: (c.zero, c.mpf(p["n"])), _bin_pmf, _bin_cdf, _bin_sf,
    ),
    "poisson": _Dist(
        "poisson", ("lam",), True, _positive("lam"),
        lambda c, p: (c.zero, c.inf), _poi_pmf, _poi_cdf, _poi_sf,
    ),
}

_DEFAULTS = {"mu": 0, "sigma": 1, "rate": 1}


def _dist(name: str, params: dict) -> tuple[_Dist, dict[str, Fraction]]:
    if name not in DISTRIBUTIONS:
        raise DomainError(
            f"unknown distribution {name!r}; known: {', '.join(DISTRIBUTIONS)}"
        )
    dist = DISTRIBUTIONS[name]
    unknown = set(params) - set(dist.params)
    if unknown:
        raise DomainError(f"{name}: unknown parameter(s) {sorted(unknown)}; it takes {dist.params}")
    exact: dict[str, Fraction] = {}
    for p in dist.params:
        if p in params:
            exact[p] = _param(params[p], f"{name}.{p}")
        elif p in _DEFAULTS:
            exact[p] = Fraction(_DEFAULTS[p])
        else:
            raise DomainError(f"{name}: parameter {p!r} is required")
    dist.validate(dict(exact))
    return dist, exact


def _mp_params(ctx: Any, exact: dict[str, Fraction]) -> dict[str, Any]:
    return {k: ctx.mpf(v.numerator) / v.denominator for k, v in exact.items()}


def _finite_or_inf(ctx: Any, v: Any) -> str:
    if v == ctx.inf:
        return "inf"
    if v == ctx.ninf:
        return "-inf"
    return ctx.nstr(v, 15)


def _close(ctx: Any, a: Any, b: Any, digits: int) -> bool:
    scale = max(abs(a), abs(b))
    return scale == 0 or abs(a - b) <= ctx.mpf(10) ** (-digits) * scale


#: Laws whose density behaves like t^(a-1) at t = 0 (singular for a < 1), with a.
_POWER_AT_ZERO: dict[str, Callable[[dict], Any]] = {
    "chi2": lambda pm: pm["df"] / 2,
    "f": lambda pm: pm["dfn"] / 2,
}


def _quad_check(ctx: Any, dist: _Dist, pm: dict, x: Any, which: str) -> Any:
    lo, hi = dist.support(ctx, pm)
    if dist.name in _POWER_AT_ZERO and x > 0:
        # t = s^(1/a) turns int_0^x t^(a-1) g(t) dt into (1/a) int_0^(x^a) g(s^(1/a)) ds:
        # the endpoint singularity that defeats tanh-sinh for a < 1 disappears
        a = _POWER_AT_ZERO[dist.name](pm)
        cut = min(x, ctx.one)

        def smooth(s: Any) -> Any:
            t = s ** (1 / a)
            return dist.density(ctx, t, pm) * t / (a * s) if s > 0 else ctx.zero

        if which == "cdf":
            near = ctx.quad(smooth, [0, cut**a])
            return near + (ctx.quad(lambda t: dist.density(ctx, t, pm), [cut, x]) if x > cut else ctx.zero)
        if x < 1:
            return ctx.quad(smooth, [x**a, 1]) + ctx.quad(lambda t: dist.density(ctx, t, pm), [ctx.one, ctx.inf])
    if which == "cdf":
        a, b = lo, x
    else:
        a, b = x, hi
    if a >= b:
        return ctx.zero
    pts = [a, b]
    for s in (pm.get("mu"), ctx.zero):
        if s is not None and a < s < b:
            pts = sorted(set([a, s, b]))
            break
    return ctx.quad(lambda t: dist.density(ctx, t, pm), pts)


def _discrete_sum(ctx: Any, dist: _Dist, pm: dict, x: Any, which: str) -> "Any | None":
    """The cdf/sf as a finite sum of the pmf — the independent route for discrete
    laws. None when the sum would be too long to be a cheap check."""
    k = int(ctx.floor(x))
    if dist.name == "binomial":
        n = int(pm["n"])
        terms = range(0, min(k, n) + 1) if which == "cdf" else range(max(k + 1, 0), n + 1)
    else:
        if which == "sf":
            return None if k > 20000 else 1 - sum((dist.density(ctx, j, pm) for j in range(0, k + 1)), ctx.zero)
        terms = range(0, k + 1)
    if len(terms) > 20000:
        return None
    return sum((dist.density(ctx, j, pm) for j in terms), ctx.zero)


def _evaluate(fn_name: str, dist: _Dist, exact: dict, x: object, dps: int) -> tuple[str, str, Proof]:
    values = []
    for prec in (dps + 15, 2 * dps + 30):
        c = _ctx(prec)
        pm = _mp_params(c, exact)
        xv = _mp(c, x, "x")
        fn = {"pdf": dist.density, "cdf": dist.cdf, "sf": dist.sf}[fn_name]
        values.append((c, pm, xv, fn(c, xv, pm)))
    (c1, _p1, _x1, v1), (c2, pm, xv, v2) = values
    if not _close(c2, v1, v2, dps + 2):
        raise CalcUndecided(
            f"{dist.name}.{fn_name}: values at {dps + 15} and {2 * dps + 30} digits disagree"
        )
    text = "precision doubling agrees"
    proof: Proof = "cross-check"
    if fn_name in ("cdf", "sf"):
        if dist.discrete:
            alt = _discrete_sum(c2, dist, pm, xv, fn_name)
            if alt is not None:
                if not _close(c2, alt, v2, dps + 2):
                    raise CalcUndecided(f"{dist.name}.{fn_name}: incomplete beta/gamma and the pmf sum disagree")
                text += "; the incomplete beta/gamma identity equals the finite sum of the pmf"
        else:
            c3 = _ctx(dps + 15)
            alt = _quad_check(c3, dist, _mp_params(c3, exact), _mp(c3, x, "x"), fn_name)
            if not _close(c3, alt, v2, dps + 1):
                raise CalcUndecided(
                    f"{dist.name}.{fn_name}: closed form and quadrature of the density disagree"
                )
            text += "; quadrature of the density agrees"
    elif not dist.discrete and fn_name == "pdf":
        text += "; closed-form density"
    return c2.nstr(v2, dps, strip_zeros=False), text, proof


_Fn = Literal["pdf", "cdf", "sf"]


def _dist_result(fn_name: _Fn, dist_name: str, x: object, dps: object, params: dict) -> CalcResult:
    from .calc import _check_dps

    d = _check_dps(dps)
    dist, exact = _dist(dist_name, params)
    _param(x, "x")
    # exact binomial pmf/cdf when p is rational: Fractions, no rounding at all
    if dist.name == "binomial":
        xv = _param(x, "x")
        n, p = int(exact["n"]), exact["p"]
        if fn_name == "pdf" or n <= 5000:
            exact_v = _binomial_exact(fn_name, n, p, xv)
            c = _ctx(d + 15)
            mpv = {"pdf": dist.density, "cdf": dist.cdf, "sf": dist.sf}[fn_name](
                c, c.mpf(xv.numerator) / xv.denominator, _mp_params(c, exact))
            if not _close(c, mpv, c.mpf(exact_v.numerator) / exact_v.denominator, d + 5):
                raise CalcUndecided(f"binomial.{fn_name}: exact sum and incomplete beta disagree")
            return CalcResult(
                f"{dist.name}.{'pmf' if fn_name == 'pdf' else fn_name}", exact_v, True,
                "exact rational sum of C(n,k) p^k (1-p)^(n-k); the incomplete beta identity agrees",
                "proof", {"params": {k: _fmt_q(v) for k, v in exact.items()}},
                approx=c.nstr(c.mpf(exact_v.numerator) / exact_v.denominator, d, strip_zeros=False),
            )
    value, text, proof = _evaluate(fn_name, dist, exact, x, d)
    label = "pmf" if dist.discrete and fn_name == "pdf" else fn_name
    return CalcResult(
        f"{dist.name}.{label}", value, False, text, proof,
        {"params": {k: _fmt_q(v) for k, v in exact.items()}, "dps": d},
    )


def _binomial_exact(fn_name: str, n: int, p: Fraction, x: Fraction) -> Fraction:
    def pmf(k: int) -> Fraction:
        return math.comb(n, k) * p**k * (1 - p) ** (n - k)

    k = math.floor(x)
    if fn_name == "pdf":
        return pmf(int(x)) if x == k and 0 <= k <= n else Fraction(0)
    if fn_name == "cdf":
        return sum((pmf(j) for j in range(0, min(k, n) + 1)), Fraction(0)) if k >= 0 else Fraction(0)
    return sum((pmf(j) for j in range(max(k + 1, 0), n + 1)), Fraction(0))


@under_budget("pdf")
def pdf(dist: str, x: object, *, dps: object = 15, **params: object) -> CalcResult:
    """Probability density of a continuous law at x (``pmf`` for discrete ones).

        >>> pdf("normal", 0, dps=20).value
        '0.39894228040143267794'
    """
    if dist in DISTRIBUTIONS and DISTRIBUTIONS[dist].discrete:
        raise DomainError(f"pdf: {dist} is discrete — use pmf")
    return _dist_result("pdf", dist, x, dps, params)


@under_budget("pmf")
def pmf(dist: str, k: object, *, dps: object = 15, **params: object) -> CalcResult:
    """Probability mass P(X = k) of a discrete law (binomial: exact rational).

        >>> pmf("binomial", 3, n=10, p="1/2").value
        Fraction(15, 128)
    """
    if dist in DISTRIBUTIONS and not DISTRIBUTIONS[dist].discrete:
        raise DomainError(f"pmf: {dist} is continuous — use pdf")
    return _dist_result("pdf", dist, k, dps, params)


@under_budget("cdf")
def cdf(dist: str, x: object, *, dps: object = 15, **params: object) -> CalcResult:
    """P(X <= x).

        >>> cdf("normal", "1.96").value
        '0.975002104851780'
    """
    return _dist_result("cdf", dist, x, dps, params)


@under_budget("sf")
def sf(dist: str, x: object, *, dps: object = 15, **params: object) -> CalcResult:
    """P(X > x), computed directly — never as 1 - cdf, which loses every digit in
    the upper tail (``sf("normal", 10)`` is 7.6e-24; ``1 - cdf`` gives 0)."""
    return _dist_result("sf", dist, x, dps, params)


def _invert(c: Any, dist: _Dist, pm: dict, p: Any) -> Any:
    """x with cdf(x) = p, solved on sf for upper-tail p (where cdf has no digits)."""
    lo_s, hi_s = dist.support(c, pm)
    upper = p > 0.5
    target = 1 - p if upper else p

    def g(x: Any) -> Any:
        return (target - dist.sf(c, x, pm)) if upper else (dist.cdf(c, x, pm) - target)

    # bracket: start at the centre and double outwards
    centre = pm.get("mu", c.zero) if lo_s == c.ninf else c.zero
    step = pm.get("sigma", c.one)
    a = centre if lo_s == c.ninf else lo_s
    b = a + step
    while g(b) < 0:
        a, b = b, b + 2 * (b - a)
    if lo_s == c.ninf:
        a = centre - step
        while g(a) > 0:
            a = a - 2 * (b - a)
    # bisection to a narrow bracket, then the secant-type Illinois method
    for _ in range(60):
        m = (a + b) / 2
        if g(m) < 0:
            a = m
        else:
            b = m
    if a == b:
        return a
    try:
        return c.findroot(g, (a, b), solver="illinois", tol=c.mpf(10) ** (-c.dps + 5), maxsteps=200)
    except ValueError:
        while b - a > c.mpf(10) ** (-c.dps + 5) * max(abs(a), 1):
            m = (a + b) / 2
            if g(m) < 0:
                a = m
            else:
                b = m
        return (a + b) / 2


@under_budget("ppf")
def ppf(dist: str, q: object, *, dps: object = 15, **params: object) -> CalcResult:
    """Quantile function: the x with P(X <= x) = q (discrete: the least k with
    P(X <= k) >= q).

    Certificate: the true quantile lies within one unit of the last returned digit —
    ``cdf(x - d) < q <= cdf(x + d)`` evaluated at doubled precision.

        >>> ppf("normal", "0.975").value
        '1.95996398454005'
    """
    from .calc import _check_dps

    d = _check_dps(dps)
    dist_, exact = _dist(dist, params)
    qv = _param(q, "q")
    if not 0 <= qv <= 1:
        raise DomainError(f"ppf: q must be in [0, 1], got {q!r}")
    c = _ctx(2 * d + 30)
    pm = _mp_params(c, exact)
    qm = c.mpf(qv.numerator) / qv.denominator
    lo_s, hi_s = dist_.support(c, pm)
    label = f"{dist_.name}.ppf"
    pstr = {k: _fmt_q(v) for k, v in exact.items()}
    if qv == 0 or qv == 1:
        edge = lo_s if qv == 0 else hi_s
        if dist_.discrete and qv == 0:
            edge = lo_s
        return CalcResult(label, _finite_or_inf(c, edge), True, "q = 0 or 1: the edge of the support", "proof", {"params": pstr})
    if dist_.discrete:
        return _discrete_ppf(c, dist_, pm, qm, label, pstr)
    x = _invert(c, dist_, pm, qm)
    text = c.nstr(x, d, strip_zeros=False)
    xr = c.mpf(text)
    delta = c.mpf(10) ** (-d + 1) * max(abs(xr), c.mpf(10) ** (-d))
    below, above = dist_.cdf(c, xr - delta, pm), dist_.cdf(c, xr + delta, pm)
    if qm > 0.5:
        below, above = 1 - dist_.sf(c, xr - delta, pm), 1 - dist_.sf(c, xr + delta, pm)
    if not (below < qm <= above):
        raise CalcUndecided(f"{label}: the bracket check failed at {text}")
    return CalcResult(
        label, text, False,
        f"bracket: cdf(x - d) < q <= cdf(x + d) with d one unit in the last digit "
        f"({d} significant digits), at {2 * d + 30}-digit precision",
        "cross-check", {"params": pstr, "dps": d},
    )


def _discrete_ppf(c: Any, dist: _Dist, pm: dict, qm: Any, label: str, pstr: dict) -> CalcResult:
    k = 0
    # galloping search then bisection on the integer line
    hi = 1
    while dist.cdf(c, c.mpf(hi), pm) < qm:
        hi *= 2
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if dist.cdf(c, c.mpf(mid), pm) < qm:
            lo = mid + 1
        else:
            hi = mid
    k = lo
    below = dist.cdf(c, c.mpf(k - 1), pm) if k > 0 else c.zero
    if not (below < qm <= dist.cdf(c, c.mpf(k), pm)):
        raise CalcUndecided(f"{label}: the bracket check failed")  # pragma: no cover
    return CalcResult(
        label, k, True, f"cdf({k - 1}) < q <= cdf({k}) (the least such k)", "cross-check",
        {"params": pstr},
    )


# =============================================================================
# A/B tests
# =============================================================================
_AB_DPS = 40
AbMethod = Literal["evanmiller", "pooled", "arcsine"]


def _z(c: Any, p: Any) -> Any:
    """Standard normal quantile, at the context's precision."""
    return c.sqrt(2) * c.erfinv(2 * p - 1)


def _phi(c: Any, z: Any) -> Any:
    return c.erfc(-z / c.sqrt(2)) / 2


def _ab_inputs(p1: object, p2: object, mde: object, relative: bool, alpha: object, power: object) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    a = _param(p1, "p1")
    if not 0 < a < 1:
        raise DomainError(f"p1 must be strictly between 0 and 1, got {p1!r}")
    if (p2 is None) == (mde is None):
        raise DomainError("give exactly one of p2 (the variant's rate) or mde (the effect)")
    if p2 is not None:
        b = _param(p2, "p2")
    else:
        m = _param(mde, "mde")
        b = a * (1 + m) if relative else a + m
    if not 0 < b < 1:
        raise DomainError(f"the variant rate must be strictly between 0 and 1, got {_fmt_q(b)}")
    if a == b:
        raise DomainError("p1 == p2: there is no effect to detect (n would be infinite)")
    al, pw = _param(alpha, "alpha"), _param(power, "power")
    if not 0 < al < 1 or not 0 < pw < 1:
        raise DomainError("alpha and power must be strictly between 0 and 1")
    return a, b, al, pw


def _ab_n_and_power(c: Any, method: str, p1: Any, p2: Any, za: Any) -> tuple[Callable[[Any], Any], Any, Any]:
    """(power(n), sd1-ish, sd2-ish) for the method — power uses the same normal
    approximation the method's sample-size formula inverts."""
    delta = abs(p2 - p1)
    if method == "evanmiller":
        sd1 = c.sqrt(2 * p1 * (1 - p1))
        sd2 = c.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        return (lambda n: _phi(c, (delta * c.sqrt(n) - za * sd1) / sd2)), sd1, sd2
    if method == "pooled":
        pb = (p1 + p2) / 2
        sd1 = c.sqrt(2 * pb * (1 - pb))
        sd2 = c.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        return (lambda n: _phi(c, (delta * c.sqrt(n) - za * sd1) / sd2)), sd1, sd2
    h = abs(2 * c.asin(c.sqrt(p2)) - 2 * c.asin(c.sqrt(p1)))
    return (lambda n: _phi(c, h * c.sqrt(n / 2) - za) + _phi(c, -h * c.sqrt(n / 2) - za)), h, None


_AB_METHOD_TEXT = {
    "evanmiller": "Evan Miller's calculator: n = (z_a sqrt(2p(1-p)) + z_b sqrt(p1q1+p2q2))^2 / delta^2",
    "pooled": "pooled-variance textbook formula (Fleiss, Levin & Paik 2003, no continuity correction)",
    "arcsine": "Cohen's arcsine effect h = 2asin(sqrt p2) - 2asin(sqrt p1), both tails (statsmodels NormalIndPower)",
}


@under_budget("ab_sample_size")
def ab_sample_size(
    p1: object,
    *,
    p2: object = None,
    mde: object = None,
    relative: bool = False,
    alpha: object = "0.05",
    power: object = "0.8",
    two_sided: bool = True,
    method: AbMethod = "evanmiller",
) -> CalcResult:
    """Visitors per group to detect a change from conversion rate p1 to p2 (or by
    ``mde``, absolute by default, ``relative=True`` for a lift such as 0.1 = +10%).

    Certificate: the closed form is evaluated at 40 digits, and the power curve of the
    SAME approximation is checked on both sides of the answer — power(n) >= target
    and power(n - 1) < target — so the rounding up is the least sufficient n.

        >>> r = ab_sample_size(0.10, p2=0.12)
        >>> r.value["per_group"], r.value["total"]
        (3623, 7246)
    """
    if method not in _AB_METHOD_TEXT:
        raise DomainError(f"ab_sample_size: method must be one of {sorted(_AB_METHOD_TEXT)}")
    a, b, al, pw = _ab_inputs(p1, p2, mde, relative, alpha, power)
    c = _ctx(_AB_DPS)
    fa, fb = c.mpf(a.numerator) / a.denominator, c.mpf(b.numerator) / b.denominator
    alpha_m = c.mpf(al.numerator) / al.denominator
    za = _z(c, 1 - alpha_m / 2 if two_sided else 1 - alpha_m)
    zb = _z(c, c.mpf(pw.numerator) / pw.denominator)
    power_fn, s1, s2 = _ab_n_and_power(c, method, fa, fb, za)
    target = c.mpf(pw.numerator) / pw.denominator
    if method == "arcsine":
        raw = 2 * ((za + zb) / s1) ** 2
        if two_sided:  # the far tail adds a sliver of power: solve exactly
            raw = c.findroot(lambda n: power_fn(n) - target, raw)
    else:
        raw = (za * s1 + zb * s2) ** 2 / (fb - fa) ** 2
    n = int(c.ceil(raw - c.mpf(10) ** -25))
    if not (power_fn(n) >= target - c.mpf(10) ** -30 and (n == 1 or power_fn(n - 1) < target)):
        raise CalcUndecided("ab_sample_size: the power check around n failed")  # pragma: no cover
    notes = (PEEKING_NOTE, MULTIPLE_COMPARISONS_NOTE, "method: " + _AB_METHOD_TEXT[method])
    return CalcResult(
        "ab_sample_size",
        {
            "per_group": n,
            "total": 2 * n,
            "n_unrounded": c.nstr(raw, 15, strip_zeros=False),
            "p1": _fmt_q(a),
            "p2": _fmt_q(b),
            "alpha": _fmt_q(al),
            "power": _fmt_q(pw),
            "two_sided": two_sided,
            "method": method,
        },
        True,
        f"power({n}) = {c.nstr(power_fn(n), 6)} >= {_fmt_q(pw)} > power({n - 1}) = "
        f"{c.nstr(power_fn(n - 1), 6) if n > 1 else '-'} under the method's own normal approximation",
        "proof",
        {"power_at_n": c.nstr(power_fn(n), 15, strip_zeros=False)},
        notes,
    )


@under_budget("ab_power")
def ab_power(
    p1: object,
    p2: object,
    n: object,
    *,
    alpha: object = "0.05",
    two_sided: bool = True,
    method: AbMethod = "evanmiller",
) -> CalcResult:
    """Power of the two-proportion test with n visitors PER GROUP. The inverse of
    ``ab_sample_size`` under the same approximation, and checked as such."""
    from .calc import _as_int

    if method not in _AB_METHOD_TEXT:
        raise DomainError(f"ab_power: method must be one of {sorted(_AB_METHOD_TEXT)}")
    nn = _as_int(n, "n")
    if nn < 1:
        raise DomainError("ab_power: n must be >= 1")
    a, b, al, _pw = _ab_inputs(p1, p2, None, False, alpha, "0.5")
    c = _ctx(_AB_DPS)
    fa, fb = c.mpf(a.numerator) / a.denominator, c.mpf(b.numerator) / b.denominator
    alpha_m = c.mpf(al.numerator) / al.denominator
    za = _z(c, 1 - alpha_m / 2 if two_sided else 1 - alpha_m)
    power_fn, _s1, _s2 = _ab_n_and_power(c, method, fa, fb, za)
    pw = power_fn(nn)
    # inverse check: the sample size needed for this power is n itself
    if method != "arcsine" and 0 < pw < 1:
        back = (za * _s1 + _z(c, pw) * _s2) ** 2 / (fb - fa) ** 2
        if not _close(c, back, c.mpf(nn), 20):
            raise CalcUndecided("ab_power: inverse check failed")  # pragma: no cover
    return CalcResult(
        "ab_power", c.nstr(pw, 15, strip_zeros=False), False,
        "inverse check: the sample-size formula at this power returns n",
        "cross-check", {"method": method},
        (PEEKING_NOTE, MULTIPLE_COMPARISONS_NOTE, "method: " + _AB_METHOD_TEXT[method]),
    )


def _counts(x: object, n: object) -> tuple[int, int]:
    from .calc import _as_int

    xi, ni = _as_int(x, "successes"), _as_int(n, "trials")
    if ni < 1 or not 0 <= xi <= ni:
        raise DomainError(f"need 0 <= successes <= trials and trials >= 1, got {xi}/{ni}")
    return xi, ni


def _confidence(conf: object) -> Fraction:
    cf = _param(conf, "confidence")
    if not 0 < cf < 1:
        raise DomainError(f"confidence must be strictly between 0 and 1, got {conf!r}")
    return cf


@under_budget("wilson_interval")
def wilson_interval(successes: object, trials: object, *, confidence: object = "0.95") -> CalcResult:
    """Wilson score interval for a proportion (better coverage than the textbook
    Wald interval, never outside [0, 1]).

    Certificate: both endpoints satisfy the score equation (p_hat - p)^2 =
    z^2 p (1 - p) / n at 40 digits, and they bracket p_hat.
    """
    x, n = _counts(successes, trials)
    cf = _confidence(confidence)
    c = _ctx(_AB_DPS)
    z = _z(c, 1 - (1 - c.mpf(cf.numerator) / cf.denominator) / 2)
    ph = c.mpf(x) / n
    denom = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / denom
    half = z / denom * c.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    # at x = 0 (x = n) the score equation's root is exactly 0 (1); the formula would
    # land a rounding error away from it, on the wrong side of p_hat
    lo = c.zero if x == 0 else max(c.zero, centre - half)
    hi = c.one if x == n else min(c.one, centre + half)
    for e in (lo, hi):
        resid = (ph - e) ** 2 - z * z * e * (1 - e) / n
        if abs(resid) > c.mpf(10) ** -30:
            raise CalcUndecided("wilson_interval: an endpoint fails the score equation")  # pragma: no cover
    if not lo <= ph <= hi:
        raise CalcUndecided("wilson_interval: the interval misses p_hat")  # pragma: no cover
    return CalcResult(
        "wilson_interval", [c.nstr(lo, 15, strip_zeros=False), c.nstr(hi, 15, strip_zeros=False)], False,
        "both endpoints solve the score equation (p_hat-p)^2 = z^2 p(1-p)/n at 40 digits",
        "cross-check", {"p_hat": _fmt_q(Fraction(x, n)), "z": c.nstr(z, 15, strip_zeros=False)},
        (PEEKING_NOTE, MULTIPLE_COMPARISONS_NOTE),
    )


def _binom_tail_ge(c: Any, n: int, x: int, p: Any) -> Any:
    """P(X >= x) for X ~ Bin(n, p) via the incomplete beta function."""
    if x <= 0:
        return c.one
    return c.betainc(x, n - x + 1, 0, p, regularized=True)


@under_budget("clopper_pearson")
def clopper_pearson(successes: object, trials: object, *, confidence: object = "0.95") -> CalcResult:
    """Clopper-Pearson ("exact") interval: the p at which the binomial tails equal
    (1 - confidence)/2.

    Certificate: at each endpoint the tail probability equals alpha/2 — checked with
    the FINITE binomial sum (independent of the incomplete beta used to solve) when
    n <= 5000, otherwise at doubled precision.
    """
    x, n = _counts(successes, trials)
    cf = _confidence(confidence)
    c = _ctx(_AB_DPS)
    half_alpha = (1 - c.mpf(cf.numerator) / cf.denominator) / 2

    def solve(g: Callable[[Any], Any]) -> Any:
        a, b = c.zero, c.one
        for _ in range(140):
            m = (a + b) / 2
            if g(m) < 0:
                a = m
            else:
                b = m
        return (a + b) / 2

    lo = c.zero if x == 0 else solve(lambda p: _binom_tail_ge(c, n, x, p) - half_alpha)
    hi = c.one if x == n else solve(lambda p: half_alpha - (1 - _binom_tail_ge(c, n, x + 1, p)))
    if n <= 5000:
        def tail_sum(p: Any, ks: range) -> Any:
            return sum((c.binomial(n, k) * p**k * (1 - p) ** (n - k) for k in ks), c.zero)

        checks = []
        if x > 0:
            checks.append(tail_sum(lo, range(x, n + 1)))
        if x < n:
            checks.append(tail_sum(hi, range(0, x + 1)))
        if any(not _close(c, v, half_alpha, 25) for v in checks):
            raise CalcUndecided("clopper_pearson: the binomial tail at an endpoint is not alpha/2")
        how = "the binomial tail sum at each endpoint equals alpha/2 (finite sum, independent of the incomplete beta)"
    else:
        how = "endpoints solve I_p(x, n-x+1) = alpha/2 by bisection at 40 digits"
    return CalcResult(
        "clopper_pearson", [c.nstr(lo, 15, strip_zeros=False), c.nstr(hi, 15, strip_zeros=False)], False, how, "cross-check",
        {"p_hat": _fmt_q(Fraction(x, n))}, (PEEKING_NOTE, MULTIPLE_COMPARISONS_NOTE),
    )


Alternative = Literal["two-sided", "larger", "smaller"]


def _p_from_z(c: Any, z: Any, alternative: str) -> Any:
    if alternative == "two-sided":
        return c.erfc(abs(z) / c.sqrt(2))
    if alternative == "larger":
        return c.erfc(z / c.sqrt(2)) / 2
    return c.erfc(-z / c.sqrt(2)) / 2


def _check_alternative(alternative: object) -> str:
    if alternative not in ("two-sided", "larger", "smaller"):
        raise DomainError("alternative must be 'two-sided', 'larger' or 'smaller'")
    return str(alternative)


@under_budget("two_proportion_ztest")
def two_proportion_ztest(
    successes1: object, trials1: object, successes2: object, trials2: object,
    *, alternative: Alternative = "two-sided",
) -> CalcResult:
    """Pooled two-proportion z test (H0: p1 == p2); ``larger`` means H1: p1 > p2.

    z^2 is exact rational; the p-value is erfc at 40 digits, cross-checked by
    quadrature of the normal density.
    """
    alt = _check_alternative(alternative)
    x1, n1 = _counts(successes1, trials1)
    x2, n2 = _counts(successes2, trials2)
    pool = Fraction(x1 + x2, n1 + n2)
    var = pool * (1 - pool) * (Fraction(1, n1) + Fraction(1, n2))
    if var == 0:
        raise DomainError("two_proportion_ztest: all trials succeeded or all failed — the z statistic is 0/0")
    diff = Fraction(x1, n1) - Fraction(x2, n2)
    z2 = diff * diff / var
    c = _ctx(_AB_DPS)
    z = c.sqrt(c.mpf(z2.numerator) / z2.denominator) * (1 if diff >= 0 else -1)
    pval = _p_from_z(c, z, alt)
    q = c.quad(lambda t: c.exp(-t * t / 2) / c.sqrt(2 * c.pi), [abs(z), c.inf])
    alt_p = 2 * q if alt == "two-sided" else (q if (alt == "larger") == (z >= 0) else 1 - q)
    if not _close(c, alt_p, pval, 20):
        raise CalcUndecided("two_proportion_ztest: erfc and quadrature disagree")  # pragma: no cover
    return CalcResult(
        "two_proportion_ztest", {"z": c.nstr(z, 15, strip_zeros=False), "p_value": c.nstr(pval, 15, strip_zeros=False)}, False,
        "z^2 is exact rational; the p-value from erfc equals the quadrature of the normal density",
        "cross-check", {"z_squared": _fmt_q(z2), "pooled_p": _fmt_q(pool), "alternative": alt},
        (PEEKING_NOTE, MULTIPLE_COMPARISONS_NOTE),
    )


@under_budget("welch_ttest")
def welch_ttest(a: Iterable[object], b: Iterable[object], *, alternative: Alternative = "two-sided") -> CalcResult:
    """Welch's unequal-variance t test (H0: equal means; ``larger``: mean(a) > mean(b)).

    The statistic t^2 and the Welch-Satterthwaite degrees of freedom are EXACT
    rationals; the p-value comes from the t distribution's incomplete-beta cdf,
    cross-checked by quadrature of its density.
    """
    alt = _check_alternative(alternative)
    da, db = _data(a, "a", min_n=2), _data(b, "b", min_n=2)
    va, vb = _exact_var(da, "sample") / da.n, _exact_var(db, "sample") / db.n
    se2 = va + vb
    if se2 == 0:
        raise DomainError("welch_ttest: both samples are constant — t is undefined")
    ma, mb = sum(da.values, Fraction(0)) / da.n, sum(db.values, Fraction(0)) / db.n
    diff = ma - mb
    t2 = diff * diff / se2
    df = se2 * se2 / (va * va / (da.n - 1) + vb * vb / (db.n - 1))
    c = _ctx(_AB_DPS)
    t = c.sqrt(c.mpf(t2.numerator) / t2.denominator) * (1 if diff >= 0 else -1)
    pm = {"df": c.mpf(df.numerator) / df.denominator}
    dist = DISTRIBUTIONS["t"]
    upper = dist.sf(c, abs(t), pm)
    if alt == "two-sided":
        pval = 2 * upper
    elif alt == "larger":
        pval = dist.sf(c, t, pm)
    else:
        pval = dist.cdf(c, t, pm)
    q = c.quad(lambda s: dist.density(c, s, pm), [abs(t), c.inf])
    if not _close(c, q, upper, 20):
        raise CalcUndecided("welch_ttest: incomplete beta and quadrature disagree")
    return CalcResult(
        "welch_ttest",
        {"t": c.nstr(t, 15, strip_zeros=False), "df": c.nstr(pm["df"], 15, strip_zeros=False), "p_value": c.nstr(pval, 15, strip_zeros=False)},
        False,
        "t^2 and the Welch-Satterthwaite df are exact rationals; the t tail from the "
        "incomplete beta equals the quadrature of the t density",
        "cross-check", {"t_squared": _fmt_q(t2), "df": _fmt_q(df), "alternative": alt},
        (PEEKING_NOTE, MULTIPLE_COMPARISONS_NOTE),
    )


@under_budget("adjust_pvalues")
def adjust_pvalues(pvalues: Sequence[object], *, method: Literal["holm", "bonferroni"] = "holm") -> CalcResult:
    """Family-wise error control for m tests: Bonferroni (min(1, m p)) or Holm's
    step-down (uniformly more powerful, same guarantee). Values in input order.

    Certificate (Holm): the step-down procedure, RUN at each adjusted level, rejects
    that hypothesis — the definition, executed, rather than the closed formula again.
    """
    if method not in ("holm", "bonferroni"):
        raise DomainError("adjust_pvalues: method must be 'holm' or 'bonferroni'")
    d = _data(pvalues, "pvalues")
    ps = list(d.values)
    if any(not 0 <= p <= 1 for p in ps):
        raise DomainError("adjust_pvalues: p-values must lie in [0, 1]")
    m = len(ps)
    if method == "bonferroni":
        adj = [min(Fraction(1), m * p) for p in ps]
        how = "min(1, m p) in exact arithmetic"
    else:
        order = sorted(range(m), key=ps.__getitem__)
        adj = [Fraction(0)] * m
        running = Fraction(0)
        for rank, i in enumerate(order):
            running = max(running, min(Fraction(1), (m - rank) * ps[i]))
            adj[i] = running
        for i in range(m):
            if not _holm_rejects(ps, adj[i], i):
                raise CalcUndecided("adjust_pvalues: step-down check failed")  # pragma: no cover
        how = "Holm step-down; running the procedure at each adjusted level rejects that hypothesis"
    value = [_out(v, d.floats) for v in adj]
    return CalcResult(
        "adjust_pvalues", value, not d.floats, how, "proof", {"method": method, "m": m},
        (PEEKING_NOTE,),
    )


def _holm_rejects(ps: Sequence[Fraction], alpha: Fraction, target: int) -> bool:
    m = len(ps)
    for rank, i in enumerate(sorted(range(m), key=ps.__getitem__)):
        if ps[i] * (m - rank) > alpha:
            return False
        if i == target:
            return True
    return False  # pragma: no cover
