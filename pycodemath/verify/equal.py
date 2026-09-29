"""``check_equal`` — is ``a == b`` an identity? Proof, counterexample, or neither.

THE ORDER: SAMPLE FIRST, THEN SIMPLIFY. The verdicts are exactly the ones the
plan asks for — VERIFIED only from a symbolic proof, REFUTED only from a point
confirmed at two precisions, UNDECIDED otherwise — but the numeric phase runs
before the symbolic one, and that is a measured choice, not a style one. Over 15
typical pairs (textbook identities and their one-character corruptions), cold:

    ``simplify(a - b)``            3.3 - 29.5 ms per pair (88 ms on first use)
    24 points x 2 sides ``evalf``   0.9 -  5.2 ms per pair, at 30 digits

Sampling is 5-10x cheaper, it is bounded by construction (a fixed number of
points), and it is the only phase that can REFUTE. ``simplify`` is neither
bounded nor able to refute: a wrong claim whose difference ``simplify`` chews on
for the whole budget would otherwise be reported UNDECIDED, when a counterexample
was one ``evalf`` away. Putting the cheap, bounded phase first costs a true
identity 1-5 ms and saves a false one the entire symbolic phase.

WHAT "EQUAL" MEANS HERE — three decisions a caller has to know about:

* Values are SymPy's PRINCIPAL BRANCH, complex allowed. ``sqrt(x^2)`` vs ``x``
  is refuted at ``x = -1`` (1 vs -1); ``log(x^2)`` vs ``2*log(x)`` is refuted at
  ``x = -1`` too (0 vs 2*pi*i) — which is the point: that rewrite is the classic
  derivation error, valid only for ``x > 0``, and a verifier that stayed quiet
  about it would be useless for checking a model's algebra. The detail carries
  both values, so a complex one is visible rather than implied.
* Equality holds WHERE BOTH SIDES ARE DEFINED. ``(x^2-1)/(x-1)`` vs ``x+1`` is
  VERIFIED: they differ only at ``x = 1``, where the left side has no value. A
  point at which either side is undefined (a pole, ``0/0``) or evaluates with no
  reliable digits is skipped, never counted as agreement or as disagreement.
* Decimal literals mean the decimal WRITTEN: ``0.1 + 0.2`` equals ``0.3``.
  Every ``Float`` is rationalised to the shortest decimal it prints as before
  anything else runs — otherwise binary rounding decides the verdict, and a
  verifier of hand-written algebra would refute ``0.1 + 0.2 = 0.3``. Text is
  read that way from the start (``_exact_decimals``): ``2^0.5`` is ``sqrt(2)``,
  not the 15-digit Float the parser would evaluate it to.

THE BUDGET, AND AN ENCLOSING ``time_budget``. On its own, ``check_equal`` runs
each phase under its own ``time_budget`` and a budget that runs out becomes an
UNDECIDED verdict with ``method="time-budget"``, never an exception. Inside a
caller's ``with time_budget(...)`` it arms NOTHING of its own, because a tighter
budget nested inside another one poisons the outer when it fires (measured: an
outer ``time_budget(10)`` block refused at 0.26 s after an inner
``time_budget(0.2)`` expired and was handled; ``core.budget`` keeps the tighter
deadline in the shared slot and re-injects every 50 ms). So there the enclosing
allowance governs the symbolic phase and, when it expires, refuses the enclosing
block exactly as it always does; ``budget`` then bounds only the sampling loop,
checked between points.
"""

from __future__ import annotations

import io
import math
import random
import re
import threading
import tokenize
import time
from typing import Iterable, Sequence

import mpmath
import sympy as sp

from ..core import budget as _budget
from ..core.budget import DEFAULT_TIME_BUDGET, time_budget
from ..core.errors import DomainError, ParseError, PycodemathError, TimeBudgetError
from ..core.ir import Expr
from ..frontend.parser import _MAX_RESULT_BITS, parse
from .verdict import Verdict, VerdictStatus

__all__ = ["check_equal"]

#: Working precision of the sampling phase, in decimal digits.
_DPS = 30
#: Precision a suspected counterexample is re-evaluated at before it is reported.
#: Double the working one, so noise at 30 digits (which shrinks by ~30 orders of
#: magnitude) and a real difference (which does not move) cannot be confused.
_RECHECK_DPS = 60
#: Relative difference above which two sampled values count as different. Ten
#: digits of margin under the 30 being computed.
_TOL = mpmath.mpf("1e-20")
#: How closely the difference must reproduce itself at ``_RECHECK_DPS`` for the
#: point to be reported. A genuine difference agrees with itself to ~20 digits;
#: six is the bar, so a difference computed with little accuracy is not reported.
_REPRODUCE = mpmath.mpf("1e-6")

#: Default number of sample points.
SAMPLES = 32

#: The first points tried, in order — so the first failing point, which is the
#: one reported, is the most readable one available. 0 and 1 come first because
#: "at x = 0: 0 != 2" is the counterexample a human checks in their head; they are
#: too special to be the ONLY evidence of agreement, which is what the random
#: points after them are for. Ten entries: coprime with the per-variable stride 3,
#: so the first ten variables of a point never share a value (``x - y`` vs
#: ``y - x`` agrees at every point where ``x == y``).
_NICE: tuple[sp.Rational, ...] = tuple(
    sp.Rational(v)
    for v in ("0", "1", "-1", "2", "-2", "3", "1/2", "-3/2", "5", "-7/3")
)
_STRIDE = 3

#: Denominators for the random points: small primes, so a random point is never
#: an integer or a half — the places where ``floor``, ``sin(pi*x)`` and friends
#: agree with things they are not equal to.
_DENOMINATORS = (7, 11, 13, 17, 19, 23, 29)
#: Ranges for the random points (a value is drawn in [-bound, bound]). Mostly
#: moderate, some wider; capped at 16 because ``exp(exp(x))`` at 16 already has
#: 3.86 million digits before the decimal point (measured: ``5.8e+3859188``).
_BOUNDS = (4, 4, 4, 1, 16)


_DECIMAL = re.compile(r"(?:\d[\d_]*)?\.?[\d_]*(?:[eE][+-]?\d[\d_]*)?")


def _exact_decimals(text: str) -> str:
    """``text`` with every decimal literal spelled as the exact fraction it
    denotes: ``2^0.5`` -> ``2^(1/2)``, ``1.5e3`` -> ``(1500)``.

    Rationalising the parsed expression is too late for a decimal EXPONENT: the
    parser evaluates ``2^0.5`` to a 15-digit Float before anything here sees it,
    and that Float is no longer the decimal written. Measured (V7 finding):
    ``check_equal("2^0.5", "sqrt(2)")`` and ``("sqrt(5^15)", "5^7.5")`` were
    REFUTED, the 60-digit re-check "confirming" a difference in the 16th digit.
    Same tokenizer the parser runs on, so only NUMBER tokens are touched (``x1e5``
    stays a name); integers, hex and imaginary literals are left as written.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return text  # the parser gives this text its own, proper refusal
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    out, at = [], 0
    for tok in tokens:
        s = tok.string
        if (
            tok.type != tokenize.NUMBER
            or not any(c in s for c in ".eE")
            or not _DECIMAL.fullmatch(s)
        ):
            continue
        begin = starts[tok.start[0] - 1] + tok.start[1]
        end = starts[tok.end[0] - 1] + tok.end[1]
        out.append(text[at:begin])
        out.append(f"({sp.Rational(s.replace('_', ''))})")
        at = end
    return "".join(out) + text[at:]


def _parse_written(text: str) -> sp.Basic:
    """``parse``, with decimal literals read as the decimals written (see
    ``_exact_decimals``). Should the exact spelling be refused where the decimal
    one is not — the parser's cost guard sees ``2^1e10`` as a 10-billion-bit
    integer, while ``2^1e10`` with a Float exponent is a Float — the text is
    parsed as written, and the Float rationalised afterwards as before V8 (or,
    when too large to read exactly, refused: ``_as_written``)."""
    exact = _exact_decimals(text)
    if exact != text:
        try:
            return parse(exact).sy
        except PycodemathError:
            pass
    return parse(text).sy


def _as_written(sy: sp.Expr, where: str) -> sp.Expr:
    """``sy`` with every Float rationalised: a decimal literal means the decimal
    written (see the module docstring).

    A Float too large (or too small) for the parser to build exactly is refused
    first. ``nsimplify`` runs before any budget is armed, and on ``2^1e10`` — the
    exact spelling ``2^10000000000`` refused by the parser's cost guard, the text
    as written a Float of ~10^(3·10^9) — its ``mpmath.identify`` never returned
    (V8 finding; measured 2^k: 0.09 s at k = 10^5, 0.73 s at 3·10^5, 4.7 s at
    10^6). The line is the parser's own: a number whose magnitude needs more than
    ``_MAX_RESULT_BITS`` bits would not be built from its exact spelling either.
    """
    if not sy.has(sp.Float):
        return sy
    for fl in sy.atoms(sp.Float):
        _sign, man, exp, bc = fl._mpf_
        if man and abs(exp + bc) > _MAX_RESULT_BITS:
            raise ParseError(
                f"{where}: the number {sp.sstr(fl, full_prec=False)} is too costly "
                f"to read as the decimal written (about 2^{exp + bc}, beyond "
                f"{_MAX_RESULT_BITS} bits)"
            )
    return sp.nsimplify(sy, rational=True)


def _to_expr(value: "Expr | str | int | float", side: str) -> sp.Expr:
    if isinstance(value, Expr):
        sy = value.sy
    elif isinstance(value, str):
        sy = _parse_written(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        sy = sp.sympify(value)
    else:
        raise DomainError(
            f"check_equal: {side} side must be an expression or text, "
            f"not {type(value).__name__}"
        )
    if not isinstance(sy, sp.Expr):
        raise DomainError(f"check_equal: {side} side is not a scalar expression")
    return _as_written(sy, f"check_equal: {side} side")


def _variables(
    a: sp.Expr, b: sp.Expr, vars: "Iterable[str] | None"
) -> tuple[sp.Symbol, ...]:
    free = a.free_symbols | b.free_symbols
    if vars is None:
        return tuple(sorted(free, key=lambda s: s.name))  # type: ignore[attr-defined]
    if isinstance(vars, str):
        names = vars.replace(",", " ").split()
    else:
        names = list(vars)
    if not all(isinstance(n, str) and n for n in names):
        raise DomainError("check_equal: vars must be variable names")
    chosen = tuple(sp.Symbol(n) for n in dict.fromkeys(names))
    missing = sorted(s.name for s in free - set(chosen))  # type: ignore[attr-defined]
    if missing:
        raise DomainError(
            f"check_equal: {', '.join(missing)} not listed in vars — a symbol "
            f"that is not sampled cannot be evaluated"
        )
    return chosen


def _check_budget(budget: "float | None") -> float:
    if budget is None:
        return DEFAULT_TIME_BUDGET
    if not isinstance(budget, (int, float)) or isinstance(budget, bool):
        raise DomainError(f"check_equal: budget must be a number, not {budget!r}")
    if math.isnan(budget) or budget <= 0:
        raise DomainError(f"check_equal: budget must be positive, got {budget!r}")
    return float(budget)


def _points(
    syms: Sequence[sp.Symbol], samples: int, seed: int
) -> list[dict[sp.Symbol, sp.Rational]]:
    """Deterministic sample points: the readable ones first, then seeded random."""
    if not syms:
        return [{}]  # a claim with no variables has exactly one point to look at
    nice = min(len(_NICE), samples // 2)
    points = [
        {s: _NICE[(i + _STRIDE * j) % len(_NICE)] for j, s in enumerate(syms)}
        for i in range(nice)
    ]
    rng = random.Random(seed)
    for _ in range(samples - nice):
        point = {}
        for s in syms:
            d = rng.choice(_DENOMINATORS)
            bound = rng.choice(_BOUNDS) * d
            n = 0
            while n == 0:
                n = rng.randint(-bound, bound)
            point[s] = sp.Rational(n, d)
        points.append(point)
    return points


def _value(sy: sp.Expr, point: dict, dps: int) -> "mpmath.mpc | None":
    """The value of ``sy`` at ``point`` to ``dps`` digits — or ``None`` when it has
    none there, or none that can be trusted.

    ``evalf`` does not raise where precision is lost: it returns a ``Float`` with
    fewer bits than asked for, and the digits it prints are not digits. Measured:
    ``(x^2-1)/(x-1)`` at ``x = 1`` evaluates to ``0.e+6`` with 1 bit of precision,
    and ``sin(x)^2 + cos(x)^2 - 1`` at ``x = 7/3`` to ``-0.e-165``, also 1 bit.
    Read as numbers, the first is a value where there is none and the second a
    nonzero where there is a zero; both are refused here instead.

    The usual cause is an EXACT zero reached by cancellation (``(x+1)^2 - (x-1)^2``
    at ``x = 0``), which no finite precision can certify. Sample points are exact
    rationals, so before giving up the point is substituted exactly and SymPy's
    own arithmetic gets the chance to produce that zero — for a rational function it
    always does. Without this step the most readable counterexample of all, the one
    at ``x = 0``, was skipped and ``x = 1`` reported instead.
    """
    value = _evaluate(sy, point, dps, exact=False)
    if value is None:
        value = _evaluate(sy, point, dps, exact=True)
    return value


def _evaluate(
    sy: sp.Expr, point: dict, dps: int, *, exact: bool
) -> "mpmath.mpc | None":
    try:
        raw = sy.xreplace(point).evalf(dps) if exact else sy.evalf(dps, subs=point)
        re, im = raw.as_real_imag()
    except Exception:  # noqa: BLE001 — a point SymPy cannot evaluate is skipped
        return None
    need = int(dps * 3.32) // 2
    parts = []
    low = []
    for part in (re, im):
        if not part.is_Number or part in (sp.nan, sp.oo, -sp.oo, sp.zoo):
            return None
        low.append(isinstance(part, sp.Float) and part._prec < need)
        parts.append(part)
    if all(low):
        return None
    if any(low):
        noise, full = (1, 0) if low[1] else (0, 1)
        if not _negligible(parts[noise], parts[full], dps):
            return None
        parts[noise] = sp.Integer(0)
    # ``mpmath.mpf``/``mpc`` round to the GLOBAL precision (15 digits unless a
    # caller raised it), so without ``workdps`` a value computed at 30 or 60 digits
    # arrived here with 53 bits. Measured before V5: ``check_equal("pi",
    # "3.141592653589793238")`` was UNDECIDED, both sides shown as the same
    # double, although they differ by 4.6e-19 — 1.5e-19 relative, over ``_TOL``.
    # The arithmetic on the converted values may stay at the global precision: a
    # difference of two exact mpfs is correct to 53 bits RELATIVE TO ITSELF, which
    # is all ``_differs`` and the re-check compare.
    with mpmath.workdps(dps):
        return mpmath.mpc(*(mpmath.mpf(sp.Float(p, dps)) for p in parts))


def _negligible(noise: sp.Float, full: sp.Number, dps: int) -> bool:
    """Is ``noise`` — a part ``evalf`` returned with almost no bits — below
    ``10**-dps`` of ``full``, the other part, even at its own uncertainty?

    SymPy's accuracy for a complex value is relative to the WHOLE number, not to
    each part. A real root written with complex radicals (casus irreducibilis:
    Cardano on a cubic with three real roots) comes back as ``re + 0.e-37*I``:
    ``re`` with the 103 bits asked for, the imaginary part as a 1-bit Float of
    magnitude 2**-123 — the evaluation's own statement that it is zero to 37 digits
    of a number of size ~5. Refusing the value for that part's missing bits
    (before V8) left ``certify_solve`` UNDECIDED "a listed root has no value" on
    ``x^3 - 8x^2 - 3x + 92`` with all 3 of 3 roots proved. Not a threshold on the
    absolute value: the bound is the part's magnitude plus its own rounding
    uncertainty, measured against the other part, which must carry full precision.
    A zero mantissa says nothing about the magnitude and is never negligible.
    """
    _, man, exp, _ = noise._mpf_
    if man == 0 or full == 0:
        return False
    with mpmath.workdps(dps + 10):
        bound = (mpmath.mpf(int(man)) + 1) * mpmath.mpf(2) ** int(exp)
        scale = abs(mpmath.mpf(sp.Float(full, dps + 10)))
        return bool(bound <= scale * mpmath.mpf(10) ** (-dps))


def _undefined_as_written(sy: sp.Expr, point: dict) -> bool:
    """Does EXACT substitution leave ``sy`` without a value at ``point`` (``nan``,
    ``zoo``)? Then the point has none, whatever ``evalf`` says.

    ``evalf(subs=)`` does not know. At a 0/0 point reached through cancellation it
    returns a tiny number and claims full precision for it. Measured (V5 fuzz):
    ``sin(x-2)/(x-2)`` at ``x = 2`` is ``nan`` exactly, and ``evalf`` gives 8.7e-19
    at 30 digits (103 bits claimed) and 2.8e-14 at 60 (203 bits);
    ``sin(sqrt(x+1))*cos(sqrt(x+1))/sqrt(x+1)`` at ``x = -1`` gives 3.05e-5 at 50,
    60 and 70 digits alike — reproducible, so a second precision does not expose
    it, and ``certify_diff`` refuted the engine's CORRECT derivative of
    ``sin(sqrt(x+1))^2`` there (the claim: "3.05e-5", finite differences: 1).
    Only called once a value is missing or about to become a counterexample:
    exact substitution can be expensive (see ``certify._num``).
    """
    try:
        return bool(sy.xreplace(point).has(sp.nan, sp.zoo))
    except Exception:  # noqa: BLE001 — no verdict about definedness, so: not known undefined
        return False


def _recheck(sy: sp.Expr, point: dict) -> "mpmath.mpc | None":
    """The value at ``_RECHECK_DPS``, exact substitution first — the re-check exists
    to rule noise out, and an exact zero from cancellation is the commonest noise.
    A point where exact substitution gives no value has none (``_undefined_as_written``)."""
    value = _evaluate(sy, point, _RECHECK_DPS, exact=True)
    if value is None and not _undefined_as_written(sy, point):
        value = _evaluate(sy, point, _RECHECK_DPS, exact=False)
    return value


def _differs(va: "mpmath.mpc", vb: "mpmath.mpc") -> bool:
    scale = max(abs(va), abs(vb))
    return bool(abs(va - vb) > _TOL * scale)


def _show(v: "mpmath.mpc", digits: int = 10) -> str:
    """``1.5``, ``6.283185307*I``, ``-0.5 + 0.8660254038*I`` — SymPy's spelling."""
    re, im = mpmath.nstr(v.real, digits), mpmath.nstr(abs(v.imag), digits)
    if v.imag == 0:
        return re
    if v.real == 0:
        return f"{'-' if v.imag < 0 else ''}{im}*I"
    return f"{re} {'-' if v.imag < 0 else '+'} {im}*I"


def _show_apart(va: "mpmath.mpc", vb: "mpmath.mpc") -> tuple[str, str]:
    """Both values with as many digits as it takes for them to READ different.

    Ten digits by default; a refutation by a difference in the 19th digit
    (``pi`` vs ``3.141592653589793238``) otherwise printed two identical numbers
    and looked like a verifier contradicting itself. Capped at the 60 digits the
    re-check is computed with.
    """
    for digits in (10, 20, 30, 45, 60):
        sa, sb = _show(va, digits), _show(vb, digits)
        if sa != sb:
            break
    return sa, sb


def _format_point(point: dict) -> dict[str, str]:
    return {s.name: str(v) for s, v in point.items()}


def _where(point: dict) -> str:
    if not point:
        return "the two sides differ"
    return "at " + ", ".join(f"{s.name} = {v}" for s, v in point.items())


def _caller_budget_active() -> bool:
    """Is this thread already inside a ``time_budget``? (See the module docstring.)

    Read-only look at ``core.budget``'s table — the one fact this module needs from
    it and that module has no public accessor for.
    """
    with _budget._LOCK:
        return threading.get_ident() in _budget._ARMED


class _Sampling:
    """The numeric phase's running tally — also the evidence an UNDECIDED quotes."""

    def __init__(self) -> None:
        self.agreed = 0
        self.skipped = 0
        self.unconfirmed = 0
        self.tried = 0
        self.refutation: "Verdict | None" = None

    def run(
        self, a: sp.Expr, b: sp.Expr, points: list, deadline: float
    ) -> None:
        for point in points:
            if time.perf_counter() >= deadline:
                return
            self.tried += 1
            va, vb = _value(a, point, _DPS), _value(b, point, _DPS)
            if va is None or vb is None:
                self.skipped += 1
                continue
            if not _differs(va, vb):
                self.agreed += 1
                continue
            ra, rb = _recheck(a, point), _recheck(b, point)
            if ra is not None and rb is not None and not _differs(ra, rb):
                # The 30-digit difference was noise: 60 digits, with the point
                # substituted exactly, see the same value on both sides.
                self.agreed += 1
                continue
            if (
                ra is None
                or rb is None
                or abs((va - vb) - (ra - rb)) > _REPRODUCE * abs(ra - rb)
            ):
                # Different at 30 digits and at 60, but not by the same amount: a
                # difference computed with too little accuracy to report. It is
                # neither agreement nor a counterexample.
                self.unconfirmed += 1
                continue
            self.refutation = Verdict(
                VerdictStatus.REFUTED,
                "numeric-sampling",
                _format_point(point),
                "{}: left = {}, right = {}".format(_where(point), *_show_apart(ra, rb)),
            )
            return

    def evidence(self) -> str:
        text = f"agree numerically at {self.agreed} of {self.tried} sample points"
        extra = []
        if self.skipped:
            extra.append(f"{self.skipped} not evaluable")
        if self.unconfirmed:
            extra.append(f"{self.unconfirmed} inconclusive")
        return text + (f" ({', '.join(extra)})" if extra else "")


def _proved(diff: sp.Expr) -> bool:
    """Does ``diff`` simplify to zero? Guarded: runs under the active budget."""
    reduced = Expr(diff).simplify().sy
    return bool(reduced == 0 or reduced.is_zero is True)


def check_equal(
    a: "Expr | str | int | float",
    b: "Expr | str | int | float",
    *,
    vars: "Iterable[str] | str | None" = None,
    budget: "float | None" = None,
    samples: int = SAMPLES,
    seed: int = 0,
) -> Verdict:
    """Decide whether ``a == b`` holds identically: VERIFIED, REFUTED or UNDECIDED.

    * REFUTED (``method="numeric-sampling"``) — a sample point where the two sides
      have different values, confirmed at 60 digits; ``counterexample`` holds it.
    * VERIFIED (``method="symbolic"``) — ``a - b`` simplifies to zero.
    * UNDECIDED — neither. ``detail`` says how many points agreed; agreement at
      every point is NOT promoted to VERIFIED.

    ``vars`` lists the variables to sample (default: every free symbol); a symbol
    that appears but is not listed is a ``DomainError``. ``budget`` is the
    wall-clock allowance in seconds (default ``DEFAULT_TIME_BUDGET``); running out
    yields UNDECIDED with ``method="time-budget"``. ``samples`` and ``seed`` fix
    the sample points, so the same call always returns the same verdict.

    Every failure is a ``PycodemathError``: text that does not parse is a
    ``ParseError``, a bad argument a ``DomainError``. See the module docstring for
    what "equal" means (principal branch, where both sides are defined, decimals
    as written) and for the behaviour inside an enclosing ``time_budget``.
    """
    total = _check_budget(budget)
    if (
        not isinstance(samples, int)
        or isinstance(samples, bool)
        or samples < 1
    ):
        raise DomainError(
            f"check_equal: samples must be a positive integer, not {samples!r}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DomainError(f"check_equal: seed must be an integer, not {seed!r}")

    left, right = _to_expr(a, "left"), _to_expr(b, "right")
    syms = _variables(left, right, vars)
    diff = left - right
    if diff == 0:
        # SymPy's automatic evaluation already cancelled the two sides: the claim
        # is an identity by construction, and sampling it would prove nothing more.
        return Verdict(
            VerdictStatus.VERIFIED, "symbolic", None,
            "the two sides are identical after automatic simplification",
        )

    # The budget's refusal is always caught below and turned into a verdict, so
    # its message is never read: no subject is built for it.
    points = _points(syms, samples, seed)
    sampling = _Sampling()
    # perf_counter, not monotonic: on Windows monotonic ticks every 15.6 ms, so a
    # budget under one tick never expired and a check fitting in it came back
    # VERIFIED (2/8 and 1/3 gate runs red). Same clock in steps.py, certify.py.
    start = time.perf_counter()
    deadline = start + total  # inf stays inf

    if _caller_budget_active():
        sampling.run(left, right, points, deadline)
        if sampling.refutation is not None:
            return sampling.refutation
        if _proved(diff):  # the ENCLOSING budget governs; its expiry is the caller's
            return _verified(sampling)
        return _undecided(sampling)

    try:
        with time_budget(total, "check_equal"):
            sampling.run(left, right, points, deadline)
    except TimeBudgetError:
        return _out_of_time(sampling, total, "numeric sampling")
    if sampling.refutation is not None:
        return sampling.refutation

    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return _out_of_time(sampling, total, "numeric sampling")
    try:
        with time_budget(remaining, "check_equal"):
            proved = _proved(diff)
    except TimeBudgetError:
        return _out_of_time(sampling, total, "symbolic simplification")
    return _verified(sampling) if proved else _undecided(sampling)


def _verified(sampling: _Sampling) -> Verdict:
    return Verdict(
        VerdictStatus.VERIFIED, "symbolic", None,
        f"left - right simplifies to 0; {sampling.evidence()}",
    )


def _undecided(sampling: _Sampling) -> Verdict:
    return Verdict(
        VerdictStatus.UNDECIDED, "numeric-sampling", None,
        f"{sampling.evidence()}, without a symbolic proof",
    )


def _out_of_time(sampling: _Sampling, total: float, phase: str) -> Verdict:
    return Verdict(
        VerdictStatus.UNDECIDED, "time-budget", None,
        f"the {total:g}s time budget ran out during {phase}; {sampling.evidence()}",
    )
