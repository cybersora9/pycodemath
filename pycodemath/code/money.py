"""Money and rounding — exact amounts, provable splits, and witnesses for lost cents (C4).

A programmer who adds up prices, invoices, instalments, taxes or a split bill gets
an exact result and a PROOF that rounding neither lost nor created a cent — or a
concrete input on which their way of computing does.

    >>> from pycodemath.code.money import allocate, round_money, float_money_hazard
    >>> [str(s) for s in allocate("100", 3, "PLN").shares]
    ['33.34', '33.33', '33.33']
    >>> str(round_money("2.675", "PLN", "HALF_UP").rounded)
    '2.68'
    >>> h = float_money_hazard("round(2.675, 2)")
    >>> h.hazard, h.float_value, h.exact_value
    (True, '2.67', '2.68')

THE CONTRACT. Every number here is computed exactly: ``fractions.Fraction`` inside,
``decimal.Decimal`` at the currency's scale outside. Nothing is ever converted
through a binary float. A ``float`` argument is REFUSED (``MoneyInputError``)
everywhere except ``float_money_hazard``, whose whole job is to look at floats:
``2.675`` as a float is 2.67499999999999982236431605997495353221893310546875, and
silently accepting it would make the library commit the very error it exists to
catch. Pass ``"2.675"`` or ``Decimal("2.675")``.

Every result carries ``checks``: the names of the independent checks that passed
before the result was returned (an exact re-derivation by a second route — a sum,
a definition, a second algorithm). A check that fails is a bug in this module and
raises ``RuntimeError``; it is never returned as an answer. Every claim that
something DIVERGES or is a HAZARD carries the concrete input that shows it.

ROUNDING MODES — names follow ``decimal`` (and Java ``RoundingMode``):

=============  ======================================  =======================================
mode           rule                                    who uses it (source)
=============  ======================================  =======================================
``HALF_UP``    nearest; a tie goes AWAY from zero      Polish VAT on invoices: "<0.5 gr dropped,
               (2.675 -> 2.68, -2.675 -> -2.68)        >=0.5 gr up" (ustawa o VAT art. 106e
                                                       ust. 11); Polish tax bases/amounts to
                                                       full zloty (Ordynacja podatkowa art. 63
                                                       par. 1); euro conversions to the nearest
                                                       cent (Reg. (EC) 1103/97 art. 5). The
                                                       DEFAULT here: it is what invoices and
                                                       tax law in PL/EU prescribe.
``HALF_EVEN``  nearest; a tie goes to the EVEN digit  "banker's rounding": IEEE 754 default
               (2.665 -> 2.66, 2.675 -> 2.68)          (roundTiesToEven), Python's ``round()``
                                                       and the default ``decimal`` context,
                                                       Mendix' default. Unbiased over many ties.
``HALF_DOWN``  nearest; a tie goes TOWARD zero         rare; listed for completeness
``DOWN``       toward zero (truncation)                ``int()``, casts, legacy code; JSR 354
                                                       warns against implicit truncation
``UP``         away from zero                          PL capital-interest tax "rounded UP to
                                                       full grosz" (Ordynacja art. 63 par. 1a)
``CEILING``    toward +infinity                        as ``UP`` for positive amounts, but a
``FLOOR``      toward -infinity                        refund (negative) goes the other way —
                                                       the difference that bites on credit notes
=============  ======================================  =======================================

``decimal.ROUND_05UP`` is deliberately not offered: no money rule found that uses it.

CURRENCIES. ``CURRENCY_EXPONENTS`` is ISO 4217 Table A.1 ("minor unit"), active
codes as of 2026 (maintenance agency: SIX), plus ``ANG`` (replaced by ``XCG`` in
2025, still found in ledgers). Cross-checked, measured on 30.10.2026: against
py-moneyed 3.0, 164/164 shared codes agree; against dinero.js 2.0.2, 164/166 shared
codes agree — the two others are ``MGA`` and ``MRU``, whose real subunit is 1/5:
dinero models that (base 5, exponent 1), ISO lists 2 decimal places, and so do we.
Codes whose minor unit is "N.A." in ISO (metals XAU/XAG/XPT/XPD, XDR, XTS, XXX, ...)
are absent on purpose — pass the number of decimal places as an ``int`` instead.

ALLOCATION. ``allocate`` uses the largest-remainder (Hamilton) method: every share
first gets the floor of its exact quota in minor units; the units left over go to
the shares with the largest fractional remainders, a tie going to the LOWER index.
Guaranteed and checked on every call: the shares sum to the total exactly, and
each share is within one minor unit of its exact quota (the "quota property").
The price, stated honestly: Hamilton exhibits the Alabama paradox — raising the
total can lower a share (weights 1:3:3, 0.03 -> 0.01/0.01/0.01 but 0.04 ->
0.00/0.02/0.02). Divisor methods (D'Hondt, Sainte-Lague) avoid it but violate the
quota property — a share can be off by more than a cent — which is the wrong
trade for money, so they are not offered. ``method="in_order"`` reproduces Fowler's
Money pattern (PoEAA) and its ports such as Money for PHP: leftover units to the
first shares in list order. It also always sums exactly, but can break the quota
property, and the result says so (``within_quota=False``): 0.06 split 2:1:1 gives
0.04/0.01/0.01 although the first quota is exactly 0.03. dinero.js 2.0.2 gives the
same 0.04/0.01/0.01 by a third rule (leftover units to the LARGEST WEIGHTS first);
measured on 20 000 random splits it breaks the quota property on 364 of them.
"""

from __future__ import annotations

import ast
import itertools
import math
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation, localcontext
import decimal
from fractions import Fraction
from numbers import Rational
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

from ..core.budget import guard
from ..core.errors import DomainError
from ..verify.verdict import VerdictStatus

__all__ = [
    "MoneyInputError",
    "ROUNDING_MODES",
    "CURRENCY_EXPONENTS",
    "currency_exponent",
    "RoundedAmount",
    "round_money",
    "Allocation",
    "allocate",
    "RoundingStrategyReport",
    "check_rounding_strategy",
    "VatComparison",
    "vat_per_line_vs_total",
    "CompoundInterest",
    "compound_interest",
    "FloatHazardReport",
    "float_money_hazard",
    "money_grid",
    "AnnuityPayment",
    "annuity_payment",
    "ScheduleRow",
    "AmortizationSchedule",
    "amortization_schedule",
    "NPVResult",
    "npv",
    "IRRResult",
    "irr",
]

#: Anything that denotes an exact amount: ``"12.34"``, ``"1/3"``, ``Decimal``, ``int``,
#: ``Fraction`` (any ``numbers.Rational``). Never ``float`` — see the module docstring.
Exact = Union[str, int, Decimal, Fraction, Rational]


class MoneyInputError(DomainError):
    """An input this module will not compute with: a float where an exact amount is
    needed, an unknown currency, a total with more decimals than its currency has,
    negative weights. Raised before any arithmetic happens; ``route`` is ``None``."""


# --- rounding modes ------------------------------------------------------------

#: Mode name -> one-line rule. The table in the module docstring says who uses which.
ROUNDING_MODES: dict[str, str] = {
    "HALF_UP": "nearest; ties away from zero",
    "HALF_EVEN": "nearest; ties to the even last digit (banker's rounding)",
    "HALF_DOWN": "nearest; ties toward zero",
    "DOWN": "toward zero (truncate)",
    "UP": "away from zero",
    "CEILING": "toward +infinity",
    "FLOOR": "toward -infinity",
}


def _mode(mode: str) -> str:
    """``"half_up"``, ``"HALF_UP"`` and ``decimal.ROUND_HALF_UP`` all mean HALF_UP."""
    if not isinstance(mode, str):
        raise MoneyInputError(f"rounding mode must be a name such as 'HALF_UP', not {mode!r}")
    name = mode.strip().upper()
    if name.startswith("ROUND_"):
        name = name[len("ROUND_"):]
    if name not in ROUNDING_MODES:
        raise MoneyInputError(
            f"unknown rounding mode {mode!r}; use one of {', '.join(ROUNDING_MODES)}"
        )
    return name


# --- currencies ------------------------------------------------------------------

_TWO = (
    "AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BMD BND BOB BOV BRL BSD "
    "BTN BWP BYN BZD CAD CDF CHE CHF CHW CNY COP COU CRC CUP CVE CZK DKK DOP DZD EGP "
    "ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GTQ GYD HKD HNL HTG HUF IDR ILS INR IRR "
    "JMD KES KGS KHR KPW KYD KZT LAK LBP LKR LRD LSL MAD MDL MGA MKD MMK MNT MOP MRU "
    "MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD PAB PEN PGK PHP PKR PLN QAR "
    "RON RSD RUB SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS "
    "TMT TOP TRY TTD TWD TZS UAH USD USN UYU UZS VED VES WST XAD XCD XCG YER ZAR ZMW ZWG"
)
_ZERO = "BIF CLP DJF GNF ISK JPY KMF KRW PYG RWF UGX UYI VND VUV XAF XOF XPF"
_THREE = "BHD IQD JOD KWD LYD OMR TND"
_FOUR = "CLF UYW"

#: ISO 4217 code -> number of decimal places of its minor unit (see module docstring).
CURRENCY_EXPONENTS: dict[str, int] = {
    **{c: 2 for c in _TWO.split()},
    **{c: 0 for c in _ZERO.split()},
    **{c: 3 for c in _THREE.split()},
    **{c: 4 for c in _FOUR.split()},
}


def currency_exponent(currency: "str | int") -> int:
    """Decimal places of ``currency``'s minor unit: ``"PLN"`` -> 2, ``"JPY"`` -> 0,
    ``"KWD"`` -> 3. An ``int`` is taken as the number of places itself (0..18), for
    units ISO does not define (``8`` for satoshis, ``4`` for an internal ledger).

        >>> currency_exponent("jpy"), currency_exponent(8)
        (0, 8)
    """
    if isinstance(currency, bool):
        raise MoneyInputError(f"currency must be an ISO 4217 code or a number of places, not {currency!r}")
    if isinstance(currency, int):
        if not 0 <= currency <= 18:
            raise MoneyInputError(f"number of decimal places must be 0..18, got {currency}")
        return currency
    if not isinstance(currency, str):
        raise MoneyInputError(f"currency must be an ISO 4217 code or a number of places, not {currency!r}")
    code = currency.strip().upper()
    if code not in CURRENCY_EXPONENTS:
        raise MoneyInputError(
            f"unknown currency {currency!r} (ISO 4217 active codes only; codes without a "
            f"minor unit such as XAU have none — pass the number of decimal places as an int)"
        )
    return CURRENCY_EXPONENTS[code]


def _currency_label(currency: "str | int") -> str:
    return currency.strip().upper() if isinstance(currency, str) else f"{currency}dp"


# --- exact values ----------------------------------------------------------------


def _exact(value: object, what: str = "value") -> Fraction:
    """The exact rational a caller MEANT. Refuses floats, bools, NaN and infinities."""
    if isinstance(value, bool):
        raise MoneyInputError(f"{what}: expected an amount, got the bool {value!r}")
    if isinstance(value, float):
        raise MoneyInputError(
            f"{what}: {value!r} is a binary float (its exact value is {Decimal(value)}); "
            f"pass it as a string {repr(str(value))} or a Decimal, or use float_money_hazard "
            f"to see what float does to it"
        )
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise MoneyInputError(f"{what}: {value} is not a finite amount")
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            dec = Decimal(text)
        except InvalidOperation:
            try:
                return Fraction(text)
            except (ValueError, ZeroDivisionError):
                raise MoneyInputError(f"{what}: cannot read {value!r} as an amount") from None
        if not dec.is_finite():
            raise MoneyInputError(f"{what}: {value!r} is not a finite amount")
        return Fraction(dec)
    if isinstance(value, Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    # SymPy Rational and friends expose p/q but are not registered as numbers.Rational.
    p, q = getattr(value, "p", None), getattr(value, "q", None)
    if isinstance(p, int) and isinstance(q, int) and q > 0:
        return Fraction(p, q)
    raise MoneyInputError(f"{what}: expected an exact amount (str, int, Decimal, Fraction), got {value!r}")


def _floor(x: Fraction) -> int:
    return x.numerator // x.denominator


def _as_units(x: Fraction, exponent: int) -> "int | None":
    """``x`` in minor units when it is a whole number of them, else ``None``."""
    scaled = x * 10**exponent
    return scaled.numerator if scaled.denominator == 1 else None


def _units_to_decimal(units: int, exponent: int) -> Decimal:
    """Minor units -> ``Decimal`` at exactly ``exponent`` places, never in E-notation."""
    sign = 1 if units < 0 else 0
    digits = tuple(int(d) for d in str(abs(units)))
    return Decimal((sign, digits, -exponent))


def _to_amount(x: Fraction, exponent: int) -> Decimal:
    units = _as_units(x, exponent)
    if units is None:  # pragma: no cover - callers only pass representable values
        raise RuntimeError(f"internal: {x} is not a whole number of 10^-{exponent} units")
    return _units_to_decimal(units, exponent)


def _terminating_decimal(x: Fraction) -> "Decimal | None":
    """``x`` as an exact ``Decimal`` when its decimal expansion terminates."""
    d, twos, fives = x.denominator, 0, 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return None
    places = max(twos, fives)
    return _units_to_decimal(x.numerator * 10**places // x.denominator, places)


def _show(x: Fraction) -> str:
    """Exact text: a terminating decimal as decimal, anything else as ``p/q``."""
    dec = _terminating_decimal(x)
    if dec is not None and len(str(dec)) <= 60:
        return str(dec)
    return str(x)


def _round_fraction(x: Fraction, step: Fraction, mode: str) -> Fraction:
    """``x`` rounded to a multiple of ``step`` (> 0) by ``mode``, exactly."""
    n = x / step
    low = _floor(n)
    frac = n - low
    if frac == 0:
        return low * step
    positive = n > 0
    if mode == "FLOOR":
        k = low
    elif mode == "CEILING":
        k = low + 1
    elif mode == "DOWN":
        k = low if positive else low + 1
    elif mode == "UP":
        k = low + 1 if positive else low
    elif frac < Fraction(1, 2):
        k = low
    elif frac > Fraction(1, 2):
        k = low + 1
    elif mode == "HALF_UP":
        k = low + 1 if positive else low
    elif mode == "HALF_DOWN":
        k = low if positive else low + 1
    else:  # HALF_EVEN
        k = low if low % 2 == 0 else low + 1
    return k * step


def _definition_holds(x: Fraction, r: Fraction, step: Fraction, mode: str) -> bool:
    """Check ``r`` against the DEFINITION of ``mode`` — no rounding arithmetic involved."""
    if (r / step).denominator != 1 or abs(r - x) >= step:
        return False
    if mode == "FLOOR":
        return r <= x
    if mode == "CEILING":
        return r >= x
    if mode == "DOWN":
        return abs(r) <= abs(x) and r * x >= 0
    if mode == "UP":
        return abs(r) >= abs(x) and r * x >= 0
    err = abs(r - x)
    if err > step / 2:
        return False
    if err < step / 2:
        return True
    if mode == "HALF_UP":
        return abs(r) > abs(x)
    if mode == "HALF_DOWN":
        return abs(r) < abs(x)
    return (r / step).numerator % 2 == 0


def _certify(ok: bool, what: str) -> str:
    if not ok:
        raise RuntimeError(f"internal: certificate failed: {what} — please report this as a bug")
    return what


def _rounder(step: Fraction, mode: str) -> Callable[[Fraction], Fraction]:
    return lambda x: _round_fraction(x, step, mode)


# --- results: shared serialisation -------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")  # never E-notation: str(Decimal("0E-12")) is "0E-12"
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, VerdictStatus):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


class _ToDict:
    """``to_dict()``: JSON-ready, exact values as strings (``"33.34"``, ``"1/3"``),
    with ``kind`` naming the operation so a certificate can be built from it."""

    kind = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        out.update(_jsonable(self))
        return out


# --- round_money -------------------------------------------------------------------


@dataclass(frozen=True)
class RoundedAmount(_ToDict):
    """``value`` (exact) rounded to ``step`` (by default the currency's minor unit).

    ``error`` is ``rounded - value`` exactly. ``checks`` lists what was verified:
    the definition of the mode (always) and agreement with ``decimal.quantize``
    (when the value is a terminating decimal and the step a power of ten)."""

    kind = "round_money"
    value: Fraction
    rounded: Decimal
    currency: str
    mode: str
    step: Decimal
    error: Fraction
    checks: tuple[str, ...]


def round_money(
    value: Exact, currency: "str | int", mode: str = "HALF_UP", *, step: "Exact | None" = None
) -> RoundedAmount:
    """Round an exact amount to its currency (or to ``step``) with a named mode.

        >>> str(round_money("2.665", "EUR", "HALF_EVEN").rounded)
        '2.66'
        >>> str(round_money("-2.675", "PLN", "CEILING").rounded)
        '-2.67'
        >>> str(round_money("12.33", "CHF", step="0.05").rounded)   # cash rounding
        '12.35'
        >>> str(round_money("1234.50", "PLN", step="1").rounded)    # tax base, full zloty
        '1235.00'

    The result keeps the currency's scale (``1235.00`` PLN, not ``1235``). A ``step``
    must be a positive whole number of minor units (``0.05`` CHF, ``1`` PLN).
    """
    m = _mode(mode)
    exp = currency_exponent(currency)
    x = _exact(value)
    unit = Fraction(1, 10**exp)
    st = unit if step is None else _exact(step, "step")
    if st <= 0 or (st / unit).denominator != 1:
        raise MoneyInputError(
            f"step must be a positive multiple of the minor unit {_units_to_decimal(1, exp)}, got {step!r}"
        )
    r = _round_fraction(x, st, m)
    checks = [_certify(_definition_holds(x, r, st, m), f"definition of {m}")]
    dec_x = _terminating_decimal(x)
    if st == unit and dec_x is not None:
        with localcontext() as ctx:
            ctx.prec = max(28, len(dec_x.as_tuple().digits) + exp + 5)
            q = dec_x.quantize(_units_to_decimal(1, exp), rounding=getattr(decimal, "ROUND_" + m))
        checks.append(_certify(Fraction(q) == r, "decimal.quantize agrees"))
    return RoundedAmount(
        value=x,
        rounded=_to_amount(r, exp),
        currency=_currency_label(currency),
        mode=m,
        step=_to_amount(st, exp),
        error=r - x,
        checks=tuple(checks),
    )


# --- allocate --------------------------------------------------------------------


@dataclass(frozen=True)
class Allocation(_ToDict):
    """A total split into ``shares`` that sum to it exactly.

    ``quotas`` are the exact (unrounded) shares; ``extra_units`` the indices that
    received one leftover minor unit; ``within_quota`` whether every share lies
    between the floor and the ceiling of its quota (always for ``hamilton``);
    ``max_spread`` the largest minus the smallest share when all weights are equal
    (at most one minor unit, checked), else ``None``."""

    kind = "allocate"
    total: Decimal
    currency: str
    method: str
    weights: tuple[Fraction, ...]
    shares: tuple[Decimal, ...]
    quotas: tuple[Fraction, ...]
    extra_units: tuple[int, ...]
    tie_rule: str
    sums_exactly: bool
    within_quota: bool
    max_spread: "Decimal | None"
    checks: tuple[str, ...]


_TIE_RULES = {
    "hamilton": "leftover units go to the largest fractional remainders; equal remainders -> lower index first",
    "in_order": "leftover units go to the positive-weight shares in list order (Fowler's Money pattern)",
}


def allocate(
    total: Exact,
    parts_or_weights: "int | Sequence[Exact]",
    currency: "str | int",
    *,
    method: str = "hamilton",
) -> Allocation:
    """Split ``total`` into ``n`` equal parts or in proportion to ``weights``,
    to the currency's minor unit, so that the shares sum to ``total`` EXACTLY.

        >>> [str(s) for s in allocate("100", 3, "PLN").shares]
        ['33.34', '33.33', '33.33']
        >>> [str(s) for s in allocate("100", [1, 2, 3], "PLN").shares]
        ['16.67', '33.33', '50.00']
        >>> [str(s) for s in allocate("-0.05", 2, "EUR").shares]   # a refund mirrors the charge
        ['-0.03', '-0.02']

    ``total`` must be a whole number of minor units (round it first with
    ``round_money``). Weights are exact, non-negative, not all zero; a zero weight
    always gets zero. A negative total is split as its absolute value and negated,
    so a refund is the mirror image of the charge. ``method``: ``"hamilton"``
    (default, largest remainder) or ``"in_order"`` (Fowler's Money pattern).
    """
    if method not in _TIE_RULES:
        raise MoneyInputError(f"allocate: method must be 'hamilton' or 'in_order', got {method!r}")
    exp = currency_exponent(currency)
    t = _exact(total, "total")
    units = _as_units(t, exp)
    if units is None:
        raise MoneyInputError(
            f"allocate: total {_show(t)} has more decimals than {_currency_label(currency)} "
            f"({exp}); round it first with round_money"
        )
    equal = isinstance(parts_or_weights, int) and not isinstance(parts_or_weights, bool)
    if equal:
        n = int(parts_or_weights)  # type: ignore[arg-type]
        if n < 1:
            raise MoneyInputError(f"allocate: number of parts must be at least 1, got {n}")
        if n > 10**7:
            raise MoneyInputError(f"allocate: {n} parts is more than this function will list (10^7)")
        weights = [Fraction(1)] * n
    else:
        if isinstance(parts_or_weights, (str, bytes)) or not isinstance(parts_or_weights, Iterable):
            raise MoneyInputError("allocate: give a number of parts or a list of weights")
        weights = [_exact(w, f"weight {i}") for i, w in enumerate(parts_or_weights)]
        if not weights:
            raise MoneyInputError("allocate: the list of weights is empty")
        if any(w < 0 for w in weights):
            raise MoneyInputError("allocate: weights must be non-negative")
        if not any(weights):
            raise MoneyInputError("allocate: all weights are zero — there is nobody to give the money to")
        equal = len(set(weights)) == 1

    sign = -1 if units < 0 else 1
    magnitude = abs(units)
    w_sum = sum(weights)
    quotas_units = [magnitude * w / w_sum for w in weights]
    floors = [_floor(q) for q in quotas_units]
    left = magnitude - sum(floors)
    if method == "hamilton":
        order = sorted(range(len(weights)), key=lambda i: (-(quotas_units[i] - floors[i]), i))
    else:
        order = [i for i, w in enumerate(weights) if w > 0]
    extra = sorted(order[:left])
    shares_units = list(floors)
    for i in extra:
        shares_units[i] += 1

    shares = tuple(_units_to_decimal(sign * s, exp) for s in shares_units)
    unit = Fraction(1, 10**exp)
    quotas = tuple(sign * q * unit for q in quotas_units)
    checks = [
        _certify(sum(Fraction(s) for s in shares) == t, "shares sum to the total exactly (rational sum)"),
        _certify(all(s == 0 for s, w in zip(shares, weights) if w == 0), "zero weights get zero"),
    ]
    within = all(_floor(q) <= s <= -_floor(-q) for s, q in zip(shares_units, quotas_units))
    if method == "hamilton":
        checks.append(_certify(within, "every share within one unit of its quota"))
    spread: "Decimal | None" = None
    if equal:
        spread = max(shares) - min(shares)
        checks.append(_certify(abs(spread) <= _units_to_decimal(1, exp), "equal split: spread <= one unit"))
    return Allocation(
        total=_units_to_decimal(units, exp),
        currency=_currency_label(currency),
        method=method,
        weights=tuple(weights),
        shares=shares,
        quotas=quotas,
        extra_units=tuple(extra),
        tie_rule=_TIE_RULES[method],
        sums_exactly=True,
        within_quota=within,
        max_spread=spread,
        checks=tuple(checks),
    )


# --- rounding on the way vs once at the end ------------------------------------------


@dataclass(frozen=True)
class RoundingStrategyReport(_ToDict):
    """Rounding at every step vs exactly once at the end, over the supplied inputs.

    ``verdict`` is about the claim "both strategies give the same amount":
    ``REFUTED`` with ``witness`` (the first input where they differ),
    ``VERIFIED`` when every supplied input agrees — a statement about exactly those
    ``checked`` inputs (``scope``), not about inputs never tried — or ``UNDECIDED``
    when ``max_cases`` cut the list short. ``once``/``stepwise``/``difference``
    (stepwise minus once) are at the witness, or at the first input if none."""

    kind = "rounding_strategy"
    verdict: VerdictStatus
    checked: int
    divergent: int
    witness: "dict[str, str] | None"
    once: Decimal
    stepwise: Decimal
    difference: Decimal
    max_difference: Decimal
    rounding_calls: int
    mode: str
    currency: str
    scope: str


def check_rounding_strategy(
    computation: Callable[..., Any],
    inputs: "Mapping[str, Exact] | Iterable[Mapping[str, Exact]]",
    currency: "str | int",
    mode: str = "HALF_UP",
    *,
    step: "Exact | None" = None,
    max_cases: int = 100_000,
) -> RoundingStrategyReport:
    """Does rounding along the way change the result? Run ``computation`` twice per
    input — once rounding wherever it calls ``rnd``, once with ``rnd`` doing nothing
    — round both results once at the end, and compare.

    ``computation(rnd, **values)`` receives the input values as exact ``Fraction``\\ s
    and must return an exact number (``Fraction``/``int``/``Decimal``). It calls
    ``rnd(x)`` wherever the real code rounds to the currency, or ``rnd(x, step="0.01")``
    to round to another step.

        >>> def vat(rnd, a, b):              # VAT per line, then summed
        ...     return rnd(a * Fraction("0.23")) + rnd(b * Fraction("0.23"))
        >>> r = check_rounding_strategy(vat, {"a": "0.02", "b": "0.02"}, "PLN")
        >>> r.verdict.value, str(r.stepwise), str(r.once)
        ('refuted', '0.00', '0.01')
    """
    m = _mode(mode)
    exp = currency_exponent(currency)
    unit = Fraction(1, 10**exp)
    final = unit if step is None else _exact(step, "step")
    if final <= 0:
        raise MoneyInputError("step must be positive")
    cases = [inputs] if isinstance(inputs, Mapping) else inputs
    if not isinstance(cases, Iterable):
        raise MoneyInputError("inputs must be a mapping or an iterable of mappings")

    def run() -> RoundingStrategyReport:
        checked = divergent = calls = 0
        witness: "dict[str, str] | None" = None
        first: "tuple[Fraction, Fraction, int] | None" = None
        worst = Fraction(0)
        truncated = False
        for case in cases:
            if checked >= max_cases:
                truncated = True
                break
            if not isinstance(case, Mapping):
                raise MoneyInputError(f"each input must be a mapping of name -> amount, got {case!r}")
            values = {str(k): _exact(v, str(k)) for k, v in case.items()}
            counter = [0]

            def stepwise_rnd(x: Any, step: "Exact | None" = None) -> Fraction:
                counter[0] += 1
                s = unit if step is None else _exact(step, "rnd step")
                return _round_fraction(_exact(x, "rnd argument"), s, m)

            def exact_rnd(x: Any, step: "Exact | None" = None) -> Fraction:
                return _exact(x, "rnd argument")

            once = _round_fraction(_exact(computation(exact_rnd, **values), "result"), final, m)
            stepwise = _round_fraction(_exact(computation(stepwise_rnd, **values), "result"), final, m)
            checked += 1
            worst = max(worst, abs(stepwise - once))
            if first is None:
                first = (once, stepwise, counter[0])
            if once != stepwise:
                divergent += 1
                if witness is None:
                    witness = {k: _show(v) for k, v in values.items()}
                    first = (once, stepwise, counter[0])
        if first is None:
            raise MoneyInputError("check_rounding_strategy: no inputs to check")
        once, stepwise, calls = first
        if witness is not None:
            verdict, scope = VerdictStatus.REFUTED, f"diverges on {divergent} of {checked} supplied inputs"
        elif truncated:
            verdict, scope = VerdictStatus.UNDECIDED, f"no divergence on the first {checked} inputs; more were not checked"
        else:
            verdict, scope = VerdictStatus.VERIFIED, f"no divergence on all {checked} supplied inputs (not a claim about others)"
        show_exp = max(exp, _places(final))
        return RoundingStrategyReport(
            verdict=verdict,
            checked=checked,
            divergent=divergent,
            witness=witness,
            once=_to_amount(once, show_exp),
            stepwise=_to_amount(stepwise, show_exp),
            difference=_to_amount(stepwise - once, show_exp),
            max_difference=_to_amount(worst, show_exp),
            rounding_calls=calls,
            mode=m,
            currency=_currency_label(currency),
            scope=scope,
        )

    return guard("check_rounding_strategy", getattr(computation, "__name__", "computation"), run)


def _places(step: Fraction) -> int:
    """Decimal places needed to write multiples of ``step`` exactly (``0.05`` -> 2)."""
    dec = _terminating_decimal(step)
    if dec is None:
        raise MoneyInputError(f"step {step} is not a terminating decimal")
    return max(0, -int(dec.as_tuple().exponent))


@dataclass(frozen=True)
class VatComparison(_ToDict):
    """VAT rounded per line and summed (``per_line``) vs VAT on the summed net
    (``on_total``); ``difference`` is ``per_line - on_total``.

    ``diverges`` is about the lines given. ``always_equal`` is about EVERY invoice
    with this rate and this number of lines, each a whole number of minor units:
    ``True`` is proved (one line; or an integer rate, where no product needs
    rounding), ``False`` comes with ``witness_lines`` (the given lines when they
    diverge, otherwise found by search), ``None`` when the search found nothing.
    ``proven_bound`` is the largest |difference| possible for this many lines."""

    kind = "vat_per_line_vs_total"
    lines: tuple[str, ...]
    rate: Fraction
    currency: str
    mode: str
    per_line: Decimal
    on_total: Decimal
    difference: Decimal
    diverges: bool
    always_equal: "bool | None"
    witness_lines: "tuple[str, ...] | None"
    witness_per_line: "Decimal | None"
    witness_on_total: "Decimal | None"
    proven_bound: Decimal
    checks: tuple[str, ...]


def vat_per_line_vs_total(
    lines: Sequence[Exact],
    rate: Exact,
    currency: "str | int",
    mode: str = "HALF_UP",
    *,
    search_limit: int = 10_000,
) -> VatComparison:
    """Classic lost cent: VAT computed per invoice line vs on the invoice total.

        >>> v = vat_per_line_vs_total(["0.01", "0.01", "0.01"], "0.23", "PLN")
        >>> str(v.per_line), str(v.on_total), v.diverges
        ('0.00', '0.01', True)

    Neither total is "the" right one in general — tax law and ERP settings choose
    (Odoo's ``round_per_line`` vs ``round_globally``); the point is to KNOW the two
    differ, by how much at most, and on which invoice.
    """
    m = _mode(mode)
    exp = currency_exponent(currency)
    unit = Fraction(1, 10**exp)
    r = _exact(rate, "rate")
    xs = [_exact(x, f"line {i}") for i, x in enumerate(lines)]
    if not xs:
        raise MoneyInputError("vat_per_line_vs_total: no lines")
    rnd = _rounder(unit, m)

    def both(amounts: Sequence[Fraction]) -> "tuple[Fraction, Fraction]":
        return sum((rnd(a * r) for a in amounts), Fraction(0)), rnd(sum(amounts, Fraction(0)) * r)

    per, tot = both(xs)
    n = len(xs)
    # |sum of n roundings - one rounding| <= (n+1) * max error; a multiple of the unit.
    half = m.startswith("HALF")
    bound_units = (n + 1) // 2 if half else n
    checks = [_certify(abs(per - tot) <= bound_units * unit, f"|difference| <= {bound_units} unit(s)")]
    always: "bool | None"
    w_lines: "list[Fraction] | None" = None
    if per != tot:
        always, w_lines = False, xs
    elif n == 1 or r.denominator == 1:
        always = True  # one rounding each side, or every product already a whole unit
    else:
        always = None
        for a in range(1, search_limit + 1):
            cand = [a * unit] * n
            p2, t2 = both(cand)
            if p2 != t2:
                always, w_lines = False, cand
                break
    wp = wt = None
    if w_lines is not None:
        p3, t3 = both(w_lines)
        checks.append(_certify(p3 != t3, "witness re-checked: per-line != on-total"))
        wp, wt = _to_amount(p3, exp), _to_amount(t3, exp)

    def fmt(x: Fraction) -> str:
        return str(_to_amount(x, exp)) if _as_units(x, exp) is not None else _show(x)

    return VatComparison(
        lines=tuple(fmt(x) for x in xs),
        rate=r,
        currency=_currency_label(currency),
        mode=m,
        per_line=_to_amount(per, exp),
        on_total=_to_amount(tot, exp),
        difference=_to_amount(per - tot, exp),
        diverges=per != tot,
        always_equal=always,
        witness_lines=None if w_lines is None else tuple(fmt(x) for x in w_lines),
        witness_per_line=wp,
        witness_on_total=wt,
        proven_bound=_units_to_decimal(bound_units, exp),
        checks=tuple(checks),
    )


@dataclass(frozen=True)
class CompoundInterest(_ToDict):
    """``principal * (1 + rate/periods_per_year) ** periods``: ``exact``, rounded
    ``once`` at the end, and ``stepwise`` (the balance rounded every period, as a
    bank account capitalising interest does). ``first_divergence`` is the first
    period after which the stepwise balance differs from the exact balance
    rounded at that period, or ``None``."""

    kind = "compound_interest"
    principal: Fraction
    periodic_rate: Fraction
    periods: int
    currency: str
    mode: str
    exact: Fraction
    once: Decimal
    stepwise: Decimal
    difference: Decimal
    first_divergence: "int | None"
    checks: tuple[str, ...]


def compound_interest(
    principal: Exact,
    rate: Exact,
    periods: int,
    currency: "str | int",
    mode: str = "HALF_UP",
    *,
    periods_per_year: int = 1,
) -> CompoundInterest:
    """Compound interest, exactly — and what rounding every period does to it.

        >>> c = compound_interest("1234.56", "0.07", 36, "PLN", periods_per_year=12)
        >>> str(c.once), str(c.stepwise), c.first_divergence
        ('1522.12', '1522.11', 2)
    """
    m = _mode(mode)
    exp = currency_exponent(currency)
    unit = Fraction(1, 10**exp)
    n = _count(periods, "periods", minimum=0)
    ppy = _count(periods_per_year, "periods_per_year", minimum=1)
    p = _exact(principal, "principal")
    r = _exact(rate, "rate") / ppy
    if r <= -1:
        raise MoneyInputError("compound_interest: the periodic rate must be greater than -100%")

    def run() -> CompoundInterest:
        exact = p * (1 + r) ** n
        growth, bal, exact_k, first = Fraction(1), _round_fraction(p, unit, m), p, None
        for k in range(1, n + 1):
            growth *= 1 + r
            exact_k = exact_k * (1 + r)
            bal = _round_fraction(bal * (1 + r), unit, m)
            if first is None and bal != _round_fraction(exact_k, unit, m):
                first = k
        checks = [_certify(p * growth == exact, "closed form == period-by-period product")]
        once = _round_fraction(exact, unit, m)
        return CompoundInterest(
            principal=p,
            periodic_rate=r,
            periods=n,
            currency=_currency_label(currency),
            mode=m,
            exact=exact,
            once=_to_amount(once, exp),
            stepwise=_to_amount(bal, exp),
            difference=_to_amount(bal - once, exp),
            first_divergence=first,
            checks=tuple(checks),
        )

    return guard("compound_interest", f"{n} periods", run)


def _count(value: object, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MoneyInputError(f"{what} must be an integer >= {minimum}, got {value!r}")
    return value


# --- float on money -------------------------------------------------------------

_FUNCS = {"round", "int", "abs", "min", "max", "float"}
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_MAX_POW = 64


def _int_literal(node: ast.AST) -> "int | None":
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _int_literal(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _check_ast(node: ast.AST, names: "set[str]") -> None:
    """Whitelist: arithmetic, comparisons, numeric literals, the given names, and
    round/int/abs/min/max/float. Anything else is refused before it runs."""
    if isinstance(node, ast.Expression):
        _check_ast(node.body, names)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, _BINOPS):
        if isinstance(node.op, ast.Pow):
            e = _int_literal(node.right)
            if e is None or abs(e) > _MAX_POW:
                raise MoneyInputError(f"float_money_hazard: '**' needs an integer literal exponent of at most {_MAX_POW}")
            # Stacked integer powers grow as 64**depth digits and CPython computes them
            # in one C call that no time budget can interrupt — refuse them up front.
            if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) for n in ast.walk(node.left)):
                raise MoneyInputError("float_money_hazard: '**' cannot be applied to an expression that already contains '**'")
        _check_ast(node.left, names)
        _check_ast(node.right, names)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _check_ast(node.operand, names)
    elif isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise MoneyInputError(f"float_money_hazard: unsupported literal {node.value!r}")
    elif isinstance(node, ast.Name):
        if node.id not in names:
            raise MoneyInputError(f"float_money_hazard: unknown name {node.id!r}; give it a value in variables=")
    elif isinstance(node, ast.Compare):
        if not all(isinstance(op, _CMPOPS) for op in node.ops):
            raise MoneyInputError("float_money_hazard: only ==, !=, <, <=, >, >= comparisons are supported")
        for sub in [node.left, *node.comparators]:
            _check_ast(sub, names)
    elif isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS) or node.keywords:
            raise MoneyInputError(f"float_money_hazard: only calls to {', '.join(sorted(_FUNCS))} are supported")
        fn, nargs = node.func.id, len(node.args)
        if (fn in ("int", "abs", "float") and nargs != 1) or (fn == "round" and nargs not in (1, 2)) \
                or (fn in ("min", "max") and nargs < 2):
            raise MoneyInputError(f"float_money_hazard: wrong number of arguments to {fn}()")
        if fn == "round" and nargs == 2 and _int_literal(node.args[1]) is None:
            raise MoneyInputError("float_money_hazard: round()'s ndigits must be an integer literal")
        for a in node.args:
            _check_ast(a, names)
    else:
        raise MoneyInputError(f"float_money_hazard: unsupported construct {type(node).__name__}")


_ExactVal = Union[Fraction, int, bool]


class _ExactEval:
    """Evaluate the whitelisted AST with every literal read as the decimal WRITTEN."""

    def __init__(self, source: str, env: "dict[str, Fraction]", mode: str) -> None:
        self.source, self.env, self.mode = source, env, mode

    def __call__(self, node: ast.AST) -> _ExactVal:
        if isinstance(node, ast.Expression):
            return self(node.body)
        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                return int(node.value)
            text = ast.get_source_segment(self.source, node) or repr(node.value)
            return Fraction(Decimal(text))
        if isinstance(node, ast.Name):
            return self.env[node.id]
        if isinstance(node, ast.UnaryOp):
            v = self(node.operand)
            return -v if isinstance(node.op, ast.USub) else +v
        if isinstance(node, ast.BinOp):
            a, b = self(node.left), self(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return Fraction(a) / Fraction(b)
            if isinstance(op, ast.FloorDiv):
                q = Fraction(a) / Fraction(b)
                return _floor(q) if isinstance(a, int) and isinstance(b, int) else Fraction(_floor(q))
            if isinstance(op, ast.Mod):
                return a - b * _floor(Fraction(a) / Fraction(b))
            exp = int(b)
            if isinstance(a, int) and exp >= 0:
                return int(a) ** exp
            return Fraction(a) ** exp
        if isinstance(node, ast.Compare):
            left = self(node.left)
            for cop, comp in zip(node.ops, node.comparators):
                right = self(comp)
                ok = {ast.Eq: left == right, ast.NotEq: left != right, ast.Lt: left < right,
                      ast.LtE: left <= right, ast.Gt: left > right, ast.GtE: left >= right}[type(cop)]
                if not ok:
                    return False
                left = right
            return True
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        fn = node.func.id
        args = [self(a) for a in node.args]
        if fn == "round":
            x = Fraction(args[0])
            if len(args) == 1:
                return int(_round_fraction(x, Fraction(1), self.mode))
            nd = int(args[1])
            step = Fraction(1, 10**nd) if nd >= 0 else Fraction(10**-nd)
            r = _round_fraction(x, step, self.mode)
            return int(r) if isinstance(args[0], int) else r
        if fn == "int":
            return int(Fraction(args[0]))
        if fn == "abs":
            v = args[0]
            return abs(int(v)) if isinstance(v, int) else abs(Fraction(v))
        if fn == "float":
            return Fraction(args[0])
        return min(args) if fn == "min" else max(args)


@dataclass(frozen=True)
class FloatHazardReport(_ToDict):
    """What binary floats do to a money expression, against exact decimal arithmetic.

    ``verdict`` is about the claim "float gives the right amount here":
    ``REFUTED`` with ``witness`` (variable values; ``{}`` for a constant
    expression), ``VERIFIED`` when every input in the declared domain agrees
    (``scope``), ``UNDECIDED`` when ``max_checks`` stopped the enumeration.

    "The right amount" is decided at the level money is seen: a comparison's
    True/False, an ``int``/``round`` result, a ``round(x, n)`` to ``n`` places, or —
    with ``currency`` — the final value rounded to the currency by ``mode``. A
    float that is merely not the exact decimal (``0.1 + 0.2``) but rounds to the
    right amount is counted in ``representation_only``, never as a hazard.

    The exact side evaluates Python's own semantics (``round`` ties-to-even) so a
    hazard is caused by float and nothing else. ``exact_mode_value`` repeats the
    exact computation with ``round()`` using ``mode`` when that is not HALF_EVEN;
    ``mode_differs`` says whether that changes the amount — a separate surprise
    (banker's rounding) from the float one. ``literal_errors`` lists the decimal
    literals a float cannot hold, with the value it holds instead.

    ``float_value``/``exact_value`` are the two AMOUNTS as compared (at the witness,
    or at the first input): ``x * 1.23`` at ``x = 0.50`` with ``currency="PLN"``
    gives ``0.61`` vs ``0.62``. ``float_raw``/``exact_raw`` are the unrounded values
    behind them (``0.615`` — the float one printed by ``repr``, which hides that it
    is really 0.61499999999999999112...)."""

    kind = "float_money_hazard"
    expression: str
    verdict: VerdictStatus
    hazard: bool
    checked: int
    hazards: int
    skipped: int
    witness: "dict[str, str] | None"
    float_value: str
    exact_value: str
    exact_mode_value: "str | None"
    mode: str
    mode_differs: bool
    mode_differences: int
    float_raw: str
    exact_raw: str
    representation_only: int
    literal_errors: tuple[tuple[str, str], ...]
    explanation: str
    scope: str


def money_grid(low: Exact, high: Exact, step: Exact = "0.01") -> tuple[str, ...]:
    """Every amount from ``low`` to ``high`` inclusive in ``step``\\ s, as decimal text.

        >>> money_grid("0.01", "0.05")
        ('0.01', '0.02', '0.03', '0.04', '0.05')
    """
    lo, hi, st = _exact(low, "low"), _exact(high, "high"), _exact(step, "step")
    if st <= 0:
        raise MoneyInputError("money_grid: step must be positive")
    count = _floor((hi - lo) / st) + 1
    if count > 10**7:
        raise MoneyInputError(f"money_grid: {count} values is more than 10^7")
    places = max(_places(lo), _places(st))
    scale = 10**places
    return tuple(str(_units_to_decimal(int((lo + k * st) * scale), places)) for k in range(max(count, 0)))


def _float_input(value: object) -> "tuple[float | int, Fraction, str]":
    """A variable's value as the running code would hold it, as the decimal meant,
    and as the caller wrote it (for the witness)."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MoneyInputError(f"float_money_hazard: {value!r} is not a finite amount")
        return value, Fraction(Decimal(repr(value))), repr(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value, Fraction(value), str(value)
    exact = _exact(value, "variable value")
    text = value.strip() if isinstance(value, str) else _show(exact)
    return float(exact), exact, text


def _money_level(
    fv: object, ev: _ExactVal, final: ast.AST, unit: "Fraction | None", mode: str
) -> "tuple[bool, bool, str, str]":
    """(hazard, representation_only, float amount, exact amount) for one input — the
    two amounts as they were compared (``0.61`` vs ``0.62``, not ``0.615`` twice)."""
    if isinstance(fv, bool) or isinstance(ev, bool) or (isinstance(fv, int) and isinstance(ev, int)):
        return fv != ev, False, str(fv), str(ev)
    assert isinstance(fv, (int, float))
    if not math.isfinite(fv):
        return True, False, repr(fv), _fmt_exact(ev)
    f = Fraction(fv)
    e = Fraction(ev)
    step: "Fraction | None" = None
    final_mode = mode
    if isinstance(final, ast.Call) and isinstance(final.func, ast.Name) and final.func.id == "round" \
            and len(final.args) == 2:
        nd = _int_literal(final.args[1])
        assert nd is not None
        step, final_mode = (Fraction(1, 10**nd) if nd >= 0 else Fraction(10**-nd)), "HALF_EVEN"
    elif unit is not None:
        step = unit
    if step is None:
        return False, f != e, repr(fv), _fmt_exact(ev)
    rf, re_ = _round_fraction(f, step, final_mode), _round_fraction(e, step, final_mode)
    places = _places(step)
    return rf != re_, rf == re_ and f != e, str(_to_amount(rf, places)), str(_to_amount(re_, places))


def float_money_hazard(
    expr: str,
    *,
    variables: "Mapping[str, Iterable[object]] | None" = None,
    currency: "str | int | None" = None,
    mode: str = "HALF_EVEN",
    max_checks: int = 100_000,
) -> FloatHazardReport:
    """Where does float give a different amount than exact decimal arithmetic?

    ``expr`` is a Python expression (arithmetic, comparisons, ``round``/``int``/
    ``abs``/``min``/``max``/``float``). It is run by CPython on floats — exactly what
    the code does — and evaluated exactly with every literal read as the decimal
    written. ``variables`` maps names to the values to try (decimal strings,
    ``Decimal``, ``int``; see ``money_grid``); all combinations are checked.

        >>> float_money_hazard("round(2.675, 2)").hazard
        True
        >>> float_money_hazard("0.1 + 0.2 == 0.3").float_value
        'False'
        >>> float_money_hazard("0.1 + 0.2", currency="USD").hazard   # rounds to 0.30 either way
        False
        >>> h = float_money_hazard("round(x * 1.23, 2)", variables={"x": money_grid("0.01", "10.00")})
        >>> h.hazards, h.checked, h.witness
        (2, 1000, {'x': '0.50'})
    """
    m = _mode(mode)
    if not isinstance(expr, str):
        raise MoneyInputError("float_money_hazard: the expression must be a string")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise MoneyInputError(f"float_money_hazard: cannot parse {expr!r}: {exc.msg}") from None
    source = expr.strip()
    variables = dict(variables or {})
    _check_ast(tree, set(variables))
    unit = None if currency is None else Fraction(1, 10 ** currency_exponent(currency))
    code = compile(tree, "<money expression>", "eval")
    builtins = {"round": round, "int": int, "abs": abs, "min": min, "max": max, "float": float}
    names = list(variables)
    pools = [[_float_input(v) for v in variables[k]] for k in names]
    total = math.prod(len(p) for p in pools)
    if total == 0:
        raise MoneyInputError("float_money_hazard: a variable has no values to try")
    literal_errors = _literal_errors(tree, source)

    def run() -> FloatHazardReport:
        checked = hazards = skipped = mode_diffs = rep_only = 0
        witness: "dict[str, str] | None" = None
        shown: "tuple[str, str, str | None, bool] | None" = None
        raw: "tuple[str, str]" = ("", "")
        for combo in itertools.islice(itertools.product(*pools), max_checks):
            env_f = {k: v[0] for k, v in zip(names, combo)}
            env_e = {k: v[1] for k, v in zip(names, combo)}
            try:
                fv: object = eval(code, {"__builtins__": builtins}, env_f)  # noqa: S307 - whitelisted AST
                f_err = None
            except (ArithmeticError, ValueError) as exc:
                fv, f_err = None, f"{type(exc).__name__}: {exc}"
            try:
                ev = _ExactEval(source, env_e, "HALF_EVEN")(tree)
                ev_mode = ev if m == "HALF_EVEN" else _ExactEval(source, env_e, m)(tree)
            except ZeroDivisionError:
                skipped += 1  # undefined for these inputs in exact arithmetic too
                continue
            checked += 1
            if f_err is not None:
                hz, rep, f_amt, e_amt = True, False, f_err, _fmt_exact(ev)
            else:
                hz, rep, f_amt, e_amt = _money_level(fv, ev, tree.body, unit, "HALF_EVEN" if unit is None else m)
            md = _differs(ev, ev_mode, tree.body, unit, m)
            mode_diffs += md
            rep_only += rep
            here = (f_amt, e_amt, None if m == "HALF_EVEN" else _fmt_exact(ev_mode), md)
            here_raw = (f_err or _fmt_float(fv), _fmt_exact(ev))
            if shown is None:
                shown, raw = here, here_raw
            if hz:
                hazards += 1
                if witness is None:
                    witness = {k: v[2] for k, v in zip(names, combo)}
                    shown, raw = here, here_raw
        if shown is None:
            raise MoneyInputError("float_money_hazard: the expression divides by zero for every input")
        exhaustive = checked + skipped == total
        if witness is not None:
            verdict = VerdictStatus.REFUTED
            scope = f"float gives a different amount on {hazards} of {checked} inputs"
        elif exhaustive:
            verdict = VerdictStatus.VERIFIED
            scope = f"float gives the right amount on all {checked} inputs" + (
                f" ({skipped} undefined)" if skipped else "")
        else:
            verdict = VerdictStatus.UNDECIDED
            scope = f"no hazard in the first {checked} of {total} inputs; the rest were not checked"
        return FloatHazardReport(
            expression=source,
            verdict=verdict,
            hazard=witness is not None,
            checked=checked,
            hazards=hazards,
            skipped=skipped,
            witness=witness,
            float_value=shown[0],
            exact_value=shown[1],
            exact_mode_value=shown[2],
            mode=m,
            mode_differs=shown[3] if witness is not None or total == 1 else mode_diffs > 0,
            float_raw=raw[0],
            exact_raw=raw[1],
            mode_differences=mode_diffs,
            representation_only=rep_only,
            literal_errors=literal_errors,
            explanation=_explain(witness is not None, literal_errors, shown, m, rep_only),
            scope=scope,
        )

    return guard("float_money_hazard", source, run)


def _differs(a: _ExactVal, b: _ExactVal, final: ast.AST, unit: "Fraction | None", mode: str) -> bool:
    if a == b:
        return False
    if isinstance(a, bool) or isinstance(b, bool) or unit is None:
        return True
    return _round_fraction(Fraction(a), unit, mode) != _round_fraction(Fraction(b), unit, mode)


def _fmt_float(v: object) -> str:
    return repr(v) if isinstance(v, float) else str(v)


def _fmt_exact(v: _ExactVal) -> str:
    return str(v) if isinstance(v, (bool, int)) else _show(v)


def _literal_errors(tree: ast.AST, source: str) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is float:
            text = ast.get_source_segment(source, node) or repr(node.value)
            if Fraction(node.value) != Fraction(Decimal(text)) and all(t != text for t, _ in out):
                out.append((text, str(Decimal(node.value))))
    return tuple(out)


def _explain(
    hazard: bool, literal_errors: "tuple[tuple[str, str], ...]",
    shown: "tuple[str, str, str | None, bool]", mode: str, rep_only: int,
) -> str:
    if hazard:
        why = ""
        if literal_errors:
            text, held = literal_errors[0]
            why = f"; e.g. the literal {text} is stored as {held}"
        return f"float gives {shown[0]} where exact decimal arithmetic gives {shown[1]}{why}"
    if shown[3]:
        return (f"float agrees with Python's ties-to-even rounding, but with {mode} the amount "
                f"is {shown[2]} instead of {shown[1]}")
    if rep_only:
        return "float is off in the last bits but rounds to the same amount"
    return "float gives the same amount"


# --- finance -------------------------------------------------------------------


def _periodic(rate: Exact, periods_per_year: int) -> Fraction:
    ppy = _count(periods_per_year, "periods_per_year", minimum=1)
    r = _exact(rate, "rate") / ppy
    if r <= -1:
        raise MoneyInputError("the periodic rate must be greater than -100%")
    return r


def _annuity_exact(p: Fraction, r: Fraction, n: int, due: str) -> Fraction:
    if r == 0:
        return p / n
    pmt = p * r / (1 - (1 + r) ** -n)
    return pmt / (1 + r) if due == "begin" else pmt


@dataclass(frozen=True)
class AnnuityPayment(_ToDict):
    """The level payment that repays ``principal`` in ``periods`` instalments.

    ``exact`` is the rational payment; ``payment`` it rounded by ``mode``.
    ``naive_total`` is ``payment * periods`` — NOT what is actually paid once the
    last instalment absorbs the rounding (see ``amortization_schedule``).
    Checked: discounting ``exact`` back over every period gives ``principal``
    exactly (a sum, independent of the closed form)."""

    kind = "annuity_payment"
    principal: Fraction
    periodic_rate: Fraction
    periods: int
    due: str
    currency: str
    mode: str
    exact: Fraction
    payment: Decimal
    naive_total: Decimal
    checks: tuple[str, ...]


def annuity_payment(
    principal: Exact,
    rate: Exact,
    periods: int,
    currency: "str | int",
    mode: str = "HALF_UP",
    *,
    periods_per_year: int = 1,
    due: str = "end",
) -> AnnuityPayment:
    """Level instalment of a loan, exactly, then rounded.

    ``rate`` is nominal per year when ``periods_per_year`` is given (``"0.072"`` with
    12 means 0.6% a month), else per period. ``due="begin"`` for payments in advance.

        >>> str(annuity_payment("1000", "0.1", 12, "PLN", periods_per_year=12).payment)
        '87.92'
    """
    m = _mode(mode)
    exp = currency_exponent(currency)
    n = _count(periods, "periods", minimum=1)
    r = _periodic(rate, periods_per_year)
    if due not in ("end", "begin"):
        raise MoneyInputError(f"due must be 'end' or 'begin', got {due!r}")
    p = _exact(principal, "principal")
    exact = _annuity_exact(p, r, n, due)
    # Discount every payment back (Horner form of sum(exact * v**k)), not the closed form.
    v, pv = 1 / (1 + r), Fraction(0)
    for _ in range(n):
        pv = (pv + exact) * v
    if due == "begin":
        pv *= 1 + r
    checks = [_certify(pv == p, "present value of the payments == principal (exact sum)")]
    pay = _round_fraction(exact, Fraction(1, 10**exp), m)
    return AnnuityPayment(
        principal=p, periodic_rate=r, periods=n, due=due, currency=_currency_label(currency), mode=m,
        exact=exact, payment=_to_amount(pay, exp), naive_total=_to_amount(pay * n, exp), checks=tuple(checks),
    )


@dataclass(frozen=True)
class ScheduleRow(_ToDict):
    """One instalment: ``payment = interest + principal``; ``balance`` is what is
    still owed after it."""

    kind = "schedule_row"
    period: int
    payment: Decimal
    interest: Decimal
    principal: Decimal
    balance: Decimal


@dataclass(frozen=True)
class AmortizationSchedule(_ToDict):
    """A repayment plan in which every row is in whole minor units and nothing is lost.

    Checked on every schedule: principal parts sum to the loan exactly; payments
    sum to interest plus principal exactly; each row's payment is its interest
    plus its principal; the balance never goes negative and ends at zero.

    ``regular_payment`` is the first instalment (the level one for ``annuity``);
    ``last_payment`` the final, correcting one. ``naive_total`` is
    ``regular_payment * periods`` and ``naive_difference`` = ``total_paid -
    naive_total``: the "instalment x count vs amount actually repaid" gap.
    ``periods_used`` is below ``periods`` only if rounding paid the loan off early."""

    kind = "amortization_schedule"
    principal: Decimal
    periodic_rate: Fraction
    periods: int
    periods_used: int
    schedule: str
    currency: str
    mode: str
    rows: tuple[ScheduleRow, ...]
    regular_payment: Decimal
    last_payment: Decimal
    total_paid: Decimal
    total_interest: Decimal
    naive_total: Decimal
    naive_difference: Decimal
    checks: tuple[str, ...]


def amortization_schedule(
    principal: Exact,
    rate: Exact,
    periods: int,
    currency: "str | int",
    mode: str = "HALF_UP",
    *,
    periods_per_year: int = 1,
    schedule: str = "annuity",
) -> AmortizationSchedule:
    """Repayment schedule, row by row, with a correcting last instalment.

    ``schedule="annuity"`` (equal instalments, Polish "raty rowne"): interest of each
    period is ``round(balance * rate)``, principal is the rest of the rounded level
    payment, and the last row repays the exact remaining balance.
    ``schedule="equal_principal"`` ("raty malejace"): the principal is split with
    ``allocate`` (so its parts sum to the loan exactly) and interest is added.

        >>> s = amortization_schedule("1000", "0.1", 12, "PLN", periods_per_year=12)
        >>> str(s.regular_payment), str(s.last_payment), str(s.total_paid), str(s.naive_total)
        ('87.92', '87.87', '1054.99', '1055.04')
    """
    m = _mode(mode)
    exp = currency_exponent(currency)
    unit = Fraction(1, 10**exp)
    n = _count(periods, "periods", minimum=1)
    r = _periodic(rate, periods_per_year)
    p = _exact(principal, "principal")
    if _as_units(p, exp) is None or p <= 0:
        raise MoneyInputError(
            f"principal must be a positive whole number of minor units of {_currency_label(currency)}"
        )
    if schedule == "annuity":
        level = _round_fraction(_annuity_exact(p, r, n, "end"), unit, m)
        parts: "list[Fraction] | None" = None
    elif schedule == "equal_principal":
        level = Fraction(0)
        parts = [Fraction(s) for s in allocate(_to_amount(p, exp), n, currency).shares]
    else:
        raise MoneyInputError(f"schedule must be 'annuity' or 'equal_principal', got {schedule!r}")

    rows: list[ScheduleRow] = []
    raw: list[tuple[Fraction, Fraction, Fraction]] = []
    bal = p
    for k in range(1, n + 1):
        interest = _round_fraction(bal * r, unit, m)
        princ = level - interest if parts is None else parts[k - 1]
        if k == n or princ >= bal:
            princ = bal
        pay = interest + princ
        bal -= princ
        raw.append((pay, interest, princ))
        rows.append(ScheduleRow(period=k, payment=_to_amount(pay, exp), interest=_to_amount(interest, exp),
                                principal=_to_amount(princ, exp), balance=_to_amount(bal, exp)))
        if bal == 0:
            break
    total_paid = sum((x[0] for x in raw), Fraction(0))
    total_int = sum((x[1] for x in raw), Fraction(0))
    total_princ = sum((x[2] for x in raw), Fraction(0))
    checks = [
        _certify(total_princ == p, "principal parts sum to the loan exactly"),
        _certify(total_paid == total_int + total_princ, "payments == interest + principal"),
        _certify(all(Fraction(row.payment) == Fraction(row.interest) + Fraction(row.principal) for row in rows),
                 "every row: payment == interest + principal"),
        _certify(all(row.balance >= 0 for row in rows) and rows[-1].balance == 0, "balance never negative, ends at 0"),
    ]
    regular = raw[0][0]
    naive = regular * n
    return AmortizationSchedule(
        principal=_to_amount(p, exp), periodic_rate=r, periods=n, periods_used=len(rows), schedule=schedule,
        currency=_currency_label(currency), mode=m, rows=tuple(rows), regular_payment=_to_amount(regular, exp),
        last_payment=_to_amount(raw[-1][0], exp), total_paid=_to_amount(total_paid, exp),
        total_interest=_to_amount(total_int, exp), naive_total=_to_amount(naive, exp),
        naive_difference=_to_amount(total_paid - naive, exp), checks=tuple(checks),
    )


@dataclass(frozen=True)
class NPVResult(_ToDict):
    """Net present value, exact. ``cashflows[0]`` is at time 0 and NOT discounted
    (numpy-financial's convention). ``excel_convention`` is the value Excel's
    ``NPV()`` returns for the same list — it discounts the first flow too, a
    frequent source of "my NPV is off" reports."""

    kind = "npv"
    rate: Fraction
    cashflows: tuple[Fraction, ...]
    exact: Fraction
    rounded: "Decimal | None"
    excel_convention: Fraction
    checks: tuple[str, ...]


def npv(
    rate: Exact, cashflows: Sequence[Exact], *, currency: "str | int | None" = None, mode: str = "HALF_UP"
) -> NPVResult:
    """``sum(c_t / (1 + rate) ** t)``, exactly.

        >>> str(npv("0.1", ["-100", "39", "59", "55", "20"]).exact)
        '573890/14641'
    """
    r = _exact(rate, "rate")
    if r <= -1:
        raise MoneyInputError("npv: rate must be greater than -100%")
    cfs = [_exact(c, f"cashflow {t}") for t, c in enumerate(cashflows)]
    if not cfs:
        raise MoneyInputError("npv: no cash flows")
    direct = sum((c / (1 + r) ** t for t, c in enumerate(cfs)), Fraction(0))
    horner = Fraction(0)
    for c in reversed(cfs):
        horner = horner / (1 + r) + c
    checks = [_certify(direct == horner, "direct sum == Horner evaluation (exact)")]
    rounded = None
    if currency is not None:
        exp = currency_exponent(currency)
        rounded = _to_amount(_round_fraction(direct, Fraction(1, 10**exp), _mode(mode)), exp)
    return NPVResult(rate=r, cashflows=tuple(cfs), exact=direct, rounded=rounded,
                     excel_convention=direct / (1 + r), checks=tuple(checks))


@dataclass(frozen=True)
class IRRResult(_ToDict):
    """Every internal rate of return of a cash-flow list, each with a proof.

    An IRR is a rate ``r > -100%`` with NPV(r) = 0, i.e. a positive root ``x`` of
    ``sum(c_t * x**t)`` with ``x = 1/(1+r)``. ALL such roots are found by exact
    real-root isolation on the integer polynomial; each is reported to ``digits``
    places with ``brackets`` — an exact interval proven to contain it (sign change
    of the square-free factor at both ends). ``count`` is cross-checked by a Sturm
    count and by Descartes' rule of signs: ``count <= sign_changes``, and counted
    with ``multiplicities`` it has the same parity.

    ``rate`` is set only when the IRR is unique; with several (``unique=False``)
    the question "what is the IRR?" has no single answer and this says so rather
    than picking one; with none, ``rates`` is empty."""

    kind = "irr"
    cashflows: tuple[Fraction, ...]
    rates: tuple[Decimal, ...]
    brackets: tuple[tuple[Fraction, Fraction], ...]
    multiplicities: tuple[int, ...]
    count: int
    unique: bool
    rate: "Decimal | None"
    sign_changes: int
    digits: int
    checks: tuple[str, ...]


def irr(cashflows: Sequence[Exact], *, digits: int = 12) -> IRRResult:
    """All internal rates of return, proven, with the uniqueness question answered.

        >>> r = irr(["-100", "39", "59", "55", "20"])
        >>> str(r.rate), r.unique
        ('0.280948421160', True)
        >>> [str(x) for x in irr(["-100", "230", "-132"]).rates]   # two IRRs: 10% and 20%
        ['0.100000000000', '0.200000000000']
    """
    import sympy as sp

    nd = _count(digits, "digits", minimum=1)
    if nd > 50:
        raise MoneyInputError("irr: digits must be at most 50")
    cfs = [_exact(c, f"cashflow {t}") for t, c in enumerate(cashflows)]
    if not any(cfs):
        raise MoneyInputError("irr: every rate is an IRR of an all-zero cash flow list")
    lcm = math.lcm(*(c.denominator for c in cfs))
    coeffs = [int(c * lcm) for c in cfs]  # coefficient of x**t
    signs = [c > 0 for c in coeffs if c != 0]
    changes = sum(a != b for a, b in zip(signs, signs[1:]))
    target = Fraction(1, 10 ** (nd + 1))

    def horner(cs: Sequence[int], x: Fraction) -> Fraction:
        acc = Fraction(0)
        for c in reversed(cs):
            acc = acc * x + c
        return acc

    def run() -> IRRResult:
        x = sp.Symbol("x")
        poly = sp.Poly(list(reversed(coeffs)), x, domain="ZZ")
        found: list[tuple[Fraction, Fraction, int]] = []
        sturm_total = 0
        for factor, mult in poly.sqf_list()[1]:
            fc = [int(c) for c in reversed(factor.all_coeffs())]
            sturm_total += int(factor.count_roots(0, None)) - (1 if fc[0] == 0 else 0)
            for (a_s, b_s), _ in factor.intervals(inf=0):
                a, b = Fraction(int(a_s.p), int(a_s.q)), Fraction(int(b_s.p), int(b_s.q))
                if a == b == 0:
                    continue  # x = 0 is r = infinity, not a rate
                while a != b and (a <= 0 or (b - a) / (a * b) > target):
                    eps = min(b - a, max(a, (b + a) / 4) ** 2 * target) / 4
                    a_s, b_s = factor.refine_root(a_s, b_s, eps=sp.Rational(eps.numerator, eps.denominator))
                    a, b = Fraction(int(a_s.p), int(a_s.q)), Fraction(int(b_s.p), int(b_s.q))
                if a == b:
                    _certify(horner(fc, a) == 0, f"exact rational root at x = {a}")
                else:
                    _certify(horner(fc, a) * horner(fc, b) < 0, f"sign change on [{a}, {b}]")
                found.append((a, b, int(mult)))
        checks = ["sign change (or exact zero) of the square-free factor on every bracket"]
        checks.append(_certify(len(found) == sturm_total, "root count == Sturm count on (0, inf)"))
        with_mult = sum(k for _, _, k in found)
        checks.append(_certify(with_mult <= changes and (changes - with_mult) % 2 == 0,
                               "Descartes' rule of signs: bound and parity"))
        found.sort(key=lambda t: -t[0])  # largest x = smallest rate
        brackets = tuple((1 / b - 1, 1 / a - 1) for a, b, _ in found)
        quantum = _units_to_decimal(1, nd)
        rates = []
        for lo, hi in brackets:
            mid = (lo + hi) / 2
            rates.append(_to_amount(_round_fraction(mid, Fraction(1, 10**nd), "HALF_EVEN"), nd))
            _certify(abs(Fraction(rates[-1]) - lo) <= Fraction(quantum) and abs(Fraction(rates[-1]) - hi)
                     <= Fraction(quantum), f"reported rate within 10^-{nd} of every point of its bracket")
        checks.append(f"each rate within 10^-{nd} of the true root")
        return IRRResult(
            cashflows=tuple(cfs), rates=tuple(rates), brackets=brackets,
            multiplicities=tuple(k for _, _, k in found), count=len(found), unique=len(found) == 1,
            rate=rates[0] if len(found) == 1 else None, sign_changes=changes, digits=nd, checks=tuple(checks),
        )

    return guard("irr", f"{len(cfs)} cash flows", run)
