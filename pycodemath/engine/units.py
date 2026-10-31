"""Units and dimensions as a VERIFICATION SIGNAL (module J1).

Converting ``5 km/h`` to ``m/s`` is the easy half. The half that matters to this
package is the other one: a calculation whose dimensions do not agree is wrong
WITHOUT evaluating a single number, and that is a refutation a verifier can make
about a step (``F = m*v``) or a line of code (``t_ms + t_s``) that no sampling of
values would ever produce.

Public surface::

    convert("5 km/h", "m/s")              -> Conversion   25/18 m/s ≈ 1.38889, certified
    Quantity.parse("20 degC")             -> Quantity     exact value + unit; arithmetic
    dimension_of("m*v**2/2", units)       -> DimensionReport   or DimensionError (located)
    check_dimensions("F", "m*v", units)   -> DimensionCheck    VERIFIED / REFUTED / UNDECIDED
    units_from_names(["t_ms", "t_s"])     -> NameUnits    suffix convention -> units
    constant("c")                         -> Constant     SI 2019 exact / CODATA 2022

--- EXACTNESS ---------------------------------------------------------------

Every factor is an exact SymPy number (``_units_table`` reads ``0.3048`` as
``381/1250``); a value typed by the caller is read the same way. ``convert``
returns the exact value AND the approximation, and its certificate holds three
independent facts: the quotient of the two dimensions is dimensionless, the
conversion back reproduces the input exactly, and — where every unit involved
exists there with an exact definition — ``sympy.physics.units``, a separate
implementation with its own table, computes the same factor.

--- TWO READINGS OF A SYMBOL: ``values="quantities"`` / ``values="numbers"`` -----

In a physics step a symbol IS a physical quantity: ``F`` is a force, and the unit
assigned to it only declares its dimension — ``F`` in N or in lbf is the same
force, so only dimensions can disagree. In code a symbol is a NUMBER IN A UNIT:
``t_ms`` holds a count of milliseconds, and ``t_ms + t_s`` is wrong although
both are times. ``values="numbers"`` tracks the implied unit of every
subexpression as well (``t_s*1000`` is in ms) and refutes a sum, an equation or
a transcendental argument whose operands are in different units of the same
dimension — but ONLY when neither operand carries a numeric coefficient. A
coefficient might be a conversion (``*1000``) or physics (``*2``), and telling
those apart would be guessing; so ``x_m + 2*y_km`` is silently accepted. A
missed bug is better than a false accusation.

--- WHAT A REFUTATION CARRIES ------------------------------------------------

A dimension mismatch is a theorem about units, and it comes with a concrete
witness checked by evaluation (Buckingham's argument made executable): an
assignment of values, then the SAME physical situation measured with one base
unit doubled. A dimensionally consistent equation or sum keeps its truth under
that change; here the two sides scale by different exact powers of 2, evaluated
at 50 digits. No witness, no REFUTED: the result is UNDECIDED with the reason.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping

import sympy as sp

from ..core.budget import guard
from ..core.errors import DomainError, ParseError
from ..verify.verdict import Verdict, VerdictStatus
from . import _units_table as table

__all__ = [
    "Dimension",
    "Unit",
    "Quantity",
    "Conversion",
    "DimensionReport",
    "DimensionCheck",
    "UnitWarning",
    "NameUnits",
    "Constant",
    "UnitParseError",
    "DimensionError",
    "AffineTemperatureError",
    "parse_unit",
    "convert",
    "dimension_of",
    "check_dimensions",
    "units_from_names",
    "read_units_spec",
    "constant",
]

#: Longest text accepted anywhere in this module (a unit, a quantity, a formula).
_MAX_TEXT = 4000
#: Largest exponent accepted in a unit (``m^1000`` is a typo, not a unit).
_MAX_UNIT_POWER = 64


# =============================================================================
# Exceptions — subclasses of the core hierarchy, as the series requires
# =============================================================================


class UnitParseError(ParseError):
    """A unit or quantity text was refused: unknown name, an ambiguous one
    (``KB``, ``month``), or a syntax that admits two readings (``J/kg K``)."""


class DimensionError(DomainError):
    """Dimensions disagree: two terms of a sum, the two sides of an equation, a
    transcendental argument that is not dimensionless, or a conversion between
    incompatible units.

    Carries the evidence as attributes: ``location`` (the subexpression where it
    happened), ``left`` / ``right`` (the two operands that disagree, as text),
    ``left_dimension`` / ``right_dimension`` and, when the error came out of an
    expression, ``witness`` — the rescaling counterexample (see module docstring).
    """

    def __init__(
        self,
        message: str,
        *,
        location: str = "",
        left: str = "",
        right: str = "",
        left_dimension: str = "",
        right_dimension: str = "",
        witness: "dict[str, str] | None" = None,
        kind: str = "dimension",
    ) -> None:
        super().__init__(message)
        self.location = location
        self.left = left
        self.right = right
        self.left_dimension = left_dimension
        self.right_dimension = right_dimension
        self.witness = witness
        self.kind = kind


class AffineTemperatureError(DomainError):
    """An absolute temperature on an offset scale (°C, °F) was scaled, multiplied,
    or added to another absolute temperature.

    20 °C is 293.15 K; doubling it is 586.3 K = 313.15 °C, not 40 °C. Pint refuses
    this too; astropy (8.0.2, measured 30.10.2026) returns ``40 deg_C``. A
    temperature DIFFERENCE is a different unit here (``delta_degC``), and
    ``30 degC - 20 degC`` is exactly ``10 delta_degC``.
    """


# =============================================================================
# Dimension
# =============================================================================

_SYMBOL_OF_BASE = {
    "length": "m", "mass": "kg", "time": "s", "current": "A", "temperature": "K",
    "amount": "mol", "luminous_intensity": "cd", "angle": "rad", "information": "bit",
    "pixel": "px",
}
#: Display order: kg·m/s² reads the way textbooks write it, mass before length.
_DISPLAY_ORDER = ("mass", "length", "time", "current", "temperature", "amount",
                  "luminous_intensity", "angle", "information", "pixel")
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _power_text(sym: str, e: Fraction) -> str:
    if e == 1:
        return sym
    if e.denominator == 1:
        return sym + str(e.numerator).translate(_SUPERSCRIPT)
    return f"{sym}^({e})"


@dataclass(frozen=True)
class Dimension:
    """A physical dimension: a vector of rational exponents over ``table.BASES``.

    >>> Dimension.of(length=1, time=-2).text
    'm/s²'
    >>> (Dimension.of(mass=1) * Dimension.of(length=1, time=-2)).name
    'force'
    """

    exponents: tuple[tuple[str, Fraction], ...] = ()

    @staticmethod
    def of(**exps: "int | Fraction") -> "Dimension":
        for base in exps:
            if base not in table.BASES:
                raise DomainError(f"Dimension: unknown base {base!r}")
        return Dimension._make({b: Fraction(e) for b, e in exps.items()})

    @staticmethod
    def _make(exps: Mapping[str, Fraction]) -> "Dimension":
        return Dimension(tuple((b, exps[b]) for b in table.BASES if exps.get(b, 0) != 0))

    def get(self, base: str) -> Fraction:
        for b, e in self.exponents:
            if b == base:
                return e
        return Fraction(0)

    def _map(self) -> dict[str, Fraction]:
        return dict(self.exponents)

    def __mul__(self, other: "Dimension") -> "Dimension":
        m = self._map()
        for b, e in other.exponents:
            m[b] = m.get(b, Fraction(0)) + e
        return Dimension._make(m)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return self * other ** -1

    def __pow__(self, power: "int | Fraction") -> "Dimension":
        p = Fraction(power)
        return Dimension._make({b: e * p for b, e in self.exponents})

    @property
    def dimensionless(self) -> bool:
        return not self.exponents

    def hard(self) -> "Dimension":
        """This dimension without the soft bases (angle) — what a refutation looks at."""
        return Dimension(tuple((b, e) for b, e in self.exponents if b not in table.SOFT_BASES))

    @property
    def text(self) -> str:
        """``kg·m/s²``, ``1/s``, ``kg·m²/(s³·A)``; ``1`` when dimensionless."""
        num = [(b, e) for b in _DISPLAY_ORDER if (e := self.get(b)) > 0]
        den = [(b, -e) for b in _DISPLAY_ORDER if (e := self.get(b)) < 0]
        top = "·".join(_power_text(_SYMBOL_OF_BASE[b], e) for b, e in num) or "1"
        if not den:
            return top
        bottom = "·".join(_power_text(_SYMBOL_OF_BASE[b], e) for b, e in den)
        return f"{top}/({bottom})" if len(den) > 1 else f"{top}/{bottom}"

    @property
    def name(self) -> "str | None":
        """The common name of this dimension (``force``, ``energy``), if it has one."""
        return _dimension_names().get(self)

    def describe(self) -> str:
        """``kg·m/s² (force)`` — the text, with the name when there is one."""
        n = self.name
        return f"{self.text} ({n})" if n else self.text

    def to_dict(self) -> dict[str, str]:
        return {b: str(e) for b, e in self.exponents}

    def __str__(self) -> str:
        return self.text


# =============================================================================
# Unit and the unit-expression parser
# =============================================================================


@dataclass(frozen=True)
class Unit:
    """A unit: an exact SI factor and a dimension, plus an offset for an absolute
    temperature on an offset scale (``T_SI = factor * v + offset``).

    ``text`` is how the unit was written. ``delta`` names the matching
    temperature-difference unit of an offset scale (``delta_degC`` for ``degC``).
    ``is_delta`` marks a unit that measures temperature DIFFERENCES.
    """

    text: str
    factor: sp.Expr
    dimension: Dimension
    offset: "sp.Expr | None" = None
    delta: "str | None" = None
    is_delta: bool = False
    notes: tuple[str, ...] = ()
    #: The same unit built from ``sympy.physics.units`` objects, or ``None`` when
    #: some piece has no exact counterpart there — the independent cross-check.
    reference: "sp.Expr | None" = field(default=None, compare=False, repr=False)

    @property
    def is_absolute_offset(self) -> bool:
        return self.offset is not None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "unit": self.text,
            "factor_si": str(self.factor),
            "dimension": self.dimension.text,
        }
        if self.offset is not None:
            d["offset_si"] = str(self.offset)
        if self.notes:
            d["notes"] = list(self.notes)
        return d

    def __str__(self) -> str:
        return self.text


_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<sup>[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)
  | (?P<name>[A-Za-z_°℃℉ΩµμÅ\u212b%′″Δ][A-Za-z0-9_°ΩµμÅ\u212b]*)
  | (?P<op>\*\*|[\^*·⋅×/()+\-])
    """,
    re.VERBOSE,
)
_SUP_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _tokens(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise UnitParseError(f"unit {text!r}: cannot read {text[pos:pos + 10]!r}")
        pos = m.end()
        kind = m.lastgroup or ""
        if kind != "ws":
            out.append((kind, m.group()))
        elif out:
            out.append(("ws", " "))
    while out and out[-1][0] == "ws":
        out.pop()
    return out


class _UnitParser:
    """Recursive descent over a unit expression.

    Grammar (``*``, ``·``, ``×`` and a space all multiply; ``/`` divides, left to
    right, so ``J/kg/K`` is ``J/(kg·K)``)::

        expr  := term ( ('*' | '/' | ' ') term )*
        term  := atom ( ('^' | '**') exponent | superscript )?
        atom  := NAME | NUMBER | '(' expr ')'

    ONE reading is refused on purpose: a space right after a division, as in
    ``J/kg K``. Half of all readers mean ``J/(kg·K)`` and the grammar says
    ``J·K/kg``; picking either silently is the kind of answer this package does not
    give, so the text is refused with both readings spelled out.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.toks = _tokens(text)
        self.i = 0
        self.affine_atoms: list[str] = []
        self.atoms = 0
        self.names = 0

    def peek(self) -> tuple[str, str]:
        return self.toks[self.i] if self.i < len(self.toks) else ("end", "")

    def take(self) -> tuple[str, str]:
        t = self.peek()
        self.i += 1
        return t

    def skip_ws(self) -> None:
        while self.peek()[0] == "ws":
            self.i += 1

    def parse(self) -> Unit:
        self.skip_ws()
        if self.peek()[0] == "end":
            return _DIMENSIONLESS
        u = self.expr()
        self.skip_ws()
        if self.peek()[0] != "end":
            raise UnitParseError(f"unit {self.text!r}: unexpected {self.peek()[1]!r}")
        return u

    def expr(self) -> Unit:
        start = self.names
        u = self.term()
        numeric = self.names == start      # no unit name read yet: "5/9 K" is a number
        after_div = False
        while True:
            had_ws = self.peek()[0] == "ws"
            self.skip_ws()
            kind, tok = self.peek()
            if kind == "op" and tok in ("*", "·", "⋅", "×"):
                self.take()
                self.skip_ws()
                before = self.names
                u = _mul(u, self.term())
                numeric = numeric and self.names == before
                after_div = False
            elif kind == "op" and tok == "/":
                self.take()
                self.skip_ws()
                before = self.names
                u = _div(u, self.term())
                numeric = numeric and self.names == before
                # "5/9 K" is a number fraction times a unit; "J/kg K" is ambiguous
                after_div = not numeric
            elif had_ws and kind in ("name", "num") or (kind == "op" and tok == "("):
                if after_div:
                    raise UnitParseError(
                        f"unit {self.text!r} is ambiguous: a space after '/' reads either "
                        f"as a/(b·c) or as (a/b)·c — write the parentheses, e.g. J/(kg*K)"
                    )
                before = self.names
                u = _mul(u, self.term())
                numeric = numeric and self.names == before
            else:
                return u

    def term(self) -> Unit:
        u = self.atom()
        kind, tok = self.peek()
        if kind == "op" and tok in ("^", "**"):
            self.take()
            return _pow(u, self.exponent())
        if kind == "sup":
            self.take()
            return _pow(u, Fraction(int(tok.translate(_SUP_DIGITS))))
        return u

    def exponent(self) -> Fraction:
        sign = 1
        kind, tok = self.peek()
        if kind == "op" and tok in ("-", "+"):
            self.take()
            sign = -1 if tok == "-" else 1
            kind, tok = self.peek()
        if kind == "num":
            self.take()
            return sign * Fraction(tok)
        if kind == "op" and tok == "(":
            self.take()
            s2 = 1
            if self.peek() == ("op", "-"):
                self.take()
                s2 = -1
            k1, n1 = self.take()
            value = Fraction(n1) if k1 == "num" else None
            if value is not None and self.peek() == ("op", "/"):
                self.take()
                k2, n2 = self.take()
                value = value / Fraction(n2) if k2 == "num" else None
            if value is None or self.take() != ("op", ")"):
                raise UnitParseError(f"unit {self.text!r}: exponent must be a number like 2 or (1/2)")
            return sign * s2 * value
        raise UnitParseError(f"unit {self.text!r}: exponent must be a number")

    def atom(self) -> Unit:
        kind, tok = self.take()
        if kind == "name":
            self.atoms += 1
            self.names += 1
            u = _lookup(tok)
            if u.offset is not None:
                self.affine_atoms.append(tok)
            return u
        if kind == "num":
            self.atoms += 1
            return Unit(tok, _exact(tok), _ONE, reference=_exact(tok))
        if kind == "op" and tok == "(":
            self.skip_ws()
            u = self.expr()
            self.skip_ws()
            if self.take() != ("op", ")"):
                raise UnitParseError(f"unit {self.text!r}: missing ')'")
            return Unit(f"({u.text})", u.factor, u.dimension, u.offset, u.delta,
                        u.is_delta, u.notes, u.reference)
        raise UnitParseError(f"unit {self.text!r}: unexpected {tok or 'end of text'!r}")


_ONE = Dimension()


def _exact(text: str) -> sp.Expr:
    """A decimal literal as the exact rational it spells: '0.1' -> 1/10."""
    return sp.Rational(text)


def _combine_notes(a: Unit, b: Unit) -> tuple[str, ...]:
    return tuple(dict.fromkeys(a.notes + b.notes))


def _as_delta(u: Unit) -> Unit:
    """An offset-scale unit inside a compound unit: only a DIFFERENCE makes sense."""
    if u.offset is None:
        return u
    note = (f"{u.text} inside a compound unit is read as a temperature difference "
            f"(1 {u.text} step = {u.factor} K)")
    return Unit(u.delta or u.text, u.factor, u.dimension, None, None, True,
                u.notes + (note,), u.reference)


def _ref_op(a: "sp.Expr | None", b: "sp.Expr | None", op: Callable[[Any, Any], Any]) -> "sp.Expr | None":
    return None if a is None or b is None else op(a, b)


def _mul(a: Unit, b: Unit) -> Unit:
    a, b = _as_delta(a), _as_delta(b)
    return Unit(f"{a.text}*{b.text}", a.factor * b.factor, a.dimension * b.dimension,
                notes=_combine_notes(a, b), is_delta=a.is_delta or b.is_delta,
                reference=_ref_op(a.reference, b.reference, lambda x, y: x * y))


def _div(a: Unit, b: Unit) -> Unit:
    a, b = _as_delta(a), _as_delta(b)
    return Unit(f"{a.text}/{b.text}", a.factor / b.factor, a.dimension / b.dimension,
                notes=_combine_notes(a, b), is_delta=a.is_delta or b.is_delta,
                reference=_ref_op(a.reference, b.reference, lambda x, y: x / y))


def _pow(a: Unit, p: Fraction) -> Unit:
    if abs(p) > _MAX_UNIT_POWER:
        raise UnitParseError(f"unit exponent {p} is out of range (|p| <= {_MAX_UNIT_POWER})")
    a = _as_delta(a)
    rp = sp.Rational(p.numerator, p.denominator)
    return Unit(f"{a.text}^{p}" if p.denominator == 1 else f"{a.text}^({p})",
                a.factor ** rp, a.dimension ** p, notes=a.notes, is_delta=a.is_delta,
                reference=None if a.reference is None else a.reference ** rp)


def _finish(parser: _UnitParser, u: Unit) -> Unit:
    """Restore the offset when the WHOLE unit is one offset-scale name."""
    if parser.affine_atoms and parser.atoms == 1 and not u.is_delta:
        return _lookup(parser.affine_atoms[0])
    return u


def parse_unit(text: str) -> Unit:
    """Read a unit expression: ``"km/h"``, ``"kg*m^2/s^2"``, ``"J/(kg*K)"``, ``"°C"``.

    >>> parse_unit("km/h").factor
    5/18
    >>> parse_unit("kWh").dimension.name
    'energy'
    """
    if not isinstance(text, str):
        raise UnitParseError(f"unit must be text, not {type(text).__name__}")
    if len(text) > _MAX_TEXT:
        raise UnitParseError(f"unit text is too long ({len(text)} characters)")
    return _parse_unit(text.strip())


@lru_cache(maxsize=4096)
def _parse_unit(stripped: str) -> Unit:
    if stripped in ("", "1", "dimensionless", "-"):
        return _DIMENSIONLESS
    p = _UnitParser(stripped)
    try:
        u = _finish(p, p.parse())
    except RecursionError:
        raise UnitParseError(f"unit {stripped[:40]!r}…: nested too deeply") from None
    if not (u.factor.is_finite and u.factor.is_positive):
        raise UnitParseError(f"unit {stripped!r} has factor {u.factor}; a unit must be a "
                             f"finite positive multiple of its SI unit")
    return Unit(stripped, u.factor, u.dimension, u.offset, u.delta, u.is_delta, u.notes, u.reference)


# --- the table -----------------------------------------------------------------

_SPECS: dict[str, table.UnitSpec] = {}
for _spec in table.UNITS:
    for _n in (_spec.name, *_spec.aliases):
        if _n in _SPECS:  # pragma: no cover - guarded by tests
            raise RuntimeError(f"duplicate unit name {_n!r}")
        _SPECS[_n] = _spec

#: Units with an exact, current definition in ``sympy.physics.units`` — the
#: independent implementation the certificate cross-checks against. Listed by
#: hand after reading each value (30.10.2026, SymPy 1.14): ``astronomical_unit``
#: (149597870691, pre-2012), ``electronvolt`` / ``psi`` / ``mmHg`` (floats),
#: ``magnetic_constant`` (pre-2019) and ``nautical_mile`` (1157478/625 m =
#: 6076 ft; the international nautical mile is 1852 m — found by this very
#: cross-check, 30.10.2026) are deliberately absent.
_SYMPY_REF: dict[str, str] = {
    "m": "meter", "kg": "kilogram", "g": "gram", "s": "second", "A": "ampere",
    "K": "kelvin", "mol": "mole", "cd": "candela", "min": "minute", "h": "hour",
    "d": "day", "in": "inch", "ft": "foot", "yd": "yard", "mi": "mile", "lb": "pound", "L": "liter", "ha": "hectare",
    "angstrom": "angstrom", "ly": "lightyear", "deg": "degree", "rad": "radian",
    "Hz": "hertz", "N": "newton", "Pa": "pascal", "J": "joule", "W": "watt",
    "C": "coulomb", "V": "volt", "ohm": "ohm", "F": "farad", "H": "henry",
    "T": "tesla", "Wb": "weber", "S": "siemens", "atm": "atmosphere", "bar": "bar",
    "bit": "bit", "B": "byte", "yr": "julian_year", "Bq": "becquerel", "Gy": "gray",
}


def _sympy_unit(name: str) -> "sp.Expr | None":
    from sympy.physics import units as su

    attr = _SYMPY_REF.get(name)
    return getattr(su, attr, None) if attr else None


_RESOLVING: set[str] = set()


@lru_cache(maxsize=None)
def _resolve_spec(name: str) -> Unit:
    spec = _SPECS[name]
    if spec.definition.startswith("base:"):
        base = spec.definition[5:]
        u = Unit(spec.name, sp.Integer(1), Dimension.of(**{base: 1}))
    else:
        if spec.name in _RESOLVING:  # pragma: no cover - table invariant
            raise RuntimeError(f"cyclic unit definition {spec.name!r}")
        _RESOLVING.add(spec.name)
        try:
            u = _parse_definition(spec.definition)
        finally:
            _RESOLVING.discard(spec.name)
    offset = sp.Rational(spec.offset) if spec.offset is not None else None
    delta = None
    if offset is not None:
        delta = "delta_" + spec.name
    notes = (spec.note,) if spec.note else ()
    is_delta = spec.name.startswith("delta_")
    return Unit(name, u.factor, u.dimension, offset, delta, is_delta, notes, _sympy_unit(spec.name))


def _parse_definition(definition: str) -> Unit:
    """Table definitions may contain pi (``pi/180 rad``)."""
    if "pi" in definition:
        head, _, rest = definition.partition(" ")
        coeff = sp.sympify(head, rational=True)
        u = parse_unit(rest)
        return Unit(definition, coeff * u.factor, u.dimension, reference=None)
    return parse_unit(definition)


@lru_cache(maxsize=4096)
def _lookup(token: str) -> Unit:
    if token in table.REFUSED_NAMES:
        raise UnitParseError(f"unit {token!r} refused: {table.REFUSED_NAMES[token]}")
    if token in _SPECS:
        return _resolve_spec(token)
    candidates: list[Unit] = []
    for prefixes, kinds in ((table.SI_PREFIXES, ("si", "si+binary")),
                            (table.BINARY_PREFIXES, ("si+binary",))):
        for p, scale in prefixes.items():
            rest = token[len(p):]
            if token.startswith(p) and rest in _SPECS and _SPECS[rest].prefixes in kinds:
                base = _resolve_spec(rest)
                f = sp.sympify(scale)
                ref = None if base.reference is None else f * base.reference
                candidates.append(Unit(token, f * base.factor, base.dimension,
                                       notes=base.notes, reference=ref))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:  # pragma: no cover - the table has no such pair today
        raise UnitParseError(f"unit {token!r} is ambiguous between prefix readings")
    close = difflib.get_close_matches(token, list(_SPECS), n=3, cutoff=0.75)
    hint = f" — did you mean {', '.join(close)}?" if close else ""
    raise UnitParseError(f"unknown unit {token!r}{hint}")


_DIMENSIONLESS = Unit("1", sp.Integer(1), _ONE, reference=sp.Integer(1))


@lru_cache(maxsize=1)
def _dimension_names() -> dict[Dimension, str]:
    names = {
        "m/s": "velocity", "m/s^2": "acceleration", "N": "force", "J": "energy",
        "W": "power", "Pa": "pressure", "kg*m/s": "momentum", "1/s": "frequency",
        "C": "charge", "V": "voltage", "ohm": "resistance", "F": "capacitance",
        "Wb": "magnetic flux", "T": "magnetic flux density", "H": "inductance",
        "m^2": "area", "m^3": "volume", "kg/m^3": "density", "J*s": "action",
        "rad/s": "angular velocity", "m": "length", "kg": "mass", "s": "time",
        "K": "temperature", "A": "current", "mol": "amount of substance",
        "rad": "angle", "bit": "information", "px": "pixel count",
        "J/K": "entropy", "N*m/rad": "torque",
    }
    return {parse_unit(u).dimension: n for u, n in names.items()}


# =============================================================================
# Quantity
# =============================================================================

_NUMBER_HEAD = re.compile(
    r"^\s*(?P<sign>[-+])?\s*(?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*/\s*(?P<den>\d+)(?![\d.]))?"
)


def _approx(value: sp.Expr, digits: int = 15) -> str:
    v = sp.N(value, digits)
    text = str(v)
    if "e" not in text and "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


@dataclass(frozen=True)
class Quantity:
    """An exact value with a unit.

    >>> q = Quantity.parse("5 km/h")
    >>> q.to("m/s").value
    25/18
    >>> (Quantity.parse("30 degC") - Quantity.parse("20 degC")).unit.text
    'delta_degC'
    """

    value: sp.Expr
    unit: Unit

    @staticmethod
    def parse(text: str) -> "Quantity":
        """``"5 km/h"``, ``"-40 °C"``, ``"3/4 in"``, ``"1.5e3 m"``, ``"km/h"`` (= 1 km/h)."""
        if not isinstance(text, str):
            raise UnitParseError(f"quantity must be text, not {type(text).__name__}")
        if len(text) > _MAX_TEXT:
            raise UnitParseError(f"quantity text is too long ({len(text)} characters)")
        if re.match(r"\s*[-+]?\s*(nan|inf)", text, re.IGNORECASE):
            raise DomainError(f"quantity {text!r}: the value is not a finite number")
        m = _NUMBER_HEAD.match(text)
        if m:
            value: sp.Expr = _exact(m.group("num"))
            if m.group("den"):
                den = int(m.group("den"))
                if den == 0:
                    raise UnitParseError(f"quantity {text!r}: division by zero")
                value = value / den
            if m.group("sign") == "-":
                value = -value
            rest = text[m.end():]
        else:
            value, rest = sp.Integer(1), text
        if rest.lstrip().startswith("/"):
            rest = "1" + rest.lstrip()
        if not rest.strip() and not m:
            raise UnitParseError(f"quantity {text!r} is empty")
        q = Quantity(value, parse_unit(rest.strip()))
        _below_absolute_zero(q.si_value(), q)
        return q

    @property
    def dimension(self) -> Dimension:
        return self.unit.dimension

    def si_value(self) -> sp.Expr:
        """The value in coherent SI base units (absolute, offsets applied)."""
        v = self.unit.factor * self.value
        return v + self.unit.offset if self.unit.offset is not None else v

    def to(self, target: "str | Unit") -> "Quantity":
        return convert(self, target).quantity

    # --- arithmetic -------------------------------------------------------------
    def _refuse_scaling(self, other: object, op: str) -> None:
        for q in (self, other):
            if isinstance(q, Quantity) and q.unit.offset is not None:
                k = q.si_value()
                raise AffineTemperatureError(
                    f"{q} {op} …: {q} is an absolute temperature on an offset scale "
                    f"({q} = {_approx(k)} K), so scaling it has no meaning in "
                    f"{q.unit.text} — convert to K first, or use {q.unit.delta} for a "
                    f"temperature difference"
                )

    def __mul__(self, other: object) -> "Quantity":
        self._refuse_scaling(other, "*")
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, _mul(self.unit, other.unit))
        return Quantity(self.value * _as_number(other), self.unit)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "Quantity":
        self._refuse_scaling(other, "/")
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, _div(self.unit, other.unit))
        return Quantity(self.value / _as_number(other), self.unit)

    def __rtruediv__(self, other: object) -> "Quantity":
        self._refuse_scaling(other, "/")
        return Quantity(_as_number(other) / self.value, _div(_DIMENSIONLESS, self.unit))

    def __pow__(self, p: "int | Fraction") -> "Quantity":
        self._refuse_scaling(None, "**")
        fp = Fraction(p)
        return Quantity(self.value ** sp.Rational(fp.numerator, fp.denominator), _pow(self.unit, fp))

    def __neg__(self) -> "Quantity":
        self._refuse_scaling(None, "negated")
        return Quantity(-self.value, self.unit)

    def __add__(self, other: object) -> "Quantity":
        return self._combine(other, +1)

    def __sub__(self, other: object) -> "Quantity":
        return self._combine(other, -1)

    def _combine(self, other: object, sign: int) -> "Quantity":
        if not isinstance(other, Quantity):
            other = Quantity(_as_number(other), _DIMENSIONLESS)
        op = "+" if sign > 0 else "-"
        if self.dimension != other.dimension:
            raise DimensionError(
                f"{self} {op} {other}: dimensions differ — {self.dimension.describe()} "
                f"vs {other.dimension.describe()}",
                location=f"{self} {op} {other}", left=str(self), right=str(other),
                left_dimension=self.dimension.text, right_dimension=other.dimension.text,
            )
        a_abs, b_abs = self.unit.offset is not None, other.unit.offset is not None
        if a_abs and b_abs:
            if sign > 0:
                raise AffineTemperatureError(
                    f"{self} + {other}: two absolute temperatures cannot be added "
                    f"({_approx(self.si_value())} K + {_approx(other.si_value())} K is not a "
                    f"temperature of anything) — subtract them for a difference, or add a "
                    f"{self.unit.delta}"
                )
            # abs - abs -> a difference, in this scale's delta unit
            diff = (self.si_value() - other.si_value()) / self.unit.factor
            return Quantity(diff, _lookup(self.unit.delta or "K"))
        if b_abs and not a_abs:
            if sign < 0:
                raise AffineTemperatureError(
                    f"{self} - {other}: subtracting an absolute temperature from a "
                    f"difference has no meaning"
                )
            return other._combine(self, +1)
        # a_abs (+/- delta) or neither: express other in self's unit, as a step
        step = other.value * other.unit.factor / self.unit.factor
        return Quantity(self.value + sign * step, self.unit)

    def __str__(self) -> str:
        return f"{self.value} {self.unit.text}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {"value": str(self.value), "approx": _approx(self.value), **self.unit.to_dict()}


def _below_absolute_zero(si: sp.Expr, q: "Quantity", unit: "Unit | None" = None) -> None:
    """An absolute temperature below 0 K is no temperature at all."""
    involved = q.unit.offset is not None or (unit is not None and unit.offset is not None)
    if involved and q.unit.dimension == Dimension.of(temperature=1) and si < 0:
        raise AffineTemperatureError(
            f"{q} is below absolute zero ({_approx(si)} K) — no absolute temperature "
            f"is; if this is a temperature difference, write delta_degC / delta_degF / K"
        )


def _as_number(x: object) -> sp.Expr:
    if isinstance(x, bool):
        raise DomainError("a bool is not a number")
    if isinstance(x, int):
        return sp.Integer(x)
    if isinstance(x, Fraction):
        return sp.Rational(x.numerator, x.denominator)
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise DomainError(f"{x!r} is not a finite number")
        return sp.Rational(repr(x))
    if isinstance(x, sp.Expr) and x.is_number:
        return x
    raise DomainError(f"cannot combine a quantity with {type(x).__name__}")


# =============================================================================
# convert
# =============================================================================


@dataclass(frozen=True)
class Conversion:
    """The certified result of ``convert``.

    ``value`` is exact (a SymPy number), ``approx`` its 15-digit decimal.
    ``certificate`` holds the independent checks (see ``convert``); ``notes``
    states every interpretation that was made (a °C inside a compound unit read
    as a difference, ``kB`` = 1000 B, ``yr`` = Julian year).
    """

    source: Quantity
    quantity: Quantity
    factor: sp.Expr
    offset: "sp.Expr | None"
    certificate: dict[str, Any]
    notes: tuple[str, ...]
    status: VerdictStatus = VerdictStatus.VERIFIED

    @property
    def value(self) -> sp.Expr:
        return self.quantity.value

    @property
    def value_text(self) -> str:
        return str(self.quantity.value)

    @property
    def approx(self) -> str:
        return _approx(self.quantity.value)

    def as_verdict(self) -> Verdict:
        """The conversion as a ``Verdict`` (X1 certificates expect one)."""
        cc = self.certificate["cross_check"]
        extra = "" if cc["agrees"] is None else f"; {cc['by']} agrees: {cc['agrees']}"
        return Verdict(VerdictStatus.VERIFIED, "exact-conversion", None,
                       f"{self.source} = {self} (dimension ratio 1, exact round trip{extra})")

    def __str__(self) -> str:
        exact, approx = self.value_text, self.approx
        unit = self.quantity.unit.text
        return f"{exact} {unit}" if exact == approx else f"{exact} {unit} ≈ {approx} {unit}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "convert",
            "status": self.status.value,
            "source": str(self.source),
            "value": self.value_text,
            "approx": self.approx,
            "unit": self.quantity.unit.text,
            "factor": str(self.factor),
            "offset": None if self.offset is None else str(self.offset),
            "certificate": self.certificate,
            "notes": list(self.notes),
        }


def _temperature_kind(u: Unit) -> str:
    if u.offset is not None:
        return "absolute-offset"
    if u.is_delta:
        return "delta"
    return "plain"


def convert(quantity: "str | Quantity", target: "str | Unit") -> Conversion:
    """Convert exactly, with a certificate.

    >>> c = convert("5 km/h", "m/s")
    >>> str(c)
    '25/18 m/s ≈ 1.38888888888889 m/s'
    >>> c.certificate["round_trip"], c.certificate["cross_check"]["agrees"]
    (True, True)

    Refusals: a different dimension (``DimensionError``; angle counts, so
    ``rad/s`` -> ``Hz`` is refused: omega = 2*pi*f), an absolute temperature to a
    difference unit or back (``AffineTemperatureError``), unknown or ambiguous
    unit text (``UnitParseError``).
    """
    src = quantity if isinstance(quantity, Quantity) else Quantity.parse(quantity)
    dst = target if isinstance(target, Unit) else parse_unit(target)
    return guard("convert", f"{src} -> {dst.text}", lambda: _convert(src, dst))


def _convert(src: Quantity, dst: Unit) -> Conversion:
    su = src.unit
    if su.dimension != dst.dimension:
        hint = ""
        if su.dimension.hard() == dst.dimension.hard():
            hint = (" — they differ only by an angle; SI calls the radian 1, but rad/s and "
                    "Hz differ by 2*pi (omega = 2*pi*f), so convert the angle explicitly")
        raise DimensionError(
            f"cannot convert {src} to {dst.text}: {su.dimension.describe()} vs "
            f"{dst.dimension.describe()}{hint}",
            location=f"{src} -> {dst.text}", left=su.text, right=dst.text,
            left_dimension=su.dimension.text, right_dimension=dst.dimension.text,
        )
    ks, kd = _temperature_kind(su), _temperature_kind(dst)
    if {ks, kd} == {"absolute-offset", "delta"}:
        raise AffineTemperatureError(
            f"cannot convert {src} to {dst.text}: one is an absolute temperature and the "
            f"other a temperature difference"
        )
    _below_absolute_zero(src.si_value(), src, dst)
    factor = su.factor / dst.factor
    if not factor.is_Rational:
        factor = sp.simplify(factor)
    offset: "sp.Expr | None" = None
    if su.offset is not None or dst.offset is not None:
        o_s = su.offset if su.offset is not None else sp.Integer(0)
        o_d = dst.offset if dst.offset is not None else sp.Integer(0)
        offset = (o_s - o_d) / dst.factor
        value = sp.simplify(factor * src.value + offset)
    else:
        value = factor * src.value
        if not value.is_Rational:
            value = sp.simplify(value)
    result = Quantity(value, dst)
    # --- certificate: three independent facts --------------------------------
    back = value * dst.factor + (dst.offset or 0) - (su.offset or 0)
    round_trip = sp.simplify(back / su.factor - src.value) == 0
    cert: dict[str, Any] = {
        "dimension_source": su.dimension.text,
        "dimension_target": dst.dimension.text,
        "dimension_ratio": (su.dimension / dst.dimension).text,
        "factor": str(factor),
        "round_trip": bool(round_trip),
        "cross_check": _cross_check(su, dst, factor),
    }
    if offset is not None:
        cert["offset"] = str(offset)
    if not round_trip:  # pragma: no cover - exact arithmetic cannot fail this
        raise DomainError(f"convert: internal round-trip failed for {src} -> {dst.text}")
    notes = tuple(dict.fromkeys(su.notes + dst.notes + _ambiguity_notes(su, dst)))
    return Conversion(src, result, factor, offset, cert, notes)


def _ambiguity_notes(*units: Unit) -> tuple[str, ...]:
    out = []
    for u in units:
        for name in re.findall(r"[A-Za-z]+", u.text):
            if name in ("kB", "MB", "GB", "TB", "PB"):
                out.append(f"{name} is decimal (SI): 1 {name} = 10^{3 * ('kMGTP'.index(name[0]) + 1)} B; "
                           f"the binary unit is {name[0].upper()}iB")
    return tuple(out)


def _cross_check(src: Unit, dst: Unit, factor: sp.Expr) -> dict[str, Any]:
    """Recompute the factor with ``sympy.physics.units`` — a separate table."""
    if src.reference is None or dst.reference is None:
        return {"by": None, "agrees": None,
                "reason": "a unit involved has no exact counterpart in sympy.physics.units"}
    from sympy.physics import units as su
    from sympy.physics.units import convert_to

    bases = [su.meter, su.kilogram, su.second, su.ampere, su.kelvin, su.mole, su.candela, su.bit]
    try:
        ratio = convert_to(src.reference / dst.reference, bases)
        ratio = ratio.subs({b: 1 for b in bases})
        if ratio.atoms(su.Quantity):
            # convert_to does not reduce angles (degree/radian); SymPy's own SI
            # scale factors do, and they are still SymPy's table, not ours.
            from sympy.physics.units.systems.si import SI

            ratio = ratio.subs({q: SI.get_quantity_scale_factor(q) for q in ratio.atoms(su.Quantity)})
        if ratio.atoms(su.Quantity) or not ratio.is_number:
            return {"by": None, "agrees": None, "reason": "sympy could not reduce the units"}
        agrees = sp.simplify(ratio - factor) == 0
    except Exception as exc:  # pragma: no cover - defensive: a foreign library
        return {"by": None, "agrees": None, "reason": f"sympy failed: {type(exc).__name__}"}
    return {"by": "sympy.physics.units", "agrees": bool(agrees), "factor": str(ratio)}


# =============================================================================
# Expressions: text -> SymPy, without eval
# =============================================================================

_FUNCS: dict[str, Callable[..., Any]] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "arcsin": sp.asin, "arccos": sp.acos,
    "arctan": sp.atan, "atan2": sp.atan2, "arctan2": sp.atan2,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh,
    "atanh": sp.atanh, "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "log10": lambda x: sp.log(x, 10), "log2": lambda x: sp.log(x, 2),
    "log1p": lambda x: sp.log(1 + x), "expm1": lambda x: sp.exp(x) - 1,
    "sqrt": sp.sqrt, "cbrt": sp.cbrt, "abs": sp.Abs, "Abs": sp.Abs, "fabs": sp.Abs,
    "floor": sp.floor, "ceil": sp.ceiling, "ceiling": sp.ceiling, "sign": sp.sign,
    "min": sp.Min, "max": sp.Max, "Min": sp.Min, "Max": sp.Max, "erf": sp.erf,
    "hypot": lambda x, y: sp.sqrt(x**2 + y**2),
    "radians": lambda x: x * sp.pi / 180, "deg2rad": lambda x: x * sp.pi / 180,
    "degrees": lambda x: x * 180 / sp.pi, "rad2deg": lambda x: x * 180 / sp.pi,
    "pow": lambda x, y: x ** y,
}
_MODULES = frozenset({"math", "np", "numpy", "cmath", "torch", "sp", "sympy", "mpmath"})
_MAX_POWER = 10_000


class _ToSympy(ast.NodeVisitor):
    """A closed-world translation of a Python expression AST into SymPy.

    Only arithmetic, calls to the whitelisted functions (optionally qualified as
    ``math.sin`` / ``np.exp``) and names are accepted; every name is a plain
    ``Symbol`` — ``E``, ``I``, ``N``, ``S``, ``gamma``, ``lambda_`` are physics
    variables here, not Euler's number, the imaginary unit or SymPy's ``N``.
    ``pi`` is the constant unless a unit was assigned to a symbol called pi.
    """

    def __init__(self, text: str, symbols: Mapping[str, sp.Symbol]) -> None:
        self.text = text
        self.symbols = symbols

    def fail(self, what: str) -> ParseError:
        return UnitParseError(f"expression {self.text!r}: {what}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        a, b = self.visit(node.left), self.visit(node.right)
        op = type(node.op)
        if op is ast.Add:
            return a + b
        if op is ast.Sub:
            return a - b
        if op is ast.Mult:
            return a * b
        if op is ast.Div:
            return a / b
        if op in (ast.Pow, ast.BitXor):
            if b.is_Number and abs(b) > _MAX_POWER:
                raise self.fail(f"exponent {b} is out of range")
            return a ** b
        raise self.fail(f"operator {op.__name__} is not supported")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise self.fail(f"operator {type(node.op).__name__} is not supported")

    def visit_Constant(self, node: ast.Constant) -> Any:
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise self.fail(f"literal {v!r} is not a number")
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            raise self.fail("non-finite literal")
        return sp.Integer(v) if isinstance(v, int) else sp.Rational(repr(v))

    def visit_Name(self, node: ast.Name) -> Any:
        name = node.id
        if name in self.symbols:
            return self.symbols[name]
        if name == "pi":
            return sp.pi
        return sp.Symbol(name)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if isinstance(node.value, ast.Name) and node.value.id in _MODULES and node.attr == "pi":
            return sp.pi
        raise self.fail("attribute access is not supported")

    def visit_Call(self, node: ast.Call) -> Any:
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in _MODULES:
            name = f.attr
        elif isinstance(f, ast.Name):
            name = f.id
        else:
            raise self.fail("only calls of known functions are supported")
        if name not in _FUNCS or node.keywords:
            raise self.fail(f"unknown function {name!r}")
        return _FUNCS[name](*[self.visit(a) for a in node.args])

    def generic_visit(self, node: ast.AST) -> Any:
        raise self.fail(f"{type(node).__name__} is not supported")


def _to_sympy(expr: object, names: Iterable[str] = ()) -> sp.Expr:
    """str / pycodemath Expr / SymPy -> SymPy (strings via the closed AST walk)."""
    if isinstance(expr, sp.Basic):
        return expr  # type: ignore[return-value]
    sy = getattr(expr, "sy", None)
    if isinstance(sy, sp.Basic):
        return sy  # type: ignore[return-value]
    if isinstance(expr, (int, float, Fraction)) and not isinstance(expr, bool):
        return _as_number(expr)
    if not isinstance(expr, str):
        raise UnitParseError(f"cannot read an expression from {type(expr).__name__}")
    text = expr.strip()
    if not text or len(text) > _MAX_TEXT:
        raise UnitParseError(f"expression {text[:40]!r}: empty or too long")
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except RecursionError:
        raise UnitParseError(f"expression {text[:40]!r}…: nested too deeply") from None
    except SyntaxError as exc:
        raise UnitParseError(f"expression {text!r}: {exc.msg}") from None
    symbols = {n: sp.Symbol(n) for n in names}
    try:
        return _ToSympy(text, symbols).visit(tree)  # type: ignore[no-any-return]
    except RecursionError:
        raise UnitParseError(f"expression {text[:40]!r}…: nested too deeply") from None


def _split_equation(text: str) -> tuple[str, str]:
    for sep in ("==", "="):
        if sep in text:
            left, _, right = text.partition(sep)
            if sep == "=" and (left.endswith(("<", ">", "!")) or right.startswith("=")):
                continue
            return left, right
    raise UnitParseError(f"{text!r} is not an equation (no '=')")


# =============================================================================
# Dimensional analysis
# =============================================================================


@dataclass(frozen=True)
class UnitWarning:
    """Something worth a look that is NOT a refutation.

    ``kind``: ``"degrees-to-trig"`` (a value in degrees reaches sin/cos/tan in
    ``numbers`` mode), ``"angle-only"`` (operands differ only by the soft angle
    dimension), ``"affine-temperature"`` (an absolute °C/°F value is scaled or
    two are added), ``"scale-with-coefficient"`` (``numbers`` mode: operands in
    different units, but a numeric coefficient may be the conversion).
    """

    kind: str
    location: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "location": self.location, "message": self.message}


@dataclass(frozen=True)
class _Info:
    dim: "Dimension | None"           # None: unknown
    factor: "sp.Expr | None"          # numbers mode: implied unit factor; None unknown
    coef_free: bool                   # no numeric coefficient other than ±1
    affine: int = 0                   # net count of absolute offset temperatures
    has_degrees: bool = False         # contains a symbol whose unit is a non-radian angle
    wildcard: bool = False            # 0 or an infinity: compatible with every dimension
    #: log() terms whose argument is dimensioned, waiting for a sum to cancel them:
    #: (the log, its argument, the argument's dimension, the weighted dimension).
    #: log(x) - log(x0) is log(x/x0) and fine; log(x) alone is not.
    pending_logs: tuple[tuple[sp.Basic, sp.Basic, Dimension, Dimension], ...] = ()


_TRANSCENDENTAL = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc,
                   sp.sinh, sp.cosh, sp.tanh, sp.coth, sp.asinh, sp.acosh, sp.atanh, sp.erf)
_TRIG = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
_INVERSE_TRIG = (sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc)
_SAME_DIM = (sp.Abs, sp.floor, sp.ceiling, sp.Min, sp.Max, sp.re, sp.im, sp.conjugate)


class _Analyzer:
    def __init__(self, units: Mapping[str, Unit | Dimension | None], numbers: bool) -> None:
        self.units = units
        self.numbers = numbers
        self.warnings: list[UnitWarning] = []
        self.unknown: set[str] = set()
        self.notes: list[str] = []

    def warn(self, kind: str, where: sp.Basic, message: str) -> None:
        w = UnitWarning(kind, str(where), message)
        if w not in self.warnings:
            self.warnings.append(w)

    def run(self, e: sp.Basic) -> _Info:
        """Analyse ``e`` as a value: a log with a dimensioned argument that no sum
        cancelled is an error here."""
        info = self.raw(e)
        if info.pending_logs:
            log_e, arg, arg_dim, _ = info.pending_logs[0]
            self.mismatch(log_e, arg, _Info(arg_dim, None, False), None, None,
                          f"log() needs a dimensionless argument, but {arg} is "
                          f"{arg_dim.describe()}", kind="argument")
        return info

    def raw(self, e: sp.Basic) -> _Info:
        if isinstance(e, sp.Symbol):
            return self.symbol(e)
        if e is sp.zoo or e is sp.nan or e in (sp.oo, -sp.oo):
            return _Info(None, None, False, wildcard=True)
        if e.is_Number or isinstance(e, sp.NumberSymbol):
            if e.is_zero:
                return _Info(None, None, False, wildcard=True)  # 0 has every dimension
            c = abs(e)
            return _Info(_ONE, sp.Integer(1) / c, c == 1)
        if isinstance(e, sp.Add):
            return self.add(e, list(e.args))
        if isinstance(e, sp.Mul):
            return self.mul(e)
        if isinstance(e, sp.Pow):
            return self.pow(e)
        if isinstance(e, sp.core.relational.Relational):
            a, b = self.run(e.lhs), self.run(e.rhs)
            self.same(e, e.lhs, a, e.rhs, b)
            return _Info(_ONE, sp.Integer(1), True)
        if isinstance(e, sp.Derivative):
            f = self.run(e.expr)
            dim, fac = f.dim, f.factor
            for var, n in e.variable_count:
                v = self.run(var)
                dim = None if dim is None or v.dim is None else dim / v.dim ** int(n)
                fac = None if fac is None or v.factor is None else fac / v.factor ** int(n)
            return _Info(dim, fac, f.coef_free)
        if isinstance(e, sp.Integral):
            f = self.run(e.function)
            dim, fac = f.dim, f.factor
            for lim in e.limits:
                v = self.run(lim[0])
                for bound in lim[1:]:
                    self.same(e, lim[0], v, bound, self.run(bound))
                dim = None if dim is None or v.dim is None else dim * v.dim
                fac = None if fac is None or v.factor is None else fac * v.factor
            return _Info(dim, fac, f.coef_free)
        if isinstance(e, sp.Piecewise):
            branches = [arg.expr for arg in e.args]
            return self.add(e, branches, piecewise=True)
        if isinstance(e, (sp.Min, sp.Max)):
            return self.add(e, list(e.args), piecewise=True)
        if isinstance(e, sp.Function):
            return self.function(e)
        return _Info(None, None, False)

    def symbol(self, s: sp.Symbol) -> _Info:
        u = self.units.get(s.name)
        if u is None:
            if s.name in self.units:   # explicitly None = unknown on purpose
                return _Info(None, None, False)
            self.unknown.add(s.name)
            return _Info(None, None, False)
        if isinstance(u, Dimension):
            return _Info(u, None, True)
        degrees = u.dimension.get("angle") != 0 and u.factor != 1
        return _Info(u.dimension, u.factor, True, 1 if u.offset is not None else 0, degrees)

    # -- sums -----------------------------------------------------------------------
    def add(self, e: sp.Basic, terms: list[sp.Basic], piecewise: bool = False) -> _Info:
        infos = [(t, self.raw(t) if not piecewise else self.run(t)) for t in terms]
        pending = tuple(p for _, i in infos for p in i.pending_logs)
        total = _ONE
        for p in pending:
            total = total * p[3]
        if total.hard().dimensionless:
            pending = ()
        infos = [(t, i) for t, i in infos if not i.wildcard]
        known = [(t, i) for t, i in infos if i.dim is not None]
        for k in range(1, len(known)):
            self.same(e, known[0][0], known[0][1], known[k][0], known[k][1])
        if not known:
            return _Info(None, None, False)
        # A term of unknown dimension makes the sum unknown: assuming it matches
        # the others would be VERIFYING something nobody checked.
        dim = known[0][1].dim if len(known) == len(infos) else None
        # affine bookkeeping: abs - abs is a difference; abs + abs is a warning
        net = 0
        for t, i in infos:
            if i.affine:
                net += -i.affine if _is_negated(t) else i.affine
        if net not in (0, 1) and not piecewise:
            self.warn("affine-temperature", e,
                      f"{e}: adds absolute temperatures on an offset scale (°C/°F); only a "
                      f"difference of two of them, or one plus a difference, has a meaning")
        factor, coef_free = self.common_factor(e, known)
        return _Info(dim, factor, coef_free, 1 if net == 1 else 0,
                     any(i.has_degrees for _, i in infos), pending_logs=pending)

    def common_factor(self, e: sp.Basic, known: list[tuple[sp.Basic, _Info]]) -> tuple["sp.Expr | None", bool]:
        if not self.numbers:
            return None, False
        with_f = [(t, i) for t, i in known if i.factor is not None]
        if len(with_f) < len(known) or not with_f:
            return None, False
        free = [(t, i) for t, i in with_f if i.coef_free]
        for t, i in free[1:]:
            self.same_scale(e, free[0][0], free[0][1], t, i)
        ref = free[0][1].factor if free else with_f[0][1].factor
        for t, i in with_f:
            if not i.coef_free and not _eq(i.factor, ref):
                self.warn("scale-with-coefficient", e,
                          f"{e}: terms are in different units of the same dimension; the "
                          f"numeric coefficient in {t} may or may not be the conversion")
        return ref, all(i.coef_free for _, i in with_f)

    # -- products and powers ----------------------------------------------------
    def mul(self, e: sp.Mul) -> _Info:
        dim: "Dimension | None" = _ONE
        fac: "sp.Expr | None" = sp.Integer(1)
        coef_free, degrees, affine = True, False, 0
        # c*log(x) is log(x**c): with only a rational coefficient beside it, a
        # pending log travels up (weighted by c) to a sum that may cancel it.
        coeff, rest = e.as_coeff_Mul()
        if coeff.is_Rational and not isinstance(rest, sp.Mul):
            inner = self.raw(rest)
            if inner.pending_logs:
                c = Fraction(int(coeff.p), int(coeff.q))
                logs = tuple((le, a, d, w ** c) for le, a, d, w in inner.pending_logs)
                return _Info(_ONE, None, False, pending_logs=logs)
        for a in e.args:
            i = self.run(a)
            dim = None if dim is None or i.dim is None else dim * i.dim
            fac = None if fac is None or i.factor is None else fac * i.factor
            coef_free = coef_free and i.coef_free
            degrees = degrees or i.has_degrees
            affine += i.affine
        if affine and not (len(e.args) == 2 and e.args[0] == -1):
            self.warn("affine-temperature", e,
                      f"{e}: an absolute temperature on an offset scale (°C/°F) is "
                      f"multiplied; 20 °C × 2 is 313.15 °C, not 40 °C — use K or a difference")
            affine = 0
        return _Info(dim, fac, coef_free, affine if len(e.args) == 2 else 0, degrees)

    def pow(self, e: sp.Pow) -> _Info:
        base, ex = e.args
        if ex.is_Float:
            ex = sp.Rational(ex)  # the exact binary value; 0.5 -> 1/2
        b = self.run(base)
        x = self.run(ex)
        if x.dim is not None and not x.dim.hard().dimensionless:
            self.mismatch(e, ex, x, None, None, f"the exponent {ex} must be dimensionless, "
                          f"but it is {x.dim.describe()}", kind="argument")
        if b.affine:
            self.warn("affine-temperature", e,
                      f"{e}: an absolute temperature on an offset scale (°C/°F) is raised "
                      f"to a power — use K")
        if ex.is_Rational:
            p = Fraction(int(ex.p), int(ex.q))
            dim = None if b.dim is None else b.dim ** p
            fac = None if b.factor is None else b.factor ** ex
            return _Info(dim, fac, b.coef_free, 0, b.has_degrees)
        if b.dim is not None and b.dim.hard().dimensionless:
            fac = sp.Integer(1) if b.factor is not None and _eq(b.factor, 1) else None
            return _Info(_ONE, fac, b.coef_free)
        return _Info(None, None, False)

    # -- functions -----------------------------------------------------------------
    def function(self, e: sp.Function) -> _Info:
        args = [(a, self.run(a)) for a in e.args]
        func = e.func
        if func is sp.log and len(args) == 1:
            (a, i), = args
            if i.dim is not None and not i.dim.hard().dimensionless:
                return _Info(_ONE, sp.Integer(1), True, pending_logs=((e, a, i.dim, i.dim),))
        if func in _TRANSCENDENTAL or (func is sp.log and len(args) == 2):
            for a, i in args:
                if i.dim is not None and not i.dim.hard().dimensionless:
                    self.mismatch(e, a, i, None, None,
                                  f"{func.__name__}() needs a dimensionless argument, but "
                                  f"{a} is {i.dim.describe()}", kind="argument")
                if self.numbers:
                    self.argument_scale(e, a, i, trig=func in _TRIG)
            return _Info(_ONE, sp.Integer(1), True)
        if func in _INVERSE_TRIG:
            for a, i in args:
                if i.dim is not None and not i.dim.hard().dimensionless:
                    self.mismatch(e, a, i, None, None,
                                  f"{func.__name__}() needs a dimensionless argument, but "
                                  f"{a} is {i.dim.describe()}", kind="argument")
            return _Info(Dimension.of(angle=1), sp.Integer(1), True)
        if func is sp.atan2:
            (a, ia), (b, ib) = args
            self.same(e, a, ia, b, ib)
            return _Info(Dimension.of(angle=1), sp.Integer(1), True)
        if func is sp.sign:
            return _Info(_ONE, sp.Integer(1), True)
        if func in _SAME_DIM:
            return self.add(e, list(e.args), piecewise=True)
        # any other function: dimensionless in, dimensionless out; otherwise unknown
        if all(i.dim is not None and i.dim.dimensionless for _, i in args):
            return _Info(_ONE, None, False)
        return _Info(None, None, False)

    def argument_scale(self, e: sp.Basic, a: sp.Basic, i: _Info, *, trig: bool) -> None:
        if i.factor is None or _eq(i.factor, 1):
            return
        if trig and i.has_degrees and not a.has(sp.pi):
            self.warn("degrees-to-trig", e,
                      f"{e}: {a} is in degrees but {e.func.__name__}() expects radians — "
                      f"multiply by pi/180 (math.radians)")
            return
        if i.coef_free:
            self.scale_mismatch(e, a, i.factor, None, sp.Integer(1),
                                f"{e.func.__name__}() receives {a}, whose value is in units "
                                f"of {_approx(i.factor, 6)} (a dimensionless ratio of different "
                                f"units), not a pure number")

    # -- the two failure paths ------------------------------------------------------
    def same(self, e: sp.Basic, a: sp.Basic, ia: _Info, b: sp.Basic, ib: _Info) -> None:
        if ia.dim is None or ib.dim is None or ia.dim == ib.dim:
            return
        if ia.dim.hard() == ib.dim.hard():
            self.warn("angle-only", e,
                      f"{e}: {a} is {ia.dim.text} and {b} is {ib.dim.text} — they differ only "
                      f"by an angle (radians are dimensionless in SI)")
            return
        self.mismatch(e, a, ia, b, ib, f"{a} is {ia.dim.describe()} but {b} is "
                      f"{ib.dim.describe()}")

    def mismatch(self, e: sp.Basic, a: sp.Basic, ia: _Info, b: "sp.Basic | None",
                 ib: "_Info | None", why: str, kind: str = "dimension") -> None:
        right = b if b is not None else sp.Integer(1)
        right_dim = ib.dim if ib is not None and ib.dim is not None else _ONE
        assert ia.dim is not None
        witness = _rescale_witness(a, ia.dim, right, right_dim, self.units)
        raise DimensionError(
            f"dimension mismatch in {e}: {why}",
            location=str(e), left=str(a), right=str(right),
            left_dimension=ia.dim.text, right_dimension=right_dim.text,
            witness=witness, kind=kind,
        )

    def same_scale(self, e: sp.Basic, a: sp.Basic, ia: _Info, b: sp.Basic, ib: _Info) -> None:
        assert ia.factor is not None and ib.factor is not None
        if ia.dim is None or ib.dim is None or ia.dim != ib.dim or _eq(ia.factor, ib.factor):
            return
        self.scale_mismatch(e, a, ia.factor, b, ib.factor,
                            f"{a} is in units of {_unit_text(ia)} but {b} is in units of "
                            f"{_unit_text(ib)}")

    def scale_mismatch(self, e: sp.Basic, a: sp.Basic, fa: sp.Expr, b: "sp.Basic | None",
                       fb: sp.Expr, why: str) -> None:
        witness = _scale_witness(a, fa, b, fb)
        raise DimensionError(
            f"unit mismatch in {e}: {why}", location=str(e), left=str(a),
            right=str(b) if b is not None else "1", left_dimension="", right_dimension="",
            witness=witness, kind="scale",
        )


def _is_negated(t: sp.Basic) -> bool:
    return isinstance(t, sp.Mul) and t.args[0].is_Number and t.args[0] < 0


def _eq(a: "sp.Expr | None", b: "sp.Expr | int") -> bool:
    if a is None:
        return False
    d = sp.simplify(a - b)
    return bool(d == 0)


def _unit_text(i: _Info) -> str:
    assert i.dim is not None and i.factor is not None
    if _eq(i.factor, 1):
        return i.dim.text
    return f"{_approx(i.factor, 10)} {i.dim.text}"


_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)


def _points(syms: list[sp.Symbol]) -> Iterable[dict[sp.Symbol, sp.Expr]]:
    for shift in range(6):
        yield {s: sp.Rational(_PRIMES[(k + shift) % len(_PRIMES)], 1 + shift % 3)
               for k, s in enumerate(syms)}


def _value(e: sp.Basic, point: Mapping[sp.Symbol, sp.Expr]) -> "sp.Expr | None":
    """``e`` at ``point`` to 50 digits; ``None`` unless a finite non-zero number.
    Complex values are fine: the scaling argument holds for them unchanged."""
    try:
        v = sp.N(e.subs(point), 50)
    except Exception:
        return None
    if not v.is_number or v.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.AccumBounds):
        return None
    return v if v != 0 else None


def _rescale_witness(a: sp.Basic, da: Dimension, b: sp.Basic, db: Dimension,
                     units: Mapping[str, Any]) -> "dict[str, str] | None":
    """Buckingham's argument, executed: double one base unit and evaluate.

    Picks a hard base in which the two dimensions differ, assigns values, then the
    same physical situation measured with that base unit doubled (every value
    with dimension exponent e in that base is divided by 2**e). The two operands
    must scale by 2**-e_a and 2**-e_b; both are evaluated at 50 digits and
    checked. Returns ``None`` when no point evaluates cleanly — the caller then
    reports UNDECIDED instead of a refutation without evidence.
    """
    diff = (da / db).hard()
    if diff.dimensionless:
        return None
    base, _ = diff.exponents[0]
    ea, eb = da.get(base), db.get(base)
    syms = sorted((a.free_symbols | b.free_symbols), key=lambda s: s.name)  # type: ignore[union-attr]

    def exponent(s: sp.Symbol) -> Fraction:
        u = units.get(s.name)
        d = u if isinstance(u, Dimension) else (u.dimension if isinstance(u, Unit) else None)
        return d.get(base) if d is not None else Fraction(0)

    tol = sp.Float(10) ** -35
    for point in _points(syms):
        scaled = {s: v / sp.Integer(2) ** sp.Rational(*_frac(exponent(s))) for s, v in point.items()}
        va, vb = _value(a, point), _value(b, point)
        wa, wb = _value(a, scaled), _value(b, scaled)
        if va is None or vb is None or wa is None or wb is None:
            continue
        ka = sp.Integer(2) ** -sp.Rational(*_frac(ea))
        kb = sp.Integer(2) ** -sp.Rational(*_frac(eb))
        if abs(wa - ka * va) > tol * abs(va) or abs(wb - kb * vb) > tol * abs(vb):
            continue
        sym = _SYMBOL_OF_BASE[base]
        out = {s.name: str(v) for s, v in point.items()}
        out.update({
            "rescale": f"{base}: 1 {sym} -> 2 {sym}",
            "left": str(a), "left_scales_by": str(ka),
            "right": str(b), "right_scales_by": str(kb),
        })
        return out
    return None


def _frac(f: Fraction) -> tuple[int, int]:
    return f.numerator, f.denominator


def _scale_witness(a: sp.Basic, fa: sp.Expr, b: "sp.Basic | None", fb: sp.Expr) -> "dict[str, str] | None":
    """Numbers mode: same dimension, different units — concrete values and the
    exact value a correct conversion would have produced."""
    syms = sorted(a.free_symbols | (b.free_symbols if b is not None else set()),  # type: ignore[union-attr]
                  key=lambda s: s.name)
    for point in _points(syms):
        va = _value(a, point)
        vb = _value(b, point) if b is not None else sp.Integer(1)
        if va is None or vb is None:
            continue
        exact_a = sp.nsimplify(a.subs(point))
        out = {s.name: str(v) for s, v in point.items()}
        out.update({
            "left": str(a), "left_value": str(exact_a), "left_unit_factor": str(fa),
            "right": str(b) if b is not None else "1", "right_unit_factor": str(fb),
            "left_in_right_units": str(sp.nsimplify(exact_a * fa / fb)),
        })
        return out
    return None


@dataclass(frozen=True)
class DimensionReport:
    """The dimension of an expression, with everything that was noticed on the way.

    ``dimension`` is ``None`` when it could not be determined (a symbol without a
    unit, a dimensioned base raised to a symbolic power); ``unknown`` names the
    symbols that had no unit. ``unit_factor`` (``numbers`` mode) is the SI factor
    of the unit the value is in.
    """

    expression: str
    dimension: "Dimension | None"
    unit_factor: "sp.Expr | None"
    warnings: tuple[UnitWarning, ...]
    unknown: tuple[str, ...]
    values: str

    @property
    def text(self) -> "str | None":
        return None if self.dimension is None else self.dimension.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "dimension_of",
            "expression": self.expression,
            "dimension": self.text,
            "dimension_name": None if self.dimension is None else self.dimension.name,
            "exponents": None if self.dimension is None else self.dimension.to_dict(),
            "unit_factor": None if self.unit_factor is None else str(self.unit_factor),
            "warnings": [w.to_dict() for w in self.warnings],
            "unknown": list(self.unknown),
            "values": self.values,
        }


UnitsArg = Mapping[str, "str | Unit | Dimension | Quantity | None"]


def _read_units(units: "UnitsArg | NameUnits | None") -> dict[str, "Unit | Dimension | None"]:
    if units is None:
        return {}
    if isinstance(units, NameUnits):
        units = units.units
    out: dict[str, Unit | Dimension | None] = {}
    for name, u in units.items():
        key = str(name)
        if u is None or isinstance(u, (Unit, Dimension)):
            out[key] = u
        elif isinstance(u, Quantity):
            out[key] = u.unit
        elif isinstance(u, str):
            out[key] = parse_unit(u)
        else:
            raise DomainError(f"units[{key!r}]: expected a unit text, Unit or Dimension, "
                              f"not {type(u).__name__}")
    return out


def _check_values(values: str) -> bool:
    if values not in ("quantities", "numbers"):
        raise DomainError(f"values must be 'quantities' or 'numbers', not {values!r}")
    return values == "numbers"


def dimension_of(expr: object, units: "UnitsArg | NameUnits | None" = None, *,
                 values: str = "quantities") -> DimensionReport:
    """The dimension of ``expr`` given units of its symbols — or a located refusal.

    >>> r = dimension_of("m*v**2/2", {"m": "kg", "v": "m/s"})
    >>> r.dimension.describe()
    'kg·m²/s² (energy)'
    >>> dimension_of("v*t + a*t", {"v": "m/s", "a": "m/s^2", "t": "s"})
    Traceback (most recent call last):
      ...
    pycodemath.engine.units.DimensionError: dimension mismatch in a*t + t*v: a*t is m/s (velocity) but t*v is m (length)

    Raises ``DimensionError`` (with ``.location``, ``.left``/``.right`` and their
    dimensions, ``.witness``) at the INNERMOST subexpression where dimensions
    disagree. ``values="numbers"`` additionally refuses mixed units of one
    dimension (see the module docstring).
    """
    numbers = _check_values(values)
    table_units = _read_units(units)
    sy = _to_sympy(expr, table_units)
    return guard("dimension_of", sy, lambda: _dimension_of(sy, table_units, numbers))


def _dimension_of(sy: sp.Basic, units: Mapping[str, Any], numbers: bool) -> DimensionReport:
    an = _Analyzer(units, numbers)
    try:
        info = an.run(sy)
    except DimensionError as exc:
        if exc.witness is not None:
            raise
        # No witness evaluated cleanly: never an accusation without evidence.
        an.warn("unwitnessed-mismatch", sy, f"{exc} — no witness point evaluated "
                f"cleanly, so the dimension is reported as unknown, not as an error")
        info = _Info(None, None, False)
    return DimensionReport(str(sy), info.dim, info.factor if numbers else None,
                           tuple(an.warnings), tuple(sorted(an.unknown)),
                           "numbers" if numbers else "quantities")


# =============================================================================
# check_dimensions — the verification step
# =============================================================================


@dataclass(frozen=True)
class DimensionCheck:
    """VERIFIED (dimensionally consistent) / REFUTED (with witness) / UNDECIDED.

    VERIFIED is a statement about DIMENSIONS only: ``E = m*v**2`` is verified
    although the kinetic energy has a 1/2 — no unit check can see a pure number.
    ``counterexample`` is set exactly when REFUTED: values for the symbols plus
    ``rescale`` (the base unit that was doubled) or the unit factors (``numbers``
    mode). ``location`` is the subexpression where the mismatch is.
    """

    status: VerdictStatus
    left: str
    right: str
    left_dimension: "str | None"
    right_dimension: "str | None"
    detail: str
    location: "str | None"
    counterexample: "dict[str, str] | None"
    warnings: tuple[UnitWarning, ...]
    unknown: tuple[str, ...]
    values: str
    method: str = "dimensional-analysis"

    def __post_init__(self) -> None:
        if (self.status is VerdictStatus.REFUTED) != (self.counterexample is not None):
            raise DomainError("DimensionCheck: counterexample must be set exactly when REFUTED")

    @property
    def verified(self) -> bool:
        return self.status is VerdictStatus.VERIFIED

    @property
    def refuted(self) -> bool:
        return self.status is VerdictStatus.REFUTED

    def as_verdict(self) -> Verdict:
        """The same outcome as a ``Verdict`` (X1 certificates, the step verifier)."""
        return Verdict(self.status, self.method, self.counterexample, self.detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "check_dimensions",
            "status": self.status.value,
            "method": self.method,
            "left": self.left,
            "right": self.right,
            "left_dimension": self.left_dimension,
            "right_dimension": self.right_dimension,
            "detail": self.detail,
            "location": self.location,
            "counterexample": self.counterexample,
            "warnings": [w.to_dict() for w in self.warnings],
            "unknown": list(self.unknown),
            "values": self.values,
        }


def check_dimensions(lhs: object, rhs: object = None,
                     units: "UnitsArg | NameUnits | None" = None, *,
                     values: str = "quantities") -> DimensionCheck:
    """Is ``lhs = rhs`` dimensionally consistent?

    >>> r = check_dimensions("F", "m*v", {"F": "N", "m": "kg", "v": "m/s"})
    >>> r.status.value, r.detail
    ('refuted', 'left is kg·m/s² (force), right is kg·m/s (momentum)')
    >>> check_dimensions("F = m*a", units={"F": "N", "m": "kg", "a": "m/s^2"}).status.value
    'verified'

    ``rhs=None`` reads ``lhs`` as an equation text (``"F = m*a"``).
    """
    numbers = _check_values(values)
    table_units = _read_units(units)
    if rhs is None:
        if not isinstance(lhs, str):
            raise DomainError("check_dimensions: give rhs, or lhs as an equation text 'a = b'")
        lhs, rhs = _split_equation(lhs)
    left = _to_sympy(lhs, table_units)
    right = _to_sympy(rhs, table_units)
    return guard("check_dimensions", left,
                 lambda: _check(left, right, table_units, numbers))


def _check(left: sp.Basic, right: sp.Basic, units: Mapping[str, Any], numbers: bool) -> DimensionCheck:
    mode = "numbers" if numbers else "quantities"
    an = _Analyzer(units, numbers)

    def result(status: VerdictStatus, li: "_Info | None", ri: "_Info | None", detail: str,
               location: "str | None" = None, cx: "dict[str, str] | None" = None) -> DimensionCheck:
        return DimensionCheck(
            status, str(left), str(right),
            None if li is None or li.dim is None else li.dim.text,
            None if ri is None or ri.dim is None else ri.dim.text,
            detail, location, cx, tuple(an.warnings), tuple(sorted(an.unknown)), mode)

    # Only its text is used (as the location of a mismatch between the sides).
    equation = sp.Symbol(f"{left} = {right}")
    li: "_Info | None" = None
    ri: "_Info | None" = None
    try:
        li = an.run(left)
        ri = an.run(right)
        an.same(equation, left, li, right, ri)
        if numbers and li.factor is not None and ri.factor is not None \
                and li.dim is not None and li.dim == ri.dim:
            if li.coef_free and ri.coef_free:
                an.same_scale(equation, left, li, right, ri)
            elif not _eq(li.factor, ri.factor):
                an.warn("scale-with-coefficient", right,
                        f"{left} and {right} are in different units of the same dimension; "
                        f"the numeric coefficient may or may not be the conversion")
    except DimensionError as exc:
        if exc.witness is None:
            return result(VerdictStatus.UNDECIDED, li, ri,
                          f"{exc} — but no witness point evaluated cleanly, so this is not "
                          f"reported as a refutation", exc.location)
        if exc.location == str(equation):
            detail = (f"left is {li.dim.describe()}, right is {ri.dim.describe()}"
                      if exc.kind == "dimension" and li and ri and li.dim and ri.dim
                      else str(exc))
            return result(VerdictStatus.REFUTED, li, ri, detail, exc.location, exc.witness)
        return result(VerdictStatus.REFUTED, li, ri, str(exc), exc.location, exc.witness)
    if li.wildcard and ri.dim is not None:      # 0 = F: zero has every dimension
        return result(VerdictStatus.VERIFIED, li, ri, f"0 = a value of {ri.dim.describe()}")
    if ri.wildcard and li.dim is not None:
        return result(VerdictStatus.VERIFIED, li, ri, f"a value of {li.dim.describe()} = 0")
    if li.dim is None or ri.dim is None:
        missing = ", ".join(sorted(an.unknown)) or "a subexpression of unknown dimension"
        return result(VerdictStatus.UNDECIDED, li, ri,
                      f"cannot decide: no unit for {missing}")
    if li.dim != ri.dim:
        return result(VerdictStatus.VERIFIED, li, ri,
                      f"both sides are {li.dim.hard().describe()} up to an angle "
                      f"(left {li.dim.text}, right {ri.dim.text}; SI treats rad as 1)")
    return result(VerdictStatus.VERIFIED, li, ri,
                  f"both sides are {li.dim.describe()}")


# =============================================================================
# Units carried in variable names (for C1 / L1)
# =============================================================================


@dataclass(frozen=True)
class NameUnits:
    """Units read off variable names by suffix: ``t_ms`` -> ms, ``delayMs`` -> ms.

    ``units`` feeds straight into ``dimension_of`` / ``check_dimensions`` (with
    ``values="numbers"`` — a variable named ``t_ms`` holds a NUMBER of ms).
    ``unmatched`` are the names that carry no recognised suffix.
    """

    units: dict[str, str]
    suffixes: dict[str, str]
    unmatched: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "units_from_names", "units": dict(self.units),
                "suffixes": dict(self.suffixes), "unmatched": list(self.unmatched)}


_CAMEL = re.compile(r"[a-z0-9]([A-Z][a-z0-9]*)$")
#: Case-sensitive suffixes, checked before the case-insensitive table.
_CASE_SENSITIVE = {"K": "K", "degC": "degC", "degF": "degF", "MB": "MB", "kB": "kB",
                   "GB": "GB", "Hz": "Hz", "kHz": "kHz", "MHz": "MHz", "GHz": "GHz",
                   "mA": "mA", "MW": "MW", "mW": "mW"}


def _suffix(name: str) -> "str | None":
    if "_" in name.strip("_"):
        stem, _, suf = name.rstrip("_").rpartition("_")
        return suf if stem.strip("_") else None
    m = _CAMEL.search(name)
    return m.group(1) if m and m.start(1) > 0 else None


def units_from_names(symbols: object, *, suffixes: "Mapping[str, str] | None" = None,
                     extra: "Mapping[str, str] | None" = None,
                     exclude: Iterable[str] = ()) -> NameUnits:
    """Map variable names to units by their suffix.

    >>> units_from_names(["t_ms", "timeout_s", "widthPx", "count"]).units
    {'t_ms': 'ms', 'timeout_s': 's', 'widthPx': 'px'}

    ``symbols``: names, SymPy symbols, or an expression (its free symbols).
    ``suffixes`` replaces the default table (``_units_table.SUFFIX_UNITS``),
    ``extra`` adds to it, ``exclude`` drops suffixes. ``extra`` may also map a
    FULL variable name to a unit. Suffixes that commonly mean something else
    (``_n``, ``_in``, ``_h``, ``_l``, ``_t``...) are absent by design.
    """
    table_ = dict(table.SUFFIX_UNITS if suffixes is None else suffixes)
    full_names: dict[str, str] = {}
    for k, v in (extra or {}).items():
        parse_unit(v)  # refuse a bad unit at configuration time
        table_[k.lower()] = v
        full_names[k] = v
    for k in exclude:
        table_.pop(k.lower(), None)
    if isinstance(symbols, str) and not symbols.isidentifier():
        names = sorted(s.name for s in _to_sympy(symbols).free_symbols)  # type: ignore[attr-defined]
    elif isinstance(symbols, str):
        names = [symbols]
    elif isinstance(symbols, (sp.Basic,)) or hasattr(symbols, "sy"):
        names = sorted(s.name for s in _to_sympy(symbols).free_symbols)  # type: ignore[attr-defined]
    elif isinstance(symbols, Iterable):
        names = [str(s) for s in symbols]
    else:
        raise DomainError(f"units_from_names: cannot read names from {type(symbols).__name__}")
    units: dict[str, str] = {}
    used: dict[str, str] = {}
    unmatched: list[str] = []
    for n in names:
        if n in full_names:
            units[n], used[n] = full_names[n], n
            continue
        suf = _suffix(n)
        unit = None
        if suf is not None:
            unit = _CASE_SENSITIVE.get(suf) if suf not in table_ else None
            unit = unit or table_.get(suf.lower())
        if unit is None:
            unmatched.append(n)
        else:
            units[n], used[n] = unit, suf or ""
    return NameUnits(units, used, tuple(unmatched))


_SPEC_ITEM = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(\S+)$")


def read_units_spec(text: str) -> dict[str, str]:
    """``"F:N m:kg v:m/s"`` -> ``{"F": "N", "m": "kg", "v": "m/s"}``.

    The one-line form for a CLI ``where`` clause or a certificate argument: pairs
    ``name:unit`` separated by spaces or commas, a unit written without spaces
    (``c:J/(kg*K)``). Every unit is parsed here, so a typo is refused at once.

    >>> read_units_spec("F:N, m:kg v:m/s")
    {'F': 'N', 'm': 'kg', 'v': 'm/s'}
    """
    if not isinstance(text, str) or len(text) > _MAX_TEXT:
        raise UnitParseError("units spec must be a text of name:unit pairs")
    out: dict[str, str] = {}
    for item in re.split(r"[\s,]+", text.strip()):
        if not item:
            continue
        m = _SPEC_ITEM.match(item)
        if not m:
            raise UnitParseError(f"units spec: {item!r} is not name:unit")
        name, unit = m.groups()
        if name in out:
            raise UnitParseError(f"units spec: {name!r} is given twice")
        parse_unit(unit)
        out[name] = unit
    return out


# =============================================================================
# Physical constants
# =============================================================================


@dataclass(frozen=True)
class Constant:
    """A physical constant: exact (SI 2019 defining constants and exact products
    of them) or measured (CODATA 2022) with its standard uncertainty."""

    name: str
    quantity: Quantity
    exact: bool
    uncertainty: "sp.Expr | None"
    source: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "exact": self.exact, "source": self.source,
                "description": self.description,
                "uncertainty": None if self.uncertainty is None else str(self.uncertainty),
                **self.quantity.to_dict()}


_CONSTANTS = {c.name: c for c in table.CONSTANTS}
_CONSTANT_ALIASES = {"h": "h_planck", "e": "e_charge", "k": "k_B", "R": "R_gas",
                     "F": "F_faraday", "sigma": "sigma_SB", "alpha": "alpha_fs"}


def constant(name: str) -> Constant:
    """A physical constant by name.

    >>> constant("c").quantity.value
    299792458
    >>> constant("G").exact
    False
    """
    key = _CONSTANT_ALIASES.get(name, name)
    spec = _CONSTANTS.get(key)
    if spec is None:
        raise DomainError(f"unknown constant {name!r}; known: {', '.join(sorted(_CONSTANTS))}")
    value = sp.sympify(spec.value, rational=True)
    unc = sp.Rational(spec.uncertainty) if spec.uncertainty else None
    return Constant(spec.name, Quantity(value, parse_unit(spec.unit)), spec.exact, unc,
                    spec.source, spec.description)

