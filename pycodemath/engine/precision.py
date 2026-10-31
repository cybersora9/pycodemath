"""Numbers you can trust: an exact result's decimal value, with digits that are
REALLY correct (module K2).

``CRootOf(x**5 - x + 1, 0)`` is an exact answer nobody can use, and
``N(root, 30)`` of the casus-irreducibilis root of ``x^3 - 3x + 1`` prints
``0.347296355333860697703433253539 - 0.e-38*I`` — a real number with an
imaginary part made of rounding noise. This module gives every closed-form value
a decimal companion whose printed digits are guaranteed, or refuses with a type.

WHAT "CORRECT DIGITS" MEANS. ``approx(e, d)`` returns the value of ``e``
CORRECTLY ROUNDED (half-even) to ``d`` significant digits, per component of a
complex number. It is decided by an ENCLOSURE: if both ends of an interval that
contains the true value round to the same ``d``-digit decimal, every point in
between does too (rounding is monotone), so the true value does. That is a
proof, not an estimate, and it is how three of the four methods work:

``exact``            the value is a rational number; rounded with exact integers.
``interval``         directed-rounding interval arithmetic (mpmath's ``libmpi``)
                     over the expression tree, at doubling precision until the
                     enclosure is narrow enough to round. Every node used has a
                     rigorous enclosure: arithmetic, ``exp``/``log``/``sin``/
                     ``cos``/``tan``/``atan``/``atan2``/``cosh``/``sinh``/
                     ``gamma``/``loggamma`` from ``libmpi``; complex values as
                     rectangles; and four enclosures written here because
                     ``mpmath.iv`` only RUNS the float algorithm on intervals for
                     them (no truncation bound — measured: ``iv.zeta(3)`` is not
                     even implemented): ``zeta(n)`` for integer n >= 2 (Borwein's
                     alternating-series acceleration with its error bound),
                     ``erf``/``erfc`` (the alternating Taylor series with the
                     first omitted term as the bound), ``LambertW`` on the real
                     branches 0 and -1 (a monotone bracket checked in intervals),
                     and ``CRootOf`` (SymPy's exact isolating intervals).
``algebraic``        the value is algebraic and an enclosure cannot decide a
                     component that is ZERO (the casus irreducibilis above: the
                     imaginary part is exactly 0, and no interval around 0 can show
                     it). The minimal polynomial is computed exactly; the one root
                     whose exact isolating box meets the enclosure IS the value,
                     and its digits come from refining that box with rationals.
``stable-doubling``  the fallback for functions with no rigorous enclosure here
                     (``polylog``, ``besselj``, ``zeta`` at a non-integer, complex
                     ``gamma`` ...): SymPy's ``evalf`` at ``d + 10`` digits and at
                     twice that must round to the same ``d`` digits. This is the
                     usual engineering check, NOT a proof — ``rigorous`` is
                     ``False`` on such a result, and the method says so.

A component whose enclosure keeps containing 0 is never printed as digits. It is
shown to be exactly zero (the algebraic route, SymPy's assumptions ``is_real`` /
``is_zero``, or ``check_equal(e, 0)`` proving it), or the call raises
``PrecisionUndecided`` with the bound that WAS proved (``|value| < 1e-9860``).
Cancellation is the case this protects: ``exp(pi*sqrt(163)) - 640320**3 - 744``
is ``-7.49927402801814e-13``, with every digit right, because the enclosure is
narrowed until it is.

DECIMALS MEAN THE DECIMAL WRITTEN, as in ``check_equal``: in text and in SymPy
input, ``0.1`` is one tenth and ``sin(1e30)`` is ``sin(10**30)``. A Python
``float`` argument is the binary number it IS (``approx(0.1)`` is
``0.100000000000000005551115123126`` at 30 digits) — the same rule K3 uses.

NO GLOBAL STATE. mpmath's ``mp.dps`` and ``iv.prec`` are never touched: every
evaluation uses private contexts, one per thread. Warnings raised by SymPy or
mpmath inside a call are swallowed and never reach stderr (``N(pi, 50)`` leaked a
``SymPyDeprecationWarning`` before K1/K2).

THE BUDGET. Every public entry point runs under ``core.budget``: the caller's
``time_budget`` when one is open, else ``budget=`` seconds, else the default.
Running out is ``TimeBudgetError``, never a half-checked number.
"""

from __future__ import annotations

import math
import re
import threading
import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Tuple

import mpmath
import sympy as sp
from mpmath.ctx_iv import MPIntervalContext
from mpmath.libmp import libmpi
from sympy.core.evalf import PrecisionExhausted
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import TokenError

from ..core.budget import guard, time_budget
from ..core.errors import DomainError, NonConvergenceError, ParseError, PycodemathError
from ..core.ir import Expr
from ..frontend import parser as _parser

if TYPE_CHECKING:
    from ..verify.verdict import Verdict

__all__ = [
    "Approx",
    "Dual",
    "PrecisionUndecided",
    "approx",
    "evalf_to",
    "dual",
    "parse_precision_request",
    "recheck_approx",
    "to_sympy",
    "DEFAULT_DIGITS",
    "MAX_DIGITS",
]

#: Significant digits when the caller names none — a double's worth.
DEFAULT_DIGITS = 15
#: The most digits one call may ask for. 10 000 digits of a transcendental cost
#: well under a second through intervals; the cap keeps a typo (``digits 10**9``)
#: from being a request for gigabytes.
MAX_DIGITS = 10_000
#: Guard bits added to the first precision level, over ``digits * log2(10)``.
_GUARD_BITS = 40
#: The enclosure loop gives up past this many bits (or 4x the first level, when
#: the request itself needs more): about 19 700 decimal digits of cancellation.
_MAX_BITS = 1 << 16
#: Expressions small enough to try the exact minimal polynomial on. Its cost grows
#: with the number and degree of the radicals; this bound keeps it to well under a
#: second on everything measured (the cubic casus irreducibilis: ~0.8 s).
_MINPOLY_MAX_OPS = 60
_MINPOLY_MAX_DEGREE = 24

_LOG2_10 = math.log2(10)


class PrecisionUndecided(NonConvergenceError):
    """The requested digits could not be certified — and none are printed.

    Raised when the precision loop reached its cap with an enclosure that still
    does not decide the rounding: typically a value that is zero (or tinier than
    the cap can resolve) and could not be PROVED zero. ``bound`` is what was
    proved instead: ``"|value| < 1e-9860"``, or ``None`` when nothing was.
    ``method`` is the method that ran out.
    """

    def __init__(
        self, message: str, *, expr: str = "", digits: int = 0,
        bound: "str | None" = None, method: str = "",
    ) -> None:
        super().__init__(message)
        self.expr = expr
        self.digits = digits
        self.bound = bound
        self.method = method


@dataclass(frozen=True)
class Approx:
    """A closed-form value with its certified decimal companion.

    ``exact``              the exact value (SymPy text, ``sstr``)
    ``value``              the decimal, correctly rounded per component to
                           ``guaranteed_digits`` significant digits; a complex
                           value reads ``a + b*I`` (parseable back)
    ``real`` / ``imag``    the components as decimals; ``imag`` is ``None`` when
                           the value is real (``"0"`` is never a guess: a zero
                           component was proved zero, see the module docstring)
    ``digits``             the digits asked for
    ``guaranteed_digits``  the digits certified (equal to ``digits`` on success)
    ``method``             ``exact | interval | algebraic | stable-doubling``
    ``rigorous``           ``True`` unless the method is ``stable-doubling``
    ``detail``             one sentence: how the digits were decided

    ``str(a)`` is ``≈ <value>``; ``to_dict()`` is JSON-ready, ``kind="approx"``.

        >>> print(approx("sqrt(2)", 20))
        ≈ 1.4142135623730950488
    """

    exact: str
    value: str
    real: str
    imag: "str | None"
    digits: int
    guaranteed_digits: int
    method: str
    rigorous: bool
    detail: str
    kind: str = "approx"

    def __str__(self) -> str:
        return f"≈ {self.value}"

    @property
    def text(self) -> str:
        """``≈ value`` with the guarantee: ``≈ 1.41421356237310 (15 digits, interval)``."""
        return f"≈ {self.value} ({self.guaranteed_digits} digits, {self.method})"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "exact": self.exact,
            "value": self.value,
            "real": self.real,
            "imag": self.imag,
            "digits": self.digits,
            "guaranteed_digits": self.guaranteed_digits,
            "method": self.method,
            "rigorous": self.rigorous,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Dual:
    """An exact result and its number: the "double result" of a calculator.

    ``approx`` is ``None`` exactly when ``reason`` says why: free symbols (there
    is no number to show) or digits that could not be certified.
    """

    exact: str
    approx: "Approx | None"
    reason: "str | None" = None

    def __str__(self) -> str:
        return self.exact if self.approx is None else f"{self.exact}  {self.approx}"

    def to_dict(self) -> dict:
        return {
            "kind": "approx",
            "exact": self.exact,
            "approx": None if self.approx is None else self.approx.to_dict(),
            "reason": self.reason,
        }


# --- input -------------------------------------------------------------------------

#: Names this module reads on top of the parser's whitelist: the special functions
#: and constants an exact result can contain. Same security boundary as ``parse``
#: (the parser's token guard and cost guard run on every text); only the
#: dictionary of resolvable names is larger.
_EXTRA_FUNCTIONS: dict[str, object] = {
    "zeta": sp.zeta, "erf": sp.erf, "erfc": sp.erfc, "erfi": sp.erfi,
    "LambertW": sp.LambertW, "lambertw": sp.LambertW,
    "polylog": sp.polylog, "polygamma": sp.polygamma, "digamma": sp.digamma,
    "loggamma": sp.loggamma, "beta": sp.beta, "besselj": sp.besselj,
    "bessely": sp.bessely, "besseli": sp.besseli, "besselk": sp.besselk,
    "Ei": sp.Ei, "Si": sp.Si, "Ci": sp.Ci, "li": sp.li, "airyai": sp.airyai,
    "CRootOf": sp.CRootOf, "RootOf": sp.CRootOf, "re": sp.re, "im": sp.im,
    "elliptic_k": sp.elliptic_k, "elliptic_e": sp.elliptic_e,
}
_EXTRA_CONSTANTS: dict[str, object] = {
    "EulerGamma": sp.EulerGamma, "Catalan": sp.Catalan, "GoldenRatio": sp.GoldenRatio,
    "phi": sp.GoldenRatio, "euler_gamma": sp.EulerGamma,
}


def _parse_extended(source: str) -> sp.Basic:
    """``parser.parse``, with the special-function names above resolvable."""
    _parser._refuse_stray_close(source, "expression")
    guard_local = dict(_parser._GUARD_LOCAL_DICT)
    guard_local.update({n: sp.Function(n) for n in _EXTRA_FUNCTIONS})
    guard_local.update({n: sp.Symbol(n) for n in _EXTRA_CONSTANTS})
    try:
        tree = parse_expr(
            source, local_dict=guard_local, global_dict=dict(_parser._GUARD_GLOBAL_DICT),
            transformations=_parser._TRANSFORMS, evaluate=False,
        )
    except PycodemathError:
        raise
    except Exception:  # noqa: BLE001 — syntax/name error: the real parse reports it
        tree = None
    if tree is not None:
        try:
            _parser._check_cost(tree)
            _check_exp_log_cost(tree)
        except _parser._CostExceeded as exc:
            raise ParseError(
                f"expression {source!r} is too costly to compute ({exc}) — the "
                f"result would exceed the safe size"
            ) from exc
        except (TypeError, AttributeError) as exc:
            # a bare function name (``zeta``): its class has no instance ``args``
            raise ParseError(
                f"cannot understand expression {source!r}: a function name without "
                f"arguments?"
            ) from exc
    local = dict(_parser._LOCAL_DICT)
    local.update(_EXTRA_FUNCTIONS)
    local.update(_EXTRA_CONSTANTS)
    try:
        return parse_expr(
            source, local_dict=local, global_dict=dict(_parser._GLOBAL_DICT),
            transformations=_parser._TRANSFORMS, evaluate=True,
        )
    except (sp.SympifyError, SyntaxError, TokenError, TypeError, ValueError,
            NameError, AttributeError, IndexError) as exc:
        raise ParseError(f"cannot understand expression {source!r}: {exc}") from exc


def _literal_bits(node: sp.Basic) -> "int | None":
    """An upper bound on the bits of the exact rational a literal subtree builds,
    or ``None`` when it is not a literal (holds a symbol, a constant, a call)."""
    if isinstance(node, sp.Integer):
        return abs(int(node)).bit_length() + 1
    if isinstance(node, sp.Rational):
        return abs(int(node.p)).bit_length() + int(node.q).bit_length() + 1
    if isinstance(node, (sp.Add, sp.Mul)):
        parts = [_literal_bits(a) for a in node.args]
        if any(p is None for p in parts):
            return None
        return 2 * sum(p for p in parts if p is not None) + 1
    if isinstance(node, sp.Pow) and node.args[1].is_Integer:
        base = _literal_bits(node.args[0])
        return None if base is None else base * max(1, abs(int(node.args[1])))
    return None


def _check_exp_log_cost(node: sp.Basic) -> None:
    """``exp(n*log(r))`` for a rational literal ``r`` is turned by SymPy into the
    exact power ``r**n`` — a number the parser's cost guard never sees, because it
    is not written as a power: ``exp(10**7*log(3/2))`` built a 5.8-million-bit
    rational and ``exp(10**20*log(1 + 10**-20))`` never returned (K2 finding)."""
    if isinstance(node, sp.Function) and type(node).__name__ == "exp" and node.args:
        arg = node.args[0]
        factors = arg.args if isinstance(arg, sp.Mul) else (arg,)
        coeff = 1
        logs = []
        for f in factors:
            n = _parser._int_value_bounded(f) if not isinstance(f, sp.Function) else None
            if n is not None:
                coeff *= abs(n)
            elif isinstance(f, sp.Function) and type(f).__name__ in ("log", "ln"):
                logs.append(f)
        for lg in logs:
            bits = _literal_bits(lg.args[0]) if lg.args else None
            if bits is not None and bits * coeff > _parser._MAX_RESULT_BITS:
                raise _parser._CostExceeded(
                    f"exp({sp.sstr(arg)}) is the power of a rational with about "
                    f"{bits * coeff} bits, beyond {_parser._MAX_RESULT_BITS}"
                )
    for a in getattr(node, "args", ()):
        if isinstance(a, sp.Basic):
            _check_exp_log_cost(a)


def _parse_text(text: str) -> sp.Basic:
    """Text with decimals read as written (``check_equal``'s rule)."""
    from ..verify.equal import _exact_decimals

    exact = _exact_decimals(text)
    if exact != text:
        try:
            return _parse_extended(exact)
        except PycodemathError:
            pass
    return _parse_extended(text)


def to_sympy(value: Any, where: str = "approx") -> sp.Expr:
    """The exact SymPy value of ``value`` — text, ``Expr``, SymPy, or a number.

    Text and SymPy ``Float`` atoms are read as the decimal written; a Python
    ``float``/``complex`` is its exact binary value; ``Fraction`` and ``Decimal``
    are exact. NaN, infinities and free symbols are a ``DomainError`` (there is no
    finite number to show), unparseable text a ``ParseError``.

        >>> to_sympy("0.1 + 0.2")
        3/10
    """
    from decimal import Decimal

    if isinstance(value, bool):
        raise DomainError(f"{where}: expected a number or an expression, not a bool")
    if isinstance(value, Expr):
        sy: Any = value.sy
    elif isinstance(value, str):
        if not value.strip():
            raise ParseError(f"{where}: empty expression")
        sy = _parse_text(value)
    elif isinstance(value, int):
        sy = sp.Integer(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise DomainError(f"{where}: {value!r} is not a finite number")
        sy = sp.Rational(value)
    elif isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise DomainError(f"{where}: {value!r} is not a finite number")
        sy = sp.Rational(value.real) + sp.I * sp.Rational(value.imag)
    elif isinstance(value, Fraction):
        sy = sp.Rational(value.numerator, value.denominator)
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise DomainError(f"{where}: {value!r} is not a finite number")
        sy = sp.Rational(str(value))
    elif isinstance(value, sp.Basic):
        sy = value
    else:
        raise DomainError(
            f"{where}: expected a number or an expression, not {type(value).__name__}"
        )
    if isinstance(sy, type) or not isinstance(sy, sp.Expr):
        raise ParseError(f"{where}: {value!r} is not a scalar expression")
    if sy.has(sp.Float):
        from ..verify.equal import _as_written

        sy = _as_written(sy, where)
    if sy.free_symbols:
        names = ", ".join(sorted(str(s) for s in sy.free_symbols))
        raise DomainError(
            f"{where}: {sp.sstr(sy)} has free symbols ({names}) — there is no number "
            f"to show; substitute values first"
        )
    if sy.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        raise DomainError(f"{where}: {sp.sstr(sy)} is not a finite number")
    return sy


def _check_digits(digits: object, where: str) -> int:
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise DomainError(f"{where}: digits must be an integer, not {digits!r}")
    if not 1 <= digits <= MAX_DIGITS:
        raise DomainError(f"{where}: digits must be between 1 and {MAX_DIGITS}, not {digits}")
    return digits


def _check_budget(budget: object, where: str) -> float:
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        raise DomainError(f"{where}: budget must be a number of seconds, not {budget!r}")
    if math.isnan(budget) or budget <= 0:
        raise DomainError(f"{where}: budget must be positive, not {budget!r}")
    return float(budget)


# --- private contexts ----------------------------------------------------------------

_LOCAL = threading.local()


def _iv(bits: int) -> MPIntervalContext:
    """This thread's interval context, at ``bits``. Never mpmath's global ``iv``."""
    ctx = getattr(_LOCAL, "iv", None)
    if ctx is None:
        ctx = _LOCAL.iv = MPIntervalContext()
    ctx.prec = bits
    return ctx


def _scratch(bits: int) -> MPIntervalContext:
    """A second interval context for helper work done INSIDE an enclosure (the
    enclosure's own context keeps its precision)."""
    ctx = getattr(_LOCAL, "scratch", None)
    if ctx is None:
        ctx = _LOCAL.scratch = MPIntervalContext()
    ctx.prec = bits
    return ctx


def _mp(bits: int) -> mpmath.ctx_mp.MPContext:
    """This thread's float context, at ``bits``. Never mpmath's global ``mp``."""
    ctx = getattr(_LOCAL, "mp", None)
    if ctx is None:
        ctx = _LOCAL.mp = mpmath.MPContext()
    ctx.prec = bits
    return ctx


# --- enclosures -----------------------------------------------------------------------

class _Unsupported(Exception):
    """No rigorous enclosure for this node: use the fallback method."""


class _Refine(Exception):
    """The enclosure is too wide to continue (a pole, a branch cut straddled):
    more precision may settle it."""


#: A complex enclosure: (real interval, imaginary interval or None = exactly real).
_Box = Tuple[Any, Optional[Any]]


def _finite(x: Any) -> Any:
    a, b = x._mpi_
    for end in (a, b):
        if end[1] == 0 and end[2] != 0:  # mpmath's inf/nan encoding
            raise _Refine()
    return x


def _nonpositive(raw: tuple) -> bool:
    return raw[1] == 0 or raw[0] == 1


def _nonnegative(raw: tuple) -> bool:
    return raw[1] == 0 or raw[0] == 0


def _contains_zero(x: Any) -> bool:
    a, b = x._mpi_
    return _nonpositive(a) and _nonnegative(b)


def _moderate(raw: tuple) -> bool:
    """Small enough to turn into an exact ``Fraction`` cheaply."""
    return raw[1] == 0 or abs(raw[2]) + raw[3] < 4096


def _frac(raw: tuple) -> Fraction:
    sign, man, exp, _bc = raw
    if man == 0:
        if exp != 0:
            raise _Refine()
        return Fraction(0)
    value = Fraction(man) * (Fraction(2) ** exp if exp < 0 else 2 ** exp)
    return -value if sign else value


def _wrap(ctx: MPIntervalContext, mpi: tuple) -> Any:
    return ctx.make_mpf(mpi)


class _Encloser:
    """Rigorous enclosure of a SymPy expression tree at ``bits`` of precision."""

    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.ctx = _iv(bits)

    # .. constants and leaves ..
    def rational(self, q: sp.Rational) -> Any:
        ctx = self.ctx
        if q.q == 1:
            return ctx.mpf(int(q.p))
        return ctx.mpf(int(q.p)) / ctx.mpf(int(q.q))

    def box(self, node: sp.Basic) -> _Box:
        ctx = self.ctx
        if isinstance(node, sp.Rational):
            return self.rational(node), None
        if node is sp.I:
            return ctx.mpf(0), ctx.mpf(1)
        if node is sp.pi:
            return +ctx.pi, None
        if node is sp.E:
            return +ctx.e, None
        if node is sp.EulerGamma:
            return +ctx.euler, None
        if node is sp.Catalan:
            return +ctx.catalan, None
        if node is sp.GoldenRatio:
            return +ctx.phi, None
        if isinstance(node, sp.Add):
            re, im = self.box(node.args[0])
            for arg in node.args[1:]:
                re2, im2 = self.box(arg)
                re, im = re + re2, _add_opt(im, im2)
            return _finite(re), None if im is None else _finite(im)
        if isinstance(node, sp.Mul):
            acc = self.box(node.args[0])
            for arg in node.args[1:]:
                acc = self.mul(acc, self.box(arg))
            return acc
        if isinstance(node, sp.Pow):
            return self.power(node)
        if isinstance(node, sp.CRootOf):
            return self.crootof(node)
        if isinstance(node, sp.Function):
            return self.function(node)
        raise _Unsupported(type(node).__name__)

    # .. complex arithmetic ..
    def mul(self, a: _Box, b: _Box) -> _Box:
        (ar, ai), (br, bi) = a, b
        if ai is None and bi is None:
            return _finite(ar * br), None
        if ai is None:
            return _finite(ar * br), _finite(ar * bi)
        if bi is None:
            return _finite(ar * br), _finite(ai * br)
        return _finite(ar * br - ai * bi), _finite(ar * bi + ai * br)

    def inverse(self, a: _Box) -> _Box:
        ar, ai = a
        if ai is None:
            if _contains_zero(ar):
                raise _Refine()
            return _finite(1 / ar), None
        den = ar * ar + ai * ai
        if _contains_zero(den):
            raise _Refine()
        return _finite(ar / den), _finite(-ai / den)

    def power(self, node: sp.Pow) -> _Box:
        base, exp = node.args
        ctx = self.ctx
        if exp.is_Integer:
            n = int(exp)
            b = self.box(base)
            if b[1] is None:
                if n < 0 and _contains_zero(b[0]):
                    raise _Refine()
                return _finite(b[0] ** n), None
            acc: _Box = (ctx.mpf(1), None)
            sq, k = b, abs(n)
            while k:
                if k & 1:
                    acc = self.mul(acc, sq)
                k >>= 1
                if k:
                    sq = self.mul(sq, sq)
            return self.inverse(acc) if n < 0 else acc
        b = self.box(base)
        if b[1] is None and not _contains_zero(b[0]) and b[0] > 0:
            if exp.is_Rational:
                return _finite(b[0] ** self.rational(exp)), None
            e = self.box(exp)
            if e[1] is None:
                return _finite(ctx.exp(e[0] * ctx.log(b[0]))), None
        # Principal branch, as SymPy defines it: z**w = exp(w * log z).
        return self.exp_box(self.mul(self.box(exp), self.log_box(b)))

    def exp_box(self, z: _Box) -> _Box:
        ctx = self.ctx
        re, im = z
        if im is None:
            return _finite(ctx.exp(re)), None
        mag = ctx.exp(re)
        return _finite(mag * ctx.cos(im)), _finite(mag * ctx.sin(im))

    def log_box(self, z: _Box) -> _Box:
        ctx = self.ctx
        re, im = z
        if im is None:
            if _contains_zero(re):
                raise _Refine()
            if re > 0:
                return _finite(ctx.log(re)), None
            return _finite(ctx.log(-re)), +ctx.pi  # log of a negative real: i*pi
        if _contains_zero(re) and _contains_zero(im):
            raise _Refine()
        if _contains_zero(im) and not re > 0:
            raise _Refine()  # the enclosure straddles the branch cut
        mod2 = re * re + im * im
        arg = _wrap(ctx, libmpi.mpi_atan2(im._mpi_, re._mpi_, self.bits))
        return _finite(ctx.log(mod2) / 2), _finite(arg)

    # .. leaves with their own enclosures ..
    def crootof(self, node: sp.CRootOf) -> _Box:
        ctx = self.ctx
        tol = sp.Rational(1, 2 ** (self.bits + 8))
        if node.is_real:
            q = node.eval_rational(dx=tol, n=max(15, int(self.bits / _LOG2_10)))
            mid = self.rational(sp.Rational(q))
            return _finite(mid + self.rational(tol) * ctx.mpf([-1, 1])), None
        q = node.eval_rational(dx=tol, dy=tol, n=max(15, int(self.bits / _LOG2_10)))
        qr, qi = sp.Rational(sp.re(q)), sp.Rational(sp.im(q))
        widen = self.rational(tol) * ctx.mpf([-1, 1])
        return _finite(self.rational(qr) + widen), _finite(self.rational(qi) + widen)

    def function(self, node: sp.Function) -> _Box:
        ctx = self.ctx
        name = type(node).__name__
        if name == "LambertW":
            return self.lambertw(node)
        if name == "zeta":
            return self.zeta(node)
        if name == "CRootOf":
            return self.crootof(node)  # type: ignore[arg-type]
        if len(node.args) != 1:
            if name in ("Max", "Min"):
                vals = [self.real(a) for a in node.args]
                return _minmax(ctx, vals, name == "Max"), None
            if name == "atan2":
                y, x = self.real(node.args[0]), self.real(node.args[1])
                if _contains_zero(x) and _contains_zero(y):
                    raise _Refine()
                if _contains_zero(y) and not x > 0:
                    raise _Refine()
                return _finite(_wrap(ctx, libmpi.mpi_atan2(y._mpi_, x._mpi_, self.bits))), None
            raise _Unsupported(name)
        z = self.box(node.args[0])
        if name == "exp":
            return self.exp_box(z)
        if name == "log":
            return self.log_box(z)
        if name in ("sin", "cos", "tan", "cot", "sec", "csc"):
            return self.trig(name, z)
        if name == "Abs":
            re, im = z
            if im is None:
                return _finite(_wrap(ctx, libmpi.mpi_abs(re._mpi_, self.bits))), None
            return _finite(ctx.sqrt(re * re + im * im)), None
        if z[1] is not None:
            raise _Unsupported(f"{name} of a complex argument")
        x = z[0]
        return self.real_function(name, x), None

    def real(self, node: sp.Basic) -> Any:
        re, im = self.box(node)
        if im is not None:
            raise _Unsupported("a real argument was expected")
        return re

    def trig(self, name: str, z: _Box) -> _Box:
        ctx = self.ctx
        re, im = z
        if im is None:
            if name == "sin":
                return _finite(ctx.sin(re)), None
            if name == "cos":
                return _finite(ctx.cos(re)), None
            s, c = ctx.sin(re), ctx.cos(re)
            num, den = {"tan": (s, c), "cot": (c, s), "sec": (ctx.mpf(1), c),
                        "csc": (ctx.mpf(1), s)}[name]
            if _contains_zero(den):
                raise _Refine()
            return _finite(num / den), None
        ch, sh = (_wrap(ctx, v) for v in libmpi.mpi_cosh_sinh(im._mpi_, self.bits))
        sin_b: _Box = (_finite(ctx.sin(re) * ch), _finite(ctx.cos(re) * sh))
        cos_b: _Box = (_finite(ctx.cos(re) * ch), _finite(-ctx.sin(re) * sh))
        if name == "sin":
            return sin_b
        if name == "cos":
            return cos_b
        one: _Box = (ctx.mpf(1), None)
        num, den = {"tan": (sin_b, cos_b), "cot": (cos_b, sin_b), "sec": (one, cos_b),
                    "csc": (one, sin_b)}[name]
        return self.mul(num, self.inverse(den))

    def real_function(self, name: str, x: Any) -> Any:
        ctx, bits = self.ctx, self.bits
        one = ctx.mpf(1)
        if name == "atan":
            return _finite(_wrap(ctx, libmpi.mpi_atan(x._mpi_, bits)))
        if name in ("sinh", "cosh", "tanh"):
            ch, sh = (_wrap(ctx, v) for v in libmpi.mpi_cosh_sinh(x._mpi_, bits))
            return _finite({"sinh": sh, "cosh": ch, "tanh": sh / ch}[name])
        if name in ("asin", "acos"):
            if not (x >= -1 and x <= 1):
                if x.b < -1 or x.a > 1:
                    raise _Unsupported(f"{name} outside [-1, 1] is complex")
                raise _Refine()
            root = ctx.sqrt(_finite(one - x * x))
            y, w = (x, root) if name == "asin" else (root, x)
            if _contains_zero(w) and _contains_zero(y):
                raise _Refine()
            return _finite(_wrap(ctx, libmpi.mpi_atan2(y._mpi_, w._mpi_, bits)))
        if name == "asinh":
            if x.b < 0:
                return -self.real_function("asinh", -x)
            return _finite(ctx.log(x + ctx.sqrt(x * x + 1)))
        if name == "acosh":
            if not x >= 1:
                if x.b < 1:
                    raise _Unsupported("acosh below 1 is complex")
                raise _Refine()
            return _finite(ctx.log(x + ctx.sqrt(x * x - 1)))
        if name == "atanh":
            if not (x > -1 and x < 1):
                if x.b <= -1 or x.a >= 1:
                    raise _Unsupported("atanh outside (-1, 1) is complex or infinite")
                raise _Refine()
            return _finite((ctx.log(one + x) - ctx.log(one - x)) / 2)
        if name in ("gamma", "factorial", "loggamma"):
            arg = x + 1 if name == "factorial" else x
            if name == "loggamma":
                if not arg > 0:
                    raise _Unsupported("loggamma at a non-positive argument")
                return _finite(ctx.loggamma(arg))
            if not arg > 0:
                raise _Unsupported("gamma at a non-positive argument")
            return _finite(ctx.gamma(arg))
        if name in ("floor", "ceiling"):
            fl = math.floor if name == "floor" else math.ceil
            if not (_moderate(x._mpi_[0]) and _moderate(x._mpi_[1])):
                raise _Unsupported(f"{name} of a huge value")
            lo, hi = fl(_frac(x._mpi_[0])), fl(_frac(x._mpi_[1]))
            if lo != hi:
                raise _Refine()
            return ctx.mpf(lo)
        if name == "sign":
            if _contains_zero(x):
                raise _Refine()
            return ctx.mpf(1 if x > 0 else -1)
        if name in ("erf", "erfc"):
            return self.erf(name, x)
        raise _Unsupported(name)

    # .. the four enclosures mpmath.iv does not give rigorously ..
    def erf(self, name: str, x: Any) -> Any:
        """erf is increasing: enclose it at the two ends. Each end by the Taylor
        series, which alternates; once its terms decrease (past ``n > x^2``) the
        error is below the first omitted term."""
        ctx = self.ctx
        if not (_moderate(x._mpi_[0]) and _moderate(x._mpi_[1])):
            raise _Unsupported("erf of a huge or tiny argument")
        lo, hi = _frac(x._mpi_[0]), _frac(x._mpi_[1])
        a = _erf_point(lo, self.bits)
        b = a if hi == lo else _erf_point(hi, self.bits)
        val = ctx.make_mpf((a[0], b[1]))
        return _finite(ctx.mpf(1) - val if name == "erfc" else val)

    def zeta(self, node: sp.Function) -> _Box:
        if len(node.args) != 1 or not node.args[0].is_Integer or int(node.args[0]) < 2:
            raise _Unsupported("zeta at a non-integer or below 2")
        return _zeta_int(int(node.args[0]), self.bits, self.ctx), None

    def lambertw(self, node: sp.Function) -> _Box:
        """W is the inverse of w*e^w, monotone on each real branch: bracket the
        mpmath value by two points and CHECK the bracket in intervals."""
        ctx, bits = self.ctx, self.bits
        k = int(node.args[1]) if len(node.args) == 2 and node.args[1].is_Integer else (
            0 if len(node.args) == 1 else None)
        if k not in (0, -1):
            raise _Unsupported("LambertW off the real branches 0 and -1")
        x = self.real(node.args[0])
        mp = _mp(bits + 20)
        lo_x, hi_x = mp.make_mpf(x._mpi_[0]), mp.make_mpf(x._mpi_[1])
        if lo_x < -mp.exp(-1) - mp.mpf(2) ** (-bits // 2) or (k == -1 and hi_x >= 0):
            raise _Unsupported("LambertW outside its real domain")
        try:
            wa, wb = mp.lambertw(lo_x, k), mp.lambertw(hi_x, k)
        except (ValueError, ZeroDivisionError) as exc:
            raise _Refine() from exc
        if mp.im(wa) != 0 or mp.im(wb) != 0:
            raise _Refine()
        wa, wb = mp.re(wa), mp.re(wb)
        # pad by the INVERSE slope of w*e^w: near the branch point w = -1 the
        # function is flat, and a pad of one ulp in w moves w*e^w by far less than
        # the width of the enclosure of x
        slope = min(abs((1 + wa) * mp.exp(wa)), abs((1 + wb) * mp.exp(wb)))
        if slope < mp.mpf(2) ** (-bits // 2):
            raise _Refine()
        # ... and by the size of w*e^w itself, whose enclosure is that wide
        pad = mp.mpf(2) ** (-bits + 8) * (1 + abs(wa) + abs(wb) + abs(lo_x) + abs(hi_x)) / slope
        minus_one = mp.mpf(-1)
        if k == 0:
            lo, hi = max(wa - pad, minus_one), wb + pad
        else:
            lo, hi = wb - pad, min(wa + pad, minus_one)

        def f(w: Any) -> Any:
            wi = ctx.make_mpf((w._mpf_, w._mpf_))
            return wi * ctx.exp(wi)

        # w*e^w - x has opposite, certain signs at lo and hi for EVERY x in the
        # enclosure, and w*e^w is monotone on [lo, hi]: the root is inside.
        f_lo, f_hi = f(lo) - x, f(hi) - x
        if k == 0:
            ok = (lo == minus_one or f_lo < 0) and f_hi > 0
        else:
            ok = f_lo > 0 and (hi == minus_one or f_hi < 0)
        if not ok:
            raise _Refine()
        return ctx.make_mpf((lo._mpf_, hi._mpf_)), None


def _add_opt(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _minmax(ctx: MPIntervalContext, vals: list, is_max: bool) -> Any:
    for v in vals:
        if not (_moderate(v._mpi_[0]) and _moderate(v._mpi_[1])):
            raise _Unsupported("Max/Min of huge values")
    pick = max if is_max else min
    lo = pick((v._mpi_[0] for v in vals), key=_frac)
    hi = pick((v._mpi_[1] for v in vals), key=_frac)
    return ctx.make_mpf((lo, hi))


def _erf_point(t: Fraction, bits: int) -> tuple:
    """A rigorous enclosure of erf(t), exact rational ``t``, as raw endpoints."""
    neg = t < 0
    t = -t if neg else t
    t2 = t * t
    if t2 > 4096:
        # erfc(t) < exp(-t^2) / (t*sqrt(pi)) for t > 0: erf(t) is 1 to far beyond
        # any precision asked here, and the bound says how far.
        ctx = _scratch(bits + 20)
        tt = ctx.mpf(t.numerator) / t.denominator
        gap = ctx.exp(-tt * tt) / (tt * ctx.sqrt(ctx.pi))
        val = ctx.make_mpf(((1 - gap)._mpi_[0], ctx.mpf(1)._mpi_[1]))
    else:
        work = bits + int(2 * float(t2)) + 32  # the terms peak near e^(t^2)
        ctx = _scratch(work)
        x = ctx.mpf(t.numerator) / t.denominator
        x2 = x * x
        term = x  # (-1)^n x^(2n+1) / n!
        total = ctx.mpf(0)
        n = 0
        eps = ctx.mpf(2) ** (-(bits + 16))
        while True:
            piece = term / (2 * n + 1)
            if n > t2 and abs(piece) < eps:
                # alternating and decreasing from here on: |tail| <= |this piece|
                bound = ctx.make_mpf((abs(piece)._mpi_[1],) * 2)
                total = total + bound * ctx.mpf([-1, 1])
                break
            total = total + piece
            n += 1
            term = -term * x2 / n
        val = total * 2 / ctx.sqrt(ctx.pi)
    out = libmpi.mpi_neg(val._mpi_) if neg else val._mpi_
    return out


def _zeta_int(s: int, bits: int, ctx: MPIntervalContext) -> Any:
    """Rigorous zeta(s), integer s >= 2: eta(s) by Borwein's acceleration of the
    alternating series (J. Borwein, "An efficient algorithm for the Riemann zeta
    function", 1991, algorithm 2), whose error is below 3 / (3 + sqrt 8)^n, then
    zeta = eta / (1 - 2^(1-s)). The coefficients d_k are exact integers; the sum
    runs in outward-rounded intervals, and the bound is added as an interval."""
    n = int((bits + 16) * math.log(2) / math.log(5.8)) + 2
    # d_k = n * sum_{i<=k} (n+i-1)! 4^i / ((n-i)! (2i)!), and each summand is
    # n * 4^i * C(n+i, 2i) / (n+i) — integers, built by the binomial's recurrence
    # (C(n+i+1, 2i+2) = C(n+i, 2i) (n+i+1)(n-i) / ((2i+1)(2i+2)), exact).
    d: list[int] = []
    acc = 0
    comb = 1  # C(n, 0)
    for i in range(n + 1):
        num = n * 4 ** i * comb
        term, rem = divmod(num, n + i)
        if rem:
            raise _Unsupported("zeta: a Borwein coefficient is not an integer")
        acc += term
        d.append(acc)
        comb = comb * (n + i + 1) * (n - i) // ((2 * i + 1) * (2 * i + 2))
    dn = d[n]
    # The sum in fixed point: each term floor((d_k - d_n) 2^P / (k+1)^s) is off by
    # less than one unit of 2^-P, so the whole sum by less than n units — an
    # error carried as an interval below, like Borwein's truncation bound.
    P = bits + 64 + n.bit_length()
    total = 0
    for k in range(n):
        piece = ((d[k] - dn) << P) // (k + 1) ** s
        total = total + piece if k % 2 == 0 else total - piece
    one = ctx.mpf([-1, 1])
    eta = (ctx.mpf(-total) + ctx.mpf(n + 1) * one) / ctx.mpf(dn) / ctx.mpf(2) ** P
    bound = ctx.mpf(3) / ctx.mpf(5.8) ** n  # 5.8 < 3 + sqrt(8): a larger bound
    eta = eta + bound * one
    return _finite(eta / (1 - ctx.mpf(2) ** (1 - s)))


# --- rounding ---------------------------------------------------------------------

#: A correctly rounded component: (negative?, significand with exactly d digits,
#: decimal exponent of the leading digit).
_Rounded = Tuple[bool, int, int]


def _round_fraction(value: Fraction, digits: int) -> "_Rounded | None":
    if value == 0:
        return None
    neg = value < 0
    a = -value if neg else value
    e10 = int(math.floor((a.numerator.bit_length() - a.denominator.bit_length()) / _LOG2_10))
    while Fraction(10) ** e10 > a:
        e10 -= 1
    while Fraction(10) ** (e10 + 1) <= a:
        e10 += 1
    n = round(a * Fraction(10) ** (digits - 1 - e10))
    if n == 10 ** digits:
        n, e10 = 10 ** (digits - 1), e10 + 1
    return neg, int(n), e10


def _round_interval(x: Any, digits: int, bits: int) -> "_Rounded | None":
    """The correctly rounded ``digits``-digit decimal of every point of ``x``, or
    ``None`` when the enclosure does not decide it (or contains 0).

    For a huge or tiny magnitude the decimal exponent and the scaled significand
    are themselves enclosed in intervals (so ``exp(exp(100))``, a number with
    10^43 digits, never becomes an integer); only the final ``d``-digit
    significand meets exact rationals."""
    lo_raw, hi_raw = x._mpi_
    for raw in (lo_raw, hi_raw):
        if raw[1] == 0 and raw[2] != 0:
            return None  # an infinite end
    if _contains_zero(x):
        return None
    if _moderate(lo_raw) and _moderate(hi_raw):
        r1 = _round_fraction(_frac(lo_raw), digits)
        r2 = _round_fraction(_frac(hi_raw), digits)
        return r1 if r1 == r2 else None
    neg = hi_raw[0] == 1
    ctx = _scratch(bits + 32)
    mag = ctx.make_mpf(libmpi.mpi_neg((lo_raw, hi_raw)) if neg else (lo_raw, hi_raw))
    l10 = ctx.log(mag) / ctx.log(10)
    e_lo = math.floor(_frac(l10._mpi_[0]))
    e_hi = math.floor(_frac(l10._mpi_[1]))
    if e_lo != e_hi:
        return None
    scaled = mag * ctx.mpf(10) ** (digits - 1 - e_lo)
    n_lo, n_hi = round(_frac(scaled._mpi_[0])), round(_frac(scaled._mpi_[1]))
    if n_lo != n_hi or not 10 ** (digits - 1) <= n_lo < 10 ** digits:
        return None
    return neg, int(n_lo), e_lo


def _format(r: _Rounded, digits: int) -> str:
    """``1.41421356237310``, ``-7.49927402801814e-13``, ``1.26765060022823e+30``.

    Fixed notation when every digit sits within ``digits`` places of the point and
    the number is not tiny (``-5 <= e < digits``), scientific otherwise. Trailing
    zeros are KEPT: they are certified digits too (``0.500000000000000``)."""
    neg, n, e10 = r
    s = str(n)
    if -5 <= e10 < digits:
        if e10 >= 0:
            head, tail = s[: e10 + 1], s[e10 + 1:]
            text = head + ("." + tail if tail else "")
        else:
            text = "0." + "0" * (-e10 - 1) + s
    else:
        text = s[0] + ("." + s[1:] if len(s) > 1 else "") + f"e{e10:+d}"
    return ("-" if neg else "") + text


def _join(re: "str | None", im: "str | None") -> str:
    if im is None:
        return re if re is not None else "0"
    if re is None:
        return f"{im}*I"
    if im.startswith("-"):
        return f"{re} - {im[1:]}*I"
    return f"{re} + {im}*I"


# --- zero components ----------------------------------------------------------------

def _is_algebraic(sy: sp.Basic) -> bool:
    """Built from rationals, ``I`` and roots of polynomials by + * and rational powers."""
    if isinstance(sy, (sp.Rational, sp.CRootOf)) or sy is sp.I:
        return True
    if isinstance(sy, (sp.Add, sp.Mul)):
        return all(_is_algebraic(a) for a in sy.args)
    if isinstance(sy, sp.Pow):
        return bool(sy.args[1].is_Rational) and _is_algebraic(sy.args[0])
    return False


def _algebraic_digits(sy: sp.Expr, box: _Box, digits: int) -> "tuple[_Rounded | None, _Rounded | None, bool] | None":
    """Locate ``sy`` among the roots of its minimal polynomial and round it there.

    Returns (real, imag, is_zero) with ``None`` for a component that is exactly
    zero, or ``None`` when the route does not apply or does not separate the roots.
    """
    if sp.count_ops(sy) > _MINPOLY_MAX_OPS or not _is_algebraic(sy):
        return None
    X = sp.Dummy("x")
    try:
        poly = sp.Poly(sp.minimal_polynomial(sy, X), X)
    except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
        return None
    if poly.degree() > _MINPOLY_MAX_DEGREE:
        return None
    if poly.degree() == 1:
        coeff = -sp.Rational(poly.nth(0), poly.nth(1))
        root = Fraction(int(coeff.p), int(coeff.q))
        if root == 0:
            return None, None, True
        return _round_fraction(root, digits), None, False
    re_iv, im_iv = box
    re_lo, re_hi = _frac(re_iv._mpi_[0]), _frac(re_iv._mpi_[1])
    if im_iv is None:
        im_lo = im_hi = Fraction(0)
    else:
        im_lo, im_hi = _frac(im_iv._mpi_[0]), _frac(im_iv._mpi_[1])

    def meets(a: Fraction, b: Fraction, c: Fraction, d: Fraction) -> bool:
        return a <= d and c <= b

    eps = sp.Rational(1, 2 ** 8)
    hit: "tuple | None" = None
    for _ in range(7):
        reals, cplx = poly.intervals(all=True, eps=eps)
        hits: list[tuple] = []
        for (a, b), _m in reals:
            fa, fb = Fraction(int(a.p), int(a.q)), Fraction(int(b.p), int(b.q))
            if meets(fa, fb, re_lo, re_hi) and im_lo <= 0 <= im_hi:
                hits.append(("real", a, b))
        for (lo_c, hi_c), _m in cplx:
            ar, ai = sp.Rational(sp.re(lo_c)), sp.Rational(sp.im(lo_c))
            br, bi = sp.Rational(sp.re(hi_c)), sp.Rational(sp.im(hi_c))
            if meets(_fr(ar), _fr(br), re_lo, re_hi) and meets(_fr(ai), _fr(bi), im_lo, im_hi):
                hits.append(("complex", ar, ai, br, bi))
        if len(hits) == 1:
            hit = hits[0]
            break
        if not hits:
            return None
        eps = eps ** 2
    if hit is None:
        return None
    if hit[0] == "real":
        a, b = hit[1], hit[2]
        width = sp.Rational(1, 10 ** (digits + 4))
        for _ in range(40):
            r1 = _round_fraction(_fr(a), digits)
            r2 = _round_fraction(_fr(b), digits)
            if r1 is not None and r1 == r2 and not (a <= 0 <= b):
                return r1, None, False
            a, b = poly.refine_root(a, b, eps=width)
            width = width / 10 ** 8
        return None
    # A non-real root: refine its isolating rectangle until both components round.
    # A component that keeps straddling 0 (``-1 + 2*(-1)**(1/3)`` is ``I*sqrt(3)``)
    # is settled by its own minimal polynomial, not by refining for ever.
    rect: "tuple | None" = None
    for k in (digits + 4, 2 * digits + 24):
        _r, cplx = poly.intervals(all=True, eps=sp.Rational(1, 10 ** k))
        for (lo_c, hi_c), _m in cplx:
            ar, ai = _fr(sp.Rational(sp.re(lo_c))), _fr(sp.Rational(sp.im(lo_c)))
            br, bi = _fr(sp.Rational(sp.re(hi_c))), _fr(sp.Rational(sp.im(hi_c)))
            if meets(ar, br, re_lo, re_hi) and meets(ai, bi, im_lo, im_hi):
                rect = (ar, br, ai, bi)
        if rect is None:
            return None
        parts = []
        for lo_f, hi_f in ((rect[0], rect[1]), (rect[2], rect[3])):
            r1, r2 = _round_fraction(lo_f, digits), _round_fraction(hi_f, digits)
            parts.append(r1 if r1 is not None and r1 == r2 and not lo_f <= 0 <= hi_f else None)
        if parts[0] is not None and parts[1] is not None:
            return parts[0], parts[1], False
    assert rect is not None
    re_part = parts[0]
    im_part = parts[1]
    if re_part is None and rect[0] <= 0 <= rect[1] and _algebraic_zero(sp.re(sy)):
        if im_part is not None:
            return None, im_part, False
    if im_part is None and rect[2] <= 0 <= rect[3] and _algebraic_zero(sp.im(sy)):
        if re_part is not None:
            return re_part, None, False
    return None


def _algebraic_zero(part: sp.Expr) -> bool:
    """Is this (algebraic) component exactly 0? Its minimal polynomial is ``x``."""
    try:
        if part == 0 or part.is_zero is True:
            return True
        if sp.count_ops(part) > 4 * _MINPOLY_MAX_OPS or not _is_algebraic(part):
            return False
        X = sp.Dummy("x")
        return bool(sp.minimal_polynomial(part, X) == X)
    except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
        return False


def _fr(q: sp.Rational) -> Fraction:
    return Fraction(int(q.p), int(q.q))


def _assumed_real(sy: sp.Expr) -> bool:
    try:
        return bool(sy.is_extended_real is True)
    except (TypeError, ValueError, RecursionError):
        return False


def _assumed_imaginary(sy: sp.Expr) -> bool:
    try:
        return bool((sy / sp.I).is_extended_real is True)
    except (TypeError, ValueError, RecursionError):
        return False


def _proved_zero(sy: sp.Expr) -> bool:
    """Is ``sy`` PROVED to be 0 — by SymPy's assumptions or by ``check_equal``?"""
    try:
        if sy.is_zero is True:
            return True
    except (TypeError, ValueError, RecursionError):
        pass
    from ..verify.equal import check_equal

    return check_equal(Expr(sy), 0).verified


# --- the loop -----------------------------------------------------------------------

def _first_bits(digits: int) -> int:
    return int(digits * _LOG2_10) + _GUARD_BITS


def _bits_levels(digits: int) -> Iterator[int]:
    bits = _first_bits(digits)
    cap = max(_MAX_BITS, 4 * bits)
    while bits <= cap:
        yield bits
        bits *= 2


def _bound_text(x: Any) -> str:
    """``|value| < 1e-123`` from an enclosure around zero: each end is below
    ``2^(exp + bitcount)`` in magnitude, read off the raw mpf without building it."""
    tops = [raw[2] + raw[3] for raw in x._mpi_ if raw[1] != 0]
    if not tops:
        return "|value| = 0"
    return f"|value| < 1e{math.ceil(max(tops) * math.log10(2))}"


class _Result:
    __slots__ = ("re", "im", "method", "rigorous", "detail")

    def __init__(self, re: "_Rounded | None", im: "_Rounded | None", method: str,
                 rigorous: bool, detail: str) -> None:
        self.re, self.im, self.method, self.rigorous, self.detail = re, im, method, rigorous, detail


def _by_intervals(sy: sp.Expr, digits: int) -> "_Result | None":
    """Rigorous enclosure at doubling precision; ``None`` when a node has none."""
    real_known = _assumed_real(sy)
    imag_known = not real_known and _assumed_imaginary(sy)
    tried_algebraic = False
    last_box: "_Box | None" = None
    last_bits = tried = 0
    for bits in _bits_levels(digits):
        tried = bits
        try:
            box = _Encloser(bits).box(sy)
        except _Refine:
            continue
        except _Unsupported:
            return None
        except (ValueError, ZeroDivisionError, ArithmeticError, TypeError):
            return None
        last_box, last_bits = box, bits
        re_iv, im_iv = box
        if real_known:
            im_iv = None
        re_r = None if imag_known else _round_interval(re_iv, digits, bits)
        im_r = None if im_iv is None else _round_interval(im_iv, digits, bits)
        re_ok = imag_known or re_r is not None
        im_ok = im_iv is None or im_r is not None
        how = ("rigorous interval enclosure at "
               f"{bits} bits rounds to the same {digits} digits at both ends")
        if real_known and box[1] is not None:
            how += "; the imaginary part is 0 by SymPy's assumptions (is_real)"
        if re_ok and im_ok:
            return _Result(re_r, im_r, "interval", True, how)
        # A component still holding 0 once the enclosure is narrow: settle it.
        undecided = []
        if not re_ok and _contains_zero(re_iv):
            undecided.append("re")
        if not im_ok and _contains_zero(im_iv):
            undecided.append("im")
        if undecided and bits >= 2 * _first_bits(digits) and not tried_algebraic:
            tried_algebraic = True
            alg = _algebraic_digits(sy, box, digits)
            if alg is not None:
                a_re, a_im, zero = alg
                if zero:
                    return _Result(None, None, "algebraic", True,
                                   "the minimal polynomial is x: the value is exactly 0")
                return _Result(a_re, a_im, "algebraic", True,
                               "the value is the one root of its exact minimal polynomial "
                               "whose isolating box meets the interval enclosure; digits "
                               "from refining that box with rationals")
            if undecided == ["re"] and im_iv is None and _proved_zero(sy):
                return _Result(None, None, "interval", True,
                               "the enclosure contains 0 and check_equal proves the value is 0")
            if undecided == ["im"] and re_r is not None and _proved_zero(sp.im(sy)):
                return _Result(re_r, None, "interval", True,
                               "real part from the enclosure; the imaginary part is "
                               "proved 0 by check_equal")
            if undecided == ["re"] and im_r is not None and _proved_zero(sp.re(sy)):
                return _Result(None, im_r, "interval", True,
                               "imaginary part from the enclosure; the real part is "
                               "proved 0 by check_equal")
    if last_box is None:
        raise PrecisionUndecided(
            f"approx: no enclosure of {_short(sy)} up to {tried} "
            f"bits (a pole or branch cut at every precision tried)",
            expr=sp.sstr(sy), digits=digits, method="interval",
        )
    re_iv, im_iv = last_box
    parts = [_bound_text(re_iv) if _contains_zero(re_iv) else None]
    if im_iv is not None and not real_known:
        parts.append(("imaginary part " + _bound_text(im_iv)) if _contains_zero(im_iv) else None)
    bound = "; ".join(p for p in parts if p) or None
    raise PrecisionUndecided(
        f"approx: could not certify {digits} digits of {_short(sy)} — the enclosure at "
        f"{last_bits} bits still does not decide the rounding"
        + (f" ({bound}, not proved to be 0)" if bound else ""),
        expr=sp.sstr(sy), digits=digits, bound=bound, method="interval",
    )


def _point_box(f: sp.Float, bits: int) -> Any:
    raw = f._mpf_
    return _scratch(bits).make_mpf((raw, raw))


def _by_doubling(sy: sp.Expr, digits: int) -> _Result:
    """SymPy ``evalf`` at two precisions must round alike — not a proof."""
    real_known = _assumed_real(sy)
    imag_known = not real_known and _assumed_imaginary(sy)
    prev: "tuple | None" = None
    dps = digits + 10
    cap = max(4 * digits + 400, 2000)
    zero_tried = False
    while dps <= cap:
        try:
            try:
                val = sy.evalf(dps, strict=True)
            except PrecisionExhausted:
                val = sy.evalf(dps)  # agreement between levels is the check anyway
        except (ValueError, TypeError, ZeroDivisionError, ArithmeticError) as exc:
            raise PrecisionUndecided(
                f"approx: {_short(sy)} cannot be evaluated numerically ({exc})",
                expr=sp.sstr(sy), digits=digits, method="stable-doubling",
            ) from exc
        re_f, im_f = val.as_real_imag()
        if not (isinstance(re_f, (sp.Float, sp.Integer, sp.Rational))
                and isinstance(im_f, (sp.Float, sp.Integer, sp.Rational))):
            raise PrecisionUndecided(
                f"approx: {_short(sy)} has no numerical value (evalf returned "
                f"{_short(val)})", expr=sp.sstr(sy), digits=digits, method="stable-doubling",
            )
        bits = int(dps * _LOG2_10) + 8
        re_r = None if imag_known else _round_point(re_f, digits, bits)
        im_r = None if real_known else _round_point(im_f, digits, bits)
        mag = max(abs(re_f), abs(im_f))
        # a component negligible against the precision carried is unresolved noise
        tiny = sp.Float(10) ** (-(dps - 5))
        if re_r is not None and mag != 0 and abs(re_f) < mag * tiny:
            re_r = None
        if im_r is not None and mag != 0 and abs(im_f) < mag * tiny:
            im_r = None
        cur = (re_r, im_r)
        if prev is not None and cur == prev:
            re_none = re_r is None and not imag_known
            im_none = im_r is None and not real_known
            if not re_none and not im_none:
                return _Result(re_r, im_r, "stable-doubling", False,
                               f"SymPy evalf at {dps // 2} and {dps} digits rounds to the "
                               f"same {digits} digits (agreement, not a proof)")
        if (re_r is None and not imag_known) or (im_r is None and not real_known):
            if not zero_tried and prev is not None:
                zero_tried = True
                if re_r is None and not imag_known and im_r is None and not real_known:
                    if _proved_zero(sy):
                        return _Result(None, None, "stable-doubling", True,
                                       "check_equal proves the value is 0")
                elif re_r is None and not imag_known and _proved_zero(sp.re(sy)):
                    imag_known = True
                elif im_r is None and not real_known and _proved_zero(sp.im(sy)):
                    real_known = True
        prev = cur
        dps *= 2
    raise PrecisionUndecided(
        f"approx: could not certify {digits} digits of {_short(sy)} — evalf did not "
        f"settle up to {cap} digits",
        expr=sp.sstr(sy), digits=digits, method="stable-doubling",
    )


def _round_point(f: sp.Expr, digits: int, bits: int) -> "_Rounded | None":
    if f == 0:
        return None
    if isinstance(f, sp.Rational):
        return _round_fraction(_fr(f), digits)
    return _round_interval(_point_box(f, bits), digits, bits)  # type: ignore[arg-type]


def _short(sy: object, limit: int = 120) -> str:
    text = sp.sstr(sy) if isinstance(sy, sp.Basic) else str(sy)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _approx(sy: sp.Expr, digits: int) -> Approx:
    exact = sp.sstr(sy)
    if isinstance(sy, sp.Rational):
        if sy == 0:
            return Approx(exact, "0", "0", None, digits, digits, "exact", True,
                          "the value is exactly 0")
        r = _round_fraction(_fr(sy), digits)
        assert r is not None
        text = _format(r, digits)
        return Approx(exact, text, text, None, digits, digits, "exact", True,
                      "a rational number, rounded with exact integers")
    result = _by_intervals(sy, digits)
    if result is None:
        result = _by_doubling(sy, digits)
    re_s = None if result.re is None else _format(result.re, digits)
    im_s = None if result.im is None else _format(result.im, digits)
    return Approx(
        exact, _join(re_s, im_s), re_s if re_s is not None else "0", im_s, digits, digits,
        result.method, result.rigorous, result.detail,
    )


def _quiet(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` with every warning swallowed — nothing reaches stderr.

    ``warnings.catch_warnings`` is process-global state; the filter is restored on
    the way out, and the only filter added is "ignore", so another thread can at
    worst miss a warning while this block runs, never see a spurious one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _run(where: str, sy: sp.Expr, budget: "float | None", fn: Callable[[], Any]) -> Any:
    if budget is None:
        return _quiet(lambda: guard(where, sy, fn))
    seconds = _check_budget(budget, where)

    def bounded() -> Any:
        with time_budget(seconds, where, sy):
            return fn()

    return _quiet(bounded)


# --- public API -----------------------------------------------------------------------

def approx(value: Any, digits: int = DEFAULT_DIGITS, *, budget: "float | None" = None) -> Approx:
    """The decimal value of a closed-form number, every printed digit certified.

    ``value`` is text (``"CRootOf(x**5 - x + 1, 0)"``, ``"zeta(3)"``), an ``Expr``,
    a SymPy expression or a Python number. The answer is correctly rounded to
    ``digits`` significant digits per component; how that was decided is in
    ``method`` and ``detail`` (module docstring). A value whose digits cannot be
    certified raises ``PrecisionUndecided`` — never a guess.

        >>> approx("CRootOf(x**5 - x + 1, 0)").value
        '-1.16730397826142'
        >>> approx("exp(pi*sqrt(163)) - 640320**3 - 744").value
        '-7.49927402801814e-13'
        >>> approx("sin(10**30)", 20).value
        '-0.090116901912138058030'
    """
    d = _check_digits(digits, "approx")
    sy = to_sympy(value, "approx")
    return _run("approx", sy, budget, lambda: _approx(sy, d))  # type: ignore[no-any-return]


def evalf_to(value: Any, digits: "int | None" = None, *, budget: "float | None" = None) -> Approx:
    """``N(e, d)`` / ``evalf e d``: ``e`` to ``d`` digits that are really correct.

    Same contract as ``approx``; ``digits`` is required — or read from the text
    itself when ``value`` is a request such as ``"N(pi, 50)"``, ``"evalf sqrt(2) 30"``
    or ``"zeta(3) digits 40"`` (see ``parse_precision_request``).

        >>> evalf_to("N(pi, 30)").value
        '3.14159265358979323846264338328'
    """
    if digits is None:
        if not isinstance(value, str):
            raise DomainError("evalf_to: digits is required")
        request = parse_precision_request(value)
        if request is None:
            raise DomainError(
                f"evalf_to: {value!r} names no digits — write N(expr, d), "
                f"'evalf expr d' or 'expr digits d'"
            )
        value, digits = request
    return approx(value, _check_digits(digits, "evalf_to"), budget=budget)


def dual(value: Any, digits: int = DEFAULT_DIGITS, *, budget: "float | None" = None) -> Dual:
    """The exact result and its certified number side by side.

    For an expression with free symbols there is no number, and ``reason`` says
    so; a number that could not be certified carries the refusal's message.

        >>> str(dual("sqrt(2)/2"))
        'sqrt(2)/2  ≈ 0.707106781186548'
    """
    d = _check_digits(digits, "dual")
    if isinstance(value, Expr):
        sy: Any = value.sy
    elif isinstance(value, sp.Basic):
        sy = value
    elif isinstance(value, str):
        sy = _quiet(lambda: _parse_text(value))
    else:
        sy = to_sympy(value, "dual")
    if isinstance(sy, sp.Expr) and sy.free_symbols:
        names = ", ".join(sorted(str(s) for s in sy.free_symbols))
        return Dual(sp.sstr(sy), None, f"free symbols ({names}): no number to show")
    try:
        return Dual(sp.sstr(sy), approx(sy, d, budget=budget))
    except PrecisionUndecided as exc:
        return Dual(sp.sstr(sy), None, str(exc))


_N_CALL = re.compile(r"^\s*(?:N|evalf|n)\s*\((?P<body>.*)\)\s*$", re.S)
_EVALF_WORD = re.compile(r"^\s*(?:evalf|N)\s+(?P<expr>.+?)\s+(?P<d>\d+)\s*$", re.S)
_DIGITS_SUFFIX = re.compile(r"^\s*(?P<expr>.+?)\s+digits\s+(?P<d>\d+)\s*$", re.S)


def parse_precision_request(text: str) -> "tuple[str, int] | None":
    """Split a precision request into ``(expression text, digits)``, or ``None``.

    Reads ``N(e, d)``, ``evalf(e, d)``, ``N(e)`` (15 digits), ``evalf e d``,
    ``N e d`` and ``e digits d``. Purely syntactic: the expression is not parsed
    here, so a caller gets the parser's own refusal for bad text.

        >>> parse_precision_request("N(pi, 50)")
        ('pi', 50)
        >>> parse_precision_request("evalf sqrt(2) 30")
        ('sqrt(2)', 30)
        >>> parse_precision_request("zeta(3) digits 40")
        ('zeta(3)', 40)
    """
    if not isinstance(text, str):
        return None
    m = _N_CALL.match(text)
    if m:
        parts = _split_top_level(m.group("body"))
        if len(parts) == 1 and parts[0].strip():
            return parts[0].strip(), DEFAULT_DIGITS
        if len(parts) == 2 and parts[0].strip() and parts[1].strip().isdigit():
            return parts[0].strip(), int(parts[1])
        return None
    m = _EVALF_WORD.match(text) or _DIGITS_SUFFIX.match(text)
    if m:
        return m.group("expr").strip(), int(m.group("d"))
    return None


def _split_top_level(body: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def recheck_approx(exact: str, value: str, digits: int, *,
                   budget: "float | None" = None) -> "Verdict":
    """Re-derive a recorded ``approx`` result — the ``run`` of an X1 ``approx`` kind.

    Recomputes ``approx(exact, digits)`` from the recorded TEXT (nothing is
    evaluated as Python) and compares it with the recorded ``value``:
    VERIFIED when equal and the method is rigorous, UNDECIDED when equal by
    ``stable-doubling`` (agreement is not a proof) or when the digits cannot be
    certified any more, REFUTED (no point: a constant has no variables) when the
    recorded digits differ from the certified ones.

        >>> recheck_approx("sqrt(2)", "1.41421356237310", 15).status.value
        'verified'
        >>> recheck_approx("sqrt(2)", "1.41421356237311", 15).status.value
        'refuted'
    """
    from ..verify.verdict import Verdict, VerdictStatus

    try:
        a = approx(exact, digits, budget=budget)
    except PrecisionUndecided as exc:
        return Verdict(VerdictStatus.UNDECIDED, exc.method or "approx", None, str(exc))
    if a.value != value:
        return Verdict(VerdictStatus.REFUTED, a.method, {},
                       f"the certified {digits} digits are {a.value}, not {value}")
    status = VerdictStatus.VERIFIED if a.rigorous else VerdictStatus.UNDECIDED
    return Verdict(status, a.method, None, a.detail)
