"""Pycodemath IR — the shared expression tree (contract between the engine and the generator).

The cornerstone of the project. Both module 1 (the math engine) and module 2
(the code generator) operate on the same ``Expr`` object. It is a thin but
proprietary layer over SymPy expressions: it gives us a concise API, a stable
interface, and a place where we will later hook in the masking stage.

The "token-cutting" goal: writing math should be short (``E("sin(x)*x")``),
and expansion into code happens only in the generator.

MODULE 9: the four methods here that reach an UNBOUNDED SymPy call —
``Expr.simplify``, ``Expr.expand``, ``Expr.integrate`` and the two ``equivalent``
methods, which run ``simplify`` on a difference — carry the wall-clock guard as
well as their ``engine.symbolic`` wrappers do. Not belt and braces: these are the
Python API a caller reaches directly (``E("1/(x^5+x+1)").integrate("x")`` never
goes through ``engine.symbolic``), and ``codegen.pipeline.generate`` calls
``Expr.simplify`` rather than the engine function. Nesting costs one dict lookup:
a guard that finds a budget already running in this thread adopts it instead of
arming a second one (``core.budget._budget_for``).

``diff``, ``subs``, ``evalf`` and ``compiled`` are deliberately NOT guarded.
Differentiation is syntax-directed and terminates in time proportional to the
expression; the other three are the numeric path, where a guard would sit inside
the loop the single-IR rule exists to keep SymPy out of.
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from typing import Callable, Iterable

import sympy as sp
from sympy.printing.codeprinter import PrintMethodNotImplementedError
from sympy.printing.str import StrPrinter

from .budget import under_budget
from .errors import (
    DomainError,
    NoClosedFormError,
    ParseError,
    PycodemathError,
    UnsupportedFormError,
)


@lru_cache(maxsize=512)
def _lambdify_cached(sy: sp.Expr, syms: tuple) -> Callable:
    """Compile (and memoize) a SymPy expression into a numeric function.

    ``lambdify`` is code generation + exec — a single call costs ~ms, so
    without a cache, methods called repeatedly on the same expression (loops,
    tests, REPL) would pay for compilation every time. The key (expression,
    symbols) is hashable, because SymPy expressions are immutable.
    """
    return sp.lambdify(syms, sy, modules=["math", "numpy"])


@lru_cache(maxsize=256)
def _reads_back(text: str, expected: sp.Basic) -> bool:
    """Does ``parse`` read the bare name ``text`` back as ``expected``?

    The parser is the oracle, not a copy of its rules: which names stay one symbol
    (``x1``, ``x_1``, ``alpha``) and which split into letters (``ab`` -> ``a*b``)
    or mean a constant (``E``) is decided in ``frontend.parser`` and would drift
    out of sync here. Imported late: the parser imports this module.
    """
    from ..frontend.parser import parse

    try:
        return bool(parse(text).sy == expected)
    except PycodemathError:
        return False


class _SourcePrinter(StrPrinter):
    """``str()``, except where ``parse`` would read the text back as something else.

    Measured on 3000 random parser expressions (module C, seed 20260929): 114 held
    ``zoo`` or ``nan``, which ``str`` prints under SymPy's names and the parser
    reads as products of letters — ``zoo`` came back as ``o**2*z``, a different
    expression with two new variables. ``1/0`` and ``0/0`` are what the parser
    itself turns into those two, so they print that way.

    A symbol or constant whose name does not read back as itself has no such text
    — ``EulerGamma`` (``parse("gamma(x)").diff("x").subs({"x": 1})``) would come
    back as ``E*G*a**2*e*l*m**2*r*u``, a ``Symbol("ab")`` as ``a*b``, an undefined
    function ``f(x)`` as ``f*x`` — so it is refused. A FUNCTION off the whitelist
    (``re``, ``polygamma``, ``AccumBounds``) is not: ``parse`` refuses the call
    itself with a ``ParseError`` ("unknown function"), which is loud, not wrong.
    """

    def _print_ComplexInfinity(self, expr: sp.Basic) -> str:
        return "(1/0)"

    def _print_NaN(self, expr: sp.Basic) -> str:
        return "(0/0)"

    def _name(self, text: str, expected: sp.Basic, what: str) -> str:
        if not _reads_back(text, expected):
            raise UnsupportedFormError(
                f"to_source: the {what} {text!r} has no text form the parser reads "
                f"back as itself"
            )
        return text

    def _print_Symbol(self, expr: sp.Symbol) -> str:
        # Compared by name: assumptions (``real=True``) are not part of any text.
        return self._name(expr.name, sp.Symbol(expr.name), "symbol")

    def _print_Dummy(self, expr: sp.Dummy) -> str:
        text = "_" + expr.name
        return self._name(text, sp.Symbol(text), "symbol")

    def _print_constant(self, expr: sp.Basic) -> str:
        return self._name(str(expr), expr, "constant")

    _print_EulerGamma = _print_Catalan = _print_constant
    _print_GoldenRatio = _print_TribonacciConstant = _print_constant

    def _print_AppliedUndef(self, expr: sp.Basic) -> str:
        # No text reads back as an undefined function: ``f(x)`` parses as the
        # product ``f*x``, a longer name as a ParseError.
        raise UnsupportedFormError(
            f"to_source: the undefined function {expr.func.__name__}(...) has no "
            f"text form the parser reads back as itself"
        )

    def _print_Integer(self, expr: sp.Integer) -> str:
        try:
            return super()._print_Integer(expr)
        except ValueError as exc:
            raise _too_long() from exc

    def _print_Rational(self, expr: sp.Rational) -> str:
        try:
            return super()._print_Rational(expr)
        except ValueError as exc:
            raise _too_long() from exc


def _too_long() -> UnsupportedFormError:
    # ``str(int)`` past ``sys.get_int_max_str_digits()`` (4300 by default) is a
    # ValueError: ``parse("2^20000").to_source()`` leaked it raw before module C.
    return UnsupportedFormError(
        f"to_source: an integer is too long to write as text "
        f"(Python's limit is {sys.get_int_max_str_digits()} digits)"
    )


class Expr:
    """Wrapper around a SymPy expression with its own concise API.

    We keep a ``sympy.Expr`` inside (``self.sy``). All mathematical operations
    are delegated to SymPy, but the user/agent sees only the stable Pycodemath
    interface.
    """

    __slots__ = ("sy",)

    sy: "sp.Expr"

    def __init__(self, expr: "sp.Expr | Expr | str | int | float"):
        if isinstance(expr, Expr):
            self.sy = expr.sy
        else:
            # str, int, float, sympy.Expr — a single entry point via sympify.
            # Note: for ``str`` this is SymPy evaluation (the same surface as
            # in the parser) — do not feed it untrusted text without being
            # aware that sympify is NOT a sandbox.
            self.sy = sp.sympify(expr)

    # --- introspection --------------------------------------------------
    @property
    def free_symbols(self) -> tuple[sp.Symbol, ...]:
        """Symbols (variables) occurring in the expression, in a stable order."""
        return tuple(sorted(self.sy.free_symbols, key=lambda s: s.name))

    def symbol_names(self) -> list[str]:
        return [s.name for s in self.free_symbols]

    # --- symbolic operations (delegated, but return Expr) ---------------
    @under_budget("simplify")
    def simplify(self) -> "Expr":
        return Expr(sp.simplify(self.sy))

    @under_budget("expand")
    def expand(self) -> "Expr":
        return Expr(sp.expand(self.sy))

    def diff(self, var: str) -> "Expr":
        # A plain (assumption-free) Symbol is not known to be real to SymPy, so
        # several of the parser's whitelisted functions (module P1) do not
        # differentiate: Abs(y).diff(y) does not reduce to sign(y), it stays an
        # unevaluated Derivative(re(y), y) — silently wrong from a bare `diff`
        # command, and a raw sympy PrintMethodNotImplementedError (not a
        # PycodemathError) crash from anything that differentiates internally
        # (min/min_nd's gradient, solve_nd/odestiff's Jacobian) on such an
        # expression. Every value this engine computes is real (the codegen
        # target is NumPy; there is no complex-analysis workflow on this
        # surface), so we differentiate under a real-valued stand-in for the
        # requested variable and substitute it back out — this changes no
        # symbol identity anywhere else (solve's fresh-Symbol name matching,
        # subs, codegen all still see the original plain Symbol), only what
        # happens inside this one call.
        sym = sp.Symbol(var)
        real_sym = sp.Symbol(var, real=True)
        result = sp.diff(self.sy.subs(sym, real_sym), real_sym).subs(real_sym, sym)
        return Expr(result)

    @under_budget("integrate")
    def integrate(self, var: str) -> "Expr":
        res = sp.integrate(self.sy, sp.Symbol(var))
        # An unevaluated Integral in the result is junk that breaks the contract
        # (analogous to the res.has(sp.Sum) guard in summation) — we refuse
        # clearly. The integrator RAN and came back empty-handed, which is
        # NoClosedFormError rather than UnsupportedFormError (module 10), and the
        # route says in data what the message has always said in prose.
        if res.has(sp.Integral):
            raise NoClosedFormError(
                "integrate: no closed form — compute numerically "
                "or simplify the expression",
                route="nintegrate",
            )
        return Expr(res)

    def subs(self, mapping: dict[str, "float | int | str"]) -> "Expr":
        # String values go through the guarded parser (whitelist of tokens
        # and names), NOT through bare sympify — otherwise subs would be a
        # second sympify surface with the full namespace, bypassing the parser.
        # Local import: frontend.parser imports core.ir (cycle).
        from ..frontend.parser import parse

        m = {
            sp.Symbol(k): parse(v).sy if isinstance(v, str) else sp.sympify(v)
            for k, v in mapping.items()
        }
        return Expr(self.sy.subs(m))

    def evalf(self, **values: float) -> float:
        """Numeric evaluation after substituting values for symbols.

        Returns a real ``float`` or raises ``DomainError`` — never a raw SymPy or
        mpmath exception, and never ``nan``. An undefined point is one answer on
        every path: ``1/x`` at ``x=0``, ``sin(1/0)``, ``exp(cos(y/0))`` all refuse
        here, while ``subs`` returns the same fact symbolically (``zoo``/``nan``
        are SymPy values; a float that is not a number is not). ``inf`` is kept:
        it is a signed real limit (``Abs(1/0)``, ``exp(10**6)`` overflowing the
        float), not an undefined one.
        """
        m = {sp.Symbol(k): v for k, v in values.items()}
        try:
            result = self.sy.evalf(subs=m)
        except (TypeError, ZeroDivisionError) as exc:
            # SymPy's evalf raises from INSIDE its own machinery when the point
            # is undefined below the top level: ``evalf_trig`` unpacks ``zoo``
            # (``exp(cos(y/0))`` -> TypeError, sympy/core/evalf.py:915) and
            # ``evalf_pow`` divides by an mpf zero (``1/x`` at ``x=0.0`` ->
            # ZeroDivisionError from mpmath's ``mpf_div``). Measured on 3000
            # fuzzed expressions (seed 1, points x=y=0.5 and x=y=0): 2 TypeError
            # + 98 ZeroDivisionError, nothing else. Only those two types are
            # caught — this is not ``except Exception``, and the budget's
            # ``_Deadline`` (a BaseException) is out of reach by construction.
            raise DomainError(
                f"cannot evaluate {self.sy} at {values}: the expression is "
                f"undefined there (division by zero or complex infinity)"
            ) from exc
        try:
            value = float(result)
        except TypeError as exc:
            # Complex or still-symbolic result — a readable error instead of a
            # raw TypeError from ``float()``.
            raise DomainError(
                f"cannot return a real number from {result} "
                f"(complex result or unsubstituted symbols)"
            ) from exc
        if math.isnan(value):
            # ``sin(1/0)``, ``0/0``, ``exp(Abs(0/0))`` reach here as SymPy ``nan``
            # and ``float()`` passes it through silently — 65 of 3000 fuzzed
            # expressions returned it before this check, while ``1/0`` (``zoo``
            # at the top) already refused. Same undefined point, same answer.
            raise DomainError(
                f"cannot evaluate {self.sy} at {values}: the result is undefined (nan)"
            )
        return value

    def compiled(self, vars: "Iterable[str]") -> Callable:
        """Compile the expression into a fast numeric function (``lambdify``).

        Numeric engine contract: compile ONCE before the loop, iterate without
        SymPy overhead (``evalf`` remains the contract for a SINGLE evaluation).
        ``math`` first — domain errors are a plain ``ValueError``, scalars are
        fast; ``numpy`` as a fallback for functions that math lacks. The
        returned function takes values positionally, in the order of ``vars``.
        Compilations are cached (LRU) by the (expression, symbols) pair.

        Building refuses with a ``PycodemathError`` subclass; CALLING keeps the
        plain-exception contract above, because that is the hot loop and
        ``engine.numerics._DOMAIN_ERRORS`` is where those are turned into data.
        """
        if self.sy.has(sp.zoo, sp.nan, sp.AccumBounds):
            # Undefined everywhere (``y/0`` -> ``zoo*y``) or set-valued
            # (``atan(1/0)`` -> ``AccumBounds(-pi/2, pi/2)``): there is no
            # function to compile. Before this check the printer failed with a
            # raw KeyError('ComplexInfinity') or PrintMethodNotImplementedError
            # — 86 of 3000 fuzzed expressions (seed 1).
            raise DomainError(
                f"cannot compile {self.sy}: the expression is undefined "
                f"(division by zero or complex infinity) and has no numeric value"
            )
        try:
            return _lambdify_cached(self.sy, tuple(sp.Symbol(v) for v in vars))
        except PrintMethodNotImplementedError as exc:
            # A node the NumPy printer has no rule for. None reached here from
            # the parser's whitelist in the fuzz once the check above was in
            # place; this keeps the contract if a future whitelist entry does.
            raise UnsupportedFormError(
                f"cannot compile {self.sy}: no numeric form for part of it"
            ) from exc

    # --- serialization (concise, token-friendly) ------------------------
    def to_source(self) -> str:
        """Short text form that ``parse`` reads back as an EQUIVALENT expression.

        The round-trip contract (module C): ``parse(e.to_source())`` is
        mathematically equivalent to ``e`` — ``check_equal`` never REFUTED — or a
        ``ParseError``; never a different expression. It is NOT promised equal
        under ``==``: ``parse`` evaluates as it reads, and SymPy distributes a
        number over a sum it multiplies, so ``1/(2*(sqrt(2) + 2))`` reads back as
        ``1/(2*sqrt(2) + 4)``. Measured on 3000 random parser expressions (the
        06.09 generator, seed 20260929): 8 changed shape this way, every one
        VERIFIED equivalent; a second pass is not a fixed point either (2 of 3000
        with a wider alphabet reshaped again). Making ``==`` hold would take a
        printer that fights SymPy's evaluation node by node, for text nobody
        would call short.

        Where it differs from ``str()`` (see ``_SourcePrinter``): ``zoo`` and
        ``nan`` print as ``(1/0)`` and ``(0/0)`` — SymPy's names read back as
        products of letters (114 of those 3000 before module C); a symbol,
        constant or undefined function whose name the parser reads as something
        else (``EulerGamma``, ``Symbol("ab")``, ``f(x)``) and an integer past
        Python's 4300-digit text limit raise ``UnsupportedFormError``.

        Boundaries, measured with a wider alphabet (decimals, ``oo``, ``I``):
        a function off the parser's whitelist (``re``, ``AccumBounds``,
        ``polygamma``) prints as SymPy does and ``parse`` refuses it (39 of 3000).
        A Float prints with its 15 significant digits: one the parser read comes
        back identical, but one COMPUTED from decimals (``-7/3 - 0.1``,
        ``sinh(0.1)``) comes back as that 15-digit decimal — the same number to
        the precision a Float claims, not the same 53 bits (``check_equal``,
        which reads decimals as written, refuted 3 of 3000 at the 16th digit).
        """
        return _SourcePrinter({"order": None}).doprint(self.sy)

    @under_budget("equivalent")
    def equivalent(self, other: "Expr | str | int | float") -> bool:
        """Mathematical equality: does ``self - other`` simplify to zero.

        Slower than ``==`` (runs ``simplify``), but catches the equivalence of
        expressions with different structure, e.g. ``(x+1)**2`` and
        ``x**2 + 2*x + 1``. ``==`` remains a structural comparison, consistent
        with ``__hash__``.
        """
        other = other if isinstance(other, Expr) else Expr(other)
        return bool(sp.simplify(self.sy - other.sy) == 0)

    def __eq__(self, other: object) -> bool:
        # STRUCTURAL comparison (consistent with ``__hash__``) — fast and
        # safe for set/dict. For mathematical equality: ``equivalent``.
        if isinstance(other, Expr):
            return bool(self.sy == other.sy)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.sy)

    def __repr__(self) -> str:
        return f"Expr({self.sy!s})"

    def __str__(self) -> str:
        return str(self.sy)


class Matrix:
    """Wrapper around a SymPy matrix/vector — IR extension for linear algebra.

    We keep a ``sympy.MatrixBase`` inside (``self.sy``). A vector is simply an
    ``n × 1`` column matrix. The notation stays concise: ``M("[[1,2],[3,4]]")``,
    ``V("[1,2,3]")`` — while the heavy operations are done by the engine
    (``engine.linalg``).
    """

    __slots__ = ("sy",)

    sy: "sp.MatrixBase"

    def __init__(self, data: "sp.MatrixBase | Matrix | str | list | tuple"):
        if isinstance(data, Matrix):
            self.sy = data.sy
        elif isinstance(data, sp.MatrixBase):
            self.sy = data
        elif isinstance(data, str):
            self.sy = _parse_matrix(data)
        else:
            try:
                self.sy = sp.Matrix(data)
            except (ValueError, TypeError) as exc:
                # e.g. rows of different lengths — a readable error instead of a
                # raw ValueError from deep inside SymPy (contract: always PycodemathError)
                raise ParseError(f"invalid matrix: {exc}") from exc

    # --- introspection --------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.sy.shape

    @property
    def rows(self) -> int:
        return self.sy.rows

    @property
    def cols(self) -> int:
        return self.sy.cols

    @property
    def free_symbols(self) -> tuple[sp.Symbol, ...]:
        return tuple(sorted(self.sy.free_symbols, key=lambda s: s.name))

    def symbol_names(self) -> list[str]:
        return [s.name for s in self.free_symbols]

    def tolist(self) -> list[list]:
        return self.sy.tolist()

    # --- operations (delegated, but return IR) --------------------------
    @property
    def T(self) -> "Matrix":
        """Transpose."""
        return Matrix(self.sy.T)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        return Matrix(self.sy * Matrix(other).sy)

    def __mul__(self, other: "Matrix | int | float | Expr") -> "Matrix":
        if isinstance(other, Matrix):
            return Matrix(self.sy * other.sy)
        if isinstance(other, Expr):
            return Matrix(self.sy * other.sy)
        return Matrix(self.sy * sp.sympify(other))

    __rmul__ = __mul__

    def __add__(self, other: "Matrix") -> "Matrix":
        return Matrix(self.sy + Matrix(other).sy)

    def __sub__(self, other: "Matrix") -> "Matrix":
        return Matrix(self.sy - Matrix(other).sy)

    def __getitem__(self, key):
        return self.sy[key]

    def __iter__(self):
        return iter(self.sy)

    # --- serialization / comparison -------------------------------------
    def to_source(self) -> str:
        """Concise literal form: ``[[1, 2], [3, 4]]`` (round-trips with ``M``)."""
        return str(self.sy.tolist())

    @under_budget("equivalent")
    def equivalent(self, other: "Matrix | sp.MatrixBase | str | list") -> bool:
        """MATHEMATICAL equality: does ``self - other`` simplify to zero.

        Slower than ``==`` (runs ``simplify`` on the entries), but catches the
        equivalence of matrices with differently structured entries. ``==``
        remains a structural comparison, consistent with ``__hash__``
        (analogous to ``Expr``).
        """
        other = other if isinstance(other, Matrix) else Matrix(other)
        if self.sy.shape != other.sy.shape:
            return False
        diff = (self.sy - other.sy).applyfunc(sp.simplify)
        return bool(diff.is_zero_matrix)

    def __eq__(self, other: object) -> bool:
        # STRUCTURAL comparison (consistent with ``__hash__``) — fast and safe
        # for set/dict. For mathematical equality: ``equivalent``.
        if isinstance(other, Matrix):
            return bool(self.sy == other.sy)
        return NotImplemented

    def __hash__(self) -> int:
        # A matrix is mutable, but we hash its immutable counterpart —
        # "structurally equal → equal hash", as for ``Expr``.
        return hash(sp.ImmutableMatrix(self.sy))

    def __repr__(self) -> str:
        return f"Matrix({self.sy.tolist()!r})"

    def __str__(self) -> str:
        return str(self.sy)


def _parse_matrix(source: str) -> sp.MatrixBase:
    """Parse a matrix/vector literal: ``[[1,2],[3,4]]`` or ``[1,2,3]``.

    Delegates to the whitelisted parser (NOT bare ``sympify``): matrix text is
    user input reachable from the REPL and MCP, so its entries must resolve
    through the same token guard/whitelist as scalar expressions — otherwise
    attribute access (``().__class__``) would evaluate here. Local import:
    ``frontend.parser`` imports ``core.ir`` (cycle), as in ``Expr.subs``.
    """
    from ..frontend.parser import parse_matrix

    return parse_matrix(source)


def E(source: "str | sp.Expr | Expr | int | float") -> Expr:
    """IR constructor shortcut: ``E("sin(x)*x")``."""
    return Expr(source)


def M(data: "sp.MatrixBase | Matrix | str | list | tuple") -> Matrix:
    """IR matrix constructor shortcut: ``M("[[1,2],[3,4]]")``."""
    return Matrix(data)


def V(data: "sp.MatrixBase | Matrix | str | list | tuple") -> Matrix:
    """IR vector constructor shortcut (column matrix): ``V("[1,2,3]")``."""
    return Matrix(data)


def symbols(names: "str | Iterable[str]") -> tuple[Expr, ...]:
    """Create symbols as IR expressions: ``x, y = symbols("x y")``."""
    syms = sp.symbols(names)
    if not isinstance(syms, (tuple, list)):
        syms = (syms,)
    return tuple(Expr(s) for s in syms)
