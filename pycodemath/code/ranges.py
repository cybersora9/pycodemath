"""C3 — value ranges, overflow, underflow, NaN and precision loss, with witnesses.

Give an expression (Python/NumPy syntax, read AS WRITTEN — evaluation order and
all), input ranges and a number type; get back the range of EVERY intermediate
value and, for each event class below, one of three statuses per operation:

``confirmed``
    A concrete input inside the ranges on which the event really happens. The
    input was EXECUTED in the type (NumPy ``int8..uint64``, ``float16``,
    ``float32``, ``float64``; ``bfloat16`` through ``ml_dtypes`` when installed,
    otherwise an emulation of its round-to-nearest-even) and the operation was
    recomputed exactly (``fractions`` / Python ints / ``mpmath``) from the
    operands the program actually saw. Both have to agree. No witness, no
    ``confirmed`` — ever.
``proved_safe``
    No input inside the ranges can trigger it. Proved by outward-rounded interval
    arithmetic (each float result enclosed in ``[RD(lo), RU(hi)]`` of the type),
    refined by bisection, by exhaustive execution of small boxes, or by Z3 for
    integer types when ``z3-solver`` is installed.
``possible``
    Neither, within the budget: an UNDECIDED, never an alarm.

Event classes::

    overflow                   result outside the type (float: +-inf from finite
                               operands; int: wraps around)
    underflow_to_zero          float result is 0 although the exact result is not
    subnormal                  float result is nonzero and below the smallest normal
    integer_precision_loss     an integer-valued float (declared integer input,
                               integer literal, + - * of those) is not representable
    nan_possible               NaN created here (0/0, inf-inf, 0*inf, sqrt/log of a
                               negative) from non-NaN operands
    division_by_zero_possible  x/0, x//0, x%0, log(0), log1p(-1)

    >>> r = analyze("(lo + hi) // 2", {"lo": (0, 2**31 - 1), "hi": (0, 2**31 - 1)}, "int32")
    >>> r.status, r.events_of("overflow")[0].node_text
    ('problems', 'lo + hi')
    >>> analyze("x * x", {"x": (1, 200)}, "float16").status
    'safe'

SEMANTICS (what "as written" means). Every operation runs in the one chosen type,
exactly as ``np.<dtype>`` arrays do under NumPy 2: literals are converted to the
type where they meet a variable (NEP 50 "weak" scalars), while a subexpression
made only of literals (``20 * 365 * 86400``) is folded by Python first, exactly as
the interpreter does. ``max``/``min`` are ``np.maximum``/``np.minimum``. The model is NumPy ARRAY code
under the NumPy that is installed: a literal that would change the type there
(NumPy 1.x value-based casting, e.g. ``float32 * 1e39`` -> float64; ml_dtypes'
``bfloat16 * 2.5`` -> float32) is refused, never analysed in the wrong type. On
NumPy 1.x, NumPy scalars and 0-d arrays promote every Python literal to
float64/int64, so for scalar code there the report does not apply. In an
integer type ``/`` is refused: NumPy turns it into float64 and C/Java truncate, so
either reading would be a guess — write ``//``. ``^`` is refused too (it is XOR in
Python, not a power). With ``ml_dtypes`` a bare Python float promotes a
``bfloat16`` array to float32; this module models the literal as a bfloat16.

ASSUMPTIONS behind ``proved_safe`` for floats: + - * / sqrt are correctly rounded
(IEEE 754, which NumPy's float16/32/64 and the bfloat16 emulation satisfy), every
other function is within ``TRANSCENDENTAL_ULPS`` of the exact result, and the CPU
does not flush subnormals to zero (NumPy's default). A proof by exhaustive
execution needs none of these, and says so in ``method``.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import random
import itertools
import math
import textwrap
import time
import warnings
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import mpmath
import numpy as np

from ..core.errors import DomainError, ParseError, UnsupportedFormError

__all__ = [
    "EVENT_KINDS",
    "DTYPES",
    "analyze",
    "max_safe_input",
    "RangeReport",
    "NodeRange",
    "Event",
    "Advice",
    "SafeInput",
    "RangeUnsupportedError",
]

#: Every event class, in report order.
EVENT_KINDS: tuple[str, ...] = (
    "overflow",
    "underflow_to_zero",
    "subnormal",
    "integer_precision_loss",
    "nan_possible",
    "division_by_zero_possible",
)

#: Classes that make a value WRONG (``max_safe_input`` default). Underflow and
#: subnormals lose precision but leave a value of the right size and sign class.
FATAL_KINDS: tuple[str, ...] = (
    "overflow",
    "integer_precision_loss",
    "nan_possible",
    "division_by_zero_possible",
)

#: Error bound assumed for NumPy's non-basic functions (exp, log, tanh, hypot,
#: power, ...). NumPy documents up to ~4 ulp for its SIMD float32 kernels.
TRANSCENDENTAL_ULPS = 4

DEFAULT_BUDGET = 10.0

#: A box with at most this many input points is executed exhaustively.
ROOT_EXHAUSTIVE = 1 << 18
LEAF_EXHAUSTIVE = 4096
MAX_BOXES = 4000

#: An extended real: a ``Fraction``, or the float ``+inf`` / ``-inf`` (never NaN).
#: Typed ``Any`` because mypy cannot order a ``Fraction | float`` union.
Ext = Any
INF = math.inf


class RangeUnsupportedError(UnsupportedFormError):
    """The expression uses a construct the range analysis does not model."""


# ---------------------------------------------------------------------------
# Number formats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FloatFmt:
    name: str
    p: int  # significand bits, implicit bit included
    emin: int
    emax: int

    @property
    def max(self) -> Fraction:
        return (2 - Fraction(1, 2 ** (self.p - 1))) * Fraction(2) ** self.emax

    @property
    def tiny(self) -> Fraction:  # smallest positive normal
        return Fraction(2) ** self.emin

    @property
    def sub(self) -> Fraction:  # smallest positive subnormal
        return Fraction(2) ** (self.emin - self.p + 1)

    @property
    def bits(self) -> int:
        return {"float16": 16, "bfloat16": 16, "float32": 32, "float64": 64}[self.name]


@dataclass(frozen=True)
class _IntFmt:
    name: str
    bits: int
    signed: bool

    @property
    def min(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def max(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1


_FORMATS: dict[str, "_FloatFmt | _IntFmt"] = {
    "float16": _FloatFmt("float16", 11, -14, 15),
    "bfloat16": _FloatFmt("bfloat16", 8, -126, 127),
    "float32": _FloatFmt("float32", 24, -126, 127),
    "float64": _FloatFmt("float64", 53, -1022, 1023),
    **{
        f"{'u' if not s else ''}int{b}": _IntFmt(f"{'u' if not s else ''}int{b}", b, s)
        for b in (8, 16, 32, 64)
        for s in (True, False)
    },
}

#: Supported type names.
DTYPES: tuple[str, ...] = tuple(_FORMATS)

_WIDER: dict[str, tuple[str, ...]] = {
    "float16": ("float32", "float64"),
    "bfloat16": ("float32", "float64"),
    "float32": ("float64",),
    "int8": ("int16", "int32", "int64"),
    "int16": ("int32", "int64"),
    "int32": ("int64",),
    "uint8": ("uint16", "uint32", "uint64"),
    "uint16": ("uint32", "uint64"),
    "uint32": ("uint64",),
}


def _fmt(dtype: object) -> "_FloatFmt | _IntFmt":
    name = dtype if isinstance(dtype, str) else getattr(dtype, "__name__", str(dtype))
    if not isinstance(dtype, str):
        try:
            name = np.dtype(dtype).name  # type: ignore[call-overload]
        except TypeError:
            pass
    if name not in _FORMATS:
        raise DomainError(
            f"ranges: unknown dtype {dtype!r}; supported: {', '.join(DTYPES)}"
        )
    return _FORMATS[name]


def _ilog2(a: Fraction) -> int:
    """floor(log2(a)) for a > 0, exactly."""
    n, d = a.numerator, a.denominator
    e = n.bit_length() - d.bit_length()
    if (n << max(0, -e)) < (d << max(0, e)):
        e -= 1
    return e


def _round(x: Ext, f: _FloatFmt, mode: str) -> Ext:
    """Round ``x`` exactly to format ``f``: ``mode`` 'n' nearest-even, 'd' down, 'u' up."""
    if isinstance(x, float):
        return x  # +-inf
    if x == 0:
        return Fraction(0)
    s = -1 if x < 0 else 1
    a = -x if x < 0 else x
    e = max(_ilog2(a), f.emin)
    q = Fraction(2) ** (e - f.p + 1)
    m = a / q
    fl = m.numerator // m.denominator
    rem = m - fl
    if mode == "n":
        up = rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1)
    elif mode == "u":
        up = rem != 0 and s > 0
    else:
        up = rem != 0 and s < 0
    mag = (fl + int(up)) * q
    if mag > f.max:
        if mode == "n" or (mode == "u" and s > 0) or (mode == "d" and s < 0):
            return s * INF
        return s * f.max
    return s * mag


def _ulp_step(x: Ext, f: _FloatFmt, k: int) -> Ext:
    """Move a representable ``x`` by ``k`` representable steps (k may be negative)."""
    if k == 0 or isinstance(x, float):
        return x
    o = _ord_of(x, f) + k
    lim = _ord_of(f.max, f)
    if o > lim:
        return INF
    if o < -lim:
        return -INF
    return _val_of(o, f)


def _ord_of(v: Fraction, f: _FloatFmt) -> int:
    """Ordinal of a representable finite value: consecutive values differ by 1, 0 -> 0."""
    if v < 0:
        return -_ord_of(-v, f)
    if v < f.tiny:
        return int(v / f.sub)
    e = _ilog2(v)
    m = v / Fraction(2) ** (e - f.p + 1)
    return (e - f.emin + 1) * (1 << (f.p - 1)) + int(m) - (1 << (f.p - 1))


def _val_of(o: int, f: _FloatFmt) -> Fraction:
    if o < 0:
        return -_val_of(-o, f)
    half = 1 << (f.p - 1)
    if o < half:
        return o * f.sub
    e = o // half - 1 + f.emin
    m = o % half + half
    return m * Fraction(2) ** (e - f.p + 1)


# ---------------------------------------------------------------------------
# The machine: executes nodes in the type (NumPy, or the bfloat16 emulation)
# ---------------------------------------------------------------------------

try:  # optional: the real bfloat16
    import ml_dtypes as _ml_dtypes  # type: ignore[import-not-found,import-untyped,unused-ignore]

    _BF16_DTYPE: "np.dtype[Any] | None" = np.dtype(_ml_dtypes.bfloat16)
except ImportError:  # pragma: no cover - depends on the environment
    _BF16_DTYPE = None


def _bf16_round(a32: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (nearest-even), kept in float32 storage.

    The same rounding ``ml_dtypes`` uses; checked against it by the test suite.
    """
    a32 = np.ascontiguousarray(a32, dtype=np.float32)
    u: np.ndarray = a32.view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    out = (u & 0xFFFFFFFF).astype(np.uint32).view(np.float32).copy()
    out[np.isnan(a32)] = np.nan
    return out


_UFUNC: dict[str, Callable[..., Any]] = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.true_divide,
    "floordiv": np.floor_divide, "mod": np.mod, "neg": np.negative,
    "abs": np.absolute, "max": np.maximum, "min": np.minimum,
    "exp": np.exp, "expm1": np.expm1, "log": np.log, "log2": np.log2,
    "log10": np.log10, "log1p": np.log1p, "sqrt": np.sqrt, "sin": np.sin,
    "cos": np.cos, "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh,
    "arctan": np.arctan, "hypot": np.hypot,
}


class _Machine:
    """Holds arrays of one type and applies NumPy operations to them."""

    def __init__(self, fmt: "_FloatFmt | _IntFmt", emulate_bf16: bool = False) -> None:
        self.fmt = fmt
        self.is_float = isinstance(fmt, _FloatFmt)
        self.emulated = fmt.name == "bfloat16" and (emulate_bf16 or _BF16_DTYPE is None)
        if self.emulated:
            self.dtype: np.dtype[Any] = np.dtype(np.float32)
        elif fmt.name == "bfloat16":
            assert _BF16_DTYPE is not None
            self.dtype = _BF16_DTYPE
        else:
            self.dtype = np.dtype(fmt.name)

    @property
    def method(self) -> str:
        return "bfloat16-emulation" if self.emulated else f"numpy-{self.fmt.name}"

    def _fix(self, a: np.ndarray) -> np.ndarray:
        return _bf16_round(a) if self.emulated else a

    def cast(self, a: np.ndarray) -> np.ndarray:
        """Values (float64 dtype values, or raw ints) -> the type, as NumPy casts."""
        with np.errstate(all="ignore"):
            if self.emulated:
                return _bf16_round(np.asarray(a).astype(np.float32))
            return np.asarray(a).astype(self.dtype)

    def const(self, value: "int | float", n: int) -> np.ndarray:
        with np.errstate(all="ignore"):
            if self.emulated:
                return _bf16_round(np.full(n, np.float32(value)))
            return np.full(n, np.asarray(value).astype(self.dtype), dtype=self.dtype)

    def apply(self, op: str, args: Sequence[np.ndarray], power: "int | float | None") -> np.ndarray:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if op == "pow":
                assert power is not None
                if self.is_float:
                    out = np.power(args[0], np.asarray(power).astype(args[0].dtype))
                else:
                    out = np.power(args[0], int(power))
            else:
                out = _UFUNC[op](*args)
            out = np.asarray(out)
            if out.dtype != self.dtype:  # e.g. an op ml_dtypes runs in float32
                out = out.astype(np.float32 if self.emulated else self.dtype)
            return self._fix(out)

    def to_f64(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(a).astype(np.float64)


def _bits_view(fmt: _FloatFmt) -> tuple[Any, Any, int]:
    """(float dtype, unsigned int dtype, shift) for bit-level ordinal arithmetic."""
    if fmt.name == "bfloat16":
        return np.float32, np.uint32, 16
    fl = {"float16": np.float16, "float32": np.float32, "float64": np.float64}[fmt.name]
    ui = {"float16": np.uint16, "float32": np.uint32, "float64": np.uint64}[fmt.name]
    return fl, ui, 0


def _ords_to_f64(ords: np.ndarray, fmt: _FloatFmt) -> np.ndarray:
    """Vectorized ``_val_of`` through the bit patterns (exact, all formats)."""
    fl, ui, shift = _bits_view(fmt)
    o = np.asarray(ords, dtype=np.int64)
    mag = np.abs(o).astype(np.uint64)
    sign = (o < 0).astype(np.uint64) << np.uint64(fmt.bits - 1)
    bits = (mag | sign).astype(ui)
    if shift:
        bits = bits.astype(np.uint32) << np.uint32(shift)
    return bits.view(fl).astype(np.float64)


def _f64_to_ords(vals: np.ndarray, fmt: _FloatFmt) -> np.ndarray:
    fl, ui, shift = _bits_view(fmt)
    with np.errstate(all="ignore"):
        bits = np.asarray(vals, dtype=np.float64).astype(fl).view(ui)
    if shift:
        bits = bits >> np.uint32(shift)
    b = bits.astype(np.uint64)
    signbit = np.uint64(1) << np.uint64(fmt.bits - 1)
    mag = (b & (signbit - np.uint64(1))).astype(np.int64)
    return np.where((b & signbit) != 0, -mag, mag)


# ---------------------------------------------------------------------------
# Parsing: the expression AS WRITTEN, one node per operation
# ---------------------------------------------------------------------------

_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div",
           ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow"}

_FUNCS = {
    "exp": "exp", "expm1": "expm1", "log": "log", "log2": "log2", "log10": "log10",
    "log1p": "log1p", "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tanh": "tanh",
    "sinh": "sinh", "cosh": "cosh", "arctan": "arctan", "atan": "arctan",
    "abs": "abs", "absolute": "abs", "fabs": "abs", "Abs": "abs",
    "max": "max", "maximum": "max", "Max": "max", "fmax": "max",
    "min": "min", "minimum": "min", "Min": "min", "fmin": "min",
    "hypot": "hypot", "square": "square", "power": "power", "pow": "power",
}
_MODULES = {"np", "numpy", "math", "mpmath", "sp", "sympy"}
_UNARY_FN = {"exp", "expm1", "log", "log2", "log10", "log1p", "sqrt", "sin", "cos",
             "tanh", "sinh", "cosh", "arctan", "abs"}
_TRANSCENDENTAL = {"exp", "expm1", "log", "log2", "log10", "log1p", "sin", "cos",
                   "tanh", "sinh", "cosh", "arctan", "hypot", "pow"}
_INT_TRACKED_OPS = {"add", "sub", "mul", "neg", "abs", "max", "min", "pow"}


@dataclass(frozen=True)
class _Node:
    id: int
    op: str  # var | const | add | sub | ... | pow
    args: tuple[int, ...]
    text: str
    name: str = ""
    value: "int | float | None" = None  # const value, or pow exponent
    int_tracked: bool = False


class _Builder:
    def __init__(self, int_vars: set[str], is_float: bool) -> None:
        self.nodes: list[_Node] = []
        self._key: dict[tuple, int] = {}
        self.int_vars = int_vars
        self.is_float = is_float

    def add(self, op: str, args: tuple[int, ...], text: str, *, name: str = "",
            value: "int | float | None" = None) -> int:
        key = (op, args, name, value if value is None else (type(value).__name__, value))
        if key in self._key:
            return self._key[key]
        if op == "var":
            tracked = name in self.int_vars or not self.is_float
        elif op == "const":
            tracked = isinstance(value, int)  # 1e38 is a float literal, not an integer claim
        elif op == "pow":
            tracked = (self.nodes[args[0]].int_tracked and isinstance(value, int) and value >= 0)
        else:
            tracked = op in _INT_TRACKED_OPS and all(self.nodes[a].int_tracked for a in args)
        node = _Node(len(self.nodes), op, args, text, name, value, tracked)
        self.nodes.append(node)
        self._key[key] = node.id
        return node.id


def _source_of(expr: object) -> str:
    """Accept text, a SymPy/pycodemath expression, or a one-``return`` function."""
    if isinstance(expr, str):
        return expr
    if callable(expr) and not hasattr(expr, "free_symbols"):
        try:
            src = textwrap.dedent(inspect.getsource(expr))
        except (OSError, TypeError) as exc:
            raise RangeUnsupportedError(f"ranges: cannot read the source of {expr!r}") from exc
        tree = ast.parse(src)
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.Lambda))), None)
        if isinstance(fn, ast.Lambda):
            return ast.unparse(fn.body)
        if fn is None:
            raise RangeUnsupportedError("ranges: no function definition found")
        body = [s for s in fn.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            raise RangeUnsupportedError(
                "ranges: only functions whose body is a single `return <expression>` are "
                "analysed directly (loops, branches and assignments arrive with module C1)"
            )
        return ast.unparse(body[0].value)
    sym = getattr(expr, "sympy", None)
    if sym is not None and not callable(sym):
        expr = sym
    if hasattr(expr, "free_symbols"):
        return str(expr)  # SymPy prints Python syntax; NOTE: its canonical order
    raise DomainError(f"ranges: cannot analyse {type(expr).__name__!r}")


def _fold(node: ast.AST) -> "int | float | None":
    """Value of a literal-only subtree, folded exactly as Python folds it."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value  # type: ignore[return-value]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _fold(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _fold(node.left), _fold(node.right)
        if a is None or b is None:
            return None
        try:
            if isinstance(node.op, ast.Pow) and isinstance(b, int) and abs(b) > 4096:
                return None
            r = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                 ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b,
                 ast.Mod: lambda: a % b, ast.Pow: lambda: a ** b}[type(node.op)]()
        except (ZeroDivisionError, OverflowError):
            return None
        return r if type(r) in (int, float) else None
    return None


def _parse(source: str, b: _Builder, int_dtype: bool) -> int:
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise ParseError(f"ranges: not a Python expression: {source!r} ({exc.msg})") from None

    def fname(f: ast.AST) -> str:
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in _MODULES:
            return f.attr
        raise RangeUnsupportedError(f"ranges: unsupported call target {ast.unparse(f)!r}")

    def text(n: ast.AST) -> str:
        return ast.unparse(_strip(n))

    def walk(n: ast.AST) -> int:
        folded = _fold(n)
        if folded is not None:
            if isinstance(folded, float) and not math.isfinite(folded):
                raise DomainError(f"ranges: constant {text(n)} is not finite")
            return b.add("const", (), text(n), value=folded)
        if isinstance(n, ast.Name):
            return b.add("var", (), n.id, name=n.id)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in _MODULES:
            if n.attr in ("pi", "e"):
                return b.add("const", (), text(n), value=getattr(math, n.attr))
            raise RangeUnsupportedError(f"ranges: unsupported name {ast.unparse(n)!r}")
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.UAdd):
                return walk(n.operand)
            if isinstance(n.op, ast.USub):
                return b.add("neg", (walk(n.operand),), text(n))
            raise RangeUnsupportedError(f"ranges: unsupported operator in {text(n)!r}")
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.BitXor):
                raise ParseError(
                    f"ranges: `^` in {text(n)!r} is bitwise XOR in Python, not a power; write **"
                )
            op = _BINOPS.get(type(n.op))
            if op is None:
                raise RangeUnsupportedError(f"ranges: unsupported operator in {text(n)!r}")
            if op == "div" and int_dtype:
                raise DomainError(
                    f"ranges: `/` in {text(n)!r} is ambiguous in an integer dtype (NumPy "
                    "promotes it to float64, C and Java truncate); write // for floor "
                    "division, or analyse the expression in a float dtype"
                )
            if op == "pow":
                k = _fold(n.right)
                if k is None:
                    raise RangeUnsupportedError(
                        f"ranges: only constant exponents are supported ({text(n)!r})")
                return _pow(walk(n.left), k, text(n))
            return b.add(op, (walk(n.left), walk(n.right)), text(n))
        if isinstance(n, ast.Call):
            if n.keywords:
                raise RangeUnsupportedError(f"ranges: keyword arguments in {text(n)!r}")
            fn = _FUNCS.get(fname(n.func))
            if fn is None:
                raise RangeUnsupportedError(
                    f"ranges: unsupported function {fname(n.func)!r}; supported: "
                    + ", ".join(sorted(set(_FUNCS))))
            args = n.args
            if fn in _UNARY_FN:
                if len(args) != 1:
                    raise RangeUnsupportedError(f"ranges: {fn} takes one argument ({text(n)!r})")
                if int_dtype and fn != "abs":
                    raise RangeUnsupportedError(
                        f"ranges: {fn} of an integer dtype promotes to float64 in NumPy; "
                        "analyse the expression in a float dtype")
                return b.add(fn, (walk(args[0]),), text(n))
            if fn == "square":
                if len(args) != 1:
                    raise RangeUnsupportedError(f"ranges: square takes one argument ({text(n)!r})")
                a = walk(args[0])
                return b.add("mul", (a, a), text(n))
            if fn == "power":
                if len(args) != 2 or _fold(args[1]) is None:
                    raise RangeUnsupportedError(f"ranges: power needs a constant exponent ({text(n)!r})")
                return _pow(walk(args[0]), _fold(args[1]), text(n))  # type: ignore[arg-type]
            if fn == "hypot":
                if len(args) != 2:
                    raise RangeUnsupportedError(f"ranges: hypot takes two arguments ({text(n)!r})")
                if int_dtype:
                    raise RangeUnsupportedError("ranges: hypot needs a float dtype")
                return b.add("hypot", (walk(args[0]), walk(args[1])), text(n))
            # max / min: n-ary folds left, as np.maximum.reduce would
            if len(args) < 2:
                raise RangeUnsupportedError(f"ranges: {fn} needs two or more arguments ({text(n)!r})")
            acc = walk(args[0])
            for i, arg in enumerate(args[1:], start=2):
                sub = text(ast.Call(func=ast.Name(id=fn, ctx=ast.Load()), args=args[:i], keywords=[])) \
                    if i < len(args) else text(n)
                acc = b.add(fn, (acc, walk(arg)), sub)
            return acc
        raise RangeUnsupportedError(f"ranges: unsupported construct {text(n)!r}")

    def _pow(base: int, k: "int | float", txt: str) -> int:
        if int_dtype and (not isinstance(k, int) or k < 0):
            raise DomainError(f"ranges: {txt!r}: integer powers need a non-negative integer exponent")
        if isinstance(k, float) and k.is_integer():
            k = int(k)
        return b.add("pow", (base,), txt, value=k)

    return walk(tree.body)


def _strip(n: ast.AST) -> ast.AST:
    """Drop ``np.`` / ``math.`` prefixes and canonical aliases, for display."""
    class T(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                and f.value.id in _MODULES else (f.id if isinstance(f, ast.Name) else None)
            if name is not None and name in _FUNCS:
                canon = {"Abs": "abs", "absolute": "abs", "fabs": "abs", "Max": "max",
                         "maximum": "max", "fmax": "max", "Min": "min", "minimum": "min",
                         "fmin": "min", "atan": "arctan"}.get(name, name)
                node.func = ast.Name(id=canon, ctx=ast.Load())
            return node
    import copy
    return T().visit(copy.deepcopy(n))  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _s(v: "Ext | int | None") -> "str | None":
    """Exact value as SymPy-readable text ('13421773/134217728', 'oo')."""
    if v is None:
        return None
    if isinstance(v, float):
        return "oo" if v > 0 else "-oo"
    return str(v)


@dataclass(frozen=True)
class NodeRange:
    """Range of one intermediate value: every value it takes lies in [lo, hi]."""

    id: int
    text: str
    op: str
    lo: str
    hi: str
    nan: bool
    integer_valued: bool

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "op": self.op, "lo": self.lo,
                "hi": self.hi, "nan": self.nan, "integer_valued": self.integer_valued}


@dataclass(frozen=True)
class Event:
    """One event class at one operation, with its status and (if confirmed) witness."""

    kind: str
    status: str  # confirmed | possible | proved_safe
    node: int
    node_text: str
    op: str
    operand_src: tuple[str, ...]
    method: str
    witness: "dict[str, str] | None" = None
    observed: "str | None" = None
    operands: tuple[str, ...] = ()
    exact: "str | None" = None
    detail: str = ""

    def __post_init__(self) -> None:
        if (self.status == "confirmed") != (self.witness is not None):
            raise DomainError("Event: a witness is required exactly when status is 'confirmed'")
        if self.status not in ("confirmed", "possible", "proved_safe"):
            raise DomainError(f"Event: bad status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "status": self.status, "node": self.node,
                "node_text": self.node_text, "op": self.op,
                "operand_src": list(self.operand_src), "method": self.method,
                "witness": self.witness, "observed": self.observed,
                "operands": list(self.operands), "exact": self.exact, "detail": self.detail}


@dataclass(frozen=True)
class Advice:
    """A rewrite that removes events, with its equivalence proof and re-analysis."""

    rewrite: str
    dtype: str
    removes: tuple[str, ...]
    equivalence: str  # always "verified": unproved rewrites are never offered
    method: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"rewrite": self.rewrite, "dtype": self.dtype, "removes": list(self.removes),
                "equivalence": self.equivalence, "method": self.method, "detail": self.detail}


@dataclass(frozen=True)
class RangeReport:
    """The outcome of ``analyze``."""

    expr: str
    dtype: str
    ranges: dict[str, tuple[str, ...]]
    nodes: tuple[NodeRange, ...]
    events: tuple[Event, ...]
    advice: tuple[Advice, ...]
    complete: bool
    machine: str
    elapsed: float
    assumptions: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "range"

    @property
    def status(self) -> str:
        """'problems' (something confirmed), 'undecided' (something possible), or 'safe'."""
        if any(e.status == "confirmed" for e in self.events):
            return "problems"
        if any(e.status == "possible" for e in self.events):
            return "undecided"
        return "safe"

    def events_of(self, kind: str, status: "str | None" = None) -> list[Event]:
        return [e for e in self.events if e.kind == kind and (status is None or e.status == status)]

    def class_status(self, kind: str) -> str:
        """Status of a whole class: confirmed anywhere > possible anywhere > proved_safe."""
        st = {e.status for e in self.events_of(kind)}
        return "confirmed" if "confirmed" in st else "possible" if "possible" in st else "proved_safe"

    @property
    def confirmed(self) -> list[Event]:
        return [e for e in self.events if e.status == "confirmed"]

    @property
    def possible(self) -> list[Event]:
        return [e for e in self.events if e.status == "possible"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "expr": self.expr, "dtype": self.dtype, "status": self.status,
            "ranges": {k: list(v) for k, v in self.ranges.items()},
            "nodes": [n.to_dict() for n in self.nodes],
            "events": [e.to_dict() for e in self.events],
            "advice": [a.to_dict() for a in self.advice],
            "complete": self.complete, "machine": self.machine,
            "elapsed": round(self.elapsed, 4), "assumptions": list(self.assumptions),
        }

    def __str__(self) -> str:
        lines = [f"{self.expr}  [{self.dtype}]  -> {self.status}"]
        for e in self.events:
            if e.status != "proved_safe":
                w = "" if e.witness is None else f" at {e.witness} -> {e.observed}"
                lines.append(f"  {e.status:9} {e.kind:26} {e.node_text}{w}")
        for a in self.advice:
            lines.append(f"  advice: {a.rewrite} [{a.dtype}] removes {', '.join(a.removes)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract values and interval transfer
# ---------------------------------------------------------------------------

@dataclass
class _AV:
    lo: Ext
    hi: Ext
    nan: bool = False
    rep: int = -1  # this value equals node ``rep`` on the current box


def _emul(a: Ext, b: Ext) -> Ext:
    if a == 0 or b == 0:
        return Fraction(0)
    return a * b  # type: ignore[return-value]


def _hull(vals: Iterable[Ext]) -> tuple[Ext, Ext]:
    vs = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return min(vs), max(vs)


def _min_abs_nz(lo: Ext, hi: Ext, smallest: Ext) -> Ext:
    """Smallest nonzero |v| for representable v in [lo, hi] (``smallest`` = least positive)."""
    if lo <= 0 <= hi:
        return smallest if (lo < 0 or hi > 0) else INF
    return min(abs(lo), abs(hi))


def _max_abs(lo: Ext, hi: Ext) -> Ext:
    return max(abs(lo), abs(hi))


def _mpf_exact(x: Fraction) -> Any:
    n, d = x.numerator, x.denominator
    if d & (d - 1) == 0:  # dyadic: exact
        return mpmath.mpf((n, -(d.bit_length() - 1)))
    return mpmath.mpf(n) / d


def _frac(m: Any) -> Ext:
    if mpmath.isinf(m):
        return INF if m > 0 else -INF
    man, exp = mpmath.mpf(m).man_exp
    return Fraction(int(man)) * Fraction(2) ** int(exp)


_MP = {"exp": "exp", "expm1": "expm1", "log": "log", "log1p": "log1p", "sqrt": "sqrt",
       "tanh": "tanh", "sinh": "sinh", "arctan": "atan", "log2": "log2", "log10": "log10"}

#: Working precision for the exact side, and the relative padding that turns a
#: high-precision value into a sound enclosure (mpmath is accurate to a few ulp of
#: its working precision; 2**-180 at 200 bits leaves a wide margin).
_PREC = 200
_PAD = Fraction(1, 2**180)


def _mp_fn(fn: str, x: Any) -> Any:
    if fn == "log2":
        return mpmath.log(x, 2)
    if fn == "log10":
        return mpmath.log10(x)
    return getattr(mpmath, _MP[fn])(x)


#: Magnitudes beyond every format (float64 tops out near 2**1024): an enclosure
#: endpoint out there is replaced by this sentinel instead of a Fraction with
#: billions of digits (exp(3e38) has a ~1e38-bit exponent).
_HUGE = Fraction(2) ** 4096
_F64_LIMIT = Fraction(2) ** 1023  # float(x) is safe below this
_TINY = Fraction(1, 2**4096)


def _enclose(v: Any) -> tuple[Ext, Ext]:
    """A sound enclosure [lo, hi] of the real number approximated by mpf ``v``."""
    if mpmath.isinf(v):
        return (INF, INF) if v > 0 else (-INF, -INF)
    if v == 0:
        return Fraction(0), Fraction(0)
    mag = int(mpmath.mpf(v).man_exp[1]) + mpmath.mpf(v).bc
    if mag > 4096:
        return (_HUGE, INF) if v > 0 else (-INF, -_HUGE)
    if mag < -4096:
        return (Fraction(0), _TINY) if v > 0 else (-_TINY, Fraction(0))
    c = _frac(v)
    assert not isinstance(c, float)
    d = abs(c) * _PAD
    return c - d, c + d


def _f_bounds(fn: str, a: Ext, b: Ext) -> tuple[Ext, Ext]:
    """Enclosure of monotone-increasing ``fn`` over [a, b] (both in its domain)."""
    def one(x: Ext, side: int) -> Ext:
        if isinstance(x, float):
            if fn == "tanh":
                return Fraction(1) if x > 0 else Fraction(-1)
            if fn == "arctan":
                with mpmath.workprec(_PREC):
                    lo, hi = _enclose(mpmath.pi / 2)
                return hi if x > 0 else -hi
            if fn == "exp":
                return INF if x > 0 else Fraction(0)
            if fn == "expm1":
                return INF if x > 0 else Fraction(-1)
            return INF if x > 0 else -INF
        if fn == "sqrt" and x == 0:
            return Fraction(0)
        if (fn in ("log", "log2", "log10") and x == 0) or (fn == "log1p" and x == -1):
            return -INF
        if fn in ("log", "log2", "log10") and x == 1:
            return Fraction(0)
        if fn in ("expm1", "tanh", "sinh", "arctan", "log1p") and x == 0:
            return Fraction(0)
        if fn == "exp" and x == 0:
            return Fraction(1)
        with mpmath.workprec(max(_PREC, x.numerator.bit_length() + 16)):
            xv = _mpf_exact(x)
        with mpmath.workprec(_PREC):
            v = _mp_fn(fn, xv)
            lo, hi = _enclose(v)
        return lo if side < 0 else hi
    return one(a, -1), one(b, 1)


@contextlib.contextmanager
def _iv_prec(bits: int) -> Iterator[None]:
    """mpmath's interval context has no ``workprec``; never leak a changed precision."""
    old = mpmath.iv.prec
    mpmath.iv.prec = bits
    try:
        yield
    finally:
        mpmath.iv.prec = old


def _gt_abs(v: Any, bound: Fraction) -> bool:
    """|v| > bound for an exact Fraction or a high-precision mpf (beyond its padding)."""
    if isinstance(v, Fraction):
        return abs(v) > bound
    lo, hi = _enclose(abs(v))
    return bool(lo > bound)


def _lt_abs(v: Any, bound: Fraction) -> bool:
    if isinstance(v, Fraction):
        return abs(v) < bound
    lo, hi = _enclose(abs(v))
    return bool(hi < bound)


def _quantum(m: Ext, f: "_FloatFmt") -> Ext:
    """Every representable value with |v| >= m is a multiple of this power of two."""
    if isinstance(m, float):
        return m
    return Fraction(2) ** (max(_ilog2(m), f.emin) - f.p + 1)


class _Analyzer:
    """One expression, one dtype: interval transfer, execution, search, proof."""

    def __init__(self, nodes: list[_Node], root: int, fmt: "_FloatFmt | _IntFmt",
                 var_kind: dict[str, str], var_box: dict[str, tuple[int, int]],
                 deadline: float) -> None:
        self.nodes = nodes
        self.root = root
        self.fmt = fmt
        self.is_float = isinstance(fmt, _FloatFmt)
        self.m = _Machine(fmt)
        self.var_kind = var_kind  # name -> "float" | "int" (raw integer input)
        self.box0 = var_box  # name -> (coord_lo, coord_hi); float coords are ordinals
        self.vars = sorted(var_box)
        self.deadline = deadline
        self.boxes_done = 0

    # ---- coordinates -------------------------------------------------------

    def coord_value(self, name: str, c: int) -> Fraction:
        if self.var_kind[name] == "float":
            assert isinstance(self.fmt, _FloatFmt)
            return _val_of(c, self.fmt)
        return Fraction(c)

    def coords_to_arrays(self, coords: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for v in self.vars:
            c = coords[v]
            if self.var_kind[v] == "float":
                assert isinstance(self.fmt, _FloatFmt)
                out[v] = _ords_to_f64(c, self.fmt)
            elif self.fmt.name == "uint64":
                out[v] = np.asarray([int(x) for x in c], dtype=object).astype(np.uint64)
            else:
                out[v] = np.asarray(c, dtype=np.int64)
        return out

    def witness_text(self, name: str, c: int) -> str:
        if self.var_kind[name] == "float":
            assert isinstance(self.fmt, _FloatFmt)
            return _exact_decimal(float(_val_of(c, self.fmt)))
        return str(int(c))

    # ---- interval transfer -------------------------------------------------

    def leaf(self, n: _Node, box: Mapping[str, tuple[int, int]]) -> _AV:
        if n.op == "var":
            lo, hi = box[n.name]
            vlo, vhi = self.coord_value(n.name, lo), self.coord_value(n.name, hi)
            if self.is_float and self.var_kind[n.name] == "int":
                assert isinstance(self.fmt, _FloatFmt)
                return _AV(_round(vlo, self.fmt, "d"), _round(vhi, self.fmt, "u"), False, n.id)
            return _AV(vlo, vhi, False, n.id)
        assert n.value is not None
        v = self.const_value(n)
        return _AV(v, v, False, n.id)

    def const_value(self, n: _Node) -> Ext:
        assert n.value is not None
        if self.is_float:
            assert isinstance(self.fmt, _FloatFmt)
            if isinstance(n.value, float):  # the float64 literal, then RN to the dtype
                return _round(Fraction(n.value), self.fmt, "n")
            return _round(Fraction(n.value), self.fmt, "n")
        return Fraction(int(n.value))

    def run_box(self, box: Mapping[str, tuple[int, int]]) -> list[_AV]:
        avs: list[_AV] = []
        for n in self.nodes:
            if n.op in ("var", "const"):
                avs.append(self.leaf(n, box))
            else:
                avs.append(self.transfer(n, [avs[a] for a in n.args]))
        return avs

    def transfer(self, n: _Node, xs: list[_AV]) -> _AV:
        if self.is_float:
            return self._transfer_float(n, xs)
        return self._transfer_int(n, xs)

    def _exact_iv(self, op: str, xs: list[_AV], power: "int | float | None") -> tuple[Ext, Ext]:
        """Exact (real) enclosure of ``op`` over operand intervals, endpoints extended."""
        a = xs[0]
        if op == "add":
            b = xs[1]
            lo = a.lo + b.lo
            hi = a.hi + b.hi
            return (-INF if lo != lo else lo), (INF if hi != hi else hi)  # type: ignore[comparison-overlap]
        if op == "sub":
            b = xs[1]
            if a.rep == b.rep and a.rep >= 0:
                return Fraction(0), Fraction(0)
            lo = a.lo - b.hi
            hi = a.hi - b.lo
            return (-INF if lo != lo else lo), (INF if hi != hi else hi)  # type: ignore[comparison-overlap]
        if op == "mul":
            b = xs[1]
            if a.rep == b.rep and a.rep >= 0:
                sq = [_emul(a.lo, a.lo), _emul(a.hi, a.hi)]
                lo = Fraction(0) if a.lo <= 0 <= a.hi else min(sq)
                return lo, max(sq)
            return _hull(_emul(p, q) for p in (a.lo, a.hi) for q in (b.lo, b.hi))
        if op == "neg":
            return -a.hi, -a.lo
        if op == "abs":
            if a.lo >= 0:
                return a.lo, a.hi
            if a.hi <= 0:
                return -a.hi, -a.lo
            return Fraction(0), max(-a.lo, a.hi)
        if op in ("max", "min"):
            b = xs[1]
            f = max if op == "max" else min
            return f(a.lo, b.lo), f(a.hi, b.hi)
        if op == "div":
            b = xs[1]
            if a.rep == b.rep and a.rep >= 0 and not (a.lo <= 0 <= a.hi) and \
                    not isinstance(a.lo, float) and not isinstance(a.hi, float):
                return Fraction(1), Fraction(1)
            if b.lo <= 0 <= b.hi:
                return -INF, INF
            qs = []
            for p in (a.lo, a.hi):
                for q in (b.lo, b.hi):
                    if isinstance(p, float) and isinstance(q, float):
                        return -INF, INF
                    qs.append(Fraction(0) if isinstance(q, float) else p / q)  # type: ignore[operator]
            return _hull(qs)
        if op in ("floordiv", "mod"):
            return self._divmod_iv(op, a, xs[1])
        if op == "pow":
            return self._pow_iv(a, power)
        if op == "hypot":
            b = xs[1]
            ma = _max_abs(a.lo, a.hi)
            mb = _max_abs(b.lo, b.hi)
            na = Fraction(0) if a.lo <= 0 <= a.hi else min(abs(a.lo), abs(a.hi))
            nb = Fraction(0) if b.lo <= 0 <= b.hi else min(abs(b.lo), abs(b.hi))
            lo2 = _f_bounds("sqrt", *(lambda s: (s, s))(na * na + nb * nb))[0]  # type: ignore[operator]
            if isinstance(ma, float) or isinstance(mb, float):
                return lo2, INF
            hi2 = _f_bounds("sqrt", ma * ma + mb * mb, ma * ma + mb * mb)[1]
            return lo2, hi2
        if op in ("sin", "cos"):
            if isinstance(a.lo, float) or isinstance(a.hi, float) or a.hi - a.lo > 7:
                return Fraction(-1), Fraction(1)
            with mpmath.workprec(max(120, a.lo.numerator.bit_length() + 16,
                                     a.hi.numerator.bit_length() + 16)):
                ends = [_mpf_exact(a.lo), _mpf_exact(a.hi)]
            with _iv_prec(120):
                r = getattr(mpmath.iv, op)(mpmath.iv.mpf(ends))
            return max(_frac(r.a), Fraction(-1)), min(_frac(r.b), Fraction(1))
        if op == "cosh":
            m = Fraction(0) if a.lo <= 0 <= a.hi else min(abs(a.lo), abs(a.hi))
            mx = _max_abs(a.lo, a.hi)
            lo = Fraction(1) if m == 0 else _f_bounds("exp", m, m)[0]
            if isinstance(mx, float):
                return lo, INF
            e1 = _f_bounds("exp", mx, mx)[1]
            e2 = _f_bounds("exp", -mx, -mx)[1]
            return max(lo, Fraction(1)), (e1 + e2) / 2  # type: ignore[operator]
        # monotone increasing, clipped to the domain
        lo, hi = a.lo, a.hi
        dom = {"log": Fraction(0), "log2": Fraction(0), "log10": Fraction(0),
               "sqrt": Fraction(0), "log1p": Fraction(-1)}.get(op)
        if dom is not None:
            lo = max(lo, dom)
            if hi < dom:
                return INF, -INF  # empty: every input is outside the domain (NaN)
        return _f_bounds(op, lo, hi)

    def _divmod_iv(self, op: str, a: _AV, b: _AV) -> tuple[Ext, Ext]:
        if self.is_float:
            if op == "mod":
                if b.lo > 0 and not isinstance(b.hi, float):
                    return Fraction(0), b.hi
                if b.hi < 0 and not isinstance(b.lo, float):
                    return b.lo, Fraction(0)
                return -INF, INF
            lo, hi = self._exact_iv("div", [a, b], None)
            fl = lambda v: v if isinstance(v, float) else Fraction(math.floor(v))  # noqa: E731
            return fl(lo), fl(hi)
        parts = []
        if b.hi >= 1:
            parts.append((max(b.lo, Fraction(1)), b.hi))
        if b.lo <= -1:
            parts.append((b.lo, min(b.hi, Fraction(-1))))
        vals: list[Ext] = []
        if b.lo <= 0 <= b.hi:
            vals.append(Fraction(0))  # NumPy: x // 0 == 0, x % 0 == 0
        for plo, phi in parts:
            if op == "floordiv":
                vals += [Fraction(math.floor(p / q)) for p in (a.lo, a.hi) for q in (plo, phi)]  # type: ignore[operator]
            elif plo > 0:
                vals += [Fraction(0), phi - 1]
            else:
                vals += [plo + 1, Fraction(0)]
        return _hull(vals)

    def _pow_iv(self, a: _AV, k: "int | float | None") -> tuple[Ext, Ext]:
        assert k is not None
        if isinstance(k, int):
            if k == 0:
                return Fraction(1), Fraction(1)
            if k > 0:
                ends = [p ** k for p in (a.lo, a.hi)]
                if k % 2 == 0:
                    lo = Fraction(0) if a.lo <= 0 <= a.hi else min(ends)
                    return lo, max(ends)
                return ends[0], ends[1]
            if a.lo <= 0 <= a.hi:
                return -INF, INF
            ends = [Fraction(0) if isinstance(p, float) else 1 / p ** (-k) for p in (a.lo, a.hi)]
            return _hull(ends)
        lo, hi = max(a.lo, Fraction(0)), a.hi
        if hi < 0:
            return INF, -INF
        def pw(x: Ext, side: int) -> Ext:
            if isinstance(x, float):
                return INF if k > 0 else Fraction(0)
            if x == 0:
                return Fraction(0) if k > 0 else INF
            with mpmath.workprec(max(120, x.numerator.bit_length() + 16)):
                xv = _mpf_exact(x)
            with _iv_prec(120):
                r = mpmath.iv.mpf([xv, xv]) ** mpmath.iv.mpf(k)
            return _frac(r.a if side < 0 else r.b)
        e1, e2 = pw(lo, -1), pw(hi, 1)
        if k > 0:
            return e1, e2
        return pw(hi, -1), pw(lo, 1)

    def _transfer_float(self, n: _Node, xs: list[_AV]) -> _AV:
        f = self.fmt
        assert isinstance(f, _FloatFmt)
        rep = n.id
        if n.op in ("max", "min"):
            a, b = xs
            if (n.op == "max" and a.lo >= b.hi) or (n.op == "min" and a.hi <= b.lo):
                rep = a.rep
            elif (n.op == "max" and b.lo >= a.hi) or (n.op == "min" and b.hi <= a.lo):
                rep = b.rep
        if n.op == "abs" and xs[0].lo >= 0:
            rep = xs[0].rep
        lo, hi = self._exact_iv(n.op, xs, n.value)
        nan = any(x.nan for x in xs) or self._nan_possible(n, xs)
        if lo > hi:  # every input outside the domain
            return _AV(Fraction(0), Fraction(0), True, -1)
        rlo, rhi = _round(lo, f, "d"), _round(hi, f, "u")
        if n.op in _TRANSCENDENTAL:
            rlo = _ulp_step(rlo, f, -TRANSCENDENTAL_ULPS)
            rhi = _ulp_step(rhi, f, TRANSCENDENTAL_ULPS)
            if n.op in ("tanh",):
                rlo, rhi = max(rlo, Fraction(-1)), min(rhi, Fraction(1))
            if n.op in ("exp", "sqrt", "cosh", "hypot") or (n.op == "pow" and n.value is not None
                                                            and isinstance(n.value, int)
                                                            and n.value % 2 == 0):
                rlo = max(rlo, Fraction(0))
        return _AV(rlo, rhi, nan, rep)

    def _transfer_int(self, n: _Node, xs: list[_AV]) -> _AV:
        f = self.fmt
        assert isinstance(f, _IntFmt)
        rep = n.id
        if n.op in ("max", "min"):
            a, b = xs
            if (n.op == "max" and a.lo >= b.hi) or (n.op == "min" and a.hi <= b.lo):
                rep = a.rep
            elif (n.op == "max" and b.lo >= a.hi) or (n.op == "min" and b.hi <= a.lo):
                rep = b.rep
        if n.op == "abs" and xs[0].lo >= 0:
            rep = xs[0].rep
        lo, hi = self._exact_iv(n.op, xs, n.value)
        if lo < f.min or hi > f.max:  # wraps: any value of the type
            return _AV(Fraction(f.min), Fraction(f.max), False, -1)
        return _AV(lo, hi, False, rep)

    def _nan_possible(self, n: _Node, xs: list[_AV]) -> bool:
        op = n.op
        a = xs[0]
        b = xs[1] if len(xs) > 1 else None
        pinf = lambda x: x.hi == INF  # noqa: E731
        ninf = lambda x: x.lo == -INF  # noqa: E731
        anyinf = lambda x: pinf(x) or ninf(x)  # noqa: E731
        zero = lambda x: x.lo <= 0 <= x.hi  # noqa: E731
        if op == "add":
            assert b is not None
            return (pinf(a) and ninf(b)) or (ninf(a) and pinf(b))
        if op == "sub":
            assert b is not None
            if a.rep == b.rep and a.rep >= 0:
                return anyinf(a)
            return (pinf(a) and pinf(b)) or (ninf(a) and ninf(b))
        if op == "mul":
            assert b is not None
            return (zero(a) and anyinf(b)) or (zero(b) and anyinf(a))
        if op == "div":
            assert b is not None
            return (zero(a) and zero(b)) or (anyinf(a) and anyinf(b))
        if op in ("floordiv", "mod"):
            assert b is not None
            return zero(b) or anyinf(a) or anyinf(b)
        if op in ("sqrt", "log", "log2", "log10"):
            return a.lo < 0
        if op == "log1p":
            return a.lo < -1
        if op in ("sin", "cos"):
            return anyinf(a)
        if op == "pow":
            k = n.value
            if isinstance(k, float):
                return a.lo < 0
            return False
        return False

    # ---- events: which classes apply where, and the interval verdict ----------

    def applicable(self, n: _Node) -> list[str]:
        op = n.op
        if not self.is_float:
            if op in ("add", "sub", "mul", "neg", "abs", "pow"):
                return ["overflow"]
            if op == "floordiv":
                return ["overflow", "division_by_zero_possible"]
            if op == "mod":
                return ["division_by_zero_possible"]
            return []
        kinds: list[str] = []
        if op in ("add", "sub", "mul", "div", "floordiv", "pow", "exp", "expm1", "sinh",
                  "cosh", "hypot"):
            kinds.append("overflow")
        if op in ("mul", "div", "pow", "exp"):
            kinds.append("underflow_to_zero")
        if op in ("add", "sub", "mul", "div", "pow", "exp"):
            kinds.append("subnormal")
        if n.int_tracked and (op in ("add", "sub", "mul", "pow", "var", "const")) and \
                (op != "var" or self.var_kind.get(n.name) == "int"):
            kinds.append("integer_precision_loss")
        if op in ("add", "sub", "mul", "div", "floordiv", "mod", "sqrt", "log", "log2",
                  "log10", "log1p", "sin", "cos") or (op == "pow" and isinstance(n.value, float)):
            kinds.append("nan_possible")
        if op in ("div", "floordiv", "mod", "log", "log2", "log10", "log1p") or \
                (op == "pow" and n.value is not None and n.value < 0):
            kinds.append("division_by_zero_possible")
        if op == "const":
            kinds.append("overflow")
        return kinds

    def possible(self, n: _Node, kind: str, xs: list[_AV], out: _AV, box: Mapping[str, tuple[int, int]]) -> bool:
        """Interval verdict: can ``kind`` happen at ``n`` on this box? (False = proved.)"""
        if not self.is_float:
            f = self.fmt
            assert isinstance(f, _IntFmt)
            if kind == "overflow":
                lo, hi = self._exact_iv(n.op, xs, n.value)
                return bool(lo < f.min or hi > f.max)
            return bool(xs[1].lo <= 0 <= xs[1].hi)
        f = self.fmt
        assert isinstance(f, _FloatFmt)
        if n.op == "const":
            v = Fraction(n.value) if n.value is not None else Fraction(0)  # type: ignore[arg-type]
            r = _round(v, f, "n")
            if kind == "overflow":
                return isinstance(r, float)
            return r != v
        if n.op == "var":  # declared integer input
            lo, hi = box[n.name]
            return _first_unrepresentable(lo, hi, f) is not None
        fin = [self._finite_part(x) for x in xs]
        if kind == "overflow":
            if any(p is None for p in fin):
                return False
            fx = [_AV(p[0], p[1], False, x.rep) for p, x in zip(fin, xs)]  # type: ignore[index]
            if n.op in ("div", "floordiv"):
                b = fx[1]
                pieces = []
                if b.hi > 0:
                    pieces.append(_AV(max(b.lo, f.sub), b.hi, False, b.rep))
                if b.lo < 0:
                    pieces.append(_AV(b.lo, min(b.hi, -f.sub), False, b.rep))
                return any(self._overflows(n, [fx[0], p]) for p in pieces)
            if n.op == "pow" and n.value is not None and n.value < 0:
                a = fx[0]
                pieces = []
                if a.hi > 0:
                    pieces.append(_AV(max(a.lo, f.sub), a.hi, False, a.rep))
                if a.lo < 0:
                    pieces.append(_AV(a.lo, min(a.hi, -f.sub), False, a.rep))
                return any(self._overflows(n, [p]) for p in pieces)
            return self._overflows(n, fx)
        if kind == "nan_possible":
            return self._nan_possible(n, xs)
        if kind == "division_by_zero_possible":
            a = xs[0]
            if n.op in ("log", "log2", "log10"):
                return bool(a.lo <= 0 <= a.hi)
            if n.op == "log1p":
                return bool(a.lo <= -1 <= a.hi)
            if n.op == "pow":
                return bool(a.lo <= 0 <= a.hi)
            return bool(xs[1].lo <= 0 <= xs[1].hi)
        if kind in ("underflow_to_zero", "subnormal"):
            thr = f.sub / 2 if kind == "underflow_to_zero" else f.tiny
            m = self._min_nonzero_result(n, fin, xs)
            if m is None:
                return False
            if kind == "underflow_to_zero":
                return bool(m <= thr)
            return bool(m < thr)
        if kind == "integer_precision_loss":
            lo, hi = self._exact_iv(n.op, xs, n.value)
            if lo > hi:
                return False
            g = self._int_grain(n, xs)
            return bool(_max_abs(lo, hi) > Fraction(2) ** f.p * g)
        raise AssertionError(kind)

    def _int_grain(self, n: _Node, xs: list[_AV]) -> Ext:
        """A power of two that divides every exact result of this integer-valued node.

        Operands are integers that the type already holds, so each is a multiple of
        the quantum of its smallest nonzero magnitude (and of 1). A sum is a multiple
        of the smaller grain, a product of the product of grains; every multiple
        j*g with |j| <= 2**p is representable, which proves e.g. that the difference
        of two float32 millisecond timestamps (both multiples of 2**17) is exact.
        """
        f = self.fmt
        assert isinstance(f, _FloatFmt)
        gs: list[Ext] = []
        for a, x in zip(n.args, xs):
            m = _min_abs_nz(x.lo, x.hi, Fraction(1))
            g = max(Fraction(1), _quantum(m, f)) if not isinstance(m, float) else Fraction(1)
            if self.nodes[a].op == "const" and x.lo == x.hi and x.lo != 0:
                v = abs(int(x.lo))
                g = max(g, Fraction((v & -v)))  # largest power of two dividing it
            gs.append(g)
        if n.op in ("add", "sub"):
            return min(gs)
        if n.op == "mul":
            return gs[0] * gs[1]  # type: ignore[operator]
        if n.op == "pow" and isinstance(n.value, int):
            return gs[0] ** n.value  # type: ignore[operator]
        return Fraction(1)

    def _finite_part(self, x: _AV) -> "tuple[Ext, Ext] | None":
        f = self.fmt
        assert isinstance(f, _FloatFmt)
        lo, hi = max(x.lo, -f.max), min(x.hi, f.max)
        if lo > hi:
            return None
        return lo, hi

    def _overflows(self, n: _Node, xs: list[_AV]) -> bool:
        f = self.fmt
        assert isinstance(f, _FloatFmt)
        lo, hi = self._exact_iv(n.op, xs, n.value)
        if lo > hi:
            return False
        rlo, rhi = _round(lo, f, "d"), _round(hi, f, "u")
        if n.op in _TRANSCENDENTAL:
            rlo = _ulp_step(rlo, f, -TRANSCENDENTAL_ULPS)
            rhi = _ulp_step(rhi, f, TRANSCENDENTAL_ULPS)
        return isinstance(rlo, float) or isinstance(rhi, float)

    def _min_nonzero_result(self, n: _Node, fin: list, xs: list[_AV]) -> "Ext | None":
        """Lower bound on |exact result| over results that are not exactly zero."""
        f = self.fmt
        assert isinstance(f, _FloatFmt)
        if any(p is None for p in fin):
            return None
        op = n.op
        sub = f.sub
        if op in ("add", "sub"):
            ms = [_min_abs_nz(p[0], p[1], sub) for p in fin]
            bound = min(_quantum(m, f) for m in ms)
            lo, hi = self._exact_iv(op, xs, n.value)
            if lo == 0 == hi:
                return None
            near = Fraction(0) if lo <= 0 <= hi else min(abs(lo), abs(hi))
            return max(bound, near)
        if op == "mul":
            (alo, ahi), (blo, bhi) = fin
            ma, mb = _min_abs_nz(alo, ahi, sub), _min_abs_nz(blo, bhi, sub)
            if xs[0].rep == xs[1].rep and xs[0].rep >= 0:
                return _emul(ma, ma)
            return _emul(ma, mb)
        if op == "div":
            (alo, ahi), (blo, bhi) = fin
            mb = _max_abs(blo, bhi)
            ma = _min_abs_nz(alo, ahi, sub)
            return ma / mb if mb != 0 else None  # type: ignore[operator]
        if op == "exp":
            lo = fin[0][0]
            return _f_bounds("exp", lo, lo)[0]
        if op == "pow":
            k = n.value
            assert k is not None
            alo, ahi = fin[0]
            if k > 0:
                m = _min_abs_nz(alo, ahi, sub)
                if isinstance(m, float):
                    return None
                return self._pow_iv(_AV(m, m), k)[0] if isinstance(k, int) else \
                    self._pow_iv(_AV(m, m), k)[0]
            mx = _max_abs(alo, ahi)
            if isinstance(mx, float):
                return Fraction(0)
            return self._pow_iv(_AV(mx, mx), k)[0]
        return None

    # ---- execution -------------------------------------------------------

    def execute(self, pts: dict[str, np.ndarray], want: "set[tuple[int, str]] | None" = None,
                scores: bool = False) -> tuple[list[np.ndarray], dict[tuple[int, str], np.ndarray],
                                               dict[tuple[int, str], np.ndarray]]:
        """Run every node on the points; return values, event masks and search scores."""
        npts = len(next(iter(pts.values()))) if pts else 1
        vals: list[np.ndarray] = []
        masks: dict[tuple[int, str], np.ndarray] = {}
        sc: dict[tuple[int, str], np.ndarray] = {}
        m = self.m
        for n in self.nodes:
            if n.op == "var":
                raw = pts[n.name]
                out = m.cast(raw)
            elif n.op == "const":
                assert n.value is not None
                out = m.const(n.value, npts)
            else:
                out = m.apply(n.op, [vals[a] for a in n.args], n.value)
            vals.append(out)
            kinds = [k for k in self.applicable(n) if want is None or (n.id, k) in want]
            if not kinds:
                continue
            ops = [vals[a] for a in n.args]
            for k in kinds:
                mk, s = self._detect(n, k, ops, out, pts, scores)
                masks[(n.id, k)] = mk
                if s is not None:
                    sc[(n.id, k)] = s
        return vals, masks, sc

    def _detect(self, n: _Node, kind: str, ops: list[np.ndarray], out: np.ndarray,
                pts: dict[str, np.ndarray], want_score: bool) -> tuple[np.ndarray, "np.ndarray | None"]:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not self.is_float:
                return self._detect_int(n, kind, ops, out, want_score)
            return self._detect_float(n, kind, ops, out, pts, want_score)

    def _detect_int(self, n: _Node, kind: str, ops: list[np.ndarray], out: np.ndarray,
                    want_score: bool) -> tuple[np.ndarray, "np.ndarray | None"]:
        f = self.fmt
        assert isinstance(f, _IntFmt)
        if kind == "division_by_zero_possible":
            b = ops[1]
            mask = b == 0
            return mask, (-np.abs(b.astype(np.float64)) if want_score else None)
        ex = _exact_int_op(n, [o.astype(object) for o in ops])
        mask = np.array([(e is not None) and (e < f.min or e > f.max) for e in ex], dtype=bool)
        score = None
        if want_score:
            score = np.array([-INF if e is None else float(max(e - f.max, f.min - e)) for e in ex])
        return mask, score

    def _detect_float(self, n: _Node, kind: str, ops: list[np.ndarray], out: np.ndarray,
                      pts: dict[str, np.ndarray], want_score: bool) -> tuple[np.ndarray, "np.ndarray | None"]:
        f = self.fmt
        assert isinstance(f, _FloatFmt)
        o64 = out.astype(np.float64)
        x64 = [o.astype(np.float64) for o in ops]
        op = n.op
        if op == "const":
            v = Fraction(n.value) if n.value is not None else Fraction(0)  # type: ignore[arg-type]
            r = _round(v, f, "n")
            hit = isinstance(r, float) if kind == "overflow" else (r != v)
            return np.full(len(o64), hit), None
        if op == "var":
            raw = pts[n.name]
            mask = np.array([Fraction(int(r)) != Fraction(float(c)) if math.isfinite(c) else True
                             for r, c in zip(raw.tolist(), o64.tolist())], dtype=bool)
            score = np.abs(raw.astype(np.float64)) if want_score else None
            return mask, score
        fin = np.ones(len(o64), dtype=bool)
        for x in x64:
            fin &= np.isfinite(x)
        nonan = np.ones(len(o64), dtype=bool)
        for x in x64:
            nonan &= ~np.isnan(x)
        pole = self._pole_mask(op, x64, n)
        if kind == "overflow":
            mask = fin & np.isinf(o64) & ~pole
            score = None
            if want_score:
                sh = _shadow(op, x64, n.value)
                with np.errstate(all="ignore"):
                    score = np.where(np.isnan(sh), -INF, np.log2(np.abs(sh) + 1e-300))
                    if op in ("add", "sub", "mul", "hypot"):
                        # tie-break: a sum or product overflows when BOTH operands are
                        # large; without this the search stalls on a plateau where
                        # the bigger operand alone decides the float64 shadow
                        small = np.minimum(np.abs(x64[0]), np.abs(x64[1]))
                        score = score + 1e-3 * np.log2(small + 1e-300)
                if op in ("exp", "expm1", "sinh", "cosh"):
                    score = np.abs(x64[0]) * np.sign(x64[0]) if op != "cosh" else np.abs(x64[0])
                # an overflow HERE needs finite operands: an inf that arrives from
                # upstream must not pull the search past the real witness
                score = np.where(fin, score, -INF)
            return mask, score
        if kind == "nan_possible":
            mask = nonan & np.isnan(o64)
            score = None
            if want_score:
                if op in ("sqrt", "log", "log2", "log10"):
                    score = -x64[0]
                elif op == "log1p":
                    score = -(x64[0] + 1)
                else:
                    score = np.zeros(len(o64))
            return mask, score
        if kind == "division_by_zero_possible":
            mask = pole
            score = None
            if want_score:
                if op in ("log", "log2", "log10", "pow"):
                    score = -np.abs(x64[0])
                elif op == "log1p":
                    score = -np.abs(x64[0] + 1)
                else:
                    score = -np.abs(x64[1])
            return mask, score
        if kind in ("underflow_to_zero", "subnormal"):
            nz = self._exact_nonzero(op, x64, n) & fin
            if kind == "underflow_to_zero":
                mask = nz & (o64 == 0)
            else:
                mask = nz & (o64 != 0) & (np.abs(o64) < float(f.tiny))
            score = None
            if want_score:
                sh = np.abs(_shadow(op, x64, n.value))
                if op == "exp":
                    sh = -x64[0]
                    score = np.where(nz, sh, -INF)
                else:
                    with np.errstate(all="ignore"):
                        score = np.where(nz & (sh > 0), -np.log2(np.where(sh > 0, sh, 1.0)), -INF)
            return mask, score
        if kind == "integer_precision_loss":
            ints = [np.array([int(v) if math.isfinite(v) else None for v in x.tolist()], dtype=object)
                    for x in x64]
            ex = _exact_int_op(n, ints)
            mask = np.array([e is not None and math.isfinite(c) and Fraction(c) != e
                             for e, c in zip(ex, o64.tolist())], dtype=bool)
            score = None
            if want_score:
                score = np.array([-INF if e is None else float(abs(e)) for e in ex])
            return mask, score
        raise AssertionError(kind)

    def _pole_mask(self, op: str, x64: list[np.ndarray], n: _Node) -> np.ndarray:
        if op in ("div", "floordiv", "mod"):
            return x64[1] == 0
        if op in ("log", "log2", "log10"):
            return x64[0] == 0
        if op == "log1p":
            return x64[0] == -1
        if op == "pow" and n.value is not None and n.value < 0:
            return x64[0] == 0
        return np.zeros(len(x64[0]) if x64 else 0, dtype=bool)

    def _exact_nonzero(self, op: str, x64: list[np.ndarray], n: _Node) -> np.ndarray:
        if op == "mul":
            return (x64[0] != 0) & (x64[1] != 0)
        if op == "div":
            return (x64[0] != 0) & np.isfinite(x64[1]) & (x64[1] != 0)
        if op == "exp":
            return np.isfinite(x64[0])
        if op == "pow":
            return x64[0] != 0
        if op in ("add", "sub"):
            return np.ones(len(x64[0]), dtype=bool)  # out == 0 only when exact == 0
        return np.zeros(len(x64[0]), dtype=bool)

    # ---- exact confirmation of one observed event ------------------------------

    def confirm(self, n: _Node, kind: str, operands: list[Any], out: Any, raw: "int | None") -> "tuple[bool, str]":
        """Recompute the operation exactly from the operands the program saw."""
        if n.op == "var":
            assert raw is not None
            c = float(out)
            return (not math.isfinite(c) or Fraction(c) != raw), str(raw)
        if not self.is_float:
            f = self.fmt
            assert isinstance(f, _IntFmt)
            xs = [int(o) for o in operands]
            if kind == "division_by_zero_possible":
                return xs[1] == 0, "undefined"
            e = _exact_int_op(n, [np.array([x], dtype=object) for x in xs])[0]
            ok = e is not None and (e < f.min or e > f.max) and e != int(out)
            return ok, str(e)
        f = self.fmt
        assert isinstance(f, _FloatFmt)
        xs_f = [float(o) for o in operands]
        o = float(out)
        op = n.op
        if kind == "division_by_zero_possible":
            if op in ("log", "log2", "log10", "pow"):
                return xs_f[0] == 0, "pole"
            if op == "log1p":
                return xs_f[0] == -1, "pole"
            return xs_f[1] == 0, "pole"
        if kind == "nan_possible":
            if not math.isnan(o) or any(math.isnan(x) for x in xs_f):
                return False, ""
            return _nan_reason(op, xs_f, n.value), "nan"
        exact = _exact_real(op, xs_f, n.value)
        if exact is None:
            return False, ""
        ex_s = _show_exact(exact)
        if kind == "overflow":
            if not (math.isinf(o) and all(math.isfinite(x) for x in xs_f)):
                return False, ex_s
            if isinstance(exact, Fraction) and op not in _TRANSCENDENTAL:
                return isinstance(_round(exact, f, "n"), float), ex_s
            return _gt_abs(exact, f.max), ex_s
        if kind == "underflow_to_zero":
            if o != 0 or exact == 0:
                return False, ex_s
            if isinstance(exact, Fraction) and op not in _TRANSCENDENTAL:
                return _round(exact, f, "n") == 0, ex_s
            return _lt_abs(exact, f.sub), ex_s
        if kind == "subnormal":
            if not (o != 0 and abs(o) < float(f.tiny)) or exact == 0:
                return False, ex_s
            return _lt_abs(exact, f.tiny), ex_s
        if kind == "integer_precision_loss":
            if not isinstance(exact, Fraction) or exact.denominator != 1 or not math.isfinite(o):
                return False, ex_s
            return Fraction(o) != exact, ex_s
        raise AssertionError(kind)


def _first_unrepresentable(lo: int, hi: int, f: _FloatFmt) -> "int | None":
    """Smallest-magnitude integer in [lo, hi] that the float format cannot hold."""
    lim = 1 << f.p
    best: "int | None" = None
    for sign in (1, -1):
        a, b = (lo, hi) if sign > 0 else (-hi, -lo)
        start = max(a, lim + 1)
        for c in (start, start + 1, start + 2):
            if c <= b and _round(Fraction(c), f, "n") != c:
                cand = sign * c
                if best is None or abs(cand) < abs(best):
                    best = cand
                break
    return best


def _exact_int_op(n: _Node, xs: list[np.ndarray]) -> list["int | None"]:
    """Exact integer result per point (object arrays of Python ints)."""
    op = n.op
    out: list["int | None"] = []
    cols = [x.tolist() for x in xs]
    for row in zip(*cols):
        if any(v is None for v in row):
            out.append(None)
            continue
        a = int(row[0])
        b = int(row[1]) if len(row) > 1 else 0
        if op == "add":
            out.append(a + b)
        elif op == "sub":
            out.append(a - b)
        elif op == "mul":
            out.append(a * b)
        elif op == "neg":
            out.append(-a)
        elif op == "abs":
            out.append(abs(a))
        elif op == "max":
            out.append(max(a, b))
        elif op == "min":
            out.append(min(a, b))
        elif op == "pow":
            out.append(a ** int(n.value))  # type: ignore[arg-type]
        elif op == "floordiv":
            out.append(None if b == 0 else a // b)
        elif op == "mod":
            out.append(None if b == 0 else a % b)
        else:
            out.append(None)
    return out


def _shadow(op: str, x64: list[np.ndarray], k: "int | float | None") -> np.ndarray:
    """The same operation in float64 (a wider-range guide for the search)."""
    with np.errstate(all="ignore"):
        if op == "pow":
            return np.power(x64[0], float(k))  # type: ignore[arg-type]
        return np.asarray(_UFUNC[op](*x64), dtype=np.float64)


def _nan_reason(op: str, xs: list[float], k: "int | float | None") -> bool:
    a = xs[0]
    b = xs[1] if len(xs) > 1 else 0.0
    if op == "add":
        return math.isinf(a) and math.isinf(b) and (a > 0) != (b > 0)
    if op == "sub":
        return math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0)
    if op == "mul":
        return (a == 0 and math.isinf(b)) or (b == 0 and math.isinf(a))
    if op == "div":
        return (a == 0 and b == 0) or (math.isinf(a) and math.isinf(b))
    if op in ("floordiv", "mod"):
        return b == 0 or math.isinf(a) or math.isinf(b)
    if op in ("sqrt", "log", "log2", "log10"):
        return a < 0
    if op == "log1p":
        return a < -1
    if op in ("sin", "cos"):
        return math.isinf(a)
    if op == "pow":
        return isinstance(k, float) and a < 0
    return False


def _exact_real(op: str, xs: list[float], k: "int | float | None") -> "Fraction | Any | None":
    """Exact result (Fraction) or a 200-bit mpmath value; None where undefined."""
    if any(not math.isfinite(x) for x in xs):
        return None
    fx = [Fraction(x) for x in xs]
    a = fx[0]
    b = fx[1] if len(fx) > 1 else Fraction(0)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return None if b == 0 else a / b
    if op == "floordiv":
        return None if b == 0 else Fraction(math.floor(a / b))
    if op == "neg":
        return -a
    if op == "abs":
        return abs(a)
    if op == "pow" and isinstance(k, int):
        if a == 0 and k < 0:
            return None
        return a ** k
    with mpmath.workprec(200):
        x = _mpf_exact(a)
        try:
            if op == "pow":
                if a < 0:
                    return None
                return x ** mpmath.mpf(k)
            if op == "hypot":
                return mpmath.sqrt(x * x + _mpf_exact(b) ** 2)
            fn = {"arctan": "atan"}.get(op, op)
            if op in ("log", "log2", "log10") and a <= 0:
                return None
            if op == "log1p" and a <= -1:
                return None
            if op == "sqrt":
                return None if a < 0 else mpmath.sqrt(x)
            if op == "log2":
                return mpmath.log(x, 2)
            if op == "log10":
                return mpmath.log10(x)
            return getattr(mpmath, fn)(x)
        except (ValueError, ZeroDivisionError):
            return None


def _show_exact(v: Any) -> str:
    if isinstance(v, Fraction):
        return str(v)
    return mpmath.nstr(v, 20)


# ---------------------------------------------------------------------------
# Search for witnesses, Z3, bisection
# ---------------------------------------------------------------------------

class _Search:
    def __init__(self, an: _Analyzer, seed: int = 0) -> None:
        self.an = an
        self.rng = random.Random(seed)

    def candidates(self, box: Mapping[str, tuple[int, int]], n_random: int = 384) -> dict[str, np.ndarray]:
        an = self.an
        vs = an.vars
        cols: dict[str, list[int]] = {v: [] for v in vs}

        def push(pt: Mapping[str, int]) -> None:
            for v in vs:
                cols[v].append(int(pt[v]))

        mids = {v: (box[v][0] + box[v][1]) // 2 for v in vs}
        corners = list(itertools.product(*[(box[v][0], box[v][1]) for v in vs])) if len(vs) <= 8 \
            else [tuple(self.rng.choice([box[v][0], box[v][1]]) for v in vs) for _ in range(256)]
        for c in corners:
            push(dict(zip(vs, c)))
        push(mids)
        specials = {v: self._specials(v, box[v]) for v in vs}
        for v in vs:
            for s in specials[v]:
                pt = dict(mids)
                pt[v] = s
                push(pt)
                for c in corners[:16]:
                    pt2 = dict(zip(vs, c))
                    pt2[v] = s
                    push(pt2)
        # all specials crossed (small), and random points
        if len(vs) <= 3:
            for combo in itertools.product(*[specials[v][:6] or [mids[v]] for v in vs]):
                push(dict(zip(vs, combo)))
        for _ in range(n_random):
            push({v: self.rng.randint(box[v][0], box[v][1]) for v in vs})
        # random in VALUE space for float coordinates (ordinals crowd near zero)
        for _ in range(n_random // 2):
            pt = {}
            for v in vs:
                lo, hi = box[v]
                if an.var_kind[v] == "float":
                    assert isinstance(an.fmt, _FloatFmt)
                    # halves first: hi - lo of a full float64 range overflows
                    x = 2 * self.rng.uniform(float(_val_of(lo, an.fmt)) / 2, float(_val_of(hi, an.fmt)) / 2)
                    o = int(_f64_to_ords(np.array([x]), an.fmt)[0])
                    pt[v] = min(max(o, lo), hi)
                else:
                    pt[v] = self.rng.randint(lo, hi)
            push(pt)
        return {v: np.array(cols[v], dtype=object) for v in vs}

    def _specials(self, v: str, rng_: tuple[int, int]) -> list[int]:
        an = self.an
        lo, hi = rng_
        out: list[int] = []
        if an.var_kind[v] == "float":
            f = an.fmt
            assert isinstance(f, _FloatFmt)
            vals = [Fraction(0), Fraction(1), Fraction(-1), f.tiny, -f.tiny, f.sub, -f.sub]
            cands = [_ord_of(x, f) for x in vals]
            cands += [lo + 1, hi - 1]
        else:
            cands = [0, 1, -1, lo + 1, hi - 1]
            if an.is_float:  # integer input of a float type: around 2**p
                assert isinstance(an.fmt, _FloatFmt)
                p2 = 1 << an.fmt.p
                cands += [p2, p2 + 1, -p2 - 1]
            else:
                f2 = an.fmt
                assert isinstance(f2, _IntFmt)
                cands += [f2.min, f2.max, f2.min + 1, f2.max - 1, -1]
        for c in cands:
            if lo <= c <= hi and c not in out:
                out.append(c)
        return out

    def _moves(self, v: str, c: int, rng_: tuple[int, int], span: int) -> list[int]:
        """Neighbours of coordinate ``c``: steps of 2**e in the coordinate (geometric for
        floats), plus, for floats, linear steps of width/2**j in value space."""
        lo, hi = rng_
        k = max(span.bit_length(), 1)
        exps = list(range(k, -1, -1)) if k < 40 else list(range(k, k - 40, -1)) + list(range(6, -1, -1))
        out = [min(max(c + sgn * (1 << e), lo), hi) for e in exps for sgn in (1, -1)]
        an = self.an
        if an.var_kind[v] == "float":
            f = an.fmt
            assert isinstance(f, _FloatFmt)
            x = float(_val_of(c, f))
            width = float(_val_of(hi, f)) - float(_val_of(lo, f))
            if math.isfinite(width) and width > 0:
                steps = np.array([x + sgn * width / 2 ** j for j in range(0, 24) for sgn in (1, -1)])
                o = _f64_to_ords(np.clip(steps, float(_val_of(lo, f)), float(_val_of(hi, f))), f)
                out += [int(min(max(int(t), lo), hi)) for t in o]
        return out

    def evaluate(self, coords: dict[str, np.ndarray], want: set[tuple[int, str]], scores: bool
                 ) -> tuple[list[np.ndarray], dict, dict, dict[str, np.ndarray]]:
        pts = self.an.coords_to_arrays(coords)
        vals, masks, sc = self.an.execute(pts, want, scores)
        return vals, masks, sc, pts

    def hill_climb(self, box: Mapping[str, tuple[int, int]], ev: tuple[int, str],
                   start: dict[str, np.ndarray], score: np.ndarray, rounds: int = 40,
                   top: int = 6) -> "dict[str, int] | None":
        """Pattern search in coordinate space toward the event, starting from good points."""
        an = self.an
        vs = an.vars
        order = np.argsort(-np.nan_to_num(score.astype(np.float64), nan=-np.inf, neginf=-1e308))[:top]
        best = [{v: int(start[v][i]) for v in vs} for i in order]
        best_s = [float(score[i]) for i in order]
        spans = {v: box[v][1] - box[v][0] for v in vs}
        for _ in range(rounds):
            if time.perf_counter() > an.deadline:
                return None
            cols: dict[str, list[int]] = {v: [] for v in vs}
            for b in best:
                for v in vs:
                    for c in self._moves(v, b[v], box[v], spans[v]):
                        for w in vs:
                            cols[w].append(c if w == v else b[w])
            coords = {v: np.array(cols[v], dtype=object) for v in vs}
            _, masks, sc, _ = self.evaluate(coords, {ev}, True)
            hit = np.flatnonzero(masks[ev])
            if hit.size:
                i = int(hit[0])
                return {v: int(coords[v][i]) for v in vs}
            s = sc[ev].astype(np.float64)
            improved = False
            idx = np.argsort(-np.nan_to_num(s, nan=-np.inf, neginf=-1e308))[:top]
            nb, ns = [], []
            for i in idx:
                nb.append({v: int(coords[v][i]) for v in vs})
                ns.append(float(s[i]))
            if ns and max(ns) > max(best_s):
                improved = True
            pool = sorted(zip(best_s + ns, range(len(best) + len(nb)), best + nb), key=lambda t: -t[0])
            seen, best, best_s = set(), [], []
            for s_, _, p in pool:
                key = tuple(p[v] for v in vs)
                if key in seen:
                    continue
                seen.add(key)
                best.append(p)
                best_s.append(s_)
                if len(best) == top:
                    break
            if not improved:
                return None
        return None


class _Run:
    """Drives one analysis: statuses for every (node, kind)."""

    def __init__(self, an: _Analyzer, use_z3: bool = True) -> None:
        self.an = an
        self.search = _Search(an)
        self.status: dict[tuple[int, str], str] = {}
        self.method: dict[tuple[int, str], str] = {}
        self.wit: dict[tuple[int, str], dict[str, Any]] = {}
        self.use_z3 = use_z3
        self.complete = True

    def count(self, box: Mapping[str, tuple[int, int]]) -> int:
        c = 1
        for v in self.an.vars:
            c *= box[v][1] - box[v][0] + 1
        return c

    def interval_possible(self, box: Mapping[str, tuple[int, int]], want: Iterable[tuple[int, str]]
                          ) -> tuple[set[tuple[int, str]], list[_AV]]:
        an = self.an
        avs = an.run_box(box)
        out = set()
        for (nid, kind) in want:
            n = an.nodes[nid]
            xs = [avs[a] for a in n.args]
            if an.possible(n, kind, xs, avs[nid], box):
                out.add((nid, kind))
        return out, avs

    def record_hits(self, coords: dict[str, np.ndarray], pts: dict[str, np.ndarray],
                    vals: list[np.ndarray], masks: dict) -> None:
        an = self.an
        for ev, mk in masks.items():
            if self.status.get(ev) == "confirmed":
                continue
            for i in np.flatnonzero(mk)[:4]:
                if self._confirm_at(ev, coords, pts, vals, int(i)):
                    break

    def _confirm_at(self, ev: tuple[int, str], coords: dict[str, np.ndarray],
                    pts: dict[str, np.ndarray], vals: list[np.ndarray], i: int) -> bool:
        an = self.an
        nid, kind = ev
        n = an.nodes[nid]
        operands = [vals[a][i] for a in n.args]
        raw = int(pts[n.name][i]) if n.op == "var" else None
        ok, exact = an.confirm(n, kind, operands, vals[nid][i], raw)
        if not ok:
            return False
        wit = {v: an.witness_text(v, int(coords[v][i])) for v in an.vars}
        self.status[ev] = "confirmed"
        self.method[ev] = f"execution ({an.m.method}) + exact recomputation"
        self.wit[ev] = {"witness": wit, "observed": _show_np(vals[nid][i]),
                        "operands": tuple(_show_np(o) for o in operands), "exact": exact}
        return True

    def run(self) -> None:
        an = self.an
        box0 = an.box0
        every = {(n.id, k) for n in an.nodes for k in an.applicable(n)}
        poss, _ = self.interval_possible(box0, every)
        for ev in every - poss:
            self.status[ev] = "proved_safe"
            self.method[ev] = "interval"
        # a literal the type cannot hold misbehaves at EVERY input: execute it once,
        # at the lower corner of the ranges, and report that corner as the witness
        for ev in list(poss):
            n = an.nodes[ev[0]]
            if n.op != "const":
                continue
            poss.discard(ev)
            assert n.value is not None
            got = an.m.const(n.value, 1)[0]
            g = float(got)
            exact = Fraction(n.value)
            hit = math.isinf(g) if ev[1] == "overflow" else (not math.isfinite(g) or Fraction(g) != exact)
            if hit:
                self.status[ev] = "confirmed"
                self.method[ev] = f"execution ({an.m.method}) of the literal"
                self.wit[ev] = {"witness": {v: an.witness_text(v, box0[v][0]) for v in an.vars},
                                "observed": _show_np(got), "operands": (), "exact": str(exact)}
        if not poss:
            return
        open_ = set(poss)
        if self.count(box0) <= ROOT_EXHAUSTIVE:
            self._exhaustive(box0, open_, "exhaustive")
            return
        self._global_search(box0, open_)
        open_ = {e for e in open_ if self.status.get(e) != "confirmed"}
        if open_ and self.use_z3 and not an.is_float:
            self._z3(open_)
            open_ = {e for e in open_ if e not in self.status}
        if open_:
            self._bisect(box0, open_)

    def _exhaustive(self, box: Mapping[str, tuple[int, int]], open_: set[tuple[int, str]], label: str) -> None:
        vs = self.an.vars
        grids = np.meshgrid(*[np.arange(box[v][0], box[v][1] + 1, dtype=object) for v in vs],
                            indexing="ij") if vs else []
        coords = {v: g.ravel() for v, g in zip(vs, grids)}
        vals, masks, _, pts = self.search.evaluate(coords, open_, False)
        self.record_hits(coords, pts, vals, masks)
        for ev in open_:
            if ev not in self.status and not masks[ev].any():
                pass  # safe on this box; caller decides globally
        if label == "exhaustive":
            for ev in open_:
                if ev not in self.status:
                    self.status[ev] = "proved_safe"
                    self.method[ev] = f"exhaustive execution of {self.count(box)} inputs"

    def _global_search(self, box: Mapping[str, tuple[int, int]], open_: set[tuple[int, str]]) -> None:
        coords = self.search.candidates(box)
        vals, masks, sc, pts = self.search.evaluate(coords, open_, True)
        self.record_hits(coords, pts, vals, masks)
        for ev in sorted(open_):
            if self.status.get(ev) == "confirmed" or ev not in sc:
                continue
            if time.perf_counter() > self.an.deadline:
                self.complete = False
                return
            pt = self.search.hill_climb(box, ev, coords, sc[ev])
            if pt is not None:
                c2 = {v: np.array([pt[v]], dtype=object) for v in self.an.vars}
                vals2, masks2, _, pts2 = self.search.evaluate(c2, open_, False)
                self.record_hits(c2, pts2, vals2, masks2)

    def _bisect(self, box0: Mapping[str, tuple[int, int]], open_: set[tuple[int, str]]) -> None:
        an = self.an
        vs = an.vars
        work: list[tuple[dict[str, tuple[int, int]], set[tuple[int, str]]]] = [(dict(box0), set(open_))]
        unresolved: set[tuple[int, str]] = set()
        while work:
            if time.perf_counter() > an.deadline or an.boxes_done >= MAX_BOXES:
                self.complete = False
                for _, evs in work:
                    unresolved |= evs
                break
            box, evs = work.pop()
            evs = {e for e in evs if self.status.get(e) != "confirmed"}
            if not evs:
                continue
            an.boxes_done += 1
            poss, _ = self.interval_possible(box, evs)
            if not poss:
                continue
            if self.count(box) <= LEAF_EXHAUSTIVE:
                vsx = vs
                grids = np.meshgrid(*[np.arange(box[v][0], box[v][1] + 1, dtype=object) for v in vsx],
                                    indexing="ij")
                coords = {v: g.ravel() for v, g in zip(vsx, grids)}
                vals, masks, _, pts = self.search.evaluate(coords, poss, False)
                self.record_hits(coords, pts, vals, masks)
                for ev in poss:
                    if masks[ev].any() and self.status.get(ev) != "confirmed":
                        unresolved.add(ev)  # observed but not confirmed exactly
                continue
            # cheap probe: corners and centre of this box
            cs = {v: np.array([box[v][0], box[v][1], (box[v][0] + box[v][1]) // 2], dtype=object)
                  for v in vs}
            vals, masks, _, pts = self.search.evaluate(cs, poss, False)
            self.record_hits(cs, pts, vals, masks)
            poss = {e for e in poss if self.status.get(e) != "confirmed"}
            if not poss:
                continue
            v = max(vs, key=lambda w: box[w][1] - box[w][0])
            lo, hi = box[v]
            mid = (lo + hi) // 2
            b1, b2 = dict(box), dict(box)
            b1[v] = (lo, mid)
            b2[v] = (mid + 1, hi)
            work.append((b2, set(poss)))
            work.append((b1, set(poss)))
        for ev in open_:
            if ev in self.status:
                continue
            if ev in unresolved:
                continue
            self.status[ev] = "proved_safe"
            self.method[ev] = "interval bisection + exhaustive leaves"

    # ---- Z3 for integer types -------------------------------------------------

    def _z3(self, open_: set[tuple[int, str]]) -> None:
        try:
            import z3  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError:
            return
        an = self.an
        f = an.fmt
        assert isinstance(f, _IntFmt)
        w = f.bits
        W = 2 * w + 4
        for ev in sorted(open_):
            remaining = an.deadline - time.perf_counter()
            if remaining <= 0.05:
                self.complete = False
                return
            nid, kind = ev
            s = z3.Solver()
            s.set("timeout", int(min(remaining, 5.0) * 1000))
            enc = _Z3Enc(z3, an, w, W)
            for v in an.vars:
                x = enc.var(v)
                lo, hi = an.box0[v]
                s.add(enc.ge(x, lo), enc.le(x, hi))
            cond = enc.event(nid, kind)
            if cond is None:
                continue
            s.add(cond)
            r = s.check()
            if r == z3.unsat:
                self.status[ev] = "proved_safe"
                self.method[ev] = "smt (z3 bit-vectors)"
            elif r == z3.sat:
                mdl = s.model()
                pt = {}
                for v in an.vars:
                    val = mdl.eval(enc.vars[v], model_completion=True).as_long()
                    if f.signed and val >= 1 << (w - 1):
                        val -= 1 << w
                    pt[v] = val
                c2 = {v: np.array([pt[v]], dtype=object) for v in an.vars}
                vals, masks, _, pts = self.search.evaluate(c2, {ev}, False)
                self.record_hits(c2, pts, vals, masks)


class _Z3Enc:
    def __init__(self, z3: Any, an: _Analyzer, w: int, W: int) -> None:
        self.z3, self.an, self.w, self.W = z3, an, w, W
        f = an.fmt
        assert isinstance(f, _IntFmt)
        self.signed = f.signed
        self.vars: dict[str, Any] = {}
        self.cache: dict[int, Any] = {}

    def var(self, name: str) -> Any:
        if name not in self.vars:
            self.vars[name] = self.z3.BitVec(name, self.w)
        return self.vars[name]

    def ext(self, x: Any) -> Any:
        z3 = self.z3
        return (z3.SignExt if self.signed else z3.ZeroExt)(self.W - self.w, x)

    def ge(self, x: Any, c: int) -> Any:
        return x >= c if self.signed else self.z3.UGE(x, c % (1 << self.w))

    def le(self, x: Any, c: int) -> Any:
        return x <= c if self.signed else self.z3.ULE(x, c % (1 << self.w))

    def exact(self, nid: int) -> "tuple[Any, Any] | None":
        """(exact W-bit value of the op on the actual operands, actual w-bit result)."""
        z3 = self.z3
        n = self.an.nodes[nid]
        if n.op == "var":
            v = self.var(n.name)
            return self.ext(v), v
        if n.op == "const":
            c = z3.BitVecVal(int(n.value) % (1 << self.w), self.w)  # type: ignore[arg-type]
            return self.ext(c), c
        args = []
        for a in n.args:
            r = self.node(a)
            if r is None:
                return None
            args.append(self.ext(r))
        a = args[0]
        b = args[1] if len(args) > 1 else None
        zero = z3.BitVecVal(0, self.W)
        if n.op == "add":
            e = a + b
        elif n.op == "sub":
            e = a - b
        elif n.op == "mul":
            e = a * b
        elif n.op == "neg":
            e = -a
        elif n.op == "abs":
            e = z3.If(a < 0, -a, a) if self.signed else a
        elif n.op == "max":
            e = z3.If(a >= b, a, b) if self.signed else z3.If(z3.UGE(a, b), a, b)
        elif n.op == "min":
            e = z3.If(a <= b, a, b) if self.signed else z3.If(z3.ULE(a, b), a, b)
        elif n.op == "pow":
            k = int(n.value)  # type: ignore[arg-type]
            if k > 3:
                return None
            e = z3.BitVecVal(1, self.W)
            for _ in range(k):
                e = e * a
        elif n.op in ("floordiv", "mod"):
            if self.signed:
                q = a / b  # truncating
                r = z3.SRem(a, b)
                assert b is not None
                adj = z3.And(r != 0, (a < 0) != (b < 0))
                fq = z3.If(adj, q - 1, q)
                fr = z3.If(adj, r + b, r)
            else:
                fq, fr = z3.UDiv(a, b), z3.URem(a, b)
            e = z3.If(b == zero, zero, fq if n.op == "floordiv" else fr)
        else:
            return None
        return e, z3.Extract(self.w - 1, 0, e)

    def node(self, nid: int) -> Any:
        if nid not in self.cache:
            r = self.exact(nid)
            self.cache[nid] = None if r is None else r[1]
        return self.cache[nid]

    def event(self, nid: int, kind: str) -> Any:
        z3 = self.z3
        r = self.exact(nid)
        if r is None:
            return None
        e, _ = r
        n = self.an.nodes[nid]
        f = self.an.fmt
        assert isinstance(f, _IntFmt)
        if kind == "division_by_zero_possible":
            b = self.node(n.args[1])
            return None if b is None else b == 0
        if n.op in ("floordiv", "mod"):
            b = self.node(n.args[1])
            if b is None:
                return None
            guard = b != 0
        else:
            guard = z3.BoolVal(True)
        if self.signed:
            return z3.And(guard, z3.Or(e < f.min, e > f.max))
        return z3.And(guard, z3.UGT(e, f.max))


def _show_np(v: Any) -> str:
    x = np.asarray(v)
    if x.dtype.kind in "iu" or x.dtype == object:
        return str(int(x))
    return _exact_decimal(float(x))


def _exact_decimal(x: float) -> str:
    """The EXACT decimal value of a binary float ('0.100000001490116119384765625').

    ``repr`` is not enough for a witness: it round-trips in float64 but names a
    different real number, and at the edge of a range that number can lie outside
    it (``repr`` of -FLT_MAX is below -FLT_MAX).
    """
    if not math.isfinite(x):
        return repr(x)
    if x == int(x) and abs(x) < 1e16:
        return str(int(x)) + ".0"
    return str(Decimal(x))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse_bound(v: object, what: str) -> Fraction:
    if isinstance(v, bool):
        raise DomainError(f"ranges: {what} must be a number, not {v!r}")
    if isinstance(v, (int, Fraction)):
        return Fraction(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise DomainError(f"ranges: {what} must be finite, not {v!r}")
        return Fraction(v)
    if isinstance(v, str):
        folded = _fold(ast.parse(v.strip(), mode="eval").body) if v.strip() else None
        if folded is None:
            try:
                return Fraction(v.strip())
            except (ValueError, ZeroDivisionError):
                raise DomainError(f"ranges: cannot read {what} {v!r}") from None
        if isinstance(folded, float) and not math.isfinite(folded):
            raise DomainError(f"ranges: {what} must be finite, not {v!r}")
        return Fraction(folded)
    raise DomainError(f"ranges: cannot read {what} {v!r}")


def _prepare(expr: object, ranges: "Mapping[str, Sequence[object]] | None", dtype: object
             ) -> tuple[str, list[_Node], int, "_FloatFmt | _IntFmt", dict[str, str],
                        dict[str, tuple[int, int]], dict[str, tuple[str, ...]], list[str]]:
    fmt = _fmt(dtype)
    is_float = isinstance(fmt, _FloatFmt)
    src = _source_of(expr)
    ranges = dict(ranges or {})
    int_vars: set[str] = set()
    for name, spec in ranges.items():
        if not isinstance(spec, (tuple, list)) or len(spec) not in (2, 3):
            raise DomainError(f"ranges: range for {name!r} must be (lo, hi) or (lo, hi, 'int')")
        if len(spec) == 3:
            if spec[2] != "int":
                raise DomainError(f"ranges: third item for {name!r} must be 'int', not {spec[2]!r}")
            int_vars.add(name)
    b = _Builder(int_vars if is_float else set(), is_float)
    root = _parse(src, b, not is_float)
    names = sorted({n.name for n in b.nodes if n.op == "var"})
    extra = set(ranges) - set(names)
    if extra:
        raise DomainError(f"ranges: {sorted(extra)} not in the expression {src!r}")
    var_kind: dict[str, str] = {}
    box: dict[str, tuple[int, int]] = {}
    shown: dict[str, tuple[str, ...]] = {}
    notes: list[str] = []
    for name in names:
        given = ranges.get(name)
        if given is None:
            if isinstance(fmt, _FloatFmt):
                lo_f, hi_f = -fmt.max, fmt.max
            else:
                lo_f, hi_f = Fraction(fmt.min), Fraction(fmt.max)
            notes.append(f"{name}: no range given, every finite value of {fmt.name} assumed")
        else:
            lo_f, hi_f = _parse_bound(given[0], f"lower bound of {name}"), \
                _parse_bound(given[1], f"upper bound of {name}")
        if lo_f > hi_f:
            raise DomainError(f"ranges: empty range for {name!r}: {lo_f} > {hi_f}")
        kind3 = name in int_vars or not is_float
        if kind3:
            if lo_f.denominator != 1 or hi_f.denominator != 1:
                raise DomainError(f"ranges: integer input {name!r} needs integer bounds")
            lo_i, hi_i = int(lo_f), int(hi_f)
            if isinstance(fmt, _IntFmt):
                if lo_i < fmt.min or hi_i > fmt.max:
                    raise DomainError(
                        f"ranges: {name} in [{lo_i}, {hi_i}] does not fit {fmt.name} "
                        f"[{fmt.min}, {fmt.max}]; a value outside the type cannot be an input")
            elif lo_i < -(1 << 63) or hi_i > (1 << 63) - 1:
                raise DomainError(f"ranges: integer input {name!r} must fit int64")
            var_kind[name] = "int"
            box[name] = (lo_i, hi_i)
        else:
            assert isinstance(fmt, _FloatFmt)
            if lo_f < -fmt.max or hi_f > fmt.max:
                raise DomainError(
                    f"ranges: {name} in [{float(lo_f):g}, {float(hi_f):g}] exceeds the finite "
                    f"range of {fmt.name} (+-{float(fmt.max):g}); such an input cannot exist")
            lo_r, hi_r = _round(lo_f, fmt, "u"), _round(hi_f, fmt, "d")
            if lo_r > hi_r:
                raise DomainError(f"ranges: no {fmt.name} value lies in the range of {name!r}")
            var_kind[name] = "float"
            box[name] = (_ord_of(lo_r, fmt), _ord_of(hi_r, fmt))  # type: ignore[arg-type]
        shown[name] = (str(lo_f), str(hi_f)) + (("int",) if name in int_vars else ())
    for n in b.nodes:
        if n.op not in ("var", "const") and all(b.nodes[a].op == "const" for a in n.args):
            raise RangeUnsupportedError(
                f"ranges: {n.text!r} has no variable in it: NumPy evaluates it to a float64/int64 "
                f"NumPy scalar, which is not a weak literal and changes the type of what it meets; "
                f"write its value as a literal")
        if n.op != "const":
            continue
        assert n.value is not None
        if not is_float:
            assert isinstance(fmt, _IntFmt)
            if not isinstance(n.value, int):
                raise DomainError(f"ranges: float literal {n.text} in integer dtype {fmt.name}")
        promoted = _literal_promotion(n.value, fmt)
        if promoted == "OverflowError":
            raise DomainError(
                f"ranges: literal {n.text} = {n.value} does not fit {fmt.name}; NumPy "
                f"{np.__version__} raises OverflowError before any arithmetic happens")
        if promoted is not None:
            raise RangeUnsupportedError(
                f"ranges: the literal {n.text} turns {fmt.name} arrays into {promoted} under NumPy "
                f"{np.__version__}{_PROMOTION_WHY.get(fmt.name if fmt.name == 'bfloat16' else _NP_MAJOR, '')}"
                f"; the expression is then not computed in one type, and a report in {fmt.name} "
                f"would describe a program that does not run. Analyse it as {promoted}, or keep the "
                f"constant in {fmt.name} explicitly")
    return src, b.nodes, root, fmt, var_kind, box, shown, notes


_NP_MAJOR = int(np.__version__.split(".")[0])
_PROMOTION_WHY = {
    1: " (value-based casting: a Python scalar that does not fit the type upcasts the array)",
    "bfloat16": " (ml_dtypes promotes bfloat16 with a Python float to float32)",
}


def _literal_promotion(value: "int | float", fmt: "_FloatFmt | _IntFmt") -> "str | None":
    """What a Python literal does to a 1-element array of the type, asked of NumPy itself.

    None when the result keeps the type (NumPy 2 / NEP 50 weak scalars, or a NumPy 1
    literal small enough for value-based casting to leave it alone); otherwise the
    type it turns into, or the name of the exception NumPy raises. Measured, not
    assumed: NumPy 1.26 turns ``float32 * 1e39`` into float64 and ``float16 * 300``
    into float32, and ml_dtypes turns ``bfloat16 * 2.5`` into float32 (NumPy 2) or
    refuses it (NumPy 1). Without ml_dtypes the bfloat16 emulation follows the
    ml_dtypes rule on NumPy 2 (a float literal promotes, an int literal does not), so
    the verdict does not depend on what happens to be installed.
    """
    if fmt.name == "bfloat16" and _BF16_DTYPE is None:
        return "float32" if isinstance(value, float) else None
    dt = _Machine(fmt).dtype
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            got = (np.ones(1, dtype=dt) * value).dtype
        except OverflowError:
            return "OverflowError"
        except Exception as exc:  # e.g. DTypePromotionError from ml_dtypes on NumPy 1
            return type(exc).__name__
    return None if got == dt else str(got.name)


#: Leaves used to tighten the REPORTED node ranges against the dependency problem.
RANGE_LEAVES = 256


def _split_point(an: _Analyzer, v: str, lo: int, hi: int) -> int:
    """Midpoint in value space inside one sign and a factor of 4; otherwise in coordinates."""
    if an.var_kind[v] == "float":
        f = an.fmt
        assert isinstance(f, _FloatFmt)
        a, b = _val_of(lo, f), _val_of(hi, f)
        if (a > 0 and b <= 4 * a) or (b < 0 and a >= 4 * b):
            m = _round((a + b) / 2, f, "d")
            return min(max(_ord_of(m, f), lo), hi - 1)  # type: ignore[arg-type]
    return (lo + hi) // 2


def _refined_ranges(an: _Analyzer, deadline: float) -> list[_AV]:
    """Union of node intervals over a bisection of the input box: sound, and much
    tighter than one interval pass where a variable occurs more than once."""
    def width(av: _AV) -> float:
        if isinstance(av.lo, float) or isinstance(av.hi, float):
            return math.inf
        d = av.hi - av.lo
        return math.inf if d >= _F64_LIMIT else float(d)

    first = an.run_box(an.box0)
    leaves: list[tuple[float, int, dict[str, tuple[int, int]], list[_AV]]] = [
        (-width(first[an.root]), 0, dict(an.box0), first)]
    tick = 1
    while len(leaves) < RANGE_LEAVES and time.perf_counter() < deadline:
        leaves.sort(key=lambda t: (t[0], t[1]))
        w, _, box, _ = leaves[0]
        cands = [v for v in an.vars if box[v][1] > box[v][0]]
        if w == 0 or not cands:
            break
        leaves.pop(0)
        v = max(cands, key=lambda u: box[u][1] - box[u][0])
        lo, hi = box[v]
        mid = _split_point(an, v, lo, hi)
        for part in ((lo, mid), (mid + 1, hi)):
            b = dict(box)
            b[v] = part
            avs = an.run_box(b)
            leaves.append((-width(avs[an.root]), tick, b, avs))
            tick += 1
    out = []
    for i in range(len(an.nodes)):
        lo = min(t[3][i].lo for t in leaves)
        hi = max(t[3][i].hi for t in leaves)
        out.append(_AV(lo, hi, any(t[3][i].nan for t in leaves), i))
    return out


def _build_report(src: str, an: _Analyzer, run: _Run, shown: dict[str, tuple[str, ...]],
                  notes: list[str], advice: tuple[Advice, ...], t0: float) -> RangeReport:
    avs = _refined_ranges(an, time.perf_counter() + 0.5)
    reachable = _reachable(an.nodes, an.root)
    nodes = tuple(
        NodeRange(n.id, n.text, n.op, _s(avs[n.id].lo) or "", _s(avs[n.id].hi) or "",
                  avs[n.id].nan, n.int_tracked)
        for n in an.nodes if n.id in reachable)
    events = []
    for n in an.nodes:
        if n.id not in reachable:
            continue
        for k in an.applicable(n):
            ev = (n.id, k)
            st = run.status.get(ev, "possible")
            w = run.wit.get(ev) if st == "confirmed" else None
            events.append(Event(
                kind=k, status=st, node=n.id, node_text=n.text, op=n.op,
                operand_src=tuple(an.nodes[a].text for a in n.args),
                method=run.method.get(ev, "undecided within the budget"),
                witness=None if w is None else w["witness"],
                observed=None if w is None else w["observed"],
                operands=() if w is None else w["operands"],
                exact=None if w is None else w["exact"],
                detail=_detail(k, st, n, w),
            ))
    order = {k: i for i, k in enumerate(EVENT_KINDS)}
    events.sort(key=lambda e: ({"confirmed": 0, "possible": 1, "proved_safe": 2}[e.status],
                               order[e.kind], e.node))
    assumptions = list(notes)
    assumptions.append(f"NumPy {np.__version__} array semantics (ndim >= 1)")
    if an.is_float:
        assumptions.append(
            f"proved_safe by intervals assumes correctly rounded + - * / sqrt and <= "
            f"{TRANSCENDENTAL_ULPS} ulp for other functions, no flush-to-zero")
    return RangeReport(src, an.fmt.name, shown, nodes, tuple(events), advice, run.complete,
                       an.m.method, time.perf_counter() - t0, tuple(assumptions))


def _reachable(nodes: list[_Node], root: int) -> set[int]:
    seen: set[int] = set()
    stack = [root]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack.extend(nodes[i].args)
    return seen


def _detail(kind: str, status: str, n: _Node, w: "dict[str, Any] | None") -> str:
    if status == "proved_safe":
        return f"{kind.replace('_', ' ')} cannot happen at {n.text}"
    if status == "possible":
        return f"{kind.replace('_', ' ')} at {n.text} neither excluded nor witnessed"
    assert w is not None
    if n.op == "const":
        return f"literal {n.text} is not representable"
    args = ", ".join(f"{k}={v}" for k, v in w["witness"].items())
    return f"{n.text} at {args}: got {w['observed']}, exact {w['exact']}"


def analyze(expr: object, ranges: "Mapping[str, Sequence[object]] | None" = None,
            dtype: object = "float32", *, budget: "float | None" = None,
            advice: bool = True, use_z3: bool = True) -> RangeReport:
    """Range of every intermediate value and every overflow-type event, with witnesses.

    ``expr`` — Python/NumPy expression text (``"(lo + hi) // 2"``, ``"np.exp(x)"``),
    a SymPy/pycodemath expression (printed by SymPy, so in ITS canonical order), or a
    function whose body is one ``return``. ``ranges`` — ``{"x": (lo, hi)}``; add a
    third item ``"int"`` for an integer-valued input of a float type (timestamps,
    counters). A variable without a range takes every finite value of the type.
    ``dtype`` — one of ``DTYPES``. ``budget`` — seconds (default 10); what is still
    open when it runs out is reported ``possible`` and ``complete`` is False.

    >>> r = analyze("t + 1", {"t": (1_600_000_000_000, 1_800_000_000_000, "int")}, "float32")
    >>> r.class_status("integer_precision_loss")
    'confirmed'
    """
    t0 = time.perf_counter()
    total = DEFAULT_BUDGET if budget is None else budget
    if not isinstance(total, (int, float)) or isinstance(total, bool) or not total > 0:
        raise DomainError(f"analyze: budget must be a positive number, not {budget!r}")
    src, nodes, root, fmt, var_kind, box, shown, notes = _prepare(expr, ranges, dtype)
    deadline = t0 + total * (0.6 if advice else 1.0)
    an = _Analyzer(nodes, root, fmt, var_kind, box, deadline)
    run = _Run(an, use_z3=use_z3)
    run.run()
    adv: tuple[Advice, ...] = ()
    if advice:
        adv = _advise(src, an, run, ranges, t0 + total)
    return _build_report(src, an, run, shown, notes, adv, t0)


# ---------------------------------------------------------------------------
# Advice: rewrites that are proved equivalent AND re-analysed
# ---------------------------------------------------------------------------

def _advise(src: str, an: _Analyzer, run: _Run, ranges: Any, deadline: float) -> tuple[Advice, ...]:
    bad = {k for (nid, k), st in run.status.items() if st in ("confirmed", "possible")}
    if not bad:
        return ()
    out: list[Advice] = []
    for cand, dt, method, detail in _rewrites(src, an, bad):
        if time.perf_counter() > deadline:
            break
        remaining = deadline - time.perf_counter()
        try:
            rep = analyze(cand, ranges, dt, budget=max(remaining, 0.2), advice=False)
        except Exception:  # a rewrite the analysis cannot read is simply not offered
            continue
        removes = tuple(k for k in EVENT_KINDS if k in bad and rep.class_status(k) == "proved_safe")
        worse = any(rep.class_status(k) != "proved_safe" for k in EVENT_KINDS
                    if k not in bad and k in {e.kind for e in rep.events})
        if removes and not worse:
            out.append(Advice(cand, dt, removes, "verified", method, detail))
    return tuple(out)


def _rewrites(src: str, an: _Analyzer, bad: set[str]) -> Iterable[tuple[str, str, str, str]]:
    tree = ast.parse(src, mode="eval")
    dt = an.fmt.name
    # 1. midpoint: (a + b) // 2 -> a + (b - a) // 2   (integers; floor division)
    if not an.is_float and "overflow" in bad:
        class Mid(ast.NodeTransformer):
            hit = False

            def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                self.generic_visit(node)
                if isinstance(node.op, ast.FloorDiv) and isinstance(node.left, ast.BinOp) and \
                        isinstance(node.left.op, ast.Add) and isinstance(node.right, ast.Constant) \
                        and node.right.value == 2:
                    a, b = node.left.left, node.left.right
                    self.hit = True
                    return ast.BinOp(a, ast.Add(), ast.BinOp(
                        ast.BinOp(b, ast.Sub(), a), ast.FloorDiv(), ast.Constant(2)))
                return node
        import copy
        t = Mid()
        new = t.visit(copy.deepcopy(tree))
        if t.hit and _midpoint_identity():
            yield (ast.unparse(new), dt, "symbolic (integer parity cases)",
                   "floor((a+b)/2) == a + floor((b-a)/2) for all integers a, b "
                   "(Bloch 2006); never forms a + b")
    # 2. hypot: sqrt(a*a + b*b) -> hypot(a, b)
    if an.is_float and "overflow" in bad:
        import copy

        class Hyp(ast.NodeTransformer):
            hit = False

            def visit_Call(self, node: ast.Call) -> ast.AST:
                self.generic_visit(node)
                name = node.func.attr if isinstance(node.func, ast.Attribute) else \
                    getattr(node.func, "id", "")
                if name == "sqrt" and len(node.args) == 1:
                    s = node.args[0]
                    if isinstance(s, ast.BinOp) and isinstance(s.op, ast.Add):
                        a, b = _square_base(s.left), _square_base(s.right)
                        if a is not None and b is not None:
                            self.hit = True
                            return ast.Call(ast.Name("hypot", ast.Load()), [a, b], [])
                return node
        t2 = Hyp()
        new = t2.visit(copy.deepcopy(tree))
        if t2.hit:
            yield (ast.unparse(new), dt, "definition (hypot(a, b) = sqrt(a**2 + b**2))",
                   "hypot scales internally, so a*a and b*b are never formed")
    # 3. max-shift for exp: softmax / log-sum-exp
    if an.is_float and bad & {"overflow", "nan_possible"}:
        yield from _shift_rewrites(src, tree, dt)
    # 4. a wider type
    for wider in _WIDER.get(dt, ()):
        yield (src, wider, "same expression",
               f"the same operations in {wider}; the value is the same exact-arithmetic "
               f"expression, only the type's range/precision changes")


def _square_base(n: ast.expr) -> "ast.expr | None":
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and \
            ast.dump(n.left) == ast.dump(n.right):
        return n.left
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) and \
            isinstance(n.right, ast.Constant) and n.right.value == 2:
        return n.left
    return None


_MID_CACHE: dict[str, bool] = {}


def _midpoint_identity() -> bool:
    """Prove floor((a+b)/2) == a + floor((b-a)/2) over the integers (two parity cases)."""
    if "v" not in _MID_CACHE:
        import sympy as sp

        a, k = sp.symbols("a k", integer=True)
        ok = True
        for d in (2 * k, 2 * k + 1):
            b = a + d
            ok = ok and sp.simplify(sp.floor((a + b) / 2) - a - sp.floor((b - a) / 2)) == 0
        _MID_CACHE["v"] = bool(ok)
    return _MID_CACHE["v"]


def _shift_rewrites(src: str, tree: ast.Expression, dt: str) -> Iterable[tuple[str, str, str, str]]:
    import copy

    from ..verify.equal import check_equal
    from ..verify.verdict import VerdictStatus

    args: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and len(n.args) == 1:
            name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            if name == "exp":
                t = ast.unparse(n.args[0])
                if t not in args:
                    args.append(t)
    if len(args) < 2 or not all(_is_name(a) for a in args):
        return
    m = f"max({', '.join(args)})"

    class Sh(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name == "exp" and len(node.args) == 1:
                inner = ast.parse(f"{ast.unparse(node.args[0])} - {m}", mode="eval").body
                return ast.Call(ast.Name("exp", ast.Load()), [inner], [])
            return node
    shifted = ast.unparse(Sh().visit(copy.deepcopy(tree)))
    cands = [shifted]
    if isinstance(tree.body, ast.Call) and getattr(tree.body.func, "id",
                                                   getattr(tree.body.func, "attr", "")) == "log":
        cands.append(f"{m} + {shifted}")
    for c in cands:
        try:
            v = check_equal(_to_sympy_text(src), _to_sympy_text(c), budget=5)
        except Exception:
            continue
        if v.status is VerdictStatus.VERIFIED:
            yield (c, dt, f"check_equal ({v.method})",
                   f"exponents shifted by {m}: every exp argument <= 0")
            return


def _is_name(t: str) -> bool:
    return t.isidentifier()


def _to_sympy_text(src: str) -> str:
    """Our syntax -> pycodemath parser syntax (Max/Min/Abs/floor)."""
    t = ast.parse(src, mode="eval")

    class S(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            ren = {"max": "Max", "maximum": "Max", "min": "Min", "minimum": "Min",
                   "abs": "Abs", "arctan": "atan"}.get(name, name)
            if ren == "hypot":
                a, b = node.args
                return ast.Call(ast.Name("sqrt", ast.Load()), [ast.BinOp(
                    ast.BinOp(a, ast.Pow(), ast.Constant(2)), ast.Add(),
                    ast.BinOp(b, ast.Pow(), ast.Constant(2)))], [])
            node.func = ast.Name(ren, ast.Load())
            return node
    return ast.unparse(S().visit(t))


# ---------------------------------------------------------------------------
# max_safe_input
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafeInput:
    """Largest value of ``var`` (from ``start`` up) for which nothing breaks."""

    var: str
    dtype: str
    start: str
    safe_up_to: "str | None"
    first_failure: "str | None"
    exact: bool
    status: str  # exact | bounded | no_failure | none_safe
    failure: "Event | None"
    method: str
    detail: str

    @property
    def kind(self) -> str:
        return "max_safe_input"

    @property
    def value(self) -> "float | None":
        return None if self.safe_up_to is None else float(Fraction(self.safe_up_to))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "var": self.var, "dtype": self.dtype, "start": self.start,
                "safe_up_to": self.safe_up_to, "first_failure": self.first_failure,
                "exact": self.exact, "status": self.status,
                "failure": None if self.failure is None else self.failure.to_dict(),
                "method": self.method, "detail": self.detail}


def max_safe_input(expr: object, var: str, dtype: object = "float32",
                   other_ranges: "Mapping[str, Sequence[object]] | None" = None, *,
                   start: object = 0, events: Sequence[str] = FATAL_KINDS,
                   budget: "float | None" = None) -> SafeInput:
    """Largest ``var`` >= ``start`` such that no event in ``events`` can happen for any
    value in ``[start, var]`` (other variables over ``other_ranges``).

    ``exact`` is True when the next representable value is CONFIRMED to fail (with a
    witness), so the boundary is pinned to one step of the type.

    >>> r = max_safe_input("exp(x)", "x", "float32")
    >>> r.exact, round(r.value, 3)
    (True, 88.723)
    """
    t0 = time.perf_counter()
    total = DEFAULT_BUDGET if budget is None else budget
    if not isinstance(total, (int, float)) or isinstance(total, bool) or not total > 0:
        raise DomainError(f"max_safe_input: budget must be a positive number, not {budget!r}")
    bad = [k for k in events if k not in EVENT_KINDS]
    if bad:
        raise DomainError(f"max_safe_input: unknown event kinds {bad}")
    ranges = dict(other_ranges or {})
    if var in ranges:
        raise DomainError(f"max_safe_input: {var!r} must not be in other_ranges")
    fmt = _fmt(dtype)
    top: object = float(fmt.max) if isinstance(fmt, _FloatFmt) else fmt.max
    if isinstance(fmt, _FloatFmt):
        top_s = str(fmt.max)
    else:
        top_s = str(fmt.max)
    ranges[var] = (start, top_s)
    del top
    src, nodes, root, fmt, var_kind, box, shown, notes = _prepare(expr, ranges, dtype)
    an = _Analyzer(nodes, root, fmt, var_kind, box, t0 + total)
    c0, cmax = box[var]
    kinds = set(events)

    def check(lo: int, hi: int) -> tuple[str, "tuple[int, Event] | None"]:
        b = dict(box)
        b[var] = (lo, hi)
        a2 = _Analyzer(nodes, root, fmt, var_kind, b, t0 + total)
        run = _Run(a2)
        run.run()
        fail: "tuple[int, Event] | None" = None
        undecided = False
        for (nid, k), st in run.status.items():
            if k not in kinds or nid not in _reachable(nodes, root):
                continue
            if st == "confirmed":
                w = run.wit[(nid, k)]
                if not w["witness"]:
                    continue
                c = _coord_from_text(an, var, w["witness"][var])
                n = nodes[nid]
                ev = Event(k, "confirmed", nid, n.text, n.op,
                           tuple(nodes[a].text for a in n.args), run.method[(nid, k)],
                           w["witness"], w["observed"], w["operands"], w["exact"],
                           _detail(k, "confirmed", n, w))
                if fail is None or c < fail[0]:
                    fail = (c, ev)
        for n in nodes:
            if n.id not in _reachable(nodes, root):
                continue
            for k in a2.applicable(n):
                if k in kinds and run.status.get((n.id, k)) not in ("confirmed", "proved_safe"):
                    undecided = True
        if fail is not None:
            return "fail", fail
        return ("undecided" if undecided else "safe"), None

    st, fail = check(c0, cmax)
    show = lambda c: str(an.coord_value(var, c))  # noqa: E731
    if st == "safe":
        return SafeInput(var, fmt.name, str(_parse_bound(start, "start")), show(cmax), None, True,
                         "no_failure", None, "proof over the whole range",
                         f"nothing in {sorted(kinds)} can happen for {var} up to the type maximum")
    if fail is None:
        return SafeInput(var, fmt.name, str(_parse_bound(start, "start")), None, None, False,
                         "bounded", None, "undecided",
                         "no failure witnessed and no proof within the budget")
    lo_ok, hi_bad = c0 - 1, fail[0]
    bad_ev = fail[1]
    while hi_bad - lo_ok > 1 and time.perf_counter() < t0 + total:
        mid = (lo_ok + hi_bad) // 2
        st, f2 = check(lo_ok + 1, mid)
        if st == "safe":
            lo_ok = mid
        elif st == "fail" and f2 is not None:
            hi_bad, bad_ev = f2
        else:
            break
    exact = hi_bad - lo_ok == 1
    if lo_ok < c0:
        return SafeInput(var, fmt.name, str(_parse_bound(start, "start")), None, show(hi_bad), exact,
                         "none_safe", bad_ev, bad_ev.method, "fails already at the start")
    return SafeInput(var, fmt.name, str(_parse_bound(start, "start")), show(lo_ok), show(hi_bad),
                     exact, "exact" if exact else "bounded", bad_ev,
                     "interval/exhaustive proof below, executed witness above",
                     f"{var} <= {float(an.coord_value(var, lo_ok))!r} is proved safe; "
                     f"{var} = {float(an.coord_value(var, hi_bad))!r} fails ({bad_ev.kind} at "
                     f"{bad_ev.node_text})")


def _coord_from_text(an: _Analyzer, var: str, text: str) -> int:
    if an.var_kind[var] == "float":
        assert isinstance(an.fmt, _FloatFmt)
        return _ord_of(Fraction(text), an.fmt)
    return int(text)
