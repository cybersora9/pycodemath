"""Precision doctor — how many bits a floating-point formula loses, where, and a proven fix (C2).

``diagnose(expr, ranges)`` measures the accuracy of evaluating ``expr`` in float64
(or float32) against the exact real value, over the whole input range, and names
the sub-expressions that lose the bits. ``improve(expr, ranges)`` searches for a
rewrite that is (a) PROVED equal to the original — symbolically, with the domain
conditions it needs written out — and (b) MEASURED more accurate on fresh sample
points, and returns ready-to-paste Python / NumPy / C. A suggestion without both
proofs is never returned; when nothing helps, the result says so.

    >>> from pycodemath.code.stability import diagnose
    >>> r = diagnose("sqrt(x + 1) - sqrt(x)", {"x": (1e10, 1e15)}, samples=64)
    >>> r.mean_bits > 20
    True
    >>> r.culprits[0].expression
    'sqrt(x + 1) - sqrt(x)'

--- THE ERROR MEASURE (as in Herbie, PLDI 2015) ----------------------------------

Bits of error at one input = ``log2(1 + U)`` where ``U`` is the number of floats
between the computed result and the correctly rounded exact result (ULP distance,
counted on the ordinal line of the format, so it is meaningful across binades and
for subnormals). 0 bits = correctly rounded; 64 bits (32 for float32) = a NaN where
a real number was due. Averages are over the sample; the maximum is reported with
its witness input. The definition, the NaN penalty and the treatment of an exact
value that overflows (it rounds to ``inf`` and is compared on the same ordinal line)
were read from Herbie's source (``src/core/points.rkt``, ``src/syntax/float.rkt``,
MIT) so the numbers are comparable with Herbie's published reports.

--- SAMPLING: UNIFORM OVER FLOAT BIT PATTERNS, NOT OVER VALUES --------------------

Each input is drawn uniformly from the ORDINALS of the floats in the user's range,
i.e. every representable float in the range is equally likely. Floats are spread
logarithmically: on ``[1e-10, 1e15]`` a uniform-by-value draw puts 99.99% of the
points in ``[1e11, 1e15]`` and never looks at the small magnitudes where
``exp(x) - 1`` or ``1 - cos(x)`` fall apart. Uniform-over-bits gives every binade
the same weight, which is the scale on which floating-point error lives (relative
error, ULPs). Special points (the range ends, 0, +-1, powers of two, the floats
next to the zeros of a denominator) are evaluated as well and reported separately,
so they can raise the maximum and show up in the map but do not bias the mean.

--- GROUND TRUTH: INTERVALS, PRECISION DOUBLED UNTIL THE ROUNDING IS DECIDED ------

The exact value is computed with ``mpmath.iv`` interval arithmetic over the written
expression at 64 bits, then 128, 256, ... up to 8192. It is accepted when BOTH ends of
the enclosing interval round to the same float of the target format — then that
float is the correctly rounded exact value, whatever the true real is inside the
interval. This is the criterion of Herbie's ground-truth engine (Rival). Functions
``mpmath.iv`` lacks (``atan``, ``sinh``, ``cbrt`` …) are monotone on the narrow
intervals that arise from point inputs and are enclosed by evaluating the ends with
40 guard bits and widening by ``2^-prec`` relative. A point is SKIPPED, never
guessed, when the value is not real there (``sqrt`` of a negative, ``log(0)``,
division by an exact zero), when an intermediate leaves ``2^(2^20)`` in magnitude, or
when 8192 bits did not decide the rounding; the report counts skipped points.

Inputs are exact floats (as in Herbie): the error measured is the error the formula
ADDS to exact inputs, not the propagation of errors already in the inputs.

--- WHAT IS MEASURED IS WHAT IS EMITTED ------------------------------------------

The formula is held in a small expression tree that keeps the WRITTEN order of
operations (SymPy would reorder ``(x + 1) - x`` into ``x - x + 1``, which is a
different floating-point program). The same tree is evaluated in NumPy for the
measurement and printed for ``to_numpy()`` / ``to_python()`` / ``to_c()``, so the
NumPy code evaluates bit-for-bit what was measured (``tests/test_code_stability``
pins it). ``to_python`` uses ``math``, whose libm may differ from NumPy's by an ulp
per call; ``to_c`` relies on ``<math.h>``.
"""

from __future__ import annotations

import ast
import functools
import itertools
import math
import re
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, Sequence

import mpmath
import numpy as np
import sympy as sp

from ..core import budget as _budget
from ..core.budget import time_budget
from ..core.errors import DomainError, ParseError, TimeBudgetError, UnsupportedFormError

__all__ = [
    "diagnose",
    "improve",
    "StabilityReport",
    "Improvement",
    "Candidate",
    "Culprit",
    "RegionError",
    "PointError",
    "Proof",
    "StabilityDomainError",
    "parse_formula",
    "from_fpcore",
    "Node",
]


class StabilityDomainError(DomainError):
    """No sample point in the given ranges has a real, decidable exact value.

    Raised instead of a report built on nothing: a range where ``sqrt(x)`` is never
    real, or every intermediate overflows ``2^(2^20)``, has no accuracy to measure.
    """


# --------------------------------------------------------------------------
# Floating-point formats
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Format:
    name: str
    dtype: Any
    int_dtype: Any
    bits: int  # total bits = the NaN penalty
    prec: int  # significand bits incl. the hidden one
    emin: int  # exponent of the smallest subnormal ULP
    emax: int  # 2**emax is the first value that overflows
    unit_roundoff: float

    @property
    def max(self) -> float:
        return float(np.finfo(self.dtype).max)


_F64 = _Format("float64", np.float64, np.int64, 64, 53, -1074, 1024, 2.0**-53)
_F32 = _Format("float32", np.float32, np.int32, 32, 24, -149, 128, 2.0**-24)
_FORMATS = {"float64": _F64, "float32": _F32}


def _format(dtype: object) -> _Format:
    key = getattr(dtype, "__name__", None) or str(dtype)
    key = {"double": "float64", "single": "float32", "float": "float64"}.get(key, key)
    if key not in _FORMATS:
        raise DomainError(f"dtype must be 'float64' or 'float32', not {dtype!r}")
    return _FORMATS[key]


def _round_rational(num: int, den: int, fmt: _Format) -> float:
    """``num/den`` correctly rounded (to nearest, ties to even) into ``fmt``.

    Exact integer arithmetic: no double rounding through float64 on the way to
    float32, subnormals and overflow handled. Returned as a Python float, which
    holds every float32 value exactly.
    """
    if num == 0:
        return 0.0
    negative = (num < 0) != (den < 0)
    num, den = abs(num), abs(den)
    e = num.bit_length() - den.bit_length()
    if e > fmt.emax + 2:
        return -math.inf if negative else math.inf
    if e < fmt.emin - 2:
        return -0.0 if negative else 0.0
    if (num << max(0, -e)) < (den << max(0, e)):
        e -= 1  # now 2**e <= num/den < 2**(e+1)
    q = max(e - (fmt.prec - 1), fmt.emin)
    n, d = (num, den << q) if q >= 0 else (num << -q, den)
    m, r = divmod(n, d)
    if 2 * r > d or (2 * r == d and m & 1):
        m += 1
    if m.bit_length() + q > fmt.emax:  # m * 2**q >= 2**emax
        return -math.inf if negative else math.inf
    value = math.ldexp(m, q)
    return -value if negative else value


def _round_mpf(parts: tuple, fmt: _Format) -> float:
    """An ``mpf`` (its ``_mpf_`` tuple) correctly rounded into ``fmt``."""
    sign, man, exp, bc = parts
    if not man:
        if exp == -456:  # mpmath's encoding of +-inf
            return -math.inf if sign else math.inf
        if exp == -123 or exp == -789:  # nan / not a finite number
            return math.nan
        return -0.0 if sign else 0.0
    top = exp + bc
    if top > fmt.emax + 2:
        return -math.inf if sign else math.inf
    if top < fmt.emin - 2:
        return -0.0 if sign else 0.0
    num = man << exp if exp >= 0 else man
    den = 1 if exp >= 0 else 1 << -exp
    return _round_rational(-num if sign else num, den, fmt)


def _ordinals(values: np.ndarray, fmt: _Format) -> np.ndarray:
    """The position of each float on the format's ordinal line (int64; +0 == -0)."""
    ints = np.ascontiguousarray(values, dtype=fmt.dtype).view(fmt.int_dtype).astype(np.int64)
    mask = np.int64((1 << (fmt.bits - 1)) - 1)
    return np.where(ints < 0, -(ints & mask), ints)


def _from_ordinals(ords: np.ndarray, fmt: _Format) -> np.ndarray:
    sign_bit = np.int64(1) << np.int64(fmt.bits - 1)
    ints = np.where(ords < 0, (-ords) | sign_bit, ords)
    if fmt.bits == 32:
        return ints.astype(np.uint32).view(np.float32).astype(np.float32)
    return ints.view(np.float64)


def _bits_of_error(out: np.ndarray, exact: np.ndarray, fmt: _Format) -> np.ndarray:
    """``log2(1 + ULP distance)`` per point; a NaN result costs ``fmt.bits``."""
    out = np.asarray(out, dtype=fmt.dtype)
    exact = np.asarray(exact, dtype=fmt.dtype)
    a, b = _ordinals(out, fmt), _ordinals(exact, fmt)
    same_sign = (a >= 0) == (b >= 0)
    with np.errstate(over="ignore"):
        exact_diff = np.abs(np.where(same_sign, a - b, 0)).astype(np.float64)
    far = np.abs(a).astype(np.float64) + np.abs(b).astype(np.float64)
    ulps = np.where(same_sign, exact_diff, far)
    bits = np.log2(1.0 + ulps)
    bits = np.where(np.isnan(out) | np.isnan(exact), float(fmt.bits), bits)
    return np.minimum(bits, float(fmt.bits))


# --------------------------------------------------------------------------
# The expression tree
# --------------------------------------------------------------------------

_UNARY = (
    "neg", "sqrt", "cbrt", "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "log1p", "expm1", "fabs",
)
_BINARY = ("+", "-", "*", "/", "pow", "hypot", "max", "min", "atan2")
_COMPARE = ("<", "<=", ">", ">=")
_ARITY = {**{op: 1 for op in _UNARY}, **{op: 2 for op in _BINARY + _COMPARE}, "fma": 3, "if": 3}

#: Relative cost of each operation, for the accuracy-vs-cost Pareto front. Rough
#: latency classes, as in Herbie's default cost model: arithmetic 1, division and
#: square root a few, transcendental functions tens.
_COST = {
    "var": 0, "const": 0, "neg": 1, "+": 1, "-": 1, "*": 1, "fabs": 1, "max": 1, "min": 1,
    "/": 4, "sqrt": 4, "fma": 2, "<": 1, "<=": 1, ">": 1, ">=": 1, "if": 1,
    "hypot": 10, "cbrt": 15, "pow": 25,
}
_DEFAULT_FN_COST = 20


@dataclass(frozen=True)
class Node:
    """One node of a floating-point formula, in the order it is evaluated.

    ``op`` is ``"var"`` (``name`` set), ``"const"`` (``value`` = the exact SymPy
    number written, e.g. ``Rational(1, 10)`` for ``0.1`` or ``pi``), or an operation
    from ``_ARITY`` applied to ``args``. Frozen and structurally hashable, so equal
    sub-formulas share cached values.
    """

    op: str
    args: tuple["Node", ...] = ()
    name: str = ""
    value: Any = None

    def __str__(self) -> str:
        return _to_text(self)

    @functools.cached_property
    def variables(self) -> tuple[str, ...]:
        found: dict[str, None] = {}
        for n in self.walk():
            if n.op == "var":
                found.setdefault(n.name)
        return tuple(sorted(found))

    def walk(self) -> tuple["Node", ...]:
        """Every node, children before parents (each distinct sub-formula once)."""
        return self._order

    @functools.cached_property
    def _order(self) -> tuple["Node", ...]:
        seen: set[Node] = set()
        out: list[Node] = []

        def visit(n: Node) -> None:
            if n in seen:
                return
            seen.add(n)
            for a in n.args:
                visit(a)
            out.append(n)

        visit(self)
        return tuple(out)

    @property
    def cost(self) -> int:
        return sum(_COST.get(n.op, _DEFAULT_FN_COST) for n in self.walk())

    def replace(self, old: "Node", new: "Node") -> "Node":
        """This formula with every occurrence of ``old`` replaced by ``new``."""
        if self == old:
            return new
        if not self.args:
            return self
        args = tuple(a.replace(old, new) for a in self.args)
        return self if args == self.args else Node(self.op, args, self.name, self.value)


def _var(name: str) -> Node:
    return Node("var", name=name)


def _const(value: Any) -> Node:
    return Node("const", value=sp.sympify(value))


def _op(op: str, *args: Node) -> Node:
    return Node(op, tuple(args))


_ONE = _const(1)
_TWO = _const(2)
_ZERO = _const(0)

# ---- printing ------------------------------------------------------------

_PREC = {"if": 0, "<": 1, "<=": 1, ">": 1, ">=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "neg": 4, "pow": 5}


def _const_text(value: Any) -> str:
    if value == sp.pi:
        return "pi"
    if value == sp.E:
        return "E"
    r = sp.Rational(value)
    if r.q == 1 and abs(r.p) < 10**16:
        return str(r.p)
    if abs(r) < 1e308:
        short = repr(float(r))  # thresholds and literals from code: the shortest spelling
        if "inf" not in short and "nan" not in short and sp.Rational(short) == r:
            return short
    if r.q == 1:
        return str(r.p)
    q, twos, fives = r.q, 0, 0
    while q % 2 == 0:
        q //= 2
        twos += 1
    while q % 5 == 0:
        q //= 5
        fives += 1
    if q != 1:  # not a terminating decimal: never produced by the readers
        return f"({r.p}/{r.q})"
    digits = max(twos, fives)
    scaled = abs(r.p) * (10**digits) // r.q
    text = str(scaled).rjust(digits + 1, "0")
    text = (text[:-digits] + "." + text[-digits:]).rstrip("0").rstrip(".")
    if len(text) > 24:
        mantissa = sp.Float(abs(r), 40)
        text = mpmath.nstr(mpmath.mpf(mantissa), 40, strip_zeros=True)
        # Exactness matters more than looks: fall back to the fraction if the
        # decimal rendering does not read back to the same rational.
        if sp.Rational(text) != abs(r):
            return f"({r.p}/{r.q})"
    return ("-" if r.p < 0 else "") + text


def _to_text(n: Node, parent: int = 0, right: bool = False) -> str:
    """Infix text in the written order; ``parse_formula`` reads it back to ``n``."""
    op = n.op
    if op == "var":
        return n.name
    if op == "const":
        text = _const_text(n.value)
        return f"({text})" if text.startswith("-") and parent > 2 else text
    if op == "if":
        c, a, b = (_to_text(x) for x in n.args)
        text = f"{a} if {c} else {b}"
        return f"({text})" if parent > 0 else text
    if op in _PREC:
        prec = _PREC[op]
        if op == "neg":
            text = "-" + _to_text(n.args[0], prec)
        elif op == "pow":
            text = f"{_to_text(n.args[0], prec + 1)}**{_to_text(n.args[1], prec, right=True)}"
        else:
            sym = {"pow": "**"}.get(op, op)
            text = f"{_to_text(n.args[0], prec)} {sym} {_to_text(n.args[1], prec, right=True)}"
        needs = prec < parent or (prec == parent and right and op != "pow")
        return f"({text})" if needs else text
    return f"{op}({', '.join(_to_text(a) for a in n.args)})"


# ---- reading text ----------------------------------------------------------

_FUNCTIONS = {
    "sqrt": "sqrt", "cbrt": "cbrt", "exp": "exp", "log": "log", "ln": "log",
    "sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos",
    "atan": "atan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "asinh": "asinh",
    "acosh": "acosh", "atanh": "atanh", "log1p": "log1p", "expm1": "expm1",
    "abs": "fabs", "fabs": "fabs", "Abs": "fabs", "hypot": "hypot", "pow": "pow",
    "max": "max", "min": "min", "Max": "max", "Min": "min", "fma": "fma", "atan2": "atan2",
}
#: ``e`` is NOT a constant here: in code it is a variable as often as not, and the
#: reader does not guess (``math.e`` is spelled ``E``).
_CONSTANTS = {"pi": sp.pi, "E": sp.E}
_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "pow"}
_CMPOPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


def parse_formula(text: str) -> Node:
    """Read a formula written in code/math notation, keeping its evaluation order.

    Python syntax read with ``ast`` (nothing is executed): ``+ - * / ** ^``, unary
    minus, calls from a fixed list (``sqrt exp log log1p expm1 sin cos tan atan
    hypot cbrt pow fabs max min fma …``), the constants ``pi`` and ``E``, and
    ``a if x < c else b``. Decimal literals mean the decimal written (``0.1`` is
    1/10, rounded to the format when evaluated, like a literal in code). Names are
    taken whole: ``eps`` is one variable. Anything else is a ``ParseError``.

        >>> str(parse_formula("(x + 1) - x"))
        'x + 1 - x'
        >>> parse_formula("(x + 1) - x") == parse_formula("x + (1 - x)")
        False
    """
    if not isinstance(text, str):
        raise ParseError(f"formula must be text, not {type(text).__name__}")
    if len(text) > 20_000:
        raise ParseError("formula is too long (more than 20000 characters)")
    source = text.replace("^", "**")
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise ParseError(f"cannot understand formula {text!r}: {exc.msg}") from None
    return _from_ast(tree.body, source.strip(), text)


def _from_ast(node: ast.AST, source: str, text: str) -> Node:
    def fail(what: str) -> ParseError:
        return ParseError(f"cannot understand formula {text!r}: {what}")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise fail(f"{node.value!r} is not a number")
        literal = ast.get_source_segment(source, node) or repr(node.value)
        try:
            return _const(sp.Rational(literal.replace("_", "")))
        except (TypeError, ValueError):
            raise fail(f"bad number {literal!r}") from None
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _const(_CONSTANTS[node.id])
        if node.id in _FUNCTIONS:
            raise fail(f"{node.id!r} is a function and needs an argument")
        if node.id.startswith("_"):
            raise fail(f"variable names may not start with '_' ({node.id!r})")
        return _var(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _from_ast(node.operand, source, text)
        if isinstance(node.op, ast.UAdd):
            return inner
        if inner.op == "const" and inner.value.is_Rational:
            return _const(-inner.value)
        return _op("neg", inner)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _op(
            _BINOPS[type(node.op)],
            _from_ast(node.left, source, text),
            _from_ast(node.right, source, text),
        )
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPOPS:
        return _op(
            _CMPOPS[type(node.ops[0])],
            _from_ast(node.left, source, text),
            _from_ast(node.comparators[0], source, text),
        )
    if isinstance(node, ast.IfExp):
        return _op(
            "if",
            _from_ast(node.test, source, text),
            _from_ast(node.body, source, text),
            _from_ast(node.orelse, source, text),
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if name not in _FUNCTIONS:
            raise fail(f"unknown function {name!r}")
        op = _FUNCTIONS[name]
        args = [_from_ast(a, source, text) for a in node.args]
        if op in ("max", "min") and len(args) >= 2:
            acc = args[0]
            for a in args[1:]:
                acc = _op(op, acc, a)
            return acc
        if len(args) != _ARITY[op]:
            raise fail(f"{name} takes {_ARITY[op]} argument(s), got {len(args)}")
        return _op(op, *args)
    raise fail(f"unsupported syntax {ast.dump(node)[:60]}")


# ---- reading FPCore ----------------------------------------------------------

_FPCORE_OPS = {
    "+": "+", "-": "-", "*": "*", "/": "/", "pow": "pow", "sqrt": "sqrt", "cbrt": "cbrt",
    "exp": "exp", "log": "log", "sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin",
    "acos": "acos", "atan": "atan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "asinh": "asinh", "acosh": "acosh", "atanh": "atanh", "log1p": "log1p",
    "expm1": "expm1", "fabs": "fabs", "hypot": "hypot", "fmax": "max", "fmin": "min",
    "fma": "fma", "atan2": "atan2", "<": "<", "<=": "<=", ">": ">", ">=": ">=", "if": "if",
}
_FPCORE_CONST = {"PI": sp.pi, "E": sp.E}


_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[()\[\]]|[^\s()\[\]"]+')


def _sexpr(text: str) -> Any:
    tokens = [{"[": "(", "]": ")"}.get(t, t) for t in _TOKEN.findall(text)]
    pos = 0

    def read() -> Any:
        nonlocal pos
        if pos >= len(tokens):
            raise ParseError("FPCore: unexpected end of input")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out = []
            while pos < len(tokens) and tokens[pos] != ")":
                out.append(read())
            if pos >= len(tokens):
                raise ParseError("FPCore: missing ')'")
            pos += 1
            return out
        if tok == ")":
            raise ParseError("FPCore: unexpected ')'")
        return tok

    result = read()
    return result


def _identifier(name: str) -> str:
    """FPCore symbols may contain ``/`` or ``-`` (``b/2``); code needs identifiers."""
    if name.isidentifier() and not name.startswith("_"):
        return name
    clean = re.sub(r"\W", "_", name).strip("_") or "v"
    return clean if clean[0].isalpha() else "v" + clean


def from_fpcore(text: str) -> tuple[Node, dict[str, Any]]:
    """Read one FPCore benchmark (``(FPCore (x) :pre ... body)``) or a bare body.

    Returns ``(formula, properties)``; ``properties`` holds the ``:name``/``:pre``
    etc. as raw s-expressions and ``"args"``. Supported: arithmetic, the functions
    in ``_FPCORE_OPS``, ``let``/``let*``, ``if`` with comparisons, ``sqr``/``cube``
    (Herbie's output spellings), ``PI``/``E``, decimal and rational literals. Others
    raise ``ParseError`` — the reader refuses rather than guesses.

        >>> f, props = from_fpcore("(FPCore (x) :name \\"t\\" (- (sqrt (+ x 1)) (sqrt x)))")
        >>> str(f), props["args"]
        ('sqrt(x + 1) - sqrt(x)', ['x'])
    """
    tree = _sexpr(text)
    props: dict[str, Any] = {}
    body = tree
    if isinstance(tree, list) and tree and tree[0] == "FPCore":
        rest = tree[1:]
        if rest and isinstance(rest[0], str):  # named FPCore
            rest = rest[1:]
        if not rest or not isinstance(rest[0], list):
            raise ParseError("FPCore: missing argument list")
        props["args"] = [_identifier(a if isinstance(a, str) else a[-1]) for a in rest[0]]
        rest = rest[1:]
        while len(rest) >= 2 and isinstance(rest[0], str) and rest[0].startswith(":"):
            props[rest[0][1:]] = rest[1]
            rest = rest[2:]
        if len(rest) != 1:
            raise ParseError("FPCore: expected exactly one body expression")
        body = rest[0]
    return _from_sexpr(body, {}), props


def _from_sexpr(s: Any, env: dict[str, Node]) -> Node:
    if isinstance(s, str):
        if s in env:
            return env[s]
        if s in _FPCORE_CONST:
            return _const(_FPCORE_CONST[s])
        try:
            return _const(sp.Rational(s))
        except (TypeError, ValueError):
            pass
        if s.startswith('"'):
            raise ParseError(f"FPCore: unexpected string {s}")
        return _var(_identifier(s))
    if not s:
        raise ParseError("FPCore: empty expression")
    head, args = s[0], s[1:]
    if head in ("let", "let*"):
        inner = dict(env)
        for name, value in args[0]:
            inner[name] = _from_sexpr(value, inner if head == "let*" else env)
        return _from_sexpr(args[1], inner)
    sub = [_from_sexpr(a, env) for a in args]
    if head == "-" and len(sub) == 1:
        return _op("neg", sub[0])
    if head == "sqr" and len(sub) == 1:
        return _op("*", sub[0], sub[0])
    if head == "cube" and len(sub) == 1:
        return _op("*", _op("*", sub[0], sub[0]), sub[0])
    if head in ("+", "*") and len(sub) > 2:
        acc = sub[0]
        for a in sub[1:]:
            acc = _op(head, acc, a)
        return acc
    if head not in _FPCORE_OPS:
        raise ParseError(f"FPCore: unsupported operation {head!r}")
    op = _FPCORE_OPS[head]
    if len(sub) != _ARITY[op]:
        raise ParseError(f"FPCore: {head} takes {_ARITY[op]} argument(s), got {len(sub)}")
    return _op(op, *sub)


# ---- SymPy bridge ----------------------------------------------------------


def _real_symbol(name: str, assumptions: Mapping[str, Mapping[str, bool]] | None = None) -> sp.Symbol:
    extra = dict((assumptions or {}).get(name, {}))
    extra.setdefault("real", True)
    return sp.Symbol(name, **extra)


def to_sympy(n: Node, assumptions: Mapping[str, Mapping[str, bool]] | None = None) -> sp.Expr:
    """The exact real function a formula denotes (helpers like ``log1p`` expanded)."""
    cache: dict[Node, Any] = {}

    def conv(m: Node) -> Any:
        if m in cache:
            return cache[m]
        op, a = m.op, [conv(x) for x in m.args]
        if op == "var":
            r = _real_symbol(m.name, assumptions)
        elif op == "const":
            r = m.value
        elif op == "neg":
            r = -a[0]
        elif op == "+":
            r = a[0] + a[1]
        elif op == "-":
            r = a[0] - a[1]
        elif op == "*":
            r = a[0] * a[1]
        elif op == "/":
            r = a[0] / a[1]
        elif op == "pow":
            r = a[0] ** a[1]
        elif op == "sqrt":
            r = sp.sqrt(a[0])
        elif op == "cbrt":
            r = sp.sign(a[0]) * sp.Abs(a[0]) ** sp.Rational(1, 3)
        elif op == "log1p":
            r = sp.log(1 + a[0])
        elif op == "expm1":
            r = sp.exp(a[0]) - 1
        elif op == "hypot":
            r = sp.sqrt(a[0] ** 2 + a[1] ** 2)
        elif op == "fma":
            r = a[0] * a[1] + a[2]
        elif op == "fabs":
            r = sp.Abs(a[0])
        elif op == "atan2":
            r = sp.atan2(a[0], a[1])
        elif op == "max":
            r = sp.Max(a[0], a[1])
        elif op == "min":
            r = sp.Min(a[0], a[1])
        elif op in _COMPARE:
            r = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](a[0], a[1])
        elif op == "if":
            r = sp.Piecewise((a[1], a[0]), (a[2], True))
        else:
            r = getattr(sp, op)(a[0])
        cache[m] = r
        return r

    return conv(n)


def from_sympy(e: Any) -> Node:
    """A SymPy expression as a formula, in the order SymPy holds it.

    Rational coefficients become divisions (``x/3``, not ``x*0.333…``), a ``-1``
    factor becomes a subtraction or negation, ``x**2`` becomes ``x*x`` and
    ``x**(1/2)`` ``sqrt(x)`` — the spellings a programmer would write.
    """
    e = sp.sympify(e)
    if e.is_Symbol:
        return _var(e.name)
    if e.is_Rational:
        if e.q == 1 or _const_text(e)[0] != "(":
            return _const(e)
        return _op("/", _const(e.p), _const(e.q))
    if e in (sp.pi, sp.E):
        return _const(e)
    if e.is_Float:
        return _const(sp.Rational(e))
    if e.is_Add:
        terms = list(e.args)
        positive = [t for t in terms if not t.could_extract_minus_sign()]
        negative = [t for t in terms if t.could_extract_minus_sign()]
        if positive:
            acc = from_sympy(positive[0])
            rest = positive[1:]
        else:
            acc = _op("neg", from_sympy(-negative[0]))
            negative = negative[1:]
            rest = []
        for t in rest:
            acc = _op("+", acc, from_sympy(t))
        for t in negative:
            acc = _op("-", acc, from_sympy(-t))
        return acc
    if e.is_Mul:
        if e.could_extract_minus_sign():
            inner = -e
            return _op("neg", from_sympy(inner))
        num, den = [], []
        for f in e.args:
            if f.is_Rational:
                if f.p != 1:
                    num.append(_const(f.p))
                if f.q != 1:
                    den.append(_const(f.q))
            elif f.is_Pow and f.exp.is_Rational and f.exp < 0:
                den.append(from_sympy(f.base ** (-f.exp)))
            else:
                num.append(from_sympy(f))
        top = num[0] if num else _ONE
        for f in num[1:]:
            top = _op("*", top, f)
        if not den:
            return top
        bottom = den[0]
        for f in den[1:]:
            bottom = _op("*", bottom, f)
        return _op("/", top, bottom)
    if e.is_Pow:
        base, ex = e.base, e.exp
        if base == sp.E:
            return _op("exp", from_sympy(ex))
        b = from_sympy(base)
        if ex == sp.Rational(1, 2):
            return _op("sqrt", b)
        if ex == sp.Rational(-1, 2):
            return _op("/", _ONE, _op("sqrt", b))
        if ex == 2:
            return _op("*", b, b)
        if ex.is_Integer and ex < 0:
            return _op("/", _ONE, from_sympy(base ** (-ex)))
        if ex.is_Rational and not ex.is_Integer and _const_text(ex)[0] == "(":
            return _op("pow", b, _op("/", _const(ex.p), _const(ex.q)))
        return _op("pow", b, from_sympy(ex))
    if isinstance(e, sp.exp):
        return _op("exp", from_sympy(e.args[0]))
    if isinstance(e, sp.log):
        if len(e.args) == 2:
            return _op("/", _op("log", from_sympy(e.args[0])), _op("log", from_sympy(e.args[1])))
        return _op("log", from_sympy(e.args[0]))
    if isinstance(e, sp.Abs):
        return _op("fabs", from_sympy(e.args[0]))
    if isinstance(e, sp.atan2):
        return _op("atan2", from_sympy(e.args[0]), from_sympy(e.args[1]))
    if isinstance(e, (sp.Max, sp.Min)):
        op = "max" if isinstance(e, sp.Max) else "min"
        args = [from_sympy(a) for a in e.args]
        acc = args[0]
        for a in args[1:]:
            acc = _op(op, acc, a)
        return acc
    from sympy.codegen import cfunctions as cf

    special = {cf.log1p: "log1p", cf.expm1: "expm1", cf.hypot: "hypot", cf.Cbrt: "cbrt", cf.fma: "fma"}
    for cls, op in special.items():
        if isinstance(e, cls):
            return _op(op, *(from_sympy(a) for a in e.args))
    if isinstance(e, sp.Piecewise):
        pieces = list(e.args)
        acc = from_sympy(pieces[-1][0])
        if pieces[-1][1] is not sp.true:
            raise UnsupportedFormError("a Piecewise without a final 'otherwise' branch has no float meaning")
        for value, cond in reversed(pieces[:-1]):
            acc = _op("if", from_sympy(cond), from_sympy(value), acc)
        return acc
    rel = {sp.StrictLessThan: "<", sp.LessThan: "<=", sp.StrictGreaterThan: ">", sp.GreaterThan: ">="}
    for cls, op in rel.items():
        if isinstance(e, cls):
            return _op(op, from_sympy(e.lhs), from_sympy(e.rhs))
    name = type(e).__name__
    if name in _FUNCTIONS and len(e.args) == 1:
        return _op(_FUNCTIONS[name], from_sympy(e.args[0]))
    raise UnsupportedFormError(f"no floating-point spelling for {name} ({e})")


def _as_formula(expr: Any) -> Node:
    if isinstance(expr, Node):
        return expr
    if isinstance(expr, str):
        return parse_formula(expr)
    sy = getattr(expr, "sy", expr)  # pycodemath.Expr
    if isinstance(sy, sp.Basic):
        return from_sympy(sy)
    raise DomainError(f"expression must be text, a SymPy expression or a Node, not {type(expr).__name__}")


# --------------------------------------------------------------------------
# Evaluation in floating point
# --------------------------------------------------------------------------

_NP_UNARY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "neg": np.negative, "sqrt": np.sqrt, "cbrt": np.cbrt, "exp": np.exp, "log": np.log,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos,
    "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh, "log1p": np.log1p,
    "expm1": np.expm1, "fabs": np.abs,
}
_NP_BINARY: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "pow": np.power,
    "hypot": np.hypot, "max": np.maximum, "min": np.minimum, "atan2": np.arctan2, "<": np.less,
    "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
}


def _const_float(value: Any, fmt: _Format) -> float:
    v = sp.sympify(value)
    if v.is_Rational:
        return _round_rational(int(v.p), int(v.q), fmt)
    with mpmath.workprec(fmt.prec + 64):
        m = mpmath.mpf(sp.N(v, 40))
    return _round_mpf(m._mpf_, fmt)


def _fma_exact(a: np.ndarray, b: np.ndarray, c: np.ndarray, fmt: _Format) -> np.ndarray:
    out = np.empty(np.broadcast(a, b, c).shape, dtype=fmt.dtype)
    for i, (x, y, z) in enumerate(np.broadcast(a, b, c)):
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            out.flat[i] = x * y + z
            continue
        r = Fraction(float(x)) * Fraction(float(y)) + Fraction(float(z))
        out.flat[i] = _round_rational(r.numerator, r.denominator, fmt)
    return out


def _eval_float(n: Node, env: Mapping[str, np.ndarray], fmt: _Format, cache: dict | None = None,
                size: "int | None" = None) -> np.ndarray:
    """Evaluate the formula in ``fmt``, every operation rounded, written order kept."""
    cache = {} if cache is None else cache
    shape = np.shape(next(iter(env.values()))) if env else ((size,) if size is not None else ())
    with np.errstate(all="ignore"):
        for m in n.walk():
            if m in cache:
                continue
            op = m.op
            if op == "var":
                r = np.asarray(env[m.name], dtype=fmt.dtype)
            elif op == "const":
                r = np.full(shape, _const_float(m.value, fmt), dtype=fmt.dtype)
            elif op in _NP_UNARY:
                r = _NP_UNARY[op](cache[m.args[0]])
            elif op in _NP_BINARY:
                r = _NP_BINARY[op](cache[m.args[0]], cache[m.args[1]])
            elif op == "fma":
                r = _fma_exact(cache[m.args[0]], cache[m.args[1]], cache[m.args[2]], fmt)
            elif op == "if":
                r = np.where(cache[m.args[0]], cache[m.args[1]], cache[m.args[2]])
            else:  # pragma: no cover - _ARITY and the tables agree
                raise UnsupportedFormError(f"cannot evaluate {op!r}")
            if r.dtype != np.bool_:
                r = r.astype(fmt.dtype, copy=False)
            cache[m] = r
    return cache[n]


def _eval_on(n: Node, sample: "_Sample", fmt: _Format) -> np.ndarray:
    """``_eval_float`` over a sample (sized even for a formula with no variables)."""
    out = _eval_float(n, sample.env, fmt, size=len(sample))
    return np.broadcast_to(out, (len(sample),)).astype(fmt.dtype) if np.ndim(out) == 0 else out


# --------------------------------------------------------------------------
# Ground truth: interval arithmetic with rising precision
# --------------------------------------------------------------------------

iv = mpmath.iv

#: Working precisions tried, in bits. The last one is the give-up point.
_PRECISIONS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
#: Intermediates beyond 2**(2**20) in magnitude are not followed (the point is skipped).
_MAX_MAG = 1 << 20
#: Periodic functions are not applied to arguments beyond 2**_MAX_TRIG_MAG.
_MAX_TRIG_MAG = 4096
#: ``exp`` of anything above this leaves the followed magnitude (an mpf, built once:
#: comparing against the integer ``2**(2**20 - 1)`` cost seconds per call).
_EXP_LIMIT = mpmath.ldexp(1, _MAX_MAG - 1)


class _Invalid(Exception):
    """The exact value is not a real number at this point (domain)."""


class _Unresolved(Exception):
    """This precision cannot decide; more bits may."""


class _OutOfReach(Exception):
    """No precision will decide here (magnitude beyond what is followed)."""


def _exact(parts: tuple) -> Any:
    """An ``mpf`` from its raw tuple WITHOUT rounding (``mpmath.mpf(t)`` rounds to 53 bits)."""
    return mpmath.mp.make_mpf(parts)


def _ends(x: Any) -> tuple[Any, Any]:
    a, b = x._mpi_
    return _exact(a), _exact(b)


def _iv(lo: Any, hi: Any) -> Any:
    return iv.mpf([lo, hi])


def _check_mag(x: Any) -> Any:
    for end in x._mpi_:
        sign, man, exp, bc = end
        if man and exp + bc > _MAX_MAG:
            raise _OutOfReach()
        if not man and exp in (-456, -123):  # inf or nan endpoint
            raise _Unresolved()
    return x


def _contains_zero(x: Any) -> bool:
    lo, hi = _ends(x)
    return lo <= 0 <= hi


def _monotone(fn: Callable[[Any], Any], x: Any, prec: int, increasing: bool = True) -> Any:
    lo, hi = _ends(x)
    with mpmath.workprec(prec + 40):
        fl, fh = fn(lo), fn(hi)
    if not increasing:
        fl, fh = fh, fl
    with mpmath.workprec(prec + 40):
        slack = mpmath.ldexp(1, -prec)
        fl = fl - abs(fl) * slack
        fh = fh + abs(fh) * slack
    return _iv(fl, fh)


def _iv_atan2(y: Any, x: Any, prec: int) -> Any:
    """``atan2(y, x)`` away from its branch cut (the negative x-axis) and the origin."""
    (yl, yh), (xl, xh) = _ends(y), _ends(x)
    _require(yl == yh == 0 and xl == xh == 0, False)
    if xl > 0:
        return _monotone(mpmath.atan, y / x, prec)
    if yl > 0:
        return iv.pi / 2 - _monotone(mpmath.atan, x / y, prec)  # arithmetic fixes iv.pi
    if yh < 0:
        return -iv.pi / 2 - _monotone(mpmath.atan, x / y, prec)
    if xh < 0 and yl == yh == 0:
        return iv.pi + 0  # exactly on the cut: atan2(+0, x < 0) = pi (``+ 0``: not lazy)
    raise _Unresolved()


def _real_cbrt(t: Any) -> Any:
    """The real cube root (``mpmath.cbrt`` is the principal, complex one for t < 0)."""
    return mpmath.cbrt(t) if t >= 0 else -mpmath.cbrt(-t)


def _require(cond_invalid: bool, cond_unresolved: bool) -> None:
    if cond_invalid:
        raise _Invalid()
    if cond_unresolved:
        raise _Unresolved()


def _iv_apply(op: str, a: list, prec: int, m: Node) -> Any:
    """One operation on intervals at ``iv.prec == prec``; domain checks first."""
    if op == "neg":
        return -a[0]
    if op == "+":
        return a[0] + a[1]
    if op == "-":
        return a[0] - a[1]
    if op == "*":
        return a[0] * a[1]
    if op == "/":
        lo, hi = _ends(a[1])
        _require(lo == 0 == hi, lo <= 0 <= hi)
        return a[0] / a[1]
    if op == "sqrt":
        lo, hi = _ends(a[0])
        _require(hi < 0, lo < 0)
        return iv.sqrt(a[0])
    if op in ("log", "log1p"):
        arg = a[0] + 1 if op == "log1p" else a[0]
        lo, hi = _ends(arg)
        _require(hi <= 0, lo <= 0)
        return iv.log(arg)
    if op == "exp":
        lo, hi = _ends(a[0])
        if hi > _EXP_LIMIT:
            raise _OutOfReach()
        return iv.exp(a[0])
    if op == "expm1":
        lo, hi = _ends(a[0])
        if hi > _EXP_LIMIT:
            raise _OutOfReach()
        return iv.exp(a[0]) - 1
    if op in ("sin", "cos", "tan"):
        lo, hi = _ends(a[0])
        if max(abs(lo), abs(hi)) > mpmath.ldexp(1, _MAX_TRIG_MAG):
            raise _OutOfReach()
        r = getattr(iv, op)(a[0])
        rl, rh = _ends(r)
        if op == "tan" and (mpmath.isinf(rl) or mpmath.isinf(rh)):
            raise _Unresolved()
        return r
    if op == "pow":
        base, ex = a
        el, eh = _ends(ex)
        bl, bh = _ends(base)
        exact_ex = m.args[1]
        if exact_ex.op == "const" and exact_ex.value.is_Integer:
            k = int(exact_ex.value)
            if k < 0:
                _require(bl == 0 == bh, bl <= 0 <= bh)
            return base ** k
        if el != eh:
            # Exponent not a point: only a positive base has a real power.
            _require(bh < 0, bl <= 0)
            return iv.exp(ex * iv.log(base))
        if bl == 0 == bh:
            _require(el <= 0, False)
            return _iv(0, 0)
        _require(bh < 0, bl <= 0)
        return iv.exp(ex * iv.log(base))
    if op == "fabs":
        return abs(a[0])
    if op == "hypot":
        return iv.sqrt(a[0] ** 2 + a[1] ** 2)
    if op == "fma":
        return a[0] * a[1] + a[2]
    if op in ("max", "min"):
        (al, ah), (bl, bh) = _ends(a[0]), _ends(a[1])
        pick = max if op == "max" else min
        return _iv(pick(al, bl), pick(ah, bh))
    if op == "cbrt":
        return _monotone(_real_cbrt, a[0], prec)
    if op == "atan2":
        return _iv_atan2(a[0], a[1], prec)
    if op == "atan":
        return _monotone(mpmath.atan, a[0], prec)
    if op == "sinh":
        return _monotone(mpmath.sinh, a[0], prec)
    if op == "tanh":
        return _monotone(mpmath.tanh, a[0], prec)
    if op == "asinh":
        return _monotone(mpmath.asinh, a[0], prec)
    if op in ("asin", "acos", "atanh"):
        lo, hi = _ends(a[0])
        if op == "atanh":
            _require(hi <= -1 or lo >= 1, lo <= -1 or hi >= 1)
            return _monotone(mpmath.atanh, a[0], prec)
        _require(hi < -1 or lo > 1, lo < -1 or hi > 1)
        return _monotone(getattr(mpmath, op), a[0], prec, increasing=(op == "asin"))
    if op == "acosh":
        lo, hi = _ends(a[0])
        _require(hi < 1, lo < 1)
        return _monotone(mpmath.acosh, a[0], prec)
    if op == "cosh":
        lo, hi = _ends(a[0])
        with mpmath.workprec(prec + 40):
            vals = [mpmath.cosh(lo), mpmath.cosh(hi)]
            slack = mpmath.ldexp(1, -prec)
            top = max(vals) * (1 + slack)
            bottom = mpmath.mpf(1) if lo <= 0 <= hi else min(vals) * (1 - slack)
        return _iv(bottom, top)
    raise UnsupportedFormError(f"no exact evaluation for {op!r}")


def _iv_compare(op: str, a: Any, b: Any) -> bool:
    (al, ah), (bl, bh) = _ends(a), _ends(b)
    if op in ("<", "<="):
        if ah < bl or (op == "<=" and ah <= bl):
            return True
        if al > bh or (op == "<" and al >= bh):
            return False
    else:
        if al > bh or (op == ">=" and al >= bh):
            return True
        if ah < bl or (op == ">" and ah <= bl):
            return False
    raise _Unresolved()


def _iv_eval(n: Node, point: Mapping[str, float], prec: int) -> dict[Node, Any]:
    """Every node's enclosure at ``prec`` bits (``_Invalid`` / ``_Unresolved`` / ``_OutOfReach``)."""
    old = iv.prec
    iv.prec = prec
    try:
        out: dict[Node, Any] = {}
        order = n.walk()
        for m in order:
            if m.op == "var":
                out[m] = iv.mpf(point[m.name])
            elif m.op == "const":
                v = m.value
                if v.is_Rational:
                    out[m] = iv.mpf(v.p) / v.q if v.q != 1 else iv.mpf(v.p)
                else:
                    # ``iv.pi`` is lazy: read later, it would be computed at whatever
                    # precision is active then. ``+ 0`` fixes it at ``prec`` now.
                    out[m] = (iv.pi if v == sp.pi else iv.e) + 0
            elif m.op == "if":
                pass  # handled lazily below
            elif m.op in _COMPARE:
                pass
            else:
                if any(a.op in _COMPARE or (a.op == "if" and a not in out) for a in m.args):
                    _iv_resolve_ifs(m, out, point, prec)
                out[m] = _check_mag(_iv_apply(m.op, [out[a] for a in m.args], prec, m))
        if n not in out:
            _iv_resolve_ifs(n, out, point, prec)
        return out
    finally:
        iv.prec = old


def _iv_resolve_ifs(m: Node, out: dict[Node, Any], point: Mapping[str, float], prec: int) -> None:
    """Fill ``out`` for the ``if`` nodes under ``m`` (only the taken branch is needed)."""
    def value(k: Node) -> Any:
        if k in out:
            return out[k]
        if k.op == "if":
            c = k.args[0]
            taken = _iv_compare(c.op, value(c.args[0]), value(c.args[1]))
            out[k] = value(k.args[1] if taken else k.args[2])
            return out[k]
        if k.op in _COMPARE:
            raise UnsupportedFormError("a comparison is only meaningful as the test of an 'if'")
        out[k] = _check_mag(_iv_apply(k.op, [value(a) for a in k.args], prec, k))
        return out[k]

    if m.op == "if":
        value(m)
    else:
        for a in m.args:
            value(a)


@dataclass
class _Truth:
    """Exact values at one point: the rounded root value, and the nodes' enclosures."""

    value: float
    nodes: dict[Node, Any]
    prec: int


def _truth(n: Node, point: Mapping[str, float], fmt: _Format) -> "_Truth | str":
    """The correctly rounded exact value, or the reason there is none."""
    for prec in _PRECISIONS:
        try:
            nodes = _iv_eval(n, point, prec)
        except _Invalid:
            return "invalid"
        except _OutOfReach:
            return "out-of-reach"
        except _Unresolved:
            continue
        except (ZeroDivisionError, ValueError, TypeError, NotImplementedError, mpmath.libmp.ComplexResult):
            continue
        lo, hi = nodes[n]._mpi_
        rl, rh = _round_mpf(lo, fmt), _round_mpf(hi, fmt)
        if rl == rh and not math.isnan(rl):
            return _Truth(rl, nodes, prec)
    return "unresolved"


def _round_node(x: Any, fmt: _Format) -> float:
    """A node's exact value rounded to ``fmt`` (midpoint of a resolved enclosure)."""
    lo, hi = x._mpi_
    rl = _round_mpf(lo, fmt)
    if rl == _round_mpf(hi, fmt):
        return rl
    mid = mpmath.ldexp(mpmath.fadd(_exact(lo), _exact(hi), exact=True), -1)
    return _round_mpf(mid._mpf_, fmt)


def _exact_text(x: Any, digits: int = 20) -> str:
    lo, hi = x._mpi_
    with mpmath.workprec(256):
        mid = (_exact(lo) + _exact(hi)) / 2
        return mpmath.nstr(mid, digits)


# --------------------------------------------------------------------------
# Bounded SymPy work
# --------------------------------------------------------------------------


def _caller_budget_active() -> bool:
    """Is this thread already inside a ``time_budget``? (as in ``verify.equal``)

    A tighter budget nested inside another poisons the outer one when it fires
    (``verify/equal.py`` measured it), so inside a caller's budget this module arms
    none of its own and lets the caller's allowance govern.
    """
    with _budget._LOCK:
        return threading.get_ident() in _budget._ARMED


def _bounded(seconds: float, fn: Callable[[], Any]) -> Any:
    """``fn()`` under its own short budget; ``None`` if it ran out or failed.

    Used for optional symbolic work (a simplification, a series): a step that does
    not finish yields no candidate, never a wrong one. Inside a caller's budget the
    caller's deadline governs and its expiry propagates as usual.
    """
    try:
        if _caller_budget_active():
            return fn()
        with time_budget(max(seconds, 1e-3), "stability", "symbolic step"):
            return fn()
    except TimeBudgetError:
        if _caller_budget_active():
            raise
        return None
    except (RecursionError, MemoryError):
        return None
    except Exception:  # noqa: BLE001 - SymPy raises many types; the step just yields nothing
        return None


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def _check_ranges(n: Node, ranges: Mapping[str, Any], fmt: _Format) -> dict[str, tuple[float, float]]:
    if not isinstance(ranges, Mapping):
        raise DomainError("ranges must be a mapping {variable: (low, high)}")
    out: dict[str, tuple[float, float]] = {}
    missing = [v for v in n.variables if v not in ranges]
    if missing:
        raise DomainError(f"no range given for variable(s) {', '.join(missing)}")
    extra = [v for v in ranges if v not in n.variables]
    if extra:
        raise DomainError(f"range given for {', '.join(map(str, extra))}, which the formula does not use")
    for name in n.variables:
        r = ranges[name]
        try:
            lo, hi = (float(r[0]), float(r[1]))
        except (TypeError, ValueError, IndexError):
            raise DomainError(f"range for {name} must be a pair of numbers, not {r!r}") from None
        if math.isnan(lo) or math.isnan(hi):
            raise DomainError(f"range for {name} contains NaN")
        lo = max(lo, -fmt.max)
        hi = min(hi, fmt.max)
        lo = float(np.asarray(lo, dtype=fmt.dtype))
        hi = float(np.asarray(hi, dtype=fmt.dtype))
        if lo > hi:
            raise DomainError(f"range for {name} is empty ({r[0]} > {r[1]})")
        out[name] = (lo, hi)
    return out


def _uniform_bits(rng: np.random.Generator, lo: float, hi: float, k: int, fmt: _Format) -> np.ndarray:
    a, b = _ordinals(np.array([lo, hi], dtype=fmt.dtype), fmt)
    ords = rng.integers(int(a), int(b), size=k, endpoint=True, dtype=np.int64)
    return _from_ordinals(ords, fmt)


@dataclass
class _Sample:
    """Points (one array per variable) with their exact values, and skip counts."""

    env: dict[str, np.ndarray]
    exact: np.ndarray
    truths: list[_Truth]
    skipped: dict[str, int]
    attempted: int

    def __len__(self) -> int:
        return len(self.exact)


def _sample(
    n: Node,
    ranges: Mapping[str, tuple[float, float]],
    k: int,
    seed: int,
    fmt: _Format,
    deadline: float,
    extra_points: Sequence[Mapping[str, float]] = (),
) -> _Sample:
    rng = np.random.default_rng(seed)
    names = n.variables
    points: list[dict[str, float]] = [dict(p) for p in extra_points]
    pending = 0
    if not extra_points:
        pending = k
    skipped = {"invalid": 0, "out-of-reach": 0, "unresolved": 0}
    kept_env: dict[str, list[float]] = {v: [] for v in names}
    exact: list[float] = []
    truths: list[_Truth] = []
    attempted = 0
    max_attempts = 20 * k + 50
    while True:
        if not points:
            if pending <= 0 or attempted >= max_attempts:
                break
            batch = max(16, pending * 2)
            cols = {v: _uniform_bits(rng, *ranges[v], batch, fmt) for v in names}
            points = [{v: float(cols[v][i]) for v in names} for i in range(batch)]
        p = points.pop(0)
        if time.perf_counter() > deadline:
            raise TimeBudgetError(
                "stability: the time budget ran out while computing exact values — "
                "raise budget= or lower samples=",
                operation="stability",
            )
        attempted += 1
        t = _truth(n, p, fmt)
        if isinstance(t, str):
            skipped[t] += 1
            continue
        for v in names:
            kept_env[v].append(p[v])
        exact.append(t.value)
        truths.append(t)
        if not extra_points:
            pending -= 1
            if pending <= 0:
                points = []
    env = {v: np.array(kept_env[v], dtype=fmt.dtype) for v in names}
    return _Sample(env, np.array(exact, dtype=fmt.dtype), truths, skipped, attempted)


def _special_points(n: Node, ranges: Mapping[str, tuple[float, float]], fmt: _Format, seed: int) -> list[dict[str, float]]:
    """Range ends, 0, +-1, powers of two and floats next to denominator zeros."""
    rng = np.random.default_rng(seed + 7919)
    names = n.variables
    pts: list[dict[str, float]] = []
    for v in names:
        lo, hi = ranges[v]
        cands = {lo, hi, 0.0, 1.0, -1.0}
        for e in (-1000, -100, -30, -10, -1, 1, 10, 30, 100, 1000):
            for s in (1.0, -1.0):
                cands.add(s * math.ldexp(1.0, e))
        for z in _denominator_zeros(n, v, lo, hi):
            f = float(np.asarray(z, dtype=fmt.dtype))
            cands.update({f, float(np.nextafter(np.asarray(f, dtype=fmt.dtype), fmt.dtype(np.inf))),
                          float(np.nextafter(np.asarray(f, dtype=fmt.dtype), fmt.dtype(-np.inf)))})
        for c in sorted(cands):
            if lo <= c <= hi and math.isfinite(c):
                p = {w: float(_uniform_bits(rng, *ranges[w], 1, fmt)[0]) for w in names if w != v}
                p[v] = float(c)
                pts.append(p)
    return pts


def _denominator_zeros(n: Node, var: str, lo: float, hi: float) -> list[float]:
    """Real zeros of single-variable denominators inside ``[lo, hi]`` (best effort)."""
    zeros: list[float] = []
    for m in n.walk():
        if m.op != "/" or m.args[1].variables != (var,) or len(m.args[1].variables) != 1:
            continue
        den = to_sympy(m.args[1])
        sols = _bounded(1.0, functools.partial(sp.solveset, den, _real_symbol(var), sp.Interval(lo, hi)))
        if isinstance(sols, sp.FiniteSet) and len(sols) <= 8:
            try:
                zeros.extend(float(s) for s in sols)
            except TypeError:  # a symbolic zero with no float value
                continue
    return zeros


# --------------------------------------------------------------------------
# Localisation: the local error of every operation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Culprit:
    """A sub-expression ranked by the bits its own operation loses.

    Local error = the operation applied in floating point to the EXACT values of
    its operands (rounded to the format), against its exact result. A large local
    error means this operation is where accuracy is lost, whatever happens around
    it. ``witness`` is the input where it is worst, with both values.
    """

    expression: str
    operation: str
    mean_bits: float
    max_bits: float
    cause: str
    hint: str
    witness: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression, "operation": self.operation,
            "mean_bits": round(self.mean_bits, 3), "max_bits": round(self.max_bits, 3),
            "cause": self.cause, "hint": self.hint, "witness": dict(self.witness),
        }


def _local_errors(n: Node, sample: _Sample, fmt: _Format) -> dict[Node, np.ndarray]:
    """Local error (bits) of each operation node at each sample point."""
    nodes = [m for m in n.walk() if m.op not in ("var", "const") and m.op not in _COMPARE and m.op != "if"]
    out: dict[Node, np.ndarray] = {}
    count = len(sample)
    rounded: dict[Node, np.ndarray] = {}

    def exact_rounded(m: Node) -> np.ndarray:
        if m not in rounded:
            vals = np.full(count, np.nan, dtype=fmt.dtype)
            for i, t in enumerate(sample.truths):
                if m in t.nodes:
                    vals[i] = _round_node(t.nodes[m], fmt)
            rounded[m] = vals
        return rounded[m]

    for m in nodes:
        if any(a.op in _COMPARE or a.op == "if" for a in m.args):
            continue
        env = {f"_a{i}": exact_rounded(a) for i, a in enumerate(m.args)}
        local = Node(m.op, tuple(_var(f"_a{i}") for i in range(len(m.args))))
        computed = _eval_float(local, env, fmt)
        out[m] = _bits_of_error(computed, exact_rounded(m), fmt)
    return out


def _cause(m: Node, sample: _Sample, i: int, fmt: _Format) -> str:
    """Why the operation at ``m`` loses bits at sample point ``i``."""
    t = sample.truths[i]
    try:
        vals = [t.nodes[a] for a in m.args]
        mine = t.nodes[m]
    except KeyError:
        return "rounding error amplified"
    with mpmath.workprec(64):
        mid = [abs((_exact(x._mpi_[0]) + _exact(x._mpi_[1])) / 2) for x in vals]
        me = abs((_exact(mine._mpi_[0]) + _exact(mine._mpi_[1])) / 2)
    rounded = [_round_node(x, fmt) for x in vals]
    if any(math.isinf(r) for r in rounded) and math.isfinite(_round_node(mine, fmt)):
        return "overflow in an intermediate result"
    if any(r == 0 and x != 0 for r, x in zip(rounded, mid)) or any(
        r != 0 and abs(r) < np.finfo(fmt.dtype).tiny for r in rounded
    ):
        return "underflow in an intermediate result"
    if m.op in ("+", "-") and max(mid) > 0 and me < max(mid) * 2.0**-4:
        return "cancellation: subtracting nearly equal values"
    if m.op in ("log",) and abs(mid[0] - 1) < 0.5:
        return "log of a value close to 1 loses the digits of the small part"
    if m.op in ("sin", "cos", "tan") and mid[0] > 2.0**30:
        return "argument reduction of a huge argument"
    if m.op == "/" and me != 0 and mid[1] != 0:
        return "rounding of the operands amplified by the division"
    return "rounding error of the operands amplified by this operation"


def _localize(n: Node, sample: _Sample, fmt: _Format, top: int = 5) -> list[Culprit]:
    if len(sample) == 0:
        return []
    errors = _local_errors(n, sample, fmt)
    ranked = sorted(errors.items(), key=lambda kv: -float(np.mean(kv[1])))
    out: list[Culprit] = []
    from . import _rewrites

    for m, err in ranked[:top]:
        mean = float(np.mean(err))
        if mean < 0.5 and float(np.max(err)) < 4:
            continue
        i = int(np.argmax(err))
        point = {v: repr(float(sample.env[v][i])) for v in n.variables}
        out.append(
            Culprit(
                expression=str(m), operation=m.op, mean_bits=mean, max_bits=float(err[i]),
                cause=_cause(m, sample, i, fmt), hint=_rewrites.hint_for(m),
                witness=point,
            )
        )
    return out


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PointError:
    """One input with the computed value, the exact value and the bits lost."""

    inputs: dict[str, str]
    computed: str
    exact: str
    bits: float

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": dict(self.inputs), "computed": self.computed, "exact": self.exact,
                "bits": round(self.bits, 3)}


@dataclass(frozen=True)
class RegionError:
    """Mean and maximum bits of error for one slice of one variable's range."""

    variable: str
    low: float
    high: float
    points: int
    mean_bits: float
    max_bits: float

    def to_dict(self) -> dict[str, Any]:
        return {"variable": self.variable, "low": repr(self.low), "high": repr(self.high),
                "points": self.points, "mean_bits": round(self.mean_bits, 3),
                "max_bits": round(self.max_bits, 3)}


@dataclass(frozen=True)
class StabilityReport:
    """How accurately a formula evaluates in floating point over the given ranges.

    ``mean_bits`` / ``max_bits`` over ``samples`` random points (uniform over float
    bit patterns), ``worst`` is the witness of the maximum (input, computed, exact).
    ``regions`` is the error map: each variable's range cut into slices of equal
    float count. ``culprits`` ranks sub-expressions by local error. ``special``
    lists the special points (ends, 0, powers of two, near denominator zeros).
    ``skipped`` counts sampled inputs with no real or no decidable exact value.
    """

    expression: str
    dtype: str
    ranges: dict[str, tuple[float, float]]
    samples: int
    skipped: dict[str, int]
    mean_bits: float
    max_bits: float
    worst: "PointError | None"
    regions: tuple[RegionError, ...]
    culprits: tuple[Culprit, ...]
    special: tuple[PointError, ...]
    verdict: str

    @property
    def accurate(self) -> bool:
        """At most one bit lost on average and three at worst."""
        return self.mean_bits <= 1.0 and self.max_bits <= 3.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "stability",
            "expression": self.expression,
            "dtype": self.dtype,
            "ranges": {k: [repr(a), repr(b)] for k, (a, b) in self.ranges.items()},
            "samples": self.samples,
            "skipped": dict(self.skipped),
            "mean_bits": round(self.mean_bits, 3),
            "max_bits": round(self.max_bits, 3),
            "worst": self.worst.to_dict() if self.worst else None,
            "regions": [r.to_dict() for r in self.regions],
            "culprits": [c.to_dict() for c in self.culprits],
            "special": [p.to_dict() for p in self.special],
            "verdict": self.verdict,
        }


def _point_errors(n: Node, sample: _Sample, fmt: _Format) -> tuple[np.ndarray, np.ndarray]:
    computed = _eval_on(n, sample, fmt) if len(sample) else np.array([], dtype=fmt.dtype)
    return computed, _bits_of_error(computed, sample.exact, fmt)


def _witness(n: Node, sample: _Sample, computed: np.ndarray, bits: np.ndarray, i: int) -> PointError:
    return PointError(
        inputs={v: repr(float(sample.env[v][i])) for v in n.variables},
        computed=repr(float(computed[i])),
        exact=_exact_text(sample.truths[i].nodes[n]),
        bits=float(bits[i]),
    )


def _regions(n: Node, sample: _Sample, bits: np.ndarray, ranges: Mapping[str, tuple[float, float]],
             fmt: _Format, slices: int = 8) -> list[RegionError]:
    out: list[RegionError] = []
    for v in n.variables:
        lo, hi = ranges[v]
        a, b = (int(o) for o in _ordinals(np.array([lo, hi], dtype=fmt.dtype), fmt))
        k = min(slices, b - a + 1)
        edges = np.linspace(a, b + 1, k + 1).astype(np.int64)
        ords = _ordinals(sample.env[v], fmt)
        for j in range(k):
            sel = (ords >= edges[j]) & (ords < edges[j + 1])
            low = float(_from_ordinals(np.array([edges[j]]), fmt)[0])
            high = float(_from_ordinals(np.array([max(edges[j], edges[j + 1] - 1)]), fmt)[0])
            cnt = int(np.sum(sel))
            out.append(RegionError(v, low, high, cnt,
                                   float(np.mean(bits[sel])) if cnt else 0.0,
                                   float(np.max(bits[sel])) if cnt else 0.0))
    return out


def _check_samples(samples: object, what: str = "samples") -> int:
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise DomainError(f"{what} must be a positive integer, not {samples!r}")
    if samples > 100_000:
        raise DomainError(f"{what} must be at most 100000, not {samples}")
    return samples


def _check_budget(budget: object) -> float:
    if budget is None:
        return 120.0
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not budget > 0:
        raise DomainError(f"budget must be a positive number of seconds, not {budget!r}")
    return float(budget)


def diagnose(
    expr: "str | Node | sp.Basic",
    ranges: Mapping[str, Any],
    *,
    dtype: str = "float64",
    samples: int = 256,
    seed: int = 0,
    budget: "float | None" = None,
) -> StabilityReport:
    """Measure the floating-point accuracy of ``expr`` over ``ranges``.

    ``expr`` is text in code notation (``"sqrt(x + 1) - sqrt(x)"``), a SymPy
    expression or a ``Node``. ``ranges`` maps every variable to ``(low, high)``.
    Returns a ``StabilityReport``: mean / max bits of error with the worst input as
    witness, an error map per variable, and the culprit sub-expressions.

    Raises ``StabilityDomainError`` when no sampled point has a real exact value,
    ``ParseError`` / ``DomainError`` for bad input, ``TimeBudgetError`` past
    ``budget`` seconds (default 120).

        >>> r = diagnose("log(1 + x)", {"x": (-1e-3, 1e-3)}, samples=64)
        >>> r.mean_bits > 10, r.culprits[0].hint
        (True, 'log(1 + u) -> log1p(u)')
    """
    fmt = _format(dtype)
    samples = _check_samples(samples)
    seconds = _check_budget(budget)
    n = _as_formula(expr)
    bounds = _check_ranges(n, ranges, fmt)
    deadline = time.perf_counter() + seconds
    sample = _sample(n, bounds, samples, seed, fmt, deadline)
    if len(sample) == 0:
        raise StabilityDomainError(
            f"stability: none of {sample.attempted} sampled inputs gives {n} a real, "
            f"decidable exact value ({_skips_text(sample.skipped)}) — check the ranges"
        )
    special = _sample(n, bounds, 0, seed, fmt, deadline, _special_points(n, bounds, fmt, seed))
    return _report(n, bounds, fmt, sample, special)


def _skips_text(skipped: Mapping[str, int]) -> str:
    parts = [f"{v} {k}" for k, v in skipped.items() if v]
    return ", ".join(parts) if parts else "none skipped"


def _report(n: Node, bounds: dict[str, tuple[float, float]], fmt: _Format,
            sample: _Sample, special: _Sample) -> StabilityReport:
    computed, bits = _point_errors(n, sample, fmt)
    i = int(np.argmax(bits))
    worst = _witness(n, sample, computed, bits, i)
    sp_points: list[PointError] = []
    max_bits = float(bits[i])
    if len(special):
        sc, sb = _point_errors(n, special, fmt)
        for j in range(len(special)):
            sp_points.append(_witness(n, special, sc, sb, j))
        j = int(np.argmax(sb))
        if sb[j] > max_bits:
            max_bits = float(sb[j])
            worst = sp_points[j]
    mean = float(np.mean(bits))
    culprits = _localize(n, sample, fmt)
    if mean <= 1.0 and max_bits <= 3.0:
        verdict = f"accurate: {mean:.2f} bits lost on average, {max_bits:.1f} at worst"
    else:
        where = f"; worst at {worst.inputs}" if worst else ""
        verdict = f"loses {mean:.1f} bits on average (max {max_bits:.1f}){where}"
    return StabilityReport(
        expression=str(n), dtype=fmt.name, ranges=dict(bounds), samples=len(sample),
        skipped=dict(sample.skipped), mean_bits=mean, max_bits=max_bits, worst=worst,
        regions=tuple(_regions(n, sample, bits, bounds, fmt)), culprits=tuple(culprits),
        special=tuple(sp_points), verdict=verdict,
    )


# --------------------------------------------------------------------------
# Proof of equivalence
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Proof:
    """Why a rewrite may replace the original.

    ``status`` is ``"verified"`` (proved) — no other status reaches a returned
    candidate. ``exact`` is True for an identity (``method`` = ``check_equal``, the
    project's verifier; ``symbolic`` = simplification under the range's sign
    assumptions; ``exp-injective`` = ``exp(a) == exp(b)`` with both sides real;
    ``case-split`` = one proof per ordering of the arguments of ``max``/``min``;
    ``piecewise`` = every branch of a regime proved). ``exact`` is False only for a
    Taylor polynomial (``method`` = ``taylor-remainder``): its relative truncation
    error is PROVED at most the unit roundoff on the stated interval — an
    approximation no worse than one extra rounding, not an identity. ``conditions``
    are the domain facts the proof used (``x > 0``), taken from the ranges.
    """

    status: str
    method: str
    conditions: tuple[str, ...] = ()
    detail: str = ""
    exact: bool = True
    branches: tuple["Proof", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "method": self.method, "exact": self.exact,
                               "conditions": list(self.conditions), "detail": self.detail}
        if self.branches:
            out["branches"] = [b.to_dict() for b in self.branches]
        return out


def _domain_facts(n: Node) -> dict[str, str]:
    """Signs a variable must have wherever the formula is real.

    ``log(N)`` is real only for ``N > 0`` and ``sqrt(x)`` only for ``x >= 0``, so on
    every input where the original has a value — the only inputs where equality is
    claimed (as in ``check_equal``) — those facts hold and a proof may use them.
    """
    facts: dict[str, str] = {}
    for m in n.walk():
        if m.op in ("log", "sqrt", "acosh") and m.args[0].op == "var":
            v = m.args[0].name
            strict = m.op in ("log", "acosh")
            if strict or facts.get(v) != "positive":
                facts[v] = "positive" if strict else facts.get(v, "nonnegative")
    return facts


def _assumptions(bounds: Mapping[str, tuple[float, float]],
                 n: "Node | None" = None) -> tuple[dict[str, dict[str, bool]], tuple[str, ...]]:
    facts: dict[str, dict[str, bool]] = {}
    text: list[str] = []
    implied = _domain_facts(n) if n is not None else {}
    for v, (lo, hi) in sorted(bounds.items()):
        if v in implied and not (lo > 0 or (lo >= 0 and implied[v] == "nonnegative")):
            facts[v] = {implied[v]: True}
            text.append(f"{v} {'>' if implied[v] == 'positive' else '>='} 0 (where the original is real)")
            continue
        if lo > 0:
            facts[v], t = {"positive": True}, f"{v} > 0"
        elif lo >= 0:
            facts[v], t = {"nonnegative": True}, f"{v} >= 0"
        elif hi < 0:
            facts[v], t = {"negative": True}, f"{v} < 0"
        elif hi <= 0:
            facts[v], t = {"nonpositive": True}, f"{v} <= 0"
        else:
            continue
        text.append(t)
    return facts, tuple(text)


def _step(deadline: float, cap: float = 4.0) -> float:
    return max(0.0, min(cap, deadline - time.perf_counter()))


def _is_zero(e: Any, deadline: float = math.inf) -> bool:
    if e == 0:
        return True
    def via_expand_log(z: Any) -> Any:
        return sp.simplify(sp.expand_log(z))

    def via_logcombine(z: Any) -> Any:
        return sp.simplify(sp.logcombine(z))

    for fn in (sp.simplify, via_expand_log, via_logcombine):
        if _step(deadline) <= 0:
            return False
        r = _bounded(_step(deadline), functools.partial(fn, e))
        if r is not None and (r == 0 or r.is_zero is True):
            return True
    return False


def _all_real(e: sp.Expr) -> bool:
    """Every log argument provably positive and every root argument nonnegative."""
    for sub in sp.preorder_traversal(e):
        if isinstance(sub, sp.log) and sub.args[0].is_positive is not True:
            return False
        if isinstance(sub, sp.Pow) and sub.exp.is_Rational and not sub.exp.is_Integer:
            if sub.base.is_nonnegative is not True:
                return False
    return True


def _case_substitutions(e: sp.Expr) -> "list[dict[sp.Basic, sp.Expr]] | None":
    """One substitution per ordering of the arguments of every ``Max``/``Min``.

    The atoms (symbols, and 0 if it occurs) are put in every total order
    ``a_1 <= a_2 <= ...`` by writing each as the previous one plus a nonnegative gap;
    when 0 is one of the atoms the values are shifted so that it stays 0. Every real
    assignment falls in some case, so an identity proved in every case holds
    everywhere. Only for arguments that are symbols or 0 (``None`` otherwise), and at
    most 5 atoms (120 cases).
    """
    groups: list[tuple[sp.Basic, ...]] = []
    for node in sp.preorder_traversal(e):
        if isinstance(node, (sp.Max, sp.Min)):
            if not all(a.is_Symbol or a == 0 for a in node.args):
                return None
            groups.append(tuple(node.args))
    if not groups:
        return None
    syms = sorted({a for g in groups for a in g if a.is_Symbol}, key=lambda z: z.name)
    has_zero = any(a == 0 for g in groups for a in g)
    atoms: list[sp.Basic] = list(syms) + ([sp.Integer(0)] if has_zero else [])
    if len(atoms) > 5:
        return None
    gaps = [sp.Symbol(f"_t{i}", nonnegative=True) for i in range(len(atoms))]
    anchor = sp.Symbol("_a", real=True)
    cases: list[dict[sp.Basic, sp.Expr]] = []
    for order in itertools.permutations(range(len(atoms))):
        values: dict[int, sp.Expr] = {order[0]: anchor}
        for j in range(1, len(order)):
            values[order[j]] = values[order[j - 1]] + gaps[j]
        if has_zero:
            shift = values[len(atoms) - 1]  # the 0 atom is last in ``atoms``
            values = {i: v - shift for i, v in values.items()}
        cases.append({atoms[i]: v for i, v in values.items() if atoms[i] != 0})
    return cases


def _prove_identity(os_: sp.Expr, cs: sp.Expr, conditions: tuple[str, ...],
                    deadline: float = math.inf) -> "Proof | None":
    if _is_zero(os_ - cs, deadline):
        return Proof("verified", "symbolic", conditions, "original - rewrite simplifies to 0")
    if _all_real(os_) and _all_real(cs) and _step(deadline) > 0:
        diff = _bounded(_step(deadline), lambda: sp.simplify(sp.exp(os_) - sp.exp(cs)))
        if diff is not None and (diff == 0 or diff.is_zero is True):
            return Proof("verified", "exp-injective", conditions,
                         "exp(original) - exp(rewrite) simplifies to 0 and both sides are real")
    for sign in (1, -1):
        if _nonnegative_form(sign * os_) and _nonnegative_form(sign * cs) and _step(deadline) > 0:
            if _is_zero(sp.expand(os_**2 - cs**2), deadline):
                return Proof("verified", "square-injective", conditions,
                             "both sides have the same sign by their form (roots, absolute values) "
                             "and equal squares, wherever they are real")
    return None


def _nonnegative_form(e: sp.Expr) -> bool:
    """A product of square roots, absolute values, even powers and positive numbers."""
    for f in sp.Mul.make_args(e):
        if f.is_number:
            if not (f.is_positive is True):
                return False
        elif isinstance(f, sp.Abs):
            continue
        elif isinstance(f, sp.Pow) and f.exp == sp.Rational(1, 2):
            continue
        elif isinstance(f, sp.Pow) and f.exp.is_Integer and f.exp % 2 == 0:
            continue
        elif f.is_nonnegative is True:
            continue
        else:
            return False
    return True


def _prove_atan_difference(os_: sp.Expr, cs: sp.Expr, bounds: Mapping[str, tuple[float, float]],
                           conditions: tuple[str, ...], deadline: float) -> "Proof | None":
    """``atan(a) - atan(b) == atan(w)`` from ``tan`` of both sides and ``1 + a*b > 0``.

    ``tan(atan a - atan b) = (a - b)/(1 + a b)``; when ``1 + a b > 0`` the difference
    lies in ``(-pi/2, pi/2)``, where ``tan`` is injective, so equal tangents mean
    equal values. Positivity is decided by ``solveset`` over the range (one
    variable) — no sampling.
    """
    if not isinstance(cs, sp.atan):
        return None
    terms = sp.Add.make_args(os_)
    if len(terms) != 2:
        return None
    pos = [t for t in terms if isinstance(t, sp.atan)]
    neg = [-t for t in terms if isinstance(-t, sp.atan)]
    if len(pos) != 1 or len(neg) != 1:
        return None
    a, b = pos[0].args[0], neg[0].args[0]
    w = cs.args[0]
    if not _is_zero(w - (a - b) / (1 + a * b), deadline):
        return None
    guard = 1 + a * b
    syms = sorted(guard.free_symbols, key=lambda z: z.name)
    if len(syms) != 1 or _step(deadline) <= 0:
        return None
    v = syms[0]
    lo, hi = bounds[v.name]
    bad = _bounded(_step(deadline), lambda: sp.solveset(guard <= 0, v, sp.Interval(lo, hi)))
    if bad is not sp.S.EmptySet:
        return None
    return Proof("verified", "tan-injective", conditions + (f"1 + ({a})*({b}) > 0",),
                 "tan of both sides is (a - b)/(1 + a*b) and 1 + a*b > 0 on the range "
                 "(solveset), so both lie in (-pi/2, pi/2) where tan is injective")


def _prove(orig: Node, cand: Node, bounds: Mapping[str, tuple[float, float]],
           deadline: float = math.inf) -> "Proof | None":
    """A proof that ``cand`` equals ``orig`` on the ranges, or None."""
    from ..core.ir import Expr
    from ..verify.equal import check_equal

    try:
        os_, cs = to_sympy(orig), to_sympy(cand)
    except Exception:  # noqa: BLE001
        return None
    if not _caller_budget_active() and _step(deadline, 6.0) > 0:
        verdict = check_equal(Expr(os_), Expr(cs), budget=_step(deadline, 6.0))
        if verdict.verified:
            return Proof("verified", "check_equal", (), verdict.detail)
    facts, conditions = _assumptions(bounds, orig)
    try:
        oa, ca = to_sympy(orig, facts), to_sympy(cand, facts)
    except Exception:  # noqa: BLE001
        return None
    proof = _prove_identity(oa, ca, conditions, deadline)
    if proof is not None:
        return proof
    atan = _prove_atan_difference(oa, ca, bounds, conditions, deadline)
    if atan is not None:
        return atan
    cases = _case_substitutions(ca)
    if cases is not None:
        branches = []
        for case in cases:
            p = _prove_identity(oa.subs(case), ca.subs(case), conditions, deadline)
            if p is None:
                return None
            branches.append(p)
        return Proof("verified", "case-split", conditions,
                     f"proved in each of {len(branches)} orderings of the max/min arguments")
    return None


def _prove_candidate(n: Node, c: "_Cand", bounds: Mapping[str, tuple[float, float]], deadline: float,
                     depth: int = 0) -> "Proof | None":
    """Prove ``c`` equal to ``n``: directly, or by congruence along its rewrite chain.

    Congruence: ``c`` is its parent with one sub-formula ``T`` replaced by ``R``. If
    the parent equals the original wherever the original is real, and ``R`` equals
    ``T`` wherever ``T`` is real, then ``c`` equals the original wherever the
    original is real — a real evaluation has real values at every sub-formula, so
    ``T`` is real there. The local claim is far smaller than the whole one, which is
    what lets SymPy close it.
    """
    if c.proof is not None:
        return c.proof
    whole = _prove(n, c.node, bounds, deadline)
    if whole is not None or c.parent is None or c.step is None or depth > 4:
        return whole
    upper = _prove_candidate(n, c.parent, bounds, deadline, depth + 1)
    if upper is None:
        return None
    c.parent.proof = upper
    target, repl = c.step
    local = _prove(target, repl, bounds, deadline)
    if local is None:
        return None
    return Proof("verified", "congruence", tuple(dict.fromkeys(upper.conditions + local.conditions)),
                 f"{target} = {repl} ({local.method}) inside a formula proved equal to the original "
                 f"({upper.method})", upper.exact and local.exact,
                 tuple(b for b in (upper, local) if b.method != "original"))


def _disagrees(orig: Node, cand: Node, points: Sequence[Mapping[str, float]]) -> "dict[str, str] | None":
    """An input in the ranges where the two formulas have provably different values.

    Disjoint interval enclosures at 256 bits are a proof of difference. Used as a
    guard before any symbolic proof is trusted (and as the witness when it fails).
    """
    for p in points:
        try:
            a = _iv_eval(orig, p, 256)[orig]
            b = _iv_eval(cand, p, 256)[cand]
        except (_Invalid, _Unresolved, _OutOfReach, ZeroDivisionError, ValueError):
            continue
        except Exception:  # noqa: BLE001
            continue
        (al, ah), (bl, bh) = _ends(a), _ends(b)
        if ah < bl or bh < al:
            return {k: repr(v) for k, v in p.items()}
    return None


# --------------------------------------------------------------------------
# Candidates and the result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A proved rewrite with its measured accuracy (bits on the TEST sample).

    ``expression`` is the formula in evaluation order (``to_numpy()`` evaluates it
    bit-for-bit as measured); ``rules`` the rewrites that produced it;
    ``conditions`` the domain facts its proof needs; ``proof`` the proof.
    """

    expression: str
    rules: tuple[str, ...]
    conditions: tuple[str, ...]
    proof: Proof
    mean_bits: float
    max_bits: float
    cost: int
    node: Node = field(repr=False, compare=False)
    dtype: str = "float64"
    variables: tuple[str, ...] = ()

    def to_python(self, name: str = "f") -> str:
        """Scalar Python using ``math`` (libm may differ from NumPy by an ulp per call)."""
        return _emit(self.node, name, "python", _format(self.dtype), self.variables)

    def to_numpy(self, name: str = "f") -> str:
        """Vectorised NumPy — evaluates exactly what was measured."""
        return _emit(self.node, name, "numpy", _format(self.dtype), self.variables)

    def to_c(self, name: str = "f") -> str:
        """C99 using ``<math.h>`` (``double``, or ``float`` with the ``f`` functions)."""
        return _emit(self.node, name, "c", _format(self.dtype), self.variables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression, "rules": list(self.rules),
            "conditions": list(self.conditions), "proof": self.proof.to_dict(),
            "mean_bits": round(self.mean_bits, 3), "max_bits": round(self.max_bits, 3),
            "cost": self.cost, "python": self.to_python(), "numpy": self.to_numpy(),
        }


@dataclass(frozen=True)
class Improvement:
    """The outcome of ``improve``: a proved, measured rewrite — or the statement that none helps.

    ``before_*`` and every candidate's bits are measured on the same TEST sample,
    drawn independently of the TRAIN sample the search and the regime split used.
    ``best`` is set only when it gains at least ``min_gain`` mean bits; otherwise
    ``improved`` is False and ``message`` says why. ``candidates`` is the Pareto
    front (fewer bits of error vs. lower cost) of the proved rewrites.
    ``witness`` shows the original's worst test input, with the best rewrite's value.
    """

    expression: str
    dtype: str
    ranges: dict[str, tuple[float, float]]
    improved: bool
    best: "Candidate | None"
    candidates: tuple[Candidate, ...]
    before_mean_bits: float
    before_max_bits: float
    samples: int
    message: str
    witness: "dict[str, str] | None"
    stats: dict[str, int]

    @property
    def gain_bits(self) -> float:
        return self.before_mean_bits - self.best.mean_bits if self.best else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "stability_improvement",
            "expression": self.expression,
            "dtype": self.dtype,
            "ranges": {k: [repr(a), repr(b)] for k, (a, b) in self.ranges.items()},
            "improved": self.improved,
            "best": self.best.to_dict() if self.best else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "before_mean_bits": round(self.before_mean_bits, 3),
            "before_max_bits": round(self.before_max_bits, 3),
            "samples": self.samples,
            "message": self.message,
            "witness": self.witness,
            "stats": dict(self.stats),
        }


@dataclass
class _Cand:
    node: Node
    rules: tuple[str, ...]
    errors: np.ndarray  # train bits
    special: "np.ndarray | None" = None  # bits at the special points (range ends, ...)
    allowed: "np.ndarray | None" = None  # train points where it may be used (series)
    proof: "Proof | None" = None
    region: "tuple[str, float, float] | None" = None  # series: (var, lo, hi) certified
    parent: "_Cand | None" = None  # the candidate this one rewrote ...
    step: "tuple[Node, Node] | None" = None  # ... by replacing step[0] with step[1]

    @property
    def mean(self) -> float:
        return float(np.mean(self.errors))


# --------------------------------------------------------------------------
# Regimes (as in Herbie): pick the best candidate per slice of one variable
# --------------------------------------------------------------------------


def _segments(errors: np.ndarray, max_segments: int, penalty: float) -> tuple[float, list[tuple[int, int, int]]]:
    """Optimal split of sorted points into <= max_segments runs, one candidate each.

    ``errors`` is (candidates, points) in sorted order (``inf`` = not allowed).
    Minimises total bits + ``penalty`` per extra segment; returns (cost, [(start,
    end, candidate)]). Dynamic programming over prefix sums.
    """
    c, n = errors.shape
    prefix = np.concatenate([np.zeros((c, 1)), np.cumsum(errors, axis=1)], axis=1)
    best = np.full((max_segments + 1, n + 1), np.inf)
    choice = np.zeros((max_segments + 1, n + 1, 2), dtype=np.int64)
    best[0, 0] = 0.0
    for s in range(1, max_segments + 1):
        for end in range(1, n + 1):
            starts = np.arange(end)
            seg = prefix[:, end][:, None] - prefix[:, starts]  # (c, end)
            total = best[s - 1, starts][None, :] + seg
            flat = int(np.argmin(total))
            ci, si = divmod(flat, end)
            best[s, end] = total[ci, si] + (penalty if s > 1 else 0.0)
            choice[s, end] = (si, ci)
    s_best = int(np.argmin(best[1:, n])) + 1
    cost = float(best[s_best, n])
    out: list[tuple[int, int, int]] = []
    end = n
    for s in range(s_best, 0, -1):
        start, ci = (int(v) for v in choice[s, end])
        out.append((start, end, ci))
        end = start
    return cost, out[::-1]


def _regime(pool: list[_Cand], train: _Sample, orig: Node, fmt: _Format,
            bounds: Mapping[str, tuple[float, float]], deadline: float,
            max_segments: int = 4) -> "tuple[Node, list[_Cand]] | None":
    """The best piecewise combination of candidates, split on one variable."""
    n = len(train)
    if n < 8 or len(pool) < 2:
        return None
    penalty = 0.25 * n  # a split must buy a quarter of a bit on average
    # Not allowed = a cost no allowed run can reach (finite, so prefix sums stay finite).
    forbidden = 1e7
    errs = np.array([np.where(c.allowed if c.allowed is not None else True, c.errors, forbidden) for c in pool])
    best_cost, best_plan, best_var = math.inf, None, None
    for v in orig.variables:
        order = np.argsort(train.env[v], kind="stable")
        cost, plan = _segments(errs[:, order], max_segments, penalty)
        if cost < best_cost:
            best_cost, best_plan, best_var = cost, (order, plan), v
    if best_plan is None or best_var is None or best_cost >= forbidden:
        return None
    order, plan = best_plan
    xs = train.env[best_var][order]
    thresholds: list[float] = []
    for (s0, e0, c0), (s1, e1, c1) in zip(plan, plan[1:]):
        left, right = float(xs[e0 - 1]), float(xs[s1])
        thresholds.append(_refine_threshold(pool[c0], pool[c1], best_var, left, right, train, orig, fmt, deadline))
    lo_v, hi_v = bounds[best_var]
    lowers, uppers = [lo_v] + thresholds, thresholds + [hi_v]
    pieces: list[tuple[float, _Cand]] = []
    for lo_i, hi_i, (_, _, ci) in zip(lowers, uppers, plan):
        for upper, c in _clip(pool[ci], best_var, lo_i, hi_i, pool[0], bounds, fmt):
            if pieces and pieces[-1][1] is c:
                pieces[-1] = (upper, c)
            else:
                pieces.append((upper, c))
    if len(pieces) == 1:
        only = pieces[0][1]
        return None if only.node == orig else (only.node, [only])
    node = pieces[-1][1].node
    for upper, c in reversed(pieces[:-1]):
        cond = _op("<=", _var(best_var), _const(sp.Rational(repr(upper))))
        node = _op("if", cond, c.node, node)
    return node, [c for _, c in pieces]


def _clip(c: _Cand, var: str, lo: float, hi: float, original: _Cand,
          bounds: Mapping[str, tuple[float, float]], fmt: _Format) -> list[tuple[float, _Cand]]:
    """Segment ``(lo, hi]`` of the split variable for ``c``, cut to its certified interval.

    A Taylor branch is proved only on its interval; the sample may simply have had
    no point outside it (bit-uniform sampling of [-1, 1] puts ~0.4% of the points
    above 1/16). So the interval is enforced on the FORMULA, not on the sample:
    outside it, the original formula takes over. Returns ``(upper bound, candidate)``
    pieces in order; a piece covers the values above the previous upper bound.
    """
    if c.region is None:
        return [(hi, c)]
    rvar, a, b = c.region
    if rvar != var:
        r_lo, r_hi = bounds[rvar]
        return [(hi, c)] if a <= r_lo and r_hi <= b else [(hi, original)]
    out: list[tuple[float, _Cand]] = []
    below_a = float(np.nextafter(fmt.dtype(a), fmt.dtype(-np.inf)))
    if lo < a:
        out.append((min(hi, below_a), original))
    if max(lo, a) <= min(hi, b):
        out.append((min(hi, b), c))
    if hi > b:
        out.append((hi, original))
    return out


def _refine_threshold(left: _Cand, right: _Cand, var: str, lo: float, hi: float, train: _Sample,
                      orig: Node, fmt: _Format, deadline: float, steps: int = 10) -> float:
    """A split point between two sorted sample values (bisection on float ordinals).

    Moves the boundary to where the right-hand candidate starts being at least as
    accurate, using exact values at the probe points (other variables fixed at the
    left point's values). Clamped to the certified interval of a Taylor branch.
    """
    a, b = (int(o) for o in _ordinals(np.array([lo, hi], dtype=fmt.dtype), fmt))
    i_left = int(np.where(train.env[var] == lo)[0][0])
    others = {w: float(train.env[w][i_left]) for w in orig.variables if w != var}
    for _ in range(steps):
        if b - a <= 1 or time.perf_counter() > deadline:
            break
        mid = (a + b) // 2
        x = float(_from_ordinals(np.array([mid], dtype=np.int64), fmt)[0])
        point = {**others, var: x}
        t = _truth(orig, point, fmt)
        if isinstance(t, str):
            break
        env = {k: np.array([v], dtype=fmt.dtype) for k, v in point.items()}
        el = _bits_of_error(_eval_float(left.node, env, fmt), np.array([t.value]), fmt)[0]
        er = _bits_of_error(_eval_float(right.node, env, fmt), np.array([t.value]), fmt)[0]
        if er <= el:
            b = mid
        else:
            a = mid
    t_ord = a
    t_val = float(_from_ordinals(np.array([t_ord], dtype=np.int64), fmt)[0])
    for c, side in ((left, "left"), (right, "right")):
        if c.region is not None and c.region[0] == var:
            lo_c, hi_c = c.region[1], c.region[2]
            if side == "left":
                t_val = min(t_val, hi_c)
            else:
                t_val = max(t_val, float(np.nextafter(fmt.dtype(lo_c), fmt.dtype(-np.inf))))
    return t_val


# --------------------------------------------------------------------------
# improve
# --------------------------------------------------------------------------


def _truths_for(cand: Node, sample: _Sample, idx: Sequence[int], fmt: _Format) -> _Sample:
    """Exact node values of ``cand`` at the chosen sample points (for localisation)."""
    env: dict[str, list[float]] = {v: [] for v in sample.env}
    truths, exact = [], []
    for i in idx:
        p = {v: float(sample.env[v][i]) for v in sample.env}
        t = _truth(cand, p, fmt)
        if isinstance(t, str):
            continue
        for v in env:
            env[v].append(p[v])
        truths.append(t)
        exact.append(t.value)
    return _Sample({v: np.array(x, dtype=fmt.dtype) for v, x in env.items()},
                   np.array(exact, dtype=fmt.dtype), truths, {}, len(idx))


def _culprit_nodes(cand: Node, sample: _Sample, fmt: _Format, top: int = 4, small: int = 12) -> list[Node]:
    """Where to apply rewrites: every operation of a small formula, else the top culprits.

    A small formula is searched everywhere because a failure confined to the edge of
    the range (an overflow at x = 700) is invisible in a subsample's local errors.
    """
    ops = [m for m in cand.walk() if m.op not in ("var", "const") and m.op not in _COMPARE]
    if len(ops) <= small:
        return ops[::-1]
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(sample), size=min(48, len(sample)), replace=False)
    sub = _truths_for(cand, sample, sorted(int(i) for i in idx), fmt)
    if len(sub) == 0:
        return [cand]
    errs = _local_errors(cand, sub, fmt)
    ranked = sorted(errs.items(), key=lambda kv: -float(np.mean(kv[1])))
    nodes = [m for m, e in ranked[:top] if float(np.max(e)) > 1.0]
    if cand not in nodes:
        nodes.append(cand)
    return nodes


def _rank(c: _Cand) -> tuple[float, float]:
    """Search order: mean bits, with the worst special point as a tie-breaker-plus.

    A candidate that removes an overflow at the edge of the range has (nearly) the
    same mean under bit-uniform sampling; the worst-case term keeps it in play.
    """
    worst = float(np.max(c.special)) if c.special is not None and len(c.special) else 0.0
    return (c.mean + worst / 64.0, c.node.cost)


def _check_min_gain(min_gain: object) -> float:
    if not isinstance(min_gain, (int, float)) or isinstance(min_gain, bool) or not min_gain >= 0:
        raise DomainError(f"min_gain must be a non-negative number of bits, not {min_gain!r}")
    return float(min_gain)


def improve(
    expr: "str | Node | sp.Basic",
    ranges: Mapping[str, Any],
    *,
    dtype: str = "float64",
    samples: int = 256,
    seed: int = 0,
    budget: "float | None" = None,
    min_gain: float = 1.0,
    rounds: int = 3,
    series: bool = True,
) -> Improvement:
    """Find a rewrite of ``expr`` that is proved equal and measured more accurate.

    Search (as in Herbie, with proofs): rank sub-expressions by local error, apply
    the targeted rules and SymPy's algebra there, iterate ``rounds`` times on the
    best candidates; add Taylor polynomials (only on the interval where their
    truncation error is proved below the unit roundoff, ``series=True``); combine
    candidates per slice of one variable (regimes). Every candidate that is kept
    has a ``Proof``; candidates whose proof fails are dropped, and one that is
    PROVABLY different somewhere in the ranges is dropped with that input counted.

    The search uses a TRAIN sample (``seed``); every number reported is measured on
    an independent TEST sample (``seed + 1``) of ``samples`` points. ``best`` is set
    only if it gains at least ``min_gain`` bits on average on the test sample.

        >>> imp = improve("sqrt(x + 1) - sqrt(x)", {"x": (1e10, 1e15)}, samples=64)
        >>> imp.improved, imp.best.expression, imp.best.proof.method
        (True, '1 / (sqrt(x + 1) + sqrt(x))', 'check_equal')
    """
    fmt = _format(dtype)
    samples = _check_samples(samples)
    seconds = _check_budget(budget)
    min_gain = _check_min_gain(min_gain)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or not 0 <= rounds <= 5:
        raise DomainError(f"rounds must be an integer from 0 to 5, not {rounds!r}")
    from . import _rewrites as rw

    n = _as_formula(expr)
    bounds = _check_ranges(n, ranges, fmt)
    start = time.perf_counter()
    deadline = start + seconds

    def time_left() -> bool:
        return time.perf_counter() < deadline

    train = _sample(n, bounds, samples, seed, fmt, deadline)
    if len(train) == 0:
        raise StabilityDomainError(
            f"stability: none of {train.attempted} sampled inputs gives {n} a real, "
            f"decidable exact value ({_skips_text(train.skipped)}) — check the ranges"
        )
    test = _sample(n, bounds, samples, seed + 1, fmt, deadline)
    special = _sample(n, bounds, 0, seed, fmt, deadline, _special_points(n, bounds, fmt, seed))
    stats = {"generated": 0, "measured": 0, "proved": 0, "unproved": 0, "refuted": 0,
             "taylor_certified": 0, "taylor_uncertified": 0}

    def train_errors(node: Node) -> np.ndarray:
        return _bits_of_error(_eval_on(node, train, fmt), train.exact, fmt)

    def special_errors(node: Node) -> np.ndarray:
        if len(special) == 0:
            return np.zeros(1)
        return _bits_of_error(_eval_on(node, special, fmt), special.exact, fmt)

    original = _Cand(n, (), train_errors(n), special_errors(n),
                     proof=Proof("verified", "original", (), "the input itself"))
    pool: dict[Node, _Cand] = {n: original}

    # Each phase gets its share of what is left after sampling, so a slow proof
    # cannot starve the Taylor search and the regimes (measured: it did).
    budget_left = max(0.0, deadline - time.perf_counter())
    search_end = time.perf_counter() + 0.35 * budget_left
    proof_end = time.perf_counter() + 0.60 * budget_left
    taylor_end = time.perf_counter() + 0.85 * budget_left

    def before(t: float) -> Callable[[], bool]:
        return lambda: time.perf_counter() < t

    def symbolic(fn: Callable[[], Any]) -> Any:
        return _bounded(min(3.0, max(0.05, deadline - time.perf_counter())), fn)

    in_search = before(search_end)
    for _ in range(rounds):
        if not in_search():
            break
        parents = sorted(pool.values(), key=_rank)[:3]
        if original not in parents:
            parents.append(original)
        for parent in parents:
            if not in_search():
                break
            for target in _culprit_nodes(parent.node, train, fmt):
                found = rw.local_rewrites(target) + rw.sympy_rewrites(target, symbolic)
                for repl, rule in found:
                    stats["generated"] += 1
                    node = parent.node.replace(target, repl)
                    if node in pool:
                        continue
                    try:
                        errs = train_errors(node)
                    except UnsupportedFormError:
                        continue
                    stats["measured"] += 1
                    pool[node] = _Cand(node, parent.rules + (rule,), errs, special_errors(node),
                                       parent=parent, step=(target, repl))

    # Which candidates deserve a proof: better on average, or clearly better somewhere.
    base = original.errors
    worst = float(np.max(original.special)) if original.special is not None else 0.0
    useful = [c for c in pool.values() if c is not original and (
        c.mean < original.mean - 0.1 or float(np.mean(c.errors < base - 1.0)) >= 0.05
        or (c.special is not None and float(np.max(c.special)) <= worst - _MAX_GAIN))]
    useful.sort(key=lambda c: c.mean)
    probe_rng = np.random.default_rng(seed + 3)
    probe_idx = probe_rng.choice(len(train), size=min(24, len(train)), replace=False)
    probes = [{v: float(train.env[v][i]) for v in n.variables} for i in probe_idx]
    proven: list[_Cand] = []
    in_proofs = before(proof_end)
    for c in useful[:16]:
        if not in_proofs():
            break
        if _disagrees(n, c.node, probes) is not None:
            stats["refuted"] += 1
            continue
        proof = _prove_candidate(n, c, bounds, proof_end)
        if proof is None:
            stats["unproved"] += 1
            continue
        stats["proved"] += 1
        c.proof = proof
        proven.append(c)

    taylor: list[_Cand] = []
    if series and time_left():
        taylor = _taylor_pool(n, [original] + proven[:2], train, bounds, fmt, stats, symbolic,
                              before(taylor_end), taylor_end)

    finalists = [original] + proven
    regime = None
    if time_left():
        found_regime = _regime([original] + proven + taylor, train, n, fmt, bounds, deadline)
        if found_regime is not None and len(found_regime[1]) == 1:
            only = found_regime[1][0]
            if only not in finalists:
                finalists.append(only)
        elif found_regime is not None:
            node, used = found_regime
            branches = tuple(c.proof for c in used if c.proof is not None)
            rules = tuple(dict.fromkeys(r for c in used for r in c.rules)) + ("regimes",)
            # A Taylor branch's interval is a fact about that branch (enforced by _clip),
            # not a condition on the whole formula; identities' domain facts are global.
            conds = tuple(dict.fromkeys(cnd for c in used if c.proof and c.proof.exact
                                        for cnd in c.proof.conditions))
            exact = all(b.exact for b in branches)
            regime = _Cand(node, rules, train_errors(node), special_errors(node),
                           proof=Proof("verified", "piecewise", conds,
                                       f"each of the {len(used)} branches is proved; split on "
                                       f"{node.args[0].args[0].name}", exact, branches))
            finalists.append(regime)

    return _finish(n, bounds, fmt, test, special, original, finalists, min_gain, stats)


def _try_from_sympy(e: Any) -> "Node | None":
    if e is None:
        return None
    try:
        return from_sympy(e)
    except Exception:  # noqa: BLE001 - no float spelling: no candidate
        return None


def _taylor_pool(n: Node, parents: list[_Cand], train: _Sample, bounds: Mapping[str, tuple[float, float]],
                 fmt: _Format, stats: dict[str, int], symbolic: Callable[[Callable[[], Any]], Any],
                 time_left: Callable[[], bool], deadline: float) -> list[_Cand]:
    """Taylor branches of one-variable sub-formulas, usable only where certified.

    Each culprit sub-formula of one variable gets its truncated series with the
    radius on which the truncation is PROVED below the unit roundoff; when several
    culprits share the variable (``(x - sin x) / (x - tan x)``), they are also
    replaced together on the common radius. Each candidate also comes in an exactly
    simplified form (``(x*x/2) / (x*x)`` -> ``1/2``), proved equal to the substituted one.
    """
    from . import _rewrites as rw

    out: list[_Cand] = []
    for parent in parents:
        options: dict[Node, list[tuple[Node, str, int, float, float]]] = {}
        for target in _culprit_nodes(parent.node, train, fmt, top=3):
            if not time_left():
                return out
            if len(target.variables) != 1 or target.op in ("var", "const"):
                continue
            var = target.variables[0]
            lo, hi = bounds[var]
            for poly_node, label, poly, order, k in rw.taylor_candidates(target, var, symbolic):
                if not time_left():
                    return out
                r_pos = rw.certified_radius(target, var, poly, order, k, hi, fmt.unit_roundoff, time_left) if hi > 0 else 0.0
                r_neg = rw.certified_radius(target, var, poly, order, k, lo, fmt.unit_roundoff, time_left) if lo < 0 else 0.0
                if r_pos == 0.0 and r_neg == 0.0:
                    stats["taylor_uncertified"] += 1
                    continue
                stats["taylor_certified"] += 1
                options.setdefault(target, []).append((poly_node, label, order, r_neg, r_pos))
                _add_taylor(out, n, parent, {target: (poly_node, label, order)}, var, r_neg, r_pos,
                            train, bounds, fmt, symbolic, time_left, deadline)
        # Several culprits of the same variable, replaced together on the common radius.
        by_var: dict[str, list[Node]] = {}
        for t in options:
            by_var.setdefault(t.variables[0], []).append(t)
        for var, targets in by_var.items():
            targets = [t for t in targets if not any(t is not u and t in u.walk() for u in targets)]
            if len(targets) < 2 or not time_left():
                continue
            depth = max(len(options[t]) for t in targets)
            for j in range(depth):
                picks = {t: options[t][min(j, len(options[t]) - 1)] for t in targets}
                r_neg = max(p[3] for p in picks.values())
                r_pos = min(p[4] for p in picks.values())
                if r_neg == 0.0 and r_pos == 0.0:
                    continue
                _add_taylor(out, n, parent, {t: (p[0], p[1], p[2]) for t, p in picks.items()}, var,
                            r_neg, r_pos, train, bounds, fmt, symbolic, time_left, deadline)
    return out


def _add_taylor(out: list[_Cand], n: Node, parent: _Cand, picks: Mapping[Node, tuple[Node, str, int]],
                var: str, r_neg: float, r_pos: float, train: _Sample,
                bounds: Mapping[str, tuple[float, float]], fmt: _Format,
                symbolic: Callable[[Callable[[], Any]], Any], time_left: Callable[[], bool],
                deadline: float) -> None:
    node = parent.node
    for target, (poly_node, _, _) in picks.items():
        node = node.replace(target, poly_node)
    xs = train.env[var]
    allowed = (xs >= r_neg) & (xs <= r_pos) & (xs != 0)
    pieces = "; ".join(f"{t} replaced by its {label.lower()} (order-{order} derivative bounded)"
                       for t, (_, label, order) in picks.items())
    condition = f"{r_neg!r} <= {var} <= {r_pos!r}"
    proof = Proof(
        "verified", "taylor-remainder", (condition,),
        f"{pieces}; relative truncation error <= 2^-{fmt.prec} proved for {var} in "
        f"[{r_neg!r}, {r_pos!r}] (Lagrange remainder, interval bounds)",
        exact=False,
    )
    if parent.proof is not None and parent.proof.method != "original":
        proof = Proof("verified", "taylor-remainder", parent.proof.conditions + proof.conditions,
                      proof.detail + f"; rest of the formula: {parent.proof.method}",
                      exact=False, branches=(parent.proof,))
    labels = tuple(dict.fromkeys(label for _, label, _ in picks.values()))
    errs = _bits_of_error(_eval_on(node, train, fmt), train.exact, fmt)
    out.append(_Cand(node, parent.rules + labels, errs, allowed=allowed, proof=proof,
                     region=(var, r_neg, r_pos)))
    # The polynomial often cancels against the rest exactly: (x*x/2 - ...)/(x*x)
    # -> 1/2 - x*x/24, which no longer underflows. Proved equal to the substituted
    # form, so the truncation bound carries over unchanged.
    alt = _try_from_sympy(symbolic(lambda: sp.cancel(to_sympy(node))))
    if alt is not None and alt != node and time_left():
        link = _prove(node, alt, bounds, deadline)
        if link is not None:
            alt_errs = _bits_of_error(_eval_on(alt, train, fmt), train.exact, fmt)
            alt_proof = Proof("verified", "taylor-remainder", tuple(dict.fromkeys(proof.conditions + link.conditions)),
                              proof.detail + f"; then simplified exactly ({link.method})",
                              exact=False, branches=(proof, link))
            out.append(_Cand(alt, parent.rules + labels + ("cancel",), alt_errs, allowed=allowed,
                             proof=alt_proof, region=(var, r_neg, r_pos)))


#: A rewrite that leaves the mean alone qualifies if it removes at least this many
#: bits from the worst case (test points and special points together) — typically
#: an overflow to inf at the edge of the range, which bit-uniform sampling rarely
#: hits (only ~0.05% of the floats in [-1000, 1000] exceed 709, where exp overflows).
_MAX_GAIN = 8.0


def _finish(n: Node, bounds: dict[str, tuple[float, float]], fmt: _Format, test: _Sample, special: _Sample,
            original: _Cand, finalists: list[_Cand], min_gain: float, stats: dict[str, int]) -> Improvement:
    if len(test) == 0:
        raise StabilityDomainError("stability: the test sample has no point with a real exact value")

    def measure(node: Node) -> tuple[np.ndarray, float]:
        bits = _bits_of_error(_eval_on(node, test, fmt), test.exact, fmt)
        worst = float(np.max(bits))
        if len(special):
            sb = _bits_of_error(_eval_on(node, special, fmt), special.exact, fmt)
            worst = max(worst, float(np.max(sb)))
        return bits, worst

    before, before_max = measure(n)
    before_mean = float(np.mean(before))
    measured: list[Candidate] = []
    for c in finalists:
        if c is original or c.proof is None:
            continue
        bits, worst = measure(c.node)
        measured.append(Candidate(
            expression=str(c.node), rules=c.rules, conditions=c.proof.conditions, proof=c.proof,
            mean_bits=float(np.mean(bits)), max_bits=worst, cost=c.node.cost,
            node=c.node, dtype=fmt.name, variables=n.variables,
        ))
    front = _pareto(measured)
    by_mean = [c for c in measured if c.mean_bits <= before_mean - min_gain]
    # Fixing the worst case never excuses a worse average (run 1 on the dev set: a
    # 0.01-bit worse lerp was proposed for its exact endpoints).
    by_max = [c for c in measured if c.max_bits <= before_max - _MAX_GAIN and c.mean_bits <= before_mean]
    best = _prefer_exact(by_mean, lambda c: c.mean_bits) or _prefer_exact(by_max, lambda c: c.max_bits)
    if best is not None:
        witness = _improvement_witness(n, best.node, test, special, fmt)
        message = (f"{best.expression}: {before_mean:.1f} -> {best.mean_bits:.1f} bits of error on "
                   f"average ({before_max:.1f} -> {best.max_bits:.1f} at worst), proved by {best.proof.method}"
                   + (f" for {', '.join(best.conditions)}" if best.conditions else ""))
        return Improvement(str(n), fmt.name, bounds, True, best, tuple(front), before_mean, before_max,
                           len(test), message, witness, stats)
    if before_mean <= 1.0 and before_max <= _MAX_GAIN:
        message = (f"already accurate: {before_mean:.2f} bits of error on average "
                   f"({before_max:.1f} at worst); no rewrite proposed")
    elif not measured:
        message = (f"no rewrite found that is both proved equal and more accurate "
                   f"({before_mean:.1f} bits of error on average, {before_max:.1f} at worst, remain)")
    else:
        top = min(measured, key=lambda c: c.mean_bits)
        message = (f"no rewrite gains {min_gain:g} bit(s) on average or {_MAX_GAIN:g} at worst: the best "
                   f"proved one has {top.mean_bits:.1f} bits of error against {before_mean:.1f}")
    return Improvement(str(n), fmt.name, bounds, False, None, tuple(front), before_mean, before_max,
                       len(test), message, None, stats)


def _prefer_exact(cands: list[Candidate], score: Callable[[Candidate], float]) -> "Candidate | None":
    """Among the candidates within half a bit of the most accurate: an exact identity
    first, then the cheapest.

    Half a bit is below what the sample can resolve, so those candidates are equally
    accurate; an identity holds on every input (a Taylor branch only on its proved
    interval), and the simpler formula is the better advice.
    """
    if not cands:
        return None
    top = min(score(c) for c in cands)
    near = [c for c in cands if score(c) <= top + 0.5]
    return min(near, key=lambda c: (not c.proof.exact, c.cost, score(c)))


def _improvement_witness(n: Node, best: Node, test: _Sample, special: _Sample, fmt: _Format) -> dict[str, str]:
    """The input where the original is worst (test or special points), both values and the exact one."""
    pools = [s for s in (test, special) if len(s)]
    worst_bits, where = -1.0, (test, 0)
    for s in pools:
        bits = _bits_of_error(_eval_on(n, s, fmt), s.exact, fmt)
        i = int(np.argmax(bits))
        if bits[i] > worst_bits:
            worst_bits, where = float(bits[i]), (s, i)
    s, i = where
    env = {v: s.env[v][i:i + 1] for v in s.env}
    return {
        **{v: repr(float(s.env[v][i])) for v in n.variables},
        "original": repr(float(np.atleast_1d(_eval_float(n, env, fmt, size=1))[0])),
        "rewrite": repr(float(np.atleast_1d(_eval_float(best, env, fmt, size=1))[0])),
        "exact": _exact_text(s.truths[i].nodes[n]),
    }


def _pareto(cands: list[Candidate]) -> list[Candidate]:
    front: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.mean_bits, c.cost)):
        if not front or c.cost < front[-1].cost:
            front.append(c)
    return front


# --------------------------------------------------------------------------
# Code emission
# --------------------------------------------------------------------------

_PY_NAMES = {
    "sqrt": "math.sqrt", "cbrt": "math.cbrt", "exp": "math.exp", "log": "math.log", "sin": "math.sin",
    "cos": "math.cos", "tan": "math.tan", "asin": "math.asin", "acos": "math.acos", "atan": "math.atan",
    "sinh": "math.sinh", "cosh": "math.cosh", "tanh": "math.tanh", "asinh": "math.asinh",
    "acosh": "math.acosh", "atanh": "math.atanh", "log1p": "math.log1p", "expm1": "math.expm1",
    "fabs": "math.fabs", "hypot": "math.hypot", "pow": "math.pow", "max": "max", "min": "min",
    "atan2": "math.atan2",
}
_NP_NAMES = {
    "sqrt": "np.sqrt", "cbrt": "np.cbrt", "exp": "np.exp", "log": "np.log", "sin": "np.sin",
    "cos": "np.cos", "tan": "np.tan", "asin": "np.arcsin", "acos": "np.arccos", "atan": "np.arctan",
    "sinh": "np.sinh", "cosh": "np.cosh", "tanh": "np.tanh", "asinh": "np.arcsinh",
    "acosh": "np.arccosh", "atanh": "np.arctanh", "log1p": "np.log1p", "expm1": "np.expm1",
    "fabs": "np.abs", "hypot": "np.hypot", "pow": "np.power", "max": "np.maximum", "min": "np.minimum",
    "atan2": "np.arctan2",
}
_C_NAMES = {
    "sqrt": "sqrt", "cbrt": "cbrt", "exp": "exp", "log": "log", "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "asinh": "asinh", "acosh": "acosh", "atanh": "atanh", "log1p": "log1p", "expm1": "expm1",
    "fabs": "fabs", "hypot": "hypot", "pow": "pow", "max": "fmax", "min": "fmin", "fma": "fma",
    "atan2": "atan2",
}


def _emit_expr(n: Node, target: str, fmt: _Format) -> str:
    op = n.op
    a = [_emit_expr(x, target, fmt) for x in n.args]
    if op == "var":
        return n.name
    if op == "const":
        v = _const_float(n.value, fmt)
        text = repr(v)
        if target == "numpy":
            return f"np.{fmt.name}({text})"
        if target == "c":
            text = text if ("e" in text or "." in text or "inf" in text) else text + ".0"
            return text + ("f" if fmt.bits == 32 else "")
        return text
    if op == "neg":
        return f"(-{a[0]})"
    if op in ("+", "-", "*", "/"):
        return f"({a[0]} {op} {a[1]})"
    if op in _COMPARE:
        return f"({a[0]} {op} {a[1]})"
    if op == "if":
        if target == "python":
            return f"({a[1]} if {a[0]} else {a[2]})"
        if target == "numpy":
            return f"np.where({a[0]}, {a[1]}, {a[2]})"
        return f"({a[0]} ? {a[1]} : {a[2]})"
    if op == "fma" and target != "c":
        raise UnsupportedFormError("fma has no exact spelling in Python < 3.13 or NumPy; use to_c()")
    names = {"python": _PY_NAMES, "numpy": _NP_NAMES, "c": _C_NAMES}[target]
    fn = names[op]
    if target == "c" and fmt.bits == 32:
        fn += "f"
    return f"{fn}({', '.join(a)})"


def _emit(n: Node, name: str, target: str, fmt: _Format, variables: Sequence[str]) -> str:
    if not name.isidentifier():
        raise DomainError(f"function name must be an identifier, not {name!r}")
    args = list(variables) or list(n.variables)
    body = _emit_expr(n, target, fmt)
    if target == "python":
        return f"import math\n\n\ndef {name}({', '.join(args)}):\n    return {body}\n"
    if target == "numpy":
        return (f"import numpy as np\n\n\ndef {name}({', '.join(args)}):\n"
                f"    with np.errstate(all=\"ignore\"):\n        return {body}\n")
    ctype = "float" if fmt.bits == 32 else "double"
    params = ", ".join(f"{ctype} {v}" for v in args)
    return f"#include <math.h>\n\n{ctype} {name}({params}) {{\n    return {body};\n}}\n"
