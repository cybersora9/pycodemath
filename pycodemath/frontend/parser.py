"""Parser — natural math notation → IR.

A thin layer over ``sympy.parsing``. Accepts standard mathematical
notation (``sin(x)*x + 2^3``) and returns an ``Expr`` from the IR core.
Guarantees round-trip: ``parse(str(parse(s))) == parse(s)``.
"""

from __future__ import annotations

import math
import token as _token
from tokenize import TokenError

import re

import sympy as sp
from sympy import SympifyError
from sympy.parsing.sympy_parser import (
    _token_splittable,
    convert_xor,
    function_exponentiation,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    split_symbols_custom,
    standard_transformations,
)

from ..core.errors import ParseError
from ..core.ir import Expr

# Tokens that make no sense in math notation but open up an eval surface:
# STRING/f-string (a string result would land in sympify with the FULL namespace) and
# OP "." = attribute access (x.__class__ etc.; fractions like 1.5 are NUMBER,
# so a decimal point does not fall in here).
_DENIED_TOKEN_TYPES = frozenset(
    t
    for t in (
        _token.STRING,
        getattr(_token, "FSTRING_START", None),  # Python 3.12+
    )
    if t is not None
)


def _deny_unsafe_tokens(tokens, local_dict, global_dict):
    """Guard transformation: rejects tokens outside math notation."""
    for tok_type, tok_val in tokens:
        if tok_type in _DENIED_TOKEN_TYPES:
            raise ParseError(
                f"text literals are not a mathematical expression: {tok_val!r}"
            )
        if tok_type == _token.OP and tok_val == ".":
            raise ParseError(
                "attribute access (dot) is not mathematical notation"
            )
    return tokens


# letters+trailing digits (v0, x1, q2, R1, t12) stay ONE symbol. Default
# split_symbols would break "q1" into Symbol('q') * <the digit '1'> — and the
# digit half is built via a bare ``Number(...)`` call the whitelist's
# global_dict does not define (only Symbol/Function/Integer/Float/Rational
# are), so this crashed with a leaked ``NameError: name 'Number' is not
# defined`` for EVERY variable name in this extremely common physics/
# engineering convention. (Plain sympy without the whitelist does not crash
# here, but is worse: it silently drops the digit — "q1" reads as "q", "q2"
# as "2*q", so "q1*q2" becomes "2*q**2".) split_symbols_custom keeps the
# documented "unknown multi-letter name splits into a product of single
# letters" behaviour (foo(x) -> f*o*o*x — the price of "2x" notation) for
# pure-letter names; only a trailing digit run is now excluded from splitting.
_LETTER_DIGIT_SUFFIX = re.compile(r"[A-Za-z]+\d+")


def _splittable(symbol: str) -> bool:
    if not _token_splittable(symbol):
        return False
    return not _LETTER_DIGIT_SUFFIX.fullmatch(symbol)


_implicit_multiplication_application = (
    split_symbols_custom(_splittable),
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
)

# Token guard BEFORE everything, then ^ as exponentiation + implicit
# multiplication (2x -> 2*x), as in math notation.
_TRANSFORMS = (
    (_deny_unsafe_tokens,)
    + standard_transformations
    + (convert_xor,)
    + _implicit_multiplication_application
)

# Matrix-literal transforms: the SAME token guard + whitelist namespace (so the
# security boundary is identical — attribute access and text literals are
# rejected, no builtin is reachable), but WITHOUT implicit multiplication or
# ``^``-as-power. A matrix entry is a plain literal (as under the previous bare
# ``sympify``): multi-letter names stay one symbol (``np`` is Symbol('np'), not
# ``n*p``) and ``^`` keeps its Python meaning — parse semantics are unchanged,
# only the eval surface is closed.
_MATRIX_TRANSFORMS = (_deny_unsafe_tokens,) + standard_transformations

# Parser whitelist: the ONLY names that resolve to SymPy objects.
# Everything off the list becomes a Symbol/undefined function (auto_symbol),
# so __import__/eval/open are not reachable at all from the input text.
_ALLOWED_FUNCTIONS = (
    # trigonometry + inverses
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc", "atan2",
    # hyperbolic + inverses
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth",
    # exponential / power
    "exp", "log", "sqrt", "cbrt", "root",
    # piecewise / discrete
    "Abs", "sign", "floor", "ceiling",
    "Min", "Max", "gamma", "factorial", "binomial",
)
_LOCAL_DICT: dict[str, object] = {
    name: getattr(sp, name) for name in _ALLOWED_FUNCTIONS
}
_LOCAL_DICT.update(
    {
        # aliases for common notation (previously captured by Python builtins)
        "ln": sp.log,
        "abs": sp.Abs,
        "min": sp.Min,
        "max": sp.Max,
        # constants
        "pi": sp.pi,
        "E": sp.E,
        "I": sp.I,
        "oo": sp.oo,
    }
)

# global_dict passed EXPLICITLY: without it parse_expr does `from sympy import *`
# and adds Python builtins (including __import__). Here only the constructors
# that the token transformations emit (auto_symbol/auto_number/…), plus
# an empty __builtins__ so eval does not get the real builtins.
_GLOBAL_DICT: dict[str, object] = {
    "Symbol": sp.Symbol,
    "Function": sp.Function,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "__builtins__": {},
}

# --- Evaluation cost guard (DoS) -------------------------------------
# The math whitelist allows exponentiation and factorial/gamma/binomial — and on
# CONCRETE numbers these can build a gigantic int: ``9**9**9`` is
# ~1.2 billion bits, ``factorial(10**8)`` likewise. This is reachable directly from the
# MCP server (``math_eval``), so a single line would hang/OOM the host. SymPy
# computes such things eagerly under ``evaluate=True`` — that is why we FIRST
# parse the same syntax WITHOUT evaluation and with "blind" functions (so the
# guard itself computes nothing), then estimate the cost and refuse in advance.

#: Upper bit limit for a CONCRETE integer produced by evaluation.
#: ~200k bits ≈ 60k digits — generous for real math, yet cuts off DoS.
_MAX_RESULT_BITS = 200_000
#: An exponent/argument with more bits than this WILL surely blow up the result.
_SAFE_EXPONENT_BITS = 64
#: The largest factorial/gamma/binomial argument (n! grows ~n·log2(n)).
_MAX_FACTORIAL_ARG = 10_000
#: Functions whose CONCRETE integer argument alone blows up the result.
_COST_FUNCS = frozenset({"factorial", "gamma", "binomial"})
_LN2 = math.log(2)


class _CostExceeded(Exception):
    """Internal signal: evaluation would build a number that is too large."""


def _int_value_bounded(node: sp.Basic) -> int | None:
    """Exact integer value of a node built EXCLUSIVELY from numbers (powers,
    products, sums, factorial/gamma/binomial), or ``None`` when the node contains
    symbols/floats/other functions. Raises ``_CostExceeded`` BEFORE computing anything
    whose bit count would exceed ``_MAX_RESULT_BITS`` — never materializes
    a huge int.
    """
    if node.is_Integer:
        return int(node)
    if isinstance(node, sp.Pow):
        base = _int_value_bounded(node.base)
        exp = _int_value_bounded(node.exp)
        if base is None or exp is None or exp < 0:
            return None  # symbolic or negative exp. (Rational) — not a "big int"
        if abs(base) <= 1:
            return base**exp  # 0/1/-1 to any power: negligible cost
        # estimate the RESULT bits before computing base**exp
        if exp > _MAX_RESULT_BITS or exp * abs(base).bit_length() > _MAX_RESULT_BITS:
            raise _CostExceeded(f"power {node} exceeds {_MAX_RESULT_BITS} bits")
        return base**exp
    if isinstance(node, sp.Mul):
        prod, bits = 1, 0
        for arg in node.args:
            val = _int_value_bounded(arg)
            if val is None:
                return None
            bits += abs(val).bit_length() or 1
            if bits > _MAX_RESULT_BITS:
                raise _CostExceeded(f"product {node} exceeds {_MAX_RESULT_BITS} bits")
            prod *= val
        return prod
    if isinstance(node, sp.Add):
        total, bits = 0, 0
        for arg in node.args:
            val = _int_value_bounded(arg)
            if val is None:
                return None
            bits = max(bits, abs(val).bit_length())
            if bits > _MAX_RESULT_BITS:
                raise _CostExceeded(f"sum {node} exceeds {_MAX_RESULT_BITS} bits")
            total += val
        return total
    if isinstance(node, sp.Function):
        name = type(node).__name__
        if name in ("factorial", "gamma") and len(node.args) == 1:
            arg = _int_value_bounded(node.args[0])
            if arg is None:
                return None
            n = arg - 1 if name == "gamma" else arg  # gamma(n) = (n-1)!
            if n < 0:
                return None
            if n > _MAX_FACTORIAL_ARG:
                raise _CostExceeded(f"{name}({arg}) — argument > {_MAX_FACTORIAL_ARG}")
            return math.factorial(n)
        if name == "binomial" and len(node.args) == 2:
            top = _int_value_bounded(node.args[0])
            bottom = _int_value_bounded(node.args[1])
            if top is None or bottom is None:
                return None
            if abs(top) > _MAX_FACTORIAL_ARG:
                raise _CostExceeded(f"binomial({top}, …) — argument > {_MAX_FACTORIAL_ARG}")
            return math.comb(top, bottom) if 0 <= bottom <= top else None
    return None  # Symbol/Float/non-integer Rational/other function → not a "big int"


def _check_cost(node: "sp.Basic | list | tuple") -> None:
    """Walk the tree and reject any CONCRETE subexpression whose evaluation
    would build a number that is too large. ``_int_value_bounded`` closes over the whole
    subgraph built from numbers (raising ``_CostExceeded``); for subgraphs with symbols
    we descend deeper to catch a dangerous nested fragment (e.g. in ``sin(9**9**9)``).

    Matrix literals parse (under ``evaluate=False``) to nested Python lists, so the
    walk also descends into ``list``/``tuple`` containers — a matrix entry gets the
    same DoS guard as a scalar (e.g. ``[[9**9**9]]`` is refused before evaluation).
    """
    if isinstance(node, (list, tuple)):
        for item in node:
            _check_cost(item)
        return
    if _int_value_bounded(node) is not None:
        return
    for arg in node.args:
        _check_cost(arg)


# The guard parses the SAME syntax but without evaluation and with dummy functions
# (computes nothing). Constants become plain symbols (their value does not affect
# the cost). The Pow/Mul/Add constructors are needed because ``evaluate=False`` emits
# explicit constructor calls.
_GUARD_FUNC_NAMES = set(_ALLOWED_FUNCTIONS) | {"ln", "abs", "min", "max"}
_GUARD_LOCAL_DICT: dict[str, object] = {name: sp.Function(name) for name in _GUARD_FUNC_NAMES}
_GUARD_LOCAL_DICT.update({c: sp.Symbol(c) for c in ("pi", "E", "I", "oo")})
_GUARD_GLOBAL_DICT: dict[str, object] = {
    **_GLOBAL_DICT,
    "Pow": sp.Pow,
    "Mul": sp.Mul,
    "Add": sp.Add,
}


def _guard_cost(source: str) -> None:
    """Refuse when evaluating ``source`` would build a gigantic number.

    Parses without evaluation and with dummy functions, so the guard itself
    computes nothing; the real (costly) ``parse_expr`` runs only after this check.
    Structurally invalid input we let through — the real parse will report it.
    """
    try:
        tree = parse_expr(
            source,
            local_dict=dict(_GUARD_LOCAL_DICT),
            global_dict=dict(_GUARD_GLOBAL_DICT),
            transformations=_TRANSFORMS,
            evaluate=False,
        )
    except Exception:  # noqa: BLE001 — syntax/name error: the real parse handles it
        return
    _check_cost(tree)


def parse(source: str) -> Expr:
    """Parse mathematical text into IR.

    A syntax error yields a readable ``PycodemathError`` (not a raw
    ``SympifyError``/``SyntaxError``) — the REPL shows a concise message.

    Security note: the parser resolves ONLY the explicit whitelist of math
    functions and constants (``_LOCAL_DICT``); unknown names become
    symbols, not Python code — ``__import__``/``eval``/``open`` are not
    resolvable, text literals and attribute access are rejected
    at the token level. Honest boundary: this raises the bar (no
    builtin is reachable from text), but evaluation still runs
    through SymPy in this process — this is not a whole-process sandbox.

    Unknown names: single-letter ``f(x)`` reads as multiplication ``f*x``,
    multi-letter ``foo(x)`` breaks up into a product of letters (``split_symbols``
    from ``implicit_multiplication_application`` — the price of ``2x`` notation);
    names with ``_`` (e.g. ``x_1``) or a trailing digit run (e.g. ``x1``, ``v0``,
    ``R1`` — the other common way to write a subscript) stay a single symbol.
    None of these paths calls Python code.
    Building ``Expr`` is in the same ``try`` because ``sympify`` in
    ``Expr.__init__`` can also raise ``SympifyError`` on the parser's output.

    Evaluation cost is bounded: expressions that would build a huge
    number (``9**9**9``, ``factorial(10**8)``) are rejected BEFORE SymPy
    computes them — otherwise a single line (reachable directly from the MCP server) could
    hang or OOM the process.
    """
    try:
        _guard_cost(source)
    except _CostExceeded as exc:
        raise ParseError(
            f"expression {source!r} is too costly to compute "
            f"({exc}) — the result would exceed the safe size"
        ) from exc
    try:
        sy = parse_expr(
            source,
            local_dict=dict(_LOCAL_DICT),
            global_dict=dict(_GLOBAL_DICT),
            transformations=_TRANSFORMS,
            evaluate=True,
        )
        return Expr(sy)
    # NameError: notation emitting a name outside _GLOBAL_DICT (e.g. lambda);
    # AttributeError: attribute access (x.y) — both are a refusal, not a crash.
    except (
        SympifyError,
        SyntaxError,
        TokenError,
        TypeError,
        ValueError,
        NameError,
        AttributeError,
    ) as exc:
        raise ParseError(f"cannot understand expression {source!r}: {exc}") from exc


def parse_matrix(source: str) -> sp.MatrixBase:
    """Parse a matrix/vector literal (``[[1,2],[3,4]]`` / ``[1,2,3]``) into a SymPy matrix.

    Matrix entries are ordinary math expressions, so this text must go through
    the SAME whitelist as scalar :func:`parse`: bare ``sympify`` here would be a
    second eval surface with the full namespace and attribute access
    (``().__class__.__bases__`` reaches ``object`` — the classic sympify gadget
    chain). We reuse the token guard + whitelisted namespace + cost guard, then
    hand the resulting nested list to ``sympy.Matrix``. Legitimate literals are
    unaffected; only the eval surface is closed.
    """
    try:
        _guard_cost(source)
    except _CostExceeded as exc:
        raise ParseError(
            f"matrix {source!r} is too costly to compute "
            f"({exc}) — the result would exceed the safe size"
        ) from exc
    try:
        obj = parse_expr(
            source,
            local_dict=dict(_LOCAL_DICT),
            global_dict=dict(_GLOBAL_DICT),
            transformations=_MATRIX_TRANSFORMS,
            evaluate=True,
        )
        return sp.Matrix(obj)
    # Same refusal set as scalar parse; the token guard's PycodemathError
    # (attribute access, text literals) propagates through unchanged.
    except (
        SympifyError,
        SyntaxError,
        TokenError,
        TypeError,
        ValueError,
        NameError,
        AttributeError,
    ) as exc:
        raise ParseError(f"cannot parse matrix {source!r}: {exc}") from exc
