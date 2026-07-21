"""Parser — natural math notation → IR.

A thin layer over ``sympy.parsing``. Accepts standard mathematical
notation (``sin(x)*x + 2^3``) and returns an ``Expr`` from the IR core.
Guarantees round-trip: ``parse(str(parse(s))) == parse(s)``.
"""

from __future__ import annotations

import math
import token as _token
from tokenize import TokenError

import sympy as sp
from sympy import SympifyError
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..core.errors import PycodemathError
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
            raise PycodemathError(
                f"text literals are not a mathematical expression: {tok_val!r}"
            )
        if tok_type == _token.OP and tok_val == ".":
            raise PycodemathError(
                "attribute access (dot) is not mathematical notation"
            )
    return tokens


# Token guard BEFORE everything, then ^ as exponentiation + implicit
# multiplication (2x -> 2*x), as in math notation.
_TRANSFORMS = (_deny_unsafe_tokens,) + standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

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


def _check_cost(node: sp.Basic) -> None:
    """Walk the tree and reject any CONCRETE subexpression whose evaluation
    would build a number that is too large. ``_int_value_bounded`` closes over the whole
    subgraph built from numbers (raising ``_CostExceeded``); for subgraphs with symbols
    we descend deeper to catch a dangerous nested fragment (e.g. in ``sin(9**9**9)``).
    """
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
    names with ``_`` (e.g. ``x_1``) stay a single symbol. None of these
    paths calls Python code.
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
        raise PycodemathError(
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
        raise PycodemathError(f"cannot understand expression {source!r}: {exc}") from exc
