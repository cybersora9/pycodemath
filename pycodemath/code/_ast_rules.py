"""The one table C1 reads Python's numeric vocabulary through (series FULL, C1).

Every ``math.*`` / ``numpy.*`` name the code frontend understands is in exactly
one of the dictionaries below, next to the SymPy expression it means. Nothing is
inferred from a name at run time: a function missing here is refused by the
frontend with its name and line, never guessed. ``tests/test_code_frontend.py``
walks every entry and compares the SymPy meaning with the real Python/NumPy
function at a point inside its domain — the coverage test the plan asks for.

Semantics, decided once here:

* ``round(x)`` is Python's: halves go to the EVEN neighbour (``round(2.5) == 2``).
  ``numpy.round`` / ``numpy.rint`` do the same. ``round(x, k)`` and
  ``numpy.round(x, k)`` are read as round-half-even of ``x*10**k`` over ``10**k``
  — the decimal meaning; the binary float may land on the other side of a half
  (``round(2.675, 2) == 2.67``), and that is a float-only effect (module C2).
* ``int(x)`` truncates toward zero; ``math.floor`` / ``math.ceil`` are floor /
  ceiling; ``float(x)`` is the identity on the reals.
* ``math.log(x, b)`` is ``log(x)/log(b)``; ``numpy.log`` is the natural log.
"""

from __future__ import annotations

from typing import Callable, Sequence

import sympy as sp

__all__ = [
    "py_round",
    "py_trunc",
    "MATH_FUNCS",
    "NUMPY_FUNCS",
    "MATH_CONSTANTS",
    "NUMPY_CONSTANTS",
    "SAFE_MODULES",
]

Builder = Callable[[Sequence[sp.Expr]], sp.Expr]


def py_round(x: sp.Expr) -> sp.Expr:
    """Python's ``round(x)``: nearest integer, ties to even.

    >>> py_round(sp.Rational(5, 2)), py_round(sp.Rational(-3, 2)), py_round(sp.Rational(7, 10))
    (2, -2, 1)
    """
    up = sp.floor(x + sp.Rational(1, 2))
    tie = sp.And(sp.Eq(x - sp.floor(x), sp.Rational(1, 2)), sp.Eq(sp.Mod(up, 2), 1))
    return sp.Piecewise((up - 1, tie), (up, True))


def py_trunc(x: sp.Expr) -> sp.Expr:
    """``int(x)`` / ``math.trunc(x)``: toward zero.

    >>> py_trunc(sp.Rational(-1, 2)), py_trunc(sp.Rational(7, 2))
    (0, 3)
    """
    return sp.Piecewise((sp.floor(x), x >= 0), (sp.ceiling(x), True))


def _round_digits(args: Sequence[sp.Expr]) -> sp.Expr:
    if len(args) == 1:
        return py_round(args[0])
    scale = sp.Integer(10) ** args[1]
    return py_round(args[0] * scale) / scale


def _log(args: Sequence[sp.Expr]) -> sp.Expr:
    if len(args) == 1:
        return sp.log(args[0])
    return sp.log(args[0]) / sp.log(args[1])


def _one(f: Callable[[sp.Expr], sp.Expr]) -> "tuple[Builder, tuple[int, ...]]":
    return (lambda a: f(a[0])), (1,)


def _two(f: Callable[[sp.Expr, sp.Expr], sp.Expr]) -> "tuple[Builder, tuple[int, ...]]":
    return (lambda a: f(a[0], a[1])), (2,)


#: name -> (builder, allowed argument counts)
_COMMON: "dict[str, tuple[Builder, tuple[int, ...]]]" = {
    "sqrt": _one(sp.sqrt),
    "exp": _one(sp.exp),
    "expm1": _one(lambda x: sp.exp(x) - 1),
    "log2": _one(lambda x: sp.log(x) / sp.log(2)),
    "log10": _one(lambda x: sp.log(x) / sp.log(10)),
    "log1p": _one(lambda x: sp.log(1 + x)),
    "sin": _one(sp.sin),
    "cos": _one(sp.cos),
    "tan": _one(sp.tan),
    "sinh": _one(sp.sinh),
    "cosh": _one(sp.cosh),
    "tanh": _one(sp.tanh),
    "floor": _one(sp.floor),
    "ceil": _one(sp.ceiling),
    "trunc": _one(py_trunc),
    "fabs": _one(sp.Abs),
    "hypot": _two(lambda a, b: sp.sqrt(a**2 + b**2)),
    "cbrt": _one(lambda x: sp.sign(x) * sp.Abs(x) ** sp.Rational(1, 3)),
    "exp2": _one(lambda x: sp.Integer(2) ** x),
}

MATH_FUNCS: "dict[str, tuple[Builder, tuple[int, ...]]]" = {
    **_COMMON,
    "log": (_log, (1, 2)),
    "asin": _one(sp.asin),
    "acos": _one(sp.acos),
    "atan": _one(sp.atan),
    "atan2": _two(sp.atan2),
    "asinh": _one(sp.asinh),
    "acosh": _one(sp.acosh),
    "atanh": _one(sp.atanh),
    "pow": _two(lambda a, b: a**b),
    "factorial": _one(sp.factorial),
    "comb": _two(sp.binomial),
    "perm": _two(lambda n, k: sp.ff(n, k)),
    "gamma": _one(sp.gamma),
    "erf": _one(sp.erf),
    "erfc": _one(sp.erfc),
    "degrees": _one(lambda x: x * 180 / sp.pi),
    "radians": _one(lambda x: x * sp.pi / 180),
    "isqrt": _one(lambda x: sp.floor(sp.sqrt(x))),
    "copysign": _two(lambda a, b: sp.Abs(a) * sp.sign(b)),
}

NUMPY_FUNCS: "dict[str, tuple[Builder, tuple[int, ...]]]" = {
    **_COMMON,
    "log": _one(sp.log),
    "arcsin": _one(sp.asin),
    "arccos": _one(sp.acos),
    "arctan": _one(sp.atan),
    "arctan2": _two(sp.atan2),
    "arcsinh": _one(sp.asinh),
    "arccosh": _one(sp.acosh),
    "arctanh": _one(sp.atanh),
    "abs": _one(sp.Abs),
    "absolute": _one(sp.Abs),
    "power": _two(lambda a, b: a**b),
    "square": _one(lambda x: x**2),
    "sign": _one(sp.sign),
    "maximum": _two(sp.Max),
    "minimum": _two(sp.Min),
    "deg2rad": _one(lambda x: x * sp.pi / 180),
    "rad2deg": _one(lambda x: x * 180 / sp.pi),
    "reciprocal": _one(lambda x: 1 / x),
    "rint": _one(py_round),
    "round": (_round_digits, (1, 2)),
    "around": (_round_digits, (1, 2)),
}

MATH_CONSTANTS: "dict[str, sp.Expr]" = {"pi": sp.pi, "e": sp.E, "tau": 2 * sp.pi}
NUMPY_CONSTANTS: "dict[str, sp.Expr]" = {"pi": sp.pi, "e": sp.E}

#: Modules whose imports do not stop the frontend from EXECUTING a function to
#: confirm a counterexample. Anything else imported (``os``, ``requests``, …)
#: means the function is read but never run.
SAFE_MODULES = frozenset(
    {"math", "numpy", "cmath", "fractions", "decimal", "statistics",
     "functools", "itertools", "operator", "typing", "__future__"}
)
