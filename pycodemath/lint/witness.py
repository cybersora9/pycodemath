"""Witnesses — the executed input behind every lint finding, and its independent replay.

A ``Witness`` is data, not prose: a Python expression (``program``), the inputs it
was run on, what it returned, and the value it should have returned (``expected``).
``replay()`` runs it AGAIN, here, with the code's own operations — Python floats
and ``math``, or NumPy arrays of the declared dtype — and recomputes the expected
value by a route that shares no code with the module that found it: SymPy exact
arithmetic (rationals; transcendental functions to 60 digits) for C2/C3 and the
static rules, an exact decimal evaluator for the money rules. A finding is emitted
only when the replay confirms it (``engine``), and ``tests/test_lint_guards.py``
replays every finding on the evaluation set once more.

Two readings of numbers, chosen per rule and recorded in ``reading``:

``binary``
    The inputs are the floats themselves; the exact value is the real number those
    floats denote (``Fraction(0.1)`` is 0.1000000000000000055...). Literals in the
    program mean the decimal written (as in C2 / Herbie). C2 and C3 findings.
``decimal``
    Inputs and literals mean the decimal written (``0.1`` is 1/10) — the number the
    programmer typed or the amount in the ledger. The program's top operation must
    then be a comparison, ``round`` or ``int``, so that "different" is a statement
    about the answer the code gives, never about the 17th digit of a float.
    Money rules and PCM501.

Comparison modes (``compare``):

``bits:<k>``         the float result is at least ``k`` bits (ULP-ordinal distance,
                     ``log2(1 + U)``) from the exact value rounded to the dtype
``event:<kind>``     C3's event classes, each with its own predicate (``_EVENTS``)
``value``            the result differs from the exact one (decimal reading)
``call``             a function call against a formula (C1), exact or within ``tolerance``

    >>> w = Witness(kind="xor", program="x ^ 2", inputs={"x": "3"}, expected_program="x ** 2",
    ...             reading="binary", compare="value", observed="1", expected="9")
    >>> w.replay().differs
    True
"""

from __future__ import annotations

import ast
import math
import struct
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable, Mapping

import numpy as np

__all__ = ["Witness", "Fix", "Replay", "exact_value", "run_program", "bits_between", "WitnessError"]


class WitnessError(Exception):
    """The program is outside the replayable subset (never reaches a user: the
    engine drops a finding whose witness cannot be replayed)."""


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Fix:
    """A replacement for ``[line:col, end_line:end_col)`` of the reported file.

    ``proof`` names how the replacement was shown equal to the original (C2's
    ``check_equal``, C3's symbolic proof, C1's verification against the spec);
    ``None`` means a suggestion whose meaning differs on purpose (``math.isclose``
    instead of ``==``) — ruff's "unsafe fix". ``applicability`` is ``"safe"``
    exactly when ``proof`` is set.
    """

    description: str
    replacement: str
    line: int
    col: int
    end_line: int
    end_col: int
    proof: "str | None" = None

    @property
    def applicability(self) -> str:
        return "safe" if self.proof else "suggestion"

    def to_dict(self) -> dict[str, Any]:
        return {"description": self.description, "replacement": self.replacement,
                "location": {"row": self.line, "column": self.col},
                "end_location": {"row": self.end_line, "column": self.end_col},
                "proof": self.proof, "applicability": self.applicability}


@dataclass(frozen=True)
class Replay:
    """The outcome of ``Witness.replay``: what ran, what it should be, whether they differ."""

    observed: str
    expected: str
    differs: bool
    detail: str = ""


@dataclass(frozen=True)
class Witness:
    """An executed input on which the code computes something other than the exact value.

    ``program`` is a Python expression over ``inputs`` (name -> literal text). ``dtype``
    (NumPy name) makes every input a one-element array of that type, as C3 models
    array code; ``None`` means Python numbers. ``expected_program`` is the program
    whose EXACT value is the reference when it is not ``program`` itself (``x ** 2``
    for ``x ^ 2``; single rounding for double rounding). ``rounding`` is the mode of
    ``round`` on the exact side (``HALF_EVEN`` = Python's, ``HALF_UP`` = invoices).
    ``observed`` / ``expected`` are what the finding reported; ``replay`` recomputes
    both. For ``compare="call"``: ``program`` is the call, ``formula`` the spec,
    ``tolerance`` C1's relative tolerance for a float result (0 = exact).
    """

    kind: str
    program: str
    inputs: dict[str, str]
    observed: str
    expected: str
    compare: str
    reading: str = "binary"
    dtype: "str | None" = None
    expected_program: "str | None" = None
    rounding: str = "HALF_EVEN"
    formula: "str | None" = None
    tolerance: float = 0.0
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """One line for a human: inputs, the code's value, the exact value."""
        args = ", ".join(f"{k}={v}" for k, v in self.inputs.items())
        where = f" at {args}" if args else ""
        if self.compare == "call":
            return f"{self.program} returns {self.observed}; the formula gives {self.expected}"
        ref = self.expected_program or "exact"
        if self.compare.startswith("bits:"):
            bits = self.extra.get("bits")
            lost = f" ({bits:.0f} bits of error)" if isinstance(bits, (int, float)) else ""
            return f"{self.program}{where} = {self.observed}, exact {self.expected}{lost}"
        if self.expected_program:
            return f"{self.program}{where} = {self.observed}, but {ref} = {self.expected}"
        return f"{self.program}{where} = {self.observed}, exact {self.expected}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind, "program": self.program, "inputs": dict(self.inputs),
            "observed": self.observed, "expected": self.expected, "compare": self.compare,
            "reading": self.reading, "dtype": self.dtype,
            "expected_program": self.expected_program, "rounding": self.rounding,
        }
        if self.formula is not None:
            d["formula"] = self.formula
            d["tolerance"] = self.tolerance
        if self.note:
            d["note"] = self.note
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Witness":
        return cls(kind=d["kind"], program=d["program"], inputs=dict(d["inputs"]),
                   observed=d["observed"], expected=d["expected"], compare=d["compare"],
                   reading=d.get("reading", "binary"), dtype=d.get("dtype"),
                   expected_program=d.get("expected_program"),
                   rounding=d.get("rounding", "HALF_EVEN"), formula=d.get("formula"),
                   tolerance=float(d.get("tolerance", 0.0)), note=d.get("note", ""),
                   extra=dict(d.get("extra", {})))

    def replay(self, module_source: "str | None" = None) -> Replay:
        """Execute the program again and recompute the expected value independently.

        ``module_source`` is needed for ``compare="call"`` (the function lives there);
        the module's definitions are executed IN THIS PROCESS, so pass only source
        you would import anyway. The engine itself never calls this for ``call``
        witnesses — C1 executed them in its isolated worker.
        """
        if self.compare == "call":
            return _replay_call(self, module_source)
        observed = run_program(self.program, self.inputs, self.dtype, self.reading)
        reference = self.expected_program or self.program
        if self.compare in ("event:nan_possible", "event:division_by_zero_possible"):
            # 0/0, log(0), sqrt(-1): there is no exact real value — the event IS its absence
            try:
                exact = exact_value(reference, self.inputs, self.reading, self.rounding)
            except (WitnessError, ArithmeticError, ValueError, TypeError):
                pred = _EVENTS[self.compare.split(":", 1)[1]]
                return Replay(_show(observed), "undefined", pred(observed, None, self.dtype))
            pred = _EVENTS[self.compare.split(":", 1)[1]]
            return Replay(_show(observed), _show_exact(exact), pred(observed, exact, self.dtype))
        exact = exact_value(reference, self.inputs, self.reading, self.rounding)
        if self.compare.startswith("bits:"):
            k = float(self.compare.split(":", 1)[1])
            bits = bits_between(observed, exact, self.dtype or "float64")
            return Replay(_show(observed), _show_exact(exact), bits >= k, f"{bits:.1f} bits")
        if self.compare.startswith("event:"):
            pred = _EVENTS[self.compare.split(":", 1)[1]]
            return Replay(_show(observed), _show_exact(exact), pred(observed, exact, self.dtype))
        if self.compare == "value":
            return Replay(_show(observed), _show_exact(exact), _values_differ(observed, exact, self.reading))
        raise WitnessError(f"unknown comparison {self.compare!r}")


# --------------------------------------------------------------------------- #
# the replayable subset                                                       #
# --------------------------------------------------------------------------- #

_MODULES = ("math", "np", "numpy")
_PY_FUNCS = {"round", "abs", "min", "max", "int", "float"}
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitXor)


def _check(tree: ast.AST, names: "set[str]", call_ok: "Callable[[ast.Call], bool] | None" = None) -> None:
    """Refuse anything but arithmetic, comparisons and whitelisted calls."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.operator, ast.unaryop, ast.cmpop,
                             ast.keyword)):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINOPS):
                raise WitnessError(f"operator {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise WitnessError(f"operator {type(node.op).__name__}")
        elif isinstance(node, ast.Compare):
            pass
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float, bool):
                raise WitnessError(f"literal {node.value!r}")
        elif isinstance(node, ast.Name):
            if node.id not in names and node.id not in _MODULES and node.id not in _PY_FUNCS:
                raise WitnessError(f"name {node.id!r}")
        elif isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id in _MODULES):
                raise WitnessError("attribute access")
        elif isinstance(node, ast.Call):
            f = node.func
            if call_ok is not None and call_ok(node):
                continue
            if isinstance(f, ast.Name) and f.id in _PY_FUNCS:
                continue
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in _MODULES:
                continue
            raise WitnessError("call")
        else:
            raise WitnessError(type(node).__name__)


def _input_value(text: str, dtype: "str | None", reading: str) -> Any:
    v = ast.literal_eval(text)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise WitnessError(f"input {text!r} is not a number")
    if dtype is not None:
        return np.array([v], dtype=np.dtype(dtype))
    return v


def run_program(program: str, inputs: Mapping[str, str], dtype: "str | None" = None,
                reading: str = "binary") -> Any:
    """Execute ``program`` on ``inputs`` with Python / NumPy semantics; return the value.

    An exception raised by the program (``TypeError`` from ``3.0 ^ 2``,
    ``ZeroDivisionError``) is returned as the string ``"<raises TypeError>"`` — it
    is an observable outcome, different from any number.
    """
    tree = ast.parse(program, mode="eval")
    _check(tree, set(inputs))
    env = {k: _input_value(v, dtype, reading) for k, v in inputs.items()}
    scope = {"__builtins__": {f: __builtins__[f] if isinstance(__builtins__, dict)  # type: ignore[index]
                              else getattr(__builtins__, f) for f in _PY_FUNCS},
             "math": math, "np": np, "numpy": np}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            out = eval(compile(tree, "<witness>", "eval"), scope, env)  # noqa: S307 - checked AST
        except Exception as exc:  # the program's own failure is the observation
            return f"<raises {type(exc).__name__}>"
    if isinstance(out, np.ndarray):
        out = out.reshape(-1)[0]
    return out


# --------------------------------------------------------------------------- #
# exact side (SymPy) — shares no code with C1-C4                              #
# --------------------------------------------------------------------------- #


def _sympy() -> Any:
    import sympy as sp

    return sp


def _sp_funcs() -> dict[str, Callable[..., Any]]:
    sp = _sympy()

    def cbrt(x: Any) -> Any:
        return sp.sign(x) * sp.Abs(x) ** sp.Rational(1, 3)

    return {
        "sqrt": sp.sqrt, "exp": sp.exp, "log": lambda x, b=None: sp.log(x) if b is None else sp.log(x) / sp.log(b),
        "log10": lambda x: sp.log(x) / sp.log(10), "log2": lambda x: sp.log(x) / sp.log(2),
        "log1p": lambda x: sp.log(1 + x), "expm1": lambda x: sp.exp(x) - 1,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
        "atan": sp.atan, "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh,
        "atanh": sp.atanh, "arcsinh": sp.asinh, "arccosh": sp.acosh, "arctanh": sp.atanh,
        "fabs": sp.Abs, "abs": sp.Abs, "absolute": sp.Abs,
        "hypot": lambda a, b: sp.sqrt(a**2 + b**2), "pow": lambda a, b: a**b, "power": lambda a, b: a**b,
        "atan2": sp.atan2, "arctan2": sp.atan2, "cbrt": cbrt, "square": lambda x: x**2,
        "floor": sp.floor, "ceil": sp.ceiling, "trunc": lambda x: sp.sign(x) * sp.floor(sp.Abs(x)),
    }


_SP_CONSTS = {"pi": "pi", "e": "E", "tau": "tau", "inf": "oo"}


def _decide(e: Any) -> Any:
    """A SymPy real as an exact Rational when it is one, else a 60-digit Float."""
    sp = _sympy()
    e = sp.nsimplify(e) if isinstance(e, sp.Float) else e
    if e.is_Rational:
        return e
    v = sp.N(e, 60)
    if not v.is_real or v.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise WitnessError("the exact value is not a finite real")
    return v


def _round_exact(x: Any, nd: int, mode: str) -> Any:
    """``round(x, nd)`` on an exact real: half to even (Python) or half away from zero."""
    sp = _sympy()
    q = _decide(x)
    frac = Fraction(int(q.p), int(q.q)) if q.is_Rational else Fraction(Decimal(str(q)))
    step = Fraction(1, 10**nd) if nd >= 0 else Fraction(10**-nd)
    units = frac / step
    fl = units.numerator // units.denominator
    rem = units - fl
    if rem > Fraction(1, 2):
        r = fl + 1
    elif rem < Fraction(1, 2):
        r = fl
    elif mode == "HALF_UP":
        r = fl + 1 if units > 0 else fl
    else:
        r = fl + 1 if fl % 2 else fl
    return sp.Rational(r) * sp.Rational(step.numerator, step.denominator)


#: Above this many bits an exact power is computed at 80 significant digits instead
#: (``(1 + r) ** 360`` of a binary float is a rational with ~19 000-digit terms).
_EXACT_POWER_BITS = 4096


def _power(a: Any, b: Any) -> Any:
    sp = _sympy()
    if getattr(a, "is_Rational", False) and getattr(b, "is_Integer", False):
        size = (int(a.p).bit_length() + int(a.q).bit_length()) * abs(int(b))
        if size <= _EXACT_POWER_BITS:
            return a**b
    if getattr(a, "is_number", False) and getattr(b, "is_number", False):
        return sp.Float(a, 80) ** sp.Float(b, 80) if not (a.is_Rational and b.is_Rational
                                                          and b.q != 1 and a < 0) else a**b
    return a**b


def exact_value(program: str, inputs: Mapping[str, str], reading: str = "binary",
                rounding: str = "HALF_EVEN") -> Any:
    """The exact value of ``program`` at ``inputs`` (a SymPy number or a bool).

    ``binary`` reading: an input is the exact real of its float (``Fraction(0.1)``);
    ``decimal``: the decimal written. Literals are always the decimal written.
    ``round`` uses ``rounding`` (``HALF_EVEN`` = Python's ``round``).
    """
    sp = _sympy()
    tree = ast.parse(program, mode="eval")
    _check(tree, set(inputs))
    funcs = _sp_funcs()
    env: dict[str, Any] = {}
    for k, text in inputs.items():
        v = ast.literal_eval(text)
        if isinstance(v, int) and not isinstance(v, bool):
            env[k] = sp.Integer(v)
        elif reading == "decimal":
            env[k] = sp.Rational(text.strip())
        else:
            f = Fraction(float(v))
            env[k] = sp.Rational(f.numerator, f.denominator)

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool):
                return n.value
            if isinstance(n.value, int):
                return sp.Integer(n.value)
            seg = ast.get_source_segment(program, n) or repr(n.value)
            return sp.Rational(seg.strip())
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            raise WitnessError(f"name {n.id!r}")
        if isinstance(n, ast.Attribute):
            assert isinstance(n.value, ast.Name)
            if n.attr in _SP_CONSTS:
                return getattr(sp, _SP_CONSTS[n.attr]) if n.attr != "tau" else 2 * sp.pi
            raise WitnessError(f"{n.value.id}.{n.attr}")
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            op = n.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                if b == 0:
                    raise WitnessError("division by zero on the exact side")
                return a / b
            if isinstance(op, ast.FloorDiv):
                return sp.floor(a / b)
            if isinstance(op, ast.Mod):
                return a - b * sp.floor(a / b)
            if isinstance(op, ast.Pow):
                return _power(a, b)
            raise WitnessError("^ on the exact side")
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for cop, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                d = _decide(sp.sympify(left - right))
                ok = {ast.Eq: d == 0, ast.NotEq: d != 0, ast.Lt: d < 0, ast.LtE: d <= 0,
                      ast.Gt: d > 0, ast.GtE: d >= 0}.get(type(cop))
                if ok is None:
                    raise WitnessError("comparison")
                if not ok:
                    return False
                left = right
            return True
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            args = [ev(a) for a in n.args]
            if name == "round":
                nd = 0 if len(args) == 1 else int(args[1])
                r = _round_exact(args[0], nd, rounding)
                return r
            if name == "int":
                return sp.sign(args[0]) * sp.floor(sp.Abs(args[0]))
            if name == "float":
                return args[0]
            if name in ("min", "max", "minimum", "maximum", "fmin", "fmax"):
                vals = [_decide(a) for a in args]
                return min(vals) if name in ("min", "minimum", "fmin") else max(vals)
            if name in funcs:
                return funcs[name](*args)
            raise WitnessError(f"function {name!r}")
        raise WitnessError(type(n).__name__)

    out = ev(tree)
    if isinstance(out, bool):
        return out
    return _decide(sp.sympify(out))


# --------------------------------------------------------------------------- #
# comparisons                                                                 #
# --------------------------------------------------------------------------- #


def _ordinal(x: float, dtype: str) -> int:
    if dtype == "float32":
        (i,) = struct.unpack("<i", struct.pack("<f", x))
    elif dtype == "float16":
        (i,) = struct.unpack("<h", np.float16(x).tobytes())
    else:
        (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & (2 ** (8 * {"float32": 4, "float16": 2}.get(dtype, 8) - 1) - 1))


def _round_to(exact: Any, dtype: str) -> float:
    sp = _sympy()
    if exact.is_Rational:
        f = float(Fraction(int(exact.p), int(exact.q)))
    else:
        f = float(sp.Float(exact, 60))
    if dtype in ("float32", "float16"):
        return float(np.dtype(dtype).type(f))
    return f


def bits_between(observed: Any, exact: Any, dtype: str = "float64") -> float:
    """Bits of error as in Herbie / C2: ``log2(1 + U)``, ``U`` the ordinal distance.

    A NaN, an infinity or a raised exception where a finite real was due counts as
    the format's full width (64 / 32 / 16 bits), as Herbie scores it.
    """
    width = {"float32": 32, "float16": 16}.get(dtype, 64)
    if isinstance(exact, bool):
        raise WitnessError("bits of error of a comparison")
    try:
        obs = float(observed)
    except (TypeError, ValueError):
        return float(width)
    target = _round_to(exact, dtype)
    if not math.isfinite(obs):
        return float(width) if math.isfinite(target) else 0.0
    if not math.isfinite(target):
        return float(width)
    u = abs(_ordinal(obs, dtype) - _ordinal(target, dtype))
    return math.log2(1 + u)


def _as_fraction(v: Any, reading: str) -> "Fraction | None":
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, np.integer)):
        return Fraction(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    # decimal reading: a float result of round/int is the decimal it prints as
    return Fraction(Decimal(repr(f))) if reading == "decimal" else Fraction(f)


def _values_differ(observed: Any, exact: Any, reading: str) -> bool:
    if isinstance(exact, bool):
        return not isinstance(observed, (bool, np.bool_)) or bool(observed) != exact
    if isinstance(observed, str):
        return True  # raised
    fo = _as_fraction(observed, reading)
    if fo is None:
        return True  # nan / inf / bool where a number is due
    sp = _sympy()
    return bool(sp.Rational(fo.numerator, fo.denominator) != exact) if exact.is_Rational \
        else abs(sp.Float(fo.numerator, 60) / fo.denominator - exact) > sp.Float(10) ** -40


def _limits(dtype: "str | None") -> tuple[Fraction, Fraction]:
    dt = np.dtype(dtype or "float64")
    if dt.kind in "iu":
        info = np.iinfo(dt)
        return Fraction(int(info.min)), Fraction(int(info.max))
    m = Fraction(float(np.finfo(dt).max))
    return -m, m


def _exact_frac(exact: Any) -> "Fraction | None":
    if isinstance(exact, bool) or not exact.is_Rational:
        return None
    return Fraction(int(exact.p), int(exact.q))


def _ev_overflow(observed: Any, exact: Any, dtype: "str | None") -> bool:
    lo, hi = _limits(dtype)
    if isinstance(exact, bool):
        return False
    sp = _sympy()
    big = (exact > sp.Rational(hi.numerator, hi.denominator)) or (exact < sp.Rational(lo.numerator, lo.denominator))
    return bool(big) and _values_differ(observed, exact, "binary")


def _ev_int_loss(observed: Any, exact: Any, dtype: "str | None") -> bool:
    f = _exact_frac(exact)
    return f is not None and f.denominator == 1 and _values_differ(observed, exact, "binary")


def _ev_underflow(observed: Any, exact: Any, dtype: "str | None") -> bool:
    try:
        return float(observed) == 0.0 and exact != 0
    except (TypeError, ValueError):
        return False


def _ev_nan(observed: Any, exact: Any, dtype: "str | None") -> bool:
    try:
        return math.isnan(float(observed))
    except (TypeError, ValueError):
        return False


def _ev_divzero(observed: Any, exact: Any, dtype: "str | None") -> bool:
    try:
        f = float(observed)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(f)


#: C3 event class -> (observed, exact, dtype) -> does the replay show the event?
_EVENTS: dict[str, Callable[[Any, Any, "str | None"], bool]] = {
    "overflow": _ev_overflow,
    "integer_precision_loss": _ev_int_loss,
    "underflow_to_zero": _ev_underflow,
    "nan_possible": _ev_nan,
    "division_by_zero_possible": _ev_divzero,
}


def _show(v: Any) -> str:
    if isinstance(v, np.generic):
        v = v.item()
    return repr(v) if isinstance(v, float) else str(v)


def _show_exact(v: Any) -> str:
    """An exact value as text: an integer, a terminating decimal, or 20 digits."""
    if isinstance(v, bool):
        return str(v)
    if v.is_Rational:
        if v.q == 1:
            return str(v.p)
        q = int(v.q)
        while q % 2 == 0:
            q //= 2
        while q % 5 == 0:
            q //= 5
        if q == 1:
            d = Decimal(int(v.p)) / Decimal(int(v.q))
            if Fraction(d) == Fraction(int(v.p), int(v.q)):
                return format(d, "f")
    return f"{_sympy().N(v, 20)}"


# --------------------------------------------------------------------------- #
# call witnesses (C1)                                                         #
# --------------------------------------------------------------------------- #

_SAFE_IMPORTS = {"math", "numpy", "cmath", "fractions", "decimal", "statistics", "functools",
                 "itertools", "operator", "typing", "__future__"}


def _definitions(source: str) -> str:
    tree = ast.parse(source)
    keep: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if all(m.split(".")[0] in _SAFE_IMPORTS or m.startswith("pycodemath") for m in mods):
                keep.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            keep.append(node)
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def _replay_call(w: Witness, module_source: "str | None") -> Replay:
    if module_source is None:
        raise WitnessError("a call witness needs the module source to replay")
    sp = _sympy()
    ns: dict[str, Any] = {}
    exec(compile(_definitions(module_source), "<lint-replay>", "exec"), ns)  # noqa: S102 - caller's own source
    call = ast.parse(w.program, mode="eval").body
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
        raise WitnessError("call witness program must be f(k=v, ...)")
    args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
    try:
        got: Any = ns[call.func.id](**args)
    except Exception as exc:
        return Replay(f"<raises {type(exc).__name__}>", w.expected, True)
    syms = {k: sp.Symbol(k) for k in args}
    assert w.formula is not None
    f = sp.sympify(w.formula.replace("^", "**"), locals=syms, rational=True)
    exact = _decide(f.subs({syms[k]: sp.nsimplify(v, rational=True) if isinstance(v, float)
                            else sp.Integer(v) for k, v in args.items()}))
    if isinstance(got, (bool, np.bool_)) or got is None:
        return Replay(str(got), _show_exact(exact), True)
    fo = _as_fraction(got, "binary")
    if fo is None:
        return Replay(_show(got), _show_exact(exact), True)
    if w.tolerance == 0:
        differs = sp.Rational(fo.numerator, fo.denominator) != exact
    else:
        ex = sp.Float(exact, 60)
        differs = abs(sp.Float(fo.numerator, 60) / fo.denominator - ex) > w.tolerance * max(abs(ex), sp.Float(1e-300))
    return Replay(_show(got), _show_exact(exact), bool(differs))

