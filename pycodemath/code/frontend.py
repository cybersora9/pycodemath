"""C1 — code <-> formula: does this Python function compute that formula?

``check_code(func_or_source, formula)`` answers VERIFIED, REFUTED or UNDECIDED:

* **VERIFIED** — the function, READ from its AST into a symbolic form, equals the
  formula on the declared domain, by a proof: SymPy simplification (``method=
  "symbolic"``), a case split over the branches of a ``Piecewise`` (``"case-split"``),
  an exact quasi-polynomial certificate for integer code with ``//``, ``%`` and
  ``if`` (``"quasi-polynomial"``), or ``check_equal``'s own symbolic proof
  (``"check_equal"``). The reading is also cross-checked by EXECUTING the function
  at sample points; a reading the execution contradicts is never used for a proof.
* **REFUTED** — with a concrete call, ``sum_to(n=1) -> 0, formula gives 1``. The
  call was really executed (in the ``isolated`` worker, under a hard deadline) and
  the formula evaluated exactly; both must agree that the values differ. A float
  result is compared against the exact reading of the code and the exact formula,
  so rounding alone never refutes: that is ``float_only_discrepancy`` (module C2).
* **UNDECIDED** — with the reason: a construct outside the supported subset
  (named, with its line), a loop that does not reduce to a closed form, a missing
  proof, the budget. Even then a counterexample is searched for by execution.

``to_symbolic`` is the reading alone, and it NEVER executes anything: the function
is parsed with ``ast`` and interpreted symbolically. The supported subset and its
semantics are in ``_Reader``'s docstring; ``emit_test`` turns a REFUTED result
into a pytest file that pins the counterexample.

    >>> src = '''
    ... def sum_to(n: int) -> int:
    ...     total = 0
    ...     for i in range(1, n):
    ...         total += i
    ...     return total
    ... '''
    >>> r = check_code(src, "n*(n+1)/2", domain={"n": "integer, n>=0"})   # doctest: +SKIP
    >>> r.verdict, r.counterexample.call                                 # doctest: +SKIP
    ('refuted', 'sum_to(n=1)')
"""

from __future__ import annotations

import ast
import datetime
import fractions
import inspect
import io
import itertools
import math
import pickle
import re
import textwrap
import time
import tokenize
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import mpmath
import sympy as sp

from ..core.backstop import isolated
from ..core.budget import DEFAULT_TIME_BUDGET, time_budget
from ..core.errors import (
    DomainError,
    IsolationError,
    ParseError,
    PycodemathError,
    TimeBudgetError,
    UnsupportedFormError,
)
from ..core.ir import Expr
from ..verify.equal import _caller_budget_active, _parse_written, check_equal
from ._ast_rules import (
    MATH_CONSTANTS,
    MATH_FUNCS,
    NUMPY_CONSTANTS,
    NUMPY_FUNCS,
    SAFE_MODULES,
    py_round,
    py_trunc,
)

__all__ = [
    "check_code",
    "to_symbolic",
    "emit_test",
    "CodeCheck",
    "Counterexample",
    "Note",
    "SymbolicFunction",
    "CodeReadError",
]


class CodeReadError(UnsupportedFormError):
    """The function uses a construct outside C1's supported subset.

    ``construct`` names it (``"while loop"``, ``"recursion"``, ``"bitwise ^"``),
    ``line`` is its line in the source (1-based, ``None`` when unknown). Never
    raised for a mathematical reason — only for what the reader cannot read.
    """

    def __init__(self, construct: str, line: "int | None", hint: str = "") -> None:
        where = f" at line {line}" if line else ""
        super().__init__(
            f"to_symbolic: unsupported construct — {construct}{where}"
            + (f" ({hint})" if hint else "")
        )
        self.construct = construct
        self.line = line

    def __reduce__(self):  # crosses the ``isolated`` boundary intact
        return (type(self), (self.construct, self.line))


class _NotReducible(Exception):
    """A ``for`` loop the reader cannot close (internal; becomes CodeReadError)."""

    def __init__(self, line: "int | None", why: str) -> None:
        super().__init__(why)
        self.line = line
        self.why = why


#: Most ``if`` paths one function may fan out to before the reader gives up.
_MAX_PATHS = 64
#: Deepest chain of inlined helper calls.
_MAX_INLINE = 8
#: Loops with a concrete trip count up to this are unrolled when they do not close.
_MAX_UNROLL = 200

# --------------------------------------------------------------------------- #
# results                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Note:
    """One finding beside the verdict: ``code`` is stable, ``message`` for humans.

    Codes: ``float_only_discrepancy`` (rounding, see module C2),
    ``suspicious_xor`` (``^`` in numeric code), ``loop_not_reducible``,
    ``unsupported``, ``undefined_path`` (a branch returns nothing / raises),
    ``reading_mismatch`` (execution contradicted the symbolic reading),
    ``unconfirmed_candidate``, ``not_executed``, ``domain``.
    """

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class Counterexample:
    """A call on which the function and the formula disagree — both confirmed.

    ``args`` are the Python values passed (ints, or floats that are exactly the
    dyadic rationals in ``args_exact``); ``function_value`` is what the real call
    returned; ``formula_value`` the formula's exact value as SymPy text, with
    ``formula_approx`` its float. ``exact`` says the two were compared exactly
    (an int or exact-rational result); otherwise the float result was compared
    with ``tolerance`` (relative) after the exact reading of the code confirmed it.
    """

    args: dict[str, Any]
    args_exact: dict[str, str]
    function_value: Any
    formula_value: str
    formula_approx: float
    exact: bool
    tolerance: float
    confirmed_by_execution: bool
    function_name: str

    @property
    def call(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.function_name}({inner})"

    def to_dict(self) -> dict[str, Any]:
        fv = self.function_value
        return {
            "call": self.call,
            "args": {k: _jsonable(v) for k, v in self.args.items()},
            "args_exact": dict(self.args_exact),
            "function_value": _jsonable(fv),
            "formula_value": self.formula_value,
            "formula_approx": self.formula_approx,
            "exact": self.exact,
            "tolerance": self.tolerance,
            "confirmed_by_execution": self.confirmed_by_execution,
        }


@dataclass(frozen=True)
class SymbolicFunction:
    """A Python function read into SymPy: ``expr`` in terms of ``symbols``.

    ``expr`` may contain ``Piecewise`` (branches; ``nan`` marks a path that
    returns nothing or raises), ``Sum``/``Product`` (closed loops) and
    ``floor``/``Mod`` (``//``, ``%``). ``params`` keeps the call order.
    """

    name: str
    params: tuple[str, ...]
    symbols: dict[str, sp.Symbol]
    expr: sp.Expr
    defaults: dict[str, Any]
    notes: tuple[Note, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": list(self.params),
            "expr": sp.sstr(self.expr),
            "assumptions": {k: _assumption_text(s) for k, s in self.symbols.items()},
            "notes": [n.to_dict() for n in self.notes],
        }


@dataclass(frozen=True)
class CodeCheck:
    """The outcome of ``check_code``. ``verdict`` is ``"verified"``,
    ``"refuted"`` or ``"undecided"``; ``counterexample`` is set exactly when it
    is ``"refuted"``. ``method`` names how it was decided, ``reason`` is one
    sentence for a human. ``kind`` is ``"code_equiv"`` (certificate kind, X1)."""

    verdict: str
    method: str
    reason: str
    function_name: str
    params: tuple[str, ...]
    formula: str
    symbolic_form: "str | None"
    counterexample: "Counterexample | None"
    notes: tuple[Note, ...]
    domain: dict[str, str]
    source: str
    module: "str | None" = None
    kind: str = "code_equiv"
    elapsed: float = 0.0
    checked_points: int = 0
    executed_calls: int = 0

    def __post_init__(self) -> None:
        if self.verdict not in ("verified", "refuted", "undecided"):
            raise DomainError(f"CodeCheck: unknown verdict {self.verdict!r}")
        if (self.verdict == "refuted") != (self.counterexample is not None):
            raise DomainError(
                "CodeCheck: counterexample must be set exactly when verdict is refuted"
            )

    @property
    def verified(self) -> bool:
        return self.verdict == "verified"

    @property
    def refuted(self) -> bool:
        return self.verdict == "refuted"

    @property
    def undecided(self) -> bool:
        return self.verdict == "undecided"

    def has_note(self, code: str) -> bool:
        return any(n.code == code for n in self.notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "verdict": self.verdict,
            "method": self.method,
            "reason": self.reason,
            "function": self.function_name,
            "params": list(self.params),
            "formula": self.formula,
            "symbolic_form": self.symbolic_form,
            "counterexample": None if self.counterexample is None else self.counterexample.to_dict(),
            "notes": [n.to_dict() for n in self.notes],
            "domain": dict(self.domain),
            "checked_points": self.checked_points,
            "executed_calls": self.executed_calls,
            "elapsed": round(self.elapsed, 4),
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int):
        return v if abs(v) < 2**53 else str(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, fractions.Fraction):
        return str(v)
    return repr(v)


def _assumption_text(s: sp.Symbol) -> str:
    for key in ("positive", "nonnegative", "negative", "nonpositive"):
        if s.assumptions0.get(key):
            kind = "integer" if s.is_integer else "real"
            return f"{kind}, {key}"
    return "integer" if s.is_integer else "real"


# --------------------------------------------------------------------------- #
# source handling                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class _Source:
    text: str  # the whole module text the function lives in (dedented)
    tree: ast.Module
    func: ast.FunctionDef
    obj: "Callable[..., Any] | None"  # the function object, when one was given
    module: "str | None"


def _load(func_or_source: "Callable[..., Any] | str", name: "str | None") -> _Source:
    obj = None
    module = None
    if isinstance(func_or_source, str):
        text = textwrap.dedent(func_or_source)
    elif callable(func_or_source):
        obj = func_or_source
        name = name or getattr(obj, "__name__", None)
        module = getattr(obj, "__module__", None)
        try:
            mod = inspect.getmodule(obj)
            text = inspect.getsource(mod) if mod is not None else ""
        except (OSError, TypeError):
            text = ""
        if not text or name is None or f"def {name}" not in text:
            try:
                text = textwrap.dedent(inspect.getsource(obj))
            except (OSError, TypeError) as exc:
                raise DomainError(
                    f"check_code: the source of {name or obj!r} is not available "
                    f"(defined interactively?) — pass the source text instead"
                ) from exc
    else:
        raise DomainError(
            f"check_code: expected a function or its source text, not {type(func_or_source).__name__}"
        )
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ParseError(f"check_code: the source does not parse as Python: {exc.msg} (line {exc.lineno})") from exc
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not defs:
        raise DomainError("check_code: no top-level function definition in the source")
    if name is None:
        func = defs[-1]  # helpers usually come first; the last def is the subject
    else:
        found = [d for d in defs if d.name == name]
        if not found:
            raise DomainError(
                f"check_code: no function named {name!r} (found: {', '.join(d.name for d in defs)})"
            )
        func = found[-1]
    return _Source(text, tree, func, obj, module)


# --------------------------------------------------------------------------- #
# domains                                                                     #
# --------------------------------------------------------------------------- #

_TYPE_WORDS = {
    "integer": "integer", "int": "integer", "real": "real", "float": "real",
    "positive": "positive", "nonnegative": "nonnegative", "negative": "negative",
    "nonpositive": "nonpositive",
}
_REL = re.compile(r"(<=|>=|!=|==|<|>)")


@dataclass
class _Domain:
    kinds: dict[str, set[str]]
    relations: list[tuple[str, str, str]]  # (lhs text, op, rhs text), chained split
    text: dict[str, str]


def _parse_domain(domain: "Mapping[str, str] | None", params: Sequence[str]) -> _Domain:
    kinds: dict[str, set[str]] = {p: set() for p in params}
    relations: list[tuple[str, str, str]] = []
    text: dict[str, str] = {}
    if domain is None:
        return _Domain(kinds, relations, text)
    if not isinstance(domain, Mapping):
        raise DomainError("check_code: domain must be a mapping like {'n': 'integer, n>=0'}")
    for var, spec in domain.items():
        if var not in kinds:
            raise DomainError(f"check_code: domain names {var!r}, which is not a parameter")
        if not isinstance(spec, str):
            raise DomainError(f"check_code: domain for {var!r} must be text, not {spec!r}")
        text[var] = spec
        for piece in (p.strip() for p in spec.split(",")):
            if not piece:
                continue
            word = _TYPE_WORDS.get(piece.lower())
            if word:
                kinds[var].add(word)
                continue
            parts = [p.strip() for p in _REL.split(piece)]
            if len(parts) < 3 or len(parts) % 2 == 0 or not all(parts):
                raise DomainError(f"check_code: cannot read domain condition {piece!r}")
            for i in range(0, len(parts) - 2, 2):
                relations.append((parts[i], parts[i + 1], parts[i + 2]))
    return _Domain(kinds, relations, text)


def _relation(lhs: sp.Expr, op: str, rhs: sp.Expr) -> sp.Basic:
    return {
        "<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge, "==": sp.Eq, "!=": sp.Ne,
    }[op](lhs, rhs)


# --------------------------------------------------------------------------- #
# the reader                                                                  #
# --------------------------------------------------------------------------- #

_BUILTIN_NAMES = {"abs", "min", "max", "round", "int", "float", "pow", "sum", "range", "divmod"}
_UNDEF = sp.nan


@dataclass
class _Path:
    cond: sp.Basic
    env: dict[str, sp.Basic]


class _Reader:
    """Symbolic interpretation of a Python function's AST.

    Supported: positional / keyword parameters (with defaults), assignment, tuple
    assignment (simultaneous, ``a, b = b, a + b``), augmented assignment,
    ``+ - * / // % **``, unary ``- + not``, ``<< >>`` by constants, comparisons
    (chained), ``and``/``or``/``not`` in conditions, ``abs min max round int float
    pow divmod``, every ``math.*``/``numpy.*`` name in ``_ast_rules``, ``if/elif/
    else``, conditional expressions, early ``return``, ``raise``/``assert`` (the
    path then has no value: ``nan``), ``for v in range(a[, b[, step]])`` loops whose
    updates are additive, multiplicative, affine, last-value or a constant-
    coefficient linear system (Fibonacci) — else unrolled when the trip count is a
    small constant — ``sum``/``math.prod``/``math.fsum`` over a generator on
    ``range``, calls to other functions of the same module (inlined), module-level
    numeric constants.

    Semantics: values are exact reals/integers. ``/`` is real division, ``//`` is
    ``floor(a/b)``, ``%`` is SymPy ``Mod`` (sign of the divisor, as Python), a float
    literal means the decimal written (``0.1`` is 1/10), ``int(x)`` truncates,
    ``round`` is half-to-even. ``^`` is Python's XOR — refused as
    ``suspicious_xor`` (it is almost always a mistaken ``**``).
    """

    def __init__(self, src: _Source, symbols: dict[str, sp.Symbol]) -> None:
        self.src = src
        self.symbols = symbols
        self.notes: list[Note] = []
        self.defs = {n.name: n for n in src.tree.body if isinstance(n, ast.FunctionDef)}
        self.aliases: dict[str, tuple[str, str | None]] = {}  # local -> (module, attr)
        self.constants: dict[str, sp.Basic] = {}
        self.stack: list[str] = []
        self.counter = itertools.count()
        for node in src.tree.body:
            self._module_stmt(node)

    # ---- module level ------------------------------------------------------

    def _module_stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                self.aliases[a.asname or root] = (a.name if a.asname else root, None)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                self.aliases[a.asname or a.name] = (node.module, a.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = self.expr(node.value, {})
            except (CodeReadError, _NotReducible):
                return
            if not value.free_symbols:
                self.constants[node.targets[0].id] = value

    # ---- entry -------------------------------------------------------------

    def read(self, func: ast.FunctionDef, args: dict[str, sp.Basic]) -> sp.Basic:
        if func.name in self.stack:
            raise CodeReadError("recursion", func.lineno, f"{func.name} calls itself")
        if len(self.stack) >= _MAX_INLINE:
            raise CodeReadError("call chain deeper than 8", func.lineno)
        self.stack.append(func.name)
        try:
            paths = [_Path(sp.true, dict(args))]
            done: list[tuple[sp.Basic, sp.Basic]] = []
            body = func.body
            first = body[0] if body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                body = body[1:]
            paths = self.block(body, paths, done)
            for p in paths:
                done.append((p.cond, _UNDEF))
                self._note("undefined_path", f"{func.name}: a path reaches the end without return (returns None)")
            return _piecewise(done)
        finally:
            self.stack.pop()

    def _note(self, code: str, message: str) -> None:
        if not any(n.code == code and n.message == message for n in self.notes):
            self.notes.append(Note(code, message))

    # ---- statements --------------------------------------------------------

    def block(self, stmts: Sequence[ast.stmt], paths: list[_Path], done: list) -> list[_Path]:
        for st in stmts:
            if not paths:
                break
            paths = self.stmt(st, paths, done)
            if len(paths) + len(done) > _MAX_PATHS:
                raise CodeReadError(f"more than {_MAX_PATHS} branch paths", st.lineno)
        return paths

    def stmt(self, st: ast.stmt, paths: list[_Path], done: list) -> list[_Path]:
        if isinstance(st, ast.Return):
            for p in paths:
                if st.value is None:
                    done.append((p.cond, _UNDEF))
                    self._note("undefined_path", f"line {st.lineno}: bare return gives None")
                else:
                    done.append((p.cond, self.value(st.value, p.env)))
            return []
        if isinstance(st, ast.Raise):
            for p in paths:
                done.append((p.cond, _UNDEF))
            self._note("undefined_path", f"line {st.lineno}: raise — no value on that path")
            return []
        if isinstance(st, ast.Assert):
            out = []
            for p in paths:
                c = self.cond(st.test, p.env)
                if c is not sp.true:
                    done.append((sp.And(p.cond, sp.Not(c)), _UNDEF))
                if c is not sp.false:
                    out.append(_Path(sp.And(p.cond, c), p.env))
            self._note("undefined_path", f"line {st.lineno}: assert — no value where it fails")
            return out
        if isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for p in paths:
                self.assign(st, p.env)
            return paths
        if isinstance(st, ast.If):
            out = []
            for p in paths:
                c = self.cond(st.test, p.env)
                if c is not sp.false:
                    out += self.block(st.body, [_Path(_and(p.cond, c), dict(p.env))], done)
                if c is not sp.true:
                    out += self.block(st.orelse, [_Path(_and(p.cond, _not(c)), dict(p.env))], done)
            return out
        if isinstance(st, ast.For):
            for p in paths:
                self.loop(st, p.env)
            return paths
        if isinstance(st, ast.Pass):
            return paths
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            return paths  # a stray string / docstring
        raise CodeReadError(_construct_name(st), getattr(st, "lineno", None))

    def assign(self, st: ast.stmt, env: dict[str, sp.Basic]) -> None:
        if isinstance(st, ast.AnnAssign):
            if st.value is None:
                return
            self._bind(st.target, self.value(st.value, env), env, st.lineno)
            return
        if isinstance(st, ast.AugAssign):
            if not isinstance(st.target, ast.Name):
                raise CodeReadError("augmented assignment to " + _construct_name(st.target), st.lineno)
            cur = self.name(st.target.id, env, st.lineno)
            env[st.target.id] = self.binop(st.op, cur, self.value(st.value, env), st.lineno)
            return
        assert isinstance(st, ast.Assign)
        value_node = st.value
        if isinstance(value_node, ast.Tuple):
            values: "sp.Basic | list[sp.Basic]" = [self.value(e, env) for e in value_node.elts]
        elif isinstance(value_node, ast.Call) and _is_divmod(value_node):
            a, b = (self.value(e, env) for e in value_node.args)
            values = [sp.floor(a / b), sp.Mod(a, b)]
        else:
            values = self.value(value_node, env)
        for target in st.targets:
            if isinstance(target, ast.Tuple):
                if not isinstance(values, list) or len(values) != len(target.elts):
                    raise CodeReadError("tuple unpacking of a non-tuple", st.lineno)
                for t, v in zip(target.elts, values):
                    self._bind(t, v, env, st.lineno)
            else:
                if isinstance(values, list):
                    raise CodeReadError("tuple value", st.lineno, "functions must return one number")
                self._bind(target, values, env, st.lineno)

    def _bind(self, target: ast.expr, value: sp.Basic, env: dict[str, sp.Basic], line: int) -> None:
        if not isinstance(target, ast.Name):
            raise CodeReadError("assignment to " + _construct_name(target), line)
        env[target.id] = value

    # ---- loops -------------------------------------------------------------

    def _range(self, call: ast.expr, env: dict[str, sp.Basic]) -> tuple[sp.Basic, sp.Basic, int]:
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range"
                and "range" not in env and not call.keywords and 1 <= len(call.args) <= 3):
            raise CodeReadError("loop over something other than range(...)", getattr(call, "lineno", None))
        args = [self.value(a, env) for a in call.args]
        if len(args) == 1:
            a, b, step = sp.Integer(0), args[0], sp.Integer(1)
        elif len(args) == 2:
            a, b, step = args[0], args[1], sp.Integer(1)
        else:
            a, b, step = args
        if not (step.is_Integer and step != 0):
            raise CodeReadError("range step that is not a nonzero integer constant", call.lineno)
        for bound in (a, b):
            if bound.is_integer is False or (bound.is_number and not bound.is_Integer):
                raise CodeReadError("range bound that is not an integer", call.lineno)
        return a, b, int(step)

    def loop(self, st: ast.For, env: dict[str, sp.Basic]) -> None:
        if st.orelse:
            raise CodeReadError("for-else", st.lineno)
        if not isinstance(st.target, ast.Name):
            raise CodeReadError("loop target that is not a single name", st.lineno)
        var = st.target.id
        if var in self.symbols:
            raise CodeReadError("loop variable shadowing a parameter", st.lineno)
        a, b, step = self._range(st.iter, env)
        carried = sorted(_assigned_names(st.body) - {var})
        if var in _assigned_names(st.body):
            raise CodeReadError("assignment to the loop variable inside the loop", st.lineno)
        for node in ast.walk(ast.Module(body=st.body, type_ignores=[])):
            if isinstance(node, (ast.Return, ast.Break, ast.Continue, ast.Raise)):
                raise _NotReducible(st.lineno, f"loop with {type(node).__name__.lower()} inside")
        trips = _trip_count(a, b, step)
        try:
            closed = self._close_loop(st, env, var, a, b, step, carried)
        except _NotReducible:
            if trips.is_Integer and int(trips) <= _MAX_UNROLL:
                closed = self._unroll(st, env, var, a, step, int(trips), carried)
            else:
                raise
        env.update(closed)
        env[var] = _POISON  # Python leaves it set; reading it is refused

    def _unroll(self, st, env, var, a, step, trips, carried) -> dict[str, sp.Basic]:
        work = dict(env)
        for t in range(max(trips, 0)):
            work[var] = a + step * t
            paths = self.block(st.body, [_Path(sp.true, work)], [])
            work = _merge_paths(paths, carried, st.lineno)
            work[var] = a + step * t
        return {v: work[v] for v in carried if v in work}

    def _close_loop(self, st, env, var, a, b, step, carried) -> dict[str, sp.Basic]:
        k = next(self.counter)
        tau = sp.Symbol(f"{var}" if step == 1 else f"_t{k}", integer=True)
        if step == 1:
            i_of_tau, tau0, tau_end = tau, a, b
            nonempty = sp.Ge(b, a)
        else:
            i_of_tau, tau0 = a + step * tau, sp.Integer(0)
            tau_end = _trip_count(a, b, step, guarded=False)
            nonempty = sp.Ge(b, a) if step > 0 else sp.Ge(a, b)
        starts = {v: sp.Dummy(f"{v}_start") for v in carried}
        body_env = dict(env)
        for v in carried:
            if v not in env:
                body_env[v] = starts[v]  # assigned before use inside the body, hopefully
            else:
                body_env[v] = starts[v]
        body_env[var] = i_of_tau
        paths = self.block(st.body, [_Path(sp.true, body_env)], [])
        merged = _merge_paths(paths, carried, st.lineno)
        updates = {v: merged[v] for v in carried if v in merged}
        init = {v: env.get(v) for v in carried}
        start_syms = {starts[v]: v for v in carried}

        known: dict[str, Callable[[sp.Basic], sp.Basic]] = {}
        pending = [v for v in carried if v in updates and updates[v] != starts[v]]
        for v in carried:
            if v not in pending:
                known[v] = _constant_value(init[v] if init[v] is not None else starts[v])

        def subst_known(e: sp.Basic, at: sp.Basic) -> sp.Basic:
            rep = {starts[w]: f(at) for w, f in known.items() if starts[w] in e.free_symbols}
            return e.xreplace(rep) if rep else e

        progress = True
        while pending and progress:
            progress = False
            for v in list(pending):
                u = updates[v]
                deps = {start_syms[s] for s in u.free_symbols if s in start_syms} - set(known)
                if deps - {v}:
                    continue
                if init[v] is None and starts[v] in u.free_symbols:
                    raise _NotReducible(st.lineno, f"{v} is used in the loop before it is assigned")
                known[v] = self._close_one(v, u, starts[v], init[v], tau, tau0, subst_known, st.lineno)
                pending.remove(v)
                progress = True
        if pending:
            known.update(self._close_linear(pending, updates, starts, init, tau, tau0, subst_known, st.lineno))
        out = {}
        for v in carried:
            if v not in updates:
                continue
            final = known[v](tau_end)
            before = init[v] if init[v] is not None else _UNDEF
            out[v] = sp.Piecewise((final, nonempty), (before, True))
        return out

    def _close_one(self, v, u, s, init, tau, tau0, subst_known, line):
        j = sp.Dummy("j", integer=True)
        u_j = subst_known(u, j).xreplace({tau: j}) if tau in u.free_symbols else subst_known(u, j)
        # the update as a function of the iteration index j (value AFTER iteration j)
        if s not in u_j.free_symbols:
            def last(t, u_j=u_j, j=j, init=init):
                return sp.Piecewise((u_j.xreplace({j: t - 1}), sp.Gt(t, tau0)),
                                    (init if init is not None else _UNDEF, True))
            return last
        h = sp.expand(_fold(u_j - s))
        if s not in h.free_symbols:
            return lambda t, h=h, j=j, init=init: init + _bounded_op(sp.Sum, h, j, tau0, t - 1)
        q = _fold(u_j / s)
        q = q if isinstance(q, sp.Piecewise) else sp.cancel(q)
        if s not in q.free_symbols and sp.expand(_fold(q * s - u_j)) == 0:
            return lambda t, q=q, j=j, init=init: init * _bounded_op(sp.Product, q, j, tau0, t - 1)
        poly = sp.Poly(sp.expand(u_j), s) if sp.expand(u_j).is_polynomial(s) else None
        if poly is not None and poly.degree() == 1:
            c, d = poly.coeff_monomial(s), poly.coeff_monomial(1)
            if j not in c.free_symbols:
                return lambda t, c=c, d=d, j=j, init=init: (
                    c ** (t - tau0) * init + sp.Sum(c ** (t - 1 - j) * d, (j, tau0, t - 1)))
        raise _NotReducible(line, f"the update of {v} is not additive, multiplicative or affine")

    def _close_linear(self, group, updates, starts, init, tau, tau0, subst_known, line):
        syms = [starts[v] for v in group]
        rows, consts = [], []
        for v in group:
            u = sp.expand(subst_known(updates[v], tau))
            if tau in u.free_symbols or init[v] is None:
                raise _NotReducible(line, "a coupled update that depends on the loop index")
            row = []
            rest = u
            for s in syms:
                c = u.coeff(s)
                if c.free_symbols & set(syms) or tau in c.free_symbols:
                    raise _NotReducible(line, "a coupled update that is not linear")
                row.append(c)
                rest -= c * s
            rest = sp.expand(rest)
            if rest.free_symbols & set(syms):
                raise _NotReducible(line, "a coupled update that is not linear")
            rows.append(row)
            consts.append(rest)
        n = len(group)
        M = sp.Matrix([r + [c] for r, c in zip(rows, consts)] + [[0] * n + [1]])
        x0 = sp.Matrix([init[v] for v in group] + [1])
        K = sp.Symbol("_K", integer=True, nonnegative=True)
        try:
            power = M**K
        except Exception as exc:  # noqa: BLE001 — SymPy has no closed power here
            raise _NotReducible(line, "a coupled update with no closed matrix power") from exc
        if isinstance(power, sp.MatrixExpr) or any(isinstance(x, sp.MatrixExpr) for x in sp.preorder_traversal(power)):
            raise _NotReducible(line, "a coupled update with no closed matrix power")
        state = power * x0
        out = {}
        for idx, v in enumerate(group):
            e = state[idx]
            out[v] = lambda t, e=e: e.xreplace({K: t - tau0})
        return out

    # ---- expressions -------------------------------------------------------

    def value(self, node: ast.expr, env: dict[str, sp.Basic]) -> sp.Basic:
        v = self.expr(node, env)
        if isinstance(v, sp.logic.boolalg.Boolean) and not isinstance(v, sp.Expr):
            return sp.Piecewise((sp.Integer(1), v), (sp.Integer(0), True))
        return v

    def cond(self, node: ast.expr, env: dict[str, sp.Basic]) -> sp.Basic:
        v = self.expr(node, env)
        if isinstance(v, sp.Expr):
            if v.is_number and v.is_finite:
                return sp.true if v != 0 else sp.false
            return sp.Ne(v, 0)
        return v

    def name(self, ident: str, env: dict[str, sp.Basic], line: int) -> sp.Basic:
        if ident in env:
            v = env[ident]
            if v is _POISON:
                raise CodeReadError("loop variable read after its loop", line)
            return v
        if ident in self.constants:
            return self.constants[ident]
        if ident in self.aliases:
            mod, attr = self.aliases[ident]
            const = _module_constant(mod, attr)
            if const is not None:
                return const
        raise CodeReadError(f"name {ident!r} that is not a parameter, local or numeric constant", line)

    def expr(self, node: ast.expr, env: dict[str, sp.Basic]) -> sp.Basic:
        line = getattr(node, "lineno", None)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return sp.true if v else sp.false
            if isinstance(v, int):
                return sp.Integer(v)
            if isinstance(v, float):
                if not math.isfinite(v):
                    raise CodeReadError("non-finite float constant", line)
                return sp.Rational(repr(v))
            raise CodeReadError(f"{type(v).__name__} constant", line)
        if isinstance(node, ast.Name):
            return self.name(node.id, env, line or 0)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.BitXor):
                self._note("suspicious_xor", f"line {line}: `^` is bitwise XOR in Python, not a power — did you mean `**`?")
                raise CodeReadError("bitwise ^ (XOR)", line, "did you mean ** ?")
            return self.binop(node.op, self.value(node.left, env), self.value(node.right, env), line)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return _not(self.cond(node.operand, env))
            x = self.value(node.operand, env)
            if isinstance(node.op, ast.USub):
                return -x
            if isinstance(node.op, ast.UAdd):
                return x
            if isinstance(node.op, ast.Invert):
                return -x - 1
        if isinstance(node, ast.Compare):
            parts = []
            left = self.value(node.left, env)
            for op, comp in zip(node.ops, node.comparators):
                right = self.value(comp, env)
                sym = _CMP.get(type(op))
                if sym is None:
                    raise CodeReadError(f"comparison {type(op).__name__}", line)
                parts.append(_relation(left, sym, right))
                left = right
            return sp.And(*parts)
        if isinstance(node, ast.BoolOp):
            vals = [self.cond(v, env) for v in node.values]
            return sp.And(*vals) if isinstance(node.op, ast.And) else sp.Or(*vals)
        if isinstance(node, ast.IfExp):
            c = self.cond(node.test, env)
            if c is sp.true:
                return self.value(node.body, env)
            if c is sp.false:
                return self.value(node.orelse, env)
            return sp.Piecewise((self.value(node.body, env), c), (self.value(node.orelse, env), True))
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in self.aliases and node.value.id not in env:
                mod, attr = self.aliases[node.value.id]
                full = mod if attr is None else f"{mod}.{attr}"
                const = _module_constant(full, node.attr)
                if const is not None:
                    return const
            raise CodeReadError(f"attribute {ast.unparse(node)}", line)
        if isinstance(node, ast.Call):
            return self.call(node, env)
        raise CodeReadError(_construct_name(node), line)

    def binop(self, op: ast.operator, a: sp.Basic, b: sp.Basic, line: "int | None") -> sp.Basic:
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            return a / b
        if isinstance(op, ast.FloorDiv):
            return sp.floor(a / b)
        if isinstance(op, ast.Mod):
            return sp.Mod(a, b)
        if isinstance(op, ast.Pow):
            return a**b
        if isinstance(op, (ast.LShift, ast.RShift)) and b.is_Integer and b >= 0:
            return a * 2**b if isinstance(op, ast.LShift) else sp.floor(a / 2**b)
        if isinstance(op, ast.BitXor):
            self._note("suspicious_xor", f"line {line}: `^` is bitwise XOR in Python, not a power — did you mean `**`?")
            raise CodeReadError("bitwise ^ (XOR)", line, "did you mean ** ?")
        raise CodeReadError(f"operator {type(op).__name__}", line)

    def call(self, node: ast.Call, env: dict[str, sp.Basic]) -> sp.Basic:
        line = node.lineno
        f = node.func
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            raise CodeReadError("*args / **kwargs in a call", line)
        if isinstance(f, ast.Name) and f.id not in env:
            if f.id in self.defs:
                return self._inline(self.defs[f.id], node, env)
            if f.id in self.aliases:
                mod, attr = self.aliases[f.id]
                return self._module_call(mod, attr, node, env)
            if f.id in _BUILTIN_NAMES:
                return self._builtin(f.id, node, env)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in self.aliases \
                and f.value.id not in env:
            mod, attr = self.aliases[f.value.id]
            full = mod if attr is None else f"{mod}.{attr}"
            return self._module_call(full, f.attr, node, env)
        raise CodeReadError(f"call to {ast.unparse(f)}", line)

    def _args(self, node: ast.Call, env) -> list[sp.Basic]:
        if node.keywords:
            raise CodeReadError(f"keyword arguments to {ast.unparse(node.func)}", node.lineno)
        return [self.value(a, env) for a in node.args]

    def _module_call(self, mod: str, attr: "str | None", node: ast.Call, env) -> sp.Basic:
        root = mod.split(".")[0]
        table = MATH_FUNCS if root == "math" else NUMPY_FUNCS if root == "numpy" else None
        if table is None or attr is None:
            raise CodeReadError(f"call to {mod}{'.' + attr if attr else ''}", node.lineno)
        if root == "math" and attr in ("prod", "fsum"):
            return self._reduce("prod" if attr == "prod" else "sum", node, env)
        if root == "numpy" and attr in ("sum", "prod"):
            return self._reduce(attr, node, env)
        if attr not in table:
            raise CodeReadError(f"{root}.{attr} (not in the supported table)", node.lineno)
        build, arities = table[attr]
        args = self._args(node, env)
        if len(args) not in arities:
            raise CodeReadError(f"{root}.{attr} with {len(args)} arguments", node.lineno)
        return build(args)

    def _builtin(self, fname: str, node: ast.Call, env) -> sp.Basic:
        line = node.lineno
        if fname == "sum":
            return self._reduce("sum", node, env)
        args = self._args(node, env)
        if fname == "abs" and len(args) == 1:
            return sp.Abs(args[0])
        if fname in ("min", "max") and len(args) >= 2:
            return (sp.Min if fname == "min" else sp.Max)(*args)
        if fname == "round" and len(args) in (1, 2):
            if len(args) == 1:
                return py_round(args[0])
            scale = sp.Integer(10) ** args[1]
            return py_round(args[0] * scale) / scale
        if fname == "int" and len(args) == 1:
            return py_trunc(args[0])
        if fname == "float" and len(args) == 1:
            return args[0]
        if fname == "pow" and len(args) == 2:
            return args[0] ** args[1]
        raise CodeReadError(f"{fname}() with {len(args)} arguments", line)

    def _reduce(self, kind: str, node: ast.Call, env) -> sp.Basic:
        """``sum``/``prod`` over ``range(...)`` or a generator / comprehension on one."""
        line = node.lineno
        if len(node.args) != 1 or node.keywords:
            raise CodeReadError(f"{kind}() with a start value or several arguments", line)
        arg = node.args[0]
        if isinstance(arg, ast.Call):  # sum(range(a, b))
            a, b, step = self._range(arg, env)
            k = sp.Dummy("k", integer=True)
            term, conds = a + step * k, list[sp.Basic]()
            trips = _trip_count(a, b, step, guarded=False)
            return self._fold(kind, term, k, trips, conds, a, b, step)
        if not isinstance(arg, (ast.GeneratorExp, ast.ListComp)) or len(arg.generators) != 1:
            raise CodeReadError(f"{kind}() over something other than one generator on range", line)
        gen = arg.generators[0]
        if not isinstance(gen.target, ast.Name) or gen.is_async:
            raise CodeReadError("generator target that is not a name", line)
        a, b, step = self._range(gen.iter, env)
        k = sp.Dummy("k", integer=True)
        inner = dict(env)
        inner[gen.target.id] = a + step * k
        term = self.value(arg.elt, inner)
        conds = [self.cond(c, inner) for c in gen.ifs]
        return self._fold(kind, term, k, _trip_count(a, b, step, guarded=False), conds, a, b, step)

    def _fold(self, kind, term, k, trips, conds, a, b, step) -> sp.Basic:
        if conds:
            c = sp.And(*conds)
            term = sp.Piecewise((term, c), (0 if kind == "sum" else 1, True))
        nonempty = sp.Ge(b, a) if step > 0 else sp.Ge(a, b)
        op = sp.Sum if kind == "sum" else sp.Product
        empty = sp.Integer(0 if kind == "sum" else 1)
        if step == 1:
            i = sp.Dummy("i", integer=True)
            closed = _bounded_op(op, term.xreplace({k: i - a}), i, a, b - 1)
        else:
            closed = _bounded_op(op, term, k, 0, trips - 1)
        return sp.Piecewise((closed, nonempty), (empty, True))

    def _inline(self, fdef: ast.FunctionDef, node: ast.Call, env) -> sp.Basic:
        params, defaults = _signature(fdef)
        bound: dict[str, sp.Basic] = {}
        if len(node.args) > len(params):
            raise CodeReadError(f"too many arguments to {fdef.name}", node.lineno)
        for p, a in zip(params, node.args):
            bound[p] = self.value(a, env)
        for kw in node.keywords:
            if kw.arg not in params or kw.arg in bound:
                raise CodeReadError(f"bad keyword {kw.arg} to {fdef.name}", node.lineno)
            bound[kw.arg] = self.value(kw.value, env)
        for p in params:
            if p not in bound:
                if p not in defaults:
                    raise CodeReadError(f"missing argument {p} to {fdef.name}", node.lineno)
                bound[p] = self.expr(defaults[p], {})
        return self.read(fdef, bound)


_POISON = sp.Symbol("__pcm_loop_variable_after_loop__")


def _fold(e: sp.Basic) -> sp.Basic:
    """``piecewise_fold``, but never across a ``Sum``/``Product`` whose body has a
    ``Piecewise`` (that would move a condition on the bound variable outside)."""
    if any(x.has(sp.Piecewise) for x in e.atoms(sp.Sum, sp.Product)):
        return e
    return sp.piecewise_fold(e)


def _bounded_op(op: type, f: sp.Basic, j: sp.Symbol, lo: sp.Basic, hi: sp.Basic) -> sp.Basic:
    """``op(f, (j, lo, hi))`` with every condition on ``j`` inside ``f`` that the
    range ``lo <= j <= hi`` decides replaced by its truth value. Keeps a loop
    closed inside another loop (``range(i, n)``) free of ``Piecewise`` arms that
    can never be taken — which ``piecewise_fold`` would otherwise drag outside the
    sum, leaving the bound variable free."""
    def decide(rel: sp.Basic) -> sp.Basic:
        if not isinstance(rel, (sp.Lt, sp.Le, sp.Gt, sp.Ge)) or j not in rel.free_symbols:
            return rel
        e = sp.expand(rel.lhs - rel.rhs)  # rel is  e < 0, e <= 0, e > 0 or e >= 0
        if not e.is_polynomial(j) or sp.degree(e, j) != 1:
            return rel
        c = e.coeff(j)
        if not c.is_number:
            return rel
        e_max = e.xreplace({j: hi if c > 0 else lo})
        e_min = e.xreplace({j: lo if c > 0 else hi})
        def holds(x: sp.Basic) -> "bool | None":
            v = x.is_nonpositive if isinstance(rel, sp.Le) else x.is_negative if isinstance(rel, sp.Lt) \
                else x.is_nonnegative if isinstance(rel, sp.Ge) else x.is_positive
            return v
        if isinstance(rel, (sp.Le, sp.Lt)):
            if holds(e_max):
                return sp.true
            if (e_min.is_positive if isinstance(rel, sp.Le) else e_min.is_nonnegative):
                return sp.false
        else:
            if holds(e_min):
                return sp.true
            if (e_max.is_negative if isinstance(rel, sp.Ge) else e_max.is_nonpositive):
                return sp.false
        return rel

    def walk(x: sp.Basic) -> sp.Basic:
        if isinstance(x, sp.Piecewise):
            arms = [(walk(v), c.replace(lambda r: isinstance(r, sp.Rel), decide)  # type: ignore[arg-type]
                     if isinstance(c, sp.Basic) else c) for v, c in x.args]
            return sp.Piecewise(*arms)
        if x.args and not isinstance(x, (sp.Sum, sp.Product)):
            return x.func(*[walk(a) for a in x.args])
        return x

    try:
        f = walk(f)
    except Exception:  # noqa: BLE001 — refinement is optional
        pass
    return op(f, (j, lo, hi))


def _constant_value(v: sp.Basic) -> Callable[[sp.Basic], sp.Basic]:
    """A loop-carried value that the body never changes, as a function of the index."""
    return lambda _t: v
_CMP = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}


def _and(a: sp.Basic, b: sp.Basic) -> sp.Basic:
    return sp.And(a, b)


def _not(c: sp.Basic) -> sp.Basic:
    return sp.Not(c)


def _is_divmod(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "divmod" and len(node.args) == 2


def _piecewise(pairs: list[tuple[sp.Basic, sp.Basic]]) -> sp.Basic:
    pairs = [(v, c) for c, v in pairs if c is not sp.false]
    if not pairs:
        return _UNDEF
    if pairs[0][1] is sp.true or len(pairs) == 1 and pairs[0][1] is sp.true:
        return pairs[0][0]
    if all(v == pairs[0][0] for v, _ in pairs):
        return pairs[0][0]
    last_v, _ = pairs[-1]
    body = [(v, c) for v, c in pairs[:-1]] + [(last_v, True)]
    return sp.Piecewise(*body)


def _merge_paths(paths: list[_Path], names: Iterable[str], line: int) -> dict[str, sp.Basic]:
    if len(paths) == 1:
        return dict(paths[0].env)
    out = dict(paths[0].env)
    for v in names:
        vals = [p.env.get(v) for p in paths]
        if any(x is None for x in vals):
            if all(x is None for x in vals):
                continue
            raise CodeReadError(f"variable {v!r} assigned on some branches only", line)
        out[v] = _piecewise([(p.cond, x) for p, x in zip(paths, vals)])  # type: ignore[misc]
    return out


def _assigned_names(stmts: Sequence[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for st in stmts:
        for node in ast.walk(st):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
    return names


def _trip_count(a: sp.Basic, b: sp.Basic, step: int, guarded: bool = True) -> sp.Basic:
    raw = sp.ceiling((b - a) / step)
    return sp.Max(0, raw) if guarded else raw


def _construct_name(node: ast.AST) -> str:
    names = {
        ast.While: "while loop", ast.List: "list", ast.Dict: "dict", ast.Set: "set",
        ast.ListComp: "list comprehension", ast.DictComp: "dict comprehension",
        ast.SetComp: "set comprehension", ast.GeneratorExp: "generator expression",
        ast.Subscript: "subscript / indexing", ast.Lambda: "lambda", ast.Global: "global",
        ast.Nonlocal: "nonlocal", ast.Try: "try/except", ast.With: "with block",
        ast.Delete: "del", ast.Break: "break", ast.Continue: "continue",
        ast.FunctionDef: "nested function", ast.ClassDef: "class", ast.Import: "import inside a function",
        ast.ImportFrom: "import inside a function", ast.Tuple: "tuple", ast.JoinedStr: "f-string",
        ast.Starred: "starred expression", ast.Yield: "yield", ast.Await: "await",
    }
    for cls, label in names.items():
        if isinstance(node, cls):
            return label
    if isinstance(node, ast.Expr):
        return "expression statement (side effect: " + ast.unparse(node)[:40] + ")"
    return type(node).__name__


def _module_constant(mod: str, attr: "str | None") -> "sp.Basic | None":
    if attr is None:
        return None
    root = mod.split(".")[0]
    if root == "math":
        return MATH_CONSTANTS.get(attr)
    if root == "numpy":
        return NUMPY_CONSTANTS.get(attr)
    return None


def _signature(fdef: ast.FunctionDef) -> tuple[list[str], dict[str, ast.expr]]:
    a = fdef.args
    if a.vararg or a.kwarg:
        raise CodeReadError("*args / **kwargs parameters", fdef.lineno)
    pos = [p.arg for p in a.posonlyargs + a.args]
    defaults = dict(zip(pos[len(pos) - len(a.defaults):], a.defaults))
    kwonly = [p.arg for p in a.kwonlyargs]
    for p, d in zip(kwonly, a.kw_defaults):
        if d is not None:
            defaults[p] = d
    return pos + kwonly, defaults


def _annotation_kind(fdef: ast.FunctionDef) -> dict[str, str]:
    out = {}
    a = fdef.args
    for p in a.posonlyargs + a.args + a.kwonlyargs:
        ann = p.annotation
        if isinstance(ann, ast.Name) and ann.id in ("int", "float"):
            out[p.arg] = "integer" if ann.id == "int" else "real"
        elif isinstance(ann, ast.Constant) and ann.value in ("int", "float"):
            out[p.arg] = "integer" if ann.value == "int" else "real"
    return out


def _range_params(fdef: ast.FunctionDef, params: Sequence[str]) -> set[str]:
    """Parameters that appear inside a ``range(...)`` call: they must be integers."""
    found: set[str] = set()
    for node in ast.walk(fdef):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id in params:
                    found.add(sub.id)
    return found


def _make_symbols(fdef: ast.FunctionDef, params: Sequence[str], dom: _Domain) -> dict[str, sp.Symbol]:
    ann = _annotation_kind(fdef)
    in_range = _range_params(fdef, params)
    syms = {}
    for p in params:
        kinds = dom.kinds.get(p, set())
        integer = "integer" in kinds or ("real" not in kinds and (ann.get(p) == "integer" or p in in_range))
        assumptions: dict[str, bool] = {"integer": True} if integer else {"real": True}
        for word in ("positive", "nonnegative", "negative", "nonpositive"):
            if word in kinds:
                assumptions[word] = True
        for lhs, op, rhs in dom.relations:  # x>0 / x>=0 with a constant become assumptions
            if lhs == p and rhs in ("0", "0.0"):
                if op == ">":
                    assumptions["positive"] = True
                elif op == ">=":
                    assumptions["nonnegative"] = True
            if rhs == p and lhs in ("0", "0.0"):
                if op == "<":
                    assumptions["positive"] = True
                elif op == "<=":
                    assumptions["nonnegative"] = True
        syms[p] = sp.Symbol(p, **assumptions)
    return syms


def to_symbolic(
    func_or_source: "Callable[..., Any] | str",
    name: "str | None" = None,
    *,
    domain: "Mapping[str, str] | None" = None,
) -> SymbolicFunction:
    """Read a Python function into SymPy, from its AST only — nothing is executed.

    >>> to_symbolic("def f(n: int):\\n    return n * (n + 1) // 2").expr
    floor(n*(n + 1)/2)

    Raises ``CodeReadError`` (an ``UnsupportedFormError``) naming the construct
    and line for anything outside the supported subset (see ``_Reader``).
    """
    src = _load(func_or_source, name)
    return _read(src, domain)


def _read(src: _Source, domain: "Mapping[str, str] | None") -> SymbolicFunction:
    params, defaults_nodes = _signature(src.func)
    dom = _parse_domain(domain, params)
    syms = _make_symbols(src.func, params, dom)
    reader = _Reader(src, syms)
    try:
        expr = reader.read(src.func, dict(syms))
    except _NotReducible as exc:
        reader._note("loop_not_reducible", f"line {exc.line}: {exc.why}")
        err = CodeReadError("loop not reducible", exc.line, exc.why)
        err.notes = tuple(reader.notes)  # type: ignore[attr-defined]
        raise err from None
    except CodeReadError as exc:
        exc.notes = tuple(reader.notes)  # type: ignore[attr-defined]
        raise
    defaults = {}
    for p, d in defaults_nodes.items():
        try:
            defaults[p] = ast.literal_eval(d)
        except ValueError:
            pass
    expr = _tidy(expr)
    return SymbolicFunction(src.func.name, tuple(params), syms, expr, defaults, tuple(reader.notes))


def _tidy(expr: sp.Basic) -> sp.Basic:
    """Evaluate closed sums/products where SymPy can, quickly; keep the rest."""
    if not expr.has(sp.Sum, sp.Product):
        return expr
    try:
        if _caller_budget_active():
            done = expr.doit()
        else:
            with time_budget(2.0, "to_symbolic"):
                done = expr.doit()
    except (TimeBudgetError, Exception):  # noqa: BLE001 — keep the unevaluated form
        return expr
    return done


# --------------------------------------------------------------------------- #
# formulas                                                                    #
# --------------------------------------------------------------------------- #

#: Functions a formula may use beyond the parser's whitelist (C1 extension).
_EXTRA_FORMULA_FUNCS: "dict[str, Callable[..., sp.Expr]]" = {
    "fibonacci": sp.fibonacci,
    "lucas": sp.lucas,
    "harmonic": sp.harmonic,
    "catalan": sp.catalan,
    "comb": sp.binomial,
    "perm": lambda n, k: sp.ff(n, k),
}


def _parse_formula(formula: "str | sp.Expr | Expr | int | float", syms: Mapping[str, sp.Symbol]) -> sp.Expr:
    if isinstance(formula, Expr):
        sy = formula.sy
    elif isinstance(formula, sp.Expr):
        sy = formula
    elif isinstance(formula, (int, float)) and not isinstance(formula, bool):
        sy = sp.Rational(repr(formula)) if isinstance(formula, float) else sp.Integer(formula)
    elif isinstance(formula, str):
        sy = _parse_formula_text(_shield_names(formula, syms), {f"pcmvar{i}": n for i, n in enumerate(syms)})
    else:
        raise DomainError(f"check_code: formula must be text or an expression, not {type(formula).__name__}")
    if sy.has(sp.Float):
        sy = sp.nsimplify(sy, rational=True)
    if isinstance(formula, str):
        sy = sy.xreplace({sp.Symbol(f"pcmvar{i}"): sp.Symbol(n) for i, n in enumerate(syms)})
    by_name = {s.name: s for s in sy.free_symbols if isinstance(s, sp.Symbol)}
    unknown = sorted(set(by_name) - set(syms))
    if unknown:
        raise DomainError(
            f"check_code: the formula uses {', '.join(unknown)}, which "
            f"{'is not a parameter' if len(unknown) == 1 else 'are not parameters'} of the function"
        )
    return sy.xreplace({by_name[n]: syms[n] for n in by_name})


def _shield_names(text: str, syms: Mapping[str, sp.Symbol]) -> str:
    """Parameter names spelled as placeholders the parser keeps whole.

    The project parser splits an unknown multi-letter name into a product
    (``rate`` -> ``r*a*t*e``) and reads ``exp`` as the function; in a formula
    written against a function's parameters, a parameter name means the parameter.
    """
    names = list(syms)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return text
    out, last = [], (1, 0)
    lines = text.splitlines(keepends=True) or [""]
    def offset(pos: tuple[int, int]) -> int:
        return sum(len(l) for l in lines[: pos[0] - 1]) + pos[1]
    for i, tok in enumerate(toks):
        if tok.type == tokenize.NAME and tok.string in names:
            nxt = toks[i + 1].string if i + 1 < len(toks) else ""
            if nxt == "(" and tok.string not in ("e",):
                if tok.string in _EXTRA_FORMULA_FUNCS:
                    continue
            out.append(text[offset(last):offset(tok.start)])
            out.append(f"pcmvar{names.index(tok.string)}")
            last = tok.end
    out.append(text[offset(last):])
    return "".join(out)


def _parse_formula_text(text: str, shielded: "Mapping[str, str] | None" = None) -> sp.Expr:
    """The project parser, plus ``fibonacci(...)`` & co. spliced in as placeholders."""
    placeholders: dict[str, sp.Expr] = {}
    out = text
    for _ in range(32):
        m = re.search(r"\b(" + "|".join(_EXTRA_FORMULA_FUNCS) + r")\s*\(", out)
        if not m:
            break
        depth, j = 0, m.end() - 1
        for j in range(m.end() - 1, len(out)):
            depth += {"(": 1, ")": -1}.get(out[j], 0)
            if depth == 0:
                break
        if depth:
            raise ParseError(f"check_code: unbalanced parentheses in formula {text!r}")
        inner = out[m.end():j]
        args = [_parse_formula_text(a, shielded) for a in _split_args(inner)]
        key = f"pcmext{len(placeholders)}"
        placeholders[key] = _EXTRA_FORMULA_FUNCS[m.group(1)](*args)
        out = out[: m.start()] + f"({key})" + out[j + 1:]
    sy = _parse_written(out)
    if placeholders:
        sy = sy.xreplace({sp.Symbol(k): v for k, v in placeholders.items()})
    return sy


def _split_args(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        depth += {"(": 1, ")": -1}.get(ch, 0)
        cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# exact numbers                                                               #
# --------------------------------------------------------------------------- #


def _exact_at(e: sp.Basic, point: Mapping[sp.Symbol, sp.Basic]) -> "sp.Expr | None":
    """``e`` at ``point``, exactly where SymPy can; ``None`` where it has no real
    value there (pole, ``nan`` path, complex)."""
    try:
        v = e.xreplace(dict(point))
        if v.has(sp.Sum, sp.Product, sp.Piecewise):
            v = v.doit()
        v = sp.sympify(v)
        if v.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or v.free_symbols:
            return None
        if v.is_real is False:
            return None
        if not v.is_Rational:
            val = sp.N(v, 60)
            if not val.is_real or val.has(sp.I):
                im = sp.im(sp.N(v, 60))
                if im != 0:
                    return None
        return v
    except Exception:  # noqa: BLE001 — a point SymPy cannot evaluate has no value here
        return None


def _mp(v: sp.Expr, dps: int = 60) -> "mpmath.mpf":
    with mpmath.workdps(dps):
        return mpmath.mpf(sp.Float(sp.N(v, dps), dps))


def _same(a: sp.Expr, b: sp.Expr) -> "bool | None":
    """Exactly equal? ``True``/``False``, ``None`` when 60 vs 100 digits disagree."""
    if a.is_Rational and b.is_Rational:
        return bool(a == b)
    d = sp.N(a - b, 60)
    if d == 0:
        return True
    scale = max(abs(sp.N(a, 30)), abs(sp.N(b, 30)), sp.Integer(1))
    if abs(d) <= sp.Float("1e-45") * scale:
        d2 = sp.N(a - b, 100)
        return True if abs(d2) <= sp.Float("1e-85") * scale else None
    d2 = sp.N(a - b, 100)
    if abs(d2 - d) > abs(d) * sp.Float("1e-20"):
        return None
    return False


# --------------------------------------------------------------------------- #
# sample points                                                               #
# --------------------------------------------------------------------------- #

_INT_POINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, 12, 15, -5, 20, 11, 13, -10, 30, 50]
_REAL_POINTS = [sp.Rational(v) for v in (
    "0", "1", "-1", "2", "1/2", "-1/2", "3", "3/2", "-2", "1/4", "5/2", "-3/2", "7/4",
    "10", "1/8", "-5/4", "5", "-3", "3/8", "100", "-7/2", "13/16", "6")]


def _bounds(sym: sp.Symbol, rels: Sequence[sp.Basic]) -> tuple["int | None", "int | None"]:
    lo: "int | None" = None
    hi: "int | None" = None
    if sym.is_nonnegative:
        lo = 0
    if sym.is_positive:
        lo = 1 if sym.is_integer else 0
    for r in rels:
        if r.free_symbols != {sym} or not isinstance(r, (sp.Ge, sp.Gt, sp.Le, sp.Lt)):
            continue
        try:
            sol = sp.solve_univariate_inequality(r, sym, relational=False)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(sol, sp.Interval):
            if sol.inf.is_finite:
                c = int(sp.ceiling(sol.inf))
                lo = c if lo is None else max(lo, c)
            if sol.sup.is_finite:
                c = int(sp.floor(sol.sup))
                hi = c if hi is None else min(hi, c)
    return lo, hi


def _points(syms: Sequence[sp.Symbol], rels: Sequence[sp.Basic], limit: int = 96) -> list[dict[sp.Symbol, sp.Basic]]:
    """Deterministic domain points, the most readable first."""
    per: list[list[sp.Basic]] = []
    for s in syms:
        lo, hi = _bounds(s, rels)
        if s.is_integer:
            vals = list(_INT_POINTS)
            if lo is not None and lo > 3:
                vals = [lo, lo + 1, lo + 2, lo + 3, lo + 5, lo + 10] + vals
            vals = [v for v in vals if (lo is None or v >= lo) and (hi is None or v <= hi)]
            cand: list[sp.Basic] = [sp.Integer(v) for v in dict.fromkeys(vals)]
        else:
            cand = [v for v in _REAL_POINTS
                    if (lo is None or v >= lo) and (hi is None or v <= hi)
                    and not (s.is_positive and v <= 0) and not (s.is_negative and v >= 0)
                    and not (s.is_nonnegative and v < 0)]
        if not cand:
            cand = [sp.Integer(lo if lo is not None else (hi or 0))]
        per.append(cand)
    if not syms:
        return [{}]
    out: list[dict[sp.Symbol, sp.Basic]] = []
    if len(syms) <= 4:
        m = max(3, round(limit ** (1 / len(syms))) + 1)
        grid = itertools.product(*[range(min(m, len(c))) for c in per])
        for idx in sorted(grid, key=lambda t: (sum(t), t)):
            out.append({s: per[j][i] for j, (s, i) in enumerate(zip(syms, idx))})
    else:
        for i in range(limit):
            out.append({s: per[j][(i + 3 * j + i // len(per[j])) % len(per[j])] for j, s in enumerate(syms)})
    keep = []
    for p in out:
        ok = True
        for r in rels:
            try:
                t = r.xreplace(p)
            except Exception:  # noqa: BLE001
                ok = False
                break
            if t is not sp.true:
                ok = False
                break
        if ok and p not in keep:
            keep.append(p)
    return keep[:limit]


# --------------------------------------------------------------------------- #
# execution (in the isolated worker)                                          #
# --------------------------------------------------------------------------- #

_DENY_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
    "setattr", "delattr", "getattr", "vars", "breakpoint", "exit", "quit",
})


def _execution_blocker(src: _Source) -> "str | None":
    """Why this module must NOT be executed, or ``None`` when it may."""
    for node in ast.walk(src.tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in SAFE_MODULES:
                    return f"imports {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in SAFE_MODULES:
                return f"imports from {node.module}"
        elif isinstance(node, ast.Name) and node.id in _DENY_NAMES:
            return f"uses {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"uses {node.attr}"
    return None


def _definitions_only(src: _Source) -> str:
    """The module's imports, function definitions and constant assignments —
    without top-level statements that would RUN something when executed."""
    keep: list[ast.stmt] = []
    for node in src.tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is not None and all(
                isinstance(n, (ast.Constant, ast.UnaryOp, ast.BinOp, ast.operator, ast.unaryop,
                               ast.Name, ast.Attribute, ast.Load, ast.Store, ast.expr_context))
                for n in ast.walk(value)
            ):
                keep.append(node)
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def _plain(v: Any) -> Any:
    """A returned value as a plain Python number (NumPy scalars unwrapped)."""
    if isinstance(v, (bool, int, float, fractions.Fraction)):
        return v
    item = getattr(v, "item", None)
    if callable(item) and getattr(v, "shape", None) == ():
        try:
            return _plain(item())
        except Exception:  # noqa: BLE001
            pass
    return ("other", type(v).__name__, repr(v)[:80])


def _run_calls(spec: tuple, calls: list[dict[str, Any]], per_call: float) -> list[tuple[str, Any]]:
    """Worker side: call the function once per argument dict; never raises."""
    import contextlib
    import warnings

    if spec[0] == "object":
        fn = spec[1]
    else:
        ns: dict[str, Any] = {"__name__": "__pcm_c1_subject__"}
        exec(compile(spec[1], "<checked function>", "exec"), ns)  # noqa: S102 — this IS the point: confirm by running
        fn = ns[spec[2]]
    out: list[tuple[str, Any]] = []
    for kwargs in calls:
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                with time_budget(per_call, "check_code execution"):
                    out.append(("ok", _plain(fn(**kwargs))))
        except TimeBudgetError:
            out.append(("timeout", per_call))
        except Exception as exc:  # noqa: BLE001 — the function's own failure is data
            out.append(("raise", f"{type(exc).__name__}: {exc}"[:200]))
        if time.perf_counter() - started > per_call:
            break
    return out


class _Executor:
    """Runs the subject function in ``isolated``; counts calls; caches results."""

    def __init__(self, src: _Source, budget: float) -> None:
        self.src = src
        self.blocked = _execution_blocker(src)
        self.calls = 0
        self.budget = budget
        self.cache: dict[tuple, tuple[str, Any]] = {}
        self.spec: "tuple | None" = None
        if src.obj is not None:
            # The caller already imported and holds this function: running it in the
            # worker adds no new exposure. Passed by reference (pickle), so module
            # globals the source alone would miss are there too.
            try:
                pickle.dumps(src.obj)
                self.spec, self.blocked = ("object", src.obj), None
            except Exception:  # noqa: BLE001 — fall back to the source text
                pass
        if self.spec is None and self.blocked is None:
            self.spec = ("source", _definitions_only(src), src.func.name)

    def run(self, calls: list[dict[str, Any]]) -> list[tuple[str, Any]]:
        if self.spec is None:
            return [("blocked", self.blocked)] * len(calls)
        todo = [c for c in calls if _key(c) not in self.cache]
        if todo:
            results = self._isolated(todo)
            for c, r in zip(todo, results):
                self.cache[_key(c)] = r
            for c in todo[len(results):]:
                self.cache[_key(c)] = ("timeout", None)
            self.calls += len(todo)
        return [self.cache[_key(c)] for c in calls]

    def _isolated(self, calls: list[dict[str, Any]]) -> list[tuple[str, Any]]:
        per_call = min(2.0, max(0.2, self.budget / 4))
        total = min(self.budget, per_call * len(calls) + 1.0)
        assert self.spec is not None
        try:
            return isolated(_run_calls, self.spec, calls, per_call, budget=max(total, 1.0))
        except (TimeBudgetError, IsolationError, EOFError, OSError) as exc:
            return [("timeout", f"{type(exc).__name__}: {exc}")] * len(calls)


def _key(c: Mapping[str, Any]) -> tuple:
    return tuple(sorted((k, type(v).__name__, repr(v)) for k, v in c.items()))


# --------------------------------------------------------------------------- #
# confirmation                                                                #
# --------------------------------------------------------------------------- #

#: Relative tolerance within which a FLOAT result counts as the exact reading of
#: the code (rounding of a short chain of double operations is ~1e-15; the margin
#: covers loops of a few hundred steps). A float result refutes only when it sits
#: within this of the exact reading AND at least ``_SEPARATION`` times further
#: from the formula.
_FLOAT_TOL = 1e-10
_SEPARATION = 1000.0
_EPS = 2.0**-53


def _float_tol(src: _Source, point: Mapping[sp.Symbol, sp.Basic]) -> float:
    """Relative rounding budget for one call: a standard a-priori model,
    ``8 * eps * ops * trips`` — ``ops`` the arithmetic/call nodes of the module's
    functions, ``trips`` a loop-count bound from the integer arguments (each loop
    level may run up to ``1 + sum |int args|`` times). Floored at 2**-50, capped at
    ``_FLOAT_TOL``. Too small a budget can only turn a refutation into
    ``float_only_discrepancy`` (a refutation also needs the float result to sit
    within the budget of the EXACT reading), never create one."""
    ops = 0
    depth = 0
    for fdef in (n for n in src.tree.body if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(fdef):
            if isinstance(node, (ast.BinOp, ast.AugAssign, ast.Call, ast.UnaryOp)):
                ops += 1
        depth = max(depth, _loop_depth(fdef))
    bound = 1 + sum(abs(int(v)) for s, v in point.items() if s.is_integer)
    trips = float(bound) ** depth
    return min(_FLOAT_TOL, max(2.0**-50, 8 * _EPS * max(ops, 1) * trips))


def _loop_depth(node: ast.AST) -> int:
    inner = max((_loop_depth(c) for c in ast.iter_child_nodes(node)), default=0)
    return inner + (1 if isinstance(node, (ast.For, ast.While, ast.comprehension)) else 0)


def _call_args(point: Mapping[sp.Symbol, sp.Basic], exact: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s, v in point.items():
        if s.is_integer:
            out[s.name] = int(v)
        elif exact:
            out[s.name] = fractions.Fraction(int(v.p), int(v.q))
        else:
            out[s.name] = float(v)
    return out


@dataclass
class _Outcome:
    status: str  # "refuted" | "agree" | "float_only" | "skip" | "candidate" | "mismatch_reading"
    message: str = ""
    cex: "Counterexample | None" = None


def _confirm(
    point: Mapping[sp.Symbol, sp.Basic],
    formula: sp.Expr,
    reading: "sp.Basic | None",
    result: tuple[str, Any],
    exact_result: "tuple[str, Any] | None",
    fname: str,
    float_tol: float = _FLOAT_TOL,
) -> _Outcome:
    F = _exact_at(formula, point)
    if F is None:
        return _Outcome("skip", "formula undefined here")
    status, A = result
    if status != "ok":
        return _Outcome("skip", f"call did not return ({status}: {A})")
    if isinstance(A, tuple):
        return _Outcome("skip", f"returned a non-number ({A[1]})")
    if isinstance(A, float) and not math.isfinite(A):
        return _Outcome("skip", "returned a non-finite float")
    args = _call_args(point, exact=False)
    args_exact = {s.name: str(v) for s, v in point.items()}
    F_approx = float(sp.N(F, 17))

    def cex(value: Any, exact: bool, tol: float) -> Counterexample:
        return Counterexample(args, args_exact, value, sp.sstr(F), F_approx, exact, tol, True, fname)

    exact_value: "sp.Expr | None" = None
    if isinstance(A, (bool, int, fractions.Fraction)):
        exact_value = sp.Rational(int(A)) if not isinstance(A, fractions.Fraction) else sp.Rational(A.numerator, A.denominator)
    elif exact_result is not None and exact_result[0] == "ok" and isinstance(exact_result[1], (int, fractions.Fraction)) \
            and not isinstance(exact_result[1], bool):
        e = exact_result[1]
        exact_value = sp.Rational(e.numerator, e.denominator) if isinstance(e, fractions.Fraction) else sp.Integer(e)
    if exact_value is not None:
        same = _same(exact_value, F)
        if same is False and isinstance(A, float):
            # exact arithmetic of the program disagrees; the float run must as well,
            # or the counterexample would show a value equal to the formula's
            if abs(mpmath.mpf(A) - _mp(F)) <= float_tol * max(abs(_mp(F)), abs(mpmath.mpf(A)), mpmath.mpf(1e-300)):
                return _Outcome("skip", "exact run differs, float run does not")
        if same is False:
            # exact arithmetic of the very program disagrees with the exact formula
            return _Outcome("refuted", "", cex(A, True, 0.0))
        if same is True:
            return _Outcome("agree")
        return _Outcome("skip", "difference not resolvable at 100 digits")
    # a float result
    a = mpmath.mpf(A)
    f = _mp(F)
    scale = max(abs(f), mpmath.mpf(1e-300))
    if abs(a - f) <= float_tol * max(scale, abs(a)):
        return _Outcome("agree")
    S = None if reading is None else _exact_at(reading, point)
    if S is None:
        return _Outcome(
            "candidate",
            f"{fname}({_fmt_args(args)}) -> {A!r}, formula gives {sp.sstr(F)} — a float result with no exact "
            f"reading of the code to rule out rounding",
        )
    s = _mp(S)
    if _same(S, F) is True:
        return _Outcome(
            "float_only",
            f"{fname}({_fmt_args(args)}) -> {A!r}, but the exact value of the code is {sp.sstr(F)} as the formula says — "
            f"the gap is floating-point rounding (see pycodemath.code.stability, C2)",
        )
    tol = float_tol * max(abs(s), mpmath.mpf(1e-300))
    if abs(a - s) <= tol and abs(a - f) > _SEPARATION * tol and _same(S, F) is False:
        rel = float(abs(a - f) / scale)
        return _Outcome("refuted", "", cex(A, False, min(1e-9, rel / 100)))
    if abs(a - s) > tol:
        return _Outcome(
            "float_only",
            f"{fname}({_fmt_args(args)}) -> {A!r}; exact code value {sp.sstr(S)}, formula {sp.sstr(F)} — the float "
            f"result is not within rounding of either, so rounding effects are suspected (C2)",
        )
    return _Outcome(
        "float_only",
        f"{fname}({_fmt_args(args)}) -> {A!r}, formula {sp.sstr(F)}: the exact code value differs from the formula, "
        f"but by less than {_SEPARATION:g} x the rounding budget ({float_tol:.1e}) — not separable from rounding",
    )


def _fmt_args(args: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


# --------------------------------------------------------------------------- #
# proofs                                                                      #
# --------------------------------------------------------------------------- #


def _zero(e: sp.Basic) -> bool:
    """Is ``e`` identically zero, by SymPy's own reasoning (under the active budget)?"""
    if e == 0:
        return True
    e = e.doit() if e.has(sp.Sum, sp.Product) else e
    if e.has(sp.fibonacci, sp.lucas):
        e = e.rewrite(sp.sqrt)
    for attempt in (sp.expand, sp.simplify):
        r = attempt(e)
        if r == 0 or r.is_zero is True:
            return True
    return _exppoly_zero(e)


def _algebraic_zero(v: sp.Expr) -> bool:
    """Is the exact number ``v`` zero — proved, not estimated."""
    if v == 0:
        return True
    if v.is_Rational:
        return False
    w = sp.radsimp(sp.expand(v))
    if w == 0:
        return True
    w = sp.nsimplify(sp.simplify(w)) if not w.is_Rational else w
    return bool(w == 0)


def _exppoly_zero(e: sp.Basic) -> bool:
    """Zero-test for an exponential polynomial in ONE integer variable ``n``:
    a sum of ``c * n**k * b**(a*n)`` terms. Such an expression with ``d`` =
    sum over distinct bases of (largest ``k`` + 1) vanishes identically iff it
    vanishes at ``d`` consecutive integers (the confluent Vandermonde system is
    nonsingular). Bases SymPy fails to recognise as equal are counted twice,
    which only raises ``d`` — the test stays sound. This is how a loop closed
    into Binet's formula is proved equal to ``fibonacci(n)``."""
    ints = [s for s in e.free_symbols if s.is_integer]
    if len(e.free_symbols) != 1 or len(ints) != 1:
        return False
    n = ints[0]
    e = sp.expand(e, power_exp=True, mul=True, multinomial=True, power_base=False)
    bases: dict[sp.Basic, int] = {}
    for term in sp.Add.make_args(e):
        base: sp.Basic = sp.Integer(1)
        degree = 0
        for f in sp.Mul.make_args(term):
            if n not in f.free_symbols:
                continue
            if f == n:
                degree += 1
            elif isinstance(f, sp.Pow) and f.base == n and f.exp.is_Integer and f.exp > 0:
                degree += int(f.exp)
            elif isinstance(f, sp.Pow) and n not in f.base.free_symbols:
                ex = sp.expand(f.exp)
                if not ex.is_polynomial(n) or sp.degree(ex, n) != 1:
                    return False
                a, c = sp.Poly(ex, n).all_coeffs()
                if c != 0 or f.base == 0:
                    return False
                base = base * f.base**a
            elif isinstance(f, sp.exp):
                ex = sp.expand(f.args[0])
                if not ex.is_polynomial(n) or sp.degree(ex, n) != 1:
                    return False
                base = base * sp.exp(sp.Poly(ex, n).all_coeffs()[0])
            else:
                return False
        base = sp.powsimp(base)
        bases[base] = max(bases.get(base, 0), degree)
    d = sum(k + 1 for k in bases.values())
    if d > 40:
        return False
    for k in range(d):
        v = _exact_at(e, {n: sp.Integer(k)})
        if v is None or not _algebraic_zero(sp.expand(v)):
            return False
    return True


def _zero_any(e: sp.Basic, syms: Sequence[sp.Symbol], rels: Sequence[sp.Basic]) -> bool:
    """``_zero``, then the quasi-polynomial certificate for one integer variable."""
    if _zero(e):
        return True
    free = [s for s in e.free_symbols if isinstance(s, sp.Symbol)]
    if len(free) == 1 and free[0].is_integer:
        try:
            return _prove_integer(e, free[0], rels)
        except (_NotQP, PycodemathError, TypeError, ValueError):
            return False
    return False


def _interval(sym: sp.Symbol, rels: Sequence[sp.Basic]) -> "sp.Set":
    lo, hi = _bounds(sym, rels)
    return sp.Interval(lo if lo is not None else -sp.oo, hi if hi is not None else sp.oo)


def _case_split(diff: sp.Basic, syms: Sequence[sp.Symbol], rels: Sequence[sp.Basic], depth: int = 0) -> bool:
    """Prove ``diff == 0`` arm by arm of its outermost ``Piecewise``.

    An arm whose condition pins an integer variable to finitely many values
    (with the domain bounds) is checked by substitution at each; otherwise the arm
    must be identically zero on its own (stronger than needed, hence sound)."""
    if depth > 4:
        return False
    if diff.has(sp.Sum, sp.Product):
        diff = diff.doit()
    if any(x.has(sp.Piecewise) for x in diff.atoms(sp.Sum, sp.Product)):
        return _zero_any(diff, syms, rels)  # folding across a bound variable is unsound
    folded = sp.piecewise_fold(diff)
    if not isinstance(folded, sp.Piecewise):
        return _zero_any(folded, syms, rels)
    covered: sp.Basic = sp.false
    for expr, cond in folded.args:
        region = sp.And(cond, sp.Not(covered))
        covered = sp.Or(covered, cond)
        if region is sp.false or expr is sp.nan:
            continue  # unreachable, or the function has no value there (raise / no return)
        pins = _pinned(region, syms, rels)
        if pins == "empty":
            continue
        if pins is not None:
            sym, values = pins
            for v in values:
                at = sp.sympify(expr).xreplace({sym: v})
                at = at.doit() if at.has(sp.Sum, sp.Product) else at
                if at.has(sp.nan, sp.zoo):
                    continue  # one side undefined there: nothing to compare
                if not _case_split(at, syms, rels, depth + 1):
                    return False
            continue
        try:
            expr = sp.refine(expr, region)
        except Exception:  # noqa: BLE001 — refine is an optional sharpening
            pass
        if not _case_split(expr, syms, rels, depth + 1):
            return False
    return True


def _pinned(region: sp.Basic, syms: Sequence[sp.Symbol], rels: Sequence[sp.Basic]):
    """``"empty"`` when ``region`` cannot hold on the domain; ``(sym, values)`` when
    it pins one variable to finitely many values; else ``None``. A conjunction is
    pinned by any one of its parts (the region is inside each part's set)."""
    parts = list(region.args) if isinstance(region, sp.And) else [region]
    if len(region.free_symbols) == 1:
        parts = [region] + parts
    best = None
    for part in parts:
        r = _pinned_one(part, rels)
        if r == "empty":
            return "empty"
        if r is not None and (best is None or len(r[1]) < len(best[1])):
            best = r
    return best


def _pinned_one(region: sp.Basic, rels: Sequence[sp.Basic]):
    free = region.free_symbols
    if len(free) != 1:
        return None
    sym = next(iter(free))
    try:
        if isinstance(region, (sp.Lt, sp.Le, sp.Gt, sp.Ge)):
            sol = sp.solve_univariate_inequality(region, sym, relational=False)
        elif isinstance(region, (sp.And, sp.Or, sp.Not)):
            sol = region.as_set()
        else:
            sol = sp.solveset(region, sym, sp.S.Reals)
    except Exception:  # noqa: BLE001
        return None
    sol = sp.Intersection(sol, _interval(sym, rels))
    if sol is sp.S.EmptySet:
        return "empty"
    if isinstance(sol, sp.FiniteSet):
        vals = [v for v in sol if not sym.is_integer or v.is_integer]
        return (sym, vals) if vals else "empty"
    if not sym.is_integer:
        return None
    ints = sp.Intersection(sol, sp.S.Integers)
    if isinstance(ints, sp.FiniteSet):
        return (sym, list(ints)) if len(ints) else "empty"
    if isinstance(sol, sp.Interval) and sol.inf.is_finite and sol.sup.is_finite:
        vals = list(range(int(sp.ceiling(sol.inf)), int(sp.floor(sol.sup)) + 1))
        if not vals:
            return "empty"
        if len(vals) <= 50:
            return sym, [sp.Integer(v) for v in vals]
    return None


class _QP:
    """A quasi-polynomial in one integer variable: on each residue class
    ``n ≡ r (mod period)`` with ``n >= threshold``, a polynomial in ``n`` over Q."""

    def __init__(self, period: int, polys: list[sp.Poly], threshold: int) -> None:
        self.period, self.polys, self.threshold = period, polys, threshold

    def at(self, r: int) -> sp.Poly:
        return self.polys[r % self.period]


class _NotQP(Exception):
    pass


def _qp(e: sp.Basic, n: sp.Symbol) -> _QP:
    """Build the quasi-polynomial of ``e`` (raises ``_NotQP`` outside the class)."""
    def poly(p: sp.Expr) -> sp.Poly:
        return sp.Poly(p, n, domain="QQ")

    def lift(a: _QP, period: int) -> list[sp.Poly]:
        return [a.at(r) for r in range(period)]

    def combine(a: _QP, b: _QP, f: Callable[[sp.Poly, sp.Poly], sp.Poly]) -> _QP:
        P = math.lcm(a.period, b.period)
        if P > 720:
            raise _NotQP("period too large")
        return _QP(P, [f(x, y) for x, y in zip(lift(a, P), lift(b, P))], max(a.threshold, b.threshold))

    if e.is_Rational:
        return _QP(1, [poly(e)], -10**9)
    if e == n:
        return _QP(1, [poly(n)], -10**9)
    if e.free_symbols - {n}:
        raise _NotQP("other symbols")
    if isinstance(e, sp.Add):
        out = _qp(e.args[0], n)
        for t in e.args[1:]:
            out = combine(out, _qp(t, n), lambda x, y: x + y)
        return out
    if isinstance(e, sp.Mul):
        out = _qp(e.args[0], n)
        for t in e.args[1:]:
            out = combine(out, _qp(t, n), lambda x, y: x * y)
        return out
    if isinstance(e, sp.Pow) and e.exp.is_Integer and e.exp >= 0:
        base = _qp(e.base, n)
        out = _QP(1, [poly(sp.Integer(1))], -10**9)
        for _ in range(int(e.exp)):
            out = combine(out, base, lambda x, y: x * y)
        return out
    if isinstance(e, sp.floor):
        return _floor_qp(_qp(e.args[0], n), n)
    if isinstance(e, sp.ceiling):
        neg = _qp(-e.args[0], n)
        fl = _floor_qp(neg, n)
        return _QP(fl.period, [-p for p in fl.polys], fl.threshold)
    if isinstance(e, sp.Mod):
        a, b = e.args
        if not b.is_Rational or b == 0:
            raise _NotQP("modulus not a constant")
        return _qp(a - b * sp.floor(a / b), n)
    if isinstance(e, (sp.Min, sp.Max, sp.Abs)):
        return _qp(e.rewrite(sp.Piecewise), n)
    if isinstance(e, sp.Piecewise):
        return _piecewise_qp(e, n)
    raise _NotQP(type(e).__name__)


def _floor_qp(a: _QP, n: sp.Symbol) -> _QP:
    k = sp.Symbol("k", integer=True)
    polys: dict[int, sp.Poly] = {}
    periods = []
    data = []
    for r in range(a.period):
        h = sp.Poly(a.at(r).as_expr().subs(n, a.period * k + r), k, domain="QQ")
        L = math.lcm(*[int(sp.Rational(c).q) for c in h.all_coeffs()]) if h.all_coeffs() else 1
        periods.append(L)
        data.append(h)
    P = a.period * math.lcm(*periods)
    if P > 720:
        raise _NotQP("period too large")
    for r in range(P):
        base = r % a.period
        j = (r - base) // a.period
        frac = data[base].eval(j) - sp.floor(data[base].eval(j))
        polys[r] = a.at(base) - sp.Poly(frac, n, domain="QQ")
    return _QP(P, [polys[r] for r in range(P)], a.threshold)


def _cauchy(p: sp.Poly) -> int:
    """Beyond this, ``p`` has no real roots (Cauchy's bound)."""
    coeffs = p.all_coeffs()
    if len(coeffs) <= 1:
        return -10**9
    lead = abs(coeffs[0])
    return int(sp.ceiling(1 + max(abs(c) / lead for c in coeffs[1:]))) + 1


def _truth(cond: sp.Basic, n: sp.Symbol, period_hint: int) -> tuple[int, list[bool], int]:
    """Eventual truth of ``cond`` on each residue class: (period, truths, threshold)."""
    if cond is sp.true or cond is sp.false:
        return 1, [cond is sp.true], -10**9
    if isinstance(cond, (sp.And, sp.Or)):
        parts = [_truth(c, n, period_hint) for c in cond.args]
        P = math.lcm(*[p[0] for p in parts])
        if P > 720:
            raise _NotQP("period too large")
        op = all if isinstance(cond, sp.And) else any
        return P, [op(p[1][r % p[0]] for p in parts) for r in range(P)], max(p[2] for p in parts)
    if isinstance(cond, sp.Not):
        P, t, th = _truth(cond.args[0], n, period_hint)
        return P, [not x for x in t], th
    if isinstance(cond, (sp.Eq, sp.Ne, sp.Lt, sp.Le, sp.Gt, sp.Ge)):
        d = _qp(cond.lhs - cond.rhs, n)
        truths, th = [], d.threshold
        for r in range(d.period):
            p = d.at(r)
            if p.is_zero:
                sign = 0
            else:
                sign = 1 if p.LC() > 0 else -1
                th = max(th, _cauchy(p))
            truths.append({
                sp.Eq: sign == 0, sp.Ne: sign != 0, sp.Lt: sign < 0, sp.Le: sign <= 0,
                sp.Gt: sign > 0, sp.Ge: sign >= 0,
            }[type(cond)])
        return d.period, truths, th
    raise _NotQP(f"condition {type(cond).__name__}")


def _piecewise_qp(e: sp.Piecewise, n: sp.Symbol) -> _QP:
    arms = [(_qp(x, n), _truth(c, n, 1)) for x, c in e.args]
    P = math.lcm(*[a.period for a, _ in arms], *[t[0] for _, t in arms])
    if P > 720:
        raise _NotQP("period too large")
    th = max(max(a.threshold, t[2]) for a, t in arms)
    polys = []
    for r in range(P):
        for a, (tp, truths, _) in arms:
            if truths[r % tp]:
                polys.append(a.at(r))
                break
        else:
            raise _NotQP("no arm holds")
    return _QP(P, polys, th)


def _qp_zero(diff: sp.Basic, n: sp.Symbol, lo: int) -> bool:
    """``diff(n) == 0`` for every integer ``n >= lo``: each class polynomial is the
    zero polynomial beyond the threshold, and the finitely many ``n`` below it are
    checked one by one, exactly."""
    q = _qp(diff, n)
    if not all(p.is_zero for p in q.polys):
        return False
    upto = max(q.threshold, lo)
    if upto - lo > 2000:
        raise _NotQP("threshold too far")
    for v in range(lo, upto + 1):
        val = _exact_at(diff, {n: sp.Integer(v)})
        if val is None:
            continue  # undefined there: equality holds where both sides are defined
        if val != 0:
            return False
    return True


def _prove_integer(diff: sp.Basic, n: sp.Symbol, rels: Sequence[sp.Basic]) -> bool:
    lo, _hi = _bounds(n, rels)
    d = diff.doit() if diff.has(sp.Sum, sp.Product) else diff
    if d.has(sp.Sum, sp.Product):
        raise _NotQP("unevaluated sum")
    d = _fold(d)
    if lo is not None:
        return _qp_zero(d, n, lo)
    m = sp.Symbol("m", integer=True)
    return _qp_zero(d, n, 0) and _qp_zero(d.xreplace({n: -m}), m, 1)


# --------------------------------------------------------------------------- #
# check_code                                                                  #
# --------------------------------------------------------------------------- #


def check_code(
    func_or_source: "Callable[..., Any] | str",
    formula: "str | sp.Expr | Expr | int | float",
    *,
    name: "str | None" = None,
    domain: "Mapping[str, str] | None" = None,
    budget: "float | None" = None,
) -> CodeCheck:
    """Is ``func_or_source`` an implementation of ``formula``? See the module docstring.

    ``domain`` maps a parameter to ``"integer, n>=0"``-style text (type words
    ``integer real positive nonnegative``, comparisons with constants or other
    parameters); without it, ``int``/``float`` annotations and use inside
    ``range(...)`` decide integer vs real. ``budget`` (seconds, default
    ``DEFAULT_TIME_BUDGET``) bounds the whole check; running out is UNDECIDED.
    """
    started = time.perf_counter()
    total = DEFAULT_TIME_BUDGET if budget is None else budget
    if not isinstance(total, (int, float)) or isinstance(total, bool) or not total > 0:
        raise DomainError(f"check_code: budget must be a positive number, not {budget!r}")
    deadline = started + float(total)
    src = _load(func_or_source, name)
    params, defaults_nodes = _signature(src.func)
    dom = _parse_domain(domain, params)
    syms = _make_symbols(src.func, params, dom)
    F = _parse_formula(formula, syms)
    formula_text = formula if isinstance(formula, str) else sp.sstr(F)
    notes: list[Note] = []
    reading: "sp.Basic | None" = None
    read_error: "CodeReadError | None" = None
    try:
        sf = _read(src, domain)
        reading = sf.expr
        notes += sf.notes
    except CodeReadError as exc:
        read_error = exc
        notes += list(getattr(exc, "notes", ()))
        if not any(n.code in ("loop_not_reducible", "suspicious_xor") for n in notes):
            notes.append(Note("unsupported", str(exc)))
        elif not any(n.code == "unsupported" for n in notes) and exc.construct != "loop not reducible":
            notes.append(Note("unsupported", str(exc)))

    # parameters that the formula leaves out but that have defaults are fixed there
    rels = []
    for lhs, op, rhs in dom.relations:
        try:
            rels.append(_relation(_parse_formula(lhs, syms), op, _parse_formula(rhs, syms)))
        except PycodemathError as exc:
            raise DomainError(f"check_code: cannot read domain condition {lhs}{op}{rhs}: {exc}") from exc
    fixed: dict[sp.Symbol, sp.Basic] = {}
    for p in params:
        if syms[p] not in F.free_symbols and p in defaults_nodes:
            try:
                lit = ast.literal_eval(defaults_nodes[p])
                if isinstance(lit, bool) or not isinstance(lit, (int, float)):
                    continue
                fixed[syms[p]] = sp.Rational(repr(lit)) if isinstance(lit, float) else sp.Integer(lit)
            except (ValueError, SyntaxError):
                pass
    sample_syms = [syms[p] for p in params if syms[p] not in fixed]
    executor = _Executor(src, max(0.5, min(5.0, float(total) / 2)))
    if executor.blocked:
        notes.append(Note("not_executed", f"the function is never run: its module {executor.blocked}"))

    def finish(verdict: str, method: str, reason: str, cex: "Counterexample | None" = None,
               points: int = 0) -> CodeCheck:
        return CodeCheck(
            verdict, method, reason, src.func.name, tuple(params), formula_text,
            None if reading is None else sp.sstr(reading), cex, tuple(_dedupe(notes)), dict(dom.text),
            src.text, src.module, elapsed=time.perf_counter() - started, checked_points=points,
            executed_calls=executor.calls,
        )

    points = _points(sample_syms, rels)
    points = [{**p, **fixed} for p in points]
    S = None if reading is None else reading.xreplace(fixed)

    # --- 1. compare the reading with the formula at the domain points ---------
    mismatches: list[dict] = []
    agreeing: list[dict] = []
    if S is not None:
        for pt in points:
            if time.perf_counter() > deadline:
                break
            s_val, f_val = _exact_at(S, pt), _exact_at(F, pt)
            if s_val is None or f_val is None:
                continue
            same = _same(s_val, f_val)
            if same is False:
                mismatches.append(pt)
            elif same is True:
                agreeing.append(pt)
        candidates = mismatches[:6] + agreeing[:6]
    else:
        candidates = points

    # --- 2. execute: confirm candidates, cross-check the reading --------------
    outcomes = _execute_and_confirm(candidates, F, S, executor, src.func.name, deadline, sample_syms)
    for o in outcomes:
        if o.status == "refuted" and o.cex is not None:
            if S is not None and _point_of(o, syms) in agreeing:
                notes.append(Note("reading_mismatch", "the symbolic reading agreed with the formula here, "
                                  "but the executed function did not"))
            return finish("refuted", "execution", _refute_reason(o.cex), o.cex, len(points))
    for o in outcomes:
        if o.status in ("float_only", "candidate"):
            notes.append(Note("float_only_discrepancy" if o.status == "float_only" else "unconfirmed_candidate",
                              o.message))
    reading_ok = S is not None and not any(o.status == "mismatch_reading" for o in outcomes)
    if S is not None and not reading_ok:
        notes.append(Note("reading_mismatch", "execution contradicted the symbolic reading; no proof is attempted"))

    if read_error is not None:
        return finish("undecided", "unsupported", _undecided_reason(read_error, outcomes, executor), None, len(points))
    if mismatches:
        return finish("undecided", "float-only" if any(n.code == "float_only_discrepancy" for n in notes)
                      else "unconfirmed", "the reading differs from the formula, but no difference was confirmed "
                      "by executing the function", None, len(points))
    if not reading_ok:
        return finish("undecided", "reading-mismatch", "the reading of the code was contradicted by its execution",
                      None, len(points))

    # --- 3. prove ---------------------------------------------------------------
    assert S is not None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return finish("undecided", "time-budget", "the budget ran out before a proof was attempted", None, len(points))
    proof = _prove(S, F, sample_syms, rels, remaining)
    if proof[0] == "verified":
        return finish("verified", proof[1], proof[2], None, len(points))
    if proof[0] == "refuted-point":
        p = proof[3]
        extra = _execute_and_confirm([p], F, S, executor, src.func.name, deadline, sample_syms)
        for o in extra:
            if o.status == "refuted" and o.cex is not None:
                return finish("refuted", "execution", _refute_reason(o.cex), o.cex, len(points))
            if o.status in ("float_only", "candidate"):
                notes.append(Note("float_only_discrepancy" if o.status == "float_only" else "unconfirmed_candidate",
                                  o.message))
        return finish("undecided", "unconfirmed", "check_equal found a differing point that execution did not "
                      "confirm", None, len(points))
    return finish("undecided", proof[1], proof[2], None, len(points))


def _point_of(o: _Outcome, syms: Mapping[str, sp.Symbol]) -> dict:
    assert o.cex is not None
    return {syms[k]: sp.Rational(v) for k, v in o.cex.args_exact.items()}


def _dedupe(notes: Iterable[Note]) -> list[Note]:
    out: list[Note] = []
    for n in notes:
        if n not in out:
            out.append(n)
    return out


def _refute_reason(c: Counterexample) -> str:
    how = "exactly" if c.exact else f"beyond float rounding (relative tolerance {c.tolerance:.1e})"
    return (f"{c.call} returns {c.function_value!r}, the formula gives {c.formula_value}"
            f" — confirmed by executing the function and evaluating the formula {how}")


def _undecided_reason(err: CodeReadError, outcomes: list[_Outcome], ex: _Executor) -> str:
    agreed = sum(o.status == "agree" for o in outcomes)
    tried = len(outcomes)
    tail = (f"; executed at {tried} points, {agreed} agree with the formula and none refutes it"
            if tried and not ex.blocked else "")
    return f"{err.construct}{f' at line {err.line}' if err.line else ''}{tail}"


def _execute_and_confirm(
    pts: list[dict], F: sp.Expr, S: "sp.Basic | None", ex: _Executor, fname: str, deadline: float,
    sample_syms: Sequence[sp.Symbol],
) -> list[_Outcome]:
    if not pts or ex.blocked or time.perf_counter() > deadline:
        return []
    real = any(not s.is_integer for p in pts for s in p)
    results = ex.run([_call_args(p, exact=False) for p in pts])
    exacts: "list[tuple[str, Any] | None]" = [None] * len(pts)
    if real:
        exacts = list(ex.run([_call_args(p, exact=True) for p in pts]))
    out = []
    for p, r, e in zip(pts, results, exacts):
        o = _confirm(p, F, S, r, e, fname, _float_tol(ex.src, p))
        if o.status == "agree" and S is not None:
            s_val, f_val = _exact_at(S, p), _exact_at(F, p)
            if s_val is not None and f_val is not None and _same(s_val, f_val) is False:
                o = _Outcome("mismatch_reading", "execution agrees with the formula, not with the reading")
        out.append(o)
    return out


def _in_domain(point: Mapping[sp.Symbol, sp.Basic], rels: Sequence[sp.Basic]) -> bool:
    for s, v in point.items():
        if (s.is_positive and not v > 0) or (s.is_nonnegative and not v >= 0) \
                or (s.is_negative and not v < 0) or (s.is_integer and not v.is_integer):
            return False
    return all(r.xreplace(dict(point)) is sp.true for r in rels)


def _prove(S: sp.Basic, F: sp.Expr, syms: Sequence[sp.Symbol], rels: Sequence[sp.Basic], remaining: float):
    diff = S - F
    integer = [s for s in syms if s.is_integer]
    real_only = not integer
    try:
        if _caller_budget_active():
            ok = _zero(diff)
        else:
            with time_budget(max(0.05, remaining * 0.5), "check_code"):
                ok = _zero(diff)
        if ok:
            return ("verified", "symbolic", "the reading of the function minus the formula simplifies to 0")
        if diff.has(sp.Abs, sp.Min, sp.Max, sp.sign):
            diff = diff.rewrite(sp.Piecewise)
        if diff.has(sp.Piecewise):
            if _caller_budget_active():
                ok = _case_split(diff, syms, rels)
            else:
                with time_budget(max(0.05, remaining * 0.3), "check_code"):
                    ok = _case_split(diff, syms, rels)
            if ok:
                return ("verified", "case-split", "every branch of the reading equals the formula on its region")
    except TimeBudgetError:
        return ("undecided", "time-budget", "the budget ran out during the symbolic proof")
    if len(integer) == 1 and len(syms) == 1:
        try:
            if _prove_integer(diff, integer[0], rels):
                return ("verified", "quasi-polynomial",
                        "exact quasi-polynomial certificate: every residue class agrees as a polynomial, "
                        "and every value below the threshold was checked")
        except (_NotQP, PycodemathError, TypeError, ValueError):
            pass
    if real_only:
        verdict = check_equal(Expr(S), Expr(F), budget=max(0.1, remaining * 0.5))  # type: ignore[arg-type]
        if verdict.verified:
            return ("verified", "check_equal", f"check_equal: {verdict.detail}")
        if verdict.refuted and verdict.counterexample:
            point = {s: sp.Rational(verdict.counterexample[s.name]) for s in syms}
            if _in_domain(point, rels):
                return ("refuted-point", "check_equal", verdict.detail, point)
            return ("undecided", "no-proof", "no proof; check_equal's differing point lies outside the domain")
        return ("undecided", verdict.method, f"no proof: {verdict.detail}")
    return ("undecided", "no-proof", "the reading agrees with the formula at every checked domain point and "
            "execution confirms it, but no proof was found")


# --------------------------------------------------------------------------- #
# emit_test                                                                   #
# --------------------------------------------------------------------------- #


def emit_test(result: CodeCheck, *, module: "str | None" = None) -> str:
    """A pytest file pinning ``result``'s counterexample: red on the current function.

    ``module`` is the import path of the function; without it (and without a
    module recorded in ``result``) the function's source is pasted into the test.
    Integer results are asserted exactly, float ones with ``pytest.approx`` and a
    relative tolerance a hundred times tighter than the discrepancy found.
    """
    if result.counterexample is None:
        raise DomainError("emit_test: only a refuted result has a counterexample to pin")
    c = result.counterexample
    mod = module or (result.module if result.module not in (None, "__main__") else None)
    lines = [
        f'"""Regression test generated by pycodemath (C1, check_code) on {datetime.date.today().isoformat()}.',
        "",
        f"{result.function_name} is meant to compute: {result.formula}",
        f"Counterexample: {c.call} returned {c.function_value!r}; the formula gives {c.formula_value}.",
        '"""',
        "",
        "import pytest",
        "",
    ]
    if mod:
        lines.append(f"from {mod} import {result.function_name}")
    else:
        lines.append(_definitions_only(_load(result.source, result.function_name)))
    lines.append("")
    lines.append("")
    args = ", ".join(f"{k}={v!r}" for k, v in c.args.items())
    fv = sp.Rational(c.formula_value) if sp.sympify(c.formula_value).is_Rational else None
    lines.append(f"def test_{result.function_name}_matches_formula_at_counterexample():")
    lines.append(f"    # formula: {result.formula}")
    exact_type = isinstance(c.function_value, (int, fractions.Fraction)) and not isinstance(c.function_value, bool)
    if c.exact and exact_type and fv is not None and fv.is_Integer:
        lines.append(f"    assert {result.function_name}({args}) == {int(fv)}")
    elif c.exact and exact_type and fv is not None:
        lines.append("    from fractions import Fraction")
        lines.append(f"    assert Fraction({result.function_name}({args})) == Fraction({fv.p}, {fv.q})")
    else:
        tol = c.tolerance if c.tolerance > 0 else 1e-12
        lines.append(f"    assert {result.function_name}({args}) == pytest.approx({c.formula_approx!r}, rel={tol:.1e})")
    return "\n".join(lines) + "\n"
