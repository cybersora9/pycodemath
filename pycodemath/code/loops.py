"""C5 — how many times does this loop run, and what is its complexity?

``count_iterations(func_or_source)`` reads the function's loops from the AST (it
never executes anything to DERIVE the count) and returns:

* **an exact formula** for the number of executions of the innermost loop bodies
  — a SymPy expression in the function's size parameters, ``Sum`` closed to a
  compact form (``n*(n - 1)/2``), with the region where it holds (``valid_for``)
  and the region where the count is exactly 0 (``zero_for``);
* **Θ(·) with a proof** — the limit of ``count / g`` as ``n -> oo`` computed
  symbolically, with every ``floor``/``ceiling`` replaced by its argument plus a
  bounded unknown, so the limit holds whatever the fractional parts are;
* **a certificate by counting** — the function is instrumented (one counter per
  loop body), EXECUTED in the ``isolated`` worker for n = -3..30 and beyond, and
  every count compared with the formula exactly. A disagreement is never hidden:
  it is the certificate's ``witness``.

Supported loop shapes (anything else is ``LoopShapeError`` with the construct and
its line — never a guess):

* ``for v in range(a)``, ``range(a, b)``, ``range(a, b, s)`` (``s`` a non-zero
  integer constant, also negative), ``reversed(range(...))``, nested to any depth,
  bounds depending on parameters, outer loop variables and earlier assignments;
* ``for x in seq`` / ``for i, x in enumerate(seq)`` over a sequence PARAMETER
  (counted in ``len_seq``);
* ``while`` with one counter updated once per iteration: ``i += s`` / ``i -= s``,
  ``i *= k`` / ``i <<= k`` (doubling), ``i //= k`` / ``i >>= k`` (halving),
  ``while i * i <= n: i += 1``, ``while i: i //= 10``; the logarithms are exact
  (``floor_log``, ``ceil_log``: integer arithmetic, no float ``log``);
* ``if`` whose condition reads only parameters and loop variables may hold loops
  (the count becomes a ``Piecewise``); ``if`` on DATA may not hold a loop;
* ``break`` / ``return`` / ``raise`` inside a loop: the formula becomes an UPPER
  BOUND (``exact=False``), explicitly — the certificate then checks ``<=``.

``check_iterations(func, "n**2")`` and ``check_complexity(func, "O(n log n)")``
turn the count into VERIFIED / REFUTED (with the executed witness: ``n``, the
counter, the claim's value) / UNDECIDED (with the reason).

    >>> src = '''
    ... def pairs(n: int) -> int:
    ...     c = 0
    ...     for i in range(n):
    ...         for j in range(i + 1, n):
    ...             c += 1
    ...     return c
    ... '''
    >>> r = count_iterations(src, certify=False)
    >>> r.main, r.theta, r.valid_for
    (n*(n - 1)/2, n**2, 'n >= 1')
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import itertools
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..core.backstop import isolated
from ..core.budget import guard, time_budget
from ..core.errors import (
    DomainError,
    IsolationError,
    ParseError,
    PycodemathError,
    TimeBudgetError,
    UnsupportedFormError,
)
from ..verify.verdict import VerdictStatus
from .frontend import _Source, _definitions_only, _execution_blocker, _load, _qp_zero

__all__ = [
    "count_iterations",
    "check_iterations",
    "check_complexity",
    "theta_of",
    "parse_complexity_claim",
    "IterationCount",
    "LoopCount",
    "CountCertificate",
    "IterationCheck",
    "ComplexityCheck",
    "LoopShapeError",
    "floor_log",
    "ceil_log",
]


class LoopShapeError(UnsupportedFormError):
    """A loop (or a statement around it) outside C5's supported shapes.

    ``construct`` names it (``"while loop on data"``, ``"loop under a data-
    dependent if"``), ``line`` is its line in the source (1-based, ``None`` when
    unknown). Raised for what the reader cannot read, never for a mathematical
    reason: a loop that is read is counted exactly or bounded explicitly.
    """

    def __init__(self, construct: str, line: "int | None", hint: str = "") -> None:
        where = f" at line {line}" if line else ""
        super().__init__(
            f"count_iterations: unsupported construct — {construct}{where}"
            + (f" ({hint})" if hint else "")
        )
        self.construct = construct
        self.line = line
        self.hint = hint

    def __reduce__(self):  # crosses the ``isolated`` boundary intact
        return (type(self), (self.construct, self.line, self.hint))


# --------------------------------------------------------------------------- #
# exact integer logarithms                                                    #
# --------------------------------------------------------------------------- #


def _ilog_floor(x: sp.Rational, k: int) -> int:
    """Largest ``j`` with ``k**j <= x`` (``x > 0``), in exact arithmetic."""
    j, p = 0, sp.Integer(1)
    if x >= 1:
        while p * k <= x:
            p *= k
            j += 1
    else:
        while p > x:
            p /= k
            j -= 1
    return j


class floor_log(sp.Function):
    """``floor(log(x, k))`` computed EXACTLY for a positive rational ``x``.

    SymPy's ``floor(log(x)/log(k))`` decides the floor numerically, which is
    fragile exactly at the powers of ``k`` where loop counts change. This one is
    integer arithmetic.

        >>> floor_log(8, 2), floor_log(7, 2), floor_log(sp.Rational(1, 3), 2)
        (3, 2, -2)
    """

    nargs = 2
    is_integer = True

    @classmethod
    def eval(cls, x, k):  # type: ignore[override]
        if x.is_Rational and k.is_Integer and k > 1 and x > 0:
            return sp.Integer(_ilog_floor(x, int(k)))
        return None

    def _sympystr(self, printer: Any) -> str:
        x, k = self.args
        return f"floor(log({printer._print(x)}, {printer._print(k)}))"


class ceil_log(sp.Function):
    """``ceiling(log(x, k))`` computed exactly: the smallest ``j`` with ``k**j >= x``.

        >>> ceil_log(8, 2), ceil_log(9, 2), ceil_log(1, 3)
        (3, 4, 0)
    """

    nargs = 2
    is_integer = True

    @classmethod
    def eval(cls, x, k):  # type: ignore[override]
        if x.is_Rational and k.is_Integer and k > 1 and x > 0:
            j = _ilog_floor(x, int(k))
            return sp.Integer(j if sp.Integer(k) ** j == x else j + 1)
        return None

    def _sympystr(self, printer: Any) -> str:
        x, k = self.args
        return f"ceiling(log({printer._print(x)}, {printer._print(k)}))"


# --------------------------------------------------------------------------- #
# results                                                                     #
# --------------------------------------------------------------------------- #


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, VerdictStatus):
        return v.value
    if isinstance(v, sp.Basic):
        return str(v)
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "__dataclass_fields__"):
        return {k: _jsonable(getattr(v, k)) for k in v.__dataclass_fields__}
    return str(v)


class _ToDict:
    kind = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        out.update(_jsonable(self))
        return out


@dataclass(frozen=True)
class LoopCount(_ToDict):
    """One loop: how many times its body runs in one call of the function.

    ``count`` is in the function's parameters (outer loops already summed);
    ``leaf`` — no loop nested in its body (the leaves add up to the total);
    ``exact`` — ``False`` when a ``break``/``return``/``raise`` makes it an upper bound.
    """

    kind = "loop"
    line: int
    loop: str  # "for" | "while"
    count: sp.Expr
    leaf: bool
    exact: bool
    shape: str


@dataclass(frozen=True)
class CountCertificate(_ToDict):
    """Formula vs a real execution counter.

    ``relation`` is ``"=="`` for an exact formula, ``"<="`` for an upper bound;
    ``agree`` holds when it held at every one of ``points`` executed calls;
    ``witness`` is the first point where it did not (``{"n": 1, "counted": 0,
    "formula": 1}``). ``skipped`` lists points not compared and why (the call
    raised, timed out, or the formula claims nothing there). ``data_independent``:
    a sequence parameter was filled in two different orders and every count was
    the same — the count does not depend on the data.
    """

    kind = "count_certificate"
    points: int
    agree: bool
    relation: str
    domain: str
    witness: "dict[str, Any] | None"
    skipped: tuple[tuple[str, str], ...]
    data_independent: "bool | None"
    seconds: float


@dataclass(frozen=True)
class IterationCount(_ToDict):
    """The answer of ``count_iterations``.

    ``main`` is the closed formula; it holds where ``valid_for`` says (a SymPy
    condition as text, ``"True"`` everywhere), the count is exactly 0 where
    ``zero_for`` says, and ``formula`` is the whole ``Piecewise``. ``exact`` is
    ``False`` for an upper bound. ``closed`` is ``False`` when a ``Sum`` could not
    be closed (the formula is still exact). ``theta`` is ``g`` in Θ(g), proved by
    the limit ``theta_constant`` = lim count/g (``None`` with ``theta_reason``
    when no proof was found). ``variables`` maps each size symbol to its meaning
    (``"n"`` -> ``"parameter n"``, ``"len_a"`` -> ``"len(a)"``).
    """

    kind = "iterations"
    function: str
    main: sp.Expr
    formula: sp.Expr
    valid_for: str
    zero_for: str
    exact: bool
    closed: bool
    theta: "sp.Expr | None"
    theta_constant: "sp.Expr | None"
    theta_reason: str
    variables: Mapping[str, str]
    loops: tuple[LoopCount, ...]
    proof: tuple[str, ...]
    certificate: "CountCertificate | None"
    notes: tuple[str, ...] = ()
    valid_condition: sp.Basic = sp.true
    valid_from: "int | None" = None


@dataclass(frozen=True)
class IterationCheck(_ToDict):
    """``check_iterations``: does the loop run exactly ``claim`` times?"""

    kind = "iterations"
    verdict: VerdictStatus
    claim: sp.Expr
    count: sp.Expr
    method: str
    reason: str
    witness: "dict[str, Any] | None"
    result: IterationCount

    @property
    def main(self) -> sp.Expr:
        return self.result.main

    @property
    def theta(self) -> "sp.Expr | None":
        return self.result.theta

    @property
    def exact(self) -> bool:
        return self.result.exact

    @property
    def certificate(self) -> "CountCertificate | None":
        return self.result.certificate


@dataclass(frozen=True)
class ComplexityCheck(_ToDict):
    """``check_complexity``: is the iteration count O / Θ / Ω of the claim?

    ``limit`` is lim count/g (n -> oo), computed symbolically; it IS the proof.
    ``samples`` are executed counts at growing n with ``count/g`` — the evidence
    a reader can see, not the argument.
    """

    kind = "complexity"
    verdict: VerdictStatus
    claim: str
    bound: str  # "O" | "Theta" | "Omega" | "o" | "omega"
    g: sp.Expr
    limit: "sp.Expr | None"
    count: sp.Expr
    theta: "sp.Expr | None"
    method: str
    reason: str
    samples: tuple[tuple[int, int, str], ...]
    result: IterationCount

    @property
    def main(self) -> sp.Expr:
        return self.result.main

    @property
    def exact(self) -> bool:
        return self.result.exact

    @property
    def certificate(self) -> "CountCertificate | None":
        return self.result.certificate


# --------------------------------------------------------------------------- #
# reading                                                                     #
# --------------------------------------------------------------------------- #

_BUILTINS_READ = {"range", "len", "enumerate", "reversed", "min", "max", "abs", "int", "isqrt"}
_LENGTH_CHANGERS = {"append", "extend", "insert", "pop", "remove", "clear", "popitem"}
_SEQ_ANNOTATIONS = {"list", "List", "Sequence", "tuple", "Tuple", "str", "bytes", "MutableSequence"}
_NUM_ANNOTATIONS = {"int"}


@dataclass
class _Entry:
    line: int
    loop: str
    count: sp.Expr
    leaf: bool
    exact: bool
    shape: str


@dataclass
class _Block:
    entries: list[_Entry] = field(default_factory=list)
    # every condition here must hold (as ``expr >= 0``) for the generic formula
    assume: list[sp.Expr] = field(default_factory=list)
    # assumptions that could not be pushed through an enclosing loop (non-affine)
    unproved: list[str] = field(default_factory=list)
    escapes: bool = False  # a return/raise somewhere inside
    zero: "sp.Basic | None" = None  # for a loop's own block: when its range is empty


class _Reader:
    """The loop reader. ``params``: int parameters; ``seqs``: sequence parameters."""

    def __init__(self, src: _Source) -> None:
        self.src = src
        self.fdef = src.func
        self.name = src.func.name
        self.params: dict[str, sp.Symbol] = {}
        self.seqs: dict[str, sp.Symbol] = {}
        self.notes: list[str] = []
        self._dummies = itertools.count()
        args = self.fdef.args
        if args.vararg or args.kwarg:
            raise LoopShapeError("*args/**kwargs parameters", self.fdef.lineno)
        names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
        seq_used = self._sequence_params(names)
        for a in args.posonlyargs + args.args + args.kwonlyargs:
            ann = a.annotation
            ann_name = None
            if isinstance(ann, ast.Name):
                ann_name = ann.id
            elif isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
                ann_name = ann.value.id
            elif isinstance(ann, ast.Attribute):
                ann_name = ann.attr
            if a.arg in seq_used or ann_name in _SEQ_ANNOTATIONS:
                self.seqs[a.arg] = sp.Symbol(f"len_{a.arg}", integer=True, nonnegative=True)
            else:
                self.params[a.arg] = sp.Symbol(a.arg, integer=True)
        for node in ast.walk(self.fdef):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == self.name:
                raise LoopShapeError("recursion", node.lineno, "use recurrence_from_code for a recursive function")
        # a module that rebinds range/len/... is not the Python the reader assumes
        for node in ast.walk(src.tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _BUILTINS_READ:
                raise LoopShapeError(f"the module redefines {node.name}()", node.lineno)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in _BUILTINS_READ:
                raise LoopShapeError(f"the module rebinds {node.id}", node.lineno)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if isinstance(node, ast.ImportFrom) and node.module == "math" and alias.name == "isqrt" \
                            and alias.asname in (None, "isqrt"):
                        continue
                    if (alias.asname or alias.name) in _BUILTINS_READ:
                        raise LoopShapeError(f"an import rebinds {alias.asname or alias.name}", node.lineno)
        # a sequence whose LENGTH changes inside the function has no fixed count
        for node in ast.walk(self.fdef):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in self.seqs \
                    and node.func.attr in _LENGTH_CHANGERS:
                raise LoopShapeError(f"{node.func.value.id}.{node.func.attr}() changes the length of a "
                                     f"sequence the loops count", node.lineno)
            if isinstance(node, (ast.Delete, ast.AugAssign, ast.Assign)):
                targets = node.targets if isinstance(node, (ast.Delete, ast.Assign)) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id in self.seqs:
                        raise LoopShapeError(f"sequence parameter {t.id!r} is reassigned", node.lineno)
                    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Slice) \
                            and isinstance(t.value, ast.Name) and t.value.id in self.seqs:
                        raise LoopShapeError(f"slice assignment to {t.value.id!r} may change its length", node.lineno)
                    if isinstance(node, ast.Delete) and isinstance(t, ast.Subscript) \
                            and isinstance(t.value, ast.Name) and t.value.id in self.seqs:
                        raise LoopShapeError(f"del on {t.value.id!r} changes its length", node.lineno)

    def _sequence_params(self, names: Sequence[str]) -> set[str]:
        out: set[str] = set()
        for node in ast.walk(self.fdef):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
                if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in names:
                    out.add(node.args[0].id)
            if isinstance(node, ast.For):
                it = node.iter
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "enumerate":
                    it = it.args[0] if it.args else it
                if isinstance(it, ast.Name) and it.id in names:
                    out.add(it.id)
        return out

    def fresh(self, base: str) -> sp.Symbol:
        return sp.Symbol(f"{base}_{next(self._dummies)}", integer=True)

    # -- expressions ---------------------------------------------------------

    def expr(self, node: ast.expr, env: Mapping[str, "sp.Expr | None"]) -> sp.Expr:
        line = getattr(node, "lineno", None)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise LoopShapeError(f"non-integer constant {node.value!r} in a loop bound", line)
            return sp.Integer(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                v = env[node.id]
                if v is None:
                    raise LoopShapeError(
                        f"variable {node.id!r} in a loop bound changes inside a loop", line,
                        "its value is not known at this point")
                return v
            if node.id in self.params:
                return self.params[node.id]
            if node.id in self.seqs:
                raise LoopShapeError(f"sequence {node.id!r} used as a number", line)
            raise LoopShapeError(f"unknown name {node.id!r} in a loop bound", line)
        if isinstance(node, ast.UnaryOp):
            v = self.expr(node.operand, env)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            raise LoopShapeError(f"operator {type(node.op).__name__} in a loop bound", line)
        if isinstance(node, ast.BinOp):
            a, b = self.expr(node.left, env), self.expr(node.right, env)
            op = node.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.FloorDiv):
                if b == 0:
                    raise LoopShapeError("division by zero in a loop bound", line)
                return sp.floor(a / b)
            if isinstance(op, ast.Mod):
                if b == 0:
                    raise LoopShapeError("modulo by zero in a loop bound", line)
                return a - b * sp.floor(a / b)
            if isinstance(op, ast.Pow):
                if not (b.is_Integer and b >= 0):
                    raise LoopShapeError("power with a non-constant or negative exponent in a loop bound", line)
                return a**b
            if isinstance(op, ast.LShift) and b.is_Integer and b >= 0:
                return a * 2**b
            if isinstance(op, ast.RShift) and b.is_Integer and b >= 0:
                return sp.floor(a / 2**b)
            if isinstance(op, ast.Div):
                raise LoopShapeError("true division '/' in a loop bound", line, "range() needs integers")
            raise LoopShapeError(f"operator {type(op).__name__} in a loop bound", line)
        if isinstance(node, ast.Call):
            fname = _call_name(node)
            if fname == "len" and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                if node.args[0].id in self.seqs:
                    return self.seqs[node.args[0].id]
                raise LoopShapeError(f"len() of {node.args[0].id!r}, which is not a parameter", line)
            if fname in {"min", "max"} and len(node.args) >= 2 and not node.keywords:
                vals = [self.expr(a, env) for a in node.args]
                return sp.Min(*vals) if fname == "min" else sp.Max(*vals)
            if fname == "abs" and len(node.args) == 1:
                return sp.Abs(self.expr(node.args[0], env))
            if fname in {"isqrt", "math.isqrt"} and len(node.args) == 1:
                return sp.floor(sp.sqrt(self.expr(node.args[0], env)))
            if fname == "int" and len(node.args) == 1:
                return self.expr(node.args[0], env)
            raise LoopShapeError(f"call {fname or '<expr>'}() in a loop bound", line)
        raise LoopShapeError(f"{type(node).__name__} in a loop bound", line)

    def cond(self, node: ast.expr, env: Mapping[str, "sp.Expr | None"]) -> sp.Basic:
        """A condition over parameters / loop variables (for gating ``if``)."""
        line = getattr(node, "lineno", None)
        if isinstance(node, ast.BoolOp):
            parts = [self.cond(v, env) for v in node.values]
            return sp.And(*parts) if isinstance(node.op, ast.And) else sp.Or(*parts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return sp.Not(self.cond(node.operand, env))
        if isinstance(node, ast.Compare):
            terms = [node.left, *node.comparators]
            out = []
            for op, l, r in zip(node.ops, terms, terms[1:]):
                a, b = self.expr(l, env), self.expr(r, env)
                rel = {ast.Lt: sp.Lt, ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge,
                       ast.Eq: sp.Eq, ast.NotEq: sp.Ne}.get(type(op))
                if rel is None:
                    raise LoopShapeError(f"comparison {type(op).__name__}", line)
                out.append(rel(a, b))
            return sp.And(*out)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return sp.true if node.value else sp.false
        v = self.expr(node, env)  # truthiness of an integer
        return sp.Ne(v, 0)

    # -- statements ----------------------------------------------------------

    def function(self) -> _Block:
        env: dict[str, "sp.Expr | None"] = {}
        return self.block(self.fdef.body, env, depth=0)

    def block(self, stmts: Sequence[ast.stmt], env: dict[str, "sp.Expr | None"], depth: int) -> _Block:
        out = _Block()
        for idx, st in enumerate(stmts):
            if isinstance(st, ast.For):
                self._merge(out, self.for_loop(st, env, depth))
                for name in _assigned(st):
                    env[name] = None
            elif isinstance(st, ast.While):
                self._merge(out, self.while_loop(st, env, depth))
                for name in _assigned(st):
                    env[name] = None
            elif isinstance(st, ast.If):
                has_loop = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(st))
                ends = _always_leaves(st.body) and not st.orelse
                if not has_loop and ends and depth == 0:
                    # ``if n < 2: return 1`` at function level: the rest runs only
                    # when the condition is false — gate it.
                    try:
                        c = self.cond(st.test, env)
                    except LoopShapeError:
                        if any(isinstance(n, (ast.For, ast.While)) for s in stmts[idx + 1:] for n in ast.walk(s)):
                            raise LoopShapeError(
                                "early return under a condition that is not on parameters", st.lineno)
                        continue
                    rest = self.block(stmts[idx + 1:], env, depth)
                    self._merge(out, self._gate(rest, sp.Not(c)))
                    return out
                if has_loop:
                    try:
                        c = self.cond(st.test, env)
                    except LoopShapeError as exc:
                        raise LoopShapeError(
                            "loop under a data-dependent if", st.lineno,
                            f"the condition is not on parameters or loop variables: {exc.construct}") from None
                    env_a, env_b = dict(env), dict(env)
                    a = self.block(st.body, env_a, depth)
                    b = self.block(st.orelse, env_b, depth)
                    self._merge(out, self._gate(a, c))
                    self._merge(out, self._gate(b, sp.Not(c)))
                    for name in set(env_a) | set(env_b):
                        if env_a.get(name, None) != env_b.get(name, None) or name not in env:
                            env[name] = None
                    for name in _assigned(st):
                        env[name] = None
                else:
                    if _escapes(st):
                        out.escapes = True
                    for name in _assigned(st):
                        env[name] = None
            elif isinstance(st, (ast.Return, ast.Raise)):
                if depth > 0:
                    out.escapes = True
                return out  # nothing after it runs
            elif isinstance(st, (ast.Break, ast.Continue)):
                return out
            elif isinstance(st, ast.Assign):
                self._assign(st.targets, st.value, env)
            elif isinstance(st, ast.AnnAssign):
                if st.value is not None:
                    self._assign([st.target], st.value, env)
            elif isinstance(st, ast.AugAssign):
                if isinstance(st.target, ast.Name):
                    name = st.target.id
                    try:
                        cur = env[name] if name in env else None
                        if cur is None:
                            raise LoopShapeError("", None)
                        node = ast.BinOp(left=ast.Name(id=name, ctx=ast.Load()), op=st.op, right=st.value)
                        ast.copy_location(node, st)
                        env[name] = self.expr(node, env)
                    except LoopShapeError:
                        env[name] = None
            elif isinstance(st, (ast.Expr, ast.Pass, ast.Assert, ast.Global, ast.Nonlocal, ast.Delete)):
                if isinstance(st, (ast.Global, ast.Nonlocal)):
                    for name in st.names:
                        env[name] = None
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                raise LoopShapeError("nested function or class definition", st.lineno)
            else:
                if any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(st)):
                    raise LoopShapeError(f"loop inside a {type(st).__name__.lower()} statement", st.lineno)
                if _escapes(st):
                    out.escapes = True
                for name in _assigned(st):
                    env[name] = None
        return out

    def _assign(self, targets: Sequence[ast.expr], value: ast.expr, env: dict[str, "sp.Expr | None"]) -> None:
        names: list[str] = []
        simple = len(targets) == 1 and isinstance(targets[0], ast.Name)
        for t in targets:
            names.extend(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        if simple:
            try:
                env[names[0]] = self.expr(value, env)
                return
            except LoopShapeError:
                pass
        for name in names:
            env[name] = None

    @staticmethod
    def _merge(out: _Block, b: _Block) -> None:
        out.entries.extend(b.entries)
        out.assume.extend(b.assume)
        out.unproved.extend(b.unproved)
        out.escapes = out.escapes or b.escapes

    @staticmethod
    def _gate(b: _Block, cond: sp.Basic) -> _Block:
        cond = sp.simplify_logic(cond) if isinstance(cond, sp.logic.boolalg.Boolean) else cond
        if cond is sp.true:
            return b
        out = _Block(assume=list(b.assume), unproved=list(b.unproved), escapes=b.escapes)
        for e in b.entries:
            out.entries.append(_Entry(e.line, e.loop, sp.Piecewise((e.count, cond), (0, True)),
                                      e.leaf, e.exact, e.shape))
        return out

    # -- for -----------------------------------------------------------------

    def for_loop(self, st: ast.For, env: dict[str, "sp.Expr | None"], depth: int) -> _Block:
        if st.orelse:
            raise LoopShapeError("for ... else", st.lineno)
        it = st.iter
        target = st.target
        line = st.lineno
        var_name: "str | None" = None
        values: "tuple[sp.Expr, sp.Expr] | None" = None  # (first, step) of the loop variable
        other_targets: list[str] = []
        if isinstance(it, ast.Call) and _call_name(it) == "reversed" and len(it.args) == 1 \
                and isinstance(it.args[0], ast.Call) and _call_name(it.args[0]) == "range":
            it = it.args[0]
        if isinstance(it, ast.Call) and _call_name(it) == "range":
            if it.keywords or not 1 <= len(it.args) <= 3:
                raise LoopShapeError("range() with unusual arguments", line)
            vals = [self.expr(a, env) for a in it.args]
            if len(vals) == 1:
                a, b, s = sp.Integer(0), vals[0], sp.Integer(1)
            elif len(vals) == 2:
                a, b, s = vals[0], vals[1], sp.Integer(1)
            else:
                a, b, s = vals
            if not s.is_Integer:
                raise LoopShapeError("range() step that is not an integer constant", line)
            if s == 0:
                raise LoopShapeError("range() step 0", line, "Python raises ValueError")
            if s > 0:
                T = (b - a) if s == 1 else sp.ceiling((b - a) / s)
                assume = b - a + s - 1
                zero = sp.Le(b - a, 0)
            else:
                T = (a - b) if s == -1 else sp.ceiling((a - b) / -s)
                assume = a - b - s - 1
                zero = sp.Le(a - b, 0)
            if not isinstance(target, ast.Name):
                raise LoopShapeError("tuple target over range()", line)
            var_name = target.id
            values = (a, s)
            shape = f"range({', '.join(str(v) for v in vals)})"
        elif isinstance(it, ast.Name) and it.id in self.seqs:
            T, assume, zero = self.seqs[it.id], sp.Integer(0), sp.Eq(self.seqs[it.id], 0)
            other_targets = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
            shape = f"for over {it.id}"
        elif isinstance(it, ast.Call) and _call_name(it) == "enumerate" and it.args \
                and isinstance(it.args[0], ast.Name) and it.args[0].id in self.seqs:
            T = self.seqs[it.args[0].id]
            assume, zero = sp.Integer(0), sp.Eq(T, 0)
            start = sp.Integer(0)
            if len(it.args) > 1:
                start = self.expr(it.args[1], env)
            for kw in it.keywords:
                if kw.arg == "start":
                    start = self.expr(kw.value, env)
            if isinstance(target, ast.Tuple) and len(target.elts) == 2 and isinstance(target.elts[0], ast.Name):
                var_name = target.elts[0].id
                values = (start, sp.Integer(1))
                other_targets = [n.id for n in ast.walk(target.elts[1]) if isinstance(n, ast.Name)]
            else:
                other_targets = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
            shape = f"enumerate({it.args[0].id})"
        elif isinstance(it, (ast.List, ast.Tuple)) and not any(isinstance(e, ast.Starred) for e in it.elts):
            T, assume, zero = sp.Integer(len(it.elts)), sp.Integer(0), sp.false
            other_targets = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
            shape = f"for over a {len(it.elts)}-element literal"
        elif isinstance(it, ast.Constant) and isinstance(it.value, str):
            T, assume, zero = sp.Integer(len(it.value)), sp.Integer(0), sp.false
            other_targets = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
            shape = "for over a string literal"
        else:
            raise LoopShapeError(f"for over {_describe(it)}", line,
                                 "supported: range(), reversed(range()), a sequence parameter, enumerate()")
        inner_env = dict(env)
        for name in _assigned_in(st.body):
            inner_env[name] = None
        for name in other_targets:
            inner_env[name] = None
        var_sym: "sp.Symbol | None" = None
        if var_name is not None:
            var_sym = self.fresh(var_name)
            inner_env[var_name] = var_sym
        body = self.block(st.body, inner_env, depth + 1)
        exact = not (_breaks(st.body) or body.escapes)
        return self._close(st, body, T, assume, zero, var_sym, values, exact, "for", shape)

    def _close(self, st: "ast.For | ast.While", body: _Block, T: sp.Expr, assume: sp.Expr, zero: sp.Basic,
               var_sym: "sp.Symbol | None", values: "tuple[sp.Expr, sp.Expr] | None",
               exact: bool, loop: str, shape: str) -> _Block:
        """Sum the body's counts over this loop; push the body's assumptions out."""
        out = _Block(escapes=body.escapes)
        t = sp.Symbol(f"t_{next(self._dummies)}", integer=True, nonnegative=True)
        Tm = sp.Symbol(f"T_{next(self._dummies)}", integer=True, nonnegative=True)
        subs_val = None
        if var_sym is not None and values is not None:
            first, step = values
            subs_val = first + step * t
        for e in body.entries:
            c = e.count
            if var_sym is not None and c.has(var_sym):
                assert subs_val is not None
                summed = _sum(c.subs(var_sym, subs_val), t, Tm, out)
                c = summed.subs(Tm, T)
            else:
                c = c * T
            out.entries.append(_Entry(e.line, e.loop, c, e.leaf, e.exact and exact, e.shape))
        leaf = not body.entries
        out.entries.append(_Entry(st.lineno, loop, T, leaf, exact, shape))
        # this loop's own assumption, then the body's — quantified over the variable
        out.assume.append(assume)
        for a in body.assume:
            if var_sym is None or not a.has(var_sym):
                out.assume.append(a)
                continue
            assert values is not None and subs_val is not None
            first, step = values
            a_t = sp.expand(a.subs(var_sym, subs_val))
            if sp.Poly(a_t, t).degree() <= 1 if a_t.is_polynomial(t) else False:
                # affine in the variable: its endpoints decide it over the whole range
                out.assume.append(sp.expand(a_t.subs(t, 0)))
                out.assume.append(sp.expand(a_t.subs(t, T - 1)))
            else:
                out.unproved.append(f"{a} >= 0 for every {var_sym} of the loop at line {st.lineno}")
        out.unproved.extend(body.unproved)
        out.zero = zero
        return out

    # -- while ---------------------------------------------------------------

    def while_loop(self, st: ast.While, env: dict[str, "sp.Expr | None"], depth: int) -> _Block:
        line = st.lineno
        if st.orelse:
            raise LoopShapeError("while ... else", line)
        if isinstance(st.test, ast.Constant):
            raise LoopShapeError("while True (the exit is data-dependent)", line)
        var, op, bound_node, squared = _while_test(st.test)
        if var is None:
            raise LoopShapeError("while loop whose condition is not 'counter <op> bound'", line,
                                 "supported: i < n, i <= n, i > c, i >= c, i * i <= n, while i")
        if var in env:
            c0 = env[var]
        else:
            c0 = self.params.get(var)  # ``while n: n //= 10`` on the parameter itself
        if c0 is None:
            raise LoopShapeError(f"while counter {var!r} has no known value before the loop", line,
                                 "a loop driven by data (binary search, two pointers) has no exact count")
        assert c0 is not None
        update = _counter_update(st.body, var)
        if update is None:
            raise LoopShapeError(f"while loop that does not update {var!r} exactly once per iteration", line,
                                 "the update must be a plain statement in the loop body, not under if")
        kind, k = update
        assigned = _assigned_in(st.body)
        if bound_node is None:
            B = sp.Integer(0)
        else:
            bound_names = {n.id for n in ast.walk(bound_node) if isinstance(n, ast.Name)}
            if bound_names & (assigned | {var}):
                raise LoopShapeError("while bound that changes inside the loop", line)
            B = self.expr(bound_node, env)
        if any(isinstance(n, ast.Continue) for n in _own_statements(st.body)):
            raise LoopShapeError("continue in a while loop", line, "it may skip the counter update")
        count, assume, zero = _while_count(op, kind, k, c0, B, squared, line)
        inner_env = dict(env)
        for name in assigned | {var}:
            inner_env[name] = None
        body = self.block(st.body, inner_env, depth + 1)
        exact = not (_breaks(st.body) or body.escapes)
        out = _Block(escapes=body.escapes)
        for e in body.entries:
            # the body cannot read the counter as a bound (it is unknown inside)
            out.entries.append(_Entry(e.line, e.loop, e.count * count, e.leaf, e.exact and exact, e.shape))
        out.entries.append(_Entry(line, "while", count, not body.entries, exact, _while_shape(op, kind, k, var)))
        out.assume.extend(assume)
        out.assume.extend(body.assume)
        out.unproved.extend(body.unproved)
        out.zero = zero
        return out


def _while_shape(op: str, kind: str, k: int, var: str) -> str:
    sym = {"add": "+=", "sub": "-=", "mul": "*=", "div": "//="}[kind]
    return f"while {var} {op} …: {var} {sym} {k}"


def _while_count(op: str, kind: str, k: int, c: sp.Expr, B: sp.Expr, squared: bool,
                 line: int) -> tuple[sp.Expr, list[sp.Expr], sp.Basic]:
    """(count, assumptions ``>= 0``, zero-condition) of a counter loop.

    Each formula is the number of ``j >= 0`` for which the condition holds on the
    ``j``-th value of the counter; derivations in the comments, exact over the
    integers (``floor(x/k**j)`` iterates because ``floor(floor(x/a)/b) = floor(x/(a*b))``).
    """
    if squared:
        if kind != "add" or k != 1 or op not in {"<", "<="}:
            raise LoopShapeError("while i*i … that is not 'i += 1' upward", line)
        if op == "<=":  # v*v <= B for v = c, c+1, …: v <= isqrt(B)
            return sp.floor(sp.sqrt(B)) - c + 1, [c, B - c**2], sp.Lt(B, c**2)
        return sp.ceiling(sp.sqrt(B)) - c, [c, B - c**2], sp.Le(B, c**2)  # v*v < B: v < ceil(sqrt B)
    if kind == "add":
        if op == "<":
            return (sp.ceiling((B - c) / k) if k != 1 else B - c), [B - c + k - 1], sp.Le(B, c)
        if op == "<=":
            return sp.floor((B - c) / k) + 1, [B - c + k], sp.Lt(B, c)
        if op == "!=":
            raise LoopShapeError("while i != n with i += k (terminates only when n - i is a multiple of k)", line)
    if kind == "sub":
        if op == ">":
            return (sp.ceiling((c - B) / k) if k != 1 else c - B), [c - B + k - 1], sp.Le(c, B)
        if op == ">=":
            return sp.floor((c - B) / k) + 1, [c - B + k], sp.Lt(c, B)
        if op == "!=":
            raise LoopShapeError("while i != n with i -= k", line)
    if kind == "mul":
        if c.is_number and c <= 0:
            raise LoopShapeError(f"while loop multiplying a counter that starts at {c}", line,
                                 "it never grows: the loop does not terminate")
        if op == "<":  # c*k**j >= B first at j = ceil(log_k(B/c))
            return ceil_log(B / c, k), [c - 1, B - c], sp.Le(B, c)
        if op == "<=":  # c*k**j > B first at j = floor(log_k(B/c)) + 1
            return floor_log(B / c, k) + 1, [c - 1, B - c], sp.Lt(B, c)
    if kind == "div":
        if op in {">", "!="}:
            if op == "!=" and B != 0:
                raise LoopShapeError("while i != n with i //= k", line)
            if B.is_number and B < 0:
                raise LoopShapeError(f"while i > {B} with i //= k", line, "floor division stops at -1: never ends")
            # floor(c/k**j) > B  <=>  c/k**j >= B+1  <=>  k**j <= c/(B+1)
            return floor_log(c / (B + 1), k) + 1, [B, c - B - 1], sp.Le(c, B)
        if op == ">=":
            if B.is_number and B <= 0:
                raise LoopShapeError(f"while i >= {B} with i //= k", line, "floor division stops at 0: never ends")
            return floor_log(c / B, k) + 1, [B - 1, c - B], sp.Lt(c, B)
    raise LoopShapeError(f"while counter loop with '{op}' and a counter that moves the other way", line,
                         "the loop either never runs or never ends")


def _while_test(test: ast.expr) -> tuple["str | None", str, "ast.expr | None", bool]:
    """``(counter, op, bound, squared)``; normalised so the counter is on the left."""
    if isinstance(test, ast.Name):
        return test.id, "!=", None, False
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None, "", None, False
    left, right, op = test.left, test.comparators[0], test.ops[0]
    flip = {ast.Lt: ">", ast.LtE: ">=", ast.Gt: "<", ast.GtE: "<=", ast.NotEq: "!="}
    keep = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.NotEq: "!="}
    if type(op) not in keep:
        return None, "", None, False

    def counter(n: ast.expr) -> "tuple[str | None, bool]":
        if isinstance(n, ast.Name):
            return n.id, False
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and isinstance(n.left, ast.Name) \
                and isinstance(n.right, ast.Name) and n.left.id == n.right.id:
            return n.left.id, True
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) and isinstance(n.left, ast.Name) \
                and isinstance(n.right, ast.Constant) and n.right.value == 2:
            return n.left.id, True
        return None, False

    v, sq = counter(left)
    if v is not None:
        return v, keep[type(op)], right, sq
    v, sq = counter(right)
    if v is not None:
        return v, flip[type(op)], left, sq
    return None, "", None, False


def _counter_update(body: Sequence[ast.stmt], var: str) -> "tuple[str, int] | None":
    """The one top-level update of ``var`` in the body: ``("mul", 2)`` etc."""
    updates = [s for s in body if _assigns(s, var)]
    nested = [n for s in body for n in ast.walk(s) if n is not s and isinstance(n, ast.stmt) and _assigns(n, var)]
    if len(updates) != 1 or nested:
        return None
    s = updates[0]
    if isinstance(s, ast.AugAssign) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, int) \
            and not isinstance(s.value.value, bool):
        k = s.value.value
        kinds = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.FloorDiv: "div",
                 ast.LShift: "mul", ast.RShift: "div"}
        kind = kinds.get(type(s.op))
        if kind is None:
            return None
        if isinstance(s.op, (ast.LShift, ast.RShift)):
            k = 2**k
        if k <= 0 or (kind in {"mul", "div"} and k < 2):
            return None
        return kind, k
    if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.value, ast.BinOp):
        b = s.value
        sides = [(b.left, b.right), (b.right, b.left)] if isinstance(b.op, (ast.Add, ast.Mult)) else [(b.left, b.right)]
        for x, kn in sides:
            if isinstance(x, ast.Name) and x.id == var and isinstance(kn, ast.Constant) \
                    and isinstance(kn.value, int) and not isinstance(kn.value, bool):
                fake = ast.AugAssign(target=ast.Name(id=var, ctx=ast.Store()), op=b.op, value=kn)
                return _counter_update([fake], var)
    return None


def _assigns(s: ast.stmt, var: str) -> bool:
    if isinstance(s, ast.AugAssign):
        return isinstance(s.target, ast.Name) and s.target.id == var
    if isinstance(s, (ast.Assign, ast.AnnAssign)):
        targets = s.targets if isinstance(s, ast.Assign) else [s.target]
        return any(isinstance(n, ast.Name) and n.id == var for t in targets for n in ast.walk(t))
    if isinstance(s, ast.For):
        return any(isinstance(n, ast.Name) and n.id == var for n in ast.walk(s.target))
    return False


def _own_statements(body: Sequence[ast.stmt]) -> Iterable[ast.AST]:
    """Nodes of the body that belong to THIS loop (not to a nested loop or def)."""
    stack: list[ast.AST] = list(body)
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def _breaks(body: Sequence[ast.stmt]) -> bool:
    return any(isinstance(n, ast.Break) for n in _own_statements(body))


def _escapes(st: ast.AST) -> bool:
    return any(isinstance(n, (ast.Return, ast.Raise)) for n in ast.walk(st))


def _always_leaves(body: Sequence[ast.stmt]) -> bool:
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise))


def _assigned(st: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(st):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
    return out


def _assigned_in(body: Sequence[ast.stmt]) -> set[str]:
    out: set[str] = set()
    for s in body:
        out |= _assigned(s)
    return out


def _call_name(node: ast.Call) -> "str | None":
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f"{f.value.id}.{f.attr}"
    return None


def _describe(node: ast.expr) -> str:
    try:
        text = ast.unparse(node)
    except Exception:  # noqa: BLE001
        text = type(node).__name__
    return text if len(text) <= 40 else text[:39] + "…"


# --------------------------------------------------------------------------- #
# closed-form sums                                                            #
# --------------------------------------------------------------------------- #


def _sum(f: sp.Expr, t: sp.Symbol, Tm: sp.Symbol, block: _Block) -> sp.Expr:
    """``Sum(f, (t, 0, Tm - 1))`` in closed form where possible (exact either way).

    Three steps beyond plain ``summation``: ``floor``/``ceiling`` of an argument
    linear in ``t`` are removed by splitting ``t`` by residue (``t = q*w + r`` makes
    ``floor(p*t/q + d) = p*w + floor(p*r/q + d)``); ``floor_log(t + d, k)`` uses
    the exact identity ``sum_{x=1..N} floor_log(x, k) = (N+1)L - (k**(L+1)-k)/(k-1)``
    with ``L = floor_log(N, k)``; a ``Piecewise`` is summed by SymPy. Anything else
    left over stays a ``Sum`` — exact, just not closed.
    """
    f = sp.expand(f) if not f.has(sp.Piecewise) else f
    if not f.has(t):
        return f * Tm
    if isinstance(f, sp.Add):
        return sp.Add(*[_sum(a, t, Tm, block) for a in f.args])
    coeff, rest = f.as_independent(t, as_Add=False)
    if coeff != 1:
        return coeff * _sum(rest, t, Tm, block)
    fl = [a for a in f.atoms(sp.floor, sp.ceiling) if a.has(t)]
    if fl:
        q = 1
        ok = True
        for a in fl:
            arg = sp.expand(a.args[0])
            if not arg.is_polynomial(t) or sp.Poly(arg, t).degree() != 1:
                ok = False
                break
            c = sp.Poly(arg, t).coeffs()[0]
            if not c.is_Rational:
                ok = False
                break
            q = int(sp.ilcm(q, sp.Rational(c).q))
        if ok and q > 1:
            w = sp.Symbol(f"w_{t.name}", integer=True, nonnegative=True)
            total = sp.Integer(0)
            for r in range(q):
                g = sp.expand(f.subs(t, q * w + r))
                W = sp.Symbol(f"W_{t.name}_{r}", integer=True, nonnegative=True)
                part = _sum(g, w, W, block)
                total += part.subs(W, sp.ceiling((Tm - r) / q))
            return total
        if ok:  # coefficient integer: floor of (integer*t + d) = integer*t + floor(d)
            g = f.replace(lambda e: isinstance(e, (sp.floor, sp.ceiling)) and e.has(t),
                          lambda e: _pull_integer_t(e, t))
            if g != f:
                return _sum(g, t, Tm, block)
    fls = [a for a in f.atoms(floor_log) if a.has(t)]
    if len(fls) == 1 and f == fls[0]:
        x, k = fls[0].args
        d = sp.expand(x - t)
        if not d.has(t) and k.is_Integer:
            # sum_{t=0}^{Tm-1} floor_log(t + d, k) = S(Tm - 1 + d) - S(d - 1)
            block.assume.append(d - 1)  # every argument is >= 1
            return _S_floor_log(Tm - 1 + d, int(k)) - _S_floor_log(d - 1, int(k))
    try:
        res = guard("summation", f, lambda: sp.summation(f, (t, 0, Tm - 1)))
    except (TimeBudgetError, PycodemathError):
        return sp.Sum(f, (t, 0, Tm - 1))
    except Exception:  # noqa: BLE001 — SymPy's own failure: leave the sum unevaluated
        return sp.Sum(f, (t, 0, Tm - 1))
    if isinstance(res, sp.Piecewise):
        # SymPy's guard for an empty range (Tm >= 1): Karr's convention makes the
        # generic branch right at Tm = 0 too, so keep the generic branch.
        res = res.args[0][0]
    return sp.expand(res) if not res.has(sp.Sum) else res


def _S_floor_log(N: sp.Expr, k: int) -> sp.Expr:
    """``sum_{x=1..N} floor_log(x, k)``, exact for ``N >= 0`` (``0`` at ``N = 0``)."""
    if N.is_Integer and N <= 0:
        return sp.Integer(0)
    L = floor_log(N, k)
    return sp.Piecewise((0, sp.Le(N, 0)), ((N + 1) * L - (sp.Integer(k) ** (L + 1) - k) / (k - 1), True))


def _pull_integer_t(e: sp.Basic, t: sp.Symbol) -> sp.Basic:
    arg = sp.expand(e.args[0])
    lin, rest = arg.as_independent(t, as_Add=True)[1], arg.as_independent(t, as_Add=True)[0]
    return lin + type(e)(rest)


# --------------------------------------------------------------------------- #
# Θ by a symbolic limit                                                       #
# --------------------------------------------------------------------------- #


def _smooth(expr: sp.Expr, n: sp.Symbol) -> tuple[sp.Expr, list[sp.Symbol]]:
    """Replace every floor/ceiling (and floor_log/ceil_log) by its argument plus a
    bounded unknown ``theta_i`` in ``[-1, 1]`` (``k**floor_log(x, k)`` by ``x`` times
    one in ``[1/k, 1]``); returns the expression and the unknowns. A limit that does not depend on them holds for every value of the
    fractional parts — that is the whole argument for floors."""
    thetas: list[sp.Symbol] = []

    def th() -> sp.Symbol:
        s = sp.Symbol(f"theta_{len(thetas)}", real=True)
        thetas.append(s)
        return s

    # harmonic(x) - log(x) - EulerGamma lies in [0, 1] for x >= 1
    for h in expr.atoms(sp.harmonic):
        if len(h.args) == 1:
            expr = expr.xreplace({h: sp.log(h.args[0]) + sp.EulerGamma + th()})
    # k**floor_log(x, k) lies in (x/k, x] and k**ceil_log(x, k) in [x, k*x): write
    # them as x times a bounded unknown, which keeps the expression polynomial in it
    for pw in sorted(expr.atoms(sp.Pow), key=sp.count_ops):
        if isinstance(pw.exp, (floor_log, ceil_log)) and pw.base == pw.exp.args[1]:
            expr = expr.xreplace({pw: pw.exp.args[0] * th()})
    changed = True
    while changed:
        changed = False
        for a in sorted(expr.atoms(sp.floor, sp.ceiling, floor_log, ceil_log), key=sp.count_ops):
            if any(b != a and a.has(b) for b in expr.atoms(sp.floor, sp.ceiling, floor_log, ceil_log)):
                continue
            if isinstance(a, (floor_log, ceil_log)):
                x, k = a.args
                rep = sp.log(x) / sp.log(k) + th()
            else:
                rep = a.args[0] + th()
            expr = expr.xreplace({a: rep})
            changed = True
            break
    return expr, thetas


def _eventual_branch(expr: sp.Expr, n: sp.Symbol) -> sp.Expr:
    """The branch of a ``Piecewise`` that holds for all large ``n``."""
    if not isinstance(expr, sp.Piecewise):
        return expr.replace(lambda e: isinstance(e, sp.Piecewise), lambda e: _eventual_branch(e, n)) \
            if expr.has(sp.Piecewise) else expr
    for e, c in expr.args:
        if c is sp.true or _eventually(c, n):
            return _eventual_branch(e, n)
    raise ValueError("no branch holds eventually")


def _eventually(c: sp.Basic, n: sp.Symbol) -> bool:
    try:
        s = sp.solveset(c, n, sp.S.Reals) if c.free_symbols == {n} else None
    except Exception:  # noqa: BLE001
        s = None
    if s is None:
        return False
    try:
        comp = sp.S.Reals - s
        sup = comp.sup if comp is not sp.S.EmptySet else -sp.oo
        return bool(sup != sp.oo and sp.Interval(sup + 1, sp.oo).is_subset(s))
    except Exception:  # noqa: BLE001
        return False


def _limit(expr: sp.Expr, n: sp.Symbol) -> "sp.Expr | None":
    x = sp.Symbol("x_lim", positive=True)
    e = expr.subs(n, x)
    try:
        return guard("limit", e, lambda: sp.limit(e, x, sp.oo))
    except Exception:  # noqa: BLE001 — SymPy could not take it: no proof
        return None


def _limit_over_thetas(expr: sp.Expr, n: sp.Symbol, thetas: Sequence[sp.Symbol]) -> "sp.Expr | None":
    """lim expr as n -> oo, required to be the same for every value of the thetas."""
    if thetas:
        e = sp.expand(expr)
        try:
            poly = sp.Poly(e, *thetas)
        except sp.PolynomialError:
            poly = None
        if poly is not None:
            # polynomial in the bounded unknowns: each coefficient converges, so the
            # convergence is uniform on the box; the limit must not depend on them
            total = None
            others: list[sp.Expr] = []
            for monom, coeff in poly.terms():
                L = _limit(coeff, n)
                if L is None or L.has(sp.AccumBounds) or not L.is_finite and L not in (sp.oo, -sp.oo):
                    return None
                if any(monom):
                    others.append(L)
                else:
                    total = L
            total = total if total is not None else sp.Integer(0)
            if total in (sp.oo, -sp.oo):
                # bounded multiples of the unknowns cannot stop a divergence
                return total if all(o.is_finite for o in others) else None
            return total if all(o == 0 for o in others) else None
    L = _limit(expr, n)
    if L is None or L.has(*thetas) if thetas else L is None:
        return None
    return L


def theta_of(expr: sp.Expr, n: "sp.Symbol | str") -> tuple["sp.Expr | None", "sp.Expr | None", str]:
    """``(g, L, reason)``: ``expr = Θ(g)`` with ``L = lim expr/g`` in ``(0, oo)``.

    ``g`` is SymPy's leading term at infinity (``Order``) with the constant
    dropped; the claim is then PROVED by the limit, which treats floors as their
    argument plus a bounded unknown. ``(None, None, reason)`` when no proof.

        >>> n = sp.Symbol("n", integer=True)
        >>> theta_of(n*(n - 1)/2, n)[:2]
        (n**2, 1/2)
        >>> theta_of(n*ceil_log(n, 2), n)[:2]
        (n*log(n), 1/log(2))
    """
    if isinstance(n, str):
        n = sp.Symbol(n, integer=True)
    try:
        expr = _eventual_branch(expr, n)
    except ValueError:
        return None, None, "no branch of the Piecewise holds for all large n"
    if expr.has(sp.Sum):
        try:
            expr = expr.doit()
        except Exception:  # noqa: BLE001
            pass
        if expr.has(sp.Sum):
            return None, None, "the count is a Sum without a closed form"
    if expr.free_symbols - {n}:
        return None, None, "Θ needs a single size variable"
    def eventual_minmax(e: sp.Basic) -> sp.Basic:
        args = list(e.args)
        best = args[0]
        for a in args[1:]:
            L = _limit(a - best, n)
            if L is None or L.has(sp.AccumBounds) or L == 0:
                return e
            if (L.is_negative or L == -sp.oo) if isinstance(e, sp.Min) else (L.is_positive or L == sp.oo):
                best = a
        return best

    expr = expr.replace(lambda e: isinstance(e, (sp.Min, sp.Max)), eventual_minmax)
    if not expr.has(n):
        return (sp.Integer(1), expr, "constant") if expr != 0 else (None, None, "the count is 0")
    smooth, thetas = _smooth(expr, n)
    plain = smooth.subs({t: 0 for t in thetas})
    x = sp.Symbol("x_lim", positive=True)
    try:
        g = guard("order", plain, lambda: sp.O(sp.expand(plain.subs(n, x)), (x, sp.oo)).expr)
    except Exception:  # noqa: BLE001
        g = None
    if g is None or g == 0:
        return None, None, "SymPy found no leading term at infinity"
    g = sp.expand_log(sp.powsimp(g), force=True).subs(x, n)
    if sp.expand(g).is_Add:
        g = _dominant_term(g, n)
    g = _monic(g)
    if g == 0:
        return None, None, "the count tends to 0"
    L = _limit_over_thetas(smooth / g, n, thetas)
    if L is not None and L.is_negative:
        g, L = -g, -L
    if L is None or not (L.is_positive and L.is_finite):
        return None, None, f"lim count/({g}) is {L}, not a positive constant"
    return g, sp.nsimplify(L) if L.is_Float else L, "limit"


def _dominant_term(g: sp.Expr, n: sp.Symbol) -> sp.Expr:
    """The term of a sum that every other term is O of (``n*log(log n)`` out of
    ``n*(log(log n) - log(log 2) + 1)``); the sum itself when there is none."""
    terms = sp.Add.make_args(sp.expand(g))
    for t in terms:
        if not t.has(n):
            continue
        if all(o is t or (lambda L: L is not None and L.is_finite)(_limit(o / t, n)) for o in terms):
            return t
    return g


def _monic(g: sp.Expr) -> sp.Expr:
    if g.has(sp.exp):
        g = sp.powsimp(sp.simplify(g))
    c, rest = g.as_coeff_Mul()
    if rest.is_Add:
        return rest
    return rest if c != 0 else g


# --------------------------------------------------------------------------- #
# certificate by counting                                                     #
# --------------------------------------------------------------------------- #


class _Instrument(ast.NodeTransformer):
    """Puts ``__pcm_counts__[line] += 1`` first in every loop body of one function,
    and ``__pcm_counts__[0] += 1`` first in the function (calls)."""

    def __init__(self, fname: str) -> None:
        self.fname = fname

    @staticmethod
    def _tick(line: int) -> ast.stmt:
        return ast.parse(f"__pcm_counts__[{line}] += 1").body[0]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        if node.name == self.fname:
            node.body.insert(_docstring_len(node), self._tick(0))
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        node.body.insert(0, self._tick(node.lineno))
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.body.insert(0, self._tick(node.lineno))
        return node


def _docstring_len(node: ast.FunctionDef) -> int:
    b = node.body
    first = b[0] if b else None
    return 1 if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
        and isinstance(first.value.value, str) else 0


def _instrumented_source(src: _Source) -> tuple[str, str]:
    """``(source to exec, how)`` — the whole module's definitions, the subject function
    instrumented. Lines in the counters are the lines of ``src.text``."""
    tree = ast.parse(src.text)
    _Instrument(src.func.name).visit(tree)
    ast.fix_missing_locations(tree)
    only = _Source(src.text, tree, src.func, src.obj, src.module)
    return _definitions_only(only), "source"


def _count_worker(spec: tuple, calls: list[dict[str, Any]], per_call: float) -> list[tuple[str, Any]]:
    """Worker side: run the instrumented function once per argument dict."""
    import collections
    import sys

    kind, code, fname, module = spec
    ns: dict[str, Any] = {}
    if kind == "module" and module:
        ns.update(vars(importlib.import_module(module)))
    ns["__name__"] = "__pcm_c5_subject__"
    counts: "collections.defaultdict[int, int]" = collections.defaultdict(int)
    ns["__pcm_counts__"] = counts
    exec(compile(code, "<counted function>", "exec"), ns)  # noqa: S102 — the certificate IS execution
    fn = ns[fname]
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 20000))
    out: list[tuple[str, Any]] = []
    try:
        for kwargs in calls:
            counts.clear()
            started = time.perf_counter()
            try:
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    warnings.simplefilter("ignore")
                    with time_budget(per_call, "count_iterations execution"):
                        fn(**kwargs)
                out.append(("ok", dict(counts)))
            except TimeBudgetError:
                out.append(("timeout", dict(counts)))
            except RecursionError:
                out.append(("raise", "RecursionError"))
            except Exception as exc:  # noqa: BLE001 — the function's own failure is data
                out.append(("raise", f"{type(exc).__name__}: {exc}"[:160]))
            del started
    finally:
        sys.setrecursionlimit(old)
    return out


def _exec_spec(src: _Source) -> "tuple[tuple, str | None]":
    """How the worker gets the instrumented function, or why it must not run it."""
    code, _ = _instrumented_source(src)
    blocked = _execution_blocker(src)
    if blocked is None:
        return ("source", code, src.func.name, None), None
    if src.obj is not None and src.module and src.module != "__main__":
        # the caller imported this module already; the worker imports it too and
        # runs the instrumented copy of the one function in that namespace
        func_only = ast.Module(body=[ast.parse(code).body[-1]], type_ignores=[])
        tree = ast.parse(src.text)
        _Instrument(src.func.name).visit(tree)
        ast.fix_missing_locations(tree)
        fdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == src.func.name]
        if fdefs:
            func_only = ast.Module(body=[fdefs[-1]], type_ignores=[])
        return ("module", ast.unparse(func_only), src.func.name, src.module), None
    return (), blocked


@dataclass
class _Run:
    point: dict[str, int]
    status: str
    counts: "dict[int, int] | None"
    detail: str = ""


def _run_points(src: _Source, reader: "_Reader", points: list[dict[str, int]],
                budget: float) -> tuple[list[_Run], "bool | None", "str | None"]:
    """Execute at every point; sequence parameters as ``list(range(L))`` and again
    reversed (``data_independent``). Returns ``(runs, data_independent, blocked)``."""
    spec, blocked = _exec_spec(src)
    if blocked is not None:
        return [], None, blocked
    defaults = _defaults(reader.fdef)
    calls: list[dict[str, Any]] = []
    for p in points:
        kw: dict[str, Any] = {}
        for name in reader.params:
            if name in p:
                kw[name] = p[name]
            elif name not in defaults:
                kw[name] = 1
        for name, sym in reader.seqs.items():
            kw[name] = list(range(p.get(sym.name, 0)))
        calls.append(kw)
    reversed_calls = []
    if reader.seqs:
        for kw in calls:
            kw2 = dict(kw)
            for name in reader.seqs:
                kw2[name] = list(reversed(kw2[name]))
            reversed_calls.append(kw2)
    per_call = min(2.0, max(0.25, budget / 6))
    total = min(budget, per_call * (len(calls) + len(reversed_calls)) + 2.0)
    runs_extra: list[_Run] = []
    try:
        res = isolated(_count_worker, spec, calls + reversed_calls, per_call, budget=max(total, 2.0))
    except (TimeBudgetError, IsolationError, EOFError, OSError) as exc:
        return [_Run(p, "timeout", None, f"{type(exc).__name__}") for p in points], None, None
    runs: list[_Run] = []
    for i, p in enumerate(points):
        j = len(points) + i
        if reversed_calls and j < len(res) and res[j][0] == "ok" and i < len(res) and res[i][0] == "ok":
            # keep the LARGER count per loop: the upper bound must cover both data orders
            merged = dict(res[i][1])
            for k, v in res[j][1].items():
                merged[k] = max(merged.get(k, 0), v)
            if merged != res[i][1]:
                runs_extra.append(_Run(p, "ok", res[j][1]))
        if i >= len(res):
            runs.append(_Run(p, "timeout", None, "the execution budget ran out"))
            continue
        st, val = res[i]
        if st == "ok":
            runs.append(_Run(p, "ok", val))
        else:
            runs.append(_Run(p, st, None, str(val) if st == "raise" else "per-call budget"))
    independent: "bool | None" = None
    if reversed_calls:
        independent = True
        for i in range(len(points)):
            j = len(points) + i
            if i < len(res) and j < len(res) and res[i][0] == "ok" and res[j][0] == "ok":
                if res[i][1] != res[j][1]:
                    independent = False
    return runs + runs_extra, independent, None


def _defaults(fdef: ast.FunctionDef) -> set[str]:
    a = fdef.args
    pos = a.posonlyargs + a.args
    names = {x.arg for x in pos[len(pos) - len(a.defaults):]} if a.defaults else set()
    names |= {x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None}
    return names


def _grid(syms: Sequence[sp.Symbol], n_max: int) -> list[dict[str, int]]:
    if not syms:
        return [{}]
    if len(syms) == 1:
        s = syms[0]
        lo = 0 if s.is_nonnegative else -3
        extra = [n_max + 3, n_max + 7, 2 * n_max + 4, 64, 100] if n_max >= 30 else []
        vals = list(range(lo, n_max + 1)) + [v for v in extra if v > n_max]
        return [{s.name: v} for v in vals]
    side = {2: 12, 3: 7}.get(len(syms), 4)
    ranges = [range(0 if s.is_nonnegative else -1, side) for s in syms]
    return [{s.name: v for s, v in zip(syms, combo)} for combo in itertools.product(*ranges)]


def _value(expr: sp.Expr, point: Mapping[str, int]) -> "int | None":
    subs = {s: point[s.name] for s in expr.free_symbols if s.name in point}
    try:
        v = expr.xreplace({k: sp.Integer(v) for k, v in subs.items()})
        if v.has(sp.Sum):
            v = v.doit()
        v = sp.nsimplify(v) if v.is_Float else v
        if v.is_Integer:
            return int(v)
        v = sp.simplify(v)
        return int(v) if v.is_Integer else None
    except Exception:  # noqa: BLE001
        return None


def _claim_value(expr: sp.Expr, point: Mapping[str, int]) -> "sp.Rational | None":
    """A claim's exact value: an integer, a non-integer rational (a real witness —
    no loop runs 1/2 times), or ``None`` where the claim is undefined / not real."""
    try:
        v = expr.xreplace({s: sp.Integer(point[s.name]) for s in expr.free_symbols if s.name in point})
        if v.has(sp.Sum):
            v = v.doit()
        if v.has(sp.nan, sp.zoo, sp.oo, -sp.oo, sp.I, floor_log, ceil_log) or not v.is_number:
            return None  # undefined there (log of 0, an unevaluated exact log): no claim
        if v.is_Rational:
            return v
        v = sp.nsimplify(sp.simplify(v)) if not v.is_Rational else v
        if v.is_Rational:
            return v
        if v.is_real is True and v.is_comparable:  # irrational: never equal to a count
            return v
        return None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# count_iterations                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class _Derived:
    reader: _Reader
    src: _Source
    entries: list[_Entry]
    total: sp.Expr  # the generic formula for the selected loops
    full: sp.Expr  # Piecewise((0, zero), (total, valid), ...)
    valid: sp.Basic
    zero: sp.Basic
    exact: bool
    unproved: list[str]
    lines: "list[int] | None"
    symbols: list[sp.Symbol]


def _derive(func_or_source: "Callable[..., Any] | str", name: "str | None", line: "int | None") -> _Derived:
    src = _load(func_or_source, name)
    reader = _Reader(src)
    block = reader.function()
    if not block.entries:
        raise LoopShapeError("no loop in the function", src.func.lineno)
    entries = block.entries
    if line is not None:
        chosen = [e for e in entries if e.line == line]
        if not chosen:
            raise DomainError(f"count_iterations: no loop at line {line} "
                              f"(loops at lines {', '.join(str(e.line) for e in entries)})")
    else:
        chosen = [e for e in entries if e.leaf]
    total = sp.Add(*[e.count for e in chosen])
    exact = all(e.exact for e in chosen)
    assume = [sp.simplify(a) for a in block.assume]
    valid = sp.And(*[sp.Ge(a, 0) for a in assume if a != 0 and not (a.is_number and a >= 0)])
    if any(a.is_number and a < 0 for a in assume):
        valid = sp.false
    # where every top-level loop has an empty range, nothing runs at all
    zeros = [b.zero for b in _top_blocks(reader)]
    zero = sp.And(*[z for z in zeros if z is not None]) if zeros and all(z is not None for z in zeros) else sp.false
    total = _tidy(total)
    full = sp.Piecewise((0, zero), (total, valid), (sp.nan, True)) if zero is not sp.false else \
        sp.Piecewise((total, valid), (sp.nan, True))
    sizes = set(reader.params.values()) | set(reader.seqs.values())
    free = set(total.free_symbols) | set(getattr(valid, "free_symbols", set())) | set(getattr(zero, "free_symbols", set()))
    symbols = sorted((s for s in free if s in sizes), key=lambda s: s.name)
    return _Derived(reader, src, entries, total, full, valid, zero, exact, block.unproved, None if line is None else [line], symbols)


def _top_blocks(reader: _Reader) -> list[_Block]:
    """The blocks of the top-level loops (re-read, cheap) — for their zero conditions."""
    out: list[_Block] = []
    env: dict[str, "sp.Expr | None"] = {}
    for st in reader.fdef.body:
        try:
            if isinstance(st, ast.For):
                out.append(reader.for_loop(st, dict(env), 0))
            elif isinstance(st, ast.While):
                out.append(reader.while_loop(st, dict(env), 0))
            elif isinstance(st, (ast.Assign, ast.AnnAssign)):
                tg = st.targets if isinstance(st, ast.Assign) else [st.target]
                if st.value is not None:
                    reader._assign(tg, st.value, env)
            elif isinstance(st, ast.If) and any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(st)):
                return []
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                env[st.target.id] = None
        except LoopShapeError:
            return []
    return out


def _tidy(e: sp.Expr) -> sp.Expr:
    if e.has(sp.Sum) or e.has(sp.Piecewise):
        return e
    try:
        f = sp.factor(sp.simplify(e))
        return f if sp.count_ops(f) <= sp.count_ops(e) + 2 else e
    except Exception:  # noqa: BLE001
        return e


def _region_text(cond: sp.Basic, syms: Sequence[sp.Symbol]) -> str:
    if cond is sp.true:
        return "True"
    if cond is sp.false:
        return "False"
    if len(syms) == 1:
        n = syms[0]
        threshold = _threshold(cond, n)
        if threshold is not None:
            return f"{n} >= {threshold}"
    return str(cond)


def _threshold(cond: sp.Basic, n: sp.Symbol, scan: int = 400) -> "int | None":
    """Smallest ``n0`` such that ``cond`` holds for every integer ``n >= n0`` —
    proved for conditions whose floors have period ``Q``: past the last sign
    change of each affine branch they hold for all larger ``n``."""
    if cond.free_symbols - {n}:
        return None
    rels = list(cond.args) if isinstance(cond, sp.And) else [cond]
    worst = -10**9
    for rel in rels:
        if not isinstance(rel, (sp.Ge, sp.Gt, sp.Le, sp.Lt)):
            return None
        expr = rel.lhs - rel.rhs if isinstance(rel, (sp.Ge, sp.Gt)) else rel.rhs - rel.lhs
        strict = isinstance(rel, (sp.Gt, sp.Lt))
        q = 1
        for a in expr.atoms(sp.floor, sp.ceiling):
            arg = sp.expand(a.args[0])
            if not arg.is_polynomial(n) or sp.Poly(arg, n).degree() > 1:
                return None
            c = sp.Poly(arg, n).coeffs()[0] if sp.Poly(arg, n).degree() == 1 else sp.Integer(1)
            if not c.is_Rational:
                return None
            q = int(sp.ilcm(q, sp.Rational(c).q))
        w = sp.Symbol("w", integer=True)
        n0 = -10**9
        for r in range(q):
            branch = sp.expand(expr.subs(n, q * w + r))
            if branch.has(sp.floor, sp.ceiling) or not branch.is_polynomial(w):
                return None
            P = sp.Poly(branch, w)
            if P.degree() > 1:
                # polynomial: holds past its largest real root when leading coeff > 0
                if P.LC() <= 0:
                    return None
                roots = [sp.re(z) for z in sp.Poly(P).nroots()] if P.degree() > 0 else []
                wmin = int(sp.floor(max(roots))) + 1 if roots else -10**6
            elif P.degree() == 1:
                a1, a0 = P.all_coeffs()
                if a1 < 0:
                    return None
                wmin = int(sp.ceiling(-a0 / a1)) + (1 if strict and (-a0 / a1).is_Integer else 0)
            else:
                ok = P.LC() > 0 if strict else P.LC() >= 0 if P.degree() == 0 else True
                if not ok:
                    return None
                wmin = -10**6
            # walk back to the first w where it fails
            wcur = wmin
            while wcur > wmin - scan:
                val = branch.subs(w, wcur - 1)
                if (val > 0) if strict else (val >= 0):
                    wcur -= 1
                else:
                    break
            n0 = max(n0, q * wcur + r)
        worst = max(worst, n0)
    if worst <= -10**8:
        return None
    # tighten: the smallest n0 with every n >= n0 satisfying (checked down to it)
    n0 = worst
    while n0 > worst - q * scan and bool(cond.subs(n, n0 - 1)):
        n0 -= 1
    return int(n0)


def count_iterations(
    func_or_source: "Callable[..., Any] | str",
    *,
    name: "str | None" = None,
    line: "int | None" = None,
    certify: bool = True,
    n_max: int = 30,
    budget: float = 30.0,
) -> IterationCount:
    """How many times do the innermost loop bodies run? Exact formula, Θ, certificate.

    ``line`` selects one loop instead (its body executions, all enclosing loops
    summed). ``certify=False`` skips the execution (no certificate). ``n_max`` is
    the largest size tried by the certificate (at least 30 is the contract; a few
    larger sizes are added when cheap).

        >>> src = '''
        ... def bsearch_steps(n: int) -> int:
        ...     i = n
        ...     c = 0
        ...     while i > 0:
        ...         i //= 2
        ...         c += 1
        ...     return c
        ... '''
        >>> r = count_iterations(src, certify=False)
        >>> r.main, r.theta
        (floor(log(n, 2)) + 1, log(n))
    """
    if n_max < 0:
        raise DomainError("count_iterations: n_max must be >= 0")
    started = time.perf_counter()
    d = _derive(func_or_source, name, line)
    syms = d.symbols
    if len(syms) == 1 and d.total.has(sp.Piecewise) and d.valid is not sp.false:
        v = 0 if d.valid is sp.true else _threshold(d.valid, syms[0])
        if v is not None:
            d.total = _tidy(_drop_dead(d.total, syms[0], v))
    proof: list[str] = ["AST reading: each loop's trip count; nested counts summed in closed form"]
    if d.unproved:
        proof.append("NOT proved (checked by counting only): " + "; ".join(d.unproved))
    else:
        proof.append(f"the formula holds where every range is non-empty: {_region_text(d.valid, syms)}")
    if not d.exact:
        proof.append("break/return/raise inside a loop: the formula is an UPPER BOUND")
    closed = not d.total.has(sp.Sum)
    theta = theta_c = None
    theta_reason = ""
    eventually_zero = len(syms) == 1 and d.zero is not sp.false and _eventually(d.zero, syms[0])
    eventually_valid = len(syms) == 1 and (d.valid is sp.true or _threshold(d.valid, syms[0]) is not None)
    if eventually_zero:
        theta, theta_c, theta_reason = sp.Integer(0), sp.Integer(0), f"the loops do not run for large {syms[0]}"
    elif len(syms) == 1 and not eventually_valid:
        theta_reason = "the formula is not proved for all large sizes"
    elif len(syms) == 1:
        if d.exact:
            theta, theta_c, theta_reason = theta_of(d.total, syms[0])
            if theta is not None:
                proof.append(f"Θ({theta}): lim count/({theta}) = {theta_c} as {syms[0]} -> oo (floors as bounded unknowns)")
        else:
            theta_reason = "an upper bound has no Θ; check_complexity proves O(...) from it"
    elif not syms:
        theta, theta_c, theta_reason = sp.Integer(1), d.total, "constant"
    else:
        theta_reason = "several size variables: no single Θ"
    cert = None
    notes: list[str] = []
    if certify:
        cert, small = _certify(d, n_max, budget)
        if cert is not None and not cert.agree:
            notes.append("THE COUNTING CERTIFICATE DISAGREES WITH THE FORMULA — do not trust the formula; "
                         "the witness is a real execution")
    variables = {s.name: (f"len({s.name[4:]})" if s.name.startswith("len_") and s.name[4:] in d.reader.seqs
                          else f"parameter {s.name}") for s in syms}
    valid_text = _region_text(d.valid, syms) if not d.unproved else f"{_region_text(d.valid, syms)} (and unproved: see proof)"
    loops = tuple(LoopCount(e.line, e.loop, _tidy(e.count), e.leaf, e.exact, e.shape) for e in d.entries)
    full = d.full
    if len(syms) == 1 and not d.unproved:
        full = _simplify_full(d, syms[0])
    del started
    return IterationCount(
        function=d.src.func.name, main=d.total, formula=full, valid_for=valid_text,
        zero_for=_zero_text(d.zero, syms), exact=d.exact, closed=closed, theta=theta,
        theta_constant=theta_c, theta_reason=theta_reason, variables=variables, loops=loops,
        proof=tuple(proof), certificate=cert, notes=tuple(notes + d.reader.notes),
        valid_condition=d.valid,
        valid_from=(0 if d.valid is sp.true else _threshold(d.valid, syms[0])) if len(syms) == 1 else None,
    )


def _zero_text(zero: sp.Basic, syms: Sequence[sp.Symbol]) -> str:
    if zero is sp.false:
        return "False"
    if len(syms) == 1:
        n = syms[0]
        try:
            s = sp.solveset(zero, n, sp.S.Reals)
            if isinstance(s, sp.Interval) and s.inf == -sp.oo:
                return f"{n} <= {_last_integer(s)}"
        except Exception:  # noqa: BLE001
            pass
    return str(zero)


def _last_integer(s: sp.Interval) -> int:
    """The largest integer in ``(-oo, sup]`` or ``(-oo, sup)``."""
    sup = s.sup
    if sup.is_Integer and s.right_open:
        return int(sup) - 1
    return int(sp.floor(sup))


def _drop_dead(expr: sp.Expr, n: sp.Symbol, v: int) -> sp.Expr:
    """Inside the region ``n >= v``: replace each ``Piecewise`` by the branch that
    holds for every ``n >= v`` (when one provably does)."""
    def pick(pw: sp.Piecewise) -> sp.Expr:
        for e, c in pw.args:
            if c is sp.true:
                return e
            try:
                sat = sp.solveset(c, n, sp.S.Reals) if c.free_symbols == {n} else None
            except Exception:  # noqa: BLE001
                return pw
            if sat is None:
                return pw
            if sp.Interval(v, sp.oo).is_subset(sat):
                return e
            if sat.intersect(sp.Interval(v, sp.oo)) is sp.S.EmptySet:
                continue
            return pw
        return pw

    return expr.replace(lambda e: isinstance(e, sp.Piecewise), pick)


def _simplify_full(d: _Derived, n: sp.Symbol) -> sp.Expr:
    """``Piecewise((0, n <= z), (formula, n >= v))`` with the region between (if any)
    left as ``nan`` — never filled in from a guess."""
    v = _threshold(d.valid, n) if d.valid is not sp.true else None
    z = None
    try:
        if d.zero is not sp.false:
            s = sp.solveset(d.zero, n, sp.S.Reals)
            if isinstance(s, sp.Interval) and s.inf == -sp.oo:
                z = _last_integer(s)
    except Exception:  # noqa: BLE001
        z = None
    if d.valid is sp.true:
        return d.total if z is None else sp.Piecewise((0, sp.Le(n, z)), (d.total, True))
    if v is None:
        return d.full
    if z is not None and z + 1 >= v:
        return sp.Piecewise((0, sp.Le(n, z)), (d.total, True))
    pieces = [(0, sp.Le(n, z))] if z is not None else []
    pieces += [(d.total, sp.Ge(n, v)), (sp.nan, True)]
    return sp.Piecewise(*pieces)


def _certify(d: _Derived, n_max: int, budget: float) -> tuple["CountCertificate | None", dict]:
    started = time.perf_counter()
    syms = d.symbols
    points = _grid(syms, n_max)
    # cap the work: skip points whose predicted count is huge
    feasible = []
    skipped: list[tuple[str, str]] = []
    for p in points:
        expected = _expected_at(d, p)
        if expected is None:
            skipped.append((_fmt(p), "outside the region the formula claims: not executed"))
            continue
        if expected > 3_000_000:
            skipped.append((_fmt(p), "predicted count above 3e6: not executed"))
            continue
        feasible.append(p)
    runs, independent, blocked = _run_points(d.src, d.reader, feasible, budget)
    if blocked is not None:
        return None, {}
    lines = d.lines if d.lines is not None else [e.line for e in d.entries if e.leaf]
    agree_n = 0
    witness = None
    rel = "==" if d.exact else "<="
    for run in runs:
        p = run.point
        if run.status != "ok" or run.counts is None:
            skipped.append((_fmt(p), f"{run.status}: {run.detail}"))
            continue
        counted = sum(run.counts.get(ln, 0) for ln in lines)
        expected = _expected_at(d, p)
        if expected is None:
            skipped.append((_fmt(p), "outside the region the formula claims"))
            continue
        ok = counted == expected if d.exact else counted <= expected
        if ok:
            agree_n += 1
        elif witness is None:
            witness = {**p, "counted": counted, "formula": expected}
    agree = witness is None and agree_n > 0 and (independent is not False or not d.exact)
    if independent is False and witness is None and d.exact:
        witness = {"data_dependent": True}
    dom = ", ".join(f"{s.name} in [{min(p[s.name] for p in points)}, {max(p[s.name] for p in points)}]"
                    for s in syms) or "no parameters"
    return CountCertificate(
        points=agree_n + (1 if witness and "counted" in witness else 0), agree=agree, relation=rel, domain=dom,
        witness=witness, skipped=tuple(skipped[:20]), data_independent=independent,
        seconds=round(time.perf_counter() - started, 3)), {}


def _expected_at(d: _Derived, p: Mapping[str, int]) -> "int | None":
    subs = {s: sp.Integer(p[s.name]) for s in d.symbols}
    try:
        if d.zero is not sp.false and bool(d.zero.xreplace(subs)):
            return 0
    except TypeError:
        pass
    try:
        ok = d.valid.xreplace(subs)
        if ok is not sp.true and not (ok == True):  # noqa: E712 — SymPy truth
            return None
    except TypeError:
        return None
    return _value(d.total, p)


def _fmt(p: Mapping[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "()"


# --------------------------------------------------------------------------- #
# claims                                                                      #
# --------------------------------------------------------------------------- #

_DOC_FIELD = re.compile(r":(iterations|complexity):\s*(.+)")


def _docstring_claim(src: _Source, field_name: str) -> "str | None":
    doc = ast.get_docstring(src.func) or ""
    for m in _DOC_FIELD.finditer(doc):
        if m.group(1) == field_name:
            return m.group(2).strip()
    return None


def _parse_claim_expr(text: str, symbols: Sequence[sp.Symbol]) -> sp.Expr:
    local = {s.name: s for s in symbols}
    local.update({"log": sp.log, "lg": lambda x: sp.log(x, 2), "ln": sp.log, "sqrt": sp.sqrt,
                  "floor": sp.floor, "ceil": sp.ceiling, "ceiling": sp.ceiling,
                  "log2": lambda x: sp.log(x, 2), "e": sp.E})
    t = text.replace("²", "**2").replace("³", "**3").replace("·", "*").replace("×", "*").replace("−", "-")
    try:
        e = parse_expr(t, local_dict=local,
                       transformations=standard_transformations + (implicit_multiplication_application, convert_xor))
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"could not read the claim {text!r}: {exc}") from exc
    return sp.sympify(e)


def _canon_logs(e: sp.Expr) -> sp.Expr:
    """``floor(log(x)/log(k))`` -> ``floor_log(x, k)`` (and ceiling): the claim in
    the same exact vocabulary as the count, so equality can be proved."""
    def fix(f: sp.Basic) -> sp.Basic:
        arg = f.args[0]
        num, den = arg.as_numer_denom()
        if isinstance(num, sp.log) and isinstance(den, sp.log) and den.args[0].is_Integer and den.args[0] > 1:
            return (floor_log if isinstance(f, sp.floor) else ceil_log)(num.args[0], den.args[0])
        return f

    return e.replace(lambda x: isinstance(x, (sp.floor, sp.ceiling)), fix)


def _bind(claim: sp.Expr, syms: Sequence[sp.Symbol]) -> sp.Expr:
    """A claim written in ``n`` for a function whose single size is ``len_a``: bind."""
    free = [s for s in claim.free_symbols if isinstance(s, sp.Symbol)]
    names = {s.name for s in syms}
    unknown = [s for s in free if s.name not in names]
    if not unknown:
        return claim.xreplace({s: next(x for x in syms if x.name == s.name) for s in free})
    if len(syms) == 1 and len(free) == 1:
        return claim.xreplace({free[0]: syms[0]})
    raise DomainError(f"the claim uses {', '.join(sorted(s.name for s in unknown))}; "
                      f"the count is in {', '.join(sorted(names)) or 'no variable'}")


def check_iterations(
    func_or_source: "Callable[..., Any] | str",
    formula: "str | sp.Expr | None" = None,
    *,
    name: "str | None" = None,
    line: "int | None" = None,
    n_max: int = 30,
    budget: float = 30.0,
    min_n: int = 0,
) -> IterationCheck:
    """Does the (innermost) loop body run exactly ``formula`` times?

    ``formula=None`` reads ``:iterations: <formula>`` from the docstring.
    REFUTED needs an EXECUTED witness: the instrumented function was run at that
    ``n`` and the counter differs from the claim (sizes ``>= min_n``, default 0:
    a claim about ``n = -1`` is not what a docstring means). VERIFIED needs a proof: the
    derived count equals the claim symbolically (or as quasi-polynomials, by
    periodic agreement), and the counting certificate agrees.

        >>> src = '''
        ... def pairs(n: int) -> int:
        ...     ":iterations: n**2"
        ...     c = 0
        ...     for i in range(n):
        ...         for j in range(i + 1, n):
        ...             c += 1
        ...     return c
        ... '''
        >>> r = check_iterations(src)            # doctest: +SKIP
        >>> r.verdict.value, r.witness            # doctest: +SKIP
        ('refuted', {'n': 1, 'counted': 0, 'claim': 1})
    """
    src = _load(func_or_source, name)
    if formula is None:
        formula = _docstring_claim(src, "iterations")
        if formula is None:
            raise DomainError("check_iterations: no formula given and no ':iterations:' in the docstring")
    res = count_iterations(func_or_source, name=name, line=line, n_max=n_max, budget=budget)
    syms = [sp.Symbol(k, integer=True, nonnegative=k.startswith("len_")) for k in res.variables]
    syms = sorted(res.main.free_symbols, key=lambda s: s.name) or syms
    claim = _parse_claim_expr(formula, syms) if isinstance(formula, str) else sp.sympify(formula)
    claim = _canon_logs(_bind(claim, syms))
    cert = res.certificate
    # 1. an executed witness against the CLAIM (independent of our formula)
    undefined: list[int] = []
    witness = _claim_witness(src, res, claim, syms, n_max, budget, min_n, undefined)
    if undefined and len(syms) == 1:
        min_n = max(min_n, max(undefined) + 1)  # the claim says nothing where it is undefined
    if witness is not None:
        return IterationCheck(VerdictStatus.REFUTED, claim, res.main, "execution",
                              f"at {_fmt({k: v for k, v in witness.items() if k not in ('counted', 'claim')})} "
                              f"the loop ran {witness['counted']} times; the claim gives {witness['claim']}",
                              witness, res)
    if cert is None:
        return IterationCheck(VerdictStatus.UNDECIDED, claim, res.main, "none",
                              "the function was not executed (blocked import or name); no witness, no certificate",
                              None, res)
    if not cert.agree:
        return IterationCheck(VerdictStatus.UNDECIDED, claim, res.main, "none",
                              "the derived formula disagrees with the counter; nothing is claimed", None, res)
    if not res.exact:
        return IterationCheck(VerdictStatus.UNDECIDED, claim, res.main, "none",
                              "the loop can exit early (break/return): only an upper bound is known; "
                              "the claim matched every executed count", None, res)
    if any(p.startswith("NOT proved") for p in res.proof):
        return IterationCheck(VerdictStatus.UNDECIDED, claim, res.main, "counting",
                              "the claim matches every executed count, but the formula's range assumptions "
                              "are not proved for all n", None, res)
    # VERIFIED: equal where the formula is proved, and every size in [min_n, v)
    # below that region was executed and matched the claim (the witness search)
    if len(syms) == 1:
        v = res.valid_from
        if v is None or v > n_max:
            return IterationCheck(VerdictStatus.UNDECIDED, claim, res.main, "counting",
                                  "the claim matches every executed count, but the formula's region "
                                  f"({res.valid_for}) leaves sizes that were not executed", None, res)
        how = _prove_equal(res.main, claim, syms, max(v, min_n))
    else:
        how = _prove_equal(res.main, claim, syms, min_n) if _orthant_inside(res.valid_condition, syms, min_n) else None
    if how is None:
        return IterationCheck(VerdictStatus.UNDECIDED, claim, res.main, "counting",
                              "the claim matches every executed count; no symbolic proof of equality found",
                              None, res)
    dom = ", ".join(f"{s.name} >= {min_n}" for s in syms)
    return IterationCheck(VerdictStatus.VERIFIED, claim, res.main, how,
                          f"count == claim for every {dom}: proved ({how}) where {res.valid_for}, "
                          f"executed below it; the counter agrees at {cert.points} points", None, res)


def _orthant_inside(cond: sp.Basic, syms: Sequence[sp.Symbol], lo: int) -> bool:
    """Every point with all ``syms >= lo`` satisfies ``cond`` (linear, coefficient-wise)."""
    if cond is sp.true:
        return True
    rels = list(cond.args) if isinstance(cond, sp.And) else [cond]
    for r in rels:
        if not isinstance(r, (sp.Ge, sp.Gt)):
            return False
        e = sp.expand((r.lhs - r.rhs).subs({s: s + lo for s in syms}))
        if not e.is_polynomial(*syms):
            return False
        P = sp.Poly(e, *syms)
        if any(c < 0 for c in P.coeffs()) or (isinstance(r, sp.Gt) and P.coeff_monomial(1) <= 0):
            return False
    return True


def _prove_equal(a: sp.Expr, b: sp.Expr, syms: Sequence[sp.Symbol], lo: int = 0) -> "str | None":
    diff = a - b
    try:
        if sp.simplify(diff) == 0:
            return "symbolic"
    except Exception:  # noqa: BLE001
        pass
    if len(syms) == 1:
        n = syms[0]
        try:
            if _qp_zero(diff, n, lo):
                return "quasi-polynomial"
        except Exception:  # noqa: BLE001
            pass
    return None


def _claim_witness(src: _Source, res: IterationCount, claim: sp.Expr, syms: Sequence[sp.Symbol],
                   n_max: int, budget: float, min_n: int = 0,
                   undefined: "list[int] | None" = None) -> "dict[str, Any] | None":
    """Run the instrumented function and compare the COUNTER with the claim; the
    first point where they differ, verified by evaluating the claim exactly."""
    reader = _Reader(src)
    points = [p for p in _grid(list(syms), n_max) if all(v >= min_n for v in p.values())]
    pts = []
    for p in points:
        v = _value(res.main, p) if not res.main.has(sp.Sum) else None
        if v is None or v <= 3_000_000:
            pts.append(p)
    runs, _, blocked = _run_points(src, reader, pts, budget)
    if blocked is not None:
        return None
    lines = [lc.line for lc in res.loops if lc.leaf]
    for run in runs:
        if run.status != "ok" or run.counts is None:
            continue
        counted = sum(run.counts.get(ln, 0) for ln in lines)
        want = _claim_value(claim, run.point)
        if want is None:
            if undefined is not None and len(run.point) == 1:
                undefined.append(next(iter(run.point.values())))
            continue
        if (want - counted) != 0 and sp.simplify(want - counted) != 0:
            return {**run.point, "counted": counted, "claim": str(want) if not want.is_Integer else int(want)}
    return None


_CLAIM = re.compile(r"^\s*(O|Θ|Theta|θ|Ω|Omega|o|ω|omega|big-?O)\s*\((.*)\)\s*$", re.S)


def parse_complexity_claim(text: str) -> tuple[str, str]:
    """``"O(n log n)"`` -> ``("O", "n log n")``; Θ/Theta, Ω/Omega, o, ω/omega."""
    m = _CLAIM.match(text)
    if not m:
        raise ParseError(f"could not read the complexity claim {text!r} (expected O(...), Θ(...) or Ω(...))")
    kind = {"Θ": "Theta", "θ": "Theta", "Theta": "Theta", "Ω": "Omega", "Omega": "Omega",
            "o": "o", "ω": "omega", "omega": "omega"}.get(m.group(1), "O")
    return kind, m.group(2)


def check_complexity(
    func_or_source: "Callable[..., Any] | str",
    claim: "str | None" = None,
    *,
    name: "str | None" = None,
    line: "int | None" = None,
    n_max: int = 30,
    budget: float = 30.0,
) -> ComplexityCheck:
    """Is the iteration count O / Θ / Ω / o / ω of the claim? Proved by a limit.

    ``claim=None`` reads ``:complexity: O(...)`` from the docstring. The decision
    is ``L = lim count/g`` (n -> oo, floors as bounded unknowns): O needs ``L < oo``,
    Ω needs ``L > 0``, Θ both, o ``L = 0``, ω ``L = oo``. An UPPER-BOUND count
    (break inside) can prove O and o, never refute them, and decides nothing else.

        >>> src = '''
        ... def bubble(a: list) -> None:
        ...     for i in range(len(a)):
        ...         for j in range(len(a) - i - 1):
        ...             if a[j] > a[j + 1]:
        ...                 a[j], a[j + 1] = a[j + 1], a[j]
        ... '''
        >>> check_complexity(src, "O(n log n)").verdict.value    # doctest: +SKIP
        'refuted'
    """
    src = _load(func_or_source, name)
    if claim is None:
        claim = _docstring_claim(src, "complexity")
        if claim is None:
            raise DomainError("check_complexity: no claim given and no ':complexity:' in the docstring")
    bound, inner = parse_complexity_claim(claim)
    res = count_iterations(func_or_source, name=name, line=line, n_max=n_max, budget=budget)
    syms = sorted(res.main.free_symbols, key=lambda s: s.name)

    def out(v: VerdictStatus, g: sp.Expr, L: "sp.Expr | None", method: str, reason: str,
            samples: tuple = ()) -> ComplexityCheck:
        return ComplexityCheck(v, claim or "", bound, g, L, res.main, res.theta, method, reason, samples, res)

    if len(syms) > 1:
        return out(VerdictStatus.UNDECIDED, sp.Integer(1), None, "none",
                   "several size variables: an asymptotic claim needs one")
    if not syms:
        n = sp.Symbol("n", integer=True)
        g = _parse_claim_expr(inner, [n])
        L = _limit(res.main / g, n) if g.has(n) else res.main / g
    else:
        n = syms[0]
        g = _bind(_parse_claim_expr(inner, [n]), [n])
        L = None
    if res.certificate is not None and not res.certificate.agree:
        return out(VerdictStatus.UNDECIDED, g, None, "none", "the derived count disagrees with the counter")
    if res.certificate is None:
        return out(VerdictStatus.UNDECIDED, g, None, "none", "the function was not executed: no certificate")
    if any(p.startswith("NOT proved") for p in res.proof):
        return out(VerdictStatus.UNDECIDED, g, None, "none",
                   "the count's range assumptions are not proved for all n")
    if syms:
        try:
            branch = _eventual_branch(res.main, n)
        except ValueError:
            return out(VerdictStatus.UNDECIDED, g, None, "none", "no formula branch holds for all large n")
        if branch.has(sp.Sum):
            return out(VerdictStatus.UNDECIDED, g, None, "none", "the count is a Sum without a closed form")
        smooth, thetas = _smooth(branch, n)
        L = _limit_over_thetas(smooth / g, n, thetas)
    if L is None or L.has(sp.AccumBounds) or L is sp.nan:
        return out(VerdictStatus.UNDECIDED, g, L, "none", "the limit of count/g could not be determined")
    finite, positive = (L.is_finite is True), (L == sp.oo or L.is_positive is True)
    samples = _samples(res, n, g) if syms else ()
    if not res.exact:
        if bound in ("O", "o") and (finite if bound == "O" else L == 0):
            return out(VerdictStatus.VERIFIED, g, L, "limit (upper bound)",
                       f"count <= {res.main} and lim ({res.main})/({g}) = {L}", samples)
        return out(VerdictStatus.UNDECIDED, g, L, "none",
                   "only an upper bound on the count is known (break/return inside a loop)", samples)
    ok = {"O": finite, "Omega": positive, "Theta": finite and positive and L != sp.oo,
          "o": L == 0, "omega": L == sp.oo}[bound]
    verdict = VerdictStatus.VERIFIED if ok else VerdictStatus.REFUTED
    why = (f"lim count/({g}) = {L} as {n} -> oo, so count is "
           + ("" if ok else "NOT ") + f"{bound}({g})" + (f"; count is Θ({res.theta})" if res.theta is not None else ""))
    return out(verdict, g, L, "limit", why, samples)


def _samples(res: IterationCount, n: sp.Symbol, g: sp.Expr) -> tuple[tuple[int, int, str], ...]:
    out = []
    for k in (4, 16, 64, 256, 1024):
        v = _value(res.main, {n.name: k})
        gv = g.subs(n, k)
        if v is None:
            continue
        try:
            ratio = sp.N(v / gv, 6)
        except Exception:  # noqa: BLE001
            continue
        out.append((k, v, str(ratio)))
    return tuple(out)
