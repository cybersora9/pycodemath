"""The lint engine: read a file, find the mathematics, ask C1-C4, keep only what is witnessed.

``lint_source(text, path, config)`` returns the findings of one file;
``lint_paths(paths, config)`` walks directories, honours excludes and the cache,
and returns a ``LintResult`` with timings. Nothing here decides a mathematical
question by itself: each rule gathers a candidate (an expression, a function, a
comparison) and hands it to the module that owns the question —

========  =========================================================================
PCM1xx    C2 ``stability.diagnose`` (+ ``improve`` for a proved rewrite)
PCM2xx    C3 ``ranges.analyze`` (+ its proved advice)
PCM3xx    C1 ``frontend.check_code`` (PCM504: + the range mutation C1 proves)
PCM4xx    C4 ``money.float_money_hazard`` / ``round_money`` / ``money_grid``
PCM501    C4 ``float_money_hazard`` (a comparison's True/False is an "amount" too)
PCM502-3  executed directly (XOR vs power; Python round vs C4 HALF_UP)
========  =========================================================================

— and then REPLAYS the witness that module returned (``witness.Witness.replay``:
execution with the code's own operations, exact value by SymPy). A candidate
whose witness does not replay is dropped. A finding is the conjunction of two
independent routes, which is what "zero alarms without a witness" means in code.

WHICH FUNCTIONS ARE "NUMERIC" (for C2/C3, the expensive analyses). A function is
analysed when its body, after the docstring, is straight-line assignments to
local names followed by ``return <expr>``, the locals inline into one expression,
and that expression uses only parameters, numeric literals, module-level numeric
constants, ``+ - * / **`` and functions from ``math`` / ``numpy`` that C2/C3 model.
Loops, branches, attribute access, subscripts and calls to other functions take a
function out (PCM902, off by default, lists them). Each analysis runs under the
per-function ``budget``; running out is PCM901, never a finding.

    >>> fs = lint_source("def f(x: float) -> float:\\n    return x ^ 2 + 1\\n", "f.py")
    >>> [(f.code, f.line, f.witness.describe()) for f in fs]
    [('PCM502', 2, 'x ^ 2 at x=3 = 1, but x ** 2 = 9')]
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import math
import os
import re
import time
import tokenize
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from ..core.errors import PycodemathError, TimeBudgetError
from .config import Config
from .rules import RULES
from .witness import Fix, Witness, WitnessError

__all__ = ["Finding", "LintResult", "FileReport", "lint_source", "lint_paths", "iter_python_files",
           "spec_functions", "check_spec", "SpecTarget", "ENGINE_VERSION"]

#: Bumped whenever results may change for the same input (invalidates the cache).
ENGINE_VERSION = "1"


# --------------------------------------------------------------------------- #
# results                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Finding:
    """One finding: where, which rule, what — and the witness that proves it.

    ``line``/``col`` are 1-based (as ruff prints them), ``end_*`` exclusive end.
    ``witness`` is ``None`` only for the ``note`` rules that report the ABSENCE of
    a verdict (PCM302, PCM9xx). ``function`` is the enclosing function, if any.
    """

    code: str
    message: str
    path: str
    line: int
    col: int
    end_line: int
    end_col: int
    witness: "Witness | None" = None
    fix: "Fix | None" = None
    function: "str | None" = None
    suppress_lines: tuple[int, ...] = ()

    @property
    def rule(self) -> Any:
        return RULES[self.code]

    @property
    def name(self) -> str:
        return RULES[self.code].name

    @property
    def level(self) -> str:
        return RULES[self.code].level

    def to_dict(self) -> dict[str, Any]:
        from .rules import rule_url

        return {
            "code": self.code, "name": self.name, "level": self.level, "message": self.message,
            "filename": self.path, "location": {"row": self.line, "column": self.col},
            "end_location": {"row": self.end_line, "column": self.end_col},
            "function": self.function, "url": rule_url(self.code),
            "witness": None if self.witness is None else self.witness.to_dict(),
            "fix": None if self.fix is None else self.fix.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], path: str) -> "Finding":
        fx = d.get("fix")
        return cls(
            code=d["code"], message=d["message"], path=path, line=d["location"]["row"],
            col=d["location"]["column"], end_line=d["end_location"]["row"],
            end_col=d["end_location"]["column"], function=d.get("function"),
            witness=None if d.get("witness") is None else Witness.from_dict(d["witness"]),
            fix=None if fx is None else Fix(fx["description"], fx["replacement"], fx["location"]["row"],
                                             fx["location"]["column"], fx["end_location"]["row"],
                                             fx["end_location"]["column"], fx.get("proof")),
            suppress_lines=tuple(d.get("suppress_lines", ())),
        )


@dataclass(frozen=True)
class FileReport:
    """Findings of one file, before selection/noqa filtering (``raw``) and after."""

    path: str
    findings: tuple[Finding, ...]
    suppressed: int
    seconds: float
    cached: bool
    error: "str | None" = None
    functions: int = 0
    analysed: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LintResult:
    """Everything ``lint_paths`` saw."""

    files: tuple[FileReport, ...]
    seconds: float

    @property
    def findings(self) -> list[Finding]:
        return [f for r in self.files for f in r.findings]

    @property
    def errors(self) -> list[tuple[str, str]]:
        return [(r.path, r.error) for r in self.files if r.error]

    def statistics(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.code] = counts.get(f.code, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


# --------------------------------------------------------------------------- #
# suppression comments                                                        #
# --------------------------------------------------------------------------- #

_IGNORE = re.compile(r"#\s*pcm:\s*ignore(?:\[([^\]]*)\])?", re.I)
_NOQA = re.compile(r"#\s*noqa(?::\s*([A-Z]+[0-9]+(?:[\s,]+[A-Z]+[0-9]+)*))?", re.I)
_FILE_IGNORE = re.compile(r"#\s*pcm:\s*file-ignore\[([^\]]*)\]", re.I)
_CODE = re.compile(r"PCM\d{0,3}|ALL", re.I)


def _suppressions(source: str) -> tuple[dict[int, set[str]], set[str]]:
    """line -> codes silenced on it (``"*"`` = all), and the file-wide codes.

    ``# pcm: ignore[PCM101, PCM5]`` silences those codes (prefixes allowed) on its
    line; ``# pcm: ignore`` silences every rule there; ``# noqa: PCM101`` is read the
    same way (a bare ``# noqa`` is ruff's/flake8's and is NOT taken as ours — it would
    silence math findings no one meant to silence). ``# pcm: file-ignore[...]``
    anywhere in the file applies to the whole file.
    """
    lines: dict[int, set[str]] = {}
    file_codes: set[str] = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return lines, file_codes
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        text = tok.string
        m = _FILE_IGNORE.search(text)
        if m:
            file_codes |= {c.upper() for c in _CODE.findall(m.group(1))}
            continue
        here: set[str] = set()
        m = _IGNORE.search(text)
        if m:
            here |= {c.upper() for c in _CODE.findall(m.group(1))} if m.group(1) is not None else {"*"}
        m = _NOQA.search(text)
        if m and m.group(1):
            here |= {c.upper() for c in _CODE.findall(m.group(1)) if c.upper().startswith("PCM")}
        if here:
            lines.setdefault(tok.start[0], set()).update(here)
    return lines, file_codes


def _silenced(code: str, codes: set[str]) -> bool:
    return "*" in codes or any(code.startswith(c) or c == "ALL" for c in codes)


# --------------------------------------------------------------------------- #
# reading the file                                                            #
# --------------------------------------------------------------------------- #

_FLOAT_TYPES = {"float", "builtins.float"}
_INT_TYPES = {"int", "builtins.int"}
_NP_DTYPES = {"float16", "float32", "float64", "int8", "int16", "int32", "int64",
              "uint8", "uint16", "uint32", "uint64", "bfloat16", "half", "single", "double"}
_NP_ALIASES = {"half": "float16", "single": "float32", "double": "float64"}
_MODULE_NAMES = {"np", "numpy", "math"}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else ""
    return ""


def _ann_kind(node: "ast.expr | None") -> "str | None":
    """``float`` / ``int`` / ``decimal`` / ``np:<dtype>`` / None for an annotation."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _ann_kind(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return None
    if isinstance(node, ast.Subscript):  # NDArray[np.float32], npt.NDArray[np.int32]
        if _dotted(node.value).split(".")[-1] in ("NDArray", "ndarray"):
            inner = node.slice
            if isinstance(inner, ast.Tuple) and inner.elts:
                inner = inner.elts[-1]
            k = _ann_kind(inner)
            return k if k and k.startswith("np:") else None
        return None
    d = _dotted(node)
    if d in _FLOAT_TYPES:
        return "float"
    if d in _INT_TYPES:
        return "int"
    if d.split(".")[-1] == "Decimal":
        return "decimal"
    parts = d.split(".")
    if len(parts) == 2 and parts[0] in ("np", "numpy") and parts[1] in _NP_DTYPES:
        return "np:" + _NP_ALIASES.get(parts[1], parts[1])
    return None


@dataclass
class _Spec:
    formula: "str | None" = None
    domain: dict[str, str] = field(default_factory=dict)
    ranges: dict[str, tuple] = field(default_factory=dict)
    line: int = 0


#: ``:math: <formula>`` alone on a docstring line. NOT Sphinx's inline role ``:math:`...```:
#: scientific docstrings are full of it (13 functions in the field run: LaTeX such as
#: ``\frac{d^2 P}{dV dV_d}``), and a fragment of documentation is not a specification.
_MATH_LINE = re.compile(r"^\s*:math:[ \t]+([^`\n]+?)\s*$", re.M)
_DOMAIN_LINE = re.compile(r"^\s*:domain\s+([A-Za-z_]\w*)\s*:\s*(.+?)\s*$", re.M)
_RANGE_LINE = re.compile(r"^\s*:range\s+([A-Za-z_]\w*)\s*:\s*\[([^\]]*)\]\s*$", re.M)


def _parse_range_text(text: str) -> "tuple | None":
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in (2, 3):
        return None
    try:
        lo, hi = float(ast.literal_eval(parts[0])), float(ast.literal_eval(parts[1]))
    except (ValueError, SyntaxError, TypeError):
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        return None
    if len(parts) == 3:
        return (lo, hi, "int") if parts[2].strip("'\" ") == "int" else None
    return (lo, hi)


_FORMULA_WORDS = {"pi", "E", "e", "I", "oo", "sqrt", "exp", "log", "ln", "sin", "cos", "tan", "asin",
                  "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "abs",
                  "floor", "ceiling", "ceil", "factorial", "binomial", "gamma", "Sum", "Product",
                  "Abs", "Min", "Max", "min", "max", "log1p", "expm1", "hypot", "cbrt", "sign", "Mod"}


def _is_formula_of(text: str, fdef: ast.FunctionDef) -> bool:
    """A spec is a formula in the function's PARAMETERS: it parses as Python (``^``
    allowed) and every other name is a known function or constant."""
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError:
        return False
    params = {a.arg for a in _params(fdef)}
    bound = {n.id for c in ast.walk(tree) if isinstance(c, ast.Call)
             for n in ast.walk(c) if isinstance(n, ast.Name) and isinstance(c.func, ast.Name)
             and c.func.id in ("Sum", "Product") and n is not c.func}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return bool(names & params) and names <= params | _FORMULA_WORDS | bound


def _spec_of(fdef: ast.FunctionDef) -> _Spec:
    """The function's spec: ``:math:`` / ``:domain`` / ``:range`` docstring fields and
    ``@spec("...", domain={...}, ranges={...})`` (the decorator wins on a conflict)."""
    s = _Spec()
    doc = ast.get_docstring(fdef, clean=True) or ""
    m = _MATH_LINE.search(doc)
    if m and _is_formula_of(m.group(1).strip(), fdef):
        s.formula = m.group(1).strip()
        s.line = fdef.body[0].lineno if fdef.body else fdef.lineno
    for name, text in _DOMAIN_LINE.findall(doc):
        s.domain[name] = text
    for name, text in _RANGE_LINE.findall(doc):
        r = _parse_range_text(text)
        if r is not None:
            s.ranges[name] = r
    for dec in fdef.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        d = _dotted(dec.func)
        if d.split(".")[-1] != "spec" or (len(d.split(".")) > 1 and d.split(".")[0] not in
                                          ("pcm", "lint", "pycodemath")):
            continue
        if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
            s.formula = dec.args[0].value
            s.line = dec.lineno
        for kw in dec.keywords:
            try:
                val = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                continue
            if kw.arg == "domain" and isinstance(val, dict):
                s.domain.update({str(k): str(v) for k, v in val.items()})
            elif kw.arg == "ranges" and isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, (list, tuple)) and len(v) in (2, 3):
                        r = _parse_range_text(", ".join(repr(x) if not isinstance(x, str) else x for x in v))
                        if r is not None:
                            s.ranges[str(k)] = r
    return s


def _params(fdef: ast.FunctionDef) -> list[ast.arg]:
    a = fdef.args
    return [*a.posonlyargs, *a.args, *a.kwonlyargs]


def _defaults(fdef: ast.FunctionDef) -> dict[str, ast.expr]:
    a = fdef.args
    pos = [*a.posonlyargs, *a.args]
    out = {p.arg: d for p, d in zip(pos[len(pos) - len(a.defaults):], a.defaults)}
    out.update({p.arg: d for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None})
    return out


def _literal_kind(node: ast.expr) -> "str | None":
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _literal_kind(node.operand)
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        if isinstance(node.value, float):
            return "float"
        if isinstance(node.value, int):
            return "int"
    return None


def _expr_kind(node: ast.expr, env: Mapping[str, "str | None"]) -> "str | None":
    """``float`` / ``int`` / None (unknown) for a Python expression."""
    k = _literal_kind(node)
    if k:
        return k
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp):
        return _expr_kind(node.operand, env)
    if isinstance(node, ast.BinOp):
        a, b = _expr_kind(node.left, env), _expr_kind(node.right, env)
        if isinstance(node.op, ast.Div):
            return "float" if a in ("float", "int") and b in ("float", "int") else None
        if "float" in (a, b) and None not in (a, b):
            return "float"
        if a == b == "int" and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            return "int"
        return None
    if isinstance(node, ast.Call):
        d = _dotted(node.func)
        if d == "float":
            return "float"
        if d in ("int", "len", "math.floor", "math.ceil", "math.trunc", "math.factorial", "math.comb"):
            return "int"
        if d == "round":
            return "int" if len(node.args) == 1 else _expr_kind(node.args[0], env) if node.args else None
        if d.startswith("math.") and d.split(".")[1] in _MATH_FLOAT_FUNCS:
            return "float"
    return None


_MATH_FLOAT_FUNCS = {"sqrt", "exp", "log", "log10", "log2", "log1p", "expm1", "sin", "cos", "tan",
                     "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh",
                     "atanh", "hypot", "pow", "fabs", "cbrt", "erf", "erfc", "gamma"}


@dataclass
class _Func:
    """What one function tells us about its names."""

    node: ast.FunctionDef
    kinds: dict[str, "str | None"]          # param/local -> float | int | decimal | np:<dt> | None
    params: list[str]
    assigns: dict[str, ast.expr]            # locals assigned exactly once (inlinable)
    spec: _Spec
    bitwise: bool                           # uses & | << >> ~ (bit manipulation)


def _single_assigns(fdef: ast.FunctionDef) -> dict[str, ast.expr]:
    counts: dict[str, int] = {}
    values: dict[str, ast.expr] = {}
    for node in ast.walk(fdef):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            if len(targets) == 1 and isinstance(targets[0], ast.Name):
                values[targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            if isinstance(node.target, ast.Name):
                values[node.target.id] = node.value
        elif isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor, ast.With, ast.NamedExpr)):
            if isinstance(node, ast.With):
                targets = [i.optional_vars for i in node.items if i.optional_vars is not None]
            else:
                targets = [node.target]
            for tt in targets:
                for n in ast.walk(tt):
                    if isinstance(n, ast.Name):
                        counts[n.id] = counts.get(n.id, 0) + 2  # never inline
            continue
        elif isinstance(node, (ast.comprehension,)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    counts[n.id] = counts.get(n.id, 0) + 2
            continue
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    counts[n.id] = counts.get(n.id, 0) + 1
    params = {a.arg for a in _params(fdef)}
    return {k: v for k, v in values.items() if counts.get(k) == 1 and k not in params}


_BITWISE = (ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift)


def _uses_bitwise(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, _BITWISE):
            return True
        if isinstance(n, ast.AugAssign) and isinstance(n.op, (*_BITWISE, ast.BitXor)):
            return True
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Invert):
            return True
    return False


def _func_info(fdef: ast.FunctionDef, module_kinds: Mapping[str, "str | None"]) -> _Func:
    kinds: dict[str, "str | None"] = dict(module_kinds)
    params = [a.arg for a in _params(fdef)]
    defaults = _defaults(fdef)
    for a in _params(fdef):
        k = _ann_kind(a.annotation)
        if k is None and a.arg in defaults:
            k = _literal_kind(defaults[a.arg])
        kinds[a.arg] = k
    assigns = _single_assigns(fdef)
    for _ in range(3):  # locals in dependency order, a few passes suffice
        for name, value in assigns.items():
            kinds[name] = _expr_kind(value, kinds)
    for node in ast.walk(fdef):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            k = _ann_kind(node.annotation)
            if k:
                kinds[node.target.id] = k
    return _Func(fdef, kinds, params, assigns, _spec_of(fdef), _uses_bitwise(fdef))


class _TooBig(Exception):
    pass


class _Inliner(ast.NodeTransformer):
    """Replace single-assignment locals by their values — at most ``_MAX_INLINE_NODES``
    nodes in total and never a name inside its own definition (``a = b; b = a``)."""

    def __init__(self, assigns: Mapping[str, ast.expr]) -> None:
        self.assigns = assigns
        self.active: list[str] = []
        self.size = 0

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.assigns and node.id not in self.active \
                and len(self.active) < 8:
            value = self.assigns[node.id]
            self.size += sum(1 for _ in ast.walk(value))
            if self.size > _MAX_INLINE_NODES:
                raise _TooBig
            self.active.append(node.id)
            try:
                return self.visit(copy.deepcopy(value))
            finally:
                self.active.pop()
        return node


_MAX_INLINE_NODES = 400


def _inline(expr: ast.expr, assigns: Mapping[str, ast.expr]) -> ast.expr:
    """``expr`` with the locals inlined; unchanged when that would exceed the size bound
    (its names then stay free, and every rule treats a free local as unknown)."""
    if not assigns:
        return expr
    try:
        out = _Inliner(assigns).visit(copy.deepcopy(expr))
    except (_TooBig, RecursionError):
        return expr
    assert isinstance(out, ast.expr)
    return ast.fix_missing_locations(out)


def _free_names(expr: ast.AST) -> list[str]:
    seen: list[str] = []
    for n in ast.walk(expr):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in seen:
            seen.append(n.id)
    callee = {id(n.func) for n in ast.walk(expr) if isinstance(n, ast.Call)}
    mods = {n.value.id for n in ast.walk(expr) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    funcs = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name) and id(n) in callee}
    return [s for s in seen if s not in funcs and s not in mods]


def _module_constants(tree: ast.Module) -> tuple[dict[str, ast.expr], dict[str, "str | None"]]:
    """Top-level ``NAME = <numeric literal>`` assigned once: inlined like literals."""
    counts: dict[str, int] = {}
    values: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        counts[n.id] = counts.get(n.id, 0) + 1
            value = getattr(node, "value", None)
            if (isinstance(node, (ast.Assign, ast.AnnAssign)) and len(targets) == 1
                    and isinstance(targets[0], ast.Name) and value is not None and _literal_kind(value)):
                values[targets[0].id] = value
    consts = {k: v for k, v in values.items() if counts.get(k) == 1}
    return consts, {k: _literal_kind(v) for k, v in consts.items()}


# --------------------------------------------------------------------------- #
# per-file context                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class _Ctx:
    source: str
    path: str
    tree: ast.Module
    config: Config
    enabled: frozenset[str]
    money: "re.Pattern[str]"
    consts: dict[str, ast.expr]
    const_kinds: dict[str, "str | None"]
    parents: dict[int, ast.AST]
    funcs: dict[int, _Func]          # id(FunctionDef) -> info
    owner: dict[int, ast.FunctionDef]  # id(node) -> enclosing function
    findings: list[Finding] = field(default_factory=list)
    analysed: dict[str, int] = field(default_factory=dict)
    module_bitwise: bool = False

    def on(self, *codes: str) -> bool:
        return any(c in self.enabled for c in codes)

    def count(self, what: str) -> None:
        self.analysed[what] = self.analysed.get(what, 0) + 1

    def add(self, code: str, node: ast.AST, message: str, witness: "Witness | None",
            fix: "Fix | None" = None, func: "ast.FunctionDef | None" = None,
            replayed: bool = False, module_source: "str | None" = None) -> None:
        """Record a finding — after replaying its witness (the second, independent route)."""
        if code not in self.enabled:
            return
        if witness is not None and not replayed:
            rep = _replay(self, witness, module_source)
            if rep is None:
                return
            if not rep.differs:
                self.count("dropped:replay-agrees")
                return
        line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", None) or line
        col = self.column(line, getattr(node, "col_offset", 0))
        end_col = self.column(end_line, getattr(node, "end_col_offset", None) or 0)
        sup = [line]
        if func is not None:
            sup.append(func.lineno)
            if func.decorator_list:
                sup.append(func.decorator_list[0].lineno)
        self.findings.append(Finding(code, message, self.path, line, col, end_line, end_col,
                                     witness, fix, func.name if func else None, tuple(sup)))

    def fix(self, description: str, replacement: str, node: ast.AST, proof: "str | None") -> Fix:
        """A ``Fix`` replacing ``node``'s source span (character columns)."""
        line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", None) or line
        return Fix(description, replacement, line, self.column(line, getattr(node, "col_offset", 0)),
                   end_line, self.column(end_line, getattr(node, "end_col_offset", None) or 0), proof)

    def column(self, line: int, byte_offset: int) -> int:
        """1-based character column of an AST (UTF-8 byte) offset."""
        if not hasattr(self, "_lines"):
            self._lines = self.source.splitlines()
        if not 1 <= line <= len(self._lines):
            return byte_offset + 1
        return len(self._lines[line - 1].encode("utf-8")[:byte_offset].decode("utf-8", "ignore")) + 1

    def money_name(self, name: str, where: "ast.AST | None" = None) -> bool:
        """Does ``name`` denote an amount? A part matching ``money-names`` (``unit_price``,
        ``totalCost``, ``taxes``) — except the generic word ``total`` (``total_min_x`` in a
        GUI is pixels; measured on arcade): ``total`` alone counts only inside a function
        whose own name is about money (``invoice_total``, ``cart_sum``, ``price_of``)."""
        parts = [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)]

        def hit(p: str) -> bool:
            return bool(self.money.fullmatch(p) or self.money.fullmatch(p.rstrip("s"))
                        or (p.endswith("es") and self.money.fullmatch(p[:-2])))

        strong = [p for p in parts if hit(p) and p.rstrip("s") not in _WEAK_MONEY]
        if strong:
            return True
        if not any(hit(p) for p in parts):
            return False
        fn = self.owner.get(id(where)) if where is not None else None
        if fn is None:
            return False
        fparts = [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", fn.name)]
        return any((hit(p) and p.rstrip("s") not in _WEAK_MONEY) or p in _MONEY_CONTEXT for p in fparts)


#: Money words too generic to mean an amount on their own.
_WEAK_MONEY = {"total", "subtotal"}
#: Function-name words that make a generic ``total`` an amount.
_MONEY_CONTEXT = {"invoice", "bill", "receipt", "order", "cart", "checkout", "payment", "pay", "money",
                  "price", "cost", "vat", "tax", "amount", "fee", "salary", "wage", "loan", "currency"}

#: Seconds one witness replay may take (SymPy exact arithmetic on the inputs).
REPLAY_BUDGET = 5.0


def _replay(ctx: "_Ctx", w: Witness, module_source: "str | None" = None) -> Any:
    """``w.replay()`` under a time budget; ``None`` (finding dropped) when it cannot run."""
    from ..core.budget import time_budget

    try:
        with time_budget(REPLAY_BUDGET, "lint-replay", w.program):
            return w.replay(module_source)
    except TimeBudgetError:
        ctx.count("dropped:replay-budget")
    except (WitnessError, ArithmeticError, ValueError, TypeError, OverflowError, RecursionError):
        ctx.count("dropped:unreplayable")
    return None


def _build_ctx(source: str, path: str, config: Config, tree: ast.Module) -> _Ctx:
    consts, const_kinds = _module_constants(tree)
    parents: dict[int, ast.AST] = {}
    owner: dict[int, ast.FunctionDef] = {}
    funcs: dict[int, _Func] = {}

    def walk(node: ast.AST, fn: "ast.FunctionDef | None") -> None:
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
            if fn is not None:
                owner[id(child)] = fn
            if isinstance(child, ast.FunctionDef):
                funcs[id(child)] = _func_info(child, const_kinds)
                walk(child, child)
            else:
                walk(child, fn)

    walk(tree, None)
    rel = os.path.relpath(path, config.root) if os.path.isabs(path) else path
    return _Ctx(source, path, tree, config, config.enabled_codes(rel),
                re.compile(f"(?:{config.money_names})", re.I), consts, const_kinds, parents, funcs,
                owner, module_bitwise=_uses_bitwise(tree))


def _kinds_at(ctx: _Ctx, node: ast.AST) -> tuple[dict[str, "str | None"], dict[str, ast.expr], "ast.FunctionDef | None"]:
    fn = ctx.owner.get(id(node))
    if fn is None:
        return dict(ctx.const_kinds), {}, None
    info = ctx.funcs[id(fn)]
    return info.kinds, info.assigns, fn


def _inline_at(ctx: _Ctx, node: ast.expr) -> ast.expr:
    _, assigns, _ = _kinds_at(ctx, node)
    return _inline(_inline(node, assigns), ctx.consts)


# --------------------------------------------------------------------------- #
# PCM502 — ^ as a power                                                       #
# --------------------------------------------------------------------------- #

_ARITH = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod)


def _int_literal(node: ast.AST) -> "int | None":
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _int_literal(node.operand)
        return None if v is None else -v
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _numeric_context(ctx: _Ctx, node: ast.expr) -> bool:
    parent = ctx.parents.get(id(node))
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, _ARITH):
        return True
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub):
        return True
    if isinstance(parent, ast.Call):
        d = _dotted(parent.func)
        if d.split(".")[0] in ("math", "np", "numpy") or d in ("sqrt", "abs", "round", "pow", "exp", "log"):
            return True
    if isinstance(node, ast.BinOp) and _int_literal(node.left) is not None and _int_literal(node.right) is not None:
        return True  # 10^-6, 2^31: a constant written as a power
    fn = ctx.owner.get(id(node))
    return fn is not None and ctx.funcs[id(fn)].spec.formula is not None


def _check_xor(ctx: _Ctx) -> None:
    """``^`` written as a power. The unit is the whole arithmetic expression around
    it: ``a^2 + b^2`` parses as ``a ^ (2 + b) ^ 2``, and what the author meant is the
    same text with every ``^`` read as ``**`` — that is the intended program."""
    seen: set[int] = set()
    for node in ast.walk(ctx.tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor)):
            continue
        top: ast.expr = node
        while isinstance(ctx.parents.get(id(top)), (ast.BinOp, ast.UnaryOp)):
            top = ctx.parents[id(top)]  # type: ignore[assignment]
        if id(top) in seen:
            continue
        seen.add(id(top))
        fn = ctx.owner.get(id(top))
        if (ctx.funcs[id(fn)].bitwise if fn is not None else ctx.module_bitwise):
            continue
        xors = [n for n in ast.walk(top) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitXor)]
        arith_inside = any(isinstance(n, ast.BinOp) and isinstance(n.op, _ARITH) for n in ast.walk(top))
        if not (arith_inside or _numeric_context(ctx, top) or any(_numeric_context(ctx, x) for x in xors)):
            continue
        if any(isinstance(n, (ast.Call, ast.Attribute, ast.Subscript)) for n in ast.walk(top)):
            continue
        segment = ast.get_source_segment(ctx.source, top)
        if not segment or "^" not in segment:
            continue
        try:
            intended_tree = ast.parse(segment.replace("^", "**").strip(), mode="eval").body
        except SyntaxError:
            continue
        if re.search(r"\b0[xXbBoO]", segment):
            continue  # a hex / binary / octal operand is bit manipulation
        pows = [n for n in ast.walk(intended_tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow)]
        if not any(2 <= (_int_literal(p.right) or 0) <= 16 or _int_literal(p.left) in (2, 10) for p in pows):
            continue  # x ^ y between names, x ^ 255: as likely a real XOR as a mistake
        if any(abs(_int_literal(p.right) or 0) > 1024 for p in pows):
            continue  # 10^1000000000: the intended value is not computable in a lint run
        kinds, _, fn = _kinds_at(ctx, top)
        names = _free_names(top)
        if any(kinds.get(n) not in (None, "int", "float") for n in names):
            continue
        inputs = {n: ("3.0" if kinds.get(n) == "float" else "3") for n in names}
        program, intended = ast.unparse(top), ast.unparse(intended_tree)
        w = Witness("xor", program, inputs, observed="", expected="", compare="value",
                    expected_program=intended, reading="binary")
        try:
            rep = w.replay()
        except (WitnessError, ArithmeticError, ValueError, OverflowError):
            continue
        if not rep.differs:
            continue
        w = replace(w, observed=rep.observed, expected=rep.expected)
        fix = ctx.fix("read `^` as the power `**` it stands for", segment.replace("^", "**"), top,
                      "the power the arithmetic context uses")
        ctx.add("PCM502", top, f"`^` is bitwise XOR in Python, not a power: {w.describe()}", w, fix, fn,
                replayed=True)


# --------------------------------------------------------------------------- #
# PCM501 — == / != on floats  (C4 decides, the witness replays)               #
# --------------------------------------------------------------------------- #

_FLOAT_POOL = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.1", "1.2", "1.3",
               "2.2", "3.3", "1", "2", "3")
_INT_POOL = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
_C4_FUNCS = {"round", "int", "abs", "min", "max", "float"}


def _c4_expr_ok(expr: ast.AST) -> bool:
    """Is ``expr`` inside C4 ``float_money_hazard``'s whitelist?"""
    for n in ast.walk(expr):
        if isinstance(n, ast.Call):
            if not (isinstance(n.func, ast.Name) and n.func.id in _C4_FUNCS) or n.keywords:
                return False
        elif isinstance(n, (ast.Attribute, ast.Subscript, ast.Starred, ast.Lambda, ast.IfExp,
                            ast.BoolOp, ast.JoinedStr, ast.List, ast.Tuple, ast.Dict, ast.Set)):
            return False
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            if _int_literal(n.right) is None or abs(_int_literal(n.right) or 0) > 64:
                return False
        elif isinstance(n, ast.BinOp) and not isinstance(n.op, _ARITH):
            return False
        elif isinstance(n, ast.Constant) and type(n.value) not in (int, float):
            return False
    return True


def _float_hazard(program: str, pools: Mapping[str, Sequence[str]], currency: "int | None" = None,
                  ) -> "tuple[dict[str, str], str, str] | None":
    from ..code.money import MoneyInputError, float_money_hazard

    try:
        h = float_money_hazard(program, variables={k: list(v) for k, v in pools.items()},
                               currency=currency, max_checks=100_000)
    except (MoneyInputError, PycodemathError, ArithmeticError, ValueError, TypeError, RecursionError):
        return None
    if not h.hazard or h.witness is None:
        return None
    return dict(h.witness), h.float_value, h.exact_value


def _frac_eval(node: ast.AST, env: Mapping[str, Fraction], source: str) -> Any:
    """Fast exact evaluation (``Fraction``) of C4-whitelisted arithmetic, for the
    witness SEARCH only — the witness found is then replayed by SymPy and
    confirmed by C4 (``float_money_hazard``) before anything is reported."""
    if isinstance(node, ast.Expression):
        return _frac_eval(node.body, env, source)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int):
            return Fraction(node.value)
        return Fraction(Decimal(ast.get_source_segment(source, node) or repr(node.value)))
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        v = _frac_eval(node.operand, env, source)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        x, y = _frac_eval(node.left, env, source), _frac_eval(node.right, env, source)
        if isinstance(node.op, ast.Add):
            return x + y
        if isinstance(node.op, ast.Sub):
            return x - y
        if isinstance(node.op, ast.Mult):
            return x * y
        if isinstance(node.op, ast.Div):
            return x / y
        if isinstance(node.op, ast.FloorDiv):
            return Fraction(math.floor(x / y))
        if isinstance(node.op, ast.Mod):
            return x - y * math.floor(x / y)
        if isinstance(node.op, ast.Pow):
            return x ** int(y)
        raise ZeroDivisionError("operator")
    if isinstance(node, ast.Compare):
        left = _frac_eval(node.left, env, source)
        right = _frac_eval(node.comparators[0], env, source)
        return left == right if isinstance(node.ops[0], ast.Eq) else left != right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_frac_eval(x, env, source) for x in node.args]
        f = node.func.id
        if f == "abs":
            return abs(args[0])
        if f in ("min", "max"):
            return min(args) if f == "min" else max(args)
        if f == "float":
            return args[0]
        if f == "int":
            return Fraction(math.trunc(args[0]))
        if f == "round":
            nd = int(args[1]) if len(args) > 1 else 0
            return Fraction(round(args[0] * 10**nd), 10**nd)  # Fraction.__round__: half to even
    raise ZeroDivisionError("unsupported")


def _check_float_equality(ctx: _Ctx) -> None:
    """``==`` / ``!=`` reported only in the direction that is a bug: the values are
    EQUAL in exact decimal arithmetic and the float comparison says they are not
    (``0.1 + 0.2 == 0.3`` is False). The other direction (float says equal, exact
    says not) is a statement about a 17-digit literal, not a mistake anyone makes."""
    import itertools

    for node in ast.walk(ctx.tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], (ast.Eq, ast.NotEq))):
            continue
        kinds, _, fn = _kinds_at(ctx, node)
        full = _inline_at(ctx, node)
        if not any(isinstance(n, ast.BinOp) for n in ast.walk(full)) or not _c4_expr_ok(full):
            continue
        names = _free_names(full)
        if len(names) > 3 or any(kinds.get(n) not in ("float", "int") for n in names):
            continue
        floaty = any(kinds.get(n) == "float" for n in names) or any(
            _literal_kind(n) == "float" for n in ast.walk(full) if isinstance(n, ast.expr)) or any(
            isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in ast.walk(full))
        if not floaty:
            continue
        program = ast.unparse(full)
        tree = ast.parse(program, mode="eval")
        code = compile(tree, "<cmp>", "eval")
        eq = isinstance(node.ops[0], ast.Eq)
        pools = [(_FLOAT_POOL if kinds.get(n) == "float" else _INT_POOL) for n in names]
        found: "dict[str, str] | None" = None
        for combo in itertools.islice(itertools.product(*pools), 20_000):
            env_e = {k: Fraction(Decimal(v)) for k, v in zip(names, combo)}
            env_f = {k: (float(v) if kinds.get(k) == "float" else int(v)) for k, v in zip(names, combo)}
            try:
                exact = _frac_eval(tree, env_e, program)
                got = eval(code, {"__builtins__": {f: __builtins__[f] if isinstance(__builtins__, dict)  # type: ignore[index]
                                                   else getattr(__builtins__, f) for f in _C4_FUNCS}}, env_f)
            except (ArithmeticError, ValueError, TypeError, OverflowError):
                continue
            if exact == eq and bool(got) != exact:
                found = dict(zip(names, combo))
                break
        if found is None:
            continue
        confirm = _float_hazard(program, {k: [v] for k, v in found.items()})  # C4, the second route
        if confirm is None:
            ctx.count("dropped:c4-disagrees")
            continue
        w = Witness("float_equality", program, found, observed=confirm[1], expected=confirm[2], compare="value",
                    reading="decimal", note="inputs and literals as the decimals written")
        left, right = ast.unparse(node.left), ast.unparse(node.comparators[0])
        sugg = f"math.isclose({left}, {right})" if eq else f"not math.isclose({left}, {right})"
        fix = ctx.fix("compare with a tolerance (choose rel_tol/abs_tol for your values)", sugg, node, None)
        ctx.add("PCM501", node, f"float `{ast.unparse(node)}` disagrees with exact arithmetic: "
                f"{w.describe()}", w, fix, fn)


# --------------------------------------------------------------------------- #
# PCM4xx / PCM503 — money                                                     #
# --------------------------------------------------------------------------- #


def _money_pools(names: Sequence[str]) -> "dict[str, tuple[str, ...]] | None":
    from ..code.money import money_grid

    hi = {1: "100.00", 2: "3.00", 3: "0.40"}.get(len(names))
    if hi is None:
        return None
    grid = money_grid("0.01", hi)
    return {n: grid for n in names}


def _money_names_ok(ctx: _Ctx, names: Sequence[str], kinds: Mapping[str, "str | None"],
                    target: "str | None" = None, where: "ast.AST | None" = None) -> bool:
    """Every free name is an amount that can hold cents: money-named (or the one
    operand of a money-named assignment), and not declared ``int``/``Decimal``."""
    if not names:
        return True
    money = [n for n in names if ctx.money_name(n, where)]
    if len(money) != len(names):
        if not (target and ctx.money_name(target, where) and len(names) == 1):
            return False
    return all(kinds.get(n) in (None, "float") for n in names)


def _assign_target(ctx: _Ctx, node: ast.AST) -> "str | None":
    parent = ctx.parents.get(id(node))
    if isinstance(parent, ast.Assign) and len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
        return parent.targets[0].id
    if isinstance(parent, (ast.AnnAssign, ast.AugAssign)) and isinstance(parent.target, ast.Name):
        return parent.target.id
    if isinstance(parent, ast.Return):
        fn = ctx.owner.get(id(node))
        return fn.name if fn is not None else None
    return None


def _round_digits(node: ast.Call) -> "int | None":
    if len(node.args) == 1 and not node.keywords:
        return 0
    if len(node.args) == 2 and not node.keywords:
        return _int_literal(node.args[1])
    return None


def _is_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _check_money(ctx: _Ctx) -> None:
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Call):
            continue
        if ctx.on("PCM403"):
            _check_decimal_literal(ctx, node)
        if _is_call(node, "round") and ctx.on("PCM402"):
            _check_double_round(ctx, node)
        float_target = _is_call(node, "round") and node.args and (_round_digits(node) or 0) >= 1
        int_target = _is_call(node, "int") and len(node.args) == 1 and isinstance(node.args[0], ast.BinOp) \
            and isinstance(node.args[0].op, ast.Mult) and any(
                _int_literal(s) in (100, 1000) for s in (node.args[0].left, node.args[0].right))
        if (float_target or int_target) and ctx.on("PCM401"):
            _check_float_money(ctx, node)
        if _is_call(node, "round") and ctx.on("PCM503"):
            _check_bankers(ctx, node)


def _check_float_money(ctx: _Ctx, node: ast.Call) -> None:
    kinds, _, fn = _kinds_at(ctx, node)
    full = _inline_at(ctx, node)
    names = _free_names(full)
    target = _assign_target(ctx, node)
    if not names or not _c4_expr_ok(full) or not _money_names_ok(ctx, names, kinds, target, node):
        return
    if not (any(ctx.money_name(n, node) for n in names) or (target and ctx.money_name(target, node))):
        return
    pools = _money_pools(names)
    if pools is None:
        return
    program = ast.unparse(full)
    found = _float_hazard(program, pools)
    if found is None:
        return
    inputs, fval, eval_ = found
    w = Witness("float_money", program, inputs, observed=fval, expected=eval_, compare="value",
                reading="decimal", note="amounts and literals as the decimals written; round() is Python's")
    ctx.add("PCM401", node, f"binary floats change this amount: {w.describe()} — use Decimal", w, None, fn)


def _check_double_round(ctx: _Ctx, node: ast.Call) -> None:
    outer = _round_digits(node)
    inner_call = node.args[0] if node.args else None
    if outer is None or not isinstance(inner_call, ast.Call) or not _is_call(inner_call, "round"):
        return
    inner = _round_digits(inner_call)
    if inner is None or inner <= outer or inner - outer > 4:
        return
    kinds, _, fn = _kinds_at(ctx, node)
    arg = inner_call.args[0]
    single = ast.Call(ast.Name("round", ast.Load()), [copy.deepcopy(arg)] + ([ast.Constant(outer)] if len(node.args) == 2 else []), [])
    # the value being rounded: search a decimal grid one digit finer than the inner rounding
    step = Fraction(1, 10 ** (inner + 1))
    program = ast.unparse(node)
    if isinstance(arg, ast.Name):
        if kinds.get(arg.id) not in (None, "float"):
            return
        var: "str | None" = arg.id
    elif _literal_kind(arg) is not None:
        var = None
    else:
        return  # the rounded value is an expression: its reachable values are not ours to choose
    candidates: list[dict[str, str]]
    if var is None:
        candidates = [{}]
    else:
        candidates = [{var: str(Decimal(k) * Decimal(step.numerator) / Decimal(step.denominator))}
                      for k in range(1, 10 ** (inner + 2) // 10 + 1)]
    for inputs in candidates[:20000]:
        w = Witness("double_rounding", program, inputs, observed="", expected="", compare="value",
                    reading="decimal", expected_program=ast.unparse(single))
        try:
            rep = w.replay()
            # the difference must hold in exact decimal arithmetic too, not only for floats
            from .witness import exact_value

            exact_double = exact_value(program, inputs, "decimal")
            exact_single = exact_value(ast.unparse(single), inputs, "decimal")
        except (WitnessError, ArithmeticError, ValueError):
            continue
        if rep.differs and exact_double != exact_single:
            w = replace(w, observed=rep.observed, expected=rep.expected)
            fix = ctx.fix("round once", ast.unparse(single), node, None)
            ctx.add("PCM402", node, f"double rounding: {w.describe()}", w, fix, fn, replayed=True)
            return


def _check_decimal_literal(ctx: _Ctx, node: ast.Call) -> None:
    if _dotted(node.func).split(".")[-1] != "Decimal" or len(node.args) != 1 or node.keywords:
        return
    arg = node.args[0]
    if _literal_kind(arg) != "float":
        return
    text = ast.get_source_segment(ctx.source, arg) or ast.unparse(arg)
    try:
        written = Decimal(text.replace("_", ""))
    except ArithmeticError:
        return
    held = Decimal(float(text.replace("_", "")))
    if held == written:
        return
    _, _, fn = _kinds_at(ctx, node)
    # program in binary reading: the float vs the literal as a decimal is exactly the question
    w = Witness("decimal_from_float", "x", {"x": repr(float(text.replace('_', '')))},
                observed=repr(float(text.replace("_", ""))), expected=str(written), compare="value",
                reading="binary", expected_program=text,
                note=f"Decimal({text}) == Decimal('{held}')", extra={"decimal_holds": str(held)})
    fix = ctx.fix("pass the literal as a string", f"{_dotted(node.func)}(\"{text}\")", node,
                  "Decimal(str) holds the decimal written")
    ctx.add("PCM403", node, f"Decimal({text}) holds {held}, not {written}", w, fix, fn)


def _check_bankers(ctx: _Ctx, node: ast.Call) -> None:
    from ..code.money import MoneyInputError, round_money

    nd = _round_digits(node)
    if nd is None or nd < 0 or nd > 4 or not node.args:
        return
    kinds, _, fn = _kinds_at(ctx, node)
    target = _assign_target(ctx, node)
    full = _inline_at(ctx, node)
    arg = full.args[0] if isinstance(full, ast.Call) else None
    if arg is None:
        return
    names = _free_names(arg)
    money_ctx = any(ctx.money_name(n, node) for n in names) or (target is not None and ctx.money_name(target, node))
    if not names or not money_ctx or not _c4_expr_ok(arg) or not _money_names_ok(ctx, names, kinds, target, node):
        return
    pools = _money_pools(names)
    if pools is None:
        return
    program = ast.unparse(full)
    arg_text = ast.unparse(arg)
    import itertools

    from .witness import exact_value, run_program

    arg_tree = ast.parse(arg_text, mode="eval")
    for combo in itertools.islice(itertools.product(*[pools[n] for n in names]), 100_000):
        inputs = dict(zip(names, combo))
        try:  # fast exact filter first: only a TIE can make the mode matter
            units = _frac_eval(arg_tree, {k: Fraction(Decimal(v)) for k, v in inputs.items()}, arg_text) * 10**nd
        except (ArithmeticError, ValueError, TypeError, KeyError):
            continue
        if not isinstance(units, Fraction) or units.denominator != 2:
            continue
        observed = run_program(program, inputs)
        try:
            e_arg = exact_value(arg_text, inputs, "decimal")
            half_even = exact_value(program, inputs, "decimal", "HALF_EVEN")
            up = round_money(str(Decimal(int(e_arg.p)) / Decimal(int(e_arg.q))), nd, "HALF_UP").rounded
        except (MoneyInputError, PycodemathError, WitnessError, ArithmeticError, AttributeError):
            continue
        obs_frac = Fraction(Decimal(repr(observed))) if isinstance(observed, float) else Fraction(observed)
        if obs_frac != Fraction(int(half_even.p), int(half_even.q)) or Fraction(up) == obs_frac:
            continue  # a float effect (PCM401's business) or no difference
        w = Witness("bankers_rounding", program, inputs, observed=repr(observed) if isinstance(observed, float)
                    else str(observed), expected=str(up), compare="value", reading="decimal",
                    rounding="HALF_UP", note="expected: C4 round_money(..., HALF_UP)")
        ctx.add("PCM503", node, f"round() is half-to-even: {w.describe()} with HALF_UP", w, None, fn)
        return


# --------------------------------------------------------------------------- #
# numeric functions: one expression (for C2 / C3)                             #
# --------------------------------------------------------------------------- #

_C2_FUNCS = {
    "sqrt": "sqrt", "cbrt": "cbrt", "exp": "exp", "log": "log", "log1p": "log1p", "expm1": "expm1",
    "sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan",
    "arcsin": "asin", "arccos": "acos", "arctan": "atan", "sinh": "sinh", "cosh": "cosh",
    "tanh": "tanh", "asinh": "asinh", "acosh": "acosh", "atanh": "atanh", "arcsinh": "asinh",
    "arccosh": "acosh", "arctanh": "atanh", "fabs": "fabs", "absolute": "fabs", "hypot": "hypot",
    "pow": "pow", "power": "pow", "atan2": "atan2", "arctan2": "atan2",
}


class _NotNumeric(Exception):
    pass


def _c2_text(node: ast.expr, params: set[str]) -> str:
    """The expression in C2's formula notation (evaluation order kept by parentheses)."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise _NotNumeric("non-finite literal")
        return repr(node.value)
    if isinstance(node, ast.Name):
        if node.id in params and not node.id.startswith("_") and node.id not in ("pi", "E"):
            return node.id
        raise _NotNumeric(f"name {node.id!r} is not a parameter")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _c2_text(node.operand, params)
        return f"(-{inner})" if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
        op = ops.get(type(node.op))
        if op is None:
            raise _NotNumeric(f"operator {type(node.op).__name__}")
        return f"({_c2_text(node.left, params)} {op} {_c2_text(node.right, params)})"
    if isinstance(node, ast.Attribute):
        d = _dotted(node)
        if d in ("math.pi", "np.pi", "numpy.pi"):
            return "pi"
        if d in ("math.e", "np.e", "numpy.e"):
            return "E"
        raise _NotNumeric(f"attribute {d or 'access'}")
    if isinstance(node, ast.Call) and not node.keywords:
        d = _dotted(node.func)
        parts = d.split(".")
        name = parts[-1]
        if d == "abs":
            name = "fabs"
        elif len(parts) != 2 or parts[0] not in _MODULE_NAMES:
            raise _NotNumeric(f"call to {d or 'a function'}")
        args = [_c2_text(a, params) for a in node.args]
        if name == "log" and len(args) == 2:
            return f"(log({args[0]}) / log({args[1]}))"
        if name == "log10" and len(args) == 1:
            return f"(log({args[0]}) / log(10))"
        if name == "log2" and len(args) == 1:
            return f"(log({args[0]}) / log(2))"
        if name == "square" and len(args) == 1:
            return f"({args[0]} * {args[0]})"
        if name not in _C2_FUNCS:
            raise _NotNumeric(f"function {d}")
        return f"{_C2_FUNCS[name]}({', '.join(args)})"
    raise _NotNumeric(type(node).__name__)


def _return_expression(info: _Func) -> "tuple[ast.Return, ast.expr] | None":
    """``return <expr>`` of a straight-line function, locals inlined; else None."""
    body = list(info.node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return None
    local: dict[str, ast.expr] = {}
    for st in body[:-1]:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            local[st.targets[0].id] = _inline(st.value, local)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            local[st.target.id] = _inline(st.value, local)
        elif isinstance(st, (ast.Assert, ast.Pass)):
            continue
        else:
            return None
    ret = body[-1]
    assert ret.value is not None
    return ret, _inline(ret.value, local)


def _dtype_of(info: _Func, names: Sequence[str]) -> "str | None":
    """One NumPy dtype for every name, or ``float64`` for plain floats, or None."""
    ks = {info.kinds.get(n) for n in names}
    np_ks = {k for k in ks if k and k.startswith("np:")}
    if len(np_ks) == 1 and ks <= np_ks | {None} and None not in ks:
        return next(iter(np_ks))[3:]
    if np_ks:
        return None
    if ks == {"float"}:
        return "float64"
    return None


# --------------------------------------------------------------------------- #
# PCM1xx — precision (C2)                                                     #
# --------------------------------------------------------------------------- #


def _boxes(info: _Func, ctx: _Ctx, names: Sequence[str]) -> tuple[list[dict[str, tuple[float, float]]], bool]:
    """Input boxes for C2: declared ranges, else the magnitude range split by sign.
    Second value: whether every range was DECLARED (then a fix is in scope)."""
    lo, hi = ctx.config.default_range
    per: list[list[tuple[float, float]]] = []
    declared = True
    for n in names:
        r = info.spec.ranges.get(n) or ctx.config.range_for(n)
        if r is not None:
            per.append([(float(r[0]), float(r[1]))])
        else:
            declared = False
            per.append([(lo, hi), (-hi, -lo)])
    import itertools

    combos = list(itertools.product(*per))
    if len(combos) > 8:
        combos = [tuple(p[0] for p in per)]
    return [dict(zip(names, c)) for c in combos], declared


def _code_module(expr: ast.AST) -> str:
    mods = {n.value.id for n in ast.walk(expr) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    return "np" if "np" in mods else "numpy" if "numpy" in mods else "math"


def _to_code(candidate: str, module: str) -> str:
    """C2's candidate (``sqrt(x)``) in the code's own module (``math.sqrt(x)``)."""
    tree = ast.parse(candidate, mode="eval")

    class T(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name == "fabs":
                    node.func = ast.Name("abs", ast.Load())
                elif name in ("max", "min"):
                    pass
                else:
                    np_names = {"asin": "arcsin", "acos": "arccos", "atan": "arctan", "atan2": "arctan2",
                                "asinh": "arcsinh", "acosh": "arccosh", "atanh": "arctanh", "pow": "power"}
                    attr = np_names.get(name, name) if module != "math" else name
                    node.func = ast.Attribute(ast.Name(module, ast.Load()), attr, ast.Load())
            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id == "pi":
                return ast.Attribute(ast.Name(module, ast.Load()), "pi", ast.Load())
            if node.id == "E":
                return ast.Attribute(ast.Name(module, ast.Load()), "e", ast.Load())
            return node

    return ast.unparse(ast.fix_missing_locations(T().visit(tree)))


def _check_precision(ctx: _Ctx, info: _Func) -> None:
    got = _return_expression(info)
    if got is None:
        return
    ret, expr = got
    expr = _inline(expr, ctx.consts)
    names = _free_names(expr)
    if not names or any(n not in info.params for n in names):
        return
    dtype = _dtype_of(info, names)
    floaty = any(isinstance(n, ast.Call) for n in ast.walk(expr)) or any(
        _literal_kind(n) == "float" for n in ast.walk(expr) if isinstance(n, ast.expr)) or any(
        isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in ast.walk(expr))
    if dtype is None:
        if any(info.kinds.get(n) is not None for n in names) or not floaty:
            return  # ints, Decimals, mixed or unknown types: not a float formula
        dtype = "float64"
    if dtype not in ("float64", "float32"):
        return
    if not any(isinstance(n, (ast.BinOp, ast.Call)) for n in ast.walk(expr)):
        return
    try:
        text = _c2_text(expr, set(info.params))
    except _NotNumeric as exc:
        ctx.add("PCM902", ret, f"{info.node.name}: not analysed for precision ({exc})", None, func=info.node)
        return
    module = _code_module(expr)
    if dtype == "float32" and module == "math":
        if any(isinstance(n, ast.Call) for n in ast.walk(expr)):
            return  # math.* computes in float64; the float32 model would not be the code
    from ..code.stability import StabilityDomainError, diagnose

    ctx.count("precision")
    k = ctx.config.precision_bits * (24 / 53 if dtype == "float32" else 1.0)
    boxes, declared = _boxes(info, ctx, names)
    budget = ctx.config.budget
    deadline = time.perf_counter() + budget
    worst = None
    for box in boxes:
        left = deadline - time.perf_counter()
        if left <= 0.05:
            ctx.add("PCM901", ret, f"{info.node.name}: precision analysis stopped at the {budget:g} s budget",
                    None, func=info.node)
            return
        try:
            rep = diagnose(text, box, dtype=dtype, samples=128, budget=left)
        except StabilityDomainError:
            continue
        except TimeBudgetError:
            ctx.add("PCM901", ret, f"{info.node.name}: precision analysis stopped at the {budget:g} s budget",
                    None, func=info.node)
            return
        except PycodemathError as exc:
            ctx.add("PCM902", ret, f"{info.node.name}: not analysed for precision ({exc})", None, func=info.node)
            return
        if rep.worst is not None and rep.max_bits >= k and (worst is None or rep.max_bits > worst[0].max_bits):
            worst = (rep, box)
    if worst is None:
        return
    rep, box = worst
    assert rep.worst is not None
    program = ast.unparse(expr)
    w = Witness("precision", program, dict(rep.worst.inputs), observed=rep.worst.computed,
                expected=rep.worst.exact, compare=f"bits:{k:g}", dtype=None if dtype == "float64" else dtype,
                extra={"bits": round(rep.worst.bits, 1), "mean_bits": round(rep.mean_bits, 2),
                       "range": {v: [repr(a), repr(b)] for v, (a, b) in box.items()}})
    r2 = _replay(ctx, w)
    if r2 is None:
        return
    if not r2.differs:
        ctx.count("dropped:replay-agrees")
        return
    fix = _precision_fix(ctx, text, box, dtype, rep.max_bits, ret, module, declared, info)
    scope = "declared range" if declared else f"default range {ctx.config.default_range[0]:g} <= |x| <= {ctx.config.default_range[1]:g}"
    ctx.add("PCM101", ret.value or ret,
            f"{info.node.name} has {rep.worst.bits:.0f} bits of error in {dtype} ({scope}): {w.describe()}", w, fix, info.node, replayed=True)


def _precision_fix(ctx: _Ctx, text: str, box: Mapping[str, tuple[float, float]], dtype: str, before: float,
                   ret: ast.Return, module: str, declared: bool, info: _Func) -> "Fix | None":
    from ..code.stability import improve

    try:
        imp = improve(text, box, dtype=dtype, samples=128, budget=max(1.0, 2 * ctx.config.budget))
    except (PycodemathError, ArithmeticError, ValueError, RecursionError):
        return None
    best = imp.best
    if not imp.improved or best is None or best.proof.status != "verified" or best.max_bits >= before:
        return None
    try:
        code = _to_code(best.expression, module)
    except SyntaxError:
        return None
    where = ", ".join(f"{v} in [{a:g}, {b:g}]" for v, (a, b) in box.items())
    conds = f"; conditions: {', '.join(best.conditions)}" if best.conditions else ""
    assert ret.value is not None
    return ctx.fix(f"proved equal on {where}{conds}; {imp.before_max_bits:.0f} -> {best.max_bits:.0f} bits "
                   "at worst" + ("" if declared else " (default range: declare :range before applying)"),
                   code, ret.value, f"C2 {best.proof.method}" if declared else None)


# --------------------------------------------------------------------------- #
# PCM2xx — ranges for a declared dtype (C3)                                   #
# --------------------------------------------------------------------------- #

_EVENT_RULE = {"overflow": "PCM201", "integer_precision_loss": "PCM202", "nan_possible": "PCM203",
               "division_by_zero_possible": "PCM204", "underflow_to_zero": "PCM205"}


def _locate(expr_root: ast.AST, text: str, fallback: ast.AST) -> ast.AST:
    for n in ast.walk(expr_root):
        if isinstance(n, ast.expr) and ast.unparse(n) == text and hasattr(n, "lineno"):
            return n
    return fallback


def _check_ranges(ctx: _Ctx, info: _Func) -> None:
    got = _return_expression(info)
    if got is None:
        return
    ret, expr = got
    expr = _inline(expr, ctx.consts)
    names = _free_names(expr)
    if not names or any(n not in info.params for n in names):
        return
    ks = {info.kinds.get(n) for n in names}
    if not all(k and k.startswith("np:") for k in ks) or len(ks) != 1:
        if not (ks == {"float"} and all(n in info.spec.ranges for n in names)):
            return
    dtype = _dtype_of(info, names)
    if dtype is None:
        return
    if dtype in ("float32", "float64") and not all(n in info.spec.ranges for n in names):
        # "every value of float64" overflows every + and * at 1.8e308: true and useless
        # (measured on TheAlgorithms, fuzzy_logic). Integers and float16 keep the whole type.
        return
    if not any(ctx.on(c) for c in _EVENT_RULE.values()):
        return
    from ..code.ranges import analyze

    ranges = {n: info.spec.ranges[n] for n in names if n in info.spec.ranges}
    ctx.count("ranges")
    try:
        rep = analyze(ast.unparse(expr), ranges or None, dtype, budget=ctx.config.budget)
    except TimeBudgetError:
        ctx.add("PCM901", ret, f"{info.node.name}: range analysis stopped at the budget", None, func=info.node)
        return
    except PycodemathError as exc:
        ctx.add("PCM902", ret, f"{info.node.name}: not analysed for ranges ({exc})", None, func=info.node)
        return
    except (ArithmeticError, ValueError, TypeError, RecursionError) as exc:
        ctx.add("PCM902", ret, f"{info.node.name}: not analysed for ranges ({type(exc).__name__})", None,
                func=info.node)
        return
    done: set[str] = set()
    scope = "declared range" if ranges else f"every value of {dtype}"
    for ev in rep.confirmed:
        code = _EVENT_RULE.get(ev.kind)
        if code is None or code in done or not ctx.on(code) or ev.witness is None:
            continue
        w = Witness(ev.kind, ev.node_text, dict(ev.witness), observed=ev.observed or "", expected=ev.exact or "",
                    compare=f"event:{ev.kind}", dtype=dtype, note=ev.method)
        fix = None
        for adv in rep.advice:
            if ev.kind in adv.removes and adv.dtype == dtype and adv.equivalence == "verified":
                assert ret.value is not None
                fix = ctx.fix(adv.detail, adv.rewrite, ret.value, f"C3 {adv.method}")
                break
        where = _locate(ret.value or ret, ev.node_text, ret.value or ret)
        words = {"overflow": "overflows", "integer_precision_loss": "loses integers",
                 "nan_possible": "creates NaN", "division_by_zero_possible": "divides by zero",
                 "underflow_to_zero": "underflows to 0"}[ev.kind]
        before = len(ctx.findings)
        ctx.add(code, where, f"`{ev.node_text}` {words} in {dtype} ({scope}): {w.describe()}", w, fix, info.node)
        if len(ctx.findings) > before:
            done.add(code)


# --------------------------------------------------------------------------- #
# PCM3xx / PCM504 — spec (C1)                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpecTarget:
    """A function with a spec, as the pytest plugin and the engine see it."""

    path: str
    name: str
    line: int
    formula: str
    domain: dict[str, str]


def spec_functions(source: str, path: str = "<string>") -> list[SpecTarget]:
    """Every top-level function with a ``:math:`` line or an ``@spec`` decorator."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            s = _spec_of(node)
            if s.formula:
                out.append(SpecTarget(path, node.name, node.lineno, s.formula, dict(s.domain)))
    return out


def _range_mutations(fdef: ast.FunctionDef) -> Iterator[tuple[ast.Call, int, int, str]]:
    """(range call, argument index, delta, new argument text) for each one-off bound."""
    for node in ast.walk(fdef):
        if isinstance(node, ast.Call) and _dotted(node.func) == "range" and 1 <= len(node.args) <= 2 \
                and not node.keywords:
            for i, a in enumerate(node.args):
                for delta in (1, -1):
                    lit = _int_literal(a)
                    if lit is not None:
                        new = str(lit + delta)
                    else:
                        new = f"{ast.unparse(a)} {'+' if delta > 0 else '-'} 1"
                        if isinstance(a, ast.BinOp) and isinstance(a.op, (ast.Add, ast.Sub)) \
                                and _int_literal(a.right) is not None:
                            lit2 = _int_literal(a.right) or 0
                            k = lit2 + (delta if isinstance(a.op, ast.Add) else -delta)
                            base = ast.unparse(a.left)
                            new = base if k == 0 else f"{base} + {k}" if k > 0 else f"{base} - {-k}"
                    yield node, i, delta, new


def _replace_segment(source: str, node: ast.AST, text: str) -> str:
    """``source`` with ``node``'s text replaced (AST columns are UTF-8 byte offsets)."""
    lb = [len(line.encode("utf-8")) for line in source.splitlines(keepends=True)]
    line, col = getattr(node, "lineno"), getattr(node, "col_offset")
    end_line, end_col = getattr(node, "end_lineno", None) or line, getattr(node, "end_col_offset", None) or 0
    start = sum(lb[: line - 1]) + col
    end = sum(lb[: end_line - 1]) + end_col
    b = source.encode("utf-8")
    return (b[:start] + text.encode("utf-8") + b[end:]).decode("utf-8")


def _is_spec_decorator(dec: ast.expr) -> bool:
    if not isinstance(dec, ast.Call):
        return False
    d = _dotted(dec.func).split(".")
    return d[-1] == "spec" and (len(d) == 1 or d[0] in ("pcm", "lint", "pycodemath"))


def _spec_source(source: str, tree: ast.Module) -> str:
    """The module as C1 should read it: ``import pycodemath...`` and ``@spec(...)``
    lines blanked (C1 refuses to execute a module that imports a package outside its
    safe list, and the decorator has no effect anyway). Lines are blanked, not
    removed, so every other position is unchanged."""
    blank: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and all(a.name.split(".")[0] == "pycodemath" for a in node.names):
            blank.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "pycodemath":
            blank.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if _is_spec_decorator(dec):
                    blank.update(range(dec.lineno, (dec.end_lineno or dec.lineno) + 1))
    if not blank:
        return source
    lines = source.splitlines(keepends=True)
    return "".join("\n" if i + 1 in blank else line for i, line in enumerate(lines))


def _run_check(source: str, name: str, formula: str, domain: Mapping[str, str], budget: float,
               execute: bool) -> Any:
    from ..code.frontend import check_code
    from ..verify.certificate import code_execution

    with code_execution(execute):
        return check_code(source, formula, name=name, domain=dict(domain) or None, budget=budget)


def check_spec(source: str, path: str, name: str, config: "Config | None" = None) -> list[Finding]:
    """The spec check of one function — PCM301 / PCM504 / PCM302 / PCM901 findings
    (unfiltered by selection). Used by the engine and by ``pytest --pcm``."""
    config = config or Config()
    tree = ast.parse(source)
    ctx = _build_ctx(source, path, replace(config, select=("PCM",)), tree)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            _check_spec_fn(ctx, ctx.funcs[id(node)])
    return ctx.findings


def _check_spec_fn(ctx: _Ctx, info: _Func) -> None:
    spec = info.spec
    if not spec.formula or not ctx.on("PCM301", "PCM302", "PCM504", "PCM901"):
        return
    ctx.count("spec")
    fdef = info.node
    budget = ctx.config.budget
    source = _spec_source(ctx.source, ctx.tree)
    try:
        res = _run_check(source, fdef.name, spec.formula, spec.domain, budget, ctx.config.execute)
    except TimeBudgetError:
        ctx.add("PCM901", fdef, f"{fdef.name}: spec check stopped at the {budget:g} s budget", None, func=fdef)
        return
    except PycodemathError as exc:
        ctx.add("PCM302", fdef, f"{fdef.name}: spec `{spec.formula}` not checked ({exc})", None, func=fdef)
        return
    if res.verdict == "undecided":
        code = "PCM901" if "budget" in res.reason.lower() else "PCM302"
        ctx.add(code, fdef, f"{fdef.name}: spec `{spec.formula}` undecided — {res.reason}", None, func=fdef)
        return
    if res.verdict != "refuted":
        return
    cx = res.counterexample
    if cx is None or not cx.confirmed_by_execution:
        ctx.add("PCM302", fdef, f"{fdef.name}: counterexample not confirmed by execution", None, func=fdef)
        return
    w = Witness("spec", cx.call, {k: repr(v) for k, v in cx.args.items()}, observed=repr(cx.function_value),
                expected=cx.formula_value, compare="call", formula=spec.formula,
                tolerance=0.0 if cx.exact else cx.tolerance, note="executed by C1 in its isolated worker")
    # PCM504: does moving one range bound by one make C1 PROVE the spec?
    if ctx.on("PCM504"):
        for call, i, delta, new in _range_mutations(fdef):
            mutated = _replace_segment(source, call.args[i], new)
            try:
                r2 = _run_check(mutated, fdef.name, spec.formula, spec.domain, budget, ctx.config.execute)
            except PycodemathError:
                continue
            if r2.verdict == "verified":
                arg = call.args[i]
                fix = ctx.fix(f"range bound {'+' if delta > 0 else '-'}1", new, arg,
                              f"C1 {r2.method}: the fixed function equals {spec.formula}")
                ctx.add("PCM504", call, f"{fdef.name}: `{ast.unparse(call)}` is off by one against "
                        f"`{spec.formula}`: {w.describe()}", w, fix, fdef, replayed=True)
                return
    ctx.add("PCM301", fdef, f"{fdef.name} does not compute `{spec.formula}`: {w.describe()}", w, None, fdef,
            replayed=True)


# --------------------------------------------------------------------------- #
# entry points                                                                #
# --------------------------------------------------------------------------- #


def _guarded(ctx: _Ctx, what: str, fn: Callable[[], None]) -> None:
    """One analysis; an unexpected exception inside it is counted, never a finding and
    never the end of the run (an internal error must not hide the other rules)."""
    try:
        fn()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - includes RecursionError and stray budget signals
        if type(exc).__name__ == "_Deadline":
            raise
        ctx.count(f"internal-error:{what}:{type(exc).__name__}")


def _analyse(ctx: _Ctx) -> None:
    if ctx.on("PCM502"):
        _guarded(ctx, "xor", lambda: _check_xor(ctx))
    if ctx.on("PCM501"):
        _guarded(ctx, "float-equality", lambda: _check_float_equality(ctx))
    if ctx.on("PCM401", "PCM402", "PCM403", "PCM503"):
        _guarded(ctx, "money", lambda: _check_money(ctx))
    for info in ctx.funcs.values():
        if ctx.on("PCM301", "PCM302", "PCM504"):
            _guarded(ctx, "spec", lambda: _check_spec_fn(ctx, info))
        if ctx.on("PCM101"):
            _guarded(ctx, "precision", lambda: _check_precision(ctx, info))
        if ctx.on("PCM201", "PCM202", "PCM203", "PCM204", "PCM205"):
            _guarded(ctx, "ranges", lambda: _check_ranges(ctx, info))


def _raw(source: str, path: str, config: Config) -> tuple[list[Finding], int, dict[str, int]]:
    source = source.lstrip("\ufeff")
    try:
        tree = ast.parse(source, filename=path)
    except (RecursionError, MemoryError):
        raise SyntaxError("too deeply nested to analyse") from None
    ctx = _build_ctx(source, path, config, tree)
    _analyse(ctx)
    ctx.findings.sort(key=lambda f: (f.line, f.col, f.code))
    return ctx.findings, len(ctx.funcs), ctx.analysed


def _filter(findings: Iterable[Finding], source: str) -> tuple[list[Finding], int]:
    lines, file_codes = _suppressions(source)
    kept, dropped = [], 0
    for f in findings:
        if _silenced(f.code, file_codes) or any(_silenced(f.code, lines.get(ln, set()))
                                                 for ln in (f.suppress_lines or (f.line,))):
            dropped += 1
        else:
            kept.append(f)
    return kept, dropped


def lint_source(source: str, path: str = "<string>", config: "Config | None" = None) -> list[Finding]:
    """Lint one file's text: findings after selection and ``# pcm: ignore`` comments."""
    config = config or Config()
    all_on = replace(config, select=("PCM",), extend_select=(), ignore=(), per_file_ignores=())
    findings, _, _ = _raw(source, path, all_on)
    rel = os.path.relpath(path, config.root) if os.path.isabs(path) else path
    enabled = config.enabled_codes(rel)
    return _filter([f for f in findings if f.code in enabled], source)[0]


def iter_python_files(paths: Sequence["str | Path"], config: Config) -> Iterator[Path]:
    """``.py`` files under ``paths`` (files given explicitly are never excluded)."""
    root = Path(config.root)
    for p in paths:
        p = Path(p)
        if p.is_file():
            yield p
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            d = Path(dirpath)
            keep = []
            for name in sorted(dirnames):
                rel = _rel(d / name, root)
                if not config.is_excluded(rel):
                    keep.append(name)
            dirnames[:] = keep
            for name in sorted(filenames):
                if name.endswith(".py") and not config.is_excluded(_rel(d / name, root)):
                    yield d / name


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _cache_path(config: Config, digest: str) -> Path:
    return Path(config.root) / config.cache_dir / ENGINE_VERSION / f"{digest}.json"


def _digest(source: str, config: Config, path: str) -> str:
    h = hashlib.sha256()
    h.update(ENGINE_VERSION.encode())
    h.update(config.fingerprint().encode())
    h.update(path.encode())
    h.update(source.encode("utf-8", "surrogatepass"))
    return h.hexdigest()


def lint_file(path: "str | Path", config: Config) -> FileReport:
    """Lint one file on disk, through the cache."""
    t0 = time.perf_counter()
    p = Path(path)
    shown = _rel(p, Path(config.root)) if p.is_absolute() or Path(config.root) != Path(".") else p.as_posix()
    try:
        source = p.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return FileReport(shown, (), 0, time.perf_counter() - t0, False, f"cannot read: {exc}")
    digest = _digest(source, config, shown)
    cache = _cache_path(config, digest)
    raw: "list[Finding] | None" = None
    cached = False
    functions = 0
    analysed: dict[str, int] = {}
    if config.cache and cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            raw = [Finding.from_dict(d, shown) for d in data["findings"]]
            functions, analysed, cached = data.get("functions", 0), data.get("analysed", {}), True
        except (OSError, ValueError, KeyError, TypeError):
            raw = None
    if raw is None:
        all_on = replace(config, select=("PCM",), extend_select=(), ignore=(), per_file_ignores=())
        try:
            raw, functions, analysed = _raw(source, shown, all_on)
        except SyntaxError as exc:
            return FileReport(shown, (), 0, time.perf_counter() - t0, False,
                              f"syntax error at line {exc.lineno}: {exc.msg}")
        if config.cache:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                marker = Path(config.root) / config.cache_dir / ".gitignore"
                if not marker.exists():  # as ruff's .ruff_cache: the cache never shows up in git
                    marker.write_text("# created by pycodemath lint\n*\n", encoding="utf-8")
                payload = {"findings": [dict(f.to_dict(), suppress_lines=list(f.suppress_lines)) for f in raw],
                           "functions": functions, "analysed": analysed}
                tmp = cache.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(tmp, cache)
            except OSError:
                pass
    enabled = config.enabled_codes(shown)
    selected = [f for f in raw if f.code in enabled]
    kept, dropped = _filter(selected, source)
    return FileReport(shown, tuple(kept), dropped, time.perf_counter() - t0, cached, None, functions, analysed)


def lint_paths(paths: Sequence["str | Path"], config: "Config | None" = None, *,
               on_file: "Callable[[FileReport], None] | None" = None) -> LintResult:
    """Lint every ``.py`` file under ``paths``. Results are cached by file hash
    (all rules are computed once; selection is applied afterwards, so changing
    ``--select`` never needs a fresh analysis)."""
    config = config or Config()
    t0 = time.perf_counter()
    reports = []
    for f in iter_python_files(paths, config):
        r = lint_file(f, config)
        reports.append(r)
        if on_file is not None:
            on_file(r)
    return LintResult(tuple(reports), time.perf_counter() - t0)
