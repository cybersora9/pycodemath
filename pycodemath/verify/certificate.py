"""``certificate`` — the record of one check, which anyone can re-check (X1).

A verdict on the screen is a claim about a run nobody else saw. A CERTIFICATE is
that run written down: the claim as typed and as parsed, the verdict and its exact
counterexample, the budget it ran under and the versions that produced it, sealed
by a sha256 of all of it. ``recheck`` takes the record ALONE, runs the same check
again, and reports every way the two disagree:

* a different verdict (status, method, counterexample, first wrong step);
* a digest that does not match the content — the file was edited by hand;
* a claim that parses differently today than when it was issued — the parser
  moved under it (the record keeps both the text and its parsed form for this);
* different versions — reported as a NOTE, not a failure: the point of a
  re-check on a newer SymPy is to find out whether the verdict survived it.

THE FORMAT (``schema: 1``), one JSON object, every key always present:

==============  ================================================================
``schema``      ``1``. A reader refuses any other number rather than guess
``kind``        what was checked: ``identity`` (``check_equal``), ``derivative``,
                ``integral``, ``roots``, ``limit``, ``ode``, ``quadrature``
                (the ``certify_*`` family), ``steps`` (``check_steps``). Open:
                ``register_kind`` adds one (``code_equiv``, ``stability``…)
``claim``       ``args`` (each argument as TEXT, by name), ``options`` (the
                keyword options that change the verdict, defaults spelled out),
                ``parsed`` (each expression as SymPy prints it after parsing),
                ``statement`` (one line for a human)
``source``      the text the claim came from (the REPL line), or ``null``
``verdict``     ``status``, ``method``, ``counterexample`` (variable -> exact
                value as text, set exactly when refuted), ``detail``; a
                ``steps`` certificate adds ``first_error``
``budget``      the wall-clock allowance in seconds the check ran under
``versions``    ``pycodemath``, ``sympy``, ``mpmath``, ``python``
``digest``      ``"sha256:"`` + hex of the canonical JSON of everything above
==============  ================================================================

The digest is over CANONICAL JSON (sorted keys, no whitespace, UTF-8), so it does
not depend on how the file was indented. It is tamper EVIDENCE, not a signature:
anyone can recompute it after an edit. That is why ``recheck`` never trusts the
recorded verdict — it re-runs the check, and a forged digest buys a forger
nothing but a mismatch reported one line lower.

WHAT A RE-CHECK DOES NOT PROMISE. The re-run is the same deterministic check
(fixed sample points), so on the same versions it reproduces exactly. A verdict
reached by ``time-budget`` depends on the machine: a re-run that runs out of time
where the original did not is reported as a mismatch, with the advice to raise
``budget``. Nothing is ever ``eval``-ed from the file: every argument goes back
through the same parser the original check used.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import re
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import mpmath
import sympy as sp

from ..core.budget import DEFAULT_TIME_BUDGET
from ..core.errors import DomainError, PycodemathError
from ..core.ir import Expr
from ..core.result import QuadratureResult
from ..frontend.parser import _check_cost, _CostExceeded, is_strict, parse_mode
from .certify import (
    certify_diff,
    certify_dsolve,
    certify_integrate,
    certify_limit,
    certify_nintegrate,
    certify_solve,
)
from .equal import SAMPLES, _parse_written, check_equal
from .steps import StepsResult, check_steps
from .verdict import Verdict, VerdictStatus

__all__ = [
    "SCHEMA_VERSION",
    "KINDS",
    "Kind",
    "Certificate",
    "Recheck",
    "CertificateMismatch",
    "issue",
    "from_verdict",
    "recheck",
    "register_kind",
    "content_digest",
    "versions",
]

#: The format version written into every certificate, and the only one read.
SCHEMA_VERSION = 1

_KEYS = ("schema", "kind", "claim", "source", "verdict", "budget", "versions", "digest")
_CLAIM_KEYS = ("args", "options", "parsed", "statement")
_STATUSES = {s.value for s in VerdictStatus}


class CertificateMismatch(DomainError):
    """A re-checked certificate did not reproduce; the message is the full report.

    A ``DomainError`` because the INPUT — the certificate — is what failed: it
    vouches for something the check no longer says (or never said).
    """


def versions() -> dict[str, str]:
    """The versions a certificate records: pycodemath, sympy, mpmath, python."""
    from .. import __version__

    return {
        "pycodemath": __version__,
        "sympy": sp.__version__,
        "mpmath": mpmath.__version__,
        "python": platform.python_version(),
    }


def _canonical(content: Mapping[str, Any]) -> bytes:
    return json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_digest(data: Mapping[str, Any]) -> str:
    """``"sha256:<hex>"`` of every field of ``data`` except ``digest`` itself."""
    content = {k: v for k, v in data.items() if k != "digest"}
    return "sha256:" + hashlib.sha256(_canonical(content)).hexdigest()


# --- kinds ---------------------------------------------------------------------
#: Argument types: ``expr`` (one expression, kept as text), ``exprs`` (a list of
#: them), ``name`` (a variable or function name), ``text`` (taken as is — a
#: derivation), ``float`` (a number kept as its exact ``repr``).
_ARG_TYPES = ("expr", "exprs", "name", "text", "float")


@dataclass(frozen=True)
class Kind:
    """What one ``kind`` of certificate checks, and how to run the check again.

    ``params`` are the positional arguments, each ``(name, type)`` with the type
    from ``expr | exprs | name | text | float``; ``options`` the keyword options
    with their defaults (recorded in full, so a re-check never depends on a
    default that later changed). ``run(args, options, budget)`` performs the
    check on the RECORDED arguments and returns a ``Verdict`` or ``StepsResult``.
    ``statement(parsed, args, options)`` is the one-line claim for a human.
    """

    params: "tuple[tuple[str, str], ...]"
    options: "Mapping[str, Any]"
    run: Callable[[dict, dict, float], "Verdict | StepsResult"]
    statement: Callable[[dict, dict, dict], str]


def _identity_run(a: dict, o: dict, budget: float) -> Verdict:
    return check_equal(a["left"], a["right"], vars=o["vars"], budget=budget,
                       samples=o["samples"], seed=o["seed"])


def _steps_run(a: dict, o: dict, budget: float) -> StepsResult:
    return check_steps(a["derivation"], vars=o["vars"], budget=budget,
                       samples=o["samples"], seed=o["seed"])


def _quadrature_run(a: dict, o: dict, budget: float) -> Verdict:
    value = float(a["value"])
    claim: "QuadratureResult | float" = value
    if o["error_estimate"] is not None:
        claim = QuadratureResult.success(value, float(o["error_estimate"]), 0, 0)
    return certify_nintegrate(a["expr"], a["var"], float(a["a"]), float(a["b"]),
                              claim, rtol=o["rtol"], budget=budget)


def _limit_statement(p: dict, a: dict, o: dict) -> str:
    return f"lim {a['var']} -> {p['to']}{o['dir']} of {p['expr']} == {p['result']}"


KINDS: dict[str, Kind] = {
    "identity": Kind(
        (("left", "expr"), ("right", "expr")),
        {"vars": None, "samples": SAMPLES, "seed": 0, "strict": True},
        _identity_run,
        lambda p, a, o: f"{p['left']} == {p['right']}",
    ),
    "derivative": Kind(
        (("expr", "expr"), ("var", "name"), ("result", "expr")),
        {"strict": True},
        lambda a, o, b: certify_diff(a["expr"], a["var"], a["result"], budget=b),
        lambda p, a, o: f"d/d{a['var']} {p['expr']} == {p['result']}",
    ),
    "integral": Kind(
        (("expr", "expr"), ("var", "name"), ("result", "expr")),
        {"strict": True},
        lambda a, o, b: certify_integrate(a["expr"], a["var"], a["result"], budget=b),
        lambda p, a, o: f"integral {p['expr']} d{a['var']} == {p['result']}",
    ),
    "roots": Kind(
        (("expr", "expr"), ("var", "name"), ("roots", "exprs")),
        {"real": True, "strict": True},
        lambda a, o, b: certify_solve(a["expr"], a["var"], a["roots"],
                                      real=o["real"], budget=b),
        lambda p, a, o: (
            f"{'real ' if o['real'] else ''}roots of {p['expr']} in {a['var']}: "
            f"{{{', '.join(p['roots'])}}}"
        ),
    ),
    "limit": Kind(
        (("expr", "expr"), ("var", "name"), ("to", "expr"), ("result", "expr")),
        {"dir": "+", "strict": True},
        lambda a, o, b: certify_limit(a["expr"], a["var"], a["to"], a["result"],
                                      dir=o["dir"], budget=b),
        _limit_statement,
    ),
    "ode": Kind(
        (("expr", "expr"), ("func", "name"), ("var", "name"), ("solutions", "exprs")),
        {"strict": True},
        lambda a, o, b: certify_dsolve(a["expr"], a["func"], a["var"],
                                       a["solutions"], budget=b),
        lambda p, a, o: (
            f"{a['func']}'({a['var']}) = {p['expr']} solved by "
            f"{', '.join(p['solutions'])}"
        ),
    ),
    "quadrature": Kind(
        (("expr", "expr"), ("var", "name"), ("a", "float"), ("b", "float"),
         ("value", "float")),
        {"error_estimate": None, "rtol": 1e-8, "strict": True},
        _quadrature_run,
        lambda p, a, o: (
            f"integral {p['expr']} d{a['var']} from {a['a']} to {a['b']} "
            f"== {a['value']}"
            + ("" if o["error_estimate"] is None else f" +- {o['error_estimate']}")
        ),
    ),
    "steps": Kind(
        (("derivation", "text"),),
        {"vars": None, "samples": SAMPLES, "seed": 0, "strict": True},
        _steps_run,
        lambda p, a, o: " -> ".join(p["steps"]),
    ),
}


def register_kind(name: str, kind: Kind) -> None:
    """Add a certificate kind (a later module's ``code_equiv``, ``stability``…).

    A name already taken is refused: a kind's meaning is part of every certificate
    ever issued under it, so it is never redefined in place.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise DomainError(f"certificate: a kind name must be an identifier, not {name!r}")
    if name in KINDS:
        raise DomainError(f"certificate: kind {name!r} already exists")
    for _, typ in kind.params:
        if typ not in _ARG_TYPES:
            raise DomainError(f"certificate: unknown argument type {typ!r}")
    KINDS[name] = kind


def _kind(name: object) -> Kind:
    kind = KINDS.get(name) if isinstance(name, str) else None
    if kind is None:
        raise DomainError(
            f"certificate: unknown kind {name!r} — known: {', '.join(KINDS)}"
        )
    return kind


# --- arguments as text -----------------------------------------------------------
# An argument is recorded as TEXT the parser reads back — the form a human writes
# and checks. Engine output does not always survive that trip: the parser does not
# know every function SymPy prints (``integrate exp(-x^2) dx`` answers with ``erf``,
# measured), and a decimal prints with fewer digits than it holds. Such an argument
# is recorded instead as ``{"sympy": <srepr>}``, the exact SymPy tree, which
# ``_read_sympy`` rebuilds WITHOUT ``eval``: only SymPy classes, SymPy constants
# and literals are accepted, integer literals are capped in length, and the tree is
# built unevaluated — a node SymPy refuses to build that way (``CRootOf`` needs its
# polynomial evaluated, measured) is rebuilt evaluated, that subtree only. So a
# hostile file cannot run code, and construction is not where it can hide a cost.
_SREPR = "sympy"
#: Longest integer literal a SymPy tree may carry, in digits.
_MAX_LITERAL_DIGITS = 1000


def _reject(why: str) -> DomainError:
    return DomainError(f"certificate: unreadable SymPy tree — {why}")


def _sympy_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
        if (
            isinstance(node.value, int)
            and node.value.bit_length() > _MAX_LITERAL_DIGITS * 3.33
        ):
            raise _reject(f"an integer literal longer than {_MAX_LITERAL_DIGITS} digits")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _sympy_node(node.operand)
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return -inner
        raise _reject("minus applies to numbers only")
    if isinstance(node, ast.Tuple):
        return sp.Tuple(*(_sympy_node(e) for e in node.elts))
    if isinstance(node, ast.Name):
        obj = getattr(sp, node.id, None)
        if isinstance(obj, sp.Basic):
            return obj  # a constant: pi, E, I, oo, zoo, nan...
        raise _reject(f"{node.id!r} is not a SymPy constant")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            obj = getattr(sp, node.func.id, None)
            if not (
                isinstance(obj, type)
                and (issubclass(obj, sp.Basic) or obj is sp.Function)
            ):
                raise _reject(f"{node.func.id!r} is not a SymPy class")
            fn: Any = obj
        elif isinstance(node.func, ast.Call):
            fn = _sympy_node(node.func)  # Function('f')(t)
            if not isinstance(fn, sp.core.function.UndefinedFunction):
                raise _reject("only an undefined function may be applied")
        else:
            raise _reject("unexpected call")
        args = [_sympy_node(a) for a in node.args]
        kwargs = {k.arg: _sympy_node(k.value) for k in node.keywords if k.arg}
        if len(kwargs) != len(node.keywords):
            raise _reject("unexpected ** in a call")
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — any refusal of SymPy's is a bad tree
            if not sp.core.parameters.global_parameters.evaluate:
                with sp.evaluate(True):
                    return _sympy_node(node)
            raise _reject(f"{type(exc).__name__}: {exc}") from exc
    raise _reject(f"unexpected {type(node).__name__}")


#: Size limits on a recorded SymPy tree (I1): a certificate can arrive from
#: anyone (``math_recheck`` over MCP), and a tree is cheap to WRITE and dear to
#: evaluate — ``Pow(10, Pow(10, 999))`` builds in microseconds unevaluated and
#: never finishes once a check touches it. Text longer than this, a tree deeper
#: or wider than this, or a concrete number the parser's own cost guard refuses
#: (the same guard ``parse`` applies to typed input), is refused on reading.
_MAX_TREE_TEXT = 100_000
_MAX_TREE_NODES = 20_000
_MAX_TREE_DEPTH = 100


def _tree_size_ok(tree: ast.AST) -> None:
    count = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > _MAX_TREE_NODES:
            raise _reject(f"more than {_MAX_TREE_NODES} nodes")
        if depth > _MAX_TREE_DEPTH:
            raise _reject(f"nested deeper than {_MAX_TREE_DEPTH} levels")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _read_sympy(text: str) -> sp.Expr:
    if len(text) > _MAX_TREE_TEXT:
        raise _reject(f"longer than {_MAX_TREE_TEXT} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise _reject(str(exc) or type(exc).__name__) from exc
    _tree_size_ok(tree)
    with sp.evaluate(False):
        sy = _sympy_node(tree.body)
    if not isinstance(sy, sp.Expr):
        raise _reject("not a scalar expression")
    try:
        _check_cost(sy)
    except _CostExceeded as exc:
        raise _reject(f"too costly to evaluate ({exc})") from exc
    return sy


def _expr_record(value: object, what: str) -> "str | dict[str, str]":
    if isinstance(value, Expr):
        text = str(value)
        try:
            if _parse_written(text) == value.sy:
                return text
        except PycodemathError:
            pass
        return {_SREPR: sp.srepr(value.sy)}
    if isinstance(value, dict) and set(value) == {_SREPR} and isinstance(value[_SREPR], str):
        _read_sympy(value[_SREPR])  # refuse a bad tree now, not inside the check
        return {_SREPR: value[_SREPR]}
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise DomainError(
        f"certificate: {what} must be an expression or text, not {type(value).__name__}"
    )


def _expr_value(recorded: "str | dict[str, str]") -> "str | Expr":
    """What the check is handed: the text, or the rebuilt exact tree."""
    return recorded if isinstance(recorded, str) else Expr(_read_sympy(recorded[_SREPR]))


def _expr_parsed(recorded: "str | dict[str, str]") -> str:
    if isinstance(recorded, str):
        return str(_parse_written(recorded))
    return str(_read_sympy(recorded[_SREPR]))


def _record_arg(value: object, name: str, typ: str) -> Any:
    if typ == "expr":
        return _expr_record(value, name)
    if typ == "exprs":
        if isinstance(value, (str, Expr)) or not isinstance(value, (list, tuple)):
            raise DomainError(f"certificate: {name} must be a list")
        return [_expr_record(v, f"each of {name}") for v in value]
    if typ == "name":
        if not isinstance(value, str) or not value.isidentifier():
            raise DomainError(f"certificate: {name} must be a name, not {value!r}")
        return value
    if typ == "text":
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            return list(value)
        raise DomainError(f"certificate: {name} must be text or a list of text")
    # float
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise DomainError(f"certificate: {name} must be a number, not {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError(f"certificate: {name} must be a number, not {value!r}")
    return repr(float(value))


def _values(kind: Kind, recorded: Mapping[str, Any]) -> dict[str, Any]:
    """The recorded arguments as the check takes them (see ``_expr_value``)."""
    out = dict(recorded)
    for name, typ in kind.params:
        if typ == "expr":
            out[name] = _expr_value(recorded[name])
        elif typ == "exprs":
            out[name] = [_expr_value(v) for v in recorded[name]]
    return out


def _record_args(kind_name: str, kind: Kind, args: Sequence[object]) -> dict[str, Any]:
    if len(args) != len(kind.params):
        names = ", ".join(n for n, _ in kind.params)
        raise DomainError(
            f"certificate {kind_name}: takes {len(kind.params)} argument(s) "
            f"({names}), got {len(args)}"
        )
    return {n: _record_arg(v, n, t) for (n, t), v in zip(kind.params, args)}


def _record_options(kind_name: str, kind: Kind, options: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(options) - set(kind.options))
    if unknown:
        raise DomainError(
            f"certificate {kind_name}: unknown option(s) {', '.join(unknown)}"
        )
    out = {**kind.options, **options}
    if "strict" in kind.options and "strict" not in options:
        # The parsing mode is part of the claim (I1): ``speed`` is one variable in
        # strict mode and s*p*e*e*d with ``implicit on``, so a certificate issued
        # in either mode must be re-read in the same one. Unless the caller says
        # otherwise, the mode is the one in force while the claim is recorded.
        out["strict"] = is_strict()
    if "strict" in out and not isinstance(out["strict"], bool):
        raise DomainError(f"certificate {kind_name}: option strict must be true or false")
    if out.get("vars") is not None:
        v = out["vars"]
        out["vars"] = v.replace(",", " ").split() if isinstance(v, str) else list(v)
    _canonical(out)  # refuses, early, an option JSON cannot carry
    return out


def _in_mode(opts: Mapping[str, Any]) -> Any:
    """The parsing mode a claim was recorded in (``nullcontext`` for a kind
    that parses nothing)."""
    if "strict" in opts:
        return parse_mode(strict=bool(opts["strict"]))
    return contextlib.nullcontext()


def _parsed(kind: Kind, args: Mapping[str, Any], result: "Verdict | StepsResult") -> dict:
    """Each expression argument as SymPy prints it after parsing — what was CHECKED."""
    if isinstance(result, StepsResult):
        return {"steps": list(result.steps)}
    out: dict[str, Any] = {}
    for name, typ in kind.params:
        if typ == "expr":
            out[name] = _expr_parsed(args[name])
        elif typ == "exprs":
            out[name] = [_expr_parsed(v) for v in args[name]]
    return out


def _verdict_record(result: "Verdict | StepsResult") -> dict[str, Any]:
    if isinstance(result, StepsResult):
        failed = result.failed
        if failed is not None:
            detail = f"step {failed.number}: {failed.verdict.detail}"
        else:
            words = ", ".join(f"{c.number}: {c.verdict.status.value}" for c in result.checks)
            detail = f"no step refuted ({words})" if words else "no step to check"
        return {
            "status": result.status.value,
            "method": "steps",
            "counterexample": result.counterexample,
            "detail": detail,
            "first_error": result.first_error,
        }
    if not isinstance(result, Verdict):
        raise DomainError(
            f"certificate: expected a Verdict or StepsResult, not {type(result).__name__}"
        )
    return {
        "status": result.status.value,
        "method": result.method,
        "counterexample": result.counterexample,
        "detail": result.detail,
    }


def _budget(budget: "float | None") -> float:
    if budget is None:
        return float(DEFAULT_TIME_BUDGET)
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not 0 < budget < math.inf
    ):
        raise DomainError(f"certificate: budget must be finite and positive, not {budget!r}")
    return float(budget)


# --- the certificate ---------------------------------------------------------
@dataclass(frozen=True)
class Certificate:
    """One check, written down. Build with ``issue`` / ``from_verdict``; read with
    ``from_json`` / ``from_dict`` / ``load``. See the module docstring for the
    fields; ``digest`` is the recorded one — ``digest_ok`` says whether it still
    matches the content."""

    schema: int
    kind: str
    claim: "dict[str, Any]"
    source: "str | None"
    verdict: "dict[str, Any]"
    budget: float
    versions: "dict[str, str]"
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps({k: getattr(self, k) for k in _KEYS}))

    def to_json(self, *, indent: "int | None" = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=False)

    def save(self, path: "str | Path") -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @property
    def digest_ok(self) -> bool:
        return content_digest(self.to_dict()) == self.digest

    @classmethod
    def from_json(cls, text: str) -> "Certificate":
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise DomainError(f"certificate: not valid JSON ({exc})") from exc
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: "str | Path") -> "Certificate":
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise DomainError(f"certificate: cannot read {str(path)!r} ({exc.strerror})") from exc
        return cls.from_json(text)

    @classmethod
    def from_dict(cls, data: object) -> "Certificate":
        """Read a certificate, checking its SHAPE (not its truth — that is
        ``recheck``). Anything that is not a schema-1 certificate is a
        ``DomainError`` naming what is wrong."""
        def bad(why: str) -> DomainError:
            return DomainError(f"certificate: {why}")

        if not isinstance(data, dict):
            raise bad("expected a JSON object")
        missing = [k for k in _KEYS if k not in data]
        extra = sorted(set(data) - set(_KEYS))
        if missing or extra:
            raise bad(
                "wrong fields"
                + (f"; missing {', '.join(missing)}" if missing else "")
                + (f"; unexpected {', '.join(extra)}" if extra else "")
            )
        if data["schema"] != SCHEMA_VERSION or isinstance(data["schema"], bool):
            raise bad(f"schema {data['schema']!r} is not supported (this reader: {SCHEMA_VERSION})")
        kind = _kind(data["kind"])
        claim = data["claim"]
        if not isinstance(claim, dict) or set(claim) != set(_CLAIM_KEYS):
            raise bad(f"claim must hold exactly {', '.join(_CLAIM_KEYS)}")
        if not isinstance(claim["args"], dict) or set(claim["args"]) != {n for n, _ in kind.params}:
            raise bad(f"claim.args must name {', '.join(n for n, _ in kind.params)}")
        if not isinstance(claim["options"], dict) or set(claim["options"]) != set(kind.options):
            raise bad(f"claim.options must name {', '.join(kind.options) or 'nothing'}")
        if not isinstance(claim["parsed"], dict) or not isinstance(claim["statement"], str):
            raise bad("claim.parsed must be an object and claim.statement text")
        if data["source"] is not None and not isinstance(data["source"], str):
            raise bad("source must be text or null")
        verdict = data["verdict"]
        if not isinstance(verdict, dict) or verdict.get("status") not in _STATUSES:
            raise bad("verdict.status must be verified, refuted or undecided")
        point = verdict.get("counterexample")
        if (verdict["status"] == "refuted") != (point is not None):
            raise bad("verdict.counterexample must be set exactly when refuted")
        if point is not None and not (
            isinstance(point, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in point.items())
        ):
            raise bad("verdict.counterexample must map names to exact values as text")
        budget = data["budget"]
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget < math.inf:
            raise bad("budget must be a finite positive number of seconds")
        vers = data["versions"]
        if not isinstance(vers, dict) or not all(isinstance(v, str) for v in vers.values()):
            raise bad("versions must map names to version text")
        digest = data["digest"]
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise bad("digest must be 'sha256:<hex>'")
        return cls(
            schema=SCHEMA_VERSION, kind=data["kind"], claim=claim, source=data["source"],
            verdict=verdict, budget=budget, versions=vers, digest=digest,
        )


def _seal(kind_name: str, claim: dict, source: "str | None", verdict: dict,
          budget: float) -> Certificate:
    content = {
        "schema": SCHEMA_VERSION, "kind": kind_name, "claim": claim, "source": source,
        "verdict": verdict, "budget": budget, "versions": versions(),
    }
    content = json.loads(json.dumps(content))  # what the file will hold, exactly
    return Certificate(**content, digest=content_digest(content))


def from_verdict(
    kind: str,
    args: Sequence[object],
    result: "Verdict | StepsResult",
    *,
    source: "str | None" = None,
    budget: "float | None" = None,
    **options: Any,
) -> Certificate:
    """Export a check ALREADY RUN as a certificate — nothing is re-run.

    ``args`` are the check's positional arguments in ``KINDS[kind].params`` order,
    ``result`` the ``Verdict`` (or, for ``steps``, the ``StepsResult``) it gave,
    ``budget`` and ``options`` what it ran with. Expression arguments are stored
    as text (an ``Expr`` as it prints), so ``recheck`` re-reads them through the
    parser: a result that does not survive that round trip shows up there.
    """
    k = _kind(kind)
    recorded = _record_args(kind, k, args)
    opts = _record_options(kind, k, options)
    with _in_mode(opts):
        parsed = _parsed(k, recorded, result)
    claim = {
        "args": recorded, "options": opts, "parsed": parsed,
        "statement": k.statement(parsed, recorded, opts),
    }
    return _seal(kind, claim, source, _verdict_record(result), _budget(budget))


def issue(
    kind: str,
    *args: object,
    source: "str | None" = None,
    budget: "float | None" = None,
    **options: Any,
) -> Certificate:
    """Run the check of ``kind`` on ``args`` and return its certificate.

        >>> issue("identity", "sqrt(x^2)", "x").verdict["counterexample"]
        {'x': '-1'}

    The check runs on the arguments AS RECORDED (text), so the certificate holds
    exactly the verdict ``recheck`` will reproduce. Kinds and their arguments:
    ``KINDS`` (``identity(left, right)``, ``derivative(expr, var, result)``,
    ``integral(expr, var, result)``, ``roots(expr, var, roots)``,
    ``limit(expr, var, to, result)``, ``ode(expr, func, var, solutions)``,
    ``quadrature(expr, var, a, b, value)``, ``steps(derivation)``).
    """
    k = _kind(kind)
    recorded = _record_args(kind, k, args)
    opts = _record_options(kind, k, options)
    total = _budget(budget)
    with _in_mode(opts):
        result = k.run(_values(k, recorded), opts, total)
    return from_verdict(kind, [recorded[n] for n, _ in k.params], result,
                        source=source, budget=total, **opts)


# --- re-checking ---------------------------------------------------------------
_COMPARED = ("status", "method", "counterexample", "first_error")


@dataclass(frozen=True)
class Recheck:
    """What re-running a certificate found.

    ``verdict`` is the re-run's verdict record, ``parsed`` the claim as it parses
    today. ``digest_ok`` (content matches the recorded digest), ``claim_ok`` (the
    claim parses as recorded) and ``reproduced`` (same status, method,
    counterexample — and first wrong step for ``steps``) are the three findings;
    ``ok`` is all three. ``problems`` says what failed, ``notes`` what differs
    without failing (versions, the wording of ``detail``).
    """

    certificate: Certificate
    verdict: "dict[str, Any]"
    parsed: "dict[str, Any]"
    digest_ok: bool
    claim_ok: bool
    reproduced: bool
    problems: "tuple[str, ...]"
    notes: "tuple[str, ...]"

    @property
    def ok(self) -> bool:
        return self.digest_ok and self.claim_ok and self.reproduced

    def text(self) -> str:
        c, v = self.certificate, self.verdict
        head = "REPRODUCED" if self.ok else "MISMATCH"
        lines = [f"{head} — {c.kind}: {c.claim['statement']}"]
        where = ""
        if v.get("counterexample"):
            where = " at " + ", ".join(f"{k} = {x}" for k, x in v["counterexample"].items())
        lines.append(
            f"  recorded: {c.verdict['status']} ({c.verdict['method']}); "
            f"re-run: {v['status']} ({v['method']}){where}"
        )
        lines.append(f"  digest: {'ok' if self.digest_ok else 'MISMATCH'}")
        lines += [f"  problem: {p}" for p in self.problems]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _show(v: Any) -> str:
    return "none" if v is None else json.dumps(v, ensure_ascii=False)


def recheck(
    certificate: "Certificate | Mapping[str, Any] | str",
    *,
    budget: "float | None" = None,
) -> Recheck:
    """Re-run a certificate's check from the record alone and compare.

    ``certificate`` is a ``Certificate``, its dict, or its JSON text (to read a
    file, ``Certificate.load(path)``). ``budget`` replaces the recorded one — for
    a re-run that ran out of time on a slower machine. A certificate that is not
    one (wrong shape, unknown kind or schema) is a ``DomainError``; a certificate
    that does not REPRODUCE is not an error but a finding, in the result.
    """
    if isinstance(certificate, Certificate):
        cert = certificate
    elif isinstance(certificate, str):
        cert = Certificate.from_json(certificate)
    else:
        cert = Certificate.from_dict(dict(certificate))
    kind = _kind(cert.kind)
    args, opts = cert.claim["args"], cert.claim["options"]
    problems: list[str] = []
    notes: list[str] = []

    digest_ok = cert.digest_ok
    if not digest_ok:
        problems.append(
            "digest does not match the content — the certificate was edited after "
            "it was issued"
        )

    # The arguments go back through the recording rules, so a hand-edited type
    # (a number where text belongs) is a typed refusal, not a crash in the check.
    recorded = _record_args(cert.kind, kind, [args[n] for n, _ in kind.params])
    opts = _record_options(cert.kind, kind, opts)
    total = _budget(cert.budget if budget is None else budget)
    with _in_mode(opts):
        result = kind.run(_values(kind, recorded), opts, total)
        parsed = _parsed(kind, recorded, result)
    verdict = _verdict_record(result)

    claim_ok = parsed == cert.claim["parsed"]
    if not claim_ok:
        problems.append(
            f"the claim parses differently now: recorded {_show(cert.claim['parsed'])}, "
            f"now {_show(parsed)}"
        )

    differ = [
        k for k in _COMPARED
        if (k in verdict or k in cert.verdict) and verdict.get(k) != cert.verdict.get(k)
    ]
    reproduced = not differ
    for k in differ:
        problems.append(
            f"{k}: recorded {_show(cert.verdict.get(k))}, re-run {_show(verdict.get(k))}"
        )
    if differ and verdict["method"] == "time-budget":
        problems.append(
            f"the re-run ran out of its {total:g}s budget — re-check with a larger budget"
        )
    if reproduced and verdict["detail"] != cert.verdict.get("detail"):
        notes.append("the verdict's detail is worded differently")

    for name, now in versions().items():
        then = cert.versions.get(name)
        if then != now:
            notes.append(f"{name} {then or '(not recorded)'} -> {now}")
    return Recheck(cert, verdict, parsed, digest_ok, claim_ok, reproduced,
                   tuple(problems), tuple(notes))


# --- the FULL-series kinds (I1) -------------------------------------------------
# One kind per module of the series, each re-checked by running the operation
# again AND comparing the recorded result with what it returns now — a hand-edited
# number is caught twice: by the digest, and by the recomputation (``recomputation``
# REFUTED, the recomputed value as the counterexample). Where a cheap route
# independent of the module exists, the re-check takes it too:
#
#   code_equiv       check_code again (a refutation is re-EXECUTED in the isolated
#                    worker); ``execute=False`` claims are re-checked without running
#   stability        the worst input is re-evaluated by the NumPy code C2 emits and
#                    the exact value recomputed by SymPy at 40 digits; the bits lost
#                    are re-counted in ULPs
#   range            analyze again; an integer witness is re-executed in NumPy and
#                    compared with Python's exact integers
#   money / calc / geometry / unit_conversion / dimensions
#                    the operation again, with its own built-in independent check
#                    (Bezout, product of factors, decimal.quantize, a second route at
#                    50 digits...); allocate: the shares re-added in Fractions
#
# Arguments are TEXT: the operation's inputs and its result as canonical JSON, so
# the digest is stable and nothing in the file is ever evaluated as code.

def _jtext(value: Any) -> str:
    """Canonical JSON text of ``value`` (sorted keys, no spaces)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _jload(text: str, what: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise DomainError(f"certificate: {what} is not valid JSON ({exc})") from exc


def _strip_volatile(value: Any) -> Any:
    """The result without wall-clock fields (``elapsed``): they never reproduce."""
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items() if k != "elapsed"}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def _same(a: Any, b: Any) -> bool:
    """JSON equality, floats within 1e-12 relative (an ulp may move between
    platforms; a hand-edited digit does not hide inside 1e-12)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-300) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return bool(a == b)


def _short(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _against_record(recorded_text: str, value: Any, verdict: Verdict) -> Verdict:
    """``verdict`` when the operation still returns the recorded result, else a
    REFUTED whose counterexample is the recomputed result."""
    recorded = _jload(recorded_text, "result")
    if _same(recorded, value):
        return verdict
    now = _jtext(value)
    return Verdict(
        VerdictStatus.REFUTED, "recomputation", {"result": now},
        f"the recorded result is not what the operation returns: recorded "
        f"{_short(recorded_text)}, recomputed {_short(now)}",
    )


#: One operation of an ``op`` kind: inputs -> (the result as JSON, its verdict).
_Op = Callable[[dict], "tuple[Any, Verdict]"]


def _op_kind(name: str, ops: Mapping[str, _Op], noun: str) -> Kind:
    def run(a: dict, o: dict, budget: float) -> Verdict:
        value, verdict = run_op(name, a["op"], _jload(a["inputs"], "inputs"), budget)
        return _against_record(a["result"], value, verdict)

    return Kind(
        (("op", "name"), ("inputs", "text"), ("result", "text")),
        {},
        run,
        lambda p, a, o: f"{noun} {a['op']}({_short(a['inputs'], 200)}) == {_short(a['result'], 200)}",
    )


_OP_TABLES: dict[str, Mapping[str, _Op]] = {}


def run_op(kind: str, op: str, inputs: Any, budget: "float | None" = None) -> "tuple[Any, Verdict]":
    """Run one operation of an ``op`` kind (``money``, ``calc``, ``geometry``) on
    JSON ``inputs`` and return ``(result as JSON, verdict)`` — the pair a
    certificate records and ``recheck`` compares."""
    from ..core.budget import time_budget

    table = _OP_TABLES[kind]
    fn = table.get(op) if isinstance(op, str) else None
    if fn is None:
        raise DomainError(f"certificate {kind}: unknown operation {op!r} — known: {', '.join(table)}")
    if not isinstance(inputs, dict):
        raise DomainError(f"certificate {kind}: inputs must be a JSON object")
    with time_budget(_budget(budget)):
        value, verdict = fn(inputs)
    return _strip_volatile(json.loads(json.dumps(value))), verdict


def certify_op(kind: str, op: str, inputs: dict, *, source: "str | None" = None,
               budget: "float | None" = None) -> "tuple[Any, Certificate]":
    """Run ``op`` once and write it down: ``(result as JSON, certificate)``."""
    value, verdict = run_op(kind, op, inputs, budget)
    cert = from_verdict(kind, [op, _jtext(inputs), _jtext(value)], verdict,
                        source=source, budget=budget)
    return value, cert


def _verified(method: str, detail: str) -> Verdict:
    return Verdict(VerdictStatus.VERIFIED, method, None, detail)


def _status_verdict(status: str, method: str, detail: str,
                    counterexample: "dict[str, str] | None" = None) -> Verdict:
    st = VerdictStatus(status)
    if st is VerdictStatus.REFUTED:
        return Verdict(st, method, counterexample or {}, detail)
    return Verdict(st, method, None, detail)


# --- calc (K3) -------------------------------------------------------------------
def _calc_op(name: str) -> _Op:
    def op(inputs: dict) -> "tuple[Any, Verdict]":
        from ..engine import calc, stats

        fn = getattr(calc, name, None) or getattr(stats, name)
        res = fn(*inputs.get("args", []), **inputs.get("kwargs", {}))
        value = res.to_dict()
        extra = _calc_extra(name, inputs, res)
        if res.proof == "probabilistic":
            return value, Verdict(VerdictStatus.UNDECIDED, "probabilistic", None, res.certificate)
        return value, _verified(res.proof, res.certificate + extra)
    return op


def _calc_extra(name: str, inputs: dict, res: Any) -> str:
    """Independent re-checks the certificate layer can afford on its own."""
    from ..frontend.parser import parse

    def integer(text: object) -> int:
        return int(parse(str(text)).sy)

    if name == "factorint" and isinstance(res.value, dict):
        n = integer(inputs["args"][0])
        prod = 1
        for p, e in res.value.items():
            prod *= int(p) ** int(e)
        if prod != n:  # K3 carries the sign as a factor -1
            raise DomainError(f"certificate calc: the factors multiply to {prod}, not {n}")
        return "; re-multiplied: the factors give the number back"
    if name == "gcd" and "bezout" in res.witness:
        args = [integer(a) for a in inputs["args"]]
        if sum(a * s for a, s in zip(args, res.witness["bezout"])) != res.value:
            raise DomainError("certificate calc: the Bezout coefficients do not give the gcd")
        return "; Bezout identity re-added"
    return ""


#: Operations of the calculator reachable through a certificate (K3's tables).
CALC_OPS = (
    "isprime", "factorint", "nextprime", "prevprime", "gcd", "lcm", "mod", "powmod",
    "modinv", "totient", "divisors", "crt", "sqrt_mod", "fibonacci", "lucas", "legendre",
    "jacobi", "binomial", "multinomial", "permutations", "combinations", "stirling1",
    "stirling2", "bell", "catalan", "partitions", "special", "primality_certificate",
    "mean", "median", "quantile", "variance", "std", "covariance", "correlation",
    "pdf", "pmf", "cdf", "sf", "ppf", "ab_sample_size", "ab_power", "wilson_interval",
    "clopper_pearson", "two_proportion_ztest", "welch_ttest", "adjust_pvalues",
)
_OP_TABLES["calc"] = {name: _calc_op(name) for name in CALC_OPS}


# --- money (C4) ------------------------------------------------------------------
def _money_checks(res: Any) -> Verdict:
    checks = "; ".join(getattr(res, "checks", ()) or ()) or "exact arithmetic"
    return _verified("exact", checks)


def _money_round(i: dict) -> "tuple[Any, Verdict]":
    from ..code import money
    res = money.round_money(i["value"], i["currency"], i.get("mode", "HALF_UP"), step=i.get("step"))
    return res.to_dict(), _money_checks(res)


def _money_allocate(i: dict) -> "tuple[Any, Verdict]":
    from fractions import Fraction

    from ..code import money
    res = money.allocate(i["total"], i["parts"], i["currency"], method=i.get("method", "hamilton"))
    shares = [Fraction(str(s)) for s in res.shares]
    if sum(shares) != Fraction(str(res.total)):
        raise DomainError("certificate money: the shares do not add up to the total")
    unit = Fraction(1, 10 ** money.currency_exponent(i["currency"]))
    if res.method == "hamilton" and any(abs(s - q) >= unit for s, q in zip(shares, res.quotas)):
        raise DomainError("certificate money: a share is a whole unit away from its quota")
    v = _money_checks(res)
    return res.to_dict(), _verified(v.method, v.detail + "; shares re-added in exact fractions")


def _money_vat(i: dict) -> "tuple[Any, Verdict]":
    from ..code import money
    res = money.vat_per_line_vs_total(i["lines"], i["rate"], i["currency"], i.get("mode", "HALF_UP"))
    return res.to_dict(), _money_checks(res)


def _money_float(i: dict) -> "tuple[Any, Verdict]":
    from ..code import money
    grid = {v: money.money_grid(lo, hi, step) for v, (lo, hi, step) in i["grid"].items()}
    res = money.float_money_hazard(i["expr"], variables=grid, currency=i.get("currency"),
                                   mode=i.get("mode", "HALF_EVEN"))
    status = res.verdict
    detail = res.explanation
    if status is VerdictStatus.REFUTED:
        return res.to_dict(), Verdict(status, "execution", dict(res.witness or {}), detail)
    return res.to_dict(), Verdict(status, "execution", None, detail)


def _money_interest(i: dict) -> "tuple[Any, Verdict]":
    from ..code import money
    res = money.compound_interest(i["principal"], i["rate"], i["periods"], i["currency"],
                                  i.get("mode", "HALF_UP"), periods_per_year=i.get("per_year", 1))
    return res.to_dict(), _money_checks(res)


def _money_loan(i: dict) -> "tuple[Any, Verdict]":
    from ..code import money
    res = money.amortization_schedule(i["principal"], i["rate"], i["periods"], i["currency"],
                                      i.get("mode", "HALF_UP"), periods_per_year=i.get("per_year", 1),
                                      schedule=i.get("schedule", "annuity"))
    return res.to_dict(), _money_checks(res)


def _money_npv(i: dict) -> "tuple[Any, Verdict]":
    from ..code import money
    res = money.npv(i["rate"], i["cashflows"], currency=i.get("currency"))
    return res.to_dict(), _money_checks(res)


def _money_irr(i: dict) -> "tuple[Any, Verdict]":
    from ..code import money
    res = money.irr(i["cashflows"])
    return res.to_dict(), _money_checks(res)


_OP_TABLES["money"] = {
    "round": _money_round, "allocate": _money_allocate, "vat": _money_vat,
    "float": _money_float, "interest": _money_interest, "loan": _money_loan,
    "npv": _money_npv, "irr": _money_irr,
}


# --- geometry (G1) ---------------------------------------------------------------
def _geo_cert(res: Any) -> Verdict:
    c = getattr(res, "certificate", None) or {}
    status = c.get("status", "verified")
    facts = "; ".join(f"{k} {_jtext(v)}" for k, v in c.items() if k not in ("status", "method"))
    return _status_verdict(status, str(c.get("method", "exact")), _short(facts or status, 400))


def _geo(fn_name: str, call: Callable[[Any, dict], Any]) -> _Op:
    def op(i: dict) -> "tuple[Any, Verdict]":
        from ..code import geometry
        res = call(geometry, i)
        if fn_name == "rotcheck":
            witness = {k: _jtext(v) for k, v in (res.witness or {}).items()}
            return res.to_dict(), _status_verdict(res.status, res.method,
                                                  "; ".join(res.explanations) or res.status, witness)
        if fn_name == "epsilon":
            first = {"input": _jtext(res.witnesses[0]["input"])} if res.witnesses else {}
            return res.to_dict(), _status_verdict(res.status, "exact-witness", res.detail, first)
        if fn_name in ("orient2d", "orient3d", "incircle"):
            return res.to_dict(), _verified(res.stage, f"sign {res.sign} ({res.stage})")
        return res.to_dict(), _geo_cert(res)
    return op


def _geo_quat(i: dict) -> Any:
    from ..code import geometry as g
    return g.QuatConvention(order=i.get("order", "wxyz"), algebra=i.get("algebra", "hamilton"))


_OP_TABLES["geometry"] = {
    "quat2mat": _geo("quat2mat", lambda g, i: g.quat_to_matrix(
        i["q"], quat=_geo_quat(i), normalize=i.get("normalize", False))),
    "euler2mat": _geo("euler2mat", lambda g, i: g.euler_to_matrix(
        i["angles"], i["seq"], frame=i["frame"], degrees=i.get("degrees", False))),
    "mat2euler": _geo("mat2euler", lambda g, i: g.matrix_to_euler(
        i["matrix"], i["seq"], frame=i["frame"], degrees=i.get("degrees", False))),
    "rotcheck": _geo("rotcheck", lambda g, i: g.compare_rotation_expressions(i["a"], i["b"])),
    "orient2d": _geo("orient2d", lambda g, i: g.classify_orient2d(*i["points"])),
    "orient3d": _geo("orient3d", lambda g, i: g.classify_orient3d(*i["points"])),
    "incircle": _geo("incircle", lambda g, i: g.classify_incircle(*i["points"])),
    "segment": _geo("segment", lambda g, i: g.segment_intersection(*i["points"])),
    "easing": _geo("easing", lambda g, i: g.css_easing(i["spec"], i["x"])),
    "arclength": _geo("arclength", lambda g, i: g.bezier_arc_length(i["control"])),
    "emit": _geo("emit", lambda g, i: g.emit(i["formula"], i["language"])),
    "epsilon": _geo("epsilon", lambda g, i: g.epsilon_witness(i["test"], float(i["epsilon"]))),
}

KINDS["calc"] = _op_kind("calc", _OP_TABLES["calc"], "calc")
KINDS["money"] = _op_kind("money", _OP_TABLES["money"], "money")
KINDS["geometry"] = _op_kind("geometry", _OP_TABLES["geometry"], "geometry")


# --- units (J1) ------------------------------------------------------------------
def _dimensions_run(a: dict, o: dict, budget: float) -> Verdict:
    from ..engine import units as U
    return U.check_dimensions(a["equation"], units=U.read_units_spec(a["units"]),
                              values=o["values"]).as_verdict()


def _conversion_run(a: dict, o: dict, budget: float) -> Verdict:
    from ..engine import units as U
    conv = U.convert(a["quantity"], a["to"])
    return _against_record(_jtext(a["value"]), conv.value_text, conv.as_verdict())


KINDS["dimensions"] = Kind(
    (("equation", "text"), ("units", "text")),
    {"values": "quantities", "strict": True},
    _dimensions_run,
    lambda p, a, o: f"{a['equation']} is dimensionally consistent given {a['units']}",
)
KINDS["unit_conversion"] = Kind(
    (("quantity", "text"), ("to", "text"), ("value", "text")),
    {},
    _conversion_run,
    lambda p, a, o: f"{a['quantity']} == {a['value']} {a['to']}",
)


# --- code_equiv (C1) -------------------------------------------------------------
import threading as _threading  # noqa: E402 — kept with the one thing that needs it

#: check_code runs one at a time while execution is being switched off, so a
#: concurrent call (the MCP server is threaded) never sees the other's setting.
_EXECUTION_LOCK = _threading.RLock()


@contextlib.contextmanager
def code_execution(allowed: bool) -> Any:
    """Inside the block ``check_code`` runs the function (``allowed``) or never
    does: source that came from outside — an agent over MCP — is read and proved,
    and a refutation it would need to EXECUTE stays UNDECIDED with the reason.

    An adapter, not a change of C1 (I1 may not touch module logic): C1's own
    "is this module safe to run" hook is pointed at a refusal for the duration.
    """
    from ..code import frontend

    with _EXECUTION_LOCK:
        if allowed:
            yield
            return
        # C5 (``loops``, and ``recurrence`` through it) imported the hook BY NAME, so
        # it is switched off there as well (I2): one switch for every worker launch.
        import sys as _sys

        holders: list[Any] = [frontend] + [m for m in (_sys.modules.get("pycodemath.code.loops"),)
                                           if m is not None and hasattr(m, "_execution_blocker")]
        originals = [h._execution_blocker for h in holders]
        for h in holders:
            h._execution_blocker = lambda src: (
                "is not executed: execution is off for source from outside "
                "(execute=true runs it in the isolated worker)"
            )
        try:
            yield
        finally:
            for h, original in zip(holders, originals):
                h._execution_blocker = original


def domain_text(domain: "Mapping[str, str] | None") -> str:
    """``{"n": "integer, n>=0"}`` -> ``"n: integer, n>=0"`` (``;`` between)."""
    return "; ".join(f"{k}: {v}" for k, v in (domain or {}).items())


def domain_spec(text: str) -> "dict[str, str]":
    """The inverse of ``domain_text``."""
    out: dict[str, str] = {}
    for part in (p.strip() for p in text.split(";")):
        if not part:
            continue
        name, sep, spec = part.partition(":")
        if not sep or not name.strip().isidentifier():
            raise DomainError(f"domain: expected 'name: spec', got {part!r}")
        out[name.strip()] = spec.strip()
    return out


def code_verdict(result: Any) -> Verdict:
    """A ``CodeCheck`` as a ``Verdict``: the counterexample is the exact call."""
    cex = result.counterexample
    point = None if cex is None else dict(cex.args_exact)
    return Verdict(VerdictStatus(result.verdict), result.method, point, result.reason)


def _code_equiv_run(a: dict, o: dict, budget: float) -> Verdict:
    from ..code import frontend
    with code_execution(bool(o["execute"])):
        result = frontend.check_code(a["source"], a["formula"], name=a["function"],
                                     domain=domain_spec(a["domain"]) or None, budget=budget)
    return code_verdict(result)


KINDS["code_equiv"] = Kind(
    (("source", "text"), ("formula", "text"), ("function", "name"), ("domain", "text")),
    {"execute": True},
    _code_equiv_run,
    lambda p, a, o: f"{a['function']}(...) computes {a['formula']}"
    + (f" for {a['domain']}" if a["domain"] else ""),
)


# --- stability (C2) --------------------------------------------------------------
def _ordinal(x: float, dtype: Any) -> int:
    """Position of ``x`` on the line of floats of ``dtype`` (0 at +-0)."""
    import numpy as np

    width = np.dtype(dtype).itemsize * 8
    as_int = {32: np.int32, 64: np.int64}[width]
    bits = int(np.array([x], dtype=dtype).view(as_int)[0])
    return bits if bits >= 0 else -(bits & ((1 << (width - 1)) - 1))


def _stability_run(a: dict, o: dict, budget: float) -> Verdict:
    import numpy as np

    from ..code import stability as S

    node = S.parse_formula(a["expr"])
    fmt = S._format(a["dtype"])
    point = _jload(a["point"], "point")
    names = list(node.variables)
    ns: dict[str, Any] = {}
    exec(S._emit(node, "f", "numpy", fmt, names), ns)  # noqa: S102 — code WE generated
    args = [np.array([float(point[v])], dtype=fmt.dtype) for v in names]
    computed = repr(float(ns["f"](*args)[0]))
    formula = S.to_sympy(node)
    exact_sy = formula.xreplace(
        {sym: sp.Rational(float(point[sym.name])) for sym in formula.free_symbols}
    )
    exact = sp.N(exact_sy, 40)
    problems = []
    if computed != a["computed"]:
        problems.append(f"the emitted NumPy code computes {computed}, recorded {a['computed']}")
    recorded_exact = sp.Float(a["exact"], 40)
    if not (exact == recorded_exact or abs(exact - recorded_exact) <= abs(exact) * sp.Float("1e-15")):
        problems.append(f"the exact value is {sp.N(exact, 20)}, recorded {a['exact']}")
    rounded = float(exact) if fmt.dtype == np.float64 else float(np.array([float(exact)], fmt.dtype)[0])
    ulps = abs(_ordinal(float(computed), fmt.dtype) - _ordinal(rounded, fmt.dtype))
    bits = math.log2(1 + ulps)
    if abs(bits - float(a["bits"])) > 0.01:
        problems.append(f"{bits:.3f} bits lost, recorded {a['bits']}")
    where = ", ".join(f"{k} = {v}" for k, v in point.items())
    seen = (f"at {where}: {a['dtype']} computes {computed} (emitted NumPy code), exact "
            f"{sp.N(exact, 20)} (SymPy, 40 digits): {bits:.1f} bits lost")
    method = "re-evaluation" if not problems else "re-evaluation-mismatch"
    if problems:
        seen += " — the record differs: " + "; ".join(problems)
    if bits > STABILITY_BITS:
        return Verdict(VerdictStatus.REFUTED, method, {k: str(v) for k, v in point.items()},
                       f"{seen} (more than {STABILITY_BITS:g})")
    return Verdict(VerdictStatus.UNDECIDED, "sampling" if not problems else method, None,
                   f"no sampled input loses more than {STABILITY_BITS:g} bits; the worst {seen} — "
                   f"sampling is not a proof")


#: The ``stability`` claim: the formula loses at most this many bits anywhere in
#: its ranges (C2's own "accurate" bar for the worst input). REFUTED by the worst
#: input, re-evaluated independently; never VERIFIED — no sample proves accuracy.
STABILITY_BITS = 3.0


KINDS["stability"] = Kind(
    (("expr", "text"), ("dtype", "name"), ("point", "text"), ("computed", "text"),
     ("exact", "text"), ("bits", "text")),
    {},
    _stability_run,
    lambda p, a, o: (
        f"{a['expr']} loses at most {STABILITY_BITS:g} bits in {a['dtype']} (worst sampled: "
        f"{a['point']} computes {a['computed']}, exact {a['exact']}, {a['bits']} bits)"
    ),
)


def stability_args(report: Any) -> "list[str] | None":
    """The ``stability`` certificate arguments of a ``StabilityReport``'s worst
    input, or ``None`` when no input was measured."""
    w = report.worst
    if w is None:
        return None
    return [report.expression, report.dtype, _jtext(w.inputs), w.computed, w.exact,
            f"{w.bits:.6f}"]


# --- range (C3) ------------------------------------------------------------------
def range_verdict(report: Any) -> Verdict:
    """A ``RangeReport`` as a ``Verdict`` on the claim "no event in this dtype
    over these ranges": REFUTED by the first confirmed event's witness."""
    confirmed = report.confirmed
    if confirmed:
        e = confirmed[0]
        others = f" (+{len(confirmed) - 1} more confirmed)" if len(confirmed) > 1 else ""
        return Verdict(VerdictStatus.REFUTED, "execution", dict(e.witness or {}),
                       f"{e.kind} at {e.node_text}: {e.detail}{others}")
    if report.possible:
        kinds = sorted({f"{e.kind} at {e.node_text}" for e in report.possible})
        return Verdict(VerdictStatus.UNDECIDED, "interval", None,
                       "not ruled out: " + "; ".join(kinds))
    return _verified("proof", f"no event in {report.dtype} over the given ranges")


def _int_reexecuted(report: Any) -> "str | None":
    """Re-execute an INTEGER witness independently of C3: NumPy in the dtype vs
    Python's exact ints. ``None`` when not applicable; text when it disagrees."""
    import numpy as np

    e = report.confirmed[0]
    dt = np.dtype(report.dtype) if report.dtype.startswith(("int", "uint")) else None
    if dt is None or e.kind != "overflow" or e.op not in ("add", "sub", "mul"):
        return None
    try:
        xs = [int(v) for v in e.operands]
    except ValueError:
        return None
    exact = {"add": xs[0] + xs[1], "sub": xs[0] - xs[1], "mul": xs[0] * xs[1]}[e.op]
    with np.errstate(all="ignore"):
        a, b = np.array([xs[0]], dt), np.array([xs[1]], dt)
        got = int({"add": a + b, "sub": a - b, "mul": a * b}[e.op][0])
    if got == exact:
        return f"re-executed {e.node_text} in {dt}: no overflow ({got})"
    return ""


def _range_run(a: dict, o: dict, budget: float) -> Verdict:
    from ..code import ranges as R

    spec = _jload(a["ranges"], "ranges")
    report = R.analyze(a["expr"], spec, a["dtype"], budget=budget, advice=False,
                       use_z3=bool(o["use_z3"]))
    verdict = range_verdict(report)
    if verdict.refuted:
        disagreement = _int_reexecuted(report)
        if disagreement:
            return Verdict(VerdictStatus.UNDECIDED, "re-execution", None,
                           f"the witness did not reproduce: {disagreement}")
    return verdict


KINDS["range"] = Kind(
    (("expr", "text"), ("ranges", "text"), ("dtype", "name")),
    {"use_z3": True},
    _range_run,
    lambda p, a, o: f"{a['expr']} in {a['dtype']} over {a['ranges']} has no overflow, NaN or lost precision",
)


# --- wave 2 (I2): K2, K5, C5, L1 ----------------------------------------------------
# Same discipline as the FULL kinds above: the arguments are TEXT (inputs and the
# result's key fields as canonical JSON), the re-check runs the module again and
# compares — a hand-edited number is caught by the digest AND by the recomputation —
# and, where the module has an independent route, takes it:
#
#   approx / identify    K2's own ``recheck_approx`` / ``recheck_identify``: the digits
#                        re-enclosed; ``rigorous=False`` is UNDECIDED, never VERIFIED
#   inequality / system / periodic / definite_integral
#                        K5 again (sign decomposition, substitution + completeness
#                        count, proved period, convergence test + quadrature); a typed
#                        refusal is recorded as UNDECIDED and must refuse again
#   iterations / complexity
#                        C5 again: the instrumented function re-COUNTED in the isolated
#                        worker (``execute`` recorded, as for ``code_equiv``), the
#                        limit re-taken
#   recurrence           substitution, initial values, theorem conditions again
#   lint                 the file re-linted with the recorded selection, and every
#                        recorded witness REPLAYED independently (program re-executed,
#                        exact value recomputed by SymPy); call witnesses only through
#                        C1's isolated worker, never in this process

def approx_verdict(a: Any) -> Verdict:
    """An ``Approx`` as a verdict: VERIFIED only when the digits are PROVED
    (``rigorous``); an agreement of two precisions is UNDECIDED (I2, note 5)."""
    if a.rigorous:
        return _verified(a.method, a.detail)
    return Verdict(VerdictStatus.UNDECIDED, a.method, None,
                   f"{a.detail} — rigorous: false, the digits are not proved")


def _approx_run(a: dict, o: dict, budget: float) -> Verdict:
    from ..engine.precision import recheck_approx
    return recheck_approx(a["exact"], a["value"], int(o["digits"]), budget=budget)


def _identify_run(a: dict, o: dict, budget: float) -> Verdict:
    from ..engine.identify import recheck_identify
    basis = None if o["basis"] is None else [b for b in o["basis"].split(",") if b]
    return recheck_identify(a["input"], a["value"] or None, digits=o["digits"], basis=basis,
                            budget=budget)


KINDS["approx"] = Kind(
    (("exact", "text"), ("value", "text")), {"digits": 15},
    _approx_run,
    lambda p, a, o: f"{a['exact']} ≈ {a['value']} (correctly rounded to {o['digits']} digits)",
)
KINDS["identify"] = Kind(
    (("input", "text"), ("value", "text")), {"digits": None, "basis": None},
    _identify_run,
    lambda p, a, o: f"{a['input']} = {a['value'] or '(no closed form recognised)'}",
)


# K5 --------------------------------------------------------------------------------
def _k5_refusals() -> "tuple[type[Exception], ...]":
    from ..engine.definite import IntegralNoClosedForm, IntegralUndecided
    from ..engine.solvesets import SolveSetUndecided
    return (SolveSetUndecided, IntegralUndecided, IntegralNoClosedForm)


def _k5_key(kind: str, d: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "inequality":
        return {"solution": d["solution"]}
    if kind == "system":
        return {"status": d["status"], "solutions": d["solutions"]}
    if kind == "periodic":
        return {"status": d["status"], "families": [f["text"] for f in d.get("families", [])]}
    return {"converges": d["converges"], "value": d["value"],
            "principal_value": d.get("principal_value", False)}


def k5_outcome(kind: str, call: Callable[[], Any]) -> "tuple[Any, dict, Verdict, Exception | None]":
    """Run a K5 operation: ``(result, key fields, verdict, typed refusal or None)``.
    A K5 refusal (no closed form, undecided) is a verdict here — UNDECIDED, with the
    quadrature number it carries — not an error."""
    try:
        res = call()
    except _k5_refusals() as exc:
        approx_value = getattr(exc, "approx", None)
        key = {"refused": type(exc).__name__, "approx": approx_value}
        why = str(exc)
        return None, key, Verdict(VerdictStatus.UNDECIDED, type(exc).__name__, None, why), exc
    d = res.to_dict()
    status = str(d.get("verdict", "undecided")).lower()
    method = str(d.get("method") or d.get("proof") or kind)
    detail: "str | None" = d["certificate"] if isinstance(d.get("certificate"), str) else None
    if not detail and "cells" in d and kind == "inequality":
        detail = (f"sign decomposition ({method}): {len(d['cells'])} cells, "
                  f"{len(d['endpoints'])} endpoints decided exactly")
    elif not detail and kind == "periodic" and d.get("period"):
        detail = (f"period {d['period']} proved ({d.get('periodicity')}); the solutions in one "
                  f"period proved complete by sign decomposition")
    if not detail:
        proof = d.get("proof")
        detail = str(proof.get("argument")) if isinstance(proof, dict) else str(proof or method)
    return res, _k5_key(kind, d), _status_verdict(status, method, str(detail)), None


def _k5_close(recorded_key: str, key: dict, now: Verdict) -> Verdict:
    """Compare the key fields; an ``approx`` number from a quadrature is compared to
    1e-12 relative (it is the integrator's float, recomputed)."""
    rec = _jload(recorded_key, "result")
    if isinstance(rec, dict) and rec.get("approx") is not None and key.get("approx") is not None:
        try:
            if rec.get("refused") == key.get("refused") and math.isclose(
                    float(rec["approx"]), float(key["approx"]), rel_tol=1e-12):
                return now
        except (TypeError, ValueError):
            pass
    return _against_record(recorded_key, key, now)


def _k5_run(kind: str) -> Callable[[dict, dict, float], Verdict]:
    def run(a: dict, o: dict, budget: float) -> Verdict:
        from ..core.budget import time_budget
        from ..engine import definite, solvesets

        i = _jload(a["inputs"], "inputs")
        if not isinstance(i, dict):
            raise DomainError(f"certificate {kind}: inputs must be a JSON object")
        calls: dict[str, Callable[[], Any]] = {
            "inequality": lambda: solvesets.solve_inequality(
                i["relation"], i["var"], within=None if i.get("within") is None else tuple(i["within"])),
            "system": lambda: solvesets.solve_system(i["equations"], i.get("vars")),
            "periodic": lambda: solvesets.periodic_solutions(i["equation"], i["var"]),
            "definite_integral": lambda: definite.definite_integral(
                i["f"], i["var"], i["a"], i["b"], principal_value=bool(i.get("principal_value"))),
        }
        with time_budget(budget):
            _, key, verdict, _ = k5_outcome(kind, calls[kind])
        return _k5_close(a["result"], key, verdict)
    return run


for _name, _noun in (("inequality", "solution set"), ("system", "solutions"),
                     ("periodic", "all real solutions"), ("definite_integral", "integral")):
    KINDS[_name] = Kind(
        (("inputs", "text"), ("result", "text")), {},
        _k5_run(_name),
        (lambda noun: lambda p, a, o: f"{noun} of {_short(a['inputs'], 200)} == {_short(a['result'], 200)}")(_noun),
    )
del _name, _noun


# C5 --------------------------------------------------------------------------------
def _witness_point(w: "Mapping[str, Any] | None") -> dict[str, str]:
    return {k: str(v) for k, v in (w or {}).items() if k not in ("counted", "claim")}


def iterations_key(res: Any) -> tuple[dict, Verdict]:
    """``(key fields, verdict)`` of a ``count_iterations`` / ``check_iterations``
    result. A count is VERIFIED when its derivation is a proof AND the run-and-count
    certificate agrees; not executed -> UNDECIDED; the counter disagrees -> REFUTED."""
    d = res.to_dict()
    if "verdict" in d:  # IterationCheck
        status = str(d["verdict"])
        return ({"count": d["count"], "claim": d["claim"]},
                _status_verdict(status, str(d["method"]), str(d["reason"]),
                                _witness_point(d.get("witness"))))
    key = {"main": d["main"], "valid_for": d["valid_for"], "exact": d["exact"]}
    cert = d.get("certificate")
    if cert is None:
        return key, Verdict(VerdictStatus.UNDECIDED, "derivation", None,
                            "the count is derived from the AST but was not counted by running it")
    if not cert["agree"]:
        return key, Verdict(VerdictStatus.REFUTED, "counting", _witness_point(cert.get("witness")),
                            "the counting certificate disagrees with the derived formula")
    if not d["exact"]:
        return key, Verdict(VerdictStatus.UNDECIDED, "counting", None,
                            f"an upper bound, counted at {cert['points']} points ({cert['domain']})")
    return key, _verified("derivation+counting",
                          f"derived from the AST and counted by running it at {cert['points']} "
                          f"points ({cert['domain']}): all agree")


def complexity_key(res: Any) -> tuple[dict, Verdict]:
    d = res.to_dict()
    return ({"limit": d.get("limit"), "theta": d.get("theta"), "count": d.get("count")},
            _status_verdict(str(d["verdict"]), str(d["method"]), str(d["reason"])))


def recurrence_key(res: Any) -> tuple[dict, Verdict]:
    d = res.to_dict()
    return ({"theta": d.get("theta"), "main": d.get("main")},
            _status_verdict(str(d["verdict"]), str(d["method"]), str(d["reason"])))


def _c5_run(which: str) -> Callable[[dict, dict, float], Verdict]:
    def run(a: dict, o: dict, budget: float) -> Verdict:
        from ..code import loops as L

        fn = a["function"] or None
        if a["claim"]:
            c5_text_guard(a["claim"], "claim")
        with code_execution(bool(o["execute"])):
            if which == "complexity":
                key, verdict = complexity_key(L.check_complexity(a["source"], a["claim"], name=fn,
                                                                 budget=budget))
            elif a["claim"]:
                key, verdict = iterations_key(L.check_iterations(a["source"], a["claim"], name=fn,
                                                                 budget=budget))
            else:
                key, verdict = iterations_key(L.count_iterations(a["source"], name=fn, budget=budget))
        return _against_record(a["result"], key, verdict)
    return run


KINDS["iterations"] = Kind(
    (("source", "text"), ("function", "text"), ("claim", "text"), ("result", "text")),
    {"execute": True},
    _c5_run("iterations"),
    lambda p, a, o: (f"{a['function']}(...) runs its loops {a['claim']} times" if a["claim"]
                     else f"{a['function']}(...) loop count == {_short(a['result'], 200)}"),
)
KINDS["complexity"] = Kind(
    (("source", "text"), ("function", "text"), ("claim", "text"), ("result", "text")),
    {"execute": True},
    _c5_run("complexity"),
    lambda p, a, o: f"{a['function']}(...) loop count is {a['claim']}",
)


def _recurrence_run(a: dict, o: dict, budget: float) -> Verdict:
    from ..code.recurrence import solve_recurrence
    key, verdict = recurrence_key(solve_recurrence(c5_text_guard(a["recurrence"], "recurrence"),
                                                   budget=budget))
    return _against_record(a["result"], key, verdict)


KINDS["recurrence"] = Kind(
    (("recurrence", "text"), ("result", "text")), {},
    _recurrence_run,
    lambda p, a, o: f"{a['recurrence']}: {_short(a['result'], 200)}",
)


# L1 --------------------------------------------------------------------------------
def lint_findings(source: str, path: str, select: "Sequence[str] | None", execute: bool,
                  budget: "float | None" = None) -> list[dict[str, Any]]:
    """``lint_source`` under an explicit config (no cache, no pyproject lookup):
    each finding as ``{"finding": to_dict(), "key": {code, line, witness}}``."""
    from dataclasses import replace as _replace

    from ..lint import Config, lint_source

    config = Config(cache=False)
    config = config.with_cli(select=list(select) if select else None, execute=execute)
    if budget is not None:
        config = _replace(config, budget=float(budget))
    with code_execution(execute):
        found = lint_source(source, path, config)
    out = []
    for f in found:
        d = f.to_dict()
        out.append({"finding": d, "key": {"code": d["code"], "line": d["location"]["row"],
                                          "witness": d.get("witness")}})
    return out


def lint_verdict(findings: Sequence[Mapping[str, Any]]) -> Verdict:
    """The claim "this file's math computes the right values": REFUTED by the first
    finding that has a witness (its inputs are the counterexample); otherwise
    UNDECIDED — a linter that finds nothing has not proved anything."""
    for f in findings:
        w = f["key"]["witness"]
        if w:
            point = {"rule": f["key"]["code"], "line": str(f["key"]["line"]),
                     **{k: str(v) for k, v in (w.get("inputs") or {}).items()}}
            more = f" (+{len(findings) - 1} more)" if len(findings) > 1 else ""
            return Verdict(VerdictStatus.REFUTED, "witness", point,
                           f"{f['finding']['code']} line {f['key']['line']}: {f['finding']['message']}{more}")
    notes = f"{len(findings)} note(s) without a witness" if findings else "no finding"
    return Verdict(VerdictStatus.UNDECIDED, "lint", None,
                   f"{notes} — the absence of findings is not a proof of correctness")


def _lint_run(a: dict, o: dict, budget: float) -> Verdict:
    from ..lint.witness import Witness

    select = [c for c in a["select"].split(",") if c] or None
    findings = lint_findings(a["source"], a["path"], select, bool(o["execute"]), None)
    verdict = lint_verdict(findings)
    recorded = _jload(a["findings"], "findings")
    if not isinstance(recorded, list):
        raise DomainError("certificate lint: findings must be a JSON list")
    for rec in recorded:  # each recorded witness, replayed on its own
        w = rec.get("witness") if isinstance(rec, dict) else None
        if not w or w.get("compare") == "call":
            continue  # a call is re-executed only by the re-lint (C1's isolated worker)
        r = Witness.from_dict(w).replay()
        if not r.differs or r.observed != w["observed"] or r.expected != w["expected"]:
            return Verdict(VerdictStatus.REFUTED, "replay",
                           {"rule": str(rec.get("code")), "line": str(rec.get("line"))},
                           f"the recorded witness does not replay: recorded {w['observed']} vs "
                           f"{w['expected']}, replayed {r.observed} vs {r.expected}")
    return _against_record(a["findings"], [f["key"] for f in findings], verdict)


KINDS["lint"] = Kind(
    (("source", "text"), ("path", "text"), ("select", "text"), ("findings", "text")),
    {"execute": True},
    _lint_run,
    lambda p, a, o: f"lint {a['path']}{' --select ' + a['select'] if a['select'] else ''}: "
                    f"{_short(a['findings'], 200)}",
)


# --- the C5 text guard (I2 adversary round) ---------------------------------------
#: C5 reads claims and recurrences with SymPy's ``sympify`` / ``parse_expr`` WITHOUT a
#: restricted global namespace — ``__import__('os').system(...)`` in a claim RAN
#: (measured in the I2 adversary round, over MCP). Until C5 reads through the strict
#: parser (I2 report, "do poprawki w C5"), every C5 text that reaches it from any
#: surface — CLI, MCP, a certificate's re-check — passes this lexical guard first:
#: math characters only, no attribute access, no strings, no dunder, identifiers that
#: are known math names or short variables, small integer literals, no numeric towers.
_C5_ALLOWED = re.compile(r"^[A-Za-z0-9_+\-*/^()., =<>ΘΩωθ]*$")
_C5_WORDS = frozenset({
    "T", "C", "n", "k", "m", "N", "O", "o", "Theta", "Omega", "omega", "Big", "log", "ln", "lg",
    "sqrt", "exp", "floor", "ceiling", "ceil", "sin", "cos", "tan", "atan", "pi", "E", "oo",
    "harmonic", "factorial", "binomial", "gamma", "Abs", "abs", "Min", "Max", "min", "max",
    "log2", "log10", "floor_log", "ceil_log",
})


def c5_text_guard(text: str, what: str) -> str:
    """``text`` unchanged when it can only be mathematics, else a ``ParseError``."""
    from ..core.errors import ParseError

    if not isinstance(text, str) or len(text) > 2_000:
        raise ParseError(f"{what}: expected a short mathematical text")
    if not _C5_ALLOWED.match(text) or "__" in text or re.search(r"\.\s*[A-Za-z_]", text):
        raise ParseError(f"{what}: {text!r} is not a mathematical expression (characters or "
                         f"attribute access outside the math notation)")
    for word in re.findall(r"[A-Za-z_]\w*", text):
        if word not in _C5_WORDS and len(word) > 2:
            raise ParseError(f"{what}: unknown name {word!r} in {text!r}")
    if any(len(d) > 12 for d in re.findall(r"\d+", text)):
        raise ParseError(f"{what}: a number literal longer than 12 digits")
    if re.search(r"\d\s*(\*\*|\^)\s*\(?\s*\d+\s*\)?\s*(\*\*|\^)", text):
        raise ParseError(f"{what}: a tower of numeric powers is too costly to read")
    return text
