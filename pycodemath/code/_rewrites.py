"""Rewrite base for the precision doctor (C2): rules that trade an inaccurate
floating-point formula for an EQUIVALENT one that rounds better.

Nothing here decides that a rewrite is correct or useful. Every candidate made
here is checked by ``stability`` twice — proved equal to the original (or, for a
Taylor polynomial, proved within the unit roundoff on an interval) and measured
more accurate on sample points — and dropped otherwise. That is what lets the
rules be generous: a rule that fires where it does not help costs a measurement,
never a wrong answer.

Three sources of candidates:

* ``local_rewrites`` — targeted rules, each one a textbook remedy (Hamming 1987,
  Goldberg 1991, Higham 2002; the same families as Herbie's rule set): the
  conjugate for ``p - sqrt(q)``, ``log1p`` / ``expm1``, sum-to-product for
  ``sin a - sin b``, ``1 - cos x = 2 sin(x/2)^2``, ``hypot``, the max-shift for
  log-sum-exp and softmax, two-pass variance, combining fractions,
  ``(1+r)^n = exp(n log1p r)``.
* ``sympy_rewrites`` — SymPy's algebra on a sub-formula (``cancel``, ``together``,
  ``radsimp``, ``trigsimp``, ``simplify`` …), read back in the order SymPy holds it.
* ``taylor_candidates`` — truncated Taylor series at 0 of a one-variable
  sub-formula, each with the radius on which ``certify_taylor`` PROVES the relative
  truncation error is at most the unit roundoff (Lagrange remainder, interval bound
  of the n-th derivative). Used only inside regimes, only inside that radius.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable, Iterable

import mpmath
import sympy as sp

from .stability import (
    Node,
    _ONE,
    _TWO,
    _ZERO,
    _const,
    _ends,
    _iv_eval,
    _op,
    _Invalid,
    _OutOfReach,
    _Unresolved,
    from_sympy,
    to_sympy,
)

__all__ = ["local_rewrites", "sympy_rewrites", "taylor_candidates", "certify_taylor", "hint_for"]

Rewrite = tuple[Node, str]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _simp(e: sp.Expr) -> sp.Expr:
    """The shortest of a few exact normal forms (used for differences like u - v)."""
    forms = [e]
    for fn in (sp.cancel, sp.expand, sp.factor, sp.together):
        try:
            forms.append(fn(e))
        except Exception:  # noqa: BLE001 - a normal form that fails is just skipped
            continue
    return min(forms, key=lambda f: (sp.count_ops(f), len(str(f))))


def _sy(n: Node) -> sp.Expr:
    return to_sympy(n)


def _back(e: sp.Expr) -> "Node | None":
    try:
        return from_sympy(e)
    except Exception:  # noqa: BLE001 - no float spelling: no candidate
        return None


def _sum_terms(n: Node) -> list[tuple[int, Node]]:
    """Flatten ``a + b - c`` into signed terms ``[(+1, a), (+1, b), (-1, c)]``."""
    if n.op == "+":
        return _sum_terms(n.args[0]) + _sum_terms(n.args[1])
    if n.op == "-":
        return _sum_terms(n.args[0]) + [(-s, t) for s, t in _sum_terms(n.args[1])]
    if n.op == "neg":
        return [(-s, t) for s, t in _sum_terms(n.args[0])]
    return [(1, n)]


def _build_sum(terms: list[tuple[int, Node]]) -> Node:
    pos = [t for s, t in terms if s > 0]
    neg = [t for s, t in terms if s < 0]
    acc = pos[0] if pos else _op("neg", neg.pop(0))
    for t in pos[1:]:
        acc = _op("+", acc, t)
    for t in neg:
        acc = _op("-", acc, t)
    return acc


def _is_sqrt(n: Node) -> bool:
    return n.op == "sqrt"


def _is_cbrt(n: Node) -> bool:
    if n.op == "cbrt":
        return True
    if n.op == "pow":
        e = n.args[1]
        if e.op == "const" and e.value == sp.Rational(1, 3):
            return True
        if e.op == "/" and all(a.op == "const" for a in e.args):
            return bool(e.args[0].value / e.args[1].value == sp.Rational(1, 3))
    return False


def _is_one(n: Node) -> bool:
    return n.op == "const" and n.value == 1


def _square_of(n: Node) -> "Node | None":
    if n.op == "*" and n.args[0] == n.args[1]:
        return n.args[0]
    if n.op == "pow" and n.args[1].op == "const" and n.args[1].value == 2:
        return n.args[0]
    return None


def _max_of(nodes: list[Node]) -> Node:
    acc = nodes[0]
    for x in nodes[1:]:
        acc = _op("max", acc, x)
    return acc


def _min_of(nodes: list[Node]) -> Node:
    acc = nodes[0]
    for x in nodes[1:]:
        acc = _op("min", acc, x)
    return acc


def _exp_arg(n: Node) -> "Node | None":
    """``w`` if ``n`` is ``exp(w)``; the constant 1 counts as ``exp(0)``."""
    if n.op == "exp":
        return n.args[0]
    if _is_one(n):
        return _ZERO
    return None


def _minus(a: Node, b: Node) -> Node:
    if b == _ZERO:
        return a
    if a == _ZERO:
        return _op("neg", b)
    return _op("-", a, b)


# --------------------------------------------------------------------------
# targeted rules (each: node -> list of (replacement, rule name))
# --------------------------------------------------------------------------


def _r_conjugate(n: Node) -> list[Rewrite]:
    """``P + s*sqrt(q) = (P^2 - q) / (P - s*sqrt(q))`` — the conjugate, numerator simplified."""
    if n.op not in ("+", "-"):
        return []
    terms = _sum_terms(n)
    out: list[Rewrite] = []
    for i, (s, t) in enumerate(terms):
        if not _is_sqrt(t):
            continue
        rest = terms[:i] + terms[i + 1:]
        if not rest:
            continue
        p = _build_sum(rest)
        num = _back(_simp(sp.expand(_sy(p) ** 2) - _sy(t.args[0])))
        if num is None:
            continue
        bottom = _op("-", p, t) if s > 0 else _op("+", p, t)
        out.append((_op("/", num, bottom), "conjugate"))
    return out


def _r_cbrt_difference(n: Node) -> list[Rewrite]:
    """``A - B = (A^3 - B^3) / (A^2 + A*B + B^2)`` for cube roots."""
    if n.op != "-" or not (_is_cbrt(n.args[0]) and _is_cbrt(n.args[1])):
        return []
    a, b = n.args
    num = _back(_simp(_sy(a.args[0]) - _sy(b.args[0])))
    if num is None:
        return []
    den = _op("+", _op("+", _op("*", a, a), _op("*", a, b)), _op("*", b, b))
    return [(_op("/", num, den), "cube-root difference")]


def _r_log1p(n: Node) -> list[Rewrite]:
    """``log(u) = log1p(u - 1)``, with ``u - 1`` simplified exactly."""
    if n.op != "log":
        return []
    w = _back(_simp(_sy(n.args[0]) - 1))
    if w is None or w.op == "const":
        return []
    return [(_op("log1p", w), "log1p")]


def _r_log_difference(n: Node) -> list[Rewrite]:
    """``log a - log b = log(a/b) = log1p((a-b)/b)`` (needs a, b > 0)."""
    if n.op != "-" or n.args[0].op != "log" or n.args[1].op != "log":
        return []
    a, b = n.args[0].args[0], n.args[1].args[0]
    out: list[Rewrite] = [(_op("log", _op("/", a, b)), "log difference")]
    d = _back(_simp(_sy(a) - _sy(b)))
    if d is not None:
        out.append((_op("log1p", _op("/", d, b)), "log difference -> log1p"))
    return out


def _r_expm1(n: Node) -> list[Rewrite]:
    """``exp(w) - 1 = expm1(w)``; ``exp(a) - exp(b) = exp(b) * expm1(a - b)``."""
    if n.op not in ("-", "+"):
        return []
    terms = _sum_terms(n)
    if len(terms) != 2:
        return []
    (s1, t1), (s2, t2) = terms
    out: list[Rewrite] = []
    for (sa, ta), (sb, tb) in (((s1, t1), (s2, t2)), ((s2, t2), (s1, t1))):
        if ta.op == "exp" and _is_one(tb) and sa == -sb:
            core = _op("expm1", ta.args[0])
            out.append((core if sa > 0 else _op("neg", core), "expm1"))
        if ta.op == "exp" and tb.op == "exp" and sa > 0 > sb:
            d = _back(_simp(_sy(ta.args[0]) - _sy(tb.args[0])))
            if d is not None:
                out.append((_op("*", tb, _op("expm1", d)), "exp difference -> expm1"))
    return out


def _half_sum_and_difference(a: Node, b: Node) -> "tuple[Node, Node] | None":
    s = _back(_simp((_sy(a) + _sy(b)) / 2))
    d = _back(_simp((_sy(a) - _sy(b)) / 2))
    if s is None or d is None:
        return None
    return s, d


def _r_trig(n: Node) -> list[Rewrite]:
    """Sum-to-product and half-angle forms of differences of sines and cosines."""
    out: list[Rewrite] = []
    if n.op == "-":
        a, b = n.args
        if a.op == b.op == "sin":
            hs = _half_sum_and_difference(a.args[0], b.args[0])
            if hs:
                out.append((_op("*", _op("*", _TWO, _op("cos", hs[0])), _op("sin", hs[1])), "sin difference"))
        if a.op == b.op == "cos":
            hs = _half_sum_and_difference(a.args[0], b.args[0])
            if hs:
                out.append((_op("neg", _op("*", _op("*", _TWO, _op("sin", hs[0])), _op("sin", hs[1]))),
                            "cos difference"))
        if a.op == b.op == "tan":
            d = _back(_simp(_sy(a.args[0]) - _sy(b.args[0])))
            if d is not None:
                out.append((_op("/", _op("sin", d), _op("*", _op("cos", a.args[0]), _op("cos", b.args[0]))),
                            "tan difference"))
        if _is_one(a) and b.op == "cos":
            w = b.args[0]
            half = _back(_simp(_sy(w) / 2))
            if half is not None:
                s = _op("sin", half)
                out.append((_op("*", _TWO, _op("*", s, s)), "1 - cos -> 2 sin^2"))
            sw = _op("sin", w)
            out.append((_op("/", _op("*", sw, sw), _op("+", _ONE, b)), "1 - cos -> sin^2/(1 + cos)"))
        if a.op == "cos" and _is_one(b):
            w = a.args[0]
            half = _back(_simp(_sy(w) / 2))
            if half is not None:
                s = _op("sin", half)
                out.append((_op("neg", _op("*", _TWO, _op("*", s, s))), "cos - 1 -> -2 sin^2"))
    return out


def _r_hypot(n: Node) -> list[Rewrite]:
    """``sqrt(x*x + y*y) = hypot(x, y)`` (no overflow or underflow of the squares)."""
    if n.op != "sqrt" or n.args[0].op != "+":
        return []
    a, b = n.args[0].args
    sa, sb = _square_of(a), _square_of(b)
    if sa is not None and sb is not None:
        return [(_op("hypot", sa, sb), "hypot")]
    for sq, other in ((sa, b), (sb, a)):
        if sq is not None and other.op == "const" and other.value > 0:
            r = sp.sqrt(other.value)
            if r.is_Rational:
                return [(_op("hypot", sq, _const(r)), "hypot")]
    return []


def _exp_terms(n: Node) -> "list[Node] | None":
    terms = _sum_terms(n)
    if len(terms) < 2 or any(s < 0 for s, _ in terms):
        return None
    args = [_exp_arg(t) for _, t in terms]
    if any(a is None for a in args) or all(a == _ZERO for a in args):
        return None
    return [a for a in args if a is not None]


def _shifted_sum(args: list[Node], m: Node) -> Node:
    acc = _op("exp", _minus(args[0], m))
    for w in args[1:]:
        acc = _op("+", acc, _op("exp", _minus(w, m)))
    return acc


def _r_logsumexp(n: Node) -> list[Rewrite]:
    """``log(sum exp(w_i)) = M + log(sum exp(w_i - M))``, ``M = max w_i``."""
    if n.op != "log":
        return []
    args = _exp_terms(n.args[0])
    if args is None:
        return []
    m = _max_of(args)
    out: list[Rewrite] = [(_op("+", m, _op("log", _shifted_sum(args, m))), "log-sum-exp max shift")]
    if len(args) == 2:
        lo = _min_of(args)
        out.append((_op("+", m, _op("log1p", _op("exp", _op("-", lo, m)))), "log-sum-exp (2 terms) with log1p"))
    return out


def _r_softmax(n: Node) -> list[Rewrite]:
    """``exp(w_k) / sum exp(w_i) = exp(w_k - M) / sum exp(w_i - M)``."""
    if n.op != "/" or n.args[0].op != "exp":
        return []
    args = _exp_terms(n.args[1])
    if args is None or n.args[0].args[0] not in args:
        return []
    m = _max_of(args)
    return [(_op("/", _op("exp", _minus(n.args[0].args[0], m)), _shifted_sum(args, m)), "softmax max shift")]


def _r_log_reciprocal(n: Node) -> list[Rewrite]:
    """``log(1/u) = -log(u)``."""
    if n.op == "log" and n.args[0].op == "/" and _is_one(n.args[0].args[0]):
        return [(_op("neg", _op("log", n.args[0].args[1])), "log of a reciprocal")]
    return []


def _r_variance(n: Node) -> list[Rewrite]:
    """One-pass variance ``E[x^2] - E[x]^2`` -> two-pass ``E[(x - mean)^2]``."""
    names = n.variables
    if len(names) < 2 or n.op != "-":
        return []
    xs = [sp.Symbol(v, real=True) for v in names]
    k = len(xs)
    mean = sum(xs) / k
    try:
        e = sp.expand(_sy(n))
    except Exception:  # noqa: BLE001
        return []
    out: list[Rewrite] = []
    for denom, label in ((k, "population"), (k - 1, "sample")):
        target = sp.expand(sum((x - mean) ** 2 for x in xs) / denom)
        if sp.expand(e - target) != 0:
            continue
        total = _var(names[0])
        for v in names[1:]:
            total = _op("+", total, _var(v))
        mean_n = _op("/", total, _const(k))
        acc = None
        for v in names:
            d = _op("-", _var(v), mean_n)
            sq = _op("*", d, d)
            acc = sq if acc is None else _op("+", acc, sq)
        assert acc is not None
        out.append((_op("/", acc, _const(denom)), f"two-pass {label} variance"))
    return out


def _var(name: str) -> Node:
    return Node("var", name=name)


def _r_fractions(n: Node) -> list[Rewrite]:
    """``a/b - c/d = (a*d - c*b) / (b*d)`` with the numerator simplified exactly."""
    if n.op not in ("+", "-"):
        return []
    terms = _sum_terms(n)
    if len(terms) < 2 or not any(t.op == "/" for _, t in terms):
        return []
    expr = sum(s * _sy(t) for s, t in terms)
    num, den = sp.fraction(sp.together(expr))
    num = _simp(sp.expand(num))
    top, bottom = _back(num), _back(sp.factor(den))
    if top is None or bottom is None:
        return []
    return [(_op("/", top, bottom), "combine fractions")]


def _r_pow_log1p(n: Node) -> list[Rewrite]:
    """``(1 + r)^n = exp(n * log1p(r))`` (needs 1 + r > 0)."""
    if n.op != "pow" or (n.args[1].op == "const" and n.args[1].value.is_Integer and abs(n.args[1].value) < 4):
        return []
    w = _back(_simp(_sy(n.args[0]) - 1))
    if w is None or w.op == "const":
        return []
    return [(_op("exp", _op("*", n.args[1], _op("log1p", w))), "power of 1+r -> exp(n log1p r)")]


def _r_atan_difference(n: Node) -> list[Rewrite]:
    """``atan a - atan b = atan((a - b) / (1 + a*b))`` (proved only where 1 + ab > 0)."""
    if n.op != "-" or n.args[0].op != "atan" or n.args[1].op != "atan":
        return []
    a, b = _sy(n.args[0].args[0]), _sy(n.args[1].args[0])
    num, den = _back(_simp(a - b)), _back(_simp(sp.expand(1 + a * b)))
    if num is None or den is None:
        return []
    return [(_op("atan", _op("/", num, den)), "atan difference")]


def _r_divide_through(n: Node) -> list[Rewrite]:
    """``a / b = (a/v) / (b/v)`` for a variable ``v`` — keeps ``x / (1 + x*x)`` from
    overflowing in ``x*x`` (``1 / (1/x + x)``); regimes keep the original where small."""
    if n.op != "/":
        return []
    out: list[Rewrite] = []
    a, b = _sy(n.args[0]), _sy(n.args[1])
    for v in sorted(n.variables):
        x = sp.Symbol(v, real=True)
        if not (a.has(x) and b.has(x)):
            continue
        top, bottom = _back(sp.expand(a / x)), _back(sp.expand(b / x))
        if top is not None and bottom is not None:
            out.append((_op("/", top, bottom), f"divide through by {v}"))
    return out


def _r_sqrt_scale(n: Node) -> list[Rewrite]:
    """``sqrt(X*X - Y) = |X| * sqrt(1 - (Y1/X)*(Y2/X))`` — the square of a huge ``X``
    never formed (the quadratic formula's ``b*b`` overflows from |b| ~ 1e154)."""
    if n.op != "sqrt" or n.args[0].op not in ("-", "+"):
        return []
    left, right = n.args[0].args
    x = _square_of(left)
    if x is None:
        return []
    y = right if n.args[0].op == "-" else _op("neg", right)
    if y.op == "*":
        ratio = _op("*", _op("/", y.args[0], x), _op("/", y.args[1], x))
    else:
        ratio = _op("/", _op("/", y, x), x)
    return [(_op("*", _op("fabs", x), _op("sqrt", _op("-", _ONE, ratio))), "scale the root by |X|")]


def _r_difference_of_squares(n: Node) -> list[Rewrite]:
    """``a*a - b*b = (a - b) * (a + b)``."""
    if n.op != "-":
        return []
    a, b = _square_of(n.args[0]), _square_of(n.args[1])
    if a is None or b is None:
        return []
    return [(_op("*", _op("-", a, b), _op("+", a, b)), "difference of squares")]


_RULES: tuple[Callable[[Node], list[Rewrite]], ...] = (
    _r_conjugate, _r_cbrt_difference, _r_log1p, _r_log_difference, _r_expm1, _r_trig,
    _r_hypot, _r_logsumexp, _r_softmax, _r_log_reciprocal, _r_variance, _r_fractions,
    _r_pow_log1p, _r_difference_of_squares, _r_atan_difference, _r_divide_through,
    _r_sqrt_scale,
)


def local_rewrites(n: Node) -> list[Rewrite]:
    """Every targeted rule that applies at ``n`` (the node itself, not below it)."""
    out: list[Rewrite] = []
    for rule in _RULES:
        try:
            out.extend(rule(n))
        except (RecursionError, MemoryError):
            raise
        except Exception:  # noqa: BLE001 - a rule that cannot build its form proposes nothing
            continue
    return [(r, name) for r, name in out if r != n]


_HINTS: tuple[tuple[Callable[[Node], bool], str], ...] = (
    (lambda n: n.op == "log" and n.args[0].op == "+" and any(_is_one(a) for a in n.args[0].args),
     "log(1 + u) -> log1p(u)"),
    (lambda n: n.op in ("+", "-") and any(a.op == "exp" for a in n.args) and any(_is_one(a) for a in n.args),
     "exp(u) - 1 -> expm1(u)"),
    (lambda n: n.op in ("+", "-") and any(_is_sqrt(a) or (a.op == "neg" and _is_sqrt(a.args[0])) for a in n.args),
     "p - sqrt(q) -> (p^2 - q) / (p + sqrt(q)) (conjugate)"),
    (lambda n: n.op == "-" and _is_one(n.args[0]) and n.args[1].op == "cos", "1 - cos(u) -> 2*sin(u/2)^2"),
    (lambda n: n.op == "-" and n.args[0].op == n.args[1].op and n.op in ("-",) and n.args[0].op in ("sin", "cos", "tan"),
     "difference of trig functions -> product form"),
    (lambda n: n.op == "sqrt" and n.args[0].op == "+" and _square_of(n.args[0].args[0]) is not None,
     "sqrt(x*x + y*y) -> hypot(x, y)"),
    (lambda n: n.op == "log" and _exp_terms(n.args[0]) is not None, "log(sum exp) -> max shift (log-sum-exp)"),
    (lambda n: n.op == "/" and n.args[0].op == "exp" and _exp_terms(n.args[1]) is not None,
     "softmax -> subtract the max before exp"),
    (lambda n: n.op == "-" and n.args[0].op == "log" and n.args[1].op == "log", "log(a) - log(b) -> log1p((a-b)/b)"),
    (lambda n: n.op in ("+", "-") and any(a.op == "/" for a in n.args), "combine the fractions"),
    (lambda n: n.op in ("exp",), "exp of a large argument overflows: shift or combine"),
)


def hint_for(n: Node) -> str:
    """A one-line textbook remedy for the operation at ``n`` ('' when none applies)."""
    for test, text in _HINTS:
        try:
            if test(n):
                return text
        except Exception:  # noqa: BLE001
            continue
    return ""


# --------------------------------------------------------------------------
# SymPy algebra on a sub-formula
# --------------------------------------------------------------------------

def _collect(e: sp.Expr) -> sp.Expr:
    """Expand, then gather terms over the function atoms (``sin(x)*(cos(e) - 1) + ...``)."""
    atoms = sorted(e.atoms(sp.Function), key=str)
    return sp.collect(sp.expand(e), atoms) if atoms else e


def _collect_vars(e: sp.Expr) -> sp.Expr:
    """Expand, then gather over the variables: ``(N+1) log(N+1) - N log N`` ->
    ``N*(log(N + 1) - log(N)) + log(N + 1)``, exposing the difference of logs."""
    syms = sorted(e.free_symbols, key=lambda z: z.name)
    return sp.collect(sp.expand(e), syms) if syms else e


def _expand_trig_collect(e: sp.Expr) -> sp.Expr:
    """Angle-addition expansion, then gathering: ``sin(x + e) - sin(x)`` ->
    ``sin(x)*(cos(e) - 1) + sin(e)*cos(x)`` — no ``x + e`` left to round."""
    return _collect(sp.expand_trig(e))


_TRANSFORMS: tuple[tuple[str, Callable[[sp.Expr], sp.Expr]], ...] = (
    ("cancel", sp.cancel),
    ("expand_trig+collect", _expand_trig_collect),
    ("collect", _collect),
    ("collect variables", _collect_vars),
    ("together", sp.together),
    ("radsimp", sp.radsimp),
    ("factor", sp.factor),
    ("expand", sp.expand),
    ("trigsimp", sp.trigsimp),
    ("simplify", sp.simplify),
)


def sympy_rewrites(n: Node, run: Callable[[Callable[[], Any]], Any]) -> list[Rewrite]:
    """``n`` rewritten by each SymPy transform (each under ``run``'s budget)."""
    try:
        e = _sy(n)
    except Exception:  # noqa: BLE001
        return []
    out: list[Rewrite] = []
    # SymPy's automatic evaluation is itself a rewrite: (x + 1) - x is already 1.
    own = _back(e)
    if own is not None and own != n:
        out.append((own, "sympy evaluation"))
    for name, fn in _TRANSFORMS:
        r = run(functools.partial(fn, e))
        if r is None or r == e:
            continue
        back = _back(r)
        if back is not None and back != n:
            out.append((back, f"sympy {name}"))
    return out


# --------------------------------------------------------------------------
# Taylor series with a proven radius
# --------------------------------------------------------------------------


def taylor_candidates(n: Node, var: str, run: Callable[[Callable[[], Any]], Any],
                      max_terms: int = 4) -> list[tuple[Node, str, sp.Expr, int, int]]:
    """Truncated series of ``n`` at ``var = 0`` with 1..max_terms nonzero terms.

    Returns ``(polynomial, label, exact polynomial, order, valuation)``; the series
    is O(x**order), i.e. its Lagrange remainder involves the ``order``-th derivative.
    """
    x = sp.Symbol(var, real=True)
    f = _sy(n)
    out: list[tuple[Node, str, sp.Expr, int, int]] = []
    full = run(lambda: sp.series(f, x, 0, 24).removeO())
    if full is None or full == 0:
        return out
    poly = sp.Poly(full, x) if full.is_polynomial(x) else None
    if poly is None:
        return out
    terms = sorted(poly.terms(), key=lambda t: t[0][0])
    k = terms[0][0][0]
    for count in range(1, min(max_terms, len(terms)) + 1):
        used = terms[:count]
        order = terms[count][0][0] if count < len(terms) else used[-1][0][0] + 1
        t = sum(c * x**d for (d,), c in used)
        ir = _back(sp.horner(t, x)) if count > 1 else _back(t)
        if ir is None:
            continue
        out.append((ir, f"Taylor series at 0 ({count} term{'s' if count > 1 else ''})", t, order, k))
    return out


def _iv_value(n: Node, var: str, lo: Any, hi: Any, prec: int) -> "Any | None":
    """Enclosure of ``n`` over ``var`` in [lo, hi], or None when undecidable."""
    try:
        nodes = _iv_eval(n, {var: mpmath.iv.mpf([lo, hi])}, prec)
    except (_Invalid, _Unresolved, _OutOfReach, ZeroDivisionError, ValueError):
        return None
    except Exception:  # noqa: BLE001 - e.g. an operation with no interval form
        return None
    return nodes[n]


@functools.lru_cache(maxsize=256)
def _prepare(f: Node, var: str, poly: sp.Expr, order: int, k: int) -> "tuple[Any, Node, Any, Node | None] | None":
    """The ``order``-th derivative of ``f`` and ``poly / x^k`` as formulas (computed once)."""
    x = sp.Symbol(var, real=True)
    try:
        deriv = sp.diff(to_sympy(f), x, order)
        d_node = from_sympy(deriv)
        p_expr = sp.expand(poly / x**k)
        p_node = from_sympy(p_expr) if p_expr.free_symbols else None
    except Exception:  # noqa: BLE001
        return None
    return deriv, d_node, p_expr, p_node


def certify_taylor(f: Node, var: str, poly: sp.Expr, order: int, k: int, r: float,
                   unit_roundoff: float, pieces: int = 16) -> bool:
    """Prove ``|f(x) - poly(x)| <= u |f(x)|`` for ``0 < |x| <= r`` (sign of ``r`` = side).

    Lagrange: ``f(x) - poly(x) = f^(order)(xi) x^order / order!`` with ``M`` an interval
    bound of ``|f^(order)|`` on the hull ``[0, r]``. With ``P = poly / x^k``,
    ``|f(x)| >= |x|^k (|P(x)| - M |x|^(order-k) / order!)``, so the relative error is at
    most ``A / (B - A)`` with ``A = M r^(order-k) / order!`` and ``B = min |P|``.
    Every bound is an interval enclosure; any undecidable piece fails the proof.
    """
    if r == 0 or order <= k:
        return False
    prepared = _prepare(f, var, poly, order, k)
    if prepared is None:
        return False
    deriv, d_node, p_expr, p_node = prepared
    prec = 113
    lo_end, hi_end = (0.0, r) if r > 0 else (r, 0.0)
    edges = [lo_end + (hi_end - lo_end) * i / pieces for i in range(pieces + 1)]
    m_bound = mpmath.mpf(0)
    if p_node is None:
        with mpmath.workprec(prec):
            p_min = abs(mpmath.mpf(sp.N(p_expr, 40))) * (1 - mpmath.ldexp(1, -100))
    else:
        p_min = mpmath.inf
    for a, b in zip(edges, edges[1:]):
        dv = _iv_value(d_node, var, a, b, prec) if d_node.variables else None
        if d_node.variables:
            if dv is None:
                return False
            dl, dh = _ends(dv)
        else:
            dl = dh = abs(mpmath.mpf(sp.N(deriv, 40))) * (1 + mpmath.ldexp(1, -100))
        if not (mpmath.isfinite(dl) and mpmath.isfinite(dh)):
            return False
        m_bound = max(m_bound, abs(dl), abs(dh))
        if p_node is not None:
            pv = _iv_value(p_node, var, a, b, prec)
            if pv is None:
                return False
            pl, ph = _ends(pv)
            if pl <= 0 <= ph:
                return False
            p_min = min(p_min, min(abs(pl), abs(ph)))
    with mpmath.workprec(prec):
        # A safety factor of 2^-60 covers the rounding of this final arithmetic.
        a_bound = m_bound * mpmath.mpf(abs(r)) ** (order - k) / mpmath.factorial(order) * (1 + mpmath.ldexp(1, -60))
        if p_min - a_bound <= 0:
            return False
        return bool(a_bound / (p_min - a_bound) <= unit_roundoff * (1 - mpmath.ldexp(1, -60)))


def certified_radius(f: Node, var: str, poly: sp.Expr, order: int, k: int, limit: float,
                     unit_roundoff: float, deadline_ok: Callable[[], bool]) -> float:
    """The largest ``2^e`` (signed like ``limit``) up to ``|limit|`` that ``certify_taylor``
    proves, or 0.0. Bisection on the exponent only, so a dozen proofs at most."""
    if limit == 0:
        return 0.0
    sign = 1.0 if limit > 0 else -1.0
    hi_e = math.frexp(abs(limit))[1]
    lo_e = -1074
    if certify_taylor(f, var, poly, order, k, limit, unit_roundoff):
        return limit
    best = 0.0
    while lo_e <= hi_e and deadline_ok():
        mid = (lo_e + hi_e) // 2
        r = sign * math.ldexp(1.0, mid)
        if abs(r) > abs(limit):
            hi_e = mid - 1
            continue
        if certify_taylor(f, var, poly, order, k, r, unit_roundoff):
            best = r
            lo_e = mid + 1
        else:
            hi_e = mid - 1
    return best


def subformulas(n: Node) -> Iterable[Node]:
    """Operation nodes of ``n``, root last."""
    return [m for m in n.walk() if m.op not in ("var", "const")]
