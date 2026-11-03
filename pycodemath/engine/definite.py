"""Certified definite and improper integrals (series FULL, module K5).

``definite_integral("exp(-x^2)", "x", "-oo", "oo")`` returns ``sqrt(pi)`` — and the
reason to believe it. The classic failure this module exists to prevent is the
Fundamental Theorem of Calculus applied THROUGH a pole::

    ∫_{-1}^{1} dx/x^2 = [-1/x]_{-1}^{1} = -2        (a negative area of a positive function)

--- THE PROCEDURE ------------------------------------------------------------------

1. SINGULAR POINTS. Every point of [a, b] where the integrand can stop being finite or
   real — zeros of denominators, of ``log`` arguments, of bases with negative or
   fractional exponents, poles of ``tan`` — is found with the cut-point machinery of
   ``engine.solvesets`` (exact real roots for rational pieces). The interval is split
   there; no quadrature and no antiderivative ever crosses such a point. A piece on
   which the integrand is not real (``log(x)`` on [-1, 0]) is a ``DomainError``.

2. CONVERGENCE AT EVERY SINGULAR END, proved by the LIMIT COMPARISON TEST with SymPy
   limits as the only computation: with ``t`` the distance to a finite end,

   * ``t^p * f`` has a FINITE limit for some p < 1  ->  that side converges
     (|f| <= C t^-p near the end, and t^-p is integrable);
   * ``t * f`` has a NONZERO limit (or ±oo)          ->  that side diverges, to the
     sign of the limit (|f| >= c/t near the end).

   At ±oo the same with ``x^p * f``, p > 1, and ``x * f``. An oscillating tail
   ``g(x) * sin(w*x + c)`` converges by DIRICHLET's test when ``g -> 0`` and ``g'``
   has a PROVED constant sign on a tail (interval enclosure); ``c * sin(w*x + c)``
   diverges without a value. Anything else is UNDECIDED, never assumed.

3. VALUE. A divergent integral is returned as such: ``value = "oo"``/``"-oo"`` when every
   divergent side has that sign, ``None`` when they disagree or oscillate (``1/x`` on
   [-1, 1]). A Cauchy principal value is computed ONLY when asked
   (``principal_value=True``) and only across interior simple poles whose two sides
   cancel. A convergent integral gets SymPy's closed form on every piece separately
   (under the time budget) and an INDEPENDENT number: mpmath tanh-sinh quadrature over
   the same pieces at 30 and at 45 digits (``quadosc`` for oscillating tails). The
   answer is returned only when the closed form agrees with the quadrature within the
   quadrature's error estimate, and that estimate is small (1e-10 relative): this is a
   ``cross-check``, recorded as such. Disagreement raises ``IntegralUndecided`` carrying
   the quadrature value — the closed form SymPy proposed is then REFUTED by a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import mpmath
import sympy as sp

from ..core.budget import under_budget
from ..core.errors import DomainError, NoClosedFormError, PycodemathError, UnsupportedFormError
from .solvesets import (
    SolveSetUndecided,
    _constraint_sign_on_cell,
    _constraints,
    _exact,
    _lt,
    _parse_expr,
    _rational_between,
    _real_symbol,
    _s,
    _sign_at,
    _sorted_unique,
    _tail_sign,
    _to_real,
    _zeros_in,
)

__all__ = [
    "DefiniteIntegralResult",
    "IntegralUndecided",
    "IntegralNoClosedForm",
    "definite_integral",
]

#: Functions accepted in an integrand: the ones whose domain ``_constraints`` knows,
#: plus entire / bounded ones that add no singular point of their own.
_INTEGRAND_FUNCS = (
    sp.exp, sp.log, sp.Abs, sp.sin, sp.cos, sp.tan, sp.atan, sp.atan2,
    sp.sinh, sp.cosh, sp.tanh, sp.erf, sp.erfc, sp.sign,
)

#: Exponents tried in the convergence comparison (finite end: t^p f; infinite end: x^p f).
_P_FINITE = (sp.Integer(0), sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(9, 10))
_P_INFINITE = (sp.Integer(2), sp.Rational(3, 2), sp.Rational(5, 4), sp.Rational(11, 10))

_DPS_LO, _DPS_HI = 30, 45
#: Relative size the quadrature error estimate must stay under for a cross-check.
_REL_ERR = mpmath.mpf("1e-10")


class IntegralUndecided(PycodemathError):
    """Neither a proved divergence nor a cross-checked value could be produced.

    Raised when the convergence at some end is not decided by the tests in the module
    docstring, when the quadrature is not accurate enough to vouch for anything, or
    when SymPy's closed form DISAGREES with the quadrature. ``approx``/``error`` carry
    the quadrature value and its estimate when one exists (as strings); ``sympy_value``
    the closed form that was refuted, if any. ``route`` is ``"nintegrate"``.
    """

    def __init__(self, message: str, *, approx: "str | None" = None, error: "str | None" = None,
                 sympy_value: "str | None" = None, reason: str = "") -> None:
        super().__init__(message, route="nintegrate")
        self.approx = approx
        self.error = error
        self.sympy_value = sympy_value
        self.reason = reason


class IntegralNoClosedForm(NoClosedFormError):
    """The integral CONVERGES (proved) but SymPy found no closed form for it.

    ``approx``/``error`` carry the quadrature value and its error estimate, so the
    numerical answer is not lost; ``route`` is ``"nintegrate"``.
    """

    def __init__(self, message: str, *, approx: "str | None" = None, error: "str | None" = None) -> None:
        super().__init__(message, route="nintegrate")
        self.approx = approx
        self.error = error


@dataclass(frozen=True, slots=True)
class DefiniteIntegralResult:
    """A definite integral with the evidence for its value — or for its divergence.

    ``converges``      False for a PROVED divergence (then ``value`` is ``"oo"``,
                       ``"-oo"`` or None = no value; ``approx`` is None)
    ``value``          exact value as SymPy text (or the divergence, above)
    ``approx``         20 significant digits of the value
    ``singularities``  points of [a, b] where the integrand is not finite/defined
    ``pieces``         the sub-intervals integrated separately
    ``behaviour``      one record per singular or infinite end: the comparison test used
    ``numeric``        the quadrature: value, error estimate, method, digits
    ``proof``          ``"proof"`` (divergence by comparison) or ``"cross-check"`` (closed
                       form agrees with an independent quadrature)
    """

    integrand: str
    var: str
    lower: str
    upper: str
    converges: bool
    value: "str | None"
    approx: "str | None"
    principal_value: bool
    singularities: tuple[str, ...]
    pieces: tuple[tuple[str, str], ...]
    behaviour: tuple[dict[str, Any], ...]
    numeric: "dict[str, str] | None"
    sympy_value: "str | None"
    proof: Literal["proof", "cross-check"]
    certificate: str
    kind: str = "definite_integral"

    @property
    def verified(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "integrand": self.integrand,
            "var": self.var,
            "lower": self.lower,
            "upper": self.upper,
            "converges": self.converges,
            "value": self.value,
            "approx": self.approx,
            "principal_value": self.principal_value,
            "singularities": list(self.singularities),
            "pieces": [list(p) for p in self.pieces],
            "behaviour": [dict(b) for b in self.behaviour],
            "numeric": dict(self.numeric) if self.numeric else None,
            "sympy_value": self.sympy_value,
            "proof": self.proof,
            "certificate": self.certificate,
            "verdict": "verified",
        }


# --- convergence ----------------------------------------------------------------------

def _limit(e: sp.Expr, t: sp.Symbol, to: sp.Expr = sp.Integer(0)) -> "sp.Expr | None":
    try:
        L = sp.limit(e, t, to, "+") if to == 0 else sp.limit(e, t, to)
    except (NotImplementedError, ValueError, sp.PoleError, TypeError, AttributeError):
        return None
    if L.has(sp.Limit, sp.nan, sp.zoo) or isinstance(L, sp.AccumBounds) or L.has(sp.AccumBounds):
        return None
    return L


def _sign_of_limit(L: sp.Expr) -> "int | None":
    if L == sp.oo:
        return 1
    if L == -sp.oo:
        return -1
    return _sign_at(L)


def _side(f: sp.Expr, x: sp.Symbol, point: sp.Expr, side: str) -> dict[str, Any]:
    """Convergence of ∫ f near ``point`` on ``side`` ('+' right of it, '-' left; for
    ±oo the side is the one inside the interval)."""
    t = sp.Symbol("t", positive=True)
    rec: dict[str, Any] = {"point": _s(point), "side": side}
    if point in (sp.oo, -sp.oo):
        g = f.subs(x, t if point == sp.oo else -t)
        for p in _P_INFINITE:
            L = _limit(t ** p * g, t, sp.oo)
            if L is not None and L.is_finite:
                rec.update(test=f"lim x^{p}·f = {_s(L)} (finite, p > 1)", verdict="converges")
                return rec
        L1 = _limit(t * g, t, sp.oo)
        if L1 is not None and not (L1.is_finite and _sign_at(L1) == 0):
            s = _sign_of_limit(L1)
            if s:
                rec.update(test=f"lim x·f = {_s(L1)} (nonzero)", verdict="diverges", sign=s)
                return rec
        return _dirichlet(g, t, rec)
    g = f.subs(x, point + t if side == "+" else point - t)
    for p in _P_FINITE:
        L = _limit(t ** p * g, t)
        if L is not None and L.is_finite:
            rec.update(test=f"lim t^{p}·f = {_s(L)} (finite, p < 1)", verdict="converges")
            return rec
    L1 = _limit(t * g, t)
    if L1 is not None and not (L1.is_finite and _sign_at(L1) == 0):
        s = _sign_of_limit(L1)
        if s:
            rec.update(test=f"lim t·f = {_s(L1)} (nonzero)", verdict="diverges", sign=s,
                       simple_pole=bool(L1.is_finite), coefficient=_s(L1))
            return rec
    rec.update(test="no comparison exponent decided", verdict="unknown")
    return rec


def _trig_split(g: sp.Expr, t: sp.Symbol) -> "tuple[sp.Expr, sp.Expr, sp.Expr] | None":
    """``g = amp * trig(w*t + c)`` with one sin/cos factor of an affine argument."""
    factors = sp.Mul.make_args(g)
    trig = [fa for fa in factors if isinstance(fa, (sp.sin, sp.cos)) and fa.args[0].has(t)]
    if len(trig) != 1:
        return None
    arg = trig[0].args[0]
    try:
        P = sp.Poly(arg, t)
    except sp.PolynomialError:
        return None
    if P.degree() != 1 or P.domain not in (sp.ZZ, sp.QQ, sp.EX) or not P.LC().is_number:
        return None
    amp = sp.Mul(*[fa for fa in factors if fa is not trig[0]])
    if amp.has(sp.sin, sp.cos):
        return None
    return amp, trig[0], P.LC()


def _dirichlet(g: sp.Expr, t: sp.Symbol, rec: dict[str, Any]) -> dict[str, Any]:
    split = _trig_split(g, t)
    if split is None:
        rec.update(test="no comparison exponent decided", verdict="unknown")
        return rec
    amp, trig, w = split
    if not amp.has(t):
        if _sign_at(amp) != 0:
            rec.update(test=f"{_s(amp)}·{_s(trig)}: the partial integrals oscillate without a limit",
                       verdict="diverges", sign=0, oscillates=True)
            return rec
    L = _limit(amp, t, sp.oo)
    if L is None or _sign_at(L) != 0:
        rec.update(test="oscillating, amplitude does not tend to 0", verdict="unknown")
        return rec
    x = sp.Symbol("x", real=True)
    damp = sp.diff(amp, t).subs(t, x)
    for M in (sp.Integer(1), sp.Integer(8), sp.Integer(64)):
        if _tail_sign(damp, x, M) is not None:
            rec.update(test=f"Dirichlet: amplitude {_s(amp)} -> 0, monotone on [{M}, oo) "
                            f"(derivative sign proved by enclosure)", verdict="converges",
                       omega=_s(abs(w)))
            return rec
    rec.update(test="oscillating, amplitude not proved monotone", verdict="unknown")
    return rec


# --- quadrature -----------------------------------------------------------------------

def _mp_point(p: sp.Expr) -> mpmath.mpf:
    if p == sp.oo:
        return mpmath.inf
    if p == -sp.oo:
        return -mpmath.inf
    return mpmath.mpf(sp.N(p, mpmath.mp.dps + 10))


def _quadrature(f: sp.Expr, x: sp.Symbol, points: list[sp.Expr], omega: "sp.Expr | None") -> tuple[mpmath.mpf, mpmath.mpf, str]:
    """Value and error estimate: tanh-sinh at 30 and 45 digits over the same pieces."""
    results = []
    for dps in (_DPS_LO, _DPS_HI):
        with mpmath.workdps(dps):
            fn = sp.lambdify(x, f, modules="mpmath")

            def F(v: Any, fn: Callable[[Any], Any] = fn) -> Any:
                y = fn(v)
                if isinstance(y, mpmath.mpc):
                    if abs(y.imag) > mpmath.mpf(10) ** (-dps + 5) * max(1, abs(y.real)):
                        raise DomainError("integrand is not real on the interval")
                    return y.real
                return y

            pts = [_mp_point(p) for p in points]
            total = mpmath.mpf(0)
            err = mpmath.mpf(0)
            for lo, hi in zip(pts, pts[1:]):
                if omega is not None and hi == mpmath.inf:
                    v = mpmath.quadosc(F, [lo, hi], omega=_mp_point(omega))
                    e = mpmath.mpf(0)
                elif omega is not None and lo == -mpmath.inf:
                    v = mpmath.quadosc(F, [lo, hi], omega=_mp_point(omega))
                    e = mpmath.mpf(0)
                else:
                    v, e = mpmath.quad(F, [lo, hi], error=True, maxdegree=8 if dps == _DPS_LO else 10)
                total += v
                err += e
            results.append((total, err))
    (v1, e1), (v2, e2) = results
    with mpmath.workdps(_DPS_HI):
        err = max(abs(v1 - v2), e2, e1 / 10)
    method = "tanh-sinh 30/45 digits" + (" + quadosc" if omega is not None else "")
    return v2, err, method


# --- the entry point ------------------------------------------------------------------

def _check_integrand(f: sp.Expr) -> None:
    for node in sp.preorder_traversal(f):
        if isinstance(node, sp.Function) and not isinstance(node, _INTEGRAND_FUNCS):
            raise UnsupportedFormError(
                f"definite_integral: the function {node.func.__name__} is not supported in a "
                f"certified integral — use nintegrate", route="nintegrate")


def _finite_closed_form(v: sp.Expr) -> bool:
    return not (v.has(sp.Integral, sp.nan, sp.zoo, sp.oo, -sp.oo, sp.Limit)
                or isinstance(v, sp.AccumBounds) or v.free_symbols)


@under_budget("integrate")
def definite_integral(
    f: "str | sp.Expr",
    var: str,
    a: "str | int | sp.Expr",
    b: "str | int | sp.Expr",
    *,
    principal_value: bool = False,
) -> DefiniteIntegralResult:
    """``∫_a^b f d(var)`` — a cross-checked value, a PROVED divergence, or a refusal.

        >>> r = definite_integral("exp(-x^2)", "x", "-oo", "oo")
        >>> r.value, r.proof
        ('sqrt(pi)', 'cross-check')
        >>> d = definite_integral("1/x^2", "x", -1, 1)
        >>> d.converges, d.value, d.singularities
        (False, 'oo', ('0',))

    ``a``/``b`` may be ``"oo"``/``"-oo"``. ``principal_value=True`` asks for the Cauchy
    principal value across interior simple poles (otherwise such an integral is
    divergent without a value). Raises ``IntegralUndecided`` (route ``nintegrate``)
    when nothing can be certified, ``IntegralNoClosedForm`` when the integral provably
    converges but has no closed form here, ``DomainError`` when the integrand is not
    real on part of the interval.
    """
    x = _real_symbol(var)
    fx = _exact(_to_real(_parse_expr(f, "definite_integral"), x), "definite_integral")
    lo = _parse_expr(a, "lower bound")
    hi = _parse_expr(b, "upper bound")
    for v, what in ((lo, "lower bound"), (hi, "upper bound")):
        if v.free_symbols:
            raise UnsupportedFormError(f"definite_integral: the {what} must be a number, not {v}")
        if v in (sp.oo, -sp.oo):
            continue
        if v.has(sp.nan, sp.zoo) or v.is_real is False or v.is_finite is False:
            raise DomainError(f"definite_integral: the {what} must be real or ±oo, not {v}")
    if fx.free_symbols - {x}:
        raise UnsupportedFormError(
            f"definite_integral: parameters are not supported "
            f"({sorted(map(str, fx.free_symbols - {x}))}) — substitute numbers")
    _check_integrand(fx)
    shown = (_s(fx), var, _s(lo), _s(hi))
    if lo == hi:
        return DefiniteIntegralResult(*shown, True, "0", "0", principal_value, (), (), (), None, "0",
                                      "proof", "empty interval: the integral is 0 by definition")
    flip = bool(_lt(hi, lo))
    if flip:
        lo, hi = hi, lo
    res = _integrate_ordered(fx, x, lo, hi, principal_value)
    if flip:
        neg = (lambda s: None if s is None else _s(-sp.sympify(s)))
        res = DefiniteIntegralResult(
            shown[0], shown[1], shown[2], shown[3], res.converges,
            neg(res.value) if res.value not in ("oo", "-oo") else ("-oo" if res.value == "oo" else "oo"),
            neg(res.approx), res.principal_value, res.singularities, res.pieces, res.behaviour,
            res.numeric, neg(res.sympy_value), res.proof,
            res.certificate + "; bounds were reversed, so the sign is flipped")
    else:
        res = DefiniteIntegralResult(shown[0], shown[1], shown[2], shown[3], *_tail_fields(res))
    return res


def _tail_fields(r: DefiniteIntegralResult) -> tuple[Any, ...]:
    return (r.converges, r.value, r.approx, r.principal_value, r.singularities, r.pieces,
            r.behaviour, r.numeric, r.sympy_value, r.proof, r.certificate)


def _integrate_ordered(f: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr,
                       pv: bool) -> DefiniteIntegralResult:
    cons = _constraints(f, x)
    cuts: list[sp.Expr] = []
    try:
        singular_set: list[sp.Expr] = []
        for c in cons:
            z, _ = _zeros_in(c.g, x, lo, hi)
            cuts += z
            if c.need != "nonneg":
                singular_set += z
        for node in sp.preorder_traversal(f):
            if isinstance(node, (sp.Abs, sp.sign)) and node.args[0].has(x):
                cuts += _zeros_in(node.args[0], x, lo, hi)[0]
            if isinstance(node, sp.atan2):
                # atan2(y, x) jumps where x < 0 and y crosses 0: split there
                cuts += _zeros_in(node.args[0], x, lo, hi)[0]
    except SolveSetUndecided as exc:
        raise IntegralUndecided(f"definite_integral: cannot locate the singular points: {exc}",
                                reason="singularities") from exc
    ends = [p for p in (lo, hi) if p.is_finite]
    pts = _sorted_unique(cuts + ends)
    seq = ([lo] if not lo.is_finite else []) + pts + ([hi] if not hi.is_finite else [])
    singular = _sorted_unique(singular_set) if singular_set else []
    # every piece must lie in the real domain
    for p, q in zip(seq, seq[1:]):
        s = _rational_between(p, q)
        for c in cons:
            try:
                sg = _constraint_sign_on_cell(c.g, x, p, q, s)
            except SolveSetUndecided:
                sg = None
            if sg is None:
                raise IntegralUndecided(
                    f"definite_integral: cannot prove that the integrand is real on ({p}, {q})",
                    reason="domain")
            if not c.holds(sg):
                raise DomainError(
                    f"definite_integral: the integrand is not real on ({_s(p)}, {_s(q)}) "
                    f"({c.text(x)} fails) — the integral is not a real number")
    # convergence at every singular or infinite end of every piece
    behaviour: list[dict[str, Any]] = []
    sing_keys = {_s(p) for p in singular}
    for p, q in zip(seq, seq[1:]):
        if p == -sp.oo:
            behaviour.append(_side(f, x, -sp.oo, "+"))
        elif _s(p) in sing_keys:
            behaviour.append(_side(f, x, p, "+"))
        if q == sp.oo:
            behaviour.append(_side(f, x, sp.oo, "-"))
        elif _s(q) in sing_keys:
            behaviour.append(_side(f, x, q, "-"))
    pieces = tuple((_s(p), _s(q)) for p, q in zip(seq, seq[1:]))
    sing_out = tuple(_s(p) for p in singular)
    divergent = [r for r in behaviour if r["verdict"] == "diverges"]
    unknown = [r for r in behaviour if r["verdict"] == "unknown"]
    if divergent and not unknown:
        return _divergent(f, x, lo, hi, seq, pieces, sing_out, behaviour, divergent, pv)
    if unknown:
        where = ", ".join(f"{r['point']}{r['side']}" for r in unknown)
        approx, err = _try_number(f, x, seq, behaviour)
        raise IntegralUndecided(
            f"definite_integral: convergence at {where} is not decided by the comparison "
            f"tests — no value is certified", approx=approx, error=err, reason="convergence")
    omega = next((sp.sympify(r["omega"]) for r in behaviour if "omega" in r), None)
    if omega is None and not (lo.is_finite and hi.is_finite):
        omega = _trig_frequency(f, x)
    return _convergent(f, x, seq, pieces, sing_out, behaviour, omega, pv)


def _trig_frequency(f: sp.Expr, x: sp.Symbol) -> "sp.Expr | None":
    """Smallest angular frequency among sin/cos/tan of affine arguments (None if none):
    an oscillating integrand on an infinite range is integrated with ``quadosc``, which
    tanh-sinh is not built for (measured: sin(x)^2/x^2 on R came back ± 8e-4)."""
    ws = []
    for n in sp.preorder_traversal(f):
        if isinstance(n, (sp.sin, sp.cos, sp.tan)) and n.args[0].has(x):
            try:
                P = sp.Poly(n.args[0], x)
            except sp.PolynomialError:
                return None
            if P.degree() != 1 or not P.LC().is_number:
                return None
            ws.append(abs(P.LC()))
    return min(ws) if ws else None


def _try_number(f: sp.Expr, x: sp.Symbol, seq: list[sp.Expr], behaviour: list[dict[str, Any]]) -> tuple["str | None", "str | None"]:
    try:
        v, e, _ = _quadrature(f, x, seq, None)
    except Exception:  # noqa: BLE001 — a number is a courtesy on the refusal path
        return None, None
    return mpmath.nstr(v, 15), mpmath.nstr(e, 3)


def _divergent(f: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr, seq: list[sp.Expr],
               pieces: tuple[tuple[str, str], ...], sing: tuple[str, ...],
               behaviour: list[dict[str, Any]], divergent: list[dict[str, Any]],
               pv: bool) -> DefiniteIntegralResult:
    shown = (_s(f), x.name, _s(lo), _s(hi))
    if pv:
        r = _principal_value(f, x, lo, hi, seq, pieces, sing, behaviour, divergent)
        if r is not None:
            return r
    signs = {r["sign"] for r in divergent}
    if signs == {1}:
        value: "str | None" = "oo"
    elif signs == {-1}:
        value = "-oo"
    else:
        value = None
    what = {"oo": "diverges to +oo", "-oo": "diverges to -oo", None: "diverges and has no value"}[value]
    detail = "; ".join(f"at {r['point']}{r['side']}: {r['test']}" for r in divergent)
    note = ""
    if value is None and any(r.get("simple_pole") for r in divergent) and not pv:
        note = " (a Cauchy principal value may exist: ask with principal_value=True)"
    return DefiniteIntegralResult(
        *shown, False, value, None, pv, sing, pieces, tuple(behaviour), None, None, "proof",
        f"the integral {what} by the limit comparison test — {detail}{note}")


def _principal_value(f: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr, seq: list[sp.Expr],
                     pieces: tuple[tuple[str, str], ...], sing: tuple[str, ...],
                     behaviour: list[dict[str, Any]], divergent: list[dict[str, Any]]) -> "DefiniteIntegralResult | None":
    """Cauchy PV across interior simple poles whose two sides cancel, else None."""
    by_point: dict[str, list[dict[str, Any]]] = {}
    for r in divergent:
        by_point.setdefault(r["point"], []).append(r)
    poles = []
    for pt, recs in by_point.items():
        if len(recs) != 2 or not all(r.get("simple_pole") for r in recs):
            return None
        cl, cr = (sp.sympify(recs[0]["coefficient"]), sp.sympify(recs[1]["coefficient"]))
        if _sign_at(cl + cr) != 0:
            return None
        poles.append(sp.sympify(pt))
    # symbolic: SymPy's own principal value
    try:
        V = sp.Integral(f, (x, lo, hi)).principal_value()
    except Exception:  # noqa: BLE001
        V = None
    # numeric: around each pole s, ∫_0^d (f(s+t) + f(s-t)) dt is regular
    t = sp.Symbol("t", positive=True)
    pts = list(seq)
    total_lo = None
    try:
        results = []
        for dps in (_DPS_LO, _DPS_HI):
            with mpmath.workdps(dps):
                tot = mpmath.mpf(0)
                errs = mpmath.mpf(0)
                # half-width around each pole: a third of the distance to its neighbours
                for s in poles:
                    i = pts.index(s)
                    left, right = pts[i - 1], pts[i + 1]
                    d = sp.Rational(1, 3) * min(s - left if left.is_finite else 1, right - s if right.is_finite else 1)
                    sym = sp.lambdify(t, sp.together(f.subs(x, s + t) + f.subs(x, s - t)), "mpmath")
                    v, e = mpmath.quad(sym, [0, _mp_point(d)], error=True)
                    tot += v
                    errs += e
                    pts[i] = (s - d, s + d)  # type: ignore[call-overload]
                flat: list[sp.Expr] = []
                for p in pts:
                    if isinstance(p, tuple):
                        flat.append(p[0])
                        flat.append(None)  # type: ignore[arg-type]
                        flat.append(p[1])
                    else:
                        flat.append(p)
                fn = sp.lambdify(x, f, "mpmath")
                seg: list[sp.Expr] = []
                for p in flat + [None]:  # type: ignore[list-item]
                    if p is None:
                        for u, w in zip(seg, seg[1:]):
                            v, e = mpmath.quad(fn, [_mp_point(u), _mp_point(w)], error=True)
                            tot += v
                            errs += e
                        seg = []
                    else:
                        seg.append(p)
                results.append((tot, errs))
                pts = list(seq)
        (v1, e1), (v2, e2) = results
        err = max(abs(v1 - v2), e2)
        total_lo = v2
    except Exception:  # noqa: BLE001
        return None
    if V is None or not _finite_closed_form(V):
        raise IntegralUndecided("definite_integral: no closed form for the principal value",
                                approx=mpmath.nstr(total_lo, 15), error=mpmath.nstr(err, 3), reason="pv")
    with mpmath.workdps(_DPS_HI):
        gap = abs(mpmath.mpf(sp.N(V, _DPS_HI)) - total_lo)
        scale = max(mpmath.mpf(1), abs(total_lo))
        if err > _REL_ERR * scale or gap > 10 * err + mpmath.mpf(10) ** -25 * scale:
            raise IntegralUndecided(
                f"definite_integral: the principal value {V} disagrees with the symmetric "
                f"quadrature {mpmath.nstr(total_lo, 15)} ± {mpmath.nstr(err, 3)}",
                approx=mpmath.nstr(total_lo, 15), error=mpmath.nstr(err, 3), sympy_value=_s(V),
                reason="disagree")
    return DefiniteIntegralResult(
        _s(f), x.name, _s(lo), _s(hi), False, _s(V), sp.sstr(sp.N(V, 20)), True, sing, pieces,
        tuple(behaviour),
        {"value": mpmath.nstr(total_lo, 25), "error": mpmath.nstr(err, 3),
         "method": "symmetric tanh-sinh around each pole, 30/45 digits"},
        _s(V), "cross-check",
        "the integral itself diverges (simple poles at " + ", ".join(_s(p) for p in poles)
        + "); its Cauchy principal value, requested explicitly, agrees with a symmetric quadrature")


def _by_antiderivative(f: sp.Expr, x: sp.Symbol, p: sp.Expr, r: sp.Expr) -> "sp.Expr | None":
    """FTC on ONE piece, used when SymPy's definite integration gives up.

    The antiderivative F must differentiate back to f (``check_equal`` VERIFIED) and
    must have no singular point strictly inside (p, r) — the pieces were cut at every
    singular point of f, and F's own are checked here — so F is continuous on the open
    piece and the value is ``F(r-) - F(p+)`` (one-sided limits). The quadrature
    cross-check still follows.
    """
    from ..core.ir import Expr
    from ..verify.equal import check_equal

    try:
        F = sp.integrate(f, x)
    except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
        return None
    if isinstance(F, sp.Piecewise):
        F = _branch_on(F, x, p, r)
        if F is None:
            return None
    if F.has(sp.Integral) or F.has(sp.Piecewise) or not F.has(x):
        return None
    if not _derivative_matches(F, f, x, p, r):
        return None
    try:
        jumps = [c.g for c in _constraints(F, x)]
        # atan2(y, x) jumps where y changes sign: such a point inside the piece makes F
        # discontinuous there, and the FTC would step over the jump
        jumps += [n.args[0] for n in sp.preorder_traversal(F) if isinstance(n, sp.atan2)]
        jumps += [n.args[0] for n in sp.preorder_traversal(F) if isinstance(n, (sp.sign, sp.floor, sp.ceiling))]
        for g in jumps:
            z, _ = _zeros_in(g, x, p, r)
            if any(_lt(p, w) and _lt(w, r) for w in z):
                return None
    except SolveSetUndecided:
        return None
    t = sp.Symbol("t", positive=True)
    # SymPy's limit cannot take atan2 (measured: NotImplementedError "MRV set ... several
    # variables"). On the OPEN piece atan2(y, x) = 2*atan(y/(sqrt(x^2+y^2) + x)), an
    # identity everywhere off the cut {y = 0, x <= 0}; the piece contains no zero of y
    # (checked above), so the one-sided limits of the rewritten F are those of F.
    F = F.replace(sp.atan2, lambda y, xx: 2 * sp.atan(y / (sp.trigsimp(sp.sqrt(xx**2 + y**2)) + xx)))
    ends = []
    for e, side in ((r, "-"), (p, "+")):
        if e in (sp.oo, -sp.oo):
            L = _limit(F.subs(x, t if e == sp.oo else -t), t, sp.oo)
        else:
            L = _limit(F.subs(x, e - t if side == "-" else e + t), t)
        if L is None or not L.is_finite:
            return None
        ends.append(L)
    return sp.simplify(ends[0] - ends[1])


def _derivative_matches(F: sp.Expr, f: sp.Expr, x: sp.Symbol, p: sp.Expr, r: sp.Expr) -> bool:
    """``F' = f`` on the piece: ``check_equal`` VERIFIED, or ``F' - f`` simplifying to 0
    under the sign the piece gives the variable (``Abs(x) = x`` on a piece in [0, oo))."""
    from ..core.ir import Expr
    from ..verify.equal import check_equal

    d = sp.diff(F, x) - f
    plain = sp.Symbol(x.name)
    v = check_equal(Expr(sp.diff(F, x).xreplace({x: plain})), Expr(f.xreplace({x: plain})),
                    vars=[x.name])
    if v.verified:
        return True
    if v.refuted:
        return False
    if not _lt(p, 0):
        y = sp.Symbol("y", positive=True)
        return sp.simplify(d.subs(x, y)) == 0
    if not _lt(0, r):
        y = sp.Symbol("y", negative=True)
        return sp.simplify(d.subs(x, y)) == 0
    return False


def _branch_on(F: sp.Piecewise, x: sp.Symbol, p: sp.Expr, r: sp.Expr) -> "sp.Expr | None":
    """The branch of a Piecewise antiderivative valid on the WHOLE open piece (p, r):
    every condition before it must be a relation whose boundary does not cross the
    piece, so that the same branch is selected at every point of it."""
    q = _rational_between(p, r)
    for expr, cond in F.args:
        if cond is sp.true:
            return expr
        if not isinstance(cond, sp.Rel):
            return None
        g = cond.lhs - cond.rhs
        try:
            z, _ = _zeros_in(g, x, p, r)
        except SolveSetUndecided:
            return None
        if any(_lt(p, w) and _lt(w, r) for w in z):
            return None
        val = cond.subs(x, q)
        if val is sp.true:
            return expr
        if val is not sp.false:
            return None
    return None


def _closed_form(f: sp.Expr, x: sp.Symbol, seq: list[sp.Expr], route: str) -> "sp.Expr | None":
    """Sum over the pieces of SymPy's definite integral (``route="definite"``, falling
    back to the antiderivative per piece when SymPy gives up) or of the FTC route alone
    (``route="antiderivative"``)."""
    V: sp.Expr = sp.Integer(0)
    for p, r in zip(seq, seq[1:]):
        piece: "sp.Expr | None" = None
        if route == "definite":
            try:
                piece = sp.integrate(f, (x, p, r))
            except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
                piece = None
            if piece is not None and not _finite_closed_form(piece):
                piece = None
        if piece is None:
            piece = _by_antiderivative(f, x, p, r)
        if piece is None:
            return None
        V = V + piece
    V = sp.nsimplify(V) if V.has(sp.Float) else V
    try:
        V = sp.simplify(V)
    except Exception:  # noqa: BLE001
        pass
    return V


def _bose_fermi(f: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr) -> "tuple[sp.Expr, str] | None":
    """The Bose–Einstein / Fermi–Dirac family on (0, oo), which SymPy does not close (0.5.1):

        ∫_0^oo x^(s-1) / (e^(c x) - 1) dx = Γ(s) ζ(s) / c^s          (s > 1)
        ∫_0^oo x^(s-1) / (e^(c x) + 1) dx = (1 - 2^(1-s)) Γ(s) ζ(s) / c^s = Γ(s) η(s) / c^s   (s > 0)

    (expand ``1/(e^t ∓ 1)`` in the geometric series of ``e^(-n t)`` and integrate
    term by term: ``∫ t^(s-1) e^(-n t) dt = Γ(s)/n^s``, summed to ζ(s) or η(s)).
    Only that exact shape — a constant times ``x^a`` over a constant times
    ``exp(c x) ± 1`` with ``c > 0``, ``s = a + 1`` real. The caller treats the value
    like SymPy's: convergence was proved first, and it must agree with the
    independent quadrature or the answer is refused. ``None`` for anything else.
    """
    if lo != 0 or hi != sp.oo:
        return None
    num, den = sp.fraction(sp.together(f))
    k_num, xa = num.as_independent(x, as_Add=False)
    if xa == 1:
        a: sp.Expr = sp.Integer(0)
    elif xa == x:
        a = sp.Integer(1)
    elif xa.is_Pow and xa.base == x and not xa.exp.has(x):
        a = xa.exp
    else:
        return None
    k_den, rest = den.as_independent(x, as_Add=False)
    if not rest.is_Add or len(rest.args) != 2:
        return None
    const, ex = rest.as_independent(x, as_Add=True)
    scale, ee = ex.as_coeff_Mul()
    if not isinstance(ee, sp.exp) or scale == 0:
        return None
    sign = const / scale
    c = sp.simplify(ee.args[0] / x)
    if c.has(x) or not (c.is_real and c.is_positive):
        return None
    s = sp.nsimplify(a + 1)
    if not s.is_real:
        return None
    if sign == -1 and s > 1:
        core, name = sp.gamma(s) * sp.zeta(s), "Bose–Einstein integral Γ(s)ζ(s)"
    elif sign == 1 and s > 0:
        core, name = sp.gamma(s) * sp.dirichlet_eta(s), "Fermi–Dirac integral (1-2^(1-s))Γ(s)ζ(s)"
    else:
        return None
    value = sp.simplify(k_num / (k_den * scale) * core / c**s)
    if not _finite_closed_form(value):
        return None
    return value, f"the {name} with s = {s}"


def _agrees(V: sp.Expr, q: mpmath.mpf, err: mpmath.mpf) -> bool:
    with mpmath.workdps(_DPS_HI):
        scale = max(mpmath.mpf(1), abs(q))
        nv = sp.N(V, _DPS_HI)
        re_, im_ = nv.as_real_imag()
        if abs(im_) > 10 * err + mpmath.mpf(10) ** -25 * scale:
            return False
        return bool(abs(mpmath.mpf(re_) - q) <= 10 * err + mpmath.mpf(10) ** -25 * scale)


def _convergent(f: sp.Expr, x: sp.Symbol, seq: list[sp.Expr], pieces: tuple[tuple[str, str], ...],
                sing: tuple[str, ...], behaviour: list[dict[str, Any]], omega: "sp.Expr | None",
                pv: bool) -> DefiniteIntegralResult:
    lo, hi = seq[0], seq[-1]
    try:
        q, err, method = _quadrature(f, x, seq, omega)
    except DomainError:
        raise
    except Exception as exc:  # noqa: BLE001 — mpmath has many ways to fail
        q, err, method = None, None, f"failed: {type(exc).__name__}"
    V = _closed_form(f, x, seq, "definite")
    route = "SymPy's definite integral"
    if V is None:
        family = _bose_fermi(f, x, lo, hi)
        if family is not None:
            V, route = family
    if q is None:
        raise IntegralUndecided(f"definite_integral: the quadrature failed ({method})",
                                sympy_value=None if V is None else _s(V), reason="quadrature")
    approx_s, err_s = mpmath.nstr(q, 20), mpmath.nstr(err, 3)
    with mpmath.workdps(_DPS_HI):
        accurate = err <= _REL_ERR * max(mpmath.mpf(1), abs(q))
    if V is None:
        raise IntegralNoClosedForm(
            f"definite_integral: the integral converges (proved) but no closed form was found; "
            f"quadrature gives {approx_s} ± {err_s}", approx=approx_s if accurate else None, error=err_s)
    if not accurate:
        raise IntegralUndecided(
            f"definite_integral: the quadrature error estimate {err_s} is too large to vouch for "
            f"{V}", approx=approx_s, error=err_s, sympy_value=_s(V), reason="accuracy")
    refuted: "sp.Expr | None" = None
    if not _agrees(V, q, err):
        # SymPy's definite answer is contradicted by the number. Measured on the
        # issue-tracker cases (#24004, #27050): its definite routine can return 0 where
        # the antiderivative, taken with ONE-SIDED limits on each cut piece, is right.
        refuted = V
        V = _closed_form(f, x, seq, "antiderivative")
        route = "the antiderivative on each piece (one-sided limits)"
        if V is None or not _agrees(V, q, err):
            raise IntegralUndecided(
                f"definite_integral: SymPy's closed form {refuted} ≈ {sp.N(refuted, 15)} DISAGREES "
                f"with the quadrature {approx_s} ± {err_s} — refusing both", approx=approx_s,
                error=err_s, sympy_value=_s(refuted), reason="disagree")
    conv = "; ".join(f"at {r['point']}{r['side']}: {r['test']}" for r in behaviour) or "proper integral"
    note = f"; SymPy's definite integral {refuted} was REFUTED by the quadrature" if refuted is not None else ""
    return DefiniteIntegralResult(
        _s(f), x.name, _s(lo), _s(hi), True, _s(V), sp.sstr(sp.N(V, 20)), pv, sing, pieces,
        tuple(behaviour), {"value": mpmath.nstr(q, 25), "error": err_s, "method": method},
        _s(refuted) if refuted is not None else _s(V), "cross-check",
        f"convergence: {conv}; the closed form from {route} agrees with an independent "
        f"quadrature ({method}) within its error estimate {err_s}{note}")
