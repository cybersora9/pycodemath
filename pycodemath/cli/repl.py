"""Pycodemath REPL — interactive entry door to the language.

You type math, you get a result. Prefixed commands let you reach
specific engine operations and the code generator.

Commands:
  <expression>            — parse and show the simplified form
  diff <expr> d<var>       — derivative with respect to a variable   (e.g. diff sin(x)*x dx)
  integrate <expr> d<var> [budget <s>] — symbolic integral   (e.g. integrate 2*x dx)
  solve <expr> for <var> [complex] [budget <s>] — solve expr = 0 for var
                                    (real only by default; 'complex' =
                                     complex too; e.g. solve x^2-4 for x)
  limit <expr> for <var> to <p> [dir +|-] [budget <s>] — limit as var → p (p may be oo/-oo)
                                          (e.g. limit sin(x)/x for x to 0)
  series <expr> for <var> [at <p>] [n <k>] [budget <s>] — Taylor/Laurent series around p
                                          (without the O term; e.g. series exp(x) for x n 4)
  sum <expr> for <var> from <lo> to <hi> [budget <s>] — symbolic summation (hi may be n or oo)
                                          (e.g. sum k for k from 1 to n)
  code <expression>        — generate optimized Python/NumPy code

  matrix <A>               — show the matrix            (e.g. matrix [[1,2],[3,4]])
  det <A>                  — determinant               (e.g. det [[1,2],[3,4]])
  inv <A>                  — inverse matrix
  transpose <A>            — transpose
  eig <A>                  — eigenvalues
  solve_system <A> = <b>   — solve A x = b   (e.g. solve_system [[2,1],[1,3]] = [3,5])
  code_system <A> = <b>    — generate NumPy code solving the system

  root <expr> for <var> at <x0>         — root f=0 (e.g. root x^2-2 for x at 1)
  nintegrate <expr> d<var> from <a> to <b> [tol <t>] — numerical integration
                                          (tol = adaptive refinement to that
                                          absolute error; fixed 100-panel
                                          Simpson without it)
                                          (e.g. nintegrate 2*x dx from 0 to 1)
  min <expr> for <var> at <x0> [method newton|bfgs] [tol <t>] [max_iter <k>]
                                        — 1D minimum (gradient descent by
                                          default; newton/bfgs = 2nd-order
                                          methods with Armijo line search)
                                          (e.g. min (x-3)^2 for x at 0 method newton)

  grad <expr> for <x,y,...>            — symbolic gradient (e.g. grad x^2*y for x,y)
  solve_nd <f1>; <f2> for <x,y> at <x0,y0> — nonlinear system via Newton
                                          (e.g. solve_nd x^2+y^2-4; x-y for x,y at 1,1)
  min_nd <expr> for <x,y> at <x0,y0> [method newton|bfgs] [tol <t>] [max_iter <k>]
                                        — minimum of several variables (bfgs
                                          copes where gradient descent
                                          stalls, e.g. the Rosenbrock valley)
                                          (e.g. min_nd (x-1)^2+(y+2)^2 for x,y at 0,0)

  dsolve <expr> for y(t) [budget <s>]   — ODE symbolically: y'(t) = f(t,y)
                                          (e.g. dsolve y for y(t))
  ode <expr> for y(t) from <t0> to <t1> at <y0> [steps <n>]
                                        — ODE numerically (RK4, fixed step)
                                          (e.g. ode y for y(t) from 0 to 1 at 1)
  odestiff <expr> for y(t) from <t0> to <t1> at <y0> [steps <n>]
                                        — STIFF ODE (BDF2, implicit method
                                          + Newton per step; large step where
                                          explicit methods blow up)
  odestiff_adaptive <expr> for y(t) from <t0> to <t1> at <y0> [rtol <r>]
                                        — STIFF ODE with ADAPTIVE step
                                          (variable-step BDF2 + error
                                          estimate; an order fewer steps than
                                          DOPRI5 on a stiff problem)
  ode_adaptive <expr> for y(t) from <t0> to <t1> at <y0> [rtol <r>]
                                        — ODE numerically (Dormand-Prince 5(4),
                                          adaptive step)
  odedense <expr> for y(t) from <t0> to <t1> at <y0> at t=<point> [rtol <r>]
                                        — ODE with dense output: read y(t)
                                          at any point (DOPRI5 interpolant)
  odeevents <expr> for y(t) from <t0> to <t1> at <y0> zero <g> [dir <+1|-1>] [rtol <r>] [stop]
                                        — ODE + event detection: moments
                                          g(t,y)=0 (bisection on the interpolant);
                                          the stop flag = TERMINAL event
                                          (stop integration at the event)

  help                     — help
  quit / exit              — exit

Trailing options (module 11): the bracketed ``<key> <value>`` pairs above go at
the END of the command, in any order, each at most once — e.g.
``min x^4 for x at 1 tol 1e-5``, ``integrate 1/(x^5+x+1) dx budget 2``.
  tol <t>       target the answer's accuracy: min/min_nd stop when the gradient
                falls under t (default 1e-9); nintegrate refines panels until
                the estimated absolute error fits inside t (fixed rule without it).
                Must be finite and positive (tol inf would certify the untouched
                starting guess as "converged")
  max_iter <k>  iteration allowance of min/min_nd (default 10000) — a
                not-converged descent that was still progressing finishes with more
  budget <s>    seconds THIS symbolic call may take (default 120) — a smaller
                budget buys a fast typed refusal instead of a long wait; a bigger
                one buys more search. Must be finite and positive here: the
                Python API (pycodemath.time_budget) also accepts math.inf
An unknown or repeated option key, or an unreadable value, is a ParseError; a
readable value outside its domain (tol 0, budget -1) is a DomainError. Options
the engine measurably does not need are deliberately absent — lr (method
newton|bfgs is the fix that works), max_iter/tol on root and solve_nd (Newton
ends its runs itself), max_evals, atol (rtol already steers the ODE error).
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .. import __version__
from ..codegen.pipeline import generate, generate_system
from ..core.budget import time_budget
from ..core.errors import DomainError, ParseError
from ..core.ir import Matrix
from ..core.result import QuadratureResult, SolveResult
from ..engine import linalg, numerics, ode, symbolic
from ..frontend.parser import parse

# One version string in the project (``pycodemath.__version__``), interpolated
# here — a banner that has to be bumped by hand is a banner that eventually lies.
BANNER = f"Pycodemath {__version__} — heavy math + Python. Type 'help' or 'quit'."


def _num(raw: str) -> float:
    """Turn a number token from a command into a ``float`` — bad input is a clear error.

    The ``_NUM`` pattern is intentionally loose (it lets through e.g. ``1e``), so
    correctness is enforced only here — with a readable message, instead of the raw
    ``ValueError`` from ``float()`` (contract: always PycodemathError).
    """
    try:
        return float(raw)
    except ValueError as exc:
        raise ParseError(f"invalid number: {raw!r}") from exc


# --- trailing options (module 11) -----------------------------------------
# The knobs the Python API always had, reachable from the grammar at last —
# modules 5-10 deferred exactly this six times. Which knobs made it in and which
# stayed Python-only was decided by MEASUREMENT, not taste (module 11 journal):
#
#   IN   tol/max_iter on min and min_nd — gradient descent is the one iterator
#        whose honest failure is "ran out of iterations while still progressing";
#        measured rescues: `min x^4 for x at 1` converges at 678 594 iterations
#        and under tol 1e-5 at 1 456, the Rosenbrock valley at 42 826.
#   IN   tol on nintegrate — the only way to AIM module 5's error_estimate;
#        measured: sqrt(x) true error 8.1e-5 -> 2e-14, and a spike integral the
#        fixed grid gets 37x wrong comes back correct to 1e-13.
#   IN   budget on the six symbolic commands — TimeBudgetError's own message
#        says "raise the budget" and the wire had no way to obey it; measured
#        in both directions (a 5 s budget refuses a 13 s integral that a 60 s
#        budget answers; the true hang is bounded in 3 s instead of 120).
#   OUT  lr — measured on the Rosenbrock valley and 1000*x^2: it only trades one
#        non-convergence for another; `method newton|bfgs` (already reachable)
#        is the fix that works.
#   OUT  max_iter/tol on root and solve_nd — Newton self-terminates (quadratic
#        convergence, or the divergence/stagnation detectors, or the bisection
#        fallback); measured: max_iter 100 000 wasted every extra iteration on
#        the x^3-2x+2 cycle and MANUFACTURED a false root of x*exp(-x) at
#        x=745 where f underflows to exact zero. A knob whose only measured
#        effects are waste and harm stays out.
#   OUT  max_evals — unexhaustible on every probed integrand (the worst took
#        83 033 of the 100 000 default); atol — no outcome moved on the
#        crossing-zero probe, rtol (already reachable) steers the error;
#        n on nintegrate — dominated by tol at 21x the cost on the same input.
#
# Syntax: ``key value`` pairs at the END of the command, ANY order, each at most
# once — the shape `method newton` and `rtol 1e-8` already have, generalized.
# A malformed option is never silently ignored (module 2): an unknown key, a
# repeated key or an unreadable value is a ParseError (the LINE could not be
# read — no engine was consulted); a readable number outside its domain is a
# DomainError (the same class the engines themselves use for a bad value).
_OPTS = r"(?P<opts>(?:\s+[A-Za-z_]+\s+\S+)*)"


def _opt_tol(raw: str) -> float:
    # math.isfinite matters here, not just decoration: ``tol inf`` passed the old
    # ``not value > 0.0`` check (inf > 0.0 is True) and made the gradient exit test
    # (``res < tol``) trivially true on iteration ONE — ``min (x-3)^2 for x at 100
    # tol inf`` came back ``SolveResult(value=100.0, residual=194.0,
    # converged=True)``: the untouched starting guess, certified converged. The
    # same class of bad input ``_opt_budget`` already guards against below.
    value = _num(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise DomainError(
            f"the tolerance tol must be a positive finite number, got {raw}"
        )
    return value


def _opt_max_iter(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ParseError(f"invalid integer: {raw!r}") from exc
    if value < 1:
        raise DomainError(f"max_iter must be at least 1, got {raw}")
    return value


def _opt_budget(raw: str) -> float:
    # Stricter than ``time_budget`` on purpose: the Python API accepts math.inf
    # for a caller who can interrupt, but this grammar IS the wire, and module 9's
    # guarantee there is precisely that no call hangs forever.
    value = _num(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise DomainError(
            f"the budget must be a positive finite number of seconds, got {raw} "
            f"(the Python API — pycodemath.time_budget — accepts math.inf for a "
            f"caller who would rather wait forever)"
        )
    return value


_MIN_OPTS: dict[str, Callable[[str], object]] = {
    "tol": _opt_tol,
    "max_iter": _opt_max_iter,
}
_NINTEGRATE_OPTS: dict[str, Callable[[str], object]] = {"tol": _opt_tol}
_BUDGET_OPTS: dict[str, Callable[[str], object]] = {"budget": _opt_budget}


def _options(m: re.Match, allowed: dict[str, Callable[[str], object]]) -> dict[str, Any]:
    """Parse a command's trailing options into keyword arguments.

    The pattern (``_OPTS``) guarantees whole ``key value`` pairs, so a dangling
    key (``min ... tol``) never reaches here — it fails the pattern and earns the
    usage message, exactly as a half-formed ``method`` suffix always has. What
    DOES reach here is decided strictly: the keys are this command's whitelist
    and nothing else, because an option that is accepted and ignored would be the
    grammar lying about what it computed.
    """
    raw = (m.groupdict().get("opts") or "").split()
    parsed: dict[str, Any] = {}
    for key, value in zip(raw[0::2], raw[1::2]):
        key = key.lower()
        if key not in allowed:
            raise ParseError(
                f"unknown option {key!r} — this command takes: "
                + ", ".join(f"{k} <value>" for k in allowed)
            )
        if key in parsed:
            raise ParseError(f"option {key!r} given twice")
        parsed[key] = allowed[key](value)
    return parsed


def _budgeted(m: re.Match, run: Callable[[], str]) -> str:
    """Run a symbolic handler under the caller's ``budget <s>``, if any.

    ``time_budget`` REPLACES the 120 s default for everything inside the block
    (module 9's re-entrancy: the entry point's own guard inherits the caller's
    allowance instead of arming a second one), so the refusal quotes the number
    the command named.
    """
    budget = _options(m, _BUDGET_OPTS).get("budget")
    if budget is None:
        return run()
    with time_budget(budget):
        return run()


# --- command handlers (receive the regex match, return text) --------
def _cmd_diff(m: re.Match) -> str:
    return str(symbolic.diff(parse(m["expr"]), m["var"]))


def _cmd_integrate(m: re.Match) -> str:
    return _budgeted(m, lambda: str(symbolic.integrate(parse(m["expr"]), m["var"])))


def _cmd_solve(m: re.Match) -> str:
    # The REPL cuts complex results by default: transcendental equations (e.g. cos(x)=2)
    # were spewing junk complex roots. The trailing 'complex' flag
    # restores the full set of roots.
    def run() -> str:
        real = not bool(m["complex"])
        roots = symbolic.solve(parse(m["expr"]), m["var"], real=real)
        return ", ".join(str(r) for r in roots) if roots else "(no solutions)"

    return _budgeted(m, run)


def _cmd_limit(m: re.Match) -> str:
    return _budgeted(
        m,
        lambda: str(
            symbolic.limit(
                parse(m["expr"]), m["var"], parse(m["to"]), dir=m["dir"] or "+"
            )
        ),
    )


def _cmd_series(m: re.Match) -> str:
    def run() -> str:
        at = parse(m["at"]) if m["at"] else 0
        n = int(m["n"]) if m["n"] else 6
        return str(symbolic.series(parse(m["expr"]), m["var"], at=at, n=n))

    return _budgeted(m, run)


def _cmd_sum(m: re.Match) -> str:
    return _budgeted(
        m,
        lambda: str(
            symbolic.summation(
                parse(m["expr"]), m["var"], parse(m["lo"]), parse(m["hi"])
            )
        ),
    )


def _cmd_code(m: re.Match) -> str:
    return generate(parse(m["expr"])).source


def _cmd_matrix(m: re.Match) -> str:
    return str(Matrix(m["a"]))


def _cmd_det(m: re.Match) -> str:
    return str(linalg.det(Matrix(m["a"])))


def _cmd_inv(m: re.Match) -> str:
    return str(linalg.inv(Matrix(m["a"])))


def _cmd_transpose(m: re.Match) -> str:
    return str(linalg.transpose(Matrix(m["a"])))


def _cmd_eig(m: re.Match) -> str:
    vals = linalg.eigenvalues(Matrix(m["a"]))
    return ", ".join(str(v) for v in vals) if vals else "(none)"


def _cmd_solve_system(m: re.Match) -> str:
    return str(linalg.solve_system(Matrix(m["a"]), Matrix(m["b"])))


def _cmd_code_system(m: re.Match) -> str:
    return generate_system(Matrix(m["a"]), Matrix(m["b"])).source


# The five numerical commands are answered TWICE — once by the plain handler the
# REPL uses and once by the evidence handler the MCP surface uses (module 6). The
# two must print the SAME text for the same run, so the text is built here and
# nowhere else; ``test_mcp`` pins the equality mechanically over every one of them.
def _fmt_scalar(value: float) -> str:
    return str(value)


def _fmt_named(names: "Sequence[str]", values: "Sequence[float]") -> str:
    return ", ".join(f"{v} = {val}" for v, val in zip(names, values))


def _cmd_root(m: re.Match) -> str:
    return _fmt_scalar(numerics.root_find(parse(m["expr"]), m["var"], _num(m["x0"])))


def _cmd_min(m: re.Match) -> str:
    method = (m["method"] or "gd").lower()
    return _fmt_scalar(
        numerics.minimize(
            parse(m["expr"]),
            m["var"],
            _num(m["x0"]),
            method=method,
            **_options(m, _MIN_OPTS),
        )
    )


def _cmd_nintegrate(m: re.Match) -> str:
    val = numerics.integrate_num(
        parse(m["expr"]),
        m["var"],
        _num(m["a"]),
        _num(m["b"]),
        **_options(m, _NINTEGRATE_OPTS),
    )
    return _fmt_scalar(val)


def _split_names(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",")]


def _split_floats(raw: str) -> list[float]:
    return [_num(v.strip()) for v in raw.split(",")]


def _cmd_grad(m: re.Match) -> str:
    return str(numerics.gradient(parse(m["expr"]), _split_names(m["vars"])))


def _cmd_solve_nd(m: re.Match) -> str:
    exprs = [parse(e.strip()) for e in m["exprs"].split(";")]
    names = _split_names(m["vars"])
    sol = numerics.root_find_nd(exprs, names, _split_floats(m["x0"]))
    return _fmt_named(names, sol)


def _cmd_min_nd(m: re.Match) -> str:
    names = _split_names(m["vars"])
    method = (m["method"] or "gd").lower()
    sol = numerics.minimize_nd(
        parse(m["expr"]),
        names,
        _split_floats(m["x0"]),
        method=method,
        **_options(m, _MIN_OPTS),
    )
    return _fmt_named(names, sol)


def _cmd_dsolve(m: re.Match) -> str:
    def run() -> str:
        sols = ode.dsolve(parse(m["expr"]), m["func"], m["var"])
        return ", ".join(f"{m['func']}({m['var']}) = {s}" for s in sols)

    return _budgeted(m, run)


def _cmd_ode(m: re.Match) -> str:
    n = int(m["n"]) if m["n"] else 100
    ts, ys = ode.solve_ode_num(
        parse(m["expr"]),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        n,
        func=m["func"],
    )
    return f"{m['func']}({ts[-1]:g}) = {ys[-1]}"


def _cmd_ode_stiff(m: re.Match) -> str:
    n = int(m["n"]) if m["n"] else 100
    ts, ys = ode.solve_ode_stiff(
        parse(m["expr"]),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        n,
        func=m["func"],
    )
    return f"{m['func']}({ts[-1]:g}) = {ys[-1]}  (BDF2, {n} implicit steps)"


def _cmd_ode_stiff_adaptive(m: re.Match) -> str:
    rtol = _num(m["rtol"]) if m["rtol"] else 1e-6
    ts, ys = ode.solve_ode_stiff_adaptive(
        parse(m["expr"]),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        rtol=rtol,
        func=m["func"],
    )
    return (
        f"{m['func']}({ts[-1]:g}) = {ys[-1]}"
        f"  (adaptive BDF, {len(ts) - 1} implicit steps)"
    )


def _cmd_ode_adaptive(m: re.Match) -> str:
    rtol = _num(m["rtol"]) if m["rtol"] else 1e-6
    ts, ys = ode.solve_ode_adaptive(
        parse(m["expr"]),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        rtol=rtol,
        func=m["func"],
    )
    return f"{m['func']}({ts[-1]:g}) = {ys[-1]}  ({len(ts) - 1} adaptive steps)"


def _cmd_ode_dense(m: re.Match) -> str:
    rtol = _num(m["rtol"]) if m["rtol"] else 1e-6
    sol = ode.solve_ode_dense(
        parse(m["expr"]),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        rtol=rtol,
        func=m["func"],
    )
    tq = _num(m["tq"])
    return f"{m['func']}({tq:g}) = {sol(tq)}  (dense output from {len(sol.ts) - 1} steps)"


def _cmd_ode_events(m: re.Match) -> str:
    direction = int(m["dir"]) if m["dir"] else 0
    rtol = _num(m["rtol"]) if m["rtol"] else 1e-6
    terminal = bool(m["stop"])
    sol = ode.solve_ode_events(
        parse(m["expr"]),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        parse(m["g"]),
        direction=direction,
        terminal=terminal,
        rtol=rtol,
        func=m["func"],
    )
    if not sol.events:
        return "(no events)"
    listed = "; ".join(f"t={t:g} ({m['func']}={y:g})" for t, y in sol.events)
    if terminal:
        return f"stop at {listed}  (integration stopped at the event)"
    return listed


# --- command table --------------------------------------------------------
# Key = the first token of the line (lowercased). Value = (full pattern,
# handler, usage message). One pattern per command instead of manual
# slicing — each command's syntax is explicit and testable in one place.
_NUM = r"[-+]?[\d.eE+-]+"  # number for float(); bad input is reported by float()
_NAMES = r"\w+(?:\s*,\s*\w+)*"  # list of variables: x,y,z
_NUMS = rf"{_NUM}(?:\s*,\s*{_NUM})*"  # list of numbers: 1,-2,0.5

_COMMANDS: dict[str, tuple[re.Pattern[str], Callable[[re.Match], str], str]] = {
    "diff": (
        re.compile(r"^diff\s+(?P<expr>.+)\s+d(?P<var>\w+)\s*$", re.IGNORECASE),
        _cmd_diff,
        "Usage: diff <expression> d<variable>   (e.g. diff sin(x)*x dx)",
    ),
    "integrate": (
        re.compile(
            rf"^integrate\s+(?P<expr>.+)\s+d(?P<var>\w+){_OPTS}\s*$", re.IGNORECASE
        ),
        _cmd_integrate,
        "Usage: integrate <expression> d<variable> [budget <s>]"
        "   (e.g. integrate 2*x dx)",
    ),
    "solve": (
        re.compile(
            rf"^solve\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)"
            rf"(?:\s+(?P<complex>complex))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_solve,
        "Usage: solve <expression> for <variable> [complex] [budget <s>]"
        "   (e.g. solve x^2-4 for x; real only by default)",
    ),
    "limit": (
        re.compile(
            rf"^limit\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)\s+to\s+(?P<to>\S+)"
            rf"(?:\s+dir\s+(?P<dir>[+-]))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_limit,
        "Usage: limit <expression> for <variable> to <point> [dir +|-] [budget <s>]"
        "   (e.g. limit sin(x)/x for x to 0)",
    ),
    "series": (
        re.compile(
            rf"^series\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)"
            rf"(?:\s+at\s+(?P<at>\S+))?(?:\s+n\s+(?P<n>\d+))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_series,
        "Usage: series <expression> for <variable> [at <point>] [n <order>] [budget <s>]"
        "   (e.g. series exp(x) for x n 4)",
    ),
    "sum": (
        re.compile(
            rf"^sum\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)"
            rf"\s+from\s+(?P<lo>\S+)\s+to\s+(?P<hi>\S+){_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_sum,
        "Usage: sum <expression> for <variable> from <lo> to <hi> [budget <s>]"
        "   (e.g. sum k for k from 1 to n)",
    ),
    "code": (
        re.compile(r"^code\s+(?P<expr>.+?)\s*$", re.IGNORECASE),
        _cmd_code,
        "Usage: code <expression>   (e.g. code (sin(x)+cos(x))^2)",
    ),
    "matrix": (
        re.compile(r"^matrix\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_matrix,
        "Usage: matrix <A>   (e.g. matrix [[1,2],[3,4]])",
    ),
    "det": (
        re.compile(r"^det\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_det,
        "Usage: det <A>   (e.g. det [[1,2],[3,4]])",
    ),
    "inv": (
        re.compile(r"^inv\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_inv,
        "Usage: inv <A>   (e.g. inv [[4,7],[2,6]])",
    ),
    "transpose": (
        re.compile(r"^transpose\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_transpose,
        "Usage: transpose <A>   (e.g. transpose [[1,2],[3,4]])",
    ),
    "eig": (
        re.compile(r"^eig\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_eig,
        "Usage: eig <A>   (e.g. eig [[2,0],[0,3]])",
    ),
    "solve_system": (
        re.compile(
            r"^solve_system\s+(?P<a>.+?)\s*=\s*(?P<b>.+?)\s*$", re.IGNORECASE
        ),
        _cmd_solve_system,
        "Usage: solve_system <A> = <b>   (e.g. solve_system [[2,1],[1,3]] = [3,5])",
    ),
    "code_system": (
        re.compile(
            r"^code_system\s+(?P<a>.+?)\s*=\s*(?P<b>.+?)\s*$", re.IGNORECASE
        ),
        _cmd_code_system,
        "Usage: code_system <A> = <b>   (e.g. code_system [[2,1],[1,3]] = [3,5])",
    ),
    "root": (
        re.compile(
            rf"^root\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)\s+at\s+(?P<x0>{_NUM})\s*$",
            re.IGNORECASE,
        ),
        _cmd_root,
        "Usage: root <expression> for <variable> at <x0>   (e.g. root x^2-2 for x at 1)",
    ),
    "min": (
        re.compile(
            rf"^min\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)\s+at\s+(?P<x0>{_NUM})"
            rf"(?:\s+method\s+(?P<method>\w+))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_min,
        "Usage: min <expression> for <variable> at <x0> [method newton|bfgs]"
        " [tol <t>] [max_iter <k>]"
        "   (e.g. min (x-3)^2 for x at 0 method newton)",
    ),
    "nintegrate": (
        re.compile(
            rf"^nintegrate\s+(?P<expr>.+)\s+d(?P<var>\w+)"
            rf"\s+from\s+(?P<a>{_NUM})\s+to\s+(?P<b>{_NUM}){_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_nintegrate,
        "Usage: nintegrate <expression> d<variable> from <a> to <b> [tol <t>]"
        "   (e.g. nintegrate 2*x dx from 0 to 1)",
    ),
    "grad": (
        re.compile(
            rf"^grad\s+(?P<expr>.+)\s+for\s+(?P<vars>{_NAMES})\s*$", re.IGNORECASE
        ),
        _cmd_grad,
        "Usage: grad <expression> for <x,y,...>   (e.g. grad x^2*y for x,y)",
    ),
    "solve_nd": (
        re.compile(
            rf"^solve_nd\s+(?P<exprs>.+)\s+for\s+(?P<vars>{_NAMES})"
            rf"\s+at\s+(?P<x0>{_NUMS})\s*$",
            re.IGNORECASE,
        ),
        _cmd_solve_nd,
        "Usage: solve_nd <f1>; <f2> for <x,y> at <x0,y0>"
        "   (e.g. solve_nd x^2+y^2-4; x-y for x,y at 1,1)",
    ),
    "min_nd": (
        re.compile(
            rf"^min_nd\s+(?P<expr>.+)\s+for\s+(?P<vars>{_NAMES})"
            rf"\s+at\s+(?P<x0>{_NUMS})(?:\s+method\s+(?P<method>\w+))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_min_nd,
        "Usage: min_nd <expression> for <x,y> at <x0,y0> [method newton|bfgs]"
        " [tol <t>] [max_iter <k>]"
        "   (e.g. min_nd (x-1)^2+(y+2)^2 for x,y at 0,0 method bfgs)",
    ),
    "dsolve": (
        re.compile(
            rf"^dsolve\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\){_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_dsolve,
        "Usage: dsolve <expression> for <y>(<t>) [budget <s>]"
        "   (e.g. dsolve y for y(t))",
    ),
    "ode": (
        re.compile(
            rf"^ode\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"(?:\s+steps\s+(?P<n>\d+))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode,
        "Usage: ode <expression> for <y>(<t>) from <t0> to <t1> at <y0> [steps <n>]"
        "   (e.g. ode y for y(t) from 0 to 1 at 1)",
    ),
    "odestiff": (
        re.compile(
            rf"^odestiff\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"(?:\s+steps\s+(?P<n>\d+))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_stiff,
        "Usage: odestiff <expression> for <y>(<t>) from <t0> to <t1> at <y0> [steps <n>]"
        "   (e.g. odestiff -1000*(y-cos(t)) for y(t) from 0 to 1 at 0)",
    ),
    "odestiff_adaptive": (
        re.compile(
            rf"^odestiff_adaptive\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"(?:\s+rtol\s+(?P<rtol>{_NUM}))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_stiff_adaptive,
        "Usage: odestiff_adaptive <expression> for <y>(<t>) from <t0> to <t1> at <y0> [rtol <r>]"
        "   (e.g. odestiff_adaptive -1000*(y-cos(t)) for y(t) from 0 to 1 at 0)",
    ),
    "ode_adaptive": (
        re.compile(
            rf"^ode_adaptive\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"(?:\s+rtol\s+(?P<rtol>{_NUM}))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_adaptive,
        "Usage: ode_adaptive <expression> for <y>(<t>) from <t0> to <t1> at <y0> [rtol <r>]"
        "   (e.g. ode_adaptive y for y(t) from 0 to 1 at 1)",
    ),
    "odedense": (
        re.compile(
            rf"^odedense\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"\s+at\s+t\s*=\s*(?P<tq>{_NUM})(?:\s+rtol\s+(?P<rtol>{_NUM}))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_dense,
        "Usage: odedense <expression> for <y>(<t>) from <t0> to <t1> at <y0> at t=<point> [rtol <r>]"
        "   (e.g. odedense y for y(t) from 0 to 2 at 1 at t=0.37)",
    ),
    "odeevents": (
        re.compile(
            rf"^odeevents\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"\s+zero\s+(?P<g>.+?)(?:\s+dir\s+(?P<dir>[-+]?1))?"
            rf"(?:\s+rtol\s+(?P<rtol>{_NUM}))?(?:\s+(?P<stop>stop))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_events,
        "Usage: odeevents <expression> for <y>(<t>) from <t0> to <t1> at <y0> zero <g> [dir <+1|-1>] [rtol <r>] [stop]"
        "   (e.g. odeevents cos(t) for y(t) from 0.5 to 10 at 0.479 zero y)",
    ),
}


# --- evidence handlers (module 6) -----------------------------------------
# The twins of the five numerical ``_cmd_*`` handlers above, asking the engine for
# the structured result modules 3-5 built (``full_result=True``). They exist for the
# MCP surface, NOT for the REPL: under ``full_result`` an ITERATION FAILURE is
# RETURNED instead of raised, so a REPL that used them would print a last iterate
# where it prints an error today. ``handle`` therefore keeps calling the plain
# handlers, and only ``handle_full`` reaches for these.


@dataclass(frozen=True, slots=True)
class Answer:
    """One line of input answered: the text, plus the EVIDENCE behind it if any.

    ``evidence`` is ``None`` for every command that has no structured result to
    give (symbolic, linear algebra, code generation, ODE) — that is the honest
    report, not a gap: module 6 carries the results that EXIST, and does not
    invent one where the engine has none.
    """

    text: str
    evidence: "SolveResult[Any] | QuadratureResult | None" = None


def _solved(result: "SolveResult[Any]", text: str) -> Answer:
    """Pair a solver result with its text — appending the caveat when it FAILED.

    A client that renders only the text block must not read a last iterate as an
    answer, so the text says so. The structured half says the same thing in
    ``converged``/``status``; the two are built here together, once.
    """
    if result.converged:
        return Answer(text, result)
    # Module 8: "NOT CONVERGED" would be false for the one outcome where the run
    # DID converge — to a stationary point that turned out to be a maximum, a
    # saddle or an inflection. The caveat has to say which of the two happened, or
    # a text-only client reads "give it more iterations" where none can help.
    headline = "NOT A MINIMUM" if result.status == "not_a_minimum" else "NOT CONVERGED"
    return Answer(
        f"{text}  ({headline} — status {result.status}, residual "
        f"{result.residual:g} after {result.iterations} iterations)",
        result,
    )


def _integrated(result: QuadratureResult, text: str) -> Answer:
    """The quadrature twin of ``_solved`` — its own fields, for module 5's reason."""
    if result.converged:
        return Answer(text, result)
    return Answer(
        f"{text}  (NOT CONVERGED — status {result.status}, error estimate "
        f"{result.error_estimate:g} after {result.evaluations} evaluations)",
        result,
    )


def _ev_root(m: re.Match) -> Answer:
    res = numerics.root_find(
        parse(m["expr"]), m["var"], _num(m["x0"]), full_result=True
    )
    return _solved(res, _fmt_scalar(res.value))


def _ev_min(m: re.Match) -> Answer:
    method = (m["method"] or "gd").lower()
    res = numerics.minimize(
        parse(m["expr"]),
        m["var"],
        _num(m["x0"]),
        method=method,
        full_result=True,
        **_options(m, _MIN_OPTS),
    )
    return _solved(res, _fmt_scalar(res.value))


def _ev_nintegrate(m: re.Match) -> Answer:
    res = numerics.integrate_num(
        parse(m["expr"]),
        m["var"],
        _num(m["a"]),
        _num(m["b"]),
        full_result=True,
        **_options(m, _NINTEGRATE_OPTS),
    )
    return _integrated(res, _fmt_scalar(res.value))


def _ev_solve_nd(m: re.Match) -> Answer:
    exprs = [parse(e.strip()) for e in m["exprs"].split(";")]
    names = _split_names(m["vars"])
    res = numerics.root_find_nd(
        exprs, names, _split_floats(m["x0"]), full_result=True
    )
    return _solved(res, _fmt_named(names, res.value))


def _ev_min_nd(m: re.Match) -> Answer:
    names = _split_names(m["vars"])
    method = (m["method"] or "gd").lower()
    res = numerics.minimize_nd(
        parse(m["expr"]),
        names,
        _split_floats(m["x0"]),
        method=method,
        full_result=True,
        **_options(m, _MIN_OPTS),
    )
    return _solved(res, _fmt_named(names, res.value))


#: Command token -> the handler that also brings back evidence. Exactly the five
#: commands the engine can answer with a ``SolveResult`` / ``QuadratureResult``;
#: every other token falls through to its plain handler and reports no evidence.
_EVIDENCE: dict[str, Callable[[re.Match], Answer]] = {
    "root": _ev_root,
    "min": _ev_min,
    "nintegrate": _ev_nintegrate,
    "solve_nd": _ev_solve_nd,
    "min_nd": _ev_min_nd,
}


def _route(line: str) -> "tuple[str, re.Match] | str":
    """Resolve one input line to ``(command token, match)`` — or to final text.

    The ONE routing rule, shared by ``handle`` and ``handle_full`` so the two
    dispatchers cannot drift on what a line means (empty line, help, an unknown
    token that falls through to simplification, a known token whose syntax does
    not match and earns the usage message). A ``str`` is a finished answer with no
    command behind it; a tuple is a matched command the caller still has to run.
    """
    line = line.strip()
    if not line:
        return ""

    if line.lower() in ("help", "?"):
        return __doc__ or ""

    token = line.split(None, 1)[0].lower()
    entry = _COMMANDS.get(token)
    if entry is None:
        # default: parse and simplify
        return str(symbolic.simplify(parse(line)))

    pattern, _handler, usage = entry
    match = pattern.match(line)
    if match is None:
        return usage
    return token, match


def handle(line: str) -> str:
    """Handle one line of REPL input and return the response text."""
    routed = _route(line)
    if isinstance(routed, str):
        return routed
    token, match = routed
    return _COMMANDS[token][1](match)


def handle_full(line: str) -> Answer:
    """Handle one line and bring back the EVIDENCE too, where the engine has any.

    The agent-facing twin of ``handle`` (module 6), used by the MCP server. Same
    line, same syntax table, same text — plus the ``SolveResult`` /
    ``QuadratureResult`` of the five numerical commands.

    ``text`` is byte-identical to ``handle``'s for every line ``handle`` answers
    without raising; the two differ only where ``handle`` RAISES an iteration
    failure, which this returns as data with the caveat appended.

    THE EVIDENCE NEVER COSTS AN ANSWER. Asking for it takes extra samples (a
    residual at the returned point, a midpoint grid for the error estimate), and
    those samples can land outside the domain where the answer itself was fine —
    the engine then refuses with ``DomainError`` saying the value is available
    without ``full_result``. So that refusal is taken literally: fall back to the
    plain handler and report the answer with no evidence. A refusal of the INPUT
    raises from the plain handler too, and propagates unchanged.
    """
    routed = _route(line)
    if isinstance(routed, str):
        return Answer(routed)
    token, match = routed
    evidence = _EVIDENCE.get(token)
    if evidence is None:
        return Answer(_COMMANDS[token][1](match))
    try:
        return evidence(match)
    except DomainError:
        return Answer(_COMMANDS[token][1](match))


def run() -> None:
    """Run the REPL — interactively or one-shot from command-line arguments.

    ``python -m pycodemath "diff sin(x)*x dx"`` (or the ``pycodemath`` script)
    runs a single command and exits: result on stdout, error on stderr with exit
    code 1 — this is what makes Pycodemath scriptable (e.g. as a skill).
    """
    # The Windows console is sometimes cp1252 — force UTF-8 so non-ASCII characters work.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:  # history and line editing on POSIX; the Windows console has this natively
        import readline  # noqa: F401
    except ImportError:
        pass

    args = sys.argv[1:]
    if args:
        try:
            out = handle(" ".join(args))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1)
        if out:
            print(out)
        return

    print(BANNER)
    while True:
        try:
            line = input("pcm> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in ("quit", "exit"):
            break
        try:
            out = handle(line)
        except Exception as exc:  # in the REPL an error must not kill the session
            out = f"error: {exc}"
        if out:
            print(out)
