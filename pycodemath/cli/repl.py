"""Pycodemath REPL — interactive entry door to the language.

You type math, you get a result. Prefixed commands let you reach
specific engine operations and the code generator.

Commands:
  <expression>            — parse and show the simplified form
  diff <expr> d<var>       — derivative with respect to a variable   (e.g. diff sin(x)*x dx)
  integrate <expr> d<var>  — symbolic integral            (e.g. integrate 2*x dx)
  solve <expr> for <var> [complex] — solve expr = 0 for var
                                    (real only by default; 'complex' =
                                     complex too; e.g. solve x^2-4 for x)
  limit <expr> for <var> to <p> [dir +|-] — limit as var → p (p may be oo/-oo)
                                          (e.g. limit sin(x)/x for x to 0)
  series <expr> for <var> [at <p>] [n <k>] — Taylor/Laurent series around p
                                          (without the O term; e.g. series exp(x) for x n 4)
  sum <expr> for <var> from <lo> to <hi> — symbolic summation (hi may be n or oo)
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
  nintegrate <expr> d<var> from <a> to <b> — numerical integration (e.g. nintegrate 2*x dx from 0 to 1)
  min <expr> for <var> at <x0> [method newton|bfgs]
                                        — 1D minimum (gradient descent by
                                          default; newton/bfgs = 2nd-order
                                          methods with Armijo line search)
                                          (e.g. min (x-3)^2 for x at 0 method newton)

  grad <expr> for <x,y,...>            — symbolic gradient (e.g. grad x^2*y for x,y)
  solve_nd <f1>; <f2> for <x,y> at <x0,y0> — nonlinear system via Newton
                                          (e.g. solve_nd x^2+y^2-4; x-y for x,y at 1,1)
  min_nd <expr> for <x,y> at <x0,y0> [method newton|bfgs]
                                        — minimum of several variables (bfgs
                                          copes where gradient descent
                                          stalls, e.g. the Rosenbrock valley)
                                          (e.g. min_nd (x-1)^2+(y+2)^2 for x,y at 0,0)

  dsolve <expr> for y(t)                — ODE symbolically: y'(t) = f(t,y)
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
"""

from __future__ import annotations

import re
import sys
from typing import Callable

from ..codegen.pipeline import generate, generate_system
from ..core.errors import PycodemathError
from ..core.ir import Matrix
from ..engine import linalg, numerics, ode, symbolic
from ..frontend.parser import parse

BANNER = "Pycodemath 0.2.0 — heavy math + Python. Type 'help' or 'quit'."


def _num(raw: str) -> float:
    """Turn a number token from a command into a ``float`` — bad input is a clear error.

    The ``_NUM`` pattern is intentionally loose (it lets through e.g. ``1e``), so
    correctness is enforced only here — with a readable message, instead of the raw
    ``ValueError`` from ``float()`` (contract: always PycodemathError).
    """
    try:
        return float(raw)
    except ValueError as exc:
        raise PycodemathError(f"invalid number: {raw!r}") from exc


# --- command handlers (receive the regex match, return text) --------
def _cmd_diff(m: re.Match) -> str:
    return str(symbolic.diff(parse(m["expr"]), m["var"]))


def _cmd_integrate(m: re.Match) -> str:
    return str(symbolic.integrate(parse(m["expr"]), m["var"]))


def _cmd_solve(m: re.Match) -> str:
    # The REPL cuts complex results by default: transcendental equations (e.g. cos(x)=2)
    # were spewing junk complex roots. The trailing 'complex' flag
    # restores the full set of roots.
    real = not bool(m["complex"])
    roots = symbolic.solve(parse(m["expr"]), m["var"], real=real)
    return ", ".join(str(r) for r in roots) if roots else "(no solutions)"


def _cmd_limit(m: re.Match) -> str:
    return str(
        symbolic.limit(
            parse(m["expr"]), m["var"], parse(m["to"]), dir=m["dir"] or "+"
        )
    )


def _cmd_series(m: re.Match) -> str:
    at = parse(m["at"]) if m["at"] else 0
    n = int(m["n"]) if m["n"] else 6
    return str(symbolic.series(parse(m["expr"]), m["var"], at=at, n=n))


def _cmd_sum(m: re.Match) -> str:
    return str(
        symbolic.summation(
            parse(m["expr"]), m["var"], parse(m["lo"]), parse(m["hi"])
        )
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


def _cmd_root(m: re.Match) -> str:
    return str(numerics.root_find(parse(m["expr"]), m["var"], _num(m["x0"])))


def _cmd_min(m: re.Match) -> str:
    method = (m["method"] or "gd").lower()
    return str(
        numerics.minimize(parse(m["expr"]), m["var"], _num(m["x0"]), method=method)
    )


def _cmd_nintegrate(m: re.Match) -> str:
    val = numerics.integrate_num(
        parse(m["expr"]), m["var"], _num(m["a"]), _num(m["b"])
    )
    return str(val)


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
    return ", ".join(f"{v} = {val}" for v, val in zip(names, sol))


def _cmd_min_nd(m: re.Match) -> str:
    names = _split_names(m["vars"])
    method = (m["method"] or "gd").lower()
    sol = numerics.minimize_nd(
        parse(m["expr"]), names, _split_floats(m["x0"]), method=method
    )
    return ", ".join(f"{v} = {val}" for v, val in zip(names, sol))


def _cmd_dsolve(m: re.Match) -> str:
    sols = ode.dsolve(parse(m["expr"]), m["func"], m["var"])
    return ", ".join(f"{m['func']}({m['var']}) = {s}" for s in sols)


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
        re.compile(r"^integrate\s+(?P<expr>.+)\s+d(?P<var>\w+)\s*$", re.IGNORECASE),
        _cmd_integrate,
        "Usage: integrate <expression> d<variable>   (e.g. integrate 2*x dx)",
    ),
    "solve": (
        re.compile(
            r"^solve\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)"
            r"(?:\s+(?P<complex>complex))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_solve,
        "Usage: solve <expression> for <variable> [complex]"
        "   (e.g. solve x^2-4 for x; real only by default)",
    ),
    "limit": (
        re.compile(
            r"^limit\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)\s+to\s+(?P<to>\S+)"
            r"(?:\s+dir\s+(?P<dir>[+-]))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_limit,
        "Usage: limit <expression> for <variable> to <point> [dir +|-]"
        "   (e.g. limit sin(x)/x for x to 0)",
    ),
    "series": (
        re.compile(
            r"^series\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)"
            r"(?:\s+at\s+(?P<at>\S+))?(?:\s+n\s+(?P<n>\d+))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_series,
        "Usage: series <expression> for <variable> [at <point>] [n <order>]"
        "   (e.g. series exp(x) for x n 4)",
    ),
    "sum": (
        re.compile(
            r"^sum\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)"
            r"\s+from\s+(?P<lo>\S+)\s+to\s+(?P<hi>\S+)\s*$",
            re.IGNORECASE,
        ),
        _cmd_sum,
        "Usage: sum <expression> for <variable> from <lo> to <hi>"
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
            rf"(?:\s+method\s+(?P<method>\w+))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_min,
        "Usage: min <expression> for <variable> at <x0> [method newton|bfgs]"
        "   (e.g. min (x-3)^2 for x at 0 method newton)",
    ),
    "nintegrate": (
        re.compile(
            rf"^nintegrate\s+(?P<expr>.+)\s+d(?P<var>\w+)"
            rf"\s+from\s+(?P<a>{_NUM})\s+to\s+(?P<b>{_NUM})\s*$",
            re.IGNORECASE,
        ),
        _cmd_nintegrate,
        "Usage: nintegrate <expression> d<variable> from <a> to <b>"
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
            rf"\s+at\s+(?P<x0>{_NUMS})(?:\s+method\s+(?P<method>\w+))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_min_nd,
        "Usage: min_nd <expression> for <x,y> at <x0,y0> [method newton|bfgs]"
        "   (e.g. min_nd (x-1)^2+(y+2)^2 for x,y at 0,0 method bfgs)",
    ),
    "dsolve": (
        re.compile(
            r"^dsolve\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)\s*$",
            re.IGNORECASE,
        ),
        _cmd_dsolve,
        "Usage: dsolve <expression> for <y>(<t>)   (e.g. dsolve y for y(t))",
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


def handle(line: str) -> str:
    """Handle one line of REPL input and return the response text."""
    line = line.strip()
    if not line:
        return ""

    if line.lower() in ("help", "?"):
        return __doc__ or ""

    token = line.split(None, 1)[0].lower()
    entry = _COMMANDS.get(token)
    if entry is not None:
        pattern, handler, usage = entry
        match = pattern.match(line)
        if match is None:
            return usage
        return handler(match)

    # default: parse and simplify
    return str(symbolic.simplify(parse(line)))


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
