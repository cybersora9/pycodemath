"""MCP server — Pycodemath as a math tool for any agent.

Exposes a single ``math_eval`` tool delegating to the REPL dispatcher
(``repl.handle``): the agent sends a concise command, gets an exact result
from SymPy/NumPy — instead of computing "in its head" or writing code. This is the
path that realizes the project goal of "cutting tokens" outside Claude Code (skill) — any
MCP client can plug in Pycodemath.

Launch (stdio transport):
    python -m pycodemath.cli.mcp_server      # or the script: pycodemath-mcp

The ``mcp`` dependency is optional (extra ``pycodemath[mcp]``).
"""

from __future__ import annotations

from . import repl

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # optional dependency missing — refusal deferred to run()
    # importing the module itself must not blow up with a traceback; a readable
    # message goes to whoever actually tries to start the server
    FastMCP = None  # type: ignore[assignment, misc]


def math_eval(command: str) -> str:
    """Compute math exactly (SymPy/NumPy engine) — one command, one result.

    Commands (power notation: ^ or **):
      <expression>                     — simplify (e.g. sin(x)^2 + cos(x)^2)
      diff <expr> d<var>               — derivative (e.g. diff sin(x)*x dx)
      integrate <expr> d<var>          — symbolic integral
      solve <expr> for <var>           — solve expr = 0
      code <expr>                      — optimized Python/NumPy code (CSE)
      matrix <A>                       — show the matrix, e.g. matrix [[1,2],[3,4]]
      det/inv/transpose/eig <A>        — matrix, e.g. det [[1,2],[3,4]]
      solve_system <A> = <b>           — linear system A x = b
      code_system <A> = <b>            — NumPy code solving the system
      root <expr> for <var> at <x0>    — root f=0 numerically
      nintegrate <expr> d<var> from <a> to <b> — numerical integration
      min <expr> for <var> at <x0> [method newton|bfgs] — 1D minimum (gradient descent by default; newton/bfgs = 2nd order + Armijo)
      grad <expr> for <x,y>            — symbolic gradient
      solve_nd <f1>; <f2> for <x,y> at <x0,y0> — nonlinear system (Newton)
      min_nd <expr> for <x,y> at <x0,y0> [method newton|bfgs] — minimum of several variables (bfgs copes e.g. with the Rosenbrock valley)
      dsolve <expr> for y(t)           — ODE symbolically: y' = f(t,y)
      ode <expr> for y(t) from <t0> to <t1> at <y0> [steps <n>] — ODE numerically (RK4)
      odestiff <expr> for y(t) from <t0> to <t1> at <y0> [steps <n>] — STIFF ODE (implicit BDF2 + Newton; e.g. odestiff -1000*(y-cos(t)) for y(t) from 0 to 1 at 0)
      odestiff_adaptive <expr> for y(t) from <t0> to <t1> at <y0> [rtol <r>] — STIFF ODE with adaptive step (adaptive BDF2; an order fewer steps than DOPRI5 on a stiff problem)
      ode_adaptive <expr> for y(t) from <t0> to <t1> at <y0> [rtol <r>] — ODE, adaptive step (DOPRI5)
      odedense <expr> for y(t) from <t0> to <t1> at <y0> at t=<point> [rtol <r>] — ODE, read y(t) at any point (dense output)
      odeevents <expr> for y(t) from <t0> to <t1> at <y0> zero <g> [dir <+1|-1>] [rtol <r>] [stop] — ODE + moments g(t,y)=0 (event detection; stop = terminal event, stop integration)

    Errors are returned as text "error: ..." (specific).
    """
    try:
        out = repl.handle(command)
    except Exception as exc:  # the agent gets text, not a traceback
        return f"error: {exc}"
    return out or "(empty result)"


server: "FastMCP | None" = None
if FastMCP is not None:
    server = FastMCP("pycodemath")
    server.tool()(math_eval)


def run() -> None:
    """Run the server over stdio (the default transport for MCP clients)."""
    if server is None:
        # SystemExit with a string: message on stderr + code 1, no traceback
        # (works the same for the pycodemath-mcp entry point and python -m)
        raise SystemExit(
            "error: the MCP server requires the 'mcp' package "
            "(install: pip install pycodemath[mcp])."
        )
    server.run()


if __name__ == "__main__":
    run()
