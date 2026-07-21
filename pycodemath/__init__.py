"""Pycodemath — heavy computational math + Python.

Top-level public API.
"""

from __future__ import annotations

from .codegen.pipeline import (
    CodeArtifact,
    generate,
    generate_ode,
    generate_ode_adaptive,
    generate_ode_dense,
    generate_system,
)
from .core.ir import E, Expr, M, Matrix, V, symbols
from .engine import linalg, numerics, ode, symbolic
from .engine.numerics import (
    gradient,
    hessian,
    jacobian,
    minimize_nd,
    root_find_nd,
)
from .engine.ode import (
    DenseSolution,
    DenseSystemSolution,
    EventSolution,
    EventSystemSolution,
    dsolve,
    solve_ode_adaptive,
    solve_ode_dense,
    solve_ode_events,
    solve_ode_num,
    solve_ode_stiff,
    solve_ode_stiff_adaptive,
    solve_ode_system_dense,
    solve_ode_system_events,
    solve_ode_system_num,
    solve_ode_system_stiff,
    solve_ode_system_stiff_adaptive,
)
from .engine.symbolic import limit, series, summation
from .frontend.parser import parse

__version__ = "0.2.0"

__all__ = [
    "E",
    "Expr",
    "M",
    "V",
    "Matrix",
    "symbols",
    "parse",
    "symbolic",
    "linalg",
    "numerics",
    "ode",
    "generate",
    "generate_system",
    "generate_ode",
    "generate_ode_adaptive",
    "generate_ode_dense",
    "CodeArtifact",
    "limit",
    "series",
    "summation",
    "gradient",
    "jacobian",
    "hessian",
    "root_find_nd",
    "minimize_nd",
    "dsolve",
    "solve_ode_num",
    "solve_ode_stiff",
    "solve_ode_stiff_adaptive",
    "solve_ode_adaptive",
    "solve_ode_dense",
    "DenseSolution",
    "solve_ode_events",
    "EventSolution",
    "solve_ode_system_num",
    "solve_ode_system_stiff",
    "solve_ode_system_stiff_adaptive",
    "solve_ode_system_dense",
    "DenseSystemSolution",
    "solve_ode_system_events",
    "EventSystemSolution",
    "__version__",
]
