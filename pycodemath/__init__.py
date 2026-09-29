"""Pycodemath — heavy computational math + Python.

Top-level public API. Everything a caller needs is importable from here:

``parse`` / ``E`` / ``M`` / ``V``
    text -> the single IR (``Expr`` / ``Matrix``) the engine and the code
    generator share.
``symbolic`` / ``linalg`` / ``numerics`` / ``ode``
    the engines. The numerical entry points that carry structured results are
    re-exported by name as well: ``root_find``, ``minimize``, ``root_find_nd``,
    ``minimize_nd``, ``integrate_num``.
``PycodemathError`` and its eight subclasses
    what a failure IS, as a type — the thing to branch on instead of reading a
    message. ``ParseError`` / ``DomainError`` are refusals of the INPUT;
    ``DivergenceError`` / ``StagnationError`` / ``NonConvergenceError`` /
    ``NotAMinimumError`` / ``BudgetExhaustedError`` are outcomes of a RUN; and
    ``TimeBudgetError`` (module 9) is the run that had no outcome at all because
    the wall clock ran out first.
``time_budget`` / ``DEFAULT_TIME_BUDGET``
    how long the symbolic engine may spend before it refuses. Every symbolic entry
    point is bounded — ``integrate 1/(x^5+x+1) dx`` used to hang forever — and
    ``with time_budget(seconds):`` is how a caller says in advance how long it is
    willing to wait.
``isolated`` / ``IsolationError``
    the HARD version of that bound (module B): ``isolated(fn, *args, budget=...)``
    runs the call in a warm worker process and kills it at the deadline, which is
    the only thing that stops a call stuck in one long C-level operation.
    Opt-in; ``time_budget`` stays the default and keeps its 3.5 us fast path.
``SolveResult`` / ``QuadratureResult`` / ``SolveStatus`` / ``SOLVE_STATUSES``
    what a run LEARNED, returned by the calls above under ``full_result=True``.
    ``converged`` is the only field to branch on — and since module 8 it answers
    the question that was ASKED (a ROOT, a MINIMUM), not merely "did an exit test
    fire". The package ships ``py.typed``,
    so the overloads that make ``full_result`` change the return TYPE are visible
    to an installed caller's type checker — no cast anywhere.

The README documents each of these with runnable examples.
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
from .core.backstop import isolated
from .core.budget import DEFAULT_TIME_BUDGET, time_budget
from .core.errors import (
    NUMERIC_ROUTES,
    ROUTES,
    BudgetExhaustedError,
    DivergenceError,
    DomainError,
    IsolationError,
    NoClosedFormError,
    NonConvergenceError,
    NotAMinimumError,
    ParseError,
    PycodemathError,
    StagnationError,
    TimeBudgetError,
    UnsupportedFormError,
)
from .core.ir import E, Expr, M, Matrix, V, symbols
from .core.result import (
    SOLVE_STATUSES,
    QuadratureResult,
    SolveResult,
    SolveStatus,
)
from .engine import linalg, numerics, ode, symbolic
from .engine.numerics import (
    gradient,
    hessian,
    integrate_num,
    jacobian,
    minimize,
    minimize_nd,
    root_find,
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

__version__ = "0.4.0"

__all__ = [
    "E",
    "Expr",
    "M",
    "V",
    "Matrix",
    "symbols",
    "parse",
    "PycodemathError",
    "ParseError",
    "DomainError",
    "DivergenceError",
    "StagnationError",
    "NonConvergenceError",
    "NotAMinimumError",
    "BudgetExhaustedError",
    "TimeBudgetError",
    "NoClosedFormError",
    "UnsupportedFormError",
    "IsolationError",
    "ROUTES",
    "NUMERIC_ROUTES",
    "time_budget",
    "DEFAULT_TIME_BUDGET",
    "isolated",
    "SolveResult",
    "QuadratureResult",
    "SolveStatus",
    "SOLVE_STATUSES",
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
    # The five functions that RETURN the types just above. Before module 7 only the
    # ``_nd`` pair was exported, so ``from pycodemath import SolveResult`` worked
    # while ``from pycodemath import root_find`` — the very call that produces one —
    # did not. Same module, same contract, same @overload: the split was an accident.
    "root_find",
    "minimize",
    "root_find_nd",
    "minimize_nd",
    "integrate_num",
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
