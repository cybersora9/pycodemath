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

The series FULL (I1) adds, loaded lazily on first use (``import pycodemath``
stays as fast as before — PEP 562):

``check_code`` / ``to_symbolic`` / ``emit_test``
    does this Python function compute that formula? (C1)
``diagnose`` / ``improve`` / ``max_safe_input``
    floating-point accuracy with proved rewrites (C2); value ranges, overflow and
    NaN with executed witnesses (C3 — ``ranges.analyze``)
``allocate`` / ``round_money`` / ``float_money_hazard``
    money without lost cents (C4)
``convert`` / ``check_dimensions`` / ``dimension_of``
    exact units and dimensional analysis (J1)
``compare_rotation_expressions`` / ``orient2d``
    geometry that never lies (G1)
``issue`` / ``recheck`` / ``Certificate``
    every check written down and re-checked from the file alone (X1)
``code`` / ``stability`` / ``ranges`` / ``money`` / ``geometry`` / ``units`` /
``calc`` / ``stats``
    the modules themselves (K3: ``calc`` number theory and special functions,
    ``stats`` exact statistics and A/B tests).

Wave 2 (I2), lazy as well:

``approx`` / ``evalf_to`` / ``dual`` / ``identify``
    decimal digits that are proved, the exact and the decimal answer side by side,
    and a constant recognised only when a coincidence is excluded (K2)
``solve_inequality`` / ``solve_system`` / ``periodic_solutions`` / ``definite_integral``
    solution sets and definite or improper integrals, every answer proved (K5)
``count_iterations`` / ``check_iterations`` / ``check_complexity`` /
``solve_recurrence`` / ``master_theorem``
    how many times a loop runs, Θ with a proof, recurrences with their conditions (C5)
``lint`` / ``lint_source`` / ``lint_paths`` / ``spec`` / ``LintConfig``
    the math-in-code linter, each finding with an executed witness (L1;
    ``LintConfig`` is ``pycodemath.lint.Config``)

The README documents each of these with runnable examples.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

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

__version__ = "0.5.1"

if TYPE_CHECKING:
    from .code import geometry, money, ranges, stability
    from .code.frontend import CodeCheck, check_code, emit_test, to_symbolic
    from .code.geometry import compare_rotation_expressions, orient2d
    from .code.money import allocate, float_money_hazard, round_money
    from .code.ranges import RangeReport, max_safe_input
    from .code.stability import Improvement, StabilityReport, diagnose, improve
    from .engine import calc, stats, units
    from .engine.calc import CalcResult
    from .engine.units import check_dimensions, convert, dimension_of
    from .verify.certificate import Certificate, issue, recheck
    from . import code, lint
    from .code.loops import check_complexity, check_iterations, count_iterations
    from .code.recurrence import master_theorem, solve_recurrence
    from .engine.definite import (
        DefiniteIntegralResult,
        IntegralNoClosedForm,
        IntegralUndecided,
        definite_integral,
    )
    from .engine.identify import Identification, identify
    from .engine.precision import Approx, PrecisionUndecided, approx, dual, evalf_to
    from .engine.solvesets import (
        InequalityResult,
        PeriodicResult,
        SolveSetUndecided,
        SystemResult,
        periodic_solutions,
        solve_inequality,
        solve_system,
    )
    from .lint import Config as LintConfig
    from .lint import lint_paths, lint_source, spec

#: The series-FULL names (I1): name -> (module, attribute or None for the module).
#: Loaded on first access, so ``import pycodemath`` does not pay for NumPy-heavy
#: modules a caller never touches (measured: +80 ms when imported eagerly).
_LAZY: dict[str, tuple[str, "str | None"]] = {
    "code": (".code", None),
    "stability": (".code.stability", None),
    "ranges": (".code.ranges", None),
    "money": (".code.money", None),
    "geometry": (".code.geometry", None),
    "units": (".engine.units", None),
    "calc": (".engine.calc", None),
    "stats": (".engine.stats", None),
    "check_code": (".code.frontend", "check_code"),
    "to_symbolic": (".code.frontend", "to_symbolic"),
    "emit_test": (".code.frontend", "emit_test"),
    "CodeCheck": (".code.frontend", "CodeCheck"),
    "diagnose": (".code.stability", "diagnose"),
    "improve": (".code.stability", "improve"),
    "StabilityReport": (".code.stability", "StabilityReport"),
    "Improvement": (".code.stability", "Improvement"),
    "max_safe_input": (".code.ranges", "max_safe_input"),
    "RangeReport": (".code.ranges", "RangeReport"),
    "allocate": (".code.money", "allocate"),
    "round_money": (".code.money", "round_money"),
    "float_money_hazard": (".code.money", "float_money_hazard"),
    "convert": (".engine.units", "convert"),
    "check_dimensions": (".engine.units", "check_dimensions"),
    "dimension_of": (".engine.units", "dimension_of"),
    "compare_rotation_expressions": (".code.geometry", "compare_rotation_expressions"),
    "orient2d": (".code.geometry", "orient2d"),
    "CalcResult": (".engine.calc", "CalcResult"),
    "Certificate": (".verify.certificate", "Certificate"),
    "issue": (".verify.certificate", "issue"),
    "recheck": (".verify.certificate", "recheck"),
    # wave 2 (I2)
    "approx": (".engine.precision", "approx"),
    "evalf_to": (".engine.precision", "evalf_to"),
    "dual": (".engine.precision", "dual"),
    "Approx": (".engine.precision", "Approx"),
    "PrecisionUndecided": (".engine.precision", "PrecisionUndecided"),
    "identify": (".engine.identify", "identify"),
    "Identification": (".engine.identify", "Identification"),
    "solve_inequality": (".engine.solvesets", "solve_inequality"),
    "solve_system": (".engine.solvesets", "solve_system"),
    "periodic_solutions": (".engine.solvesets", "periodic_solutions"),
    "InequalityResult": (".engine.solvesets", "InequalityResult"),
    "SystemResult": (".engine.solvesets", "SystemResult"),
    "PeriodicResult": (".engine.solvesets", "PeriodicResult"),
    "SolveSetUndecided": (".engine.solvesets", "SolveSetUndecided"),
    "definite_integral": (".engine.definite", "definite_integral"),
    "DefiniteIntegralResult": (".engine.definite", "DefiniteIntegralResult"),
    "IntegralUndecided": (".engine.definite", "IntegralUndecided"),
    "IntegralNoClosedForm": (".engine.definite", "IntegralNoClosedForm"),
    "count_iterations": (".code.loops", "count_iterations"),
    "check_iterations": (".code.loops", "check_iterations"),
    "check_complexity": (".code.loops", "check_complexity"),
    "solve_recurrence": (".code.recurrence", "solve_recurrence"),
    "master_theorem": (".code.recurrence", "master_theorem"),
    "lint": (".lint", None),
    "lint_source": (".lint", "lint_source"),
    "lint_paths": (".lint", "lint_paths"),
    "spec": (".lint", "spec"),
    "LintConfig": (".lint", "Config"),
}


def __getattr__(name: str) -> Any:
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(entry[0], __name__)
    value = module if entry[1] is None else getattr(module, entry[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))

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
    # series FULL (I1) and wave 2 (I2), loaded lazily — see ``_LAZY``
    *_LAZY,
]
