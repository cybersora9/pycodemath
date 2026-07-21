"""Code generator pipeline (module 2): IR → simplify → CSE → code + callable.

Ties source emission (``emit``) to compilation into a live function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.ir import Expr
from .emit import (
    GeneratedCode,
    emit_function,
    emit_ode,
    emit_ode_adaptive,
    emit_ode_dense,
    emit_solver,
)


@dataclass
class CodeArtifact:
    """Full generation result: source, live function and metadata."""

    code: GeneratedCode
    func: Callable

    @property
    def source(self) -> str:
        return self.code.source

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def generate(
    expr: Expr,
    func_name: str = "f",
    simplify: bool = True,
    use_cse: bool = True,
) -> CodeArtifact:
    """Generate an optimized Python/NumPy function from an IR expression.

    Steps: (optional) simplify → CSE → source emission → compilation.
    """
    work = expr.simplify() if simplify else expr
    generated = emit_function(work, func_name=func_name, use_cse=use_cse)
    return CodeArtifact(code=generated, func=_compile(generated))


def generate_system(
    A,
    b,
    func_name: str = "solve",
    use_cse: bool = True,
) -> CodeArtifact:
    """Generate an optimized NumPy function solving the system ``A x = b``.

    Steps: solver emission (CSE over the matrix entries) → compilation. The
    returned function takes the system parameters and returns the solution vector.
    """
    generated = emit_solver(A, b, func_name=func_name, use_cse=use_cse)
    return CodeArtifact(code=generated, func=_compile(generated))


def generate_ode(
    rhs: Expr,
    var: str = "t",
    func: str = "y",
    func_name: str = "solve_ode",
    use_cse: bool = True,
) -> CodeArtifact:
    """Generate a standalone RK4 integrator (Python/NumPy) for ``y' = f(t, y)``.

    Steps: integrator emission (CSE over the right-hand side) → compilation. The
    returned function takes ``(y0, t0, t1, n)`` and returns samples ``(ts, ys)``
    — the same as ``engine.ode.solve_ode_num``.
    """
    generated = emit_ode(
        Expr(rhs), var=var, func=func, func_name=func_name, use_cse=use_cse
    )
    return CodeArtifact(code=generated, func=_compile(generated))


def generate_ode_adaptive(
    rhs: Expr,
    var: str = "t",
    func: str = "y",
    func_name: str = "solve_ode_adaptive",
    use_cse: bool = True,
) -> CodeArtifact:
    """Generate a standalone ADAPTIVE DOPRI5 integrator for ``y' = f(t, y)``.

    Steps: integrator emission (CSE over the right-hand side) → compilation. The
    returned function takes ``(y0, t0, t1, rtol=1e-6, atol=1e-9,
    max_steps=100000)`` and returns samples ``(ts, ys)`` — node parity with
    ``engine.ode.solve_ode_adaptive`` (the same coefficients and step control,
    including backward integration).
    """
    generated = emit_ode_adaptive(
        Expr(rhs), var=var, func=func, func_name=func_name, use_cse=use_cse
    )
    return CodeArtifact(code=generated, func=_compile(generated))


def generate_ode_dense(
    rhs: Expr,
    var: str = "t",
    func: str = "y",
    func_name: str = "solve_ode_dense",
    use_cse: bool = True,
) -> CodeArtifact:
    """Generate a standalone solver with DENSE OUTPUT (DOPRI5) for ``y' = f(t, y)``.

    Steps: integrator emission with an interpolant (CSE over the right-hand
    side) → compilation. The returned function takes ``(y0, t0, t1, rtol=1e-6,
    atol=1e-9, max_steps=100000)`` and returns an INTERPOLANT FUNCTION ``sol`` —
    ``sol(t)`` reads ``y(t)`` at any point (bitwise parity of nodes and
    interpolant with ``engine.ode.solve_ode_dense``); nodes: ``sol.ts`` / ``sol.ys``.
    """
    generated = emit_ode_dense(
        Expr(rhs), var=var, func=func, func_name=func_name, use_cse=use_cse
    )
    return CodeArtifact(code=generated, func=_compile(generated))


def _compile(generated: GeneratedCode) -> Callable:
    """Compile the emitted source into a live function.

    The emitted code uses the ``np.``/``numpy.`` prefix (and imports NumPy
    itself), so the namespace is limited to the module aliases — without
    injecting hundreds of names from ``dir(np)``.
    """
    namespace: dict = {"np": np, "numpy": np}
    exec(compile(generated.source, "<pycodemath>", "exec"), namespace)
    return namespace[generated.func_name]
