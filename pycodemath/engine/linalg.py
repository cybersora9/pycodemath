"""Linear algebra engine (module 3) — matrices and vectors on IR.

Operates on ``Matrix`` from the IR core. By default it computes symbolically
(SymPy — exact, rational results, with parameters), and when that is not
possible (or when you explicitly ask for ``numeric=True``) — it falls back to
a numeric NumPy path.

Operations: multiplication, transpose, determinant, inverse, solving systems
``A x = b``, and eigenvalues and eigenvectors.

Every entry point here runs under the wall-clock budget of module 9 (see
``core.budget``): the symbolic path is SymPy, and a symbolic determinant or
eigenvalue problem is as capable of not returning as an integral is. Note that
``_FALLBACK_ERRORS`` below catches ``ValueError``, ``TypeError`` and
``NotImplementedError`` in order to drop to NumPy — which is precisely why the
guard's interrupt derives from ``BaseException`` and not from ``Exception``: an
``Exception`` would be caught here, and a timed-out symbolic run would silently
become a numeric one instead of a refusal.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

try:  # SymPy >= 1.13
    from sympy.matrices.exceptions import MatrixError
except ImportError:  # SymPy 1.12
    from sympy.matrices.common import MatrixError

from ..core.budget import under_budget
from ..core.errors import DomainError
from ..core.ir import Expr, Matrix

# Exceptions after which we may drop to the numeric fallback: these are
# signals that "SymPy cannot compute this symbolically" (singularity, wrong
# shape, missing algorithm). Everything else (e.g. a programmer error) should
# blow up loudly, not vanish into the fallback.
_FALLBACK_ERRORS = (MatrixError, ValueError, TypeError, NotImplementedError)


def _mat(x: "Matrix | sp.MatrixBase | str | list") -> Matrix:
    return x if isinstance(x, Matrix) else Matrix(x)


def _to_float(m: Matrix) -> np.ndarray:
    """Cast an IR matrix to a ``float`` array (for the numeric path).

    The numeric path requires concrete numbers — a matrix with parameters would
    raise a raw ``TypeError`` from deep inside NumPy, so instead we raise a
    readable error.
    """
    if m.sy.free_symbols:
        names = ", ".join(sorted(s.name for s in m.sy.free_symbols))
        raise DomainError(
            f"the numeric path requires a matrix without symbols; parameters given: {names}"
        )
    return np.array(m.sy.tolist(), dtype=float)


def _num_to_expr(v) -> Expr:
    """Wrap a number (float/complex from NumPy) back into a scalar IR."""
    v = complex(v)
    if abs(v.imag) < 1e-12:
        return Expr(sp.Float(v.real))
    return Expr(sp.Float(v.real) + sp.Float(v.imag) * sp.I)


def _numeric(fn, *args, **kwargs):
    """Run an ``np.linalg`` operation, translating ``LinAlgError`` to the contract.

    The numeric path on a singular/ill-conditioned matrix (or one whose
    spectrum does not converge) raises a raw ``numpy.linalg.LinAlgError`` — in
    English and outside the contract. We catch it and return a consistent
    ``PycodemathError``, like the rest of the engine (cf. ``numerics`` with a
    singular Jacobian).
    """
    try:
        return fn(*args, **kwargs)
    except np.linalg.LinAlgError as exc:
        raise DomainError(
            f"numeric linear algebra failed "
            f"(singular or ill-conditioned matrix): {exc}"
        ) from exc


# --- basic operations --------------------------------------------------
@under_budget("multiply")
def multiply(a, b) -> Matrix:
    """Matrix product ``A · B`` (also works for matrix × vector)."""
    A, B = _mat(a), _mat(b)
    # a mismatched shape used to raise a raw sympy.ShapeError outside the contract
    if A.sy.cols != B.sy.rows:
        raise DomainError(
            f"multiply: incompatible dimensions {A.sy.rows}×{A.sy.cols} · "
            f"{B.sy.rows}×{B.sy.cols} — the number of columns of A must equal "
            f"the number of rows of B"
        )
    return Matrix(A.sy * B.sy)


@under_budget("transpose")
def transpose(a) -> Matrix:
    """Transpose ``Aᵀ``."""
    return Matrix(_mat(a).sy.T)


@under_budget("det")
def det(a, numeric: bool = False) -> Expr:
    """Determinant ``det(A)`` (IR scalar). Symbolic, with a numeric fallback."""
    A = _mat(a)
    # a non-square matrix used to raise a raw NonSquareMatrixError, and after
    # the numeric fallback — a misleading message about singularity
    if A.sy.rows != A.sy.cols:
        raise DomainError(
            f"det: the matrix must be square, but its dimensions are "
            f"{A.sy.rows}×{A.sy.cols}"
        )
    if not numeric:
        try:
            return Expr(A.sy.det())
        except _FALLBACK_ERRORS:
            pass
    return _num_to_expr(_numeric(np.linalg.det, _to_float(A)))


@under_budget("inv")
def inv(a, numeric: bool = False) -> Matrix:
    """Inverse matrix ``A⁻¹``. Symbolic, with a numeric fallback."""
    A = _mat(a)
    if not numeric:
        try:
            return Matrix(A.sy.inv())
        except _FALLBACK_ERRORS:
            pass
    return Matrix(sp.Matrix(_numeric(np.linalg.inv, _to_float(A))))


# --- systems of equations ----------------------------------------------
@under_budget("solve_system")
def solve_system(A, b, numeric: bool = False) -> Matrix:
    """Solve the system ``A x = b`` and return the solution vector ``x``.

    ``b`` may be a vector (column) or a list. For a square matrix — the exact
    solution (``LUsolve`` / ``numpy.linalg.solve``). For an overdetermined
    system (more equations than unknowns) — the least-squares solution
    (``solve_least_squares`` / ``numpy.linalg.lstsq``). Symbolic, with a
    numeric NumPy fallback.
    """
    A = _mat(A)
    b = _mat(b)
    # a mismatched number of rows used to produce, after the fallback, a
    # misleading message about parameters/singularity instead of pointing to
    # the wrong dimensions
    if A.sy.rows != b.sy.rows:
        raise DomainError(
            f"solve_system: incompatible dimensions — A has {A.sy.rows} rows, "
            f"but b has {b.sy.rows} elements"
        )
    square = A.sy.rows == A.sy.cols
    if not numeric:
        try:
            if square:
                return Matrix(A.sy.LUsolve(b.sy))
            return Matrix(A.sy.solve_least_squares(b.sy))
        except _FALLBACK_ERRORS:
            pass
    An = _to_float(A)
    bn = _to_float(b).reshape(-1)
    if square:
        x = _numeric(np.linalg.solve, An, bn)
    else:
        x = _numeric(np.linalg.lstsq, An, bn, rcond=None)[0]
    return Matrix(sp.Matrix(np.asarray(x).reshape(-1, 1)))


# --- spectrum ----------------------------------------------------------
@under_budget("eigenvalues")
def eigenvalues(a, numeric: bool = False) -> list[Expr]:
    """Eigenvalues — a list of IR scalars (with multiplicities)."""
    A = _mat(a)
    if not numeric:
        try:
            out: list[Expr] = []
            for val, mult in A.sy.eigenvals().items():
                out.extend([Expr(val)] * int(mult))
            return out
        except _FALLBACK_ERRORS:
            pass
    return [_num_to_expr(v) for v in _numeric(np.linalg.eigvals, _to_float(A))]


@under_budget("eigenvectors")
def eigenvectors(a, numeric: bool = False) -> list[tuple[Expr, Matrix]]:
    """Eigenvectors — a list of pairs ``(eigenvalue, vector)``.

    The symbolic path returns exact vectors (one per eigen-direction); the
    numeric fallback — one normalized vector per value.
    """
    A = _mat(a)
    if not numeric:
        try:
            out: list[tuple[Expr, Matrix]] = []
            for val, _mult, vecs in A.sy.eigenvects():
                for vec in vecs:
                    out.append((Expr(val), Matrix(vec)))
            return out
        except _FALLBACK_ERRORS:
            pass
    w, vecs = _numeric(np.linalg.eig, _to_float(A))
    return [
        (_num_to_expr(w[i]), Matrix(sp.Matrix(vecs[:, i].reshape(-1, 1))))
        for i in range(len(w))
    ]
