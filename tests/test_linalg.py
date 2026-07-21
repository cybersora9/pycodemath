"""Round 2 tests (Block C): dimension validation and numeric det in linalg.

Contract: edge cases always raise PycodemathError (not a raw
sympy.ShapeError/NonSquareMatrixError nor a misleading message after fallback).
Numeric expectations come from the closed form (by hand), not from pycodemath.
"""

from __future__ import annotations

import pytest

from pycodemath import M, V
from pycodemath.core.errors import PycodemathError
from pycodemath.engine import linalg


# --- dimension validation ------------------------------------------------
def test_multiply_shape_mismatch_raises_pycodemath():
    # 2×3 · 2×3 — columns of A (3) ≠ rows of B (2): previously a raw ShapeError
    A = M("[[1,2,3],[4,5,6]]")
    with pytest.raises(PycodemathError, match="incompatible dimensions"):
        linalg.multiply(A, A)


def test_multiply_compatible_still_works():
    A = M("[[1,2,3],[4,5,6]]")  # 2×3 · 3×2 = 2×2 (validation regression)
    assert linalg.multiply(A, linalg.transpose(A)).shape == (2, 2)


def test_solve_system_row_mismatch_raises_pycodemath():
    # A 3×3, b of length 2 — previously a misleading message after fallback
    A = M("[[1,0,0],[0,1,0],[0,0,1]]")
    b = V("[1,2]")
    with pytest.raises(PycodemathError, match="incompatible dimensions"):
        linalg.solve_system(A, b)


# --- det: squareness + numeric path ------------------------------
def test_det_non_square_raises_pycodemath():
    with pytest.raises(PycodemathError, match="square"):
        linalg.det(M("[[1,2,3],[4,5,6]]"))


def test_det_numeric_matches_closed_form():
    # det [[4,7],[2,6]] = 4·6 − 7·2 = 10 (by hand, 2×2 formula)
    A = M("[[4,7],[2,6]]")
    assert float(str(linalg.det(A, numeric=True))) == pytest.approx(10.0)


def test_det_numeric_rejects_symbolic():
    # consistent with inv/eig: numeric=True refuses when parameters are present
    with pytest.raises(PycodemathError):
        linalg.det(M("[[a,0],[0,1]]"), numeric=True)
