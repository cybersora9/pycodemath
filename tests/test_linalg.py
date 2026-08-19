"""Testy rundy 2 (Blok C): walidacja wymiarów i det numeryczny w linalg.

Kontrakt: brzegi zawsze rzucają PycodemathError (nie surowy
sympy.ShapeError/NonSquareMatrixError ani mylący komunikat po fallbacku).
Oczekiwania liczbowe z formy zamkniętej (ręcznie), nie z pycodemath.
"""

from __future__ import annotations

import pytest

from pycodemath import M, V
from pycodemath.core.errors import DomainError
from pycodemath.engine import linalg


# --- walidacja wymiarów ------------------------------------------------
def test_multiply_shape_mismatch_raises_pycodemath():
    # 2×3 · 2×3 — kolumny A (3) ≠ wiersze B (2): dotąd surowy ShapeError
    A = M("[[1,2,3],[4,5,6]]")
    with pytest.raises(DomainError, match="incompatible dimensions"):
        linalg.multiply(A, A)


def test_multiply_compatible_still_works():
    A = M("[[1,2,3],[4,5,6]]")  # 2×3 · 3×2 = 2×2 (regresja walidacji)
    assert linalg.multiply(A, linalg.transpose(A)).shape == (2, 2)


def test_solve_system_row_mismatch_raises_pycodemath():
    # A 3×3, b długości 2 — dotąd mylący komunikat po fallbacku
    A = M("[[1,0,0],[0,1,0],[0,0,1]]")
    b = V("[1,2]")
    with pytest.raises(DomainError, match="incompatible dimensions"):
        linalg.solve_system(A, b)


# --- det: kwadratowość + ścieżka numeryczna ------------------------------
def test_det_non_square_raises_pycodemath():
    with pytest.raises(DomainError, match="square"):
        linalg.det(M("[[1,2,3],[4,5,6]]"))


def test_det_numeric_matches_closed_form():
    # det [[4,7],[2,6]] = 4·6 − 7·2 = 10 (ręcznie, wzór 2×2)
    A = M("[[4,7],[2,6]]")
    assert float(str(linalg.det(A, numeric=True))) == pytest.approx(10.0)


def test_det_numeric_rejects_symbolic():
    # spójnie z inv/eig: numeric=True odmawia przy parametrach
    with pytest.raises(DomainError):
        linalg.det(M("[[a,0],[0,1]]"), numeric=True)
