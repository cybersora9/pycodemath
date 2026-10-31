"""G1 — geometry and graphics maths for programmers, every answer with its proof.

Game, graphics, CAD and robotics code is full of small formulas that are easy to
get almost right: a quaternion product in the wrong order, a JPL quaternion read
as Hamilton, a row-vector matrix read as column-vector, an unnormalised
quaternion fed to ``1 - 2(y^2 + z^2)``, an ``EPSILON`` in a ray/triangle test, a
naive orientation determinant whose sign is wrong for nearly collinear points.
Each of these fails silently. This module gives the formulas with explicit
conventions, a certificate for every result, and detectors that turn a suspected
bug into a concrete input where the two answers differ.

Sections (each result is a frozen dataclass with ``to_dict()``):

1. **Rotations** — quaternion <-> 3x3 matrix <-> Euler angles (all 12 sequences,
   ``frame="intrinsic"|"extrinsic"`` is a required argument) <-> axis-angle.
   Conventions are values, never defaults hidden in a name: :class:`QuatConvention`
   (component ``order`` ``wxyz``/``xyzw``, ``algebra`` Hamilton/JPL) and
   :class:`MatrixConvention` (``vectors`` column/row, ``mapping`` active/passive).
   Every conversion is certified by an INDEPENDENT route: the matrix of the source
   (elementary-rotation products for Euler, Rodrigues for axis-angle, the
   homogeneous quadratic form for quaternions) is compared with the matrix of the
   target at 50 significant digits. The underlying formulas are proved symbolically
   once (:func:`prove_rotation_identities`). :func:`slerp` takes the shorter arc and
   stays accurate for nearly equal endpoints (sinc form, no lerp fallback).
2. **Convention detector** — :func:`compare_rotation_expressions` decides whether
   ``"q1*q2"`` and ``"R1@R2"`` denote the same rotation for EVERY input (symbolic
   proof over polynomial rings) or REFUTES it with integer quaternions at which the
   two differ exactly, naming the likely cause (reversed order, transpose,
   row/column vectors, JPL/Hamilton, component order, handedness mirror).
   :func:`identify_euler_convention` tells which of the 24 Euler conventions
   (and transposition) a library used.
3. **Predicates and intersections** — :func:`orient2d`, :func:`orient3d`,
   :func:`incircle` on floats with Shewchuk's stage-A error filter and an exact
   rational fallback (the sign is never wrong); segment/segment, ray/triangle
   (Möller–Trumbore in exact arithmetic), ray/sphere, point in polygon (boundary
   reporting and the half-open ownership rule). :func:`epsilon_witness` shows the
   concrete data at which an ``abs(x) < EPSILON`` test answers wrongly.
4. **Curves** — Bézier of any degree (de Casteljau, Bernstein form, derivatives,
   arc length with a RIGOROUS bracket), CSS ``cubic-bezier`` easing (y for x, with
   an exact enclosure of the answer), Catmull-Rom (uniform/centripetal/chordal),
   cubic Hermite, ``smoothstep``/``smootherstep``.
5. **Code emission** — :func:`emit` prints each formula as Python, NumPy, C or
   GLSL; the test-suite executes Python/NumPy, compiles C, and compiles GLSL as C
   under a shim that implements the GLSL constructor rules.

Exact inputs (``int``, ``Fraction``, rational strings like ``"1/3"``) stay exact
where the mathematics allows it; ``float`` inputs are taken at their exact binary
value, so an exact answer about a float input is an answer about THAT float.
"""

from __future__ import annotations

import ast
import itertools
import math
import re
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from functools import lru_cache
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Union

import mpmath

from ..core.budget import guard
from ..core.errors import DomainError, NonConvergenceError

__all__ = [
    # errors
    "NotARotationError",
    "NonUnitQuaternionError",
    "CertificationError",
    # conventions
    "QuatConvention",
    "MatrixConvention",
    "EULER_SEQUENCES",
    # rotations
    "RotationResult",
    "convert_rotation",
    "quat_to_matrix",
    "matrix_to_quat",
    "euler_to_matrix",
    "matrix_to_euler",
    "euler_to_quat",
    "quat_to_euler",
    "axis_angle_to_quat",
    "axis_angle_to_matrix",
    "quat_to_axis_angle",
    "matrix_to_axis_angle",
    "quat_multiply",
    "slerp",
    "mirror_rotation",
    "prove_rotation_identities",
    # detector
    "ConventionReport",
    "compare_rotation_expressions",
    "EulerMatch",
    "identify_euler_convention",
    # predicates / intersections
    "PredicateResult",
    "orient2d",
    "orient3d",
    "incircle",
    "classify_orient2d",
    "classify_orient3d",
    "classify_incircle",
    "orient2d_exact",
    "orient3d_exact",
    "incircle_exact",
    "SegmentIntersection",
    "segment_intersection",
    "RayHit",
    "ray_triangle",
    "moller_trumbore_float",
    "RaySphere",
    "ray_sphere",
    "PolygonLocation",
    "point_in_polygon",
    "EpsilonReport",
    "epsilon_witness",
    # curves
    "CurveValue",
    "bezier_point",
    "bezier_derivative",
    "bezier_polynomial",
    "ArcLength",
    "bezier_arc_length",
    "EasingResult",
    "css_cubic_bezier",
    "css_easing",
    "CSS_EASING_PRESETS",
    "catmull_rom",
    "hermite",
    "smoothstep",
    "smootherstep",
    # code
    "EmittedCode",
    "emit",
    "FORMULAS",
    "LANGUAGES",
]

Num = Union[int, float, Fraction]
Vec = tuple[float, ...]
Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

_EPS = 2.0**-53
#: Working precision (significant digits) of every numeric certificate.
CERT_DPS = 50


# ======================================================================= errors
class NotARotationError(DomainError):
    """A 3x3 matrix that is not a proper rotation (not orthogonal, or det = -1)."""


class NonUnitQuaternionError(DomainError):
    """A quaternion whose norm is not 1 where a unit quaternion was required.

    Carries ``norm`` (float). The message also states what the common
    non-homogeneous formula ``1 - 2(y^2+z^2)`` would have produced from it.
    """

    def __init__(self, message: str, *, norm: float) -> None:
        super().__init__(message)
        self.norm = norm


class CertificationError(NonConvergenceError):
    """A computed result failed its independent check — reported, never returned.

    This is the refusal the truth contract demands instead of an unproved value.
    It should not happen for valid input; if it does it carries the measured
    ``residual`` and the ``tolerance`` it exceeded.
    """

    def __init__(self, message: str, *, residual: float, tolerance: float) -> None:
        super().__init__(message)
        self.residual = residual
        self.tolerance = tolerance


# ====================================================================== helpers
def _jsonable(v: Any) -> Any:
    if isinstance(v, Fraction):
        return _fstr(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "to_dict"):
        return v.to_dict()
    return str(v)


class _Result:
    """Mixin: ``to_dict()`` for the frozen result dataclasses (JSON-able)."""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]


def _fstr(q: Fraction) -> str:
    """Exact rational as text SymPy/pycodemath read back: ``"3"``, ``"-7/2"``."""
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


_RATIONAL_RE = re.compile(r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?(\s*/\s*[-+]?\d+)?\s*$")


def _scalar(v: Any, what: str) -> Num:
    """One real input: ``int``/``Fraction``/rational text stay exact, ``float`` stays float.

    numpy scalars are accepted by value. NaN and infinities are refused — every
    formula here needs a finite real.
    """
    if isinstance(v, bool):
        raise DomainError(f"{what}: expected a real number, got a bool")
    if isinstance(v, (int, Fraction)):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            raise DomainError(f"{what}: expected a finite number, got {v!r}")
        return v
    if isinstance(v, str):
        if not _RATIONAL_RE.match(v):
            raise DomainError(f"{what}: {v!r} is not a rational number (write 1/3, 0.25, 2e-3)")
        num, _, den = v.partition("/")
        d = Fraction(den.strip()) if den else Fraction(1)
        if d == 0:
            raise DomainError(f"{what}: {v!r} divides by zero")
        return Fraction(num.strip()) / d
    try:  # numpy scalars, sympy Rationals, decimal.Decimal
        import numbers

        if isinstance(v, numbers.Integral):
            return int(v)
        if isinstance(v, numbers.Rational):
            return Fraction(int(v.numerator), int(v.denominator))
        if hasattr(v, "is_Rational") and getattr(v, "is_Rational"):
            return Fraction(int(v.p), int(v.q))
        if isinstance(v, numbers.Real):
            f = float(v)
            return _scalar(f, what)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DomainError(f"{what}: cannot read {v!r} as a real number") from exc
    raise DomainError(f"{what}: expected a real number, got {type(v).__name__}")


def _vector(v: Any, n: "int | None", what: str) -> tuple[Num, ...]:
    try:
        items = list(v)
    except TypeError as exc:
        raise DomainError(f"{what}: expected a sequence of numbers, got {v!r}") from exc
    if n is not None and len(items) != n:
        raise DomainError(f"{what}: expected {n} components, got {len(items)}")
    if not items:
        raise DomainError(f"{what}: empty vector")
    return tuple(_scalar(x, f"{what}[{i}]") for i, x in enumerate(items))


def _is_exact(values: Iterable[Num]) -> bool:
    return all(not isinstance(x, float) for x in values)


def _F(x: Num) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def _mp(x: Num) -> Any:
    if isinstance(x, Fraction):
        return mpmath.mpf(x.numerator) / x.denominator
    return mpmath.mpf(x)


def _sign(x: Any) -> int:
    return (x > 0) - (x < 0)


def _float_down(q: Fraction) -> float:
    f = float(q)
    return math.nextafter(f, -math.inf) if Fraction(f) > q else f


def _float_up(q: Fraction) -> float:
    f = float(q)
    return math.nextafter(f, math.inf) if Fraction(f) < q else f


# ================================================================== conventions
@dataclass(frozen=True)
class QuatConvention(_Result):
    """How four numbers are read as a rotation quaternion.

    ``order``: ``"wxyz"`` (scalar first: Eigen's constructor, glm's constructor,
    numpy-quaternion, pyquaternion, transforms3d) or ``"xyzw"`` (scalar last:
    SciPy, ROS ``geometry_msgs``, Unity, GLSL ``vec4``).

    ``algebra``: ``"hamilton"`` (i*j = k; ``q`` maps body to world; Eigen, glm,
    SciPy, ROS) or ``"jpl"`` (i*j = -k; ``q`` maps world to body; JPL/NASA
    navigation literature, MSCKF, Trawny & Roumeliotis 2005). For the same four
    numbers the JPL matrix is the TRANSPOSE of the Hamilton matrix, and
    ``p (x)_JPL q = q (x)_Hamilton p`` (Sola 2017, "Quaternion kinematics for the
    error-state Kalman filter", §2.6).
    """

    order: Literal["wxyz", "xyzw"] = "wxyz"
    algebra: Literal["hamilton", "jpl"] = "hamilton"

    def __post_init__(self) -> None:
        if self.order not in ("wxyz", "xyzw"):
            raise DomainError(f"QuatConvention.order must be 'wxyz' or 'xyzw', got {self.order!r}")
        if self.algebra not in ("hamilton", "jpl"):
            raise DomainError(f"QuatConvention.algebra must be 'hamilton' or 'jpl', got {self.algebra!r}")


@dataclass(frozen=True)
class MatrixConvention(_Result):
    """How a 3x3 matrix is read as a rotation.

    ``vectors="column"``: ``v' = R @ v`` (math, OpenGL/glm, Eigen, SciPy).
    ``vectors="row"``: ``v' = v @ M`` (DirectX/XNA, Unity's serialized data,
    many game engines) — numerically ``M = R.T``, and products compose in the
    REVERSE order (``R1 @ R2`` becomes ``M2 @ M1``).

    ``mapping="active"`` rotates the vector (alibi); ``"passive"`` re-expresses a
    fixed vector in a rotated frame (alias) — numerically also the transpose.
    Both flags together cancel.

    Handedness is deliberately NOT a field: a rotation matrix has the same numbers
    in a left-handed frame; only the picture changes (positive angles look
    clockwise). Moving data BETWEEN a right- and a left-handed frame is a change of
    basis by a mirror, ``S @ R @ S`` — that is :func:`mirror_rotation`.
    """

    vectors: Literal["column", "row"] = "column"
    mapping: Literal["active", "passive"] = "active"

    def __post_init__(self) -> None:
        if self.vectors not in ("column", "row"):
            raise DomainError(f"MatrixConvention.vectors must be 'column' or 'row', got {self.vectors!r}")
        if self.mapping not in ("active", "passive"):
            raise DomainError(f"MatrixConvention.mapping must be 'active' or 'passive', got {self.mapping!r}")

    @property
    def transposed(self) -> bool:
        """True when the stored matrix is the transpose of the active column-vector one."""
        return (self.vectors == "row") != (self.mapping == "passive")


#: The 12 Euler sequences: 6 Tait-Bryan (all axes differ) + 6 proper Euler (first = last).
EULER_SEQUENCES: tuple[str, ...] = (
    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
    "xyx", "xzx", "yxy", "yzy", "zxz", "zyz",
)

_AXIS = {"x": 0, "y": 1, "z": 2}


def _check_seq(seq: Any, frame: Any) -> tuple[str, str]:
    if not isinstance(seq, str) or len(seq) != 3 or seq.lower() not in EULER_SEQUENCES:
        raise DomainError(
            f"Euler sequence must be one of {', '.join(EULER_SEQUENCES)} (any case), got {seq!r}"
        )
    if frame not in ("intrinsic", "extrinsic"):
        raise DomainError(
            "Euler frame must be stated: frame='intrinsic' (rotating axes, R = R1 @ R2 @ R3) "
            f"or frame='extrinsic' (fixed axes, R = R3 @ R2 @ R1); got {frame!r}"
        )
    return seq.lower(), frame


# =============================================================== quaternion core
# Internal representation: Hamilton, (w, x, y, z), active, column vectors.
# Every function below is generic over the number type (float, Fraction,
# mpmath.mpf, polynomial-ring elements) — only +, -, * (and / for scaling).


def _qmul(a: Sequence[Any], b: Sequence[Any]) -> tuple[Any, Any, Any, Any]:
    """Hamilton product a (x) b, components (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qconj(q: Sequence[Any]) -> tuple[Any, Any, Any, Any]:
    return (q[0], -q[1], -q[2], -q[3])


def _qnorm2(q: Sequence[Any]) -> Any:
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]


def _quat_numerator(q: Sequence[Any]) -> list[list[Any]]:
    """|q|^2 * R(q): the homogeneous form, a rotation times |q|^2 for ANY q != 0."""
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    return [
        [ww + xx - yy - zz, 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), ww - xx + yy - zz, 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), ww - xx - yy + zz],
    ]


def _quat_matrix(q: Sequence[Any]) -> list[list[Any]]:
    """R(q) = homogeneous form / |q|^2 — orthogonal even when |q| drifted from 1."""
    n = _qnorm2(q)
    return [[v / n for v in row] for row in _quat_numerator(q)]


def _transpose(m: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def _matmul(a: Sequence[Sequence[Any]], b: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), start=a[i][0] * 0) for j in range(len(b[0]))]
            for i in range(len(a))]


def _det3(m: Sequence[Sequence[Any]]) -> Any:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _elem_matrix(axis: int, c: Any, s: Any) -> list[list[Any]]:
    one, zero = c * 0 + 1, c * 0
    if axis == 0:
        return [[one, zero, zero], [zero, c, -s], [zero, s, c]]
    if axis == 1:
        return [[c, zero, s], [zero, one, zero], [-s, zero, c]]
    return [[c, -s, zero], [s, c, zero], [zero, zero, one]]


def _elem_quat(axis: int, c_half: Any, s_half: Any) -> tuple[Any, Any, Any, Any]:
    q = [c_half, c_half * 0, c_half * 0, c_half * 0]
    q[1 + axis] = s_half
    return (q[0], q[1], q[2], q[3])


def _to_internal(q: tuple[Num, ...], conv: QuatConvention) -> tuple[Num, Num, Num, Num]:
    """User components -> internal Hamilton (w, x, y, z) of the SAME rotation."""
    if conv.order == "xyzw":
        w, x, y, z = q[3], q[0], q[1], q[2]
    else:
        w, x, y, z = q
    if conv.algebra == "jpl":  # R_JPL(q) = R_H(q)^T = R_H(conj q)
        x, y, z = -x, -y, -z
    return (w, x, y, z)


def _from_internal(q: Sequence[Any], conv: QuatConvention) -> tuple[Any, ...]:
    w, x, y, z = q
    if conv.algebra == "jpl":
        x, y, z = -x, -y, -z
    return (x, y, z, w) if conv.order == "xyzw" else (w, x, y, z)


def _canonical(q: Sequence[float]) -> tuple[float, float, float, float]:
    """Pick the representative with w > 0 (or, for w == 0, first non-zero > 0)."""
    for v in q:
        if v != 0:
            if v < 0:
                return (-q[0], -q[1], -q[2], -q[3])
            break
    return (q[0], q[1], q[2], q[3])


# ======================================================== rotation: certificates
def _mp_matrix_of(kind: str, data: Any) -> list[list[Any]]:
    """Active column-vector matrix of a representation, at CERT_DPS, by ITS OWN route.

    quat   -> homogeneous quadratic form / |q|^2
    matrix -> the numbers themselves
    euler  -> product of elementary rotation matrices (cos/sin of each angle)
    axis   -> Rodrigues' formula  I + sin(a) K + (1 - cos(a)) K^2
    """
    with mpmath.workdps(CERT_DPS):
        if kind == "quat":
            return _quat_matrix([_mp(v) for v in data])
        if kind == "matrix":
            return [[_mp(v) for v in row] for row in data]
        if kind == "euler":
            seq, frame, angles = data
            mats = [_elem_matrix(_AXIS[a], mpmath.cos(_mp(t)), mpmath.sin(_mp(t))) for a, t in zip(seq, angles)]
            if frame == "extrinsic":
                mats = mats[::-1]
            return _matmul(_matmul(mats[0], mats[1]), mats[2])
        if kind == "axis":
            axis, angle = data
            ax = [_mp(v) for v in axis]
            n = mpmath.sqrt(sum(v * v for v in ax))
            kx, ky, kz = (v / n for v in ax)
            K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
            K2 = _matmul(K, K)
            s, c = mpmath.sin(_mp(angle)), mpmath.cos(_mp(angle))
            return [[(1 if i == j else 0) + s * K[i][j] + (1 - c) * K2[i][j] for j in range(3)] for i in range(3)]
    raise AssertionError(kind)


def _mp_residual(a: list[list[Any]], b: list[list[Any]]) -> float:
    with mpmath.workdps(CERT_DPS):
        return float(max(abs(a[i][j] - b[i][j]) for i in range(3) for j in range(3)))


def _orthogonality(m: Sequence[Sequence[Num]]) -> tuple[float, float]:
    """(max |M^T M - I|, det M - 1) of the numbers as given, at CERT_DPS."""
    with mpmath.workdps(CERT_DPS):
        mm = [[_mp(v) for v in row] for row in m]
        g = _matmul(_transpose(mm), mm)
        err = max(abs(g[i][j] - (1 if i == j else 0)) for i in range(3) for j in range(3))
        return float(err), float(_det3(mm) - 1)


@dataclass(frozen=True)
class RotationResult(_Result):
    """One certified rotation conversion (or product / slerp).

    ``value``: the target representation as floats — a quaternion tuple in the
    requested :class:`QuatConvention` order, a 3x3 tuple of rows in the requested
    :class:`MatrixConvention`, Euler angles in the sequence order, or
    ``(axis, angle)``. ``exact`` is the same as exact rational strings when the
    input was exact and the target is rational (quaternion -> matrix, products).

    ``certificate``: ``status`` (``"verified"``), ``method``, the measured
    ``residual`` between the source's matrix and the target's matrix (each by its
    own route, at 50 digits), the ``tolerance`` it had to meet, and for matrix
    outputs ``orthogonality_error`` and ``det_minus_1`` of the returned numbers.
    """

    kind: str
    source: str
    target: str
    value: Any
    conventions: dict[str, Any]
    certificate: dict[str, Any]
    exact: Any = None
    gimbal_lock: bool = False
    notes: tuple[str, ...] = ()


def _certify(source: tuple[str, Any], target: tuple[str, Any], tol: float,
             what: str, extra: "dict[str, Any] | None" = None) -> dict[str, Any]:
    res = _mp_residual(_mp_matrix_of(*source), _mp_matrix_of(*target))
    if not res <= tol:
        raise CertificationError(
            f"{what}: the result does not reproduce the input rotation "
            f"(residual {res:.3g} > tolerance {tol:.3g}); refusing to return it",
            residual=res, tolerance=tol,
        )
    cert: dict[str, Any] = {
        "status": "verified",
        "method": f"independent-route matrix comparison at {CERT_DPS} digits",
        "routes": [source[0], target[0]],
        "residual": res,
        "tolerance": tol,
        "formulas_proved": "prove_rotation_identities()",
    }
    if extra:
        cert.update(extra)
    return cert


_FLOAT_TOL = 64 * _EPS  # ~1.4e-14: a few roundings per entry, with margin
_UNIT_TOL = 1e-6


# ======================================================= rotation: input parsing
def _read_quat(q: Any, conv: QuatConvention, normalize: bool, what: str) -> tuple[tuple[Num, ...], tuple[Num, ...], float]:
    """-> (raw components, internal Hamilton wxyz (NOT normalised), norm)."""
    raw = _vector(q, 4, what)
    internal = _to_internal(raw, conv)
    exact = _is_exact(raw)
    if exact:
        n2: Any = _qnorm2([_F(v) for v in internal])
        if n2 == 0:
            raise DomainError(f"{what}: the zero quaternion is not a rotation")
        norm = math.sqrt(float(n2)) if float(n2) != math.inf else math.inf
    else:
        # Scale by a power of two (exact) so |q|^2 neither underflows nor overflows;
        # the rotation does not depend on the scale.
        big = max(abs(float(v)) for v in internal)
        if big == 0:
            raise DomainError(f"{what}: the zero quaternion is not a rotation")
        e = math.frexp(big)[1]
        internal = tuple(math.ldexp(float(v), -e) for v in internal)  # type: ignore[assignment]
        n2 = _qnorm2(internal)
        norm = math.ldexp(math.sqrt(n2), e) if e < 1000 else math.inf
    if not normalize:
        if (n2 != 1) if exact else (abs(norm - 1.0) > _UNIT_TOL):
            # What the common non-homogeneous formula would have produced:
            # R' = R(q/|q|) + (1 - |q|^2) * (I - 2 * diag-ish) -> its determinant.
            w, x, y, z = (float(v) for v in internal)
            naive = [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
            oe, dm = _orthogonality(naive)
            raise NonUnitQuaternionError(
                f"{what}: quaternion norm is {norm:.17g}, not 1. The common formula "
                f"1-2(y^2+z^2), ... would give a matrix with det {1 + dm:.6g} and "
                f"orthogonality error {oe:.3g} — a scaling/shear, not a rotation. "
                f"Pass normalize=True to use q/|q| (done exactly via the homogeneous form)",
                norm=norm,
            )
    return raw, internal, norm


def _read_matrix(m: Any, conv: MatrixConvention, what: str, tol: float = _UNIT_TOL) -> tuple[list[list[Num]], float, float]:
    """-> (active column-vector matrix as given numbers, orthogonality error, det-1)."""
    try:
        rows = [list(r) for r in m]
    except TypeError as exc:
        raise DomainError(f"{what}: expected a 3x3 matrix (sequence of 3 rows)") from exc
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise DomainError(f"{what}: expected a 3x3 matrix, got shape {len(rows)}x{[len(r) for r in rows]}")
    M = [[_scalar(v, f"{what}[{i}][{j}]") for j, v in enumerate(r)] for i, r in enumerate(rows)]
    if conv.transposed:
        M = _transpose(M)
    oe, dm = _orthogonality(M)
    if dm < -1 + 1e-6 or (1 + dm) < 0:
        raise NotARotationError(
            f"{what}: det = {1 + dm:.6g} < 0 — this is a reflection (an improper rotation), "
            "typically a right/left-handed frame mix-up; mirror it first (mirror_rotation)"
        )
    if oe > tol or abs(dm) > tol:
        raise NotARotationError(
            f"{what}: not a rotation matrix — max|M^T M - I| = {oe:.3g}, det - 1 = {dm:.3g} "
            f"(tolerance {tol:g}); a scaled or sheared matrix has no rotation to convert"
        )
    return M, oe, dm


def _read_euler(angles: Any, seq: Any, frame: Any, degrees: bool, what: str) -> tuple[str, str, tuple[float, float, float]]:
    s, fr = _check_seq(seq, frame)
    a = _vector(angles, 3, what)
    rad = tuple(math.radians(float(v)) if degrees else float(v) for v in a)
    if degrees and _is_exact(a):
        # exact degrees -> radians via mpmath only in the certificate; floats here
        pass
    return s, fr, (rad[0], rad[1], rad[2])


def _read_axis_angle(axis: Any, angle: Any, degrees: bool, what: str) -> tuple[tuple[float, float, float], float]:
    ax = _vector(axis, 3, f"{what} axis")
    ang = float(_scalar(angle, f"{what} angle"))
    if degrees:
        ang = math.radians(ang)
    n = math.sqrt(sum(float(v) ** 2 for v in ax))
    if n == 0 or not math.isfinite(n):
        raise DomainError(f"{what}: the rotation axis must be a non-zero finite vector, got {ax}")
    return (float(ax[0]) / n, float(ax[1]) / n, float(ax[2]) / n), ang


# ===================================================== rotation: float routes
def _float_quat_to_matrix(q: Sequence[float]) -> list[list[float]]:
    return _quat_matrix([float(v) for v in q])


def _float_matrix_to_quat(m: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Shepperd's method (1978): branch on the largest of trace and diagonal."""
    m00, m01, m02 = (float(v) for v in m[0])
    m10, m11, m12 = (float(v) for v in m[1])
    m20, m21, m22 = (float(v) for v in m[2])
    tr = m00 + m11 + m22
    k = max(range(4), key=lambda i: (tr, m00, m11, m22)[i])
    if k == 0:
        s = 2.0 * math.sqrt(max(1.0 + tr, 0.0))
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif k == 1:
        s = 2.0 * math.sqrt(max(1.0 + m00 - m11 - m22, 0.0))
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif k == 2:
        s = 2.0 * math.sqrt(max(1.0 + m11 - m00 - m22, 0.0))
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = 2.0 * math.sqrt(max(1.0 + m22 - m00 - m11, 0.0))
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    n = math.sqrt(_qnorm2(q))
    return _canonical(tuple(v / n for v in q))  # type: ignore[arg-type]


def _float_euler_to_quat(seq: str, frame: str, angles: Sequence[float]) -> tuple[float, float, float, float]:
    qs = [_elem_quat(_AXIS[a], math.cos(t / 2), math.sin(t / 2)) for a, t in zip(seq, angles)]
    if frame == "extrinsic":
        qs = qs[::-1]
    q = _qmul(_qmul(qs[0], qs[1]), qs[2])
    return q


def _float_euler_to_matrix(seq: str, frame: str, angles: Sequence[float]) -> list[list[float]]:
    mats = [_elem_matrix(_AXIS[a], math.cos(t), math.sin(t)) for a, t in zip(seq, angles)]
    if frame == "extrinsic":
        mats = mats[::-1]
    return _matmul(_matmul(mats[0], mats[1]), mats[2])


def _float_quat_to_euler(q: Sequence[float], seq: str, frame: str) -> tuple[tuple[float, float, float], bool, bool]:
    """Bernardes & Viollet (2022, PLOS ONE 17(11): e0276302), as in SciPy (BSD-3).

    Difference from SciPy: SciPy declares gimbal lock whenever the middle angle is
    within 1e-7 rad of the singularity and then forces the third angle to zero,
    which makes the recomposed rotation wrong by up to ~6e-8 (measured, see the G1
    report). Here only the singularity at the resolution of the data (one pair
    within 8 ulps of zero relative to the other) is treated specially; near it the closed form is still accurate because the rotation is
    insensitive to the ill-determined angle difference. Returns
    (angles, gimbal_lock, near_gimbal).
    """
    w, x, y, z = (float(v) for v in q)
    qq = (x, y, z, w)  # SciPy's scalar-last indexing
    extrinsic = frame == "extrinsic"
    axes = [_AXIS[a] for a in seq]
    if not extrinsic:
        axes = axes[::-1]
    i, j, k = axes
    symmetric = i == k
    if symmetric:
        k = 3 - i - j
    sign = (i - j) * (j - k) * (k - i) // 2
    if symmetric:
        a, b, c, d = qq[3], qq[i], qq[j], qq[k] * sign
    else:
        a, b, c, d = qq[3] - qq[j], qq[i] + qq[k] * sign, qq[j] + qq[3], qq[k] * sign - qq[i]
    hab, hcd = math.hypot(a, b), math.hypot(c, d)
    angles = [0.0, 2 * math.atan2(hcd, hab), 0.0]
    half_sum = math.atan2(b, a)
    half_diff = math.atan2(d, c)
    first, third = (0, 2) if extrinsic else (2, 0)
    # Singular only when one pair vanishes AT THE RESOLUTION OF THE DATA (a few ulps
    # relative to the other pair): zeroing an angle then moves the rotation by
    # O(1e-15), not by SciPy's O(1e-7).
    small_cd = hcd <= 8 * _EPS * hab
    gimbal = small_cd or hab <= 8 * _EPS * hcd
    if gimbal:
        # SciPy's rule: the free angle goes to index 0, the LAST angle of the sequence is 0.
        angles[0] = 2 * half_sum if small_cd else 2 * half_diff * (-1 if extrinsic else 1)
        a3 = angles[third]
    else:
        angles[first] = half_sum - half_diff
        a3 = half_sum + half_diff
    if not symmetric:
        a3 *= sign
        angles[1] -= math.pi / 2
    angles[third] = a3
    angles = [((v + math.pi) % (2 * math.pi)) - math.pi for v in angles]
    near = not gimbal and (abs(angles[1] - (0.0 if symmetric else math.pi / 2)) < 1e-7
                           or abs(angles[1] + (0.0 if symmetric else math.pi / 2)) < 1e-7
                           or (symmetric and abs(abs(angles[1]) - math.pi) < 1e-7))
    return (angles[0], angles[1], angles[2]), gimbal, near


def _float_axis_angle_to_quat(axis: Sequence[float], angle: float) -> tuple[float, float, float, float]:
    s, c = math.sin(angle / 2), math.cos(angle / 2)
    return (c, axis[0] * s, axis[1] * s, axis[2] * s)


def _float_quat_to_axis_angle(q: Sequence[float]) -> tuple[tuple[float, float, float], float, bool]:
    w, x, y, z = _canonical([float(v) for v in q])
    vn = math.sqrt(x * x + y * y + z * z)
    if vn == 0.0:
        return (1.0, 0.0, 0.0), 0.0, True
    angle = 2 * math.atan2(vn, w)
    return (x / vn, y / vn, z / vn), angle, False


# ===================================================== rotation: public API
_KINDS = ("quat", "matrix", "euler", "axis_angle")


def convert_rotation(
    value: Any,
    source: str,
    target: str,
    *,
    seq: "str | None" = None,
    frame: "str | None" = None,
    target_seq: "str | None" = None,
    target_frame: "str | None" = None,
    degrees: bool = False,
    quat: QuatConvention = QuatConvention(),
    matrix: MatrixConvention = MatrixConvention(),
    normalize: bool = False,
) -> RotationResult:
    """Convert a rotation between representations, with a certificate.

    ``source``/``target`` are ``"quat"``, ``"matrix"``, ``"euler"``, ``"axis_angle"``.
    Euler needs ``seq`` (one of :data:`EULER_SEQUENCES`) and ``frame``
    (``"intrinsic"``/``"extrinsic"``) — both required, there is no default
    convention to be silently wrong about; ``target_seq``/``target_frame`` default
    to ``seq``/``frame``. Axis-angle input is ``(axis, angle)``; output is
    ``((ax, ay, az), angle)`` with angle in ``[0, pi]`` (degrees if ``degrees``).
    Euler angles come out in ``[-pi, pi)`` (as SciPy), the middle one in
    ``[-pi/2, pi/2]`` (Tait-Bryan) or ``[0, pi]`` (proper Euler).

        >>> r = convert_rotation([1, 0, 0, 1], "quat", "euler", seq="zyx", frame="intrinsic",
        ...                      degrees=True, normalize=True)
        >>> [round(a, 12) for a in r.value]        # 90 deg yaw
        [90.0, 0.0, 0.0]
        >>> r.certificate["status"]
        'verified'
    """
    if source not in _KINDS or target not in _KINDS:
        raise DomainError(f"rotation kinds are {', '.join(_KINDS)}; got {source!r} -> {target!r}")
    what = f"{source}->{target}"
    notes: list[str] = []
    exact_out: Any = None
    src_oe = 0.0
    # ---- read the source into (certificate route, internal float quaternion)
    if source == "quat":
        raw, internal, norm = _read_quat(value, quat, normalize, f"{what}: quaternion")
        src_route: tuple[str, Any] = ("quat", internal)
        qf = tuple(float(v) for v in internal)
        n = math.sqrt(_qnorm2(qf))
        qf = tuple(v / n for v in qf)
        if norm != 1.0:
            notes.append(f"input norm {norm:.17g}; used q/|q|")
    elif source == "matrix":
        M, src_oe, _ = _read_matrix(value, matrix, f"{what}: matrix")
        src_route = ("matrix", M)
        qf = _float_matrix_to_quat([[float(v) for v in r] for r in M])
    elif source == "euler":
        s, fr, ang = _read_euler(value, seq, frame, degrees, f"{what}: angles")
        src_route = ("euler", (s, fr, ang))
        qf = _float_euler_to_quat(s, fr, ang)
    else:
        try:
            axis_in, angle_in = value
        except (TypeError, ValueError) as exc:
            raise DomainError(f"{what}: axis-angle input is (axis, angle)") from exc
        ax, ang1 = _read_axis_angle(axis_in, angle_in, degrees, what)
        src_route = ("axis", (ax, ang1))
        qf = _float_axis_angle_to_quat(ax, ang1)

    gimbal = False
    extra: dict[str, Any] = {}
    # ---- produce the target, and its OWN route for the certificate
    if target == "quat":
        q_out = _canonical(qf) if source != "quat" else qf
        tgt_route: tuple[str, Any] = ("quat", q_out)
        out: Any = tuple(float(v) for v in _from_internal(q_out, quat))
    elif target == "matrix":
        if source == "euler":
            Mf = _float_euler_to_matrix(*src_route[1])  # direct, more accurate than via q
        elif source == "quat" and _is_exact(src_route[1]):
            Mex = _quat_matrix([_F(v) for v in src_route[1]])
            Mf = [[float(v) for v in r] for r in Mex]
            exact_rows = _transpose(Mex) if matrix.transposed else Mex
            exact_out = tuple(tuple(_fstr(v) for v in r) for r in exact_rows)
            g = _matmul(_transpose(Mex), Mex)
            extra["exact_orthogonal"] = all(g[i][j] == (1 if i == j else 0) for i in range(3) for j in range(3))
            extra["exact_det"] = _fstr(_det3(Mex))
        elif source == "quat":
            Mf = _float_quat_to_matrix(src_route[1])
        else:
            Mf = _float_quat_to_matrix(qf)
        tgt_route = ("matrix", Mf)
        oe, dm = _orthogonality(Mf)
        extra["orthogonality_error"], extra["det_minus_1"] = oe, dm
        Mo = _transpose(Mf) if matrix.transposed else Mf
        out = tuple(tuple(float(v) for v in r) for r in Mo)
    elif target == "euler":
        ts, tf = _check_seq(target_seq or seq, target_frame or frame)
        angs, gimbal, near = _float_quat_to_euler(qf, ts, tf)
        tgt_route = ("euler", (ts, tf, angs))
        if gimbal:
            notes.append("gimbal lock: the middle angle is at its singularity, the first and third "
                         "angles are not unique; the third was set to 0")
        elif near:
            notes.append("near gimbal lock (middle angle within 1e-7 rad of the singularity): "
                         "angles are accurate but ill-conditioned")
        out = tuple(math.degrees(a) if degrees else a for a in angs)
        extra["sequence"], extra["frame"] = ts, tf
    else:
        axis, angle, ident = _float_quat_to_axis_angle(qf)
        tgt_route = ("axis", (axis, angle))
        if ident:
            notes.append("identity rotation: the axis is arbitrary; (1, 0, 0) returned")
        out = (axis, math.degrees(angle) if degrees else angle)

    tol = _FLOAT_TOL
    if source == "matrix":
        # the input's own drift from orthogonality is inherited by any exact rotation
        tol += 4 * src_oe
        extra["input_orthogonality_error"] = src_oe
    cert = _certify(src_route, tgt_route, tol, what, extra)
    return RotationResult(
        kind="rotation_convert", source=source, target=target, value=out,
        conventions={"quat": quat.to_dict(), "matrix": matrix.to_dict(),
                     "euler": {"seq": seq, "frame": frame} if "euler" in (source, target) else None,
                     "degrees": degrees},
        certificate=cert, exact=exact_out, gimbal_lock=gimbal, notes=tuple(notes),
    )


def quat_to_matrix(q: Any, *, quat: QuatConvention = QuatConvention(),
                   matrix: MatrixConvention = MatrixConvention(), normalize: bool = False) -> RotationResult:
    """Quaternion -> 3x3 rotation matrix (see :func:`convert_rotation`).

        >>> quat_to_matrix([1, 0, 0, 1], normalize=True).exact
        (('0', '-1', '0'), ('1', '0', '0'), ('0', '0', '1'))
    """
    return convert_rotation(q, "quat", "matrix", quat=quat, matrix=matrix, normalize=normalize)


def matrix_to_quat(m: Any, *, quat: QuatConvention = QuatConvention(),
                   matrix: MatrixConvention = MatrixConvention()) -> RotationResult:
    """3x3 rotation matrix -> unit quaternion with w >= 0 (Shepperd's method)."""
    return convert_rotation(m, "matrix", "quat", quat=quat, matrix=matrix)


def euler_to_matrix(angles: Any, seq: str, *, frame: str, degrees: bool = False,
                    matrix: MatrixConvention = MatrixConvention()) -> RotationResult:
    """Euler angles -> matrix. ``frame`` is required: intrinsic ``R1@R2@R3``, extrinsic ``R3@R2@R1``."""
    return convert_rotation(angles, "euler", "matrix", seq=seq, frame=frame, degrees=degrees, matrix=matrix)


def matrix_to_euler(m: Any, seq: str, *, frame: str, degrees: bool = False,
                    matrix: MatrixConvention = MatrixConvention()) -> RotationResult:
    """Matrix -> Euler angles; ``gimbal_lock`` says when the decomposition is not unique."""
    return convert_rotation(m, "matrix", "euler", seq=seq, frame=frame, degrees=degrees, matrix=matrix)


def euler_to_quat(angles: Any, seq: str, *, frame: str, degrees: bool = False,
                  quat: QuatConvention = QuatConvention()) -> RotationResult:
    return convert_rotation(angles, "euler", "quat", seq=seq, frame=frame, degrees=degrees, quat=quat)


def quat_to_euler(q: Any, seq: str, *, frame: str, degrees: bool = False,
                  quat: QuatConvention = QuatConvention(), normalize: bool = False) -> RotationResult:
    return convert_rotation(q, "quat", "euler", seq=seq, frame=frame, degrees=degrees, quat=quat,
                            normalize=normalize)


def axis_angle_to_quat(axis: Any, angle: Any, *, degrees: bool = False,
                       quat: QuatConvention = QuatConvention()) -> RotationResult:
    return convert_rotation((axis, angle), "axis_angle", "quat", degrees=degrees, quat=quat)


def axis_angle_to_matrix(axis: Any, angle: Any, *, degrees: bool = False,
                         matrix: MatrixConvention = MatrixConvention()) -> RotationResult:
    return convert_rotation((axis, angle), "axis_angle", "matrix", degrees=degrees, matrix=matrix)


def quat_to_axis_angle(q: Any, *, degrees: bool = False, quat: QuatConvention = QuatConvention(),
                       normalize: bool = False) -> RotationResult:
    return convert_rotation(q, "quat", "axis_angle", degrees=degrees, quat=quat, normalize=normalize)


def matrix_to_axis_angle(m: Any, *, degrees: bool = False,
                         matrix: MatrixConvention = MatrixConvention()) -> RotationResult:
    return convert_rotation(m, "matrix", "axis_angle", degrees=degrees, matrix=matrix)


def quat_multiply(a: Any, b: Any, *, quat: QuatConvention = QuatConvention()) -> RotationResult:
    """Quaternion product in the stated algebra (exact for exact input).

    Hamilton: ``R(a*b) = R(a) @ R(b)`` — ``b`` is applied first. JPL:
    ``a (x)_JPL b = b (x)_H a`` on the numbers, and with JPL's own matrix map the
    composition rule is the same, ``C(a*b) = C(a) @ C(b)``.

        >>> quat_multiply([0, 1, 0, 0], [0, 0, 1, 0]).exact      # i*j = k
        ('0', '0', '0', '1')
        >>> quat_multiply([0, 1, 0, 0], [0, 0, 1, 0], quat=QuatConvention(algebra="jpl")).exact
        ('0', '0', '0', '-1')
    """
    ra = _vector(a, 4, "quat_multiply: a")
    rb = _vector(b, 4, "quat_multiply: b")
    exact = _is_exact(ra + rb)
    conv_ord = QuatConvention(order=quat.order)
    wa = _to_internal(ra, conv_ord)  # reorder only; algebra handled by the product rule
    wb = _to_internal(rb, conv_ord)
    if exact:
        wa = tuple(_F(v) for v in wa)  # type: ignore[assignment]
        wb = tuple(_F(v) for v in wb)  # type: ignore[assignment]
    else:
        wa = tuple(float(v) for v in wa)  # type: ignore[assignment]
        wb = tuple(float(v) for v in wb)  # type: ignore[assignment]
    prod = _qmul(wa, wb) if quat.algebra == "hamilton" else _qmul(wb, wa)
    out = _from_internal(prod, conv_ord)
    # certificate: matrix of the product == product of matrices (independent route)
    with mpmath.workdps(CERT_DPS):
        conv_m = (lambda q: _quat_matrix(q)) if quat.algebra == "hamilton" else (lambda q: _transpose(_quat_matrix(q)))
        lhs = conv_m([_mp(v) for v in prod])
        rhs = _matmul(conv_m([_mp(v) for v in wa]), conv_m([_mp(v) for v in wb]))
        res = _mp_residual(lhs, rhs)
    tol = 0.0 if exact else _FLOAT_TOL
    if not res <= tol + (1e-40 if exact else 0):
        raise CertificationError(f"quat_multiply: R(a*b) != R(a)R(b) (residual {res:.3g})", residual=res, tolerance=tol)
    return RotationResult(
        kind="quat_multiply", source="quat", target="quat",
        value=tuple(float(v) for v in out), conventions={"quat": quat.to_dict()},
        certificate={"status": "verified", "method": "R(a*b) == R(a) @ R(b) at 50 digits", "residual": res,
                     "tolerance": tol, "formulas_proved": "prove_rotation_identities()"},
        exact=tuple(_fstr(v) for v in out) if exact else None,
    )


def mirror_rotation(m: Any, axis: str = "z") -> RotationResult:
    """Re-express a rotation matrix in the mirrored (other-handed) frame: ``S @ R @ S``.

    ``S`` flips one axis (``"z"`` is the OpenGL <-> Direct3D/Unity change). The
    result is again a proper rotation: the same physical motion seen in a frame
    of the other handedness. Exact for exact input.
    """
    if axis not in _AXIS:
        raise DomainError(f"mirror_rotation: axis must be x, y or z, got {axis!r}")
    M, _, _ = _read_matrix(m, MatrixConvention(), "mirror_rotation")
    k = _AXIS[axis]
    out = [[(-v if (i == k) != (j == k) else v) for j, v in enumerate(r)] for i, r in enumerate(M)]
    exact = _is_exact(v for r in M for v in r)
    cert = _certify(("matrix", _matmul(_matmul(_mirror(k), M), _mirror(k))), ("matrix", out),
                    0.0 if exact else _FLOAT_TOL, "mirror_rotation")
    return RotationResult(kind="mirror_rotation", source="matrix", target="matrix",
                          value=tuple(tuple(float(v) for v in r) for r in out),
                          conventions={"mirror_axis": axis}, certificate=cert,
                          exact=tuple(tuple(_fstr(_F(v)) for v in r) for r in out) if exact else None)


def _mirror(k: int) -> list[list[int]]:
    return [[(-1 if i == k else 1) if i == j else 0 for j in range(3)] for i in range(3)]


# ======================================================================== slerp
def _sinc(x: Any) -> Any:
    """sin(x)/x, accurate through 0 (works for float and mpf)."""
    if abs(x) < 1e-4:
        x2 = x * x
        return 1 - x2 / 6 + x2 * x2 / 120
    return (mpmath.sin(x) if isinstance(x, mpmath.mpf) else math.sin(x)) / x


def _slerp_core(a: Sequence[Any], b: Sequence[Any], t: Any, sqrt: Callable[[Any], Any],
                atan2: Callable[[Any, Any], Any]) -> tuple[list[Any], Any]:
    """Great-arc interpolation on S^3 in the sinc form (no division by sin(theta))."""
    d = [x - y for x, y in zip(a, b)]
    s = [x + y for x, y in zip(a, b)]
    theta = 2 * atan2(sqrt(sum(v * v for v in d)), sqrt(sum(v * v for v in s)))
    st = _sinc(theta)
    c0 = (1 - t) * _sinc((1 - t) * theta) / st
    c1 = t * _sinc(t * theta) / st
    out = [c0 * x + c1 * y for x, y in zip(a, b)]
    n = sqrt(sum(v * v for v in out))
    return [v / n for v in out], theta


def slerp(q0: Any, q1: Any, t: Any, *, shortest: bool = True,
          quat: QuatConvention = QuatConvention()) -> RotationResult:
    """Spherical linear interpolation between unit quaternions, certified.

    ``shortest=True`` (default) replaces ``q1`` by ``-q1`` when ``q0 . q1 < 0`` —
    the same rotation, but the arc through it is the shorter one (forgetting this
    is the classic "the character spins the long way round" bug). The angle is
    taken as ``2*atan2(|q0-q1|, |q0+q1|)`` (accurate for nearly equal AND nearly
    opposite endpoints) and the weights in the sinc form
    ``(1-t) sinc((1-t)θ)/sinc(θ)``, so nearly parallel endpoints need no lerp
    fallback and keep constant angular speed. ``t`` outside [0, 1] extrapolates.

    Certificate: the result is recomputed at 50 digits AND checked to lie on the
    geodesic: angle(q0, q_t) = t·θ and |q_t| = 1.

        >>> r = slerp([1, 0, 0, 0], [0, 0, 0, 1], 0.5)
        >>> [round(v, 12) for v in r.value]
        [0.707106781187, 0.0, 0.0, 0.707106781187]
    """
    _, a, _ = _read_quat(q0, quat, False, "slerp: q0")
    _, b, _ = _read_quat(q1, quat, False, "slerp: q1")
    tt = _scalar(t, "slerp: t")
    af = [float(v) for v in a]
    bf = [float(v) for v in b]
    na, nb = math.sqrt(_qnorm2(af)), math.sqrt(_qnorm2(bf))
    af, bf = [v / na for v in af], [v / nb for v in bf]
    dot = sum(x * y for x, y in zip(af, bf))
    flipped = shortest and dot < 0
    if flipped:
        bf = [-v for v in bf]
    if not shortest and sum((x + y) ** 2 for x, y in zip(af, bf)) == 0:
        raise DomainError("slerp: q1 = -q0 with shortest=False — the great circle through "
                          "antipodal quaternions is not unique (it is a full turn about ANY axis)")
    out, theta = _slerp_core(af, bf, float(tt), math.sqrt, math.atan2)
    with mpmath.workdps(CERT_DPS):
        A, B, T = [_mp(v) for v in af], [_mp(v) for v in bf], _mp(tt)
        ref, th = _slerp_core(A, B, T, mpmath.sqrt, mpmath.atan2)
        res = max(abs(_mp(x) - y) for x, y in zip(out, ref))
        o = [_mp(v) for v in out]
        dist = [x - y for x, y in zip(A, o)]
        summ = [x + y for x, y in zip(A, o)]
        ang = 2 * mpmath.atan2(mpmath.sqrt(sum(v * v for v in dist)), mpmath.sqrt(sum(v * v for v in summ)))
        geod = abs(ang - abs(T) * th) if 0 <= T <= 1 else mpmath.mpf(0)
        unit = abs(mpmath.sqrt(sum(v * v for v in o)) - 1)
    tol = _FLOAT_TOL * (1 + abs(float(tt)))
    worst = float(max(res, unit, geod))
    if not worst <= tol:
        raise CertificationError(f"slerp: result off the geodesic by {worst:.3g}", residual=worst, tolerance=tol)
    q_out = _from_internal(out, quat)
    return RotationResult(
        kind="slerp", source="quat", target="quat", value=tuple(float(v) for v in q_out),
        conventions={"quat": quat.to_dict(), "shortest": shortest},
        certificate={"status": "verified", "method": "50-digit recomputation + geodesic check",
                     "residual": float(res), "geodesic_error": float(geod), "unit_error": float(unit),
                     "tolerance": tol},
        notes=(("q1 negated: shorter arc taken",) if flipped else ())
        + ((f"endpoints {math.degrees(2 * theta):.6g} deg apart as rotations",)),
    )


# ======================================================= symbolic formula proofs
@lru_cache(maxsize=1)
def prove_rotation_identities() -> tuple[dict[str, Any], ...]:
    """Prove, once, the identities every rotation certificate leans on.

    All proofs are polynomial identities checked by exact expansion over the
    integers (no floating point, no sampling). Trigonometric statements use the
    rational parametrisation ``cos a = (1-u^2)/(1+u^2)``, ``sin a = 2u/(1+u^2)``
    (``u = tan(a/2)``), which turns them into polynomial identities as well.

    Returns one record per identity: ``{"id", "statement", "proved": True}``.
    """
    return guard("geometry.prove_rotation_identities", "rotation formulas", _prove_rotation_identities)


def _prove_rotation_identities() -> tuple[dict[str, Any], ...]:
    from sympy import QQ
    from sympy.polys.rings import ring

    R, w1, x1, y1, z1, w2, x2, y2, z2, u1, u2, u3, a1, b1, c1 = ring(
        "w1 x1 y1 z1 w2 x2 y2 z2 u1 u2 u3 a1 b1 c1", QQ)
    q1, q2 = (w1, x1, y1, z1), (w2, x2, y2, z2)
    n1, n2 = _qnorm2(q1), _qnorm2(q2)
    N1, N2 = _quat_numerator(q1), _quat_numerator(q2)
    I3 = [[R(1) if i == j else R(0) for j in range(3)] for i in range(3)]
    out: list[dict[str, Any]] = []

    def record(ident: str, statement: str, ok: bool) -> None:
        if not ok:  # pragma: no cover - a failed proof is a bug in this file
            raise AssertionError(f"identity {ident} failed to prove")
        out.append({"id": ident, "statement": statement, "proved": True})

    def eq(A: Any, B: Any) -> bool:
        return all(A[i][j] == B[i][j] for i in range(3) for j in range(3))

    g = _matmul(_transpose(N1), N1)
    record("orthogonal", "N(q)^T N(q) = |q|^4 I, so R(q) = N(q)/|q|^2 is orthogonal for every q != 0",
           eq(g, [[v * n1 * n1 for v in r] for r in I3]))
    record("det", "det N(q) = |q|^6, so det R(q) = +1", _det3(N1) == n1 ** 3)
    p = _qmul(q1, q2)
    record("homomorphism", "N(q1 (x) q2) = N(q1) N(q2): Hamilton product composes like column-vector matrices",
           eq(_quat_numerator(p), _matmul(N1, N2)))
    record("double_cover", "N(-q) = N(q)", eq(_quat_numerator(tuple(-v for v in q1)), N1))
    record("inverse", "N(conj q) = N(q)^T", eq(_quat_numerator(_qconj(q1)), _transpose(N1)))
    pj = _qmul(q2, q1)
    record("jpl_homomorphism", "with C(q) = N(q)^T and p (x)_JPL q = q (x)_H p: C(q1 (x)_JPL q2) = C(q1) C(q2)",
           eq(_transpose(_quat_numerator(pj)), _matmul(_transpose(N1), _transpose(N2))))
    # Euler: elementary quaternion (half angle) maps to elementary matrix (full angle).
    # half angle via u = tan(a/4): ch = (1-u^2)/(1+u^2), sh = 2u/(1+u^2); then the full
    # angle cosine/sine follow from the double-angle polynomials of (ch, sh).
    for axis in range(3):
        ch, sh = 1 - u1 * u1, 2 * u1  # scaled by (1+u^2); scale^2 cancels in N/|q|^2
        qe = _elem_quat(axis, ch, sh)
        nq = _qnorm2(qe)  # (1+u^2)^2
        c_full = ch * ch - sh * sh  # cos(a) * (1+u^2)^2
        s_full = 2 * ch * sh  # sin(a) * (1+u^2)^2
        E = _elem_matrix(axis, c_full, s_full)
        # E has 1 on the axis diagonal; scale it to (1+u^2)^2 to compare with N
        E = [[(nq if (i == j == axis) else v) if (i == axis or j == axis) else v for j, v in enumerate(r)]
             for i, r in enumerate(E)]
        record(f"elementary_{'xyz'[axis]}", f"N(cos(a/2) + sin(a/2) e_{'xyz'[axis]}) = |q|^2 R_{'xyz'[axis]}(a)",
               eq(_quat_numerator(qe), E))
    for seq in EULER_SEQUENCES:
        for frame in ("intrinsic", "extrinsic"):
            qs = [_elem_quat(_AXIS[a], 1 - u * u, 2 * u) for a, u in zip(seq, (u1, u2, u3))]
            ms = [_quat_numerator(qe) for qe in qs]
            if frame == "extrinsic":
                qs, ms = qs[::-1], ms[::-1]
            qprod = _qmul(_qmul(qs[0], qs[1]), qs[2])
            record(f"euler_{seq}_{frame}",
                   f"{seq} {frame}: quaternion route == elementary-matrix route",
                   eq(_quat_numerator(qprod), _matmul(_matmul(ms[0], ms[1]), ms[2])))
    # Rodrigues: axis (a1,b1,c1) (not normalised), angle via u = tan(angle/2)
    k2 = a1 * a1 + b1 * b1 + c1 * c1
    qa = (1 - u1 * u1, 2 * u1 * a1, 2 * u1 * b1, 2 * u1 * c1)  # needs k2 = 1 -> use homogeneous check
    K = [[R(0), -c1, b1], [c1, R(0), -a1], [-b1, a1, R(0)]]
    K2 = _matmul(K, K)
    # with k2 = 1: |q|^2 = (1+u^2)^2, |q|^2 R = (1+u^2)^2 I + 2u(1-u^2)*2 K + 8u^2 K^2 (since
    # sin a = 2u/(1+u^2)·(1-u^2)/(1+u^2)*2..., written out via the double angle)
    nq = (1 - u1 * u1) ** 2 + 4 * u1 * u1 * k2
    rod = [[(nq if i == j else R(0)) + 4 * u1 * (1 - u1 * u1) * K[i][j] + 8 * u1 * u1 * K2[i][j]
            for j in range(3)] for i in range(3)]
    Nq = _quat_numerator(qa)
    # N(q) and Rodrigues agree modulo (k2 - 1): substitute a1^2 = 1 - b1^2 - c1^2
    diff = [[(Nq[i][j] - rod[i][j]) for j in range(3)] for i in range(3)]
    sub = 1 - b1 * b1 - c1 * c1
    ok = all(_reduce_a2(diff[i][j], a1, sub) == 0 for i in range(3) for j in range(3))
    record("rodrigues", "N(cos(a/2), sin(a/2) k) = |q|^2 (I + sin a K + (1 - cos a) K^2) for |k| = 1", ok)
    return tuple(out)


def _reduce_a2(p: Any, a: Any, sub: Any) -> Any:
    """Replace a^2 by ``sub`` repeatedly (reduction modulo a^2 - sub)."""
    gens = p.ring.gens
    idx = gens.index(a)
    result = p.ring(0)
    for monom, coeff in p.terms():
        e = monom[idx]
        rest = list(monom)
        rest[idx] = e % 2
        term = p.ring({tuple(rest): coeff})
        result += term * sub ** (e // 2)
    return result


# ====================================================== convention detector
_NAME_RE = re.compile(r"^(q|R|M)_?(\w+)$")
_CONSTS = {"Sx": 0, "Sy": 1, "Sz": 2}
_CAUSE_TEXT = {
    "reversed_order": "multiplication order reversed (q1*q2 applies q2 FIRST; R1@R2 applies R2 first)",
    "transposed": "one side is the transpose/inverse of the other (active vs passive, or a missing .T)",
    "row_vs_column_vectors": "row-vector (v@M) matrices read as column-vector (M@v) ones, or vice versa",
    "jpl_vs_hamilton": "JPL quaternion algebra (i*j = -k) mixed with Hamilton (i*j = k)",
    "component_order": "quaternion components read as wxyz on one side and xyzw on the other",
    "handedness_mirror_x": "a handedness change (mirror S_x R S_x) applied to one side only",
    "handedness_mirror_y": "a handedness change (mirror S_y R S_y) applied to one side only",
    "handedness_mirror_z": "a handedness change (mirror S_z R S_z) applied to one side only",
}

#: integer quaternions used as witnesses (component-distinct, non-special)
_WITNESS_QUATS = ((1, 2, 3, 4), (2, -1, 1, 3), (3, 1, -2, 1), (1, -3, 2, -2), (4, 1, 1, -1), (2, 3, -1, 1))


@dataclass(frozen=True)
class _Side:
    tree: ast.AST
    order: str
    algebra: str
    transposed: bool


def _parse_rot_expr(text: Any, names: "dict[str, tuple[str, str]] | None") -> tuple[ast.AST, dict[str, tuple[str, str]]]:
    if not isinstance(text, str) or not text.strip():
        raise DomainError("compare_rotation_expressions: expressions are non-empty strings like 'q1*q2' or 'R1@R2'")
    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as exc:
        raise DomainError(f"compare_rotation_expressions: cannot parse {text!r}") from exc
    found: dict[str, tuple[str, str]] = {}

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if node.id in _CONSTS or node.id == "I":
                return
            if names and node.id in names:
                found[node.id] = names[node.id]
                return
            m = _NAME_RE.match(node.id)
            if not m:
                raise DomainError(
                    f"compare_rotation_expressions: cannot tell what {node.id!r} is — name quaternions "
                    "q1, q2, ... and matrices R1, R2, ... (same suffix = same rotation), or pass names=")
            found[node.id] = ("quat" if m.group(1) == "q" else "matrix", m.group(2))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.MatMult)):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            walk(node.operand)
        elif isinstance(node, ast.Attribute) and node.attr in ("T", "conj", "inverse"):
            walk(node.value)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in ("conj", "inv", "inverse", "transpose") and len(node.args) == 1 and not node.keywords):
            walk(node.args[0])
        else:
            raise DomainError(
                f"compare_rotation_expressions: unsupported syntax {ast.unparse(node)!r} — use names, "
                "'*' (quaternion product), '@' (matrix product), '.T', conj(), inv(), unary '-', "
                "and the mirrors Sx, Sy, Sz")
    walk(tree)
    return tree, found


def _eval_side(side: _Side, kinds: dict[str, tuple[str, str]], quats: Mapping[str, Sequence[Any]],
               reverse: bool = False) -> tuple[list[list[Any]], Any]:
    """Evaluate one side to (N, scale): the rotation it denotes is N / scale (active, column)."""
    one = next(iter(quats.values()))[0] * 0 + 1 if quats else 1

    def interp(stored: Sequence[Any]) -> tuple[Any, Any, Any, Any]:
        # stored array -> Hamilton (w,x,y,z) of the rotation this side believes it is
        w, x, y, z = (stored[3], stored[0], stored[1], stored[2]) if side.order == "xyzw" else tuple(stored)
        return (w, x, y, z) if side.algebra == "hamilton" else (w, -x, -y, -z)

    def mat_of(stored: Sequence[Any]) -> tuple[list[list[Any]], Any]:
        h = interp(stored)
        N = _quat_numerator(h)
        return (_transpose(N) if side.transposed else N), _qnorm2(h)

    def ev(node: ast.AST) -> tuple[str, Any, Any]:
        if isinstance(node, ast.Name):
            if node.id in _CONSTS:
                return ("matrix", [[one * v for v in r] for r in _mirror(_CONSTS[node.id])], one)
            if node.id == "I":
                return ("matrix", [[one * (i == j) for j in range(3)] for i in range(3)], one)
            rep, ident = kinds[node.id]
            if rep == "quat":
                return ("quat", tuple(quats[ident]), None)
            N, s = mat_of(quats[ident])
            return ("matrix", N, s)
        if isinstance(node, ast.UnaryOp):
            k, v, s = ev(node.operand)
            if k != "quat":
                raise DomainError("compare_rotation_expressions: unary minus on a matrix is not a rotation")
            return ("quat", tuple(-c for c in v), None)
        if isinstance(node, ast.Attribute) or isinstance(node, ast.Call):
            arg = node.value if isinstance(node, ast.Attribute) else node.args[0]
            k, v, s = ev(arg)
            if k == "quat":
                return ("quat", (v[0], -v[1], -v[2], -v[3]), None)
            return ("matrix", _transpose(v), s)
        assert isinstance(node, ast.BinOp)
        lk, lv, ls = ev(node.left)
        rk, rv, rs = ev(node.right)
        if reverse:
            lk, lv, ls, rk, rv, rs = rk, rv, rs, lk, lv, ls
        if isinstance(node.op, ast.Mult):
            if lk == rk == "quat":
                return ("quat", _qmul_side(lv, rv), None)
            if lk == rk == "matrix":
                raise DomainError(
                    "compare_rotation_expressions: '*' between matrices is the ELEMENTWISE product in NumPy "
                    "(not a rotation) — use '@'")
            raise DomainError("compare_rotation_expressions: cannot multiply a quaternion by a matrix")
        if lk == rk == "matrix":
            return ("matrix", _matmul(lv, rv), ls * rs)
        raise DomainError("compare_rotation_expressions: '@' needs matrices on both sides (use '*' for quaternions)")

    def _qmul_side(a: Any, b: Any) -> tuple[Any, ...]:
        # product of STORED arrays in this side's algebra & order
        ha = (a[3], a[0], a[1], a[2]) if side.order == "xyzw" else tuple(a)
        hb = (b[3], b[0], b[1], b[2]) if side.order == "xyzw" else tuple(b)
        p = _qmul(ha, hb) if side.algebra == "hamilton" else _qmul(hb, ha)
        return (p[1], p[2], p[3], p[0]) if side.order == "xyzw" else p

    k, v, s = ev(side.tree)
    if k == "quat":
        h = interp(v)
        return _quat_numerator(h), _qnorm2(h)  # a quaternion result is a rotation already
    return (_transpose(v) if side.transposed else v), s


def _same(a: tuple[list[list[Any]], Any], b: tuple[list[list[Any]], Any]) -> bool:
    (Na, sa), (Nb, sb) = a, b
    return all(Na[i][j] * sb == Nb[i][j] * sa for i in range(3) for j in range(3))


@dataclass(frozen=True)
class ConventionReport(_Result):
    """Outcome of :func:`compare_rotation_expressions`.

    ``status``: ``"verified"`` (proved equal for every input, ``method="symbolic"``),
    ``"refuted"`` (``witness`` gives, for every rotation id, integer quaternion
    components at which the two sides differ EXACTLY; ``matrix_a``/``matrix_b``
    are the two exact rotation matrices there), or ``"undecided"``.
    ``causes``: names of the convention changes that make the two sides agree at
    every witness tried — the likely bug (see ``explanations``).
    """

    kind: str
    status: str
    method: str
    a: str
    b: str
    conventions: dict[str, Any]
    witness: dict[str, dict[str, Any]]
    matrix_a: Any
    matrix_b: Any
    max_abs_difference: "float | None"
    causes: tuple[str, ...]
    explanations: tuple[str, ...]
    detail: str


def _quat_text(q: Sequence[int]) -> str:
    n2 = sum(v * v for v in q)
    r = math.isqrt(n2)
    norm = str(r) if r * r == n2 else f"sqrt({n2})"
    return f"({', '.join(str(v) for v in q)})" + ("" if norm == "1" else f" / {norm}")


def _describe(q: Sequence[int], order: str, algebra: str) -> dict[str, Any]:
    w, x, y, z = (q[3], q[0], q[1], q[2]) if order == "xyzw" else tuple(q)
    if algebra == "jpl":
        x, y, z = -x, -y, -z
    axis, angle, _ = _float_quat_to_axis_angle([float(v) for v in (w, x, y, z)])
    return {"components": _quat_text(q), "order": order, "algebra": algebra,
            "axis": [round(v, 6) for v in axis], "angle_deg": round(math.degrees(angle), 6)}


def compare_rotation_expressions(
    a: str,
    b: str,
    *,
    quat: QuatConvention = QuatConvention(),
    matrix: MatrixConvention = MatrixConvention(),
    quat_b: "QuatConvention | None" = None,
    matrix_b: "MatrixConvention | None" = None,
    names: "dict[str, tuple[str, str]] | None" = None,
) -> ConventionReport:
    """Do two rotation formulas denote the same rotation — for EVERY input?

    Each id (``1`` in ``q1``/``R1``) is one rotation, stored as four quaternion
    components; ``R1`` is ITS matrix under that side's conventions (``quat``,
    ``matrix`` for side a; ``quat_b``/``matrix_b`` for side b, defaulting to a's).
    Syntax: ``*`` quaternion product, ``@`` matrix product, ``.T``, ``conj()``,
    ``inv()``, unary ``-``, and the mirrors ``Sx``, ``Sy``, ``Sz``.

    VERIFIED is a proof: both sides are reduced to matrices of polynomials in the
    symbolic components and compared after cross-multiplying the norms. REFUTED
    gives integer quaternions where the two exact rational matrices differ, and
    ``causes`` names which convention change would reconcile them.

        >>> compare_rotation_expressions("q1*q2", "R1@R2").status
        'verified'
        >>> r = compare_rotation_expressions("q1*q2", "R2@R1")
        >>> r.status, r.causes[0]
        ('refuted', 'reversed_order')
    """
    qa, ma = quat, matrix
    qb, mb = quat_b or quat, matrix_b or matrix
    ta, kinds_a = _parse_rot_expr(a, names)
    tb, kinds_b = _parse_rot_expr(b, names)
    ids = sorted({v[1] for v in kinds_a.values()} | {v[1] for v in kinds_b.values()})
    if not ids:
        raise DomainError("compare_rotation_expressions: no rotation variables in either expression")
    side_a = _Side(ta, qa.order, qa.algebra, ma.transposed)
    side_b = _Side(tb, qb.order, qb.algebra, mb.transposed)
    conv = {"a": {"quat": qa.to_dict(), "matrix": ma.to_dict()}, "b": {"quat": qb.to_dict(), "matrix": mb.to_dict()}}
    if len(ids) > len(_WITNESS_QUATS):
        raise DomainError(f"compare_rotation_expressions: at most {len(_WITNESS_QUATS)} distinct rotations")

    def witness_sets() -> Iterable[dict[str, tuple[Fraction, ...]]]:
        for shift in range(len(_WITNESS_QUATS)):
            yield {ident: tuple(Fraction(v) for v in _WITNESS_QUATS[(i + shift) % len(_WITNESS_QUATS)])
                   for i, ident in enumerate(ids)}

    sets = list(witness_sets())
    first_diff = None
    for ws in sets:
        ra, rb = _eval_side(side_a, kinds_a, ws), _eval_side(side_b, kinds_b, ws)
        if not _same(ra, rb):
            first_diff = (ws, ra, rb)
            break
    if first_diff is None:
        proved = guard("geometry.compare_rotation_expressions", f"{a} vs {b}",
                       lambda: _prove_same(side_a, side_b, kinds_a, kinds_b, ids))
        if proved:
            return ConventionReport(
                kind="compare_rotation_expressions", status="verified", method="symbolic",
                a=a, b=b, conventions=conv, witness={}, matrix_a=None, matrix_b=None,
                max_abs_difference=None, causes=(), explanations=(),
                detail="equal as rational functions of the quaternion components (polynomial identity)")
        return ConventionReport(  # pragma: no cover - agreement at all witnesses but no proof
            kind="compare_rotation_expressions", status="undecided", method="sampling",
            a=a, b=b, conventions=conv, witness={}, matrix_a=None, matrix_b=None,
            max_abs_difference=None, causes=(), explanations=(),
            detail=f"equal at {len(sets)} exact integer witnesses, but the symbolic identity did not close")
    ws, ra, rb = first_diff
    Ma = [[v / ra[1] for v in r] for r in ra[0]]
    Mb = [[v / rb[1] for v in r] for r in rb[0]]
    diff = max(abs(float(Ma[i][j] - Mb[i][j])) for i in range(3) for j in range(3))
    causes = _diagnose(side_a, side_b, kinds_a, kinds_b, sets)
    witness = {ident: _describe([int(v) for v in ws[ident]], qa.order, qa.algebra) for ident in ids}
    return ConventionReport(
        kind="compare_rotation_expressions", status="refuted", method="exact-witness",
        a=a, b=b, conventions=conv, witness=witness,
        matrix_a=tuple(tuple(_fstr(v) for v in r) for r in Ma),
        matrix_b=tuple(tuple(_fstr(v) for v in r) for r in Mb),
        max_abs_difference=diff, causes=causes, explanations=tuple(_CAUSE_TEXT[c] for c in causes),
        detail=(f"different rotations at the witness (max entry difference {diff:.6g}, exact rationals)"
                + (f"; likely cause: {_CAUSE_TEXT[causes[0]]}" if causes else "; no single convention change explains it")),
    )


def _diagnose(sa: _Side, sb: _Side, ka: dict[str, tuple[str, str]], kb: dict[str, tuple[str, str]],
              sets: list[dict[str, tuple[Fraction, ...]]]) -> tuple[str, ...]:
    has_m_b = any(v[0] == "matrix" for v in kb.values())
    has_m_a = any(v[0] == "matrix" for v in ka.values())
    hyps: list[tuple[str, Callable[[dict[str, tuple[Fraction, ...]]], bool]]] = []

    def flip(side: _Side, **kw: Any) -> _Side:
        return replace(side, **kw)

    hyps.append(("reversed_order", lambda ws: _same(_eval_side(sa, ka, ws), _eval_side(sb, kb, ws, reverse=True))))
    hyps.append(("transposed", lambda ws: _same(_eval_side(sa, ka, ws),
                                                _t(_eval_side(sb, kb, ws)))))
    if has_m_a or has_m_b:
        hyps.append(("row_vs_column_vectors", lambda ws: (
            _same(_eval_side(sa, ka, ws), _eval_side(flip(sb, transposed=not sb.transposed), kb, ws)) if has_m_b
            else _same(_eval_side(flip(sa, transposed=not sa.transposed), ka, ws), _eval_side(sb, kb, ws)))))
    other = {"hamilton": "jpl", "jpl": "hamilton"}
    hyps.append(("jpl_vs_hamilton", lambda ws: (
        _same(_eval_side(sa, ka, ws), _eval_side(flip(sb, algebra=other[sb.algebra]), kb, ws))
        or _same(_eval_side(flip(sa, algebra=other[sa.algebra]), ka, ws), _eval_side(sb, kb, ws)))))
    oth = {"wxyz": "xyzw", "xyzw": "wxyz"}
    hyps.append(("component_order", lambda ws: (
        _same(_eval_side(sa, ka, ws), _eval_side(flip(sb, order=oth[sb.order]), kb, ws))
        or _same(_eval_side(flip(sa, order=oth[sa.order]), ka, ws), _eval_side(sb, kb, ws)))))
    def mirrored(S: list[list[int]]) -> Callable[[dict[str, tuple[Fraction, ...]]], bool]:
        return lambda ws: _same(_eval_side(sa, ka, ws), _conj_by(S, _eval_side(sb, kb, ws)))

    for k, name in enumerate(("x", "y", "z")):
        hyps.append((f"handedness_mirror_{name}", mirrored(_mirror(k))))
    return tuple(name for name, test in hyps if all(test(ws) for ws in sets))


def _t(r: tuple[list[list[Any]], Any]) -> tuple[list[list[Any]], Any]:
    return _transpose(r[0]), r[1]


def _conj_by(S: list[list[int]], r: tuple[list[list[Any]], Any]) -> tuple[list[list[Any]], Any]:
    return _matmul(_matmul(S, r[0]), S), r[1]


def _prove_same(sa: _Side, sb: _Side, ka: dict[str, tuple[str, str]], kb: dict[str, tuple[str, str]],
                ids: list[str]) -> bool:
    from sympy import ZZ
    from sympy.polys.rings import ring

    gens = " ".join(f"{c}_{i}" for i in range(len(ids)) for c in "abcd")
    Rg, *syms = ring(gens, ZZ)
    quats = {ident: tuple(syms[4 * i: 4 * i + 4]) for i, ident in enumerate(ids)}
    return _same(_eval_side(sa, ka, quats), _eval_side(sb, kb, quats))


@dataclass(frozen=True)
class EulerMatch(_Result):
    """One Euler convention consistent with given angles and matrix."""

    seq: str
    frame: str
    transposed: bool
    residual: float
    scipy_seq: str


def identify_euler_convention(angles: Any, matrix: Any, *, degrees: bool = False,
                              tol: float = 1e-9) -> tuple[EulerMatch, ...]:
    """Which Euler conventions turn ``angles`` into ``matrix``?

    Tries all 12 sequences x intrinsic/extrinsic x (column-vector, transposed),
    i.e. 48 readings, and returns every one whose matrix (elementary-product
    route, 50 digits) is within ``tol`` of the given matrix — usually exactly one,
    or a few for special angles. An empty tuple means no standard convention fits.
    ``scipy_seq`` is the matching SciPy sequence string (upper case = intrinsic).
    """
    ang = _vector(angles, 3, "identify_euler_convention: angles")
    rad = tuple(math.radians(float(v)) if degrees else float(v) for v in ang)
    rows = [list(r) for r in matrix]
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise DomainError("identify_euler_convention: matrix must be 3x3")
    M = [[_scalar(v, "identify_euler_convention: matrix") for v in r] for r in rows]
    with mpmath.workdps(CERT_DPS):
        Mm = [[_mp(v) for v in r] for r in M]
        MmT = _transpose(Mm)
        out = []
        for seq in EULER_SEQUENCES:
            for frame in ("intrinsic", "extrinsic"):
                E = _mp_matrix_of("euler", (seq, frame, rad))
                for tr, target in ((False, Mm), (True, MmT)):
                    res = _mp_residual(E, target)
                    if res <= tol:
                        out.append(EulerMatch(seq=seq, frame=frame, transposed=tr, residual=res,
                                              scipy_seq=seq.upper() if frame == "intrinsic" else seq))
    return tuple(sorted(out, key=lambda m: m.residual))


# ==================================================================== predicates
# Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
# Predicates", Discrete Comput. Geom. 18 (1997) 305-363: stage-A error bounds, with
# epsilon = 2^-53 (half an ulp of 1).
_CCW_A = (3.0 + 16.0 * _EPS) * _EPS
_O3D_A = (7.0 + 56.0 * _EPS) * _EPS
_ICC_A = (10.0 + 96.0 * _EPS) * _EPS
# Shewchuk's bounds assume no underflow or overflow. If every non-zero coordinate
# DIFFERENCE lies in [2^-240, 2^240], no product of up to four of them leaves the
# normal range (2^-960 > 2^-1022, 2^960 < 2^1024), so the bound is valid; otherwise
# the filter is skipped and the exact rational path decides. Differences of floats
# are never rounded to zero (gradual underflow), so "zero" is exact.
_SAFE_LO, _SAFE_HI = 2.0**-240, 2.0**240

_STATS = {"filter": 0, "exact": 0}


def _safe(*vals: float) -> bool:
    for v in vals:
        a = abs(v)
        if a != 0.0 and not (_SAFE_LO <= a <= _SAFE_HI):
            return False
    return True


def _pts(points: Sequence[Any], dim: int, what: str) -> list[tuple[Num, ...]]:
    return [_vector(p, dim, f"{what}: point {i}") for i, p in enumerate(points)]


def orient2d_exact(a: Any, b: Any, c: Any) -> int:
    """Exact sign of det[[ax-cx, ay-cy], [bx-cx, by-cy]] in rationals.

    +1: a, b, c counter-clockwise; -1: clockwise; 0: collinear (Shewchuk's sign).
    """
    A, B, C = (tuple(_F(v) for v in p) for p in _pts((a, b, c), 2, "orient2d"))
    return _sign((A[0] - C[0]) * (B[1] - C[1]) - (A[1] - C[1]) * (B[0] - C[0]))


def orient3d_exact(a: Any, b: Any, c: Any, d: Any) -> int:
    """Exact sign of det[a-d; b-d; c-d]: +1 when d is BELOW the plane of a, b, c
    (a, b, c counter-clockwise seen from above) — Shewchuk's convention."""
    P = [tuple(_F(v) for v in p) for p in _pts((a, b, c, d), 3, "orient3d")]
    D = P[3]
    m = [[P[r][k] - D[k] for k in range(3)] for r in range(3)]
    return _sign(_det3(m))


def incircle_exact(a: Any, b: Any, c: Any, d: Any) -> int:
    """Exact sign: +1 when d is INSIDE the circle through a, b, c (counter-clockwise)."""
    P = [tuple(_F(v) for v in p) for p in _pts((a, b, c, d), 2, "incircle")]
    D = P[3]
    rows = []
    for r in range(3):
        dx, dy = P[r][0] - D[0], P[r][1] - D[1]
        rows.append([dx, dy, dx * dx + dy * dy])
    return _sign(_det3(rows))


@dataclass(frozen=True)
class PredicateResult(_Result):
    """A predicate sign with how it was decided.

    ``stage``: ``"filter"`` (float determinant beyond Shewchuk's error bound — sign
    proved by the bound), ``"exact"`` (rational arithmetic), or ``"exact-input"``
    (the input was already exact). ``approx``: the float determinant (or None),
    ``error_bound``: the bound it was compared with.
    """

    kind: str
    sign: int
    stage: str
    approx: "float | None"
    error_bound: "float | None"


def _orient2d_filter(A: Sequence[float], B: Sequence[float], C: Sequence[float]) -> "tuple[int, float, float] | None":
    acx, bcx, acy, bcy = A[0] - C[0], B[0] - C[0], A[1] - C[1], B[1] - C[1]
    if not _safe(acx, bcx, acy, bcy):
        return None
    detleft = acx * bcy
    detright = acy * bcx
    det = detleft - detright
    if detleft > 0.0:
        if detright <= 0.0:
            return _sign(det), det, 0.0
        detsum = detleft + detright
    elif detleft < 0.0:
        if detright >= 0.0:
            return _sign(det), det, 0.0
        detsum = -detleft - detright
    else:
        return _sign(det), det, 0.0
    err = _CCW_A * detsum
    if det >= err or -det >= err:
        return _sign(det), det, err
    return None


def classify_orient2d(a: Any, b: Any, c: Any) -> PredicateResult:
    """orient2d with the evidence (see :func:`orient2d`)."""
    P = _pts((a, b, c), 2, "orient2d")
    flat = [v for p in P for v in p]
    if _is_exact(flat):
        return PredicateResult("orient2d", orient2d_exact(*P), "exact-input", None, None)
    Pf = [tuple(float(v) for v in p) for p in P]
    r = _orient2d_filter(*Pf)
    if r is not None:
        _STATS["filter"] += 1
        return PredicateResult("orient2d", r[0], "filter", r[1], r[2])
    _STATS["exact"] += 1
    return PredicateResult("orient2d", orient2d_exact(*P), "exact", None, None)


def orient2d(a: Any, b: Any, c: Any) -> int:
    """Robust orientation of three 2-D points: +1 ccw, -1 cw, 0 collinear.

    Float inputs go through Shewchuk's stage-A filter (one float determinant and
    an a-priori error bound); only when the bound cannot certify the sign does it
    fall back to exact rational arithmetic on the exact binary values. The sign is
    therefore always the true sign for the given floats — on the Kettner et al.
    (2008) 256x256 grid, where the naive determinant is wrong 11 492 times, it is
    never wrong.

        >>> orient2d((0, 0), (1, 0), (0, 1))
        1
        >>> orient2d((0.5, 0.5), (12.0, 12.0), (24.0, 24.0))
        0
    """
    # Fast path: three pairs of finite Python floats (x - x == 0 rejects inf/nan).
    try:
        ax, ay = a
        bx, by = b
        cx, cy = c
    except (TypeError, ValueError):
        return classify_orient2d(a, b, c).sign
    if (type(ax) is float and type(ay) is float and type(bx) is float and type(by) is float
            and type(cx) is float and type(cy) is float
            and ax - ax == 0.0 and ay - ay == 0.0 and bx - bx == 0.0 and by - by == 0.0
            and cx - cx == 0.0 and cy - cy == 0.0):
        r = _orient2d_filter((ax, ay), (bx, by), (cx, cy))
        if r is not None:
            _STATS["filter"] += 1
            return r[0]
    return classify_orient2d(a, b, c).sign


def classify_orient3d(a: Any, b: Any, c: Any, d: Any) -> PredicateResult:
    P = _pts((a, b, c, d), 3, "orient3d")
    if _is_exact(v for p in P for v in p):
        return PredicateResult("orient3d", orient3d_exact(*P), "exact-input", None, None)
    A, B, C, D = [tuple(float(v) for v in p) for p in P]
    adx, bdx, cdx = A[0] - D[0], B[0] - D[0], C[0] - D[0]
    ady, bdy, cdy = A[1] - D[1], B[1] - D[1], C[1] - D[1]
    adz, bdz, cdz = A[2] - D[2], B[2] - D[2], C[2] - D[2]
    if _safe(adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz):
        bdxcdy, cdxbdy = bdx * cdy, cdx * bdy
        cdxady, adxcdy = cdx * ady, adx * cdy
        adxbdy, bdxady = adx * bdy, bdx * ady
        det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady)
        perm = ((abs(bdxcdy) + abs(cdxbdy)) * abs(adz) + (abs(cdxady) + abs(adxcdy)) * abs(bdz)
                + (abs(adxbdy) + abs(bdxady)) * abs(cdz))
        err = _O3D_A * perm
        if det > err or -det > err:
            _STATS["filter"] += 1
            return PredicateResult("orient3d", _sign(det), "filter", det, err)
    _STATS["exact"] += 1
    return PredicateResult("orient3d", orient3d_exact(*P), "exact", None, None)


def orient3d(a: Any, b: Any, c: Any, d: Any) -> int:
    """Robust 3-D orientation: +1 if d is below the plane through a, b, c (Shewchuk)."""
    return classify_orient3d(a, b, c, d).sign


def classify_incircle(a: Any, b: Any, c: Any, d: Any) -> PredicateResult:
    P = _pts((a, b, c, d), 2, "incircle")
    if _is_exact(v for p in P for v in p):
        return PredicateResult("incircle", incircle_exact(*P), "exact-input", None, None)
    A, B, C, D = [tuple(float(v) for v in p) for p in P]
    adx, bdx, cdx = A[0] - D[0], B[0] - D[0], C[0] - D[0]
    ady, bdy, cdy = A[1] - D[1], B[1] - D[1], C[1] - D[1]
    if _safe(adx, bdx, cdx, ady, bdy, cdy):
        bdxcdy, cdxbdy = bdx * cdy, cdx * bdy
        alift = adx * adx + ady * ady
        cdxady, adxcdy = cdx * ady, adx * cdy
        blift = bdx * bdx + bdy * bdy
        adxbdy, bdxady = adx * bdy, bdx * ady
        clift = cdx * cdx + cdy * cdy
        det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady)
        perm = ((abs(bdxcdy) + abs(cdxbdy)) * alift + (abs(cdxady) + abs(adxcdy)) * blift
                + (abs(adxbdy) + abs(bdxady)) * clift)
        err = _ICC_A * perm
        if det > err or -det > err:
            _STATS["filter"] += 1
            return PredicateResult("incircle", _sign(det), "filter", det, err)
    _STATS["exact"] += 1
    return PredicateResult("incircle", incircle_exact(*P), "exact", None, None)


def incircle(a: Any, b: Any, c: Any, d: Any) -> int:
    """Robust in-circle test: +1 if d is inside the circle through ccw a, b, c."""
    return classify_incircle(a, b, c, d).sign


# ================================================================ intersections
def _fpt(p: Sequence[Fraction]) -> tuple[str, ...]:
    return tuple(_fstr(v) for v in p)


@dataclass(frozen=True)
class SegmentIntersection(_Result):
    """Exact segment/segment intersection.

    ``kind``: ``"proper"`` (interiors cross at one point), ``"touching"`` (one
    point, an endpoint involved), ``"collinear_overlap"`` (``segment`` is the
    shared piece, possibly a single point), ``"disjoint"``.
    ``point``/``segment`` are exact rational strings; ``approx`` the nearest floats.
    """

    kind: str
    point: "tuple[str, ...] | None"
    segment: "tuple[tuple[str, ...], tuple[str, ...]] | None"
    approx: Any
    certificate: dict[str, Any]


def _on_segment(p: Sequence[Fraction], a: Sequence[Fraction], b: Sequence[Fraction]) -> bool:
    if _sign((a[0] - p[0]) * (b[1] - p[1]) - (a[1] - p[1]) * (b[0] - p[0])) != 0:
        return False
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segment_intersection(p1: Any, p2: Any, q1: Any, q2: Any) -> SegmentIntersection:
    """Intersect segments p1p2 and q1q2 exactly (floats taken at their binary value).

    Built on exact orientation signs, so touching, collinear and one-ulp-miss
    configurations are classified correctly.

        >>> segment_intersection((0, 0), (2, 2), (0, 2), (2, 0)).point
        ('1', '1')
    """
    P1, P2, Q1, Q2 = (tuple(_F(v) for v in p) for p in _pts((p1, p2, q1, q2), 2, "segment_intersection"))

    def o(a: Sequence[Fraction], b: Sequence[Fraction], c: Sequence[Fraction]) -> int:
        return _sign((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))

    cert = {"status": "verified", "method": "exact rational orientation signs"}

    def result(kind: str, point: "Sequence[Fraction] | None" = None,
               seg: "tuple[Sequence[Fraction], Sequence[Fraction]] | None" = None) -> SegmentIntersection:
        approx: Any = None
        if point is not None:
            approx = tuple(float(v) for v in point)
            # independent check: the point lies on both segments
            ok = _on_segment(point, P1, P2) and _on_segment(point, Q1, Q2)
            if not ok:  # pragma: no cover
                raise CertificationError("segment_intersection: point not on both segments", residual=1.0, tolerance=0.0)
        if seg is not None:
            approx = tuple(tuple(float(v) for v in s) for s in seg)
        return SegmentIntersection(kind=kind, point=_fpt(point) if point is not None else None,
                                   segment=(_fpt(seg[0]), _fpt(seg[1])) if seg is not None else None,
                                   approx=approx, certificate=cert)

    pdeg, qdeg = P1 == P2, Q1 == Q2
    if pdeg or qdeg:
        if pdeg and qdeg:
            return result("touching", P1) if P1 == Q1 else result("disjoint")
        pt, a, b = (P1, Q1, Q2) if pdeg else (Q1, P1, P2)
        return result("touching", pt) if _on_segment(pt, a, b) else result("disjoint")
    o1, o2, o3, o4 = o(P1, P2, Q1), o(P1, P2, Q2), o(Q1, Q2, P1), o(Q1, Q2, P2)
    if o1 == o2 == 0:
        axis = 0 if P1[0] != P2[0] else 1
        lo_p, hi_p = sorted((P1, P2), key=lambda p: p[axis])
        lo_q, hi_q = sorted((Q1, Q2), key=lambda p: p[axis])
        lo = max(lo_p, lo_q, key=lambda p: p[axis])
        hi = min(hi_p, hi_q, key=lambda p: p[axis])
        if lo[axis] > hi[axis]:
            return result("disjoint")
        return result("collinear_overlap", seg=(lo, hi))
    if o1 * o2 <= 0 and o3 * o4 <= 0:
        rx, ry = P2[0] - P1[0], P2[1] - P1[1]
        sx, sy = Q2[0] - Q1[0], Q2[1] - Q1[1]
        den = rx * sy - ry * sx
        t = ((Q1[0] - P1[0]) * sy - (Q1[1] - P1[1]) * sx) / den
        pt = (P1[0] + t * rx, P1[1] + t * ry)
        return result("touching" if 0 in (o1, o2, o3, o4) else "proper", pt)
    return result("disjoint")


@dataclass(frozen=True)
class RayHit(_Result):
    """Exact ray/triangle result (Möller–Trumbore's equations solved in rationals).

    ``hit``; ``t`` (ray parameter), ``u``, ``v`` (barycentric of v1, v2) as exact
    strings; ``point``; ``on_edge``/``on_vertex`` when the hit is on the boundary;
    ``reason`` for a miss (``"parallel"``, ``"outside"``, ``"behind"``,
    ``"backface"``, ``"degenerate_triangle"``).
    """

    hit: bool
    t: "str | None"
    u: "str | None"
    v: "str | None"
    point: "tuple[str, ...] | None"
    approx_t: "float | None"
    on_edge: bool
    on_vertex: bool
    reason: str
    certificate: dict[str, Any]


def _sub3(a: Sequence[Any], b: Sequence[Any]) -> tuple[Any, Any, Any]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Sequence[Any], b: Sequence[Any]) -> tuple[Any, Any, Any]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Sequence[Any], b: Sequence[Any]) -> Any:
    return sum((x * y for x, y in zip(a, b)), start=a[0] * 0)


def _mt_exact(O: Sequence[Fraction], D: Sequence[Fraction], V0: Sequence[Fraction], V1: Sequence[Fraction],
              V2: Sequence[Fraction]) -> "tuple[Fraction, Fraction, Fraction, Fraction] | None":
    e1, e2 = _sub3(V1, V0), _sub3(V2, V0)
    pvec = _cross(D, e2)
    det = _dot(e1, pvec)
    if det == 0:
        return None
    tvec = _sub3(O, V0)
    u = _dot(tvec, pvec) / det
    qvec = _cross(tvec, e1)
    v = _dot(D, qvec) / det
    t = _dot(e2, qvec) / det
    return det, t, u, v


def ray_triangle(origin: Any, direction: Any, v0: Any, v1: Any, v2: Any, *,
                 cull_backface: bool = False, t_min: Any = 0) -> RayHit:
    """Exact ray/triangle intersection; hits on edges and vertices count.

    A hit needs ``u >= 0, v >= 0, u + v <= 1`` and ``t >= t_min`` — all decided in
    exact rationals, so there is no EPSILON and no leak along shared edges. The
    certificate re-derives the hit point from ``t`` on the ray and from ``(u, v)``
    on the triangle and checks they coincide exactly.

        >>> ray_triangle((0.25, 0.25, 1), (0, 0, -1), (0, 0, 0), (1, 0, 0), (0, 1, 0)).t
        '1'
    """
    O, D, V0, V1, V2 = (tuple(_F(v) for v in p) for p in _pts((origin, direction, v0, v1, v2), 3, "ray_triangle"))
    tmin = _F(_scalar(t_min, "ray_triangle: t_min"))
    if D == (0, 0, 0):
        raise DomainError("ray_triangle: the ray direction is the zero vector")
    cert = {"status": "verified", "method": "exact rationals; point(t) on ray == point(u,v) on triangle"}

    def miss(reason: str) -> RayHit:
        return RayHit(False, None, None, None, None, None, False, False, reason, cert)

    if _cross(_sub3(V1, V0), _sub3(V2, V0)) == (0, 0, 0):
        return miss("degenerate_triangle")
    r = _mt_exact(O, D, V0, V1, V2)
    if r is None:
        return miss("parallel")
    det, t, u, v = r
    if cull_backface and det < 0:
        return miss("backface")
    if u < 0 or v < 0 or u + v > 1:
        return miss("outside")
    if t < tmin:
        return miss("behind")
    p_ray = tuple(O[k] + t * D[k] for k in range(3))
    p_tri = tuple(V0[k] + u * (V1[k] - V0[k]) + v * (V2[k] - V0[k]) for k in range(3))
    if p_ray != p_tri:  # pragma: no cover
        raise CertificationError("ray_triangle: ray point != triangle point", residual=1.0, tolerance=0.0)
    zeros = (u == 0) + (v == 0) + (u + v == 1)
    return RayHit(True, _fstr(t), _fstr(u), _fstr(v), _fpt(p_ray), float(t), zeros >= 1, zeros >= 2, "hit", cert)


def moller_trumbore_float(origin: Sequence[float], direction: Sequence[float], v0: Sequence[float],
                          v1: Sequence[float], v2: Sequence[float], epsilon: float = 1e-8) -> "float | None":
    """The textbook float Möller–Trumbore (1997) with its EPSILON — as commonly written.

    Kept as the REFERENCE of what typical code does (used by :func:`epsilon_witness`
    and the code emitter); returns ``t`` or ``None``. Not robust: see the witness.
    """
    e1 = _sub3(v1, v0)
    e2 = _sub3(v2, v0)
    pvec = _cross(direction, e2)
    det = _dot(e1, pvec)
    if -epsilon < det < epsilon:
        return None
    inv = 1.0 / det
    tvec = _sub3(origin, v0)
    u = _dot(tvec, pvec) * inv
    if u < 0.0 or u > 1.0:
        return None
    qvec = _cross(tvec, e1)
    v = _dot(direction, qvec) * inv
    if v < 0.0 or u + v > 1.0:
        return None
    t = _dot(e2, qvec) * inv
    return t if t > epsilon else None


@dataclass(frozen=True)
class RaySphere(_Result):
    """Exact ray/sphere classification: ``"miss"``, ``"tangent"``, ``"secant"``,
    ``"inside"`` (origin strictly inside: one root behind, one ahead),
    ``"on_surface"`` (the origin lies on the sphere: ``t = 0`` is a root; the
    other root is listed too when it is ahead).
    ``t`` holds exact roots (SymPy text, may contain ``sqrt``) of the ray
    parameter, ``approx_t`` the floats; ``discriminant`` the exact quarter
    discriminant ``(oc.d)^2 - |d|^2 (|oc|^2 - r^2)``."""

    kind: str
    t: tuple[str, ...]
    approx_t: tuple[float, ...]
    discriminant: str
    certificate: dict[str, Any]


def ray_sphere(origin: Any, direction: Any, center: Any, radius: Any) -> RaySphere:
    """Exact ray/sphere intersection for the ray ``origin + t*direction, t >= 0``.

    The direction need not be normalised. Classification uses the exact sign of
    the discriminant, so a ray missing by one ulp is ``"miss"``, not ``"tangent"``.

        >>> ray_sphere((0, 0, -5), (0, 0, 1), (0, 0, 0), 1).t
        ('4', '6')
    """
    import sympy as sp

    O, D, C = (tuple(_F(v) for v in p) for p in _pts((origin, direction, center), 3, "ray_sphere"))
    r = _F(_scalar(radius, "ray_sphere: radius"))
    if r <= 0:
        raise DomainError(f"ray_sphere: radius must be positive, got {_fstr(r)}")
    if D == (0, 0, 0):
        raise DomainError("ray_sphere: the ray direction is the zero vector")
    oc = _sub3(O, C)
    a = _dot(D, D)
    hb = _dot(oc, D)
    c = _dot(oc, oc) - r * r
    disc = hb * hb - a * c
    cert = {"status": "verified", "method": "exact discriminant sign; roots substituted back exactly"}

    def roots(signs: Sequence[int]) -> list[Any]:
        return [(sp.Rational(-hb.numerator, hb.denominator) + s * sp.sqrt(sp.Rational(disc.numerator, disc.denominator)))
                / sp.Rational(a.numerator, a.denominator) for s in signs]

    def done(kind: str, ts: list[Any]) -> RaySphere:
        for tv in ts:  # independent check: |O + tD - C|^2 = r^2 exactly
            pt = [sp.Rational(oc[k].numerator, oc[k].denominator) + tv * sp.Rational(D[k].numerator, D[k].denominator)
                  for k in range(3)]
            if sp.expand(sum(v * v for v in pt) - sp.Rational(r.numerator, r.denominator) ** 2) != 0:  # pragma: no cover
                raise CertificationError("ray_sphere: root not on the sphere", residual=1.0, tolerance=0.0)
        return RaySphere(kind, tuple(str(sp.nsimplify(tv)) for tv in ts),
                         tuple(float(sp.N(tv, 20)) for tv in ts), _fstr(disc), cert)

    if disc < 0:
        return done("miss", [])
    if disc == 0:
        t0 = -hb / a
        return done("tangent", roots([0])) if t0 >= 0 else done("miss", [])
    t_lo, t_hi = roots([-1, 1])
    if c < 0:
        return done("inside", [t_lo, t_hi])
    if c == 0:  # the origin is ON the sphere: t = 0 is a root
        return done("on_surface", [t_lo, t_hi] if hb < 0 else [t_lo, t_hi][1:])
    # c >= 0: origin outside or on the sphere; both roots have the sign of -hb
    if hb > 0:
        return done("miss", [])
    return done("secant", [t_lo, t_hi])


@dataclass(frozen=True)
class PolygonLocation(_Result):
    """Point-in-polygon result: ``location`` ``"inside"``/``"outside"``/``"boundary"``
    (exact), ``winding`` number, and ``half_open`` — the half-open ownership rule:
    a point on a shared edge belongs to exactly one of the polygons that share it
    (left/bottom edges of an axis-aligned square are owned, right/top are not),
    so a tiling counts every point exactly once."""

    location: str
    winding: int
    half_open: bool
    edge: "int | None"
    certificate: dict[str, Any]


def point_in_polygon(point: Any, polygon: Any, *, fill_rule: str = "nonzero") -> PolygonLocation:
    """Locate a point against a polygon (any orientation, simple or not), exactly.

    ``location`` reports the boundary explicitly; ``half_open`` is the boolean a
    renderer or a spatial partition needs (crossing test with half-open edge
    intervals in y and a strict side test). ``fill_rule`` is ``"nonzero"`` or
    ``"evenodd"`` for self-intersecting polygons.

        >>> point_in_polygon((1, 0.5), [(0, 0), (1, 0), (1, 1), (0, 1)]).location
        'boundary'
    """
    if fill_rule not in ("nonzero", "evenodd"):
        raise DomainError(f"point_in_polygon: fill_rule is 'nonzero' or 'evenodd', got {fill_rule!r}")
    p = tuple(_F(v) for v in _vector(point, 2, "point_in_polygon: point"))
    try:
        raw = list(polygon)
    except TypeError as exc:
        raise DomainError("point_in_polygon: polygon must be a sequence of 2-D points") from exc
    if len(raw) < 3:
        raise DomainError(f"point_in_polygon: a polygon needs at least 3 vertices, got {len(raw)}")
    V = [tuple(_F(v) for v in q) for q in _pts(raw, 2, "point_in_polygon")]
    boundary_edge = None
    wn = 0
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        if boundary_edge is None and (_on_segment(p, a, b) if a != b else p == a):
            boundary_edge = i
        s = _sign((a[0] - p[0]) * (b[1] - p[1]) - (a[1] - p[1]) * (b[0] - p[0]))
        if a[1] <= p[1] < b[1] and s > 0:
            wn += 1
        elif b[1] <= p[1] < a[1] and s < 0:
            wn -= 1
    inside = (wn != 0) if fill_rule == "nonzero" else (wn % 2 != 0)
    loc = "boundary" if boundary_edge is not None else ("inside" if inside else "outside")
    return PolygonLocation(loc, wn, inside, boundary_edge,
                           {"status": "verified", "method": "exact winding number (Sunday 2001) on rationals"})


# ============================================================ epsilon detector
@dataclass(frozen=True)
class EpsilonReport(_Result):
    """Concrete inputs where an EPSILON-based test answers wrongly.

    ``status`` is ``"refuted"`` when at least one witness was found (every witness
    re-executed in floats AND decided exactly), ``"undecided"`` when the bounded
    search found none — never "verified": an epsilon test is not proved correct by
    failing to break it. Each witness: ``kind`` (``"false_degenerate"``,
    ``"wrong_sign"``, ``"missed_hit"``, ``"edge_leak"``), ``input``,
    ``epsilon_test`` (what the float test says), ``exact`` (the truth), ``note``.
    """

    kind: str
    test: str
    epsilon: float
    status: str
    witnesses: tuple[dict[str, Any], ...]
    detail: str


def _naive_orient(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """The orientation determinant as typically written: cross(b - a, c - a)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


#: Found by :func:`_kettner_search` over the whole 256x256 grid (x 3 argument
#: rotations): p = (0.5 + 41 ulp, 0.5 + 48 ulp) with (p, q, r) gives the wrong
#: NON-ZERO sign of largest magnitude. Re-verified (float + exact) on every use.
_KETTNER_IJ = (41, 48)


def _kettner_search() -> "tuple[tuple[tuple[float, float], ...], float] | None":
    """Kettner et al. (2008): p near (0.5, 0.5) moved by 0..255 ulps per axis,
    q = (12, 12), r = (24, 24); the point where the typical float determinant has
    the wrong NON-ZERO sign of largest magnitude (about 3 s; used in the tests)."""
    q, r = (12.0, 12.0), (24.0, 24.0)
    best = None
    for i in range(256):
        for j in range(256):
            p = (0.5 + i * _EPS, 0.5 + j * _EPS)
            for pts in ((p, q, r), (q, r, p), (r, p, q)):
                d = _naive_orient(*pts)
                if d != 0 and (best is None or abs(d) > abs(best[1])) and _sign(d) != orient2d_exact(*pts):
                    best = (pts, d)
    return best


def _kettner_wrong_sign() -> "tuple[tuple[tuple[float, float], ...], float] | None":
    i, j = _KETTNER_IJ
    pts = ((0.5 + i * _EPS, 0.5 + j * _EPS), (12.0, 12.0), (24.0, 24.0))
    d = _naive_orient(*pts)
    if d != 0 and _sign(d) != orient2d_exact(*pts):
        return pts, d
    return _kettner_search()  # pragma: no cover - only if float semantics changed


def epsilon_witness(test: str, epsilon: float) -> EpsilonReport:
    """Find data on which a typical EPSILON test gives the wrong answer.

    ``test``:

    * ``"orient2d"`` — ``d = det; if abs(d) < eps: collinear else sign(d)``.
      ``d = cross(b - a, c - a)`` as typically written.
      Witnesses: non-collinear points with ``|d| < eps`` (``false_degenerate``),
      and points where the float ``d`` has the WRONG SIGN although ``|d| >= eps``
      (``wrong_sign``; the Kettner et al. grid scaled by a power of two, which
      scales every rounding exactly).
    * ``"ray_triangle"`` — textbook float Möller–Trumbore with ``abs(det) < eps``
      -> miss. Witnesses: a small triangle that is hit (``missed_hit``), and a ray
      through the shared edge of two triangles that misses BOTH in floats while
      it exactly hits (``edge_leak`` — the watertightness problem, Woop et al. 2013).
    """
    eps = float(_scalar(epsilon, "epsilon_witness: epsilon"))
    if eps < 0:
        raise DomainError("epsilon_witness: epsilon must be >= 0")
    if test == "orient2d":
        ws = _orient_eps_witnesses(eps)
    elif test == "ray_triangle":
        ws = _ray_eps_witnesses(eps)
    else:
        raise DomainError(f"epsilon_witness: test is 'orient2d' or 'ray_triangle', got {test!r}")
    status = "refuted" if ws else "undecided"
    detail = (f"{len(ws)} witness(es), each re-executed in floats and decided exactly" if ws
              else "no witness found by the bounded search; this is NOT a proof that the test is correct")
    return EpsilonReport("epsilon_witness", test, eps, status, tuple(ws), detail)


def _orient_eps_witnesses(eps: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if eps > 0:
        h = 2.0 ** (math.frexp(eps)[1] - 2)  # a power of two strictly below eps
        while h >= eps:
            h /= 2
        a, b, c = (0.0, 0.0), (1.0, 0.0), (0.5, h)
        d = _naive_orient(a, b, c)
        ex = orient2d_exact(a, b, c)
        if abs(d) < eps and ex != 0:
            out.append({"kind": "false_degenerate", "input": [list(a), list(b), list(c)],
                        "epsilon_test": "collinear", "exact": "ccw" if ex > 0 else "cw",
                        "note": f"triangle of area {h / 2!r} called collinear: eps is an ABSOLUTE "
                                "threshold on a quantity that scales with the square of the coordinates"})
    best = _kettner_wrong_sign()
    if best is not None:
        (p, q, r), d = best
        k = 0
        while abs(d) * 4.0**k < eps and k < 500:
            k += 1
        sc = 2.0**k
        P, Q, Rr = (p[0] * sc, p[1] * sc), (q[0] * sc, q[1] * sc), (r[0] * sc, r[1] * sc)
        ds = _naive_orient(P, Q, Rr)
        ex = orient2d_exact(P, Q, Rr)
        if abs(ds) >= eps and _sign(ds) != ex and math.isfinite(ds):
            out.append({"kind": "wrong_sign", "input": [list(P), list(Q), list(Rr)],
                        "epsilon_test": {1: "ccw", -1: "cw"}[_sign(ds)], "float_det": ds,
                        "exact": {1: "ccw", -1: "cw", 0: "collinear"}[ex],
                        "note": "the float determinant clears eps with the WRONG sign "
                                "(Kettner et al. 2008 grid, scaled by 2^%d)" % k})
    return out


def _ray_eps_witnesses(eps: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if eps > 0:
        s = 2.0 ** (math.frexp(math.sqrt(eps))[1] - 1)
        while s * s >= eps:
            s /= 2
        O, D = (s / 4, s / 4, 1.0), (0.0, 0.0, -1.0)
        tri = ((0.0, 0.0, 0.0), (s, 0.0, 0.0), (0.0, s, 0.0))
        ft = moller_trumbore_float(O, D, *tri, epsilon=eps)
        ex = ray_triangle(O, D, *tri)
        if ft is None and ex.hit:
            out.append({"kind": "missed_hit", "input": {"origin": list(O), "direction": list(D),
                                                        "triangle": [list(v) for v in tri]},
                        "epsilon_test": "miss", "exact": f"hit at t = {ex.t}",
                        "note": f"det = {s * s!r} < eps: the parallel test is absolute, so a triangle "
                                f"of side {s!r} (e.g. millimetres in metre units) is never hit"})
    # watertightness: shared edge v1-v2 of two triangles, rays through points on it
    A, B = (0.1, 0.2, 0.0), (0.7, 0.9, 0.0)
    T1 = ((0.0, 0.0, 0.0), A, B)
    T2 = (B, A, (0.0, 1.0, 0.0))  # opposite side of AB from T1
    for k in range(1, 4000):
        lam = k / 4001
        tx = A[0] + lam * (B[0] - A[0])
        ty = A[1] + lam * (B[1] - A[1])
        O, D = (tx, ty, 1.0), (0.0, 0.0, -1.0)
        if moller_trumbore_float(O, D, *T1, epsilon=eps) is None and moller_trumbore_float(O, D, *T2, epsilon=eps) is None:
            h1, h2 = ray_triangle(O, D, *T1), ray_triangle(O, D, *T2)
            if h1.hit or h2.hit:
                out.append({"kind": "edge_leak", "input": {"origin": list(O), "direction": list(D),
                                                           "triangle_1": [list(v) for v in T1],
                                                           "triangle_2": [list(v) for v in T2]},
                            "epsilon_test": "miss (both triangles)",
                            "exact": "hit triangle " + ("1" if h1.hit else "2") + f" at t = {(h1 if h1.hit else h2).t}",
                            "note": "the ray passes through the edge the two triangles share; float "
                                    "barycentrics round to just outside BOTH (Woop, Benthin, Wald 2013)"})
                break
    return out


# ======================================================================== curves
@dataclass(frozen=True)
class CurveValue(_Result):
    """A point (or scalar) on a curve, certified.

    ``value``: floats. ``exact``: exact rational strings when every input was
    exact. ``certificate``: for exact input, two independent algorithms agree
    EXACTLY; for float input, ``error`` is the measured distance between the
    float result and the exact value of the same formula at the given floats.
    """

    kind: str
    value: tuple[float, ...]
    exact: "tuple[str, ...] | None"
    certificate: dict[str, Any]


def _ctrl(control: Any, what: str) -> list[tuple[Num, ...]]:
    try:
        pts = list(control)
    except TypeError as exc:
        raise DomainError(f"{what}: control points must be a sequence of points") from exc
    if not pts:
        raise DomainError(f"{what}: at least one control point is needed")
    first = _vector(pts[0], None, f"{what}: control point 0")
    return [first] + [_vector(p, len(first), f"{what}: control point {i + 1}") for i, p in enumerate(pts[1:])]


def _casteljau(P: Sequence[Sequence[Any]], t: Any) -> tuple[Any, ...]:
    pts = [list(p) for p in P]
    for r in range(1, len(pts)):
        pts = [[(1 - t) * a + t * b for a, b in zip(pts[i], pts[i + 1])] for i in range(len(pts) - 1)]
    return tuple(pts[0])


def _bernstein(P: Sequence[Sequence[Any]], t: Any) -> tuple[Any, ...]:
    n = len(P) - 1
    out = [P[0][0] * 0 for _ in P[0]]
    for i, p in enumerate(P):
        b = math.comb(n, i) * t**i * (1 - t) ** (n - i)
        out = [o + b * c for o, c in zip(out, p)]
    return tuple(out)


def _split(P: Sequence[Sequence[Any]], t: Any) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    left, right = [], []
    pts = [tuple(p) for p in P]
    while pts:
        left.append(pts[0])
        right.append(pts[-1])
        pts = [tuple((1 - t) * a + t * b for a, b in zip(pts[i], pts[i + 1])) for i in range(len(pts) - 1)]
    return left, right[::-1]


def _point_result(kind: str, exact_inputs: bool, fast: Callable[[Callable[[Num], Any]], tuple[Any, ...]],
                  slow: Callable[[], tuple[Fraction, ...]]) -> CurveValue:
    """Run ``fast`` in the input's arithmetic and certify against ``slow`` (exact)."""
    ex = slow()
    if exact_inputs:
        other = fast(_F)
        if tuple(other) != tuple(ex):  # pragma: no cover
            raise CertificationError(f"{kind}: the two exact algorithms disagree", residual=1.0, tolerance=0.0)
        return CurveValue(kind, tuple(float(v) for v in ex), tuple(_fstr(v) for v in ex),
                          {"status": "verified", "method": "two exact algorithms agree exactly", "error": 0.0})
    val = tuple(float(v) for v in fast(float))
    err = max((abs(Fraction(v) - e) for v, e in zip(val, ex)), default=Fraction(0))
    scale = max((abs(e) for e in ex), default=Fraction(0))
    return CurveValue(kind, val, None,
                      {"status": "verified", "method": "float result vs exact rational evaluation of the same inputs",
                       "error": float(err), "relative_error": float(err / scale) if scale else float(err)})


def bezier_point(control: Any, t: Any) -> CurveValue:
    """Point on a Bézier curve of any degree and dimension.

    Computed by de Casteljau (stable, the way to evaluate in floats); certified
    against the Bernstein closed form in exact rationals.

        >>> bezier_point([(0, 0), (1, 2), (3, 3), (4, 0)], "1/3").exact
        ('34/27', '14/9')
    """
    P = _ctrl(control, "bezier_point")
    tt = _scalar(t, "bezier_point: t")
    flat = [v for p in P for v in p] + [tt]
    return _point_result("bezier_point", _is_exact(flat),
                         lambda cv: _casteljau([[cv(v) for v in p] for p in P], cv(tt)),
                         lambda: _bernstein([[_F(v) for v in p] for p in P], _F(tt)))


def bezier_derivative(control: Any, order: int = 1) -> tuple[tuple[str, ...], ...]:
    """Control points of the ``order``-th derivative (hodograph): n(P_{i+1} - P_i), repeated.

    Exact rational strings (float inputs taken exactly).

        >>> bezier_derivative([(0, 0), (1, 2), (3, 3), (4, 0)])
        (('3', '6'), ('6', '3'), ('3', '-9'))
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        raise DomainError(f"bezier_derivative: order must be a non-negative integer, got {order!r}")
    P = [[_F(v) for v in p] for p in _ctrl(control, "bezier_derivative")]
    for _ in range(order):
        n = len(P) - 1
        if n == 0:
            P = [[Fraction(0) for _ in P[0]]]
            break
        P = [[n * (b - a) for a, b in zip(P[i], P[i + 1])] for i in range(n)]
    return tuple(tuple(_fstr(v) for v in p) for p in P)


def bezier_polynomial(control: Any, var: str = "t") -> tuple[str, ...]:
    """Power-basis polynomial of each coordinate, as SymPy text (exact).

        >>> bezier_polynomial([(0, 0), (1, 1)])
        ('t', 't')
    """
    import sympy as sp

    if not isinstance(var, str) or not var.isidentifier():
        raise DomainError(f"bezier_polynomial: var must be an identifier, got {var!r}")
    t = sp.Symbol(var)
    P = [[sp.Rational(_F(v).numerator, _F(v).denominator) for v in p] for p in _ctrl(control, "bezier_polynomial")]
    return tuple(str(sp.expand(c)) for c in _bernstein(P, t))


@dataclass(frozen=True)
class ArcLength(_Result):
    """Arc length with a RIGOROUS bracket ``lower <= true length <= upper``.

    The bracket is the sum over adaptive pieces of chord (below) and control
    polygon (above) — valid for every Bézier curve. ``value`` is the 50-digit
    Gauss–Legendre estimate, required to lie inside the bracket.
    """

    kind: str
    value: float
    lower: float
    upper: float
    pieces: int
    certificate: dict[str, Any]


def bezier_arc_length(control: Any, *, rel_tol: float = 1e-7, max_pieces: int = 20000) -> ArcLength:
    """Length of a Bézier curve over t in [0, 1], with a guaranteed enclosure.

    Subdivides at t = 1/2 (exact in binary, so the pieces are the exact
    sub-curves of the given control points), always refining the piece whose
    control polygon exceeds its chord the most, until
    ``upper - lower <= rel_tol * lower``. Piece lengths are summed in interval
    arithmetic at 50 digits, so the bracket is rigorous — not an error ESTIMATE
    (what adaptive quadrature and the ``bezier`` package report). ``value`` is a
    50-digit Gauss–Legendre estimate that must lie inside the bracket.

        >>> r = bezier_arc_length([(0, 0), (3, 4)])
        >>> r.lower <= 5.0 <= r.upper
        True
    """
    P = [[_F(v) for v in p] for p in _ctrl(control, "bezier_arc_length")]
    tol = float(_scalar(rel_tol, "bezier_arc_length: rel_tol"))
    if not 1e-11 <= tol <= 1:  # below ~1e-11 the 2^-40 rounding margin dominates
        raise DomainError("bezier_arc_length: rel_tol must be in [1e-11, 1]")
    if not isinstance(max_pieces, int) or max_pieces < 1:
        raise DomainError("bezier_arc_length: max_pieces must be a positive integer")
    if all(p == P[0] for p in P):
        return ArcLength("bezier_arc_length", 0.0, 0.0, 0.0, 1,
                         {"status": "verified", "method": "all control points coincide: length 0",
                          "relative_width": 0.0})
    return guard("geometry.bezier_arc_length", "Bezier curve", lambda: _arc_length(P, tol, max_pieces))


#: Relative margin applied to every float segment length. A length is computed
#: from EXACT coordinate differences (Fractions), each rounded once to float
#: (0.5 ulp), squared and summed by ``math.hypot`` (<= 1 ulp) — a few ulps in
#: all, far inside 2^-40 ~ 9e-13. So chord*(1-m) and polygon*(1+m) are
#: rigorous bounds; the final sums use ``math.fsum`` (correctly rounded) and get
#: the same margin again.
_LEN_MARGIN = 2.0**-40


def _arc_length(P: list[list[Fraction]], tol: float, max_pieces: int) -> ArcLength:
    import heapq

    def seglen(a: Sequence[Fraction], b: Sequence[Fraction]) -> float:
        return math.hypot(*(float(x - y) for x, y in zip(a, b)))

    counter = itertools.count()

    def piece(Q: list[Any]) -> tuple[float, int, list[Any], float, float]:
        chord = seglen(Q[0], Q[-1]) * (1 - _LEN_MARGIN)
        poly = math.fsum(seglen(Q[i], Q[i + 1]) for i in range(len(Q) - 1)) * (1 + _LEN_MARGIN)
        return (-(poly - chord), next(counter), Q, chord, poly)

    heap = [piece(P)]
    rel = math.inf
    while True:
        lower = math.fsum(item[3] for item in heap) * (1 - _LEN_MARGIN)
        upper = math.fsum(item[4] for item in heap) * (1 + _LEN_MARGIN)
        rel = (upper - lower) / lower if lower > 0 else math.inf
        if rel <= tol or len(heap) >= max_pieces:
            break
        budget = -math.fsum(item[0] for item in heap) - tol * lower * 0.5
        while budget > 0 and len(heap) < max_pieces:
            g, _, Q, _, _ = heapq.heappop(heap)
            left, right = _split(Q, Fraction(1, 2))
            nl, nr = piece(left), piece(right)
            heapq.heappush(heap, nl)
            heapq.heappush(heap, nr)
            budget -= -g - (-nl[0] - nr[0])
    pieces = len(heap)
    if not rel <= tol:
        raise CertificationError(
            f"bezier_arc_length: bracket width {float(rel):.3g} (relative) after {pieces} pieces exceeds "
            f"rel_tol {tol:g}; raise max_pieces or rel_tol", residual=float(rel), tolerance=tol)
    with mpmath.workdps(CERT_DPS):
        n = len(P) - 1
        Dm = [[mpmath.mpf((n * (b - a)).numerator) / (n * (b - a)).denominator for a, b in zip(P[i], P[i + 1])]
              for i in range(n)]

        def speed(t: Any) -> Any:
            return mpmath.sqrt(sum(c * c for c in _casteljau(Dm, t)))

        est = mpmath.quad(speed, mpmath.linspace(0, 1, 17))
        inside = lower <= est <= upper
        if not inside:  # pragma: no cover - quadrature outside a rigorous bracket: use the midpoint
            est = (mpmath.mpf(lower) + upper) / 2
    return ArcLength("bezier_arc_length", float(est), math.nextafter(lower, -math.inf),
                     math.nextafter(upper, math.inf), pieces,
                     {"status": "verified", "method": "chord <= arc <= control polygon on exact dyadic pieces; "
                      "float lengths of exact differences with a 2^-40 relative margin",
                      "relative_width": float(rel),
                      "estimate_inside_bracket": bool(inside)})


# ----------------------------------------------------------------- CSS easing
CSS_EASING_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "ease": (0.25, 0.1, 0.25, 1.0),
    "ease-in": (0.42, 0.0, 1.0, 1.0),
    "ease-out": (0.0, 0.0, 0.58, 1.0),
    "ease-in-out": (0.42, 0.0, 0.58, 1.0),
}


@dataclass(frozen=True)
class EasingResult(_Result):
    """CSS ``cubic-bezier(x1, y1, x2, y2)`` at input progress ``x``.

    ``y``: the output progress (float, within an ulp or two of the exact value);
    ``t``: the curve parameter with B_x(t) = x; ``enclosure``: floats
    ``(lo, hi)`` that PROVABLY contain the exact y (exact rational bracketing of
    t and the Bernstein convex hull of B_y on that bracket);
    ``exact``: optional SymPy text of the exact algebraic y (``exact=True``).
    """

    kind: str
    x: float
    y: float
    t: "float | None"
    enclosure: tuple[float, float]
    control: tuple[float, float, float, float]
    method: str
    exact: "str | None"
    certificate: dict[str, Any]


def _cubic_coeffs(p1: Num, p2: Num) -> tuple[Any, Any, Any]:
    c = 3 * p1
    b = 3 * (p2 - p1) - c
    a = 1 - c - b
    return a, b, c


def _cardano_unit_root(a: float, b: float, c: float, d: float) -> "float | None":
    """A real root in [0, 1] of a t^3 + b t^2 + c t + d (float Cardano / trig branch)."""
    roots: list[float] = []
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            if c != 0:
                roots = [-d / c]
        else:
            disc = c * c - 4 * b * d
            if disc >= 0:
                sq = math.sqrt(disc)
                roots = [(-c + sq) / (2 * b), (-c - sq) / (2 * b)]
    else:
        B, C, D = b / a, c / a, d / a
        p = C - B * B / 3
        q = 2 * B**3 / 27 - B * C / 3 + D
        disc = q * q / 4 + p**3 / 27
        if disc > 0:
            s = math.sqrt(disc)
            roots = [math.copysign(abs(-q / 2 + s) ** (1 / 3), -q / 2 + s)
                     + math.copysign(abs(-q / 2 - s) ** (1 / 3), -q / 2 - s) - B / 3]
        elif p < 0:  # three real roots: trigonometric branch
            m = 2 * math.sqrt(-p / 3)
            arg = max(-1.0, min(1.0, 3 * q / (p * m)))
            th = math.acos(arg) / 3
            roots = [m * math.cos(th - 2 * math.pi * k / 3) - B / 3 for k in range(3)]
        else:
            roots = [-B / 3]
    good = [r for r in roots if -1e-9 <= r <= 1 + 1e-9 and math.isfinite(r)]
    return min(max(good[0], 0.0), 1.0) if good else None


def _poly3(a: Any, b: Any, c: Any, t: Any) -> Any:
    return ((a * t + b) * t + c) * t


def css_cubic_bezier(x1: Any, y1: Any, x2: Any, y2: Any, x: Any, *, exact: bool = False) -> EasingResult:
    """y for a given x on the CSS easing curve P0=(0,0), P1, P2, P3=(1,1).

    1. ``x1``, ``x2`` must lie in [0, 1] (CSS Easing Level 1 — then B_x is
       monotone and t is unique); ``y1``, ``y2`` are free (overshoot allowed).
    2. Outside [0, 1] the spec extends the curve along the end tangent (P1 or P2
       towards P0/P3); that is done exactly.
    3. Inside: Cardano (real branch, trigonometric form when three roots are real)
       gives t, safeguarded Newton with a bisection bracket polishes it to the last
       ulp; then t is BRACKETED EXACTLY (B_x evaluated in rationals at two floats)
       and y is enclosed by the convex hull of B_y's Bézier control points on that
       bracket.

        >>> round(css_cubic_bezier(0.25, 0.1, 0.25, 1.0, 0.5).y, 15)  # 'ease' at 50%
        0.802403387584857
    """
    X1, Y1, X2, Y2, X = (_scalar(v, n) for v, n in zip((x1, y1, x2, y2, x), ("x1", "y1", "x2", "y2", "x")))
    for v, n in ((X1, "x1"), (X2, "x2")):
        if not 0 <= v <= 1:
            raise DomainError(f"css_cubic_bezier: {n} = {v} is outside [0, 1] — CSS requires the x "
                              "coordinates of P1 and P2 in [0, 1] (otherwise the curve is not a function of x)")
    ctrl = (float(X1), float(Y1), float(X2), float(Y2))
    FX1, FY1, FX2, FY2, FX = (_F(v) for v in (X1, Y1, X2, Y2, X))
    if FX < 0 or FX > 1:
        return _css_extrapolate(FX1, FY1, FX2, FY2, FX, ctrl)
    ax, bx, cx = _cubic_coeffs(FX1, FX2)
    ay, by, cy = _cubic_coeffs(FY1, FY2)
    # ---- float solve: Cardano seed, safeguarded Newton/bisection to the last ulp
    fa, fb, fc, xf = float(ax), float(bx), float(cx), float(FX)
    t = _cardano_unit_root(fa, fb, fc, -xf)
    lo_t, hi_t = 0.0, 1.0
    if t is None:
        t = xf
    for _ in range(100):
        fx = _poly3(fa, fb, fc, t) - xf
        if fx == 0:
            break
        if fx < 0:
            lo_t = max(lo_t, t)
        else:
            hi_t = min(hi_t, t)
        d = (3 * fa * t + 2 * fb) * t + fc
        nt = t - fx / d if d > 0 else (lo_t + hi_t) / 2
        if not lo_t <= nt <= hi_t or nt == t:
            nt = (lo_t + hi_t) / 2
        if nt == t or hi_t - lo_t <= 2 * _EPS * max(1.0, abs(t)):
            break
        t = nt
    # ---- exact bracket of the true t: B_x(tl) <= x <= B_x(th), floats tl <= th
    t_lo, t_hi = _bracket_t(ax, bx, cx, FX, t)
    # ---- y enclosure: Bezier sub-curve of B_y on [t_lo, t_hi], convex hull
    Py = [Fraction(0), FY1, FY2, Fraction(1)]
    _, right = _split([[v] for v in Py], t_lo)
    if t_hi < 1 and t_lo < t_hi:
        sub, _ = _split(right, (t_hi - t_lo) / (1 - t_lo))
    else:
        sub = right if t_lo < t_hi else [(_bernstein([[v] for v in Py], t_lo)[0],)]
    ys = [p[0] for p in sub]
    ylo, yhi = min(ys), max(ys)
    tm = Fraction(t)
    yv = float(_poly3(ay, by, cy, tm))
    enc = (_float_down(ylo), _float_up(yhi))
    yv = min(max(yv, enc[0]), enc[1])
    ex_text = _css_exact(ax, bx, cx, ay, by, cy, FX, ylo, yhi) if exact else None
    return EasingResult(
        "css_cubic_bezier", float(FX), yv, t, enc, ctrl, "cardano+newton", ex_text,
        {"status": "verified", "method": "exact rational bracket of t, Bernstein hull of B_y",
         "t_bracket": (float(t_lo), float(t_hi)), "y_width": float(yhi - ylo)})


def _bracket_t(ax: Fraction, bx: Fraction, cx: Fraction, X: Fraction, t: float) -> tuple[Fraction, Fraction]:
    def bxf(s: float) -> Fraction:
        S = Fraction(s)
        return ((ax * S + bx) * S + cx) * S

    lo = hi = t
    v = bxf(t)
    if v == X:
        return Fraction(t), Fraction(t)
    if v < X:
        step = 0
        hi = t
        while bxf(hi) < X:
            hi = min(1.0, math.nextafter(hi, 2.0) if step < 8 else hi + (1 - hi) / 2 + 2 * _EPS)
            step += 1
            if hi == 1.0:
                break
        lo = math.nextafter(hi, -1.0) if bxf(math.nextafter(hi, -1.0)) <= X else t
    else:
        step = 0
        lo = t
        while bxf(lo) > X:
            lo = max(0.0, math.nextafter(lo, -1.0) if step < 8 else lo / 2 - 2 * _EPS)
            step += 1
            if lo == 0.0:
                break
        hi = math.nextafter(lo, 2.0) if bxf(math.nextafter(lo, 2.0)) >= X else t
    L, H = Fraction(lo), Fraction(hi)
    if not (bxf(lo) <= X <= bxf(hi)):  # pragma: no cover - monotone B_x makes this unreachable
        raise CertificationError("css_cubic_bezier: could not bracket t exactly", residual=1.0, tolerance=0.0)
    return L, H


def _css_exact(ax: Fraction, bx: Fraction, cx: Fraction, ay: Fraction, by: Fraction, cy: Fraction,
               X: Fraction, ylo: Fraction, yhi: Fraction) -> str:
    import sympy as sp

    def R(q: Fraction) -> Any:
        return sp.Rational(q.numerator, q.denominator)

    T = sp.Symbol("t")

    def work() -> str:
        poly = sp.Poly(R(ax) * T**3 + R(bx) * T**2 + R(cx) * T - R(X), T)
        for root in poly.real_roots():
            if 0 <= root <= 1:
                y = sp.expand(R(ay) * root**3 + R(by) * root**2 + R(cy) * root)
                yv = sp.Rational(str(sp.N(y, 40)))
                if not R(ylo) - sp.Rational(1, 10**30) <= yv <= R(yhi) + sp.Rational(1, 10**30):  # pragma: no cover
                    raise CertificationError("css_cubic_bezier: exact root outside the enclosure",
                                             residual=1.0, tolerance=0.0)
                return str(y)
        raise CertificationError("css_cubic_bezier: no exact root in [0, 1]", residual=1.0,  # pragma: no cover
                                 tolerance=0.0)

    return guard("geometry.css_cubic_bezier", "exact easing root", work)


def _css_extrapolate(x1: Fraction, y1: Fraction, x2: Fraction, y2: Fraction, X: Fraction,
                     ctrl: tuple[float, float, float, float]) -> EasingResult:
    """CSS Easing Level 1, §2.2 (cubic Bézier easing functions), outside [0, 1]."""
    if X < 0:
        if x1 > 0:
            y = y1 / x1 * X
        elif x2 > 0:
            y = y2 / x2 * X
        else:
            y = Fraction(0)
    else:
        if x2 < 1:
            y = 1 + (y2 - 1) / (x2 - 1) * (X - 1)
        elif x1 < 1:
            y = 1 + (y1 - 1) / (x1 - 1) * (X - 1)
        else:
            y = Fraction(1)
    return EasingResult("css_cubic_bezier", float(X), float(y), None, (_float_down(y), _float_up(y)), ctrl,
                        "tangent-extrapolation", _fstr(y),
                        {"status": "verified", "method": "exact rational tangent line (CSS Easing 1, §2.2)",
                         "y_width": 0.0})


_CSS_FN = re.compile(r"^\s*cubic-bezier\(\s*([^,]+),([^,]+),([^,]+),([^)]+)\)\s*$")


def css_easing(spec: str, x: Any, *, exact: bool = False) -> EasingResult:
    """``css_easing("ease", 0.5)`` or ``css_easing("cubic-bezier(.68,-.55,.27,1.55)", 0.3)``."""
    if not isinstance(spec, str):
        raise DomainError("css_easing: spec is a CSS keyword or 'cubic-bezier(x1, y1, x2, y2)'")
    key = spec.strip().lower()
    if key in CSS_EASING_PRESETS:
        x1, y1, x2, y2 = CSS_EASING_PRESETS[key]
        return css_cubic_bezier(x1, y1, x2, y2, x, exact=exact)
    m = _CSS_FN.match(key)
    if not m:
        raise DomainError(f"css_easing: unknown easing {spec!r}; use one of "
                          f"{', '.join(CSS_EASING_PRESETS)} or cubic-bezier(x1, y1, x2, y2)")
    vals = [g.strip() for g in m.groups()]
    for v in vals:
        if not _RATIONAL_RE.match(v):
            raise DomainError(f"css_easing: {v!r} is not a number")
    a, b, c, d = (float(v) for v in vals)
    return css_cubic_bezier(a, b, c, d, x, exact=exact)


# ---------------------------------------------------------- splines and steps
def _knot(a: Sequence[float], b: Sequence[float], alpha: float) -> float:
    d = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return d**alpha if alpha else 1.0


def catmull_rom(p0: Any, p1: Any, p2: Any, p3: Any, t: Any, *, alpha: float = 0.0) -> CurveValue:
    """Catmull-Rom segment between p1 (t=0) and p2 (t=1).

    ``alpha``: 0 uniform (the classic matrix form), 0.5 centripetal (no cusps or
    self-intersections inside a segment — Yuksel, Schaefer, Keyser 2011), 1
    chordal. Evaluated with the Barry–Goldman pyramid; certified against the
    cubic Hermite form with the non-uniform tangents (uniform + exact input:
    exactly; otherwise at 50 digits).

        >>> catmull_rom((0,), (1,), (2,), (3,), "1/2").exact   # collinear, evenly spaced
        ('3/2',)
    """
    P = _ctrl([p0, p1, p2, p3], "catmull_rom")
    tt = _scalar(t, "catmull_rom: t")
    a = float(_scalar(alpha, "catmull_rom: alpha"))
    if not 0 <= a <= 1:
        raise DomainError(f"catmull_rom: alpha must be in [0, 1], got {a}")
    flat = [v for p in P for v in p] + [tt]
    if a == 0:
        def uniform_cr(cv: Callable[[Num], Any]) -> tuple[Any, ...]:
            Q = [[cv(v) for v in p] for p in P]
            s = cv(tt)
            return tuple((2 * b + (-a_ + c) * s + (2 * a_ - 5 * b + 4 * c - d) * s * s
                          + (-a_ + 3 * b - 3 * c + d) * s * s * s) / 2 for a_, b, c, d in zip(*Q))

        def herm(cv: Callable[[Num], Any] = _F) -> tuple[Any, ...]:
            Q = [[cv(v) for v in p] for p in P]
            m1 = [(c - a_) / 2 for a_, c in zip(Q[0], Q[2])]
            m2 = [(d - b) / 2 for b, d in zip(Q[1], Q[3])]
            return _hermite_eval(Q[1], m1, Q[2], m2, cv(tt))

        return _point_result("catmull_rom", _is_exact(flat), uniform_cr, lambda: herm(_F))
    Pf = [[float(v) for v in p] for p in P]
    k1, k2, k3 = _knot(Pf[0], Pf[1], a), _knot(Pf[1], Pf[2], a), _knot(Pf[2], Pf[3], a)
    if 0.0 in (k1, k2, k3):
        raise DomainError("catmull_rom: two consecutive control points coincide — the knot interval is 0 "
                          "for alpha > 0 (remove the duplicate point)")
    val = _barry_goldman(Pf, k1, k2, k3, float(tt))
    with mpmath.workdps(CERT_DPS):
        Pm = [[_mp(v) for v in p] for p in P]
        K1, K2, K3 = (_mp(k) for k in (k1, k2, k3))
        m1 = [((b - a_) / K1 - (c - a_) / (K1 + K2) + (c - b) / K2) * K2 for a_, b, c in zip(Pm[0], Pm[1], Pm[2])]
        m2 = [((c - b) / K2 - (d - b) / (K2 + K3) + (d - c) / K3) * K2 for b, c, d in zip(Pm[1], Pm[2], Pm[3])]
        ref = _hermite_eval(Pm[1], m1, Pm[2], m2, _mp(tt))
        err = float(max(abs(_mp(v) - r) for v, r in zip(val, ref)))
    tol = 1e-12 * max(1.0, max(abs(v) for p in Pf for v in p))
    if not err <= tol:
        raise CertificationError(f"catmull_rom: Barry-Goldman and Hermite forms differ by {err:.3g}",
                                 residual=err, tolerance=tol)
    return CurveValue("catmull_rom", tuple(val), None,
                      {"status": "verified", "method": "Barry-Goldman vs non-uniform Hermite form at 50 digits",
                       "error": err, "knots": (0.0, k1, k1 + k2, k1 + k2 + k3)})


def _barry_goldman(P: list[list[float]], k1: float, k2: float, k3: float, s: float) -> tuple[float, ...]:
    t0, t1, t2, t3 = 0.0, k1, k1 + k2, k1 + k2 + k3
    t = t1 + s * (t2 - t1)

    def lerp(a: list[float], b: list[float], ta: float, tb: float) -> list[float]:
        return [((tb - t) * x + (t - ta) * y) / (tb - ta) for x, y in zip(a, b)]

    A1, A2, A3 = lerp(P[0], P[1], t0, t1), lerp(P[1], P[2], t1, t2), lerp(P[2], P[3], t2, t3)
    B1, B2 = lerp(A1, A2, t0, t2), lerp(A2, A3, t1, t3)
    return tuple(lerp(B1, B2, t1, t2))


def _hermite_eval(p0: Sequence[Any], m0: Sequence[Any], p1: Sequence[Any], m1: Sequence[Any], t: Any) -> tuple[Any, ...]:
    t2, t3 = t * t, t * t * t
    h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2
    return tuple(h00 * a + h10 * b + h01 * c + h11 * d for a, b, c, d in zip(p0, m0, p1, m1))


def hermite(p0: Any, m0: Any, p1: Any, m1: Any, t: Any) -> CurveValue:
    """Cubic Hermite: position p0 -> p1, tangents m0, m1, t in [0, 1] (extrapolates outside).

    Certified against its Bézier form (p0, p0 + m0/3, p1 - m1/3, p1) by de Casteljau.

        >>> hermite((0,), (0,), (1,), (0,), "1/2").exact
        ('1/2',)
    """
    pts = _ctrl([p0, m0, p1, m1], "hermite")
    tt = _scalar(t, "hermite")
    flat = [v for p in pts for v in p] + [tt]

    def bez() -> tuple[Fraction, ...]:
        a, ma, b, mb = ([_F(v) for v in p] for p in pts)
        ctrl = [a, [x + y / 3 for x, y in zip(a, ma)], [x - y / 3 for x, y in zip(b, mb)], b]
        return _casteljau(ctrl, _F(tt))

    def fast(cv: Callable[[Num], Any]) -> tuple[Any, ...]:
        a, ma, b, mb = ([cv(v) for v in p] for p in pts)
        return _hermite_eval(a, ma, b, mb, cv(tt))

    return _point_result("hermite", _is_exact(flat), fast, bez)


def _step_input(e0: Any, e1: Any, x: Any, name: str) -> tuple[Num, Num, Num]:
    a, b, v = _scalar(e0, f"{name}: edge0"), _scalar(e1, f"{name}: edge1"), _scalar(x, f"{name}: x")
    if a == b:
        raise DomainError(f"{name}: edge0 == edge1 — the transition has zero width (GLSL leaves this undefined)")
    return a, b, v


def _clamp01(t: Any) -> Any:
    return t * 0 if t < 0 else (t * 0 + 1 if t > 1 else t)


def smoothstep(edge0: Any, edge1: Any, x: Any) -> CurveValue:
    """GLSL ``smoothstep``: t = clamp((x-e0)/(e1-e0), 0, 1); t^2 (3 - 2t).

    edge0 > edge1 is allowed (a falling step, as widely used in shaders), equal
    edges are refused. Certified against the Bézier form (0, 0, 1, 1).

        >>> smoothstep(0, 1, "1/4").exact
        ('5/32',)
    """
    a, b, v = _step_input(edge0, edge1, x, "smoothstep")
    flat = [a, b, v]

    def fast(cv: Callable[[Num], Any]) -> tuple[Any, ...]:
        t = _clamp01((cv(v) - cv(a)) / (cv(b) - cv(a)))
        return (t * t * (3 - 2 * t),)

    def slow() -> tuple[Fraction, ...]:
        t = _clamp01((_F(v) - _F(a)) / (_F(b) - _F(a)))
        return _casteljau([[Fraction(0)], [Fraction(0)], [Fraction(1)], [Fraction(1)]], t)

    return _point_result("smoothstep", _is_exact(flat), fast, slow)


def smootherstep(edge0: Any, edge1: Any, x: Any) -> CurveValue:
    """Perlin's ``smootherstep``: t^3 (t (6t - 15) + 10), zero 1st and 2nd derivative at both ends.

    Certified against the quintic Bézier form (0, 0, 0, 1, 1, 1).
    """
    a, b, v = _step_input(edge0, edge1, x, "smootherstep")

    def fast(cv: Callable[[Num], Any]) -> tuple[Any, ...]:
        t = _clamp01((cv(v) - cv(a)) / (cv(b) - cv(a)))
        return (t * t * t * (t * (6 * t - 15) + 10),)

    def slow() -> tuple[Fraction, ...]:
        t = _clamp01((_F(v) - _F(a)) / (_F(b) - _F(a)))
        return _casteljau([[Fraction(k)] for k in (0, 0, 0, 1, 1, 1)], t)

    return _point_result("smootherstep", _is_exact([a, b, v]), fast, slow)


# ================================================================ code emission
LANGUAGES: tuple[str, ...] = ("python", "numpy", "c", "glsl")


@dataclass(frozen=True)
class _Formula:
    name: str
    doc: str
    params: tuple[tuple[str, tuple[str, ...]], ...]  # (GLSL parameter, its scalar names)
    out_kind: str  # scalar | vec3 | quat | mat3
    build: Callable[[dict[str, Any]], list[Any]]
    reference: Callable[..., tuple[float, ...]]
    sample: Callable[[Any], tuple[float, ...]]
    glsl_components: tuple[str, ...] = ()  # component names for vector params


def _clamp_sym(e: Any) -> Any:
    import sympy as sp

    return sp.Min(sp.Max(e, 0), 1)


def _f_smoothstep(s: dict[str, Any]) -> list[Any]:
    t = _clamp_sym((s["x"] - s["edge0"]) / (s["edge1"] - s["edge0"]))
    return [t * t * (3 - 2 * t)]


def _f_smootherstep(s: dict[str, Any]) -> list[Any]:
    t = _clamp_sym((s["x"] - s["edge0"]) / (s["edge1"] - s["edge0"]))
    return [t * t * t * (t * (6 * t - 15) + 10)]


def _f_hermite(s: dict[str, Any]) -> list[Any]:
    t = s["t"]
    return [(2 * t**3 - 3 * t**2 + 1) * s["p0"] + (t**3 - 2 * t**2 + t) * s["m0"]
            + (-2 * t**3 + 3 * t**2) * s["p1"] + (t**3 - t**2) * s["m1"]]


def _f_catmull_rom(s: dict[str, Any]) -> list[Any]:
    import sympy as sp

    t, a, b, c, d = s["t"], s["p0"], s["p1"], s["p2"], s["p3"]
    return [sp.Rational(1, 2) * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t**2
                                 + (-a + 3 * b - 3 * c + d) * t**3)]


def _f_bezier3(s: dict[str, Any]) -> list[Any]:
    t = s["t"]
    u = 1 - t
    return [u**3 * s["p0"] + 3 * u**2 * t * s["p1"] + 3 * u * t**2 * s["p2"] + t**3 * s["p3"]]


def _f_quat_to_mat3(s: dict[str, Any]) -> list[Any]:
    q = (s["qw"], s["qx"], s["qy"], s["qz"])
    n = _qnorm2(q)
    return [v / n for row in _quat_numerator(q) for v in row]


def _f_quat_mul(s: dict[str, Any]) -> list[Any]:
    return list(_qmul((s["aw"], s["ax"], s["ay"], s["az"]), (s["bw"], s["bx"], s["by"], s["bz"])))


def _f_quat_rotate(s: dict[str, Any]) -> list[Any]:
    q = (s["qw"], s["qx"], s["qy"], s["qz"])
    n = _qnorm2(q)
    N = _quat_numerator(q)
    v = (s["vx"], s["vy"], s["vz"])
    return [sum(N[i][k] * v[k] for k in range(3)) / n for i in range(3)]


def _ref_quat_mat(qw: float, qx: float, qy: float, qz: float) -> tuple[float, ...]:
    m = quat_to_matrix([qw, qx, qy, qz], normalize=True).value
    return tuple(v for row in m for v in row)


def _ref_quat_rotate(qw: float, qx: float, qy: float, qz: float, vx: float, vy: float, vz: float) -> tuple[float, ...]:
    m = _ref_quat_mat(qw, qx, qy, qz)
    v = (vx, vy, vz)
    return tuple(math.fsum(m[3 * i + k] * v[k] for k in range(3)) for i in range(3))


def _ref_css(x1: float, y1: float, x2: float, y2: float, x: float) -> tuple[float, ...]:
    return (css_cubic_bezier(x1, y1, x2, y2, x).y,)


def _u(rng: Any, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


FORMULAS: dict[str, _Formula] = {
    f.name: f for f in (
        _Formula("smoothstep", "GLSL smoothstep: t = clamp((x-e0)/(e1-e0), 0, 1); t*t*(3-2t)",
                 (("edge0", ("edge0",)), ("edge1", ("edge1",)), ("x", ("x",))), "scalar", _f_smoothstep,
                 lambda e0, e1, x: smoothstep(e0, e1, x).value,
                 lambda r: (_u(r, -2, 0), _u(r, 0.5, 3), _u(r, -3, 4))),
        _Formula("smootherstep", "Perlin smootherstep: t^3 (t (6t - 15) + 10)",
                 (("edge0", ("edge0",)), ("edge1", ("edge1",)), ("x", ("x",))), "scalar", _f_smootherstep,
                 lambda e0, e1, x: smootherstep(e0, e1, x).value,
                 lambda r: (_u(r, -2, 0), _u(r, 0.5, 3), _u(r, -3, 4))),
        _Formula("hermite", "cubic Hermite: p0, tangent m0 at t=0; p1, tangent m1 at t=1",
                 (("p0", ("p0",)), ("m0", ("m0",)), ("p1", ("p1",)), ("m1", ("m1",)), ("t", ("t",))), "scalar",
                 _f_hermite, lambda p0, m0, p1, m1, t: hermite((p0,), (m0,), (p1,), (m1,), t).value,
                 lambda r: tuple(_u(r, -5, 5) for _ in range(4)) + (_u(r, 0, 1),)),
        _Formula("catmull_rom", "uniform Catmull-Rom between p1 (t=0) and p2 (t=1)",
                 (("p0", ("p0",)), ("p1", ("p1",)), ("p2", ("p2",)), ("p3", ("p3",)), ("t", ("t",))), "scalar",
                 _f_catmull_rom, lambda a, b, c, d, t: catmull_rom((a,), (b,), (c,), (d,), t).value,
                 lambda r: tuple(_u(r, -5, 5) for _ in range(4)) + (_u(r, 0, 1),)),
        _Formula("bezier3", "cubic Bezier (one coordinate), Bernstein form",
                 (("p0", ("p0",)), ("p1", ("p1",)), ("p2", ("p2",)), ("p3", ("p3",)), ("t", ("t",))), "scalar",
                 _f_bezier3, lambda a, b, c, d, t: bezier_point([(a,), (b,), (c,), (d,)], t).value,
                 lambda r: tuple(_u(r, -5, 5) for _ in range(4)) + (_u(r, 0, 1),)),
        _Formula("quat_to_mat3", "rotation matrix of a Hamilton quaternion; homogeneous form, "
                 "exact rotation of q/|q| even when |q| != 1",
                 (("q", ("qx", "qy", "qz", "qw")),), "mat3", _f_quat_to_mat3,
                 lambda qx, qy, qz, qw: _ref_quat_mat(qw, qx, qy, qz),
                 lambda r: tuple(_u(r, -1, 1) for _ in range(4)), ("x", "y", "z", "w")),
        _Formula("quat_mul", "Hamilton product a*b (apply b first, then a)",
                 (("a", ("ax", "ay", "az", "aw")), ("b", ("bx", "by", "bz", "bw"))), "quat", _f_quat_mul,
                 lambda ax, ay, az, aw, bx, by, bz, bw: quat_multiply([aw, ax, ay, az], [bw, bx, by, bz]).value,
                 lambda r: tuple(_u(r, -1, 1) for _ in range(8)), ("x", "y", "z", "w")),
        _Formula("quat_rotate", "rotate vector v by Hamilton quaternion q (q/|q| used)",
                 (("q", ("qx", "qy", "qz", "qw")), ("v", ("vx", "vy", "vz"))), "vec3", _f_quat_rotate,
                 lambda qx, qy, qz, qw, vx, vy, vz: _ref_quat_rotate(qw, qx, qy, qz, vx, vy, vz),
                 lambda r: tuple(_u(r, -1, 1) for _ in range(4)) + tuple(_u(r, -10, 10) for _ in range(3)),
                 ("x", "y", "z", "w")),
        _Formula("css_cubic_bezier", "CSS cubic-bezier easing: y at progress x",
                 (("x1", ("x1",)), ("y1", ("y1",)), ("x2", ("x2",)), ("y2", ("y2",)), ("x", ("x",))), "scalar",
                 lambda s: [], _ref_css,
                 lambda r: (_u(r, 0, 1), _u(r, -1, 2), _u(r, 0, 1), _u(r, -1, 2), _u(r, -0.2, 1.2))),
    )
}


@dataclass(frozen=True)
class EmittedCode(_Result):
    """Source code for one formula in one language, with its parity check.

    ``certificate["status"]``: ``"verified"`` when the code was executed (Python,
    NumPy) or compiled and run (C; GLSL through ``_glsl.GLSL_AS_C_SHIM``) on
    ``points`` seeded inputs and matched the certified Python implementation to
    ``max_rel_error``; ``"undecided"`` when no C compiler was available (nothing
    is claimed then).
    """

    kind: str
    formula: str
    language: str
    code: str
    entry: str
    signature: str
    notes: tuple[str, ...]
    certificate: dict[str, Any]


def _render(f: _Formula, lang: str) -> str:
    import sympy as sp

    from . import _glsl

    if f.name == "css_cubic_bezier":
        return _glsl.EASING_TEMPLATES[lang]
    syms = {n: sp.Symbol(n, real=True) for _, ns in f.params for n in ns}
    gnames: dict[Any, str] = {}
    for pname, ns in f.params:
        if len(ns) > 1:
            for n, comp in zip(ns, f.glsl_components):
                gnames[syms[n]] = f"{pname}.{comp}"
    outputs = [sp.sympify(e) for e in f.build(syms)]
    params = [(pname, [syms[n] for n in ns]) for pname, ns in f.params]
    return _glsl.render(f.name, lang, params, outputs, f.out_kind, gnames, f.doc)


def _flat_args(f: _Formula) -> list[str]:
    return [n for _, ns in f.params for n in ns]


def _out_len(f: _Formula) -> int:
    return {"scalar": 1, "vec3": 3, "quat": 4, "mat3": 9}[f.out_kind]


def emit(formula: str, language: str, *, verify: bool = True, points: int = 64, seed: int = 0) -> EmittedCode:
    """Emit ``formula`` (a key of :data:`FORMULAS`) as ``language`` source code.

    Languages: ``python`` (``math`` only), ``numpy`` (vectorised; matrices come
    out with shape ``(..., 3, 3)``), ``c`` (C99, ``double``; vector results via
    ``out[]``: quaternions as w, x, y, z, matrices row-major), ``glsl`` (vector
    parameters as ``vec4``/``vec3``; quaternions are ``vec4(x, y, z, w)`` — scalar
    LAST, as GLSL code conventionally stores them; ``mat3`` is built column by
    column, as the GLSL constructor requires).

    With ``verify=True`` the code is run on ``points`` seeded inputs and compared
    with this module's certified implementation.

        >>> print(emit("smoothstep", "glsl", verify=False).code.splitlines()[1])
        float smoothstep(float edge0, float edge1, float x) {
    """
    if formula not in FORMULAS:
        raise DomainError(f"emit: unknown formula {formula!r}; available: {', '.join(FORMULAS)}")
    if language not in LANGUAGES:
        raise DomainError(f"emit: language must be one of {', '.join(LANGUAGES)}, got {language!r}")
    if not isinstance(points, int) or points < 1:
        raise DomainError("emit: points must be a positive integer")
    f = FORMULAS[formula]
    code = _render(f, language)
    sig = code.splitlines()[1 if language in ("c", "glsl") else 0]
    notes: list[str] = []
    if language == "glsl":
        notes.append("GPU float is fp32: expect ~1e-7 relative accuracy; the check below ran in fp64")
        if f.name == "smoothstep":
            notes.append("GLSL already has a built-in smoothstep; this is the explicit formula "
                         "(the built-in is undefined for edge0 >= edge1)")
    if f.out_kind == "quat" and language in ("python", "numpy", "c"):
        notes.append("quaternion parameters and results are (w, x, y, z)" if language == "c"
                     else "quaternion results are (w, x, y, z); parameters are named")
    cert: dict[str, Any] = {"status": "unchecked"}
    if verify:
        cert = _verify_emitted(f, language, code, points, seed)
    return EmittedCode("emit", formula, language, code, formula, sig.strip(), tuple(notes), cert)


def _samples(f: _Formula, points: int, seed: int) -> list[tuple[float, ...]]:
    import random

    rng = random.Random(seed)
    out = [f.sample(rng) for _ in range(points)]
    # GLSL-ordered params are (x, y, z, w) for quaternions — sample tuples follow _flat_args order
    return out


def _compare(f: _Formula, got: Sequence[Sequence[float]], args: list[tuple[float, ...]]) -> dict[str, Any]:
    worst = 0.0
    for a, g in zip(args, got):
        ref = f.reference(*a)
        for x, y in zip(g, ref):
            worst = max(worst, abs(x - y) / max(1.0, abs(y)))
    ok = worst <= 1e-12
    return {"status": "verified" if ok else "refuted", "points": len(args), "max_rel_error": worst,
            "tolerance": 1e-12, "reference": "pycodemath.code.geometry certified implementation"}


def _verify_emitted(f: _Formula, lang: str, code: str, points: int, seed: int) -> dict[str, Any]:
    args = _samples(f, points, seed)
    n_out = _out_len(f)
    if lang in ("python", "numpy"):
        import numpy

        ns: dict[str, Any] = {"math": math, "numpy": numpy}
        exec(compile(code, f"<emit {f.name} {lang}>", "exec"), ns)  # noqa: S102 - our own generated code
        fn = ns[f.name]
        if lang == "python":
            got = [_flatten(fn(*a)) for a in args]
        else:
            cols = [numpy.array([a[i] for a in args]) for i in range(len(args[0]))]
            res = numpy.asarray(fn(*cols)).reshape(len(args), -1)
            got = [tuple(float(v) for v in row) for row in res]
        cert = _compare(f, got, args)
        cert["method"] = "executed"
        return cert
    return _verify_compiled(f, lang, code, args, n_out)


def _flatten(v: Any) -> tuple[float, ...]:
    if isinstance(v, (tuple, list)):
        return tuple(x for item in v for x in _flatten(item))
    return (float(v),)


def _verify_compiled(f: _Formula, lang: str, code: str, args: list[tuple[float, ...]], n_out: int) -> dict[str, Any]:
    import shutil
    import subprocess
    import tempfile

    from . import _glsl

    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        return {"status": "undecided", "reason": "no C compiler found; the code was not compiled", "points": 0}
    names = _flat_args(f)
    lines = ["#include <stdio.h>", "#include <math.h>"]
    if lang == "glsl":
        lines.append(_glsl.GLSL_AS_C_SHIM)
    lines.append(code)
    lines.append("int main(void) {")
    for a in args:
        lits = [repr(float(v)) for v in a]
        if lang == "c":
            if f.out_kind == "scalar":
                lines.append(f'    printf("%.17g\\n", {f.name}({", ".join(lits)}));')
            else:
                lines.append("    { double out[%d]; %s(%s, out);" % (n_out, f.name, ", ".join(lits)))
                lines.append('      for (int i = 0; i < %d; i++) { printf("%%.17g ", out[i]); } printf("\\n"); }' % n_out)
        else:
            gargs, k = [], 0
            for pname, ns in f.params:
                if len(ns) == 1:
                    gargs.append(lits[k])
                else:
                    gargs.append(f"vec{len(ns)}({', '.join(lits[k:k + len(ns)])})")
                k += len(ns)
            call = f"{f.name}({', '.join(gargs)})"
            if f.out_kind == "scalar":
                lines.append(f'    printf("%.17g\\n", {call});')
            elif f.out_kind == "quat":  # vec4 (x,y,z,w) -> print w x y z
                lines.append(f'    {{ vec4 r = {call}; printf("%.17g %.17g %.17g %.17g\\n", r.w, r.x, r.y, r.z); }}')
            elif f.out_kind == "vec3":
                lines.append(f'    {{ vec3 r = {call}; printf("%.17g %.17g %.17g\\n", r.x, r.y, r.z); }}')
            else:  # mat3: print row-major from m.c[col][row]
                lines.append(f"    {{ mat3 m = {call}; for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) "
                             'printf("%.17g ", m.c[c][r]); printf("\\n"); }')
    lines.append("    return 0;\n}")
    src = "\n".join(lines)
    del names
    with tempfile.TemporaryDirectory() as d:
        cpath, exe = f"{d}/emit.c", f"{d}/emit"
        with open(cpath, "w") as fh:
            fh.write(src)
        comp = subprocess.run([cc, "-std=c99", "-O1", "-Wall", "-Werror", "-Wno-unused-function", "-o", exe, cpath,
                               "-lm"], capture_output=True, text=True, timeout=60)
        if comp.returncode != 0:
            return {"status": "refuted", "method": "compile", "points": 0,
                    "compiler_error": comp.stderr.strip()[:2000]}
        run = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    got = [tuple(float(v) for v in line.split()) for line in run.stdout.strip().splitlines()]
    cert = _compare(f, got, args)
    cert["method"] = "compiled with cc and run" + (" (GLSL through the GLSL-as-C shim)" if lang == "glsl" else "")
    return cert
