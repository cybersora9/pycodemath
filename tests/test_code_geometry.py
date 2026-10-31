"""G1 — geometry and graphics maths for programmers (pycodemath.code.geometry)."""
from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from pycodemath.code import geometry as g


# --------------------------------------------------------------------- guards
def test_quaternion_matrix_roundtrip_and_order() -> None:
    """quat -> matrix -> quat is the identity (up to the q/-q cover), the matrix is a
    proper rotation, and the Hamilton product composes like column-vector matrices —
    while the reversed order is REFUTED with a concrete witness."""
    rng = random.Random(1)
    for _ in range(200):
        q = [rng.uniform(-1, 1) for _ in range(4)]
        n = math.sqrt(sum(v * v for v in q))
        q = [v / n for v in q]
        m = g.quat_to_matrix(q)
        R = m.value
        for i in range(3):
            for j in range(3):
                dot = sum(R[k][i] * R[k][j] for k in range(3))
                assert abs(dot - (i == j)) < 1e-14
        back = g.matrix_to_quat(R).value
        sgn = 1 if sum(a * b for a, b in zip(back, q)) > 0 else -1
        assert max(abs(a - sgn * b) for a, b in zip(back, q)) < 1e-14
        assert m.certificate["status"] == "verified"
    same = g.compare_rotation_expressions("q1*q2", "R1@R2")
    assert same.status == "verified" and same.method == "symbolic"
    rev = g.compare_rotation_expressions("q1*q2", "R2@R1")
    assert rev.status == "refuted"
    assert "reversed_order" in rev.causes
    assert set(rev.witness) >= {"1", "2"}


def _exact_orient(a, b, c) -> int:
    a, b, c = [(Fraction(x), Fraction(y)) for x, y in (a, b, c)]
    d = (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])
    return (d > 0) - (d < 0)


def test_filtered_orient2d_never_wrong_sign() -> None:
    """The float orient2d with Shewchuk's error filter never returns a wrong sign:
    on the whole 256x256 Kettner et al. (CGTA 2008) ulp grid — where the naive
    float determinant is wrong 11 492 times — and on near-degenerate random data."""
    q, r = (12.0, 12.0), (24.0, 24.0)
    naive_wrong = 0
    for i in range(256):
        for j in range(256):
            p = (0.5 + i * 2.0**-53, 0.5 + j * 2.0**-53)
            exact = _exact_orient(p, q, r)
            assert g.orient2d(p, q, r) == exact
            d = (p[0] - r[0]) * (q[1] - r[1]) - (p[1] - r[1]) * (q[0] - r[0])
            naive_wrong += ((d > 0) - (d < 0)) != exact
    assert naive_wrong == 11492
    rng = random.Random(7)
    for _ in range(3000):
        a = (rng.uniform(-1, 1), rng.uniform(-1, 1))
        b = (rng.uniform(-1, 1), rng.uniform(-1, 1))
        t = rng.uniform(-2, 2)
        c = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))  # rounded onto the line
        scale = 2.0 ** rng.randint(-200, 200)
        a, b, c = [(x * scale, y * scale) for x, y in (a, b, c)]
        assert g.orient2d(a, b, c) == _exact_orient(a, b, c)


def _chromium_unit_bezier(x1, y1, x2, y2, x):
    """Port of Chromium ui/gfx/geometry/cubic_bezier.cc (BSD) SolveCurveX/SampleCurveY."""
    cx = 3.0 * x1; bx = 3.0 * (x2 - x1) - cx; ax = 1.0 - cx - bx
    cy = 3.0 * y1; by = 3.0 * (y2 - y1) - cy; ay = 1.0 - cy - by
    sx = lambda t: ((ax * t + bx) * t + cx) * t
    sy = lambda t: ((ay * t + by) * t + cy) * t
    dx = lambda t: (3.0 * ax * t + 2.0 * bx) * t + cx
    eps = 1e-7
    samples = [sx(i / 10) for i in range(11)]
    t0 = 0.0; t1 = 1.0; t2 = x
    for i in range(1, 11):
        if x <= samples[i]:
            t1 = i / 10; t0 = t1 - 0.1
            t2 = t0 + (t1 - t0) * (x - samples[i - 1]) / (samples[i] - samples[i - 1])
            break
    x2_ = 0.0
    for _ in range(4):
        x2_ = sx(t2) - x
        if abs(x2_) < eps:
            return sy(t2)
        d2 = dx(t2)
        if abs(d2) < eps:
            break
        t2 = t2 - x2_ / d2
    if abs(x2_) < eps:
        return sy(t2)
    while t0 < t1:
        x2_ = sx(t2)
        if abs(x2_ - x) < eps:
            return sy(t2)
        if x > x2_:
            t0 = t2
        else:
            t1 = t2
        t2 = (t1 + t0) * 0.5
    return sy(t2)


def test_css_easing_matches_reference() -> None:
    """y for a given x on CSS cubic-bezier: agrees with Chromium's UnitBezier within
    Chromium's own tolerance, and with a 60-digit bisection to ~1e-15; the returned
    enclosure always contains the exact answer."""
    presets = {"ease": (0.25, 0.1, 0.25, 1.0), "ease-in": (0.42, 0.0, 1.0, 1.0),
               "ease-out": (0.0, 0.0, 0.58, 1.0), "ease-in-out": (0.42, 0.0, 0.58, 1.0),
               "back": (0.68, -0.55, 0.265, 1.55)}
    for name, (x1, y1, x2, y2) in presets.items():
        for k in range(101):
            x = k / 100
            res = g.css_cubic_bezier(x1, y1, x2, y2, x)
            ref = _chromium_unit_bezier(x1, y1, x2, y2, x)
            # Chromium stops at |B_x(t)-x| < 1e-7; |dy/dx| <= 3.5 on these curves
            assert abs(res.y - ref) < 4e-7, (name, x)
            lo, hi = res.enclosure
            assert lo <= res.y <= hi and hi - lo < 1e-14
    ease = g.css_cubic_bezier(0.25, 0.1, 0.25, 1.0, 0.5)
    # 60-digit mpmath bisection on B_x(t) = 0.5 (dev-journal/full/G1/make_cases.py)
    assert abs(ease.y - 0.802403387584856990570789101035) < 2e-16
    assert g.css_easing("ease-in-out", 0.5).y == 0.5


# ===================================================================== rotations
SEQS = g.EULER_SEQUENCES


def _scipy():
    return pytest.importorskip("scipy.spatial.transform")


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("frame", ["intrinsic", "extrinsic"])
def test_euler_matches_scipy_all_24_conventions(seq: str, frame: str) -> None:
    st = _scipy()
    rng = random.Random(hash((seq, frame)) & 0xFFFF)
    sp_seq = seq.upper() if frame == "intrinsic" else seq
    for _ in range(20):
        ang = [rng.uniform(-3, 3), rng.uniform(0.1, 1.4) if seq[0] != seq[2] else rng.uniform(0.1, 3.0),
               rng.uniform(-3, 3)]
        mine = g.euler_to_matrix(ang, seq, frame=frame).value
        ref = st.Rotation.from_euler(sp_seq, ang).as_matrix()
        assert max(abs(mine[i][j] - ref[i][j]) for i in range(3) for j in range(3)) < 1e-14
        back = g.matrix_to_euler(mine, seq, frame=frame)
        assert back.certificate["status"] == "verified"
        assert max(abs(a - b) for a, b in zip(back.value, ang)) < 1e-12


@pytest.mark.parametrize("seq", SEQS)
def test_gimbal_lock_roundtrip_is_exact_where_scipy_drifts(seq: str) -> None:
    """Within 1e-7 rad of gimbal lock SciPy zeroes the third angle and the
    recomposed rotation drifts by up to ~6e-8; here the round trip stays ~1e-15."""
    for frame in ("intrinsic", "extrinsic"):
        for mid_off in (0.0, 1e-12, 1e-9, 5e-8):
            mid = (mid_off if seq[0] == seq[2] else math.pi / 2 - mid_off)
            m = g.euler_to_matrix([0.3, mid, 0.5], seq, frame=frame).value
            r = g.matrix_to_euler(m, seq, frame=frame)
            back = g.euler_to_matrix(r.value, seq, frame=frame).value
            assert max(abs(back[i][j] - m[i][j]) for i in range(3) for j in range(3)) < 1e-14
            assert r.gimbal_lock == (mid_off == 0.0) or mid_off < 1e-12


def test_frame_is_required_and_seq_validated() -> None:
    with pytest.raises(g.DomainError, match="frame must be stated"):
        g.euler_to_matrix([0, 0, 0], "xyz", frame="fixed")
    with pytest.raises(g.DomainError, match="Euler sequence"):
        g.euler_to_matrix([0, 0, 0], "xxz", frame="intrinsic")


def test_quaternion_conventions_order_and_jpl() -> None:
    q = [0.9, 0.1, -0.3, 0.2]
    n = math.sqrt(sum(v * v for v in q))
    q = [v / n for v in q]
    h = g.quat_to_matrix(q).value
    xyzw = g.quat_to_matrix([q[1], q[2], q[3], q[0]], quat=g.QuatConvention(order="xyzw")).value
    assert h == xyzw
    jpl = g.quat_to_matrix(q, quat=g.QuatConvention(algebra="jpl")).value
    assert all(abs(jpl[i][j] - h[j][i]) < 1e-15 for i in range(3) for j in range(3))
    row = g.quat_to_matrix(q, matrix=g.MatrixConvention(vectors="row")).value
    assert row == tuple(zip(*h))
    both = g.quat_to_matrix(q, matrix=g.MatrixConvention(vectors="row", mapping="passive")).value
    assert both == h


def test_quat_multiply_exact_and_jpl_reverses() -> None:
    assert g.quat_multiply([0, 1, 0, 0], [0, 0, 1, 0]).exact == ("0", "0", "0", "1")
    assert g.quat_multiply([0, 1, 0, 0], [0, 0, 1, 0], quat=g.QuatConvention(algebra="jpl")).exact == ("0", "0", "0", "-1")
    r = g.quat_multiply([0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0])
    assert r.certificate["status"] == "verified" and r.certificate["residual"] < 1e-15


def test_non_unit_quaternion_refused_with_the_naive_formulas_damage() -> None:
    with pytest.raises(g.NonUnitQuaternionError) as e:
        g.quat_to_matrix([1, 1, 0, 0])
    assert e.value.norm == pytest.approx(math.sqrt(2))
    assert "det" in str(e.value)
    ok = g.quat_to_matrix([1, 1, 0, 0], normalize=True)
    assert ok.exact == (("1", "0", "0"), ("0", "0", "-1"), ("0", "1", "0"))
    assert ok.certificate["exact_orthogonal"] is True and ok.certificate["exact_det"] == "1"


def test_tiny_and_huge_float_quaternions_regression() -> None:
    """Adversarial round: |q|^2 underflowed (false 'zero quaternion') / overflowed."""
    assert g.quat_to_matrix([1e-200, 1e-200, 0, 0], normalize=True).value[1][2] == pytest.approx(-1.0, abs=1e-15)
    assert g.quat_to_matrix([1e200, 0, 1e200, 0], normalize=True).value[0][2] == pytest.approx(1.0, abs=1e-15)


def test_not_a_rotation_matrix() -> None:
    with pytest.raises(g.NotARotationError, match="reflection"):
        g.matrix_to_quat([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    with pytest.raises(g.NotARotationError, match="not a rotation"):
        g.matrix_to_quat([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])


def test_axis_angle_roundtrip_and_identity() -> None:
    r = g.axis_angle_to_quat([0, 0, 2], 90, degrees=True)
    assert r.value == pytest.approx((math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
    back = g.quat_to_axis_angle(r.value, degrees=True)
    assert back.value[1] == pytest.approx(90)
    ident = g.quat_to_axis_angle([1, 0, 0, 0])
    assert ident.value == ((1.0, 0.0, 0.0), 0.0) and "arbitrary" in ident.notes[0]
    m = g.matrix_to_axis_angle([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert m.value[1] == pytest.approx(math.pi)


def test_slerp_shortest_path_and_near_parallel_constant_speed() -> None:
    q0 = [1.0, 0.0, 0.0, 0.0]
    q1 = [-math.cos(0.2), 0.0, 0.0, -math.sin(0.2)]  # same rotation as +q1: long way round if not flipped
    r = g.slerp(q0, q1, 0.5)
    assert "negated" in r.notes[0]
    assert r.value == pytest.approx((math.cos(0.1), 0, 0, math.sin(0.1)), abs=1e-15)
    long = g.slerp(q0, q1, 0.5, shortest=False)
    assert abs(long.value[0]) < 0.2  # through the far side
    # near-parallel: angle 2e-9 rad; constant angular speed, no lerp fallback
    for t in (0.1, 0.5, 0.9):
        v = g.slerp(q0, [math.cos(1e-9), math.sin(1e-9), 0, 0], t).value
        assert v[1] == pytest.approx(math.sin(t * 1e-9), rel=1e-15)


def test_rotation_identities_are_proved() -> None:
    ids = g.prove_rotation_identities()
    names = {r["id"] for r in ids}
    assert {"orthogonal", "det", "homomorphism", "jpl_homomorphism", "rodrigues"} <= names
    assert sum(n.startswith("euler_") for n in names) == 24
    assert all(r["proved"] for r in ids)


def test_mirror_rotation_handedness() -> None:
    r = g.mirror_rotation([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "z")
    assert r.exact == (("0", "-1", "0"), ("1", "0", "0"), ("0", "0", "1"))  # about z: unchanged
    r = g.mirror_rotation([[1, 0, 0], [0, 0, -1], [0, 1, 0]], "z")
    assert r.exact == (("1", "0", "0"), ("0", "0", "1"), ("0", "-1", "0"))  # about x: reversed


def test_identify_euler_convention() -> None:
    st = _scipy()
    m = st.Rotation.from_euler("ZYX", [0.3, 0.2, 0.1]).as_matrix()
    found = g.identify_euler_convention([0.3, 0.2, 0.1], m)
    assert [(f.scipy_seq, f.transposed) for f in found] == [("ZYX", False)]
    found_t = g.identify_euler_convention([0.3, 0.2, 0.1], m.T)
    assert ("ZYX", True) in [(f.scipy_seq, f.transposed) for f in found_t]


# ================================================================ detector
@pytest.mark.parametrize("a,b,status,cause", [
    ("q1*q2*q3", "R1@R2@R3", "verified", None),
    ("conj(q2)*conj(q1)", "conj(q1*q2)", "verified", None),
    ("q1*q2*conj(q1)", "R1@R2@R1.T", "verified", None),
    ("R1.T.T", "q1", "verified", None),
    ("Sz@R1@Sz", "Sz@R1@Sz", "verified", None),
    ("q1*q2", "R2@R1", "refuted", "reversed_order"),
    ("q1*q2", "(R1@R2).T", "refuted", "transposed"),
    ("q1", "Sx@R1@Sx", "refuted", "handedness_mirror_x"),
    ("q1*q2", "R1@R1", "refuted", None),
])
def test_compare_rotation_expressions(a: str, b: str, status: str, cause: "str | None") -> None:
    r = g.compare_rotation_expressions(a, b)
    assert r.status == status
    if cause:
        assert cause in r.causes
    if status == "refuted":
        # the witness is real: re-evaluate both sides exactly at it
        assert r.matrix_a != r.matrix_b and r.max_abs_difference > 0
        d = r.to_dict()
        assert d["status"] == "refuted" and d["witness"]


def test_compare_conventions_jpl_row_order() -> None:
    jpl = g.QuatConvention(algebra="jpl")
    assert g.compare_rotation_expressions("q1*q2", "R1@R2", quat=jpl).status == "verified"
    mixed = g.compare_rotation_expressions("q1*q2", "R1@R2", quat=jpl, quat_b=g.QuatConvention())
    assert mixed.status == "refuted" and "jpl_vs_hamilton" in mixed.causes
    row = g.MatrixConvention(vectors="row")
    assert g.compare_rotation_expressions("q1*q2", "R2@R1", matrix_b=row).status == "verified"
    bad = g.compare_rotation_expressions("q1*q2", "R1@R2", matrix_b=row)
    assert "row_vs_column_vectors" in bad.causes
    order = g.compare_rotation_expressions("q1*q2", "q1*q2", quat_b=g.QuatConvention(order="xyzw"))
    assert order.causes == ("component_order",)


def test_compare_never_false_alarm_random_expressions() -> None:
    """Fuzz: random products of 1-3 rotations with random transposes; the verdict
    must agree with a direct float evaluation at random rotations."""
    rng = random.Random(3)
    for _ in range(60):
        k = rng.randint(1, 3)
        ids = [str(rng.randint(1, 3)) for _ in range(k)]
        qa = "*".join(("conj(q%s)" if rng.random() < 0.3 else "q%s") % i for i in ids)
        perm = ids[:] if rng.random() < 0.6 else ids[::-1]
        rb = "@".join(f"R{i}" for i in perm)
        if rng.random() < 0.3:
            rb = f"({rb}).T"
        r = g.compare_rotation_expressions(qa, rb)
        # float check at 3 random rotations
        same = True
        for _ in range(3):
            qs = {}
            for i in "123":
                v = [rng.gauss(0, 1) for _ in range(4)]
                qs[i] = v
            def qm(i):
                return [[x for x in row] for row in g.quat_to_matrix(qs[i], normalize=True).value]
            def mm(a, b):
                return [[sum(a[r_][k_] * b[k_][c] for k_ in range(3)) for c in range(3)] for r_ in range(3)]
            A = None
            for tok in qa.split("*"):
                i = tok.strip("conj(q)")
                M = qm(i)
                if tok.startswith("conj"):
                    M = [list(x) for x in zip(*M)]
                A = M if A is None else mm(A, M)
            B = None
            for i in perm:
                B = qm(i) if B is None else mm(B, qm(i))
            if rb.endswith(".T"):
                B = [list(x) for x in zip(*B)]
            if max(abs(A[i_][j] - B[i_][j]) for i_ in range(3) for j in range(3)) > 1e-9:
                same = False
        assert r.status == ("verified" if same else "refuted"), (qa, rb)


def test_compare_refuses_bad_syntax() -> None:
    for a, b in (("R1*R2", "q1*q2"), ("q1*R1", "R1"), ("__import__('os')", "q1"), ("q1+q2", "q1"), ("", "q1")):
        with pytest.raises(g.DomainError):
            g.compare_rotation_expressions(a, b)


# =============================================================== predicates
def _exact3(a, b, c, d) -> int:
    m = [[Fraction(p[k]) - Fraction(d[k]) for k in range(3)] for p in (a, b, c)]
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    return (det > 0) - (det < 0)


def test_orient3d_and_incircle_never_wrong_sign() -> None:
    rng = random.Random(11)
    for _ in range(1500):
        a, b, c = ([rng.uniform(-1, 1) for _ in range(3)] for _ in range(3))
        s, t = rng.uniform(-1, 2), rng.uniform(-1, 2)
        d = [a[k] + s * (b[k] - a[k]) + t * (c[k] - a[k]) for k in range(3)]  # nearly coplanar
        scale = 2.0 ** rng.randint(-60, 60)
        P = [[v * scale for v in p] for p in (a, b, c, d)]
        assert g.orient3d(*P) == _exact3(*P)
    for _ in range(1500):
        th = [rng.uniform(0, 2 * math.pi) for _ in range(4)]
        pts = [(math.cos(x), math.sin(x)) for x in th]  # nearly cocircular
        assert g.incircle(*pts) == g.incircle_exact(*pts)


def test_filter_handles_underflow_and_overflow() -> None:
    assert g.classify_orient2d((5e-324, 0), (0, 5e-324), (0, 0)).stage == "exact"
    assert g.orient2d((5e-324, 0), (0, 5e-324), (0, 0)) == 1
    assert g.orient2d((1.7e308, 1.7e308), (-1.7e308, 1.7e308), (0, -1.7e308)) == 1
    assert g.classify_orient2d((0, 0), (1, 0), (0, 1)).stage == "exact-input"
    assert g.classify_orient2d((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)).stage == "filter"


def test_segment_intersections() -> None:
    assert g.segment_intersection((0, 0), (2, 2), (0, 2), (2, 0)).kind == "proper"
    assert g.segment_intersection((0, 0), (2, 0), (1, 0), (1, 5)).kind == "touching"
    r = g.segment_intersection((0, 0), (2, 0), (1, 0), (3, 0))
    assert r.kind == "collinear_overlap" and r.segment == (("1", "0"), ("2", "0"))
    # one ulp miss
    assert g.segment_intersection((0, 0), (1, 1), (0, 1), (0.4, 0.6000000000000001)).kind == "disjoint"
    assert g.segment_intersection((0.1, 0.1), (0.3, 0.3), (0.1, 0.3), (0.3, 0.1)).kind == "proper"


def test_ray_triangle_exact_edges_and_float_reference() -> None:
    tri = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    r = g.ray_triangle((0.5, 0.5, 1), (0, 0, -1), *tri)
    assert r.hit and r.on_edge and not r.on_vertex
    assert g.ray_triangle((0, 0, 1), (0, 0, -1), *tri).on_vertex
    assert g.ray_triangle((0.25, 0.25, 1), (0, 0, 1), *tri).reason == "behind"
    assert g.ray_triangle((0.25, 0.25, 1), (0, 0, -1), (0, 0, 0), (0, 1, 0), (1, 0, 0),
                          cull_backface=True).reason == "backface"
    assert g.moller_trumbore_float((0.25, 0.25, 1.0), (0.0, 0.0, -1.0), *tri) == 1.0


def test_ray_sphere_exact_classification() -> None:
    assert g.ray_sphere((1, 0, -5), (0, 0, 1), (0, 0, 0), 1).kind == "tangent"
    assert g.ray_sphere((1.0000000000000002, 0, -5), (0, 0, 1), (0, 0, 0), 1).kind == "miss"
    r = g.ray_sphere((0, 0, -5), (0, 0, 1), (0, 0, 0), 2)
    assert r.t == ("3", "7") and r.kind == "secant"
    r = g.ray_sphere((0.5, 0, -5), (0, 0, 1), (0, 0, 0), 1)  # t = 5 -+ sqrt(3)/2
    assert "sqrt(3)" in r.t[0] and r.kind == "secant"
    assert r.approx_t == pytest.approx((5 - math.sqrt(3) / 2, 5 + math.sqrt(3) / 2))
    # adversarial-round regression: origin on the sphere
    assert g.ray_sphere((1, 0, 0), (1, 0, 0), (0, 0, 0), 1).t == ("0",)


def test_point_in_polygon_half_open_tiling() -> None:
    """Every point of a 3x3 tiling (incl. shared edges and corners) is owned by
    exactly one unit square under the half-open rule."""
    squares = [[(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)] for i in range(3) for j in range(3)]
    for k in range(1, 12):
        for m in range(1, 12):
            p = (Fraction(k, 4), Fraction(m, 4))
            owners = sum(g.point_in_polygon(p, sq).half_open for sq in squares)
            assert owners == 1, p
    assert g.point_in_polygon((1, 0.5), squares[0]).location == "boundary"
    bow = [(0, 0), (2, 2), (2, 0), (0, 2)]  # self-intersecting
    assert g.point_in_polygon((0.5, 1), bow, fill_rule="evenodd").location == "inside"   # left lobe
    assert g.point_in_polygon((1, 0.5), bow, fill_rule="evenodd").location == "outside"  # the notch


def test_epsilon_witnesses_are_real() -> None:
    for eps in (0.0, 1e-12, 1e-6, 1e-3):
        r = g.epsilon_witness("orient2d", eps)
        assert r.status == "refuted"
        for w in r.witnesses:
            a, b, c = w["input"]
            d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            exact = g.orient2d_exact(a, b, c)
            if w["kind"] == "false_degenerate":
                assert abs(d) < eps and exact != 0
            else:
                assert abs(d) >= eps and ((d > 0) - (d < 0)) != exact
        rt = g.epsilon_witness("ray_triangle", eps)
        assert rt.status == "refuted"
        kinds = {w["kind"] for w in rt.witnesses}
        assert "edge_leak" in kinds and (eps == 0 or "missed_hit" in kinds)
    q, r_ = (12.0, 12.0), (24.0, 24.0)
    assert g._kettner_search() == g._kettner_wrong_sign()  # the recorded witness is the grid's worst


# =================================================================== curves
def test_bezier_point_derivative_polynomial() -> None:
    assert g.bezier_point([(0, 0), (1, 2), (3, 3), (4, 0)], "1/3").exact == ("34/27", "14/9")
    assert g.bezier_derivative([(0, 0), (1, 2), (3, 3), (4, 0)], 2) == (("6", "-6"), ("-6", "-24"))
    assert g.bezier_polynomial([(0,), (0,), (1,), (1,)]) == ("-2*t**3 + 3*t**2",)
    f = g.bezier_point([(0.1, 0.2), (1, 2), (3, 3), (4, 0)], 0.3)
    assert f.certificate["relative_error"] < 1e-15


def test_bezier_arc_length_bracket_is_rigorous() -> None:
    import mpmath

    ctrl = [(0, 0), (1, 2), (3, 3), (4, 0)]
    r = g.bezier_arc_length(ctrl)
    mpmath.mp.dps = 40
    ref = mpmath.quad(lambda t: mpmath.sqrt((3 * (1 - t) ** 2 * 1 + 6 * (1 - t) * t * 2 + 3 * t * t * 1) ** 2
                                            + (3 * (1 - t) ** 2 * 2 + 6 * (1 - t) * t * 1 + 3 * t * t * -3) ** 2),
                      [0, 0.5, 1])
    mpmath.mp.dps = 15
    assert r.lower <= float(ref) <= r.upper
    assert r.upper - r.lower <= 1e-7 * r.lower
    assert abs(r.value - float(ref)) < 1e-12


def test_css_easing_extrapolation_and_presets() -> None:
    assert g.css_easing("ease", -0.5).exact == "-3602879701896397/18014398509481984"  # -2 * binary(0.1)
    assert g.css_easing("ease", 1.5).y == 1.0
    assert g.css_easing("cubic-bezier(0, 0.5, 1, 0.5)", -1).y == -0.5
    assert g.css_easing("linear", 0.37).y == pytest.approx(0.37, abs=1e-16)
    with pytest.raises(g.DomainError):
        g.css_cubic_bezier(1.2, 0, 0.5, 1, 0.5)
    ex = g.css_cubic_bezier(0.42, 0, 0.58, 1, 0.3, exact=True)
    assert "CRootOf" in ex.exact or "sqrt" in ex.exact or ex.exact


def test_splines_and_steps() -> None:
    assert g.catmull_rom((0,), (1,), (2,), (3,), "1/2").exact == ("3/2",)
    assert g.catmull_rom((0, 0), (1, 1), (2, 0), (3, 1), 0.0, alpha=0.5).value == pytest.approx((1, 1))
    assert g.catmull_rom((0, 0), (1, 1), (2, 0), (3, 1), 1.0, alpha=1.0).value == pytest.approx((2, 0))
    assert g.hermite((0,), (0,), (1,), (0,), "1/2").exact == ("1/2",)
    assert g.smoothstep(0, 1, "1/4").exact == ("5/32",)
    assert g.smoothstep(0, 1, 2).exact == ("1",) and g.smoothstep(0, 1, -2).exact == ("0",)
    assert g.smootherstep(0, 1, "1/2").exact == ("1/2",)
    with pytest.raises(g.DomainError):
        g.smoothstep(1, 1, 0.5)


# ============================================================== code emission
@pytest.mark.parametrize("formula", sorted(g.FORMULAS))
@pytest.mark.parametrize("lang", ["python", "numpy"])
def test_emitted_python_numpy_match(formula: str, lang: str) -> None:
    r = g.emit(formula, lang, points=48)
    assert r.certificate["status"] == "verified", r.certificate


@pytest.mark.parametrize("formula", sorted(g.FORMULAS))
@pytest.mark.parametrize("lang", ["c", "glsl"])
def test_emitted_c_glsl_compile_and_match(formula: str, lang: str) -> None:
    import shutil

    if not (shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")):
        pytest.skip("no C compiler")
    r = g.emit(formula, lang, points=32)
    assert r.certificate["status"] == "verified", r.certificate


@pytest.mark.parametrize("formula", sorted(g.FORMULAS))
def test_glsl_has_no_pow_and_no_integer_literals(formula: str) -> None:
    import re

    code = g.emit(formula, "glsl", verify=False).code
    body = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("//"))
    assert "pow(" not in body
    # integer literal = digits not part of a float literal, an identifier, or a loop counter
    bad = [m.group(0) for m in re.finditer(r"(?<![\w.])\d+(?![\w.])", body)]
    assert set(bad) <= {"0", "64"}, bad  # the loop 'for (int i = 0; i < 64; i++)' only
    if bad:
        assert "for (int i = 0; i < 64; i++)" in body


def test_glsl_mat3_column_order_is_checked() -> None:
    """Mutation test: emitting mat3 row-by-row (the classic GLSL bug) is caught."""
    import shutil

    if not shutil.which("cc"):
        pytest.skip("no C compiler")
    f = g.FORMULAS["quat_to_mat3"]
    good = g._render(f, "glsl")
    inner = good[good.index("return mat3(") + len("return mat3("): good.rindex(");")]
    parts = [p.strip() for p in _split_args(inner)]
    rowwise = [parts[3 * c + r] for r in range(3) for c in range(3)]  # transpose back
    bad = good.replace(inner, ", ".join(rowwise))
    cert = g._verify_compiled(f, "glsl", bad, g._samples(f, 8, 0), 9)
    assert cert["status"] == "refuted"


def _split_args(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    return out + [cur]


def test_emit_refusals() -> None:
    with pytest.raises(g.DomainError):
        g.emit("nope", "c")
    with pytest.raises(g.DomainError):
        g.emit("smoothstep", "hlsl")


# ================================================================== results
def test_results_serialise_to_json() -> None:
    import json

    objs = [g.quat_to_matrix([1, 0, 0, 0]), g.compare_rotation_expressions("q1*q2", "R2@R1"),
            g.classify_orient2d((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), g.segment_intersection((0, 0), (1, 1), (0, 1), (1, 0)),
            g.ray_triangle((0.25, 0.25, 1), (0, 0, -1), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
            g.ray_sphere((0, 0, -5), (0, 0, 1), (0, 0, 0), 1), g.point_in_polygon((0, 0), [(0, 0), (1, 0), (0, 1)]),
            g.epsilon_witness("ray_triangle", 1e-6), g.bezier_point([(0, 0), (1, 1)], 0.5),
            g.bezier_arc_length([(0, 0), (1, 1)]), g.css_easing("ease", 0.5), g.emit("smoothstep", "c", verify=False)]
    for o in objs:
        d = o.to_dict()
        json.loads(json.dumps(d))
        assert "kind" in d or "location" in d or "hit" in d


# ========================================================== evaluation set
def _runner():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent / "evalsets" / "g1_run_cases.py"
    spec = importlib.util.spec_from_file_location("g1_run_cases", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("split", ["dev", "held"])
def test_evaluation_set_zero_wrong(split: str) -> None:
    r = _runner().run(split)
    assert r["wrong"] == 0 and r["refused"] == 0, r["failures"]
    assert r["total"] >= 55
