"""Run the G1 evaluation set (pycodemath/code/data/G1_cases.json) against pycodemath.code.geometry.

    python tests/evalsets/g1_run_cases.py [dev|held|all]

A case PASSES when the answer matches the independent expected value; it is WRONG
when an answer was given and it does not match; it is a REFUSAL when a typed
pycodemath error was raised instead (never counted as correct).
"""
from __future__ import annotations

import json
import math
import os
import sys
from fractions import Fraction

import sympy as sp

from pycodemath.code import geometry as g
from pycodemath.core.errors import PycodemathError

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "pycodemath", "code", "data", "G1_cases.json")


def _close(m, e, tol):
    return all(abs(a - b) <= tol for ra, rb in zip(m, e) for a, b in zip(ra, rb))


def check(case: dict) -> tuple[bool, str]:
    op, i, e = case["op"], case["input"], case["expected"]
    if op == "euler_to_matrix":
        m = g.euler_to_matrix(i["angles_deg"], i["seq"], frame=i["frame"], degrees=True).value
        return _close(m, e["matrix"], e["tol"]), ""
    if op == "matrix_to_euler_roundtrip":
        r = g.matrix_to_euler(i["matrix"], i["seq"], frame=i["frame"])
        back = g.euler_to_matrix(r.value, i["seq"], frame=i["frame"]).value
        return _close(back, i["matrix"], e["recompose_tol"]), f"gimbal={r.gimbal_lock}"
    if op == "quat_to_matrix":
        conv = g.QuatConvention(order=i["order"], algebra=i["algebra"])
        m = g.quat_to_matrix(i["quat_wxyz"], quat=conv).value
        return _close(m, e["matrix"], e["tol"]), ""
    if op == "matrix_to_quat":
        q = g.matrix_to_quat(i["matrix"]).value
        ex = e["quat_wxyz_up_to_sign"]
        s = 1 if sum(a * b for a, b in zip(q, ex)) >= 0 else -1
        return max(abs(a - s * b) for a, b in zip(q, ex)) <= e["tol"], ""
    if op == "axis_angle_to_matrix":
        m = g.axis_angle_to_matrix(i["axis"], i["angle_deg"], degrees=True).value
        return _close(m, e["matrix"], e["tol"]), ""
    if op == "slerp":
        q = g.slerp(i["q0"], i["q1"], i["t"]).value
        m = g.quat_to_matrix(q, normalize=True).value
        return _close(m, e["matrix"], e["tol"]), ""
    if op == "compare_rotation_expressions":
        c = i["conventions"]
        alg = c.get("algebra", "hamilton")
        qa = g.QuatConvention(order=c.get("order_a", "wxyz"), algebra=c.get("algebra_a", alg))
        qb = g.QuatConvention(order=c.get("order_b", "wxyz"), algebra=c.get("algebra_b", alg))
        mb = g.MatrixConvention(vectors=c.get("vectors_b", "column"))
        r = g.compare_rotation_expressions(i["a"], i["b"], quat=qa, quat_b=qb, matrix_b=mb)
        ok = r.status == e["status"]
        if e["cause"]:
            ok = ok and any(cz.startswith(e["cause"]) for cz in r.causes)
        return ok, f"{r.status} {r.causes}"
    if op in ("orient2d", "orient3d", "incircle"):
        fn = getattr(g, op)
        pts = [i[k] for k in ("a", "b", "c", "d") if k in i]
        s = fn(*pts)
        return s == e["sign"], f"sign={s}"
    if op == "segment_intersection":
        r = g.segment_intersection(i["p1"], i["p2"], i["q1"], i["q2"])
        ok = r.kind == e["kind"]
        if "point" in e:
            ok = ok and r.point == tuple(e["point"])
        if "point_float" in e:
            ok = ok and r.approx is not None and max(abs(a - b) for a, b in zip(r.approx, e["point_float"])) < 1e-15
        if "segment" in e:
            ok = ok and r.segment == tuple(tuple(p) for p in e["segment"])
        return ok, r.kind
    if op == "ray_triangle":
        r = g.ray_triangle(i["origin"], i["direction"], *i["triangle"])
        ok = r.hit == e["hit"]
        for k in ("t", "u", "v"):
            if k in e:
                ok = ok and getattr(r, k) == e[k]
        for k in ("on_edge", "on_vertex"):
            if k in e:
                ok = ok and getattr(r, k) == e[k]
        return ok, f"hit={r.hit} t={r.t} {r.reason}"
    if op == "ray_sphere":
        r = g.ray_sphere(i["origin"], i["direction"], i["center"], i["radius"])
        ok = r.kind == e["kind"]
        if "t" in e:
            ok = ok and len(r.t) == len(e["t"]) and all(sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0
                                                         for a, b in zip(r.t, e["t"]))
        return ok, f"{r.kind} {r.t}"
    if op == "point_in_polygon":
        r = g.point_in_polygon(i["point"], i["polygon"])
        ok = r.location == e["boundary_rule"]
        if e["half_open"] is not None:
            ok = ok and r.half_open == e["half_open"]
        return ok, f"{r.location} {r.half_open}"
    if op == "css_easing":
        r = g.css_cubic_bezier(i["x1"], i["y1"], i["x2"], i["y2"], i["x"])
        ref = Fraction(e["y"])
        # the reference is a 60-digit bisection: its own error is < 1e-40, except where
        # B_x is stationary (cube-root conditioning, ~1e-20) — allow 1e-18
        ok = abs(r.y - float(ref)) <= e["tol"] and (Fraction(r.enclosure[0]) - Fraction(1, 10**18)
                                                     <= ref <= Fraction(r.enclosure[1]) + Fraction(1, 10**18))
        return ok, f"y={r.y!r}"
    if op == "bezier_point":
        r = g.bezier_point(i["control"], i["t"])
        return r.exact == tuple(e["point"]), str(r.exact)
    if op == "bezier_arc_length":
        r = g.bezier_arc_length(i["control"])
        ref = float(e["length"])
        return abs(r.value - ref) <= e["tol"] * ref and r.lower <= ref <= r.upper, f"{r.lower} {r.value} {r.upper}"
    raise KeyError(op)


def run(split: str = "all") -> dict:
    cases = json.load(open(DATA))["cases"]
    res = {"pass": 0, "wrong": 0, "refused": 0, "failures": []}
    for c in cases:
        if split != "all" and c["split"] != split:
            continue
        try:
            ok, info = check(c)
        except PycodemathError as exc:
            res["refused"] += 1
            res["failures"].append((c["id"], c["op"], f"refused: {type(exc).__name__}: {exc}"))
            continue
        if ok:
            res["pass"] += 1
        else:
            res["wrong"] += 1
            res["failures"].append((c["id"], c["op"], info))
    res["total"] = res["pass"] + res["wrong"] + res["refused"]
    return res


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    r = run(split)
    print(f"{split}: {r['pass']}/{r['total']} pass, {r['wrong']} wrong, {r['refused']} refused")
    for f in r["failures"]:
        print("  ", f)
