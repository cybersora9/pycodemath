"""C3 — value ranges and overflow / underflow / NaN / precision-loss with witnesses.

The guards first: every ``confirmed`` event in the whole evaluation set is
re-executed HERE, with a reproducer that shares no code with the analyzer (plain
NumPy for the dtype result, ``fractions``/``mpmath`` for the exact one).
"""

from __future__ import annotations

import json
import math
import warnings
from fractions import Fraction
from pathlib import Path

import mpmath
import numpy as np
import pytest

from pycodemath.code import ranges as R
from pycodemath.core.errors import DomainError, ParseError

DATA = Path(R.__file__).parent / "data" / "C3_cases.json"
CASES = json.loads(DATA.read_text(encoding="utf-8"))["cases"]

try:  # bfloat16 reproduction needs the real type; the analyzer itself does not
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover - exercised only without ml_dtypes
    _BF16 = None


# --------------------------------------------------------------------------
# Independent reproducer (no import from pycodemath.code.ranges below this line
# except the report objects themselves)
# --------------------------------------------------------------------------

def _np_dtype(name: str):
    if name == "bfloat16":
        if _BF16 is None:
            pytest.skip("ml_dtypes not installed")
        return np.dtype(_BF16)
    return np.dtype(name)


_NS = {
    "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10,
    "log1p": np.log1p, "expm1": np.expm1, "sqrt": np.sqrt, "hypot": np.hypot,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
    "sinh": np.sinh, "cosh": np.cosh, "arctan": np.arctan, "atan": np.arctan,
    "abs": np.abs, "max": np.maximum, "min": np.minimum,
}


def _np_eval(src: str, witness: dict, dtype: np.dtype, kinds: dict):
    # 1-element ARRAYS, the program the analysis models: on NumPy 1.x a 0-d array or a
    # NumPy scalar promotes every Python literal to float64/int64 (value-based casting)
    env = dict(_NS)
    for name, text in witness.items():
        if kinds.get(name) == "int" and dtype.kind == "f":
            env[name] = np.array([int(text)]).astype(dtype)  # the cast is the program's
        elif dtype.kind in "iu":
            env[name] = np.array([int(text)], dtype=dtype)
        else:
            env[name] = np.array([float(text)], dtype=dtype)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        out = np.asarray(eval(src, {"__builtins__": {}}, env))  # noqa: S307 - test-only, own strings
        if out.ndim:
            # the reproducer must see the SAME type the analysis claims; a promoted
            # result here would mean the report described a program that does not run
            assert out.dtype == dtype, (src, out.dtype, dtype)
        return out.astype(dtype).reshape(())


def _exact(v) -> Fraction:
    return Fraction(int(v)) if np.asarray(v).dtype.kind in "iu" else Fraction(float(v))


def _exact_op(op: str, xs: list[Fraction]):
    """The real-number result of ``op`` on the observed operands (mpf or Fraction)."""
    a = xs[0] if xs else None
    b = xs[1] if len(xs) > 1 else None
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return None if b == 0 else a / b
    if op == "floordiv":
        return None if b == 0 else Fraction(math.floor(a / b))
    if op == "neg":
        return -a
    if op == "abs":
        return abs(a)
    if op == "pow":
        k = xs[2]
        if isinstance(k, int):
            return None if (a == 0 and k < 0) else a ** k
        if a < 0:
            return None
        with mpmath.workprec(300):
            return (mpmath.mpf(a.numerator) / a.denominator) ** mpmath.mpf(k)
    if op in ("exp", "log", "sqrt", "tanh", "log1p", "expm1"):
        f = getattr(mpmath, op)
        if op in ("log", "sqrt") and a <= 0:
            return None
        if op == "log1p" and a <= -1:
            return None
        with mpmath.workprec(300):
            return f(mpmath.mpf(a.numerator) / a.denominator)
    raise AssertionError(f"reproducer has no exact rule for {op}")


def _reproduce(report, ev, case_kinds) -> None:
    """Re-execute one confirmed event and assert it really happens."""
    dt = _np_dtype(report.dtype)
    assert ev.witness, f"{ev.kind} at {ev.node_text}: confirmed without a witness"
    # the witness must lie inside the declared ranges
    for name, text in ev.witness.items():
        lo, hi = report.ranges[name][0], report.ranges[name][1]
        v = Fraction(text) if dt.kind != "f" or "." in text or "e" in text else Fraction(int(text))
        assert Fraction(lo) <= v <= Fraction(hi), (name, text, lo, hi)
    if ev.op == "var":  # cast of a declared-integer input
        name = ev.node_text
        raw = int(ev.witness[name])
        got = np.array(raw).astype(dt)
        assert ev.kind == "integer_precision_loss"
        assert Fraction(float(got)) != raw, (raw, got)
        return
    if ev.op == "const":  # a literal the type cannot hold: the same at every input
        import ast as _ast

        lit = _ast.literal_eval(ev.node_text)
        with np.errstate(all="ignore"):
            got = np.array(lit).astype(dt)
        if ev.kind == "overflow":
            assert np.isinf(got)
        else:
            assert ev.kind == "integer_precision_loss" and Fraction(float(got)) != lit
        return
    operands = [_np_eval(s, ev.witness, dt, case_kinds) for s in ev.operand_src]
    extra = []
    if ev.op == "pow":
        import ast as _ast

        tree = _ast.parse(ev.node_text, mode="eval").body
        k_node = tree.right if isinstance(tree, _ast.BinOp) else tree.args[1]
        extra = [None, _ast.literal_eval(k_node)]
    whole = _np_eval(ev.node_text, ev.witness, dt, case_kinds)
    finite_in = all(np.isfinite(o).all() for o in operands) if dt.kind == "f" else True
    if ev.kind == "overflow":
        if dt.kind in "iu":
            exact = _exact_op(ev.op, [Fraction(int(o)) for o in operands] + extra)
            info = np.iinfo(dt)
            assert exact is not None and not (info.min <= exact <= info.max), (exact, whole)
            assert Fraction(int(whole)) != exact
        else:
            assert finite_in and np.isinf(whole), (operands, whole)
    elif ev.kind == "nan_possible":
        assert not any(np.isnan(o).any() for o in operands) and np.isnan(whole), (operands, whole)
    elif ev.kind == "division_by_zero_possible":
        if dt.kind in "iu":
            assert int(operands[1]) == 0
        elif ev.op in ("log", "log2", "log10"):
            assert float(operands[0]) == 0.0 and np.isinf(whole)
        elif ev.op == "log1p":
            assert float(operands[0]) == -1.0 and np.isinf(whole)
        elif ev.op == "pow":
            assert float(operands[0]) == 0.0
        else:
            assert float(operands[1]) == 0.0
    elif ev.kind == "underflow_to_zero":
        assert float(whole) == 0.0
        exact = _exact_op(ev.op, [_exact(o) for o in operands] + extra)
        assert exact is not None and exact != 0, (operands, exact)
    elif ev.kind == "subnormal":
        tiny = float(np.finfo(dt).tiny) if dt != np.dtype(_BF16 or "f4") or _BF16 is None \
            else float(ml_dtypes.finfo(_BF16).tiny)
        assert 0 < abs(float(whole)) < tiny, whole
    elif ev.kind == "integer_precision_loss":
        exact = _exact_op(ev.op, [_exact(o) for o in operands] + extra)
        assert exact is not None and Fraction(exact).denominator == 1
        assert Fraction(float(whole)) != exact, (operands, whole, exact)
    else:  # pragma: no cover
        raise AssertionError(ev.kind)


def _kinds(case) -> dict:
    return {k: (v[2] if len(v) > 2 else None) for k, v in case["ranges"].items()}


def _ranges(case) -> dict:
    return {k: tuple(v) for k, v in case["ranges"].items()}


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_overflow_warning_has_a_witness(case):
    """Every confirmed event in the whole set is reproduced by NumPy execution."""
    if case["dtype"] == "bfloat16" and _BF16 is None:
        pytest.skip("ml_dtypes not installed")
    report = R.analyze(case["expr"], _ranges(case), case["dtype"], budget=60)
    for ev in report.events:
        if ev.status == "confirmed":
            _reproduce(report, ev, _kinds(case))
        else:
            assert ev.status in ("possible", "proved_safe")
            if ev.status == "proved_safe":
                assert ev.witness is None


def test_binary_search_midpoint_int32_confirmed():
    report = R.analyze("(lo + hi) // 2", {"lo": (0, 2**31 - 1), "hi": (0, 2**31 - 1)}, "int32")
    over = [e for e in report.events if e.kind == "overflow" and e.status == "confirmed"]
    assert over, report.events
    ev = over[0]
    assert ev.node_text == "lo + hi"
    lo, hi = int(ev.witness["lo"]), int(ev.witness["hi"])
    assert lo + hi > 2**31 - 1
    wrapped = np.array([lo], np.int32) + np.array([hi], np.int32)
    assert int(wrapped[0]) < 0  # the famous negative midpoint
    assert report.status == "problems"
    rewrites = [a.rewrite for a in report.advice]
    assert "lo + (hi - lo) // 2" in rewrites
    fix = next(a for a in report.advice if a.rewrite == "lo + (hi - lo) // 2")
    assert fix.equivalence == "verified" and "overflow" in fix.removes


def test_safe_expression_is_proved_safe():
    report = R.analyze("x * x", {"x": (1, 200)}, "float16")
    assert report.status == "safe"
    assert report.events and all(e.status == "proved_safe" for e in report.events)
    sq = next(n for n in report.nodes if n.text == "x * x")
    assert (Fraction(sq.lo), Fraction(sq.hi)) == (1, 40000)
    # around zero x*x is still overflow- and NaN-free; the tiny squares that flush to
    # zero are REAL (witnessed), not a false alarm
    wide = R.analyze("x * x", {"x": (-200, 200)}, "float16")
    assert wide.class_status("overflow") == "proved_safe"
    assert wide.class_status("nan_possible") == "proved_safe"
    assert wide.class_status("underflow_to_zero") == "confirmed"


# --------------------------------------------------------------------------
# Adversarial round (C3 session): inputs built to provoke a false alarm, a crash
# or a wrong proof. Each row: expression, ranges, dtype, {class: required status}
# where "!confirmed" means "anything but confirmed" (possible is acceptable).
# --------------------------------------------------------------------------

F32MAX = "340282346638528859811704183484516925440"
ADVERSARIAL = [
    ("x - x", {}, "float32", {"overflow": "proved_safe", "nan_possible": "proved_safe"}),
    ("x / x", {"x": ("1e-30", "1e30")}, "float32", {"overflow": "proved_safe", "nan_possible": "proved_safe"}),
    ("x * 0", {}, "float32", {"overflow": "proved_safe", "nan_possible": "proved_safe"}),
    ("0 * exp(x)", {"x": (0, 100)}, "float32", {"nan_possible": "confirmed"}),
    ("abs(x) - x", {}, "int8", {"overflow": "confirmed"}),
    ("-x", {}, "int32", {"overflow": "confirmed"}),
    ("-x", {"x": (0, 0)}, "uint8", {"overflow": "proved_safe"}),
    ("-x", {"x": (1, 1)}, "uint8", {"overflow": "confirmed"}),
    ("x * x", {"x": (5, 5)}, "float32", {"overflow": "proved_safe"}),
    ("x * 2", {"x": ("1e-50", "1e-40")}, "float32", {"subnormal": "confirmed"}),
    ("x + y", {}, "float64", {"overflow": "confirmed"}),
    ("x ** 0.5", {"x": (-1, 1)}, "float32", {"nan_possible": "confirmed"}),
    ("x ** -1", {"x": (-1, 1)}, "float32", {"division_by_zero_possible": "confirmed", "overflow": "confirmed"}),
    ("x ** 3", {"x": (-40, 40)}, "int16", {"overflow": "confirmed"}),
    ("x ** 2", {"x": (-11, 11)}, "int8", {"overflow": "proved_safe"}),
    ("x // y", {"y": (1, 127)}, "int8", {"overflow": "proved_safe", "division_by_zero_possible": "proved_safe"}),
    ("x % y", {"y": (-5, -1)}, "int32", {"division_by_zero_possible": "proved_safe"}),
    ("log1p(x)", {"x": (-1, 0)}, "float32", {"division_by_zero_possible": "confirmed"}),
    ("log1p(x)", {"x": ("-0.99", 1)}, "float32", {"division_by_zero_possible": "proved_safe", "nan_possible": "proved_safe"}),
    ("sin(x)", {}, "float32", {"nan_possible": "proved_safe"}),
    ("cos(x) / cos(x)", {"x": (1, 2)}, "float32", {"division_by_zero_possible": "!confirmed"}),
    ("1 / (x - 0.1)", {"x": (0, 1)}, "float32", {"division_by_zero_possible": "confirmed"}),
    ("1 / (x - 0.1)", {"x": (0, 1)}, "float64", {"division_by_zero_possible": "confirmed"}),
    ("x * y", {}, "int64", {"overflow": "confirmed"}),
    ("x + 1", {"x": (0, 2**64 - 1)}, "uint64", {"overflow": "confirmed"}),
    ("x * x", {"x": (0, 2**32 - 1)}, "uint64", {"overflow": "proved_safe"}),
    ("x * x", {"x": (0, 2**32)}, "uint64", {"overflow": "confirmed"}),
    ("exp(exp(x))", {"x": (-10, 5)}, "float32", {"overflow": "confirmed"}),
    ("sqrt(x) * sqrt(x)", {"x": (0, F32MAX)}, "float32", {"overflow": "!confirmed", "nan_possible": "proved_safe"}),
    ("hypot(x, y)", {}, "float32", {"overflow": "confirmed"}),
    ("tanh(x) * 1e38", {}, "float32", {"overflow": "proved_safe"}),
    ("x - y", {"x": (-65504, 65504), "y": (-65504, 65504)}, "float16", {"overflow": "confirmed"}),
    ("x * x - x * x", {}, "float32", {"overflow": "confirmed", "nan_possible": "confirmed"}),
    ("(x + 1) - x", {"x": ("-1e38", "1e38")}, "float32", {"overflow": "proved_safe"}),
    ("np.maximum(x, y) - np.minimum(x, y)", {"x": (0, 2**31 - 1), "y": (0, 2**31 - 1)}, "int32",
     {"overflow": "proved_safe"}),
    ("max(x, y) - min(x, y)", {}, "int32", {"overflow": "confirmed"}),
    ("x * x + 1", {"x": (-46340, 46340)}, "int32", {"overflow": "proved_safe"}),
    ("x * x + 1", {"x": (-46341, 46341)}, "int32", {"overflow": "confirmed"}),
    ("exp", {"exp": (0, 1)}, "float32", {}),  # a variable that happens to be named exp
    ("i + j", {"i": (0, 2**24, "int"), "j": (0, 0, "int")}, "float32", {"integer_precision_loss": "proved_safe"}),
    ("i * 2", {"i": (0, 2**24, "int")}, "float32", {"integer_precision_loss": "proved_safe"}),
    ("i * 3", {"i": (0, 2**24, "int")}, "float32", {"integer_precision_loss": "confirmed"}),
]


@pytest.mark.parametrize("expr,rng,dtype,expect", ADVERSARIAL, ids=[f"{a[0]}@{a[2]}" for a in ADVERSARIAL])
def test_adversarial(expr, rng, dtype, expect):
    report = R.analyze(expr, rng, dtype, budget=30)
    kinds = {k: (v[2] if len(v) > 2 else None) for k, v in rng.items()}
    for ev in report.confirmed:
        _reproduce(report, ev, kinds)
    for k, want in expect.items():
        got = report.class_status(k)
        if want == "!confirmed":
            assert got != "confirmed", (k, got)
        else:
            assert got == want, (k, got, [e for e in report.events_of(k)])


@pytest.mark.parametrize("expr,rng,dtype,exc", [
    ("foo(x)", {}, "float32", R.RangeUnsupportedError),
    ("x ^ 2", {}, "float32", ParseError),
    ("x / 2", {}, "int32", DomainError),
    ("x + 300", {}, "int8", Exception),
    ("x if x > 0 else -x", {}, "float32", R.RangeUnsupportedError),
    ("__import__('os').system('echo pwned')", {}, "float32", R.RangeUnsupportedError),
    ("x", {"x": ("nan", 1)}, "float32", Exception),
    ("x", {"x": (0, float("inf"))}, "float32", Exception),
    ("x", {"x": (2, 1)}, "float32", Exception),
    ("x", {"y": (0, 1)}, "float32", Exception),
    ("x", {"x": ("0.1", "0.1")}, "float16", Exception),
    ("x", {"x": (0, 1e39)}, "float32", Exception),
    ("x", {"x": (0, 300)}, "uint8", Exception),
    ("x", {}, "complex64", Exception),
    ("x(", {}, "float32", Exception),
    ("2 ** 70 + x", {}, "int64", Exception),
    ("x ** y", {}, "float32", R.RangeUnsupportedError),
    ("exp(x)", {}, "int32", R.RangeUnsupportedError),
])
def test_refusals_are_typed(expr, rng, dtype, exc):
    from pycodemath.core.errors import PycodemathError

    with pytest.raises(PycodemathError) as info:
        R.analyze(expr, rng, dtype)
    assert isinstance(info.value, exc)


# --------------------------------------------------------------------------
# The machinery the guarantees rest on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["float16", "float32", "float64"])
def test_exact_rounding_matches_numpy_casts(name):
    """``_round`` (exact, Fraction-based) is the independent check of every float event."""
    fmt = R._FORMATS[name]
    rng = np.random.default_rng(1)
    xs = np.concatenate([
        rng.standard_normal(3000) * 10.0 ** rng.integers(-320, 308, 3000),
        [0.1, 1 / 3, 65519.99, 65520.0, 65504.0, 6e-8, 2.98e-8, 2.9802322387695312e-08],
        [float(np.finfo(np.float32).max) * (1 + 2.0**-25)],
    ])
    with np.errstate(all="ignore"):
        cast = xs.astype(name).astype(np.float64)
    for x, c in zip(xs.tolist(), cast.tolist()):
        if not math.isfinite(x):
            continue
        r = R._round(Fraction(x), fmt, "n")
        if isinstance(r, float):
            assert math.isinf(c) and (r > 0) == (c > 0), x
        else:
            assert r == Fraction(c), (x, r, c)
        lo, hi = R._round(Fraction(x), fmt, "d"), R._round(Fraction(x), fmt, "u")
        assert lo <= Fraction(x) <= hi


def test_ordinals_are_a_bijection_on_every_float16():
    fmt = R._FORMATS["float16"]
    allv = np.arange(0, 0x7C00, dtype=np.uint16).view(np.float16).astype(np.float64)  # finite >= 0
    ords = R._f64_to_ords(allv, fmt)
    assert (ords == np.arange(len(allv))).all()
    assert (R._ords_to_f64(ords, fmt) == allv).all()
    assert (R._ords_to_f64(-ords, fmt) == -allv).all()
    for o in (0, 1, 1023, 1024, 0x7BFF):
        assert R._ord_of(R._val_of(o, fmt), fmt) == o


@pytest.mark.skipif(_BF16 is None, reason="ml_dtypes not installed")
def test_bfloat16_emulation_matches_ml_dtypes():
    """Without ml_dtypes the analysis runs on the emulation; it must agree bit for bit."""
    allbits = np.arange(0, 1 << 16, dtype=np.uint32) << 16
    a32 = allbits.view(np.float32)
    with np.errstate(all="ignore"):  # the NaN bit patterns are part of the sweep
        real = a32.astype(_BF16).astype(np.float32)
    emu = R._bf16_round(a32)
    same = (real == emu) | (np.isnan(real) & np.isnan(emu))
    assert same.all()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200000).astype(np.float32) * np.float32(1e3)
    y = rng.standard_normal(200000).astype(np.float32) * np.float32(1e-2)
    em = R._Machine(R._FORMATS["bfloat16"], emulate_bf16=True)
    nat = R._Machine(R._FORMATS["bfloat16"])
    xe, ye = em.cast(x), em.cast(y)
    xn, yn = nat.cast(x), nat.cast(y)
    with np.errstate(all="ignore"):
        for op in ("add", "sub", "mul", "div"):
            e = em.apply(op, [xe, ye], None)
            n = nat.apply(op, [xn, yn], None).astype(np.float32)
            assert ((e == n) | (np.isnan(e) & np.isnan(n))).all(), op
        assert (em.apply("sqrt", [np.abs(xe)], None) == nat.apply("sqrt", [np.abs(xn)], None).astype(np.float32)).all()


def test_bfloat16_analysis_without_ml_dtypes(monkeypatch):
    monkeypatch.setattr(R, "_BF16_DTYPE", None)
    r = R.analyze("i + 1", {"i": (0, 1000, "int")}, "bfloat16")
    assert r.machine == "bfloat16-emulation"
    assert r.class_status("integer_precision_loss") == "confirmed"
    ev = r.events_of("integer_precision_loss", "confirmed")[0]
    assert int(ev.witness["i"]) > 256


# --------------------------------------------------------------------------
# API behaviour
# --------------------------------------------------------------------------

def test_max_safe_input_exp_float32_is_pinned_to_one_ulp():
    r = R.max_safe_input("exp(x)", "x", "float32")
    assert r.exact and r.status == "exact"
    x = np.float32(float(Fraction(r.safe_up_to)))
    with np.errstate(all="ignore"):
        assert np.isfinite(np.exp(x))
        assert np.isinf(np.exp(np.nextafter(x, np.float32(np.inf))))
    assert float(Fraction(r.first_failure)) == float(np.nextafter(x, np.float32(np.inf)))
    assert abs(r.value - 88.7228) < 1e-3
    assert json.dumps(r.to_dict())


@pytest.mark.parametrize("expr,dtype,limit", [
    ("x * x", "int32", 46340), ("x * x", "float16", 255.875), ("exp(x)", "float16", 11.0859375),
])
def test_max_safe_input_known_limits(expr, dtype, limit):
    r = R.max_safe_input(expr, "x", dtype)
    assert r.exact and float(Fraction(r.safe_up_to)) == limit


def test_max_safe_input_with_other_variables():
    r = R.max_safe_input("a + b", "a", "int16", {"b": (0, 100)})
    assert r.exact and int(r.safe_up_to) == 32767 - 100


def test_report_to_dict_is_json_and_exact():
    r = R.analyze("(lo + hi) // 2", {"lo": (0, 2**31 - 1), "hi": (0, 2**31 - 1)}, "int32")
    d = json.loads(json.dumps(r.to_dict()))
    assert d["kind"] == "range" and d["status"] == "problems" and d["dtype"] == "int32"
    add = next(n for n in d["nodes"] if n["text"] == "lo + hi")
    assert (add["lo"], add["hi"]) == ("-2147483648", "2147483647")  # wraps: anything
    ev = next(e for e in d["events"] if e["status"] == "confirmed")
    assert int(ev["exact"]) > 2**31 - 1 and int(ev["observed"]) < 0
    assert str(r).startswith("(lo + hi) // 2  [int32]  -> problems")


def test_node_ranges_are_exact_and_cover_every_intermediate():
    r = R.analyze("exp(x) / (exp(x) + 1)", {"x": (-1, 1)}, "float64")
    assert r.status == "safe"
    texts = [n.text for n in r.nodes]
    assert texts == ["x", "exp(x)", "1", "exp(x) + 1", "exp(x) / (exp(x) + 1)"]
    sig = next(n for n in r.nodes if n.text == "exp(x) / (exp(x) + 1)")
    assert 0.26 < float(Fraction(sig.lo)) < 0.27 and 0.73 < float(Fraction(sig.hi)) < 0.74


def test_expression_is_read_as_written_not_simplified():
    # SymPy would turn (a + b)/2 into a/2 + b/2 and x - x into 0
    r = R.analyze("(a + b) // 2", {"a": (0, 200), "b": (0, 200)}, "uint8")
    assert r.events_of("overflow", "confirmed")[0].node_text == "a + b"
    assert R.analyze("x - x", {}, "float32").class_status("overflow") == "proved_safe"


def test_literal_folding_follows_python():
    # 20 * 365 * 86400 is folded by Python (630720000 fits int32), only now + it wraps
    r = R.analyze("now + 20 * 365 * 86400", {"now": (1_700_000_000, 1_800_000_000)}, "int32")
    assert [n.text for n in r.nodes] == ["now", "20 * 365 * 86400", "now + 20 * 365 * 86400"]
    # ... whereas now * 20 * 365 multiplies in int32 step by step
    r2 = R.analyze("now * 20 * 365", {"now": (0, 1000)}, "int32")
    assert len(r2.nodes) == 5


def test_function_and_sympy_inputs():
    def mid(lo, hi):
        """Binary-search midpoint."""
        return (lo + hi) // 2

    r = R.analyze(mid, {"lo": (0, 2**31 - 1), "hi": (0, 2**31 - 1)}, "int32")
    assert r.class_status("overflow") == "confirmed"
    r2 = R.analyze(lambda x: np.exp(x), {"x": (0, 100)}, "float32")
    assert r2.class_status("overflow") == "confirmed"
    from pycodemath import E

    r3 = R.analyze(E("x**2"), {"x": (-300, 300)}, "float16")
    assert r3.class_status("overflow") == "confirmed"

    def loop(n):
        s = 0
        for k in range(n):
            s += k
        return s

    with pytest.raises(R.RangeUnsupportedError):
        R.analyze(loop, {"n": (0, 10)}, "int32")


def test_advice_is_verified_and_removes_the_event():
    r = R.analyze("sqrt(dx * dx + dy * dy)", {"dx": ("-1e20", "1e20"), "dy": ("-1e20", "1e20")}, "float32")
    hyp = next(a for a in r.advice if a.rewrite == "hypot(dx, dy)")
    assert hyp.dtype == "float32" and "overflow" in hyp.removes and hyp.equivalence == "verified"
    again = R.analyze(hyp.rewrite, {"dx": ("-1e20", "1e20"), "dy": ("-1e20", "1e20")}, "float32")
    assert again.class_status("overflow") == "proved_safe"

    soft = R.analyze("exp(x) / (exp(x) + exp(y))", {"x": (-20, 20), "y": (-20, 20)}, "float16")
    shifted = [a for a in soft.advice if "max(x, y)" in a.rewrite]
    assert shifted and {"overflow", "nan_possible"} <= set(shifted[0].removes)
    assert shifted[0].method.startswith("check_equal")

    ts = R.analyze("t + 1", {"t": (1_600_000_000_000, 1_800_000_000_000, "int")}, "float32")
    assert [(a.rewrite, a.dtype) for a in ts.advice] == [("t + 1", "float64")]


def test_no_advice_when_nothing_proves_it():
    # no wider signed type than int64, and no equivalent rewrite is known
    r = R.analyze("n * (n - 1) * (n - 2) // 6", {"n": (0, 3_000_000)}, "int64")
    assert r.class_status("overflow") == "confirmed" and r.advice == ()
    # uint32 a - b wraps for b > a; uint64 wraps too, so no "wider type" advice
    r2 = R.analyze("a - b", {"a": (0, 100), "b": (0, 100)}, "uint32")
    assert all(a.dtype != "uint64" for a in r2.advice)


def test_budget_exhaustion_is_reported_never_guessed():
    r = R.analyze("cos(x) / cos(x)", {"x": (1, 2)}, "float32", budget=0.05, advice=False)
    assert r.class_status("division_by_zero_possible") in ("possible", "proved_safe")
    for e in r.events:
        if e.status == "confirmed":
            _reproduce(r, e, {})
    with pytest.raises(DomainError):
        R.analyze("x", {}, "float32", budget=0)


def test_analysis_is_deterministic():
    # I2: a fixed, generous budget — under ``-n 4`` load the default one ran out in
    # one of the two runs once (30.10), and a run cut by the clock answers differently;
    # determinism is a property of the analysis, not of the machine's load
    args = ("log(exp(x) + exp(y))", {"x": (-1000, 1000), "y": (-1000, 1000)}, "float32")
    a, b = R.analyze(*args, budget=600.0), R.analyze(*args, budget=600.0)
    strip = lambda d: {k: v for k, v in d.items() if k != "elapsed"}  # noqa: E731
    assert strip(a.to_dict()) == strip(b.to_dict())


def test_event_invariant_witness_iff_confirmed():
    with pytest.raises(DomainError):
        R.Event("overflow", "confirmed", 0, "x", "add", (), "m", witness=None)
    with pytest.raises(DomainError):
        R.Event("overflow", "possible", 0, "x", "add", (), "m", witness={"x": "1"})


def test_without_z3_integer_proofs_still_close(monkeypatch):
    r = R.analyze("lo + (hi - lo) // 2", {"lo": (0, 2**31 - 1), "hi": (0, 2**31 - 1)}, "int32",
                  use_z3=False)
    assert r.class_status("overflow") == "proved_safe"
    assert any("bisection" in e.method for e in r.events_of("overflow"))


def test_z3_proof_when_available():
    pytest.importorskip("z3")
    r = R.analyze("lo + (hi - lo) // 2", {"lo": (0, 2**31 - 1), "hi": (0, 2**31 - 1)}, "int32")
    assert r.class_status("overflow") == "proved_safe"
    assert any(e.method.startswith("smt") for e in r.events_of("overflow"))


def test_literal_that_changes_the_type_is_refused_not_analysed():
    """Reviewer's catch: NumPy 1.x value-based casting turns float32 * 1e39 into float64,
    so a 'confirmed' overflow of the literal would be a false alarm there."""
    from pycodemath.core.errors import PycodemathError

    for expr, dtype in [("x * 1e39", "float32"), ("x * 300", "float16"), ("x * 1e38", "float16"),
                        ("x * 2.5", "bfloat16"), ("x + 300", "int8"), ("x + 16777217", "float32")]:
        if dtype == "bfloat16" and _BF16 is None:
            continue
        dt = _np_dtype(dtype)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                same = (np.ones(1, dt) * (eval(expr.split()[-1]))).dtype == dt
            except Exception:
                same = False
        if same:  # NumPy 2: the literal is weak and the array keeps its type
            r = R.analyze(expr, {"x": (0, 1)}, dtype)
            assert r.confirmed  # each of these literals misbehaves in the type
            for ev in r.confirmed:
                _reproduce(r, ev, {})
        else:
            with pytest.raises(PycodemathError):
                R.analyze(expr, {"x": (0, 1)}, dtype)


@pytest.mark.skipif(int(np.__version__.split(".")[0]) >= 2, reason="NumPy 1.x promotion only")
def test_numpy1_value_based_casting_is_refused():
    with pytest.raises(R.RangeUnsupportedError, match="float64"):
        R.analyze("x * 1e39", {"x": (0, 1)}, "float32")


@pytest.mark.skipif(int(np.__version__.split(".")[0]) < 2, reason="NumPy 2 (NEP 50) only")
def test_numpy2_weak_literal_overflow_is_real():
    r = R.analyze("x * 1e39", {"x": (0, 1)}, "float32")
    assert r.class_status("overflow") == "confirmed"
    for ev in r.confirmed:
        _reproduce(r, ev, {})


def test_bfloat16_float_literal_is_refused_without_ml_dtypes_too(monkeypatch):
    monkeypatch.setattr(R, "_BF16_DTYPE", None)
    with pytest.raises(R.RangeUnsupportedError, match="float32"):
        R.analyze("x * 2.5", {"x": (0, 1)}, "bfloat16")
    assert R.analyze("x + 1", {"x": (0, 1)}, "bfloat16").class_status("overflow") == "proved_safe"


def test_constant_only_call_is_refused():
    with pytest.raises(R.RangeUnsupportedError, match="no variable"):
        R.analyze("x * exp(2.0)", {"x": (0, 1)}, "float32")
