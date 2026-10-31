"""Code printers for G1 formulas: Python, NumPy, C and GLSL from one expression.

Why not SymPy's own printers: measured on SymPy 1.14, ``glsl_code`` prints
``2*pow(t, 3.0)`` — an INTEGER literal (a compile error in GLSL ES 1.00, which
has no implicit int->float conversion) and ``pow`` with a base that may be
negative (undefined in GLSL: ``pow(x, y)`` is specified only for ``x >= 0``) —
and turns ``Min``/``Max`` into nested ternaries. The formulas here only need
``+ - * /``, integer powers, ``sqrt``, ``min``/``max``; this printer handles
exactly that subset, the same way for all four languages:

* every numeric literal is a floating-point literal (``3.0``, ``(1.0/3.0)``);
* integer powers become repeated products (``t*t*t``), never ``pow``;
* ``Min``/``Max`` become ``fmin``/``fmax`` (C), ``min``/``max`` (GLSL, Python),
  ``numpy.minimum``/``numpy.maximum`` (NumPy, elementwise on arrays).

Also here: the C shim that compiles emitted GLSL as C (``GLSL_AS_C_SHIM``) for
the test-suite, implementing the GLSL constructor rules that matter — notably
that ``mat3(a0, ..., a8)`` fills COLUMNS first (GLSL 4.60 spec §5.4.2) — and the
hand-written CSS easing solver in the four languages.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

__all__ = ["print_expr", "render", "GLSL_AS_C_SHIM", "EASING_TEMPLATES"]


def _lit(v: Any) -> str:
    if isinstance(v, sp.Integer) or (isinstance(v, int) and not isinstance(v, bool)):
        return f"{int(v)}.0"
    if isinstance(v, sp.Rational):
        return f"({int(v.p)}.0/{int(v.q)}.0)"
    if isinstance(v, sp.Float):
        return repr(float(v))
    raise TypeError(v)


_FUNCS = {
    "python": {"min": "min", "max": "max", "sqrt": "math.sqrt"},
    "numpy": {"min": "numpy.minimum", "max": "numpy.maximum", "sqrt": "numpy.sqrt"},
    "c": {"min": "fmin", "max": "fmax", "sqrt": "sqrt"},
    "glsl": {"min": "min", "max": "max", "sqrt": "sqrt"},
}


def print_expr(expr: Any, lang: str, names: "dict[sp.Symbol, str] | None" = None) -> str:
    """Print a SymPy expression of the supported subset in ``lang``."""
    names = names or {}
    fn = _FUNCS[lang]

    def atom(e: Any) -> bool:
        return isinstance(e, (sp.Symbol, sp.Integer)) or (isinstance(e, sp.Rational) and e.q == 1)

    def p(e: Any) -> str:
        if isinstance(e, sp.Symbol):
            return names.get(e, e.name)
        if isinstance(e, sp.Number):
            if e < 0:
                return f"(-{_lit(-e)})"
            return _lit(e)
        if isinstance(e, sp.Add):
            terms = sorted(e.args, key=sp.default_sort_key)
            out = ""
            for i, term in enumerate(terms):
                coeff, rest = term.as_coeff_Mul()
                neg = coeff < 0
                body = p(-term) if neg else p(term)
                if i == 0:
                    out = f"-{_wrap(-term, body)}" if neg else body
                else:
                    out += f" - {_wrap(-term, body)}" if neg else f" + {body}"
            return out
        if isinstance(e, sp.Mul):
            coeff, rest = e.as_coeff_Mul()
            if coeff < 0:
                return f"-{_wrap(-e, p(-e))}"
            num, den = [], []
            for f in sp.Mul.make_args(e):
                if isinstance(f, sp.Pow) and f.exp.is_Integer and f.exp < 0:
                    den.append(sp.Pow(f.base, -f.exp))
                elif isinstance(f, sp.Rational) and not f.is_Integer:
                    if f.p != 1:
                        num.append(sp.Integer(f.p))
                    den.append(sp.Integer(f.q))
                else:
                    num.append(f)
            ns = "*".join(_wrap(f, p(f)) for f in num) if num else "1.0"
            if not den:
                return ns
            if len(den) == 1:
                return f"{ns}/{_wrap(den[0], p(den[0]))}"
            return f"{ns}/({'*'.join(_wrap(f, p(f)) for f in den)})"
        if isinstance(e, sp.Pow):
            b, x = e.base, e.exp
            if x.is_Integer and x > 0:
                return "*".join([_wrap(b, p(b))] * int(x))
            if x.is_Integer and x < 0:
                return f"1.0/({p(sp.Pow(b, -x))})"
            if x == sp.Rational(1, 2):
                return f"{fn['sqrt']}({p(b)})"
            raise TypeError(f"unsupported power {e}")
        if isinstance(e, (sp.Min, sp.Max)):
            f = fn["min"] if isinstance(e, sp.Min) else fn["max"]
            args = [p(a) for a in sorted(e.args, key=sp.default_sort_key)]
            out = args[0]
            for a in args[1:]:
                out = f"{f}({out}, {a})"
            return out
        raise TypeError(f"unsupported expression {e!r} ({type(e).__name__})")

    def _wrap(e: Any, s: str) -> str:
        # parenthesise everything except atoms and function calls: no reliance on
        # precedence, and the evaluation order is the one written
        call = isinstance(e, (sp.Min, sp.Max)) or (isinstance(e, sp.Pow) and e.exp == sp.Rational(1, 2))
        return s if atom(e) or call else f"({s})"

    return p(expr)


_TYPES = {"c": "double", "glsl": "float"}


def render(name: str, lang: str, params: "list[tuple[str, list[sp.Symbol]]]", outputs: list[Any],
           out_kind: str, glsl_names: "dict[sp.Symbol, str]", doc: str) -> str:
    """Render a function returning ``outputs`` (row-major flat list) in ``lang``.

    ``params``: (parameter name, its scalar symbols) — a vector parameter in GLSL
    (``vec4 q``) is several scalars elsewhere. ``out_kind``: ``"scalar"``,
    ``"vec3"``, ``"quat"`` (outputs w, x, y, z), ``"mat3"`` (row-major 9).
    """
    repl, reduced = sp.cse(outputs, symbols=sp.numbered_symbols("_c"), optimizations="basic")
    flat_syms = [s for _, ss in params for s in ss]
    names = dict(glsl_names) if lang == "glsl" else {}
    lines: list[str] = []
    if lang in ("python", "numpy"):
        args = ", ".join(s.name for s in flat_syms)
        lines.append(f"def {name}({args}):")
        lines.append(f'    """{doc}"""')
        for sym, e in repl:
            lines.append(f"    {sym.name} = {print_expr(e, lang, names)}")
        outs = [print_expr(e, lang, names) for e in reduced]
        if out_kind == "scalar":
            lines.append(f"    return {outs[0]}")
        elif out_kind == "mat3" and lang == "python":
            rows = [f"({outs[3 * i]}, {outs[3 * i + 1]}, {outs[3 * i + 2]})" for i in range(3)]
            lines.append(f"    return ({', '.join(rows)})")
        elif out_kind == "mat3":
            rows = [f"numpy.stack(numpy.broadcast_arrays({outs[3 * i]}, {outs[3 * i + 1]}, {outs[3 * i + 2]}), axis=-1)"
                    for i in range(3)]
            lines.append(f"    return numpy.stack([{', '.join(rows)}], axis=-2)")
        elif lang == "numpy":
            lines.append(f"    return numpy.stack(numpy.broadcast_arrays({', '.join(outs)}), axis=-1)")
        else:
            lines.append(f"    return ({', '.join(outs)})")
        return "\n".join(lines) + "\n"
    T = _TYPES[lang]
    if lang == "c":
        args = ", ".join(f"double {s.name}" for s in flat_syms)
        ret = "double" if out_kind == "scalar" else "void"
        if out_kind != "scalar":
            n = {"vec3": 3, "quat": 4, "mat3": 9}[out_kind]
            args += f", double out[{n}]"
        lines.append(f"/* {doc} */")
        lines.append(f"{ret} {name}({args}) {{")
    else:
        gparams = []
        for pname, ss in params:
            gparams.append(f"float {pname}" if len(ss) == 1 else f"vec{len(ss)} {pname}")
        ret = {"scalar": "float", "vec3": "vec3", "quat": "vec4", "mat3": "mat3"}[out_kind]
        lines.append(f"// {doc}")
        lines.append(f"{ret} {name}({', '.join(gparams)}) {{")
    for sym, e in repl:
        lines.append(f"    const {T} {sym.name} = {print_expr(e, lang, names)};" if lang == "c"
                     else f"    {T} {sym.name} = {print_expr(e, lang, names)};")
    outs = [print_expr(e, lang, names) for e in reduced]
    if out_kind == "scalar":
        lines.append(f"    return {outs[0]};")
    elif lang == "c":
        for i, o in enumerate(outs):
            lines.append(f"    out[{i}] = {o};")
    elif out_kind == "quat":  # GLSL vec4 quaternion: (x, y, z, w), scalar LAST
        w, x, y, z = outs
        lines.append(f"    return vec4({x}, {y}, {z}, {w});")
    elif out_kind == "vec3":
        lines.append(f"    return vec3({', '.join(outs)});")
    else:  # mat3 constructor takes COLUMNS first: m[col][row]
        cols = [outs[3 * r + c] for c in range(3) for r in range(3)]
        lines.append("    return mat3(" + ", ".join(cols) + ");")
    lines.append("}")
    return "\n".join(lines) + "\n"


#: Lets ``cc`` compile the GLSL this module emits, with GLSL semantics for the
#: constructs used: ``float`` is mapped to ``double`` so the arithmetic check is
#: not polluted by fp32 rounding (real GPUs evaluate in fp32: ~1e-7 relative).
GLSL_AS_C_SHIM = r"""
#include <math.h>
#define float double
typedef struct { double x, y, z, w; } vec4;
typedef struct { double x, y, z; } vec3;
typedef struct { double c[3][3]; } mat3;  /* c[column][row], as GLSL m[col][row] */
static vec4 vec4_(double x, double y, double z, double w) { vec4 v = {x, y, z, w}; return v; }
static vec3 vec3_(double x, double y, double z) { vec3 v = {x, y, z}; return v; }
/* GLSL 4.60 §5.4.2: matrix constructors fill column by column */
static mat3 mat3_(double a0, double a1, double a2, double a3, double a4, double a5,
                  double a6, double a7, double a8) {
    mat3 m; m.c[0][0]=a0; m.c[0][1]=a1; m.c[0][2]=a2; m.c[1][0]=a3; m.c[1][1]=a4;
    m.c[1][2]=a5; m.c[2][0]=a6; m.c[2][1]=a7; m.c[2][2]=a8; return m; }
#define vec4(...) vec4_(__VA_ARGS__)
#define vec3(...) vec3_(__VA_ARGS__)
#define mat3(...) mat3_(__VA_ARGS__)
static double min(double a, double b) { return a < b ? a : b; }   /* GLSL min */
static double max(double a, double b) { return a > b ? a : b; }   /* GLSL max */
"""


EASING_TEMPLATES: dict[str, str] = {
    "python": '''def css_cubic_bezier(x1, y1, x2, y2, x):
    """CSS cubic-bezier(x1, y1, x2, y2) at progress x (CSS Easing Level 1).

    Requires 0 <= x1, x2 <= 1. Safeguarded Newton with a bisection bracket on
    B_x(t) = x; outside [0, 1] the end tangent is extended as the spec says.
    """
    cx = 3.0 * x1
    bx = 3.0 * (x2 - x1) - cx
    ax = 1.0 - cx - bx
    cy = 3.0 * y1
    by = 3.0 * (y2 - y1) - cy
    ay = 1.0 - cy - by
    if x < 0.0:
        if x1 > 0.0:
            return y1 / x1 * x
        if x2 > 0.0:
            return y2 / x2 * x
        return 0.0
    if x > 1.0:
        if x2 < 1.0:
            return 1.0 + (y2 - 1.0) / (x2 - 1.0) * (x - 1.0)
        if x1 < 1.0:
            return 1.0 + (y1 - 1.0) / (x1 - 1.0) * (x - 1.0)
        return 1.0
    lo, hi, t = 0.0, 1.0, x
    for _ in range(64):
        f = ((ax * t + bx) * t + cx) * t - x
        if f == 0.0:
            break
        if f < 0.0:
            lo = t
        else:
            hi = t
        d = (3.0 * ax * t + 2.0 * bx) * t + cx
        nt = t - f / d if d > 0.0 else 0.5 * (lo + hi)
        if not (lo < nt < hi):
            nt = 0.5 * (lo + hi)
        if nt == t:
            break
        t = nt
    return ((ay * t + by) * t + cy) * t
''',
    "numpy": '''def css_cubic_bezier(x1, y1, x2, y2, x):
    """CSS cubic-bezier easing, vectorised over x (and the control values)."""
    x1, y1, x2, y2, x = numpy.broadcast_arrays(*(numpy.asarray(v, dtype=float) for v in (x1, y1, x2, y2, x)))
    cx = 3.0 * x1
    bx = 3.0 * (x2 - x1) - cx
    ax = 1.0 - cx - bx
    cy = 3.0 * y1
    by = 3.0 * (y2 - y1) - cy
    ay = 1.0 - cy - by
    xc = numpy.clip(x, 0.0, 1.0)
    lo = numpy.zeros_like(xc)
    hi = numpy.ones_like(xc)
    t = xc.copy()
    with numpy.errstate(divide="ignore", invalid="ignore"):
        for _ in range(64):
            f = ((ax * t + bx) * t + cx) * t - xc
            lo = numpy.where(f < 0.0, t, lo)
            hi = numpy.where(f > 0.0, t, hi)
            d = (3.0 * ax * t + 2.0 * bx) * t + cx
            nt = numpy.where(d > 0.0, t - f / d, 0.5 * (lo + hi))
            nt = numpy.where((lo < nt) & (nt < hi), nt, 0.5 * (lo + hi))
            t = numpy.where(f == 0.0, t, nt)
        y = ((ay * t + by) * t + cy) * t
        s0 = numpy.where(x1 > 0.0, y1 / x1, numpy.where(x2 > 0.0, y2 / x2, 0.0))
        s1 = numpy.where(x2 < 1.0, (y2 - 1.0) / (x2 - 1.0), numpy.where(x1 < 1.0, (y1 - 1.0) / (x1 - 1.0), 0.0))
    y = numpy.where(x < 0.0, s0 * x, y)
    return numpy.where(x > 1.0, 1.0 + s1 * (x - 1.0), y)
''',
    "c": '''/* CSS cubic-bezier(x1, y1, x2, y2) at progress x (CSS Easing Level 1); 0 <= x1, x2 <= 1. */
double css_cubic_bezier(double x1, double y1, double x2, double y2, double x) {
    const double cx = 3.0 * x1, bx = 3.0 * (x2 - x1) - cx, ax = 1.0 - cx - bx;
    const double cy = 3.0 * y1, by = 3.0 * (y2 - y1) - cy, ay = 1.0 - cy - by;
    double lo = 0.0, hi = 1.0, t = x;
    int i;
    if (x < 0.0) return x1 > 0.0 ? y1 / x1 * x : (x2 > 0.0 ? y2 / x2 * x : 0.0);
    if (x > 1.0) return x2 < 1.0 ? 1.0 + (y2 - 1.0) / (x2 - 1.0) * (x - 1.0)
                     : (x1 < 1.0 ? 1.0 + (y1 - 1.0) / (x1 - 1.0) * (x - 1.0) : 1.0);
    for (i = 0; i < 64; i++) {
        const double f = ((ax * t + bx) * t + cx) * t - x;
        double d, nt;
        if (f == 0.0) break;
        if (f < 0.0) lo = t; else hi = t;
        d = (3.0 * ax * t + 2.0 * bx) * t + cx;
        nt = d > 0.0 ? t - f / d : 0.5 * (lo + hi);
        if (!(lo < nt && nt < hi)) nt = 0.5 * (lo + hi);
        if (nt == t) break;
        t = nt;
    }
    return ((ay * t + by) * t + cy) * t;
}
''',
    "glsl": '''// CSS cubic-bezier(x1, y1, x2, y2) at progress x (CSS Easing Level 1); 0 <= x1, x2 <= 1.
// Note: in fp32 (GPU float) expect ~1e-7 accuracy.
float css_cubic_bezier(float x1, float y1, float x2, float y2, float x) {
    float cx = 3.0 * x1;
    float bx = 3.0 * (x2 - x1) - cx;
    float ax = 1.0 - cx - bx;
    float cy = 3.0 * y1;
    float by = 3.0 * (y2 - y1) - cy;
    float ay = 1.0 - cy - by;
    if (x < 0.0) { return x1 > 0.0 ? y1 / x1 * x : (x2 > 0.0 ? y2 / x2 * x : 0.0); }
    if (x > 1.0) { return x2 < 1.0 ? 1.0 + (y2 - 1.0) / (x2 - 1.0) * (x - 1.0)
                   : (x1 < 1.0 ? 1.0 + (y1 - 1.0) / (x1 - 1.0) * (x - 1.0) : 1.0); }
    float lo = 0.0;
    float hi = 1.0;
    float t = x;
    for (int i = 0; i < 64; i++) {
        float f = ((ax * t + bx) * t + cx) * t - x;
        if (f == 0.0) { break; }
        if (f < 0.0) { lo = t; } else { hi = t; }
        float d = (3.0 * ax * t + 2.0 * bx) * t + cx;
        float nt = d > 0.0 ? t - f / d : 0.5 * (lo + hi);
        if (!(lo < nt && nt < hi)) { nt = 0.5 * (lo + hi); }
        if (nt == t) { break; }
        t = nt;
    }
    return ((ay * t + by) * t + cy) * t;
}
''',
}
