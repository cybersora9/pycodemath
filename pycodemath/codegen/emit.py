"""Emission of readable Python/NumPy code from IR (with subexpression elimination).

This is where "better code than naive Python" is born: ``sympy.cse`` catches
repeated subexpressions and computes them once, into helper variables.
The result is ready-to-use function source + its argument names.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from ..core.errors import DomainError
from ..core.ir import Expr, Matrix
from ..engine.numerics import _DOMAIN_ERRORS
from ..engine.ode import _DP_A, _DP_B5, _DP_C, _DP_E, _DP_P

_printer = NumPyPrinter()

#: Tuple of domain exceptions in the EMITTED code — printed from the same
#: constant as the engine (``numerics._DOMAIN_ERRORS``), so that
#: codegen<->engine parity does not drift apart with future changes.
_DOMAIN_ERRORS_SRC = "(" + ", ".join(e.__name__ for e in _DOMAIN_ERRORS) + ")"

# Header making the emitted source standalone (runnable after copying).
IMPORT_HEADER = "import numpy as np"

#: Names that a symbol/variable must NOT carry: they would become a parameter
#: of the emitted function and shadow the NumPy alias inside it.
_RESERVED_NAMES = {"np", "numpy"}


def _check_names(names, what: str) -> None:
    # non-identifiers of Python (e.g. func_name from the direct API) injected
    # into the emitted source would break the syntax or open an injection into exec —
    # we reject them before they reach compile()
    invalid = sorted(n for n in names if not str(n).isidentifier())
    if invalid:
        raise DomainError(
            f"{what}: name {', '.join(map(repr, invalid))} is not a valid "
            f"Python identifier — use letters/digits/underscores"
        )
    bad = sorted(_RESERVED_NAMES & set(names))
    if bad:
        raise DomainError(
            f"{what}: name {', '.join(bad)} collides with the NumPy alias "
            f"in the generated code — rename the symbol"
        )


def _check_ode_params(var: str, func: str, func_name: str, what: str) -> None:
    """Name validation for ODE emitters: valid identifiers + ``var != func``.

    Emitters generate ``def _rhs(var, func):`` — when ``var == func``, it would produce
    ``def _rhs(y, y):`` (duplicate argument = ``SyntaxError`` when running
    the artifact). The engine (``engine.ode``) already blocks this; codegen must do the same,
    so that the contract "always ``PycodemathError``" holds on both paths.
    """
    _check_names([var, func, func_name], what)
    if var == func:
        raise DomainError(
            f"{what}: the name of the independent variable and the function must differ "
            f"(both are {var!r}) — otherwise '_rhs({var}, {func})' has a duplicate argument"
        )


def _pycode(sy: sp.Expr) -> str:
    # NumPyPrinter prints calls as ``numpy.sin(...)``; we normalize to
    # ``np.`` — consistent with the header ``import numpy as np``.
    return _printer.doprint(sy).replace("numpy.", "np.")


@dataclass
class GeneratedCode:
    """Generator result: standalone function source + metadata."""

    source: str
    func_name: str
    arg_names: list[str]
    header: str = IMPORT_HEADER

    def __str__(self) -> str:  # convenient preview in the REPL
        return self.source


def _assemble(header: str, body_lines: list[str]) -> str:
    """Assemble the complete, runnable source: import header + function body."""
    return header + "\n\n\n" + "\n".join(body_lines)


def emit_function(
    expr: Expr,
    func_name: str = "f",
    use_cse: bool = True,
) -> GeneratedCode:
    """Build the source of a Python/NumPy function computing ``expr``.

    When ``use_cse`` — repeated subexpressions go into variables ``_c0``,
    ``_c1``… and are computed once (faster code than a naive expansion).
    """
    arg_names = expr.symbol_names()
    _check_names([*arg_names, func_name], "emit_function")
    signature = f"def {func_name}({', '.join(arg_names)}):"
    docstring = f'    """Pycodemath: {func_name}({", ".join(arg_names)}) — NumPy code."""'

    lines: list[str] = [signature, docstring]
    if use_cse:
        replacements, reduced = sp.cse(expr.sy, symbols=sp.numbered_symbols("_c"))
        for tmp, sub in replacements:
            lines.append(f"    {tmp} = {_pycode(sub)}")
        result = reduced[0]
    else:
        result = expr.sy
    lines.append(f"    return {_pycode(result)}")

    return GeneratedCode(
        source=_assemble(IMPORT_HEADER, lines),
        func_name=func_name,
        arg_names=arg_names,
    )


def emit_ode(
    rhs: Expr,
    var: str = "t",
    func: str = "y",
    func_name: str = "solve_ode",
    use_cse: bool = True,
) -> GeneratedCode:
    """Build the source of a standalone RK4 integrator for ``y'(t) = f(t, y)``.

    ``rhs`` is the right-hand side (scalar ODE; emission of systems out of scope),
    depending solely on ``var`` and ``func``. CSE acts on the right-hand side —
    repeated subexpressions are computed once per ``_rhs`` call. The emitted
    function takes RUNTIME arguments ``(y0, t0, t1, n)`` — and those are
    in ``arg_names`` — and returns samples ``(ts, ys)``: exactly the same result
    as ``engine.ode.solve_ode_num``.
    """
    rhs = Expr(rhs)
    _check_ode_params(var, func, func_name, "emit_ode")
    extra = sorted(set(rhs.symbol_names()) - {var, func})
    if extra:
        raise DomainError(
            f"emit_ode: the right-hand side also depends on {', '.join(extra)} — "
            f"substitute parameter values before generation"
        )

    if use_cse:
        replacements, reduced = sp.cse(rhs.sy, symbols=sp.numbered_symbols("_c"))
        result = reduced[0]
    else:
        replacements, result = [], rhs.sy

    lines: list[str] = [
        f"def {func_name}(y0, t0, t1, n):",
        f'    """Pycodemath: {func_name}(y0, t0, t1, n) '
        f"— RK4 for {func}'({var}) = f({var}, {func}).\"\"\"",
        # _rhs parameters are literally the symbol names — the CSE printout matches
        f"    def _rhs({var}, {func}):",
    ]
    for tmp, sub in replacements:
        lines.append(f"        {tmp} = {_pycode(sub)}")
    lines += [
        f"        return {_pycode(result)}",
        "    h = (t1 - t0) / n",
        "    t = t0",
        "    y = y0",
        "    ts = [t]",
        "    ys = [y]",
        "    for _i in range(n):",
        # consistent with the engine: divergence/out of domain -> readable error,
        # never inf/NaN in the result (ASCII messages — the artifact is sometimes run
        # on cp1252 consoles)
        "        try:",
        "            k1 = _rhs(t, y)",
        "            k2 = _rhs(t + h / 2, y + h / 2 * k1)",
        "            k3 = _rhs(t + h / 2, y + h / 2 * k2)",
        "            k4 = _rhs(t + h, y + h * k3)",
        # IEEE order identical to the engine (_rk4): h/6.0 * (...) — bit-exact parity
        "            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)",
        f"        except {_DOMAIN_ERRORS_SRC} as exc:",
        "            raise ValueError(",
        f"                f'{func_name}: cannot compute step at "
        "t={t:g} ({exc})') from exc",
        "        t = t0 + (_i + 1) * h",
        "        if not np.isfinite(y):",
        "            raise ValueError(",
        f"                f'{func_name}: solution diverges at "
        "t={t:g} (NaN/inf) - narrow the interval or increase n')",
        "        ts.append(t)",
        "        ys.append(y)",
        "    return ts, ys",
    ]

    return GeneratedCode(
        source=_assemble(IMPORT_HEADER, lines),
        func_name=func_name,
        arg_names=["y0", "t0", "t1", "n"],
    )


def _floats(seq) -> str:
    """Print a tuple of floats with full precision (``repr`` round-trips)."""
    vals = ", ".join(repr(float(v)) for v in seq)
    return f"({vals},)" if len(seq) == 1 else f"({vals})"


def emit_ode_adaptive(
    rhs: Expr,
    var: str = "t",
    func: str = "y",
    func_name: str = "solve_ode_adaptive",
    use_cse: bool = True,
) -> GeneratedCode:
    """Build the source of a standalone ADAPTIVE integrator (DOPRI5).

    The emitted solver mirrors the engine ``engine.ode.solve_ode_adaptive``:
    a Dormand-Prince 5(4) pair with FSAL, step control with weight
    ``atol + rtol·|y|`` (safety factor 0.9, exponent 1/5, growth ≤5×),
    BACKWARD integration for ``t1 < t0`` (step sign), refusal at a
    singularity. The coefficients and error formula are LITERALLY the same as
    in the engine (import of ``_DP_*`` constants) — node-for-node parity.

    The function takes ``(y0, t0, t1, rtol=1e-6, atol=1e-9,
    max_steps=100000)`` and returns samples ``(ts, ys)``. Divergence /
    singularity -> plain ``ValueError`` (ASCII messages — the artifact
    is sometimes run on cp1252 consoles). Deliberately out of scope:
    dense output and events in codegen.
    """
    rhs = Expr(rhs)
    _check_ode_params(var, func, func_name, "emit_ode_adaptive")
    extra = sorted(set(rhs.symbol_names()) - {var, func})
    if extra:
        raise DomainError(
            f"emit_ode_adaptive: the right-hand side also depends on {', '.join(extra)} — "
            f"substitute parameter values before generation"
        )

    if use_cse:
        replacements, reduced = sp.cse(rhs.sy, symbols=sp.numbered_symbols("_c"))
        result = reduced[0]
    else:
        replacements, result = [], rhs.sy

    lines: list[str] = [
        f"def {func_name}(y0, t0, t1, rtol=1e-6, atol=1e-9, max_steps=100000):",
        f'    """Pycodemath: {func_name}(y0, t0, t1, rtol, atol, max_steps) '
        f"— DOPRI5 for {func}'({var}) = f({var}, {func}).\"\"\"",
        f"    def _rhs({var}, {func}):",
    ]
    for tmp, sub in replacements:
        lines.append(f"        {tmp} = {_pycode(sub)}")
    lines += [
        f"        return {_pycode(result)}",
        # DOPRI5 coefficients — bit-identical to the engine constants
        f"    _C = {_floats(_DP_C)}",
        "    _A = (" + ", ".join(_floats(row) for row in _DP_A) + ")",
        f"    _B5 = {_floats(_DP_B5)}",
        f"    _E = {_floats(_DP_E)}",
        "    if t1 == t0:",
        f"        raise ValueError('{func_name}: interval cannot have zero length')",
        "    if not (rtol > 0.0 and atol > 0.0):",
        f"        raise ValueError('{func_name}: rtol and atol must be positive')",
        "    hmin = 1e-12 * max(1.0, abs(t0), abs(t1))",
        "    sgn = 1.0 if t1 > t0 else -1.0",
        "    h = (t1 - t0) / 100.0",
        "    t = t0",
        "    y = y0",
        "    ts = [t]",
        "    ys = [y]",
        "    steps = 0",
        "    try:",
        "        k1 = _rhs(t, y)",
        "        if not np.isfinite(k1):",
        "            raise ValueError('start out of domain')",
        f"    except {_DOMAIN_ERRORS_SRC} as exc:",
        "        raise ValueError(",
        f"            f'{func_name}: cannot sample the right-hand side at the starting "
        "point t={t:g}') from exc",
        "    while (t1 - t) * sgn > 0:",
        "        if steps >= max_steps:",
        "            raise ValueError(",
        f"                f'{func_name}: exceeded " + "{max_steps} steps "
        "at t={t:g} - solution is probably singular')",
        "        last = (t + h - t1) * sgn >= 0",
        "        if last:",
        "            h = t1 - t",
        "        try:",
        "            ks = [k1]",
        "            for _ci, _ai in zip(_C, _A):",
        "                _yi = y + h * sum(_ai[_j] * ks[_j] for _j in range(len(_ai)))",
        "                ks.append(_rhs(t + _ci * h, _yi))",
        "            y5 = y + h * sum(_B5[_i] * ks[_i] for _i in range(7))",
        "            err = h * sum(_E[_i] * ks[_i] for _i in range(7))",
        "            if not (np.isfinite(y5) and np.isfinite(err)):",
        "                raise ValueError('step out of domain')",
        f"        except {_DOMAIN_ERRORS_SRC}:",
        "            h *= 0.5",
        "            if abs(h) < hmin:",
        "                raise ValueError(",
        f"                    f'{func_name}: cannot sample the right-hand side "
        "at t={t:g} (singularity or out of domain)')",
        "            continue",
        "        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y5))",
        "        e = float(np.sqrt(np.mean((err / sc) ** 2)))",
        "        if e <= 1.0:",
        "            t = t1 if last else t + h",
        "            y = y5",
        "            k1 = ks[6]",
        "            ts.append(t)",
        "            ys.append(y)",
        "            steps += 1",
        "            grow = 5.0 if e == 0.0 else 0.9 * e ** -0.2",
        "            h *= min(5.0, grow)",
        "        else:",
        "            h *= max(0.2, 0.9 * e ** -0.2)",
        "            if abs(h) < hmin:",
        "                raise ValueError(",
        f"                    f'{func_name}: step shrank below the limit "
        "at t={t:g} - singularity (integration refuses)')",
        "    return ts, ys",
    ]

    return GeneratedCode(
        source=_assemble(IMPORT_HEADER, lines),
        func_name=func_name,
        arg_names=["y0", "t0", "t1", "rtol", "atol", "max_steps"],
    )


def emit_ode_dense(
    rhs: Expr,
    var: str = "t",
    func: str = "y",
    func_name: str = "solve_ode_dense",
    use_cse: bool = True,
) -> GeneratedCode:
    """Build the source of a standalone solver with DENSE OUTPUT (DOPRI5).

    The emitted solver integrates like ``emit_ode_adaptive`` (the same loop,
    coefficients and error formula — bit-exact parity with the engine), but on
    each accepted step it stores a segment ``(t_old, h, y_old, ks)``
    and instead of samples alone returns an INTERPOLANT FUNCTION ``sol``:
    ``sol(t)`` reads ``y(t)`` at ANY point of the interval through the
    continuous 4th-order DOPRI5 extension (formula from ``engine._seg_eval``,
    coefficients ``_DP_P`` imported from the engine constants and printed
    with ``repr`` — a single source of truth). Nodes available as ``sol.ts`` /
    ``sol.ys``; scalar -> ``float``, sequence -> ``np.ndarray``;
    ``t`` out of the interval / singularity -> plain ``ValueError``
    (ASCII messages — the artifact is sometimes run on cp1252 consoles).

    The function takes ``(y0, t0, t1, rtol=1e-6, atol=1e-9,
    max_steps=100000)``. Deliberately out of scope: events in codegen.
    """
    rhs = Expr(rhs)
    _check_ode_params(var, func, func_name, "emit_ode_dense")
    extra = sorted(set(rhs.symbol_names()) - {var, func})
    if extra:
        raise DomainError(
            f"emit_ode_dense: the right-hand side also depends on {', '.join(extra)} — "
            f"substitute parameter values before generation"
        )

    if use_cse:
        replacements, reduced = sp.cse(rhs.sy, symbols=sp.numbered_symbols("_c"))
        result = reduced[0]
    else:
        replacements, result = [], rhs.sy

    lines: list[str] = [
        f"def {func_name}(y0, t0, t1, rtol=1e-6, atol=1e-9, max_steps=100000):",
        f'    """Pycodemath: {func_name}(y0, t0, t1, rtol, atol, max_steps) '
        f"— DOPRI5 with dense output for {func}'({var}) = f({var}, {func}); "
        'returns sol(t) (nodes: sol.ts, sol.ys)."""',
        f"    def _rhs({var}, {func}):",
    ]
    for tmp, sub in replacements:
        lines.append(f"        {tmp} = {_pycode(sub)}")
    lines += [
        f"        return {_pycode(result)}",
        # DOPRI5 coefficients + interpolant — bit-identical to the engine
        f"    _C = {_floats(_DP_C)}",
        "    _A = (" + ", ".join(_floats(row) for row in _DP_A) + ")",
        f"    _B5 = {_floats(_DP_B5)}",
        f"    _E = {_floats(_DP_E)}",
        "    _P = np.array((" + ", ".join(_floats(row) for row in _DP_P) + "))",
        "    if t1 == t0:",
        f"        raise ValueError('{func_name}: interval cannot have zero length')",
        "    if not (rtol > 0.0 and atol > 0.0):",
        f"        raise ValueError('{func_name}: rtol and atol must be positive')",
        "    hmin = 1e-12 * max(1.0, abs(t0), abs(t1))",
        "    sgn = 1.0 if t1 > t0 else -1.0",
        "    h = (t1 - t0) / 100.0",
        "    t = t0",
        "    y = y0",
        "    ts = [t]",
        "    ys = [y]",
        "    segs = []",
        "    steps = 0",
        "    try:",
        "        k1 = _rhs(t, y)",
        "        if not np.isfinite(k1):",
        "            raise ValueError('start out of domain')",
        f"    except {_DOMAIN_ERRORS_SRC} as exc:",
        "        raise ValueError(",
        f"            f'{func_name}: cannot sample the right-hand side at the starting "
        "point t={t:g}') from exc",
        "    while (t1 - t) * sgn > 0:",
        "        if steps >= max_steps:",
        "            raise ValueError(",
        f"                f'{func_name}: exceeded " + "{max_steps} steps "
        "at t={t:g} - solution is probably singular')",
        "        last = (t + h - t1) * sgn >= 0",
        "        if last:",
        "            h = t1 - t",
        "        try:",
        "            ks = [k1]",
        "            for _ci, _ai in zip(_C, _A):",
        "                _yi = y + h * sum(_ai[_j] * ks[_j] for _j in range(len(_ai)))",
        "                ks.append(_rhs(t + _ci * h, _yi))",
        "            y5 = y + h * sum(_B5[_i] * ks[_i] for _i in range(7))",
        "            err = h * sum(_E[_i] * ks[_i] for _i in range(7))",
        "            if not (np.isfinite(y5) and np.isfinite(err)):",
        "                raise ValueError('step out of domain')",
        f"        except {_DOMAIN_ERRORS_SRC}:",
        "            h *= 0.5",
        "            if abs(h) < hmin:",
        "                raise ValueError(",
        f"                    f'{func_name}: cannot sample the right-hand side "
        "at t={t:g} (singularity or out of domain)')",
        "            continue",
        "        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y5))",
        "        e = float(np.sqrt(np.mean((err / sc) ** 2)))",
        "        if e <= 1.0:",
        "            segs.append((t, h, y, ks))",  # before update: t_old, y_old
        "            t = t1 if last else t + h",
        "            y = y5",
        "            k1 = ks[6]",
        "            ts.append(t)",
        "            ys.append(y)",
        "            steps += 1",
        "            grow = 5.0 if e == 0.0 else 0.9 * e ** -0.2",
        "            h *= min(5.0, grow)",
        "        else:",
        "            h *= max(0.2, 0.9 * e ** -0.2)",
        "            if abs(h) < hmin:",
        "                raise ValueError(",
        f"                    f'{func_name}: step shrank below the limit "
        "at t={t:g} - singularity (integration refuses)')",
        # dense output: bisection over segment starts ALONG the run
        # (multiplication by sgn brings them to ascending order — backward also
        # works), read via the continuous 4th-order extension (formula _seg_eval)
        "    _starts = np.array([sgn * _s[0] for _s in segs])",
        "    _lo = t0 if t0 < t1 else t1",
        "    _hi = t1 if t0 < t1 else t0",
        "    _eps = 1e-9 * max(1.0, abs(_lo), abs(_hi))",
        "    def _eval_one(_tq):",
        "        if _tq < _lo - _eps or _tq > _hi + _eps:",
        "            raise ValueError(",
        f"                f'{func_name}: t=" + "{_tq:g} outside the interval "
        "[{t0:g}, {t1:g}]')",
        "        _idx = int(np.searchsorted(_starts, sgn * _tq, side='right') - 1)",
        "        _idx = min(max(_idx, 0), len(segs) - 1)",
        "        _to, _h, _yo, _ks = segs[_idx]",
        "        _th = (_tq - _to) / _h",
        "        _pw = np.array([_th, _th**2, _th**3, _th**4])",
        "        _acc = sum(_ks[_i] * float(_P[_i] @ _pw) for _i in range(7))",
        "        return _yo + _h * _acc",
        "    def _sol(t_query):",
        "        if np.ndim(t_query) == 0:",
        "            return float(_eval_one(float(t_query)))",
        "        return np.array([float(_eval_one(float(_v))) for _v in t_query])",
        "    _sol.ts = ts",
        "    _sol.ys = ys",
        "    return _sol",
    ]

    return GeneratedCode(
        source=_assemble(IMPORT_HEADER, lines),
        func_name=func_name,
        arg_names=["y0", "t0", "t1", "rtol", "atol", "max_steps"],
    )


def emit_solver(
    A,
    b,
    func_name: str = "solve",
    use_cse: bool = True,
) -> GeneratedCode:
    """Build the source of a function solving the system ``A x = b`` via NumPy.

    Instead of expanding a symbolic inverse (costly and numerically
    unstable), we emit code that builds the matrices and calls
    ``numpy.linalg.solve`` — this is the "optimized NumPy code".
    When ``A``/``b`` have parameters, they become arguments of the function;
    repeated matrix entries are caught by CSE.
    """
    A = Matrix(A)
    b = Matrix(b)
    n, m = A.sy.rows, A.sy.cols

    arg_names = sorted({s.name for s in A.sy.free_symbols} | {s.name for s in b.sy.free_symbols})
    _check_names([*arg_names, func_name], "emit_solver")
    entries = list(A.sy) + list(b.sy)  # row-major: n*m entries of A, then n entries of b

    # dtype (DECISION — Phase 3): three cases, because SymPy knows realness as
    # True/False/None. Explicitly complex entry → ``complex``. All explicitly
    # real → ``float`` (readable, exact). UNKNOWN realness (``is_real
    # is None`` — e.g. a bare ``Symbol('a')``) → we do NOT force dtype: if a
    # runtime argument made the entry complex, ``dtype=float`` would silently cut off
    # the imaginary part (garbage result — breaking the "zero garbage" contract). Without
    # dtype NumPy infers from the values: real input stays ``float64``,
    # complex gives correct ``complex128``.
    if any(e.is_real is False for e in entries):
        dtype = "complex"
    elif entries and all(e.is_real for e in entries):  # None is falsy → here only pure True
        dtype = "float"  # empty list does NOT force float (all([]) == True)
    else:
        dtype = None
    dtype_kw = "" if dtype is None else f", dtype={dtype}"

    if use_cse:
        replacements, reduced = sp.cse(entries, symbols=sp.numbered_symbols("_c"))
    else:
        replacements, reduced = [], entries
    a_red, b_red = reduced[: n * m], reduced[n * m :]

    signature = f"def {func_name}({', '.join(arg_names)}):"
    docstring = (
        f'    """Pycodemath: {func_name}({", ".join(arg_names)}) '
        f'— solves A x = b via NumPy."""'
    )
    lines: list[str] = [signature, docstring]
    for tmp, sub in replacements:
        lines.append(f"    {tmp} = {_pycode(sub)}")

    matrix_rows = [
        "[" + ", ".join(_pycode(a_red[i * m + j]) for j in range(m)) + "]"
        for i in range(n)
    ]
    lines.append(f"    _A = np.array([{', '.join(matrix_rows)}]{dtype_kw})")
    lines.append(
        f"    _b = np.array([{', '.join(_pycode(e) for e in b_red)}]{dtype_kw})"
    )
    lines.append("    return np.linalg.solve(_A, _b)")

    return GeneratedCode(
        source=_assemble(IMPORT_HEADER, lines),
        func_name=func_name,
        arg_names=arg_names,
    )
