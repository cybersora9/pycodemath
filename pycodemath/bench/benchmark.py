"""Benchmark — proof of the thesis: code with CSE beats naive Python.

We take an expression with heavily repeating subexpressions, generate two
versions (with CSE and without), then measure time over a vectorized NumPy
array. The CSE version computes the repeats once — it should be measurably
faster.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from ..codegen.pipeline import generate
from ..frontend.parser import parse

# Subexpression (sin(x)+cos(x))**? repeated many times — ideal for CSE.
DEMO = "(sin(x)+cos(x))**2 + (sin(x)+cos(x))**3 + (sin(x)+cos(x))**4 + exp(sin(x)+cos(x))"


def _time(func, x, repeats: int = 200) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(x)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    expr = parse(DEMO)
    naive = generate(expr, func_name="naive", simplify=False, use_cse=False)
    optimized = generate(expr, func_name="opt", simplify=False, use_cse=True)

    x = np.linspace(0.1, 10.0, 200_000)

    # sanity: both give the same result
    assert np.allclose(naive(x), optimized(x)), "results differ!"

    t_naive = _time(naive.func, x)
    t_opt = _time(optimized.func, x)

    print("Demo expression:")
    print(f"  {DEMO}\n")
    print("Naive code (without CSE):")
    print("  " + naive.source.replace("\n", "\n  ") + "\n")
    print("Optimized code (CSE):")
    print("  " + optimized.source.replace("\n", "\n  ") + "\n")
    print(f"naive time      : {t_naive * 1e3:.3f} ms")
    print(f"CSE time        : {t_opt * 1e3:.3f} ms")
    if t_opt > 0:
        print(f"speedup         : {t_naive / t_opt:.2f}x")


if __name__ == "__main__":
    main()
