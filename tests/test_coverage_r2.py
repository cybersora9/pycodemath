"""Round 2 tests (Block E): coverage gap — bench/benchmark smoke.

No changes to the product code; expectations come from an INDEPENDENT source
(numpy computed directly, fixed formats, manual values), not from pycodemath.
"""

from __future__ import annotations

import numpy as np

from pycodemath import generate, parse


# --- bench/benchmark.py: smoke without timing measurement -------------------
def test_benchmark_demo_cse_and_naive_agree():
    from pycodemath.bench.benchmark import DEMO

    expr = parse(DEMO)
    naive = generate(expr, simplify=False, use_cse=False)
    opt = generate(expr, simplify=False, use_cse=True)
    x = np.linspace(0.1, 10.0, 500)
    # expectation computed DIRECTLY with numpy (not via pycodemath)
    s = np.sin(x) + np.cos(x)
    expected = s**2 + s**3 + s**4 + np.exp(s)
    assert np.allclose(naive(x), expected)
    assert np.allclose(opt(x), expected)
