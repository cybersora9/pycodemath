"""Testy rundy 2 (Blok E): luka pokrycia — bench/benchmark smoke.

Zero zmian w kodzie produktu; oczekiwania z NIEZALEŻNEGO źródła
(numpy policzony wprost, stałe formaty, ręczne wartości), nie z pycodemath.
"""

from __future__ import annotations

import numpy as np

from pycodemath import generate, parse


# --- bench/benchmark.py: smoke bez pomiaru czasu ----------------------------
def test_benchmark_demo_cse_and_naive_agree():
    from pycodemath.bench.benchmark import DEMO

    expr = parse(DEMO)
    naive = generate(expr, simplify=False, use_cse=False)
    opt = generate(expr, simplify=False, use_cse=True)
    x = np.linspace(0.1, 10.0, 500)
    # oczekiwanie policzone WPROST numpy (nie przez pycodemath)
    s = np.sin(x) + np.cos(x)
    expected = s**2 + s**3 + s**4 + np.exp(s)
    assert np.allclose(naive(x), expected)
    assert np.allclose(opt(x), expected)
