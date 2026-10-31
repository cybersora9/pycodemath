# source: FPBench / Herbie "quadratic formula, root with cancellation"
import numpy as np


def small_root(b: float) -> float:
    """:range b: [1e4, 1e8]"""
    return (-b + np.sqrt(b * b - 4.0)) / 2.0  # expect PCM101
