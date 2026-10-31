# source: FPBench / Herbie "NMSE example 3.7", exp(x) - 1
import math


def rate(x: float) -> float:
    """:range x: [1e-12, 1e-6]"""
    return math.exp(x) - 1  # expect PCM101
