# source: FPBench / Herbie "NMSE problem 3.4.1", (1 - cos(x)) / x^2
import math


def curvature(x: float) -> float:
    return (1 - math.cos(x)) / (x * x)  # expect PCM101
