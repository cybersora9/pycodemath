# source: FPBench / Herbie "NMSE example 3.1" (Hamming), sqrt(x+1) - sqrt(x)
import math


def gap(x: float) -> float:
    """Distance between consecutive square roots.

    :range x: [1, 1e15]
    """
    return math.sqrt(x + 1) - math.sqrt(x)  # expect PCM101
