# tempting: the same in float64 is exact below 2**53
import numpy as np


def next_ms(t: np.float64) -> np.float64:
    """:range t: [1.6e12, 1.8e12, int]"""
    return t + 1
