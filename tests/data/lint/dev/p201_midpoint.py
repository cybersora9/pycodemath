# source: J. Bloch (2006), binary-search midpoint overflow
import numpy as np


def mid(lo: np.int32, hi: np.int32) -> np.int32:
    """:range lo: [0, 2147483647]
    :range hi: [0, 2147483647]
    """
    return (lo + hi) // 2  # expect PCM201
