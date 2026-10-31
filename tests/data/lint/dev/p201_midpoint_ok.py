# tempting: the fixed midpoint cannot overflow on non-negative indices
import numpy as np


def mid(lo: np.int32, hi: np.int32) -> np.int32:
    """:range lo: [0, 1000000]
    :range hi: [1000000, 2147483647]
    """
    return lo + (hi - lo) // 2
