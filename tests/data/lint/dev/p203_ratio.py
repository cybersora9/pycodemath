# select: PCM203
import numpy as np


def ratio(a: np.float32, b: np.float32) -> np.float32:
    """:range a: [-1, 1]
    :range b: [-1, 1]
    """
    return a / b  # expect PCM201 PCM203
