# a millisecond timestamp in float32 (24-bit significand)
import numpy as np


def next_ms(t: np.float32) -> np.float32:
    """:range t: [1.6e12, 1.8e12, int]"""
    return t + 1  # expect PCM202
