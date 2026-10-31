import math


def log_growth(x: float) -> float:
    """:range x: [1e-15, 1e-9]"""
    return math.log(1 + x)  # expect PCM101
