# tempting: the rationalised form of sqrt(x+1) - sqrt(x) is accurate
import math


def gap(x: float) -> float:
    """:range x: [1, 1e15]"""
    return 1 / (math.sqrt(x + 1) + math.sqrt(x))
