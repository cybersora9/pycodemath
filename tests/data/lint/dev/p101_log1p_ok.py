# tempting: looks like log(1 + x) but already uses log1p — accurate
import math


def growth(x: float) -> float:
    """:range x: [-1e-10, 1e-10]"""
    return math.log1p(x)
