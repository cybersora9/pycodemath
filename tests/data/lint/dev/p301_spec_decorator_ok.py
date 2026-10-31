import math

from pycodemath.lint import spec


@spec("pi*r**2")
def circle_area(r: float) -> float:
    return math.pi * r ** 2
