# tempting: rounding to MORE digits after fewer is harmless
def settle(x: float) -> float:
    return round(round(x, 2), 3)
