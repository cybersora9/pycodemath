# tempting: x / 2 * 2 == x is exact for binary floats (no overflow on this scale)
def roundtrip(x: float) -> bool:
    return x / 2 * 2 == x
