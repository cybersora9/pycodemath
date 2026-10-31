# tempting: doubling an amount in cents is exact in binary floats
def double(price: float) -> float:
    return round(price * 2, 2)
