# tempting: rounding an amount that already has 2 decimals to 2 decimals has no ties
def normalise(price: float) -> float:
    return round(price, 2)
