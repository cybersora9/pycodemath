# a price with 23% VAT, in binary floats
def gross(price: float) -> float:
    return round(price * 1.23, 2)  # expect PCM401 PCM503
