def vat(net: float) -> float:
    tax = round(net * 0.23, 2)  # expect PCM401 PCM503
    return tax
