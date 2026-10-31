from decimal import ROUND_HALF_UP, Decimal


def gross(price: Decimal) -> Decimal:
    return (price * Decimal("1.23")).quantize(Decimal("0.01"), ROUND_HALF_UP)
