# source: classic int(0.29 * 100) == 28
def to_cents(amount: float) -> int:
    return int(amount * 100)  # expect PCM401
