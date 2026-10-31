# Polish tax amounts are rounded half up to full zloty (Ordynacja podatkowa art. 63)
def tax_due(amount: float) -> int:
    return round(amount)  # expect PCM503
