# tempting: banker's rounding of a non-money value is not flagged
def grade(score: float) -> int:
    return round(score)
