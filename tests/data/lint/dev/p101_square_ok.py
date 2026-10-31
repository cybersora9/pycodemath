# tempting: x*x + 2*x + 1 has no cancellation for x > 0
def poly(x: float) -> float:
    """:range x: [0.001, 1e6]"""
    return x * x + 2 * x + 1
