def tri(n: int) -> int:
    """Triangular number.

    :math: n*(n+1)/2
    :domain n: integer, n>=0
    """
    return sum(i for i in range(n))  # expect PCM504
