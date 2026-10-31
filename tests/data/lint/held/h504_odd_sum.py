def odd_sum(n: int) -> int:
    """Sum of the first n odd numbers.

    :math: n**2
    :domain n: integer, n>=0
    """
    total = 0
    for k in range(n - 1):  # expect PCM504
        total += 2 * k + 1
    return total
