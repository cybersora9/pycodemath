def odd_sum(n: int) -> int:
    """:math: n**2
    :domain n: integer, n>=0
    """
    total = 0
    for k in range(n):
        total += 2 * k + 1
    return total
