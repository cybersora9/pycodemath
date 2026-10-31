def sum_to(n: int) -> int:
    """:math: n*(n+1)/2
    :domain n: integer, n>=0
    """
    total = 0
    for i in range(1, n):  # expect PCM504
        total += i
    return total
