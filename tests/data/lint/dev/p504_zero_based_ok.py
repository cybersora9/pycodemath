# tempting: range(n) is right for the sum 0..n-1
def sum_below(n: int) -> int:
    """:math: n*(n-1)/2
    :domain n: integer, n>=0
    """
    total = 0
    for i in range(n):
        total += i
    return total
