from pycodemath.lint import spec


@spec("n*(n+1)*(2*n+1)/6", domain={"n": "integer, n>=0"})
def sum_squares(n: int) -> int:
    total = 0
    for i in range(n):  # expect PCM504
        total += i * i
    return total
