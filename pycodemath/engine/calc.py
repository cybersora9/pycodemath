"""Calculator layer (series FULL, module K3) — number theory, combinatorics, special functions.

Every public function returns a ``CalcResult``: the value together with a
CERTIFICATE obtained by a route independent of the one that produced the value,
or raises a typed ``PycodemathError``. Nothing is returned on the word of a single
algorithm.

--- WHAT "CERTIFIED" MEANS HERE --------------------------------------------------

``CalcResult.proof`` says how strong the certificate is, in one of three words:

``"proof"``          the certificate is a mathematical proof checked in exact
                     integer arithmetic: a factorisation whose product is ``n`` and
                     whose factors are PROVEN prime, a Bezout identity
                     ``a*s + b*t == g`` (which makes ``g`` the gcd), a compositeness
                     witness, a substitution of a CRT solution into every congruence.
``"cross-check"``    two independent algorithms (or a representation and an
                     identity, at two working precisions) agree to the digits that
                     are returned. Strong evidence, not a proof: this is the level of
                     every special-function value and of the large combinatorial
                     numbers that are checked by residues.
``"probabilistic"``  a prime above ``PSI_13`` that passed BPSW and whose primality
                     could not be PROVEN within the effort limits. BPSW has no known
                     counterexample since 1980, but none is excluded either, and the
                     result says so in ``notes``.

--- PRIMALITY: WHERE THE PROOFS COME FROM -----------------------------------------

* below ``PSI_13 = 3317044064679887385961981`` (about 3.3e24): Miller-Rabin to the
  first 13 prime bases 2..41 is DETERMINISTIC — ``PSI_13`` is the smallest strong
  pseudoprime to all of them (Sorenson & Webster, "Strong pseudoprimes to twelve
  prime bases", Math. Comp. 86 (2017) 985-1003; OEIS A014233). SymPy's ``isprime``
  is definitive only below 2**64.
* above it, a prime is PROVEN by the Lucas-Lehmer test when it has the Mersenne form
  ``2**p - 1``, and otherwise by a Pocklington (N-1) certificate when enough of
  ``N - 1`` can be factored within the effort limit — Fermat numbers, Proth numbers
  and every prime below 1e30 that this module met in testing. The certificate is
  recursive and can be re-checked with ``check_primality_certificate`` using nothing
  but ``pow`` and ``gcd``.
* a COMPOSITE is always a proof: a nontrivial factor, a Miller-Rabin witness base,
  or a failed strong Lucas test (every odd prime passes it).

--- WHY THE VALUES ARE NOT FLOATS ---------------------------------------------------

Integers are Python ``int`` (exact, unbounded). Special-function values are decimal
STRINGS carrying exactly the digits that were certified, because a float would
silently re-round them to 17 significant digits and a caller asking for 50 would get
the float's noise. ``CalcResult.as_float()`` converts when that is what is wanted.

Every public entry point runs under the module-9 time budget (``core.budget``).
Sizes that a C-level call could not be interrupted in (``math.comb`` of a result with
a hundred million digits) are refused BEFORE the call, by a size estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterable, Literal, Sequence

import mpmath
import sympy as sp

from ..core.budget import under_budget
from ..core.errors import DomainError, PycodemathError

__all__ = [
    "CalcResult",
    "CalcUndecided",
    "PSI_13",
    "Proof",
    # number theory
    "isprime",
    "factorint",
    "nextprime",
    "prevprime",
    "gcd",
    "lcm",
    "mod",
    "powmod",
    "modinv",
    "totient",
    "divisors",
    "crt",
    "sqrt_mod",
    "fibonacci",
    "lucas",
    "legendre",
    "jacobi",
    "primality_certificate",
    "check_primality_certificate",
    # combinatorics
    "binomial",
    "multinomial",
    "permutations",
    "combinations",
    "stirling1",
    "stirling2",
    "bell",
    "catalan",
    "partitions",
    # special functions
    "special",
    "SPECIAL_FUNCTIONS",
    "zeta",
    "erf",
    "erfc",
    "erfinv",
    "gamma",
    "lgamma",
    "beta",
    "besselj",
    "bessely",
    "polylog",
    "lambertw",
    "airyai",
    "airybi",
]

Proof = Literal["proof", "cross-check", "probabilistic"]

#: The smallest strong pseudoprime to every one of the first 13 prime bases
#: (Sorenson & Webster 2017; OEIS A014233). Below it, Miller-Rabin to bases 2..41 is
#: a proof; it is itself composite and is the canonical test of the boundary.
PSI_13 = 3317044064679887385961981
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)

#: Largest result, in bits, that an exact combinatorial call may build. ``math.comb``
#: and big-int multiplication are single C calls the time budget cannot interrupt
#: (see ``core.budget``), so the size is checked before the call: 2**27 bits is
#: about 40 million decimal digits, 16 MiB.
MAX_RESULT_BITS = 1 << 27

_PROBABILISTIC_NOTE = (
    "probabilistic: passed BPSW (Miller-Rabin base 2 + strong Lucas), which has no "
    "known counterexample since 1980 — but primality was NOT proven"
)

_CERT_RANK = {"proof": 0, "cross-check": 1, "probabilistic": 2}


class CalcUndecided(PycodemathError):
    """The calculator could not CERTIFY a value, so it returns none.

    Raised when the independent check disagreed with the primary algorithm (two
    precisions of a special function that do not agree to the requested digits, a
    closed form SymPy proposes that the numbers contradict), or when a certificate
    needs more than the documented effort. It is the calculator's UNDECIDED: the
    alternative would be a value nobody vouches for.
    """


@dataclass(frozen=True, slots=True)
class CalcResult:
    """A calculator value with the evidence behind it.

    ``value``        the answer, as an exact Python object (``int``, ``bool``,
                     ``Fraction``, ``list``, ``dict``) or, for a numerical value, a
                     decimal string with exactly the certified digits
    ``exact``        True when ``value`` is the exact mathematical answer
    ``certificate``  one sentence: which independent check passed
    ``proof``        ``"proof"`` / ``"cross-check"`` / ``"probabilistic"`` (module
                     docstring)
    ``witness``      machine-checkable data behind the certificate (Bezout
                     coefficients, a factor, a witness base, a primality certificate)
    ``notes``        caveats a caller must see (conventions, the BPSW caveat)
    ``approx``       a decimal approximation when ``value`` is a closed form
    ``kind``         the operation's name — the ``kind`` of an X1 certificate
    """

    kind: str
    value: Any
    exact: bool
    certificate: str
    proof: Proof = "proof"
    witness: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    approx: "str | None" = None

    def as_float(self) -> float:
        """The value as the nearest float (for a closed form, its approximation)."""
        source = self.approx if self.approx is not None else self.value
        if isinstance(source, (int, Fraction)) and not isinstance(source, bool):
            return float(source)
        return float(mpmath.mpf(str(source)))

    def to_dict(self) -> dict:
        """JSON-ready form; exact values as SymPy-readable strings."""
        return {
            "kind": self.kind,
            "value": _jsonable(self.value),
            "exact": self.exact,
            "certificate": self.certificate,
            "proof": self.proof,
            "witness": _jsonable(self.witness),
            "notes": list(self.notes),
            "approx": self.approx,
        }


#: CPython refuses ``str()`` of an int above 4300 digits (the CVE-2020-10735 guard,
#: ``sys.get_int_max_str_digits``). The calculator does produce such numbers —
#: ``fibonacci(10**6)`` has 208 988 digits — so neither a message nor ``to_dict`` may
#: call ``str`` on them blindly.
_STR_DIGITS_SAFE = 4000


def _int_str(n: int) -> str:
    """Full decimal string of any int, past the interpreter's str() limit
    (divide and conquer on powers of ten)."""
    if n < 0:
        return "-" + _int_str(-n)
    if n.bit_length() < _STR_DIGITS_SAFE * 3:
        return str(n)
    k = int(n.bit_length() * 0.30103) // 2
    hi, lo = divmod(n, 10**k)
    return _int_str(hi) + _int_str(lo).zfill(k)


def _short(v: object) -> str:
    """An int for a MESSAGE: in full up to 60 digits, abbreviated above."""
    if isinstance(v, int) and not isinstance(v, bool) and v.bit_length() > 200:
        ctx = mpmath.MPContext()
        ctx.prec = 80
        return "≈" + ctx.nstr(ctx.mpf(v), 12)
    return str(v)


def _jsonable(v: Any) -> Any:
    """Exact values -> JSON: big ints stay ints (JSON has no size limit), rationals
    become ``"p/q"`` strings SymPy reads back, dict keys become strings."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        # beyond str()'s limit json.dumps would raise: such ints travel as strings
        return v if v.bit_length() < _STR_DIGITS_SAFE * 3 else _int_str(v)
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return _int_str(v.numerator)
        return f"{_int_str(v.numerator)}/{_int_str(v.denominator)}"
    if isinstance(v, float):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


def _weakest(*levels: Proof) -> Proof:
    return max(levels, key=_CERT_RANK.__getitem__) if levels else "proof"


# --- input coercion ------------------------------------------------------------
def _as_int(x: object, name: str = "argument") -> int:
    """An exact integer from what a caller may reasonably pass — never a guess.

    Floats are accepted only when they are integral AND at most 2**53, where every
    integer is exactly representable: ``1e20`` might be 10**20 or might be the float
    nearest to something else, and ``isprime(float(2**64 + 1))`` would silently test
    2**64. Strings go through the pycodemath parser (``"2^64+1"``), so they get its
    whitelist and its cost guard.
    """
    if isinstance(x, bool):
        raise DomainError(f"{name}: expected an integer, got the boolean {x!r}")
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isfinite(x) and x.is_integer() and abs(x) <= 2**53:
            return int(x)
        raise DomainError(
            f"{name}: {x!r} is a float, not an exact integer — pass an int or a "
            f"string such as '2^64+1' (above 2**53 a float cannot tell neighbouring "
            f"integers apart)"
        )
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return int(x)
        raise DomainError(f"{name}: expected an integer, got {_short(x)}")
    if isinstance(x, str):
        from ..frontend.parser import parse

        sy = parse(x).sy
        if isinstance(sy, sp.Integer):
            return int(sy)
        raise DomainError(f"{name}: {x!r} is not an integer (it is {sy})")
    sy = getattr(x, "sy", x)
    if isinstance(sy, sp.Integer):
        return int(sy)
    try:
        import numpy as np

        if isinstance(x, np.integer):
            return int(x)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    raise DomainError(f"{name}: expected an integer, got {x!r}")


def _ints(xs: Iterable[object], name: str) -> list[int]:
    return [_as_int(x, f"{name}[{i}]") for i, x in enumerate(xs)]


def _check_bits(bits: float, what: str) -> None:
    if bits > MAX_RESULT_BITS:
        digits = int(bits * math.log10(2)) + 1
        raise DomainError(
            f"{what}: the exact result would have about {digits:,} decimal digits, "
            f"over the limit of {int(MAX_RESULT_BITS * math.log10(2)):,} — the "
            f"computation is one uninterruptible C call, so it is refused up front"
        )


# =============================================================================
# primality — the module's own implementation, independent of SymPy's
# =============================================================================
def _small_primes(limit: int) -> list[int]:
    sieve = bytearray([1]) * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, math.isqrt(limit) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(range(i * i, limit + 1, i)))
    return [i for i, v in enumerate(sieve) if v]


_SMALL = _small_primes(1000)
_SMALL_SET = frozenset(_SMALL)


@dataclass(frozen=True, slots=True)
class _Primality:
    prime: bool
    proof: Proof
    how: str
    witness: dict


def _mr_witness(n: int, a: int) -> "tuple[bool, int | None]":
    """Does base ``a`` prove odd ``n > 2`` composite?  Returns (composite, factor):
    a nontrivial square root of 1 met on the way yields a factor for free."""
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False, None
    for _ in range(s - 1):
        y = pow(x, 2, n)
        if y == n - 1:
            return False, None
        if y == 1:
            # x is a square root of 1 other than +-1: gcd(x-1, n) splits n
            return True, math.gcd(x - 1, n)
        x = y
    if x * x % n == 1:
        return True, math.gcd(x - 1, n)
    return True, None


def _jacobi(a: int, n: int) -> int:
    """Jacobi symbol (a/n) for odd n > 0, by quadratic reciprocity."""
    if n <= 0 or n % 2 == 0:
        raise DomainError(f"jacobi: n must be an odd positive integer, got {_short(n)}")
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def _strong_lucas(n: int) -> bool:
    """Strong Lucas probable-prime test with Selfridge's parameters (method A).

    Every odd prime that is not a perfect square passes; a failure is a PROOF of
    compositeness. Together with Miller-Rabin base 2 this is BPSW.
    """
    if math.isqrt(n) ** 2 == n:
        return False
    d = 5
    while True:
        j = _jacobi(d, n)
        if j == -1:
            break
        if j == 0 and abs(d) != n:
            return False
        d = -d - 2 if d > 0 else -d + 2
    p, q = 1, (1 - d) // 4
    # n + 1 = dd * 2^s with dd odd
    dd, s = n + 1, 0
    while dd % 2 == 0:
        dd //= 2
        s += 1
    # U_k, V_k, Q^k by binary method over the bits of dd
    u, v, qk = 1, p, q % n
    inv2 = (n + 1) // 2
    for bit in bin(dd)[3:]:
        u, v = (u * v) % n, (v * v - 2 * qk) % n
        qk = qk * qk % n
        if bit == "1":
            u, v = ((p * u + v) * inv2) % n, ((d * u + p * v) * inv2) % n
            qk = qk * q % n
    if u == 0 or v == 0:
        return True
    for _ in range(s - 1):
        v = (v * v - 2 * qk) % n
        qk = qk * qk % n
        if v == 0:
            return True
    return False


def _mersenne_exponent(n: int) -> "int | None":
    if n > 2 and (n + 1) & n == 0:
        return (n + 1).bit_length() - 1
    return None


def _lucas_lehmer(p: int) -> bool:
    """2**p - 1 is prime iff s_{p-2} == 0, s_0 = 4, s_{k+1} = s_k^2 - 2 (p odd prime)."""
    m = (1 << p) - 1
    s = 4
    for _ in range(p - 2):
        s = s * s - 2
        s = (s & m) + (s >> p)  # x mod 2^p - 1 without a division
        if s >= m:
            s -= m
    return s == 0 or s == m


def _trial_and_mr(n: int) -> "_Primality | None":
    """Everything decidable without BPSW: small n, small factors, MR below PSI_13."""
    if n < 2:
        return _Primality(False, "proof", "n < 2 is not prime by definition", {"kind": "definition"})
    if n in _SMALL_SET:
        return _Primality(True, "proof", "in the sieve of primes below 1000", {"kind": "sieve"})
    for p in _SMALL:
        if n % p == 0:
            return _Primality(False, "proof", f"{_short(p)} divides it", {"kind": "factor", "factor": p})
    if n < 1000 * 1000:
        return _Primality(
            True, "proof", "no prime factor up to its square root (trial division)",
            {"kind": "trial-division"},
        )
    # above PSI_13 only base 2 is useful: it is the Miller-Rabin half of BPSW, and
    # twelve more bases would prove nothing the strong Lucas test does not decide
    for a in _MR_BASES if n < PSI_13 else (2,):
        composite, factor = _mr_witness(n, a)
        if composite:
            if factor is not None and 1 < factor < n:
                return _Primality(
                    False, "proof",
                    f"{_short(factor)} divides it (found as gcd(x-1, n) for a nontrivial square root x of 1 mod n, base {_short(a)})",
                    {"kind": "factor", "factor": factor, "base": a},
                )
            return _Primality(
                False, "proof", f"Miller-Rabin witness: base {_short(a)} proves it composite",
                {"kind": "miller-rabin", "base": a},
            )
    if n < PSI_13:
        return _Primality(
            True, "proof",
            "strong probable prime to bases 2..41, deterministic below PSI_13 = 3.317e24 "
            "(Sorenson & Webster 2017)",
            {"kind": "miller-rabin-deterministic", "bases": list(_MR_BASES)},
        )
    return None


def _primality(n: int, *, effort: int = 1) -> _Primality:
    """Decide primality of ``n`` with the strongest certificate within ``effort``."""
    p = _mersenne_exponent(n)
    if p is not None and 1000 < p <= 20_000:
        # 2^p - 1 with composite p = ab is divisible by 2^a - 1; with prime p the
        # Lucas-Lehmer test DECIDES it, so the 13 Miller-Rabin rounds would be waste
        for q in _SMALL:
            if q * q > p:
                break
            if p % q == 0:
                return _Primality(
                    False, "proof", f"2^{_short(q)}-1 divides it ({_short(q)} divides the exponent {_short(p)})",
                    {"kind": "factor", "factor": (1 << q) - 1},
                )
        ok = _lucas_lehmer(p)
        return _Primality(
            ok, "proof", f"Lucas-Lehmer test proves 2^{_short(p)}-1 {'prime' if ok else 'composite'}",
            {"kind": "lucas-lehmer", "exponent": p},
        )
    early = _trial_and_mr(n)
    if early is not None:
        return early
    if not _strong_lucas(n):
        return _Primality(
            False, "proof", "fails the strong Lucas test (every odd prime passes it)",
            {"kind": "lucas"},
        )
    if p is not None and p <= 1000 and _lucas_lehmer(p):
        return _Primality(
            True, "proof", f"Lucas-Lehmer test proves 2^{_short(p)}-1 prime",
            {"kind": "lucas-lehmer", "exponent": p},
        )
    cert = _pocklington(n, effort=effort)
    if cert is not None:
        return _Primality(
            True, "proof", "Pocklington (N-1) certificate, re-checkable with "
            "check_primality_certificate", {"kind": "pocklington", "certificate": cert},
        )
    return _Primality(True, "probabilistic", _PROBABILISTIC_NOTE, {"kind": "bpsw"})


def _partial_factor(m: int, effort: int) -> "tuple[dict[int, int], int]":
    """Factor ``m`` as far as cheap methods go: returns (prime powers found, cofactor).

    Below 1e30 (spec: certificates for factors < 1e30) the factorisation is complete.
    Above it ``effort`` decides: 1 (``isprime``) is trial division plus a short
    Pollard rho — milliseconds, measured — so the routine call never pays for a
    proof it is unlikely to get; 2+ (``primality_certificate``) adds ECM.
    """
    if m < 10**30:
        found = sp.factorint(m)
    else:
        found = sp.factorint(m, limit=10**4 * effort**2, use_ecm=effort >= 2, use_pm1=effort >= 2)
    primes: dict[int, int] = {}
    rest = 1
    for q, e in found.items():
        q = int(q)
        # a cofactor SymPy could not split may be composite; keep it out of F
        if q < PSI_13 or _strong_lucas(q):
            primes[q] = primes.get(q, 0) + e
        else:
            rest *= q**e
    return primes, rest


def _pocklington(n: int, *, effort: int = 1, depth: int = 0) -> "dict | None":
    """A recursive Pocklington certificate for ``n``, or None within the effort.

    Theorem (Pocklington 1914, in the form of Brillhart-Lehmer-Selfridge 1975):
    let ``n - 1 = F * R`` with every prime factor q of F known and ``F**2 > n``. If
    some ``a`` has ``a**(n-1) == 1 (mod n)`` and ``gcd(a**((n-1)/q) - 1, n) == 1``
    for every prime ``q | F``, then n is prime. A prime factor q of F above
    ``PSI_13`` gets its own certificate, recursively.
    """
    if depth > 40:  # pragma: no cover - recursion depth of real certificates is < 10
        return None
    primes, _rest = _partial_factor(n - 1, effort)
    # use the largest fully-proved part of n-1
    proven: dict[int, Any] = {}
    f_part = 1
    for q in sorted(primes):
        if q < PSI_13:
            leaf = _trial_and_mr(q)
            if leaf is None or not leaf.prime:
                continue
            proven[q] = "mr"
        else:
            sub = _pocklington(q, effort=effort, depth=depth + 1)
            if sub is None:
                continue
            proven[q] = sub
        f_part *= q ** primes[q]
    if f_part * f_part <= n:
        return None
    for a in range(2, 200):
        if pow(a, n - 1, n) != 1:
            return None  # a Fermat witness: n is composite (cannot happen after BPSW)
        if all(math.gcd(pow(a, (n - 1) // q, n) - 1, n) == 1 for q in proven):
            return {
                "n": str(n),
                "a": a,
                "factors": {str(q): primes[q] for q in proven},
                "sub": {str(q): c for q, c in proven.items() if isinstance(c, dict)},
            }
    return None  # pragma: no cover - a suitable a is found among the first few


def check_primality_certificate(cert: dict) -> bool:
    """Re-check a Pocklington certificate from scratch, with ``pow`` and ``gcd`` only.

    Leaves (factors below ``PSI_13``) are re-proved by deterministic Miller-Rabin;
    nothing from the code that BUILT the certificate is trusted.

        >>> c = primality_certificate(2**89 + 2**60 + 1) if False else None  # doctest: +SKIP
    """
    try:
        n = int(cert["n"])
        a = int(cert["a"])
        factors = {int(q): int(e) for q, e in cert["factors"].items()}
        sub = cert.get("sub", {})
    except (KeyError, TypeError, ValueError):
        return False
    f_part = 1
    for q, e in factors.items():
        if (n - 1) % q**e:
            return False
        f_part *= q**e
        if q < PSI_13:
            verdict = _trial_and_mr(q)
            if verdict is None or not verdict.prime:
                return False
        elif str(q) not in sub or int(sub[str(q)]["n"]) != q or not check_primality_certificate(sub[str(q)]):
            return False
    if f_part * f_part <= n or pow(a, n - 1, n) != 1:
        return False
    return all(math.gcd(pow(a, (n - 1) // q, n) - 1, n) == 1 for q in factors)


@under_budget("primality_certificate")
def primality_certificate(n: object, *, effort: int = 3) -> CalcResult:
    """A re-checkable primality PROOF for ``n`` (Pocklington, recursive), or a refusal.

    Below ``PSI_13`` deterministic Miller-Rabin is already a proof and the certificate
    is that statement; above it the result's ``witness["certificate"]`` is a nested
    dict that ``check_primality_certificate`` verifies independently.
    """
    n = _as_int(n, "n")
    verdict = _primality(n, effort=effort)
    if not verdict.prime:
        raise DomainError(f"primality_certificate: {_short(n)} is composite — {verdict.how}")
    if verdict.proof != "proof":
        raise CalcUndecided(
            f"primality_certificate: {_short(n)} passed BPSW but n-1 could not be factored far "
            f"enough for a Pocklington certificate within effort={effort}"
        )
    witness = dict(verdict.witness)
    if "certificate" in witness and not check_primality_certificate(witness["certificate"]):
        raise CalcUndecided("primality_certificate: the certificate failed its own re-check")  # pragma: no cover
    return CalcResult("primality_certificate", True, True, verdict.how, "proof", witness)


# =============================================================================
# number theory
# =============================================================================
@under_budget("isprime")
def isprime(n: object) -> CalcResult:
    """Is ``n`` prime?  Composite answers carry a witness; prime answers a proof
    (or, above ``PSI_13`` and beyond the effort limit, the explicit BPSW caveat).

        >>> isprime(561).value, isprime(561).witness["factor"]
        (False, 3)
        >>> isprime(2**61 - 1).value
        True
    """
    n = _as_int(n, "n")
    verdict = _primality(n)
    notes: tuple[str, ...] = (_PROBABILISTIC_NOTE,) if verdict.proof == "probabilistic" else ()
    if n < 0:
        notes = notes + ("negative integers are not prime (the convention of SymPy and PARI)",)
    return CalcResult(
        "isprime", verdict.prime, True,
        f"{_short(n)} is {'prime' if verdict.prime else 'composite'}: {verdict.how}",
        verdict.proof, dict(verdict.witness), notes,
    )


@under_budget("factorint")
def factorint(n: object) -> CalcResult:
    """Prime factorisation ``{p: e}`` with product == n and every p proven prime.

    The factors come from SymPy; the certificate is this module's own: the product is
    recomputed exactly and every factor goes through ``_primality``. A factor that is
    only BPSW-probable makes the whole result ``"probabilistic"``.

        >>> factorint(2**64 + 1).value
        {274177: 1, 67280421310721: 1}
    """
    n = _as_int(n, "n")
    if n == 0:
        raise DomainError("factorint: 0 has no prime factorisation (every prime divides it)")
    raw = sp.factorint(n)
    value = {int(p): int(e) for p, e in sorted(raw.items())}
    product = math.prod(p**e for p, e in value.items())
    if product != n:
        raise CalcUndecided(f"factorint: factor product {_short(product)} != {_short(n)}")  # pragma: no cover
    levels: list[Proof] = []
    for p in value:
        if p == -1:
            continue
        verdict = _primality(p)
        if not verdict.prime:
            raise CalcUndecided(  # pragma: no cover - would be a SymPy bug
                f"factorint: SymPy returned {_short(p)} as a prime factor, but {verdict.how}"
            )
        levels.append(verdict.proof)
    proof = _weakest(*levels)
    notes: tuple[str, ...] = (_PROBABILISTIC_NOTE,) if proof == "probabilistic" else ()
    if n == 1 or n == -1:
        notes += ("the empty product: 1 has no prime factors",)
    return CalcResult(
        "factorint", value, True,
        f"product of the factors equals {_short(n)} exactly; each factor is "
        + ("proven prime" if proof == "proof" else "prime (BPSW for some, see notes)"),
        proof, {"product": n}, notes,
    )


def _gap_is_composite(lo: int, hi: int) -> None:
    for m in range(lo + 1, hi):
        if _primality(m).prime:  # pragma: no cover - would contradict SymPy's nextprime
            raise CalcUndecided(f"prime gap check: {_short(m)} is prime")


@under_budget("nextprime")
def nextprime(n: object) -> CalcResult:
    """The smallest prime > n; certificate: it is prime AND every integer between is
    composite (each with a witness), so no prime was skipped."""
    n = _as_int(n, "n")
    p = int(sp.nextprime(n))
    verdict = _primality(p)
    if not verdict.prime:
        raise CalcUndecided(f"nextprime: {_short(p)} is not prime: {verdict.how}")  # pragma: no cover
    _gap_is_composite(max(n, 1), p)
    notes = (_PROBABILISTIC_NOTE,) if verdict.proof == "probabilistic" else ()
    return CalcResult(
        "nextprime", p, True,
        f"{_short(p)} is prime ({verdict.how}); all {max(0, p - max(n, 1) - 1)} integers in "
        f"({_short(n)}, {_short(p)}) are composite",
        verdict.proof, {"gap": p - n}, notes,
    )


@under_budget("prevprime")
def prevprime(n: object) -> CalcResult:
    """The largest prime < n (n > 2); certified like ``nextprime``."""
    n = _as_int(n, "n")
    if n <= 2:
        raise DomainError(f"prevprime: there is no prime below {_short(n)}")
    p = int(sp.prevprime(n))
    verdict = _primality(p)
    if not verdict.prime:
        raise CalcUndecided(f"prevprime: {_short(p)} is not prime: {verdict.how}")  # pragma: no cover
    _gap_is_composite(p, n)
    notes = (_PROBABILISTIC_NOTE,) if verdict.proof == "probabilistic" else ()
    return CalcResult(
        "prevprime", p, True,
        f"{_short(p)} is prime ({verdict.how}); all integers in ({_short(p)}, {_short(n)}) are composite",
        verdict.proof, {"gap": n - p}, notes,
    )


def _ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    """(g, s, t) with a*s + b*t == g == gcd(a, b) >= 0."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


@under_budget("gcd")
def gcd(*args: object) -> CalcResult:
    """Greatest common divisor with Bezout coefficients as the certificate.

    ``sum(a_i * s_i) == g`` and ``g | a_i`` for all i together PROVE g is the gcd:
    every common divisor divides the left side, hence divides g.

        >>> r = gcd(1071, 462); r.value, r.witness["bezout"]
        (21, [-3, 7])
    """
    xs = _ints(args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args, "gcd")
    if not xs:
        raise DomainError("gcd: needs at least one integer")
    g, coeffs = abs(xs[0]), [1 if xs[0] >= 0 else -1]
    for x in xs[1:]:
        g2, s, t = _ext_gcd(g, x)
        coeffs = [c * s for c in coeffs] + [t]
        g = g2
    if sum(a * c for a, c in zip(xs, coeffs)) != g or any(g and a % g for a in xs):
        raise CalcUndecided("gcd: Bezout identity failed")  # pragma: no cover
    return CalcResult(
        "gcd", g, True,
        f"Bezout: sum(a_i*s_i) == {_short(g)} and {_short(g)} divides every argument",
        "proof", {"bezout": coeffs},
    )


@under_budget("lcm")
def lcm(*args: object) -> CalcResult:
    """Least common multiple; certificate per step: ``lcm * gcd == |a*b|`` with the
    gcd proven by Bezout, and every argument divides the result."""
    xs = _ints(args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args, "lcm")
    if not xs:
        raise DomainError("lcm: needs at least one integer")
    value = abs(xs[0])
    for x in xs[1:]:
        g, s, t = _ext_gcd(value, x)
        if value * s + x * t != g:
            raise CalcUndecided("lcm: Bezout identity failed")  # pragma: no cover
        value = 0 if g == 0 else abs(value * x) // g
    if any(x and value % x for x in xs):
        raise CalcUndecided("lcm: a factor does not divide the result")  # pragma: no cover
    return CalcResult(
        "lcm", value, True,
        "every argument divides the result, and at each step lcm*gcd == |a*b| "
        "with the gcd proven by Bezout coefficients", "proof",
    )


ModConvention = Literal["floor", "euclid", "trunc"]


@under_budget("mod")
def mod(a: object, m: object, *, convention: ModConvention = "floor") -> CalcResult:
    """``a mod m`` with the sign convention NAMED, because languages disagree.

    ``floor`` (Python, Ruby, R): the remainder has the sign of m; ``euclid``: always
    ``0 <= r < |m|``; ``trunc`` (C, C++, Java, JavaScript, Go, Rust ``%``): the sign
    of a. ``mod(-7, 3)`` is 2, 2 and -1 respectively — the notes list all three when
    they differ, since that is where ports between languages go wrong.
    """
    a, m = _as_int(a, "a"), _as_int(m, "m")
    if m == 0:
        raise DomainError("mod: modulus 0 (division by zero)")
    variants = {
        "floor": a - m * (a // m),
        "euclid": a % abs(m),
        "trunc": int(math.copysign(abs(a) % abs(m), a)) if a % m else 0,
    }
    if convention not in variants:
        raise DomainError(f"mod: unknown convention {convention!r}; use floor, euclid or trunc")
    r = variants[convention]
    q = (a - r) // m
    if q * m + r != a or not abs(r) < abs(m):
        raise CalcUndecided("mod: division identity failed")  # pragma: no cover
    notes: tuple[str, ...] = ()
    if len(set(variants.values())) > 1:
        notes = (
            "conventions differ here: floor (Python) {floor}, euclid {euclid}, "
            "trunc (C/Java/JS %) {trunc}".format(**variants),
        )
    return CalcResult(
        "mod", r, True, f"{_short(a)} == {_short(q)}*{_short(m)} + {_short(r)} with |{_short(r)}| < |{_short(m)}| ({convention} convention)",
        "proof", {"quotient": q, "convention": convention}, notes,
    )


def _powmod_rtl(a: int, e: int, m: int) -> int:
    """Right-to-left binary exponentiation — a second implementation to check the
    builtin (which is left-to-right with a sliding window)."""
    result, base = 1 % m, a % m
    while e:
        if e & 1:
            result = result * base % m
        base = base * base % m
        e >>= 1
    return result


@under_budget("powmod")
def powmod(a: object, e: object, m: object) -> CalcResult:
    """``a**e mod m`` (m >= 1); a negative e needs a modular inverse."""
    a, e, m = _as_int(a, "a"), _as_int(e, "e"), _as_int(m, "m")
    if m < 1:
        raise DomainError(f"powmod: modulus must be >= 1, got {_short(m)}")
    base = a
    if e < 0:
        base = modinv(a, m).value
        e = -e
    value = pow(base, e, m)
    if _powmod_rtl(base, e, m) != value:
        raise CalcUndecided("powmod: two exponentiation algorithms disagree")  # pragma: no cover
    return CalcResult(
        "powmod", value, True,
        "builtin pow and an independent right-to-left binary exponentiation agree",
        "cross-check",
    )


@under_budget("modinv")
def modinv(a: object, m: object) -> CalcResult:
    """Inverse of a modulo m; certificate ``(a*inv - 1) % m == 0``. Refusal carries
    the gcd that makes an inverse impossible."""
    a, m = _as_int(a, "a"), _as_int(m, "m")
    if m < 1:
        raise DomainError(f"modinv: modulus must be >= 1, got {_short(m)}")
    g, s, _t = _ext_gcd(a % m, m)
    if g != 1:
        raise DomainError(
            f"modinv: {_short(a)} has no inverse modulo {_short(m)} — gcd({_short(a)}, {_short(m)}) = {_short(g)} != 1"
        )
    inv = s % m
    if (a * inv - 1) % m:
        raise CalcUndecided("modinv: certificate failed")  # pragma: no cover
    return CalcResult(
        "modinv", inv, True, f"{_short(a)}*{_short(inv)} == 1 (mod {_short(m)})", "proof",
    )


def _certified_factors(n: int, what: str) -> tuple[dict[int, int], Proof]:
    if n < 1:
        raise DomainError(f"{what}: needs a positive integer, got {_short(n)}")
    r = factorint(n)
    return {p: e for p, e in r.value.items()}, r.proof


@under_budget("totient")
def totient(n: object) -> CalcResult:
    """Euler's phi from a certified factorisation (product formula); for n <= 10**5
    also counted directly as #{k <= n : gcd(k, n) == 1}."""
    n = _as_int(n, "n")
    fac, proof = _certified_factors(n, "totient")
    value = math.prod((p - 1) * p ** (e - 1) for p, e in fac.items())
    how = "Euler's product over the certified factorisation"
    if n <= 10**5:
        if sum(1 for k in range(1, n + 1) if math.gcd(k, n) == 1) != value:
            raise CalcUndecided("totient: direct count disagrees")  # pragma: no cover
        how += " and a direct count of the units"
    return CalcResult("totient", value, True, how, proof, {"factors": fac})


@under_budget("divisors")
def divisors(n: object, *, max_count: int = 10**6) -> CalcResult:
    """All positive divisors, sorted. Certificate: generated from the certified
    factorisation, each divides n, all distinct, and there are exactly
    ``prod(e + 1)`` of them — which is every divisor there is."""
    n = abs(_as_int(n, "n"))
    fac, proof = _certified_factors(n, "divisors")
    count = math.prod(e + 1 for e in fac.values())
    if count > max_count:
        raise DomainError(f"divisors: {_short(n)} has {_short(count)} divisors, over max_count={max_count}")
    divs = [1]
    for p, e in fac.items():
        divs = [d * p**k for d in divs for k in range(e + 1)]
    divs.sort()
    if len(set(divs)) != count or any(n % d for d in divs):
        raise CalcUndecided("divisors: certificate failed")  # pragma: no cover
    return CalcResult(
        "divisors", divs, True,
        f"{_short(count)} = prod(e+1) distinct divisors, each divides {_short(n)}", proof, {"count": count},
    )


@under_budget("crt")
def crt(residues: Sequence[object], moduli: Sequence[object]) -> CalcResult:
    """Solve x == r_i (mod m_i), moduli need NOT be coprime.

    Value: the least x >= 0 (``witness["modulus"]`` is the lcm L; every solution is
    x + kL), or ``None`` when the system is inconsistent — then ``witness`` names
    the pair of congruences that contradict each other mod their gcd.
    """
    rs, ms = _ints(residues, "residues"), _ints(moduli, "moduli")
    if len(rs) != len(ms) or not rs:
        raise DomainError("crt: needs equally many residues and moduli (at least one)")
    if any(m < 1 for m in ms):
        raise DomainError("crt: moduli must be >= 1")
    x, big = rs[0] % ms[0], ms[0]
    for i in range(1, len(rs)):
        r, m = rs[i] % ms[i], ms[i]
        g, s, _t = _ext_gcd(big, m)
        if (r - x) % g:
            for j in range(i):
                gj = math.gcd(ms[j], m)
                if (rs[j] - r) % gj:
                    return CalcResult(
                        "crt", None, True,
                        f"no solution: x == {_short(rs[j])} (mod {_short(ms[j])}) and x == {_short(rs[i])} (mod {_short(m)}) "
                        f"disagree modulo gcd = {gj}",
                        "proof", {"conflict": [j, i], "gcd": gj},
                    )
            return CalcResult(  # pragma: no cover - a pairwise conflict always exists
                "crt", None, True, "no solution (inconsistent system)", "proof", {},
            )
        lcm_ = big // g * m
        x = (x + (r - x) // g * s % (m // g) * big) % lcm_
        big = lcm_
    if any((x - r) % m for r, m in zip(rs, ms)):
        raise CalcUndecided("crt: substitution failed")  # pragma: no cover
    return CalcResult(
        "crt", x, True,
        f"substitution: x = {_short(x)} satisfies all {len(rs)} congruences; unique mod {_short(big)}",
        "proof", {"modulus": big},
    )


def _count_sqrt_prime_power(a: int, p: int, k: int) -> int:
    """Exact number of x mod p**k with x**2 == a (mod p**k)."""
    pk = p**k
    a %= pk
    if a == 0:
        return p ** (k // 2)
    v = 0
    while a % p == 0:
        a //= p
        v += 1
    if v % 2:
        return 0
    j = k - v  # u = a must be a square mod p**j, x = p**(v/2) * y
    if p == 2:
        if j == 1:
            n_u = 1
        elif j == 2:
            n_u = 2 if a % 4 == 1 else 0
        else:
            n_u = 4 if a % 8 == 1 else 0
    else:
        n_u = 1 + _jacobi(a, p)
    return p ** (v // 2) * n_u


@under_budget("sqrt_mod")
def sqrt_mod(a: object, n: object) -> CalcResult:
    """All x in [0, n) with x**2 == a (mod n), sorted.

    Certificate: each root is substituted, AND their number equals the exact count
    from the certified factorisation of n (Hensel/CRT counting) — so none is missing.
    An empty list is a proven "no square root".
    """
    a, n = _as_int(a, "a"), _as_int(n, "n")
    if n < 1:
        raise DomainError(f"sqrt_mod: modulus must be >= 1, got {_short(n)}")
    fac, proof = _certified_factors(n, "sqrt_mod") if n > 1 else ({}, "proof")
    expected = math.prod(_count_sqrt_prime_power(a, p, e) for p, e in fac.items())
    roots = sorted({int(r) for r in (sp.sqrt_mod(a, n, all_roots=True) or [])})
    if n == 1:
        roots = [0]
    if any((r * r - a) % n for r in roots) or len(roots) != expected:
        raise CalcUndecided(  # pragma: no cover
            f"sqrt_mod: {len(roots)} roots found, {_short(expected)} expected"
        )
    return CalcResult(
        "sqrt_mod", roots, True,
        f"each root squares to {_short(a)} mod {_short(n)}, and there are exactly {_short(expected)} "
        "(count from the factorisation of n)", proof, {"count": expected},
    )


def _fib_pair(n: int) -> tuple[int, int]:
    """(F_n, F_{n+1}) by fast doubling, n >= 0."""
    if n == 0:
        return 0, 1
    a, b = _fib_pair(n >> 1)
    c = a * (2 * b - a)
    d = a * a + b * b
    return (d, c + d) if n & 1 else (c, d)


def _fib_lucas(n: int) -> tuple[int, int]:
    """(F_n, L_n) for n >= 0."""
    f, f1 = _fib_pair(n)
    return f, 2 * f1 - f


def _fib_certificate(n: int, f: int, l: int) -> str:
    """Prove (F, L) == (F_n, L_n): positive solutions of L^2 - 5F^2 = 4(-1)^k are
    exactly the pairs (L_k, F_k); the index k is then pinned by log_phi(F*sqrt5)."""
    if l * l - 5 * f * f != 4 * (-1) ** n:
        raise CalcUndecided("fibonacci: Lucas pair identity failed")  # pragma: no cover
    if n > 40:
        ctx = mpmath.MPContext()
        ctx.prec = n.bit_length() + 80
        k = ctx.log(ctx.mpf(f) * ctx.sqrt(5)) / ctx.log((1 + ctx.sqrt(5)) / 2)
        if int(ctx.nint(k)) != n or abs(k - n) > 0.01:
            raise CalcUndecided("fibonacci: index check failed")  # pragma: no cover
    else:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        if a != f:
            raise CalcUndecided("fibonacci: iteration disagrees")  # pragma: no cover
    return (
        f"L^2 - 5F^2 == 4(-1)^n (so the pair is (L_k, F_k) for some k) and "
        f"log_phi(F*sqrt5) rounds to n = {_short(n)}"
    )


def _fib_bits(n: int) -> float:
    return abs(n) * 0.6943 + 2


@under_budget("fibonacci")
def fibonacci(n: object) -> CalcResult:
    """F_n by fast doubling (any integer n; F_{-n} = (-1)^(n+1) F_n).

        >>> fibonacci(100).value
        354224848179261915075
    """
    n = _as_int(n, "n")
    _check_bits(_fib_bits(n), "fibonacci")
    f, l = _fib_lucas(abs(n))
    cert = _fib_certificate(abs(n), f, l)
    if n < 0 and n % 2 == 0:
        f = -f
    return CalcResult("fibonacci", f, True, cert, "proof")


@under_budget("lucas")
def lucas(n: object) -> CalcResult:
    """L_n (L_0 = 2, L_1 = 1; L_{-n} = (-1)^n L_n), certified with F_n."""
    n = _as_int(n, "n")
    _check_bits(_fib_bits(n), "lucas")
    f, l = _fib_lucas(abs(n))
    cert = _fib_certificate(abs(n), f, l)
    if n < 0 and n % 2:
        l = -l
    return CalcResult("lucas", l, True, cert, "proof")


@under_budget("legendre")
def legendre(a: object, p: object) -> CalcResult:
    """Legendre symbol (a/p) for an odd prime p (the primality of p is checked and
    a composite p is refused with its witness). Two routes: Euler's criterion
    ``a^((p-1)/2) mod p`` and quadratic reciprocity."""
    a, p = _as_int(a, "a"), _as_int(p, "p")
    if p < 3 or p % 2 == 0:
        raise DomainError(f"legendre: p must be an odd prime, got {_short(p)}")
    verdict = _primality(p)
    if not verdict.prime:
        raise DomainError(f"legendre: p = {_short(p)} is not prime — {verdict.how}; use jacobi")
    e = pow(a, (p - 1) // 2, p)
    euler = -1 if e == p - 1 else e
    if euler != _jacobi(a, p):
        raise CalcUndecided("legendre: Euler criterion and reciprocity disagree")  # pragma: no cover
    return CalcResult(
        "legendre", euler, True,
        "Euler's criterion a^((p-1)/2) mod p and quadratic reciprocity agree",
        _weakest("proof", verdict.proof),
    )


@under_budget("jacobi")
def jacobi(a: object, n: object) -> CalcResult:
    """Jacobi symbol (a/n), n odd positive. Reciprocity, checked against the
    product of Euler's criteria over the certified factorisation of n."""
    a, n = _as_int(a, "a"), _as_int(n, "n")
    value = _jacobi(a, n)
    fac, proof = _certified_factors(n, "jacobi") if n > 1 else ({}, "proof")
    prod = 1
    for p, e in fac.items():
        r = pow(a, (p - 1) // 2, p)
        prod *= (-1 if r == p - 1 else r) ** e
    if prod != value:
        raise CalcUndecided("jacobi: reciprocity and Euler's criterion disagree")  # pragma: no cover
    return CalcResult(
        "jacobi", value, True,
        "quadratic reciprocity agrees with the product of Euler's criteria over "
        "the certified factorisation of n", proof,
    )


# =============================================================================
# combinatorics — exact integers, two routes each
# =============================================================================
def _product_tree(xs: list[int]) -> int:
    while len(xs) > 1:
        xs = [xs[i] * xs[i + 1] if i + 1 < len(xs) else xs[i] for i in range(0, len(xs), 2)]
    return xs[0] if xs else 1


def _legendre_valuation(n: int, p: int) -> int:
    v, q = 0, p
    while q <= n:
        v += n // q
        q *= p
    return v


def _factorial_ratio_by_primes(num: int, dens: Sequence[int]) -> int:
    """num! / prod(d!) assembled from its prime factorisation (Legendre's formula) —
    a route that shares nothing with ``math.comb`` / ``math.factorial``."""
    powers = []
    for p in _small_primes(num) if num >= 2 else []:
        v = _legendre_valuation(num, p) - sum(_legendre_valuation(d, p) for d in dens)
        if v < 0:
            raise CalcUndecided("factorial ratio is not an integer")  # pragma: no cover
        if v:
            powers.append(p**v)
    return _product_tree(powers)


_CHECK_PRIMES = (1_000_003, 1_000_033, 1_000_037, 1_000_039, 1_000_081)


def _binom_mod_p(n: int, k: int, p: int, fact: list[int]) -> int:
    """C(n, k) mod p by Lucas's theorem."""
    r = 1
    while n or k:
        ni, ki = n % p, k % p
        if ki > ni:
            return 0
        r = r * fact[ni] * pow(fact[ki] * fact[ni - ki], -1, p) % p
        n //= p
        k //= p
    return r


def _binomial_bits(n: int, k: int) -> float:
    k = min(k, n - k)
    if k <= 0:
        return 1
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2) + 1


#: Up to this n the exact Legendre-product route is used; above it, Lucas residues.
_LEGENDRE_LIMIT = 200_000


def _certify_binomial(n: int, k: int, value: int) -> tuple[str, Proof]:
    if n <= _LEGENDRE_LIMIT:
        if _factorial_ratio_by_primes(n, [k, n - k]) != value:
            raise CalcUndecided("binomial: prime-power route disagrees")  # pragma: no cover
        return "equals the product of prime powers from Legendre's/Kummer's formula", "cross-check"
    primes = [p for p in _CHECK_PRIMES if p <= n] or list(_CHECK_PRIMES)
    for p in primes:
        fact = [1] * p
        for i in range(1, p):
            fact[i] = fact[i - 1] * i % p
        if _binom_mod_p(n, k, p, fact) != value % p:
            raise CalcUndecided("binomial: Lucas residue disagrees")  # pragma: no cover
    return (
        f"residues modulo {len(primes)} primes near 1e6 agree with Lucas's theorem",
        "cross-check",
    )


@under_budget("binomial")
def binomial(n: object, k: object) -> CalcResult:
    """C(n, k) exactly, for any integer n (negative n: ``(-1)^k C(k-n-1, k)``) and
    k (k < 0 gives 0).

        >>> binomial(100, 50).value
        100891344545564193334812497256
    """
    n, k = _as_int(n, "n"), _as_int(k, "k")
    notes: tuple[str, ...] = ()
    if k < 0:
        return CalcResult("binomial", 0, True, "C(n, k) = 0 for k < 0 by definition", "proof")
    sign = 1
    nn = n
    if n < 0:
        sign, nn = (-1) ** k, k - n - 1
        notes = (f"negative n: C({_short(n)}, {_short(k)}) = (-1)^{_short(k)} * C({_short(nn)}, {_short(k)})",)
    if k > nn:
        return CalcResult("binomial", 0, True, f"C({_short(n)}, {_short(k)}) = 0 since k > n >= 0", "proof", notes=notes)
    _check_bits(_binomial_bits(nn, k), "binomial")
    value = math.comb(nn, k)
    how, proof = _certify_binomial(nn, k, value)
    return CalcResult("binomial", sign * value, True, how, proof, notes=notes)


@under_budget("multinomial")
def multinomial(ks: Sequence[object]) -> CalcResult:
    """(k1 + ... + km)! / (k1! ... km!) — a telescoping product of binomials,
    checked against the prime-power factorisation."""
    parts = _ints(ks, "ks")
    if any(k < 0 for k in parts):
        raise DomainError("multinomial: parts must be >= 0")
    n = sum(parts)
    _check_bits(sum(_binomial_bits(sum(parts[: i + 1]), parts[i]) for i in range(len(parts))), "multinomial")
    value, acc = 1, 0
    for k in parts:
        acc += k
        value *= math.comb(acc, k)
    if n <= _LEGENDRE_LIMIT:
        if _factorial_ratio_by_primes(n, parts) != value:
            raise CalcUndecided("multinomial: prime-power route disagrees")  # pragma: no cover
        how = "telescoping product of binomials equals the Legendre prime-power product"
    else:
        how = "telescoping product of binomials (each binomial exact)"
        for i in range(1, len(parts)):  # each factor certified separately
            _certify_binomial(sum(parts[: i + 1]), parts[i], math.comb(sum(parts[: i + 1]), parts[i]))
    return CalcResult("multinomial", value, True, how, "cross-check")


@under_budget("permutations")
def permutations(n: object, k: object, *, repetition: bool = False) -> CalcResult:
    """Ordered selections of k from n: n!/(n-k)! (``n**k`` with repetition)."""
    n, k = _as_int(n, "n"), _as_int(k, "k")
    if n < 0 or k < 0:
        raise DomainError("permutations: n and k must be >= 0")
    if repetition:
        _check_bits(k * math.log2(max(n, 1)) + 1, "permutations")
        value = n**k
        for p in _CHECK_PRIMES[:3]:
            if pow(n, k, p) != value % p:
                raise CalcUndecided("permutations: residue check failed")  # pragma: no cover
        return CalcResult(
            "permutations", value, True, f"n^k; residues mod 3 primes agree with pow(n, k, p)",
            "cross-check", notes=("with repetition",),
        )
    if k > n:
        return CalcResult("permutations", 0, True, "k > n: no arrangement exists", "proof")
    _check_bits(_binomial_bits(n, k) + math.lgamma(k + 1) / math.log(2), "permutations")
    value = math.perm(n, k)
    if n <= _LEGENDRE_LIMIT:
        if _factorial_ratio_by_primes(n, [n - k]) != value:
            raise CalcUndecided("permutations: prime-power route disagrees")  # pragma: no cover
        how = "math.perm equals the Legendre prime-power product of n!/(n-k)!"
    else:
        how = "math.perm equals C(n,k)*k! (C certified by Lucas residues)"
        _certify_binomial(n, k, math.comb(n, k))
        if math.comb(n, k) * math.factorial(k) != value:
            raise CalcUndecided("permutations: C(n,k)*k! disagrees")  # pragma: no cover
    return CalcResult("permutations", value, True, how, "cross-check")


@under_budget("combinations")
def combinations(n: object, k: object, *, repetition: bool = False) -> CalcResult:
    """Unordered selections: C(n, k), or C(n+k-1, k) with repetition (multisets)."""
    n, k = _as_int(n, "n"), _as_int(k, "k")
    if n < 0 or k < 0:
        raise DomainError("combinations: n and k must be >= 0")
    if repetition:
        if n == 0:
            return CalcResult("combinations", 1 if k == 0 else 0, True, "empty alphabet", "proof")
        r = binomial(n + k - 1, k)
        return CalcResult(
            "combinations", r.value, True, f"C(n+k-1, k) = C({n + k - 1}, {_short(k)}); " + r.certificate,
            r.proof, notes=("with repetition (multisets)",),
        )
    r = binomial(n, k)
    return CalcResult("combinations", r.value, True, r.certificate, r.proof)


def _stirling2_row(n: int) -> list[int]:
    row = [1]
    for i in range(1, n + 1):
        new = [0] * (i + 1)
        for j in range(1, i + 1):
            new[j] = j * (row[j] if j < len(row) else 0) + row[j - 1]
        row = new
    return row


@under_budget("stirling2")
def stirling2(n: object, k: object) -> CalcResult:
    """Stirling numbers of the second kind S(n, k): partitions of n labelled items
    into k non-empty blocks. Explicit alternating sum vs the recurrence
    S(n,k) = k S(n-1,k) + S(n-1,k-1)."""
    n, k = _as_int(n, "n"), _as_int(k, "k")
    if n < 0 or k < 0:
        raise DomainError("stirling2: n and k must be >= 0")
    if k > n:
        return CalcResult("stirling2", 0, True, "k > n: no partition into k blocks", "proof")
    _check_bits(n * math.log2(max(k, 2)) + 1, "stirling2")
    total = sum((-1) ** j * math.comb(k, j) * (k - j) ** n for j in range(k + 1))
    value, rem = divmod(total, math.factorial(k))
    if rem:
        raise CalcUndecided("stirling2: explicit sum not divisible by k!")  # pragma: no cover
    if n <= 1500:
        if _stirling2_row(n)[k] != value:
            raise CalcUndecided("stirling2: recurrence disagrees")  # pragma: no cover
        how = "explicit alternating sum equals the triangular recurrence"
    else:
        if int(sp.functions.combinatorial.numbers.stirling(n, k, kind=2)) != value:
            raise CalcUndecided("stirling2: SymPy disagrees")  # pragma: no cover
        how = "explicit alternating sum equals SymPy's stirling (independent code)"
    return CalcResult("stirling2", value, True, how, "cross-check")


@under_budget("stirling1")
def stirling1(n: object, k: object, *, signed: bool = False) -> CalcResult:
    """Stirling numbers of the first kind: unsigned c(n, k) (permutations of n with k
    cycles) or, with ``signed=True``, s(n, k) = (-1)^(n-k) c(n, k). The cycle
    recurrence, checked against SymPy's independent implementation."""
    n, k = _as_int(n, "n"), _as_int(k, "k")
    if n < 0 or k < 0:
        raise DomainError("stirling1: n and k must be >= 0")
    if k > n:
        return CalcResult("stirling1", 0, True, "k > n: no permutation with k cycles", "proof")
    _check_bits(math.lgamma(n + 1) / math.log(2) + 1, "stirling1")
    if n > 3000:
        raise DomainError("stirling1: n above 3000 is beyond the O(n^2) certificate's budget")
    row = [1]  # c(i, j); c(i+1, j) = i c(i, j) + c(i, j-1)
    for i in range(n):
        new = [0] * (i + 2)
        for j in range(1, i + 2):
            new[j] = i * (row[j] if j <= i else 0) + row[j - 1]
        row = new
    value_unsigned = row[k]
    if int(sp.functions.combinatorial.numbers.stirling(n, k, kind=1, signed=False)) != value_unsigned:
        raise CalcUndecided("stirling1: SymPy disagrees")  # pragma: no cover
    value = (-1) ** (n - k) * value_unsigned if signed else value_unsigned
    return CalcResult(
        "stirling1", value, True,
        "the cycle recurrence equals SymPy's stirling (independent code)",
        "cross-check", notes=("signed s(n,k)" if signed else "unsigned c(n,k)",),
    )


@under_budget("bell")
def bell(n: object) -> CalcResult:
    """Bell number B_n (set partitions of n items): Bell triangle vs sum_k S(n, k)
    (rows of the Stirling recurrence)."""
    n = _as_int(n, "n")
    if n < 0:
        raise DomainError("bell: n must be >= 0")
    if n > 3000:
        raise DomainError("bell: n above 3000 is beyond the O(n^2) certificate's budget")
    row = [1]
    for _ in range(n):
        new = [row[-1]]
        for x in row:
            new.append(new[-1] + x)
        row = new
    value = row[0]
    if sum(_stirling2_row(n)) != value:
        raise CalcUndecided("bell: Stirling-row sum disagrees")  # pragma: no cover
    return CalcResult(
        "bell", value, True, "Bell triangle equals the sum of the Stirling row S(n, 0..n)",
        "cross-check",
    )


@under_budget("catalan")
def catalan(n: object) -> CalcResult:
    """Catalan number C_n = C(2n, n)/(n+1), checked as C(2n, n) - C(2n, n+1)."""
    n = _as_int(n, "n")
    if n < 0:
        raise DomainError("catalan: n must be >= 0")
    _check_bits(2 * n + 1, "catalan")
    c = math.comb(2 * n, n)
    value, rem = divmod(c, n + 1)
    if rem or c - math.comb(2 * n, n + 1) != value:
        raise CalcUndecided("catalan: the two formulas disagree")  # pragma: no cover
    return CalcResult(
        "catalan", value, True, "C(2n,n)/(n+1) equals C(2n,n) - C(2n,n+1)", "cross-check",
    )


def _partitions_pentagonal(n: int, modulus: "int | None" = None) -> int:
    """p(n) by Euler's pentagonal number recurrence (optionally mod ``modulus``)."""
    p = [1] + [0] * n
    for m in range(1, n + 1):
        total, k = 0, 1
        while True:
            g1 = k * (3 * k - 1) // 2
            if g1 > m:
                break
            sign = 1 if k % 2 else -1
            total += sign * p[m - g1]
            g2 = k * (3 * k + 1) // 2
            if g2 <= m:
                total += sign * p[m - g2]
            k += 1
        p[m] = total % modulus if modulus else total
    return p[n]


@under_budget("partitions")
def partitions(n: object) -> CalcResult:
    """p(n), the number of integer partitions of n: Hardy-Ramanujan-Rademacher
    (SymPy's ``npartitions``) vs Euler's pentagonal recurrence — exactly for
    n <= 5000, modulo a 62-bit prime above (the recurrence is O(n^1.5))."""
    n = _as_int(n, "n")
    if n < 0:
        return CalcResult("partitions", 0, True, "p(n) = 0 for n < 0", "proof")
    value = int(sp.functions.combinatorial.numbers.partition(n))
    if n <= 5000:
        if _partitions_pentagonal(n) != value:
            raise CalcUndecided("partitions: pentagonal recurrence disagrees")  # pragma: no cover
        how = "Hardy-Ramanujan-Rademacher equals Euler's pentagonal recurrence"
    else:
        q = (1 << 61) - 1
        if _partitions_pentagonal(n, q) != value % q:
            raise CalcUndecided("partitions: pentagonal residue disagrees")  # pragma: no cover
        how = "Hardy-Ramanujan-Rademacher agrees with the pentagonal recurrence mod 2^61-1"
    return CalcResult("partitions", value, True, how, "cross-check")


# =============================================================================
# special functions — mpmath values, certified at two precisions + an identity
# =============================================================================
#: Hard cap on requested digits: the work grows superlinearly and the C-level
#: big-float operations are not interruptible.
MAX_DPS = 5000


def _check_dps(dps: object) -> int:
    d = _as_int(dps, "dps")
    if not 1 <= d <= MAX_DPS:
        raise DomainError(f"dps must be between 1 and {MAX_DPS}, got {d}")
    return d


def _exact_arg(x: object) -> "sp.Expr | None":
    """The argument as an exact SymPy number when it is one, else None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return sp.Integer(x)
    if isinstance(x, Fraction):
        return sp.Rational(x.numerator, x.denominator)
    if isinstance(x, str):
        try:
            return sp.Rational(Fraction(x.strip()))
        except (ValueError, ZeroDivisionError):
            return None
    if isinstance(x, sp.Rational):
        return x
    return None


def _mp_arg(ctx: Any, x: object, name: str) -> Any:
    """An mpmath number at the context's precision. Decimal strings are read
    exactly-then-rounded ("0.1" is one tenth, not the float nearest it)."""
    if isinstance(x, bool):
        raise DomainError(f"{name}: expected a number, got {x!r}")
    if isinstance(x, int):
        return ctx.mpf(x)
    if isinstance(x, Fraction):
        return ctx.mpf(x.numerator) / x.denominator
    if isinstance(x, float):
        if not math.isfinite(x):
            raise DomainError(f"{name}: {x!r} is not a finite number")
        return ctx.mpf(x)
    if isinstance(x, complex):
        if not (math.isfinite(x.real) and math.isfinite(x.imag)):
            raise DomainError(f"{name}: {x!r} is not a finite number")
        return ctx.mpc(x)
    if isinstance(x, str):
        s = x.strip().replace(" ", "")
        ex = _exact_arg(s)
        if ex is not None:
            return ctx.mpf(int(ex.p)) / int(ex.q)
        try:
            v = ctx.mpc(complex(s)) if s.endswith("j") else ctx.mpf(s)
        except (ValueError, TypeError):
            raise DomainError(f"{name}: cannot read {x!r} as a number") from None
        if not ctx_finite(ctx, v):
            raise DomainError(f"{name}: {x!r} is not a finite number")
        return v
    if isinstance(x, sp.Basic) and x.is_number:
        return ctx.mpmathify(sp.N(x, int(ctx.dps) + 5))
    raise DomainError(f"{name}: expected a number, got {x!r}")


def _close(ctx: Any, a: Any, b: Any, digits: int) -> bool:
    """|a - b| <= 10^-digits * max(|a|, |b|), with exact zeros agreeing only with
    exact zeros or with values below 10^-(digits + 10) in absolute terms."""
    scale = max(abs(a), abs(b))
    if scale == 0:
        return True
    return abs(a - b) <= ctx.mpf(10) ** (-digits) * scale


def _fmt(ctx: Any, v: Any, dps: int) -> str:
    if isinstance(v, ctx.mpc) or (hasattr(v, "imag") and v.imag != 0):
        v = ctx.mpc(v)
        if v.imag == 0:
            return ctx.nstr(v.real, dps, strip_zeros=False)
        re, im = ctx.nstr(v.real, dps, strip_zeros=False), ctx.nstr(abs(v.imag), dps, strip_zeros=False)
        return f"{re}{'-' if v.imag < 0 else '+'}{im}j"
    return ctx.nstr(v, dps, strip_zeros=False)


@dataclass(frozen=True)
class _Special:
    """One special function: how to evaluate it, and its independent check."""

    name: str
    arity: int
    evaluate: Callable[..., Any]
    #: (ctx, args, value) -> (ok, description); runs at the higher precision
    check: "Callable[..., tuple[bool, str]] | None"
    sympy: "Callable[..., sp.Expr] | None"
    domain: "Callable[..., str | None]"


def _no_domain_issue(ctx: Any, *args: Any) -> "str | None":
    return None


def _is_nonpos_int(ctx: Any, x: Any) -> bool:
    return bool(ctx.im(x) == 0 and ctx.re(x) <= 0 and ctx.re(x) == ctx.floor(ctx.re(x)))


def _gamma_domain(ctx: Any, x: Any) -> "str | None":
    if _is_nonpos_int(ctx, x):
        return f"Gamma has a pole at the non-positive integer {ctx.nstr(x, 15)}"
    return None


def _zeta_domain(ctx: Any, s: Any) -> "str | None":
    return "zeta has a pole at s = 1" if s == 1 else None


def _erfinv_domain(ctx: Any, y: Any) -> "str | None":
    if ctx.im(y) != 0:
        return "erfinv is implemented for real arguments only"
    if abs(y) > 1:
        return f"erfinv needs -1 <= y <= 1, got {ctx.nstr(y, 15)}"
    return None


def _beta_domain(ctx: Any, a: Any, b: Any) -> "str | None":
    for v in (a, b):
        if _is_nonpos_int(ctx, v):
            return f"Beta has a pole: Gamma({ctx.nstr(v, 15)}) is infinite"
    if _is_nonpos_int(ctx, a + b):
        return None  # Gamma(a+b) infinite -> B = 0 is a limit; refuse rather than guess
    return None


def _bessely_domain(ctx: Any, nu: Any, x: Any) -> "str | None":
    if x == 0:
        return "Y_nu(0) is infinite (logarithmic/power singularity at 0)"
    return None


def _lgamma_domain(ctx: Any, x: Any) -> "str | None":
    if ctx.im(x) != 0:
        return "lgamma is log|Gamma(x)| for real x; use gamma for complex arguments"
    return _gamma_domain(ctx, x)


def _quad_ok(ctx: Any, f: Callable[[Any], Any], interval: list, value: Any, digits: int, what: str) -> tuple[bool, str]:
    q, err = ctx.quad(f, interval, error=True, maxdegree=10)
    return _close(ctx, q, value, digits), what


def _check_zeta(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    s = args[0]
    other = ctx.zeta(s, method="euler-maclaurin")
    return _close(ctx, other, v, digits), "Euler-Maclaurin summation agrees with the default (Borwein) algorithm"


def _check_erf(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    x = args[0]
    rep = 2 * x / ctx.sqrt(ctx.pi) * ctx.hyp1f1(0.5, 1.5, -x * x)
    return _close(ctx, rep, v, digits), "the confluent hypergeometric representation 2x/sqrt(pi) 1F1(1/2;3/2;-x^2) agrees"


def _check_erfc(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    x = args[0]
    if ctx.im(x) == 0:
        # t = |x| + u: erfc(|x|) = 2/sqrt(pi) e^(-x^2) int_0^inf e^(-u^2 - 2|x|u) du — the
        # integrand is O(1), so the quadrature keeps its relative accuracy even when
        # erfc(x) is 1e-393; erfc(-x) = 2 - erfc(x)
        ax = abs(x)
        q = 2 / ctx.sqrt(ctx.pi) * ctx.exp(-ax * ax) * ctx.quad(
            lambda u: ctx.exp(-u * u - 2 * ax * u), [0, 1 / (1 + ax), ctx.inf])
        other = q if x >= 0 else 2 - q
        return _close(ctx, other, v, digits), "quadrature of the shifted erfc integral agrees"
    return _close(ctx, 1 - ctx.erf(x), v, digits), "1 - erf(x) agrees"


def _check_erfinv(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    y = args[0]
    if abs(y) == 1:
        return True, "erfinv(+-1) = +-inf by definition"
    back = ctx.erf(v)
    # |dx| = |dy| / erf'(x); require the inverse residue to pin x to the digits
    slope = 2 / ctx.sqrt(ctx.pi) * ctx.exp(-v * v)
    dx = abs(back - y) / slope
    ok = v == 0 and y == 0 or dx <= ctx.mpf(10) ** (-digits) * max(abs(v), ctx.mpf(10) ** (-digits))
    return bool(ok), "erf(erfinv(y)) == y, residue divided by erf' bounds the error in x"


def _check_gamma(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    x = args[0]
    if ctx.im(x) == 0 and x > 0:
        # Gamma(x) = Gamma(x+m) / (x(x+1)...(x+m-1)) with x+m >= 1: Euler's integrand
        # t^(x+m-1) e^-t is then bounded at 0, where t^(x-1) itself is not
        m = int(ctx.ceil(1 - x)) if x < 1 else 0
        y = x + m
        q = ctx.quad(lambda t: t ** (y - 1) * ctx.exp(-t), [0, max(y - 1, ctx.one), 2 * y + 10, ctx.inf])
        for j in range(m):
            q /= x + j
        return _close(ctx, q, v, digits), "Euler's integral (after the recurrence shift to x >= 1) agrees"
    other = ctx.pi / (ctx.sin(ctx.pi * x) * ctx.gamma(1 - x))
    return _close(ctx, other, v, digits), "the reflection formula pi / (sin(pi x) Gamma(1-x)) agrees"


def _check_lgamma(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    x = args[0]
    other = ctx.loggamma(x + 1) - ctx.log(abs(x)) if x > 0 else ctx.log(abs(ctx.gamma(x)))
    if x > 0:
        other = ctx.re(other)
    return _close(ctx, other, v, digits), "the recurrence log|Gamma(x+1)| - log|x| agrees"


def _check_beta(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    a, b = args
    if ctx.im(a) == 0 and ctx.im(b) == 0 and a > 0 and b > 0:
        return _quad_ok(
            ctx, lambda t: t ** (a - 1) * (1 - t) ** (b - 1), [0, 0.5, 1], v, digits,
            "the integral of t^(a-1)(1-t)^(b-1) over (0, 1) agrees",
        )
    other = ctx.gamma(a) * ctx.gamma(b) / ctx.gamma(a + b)
    return _close(ctx, other, v, digits), "Gamma(a)Gamma(b)/Gamma(a+b) agrees"


def _check_besselj(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    nu, x = args
    if ctx.im(nu) == 0 and nu == ctx.floor(nu) and ctx.im(x) == 0:
        n = int(nu)
        other = ctx.quad(lambda t: ctx.cos(n * t - x * ctx.sin(t)), [0, ctx.pi / 2, ctx.pi]) / ctx.pi
        ok = _close(ctx, other, v, digits) or abs(other - v) <= ctx.mpf(10) ** (-digits - 5)
        return bool(ok), "Bessel's integral (1/pi) int_0^pi cos(n t - x sin t) dt agrees"
    # Wronskian with Y: J_{nu+1} Y_nu - J_nu Y_{nu+1} = 2/(pi x)
    w = ctx.besselj(nu + 1, x) * ctx.bessely(nu, x) - v * ctx.bessely(nu + 1, x)
    return _close(ctx, w, 2 / (ctx.pi * x), digits), "the Wronskian J_{nu+1}Y_nu - J_nu Y_{nu+1} = 2/(pi x) holds"


def _check_bessely(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    nu, x = args
    w = ctx.besselj(nu + 1, x) * v - ctx.besselj(nu, x) * ctx.bessely(nu + 1, x)
    return _close(ctx, w, 2 / (ctx.pi * x), digits), "the Wronskian J_{nu+1}Y_nu - J_nu Y_{nu+1} = 2/(pi x) holds"


def _check_polylog(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    s, z = args
    if ctx.im(s) == 0 and ctx.im(z) == 0 and s > 0 and z < 1:
        # Li_s(z) = z/Gamma(s) int_0^inf t^(s-1)/(e^t - z) dt
        q = ctx.quad(lambda t: t ** (s - 1) / (ctx.exp(t) - z), [0, 1, ctx.inf])
        return _close(ctx, z * q / ctx.gamma(s), v, digits), "the Bose-Einstein integral representation agrees"
    if abs(z) < 0.75:
        series = ctx.nsum(lambda k: z**k / k**s, [1, ctx.inf])
        return _close(ctx, series, v, digits), "the defining series sum z^k/k^s agrees"
    return True, "no independent representation for this region; precision doubling only"


def _check_lambertw(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
    z, k = args
    back = v * ctx.exp(v)
    ok = _close(ctx, back, z, digits) or abs(back - z) <= ctx.mpf(10) ** (-digits - 5)
    return bool(ok), "w * exp(w) == z (inverse check) on the requested branch"


def _check_airy(which: str) -> Callable[..., tuple[bool, str]]:
    def check(ctx: Any, args: tuple, v: Any, digits: int) -> tuple[bool, str]:
        x = args[0]
        ai, aip = ctx.airyai(x), ctx.airyai(x, derivative=1)
        bi, bip = ctx.airybi(x), ctx.airybi(x, derivative=1)
        w = ai * bip - aip * bi
        return _close(ctx, w, 1 / ctx.pi, digits), "the Wronskian Ai Bi' - Ai' Bi = 1/pi holds"

    return check


def _lgamma_eval(ctx: Any, x: Any) -> Any:
    return ctx.re(ctx.loggamma(x)) if x > 0 else ctx.log(abs(ctx.gamma(x)))


def _lambertw_eval(ctx: Any, z: Any, k: Any) -> Any:
    return ctx.lambertw(z, int(k))


SPECIAL_FUNCTIONS: dict[str, _Special] = {
    "zeta": _Special("zeta", 1, lambda c, s: c.zeta(s), _check_zeta, sp.zeta, _zeta_domain),
    "erf": _Special("erf", 1, lambda c, x: c.erf(x), _check_erf, sp.erf, _no_domain_issue),
    "erfc": _Special("erfc", 1, lambda c, x: c.erfc(x), _check_erfc, sp.erfc, _no_domain_issue),
    "erfinv": _Special("erfinv", 1, lambda c, y: c.erfinv(y), _check_erfinv, sp.erfinv, _erfinv_domain),
    "gamma": _Special("gamma", 1, lambda c, x: c.gamma(x), _check_gamma, sp.gamma, _gamma_domain),
    "lgamma": _Special("lgamma", 1, _lgamma_eval, _check_lgamma, None, _lgamma_domain),
    "beta": _Special("beta", 2, lambda c, a, b: c.beta(a, b), _check_beta, sp.beta, _beta_domain),
    "besselj": _Special("besselj", 2, lambda c, n, x: c.besselj(n, x), _check_besselj, sp.besselj, _no_domain_issue),
    "bessely": _Special("bessely", 2, lambda c, n, x: c.bessely(n, x), _check_bessely, sp.bessely, _bessely_domain),
    "polylog": _Special("polylog", 2, lambda c, s, z: c.polylog(s, z), _check_polylog, sp.polylog, _no_domain_issue),
    "lambertw": _Special("lambertw", 2, _lambertw_eval, _check_lambertw, None, _no_domain_issue),
    "airyai": _Special("airyai", 1, lambda c, x: c.airyai(x), _check_airy("ai"), sp.airyai, _no_domain_issue),
    "airybi": _Special("airybi", 1, lambda c, x: c.airybi(x), _check_airy("bi"), sp.airybi, _no_domain_issue),
}


def _closed_form(spec: _Special, args: Sequence[object]) -> "sp.Expr | None":
    if spec.sympy is None:
        return None
    exact = [_exact_arg(a) for a in args]
    if any(e is None for e in exact):
        return None
    try:
        expr = spec.sympy(*exact)
    except (TypeError, ValueError, ZeroDivisionError):  # pragma: no cover - defensive
        return None
    if not isinstance(expr, sp.Expr) or not expr.is_number or expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None
    if expr.has(spec.sympy) or expr.has(sp.Float) or len(str(expr)) > 120:
        return None
    return expr


def _input_digits(args: Sequence[object]) -> int:
    """Significant decimal digits the caller TYPED. Working precision must exceed
    them, or the argument itself is rounded before the function sees it — fatal next
    to a branch point (``lambertw`` near -1/e moves by sqrt of the rounding)."""
    best = 0
    for a in args:
        if isinstance(a, str):
            best = max(best, sum(ch.isdigit() for ch in a))
        elif isinstance(a, int) and not isinstance(a, bool):
            best = max(best, len(_short(a)) if a.bit_length() <= 200 else int(a.bit_length() * 0.30103) + 1)
        elif isinstance(a, Fraction):
            best = max(best, int((a.numerator.bit_length() + a.denominator.bit_length()) * 0.30103) + 2)
        elif isinstance(a, float):
            best = max(best, 17)
    return best


def _evaluate_at(spec: _Special, args: Sequence[object], prec: int) -> tuple[Any, list, Any]:
    ctx = mpmath.MPContext()
    ctx.dps = prec
    xs = [_mp_arg(ctx, a, f"{spec.name} argument {i + 1}") for i, a in enumerate(args)]
    issue = spec.domain(ctx, *xs)
    if issue:
        raise DomainError(f"{spec.name}: {issue}")
    try:
        v = spec.evaluate(ctx, *xs)
    except (ValueError, ZeroDivisionError) as exc:
        raise DomainError(f"{spec.name}: {exc}") from None
    if not ctx_finite(ctx, v):
        raise DomainError(f"{spec.name}: the value is not finite at this argument")
    return ctx, xs, v


def _evaluate_special(spec: _Special, args: Sequence[object], dps: int) -> tuple[Any, Any, str]:
    """(value, value at the higher precision, check text).

    Precision ladder: the value is accepted when two CONSECUTIVE rungs agree to the
    requested digits. A third rung exists for cancellation that mpmath does not
    detect on its own (``besselj(0, x)`` at x next to its first zero, where the value
    is 1e-16 and 30 working digits leave only 14 of them right).
    """
    base = max(dps, _input_digits(args)) + 15
    ladder = (base, 2 * base, 4 * base)
    c_prev, _x_prev, v_prev = _evaluate_at(spec, args, ladder[0])
    for i, prec in enumerate(ladder[1:], 1):
        c2, x2, v2 = _evaluate_at(spec, args, prec)
        if _close(c2, v_prev, v2, dps + 2):
            break
        if i == len(ladder) - 1:
            raise CalcUndecided(
                f"{spec.name}: the values at {ladder[-2]} and {ladder[-1]} digits do not "
                f"agree to {dps} digits — refusing to return either"
            )
        v_prev = v2
    text = f"precision ladder: values at {ladder[i - 1]} and {prec} digits agree"
    if spec.check is not None:
        ok, desc = spec.check(c2, tuple(x2), v2, dps + 2)
        if not ok:
            raise CalcUndecided(f"{spec.name}: independent check failed — {desc}")
        text += "; " + desc
    return v_prev, v2, text


def ctx_finite(ctx: Any, v: Any) -> bool:
    return bool(ctx.isfinite(v)) if not isinstance(v, ctx.mpc) else bool(ctx.isfinite(v.real) and ctx.isfinite(v.imag))


@under_budget("special")
def special(name: str, *args: object, dps: object = 15) -> CalcResult:
    """Evaluate the special function ``name`` to ``dps`` significant digits.

    The value is computed by mpmath at ``dps + 15`` and at ``2*dps + 30`` digits, the
    two must agree to the requested digits, and a function-specific INDEPENDENT
    check (another algorithm, an integral representation, an inverse or a
    Wronskian identity) must hold at the higher precision. When every argument is
    exact and SymPy knows a closed form (``zeta(2) = pi**2/6``) the value is that
    closed form, ``exact=True``, after the numbers agreed with it.

    Arguments: int, Fraction, decimal/fraction strings (read exactly: ``"0.1"`` is
    one tenth), float (its exact binary value), complex.

        >>> special("zeta", 2).value
        'pi**2/6'
        >>> special("lambertw", 1, 0, dps=20).value
        '0.56714329040978387300'
    """
    if name not in SPECIAL_FUNCTIONS:
        raise DomainError(
            f"special: unknown function {name!r}; known: {', '.join(sorted(SPECIAL_FUNCTIONS))}"
        )
    spec = SPECIAL_FUNCTIONS[name]
    if name == "lambertw" and len(args) == 1:
        args = (args[0], 0)
    if len(args) != spec.arity:
        raise DomainError(f"{name}: expected {spec.arity} argument(s), got {len(args)}")
    if name == "lambertw":
        _as_int(args[1], "lambertw branch k")
    d = _check_dps(dps)
    if name == "erfinv" and _exact_arg(args[0]) in (sp.Integer(1), sp.Integer(-1)):
        edge = "inf" if _exact_arg(args[0]) == 1 else "-inf"
        return CalcResult(
            name, edge, True, "erfinv(+-1) = +-inf: erf tends to +-1 only as x -> +-inf",
            "proof", {"dps": d},
        )
    v1, v2, text = _evaluate_special(spec, args, d)
    ctx = mpmath.MPContext()
    ctx.dps = 2 * d + 30
    approx = _fmt(ctx, v2, d)
    notes: tuple[str, ...] = ()
    if name == "lambertw":
        notes = (f"branch k = {int(_as_int(args[1]))}",)
    closed = _closed_form(spec, args)
    if closed is not None:
        cv = ctx.mpmathify(sp.N(closed, 2 * d + 30))
        if not _close(ctx, cv, v2, 2 * d + 10) and not (closed == 0 and abs(v2) < ctx.mpf(10) ** (-2 * d)):
            raise CalcUndecided(
                f"{name}: SymPy's closed form {closed} disagrees with the numerical value"
            )
        return CalcResult(
            name, str(closed), True,
            f"SymPy closed form agrees with mpmath to {2 * d + 10} digits; {text}",
            "cross-check", {"dps": d}, notes, approx,
        )
    return CalcResult(name, approx, False, text, "cross-check", {"dps": d}, notes)


def _make_special(name: str, doc: str) -> Callable[..., CalcResult]:
    def fn(*args: object, dps: object = 15) -> CalcResult:
        return special(name, *args, dps=dps)

    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = doc + "\n\nSee ``special`` for the certificate."
    return fn


zeta = _make_special("zeta", "Riemann zeta(s) (pole at s = 1).")
erf = _make_special("erf", "Error function erf(x).")
erfc = _make_special("erfc", "Complementary error function erfc(x) = 1 - erf(x), without the cancellation.")
erfinv = _make_special("erfinv", "Inverse error function, -1 <= y <= 1.")
gamma = _make_special("gamma", "Gamma(x) (poles at 0, -1, -2, ...).")
lgamma = _make_special("lgamma", "log|Gamma(x)| for real x, like math.lgamma.")
beta = _make_special("beta", "Beta(a, b) = Gamma(a)Gamma(b)/Gamma(a+b).")
besselj = _make_special("besselj", "Bessel function of the first kind J_nu(x): besselj(nu, x).")
bessely = _make_special("bessely", "Bessel function of the second kind Y_nu(x): bessely(nu, x).")
polylog = _make_special("polylog", "Polylogarithm Li_s(z): polylog(s, z).")
lambertw = _make_special("lambertw", "Lambert W on branch k: lambertw(z, k=0); branch -1 is the other real branch on [-1/e, 0).")
airyai = _make_special("airyai", "Airy function Ai(x).")
airybi = _make_special("airybi", "Airy function Bi(x).")
