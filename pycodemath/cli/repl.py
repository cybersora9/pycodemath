"""Pycodemath REPL — interactive entry door to the language.

You type math, you get a result. Prefixed commands let you reach
specific engine operations and the code generator.

Commands:
  <expression>            — parse and show the simplified form
  diff <expr> d<var>       — derivative with respect to a variable   (e.g. diff sin(x)*x dx)
  integrate <expr> d<var> [budget <s>] — symbolic integral   (e.g. integrate 2*x dx)
  solve <expr> for <var> [complex] [budget <s>] — solve expr = 0 for var
                                    (real only by default; 'complex' =
                                     complex too; e.g. solve x^2-4 for x)
  limit <expr> for <var> to <p> [dir +|-] [budget <s>] — limit as var → p (p may be oo/-oo)
                                          (e.g. limit sin(x)/x for x to 0)
  series <expr> for <var> [at <p>] [n <k>] [budget <s>] — Taylor/Laurent series around p
                                          (without the O term; e.g. series exp(x) for x n 4)
  sum <expr> for <var> from <lo> to <hi> [budget <s>] — symbolic summation (hi may be n or oo)
                                          (e.g. sum k for k from 1 to n)
  code <expression>        — generate optimized Python/NumPy code

  matrix <A>               — show the matrix            (e.g. matrix [[1,2],[3,4]])
  det <A>                  — determinant               (e.g. det [[1,2],[3,4]])
  inv <A>                  — inverse matrix
  transpose <A>            — transpose
  eig <A>                  — eigenvalues
  solve_system <A> = <b>   — solve A x = b   (e.g. solve_system [[2,1],[1,3]] = [3,5])
  code_system <A> = <b>    — generate NumPy code solving the system

  root <expr> for <var> at <x0>         — root f=0 (e.g. root x^2-2 for x at 1)
  nintegrate <expr> d<var> from <a> to <b> [tol <t>] — numerical integration
                                          (tol = adaptive refinement to that
                                          absolute error; fixed 100-panel
                                          Simpson without it)
                                          (e.g. nintegrate 2*x dx from 0 to 1)
  min <expr> for <var> at <x0> [method newton|bfgs] [tol <t>] [max_iter <k>]
                                        — 1D minimum (gradient descent by
                                          default; newton/bfgs = 2nd-order
                                          methods with Armijo line search)
                                          (e.g. min (x-3)^2 for x at 0 method newton)

  grad <expr> for <x,y,...>            — symbolic gradient (e.g. grad x^2*y for x,y)
  solve_nd <f1>; <f2> for <x,y> at <x0,y0> — nonlinear system via Newton
                                          (e.g. solve_nd x^2+y^2-4; x-y for x,y at 1,1)
  min_nd <expr> for <x,y> at <x0,y0> [method newton|bfgs] [tol <t>] [max_iter <k>]
                                        — minimum of several variables (bfgs
                                          copes where gradient descent
                                          stalls, e.g. the Rosenbrock valley)
                                          (e.g. min_nd (x-1)^2+(y+2)^2 for x,y at 0,0)

  dsolve <expr> for y(t) [budget <s>]   — ODE symbolically: y'(t) = f(t,y)
                                          (e.g. dsolve y for y(t))
  ode <expr> for y(t) from <t0> to <t1> at <y0> [steps <n>]
                                        — ODE numerically (RK4, fixed step)
                                          (e.g. ode y for y(t) from 0 to 1 at 1)
  odestiff <expr> for y(t) from <t0> to <t1> at <y0> [steps <n>]
                                        — STIFF ODE (BDF2, implicit method
                                          + Newton per step; large step where
                                          explicit methods blow up)
  odestiff_adaptive <expr> for y(t) from <t0> to <t1> at <y0> [rtol <r>]
                                        — STIFF ODE with ADAPTIVE step
                                          (variable-step BDF2 + error
                                          estimate; an order fewer steps than
                                          DOPRI5 on a stiff problem)
  ode_adaptive <expr> for y(t) from <t0> to <t1> at <y0> [rtol <r>]
                                        — ODE numerically (Dormand-Prince 5(4),
                                          adaptive step)
  odedense <expr> for y(t) from <t0> to <t1> at <y0> at t=<point> [rtol <r>]
                                        — ODE with dense output: read y(t)
                                          at any point (DOPRI5 interpolant)
  odeevents <expr> for y(t) from <t0> to <t1> at <y0> zero <g> [dir <+1|-1>] [rtol <r>] [stop]
                                        — ODE + event detection: moments
                                          g(t,y)=0 (bisection on the interpolant);
                                          the stop flag = TERMINAL event
                                          (stop integration at the event)

  verify <a> == <b> [budget <s>]        — is a == b an identity? VERIFIED (proved),
                                          REFUTED (with a counterexample) or
                                          UNDECIDED (e.g. verify sqrt(x^2) == x)
  verify steps <derivation> [budget <s>] — check a derivation step by step; the
                                          first wrong step and a counterexample
                                          (one line: a = b = c, or equations
                                          joined by ->; typing just 'verify steps'
                                          reads one step per line until an empty line)
  certify <command>                     — run integrate/diff/solve/limit/dsolve/
                                          nintegrate, then check the answer by an
                                          independent route (e.g. certify integrate
                                          x*cos(x) dx); the command's own budget
                                          also bounds the certificate
  certificate <verify/certify line>     — the same check, written down as JSON
                                          (schema 1: claim, verdict, counterexample,
                                          budget, versions, sha256) — save it with
                                          > file.json (e.g. certificate verify
                                          sqrt(x^2) == x)
  recheck <file.json> [budget <s>]      — re-run a certificate from the file alone:
                                          REPRODUCED, or an error naming the mismatch
                                          (another verdict, a digest broken by a hand
                                          edit, a claim that now parses differently)

CHECK THE MATH IN YOUR CODE (series FULL; every answer ends with a certificate
line — the same check written down, see `certificate` / `recheck`):
  verify code <file.py>[:<f>] == <formula> [where n: integer, n>=0] [--emit-test]
                                        — does the function compute the formula?
                                          REFUTED comes with the call that was really
                                          executed; --emit-test prints the pytest
                                          (e.g. verify code sums.py:sum_to == n*(n+1)/2)
  stability <expr> for x in [a, b] [as float32] [samples N] — bits of float error,
                                          the worst input, the culprit operation
                                          (e.g. stability sqrt(x + 1) - sqrt(x) for x in [1e10, 1e15])
  improve <expr> for x in [a, b] [as float32] [emit python|numpy|c]
                                        — a rewrite PROVED equal and MEASURED better
  ranges <expr> for x in [a, b][ int], y in [c, d] as <dtype>
                                        — overflow / underflow / NaN / lost precision
                                          with an executed witness, or a proof of safety
                                          (e.g. ranges (lo + hi) // 2 for lo in [0, 2^31-1],
                                          hi in [0, 2^31-1] as int32)
  maxsafe <expr> for <x> as <dtype>     — the largest safe input (e.g. maxsafe exp(x) for x as float32)
  iterations <file.py>[:<f>] [== <count>] — how many times the loops run: the exact count,
                                          where it holds, Θ proved by a limit, then counted
                                          by running it; a claimed count REFUTED with the
                                          n where it fails (e.g. iterations sums.py:f == n)
  complexity <file.py>[:<f>] is O(n log n) — O / Theta / Omega / o / omega, decided by a limit
  recurrence T(n) = 2*T(n/2) + n, T(1) = 1 — Θ and the exact form, every condition checked
  master <a> <b> <f(n)>                 — the master theorem, or why it does not apply
  lint <paths> [--select PCM4] [--format sarif] [--no-execute]
                                        — the math-in-code linter (python -m pycodemath.lint):
                                          every finding with an executed witness; exit codes
                                          0 clean / 1 findings / 2 error, as ruff

NUMBERS YOU CAN TRUST (exact answer + proved decimal):
  <exact number>                        — shown with its proved decimal: sqrt(2)/2  ≈ 0.707…
  evalf <expr> <digits>   N(<expr>, <d>) — digits that are PROVED (interval arithmetic) or
                                          marked "NOT rigorous" (e.g. evalf sqrt(2) 50)
  identify <number> [digits <d>] [basis pi,log(3)] — a closed form, only when a coincidence
                                          is excluded (>= 50 digits; e.g. identify zeta(2))
  latex <expr>                          — LaTeX of the parsed expression

SOLUTION SETS AND DEFINITE INTEGRALS (every answer proved):
  solve <f> < <g> [for x [in [a, b]]]   — an inequality: the real line cut at every zero,
                                          each cell's sign proved (e.g. solve x^2-4 > 0 for x)
  solve <eq1>, <eq2> for x, y           — a system, with a completeness proof
  solve <eq> all                        — every real solution: periodic families, period proved
  reduce <relation> [for x]             — the same as solve for relations and systems
  integrate <expr> d<var> from <a> to <b> [pv] [budget <s>]
                                        — definite or improper: convergence proved, the value
                                          cross-checked by quadrature; a divergence proved;
                                          pv = Cauchy principal value, only on request

CALCULATOR (exact; each answer certified):
  factor | isprime | nextprime | prevprime | totient | divisors | fibonacci |
  lucas | bell | catalan | partitions <n>           (e.g. factor 2^64+1)
  gcd | lcm <a> <b> ...;  powmod <a> <e> <m>;  modinv <a> <m>;  sqrtmod <a> <n>;
  legendre | jacobi <a> <n>;  binomial | permutations | combinations |
  stirling1 | stirling2 <n> <k>;  mod <a> <m> [floor|euclid|trunc];
  crt <r1,r2,...> mod <m1,m2,...>       (e.g. crt 2,3,2 mod 3,5,7)
  special <function> <args> [digits <d>] — zeta, gamma, erf, besselj, lambertw...
                                          (e.g. special zeta 3 digits 50)
  mean | median <x1> <x2> ...;  var | std sample|population <x1> ...;
  quantile <q> [type 1..9] <x1> ...
  cdf | sf | pdf | pmf | ppf <distribution> <x> [param=value ...] (e.g. ppf t 0.975 df=10)
  samplesize p1=<p> p2=<p> [power=0.8] [alpha=0.05] [method=evanmiller|pooled|arcsine] [relative]
  power p1= p2= n=;  ci wilson|exact <s>/<n>;  ztest <s1>/<n1> <s2>/<n2>;
  welch [x...] [y...];  holm | bonferroni <p1> <p2> ...

UNITS:
  convert <quantity> to <unit>          — exact (e.g. convert 5 km/h to m/s; the
                                          bare line 5 km/h to m/s works too)
  units <lhs> = <rhs> where F:N m:kg v:m/s [numbers]
                                        — dimensional check: REFUTED with a witness
  dims <expr> where m:kg v:m/s          — the dimension of an expression

MONEY (exact decimals at the currency's ISO 4217 scale):
  money round <amount> <CUR> [mode half_even] [step 0.05]
  money split <amount> <CUR> into <n> | by 1:2:3   (e.g. money split 100 PLN into 3)
  money vat <net1,net2,...> at <rate>% <CUR> — VAT per line vs on the total
  money float <expr> for <x> in <lo>..<hi> [step 0.01] [currency <CUR>]
                                        — which input loses a cent in floats?
  money interest | loan <amount> <CUR> at <rate>% for <n> months|years [equal_principal] [schedule]
  money npv <rate>% <c0,c1,...>;  money irr <c0,c1,...> — every IRR, not one

GEOMETRY:
  quat2mat <w> <x> <y> <z> [order xyzw] [algebra jpl] [normalize]
  euler2mat <a> <b> <c> seq zyx frame intrinsic|extrinsic [deg]
  mat2euler [[...],[...],[...]] seq zyx frame intrinsic [deg]
  rotcheck <expr> == <expr>             — two rotation formulas equal, or integer
                                          quaternions where they differ + the likely bug
  orient2d | orient3d | incircle <points>;  segment p1 p2 q1 q2
  easing <ease|cubic-bezier(...)> <x>;  arclength (x,y) (x,y) ...
  emit <formula> python|numpy|c|glsl;   epsilon <orient2d|...> <epsilon>

  implicit on|off          — REPL setting: off (default, strict) reads a
                             multi-letter name as ONE variable (speed*time);
                             on splits it into letters (2xy = 2*x*y). One-shot:
                             python -m pycodemath --implicit "<command>"
  intro [--static] [--no-logo] [--once]
                           — the presentation: the logo, one live demo, what it
                             can do (shown by itself once per installed version;
                             --once: only if not shown for this version yet)
  commands                 — every command, grouped by topic, one line each
  help <topic>             — the help of one topic: solve, calculus, verify, code,
                             numbers, money, units, geometry, ode, stats, start
  examples <topic>         — 3-5 runnable examples of a topic
  tour                     — five short live demos (Enter = next, q = quit)
  help                     — help
  quit / exit              — exit

Strict parsing (K1): an unknown command word (factor 2^64+1) or an unknown word
next to another term (5 km/h to m/s) is a ParseError with a hint, never a
product of letters; so is a command whose variable shows up in only one of the
two readings (diff xy dx).

Trailing options (module 11): the bracketed ``<key> <value>`` pairs above go at
the END of the command, in any order, each at most once — e.g.
``min x^4 for x at 1 tol 1e-5``, ``integrate 1/(x^5+x+1) dx budget 2``.
  tol <t>       target the answer's accuracy: min/min_nd stop when the gradient
                falls under t (default 1e-9); nintegrate refines panels until
                the estimated absolute error fits inside t (fixed rule without it).
                Must be finite and positive (tol inf would certify the untouched
                starting guess as "converged")
  max_iter <k>  iteration allowance of min/min_nd (default 10000) — a
                not-converged descent that was still progressing finishes with more
  budget <s>    seconds THIS symbolic call may take (default 120) — a smaller
                budget buys a fast typed refusal instead of a long wait; a bigger
                one buys more search. Must be finite and positive here: the
                Python API (pycodemath.time_budget) also accepts math.inf
An unknown or repeated option key, or an unreadable value, is a ParseError; a
readable value outside its domain (tol 0, budget -1) is a DomainError. Options
the engine measurably does not need are deliberately absent — lr (method
newton|bfgs is the fix that works), max_iter/tol on root and solve_nd (Newton
ends its runs itself), max_evals, atol (rtol already steers the ODE error).
"""

from __future__ import annotations

import difflib
import math
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from sympy.utilities.exceptions import SymPyDeprecationWarning

from .. import __version__
from ..codegen.pipeline import generate, generate_system
from ..core.budget import time_budget
from ..core.errors import DomainError, ParseError, PycodemathError
from ..core.ir import Matrix
from ..core.result import QuadratureResult, SolveResult
from ..engine import linalg, numerics, ode, symbolic
from ..frontend.parser import KNOWN_NAMES, is_strict, parse, parse_mode, set_strict
from . import full
from ..verify import (
    Certificate,
    CertificateMismatch,
    StepsResult,
    Verdict,
    VerdictStatus,
    certify_diff,
    certify_dsolve,
    certify_integrate,
    certify_limit,
    certify_nintegrate,
    certify_solve,
    check_equal,
    check_steps,
    from_verdict,
    recheck,
)

_T = TypeVar("_T")

# One version string in the project (``pycodemath.__version__``), interpolated
# here — a banner that has to be bumped by hand is a banner that eventually lies.
BANNER = f"Pycodemath {__version__} — heavy math + Python. Type 'help' or 'quit'."


def _num(raw: str) -> float:
    """Turn a number token from a command into a ``float`` — bad input is a clear error.

    The ``_NUM`` pattern is intentionally loose (it lets through e.g. ``1e``), so
    correctness is enforced only here — with a readable message, instead of the raw
    ``ValueError`` from ``float()`` (contract: always PycodemathError).
    """
    try:
        return float(raw)
    except ValueError as exc:
        raise ParseError(f"invalid number: {raw!r}") from exc


# --- trailing options (module 11) -----------------------------------------
# The knobs the Python API always had, reachable from the grammar at last —
# modules 5-10 deferred exactly this six times. Which knobs made it in and which
# stayed Python-only was decided by MEASUREMENT, not taste (module 11 journal):
#
#   IN   tol/max_iter on min and min_nd — gradient descent is the one iterator
#        whose honest failure is "ran out of iterations while still progressing";
#        measured rescues: `min x^4 for x at 1` converges at 678 594 iterations
#        and under tol 1e-5 at 1 456, the Rosenbrock valley at 42 826.
#   IN   tol on nintegrate — the only way to AIM module 5's error_estimate;
#        measured: sqrt(x) true error 8.1e-5 -> 2e-14, and a spike integral the
#        fixed grid gets 37x wrong comes back correct to 1e-13.
#   IN   budget on the six symbolic commands — TimeBudgetError's own message
#        says "raise the budget" and the wire had no way to obey it; measured
#        in both directions (a 5 s budget refuses a 13 s integral that a 60 s
#        budget answers; the true hang is bounded in 3 s instead of 120).
#   OUT  lr — measured on the Rosenbrock valley and 1000*x^2: it only trades one
#        non-convergence for another; `method newton|bfgs` (already reachable)
#        is the fix that works.
#   OUT  max_iter/tol on root and solve_nd — Newton self-terminates (quadratic
#        convergence, or the divergence/stagnation detectors, or the bisection
#        fallback); measured: max_iter 100 000 wasted every extra iteration on
#        the x^3-2x+2 cycle and MANUFACTURED a false root of x*exp(-x) at
#        x=745 where f underflows to exact zero. A knob whose only measured
#        effects are waste and harm stays out.
#   OUT  max_evals — unexhaustible on every probed integrand (the worst took
#        83 033 of the 100 000 default); atol — no outcome moved on the
#        crossing-zero probe, rtol (already reachable) steers the error;
#        n on nintegrate — dominated by tol at 21x the cost on the same input.
#
# Syntax: ``key value`` pairs at the END of the command, ANY order, each at most
# once — the shape `method newton` and `rtol 1e-8` already have, generalized.
# A malformed option is never silently ignored (module 2): an unknown key, a
# repeated key or an unreadable value is a ParseError (the LINE could not be
# read — no engine was consulted); a readable number outside its domain is a
# DomainError (the same class the engines themselves use for a bad value).
_OPTS = r"(?P<opts>(?:\s+[A-Za-z_]+\s+\S+)*)"


def _opt_tol(raw: str) -> float:
    # math.isfinite matters here, not just decoration: ``tol inf`` passed the old
    # ``not value > 0.0`` check (inf > 0.0 is True) and made the gradient exit test
    # (``res < tol``) trivially true on iteration ONE — ``min (x-3)^2 for x at 100
    # tol inf`` came back ``SolveResult(value=100.0, residual=194.0,
    # converged=True)``: the untouched starting guess, certified converged. The
    # same class of bad input ``_opt_budget`` already guards against below.
    value = _num(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise DomainError(
            f"the tolerance tol must be a positive finite number, got {raw}"
        )
    return value


def _opt_max_iter(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ParseError(f"invalid integer: {raw!r}") from exc
    if value < 1:
        raise DomainError(f"max_iter must be at least 1, got {raw}")
    return value


def _opt_budget(raw: str) -> float:
    # Stricter than ``time_budget`` on purpose: the Python API accepts math.inf
    # for a caller who can interrupt, but this grammar IS the wire, and module 9's
    # guarantee there is precisely that no call hangs forever.
    value = _num(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise DomainError(
            f"the budget must be a positive finite number of seconds, got {raw} "
            f"(the Python API — pycodemath.time_budget — accepts math.inf for a "
            f"caller who would rather wait forever)"
        )
    return value


_MIN_OPTS: dict[str, Callable[[str], object]] = {
    "tol": _opt_tol,
    "max_iter": _opt_max_iter,
}
_NINTEGRATE_OPTS: dict[str, Callable[[str], object]] = {"tol": _opt_tol}
_BUDGET_OPTS: dict[str, Callable[[str], object]] = {"budget": _opt_budget}


def _options(m: re.Match, allowed: dict[str, Callable[[str], object]]) -> dict[str, Any]:
    """Parse a command's trailing options into keyword arguments.

    The pattern (``_OPTS``) guarantees whole ``key value`` pairs, so a dangling
    key (``min ... tol``) never reaches here — it fails the pattern and earns the
    usage message, exactly as a half-formed ``method`` suffix always has. What
    DOES reach here is decided strictly: the keys are this command's whitelist
    and nothing else, because an option that is accepted and ignored would be the
    grammar lying about what it computed.
    """
    raw = (m.groupdict().get("opts") or "").split()
    parsed: dict[str, Any] = {}
    for key, value in zip(raw[0::2], raw[1::2]):
        key = key.lower()
        if key not in allowed:
            raise ParseError(
                f"unknown option {key!r} — this command takes: "
                + ", ".join(f"{k} <value>" for k in allowed)
            )
        if key in parsed:
            raise ParseError(f"option {key!r} given twice")
        parsed[key] = allowed[key](value)
    return parsed


def _budgeted(m: re.Match, run: Callable[[], _T]) -> _T:
    """Run a symbolic handler under the caller's ``budget <s>``, if any.

    ``time_budget`` REPLACES the 120 s default for everything inside the block
    (module 9's re-entrancy: the entry point's own guard inherits the caller's
    allowance instead of arming a second one), so the refusal quotes the number
    the command named.
    """
    budget = _options(m, _BUDGET_OPTS).get("budget")
    if budget is None:
        return run()
    with time_budget(budget):
        return run()


# --- parsing a command's expression (K1) -------------------------------------
def _parse_for(texts: "Sequence[str]", names: "Sequence[str]") -> list:
    """Parse ``texts`` and refuse when a command variable is visible in only ONE
    of the two readings of a multi-letter name.

    Strict mode reads ``xy`` as one variable, implicit mode as ``x*y``. Either
    reading alone can turn a command into a silent zero: ``diff xy dx`` is 0 in
    strict mode, ``diff speed*time dtime`` was 0 with splitting (the worst of the
    30.10 probes). When the variable is absent from the expression as parsed but
    present under the other reading, the line is ambiguous — a ``ParseError``
    naming the setting that changes it, not an answer.
    """
    exprs = [parse(t) for t in texts]
    present = {str(sym) for e in exprs for sym in e.sy.free_symbols}
    for name in names:
        if name in present:
            continue
        strict = is_strict()
        try:
            with parse_mode(strict=not strict):
                other = {str(sym) for t in texts for sym in parse(t).sy.free_symbols}
        except PycodemathError:
            continue
        if name not in other:
            continue
        if strict:
            raise ParseError(
                f"'{name}' does not appear in {' ; '.join(texts)!r}: multi-letter "
                f"names are read as ONE variable (strict parsing) — write the "
                f"product with '*' (e.g. x*y), or switch 'implicit on' to split names"
            )
        raise ParseError(
            f"'{name}' does not appear in {' ; '.join(texts)!r}: with 'implicit on' "
            f"multi-letter names are split into letters — switch 'implicit off' "
            f"to keep '{name}' one variable"
        )
    return exprs


def _expr(m: re.Match) -> Any:
    """The command's ``expr`` group, parsed and checked against its variable(s)."""
    groups = m.groupdict()
    names = [groups["var"]] if groups.get("var") else []
    if groups.get("vars"):
        names = _split_names(groups["vars"])
    return _parse_for([m["expr"]], names)[0]


# --- command handlers (receive the regex match, return text) --------
def _cmd_diff(m: re.Match) -> str:
    return str(symbolic.diff(_expr(m), m["var"]))


def _cmd_integrate(m: re.Match) -> str:
    return _budgeted(m, lambda: str(symbolic.integrate(_expr(m), m["var"])))


def _cmd_solve(m: re.Match) -> str:
    # The REPL cuts complex results by default: transcendental equations (e.g. cos(x)=2)
    # were spewing junk complex roots. The trailing 'complex' flag
    # restores the full set of roots.
    def run() -> str:
        real = not bool(m["complex"])
        roots = symbolic.solve(_expr(m), m["var"], real=real)
        return ", ".join(str(r) for r in roots) if roots else "(no solutions)"

    return _budgeted(m, run)


def _cmd_limit(m: re.Match) -> str:
    return _budgeted(
        m,
        lambda: str(
            symbolic.limit(
                _expr(m), m["var"], parse(m["to"]), dir=m["dir"] or "+"
            )
        ),
    )


def _cmd_series(m: re.Match) -> str:
    def run() -> str:
        at = parse(m["at"]) if m["at"] else 0
        n = int(m["n"]) if m["n"] else 6
        return str(symbolic.series(_expr(m), m["var"], at=at, n=n))

    return _budgeted(m, run)


def _cmd_sum(m: re.Match) -> str:
    return _budgeted(
        m,
        lambda: str(
            symbolic.summation(
                _expr(m), m["var"], parse(m["lo"]), parse(m["hi"])
            )
        ),
    )


def _cmd_code(m: re.Match) -> str:
    return generate(_expr(m)).source


def _cmd_matrix(m: re.Match) -> str:
    return str(Matrix(m["a"]))


def _cmd_det(m: re.Match) -> str:
    return str(linalg.det(Matrix(m["a"])))


def _cmd_inv(m: re.Match) -> str:
    return str(linalg.inv(Matrix(m["a"])))


def _cmd_transpose(m: re.Match) -> str:
    return str(linalg.transpose(Matrix(m["a"])))


def _cmd_eig(m: re.Match) -> str:
    vals = linalg.eigenvalues(Matrix(m["a"]))
    return ", ".join(str(v) for v in vals) if vals else "(none)"


def _cmd_solve_system(m: re.Match) -> str:
    return str(linalg.solve_system(Matrix(m["a"]), Matrix(m["b"])))


def _cmd_code_system(m: re.Match) -> str:
    return generate_system(Matrix(m["a"]), Matrix(m["b"])).source


# The five numerical commands are answered TWICE — once by the plain handler the
# REPL uses and once by the evidence handler the MCP surface uses (module 6). The
# two must print the SAME text for the same run, so the text is built here and
# nowhere else; ``test_mcp`` pins the equality mechanically over every one of them.
def _fmt_scalar(value: float) -> str:
    return str(value)


def _fmt_named(names: "Sequence[str]", values: "Sequence[float]") -> str:
    return ", ".join(f"{v} = {val}" for v, val in zip(names, values))


def _cmd_root(m: re.Match) -> str:
    return _fmt_scalar(numerics.root_find(_expr(m), m["var"], _num(m["x0"])))


def _cmd_min(m: re.Match) -> str:
    method = (m["method"] or "gd").lower()
    return _fmt_scalar(
        numerics.minimize(
            _expr(m),
            m["var"],
            _num(m["x0"]),
            method=method,
            **_options(m, _MIN_OPTS),
        )
    )


def _cmd_nintegrate(m: re.Match) -> str:
    val = numerics.integrate_num(
        _expr(m),
        m["var"],
        _num(m["a"]),
        _num(m["b"]),
        **_options(m, _NINTEGRATE_OPTS),
    )
    return _fmt_scalar(val)


def _split_names(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",")]


def _split_floats(raw: str) -> list[float]:
    return [_num(v.strip()) for v in raw.split(",")]


def _cmd_grad(m: re.Match) -> str:
    return str(numerics.gradient(_expr(m), _split_names(m["vars"])))


def _cmd_solve_nd(m: re.Match) -> str:
    names = _split_names(m["vars"])
    exprs = _parse_for([e.strip() for e in m["exprs"].split(";")], names)
    sol = numerics.root_find_nd(exprs, names, _split_floats(m["x0"]))
    return _fmt_named(names, sol)


def _cmd_min_nd(m: re.Match) -> str:
    names = _split_names(m["vars"])
    method = (m["method"] or "gd").lower()
    sol = numerics.minimize_nd(
        _expr(m),
        names,
        _split_floats(m["x0"]),
        method=method,
        **_options(m, _MIN_OPTS),
    )
    return _fmt_named(names, sol)


def _cmd_dsolve(m: re.Match) -> str:
    def run() -> str:
        sols = ode.dsolve(_expr(m), m["func"], m["var"])
        return ", ".join(f"{m['func']}({m['var']}) = {s}" for s in sols)

    return _budgeted(m, run)


def _cmd_ode(m: re.Match) -> str:
    n = int(m["n"]) if m["n"] else 100
    ts, ys = ode.solve_ode_num(
        _expr(m),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        n,
        func=m["func"],
    )
    return f"{m['func']}({ts[-1]:g}) = {ys[-1]}"


def _cmd_ode_stiff(m: re.Match) -> str:
    n = int(m["n"]) if m["n"] else 100
    ts, ys = ode.solve_ode_stiff(
        _expr(m),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        n,
        func=m["func"],
    )
    return f"{m['func']}({ts[-1]:g}) = {ys[-1]}  (BDF2, {n} implicit steps)"


def _cmd_ode_stiff_adaptive(m: re.Match) -> str:
    rtol = _num(m["rtol"]) if m["rtol"] else 1e-6
    ts, ys = ode.solve_ode_stiff_adaptive(
        _expr(m),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        rtol=rtol,
        func=m["func"],
    )
    return (
        f"{m['func']}({ts[-1]:g}) = {ys[-1]}"
        f"  (adaptive BDF, {len(ts) - 1} implicit steps)"
    )


def _cmd_ode_adaptive(m: re.Match) -> str:
    rtol = _num(m["rtol"]) if m["rtol"] else 1e-6
    ts, ys = ode.solve_ode_adaptive(
        _expr(m),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        rtol=rtol,
        func=m["func"],
    )
    return f"{m['func']}({ts[-1]:g}) = {ys[-1]}  ({len(ts) - 1} adaptive steps)"


def _cmd_ode_dense(m: re.Match) -> str:
    rtol = _num(m["rtol"]) if m["rtol"] else 1e-6
    sol = ode.solve_ode_dense(
        _expr(m),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        rtol=rtol,
        func=m["func"],
    )
    tq = _num(m["tq"])
    return f"{m['func']}({tq:g}) = {sol(tq)}  (dense output from {len(sol.ts) - 1} steps)"


def _cmd_ode_events(m: re.Match) -> str:
    direction = int(m["dir"]) if m["dir"] else 0
    rtol = _num(m["rtol"]) if m["rtol"] else 1e-6
    terminal = bool(m["stop"])
    sol = ode.solve_ode_events(
        _expr(m),
        m["var"],
        _num(m["y0"]),
        (_num(m["t0"]), _num(m["t1"])),
        parse(m["g"]),
        direction=direction,
        terminal=terminal,
        rtol=rtol,
        func=m["func"],
    )
    if not sol.events:
        return "(no events)"
    listed = "; ".join(f"t={t:g} ({m['func']}={y:g})" for t, y in sol.events)
    if terminal:
        return f"stop at {listed}  (integration stopped at the event)"
    return listed


# --- verify and certify (VERIFY V4) ----------------------------------------
# The verifier's three words reach the grammar. Nothing here decides anything:
# ``check_equal``, ``check_steps`` and the ``certify_*`` functions keep their
# semantics exactly (principal branch, equality where both sides are defined,
# decimals as written, UNDECIDED never rounded up) — this layer only reads the
# line, calls them, and renders the verdict. The verdict OBJECTS travel with the
# text (``Checked``) so the MCP surface can put them on the wire as data.
#
# ``budget`` is the one option, and it is matched by NAME, not by ``_OPTS``'s
# generic ``key value`` shape: an expression may end in juxtaposed names
# (``verify x*y*z == x y z`` — the parser reads ``x y z`` as a product), and the
# generic shape would read ``y z`` there as an option ``y`` with the value ``z``.
_VERIFY_OPTS = r"(?P<opts>(?:\s+budget\s+\S+)?)"


@dataclass(frozen=True, slots=True)
class Checked:
    """One ``verify`` / ``certify`` line answered: the text, plus the verdicts.

    ``verdict`` is set for ``verify <a> == <b>`` and ``certify`` (the certificate),
    ``steps`` for ``verify steps``, ``answer`` for ``certify`` (the engine's answer
    that was certified, as ``handle`` prints it). ``certificate`` (X1) is the same
    check written down for ``recheck`` — built from the verdict already in hand,
    nothing is re-run. All four are ``None`` on a usage message — the line did not
    fit the grammar, so nothing was checked.
    """

    text: str
    verdict: "Verdict | None" = None
    steps: "StepsResult | None" = None
    answer: "str | None" = None
    certificate: "Certificate | None" = None


def _verdict_line(v: Verdict) -> str:
    # ``detail`` already names the point of a refutation ("at x = -1: ...").
    return f"{v.status.name} ({v.method}) — {v.detail}"


def _point(counterexample: "dict[str, str]") -> str:
    return ", ".join(f"{k} = {v}" for k, v in counterexample.items())


def _steps_text(result: StepsResult) -> str:
    lines = []
    for c in result.checks:
        lines.append(f"step {c.number}: {_verdict_line(c.verdict)}")
        if c.warning:
            lines.append(f"  warning: {c.warning}")
    status = result.status.name
    if result.first_error is not None:
        where = _point(result.counterexample or {})
        summary = f"first wrong step: {result.first_error}" + (
            f" (counterexample {where})" if where else ""
        )
    elif result.status is VerdictStatus.VERIFIED:
        summary = f"all {len(result.checks)} steps proved"
    else:
        open_ = sum(1 for c in result.checks if not c.verdict.verified)
        summary = f"no step refuted, {open_} of {len(result.checks)} not proved"
    lines.append(f"derivation: {status} — {summary}")
    return "\n".join(lines)


def _source(m: re.Match) -> str:
    return " ".join(m.string.split()) if "\n" not in m.string else m.string.strip()


def _written_down(*args: Any, **kwargs: Any) -> "Certificate | None":
    """``from_verdict``, or ``None`` when the claim cannot be recorded — the
    certificate is a by-product, and it never costs the line its verdict."""
    try:
        return from_verdict(*args, **kwargs)
    except PycodemathError:
        return None


def _check_verify(m: re.Match) -> Checked:
    if m["code"] is not None:  # C1: a function in a file against a formula (I1)
        done = full.check_code_line(m["code"] + (m["opts"] or ""), source_line=_source(m))
        return Checked(done.text, verdict=done.verdict, certificate=done.certificate)
    budget = _options(m, _BUDGET_OPTS).get("budget")
    if m["steps"] is not None:
        result = check_steps(m["steps"], budget=budget)
        cert = _written_down("steps", [m["steps"]], result, source=_source(m), budget=budget)
        return Checked(_steps_text(result), steps=result, certificate=cert)
    verdict = check_equal(m["a"], m["b"], budget=budget)
    cert = _written_down(
        "identity", [m["a"], m["b"]], verdict, source=_source(m), budget=budget
    )
    return Checked(_verdict_line(verdict), verdict=verdict, certificate=cert)


# Each certifier runs the SAME engine call its plain ``_cmd_*`` twin makes and
# returns (the answer as ``handle`` prints it, the certificate). The command's own
# ``budget`` (on the four commands that take one) bounds the engine call as always
# and, SEPARATELY, the certificate — elsewhere the certificate gets the default:
# nesting the certificate inside the command's ``time_budget`` would let its expiry
# refuse the whole line and lose an answer already computed.
def _cert_budget(m: re.Match) -> "float | None":
    return _options(m, _BUDGET_OPTS).get("budget")


@dataclass(frozen=True, slots=True)
class _Certified:
    """A certifier's answer, its verdict, and the claim as a certificate records
    it (X1): the kind, the arguments in the kind's order, the options, the budget."""

    answer: str
    verdict: Verdict
    kind: str
    args: "list[Any]"
    options: "dict[str, Any]"
    budget: "float | None"


def _cert_integrate(m: re.Match) -> _Certified:
    expr, budget = _expr(m), _cert_budget(m)
    result = _budgeted(m, lambda: symbolic.integrate(expr, m["var"]))
    verdict = certify_integrate(expr, m["var"], result, budget=budget)
    return _Certified(str(result), verdict, "integral",
                      [m["expr"], m["var"], result], {}, budget)


def _cert_diff(m: re.Match) -> _Certified:
    expr = _expr(m)
    result = symbolic.diff(expr, m["var"])
    verdict = certify_diff(expr, m["var"], result)
    return _Certified(str(result), verdict, "derivative",
                      [m["expr"], m["var"], result], {}, None)


def _cert_solve(m: re.Match) -> _Certified:
    expr, real, budget = _expr(m), not bool(m["complex"]), _cert_budget(m)
    roots = _budgeted(m, lambda: symbolic.solve(expr, m["var"], real=real))
    text = ", ".join(str(r) for r in roots) if roots else "(no solutions)"
    verdict = certify_solve(expr, m["var"], roots, real=real, budget=budget)
    return _Certified(text, verdict, "roots",
                      [m["expr"], m["var"], list(roots)], {"real": real}, budget)


def _cert_limit(m: re.Match) -> _Certified:
    expr, to, side = _expr(m), parse(m["to"]), m["dir"] or "+"
    budget = _cert_budget(m)
    result = _budgeted(m, lambda: symbolic.limit(expr, m["var"], to, dir=side))
    verdict = certify_limit(expr, m["var"], to, result, dir=side, budget=budget)
    return _Certified(str(result), verdict, "limit",
                      [m["expr"], m["var"], m["to"], result], {"dir": side}, budget)


def _cert_dsolve(m: re.Match) -> _Certified:
    expr, budget = _expr(m), _cert_budget(m)
    sols = _budgeted(m, lambda: ode.dsolve(expr, m["func"], m["var"]))
    text = ", ".join(f"{m['func']}({m['var']}) = {s}" for s in sols)
    verdict = certify_dsolve(expr, m["func"], m["var"], sols, budget=budget)
    return _Certified(text, verdict, "ode",
                      [m["expr"], m["func"], m["var"], list(sols)], {}, budget)


def _cert_nintegrate(m: re.Match) -> _Certified:
    # The certificate checks the claim the run MADE: its own ``error_estimate``
    # (the engine documents that estimate as able to lie — this is what catches
    # it). Where the evidence cannot be had (``handle_full``'s fallback case) the
    # bare value is held to ``certify_nintegrate``'s default relative tolerance.
    expr, a, b = _expr(m), _num(m["a"]), _num(m["b"])
    claim: "QuadratureResult | float"
    try:
        answer = _ev_nintegrate(m)
        assert isinstance(answer.evidence, QuadratureResult)
        text, claim = answer.text, answer.evidence
    except DomainError:
        text = _cmd_nintegrate(m)
        claim = float(text)
    verdict = certify_nintegrate(expr, m["var"], a, b, claim)
    if isinstance(claim, QuadratureResult):
        value, options = claim.value, {"error_estimate": claim.error_estimate}
    else:
        value, options = claim, {}
    return _Certified(text, verdict, "quadrature",
                      [m["expr"], m["var"], a, b, value], options, None)


_CERTIFY: dict[str, Callable[[re.Match], _Certified]] = {
    "integrate": _cert_integrate,
    "diff": _cert_diff,
    "solve": _cert_solve,
    "limit": _cert_limit,
    "dsolve": _cert_dsolve,
    "nintegrate": _cert_nintegrate,
}


def _check_certify(m: re.Match) -> Checked:
    line = m["command"]
    token = line.split(None, 1)[0].lower()
    certifier = _CERTIFY.get(token)
    if certifier is None:
        raise DomainError(
            f"certify: no certificate for {token!r} — available: "
            f"{', '.join(_CERTIFY)}"
        )
    pattern, _handler, usage = _COMMANDS[token]
    match = pattern.match(line)
    if match is None:
        return Checked(
            usage.replace("Usage: ", "Usage: certify ", 1).replace(
                "(e.g. ", "(e.g. certify ", 1
            )
        )
    done = certifier(match)
    cert = _written_down(
        done.kind, done.args, done.verdict,
        source=_source(m), budget=done.budget, **done.options,
    )
    return Checked(
        f"{done.answer}\ncertificate: {_verdict_line(done.verdict)}",
        verdict=done.verdict,
        answer=done.answer,
        certificate=cert,
    )


def _cmd_verify(m: re.Match) -> str:
    return _check_verify(m).text


def _cmd_certify(m: re.Match) -> str:
    return _check_certify(m).text


# --- certificates (X1) -----------------------------------------------------
# ``certificate`` prints a verify/certify line's check as JSON; ``recheck`` reads
# one back and re-runs it. A certificate that does not reproduce RAISES
# (``CertificateMismatch``, the report as its message) — so a one-shot
# ``pycodemath recheck file.json`` exits 1 on a mismatch and a script can rely on
# the exit code, while the REPL prints the report as an error.
def _cmd_certificate(m: re.Match) -> str:
    checked = handle_checked(m["line"])
    if checked.certificate is not None:
        return checked.certificate.to_json()
    if checked.verdict is None and checked.steps is None:
        return checked.text  # the inner line's usage message: nothing was checked
    raise DomainError(
        f"certificate: the check ran ({checked.text.splitlines()[-1]}) but its "
        f"claim cannot be recorded for a re-check"
    )


def _cmd_recheck(m: re.Match) -> str:
    budget = _options(m, _BUDGET_OPTS).get("budget")
    result = recheck(Certificate.load(m["path"]), budget=budget)
    if not result.ok:
        raise CertificateMismatch(result.text())
    return result.text()


# --- command table --------------------------------------------------------
# Key = the first token of the line (lowercased). Value = (full pattern,
# handler, usage message). One pattern per command instead of manual
# slicing — each command's syntax is explicit and testable in one place.
_NUM = r"[-+]?[\d.eE+-]+"  # number for float(); bad input is reported by float()
_NAMES = r"\w+(?:\s*,\s*\w+)*"  # list of variables: x,y,z
_NUMS = rf"{_NUM}(?:\s*,\s*{_NUM})*"  # list of numbers: 1,-2,0.5

_COMMANDS: dict[str, tuple[re.Pattern[str], Callable[[re.Match], str], str]] = {
    "diff": (
        re.compile(r"^diff\s+(?P<expr>.+)\s+d(?P<var>\w+)\s*$", re.IGNORECASE),
        _cmd_diff,
        "Usage: diff <expression> d<variable>   (e.g. diff sin(x)*x dx)",
    ),
    "integrate": (
        re.compile(
            rf"^integrate\s+(?P<expr>.+)\s+d(?P<var>\w+){_OPTS}\s*$", re.IGNORECASE
        ),
        _cmd_integrate,
        "Usage: integrate <expression> d<variable> [budget <s>]"
        "   (e.g. integrate 2*x dx)",
    ),
    "solve": (
        re.compile(
            rf"^solve\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)"
            rf"(?:\s+(?P<complex>complex))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_solve,
        "Usage: solve <expression> for <variable> [complex] [budget <s>]"
        "   (e.g. solve x^2-4 for x; real only by default)",
    ),
    "limit": (
        re.compile(
            rf"^limit\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)\s+to\s+(?P<to>\S+)"
            rf"(?:\s+dir\s+(?P<dir>[+-]))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_limit,
        "Usage: limit <expression> for <variable> to <point> [dir +|-] [budget <s>]"
        "   (e.g. limit sin(x)/x for x to 0)",
    ),
    "series": (
        re.compile(
            rf"^series\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)"
            rf"(?:\s+at\s+(?P<at>\S+))?(?:\s+n\s+(?P<n>\d+))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_series,
        "Usage: series <expression> for <variable> [at <point>] [n <order>] [budget <s>]"
        "   (e.g. series exp(x) for x n 4)",
    ),
    "sum": (
        re.compile(
            rf"^sum\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)"
            rf"\s+from\s+(?P<lo>\S+)\s+to\s+(?P<hi>\S+){_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_sum,
        "Usage: sum <expression> for <variable> from <lo> to <hi> [budget <s>]"
        "   (e.g. sum k for k from 1 to n)",
    ),
    "code": (
        re.compile(r"^code\s+(?P<expr>.+?)\s*$", re.IGNORECASE),
        _cmd_code,
        "Usage: code <expression>   (e.g. code (sin(x)+cos(x))^2)",
    ),
    "matrix": (
        re.compile(r"^matrix\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_matrix,
        "Usage: matrix <A>   (e.g. matrix [[1,2],[3,4]])",
    ),
    "det": (
        re.compile(r"^det\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_det,
        "Usage: det <A>   (e.g. det [[1,2],[3,4]])",
    ),
    "inv": (
        re.compile(r"^inv\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_inv,
        "Usage: inv <A>   (e.g. inv [[4,7],[2,6]])",
    ),
    "transpose": (
        re.compile(r"^transpose\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_transpose,
        "Usage: transpose <A>   (e.g. transpose [[1,2],[3,4]])",
    ),
    "eig": (
        re.compile(r"^eig\s+(?P<a>.+?)\s*$", re.IGNORECASE),
        _cmd_eig,
        "Usage: eig <A>   (e.g. eig [[2,0],[0,3]])",
    ),
    "solve_system": (
        re.compile(
            r"^solve_system\s+(?P<a>.+?)\s*=\s*(?P<b>.+?)\s*$", re.IGNORECASE
        ),
        _cmd_solve_system,
        "Usage: solve_system <A> = <b>   (e.g. solve_system [[2,1],[1,3]] = [3,5])",
    ),
    "code_system": (
        re.compile(
            r"^code_system\s+(?P<a>.+?)\s*=\s*(?P<b>.+?)\s*$", re.IGNORECASE
        ),
        _cmd_code_system,
        "Usage: code_system <A> = <b>   (e.g. code_system [[2,1],[1,3]] = [3,5])",
    ),
    "root": (
        re.compile(
            rf"^root\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)\s+at\s+(?P<x0>{_NUM})\s*$",
            re.IGNORECASE,
        ),
        _cmd_root,
        "Usage: root <expression> for <variable> at <x0>   (e.g. root x^2-2 for x at 1)",
    ),
    "min": (
        re.compile(
            rf"^min\s+(?P<expr>.+)\s+for\s+(?P<var>\w+)\s+at\s+(?P<x0>{_NUM})"
            rf"(?:\s+method\s+(?P<method>\w+))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_min,
        "Usage: min <expression> for <variable> at <x0> [method newton|bfgs]"
        " [tol <t>] [max_iter <k>]"
        "   (e.g. min (x-3)^2 for x at 0 method newton)",
    ),
    "nintegrate": (
        re.compile(
            rf"^nintegrate\s+(?P<expr>.+)\s+d(?P<var>\w+)"
            rf"\s+from\s+(?P<a>{_NUM})\s+to\s+(?P<b>{_NUM}){_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_nintegrate,
        "Usage: nintegrate <expression> d<variable> from <a> to <b> [tol <t>]"
        "   (e.g. nintegrate 2*x dx from 0 to 1)",
    ),
    "grad": (
        re.compile(
            rf"^grad\s+(?P<expr>.+)\s+for\s+(?P<vars>{_NAMES})\s*$", re.IGNORECASE
        ),
        _cmd_grad,
        "Usage: grad <expression> for <x,y,...>   (e.g. grad x^2*y for x,y)",
    ),
    "solve_nd": (
        re.compile(
            rf"^solve_nd\s+(?P<exprs>.+)\s+for\s+(?P<vars>{_NAMES})"
            rf"\s+at\s+(?P<x0>{_NUMS})\s*$",
            re.IGNORECASE,
        ),
        _cmd_solve_nd,
        "Usage: solve_nd <f1>; <f2> for <x,y> at <x0,y0>"
        "   (e.g. solve_nd x^2+y^2-4; x-y for x,y at 1,1)",
    ),
    "min_nd": (
        re.compile(
            rf"^min_nd\s+(?P<expr>.+)\s+for\s+(?P<vars>{_NAMES})"
            rf"\s+at\s+(?P<x0>{_NUMS})(?:\s+method\s+(?P<method>\w+))?{_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_min_nd,
        "Usage: min_nd <expression> for <x,y> at <x0,y0> [method newton|bfgs]"
        " [tol <t>] [max_iter <k>]"
        "   (e.g. min_nd (x-1)^2+(y+2)^2 for x,y at 0,0 method bfgs)",
    ),
    "dsolve": (
        re.compile(
            rf"^dsolve\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\){_OPTS}\s*$",
            re.IGNORECASE,
        ),
        _cmd_dsolve,
        "Usage: dsolve <expression> for <y>(<t>) [budget <s>]"
        "   (e.g. dsolve y for y(t))",
    ),
    "ode": (
        re.compile(
            rf"^ode\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"(?:\s+steps\s+(?P<n>\d+))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode,
        "Usage: ode <expression> for <y>(<t>) from <t0> to <t1> at <y0> [steps <n>]"
        "   (e.g. ode y for y(t) from 0 to 1 at 1)",
    ),
    "odestiff": (
        re.compile(
            rf"^odestiff\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"(?:\s+steps\s+(?P<n>\d+))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_stiff,
        "Usage: odestiff <expression> for <y>(<t>) from <t0> to <t1> at <y0> [steps <n>]"
        "   (e.g. odestiff -1000*(y-cos(t)) for y(t) from 0 to 1 at 0)",
    ),
    "odestiff_adaptive": (
        re.compile(
            rf"^odestiff_adaptive\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"(?:\s+rtol\s+(?P<rtol>{_NUM}))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_stiff_adaptive,
        "Usage: odestiff_adaptive <expression> for <y>(<t>) from <t0> to <t1> at <y0> [rtol <r>]"
        "   (e.g. odestiff_adaptive -1000*(y-cos(t)) for y(t) from 0 to 1 at 0)",
    ),
    "ode_adaptive": (
        re.compile(
            rf"^ode_adaptive\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"(?:\s+rtol\s+(?P<rtol>{_NUM}))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_adaptive,
        "Usage: ode_adaptive <expression> for <y>(<t>) from <t0> to <t1> at <y0> [rtol <r>]"
        "   (e.g. ode_adaptive y for y(t) from 0 to 1 at 1)",
    ),
    "odedense": (
        re.compile(
            rf"^odedense\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"\s+at\s+t\s*=\s*(?P<tq>{_NUM})(?:\s+rtol\s+(?P<rtol>{_NUM}))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_dense,
        "Usage: odedense <expression> for <y>(<t>) from <t0> to <t1> at <y0> at t=<point> [rtol <r>]"
        "   (e.g. odedense y for y(t) from 0 to 2 at 1 at t=0.37)",
    ),
    "odeevents": (
        re.compile(
            rf"^odeevents\s+(?P<expr>.+)\s+for\s+(?P<func>\w+)\((?P<var>\w+)\)"
            rf"\s+from\s+(?P<t0>{_NUM})\s+to\s+(?P<t1>{_NUM})\s+at\s+(?P<y0>{_NUM})"
            rf"\s+zero\s+(?P<g>.+?)(?:\s+dir\s+(?P<dir>[-+]?1))?"
            rf"(?:\s+rtol\s+(?P<rtol>{_NUM}))?(?:\s+(?P<stop>stop))?\s*$",
            re.IGNORECASE,
        ),
        _cmd_ode_events,
        "Usage: odeevents <expression> for <y>(<t>) from <t0> to <t1> at <y0> zero <g> [dir <+1|-1>] [rtol <r>] [stop]"
        "   (e.g. odeevents cos(t) for y(t) from 0.5 to 10 at 0.479 zero y)",
    ),
    # DOTALL: ``verify steps`` carries a derivation of one step per line.
    # Exactly one ``==`` on the equality form — a chain is a derivation, and
    # reading ``a == b == c`` as one claim would silently drop a step.
    "verify": (
        re.compile(
            r"^verify\s+(?:code\s+(?P<code>\S+?\.py\b.*?)"
            r"|steps\s+(?P<steps>.+?)"
            r"|(?P<a>(?:(?!==).)+?)\s*==\s*(?P<b>(?:(?!==).)+?))"
            rf"{_VERIFY_OPTS}\s*$",
            re.IGNORECASE | re.DOTALL,
        ),
        _cmd_verify,
        "Usage: verify <a> == <b> [budget <s>]   (e.g. verify sqrt(x^2) == x)\n"
        "       verify code <file.py>[:<function>] == <formula> [where <param>: <spec>]"
        " [--emit-test] [budget <s>]\n"
        "       verify steps <derivation> [budget <s>]"
        "   (e.g. verify steps 2x + 3 = 7 -> 2x = 4 -> x = 2)\n"
        "       certify <command>   (e.g. certify integrate x*cos(x) dx)",
    ),
    "certify": (
        re.compile(r"^certify\s+(?P<command>.+?)\s*$", re.IGNORECASE),
        _cmd_certify,
        "Usage: certify <command>   — one of: integrate, diff, solve, limit,"
        " dsolve, nintegrate   (e.g. certify integrate x*cos(x) dx)",
    ),
    "certificate": (
        re.compile(
            r"^certificate\s+(?P<line>(?:verify|certify)\s.+?)\s*$",
            re.IGNORECASE | re.DOTALL,
        ),
        _cmd_certificate,
        "Usage: certificate <verify or certify line>"
        "   (e.g. certificate verify sqrt(x^2) == x > c.json)",
    ),
    "recheck": (
        re.compile(rf"^recheck\s+(?P<path>\S+){_VERIFY_OPTS}\s*$", re.IGNORECASE),
        _cmd_recheck,
        "Usage: recheck <file.json> [budget <s>]   (e.g. recheck c.json)",
    ),
}


# --- the FULL-series commands (I1) ------------------------------------------
# Nine modules, one table: ``full.COMMANDS`` holds their patterns and handlers
# (each returns an ``Outcome`` — text, the module's ``to_dict()``, a certificate),
# merged here so the REPL, the one-shot CLI and MCP ``math_eval`` route them with
# the same rule as every other command, and K1's "did you mean" knows them too.
def _text_of(handler: Callable[[re.Match], "full.Outcome"]) -> Callable[[re.Match], str]:
    return lambda m: handler(m).text


for _word, (_pattern, _full_handler, _usage) in full.COMMANDS.items():
    assert _word not in _COMMANDS, f"I1: command {_word!r} defined twice"
    _COMMANDS[_word] = (_pattern, _text_of(_full_handler), _usage)
del _word, _pattern, _full_handler, _usage

# --- evidence handlers (module 6) -----------------------------------------
# The twins of the five numerical ``_cmd_*`` handlers above, asking the engine for
# the structured result modules 3-5 built (``full_result=True``). They exist for the
# MCP surface, NOT for the REPL: under ``full_result`` an ITERATION FAILURE is
# RETURNED instead of raised, so a REPL that used them would print a last iterate
# where it prints an error today. ``handle`` therefore keeps calling the plain
# handlers, and only ``handle_full`` reaches for these.


@dataclass(frozen=True, slots=True)
class Answer:
    """One line of input answered: the text, plus the EVIDENCE behind it if any.

    ``evidence`` is ``None`` for every command that has no structured result to
    give (symbolic, linear algebra, code generation, ODE) — that is the honest
    report, not a gap: module 6 carries the results that EXIST, and does not
    invent one where the engine has none.
    """

    text: str
    evidence: "SolveResult[Any] | QuadratureResult | None" = None
    outcome: "full.Outcome | None" = None


def _solved(result: "SolveResult[Any]", text: str) -> Answer:
    """Pair a solver result with its text — appending the caveat when it FAILED.

    A client that renders only the text block must not read a last iterate as an
    answer, so the text says so. The structured half says the same thing in
    ``converged``/``status``; the two are built here together, once.
    """
    if result.converged:
        return Answer(text, result)
    # Module 8: "NOT CONVERGED" would be false for the one outcome where the run
    # DID converge — to a stationary point that turned out to be a maximum, a
    # saddle or an inflection. The caveat has to say which of the two happened, or
    # a text-only client reads "give it more iterations" where none can help.
    headline = "NOT A MINIMUM" if result.status == "not_a_minimum" else "NOT CONVERGED"
    return Answer(
        f"{text}  ({headline} — status {result.status}, residual "
        f"{result.residual:g} after {result.iterations} iterations)",
        result,
    )


def _integrated(result: QuadratureResult, text: str) -> Answer:
    """The quadrature twin of ``_solved`` — its own fields, for module 5's reason."""
    if result.converged:
        return Answer(text, result)
    return Answer(
        f"{text}  (NOT CONVERGED — status {result.status}, error estimate "
        f"{result.error_estimate:g} after {result.evaluations} evaluations)",
        result,
    )


def _ev_root(m: re.Match) -> Answer:
    res = numerics.root_find(
        _expr(m), m["var"], _num(m["x0"]), full_result=True
    )
    return _solved(res, _fmt_scalar(res.value))


def _ev_min(m: re.Match) -> Answer:
    method = (m["method"] or "gd").lower()
    res = numerics.minimize(
        _expr(m),
        m["var"],
        _num(m["x0"]),
        method=method,
        full_result=True,
        **_options(m, _MIN_OPTS),
    )
    return _solved(res, _fmt_scalar(res.value))


def _ev_nintegrate(m: re.Match) -> Answer:
    res = numerics.integrate_num(
        _expr(m),
        m["var"],
        _num(m["a"]),
        _num(m["b"]),
        full_result=True,
        **_options(m, _NINTEGRATE_OPTS),
    )
    return _integrated(res, _fmt_scalar(res.value))


def _ev_solve_nd(m: re.Match) -> Answer:
    names = _split_names(m["vars"])
    exprs = _parse_for([e.strip() for e in m["exprs"].split(";")], names)
    res = numerics.root_find_nd(
        exprs, names, _split_floats(m["x0"]), full_result=True
    )
    return _solved(res, _fmt_named(names, res.value))


def _ev_min_nd(m: re.Match) -> Answer:
    names = _split_names(m["vars"])
    method = (m["method"] or "gd").lower()
    res = numerics.minimize_nd(
        _expr(m),
        names,
        _split_floats(m["x0"]),
        method=method,
        full_result=True,
        **_options(m, _MIN_OPTS),
    )
    return _solved(res, _fmt_named(names, res.value))


#: Command token -> the handler that also brings back evidence. Exactly the five
#: commands the engine can answer with a ``SolveResult`` / ``QuadratureResult``;
#: every other token falls through to its plain handler and reports no evidence.
_EVIDENCE: dict[str, Callable[[re.Match], Answer]] = {
    "root": _ev_root,
    "min": _ev_min,
    "nintegrate": _ev_nintegrate,
    "solve_nd": _ev_solve_nd,
    "min_nd": _ev_min_nd,
}


def _route(line: str) -> "tuple[str, re.Match] | str":
    """Resolve one input line to ``(command token, match)`` — or to final text.

    The ONE routing rule, shared by ``handle`` and ``handle_full`` so the two
    dispatchers cannot drift on what a line means (empty line, help, an unknown
    token that falls through to simplification, a known token whose syntax does
    not match and earns the usage message). A ``str`` is a finished answer with no
    command behind it; a tuple is a matched command the caller still has to run.
    """
    line = line.strip()
    if not line:
        return ""

    if line.lower() in ("help", "?"):
        return __doc__ or ""

    token = line.split(None, 1)[0].lower()
    if token in _WAVE2_SHAPES:
        # K5 (I2): a relation, a system, ``all`` or ``from a to b`` picks the module;
        # every other ``solve`` / ``integrate`` line keeps its old meaning
        outcome = full.wave2_route(line)
        if outcome is not None:
            return "outcome", outcome  # type: ignore[return-value]
    entry = _COMMANDS.get(token)
    if entry is None:
        bare = full.BARE_CONVERSION.match(line)
        if bare is not None and _reads_as_units(bare["q"], bare["to"]):
            return "convert", _COMMANDS["convert"][0].match(f"convert {bare['q']} to {bare['to']}")  # type: ignore[return-value]
        precision = _precision_request(line)
        if precision is not None:  # N(pi, 50), evalf(e, 30), pi digits 50 (K2, I2)
            return "outcome", full.approx_outcome(*precision, source=line)  # type: ignore[return-value]
        _refuse_unknown_command(line)
        # default: parse and simplify — with the proved decimal beside an exact
        # number (K2's dual answer, I2), the way SymPy Gamma and W|A show both
        try:
            value = symbolic.simplify(parse(line))
            text = str(value)
            dual = full.dual_outcome(text, value, source=line)
            return text if dual is None else ("outcome", dual)  # type: ignore[return-value]
        except ParseError as exc:
            # ``gcd(12, 18)``: a function the parser does not know, the calculator
            # does (I1) — only once the parser has refused, so nothing it reads moves
            if "unknown function" not in str(exc) or full.call_route(line) is None:
                raise
            return "call", full._CALL.match(line.strip())  # type: ignore[return-value]

    pattern, _handler, usage = entry
    match = pattern.match(line)
    if match is None:
        rest = line[len(token):].lstrip()
        if token in full.COMMANDS and rest and rest[0] in _EXPRESSION_CONTINUES:
            # ``mean + 1``, ``power(2, 3)``: an expression that happens to open
            # with a FULL command word (rule 3 of ``full``) — not a command line
            return str(symbolic.simplify(parse(line)))
        return usage
    return token, match


#: First words whose line K5 may take by its shape (``full.wave2_route``).
_WAVE2_SHAPES = ("solve", "integrate", "nintegrate", "reduce")


def _precision_request(line: str) -> "tuple[str, int] | None":
    """``N(e, d)`` / ``evalf(e, d)`` / ``e digits d`` -> ``(e, d)`` — K2's purely
    syntactic reader, consulted only for a line that is not a command."""
    from ..engine.precision import parse_precision_request

    call = re.match(r"^\s*(?:N|evalf)\s*\(", line)
    if not call and not re.search(r"\sdigits\s+\d+\s*$", line):
        return None
    request = parse_precision_request(line)
    if request is None and call:
        raise ParseError(f"{line.strip()!r}: write N(<expression>, <digits>) with digits a whole "
                         f"number from 1 to 10000 (e.g. N(pi, 50))")
    return request


def _reads_as_units(quantity: str, target: str) -> bool:
    """Do both sides of ``<number> <unit> to <unit>`` read as units (J1)?"""
    from ..engine import units

    try:
        units.parse_unit(target)
        units.convert(quantity, target)
    except PycodemathError:
        return False
    return True


#: Characters that, right after a first word, make it part of an expression
#: (``ab + c``, ``zeta (2)``) rather than a command with arguments.
_EXPRESSION_CONTINUES = "+-*/^=<>!),]([."


def _refuse_unknown_command(line: str) -> None:
    """A line opening with an unknown WORD and its arguments is a ``ParseError``.

    ``factor 2^64+1``, ``isprime 2^61-1``, ``latex x^2/2``, ``evalf sqrt(2) 50``
    used to fall through to "parse and simplify" and came back as products of
    letters with exit code 0 (K1). A first word that is not a command, not a
    known function or constant, followed by more text that does not continue an
    expression, is a command the grammar does not have: say so, with the closest
    command names (difflib) when there are any.
    """
    parts = line.split(None, 1)
    word = parts[0]
    if len(parts) < 2 or not word.isalpha() or len(word) < 2 or word in KNOWN_NAMES:
        return
    if parts[1][0] in _EXPRESSION_CONTINUES:
        return
    if word.lower() == "implicit":
        raise ParseError(
            "'implicit on|off' is a setting of the interactive REPL (one-shot: "
            "--implicit); this surface always parses strictly"
        )
    close = difflib.get_close_matches(word.lower(), list(_COMMANDS) + ["help"], n=3, cutoff=0.75)
    hint = (
        " — did you mean " + " or ".join(f"'{c}'" for c in close) + "?"
        if close
        else ""
    )
    raise ParseError(
        f"unknown command {word!r}{hint} (type 'help' for the list of commands; "
        f"write {word}*(...) if a product is meant)"
    )


def handle(line: str) -> str:
    """Handle one line of REPL input and return the response text."""
    routed = _route(line)
    if isinstance(routed, str):
        return routed
    token, match = routed
    if token == "call":
        return _called(match).text
    if token == "outcome":
        return match.text  # type: ignore[attr-defined]
    return _COMMANDS[token][1](match)


def _called(match: re.Match) -> "full.Outcome":
    outcome = full.call_route(match.string)
    assert outcome is not None  # ``_route`` only returns "call" for a routed name
    return outcome


def handle_full(line: str) -> Answer:
    """Handle one line and bring back the EVIDENCE too, where the engine has any.

    The agent-facing twin of ``handle`` (module 6), used by the MCP server. Same
    line, same syntax table, same text — plus the ``SolveResult`` /
    ``QuadratureResult`` of the five numerical commands.

    ``text`` is byte-identical to ``handle``'s for every line ``handle`` answers
    without raising; the two differ only where ``handle`` RAISES an iteration
    failure, which this returns as data with the caveat appended.

    THE EVIDENCE NEVER COSTS AN ANSWER. Asking for it takes extra samples (a
    residual at the returned point, a midpoint grid for the error estimate), and
    those samples can land outside the domain where the answer itself was fine —
    the engine then refuses with ``DomainError`` saying the value is available
    without ``full_result``. So that refusal is taken literally: fall back to the
    plain handler and report the answer with no evidence. A refusal of the INPUT
    raises from the plain handler too, and propagates unchanged.
    """
    routed = _route(line)
    if isinstance(routed, str):
        return Answer(routed)
    token, match = routed
    if token == "call":
        outcome = _called(match)
        return Answer(outcome.text, outcome=outcome)
    if token == "outcome":
        return Answer(match.text, outcome=match)  # type: ignore[arg-type,attr-defined]
    if token in full.COMMANDS:
        outcome = full.COMMANDS[token][1](match)
        return Answer(outcome.text, outcome=outcome)
    evidence = _EVIDENCE.get(token)
    if evidence is None:
        return Answer(_COMMANDS[token][1](match))
    try:
        return evidence(match)
    except DomainError:
        return Answer(_COMMANDS[token][1](match))


#: Command token -> the handler that also brings back the VERDICT objects.
_CHECKS: dict[str, Callable[[re.Match], Checked]] = {
    "verify": _check_verify,
    "certify": _check_certify,
}


def handle_checked(line: str) -> Checked:
    """Handle one ``verify`` / ``certify`` line and bring back the verdicts too.

    The twin of ``handle`` for the MCP ``math_verify`` tool, as ``handle_full`` is
    for ``math_eval``: same line, same syntax table, byte-identical ``text``. A
    line that does not start with ``verify`` or ``certify`` is refused with a
    ``ParseError`` — this entry point checks, it does not compute.
    """
    routed = _route(line)
    if isinstance(routed, str):
        return Checked(routed)
    token, match = routed
    check = _CHECKS.get(token)
    if check is None:
        raise ParseError(
            f"not a verify/certify line: {line.strip()!r} — "
            f"expected 'verify <a> == <b>', 'verify steps <derivation>' "
            f"or 'certify <command>'"
        )
    return check(match)


def _read_steps(line: str, read: Callable[[str], str]) -> str:
    """``line`` itself — or, for a bare ``verify steps``, the derivation after it.

    Interactive only: the steps are read one per line with ``read`` until an
    empty line (or end of input), and joined back into the one multi-line
    command the dispatcher takes — so the interactive form and a one-shot
    ``verify steps a = b = c`` reach ``check_steps`` the same way.
    """
    if line.strip().lower() != "verify steps":
        return line
    lines = [line.strip()]
    while True:
        try:
            step = read("...> ")
        except EOFError:
            break
        if not step.strip():
            break
        lines.append(step)
    return "\n".join(lines)


def _implicit_setting(line: str) -> "str | None":
    """``implicit on|off`` (or bare ``implicit``): the REPL session's parsing mode.

    Off (strict, the default) reads ``speed`` as one variable; on splits a
    multi-letter name into a product of letters (``2xy`` = 2*x*y). Returns the
    text to print, or ``None`` when the line is not this setting.
    """
    words = line.strip().lower().split()
    if not words or words[0] != "implicit" or len(words) > 2:
        return None
    if len(words) == 2:
        if words[1] not in ("on", "off"):
            return "Usage: implicit on|off"
        set_strict(words[1] == "off")
    state = "off (strict: 'speed' is one variable)" if is_strict() else (
        "on ('xy' is x*y — multi-letter names split into letters)"
    )
    return f"implicit multiplication of names: {state}"


def run() -> None:
    """Run the REPL — interactively or one-shot from command-line arguments.

    ``python -m pycodemath "diff sin(x)*x dx"`` (or the ``pycodemath`` script)
    runs a single command and exits: result on stdout, error on stderr with exit
    code 1 — this is what makes Pycodemath scriptable (e.g. as a skill).
    """
    # The Windows console is sometimes cp1252 — force UTF-8 so non-ASCII characters work.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:  # history and line editing on POSIX; the Windows console has this natively
        import readline  # noqa: F401
    except ImportError:
        pass

    # SymPy's deprecation warnings are about SymPy's own API, never about the
    # user's math, and on stderr they buried the actual answer or refusal
    # (``N(pi, 50)`` printed a SymPyDeprecationWarning before its error, K1).
    warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)

    args = sys.argv[1:]
    if args and args[0] == "--implicit":  # one-shot opt-in to name splitting
        set_strict(False)
        args = args[1:]
    from . import discover, intro  # intro / commands / help <topic> / examples / tour

    # Once per installed version, on a TTY only: the intro, then the command as usual.
    # Never off a TTY (scripts, pipes, agents) — their stdout is the answer alone.
    if not (args and discover.handles(args) and discover._words(args)[0].lower() == "intro"):
        intro.first_run(sys.stdout)
    if args and args[0].split(None, 1)[0].lower() == "lint":
        # L1 (I2): ruff's exit codes for ``lint`` ONLY — 0 clean, 1 findings, 2 usage
        # or error; every other command keeps 0 = answered, 1 = refused. One quoted
        # line ("lint a.py --concise") or separate words (lint a.py --concise).
        import shlex

        line = args[0] if len(args) == 1 else "lint " + " ".join(shlex.quote(a) for a in args[1:])
        try:
            answer = handle_full(line)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2)
        print(answer.text)
        result = answer.outcome.result if answer.outcome is not None else None
        raise SystemExit(int((result or {}).get("exit_code", 2)))
    if args and discover.handles(args):
        raise SystemExit(discover.main(args))
    if args:
        try:
            out = handle(" ".join(args))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1)
        if out:
            print(out)
        return

    print(intro.repl_banner())
    while True:
        try:
            line = input("pcm> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in ("quit", "exit"):
            break
        setting = _implicit_setting(line)
        if setting is not None:
            print(setting)
            continue
        if discover.handles([line]):
            try:
                discover.main([line])
            except KeyboardInterrupt:
                print()
            except Exception as exc:  # in the REPL an error must not kill the session
                print(f"error: {exc}")
            continue
        try:
            line = _read_steps(line, input)
        except KeyboardInterrupt:  # abandons the derivation, not the session
            print()
            continue
        try:
            out = handle(line)
        except Exception as exc:  # in the REPL an error must not kill the session
            out = f"error: {exc}"
        if out:
            print(out)
