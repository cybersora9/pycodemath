"""Structured results — what a numerical run LEARNED, not just where it landed.

Two types live here, deliberately kept apart: ``SolveResult`` for the four iterative
solvers (modules 3 and 4) and ``QuadratureResult`` for the integral (module 5). They
share the status vocabulary and the constructor discipline, and nothing else — see
``QuadratureResult`` for why reusing one type for both would have corrupted a field.

--- SolveResult ---------------------------------------------------------------

Module 2 of the v0.3 series made failure TYPES distinguishable (the
``PycodemathError`` hierarchy). On the SUCCESS path a solver still returned a bare
number: an agent got the root, but not whether it was a rock-solid one or a value
that scraped in at the tolerance limit on the last allowed iteration. Everything
the loop knew — how many iterations it took, how big the residual still was —
was thrown away at the ``return``.

``SolveResult`` carries that depth. It is OPT-IN: every solver keeps its current
default return type and value, and only ``full_result=True`` asks for the
structured form (see ``engine.numerics``). Frozen and slotted, so a result cannot
be edited after the fact and stays cheap to pass around.

Field contract:

==============  ==============================================================
``value``       exactly what the solver returns without ``full_result`` —
                ``float`` for the 1D solvers, ``list[float]`` for the nd ones.
                ON A FAILURE it is the LAST ITERATE — the point the solver gave
                up at, NOT a solution (see below)
``iterations``  how many iterations actually ran, counting the one that
                produced the answer (so ``1`` when the starting point already
                satisfied the tolerance); never above the solver's ``max_iter``
``residual``    how far from solved the returned point still is: ``|f(x)|``
                for root finding (``‖F(x)‖∞`` for a system) and ``‖∇f(x)‖∞``
                for minimization — ONE quantity per family, recomputed AT the
                returned point, not the smallest value seen along the way.
                ``math.inf`` when it cannot be sampled at a failed run's last
                iterate (module 4; see ``engine.numerics``)
``converged``   ``True`` only on the success path — the ONLY field a caller may
                branch on to decide whether ``value`` is an answer TO THE QUESTION
                THAT WAS ASKED: a ROOT for the root finders, a MINIMUM for the
                minimizers. Never a proxy for it: a small ``residual`` is NOT the
                same claim (module 8 — ``root_find(exp(x))`` used to walk down the
                tail until ``|f|`` fell under ``tol`` and report success at a point
                where no root exists), and neither is a vanished gradient (every
                MAXIMUM and every SADDLE satisfies it)
``status``      ``"converged"`` on success; the remaining values name the
                Module-2 failure vocabulary
==============  ==============================================================

TWO forms, two constructors — and no third way in. ``success()`` builds the form
with ``converged=True`` / ``status="converged"``; ``failure()`` (module 4) builds
the form with ``converged=False`` and one of the four failure statuses. Both
pairings live in exactly one place each, so a result carrying ``converged=True``
with a status saying otherwise cannot come out of any solver.

Module 3 declared ``"diverged"``, ``"stagnated"`` and ``"not_converged"`` with no
producer: a failing solver simply raised. Module 4 gave them one — under
``full_result=True`` the four iterative solvers RETURN the failure form instead of
raising (the default call still raises the Module-2 exception, message unchanged).
What did NOT change: an INPUT refusal (``DomainError`` — a bad variable list, a
non-square system, a singular Jacobian, an unknown method, a point outside the
domain) raises either way. Those are not iteration outcomes, so they have no
status to report, and there is no last iterate to hand back.

Module 8 made ``converged`` TRUE to that promise. Until then it answered a WEAKER
question than the one it was documented to answer — "did an exit test fire" rather
than "is ``value`` an answer" — and the two came apart in three measured ways: a
maximum or a saddle passed the gradient test (``minimize(-x^2, x0=0)`` →
``converged=True``), a decaying function passed the residual test with no root
anywhere (``root_find(exp(x))`` → ``converged=True`` at ``x=-24``), and a correct
minimum reached slowly was reported as a stall (``minimize(sin(x), x0=0)`` →
``converged=False`` at a value right to 7 significant figures). The exit tests are
now CORROBORATED — see ``engine.numerics`` for the two certificates and for the
one thing they still do not promise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

__all__ = [
    "SolveResult",
    "QuadratureResult",
    "SolveStatus",
    "SOLVE_STATUSES",
]

#: How the iteration ENDED. ``"converged"`` comes from ``success()``; the other
#: four mirror the Module-2 exception hierarchy (``DivergenceError`` /
#: ``StagnationError`` / ``NonConvergenceError`` / ``NotAMinimumError``) and come
#: from ``failure()``, so a solver hands back the same outcomes it would otherwise
#: raise — without inventing a second vocabulary for them.
#:
#: ``"not_a_minimum"`` joined in module 8, together with its exception, and the
#: pairing stayed 1:1 on purpose. It is the one outcome where the run had no
#: trouble at all — it ARRIVED, at a stationary point that turned out to be a
#: maximum, a saddle or an inflection. None of the other three describes that: two
#: of them say the iterate never got anywhere (escaped / ran out) and one says it
#: stopped moving while still off target. The distinction is not cosmetic, it is
#: the REMEDY: an agent reading ``"not_converged"`` raises ``max_iter``, and more
#: iterations can never help here — gradient descent cannot leave a stationary
#: point, because the step it would take there is zero. Only a different starting
#: point can.
SolveStatus = Literal[
    "converged", "diverged", "stagnated", "not_converged", "not_a_minimum"
]

#: The ``SolveStatus`` values as data — for validation and for a caller that wants
#: to enumerate them at runtime (``Literal`` alone is invisible to ``in``).
SOLVE_STATUSES: tuple[SolveStatus, ...] = (
    "converged",
    "diverged",
    "stagnated",
    "not_converged",
    "not_a_minimum",
)

#: The FAILURE half of the vocabulary — what ``failure()`` accepts. Derived from
#: ``SOLVE_STATUSES``, never spelled out a second time: adding a status to the
#: ``Literal`` must not silently leave this list behind.
_FAILURE_STATUSES: tuple[SolveStatus, ...] = tuple(
    s for s in SOLVE_STATUSES if s != "converged"
)

#: The answer type, so ``SolveResult[float]`` (1D) and ``SolveResult[list[float]]``
#: (nd) stay distinguishable to a type checker — the caller of a 1D solver must not
#: have to cast to get a ``float`` back out.
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SolveResult(Generic[T]):
    """A solver's answer together with the evidence behind it (see module docstring)."""

    value: T
    iterations: int
    residual: float
    converged: bool
    status: SolveStatus

    @classmethod
    def success(cls, value: T, iterations: int, residual: float) -> "SolveResult[T]":
        """Build the SUCCESS form: ``converged=True`` and ``status="converged"``.

        A single constructor for the whole success path, so the pairing of the two
        cannot drift apart across solvers (a result with ``converged=True`` and a
        status saying otherwise would be a contradiction an agent may branch on).
        """
        return cls(
            value=value,
            iterations=iterations,
            residual=residual,
            converged=True,
            status="converged",
        )

    @classmethod
    def failure(
        cls, value: T, iterations: int, residual: float, status: SolveStatus
    ) -> "SolveResult[T]":
        """Build the FAILURE form: ``converged=False`` and the given ``status`` (module 4).

        The twin of ``success()`` and the ONLY constructor of the failure form, for
        the same reason: the pairing of the flag with the status stays in one place.
        ``value`` is the LAST ITERATE — where the solver gave up, not an answer;
        ``converged`` is what a caller branches on.

        ``status`` must be one of the failure statuses. A wrong one is a
        programming error inside a solver, not a mathematical outcome, so it gets a
        plain ``ValueError`` — this module deliberately stays independent of
        ``core.errors``, and the check is the mechanical guarantee that every
        status a solver can produce belongs to ``SOLVE_STATUSES``.
        """
        if status not in _FAILURE_STATUSES:
            raise ValueError(
                f"SolveResult.failure needs a failure status, got {status!r} — "
                f"available: {', '.join(_FAILURE_STATUSES)} "
                f"(use SolveResult.success() for the converged form)"
            )
        return cls(
            value=value,
            iterations=iterations,
            residual=residual,
            converged=False,
            status=status,
        )


@dataclass(frozen=True, slots=True)
class QuadratureResult:
    """A numerical integral together with how good it is (module 5).

    WHY A SEPARATE TYPE, and not ``SolveResult``. Both were defensible and the wrong
    one silently corrupts a contract, so here is the reasoning, once:

      * ``SolveResult.residual`` means "how far the returned point still is from
        solved", MEASURED at that point: ``|f(x)|`` or ``‖∇f(x)‖∞``. It is a fact
        about the answer — exact, verifiable, and always finite when measurable.
        A quadrature error estimate is a different KIND of number: nothing is being
        zeroed (there is no equation to satisfy), and the value is not measured but
        INFERRED from two grids under a smoothness assumption — an assumption an
        integrand may violate, in which case the estimate is optimistic and lies
        (see ``error_estimate``). Putting a fact and a guess in one field, with the
        meaning depending on which function produced the result, is exactly the
        corruption an agent cannot see and cannot defend against.
      * ``SolveResult.iterations`` counts iterations of a method that walks toward a
        point. Quadrature does not iterate toward anything: its work is SAMPLES of
        the integrand, and its refinement is a tree of subdivisions. Reusing the
        field would force one of the two meanings ("refinement levels", "panels")
        onto a name that promises neither.

    What IS shared with ``SolveResult`` — deliberately, so there is one discipline
    and not two: the status vocabulary (``SOLVE_STATUSES``, not a private second
    list), one constructor per form (``success()`` / ``failure()``, no third way in),
    a runtime-validated failure status, ``converged`` as the ONLY field a caller may
    branch on, and frozen + slotted so a result cannot be edited after the fact.

    Field contract:

    ==================  ==========================================================
    ``value``           the integral — exactly what ``integrate_num`` returns
                        without ``full_result``. ON A FAILURE it is the best
                        composite the run had reached, NOT an answer to the
                        requested tolerance
    ``error_estimate``  estimated ABSOLUTE error of ``value``, from the classical
                        Richardson comparison of two grids. AN ESTIMATE, NOT A
                        BOUND: an integrand that violates the smoothness the
                        comparison assumes can get a small estimate with a large
                        true error (``engine.numerics.integrate_num`` documents the
                        case and the test suite pins one where the estimate is 0.0
                        and the true error is 100%). ``math.inf`` when the run gave
                        up with part of the interval unrefined, so it cannot be
                        measured at all — the same rule, and the same reasoning, as
                        a failed solver's residual
    ``evaluations``     how many times the COMPILED integrand was sampled — the
                        honest unit of work for quadrature (never above
                        ``max_evals`` plus the samples of the panel in progress)
    ``refinements``     how many panel subdivisions the adaptive path performed;
                        ``0`` on the fixed-``n`` path, which subdivides nothing
    ``converged``       whether the run delivered the accuracy that was ASKED FOR.
                        With no ``tol`` requested nothing can be missed, so it is
                        ``True`` — it is NOT a quality claim: read
                        ``error_estimate`` for quality. The ONLY field to branch on
    ``status``          ``"converged"`` on success; ``"stagnated"`` /
                        ``"not_converged"`` name the failure, from the same
                        vocabulary the solvers use
    ==================  ==========================================================

    ``"diverged"`` is never produced here, and that is not an omission: an integrand
    that escapes to infinity is refused at the SAMPLE (``DomainError``, from
    ``_sample``) rather than integrated into a diverging total, so quadrature has no
    outcome that word describes. The same holds for ``"not_a_minimum"`` (module 8):
    quadrature is not asked for a point, so no point it returns can be of the wrong
    kind. Sharing ONE vocabulary means each type can be handed a word it never says
    — that is the price of not maintaining two lists that drift, and it is paid
    knowingly: what a type PRODUCES is pinned by tests, not by the vocabulary.
    """

    value: float
    error_estimate: float
    evaluations: int
    refinements: int
    converged: bool
    status: SolveStatus

    @classmethod
    def success(
        cls, value: float, error_estimate: float, evaluations: int, refinements: int
    ) -> "QuadratureResult":
        """Build the SUCCESS form: ``converged=True`` and ``status="converged"``.

        One constructor for the whole success path, for the same reason as in
        ``SolveResult``: the pairing of the flag with the status cannot drift.
        """
        return cls(
            value=value,
            error_estimate=error_estimate,
            evaluations=evaluations,
            refinements=refinements,
            converged=True,
            status="converged",
        )

    @classmethod
    def failure(
        cls,
        value: float,
        error_estimate: float,
        evaluations: int,
        refinements: int,
        status: SolveStatus,
    ) -> "QuadratureResult":
        """Build the FAILURE form: ``converged=False`` and the given ``status``.

        The requested tolerance was not reached. ``value`` is the best composite the
        run had — usable as a rough number, but NOT accurate to ``tol``; ``converged``
        is what a caller branches on.

        ``status`` must be a failure status (validated at runtime, as in
        ``SolveResult.failure`` and against the same derived list, so neither type can
        invent a status the vocabulary does not declare).
        """
        if status not in _FAILURE_STATUSES:
            raise ValueError(
                f"QuadratureResult.failure needs a failure status, got {status!r} — "
                f"available: {', '.join(_FAILURE_STATUSES)} "
                f"(use QuadratureResult.success() for the converged form)"
            )
        return cls(
            value=value,
            error_estimate=error_estimate,
            evaluations=evaluations,
            refinements=refinements,
            converged=False,
            status=status,
        )
