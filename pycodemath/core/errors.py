"""Pycodemath exceptions — one shared base plus a typed hierarchy beneath it.

Goal: the user/agent and the REPL get a readable Pycodemath message instead of
a raw SymPy exception (``SympifyError``, ``TypeError`` from ``float``, etc.).

Every failure in the package raises ``PycodemathError`` or one of its
subclasses, so ``except PycodemathError`` stays a complete catch. The subclass
says WHAT KIND of failure it was, so an agent can branch on the class instead of
string-matching the English message.

The taxonomy is by mathematical OUTCOME, not by mechanism:

======================  ====================================================
``ParseError``          the text was refused before any math happened
``DomainError``         the input was rejected: bad value, point/bound, shape,
                        option, or a sample outside the function's domain
``DivergenceError``     the quantity escaped to infinity (iterate, ODE
                        solution, or a divergent series)
``StagnationError``     an iteration stopped making progress while still off
                        target (gradient descent stalls, ODE step collapse,
                        a quadrature panel at the resolution of the float grid)
``NonConvergenceError`` an iteration ended without a value, neither blowing up
                        nor stalling (iteration/step cap, oscillation)
``NotAMinimumError``    an iteration ARRIVED — at a stationary point that is not
                        a minimum (a maximum, a saddle, an inflection)
``BudgetExhaustedError``an explicit evaluation/iteration budget ran out
                        (adaptive quadrature's ``max_evals``)
``TimeBudgetError``     the symbolic engine was still working when the wall-clock
                        budget ran out — no answer, and no partial one either
``NoClosedFormError``   the symbolic engine SEARCHED and found no closed form
                        (an integral, an infinite sum, an ODE, a limit)
``UnsupportedFormError``the symbolic engine has no METHOD for this SHAPE — it
                        never got as far as searching
``IsolationError``      the worker process of ``isolated`` (module B) died or
                        could not hand its answer back — no math verdict at all
======================  ====================================================

``PycodemathError`` itself is never raised directly. Until module 10 it was: ten
``raise PycodemathError(...)`` statements, every one of them on the symbolic
surface, carrying four different mathematical outcomes with four different
remedies under one class name. See ``NoClosedFormError`` for the measurement and
for why the split is not cosmetic.

--- THE ROUTE (module 10) ------------------------------------------------------

Every exception here carries ``route``: the name of the pycodemath command that
attacks THE SAME PROBLEM numerically, or ``None`` when there is no such command.
``NoClosedFormError`` from ``integrate`` carries ``"nintegrate"``;
``UnsupportedFormError`` from ``solve`` carries ``"root"``; a ``TimeBudgetError``
that expired inside ``integrate`` carries ``"nintegrate"`` too.

It is a fact about the API, NOT a promise about the answer. The route may fail,
and it may need input the symbolic call did not need — ``nintegrate`` wants
bounds, because a definite integral is a different question from an
antiderivative, and ``root`` wants a starting point. What the route removes is the
step where an agent had to read an English sentence to learn that a numerical
path exists at all. ``ROUTES`` below is the closed vocabulary, and
``tests/test_errors.py`` pins every one of its values to a command the REPL
actually dispatches — a route naming a command that does not exist would be worse
than no route.
"""

from __future__ import annotations

__all__ = [
    "PycodemathError",
    "ParseError",
    "DomainError",
    "DivergenceError",
    "StagnationError",
    "NonConvergenceError",
    "NotAMinimumError",
    "BudgetExhaustedError",
    "TimeBudgetError",
    "NoClosedFormError",
    "UnsupportedFormError",
    "IsolationError",
    "ROUTES",
    "NUMERIC_ROUTES",
]

#: The closed vocabulary of numerical routes. Every value is a REPL/MCP command
#: token, pinned as such by ``tests/test_errors.py`` — the route must name a call
#: an agent can actually make, and the check is mechanical rather than a promise.
ROUTES: tuple[str, ...] = ("nintegrate", "root", "ode", "solve_nd")

#: Guarded OPERATION (``core.budget.under_budget``'s label) -> the numerical route
#: for that operation. Used where the refusal is raised by the budget rather than
#: by the engine, so ``TimeBudgetError`` carries the same datum as the refusals it
#: sits beside. An operation absent here has no numerical counterpart in the
#: command grammar (``simplify``, ``limit``, ``series``, ``summation``, and the
#: matrix entry points, whose numeric fallback is not a separate command).
NUMERIC_ROUTES: dict[str, str] = {
    "integrate": "nintegrate",
    "solve": "root",
}


class PycodemathError(Exception):
    """Pycodemath domain error (parsing, evaluation, generation).

    Base of the whole hierarchy, and never raised on its own — every failure in
    the package raises one of the subclasses below, so ``except PycodemathError``
    stays a complete catch while the CLASS always says what kind of failure it
    was. Always carries a concise, readable message; the original exception is
    attached via ``raise ... from exc`` for inspection while debugging.

    ``route`` (module 10) is the name of the pycodemath command that attacks the
    same problem numerically, or ``None``. It lives on the base class so that any
    refusal can carry one — including ``TimeBudgetError``, which is raised by the
    budget rather than by the engine. See the module docstring for what a route
    does and does not promise.
    """

    def __init__(self, message: str, *, route: "str | None" = None) -> None:
        super().__init__(message)
        self.route = route


class ParseError(PycodemathError):
    """The input text was refused by the frontend, before any math ran.

    Covers whitelist refusals (text literals, attribute access), syntax the
    parser cannot understand, matrix/vector literals that are not a matrix,
    and cost-guard refusals (an expression whose result would exceed the safe
    size).
    """


class DomainError(PycodemathError):
    """The input was rejected as outside what the operation accepts.

    Covers a sample outside a function's real domain (log of a non-positive
    number, a complex result), an invalid point or bound, a matrix shape or
    dimension mismatch, a singular matrix, a bad variable list, and an
    unrecognised option value.
    """


class DivergenceError(PycodemathError):
    """The quantity escaped to infinity.

    Raised by the ``_has_diverged`` family in ``engine.numerics``
    (``root_find``, ``root_find_nd``, ``minimize``, ``minimize_nd``), by an ODE
    solution that reaches NaN/inf, and by a divergent symbolic sum. Distinct
    from ``NonConvergenceError``: here the iterate provably blew up, rather
    than merely failing to settle.
    """


class StagnationError(PycodemathError):
    """The iteration stopped making progress while still off target.

    Raised where gradient descent stalls (the function value stops dropping
    while the gradient is still above tolerance) and where an adaptive ODE step
    shrinks below the minimum instead of stepping over a singularity. Distinct
    from ``DivergenceError``: nothing blew up — the method simply cannot
    advance from here.
    """


class NonConvergenceError(PycodemathError):
    """The iteration ended without reaching a value.

    Raised when a solver exhausts its iteration or step count, or when the
    iterate oscillates without dropping below tolerance — neither divergence
    nor stagnation. Also raised for a symbolic limit that does not exist
    because the expression oscillates.
    """


class NotAMinimumError(PycodemathError):
    """The iteration ARRIVED — at a point that is not the kind of point requested.

    Raised by ``minimize`` / ``minimize_nd`` (module 8) when the run reaches a
    STATIONARY point (``‖∇f‖`` under the tolerance, or an iterate that stopped
    moving) and a probe then finds a strictly lower value right next to it: a
    maximum, a saddle, or an inflection. The gradient test alone cannot tell those
    apart from a minimum — it is satisfied at EVERY stationary point — so before
    module 8 they came back as ``converged=True``.

    Distinct from all three iteration failures above, and the distinction is the
    remedy an agent should reach for:

      * ``DivergenceError`` — the iterate escaped; there may be no minimum at all.
      * ``StagnationError`` — the iterate stopped moving while still OFF target;
        the remedy is a smaller ``lr``.
      * ``NonConvergenceError`` — the budget ran out with no value to show; the
        remedy is more iterations.
      * here the run had NO trouble converging and MORE iterations cannot help:
        gradient descent can never leave a stationary point, because the step it
        would take there is zero. The remedy is a different STARTING POINT (beside
        the point, not on it) — or the acceptance that this function has no minimum
        here at all.

    The message names the point, the value there, and the nearby point that is
    lower — the evidence, not a verdict pulled out of a curvature sign. See
    ``engine.numerics._lower_value_nearby`` for what is probed and what the
    certificate does NOT promise.
    """


class BudgetExhaustedError(PycodemathError):
    """An explicit evaluation or iteration budget ran out.

    Raised by adaptive quadrature (``integrate_num`` with a ``tol``) when its
    ``max_evals`` allowance is spent before the requested tolerance is reached.
    Declared empty by module 2 and given this one producer in module 5, on a
    deliberate distinction: an iterative solver that stops at its own
    ``max_iter`` / ``max_steps`` cap raises ``NonConvergenceError``, because
    there the cap is a DIAGNOSIS — the run has no value to show and the problem
    itself is suspect. Here nothing is wrong with the integrand: an answer and a
    refinement in progress both exist, the estimate was still falling, and only
    the caller's evaluation allowance ended. That is a budget, not a diagnosis,
    and raising it again with the same ``tol`` finishes the job.
    """


class TimeBudgetError(PycodemathError):
    """The symbolic engine ran out of WALL CLOCK with nothing to show.

    Raised by every entry point in ``engine.symbolic`` and ``engine.linalg``
    (module 9) when SymPy was still working at the deadline: ``integrate
    1/(x^5+x+1) dx`` did not finish in 500 s, and before module 9 that was not a
    slow call — it was a call that never returned, with no result, no exception and
    no payload for an agent to branch on.

    WHY THIS IS NOT ``BudgetExhaustedError``, though both are called a budget. The
    words that separate them are in that class's own docstring: "raising it again
    with the same ``tol`` finishes the job". There, the run is making MEASURABLE
    progress — a composite value exists, the error estimate was falling, and the
    allowance is the only thing that ended, so more allowance ends the problem. Here
    there is no progress to measure and no partial answer to hand back: SymPy's
    ``integrate`` either returns an antiderivative or it does not, and doubling the
    budget on the reproducer buys another interval of nothing. The two also differ
    in what they CARRY — a ``BudgetExhaustedError`` run returns as data under
    ``full_result=True``, with ``evaluations`` and an ``error_estimate`` an agent can
    read; a symbolic run has no structured result at all, so this one travels the
    input-refusal road on the MCP wire (``error.type``, with ``solve`` and
    ``quadrature`` null). Sharing one class would file a remedy that works beside a
    remedy that does not, under a name that promises the first.

    Deliberately NOT paired with a ``SolveStatus``, and the asymmetry is the point:
    ``SOLVE_STATUSES`` is the vocabulary of ITERATION OUTCOMES, and every word in it
    describes where an iterate ended up. A symbolic run has no iterate, no last
    point and no residual — a status for it would be a word in ``SolveResult`` that
    no solver can produce and no caller can read off a result. See ``core.result``
    and ``tests/test_result.py``, which name this class among those left unpaired on
    purpose, beside ``ParseError`` and ``DomainError``.

    Carries the evidence of what it spent as ATTRIBUTES rather than in the message:
    ``operation`` (which entry point), ``budget`` (the allowance, in seconds) and
    ``spent`` (the wall clock actually used). The message quotes the budget, which is
    a number the CALLER chose, and never the time spent, which is a number the
    MACHINE chose — see ``core.budget.guard`` for why that keeps the message pinnable.

    Module 10 added ``route`` to that evidence, derived from ``operation`` through
    ``NUMERIC_ROUTES``: a budget that expired inside ``integrate`` carries
    ``"nintegrate"``, one that expired inside ``solve`` carries ``"root"``, and one
    that expired inside ``simplify`` carries ``None``. The message has told a reader
    to "compute numerically" since module 9; the route is the same advice as DATA,
    naming WHICH command, which the sentence never did.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str = "",
        budget: float = 0.0,
        spent: float = 0.0,
        route: "str | None" = None,
    ) -> None:
        super().__init__(message, route=route)
        self.operation = operation
        self.budget = budget
        self.spent = spent


class NoClosedFormError(PycodemathError):
    """The symbolic engine SEARCHED for an answer and did not find one.

    Raised where SymPy ran its method to the end and handed back nothing usable:
    an unevaluated ``Integral`` (``integrate exp(sin(x)) dx``), an unevaluated
    ``Sum`` (``summation 1/(k^2+k+1) k=1..oo``), an unevaluated ``Limit``, and a
    ``dsolve`` that cannot express ``y(t)`` explicitly (``y' = t^2 + y^2``).

    READ IT AS A STATEMENT ABOUT REACH, NOT AS A THEOREM. "No closed form" here
    means "not within this engine's methods", not "provably does not exist" —
    SymPy's integrator is not a complete decision procedure, and a rearranged or
    simplified input sometimes succeeds where the original did not. The class is
    honest about the difference; the message does not claim more.

    WHY THIS IS NOT ``UnsupportedFormError``, which sits next to it. There the
    engine refused BEFORE searching: it has no algorithm for the shape it was
    handed, and says so through a raised SymPy exception. Here the algorithm ran.
    The two differ in what a retry buys — reshaping the input can move a problem
    from the second class into the first, or out of both, while re-asking the same
    question in the same form cannot.

    Carries ``route`` where the numerical path asks the same question
    (``"nintegrate"`` for an integral, ``"ode"`` for an initial-value problem) and
    ``None`` where the command grammar has no counterpart — an infinite sum has
    none, and saying so in DATA is the point: the agent learns "there is nothing
    else to try here" without reading a sentence.

    WHY IT EXISTS (module 10's measurement). Before it, this outcome and three
    unrelated ones — no method for the shape, a limit that does not exist, and a
    call to the wrong function — all arrived as a bare ``PycodemathError`` with
    prose. All ten of the package's bare ``PycodemathError`` raises were on the
    symbolic surface, and an agent that wanted to know whether to STOP or to
    RE-ASK NUMERICALLY had to parse the English to find out. That is the thing the
    v0.3 series has spent eight modules removing everywhere else.
    """


class UnsupportedFormError(PycodemathError):
    """The symbolic engine has no METHOD for this shape — it never got to search.

    Raised where SymPy refused by raising rather than by returning: ``solve`` on
    an equation that mixes generators (``exp(x) + x^5 - 3`` — "multiple generators
    [x, exp(x)]"), or that it has no algorithm for at all (``besselj(0,x)``);
    ``limit`` hitting a ``PoleError``; ``series`` with no expansion available at
    the requested point (``besselj(0,1/x)`` around 0); ``summation`` raising out of
    its own machinery.

    THE SYMPY VOCABULARY STOPS HERE, and that is half the reason the class exists.
    "multiple generators [x, exp(x)]" is SymPy's internal wording for its own
    solver's limits — module 2 exists so that an agent never has to read a message
    to learn what happened, and that message was the loudest remaining leak. The
    original exception is still attached (``raise ... from exc``) for whoever is
    debugging; it is no longer in the text an agent receives, where it was also the
    one message whose exact bytes depended on the installed SymPy version.

    Carries ``route`` where a numerical command answers the same question —
    ``"root"`` for ``solve``, because a transcendental equation SymPy cannot
    rearrange is exactly what a numerical root finder is for. ``limit``, ``series``
    and ``summation`` carry ``None``: there is no numerical command for them in the
    grammar, and inventing one in a hint would be the dangerous kind of wrong.
    """


class IsolationError(PycodemathError):
    """The worker process behind ``isolated`` failed as a PROCESS (module B).

    ``core.backstop.isolated`` runs a call in a second process so that a call
    stuck in C can be killed. That buys a failure mode the in-process package never
    had: the worker can die without answering — killed from outside, out of
    memory, crashed inside a compiled extension — or finish with a result that
    cannot be pickled back. None of those is a statement about the mathematics.

    WHY THIS IS NOT ``TimeBudgetError``. There, the clock ran out and the remedy is
    a larger budget or a numerical route. Here the clock did not run out; the
    process did, and retrying the same call may kill the fresh worker the same way.
    Filing a crash under "slow" would send an agent to raise a budget that was
    never the problem.

    Carries ``exitcode`` — the worker's exit code when it died, ``None`` when it
    is still alive (the answer could not be sent back) or the code is unknown.
    ``route`` is always ``None``: there is no numerical counterpart to a crash.
    """

    def __init__(self, message: str, *, exitcode: "int | None" = None) -> None:
        super().__init__(message)
        self.exitcode = exitcode
