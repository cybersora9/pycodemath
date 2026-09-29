"""MCP server — Pycodemath as a math tool for any agent.

Exposes ``math_eval``, delegating to the REPL dispatcher (``repl.handle_full``):
the agent sends a concise command, gets an exact result from SymPy/NumPy —
instead of computing "in its head" or writing code. Its twin ``math_verify``
(``repl.handle_checked``, see the last section) CHECKS instead of computing. This is the
path that realizes the project goal of "cutting tokens" outside Claude Code (skill) — any
MCP client can plug in Pycodemath.

Launch (stdio transport):
    python -m pycodemath.cli.mcp_server      # or the script: pycodemath-mcp

The ``mcp`` dependency is optional (extra ``pycodemath[mcp]``). Everything below
except the registration itself works without it, so the payload an agent receives
is testable — and this module importable — with no MCP installed.

--- THE RESPONSE SHAPE (module 6) --------------------------------------------

Modules 2-5 built evidence an agent can branch on — a failure CLASS, a solver's
``SolveResult``, quadrature's ``QuadratureResult`` — and all of it used to die
here: the tool returned ``str``, so the model got prose to string-match. Now the
tool returns ``MathResult`` and MCP carries it as ``structuredContent``.

Four measurements shaped it, and each one closed off an alternative:

1. Module 1 proved ``structuredContent`` reaches the model, and that on Claude
   Code 2.1.220 it DISPLACES the text block — when a tool returns both, only the
   structured JSON is shown. So the human-readable answer cannot live in the text
   block alone; it is a FIELD (``text``), always present. That is also why the
   text and the structure are not alternatives here: both travel, in one object.
2. A tool that RAISES arrives as ``isError=true`` with ``structuredContent=null``
   — measured against a live client session. The MCP error channel has no
   structured half in this stack, so a refusal that raised would reach the model
   as exactly the prose this module exists to replace. Every outcome therefore
   RETURNS, and ``error`` is a field. (This is also what the tool did before, so
   nothing regressed: refusals already came back as ``"error: ..."`` text.)
3. Omitting a field is not available: ``NotRequired`` keys are still emitted as
   ``null`` by the serializer and then fail the client's own output validation.
   So every key is always present, and "not applicable" is spelled ``null``.
4. ``math.inf`` — module 4's "residual not measurable" and module 5's "error
   estimate not measurable" — is not expressible in JSON. It reaches the client
   as ``null`` and FAILS validation against a ``number`` field, killing the whole
   call. The two fields are therefore nullable, and ``inf`` is translated to
   ``null``: JSON's own word for "no value", carrying the meaning module 4 gave
   it. Nothing else in the payload may be non-finite.

The fields:

==============  ================================================================
``text``        the human-readable answer — byte-identical to what the REPL
                prints for the same command. On a run that did not converge it
                carries a caveat, so a client that renders only text cannot read
                a last iterate as an answer
``solve``       the ``SolveResult`` of ``root`` / ``min`` / ``solve_nd`` /
                ``min_nd``: ``value``, ``iterations``, ``residual``,
                ``converged``, ``status``. ``null`` for every other command
``quadrature``  the ``QuadratureResult`` of ``nintegrate``: ``value``,
                ``error_estimate``, ``evaluations``, ``refinements``,
                ``converged``, ``status``. ``null`` for every other command
``error``       the REFUSAL: ``type`` (the exception CLASS NAME — the
                agent-readable part of module 2's hierarchy), ``message``, and
                ``route`` (module 10 — the command that asks the same question
                numerically, or ``null``). ``null`` when the request was not refused
==============  ================================================================

--- WHY ``route`` EXTENDS ``error`` RATHER THAN BEING A THIRD FIELD (module 10) -

``solve`` and ``quadrature`` are separate fields because each is the RESULT of a
run and the field NAME discriminates which kind. A symbolic refusal is not a
result, and module 10 measured that it has no fields to be one: at the moment
``integrate`` refuses there is no iterate, no residual, no evaluation count and no
partial value — only the fact that the search ended. A third result field would
therefore have carried a status and nothing else, which is ``error.type`` wearing
a dataclass.

What a symbolic refusal DOES have that ``error`` could not previously carry is
what to do NEXT, and that is one datum: ``route``. It is the one thing an agent
could not derive from ``type`` plus ``message`` without reading English — the same
class refuses ``integrate`` (where ``nintegrate`` asks the same question) and an
infinite ``summation`` (where nothing in this grammar does), and telling those
apart is the difference between retrying usefully and retrying pointlessly.

A route is a fact about the API, never a promise about the answer: the command may
fail, and it may need input the symbolic call did not (``nintegrate`` needs
bounds — a definite integral is a different question from an antiderivative;
``root`` needs a starting point). ``core.errors.ROUTES`` is the closed vocabulary
and every value in it is pinned to a command this server actually dispatches.

``solve`` and ``quadrature`` are separate fields on purpose. Module 5 gave
quadrature its own type because an error ESTIMATE is not a residual — a residual
is measured, an estimate is inferred under an assumption that can fail — and
flattening both into one loose ``result`` object would undo exactly that
distinction. Here the field NAME is the discriminator, so an agent never has to
sniff which kind of number it got.

There is no ``ok`` flag: ``error is null`` already says whether the request was
refused, and ``converged`` already says whether the value is an answer. A third
boolean would be a second encoding of a fact these fields already carry — the
kind of drift modules 3-5 spent one constructor per form to prevent. The same
discipline applies here: ``_answer`` and ``_refusal`` are the only two ways a
payload is built.

--- WHICH FAILURES TRAVEL AS DATA, AND WHICH AS ERRORS -----------------------

The line is module 4's, unchanged: an ITERATION OUTCOME is data, an INPUT REFUSAL
is an error.

  * ITERATION OUTCOMES — divergence, stagnation, non-convergence, an exhausted
    quadrature budget — arrive with ``error=null`` and a populated
    ``solve``/``quadrature`` whose ``converged`` is ``false``. There IS something
    to report: the last iterate, the residual, the cost. Modules 4-5 made exactly
    these returnable, and this is where that pays off.
  * INPUT REFUSALS — ``ParseError`` (the text was refused before any math) and
    ``DomainError`` (a bad value, shape, option or point) — arrive with
    ``error.type`` naming the class and ``solve``/``quadrature`` ``null``. There
    is no last iterate, no residual, no run: an input refusal is not an outcome,
    and inventing a status for it would be a lie.

The boundary is visible on the wire: the SAME command can produce either form,
and the shapes differ completely.

One honest limit. A failure travels as data only where modules 4-5 built a
returnable form — the four solvers and quadrature. Everywhere else (ODE,
symbolic, linear algebra, code generation) a failure still travels as an error —
but with its module-2 class name in ``error.type``, which is already the
agent-readable half. Those engines have no structured result today; giving them
one is engine work, and this module does not touch the engine.

--- THE SECOND TOOL: ``math_verify`` (VERIFY V4) ------------------------------

``math_eval`` COMPUTES; ``math_verify`` CHECKS — an identity, a derivation, or an
engine answer — and its answer is a VERDICT: ``verified`` (proved), ``refuted``
(a counterexample, confirmed at two precisions) or ``undecided`` (neither — a
first-class result, never rounded up). It is a second tool rather than more
fields on ``math_eval`` because the two answer different questions, and one
schema carrying both would put ``solve``/``quadrature`` next to verdict fields
that can never be set together — every client would branch around nulls that
say nothing.

Every rule above is kept, not restated differently: one-line docstrings on the
types, every key always present with ``null`` for "not applicable", a refusal
RETURNED as the same ``Refusal`` object (never raised: ``isError`` carries no
structure), and no ``ok`` flag. The same discipline on the new fields:

==============  ================================================================
``text``        the verdict in words — byte-identical to what the REPL prints
                for the same ``verify`` / ``certify`` line
``answer``      ``certify`` only: the engine's answer that was certified, exactly
                as ``math_eval`` would print it. ``null`` otherwise
``verdict``     ``verify <a> == <b>`` and ``certify``: ``status``, ``method``,
                ``counterexample`` (variable -> value, set exactly when
                ``refuted``), ``detail``. ``null`` for a derivation
``steps``       ``verify steps``: the derivation's ``status``, ``first_error``
                (step number), ``counterexample``, and one ``checks`` entry per
                step with its own verdict, ``effect`` and ``warning``
``error``       the refusal, as in ``math_eval``
==============  ================================================================

``verdict`` and ``steps`` are separate for the reason ``solve`` and
``quadrature`` are: the field NAME says which kind of evidence arrived. A
derivation's ``status``/``first_error``/``counterexample`` are the same
properties ``StepsResult`` exposes, carried over, not a second encoding of the
per-step verdicts. A counterexample holds only strings (``{"x": "-1"}``), so no
verdict payload ever carries a number JSON cannot express.
"""

from __future__ import annotations

import math
try:  # pydantic (pulled in by mcp) refuses typing.TypedDict below Python 3.12:
    # on 3.11 the module did not import at all, and CI was red on both 3.11 jobs.
    from typing_extensions import TypedDict
except ImportError:  # no mcp installed, so no pydantic to satisfy
    from typing import TypedDict  # type: ignore[assignment]

from ..core.result import QuadratureResult, SolveResult
from ..verify import StepsResult, Verdict
from . import repl

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # optional dependency missing — refusal deferred to run()
    # importing the module itself must not blow up with a traceback; a readable
    # message goes to whoever actually tries to start the server
    FastMCP = None  # type: ignore[assignment, misc]


# NOTE — these docstrings are SERIALIZED into the tool's output schema and travel
# to every client that lists the tool, so they stay one line each. The reasoning
# behind the shape lives in the module docstring above, which does not travel.
class SolveEvidence(TypedDict):
    """A solver run: value, iterations, residual, converged, status."""

    value: "float | list[float | None] | None"
    iterations: int
    residual: "float | None"
    converged: bool
    status: str


class QuadratureEvidence(TypedDict):
    """An integral: value, error_estimate (an ESTIMATE, never a bound), cost, status."""

    value: "float | None"
    error_estimate: "float | None"
    evaluations: int
    refinements: int
    converged: bool
    status: str


class Refusal(TypedDict):
    """A refused request: type (exception class), message, route (numerical retry)."""

    type: str
    message: str
    route: "str | None"


class MathResult(TypedDict):
    """The answer (text) plus the evidence behind it; error is set when refused."""

    text: str
    solve: "SolveEvidence | None"
    quadrature: "QuadratureEvidence | None"
    error: "Refusal | None"


class VerdictEvidence(TypedDict):
    """A verdict: status (verified/refuted/undecided), method, counterexample, detail."""

    status: str
    method: str
    counterexample: "dict[str, str] | None"
    detail: str


class StepEvidence(TypedDict):
    """One step judged against the one before: number, text, verdict, effect, warning."""

    number: int
    text: str
    verdict: VerdictEvidence
    effect: "str | None"
    warning: "str | None"


class StepsEvidence(TypedDict):
    """A derivation: kind, status, first_error (step number), counterexample, checks."""

    kind: str
    status: str
    first_error: "int | None"
    counterexample: "dict[str, str] | None"
    checks: "list[StepEvidence]"


class VerifyResult(TypedDict):
    """The verdict in words (text) and as data; error is set when refused."""

    text: str
    answer: "str | None"
    verdict: "VerdictEvidence | None"
    steps: "StepsEvidence | None"
    error: "Refusal | None"


def _finite(x: float) -> "float | None":
    """``None`` for anything JSON cannot carry — i.e. for "not measurable".

    ``math.inf`` is module 4's rule for a residual that could not be sampled and
    module 5's for an error estimate over an unrefined interval. JSON has no
    infinity; sending it produces ``null`` anyway and then fails the client's own
    schema validation, so the translation is made here, deliberately and once.
    """
    return x if math.isfinite(x) else None


def _answer(
    text: str,
    *,
    solve: "SolveEvidence | None" = None,
    quadrature: "QuadratureEvidence | None" = None,
) -> MathResult:
    """Build the ANSWERED form: ``error`` is ``null``, whatever else it carries."""
    return {"text": text, "solve": solve, "quadrature": quadrature, "error": None}


def _refusal(exc: Exception) -> MathResult:
    """Build the REFUSED form: no evidence, and the exception CLASS on the wire.

    ``route`` (module 10) is read with ``getattr``, not ``exc.route``, and the
    default is not defensive padding: ``math_eval`` catches ``Exception``, so a bug
    in the library arrives here as some non-Pycodemath class that has no such
    attribute. Those get ``null``, which is the truth about them.
    """
    return {
        "text": f"error: {exc}",
        "solve": None,
        "quadrature": None,
        "error": _refused(exc),
    }


def _refused(exc: Exception) -> Refusal:
    """The ``error`` object — one builder for both tools, so they cannot drift."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "route": getattr(exc, "route", None),
    }


def _solve_evidence(result: "SolveResult[object]") -> SolveEvidence:
    value = result.value
    return {
        "value": (
            [_finite(float(v)) for v in value]
            if isinstance(value, list)
            else _finite(float(value))  # type: ignore[arg-type]
        ),
        "iterations": result.iterations,
        "residual": _finite(result.residual),
        "converged": result.converged,
        "status": result.status,
    }


def _quadrature_evidence(result: QuadratureResult) -> QuadratureEvidence:
    return {
        "value": _finite(result.value),
        "error_estimate": _finite(result.error_estimate),
        "evaluations": result.evaluations,
        "refinements": result.refinements,
        "converged": result.converged,
        "status": result.status,
    }


def math_eval(command: str) -> MathResult:
    """Compute math exactly (SymPy/NumPy engine) — one command, one result.

    Commands (power notation: ^ or **):
      <expression>                     — simplify (e.g. sin(x)^2 + cos(x)^2)
      diff <expr> d<var>               — derivative (e.g. diff sin(x)*x dx)
      integrate <expr> d<var> [budget <s>] — symbolic integral
      solve <expr> for <var> [budget <s>]  — solve expr = 0
      limit / series / sum             — as in the full grammar; each also takes [budget <s>]
      code <expr>                      — optimized Python/NumPy code (CSE)
      matrix <A>                       — show the matrix, e.g. matrix [[1,2],[3,4]]
      det/inv/transpose/eig <A>        — matrix, e.g. det [[1,2],[3,4]]
      solve_system <A> = <b>           — linear system A x = b
      code_system <A> = <b>            — NumPy code solving the system
      root <expr> for <var> at <x0>    — root f=0 numerically
      nintegrate <expr> d<var> from <a> to <b> [tol <t>] — numerical integration
      min <expr> for <var> at <x0> [method newton|bfgs] [tol <t>] [max_iter <k>] — 1D minimum (gradient descent by default; newton/bfgs = 2nd order + Armijo)
      grad <expr> for <x,y>            — symbolic gradient
      solve_nd <f1>; <f2> for <x,y> at <x0,y0> — nonlinear system (Newton)
      min_nd <expr> for <x,y> at <x0,y0> [method newton|bfgs] [tol <t>] [max_iter <k>] — minimum of several variables (bfgs copes e.g. with the Rosenbrock valley)
      dsolve <expr> for y(t) [budget <s>]  — ODE symbolically: y' = f(t,y)
      ode <expr> for y(t) from <t0> to <t1> at <y0> [steps <n>] — ODE numerically (RK4)
      odestiff <expr> for y(t) from <t0> to <t1> at <y0> [steps <n>] — STIFF ODE (implicit BDF2 + Newton; e.g. odestiff -1000*(y-cos(t)) for y(t) from 0 to 1 at 0)
      odestiff_adaptive <expr> for y(t) from <t0> to <t1> at <y0> [rtol <r>] — STIFF ODE with adaptive step (adaptive BDF2; an order fewer steps than DOPRI5 on a stiff problem)
      ode_adaptive <expr> for y(t) from <t0> to <t1> at <y0> [rtol <r>] — ODE, adaptive step (DOPRI5)
      odedense <expr> for y(t) from <t0> to <t1> at <y0> at t=<point> [rtol <r>] — ODE, read y(t) at any point (dense output)
      odeevents <expr> for y(t) from <t0> to <t1> at <y0> zero <g> [dir <+1|-1>] [rtol <r>] [stop] — ODE + moments g(t,y)=0 (event detection; stop = terminal event, stop integration)
      To CHECK an identity, a derivation or one of these answers, use the math_verify tool.

    TRAILING OPTIONS — ``key value`` pairs at the END of a command, any order,
    each at most once. They are the knobs behind the evidence fields, so a result
    you can READ is now also one you can ACT on:
      tol <t>       min/min_nd: accept when the gradient falls under t (default
                    1e-9) — a "not_converged" descent often converges under a
                    looser t. nintegrate: refine until the estimated absolute
                    error fits inside t (default: fixed 100-panel rule, no
                    target) — this is how you AIM ``error_estimate``. Tighter
                    costs more evaluations/iterations, looser costs accuracy.
                    Must be finite and positive: "tol inf" would certify the
                    untouched starting guess as "converged".
      max_iter <k>  min/min_nd iteration allowance (default 10000). A
                    "not_converged" run whose residual was still falling
                    finishes with more; "not_a_minimum" never does (see below).
      budget <s>    seconds THIS symbolic call may take (default 120). Smaller
                    buys a fast typed refusal instead of a two-minute wait;
                    larger buys more search after a "TimeBudgetError". Must be
                    finite and positive on this surface.
    A malformed option is never silently ignored: an unknown or repeated key or
    an unreadable value is a "ParseError"; a readable value outside its domain
    (tol 0, budget -1) is a "DomainError".

    THE RESPONSE — read the fields, do not parse the prose:
      text          the readable answer (always)
      solve         root/min/solve_nd/min_nd: value, iterations, residual,
                    converged, status. null for other commands
      quadrature    nintegrate: value, error_estimate, evaluations, refinements,
                    converged, status. null for other commands
      error         the request was REFUSED: type (exception class), message, and
                    route — the command that asks the same question numerically,
                    or null when there is none. null otherwise

    Branch on ``error`` (was it refused?) and on ``converged`` (is the value an
    answer?). A run that did not converge is NOT an error: it reports its last
    iterate and its status ("diverged" / "stagnated" / "not_converged" /
    "not_a_minimum"). A ``residual`` or ``error_estimate`` of null means the
    quantity could not be measured. ``error_estimate`` is an estimate, never a bound.

    ``converged`` answers the question that was ASKED — is this a ROOT, is this a
    MINIMUM — and NOT the weaker "did an exit test fire". Two consequences worth
    knowing before you branch on anything else:
      * ``residual`` is not a substitute for it, IN EITHER DIRECTION. A refused run
        can carry a residual of 1e-44 (``root exp(x) for x at 0``: no root exists,
        the function merely DECAYED below the tolerance on the way down), and an
        accepted minimum can carry a residual well above ``tol`` when plain gradient
        descent has driven the value as far as float64 allows.
      * "not_a_minimum" means the run ARRIVED — at a stationary point that is a
        maximum, a saddle or an inflection (``min -x^2 for x at 0``). Do NOT retry
        it with more iterations: the step at a stationary point is zero, so the run
        cannot move however long you let it. Retry from a DIFFERENT starting point,
        beside that one.

    EVERY SYMBOLIC COMMAND IS BOUNDED IN TIME (120 s by default; ``budget <s>``
    replaces that for one call). Some inputs make SymPy work
    forever — ``integrate 1/(x^5+x+1) dx`` did not finish in 500 s — and this call
    used to hang with them, returning nothing you could branch on. It now comes back
    with ``error.type = "TimeBudgetError"``, and ``solve``/``quadrature`` null,
    because a symbolic run has no last iterate and no residual to report.

    Read that type literally, and do NOT confuse it with the refusals next to it:
      * "TimeBudgetError" says nothing about whether an answer EXISTS — only that we
        stopped looking. Retrying the same command unchanged changes nothing;
        retry with a bigger ``budget <s>`` if the wait is worth it to you
        (measured: an integral refused at 5 s answers at 60), or take
        ``error.route``.
      * "NoClosedFormError" is a different claim: the engine SEARCHED and found no
        closed form. It is reached in seconds, on purpose, rather than being
        replaced by a timeout. Read it as a limit of this engine's reach, not as a
        proof that no answer exists — a simplified or rearranged input sometimes
        succeeds where the original did not.
      * "UnsupportedFormError" is the third: the engine has no METHOD for this
        shape and never got as far as searching (``solve exp(x)+x^5-3 for x`` mixes
        ``x`` with ``exp(x)``, which its algebraic solver cannot separate). This is
        the one where the numerical route is most often the right answer.

    A SYMBOLIC REFUSAL TELLS YOU WHAT TO DO NEXT, IN ``error.route``. It names the
    command that asks the same question numerically, or is null when this grammar
    has none:

      integrate exp(sin(x)) dx    NoClosedFormError     route "nintegrate"
      solve exp(x)+x^5-3 for x    UnsupportedFormError  route "root"
      dsolve y^2+t^2 for y(t)     NoClosedFormError     route "ode"
      summation over an infinite range                  route null — nothing else
                                                        to try; change the question

    A route is a fact about this API, not a promise about the answer. The command
    may itself fail, and it may need input the symbolic call did not: ``nintegrate``
    needs BOUNDS (a definite integral is a different question from an
    antiderivative) and ``root`` needs a STARTING POINT. A null route is
    informative in its own right — it means retrying numerically is not available,
    so reformulate rather than retry.

    A route names a COMMAND, never a command with options attached. Which command
    asks the same question is a fact about this API; how tightly to ask it
    (``tol``, ``budget``) depends on what you need the answer for, which the
    refusal cannot know — and a route with prose attached could no longer be
    mechanically checked against the dispatch table. The routed command's own
    options are yours to add: ``nintegrate`` takes ``tol``.
    """
    try:
        answer = repl.handle_full(command)
    except Exception as exc:  # the agent gets DATA, not a traceback (measurement 2)
        # ``PycodemathError`` subclasses are the refusals module 2 typed; anything
        # else would be a bug in the library, and its class name says so just as
        # plainly. Both take the same road, because both leave nothing to report.
        return _refusal(exc)

    evidence = answer.evidence
    text = answer.text or "(empty result)"
    if isinstance(evidence, SolveResult):
        return _answer(text, solve=_solve_evidence(evidence))
    if isinstance(evidence, QuadratureResult):
        return _answer(text, quadrature=_quadrature_evidence(evidence))
    return _answer(text)


def _verdict_evidence(verdict: Verdict) -> VerdictEvidence:
    return {
        "status": verdict.status.value,
        "method": verdict.method,
        "counterexample": verdict.counterexample,
        "detail": verdict.detail,
    }


def _steps_evidence(result: StepsResult) -> StepsEvidence:
    return {
        "kind": result.kind,
        "status": result.status.value,
        "first_error": result.first_error,
        "counterexample": result.counterexample,
        "checks": [
            {
                "number": c.number,
                "text": c.text,
                "verdict": _verdict_evidence(c.verdict),
                "effect": c.effect,
                "warning": c.warning,
            }
            for c in result.checks
        ],
    }


def _checked(checked: "repl.Checked") -> VerifyResult:
    """Build the CHECKED form of ``math_verify``: ``error`` is ``null``."""
    return {
        "text": checked.text or "(empty result)",
        "answer": checked.answer,
        "verdict": (
            None if checked.verdict is None else _verdict_evidence(checked.verdict)
        ),
        "steps": None if checked.steps is None else _steps_evidence(checked.steps),
        "error": None,
    }


def _verify_refusal(exc: Exception) -> VerifyResult:
    """Build the REFUSED form of ``math_verify`` — ``_refusal``'s twin."""
    return {
        "text": f"error: {exc}",
        "answer": None,
        "verdict": None,
        "steps": None,
        "error": _refused(exc),
    }


def math_verify(command: str) -> VerifyResult:
    """Check math instead of computing it — a verdict: verified, refuted or undecided.

    Commands (power notation: ^ or **; the leading word "verify" is optional):
      <a> == <b> [budget <s>]          — is a == b an identity? (e.g. sqrt(x^2) == x)
      steps <derivation> [budget <s>]  — check a derivation step by step: an
                                         expression chain "a = b = c" (one line,
                                         or one step per line starting with =),
                                         or equations one per line / joined by
                                         -> (e.g. steps 2x + 3 = 7 -> 2x = 4 -> x = 2)
      certify <command>                — run a math_eval command (integrate, diff,
                                         solve, limit, dsolve, nintegrate) and
                                         check its answer by an independent route
                                         (e.g. certify integrate x*cos(x) dx)

    THE THREE WORDS — read them literally:
      verified   PROVED (a symbolic argument closed), not "probably true".
      refuted    FALSE, with a counterexample: a point (variable -> value, as
                 text the parser reads back) confirmed at two precisions.
      undecided  neither. Agreement at every sampled point is undecided, never
                 verified. A first-class answer, not a failure: the same command
                 gives the same word (the sample points are fixed) — except an
                 undecided with method "time-budget", which a larger budget may
                 settle.

    THE RESPONSE — read the fields, do not parse the prose:
      text      the verdict in words (always)
      answer    certify: the engine's answer that was certified. null otherwise
      verdict   <a> == <b> and certify: status, method, counterexample (set
                exactly when refuted), detail. null for steps
      steps     steps: kind, status, first_error (the number of the first
                refuted step, from 1; step 1 is the starting point), its
                counterexample, and checks — one per step with its own verdict,
                effect ("gains-roots" / "loses-roots" on an equation step that
                changes the solution set) and warning. null otherwise
      error     the request was REFUSED: type (exception class), message, route.
                null otherwise

    Branch on ``error``, then on ``verdict.status`` / ``steps.status``.

    WHAT EQUAL MEANS: values on the principal branch, complex allowed —
    ``log(x^2) == 2*log(x)`` is refuted at x = -1; equality where BOTH sides are
    defined — ``(x^2-1)/(x-1) == x+1`` is verified; decimals mean the decimal
    written — ``0.1 + 0.2 == 0.3`` holds. Equation steps compare solution sets
    over the reals: squaring both sides is refuted with effect "gains-roots".

    A CERTIFICATE certifies what the table says and nothing more: dsolve — every
    returned function solves the equation, NOT that they are all the solutions;
    solve — completeness is proved only for rational functions, elsewhere a
    missing real root can be found but not ruled out; nintegrate — the value is
    within the run's own error_estimate.

    TIME: every check is bounded (120 s by default; ``budget <s>`` replaces it).
    Running out is an undecided verdict with method "time-budget", not an error.
    For certify, the command's own ``budget`` bounds the computation and,
    separately, the certificate. A refusal of the computation itself
    (NoClosedFormError, TimeBudgetError...) arrives in ``error`` with its route,
    exactly as math_eval returns it.
    """
    line = command.strip()
    head = line.split(None, 1)[0].lower() if line else ""
    if head not in ("verify", "certify"):
        line = f"verify {line}"
    try:
        checked = repl.handle_checked(line)
    except Exception as exc:  # the agent gets DATA, not a traceback (measurement 2)
        return _verify_refusal(exc)
    return _checked(checked)


server: "FastMCP | None" = None
if FastMCP is not None:
    server = FastMCP("pycodemath")
    server.tool()(math_eval)
    server.tool()(math_verify)


def run() -> None:
    """Run the server over stdio (the default transport for MCP clients)."""
    if server is None:
        # SystemExit with a string: message on stderr + code 1, no traceback
        # (works the same for the pycodemath-mcp entry point and python -m)
        raise SystemExit(
            "error: the MCP server requires the 'mcp' package "
            "(install: pip install pycodemath[mcp])."
        )
    server.run()


if __name__ == "__main__":
    run()
