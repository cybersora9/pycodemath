"""Time budget for the symbolic engine — the call that never returns (module 9).

Modules 2-8 built one guarantee: a caller can always tell what happened. Failure
is a TYPE (module 2), success and failure both carry their EVIDENCE (3-5), the MCP
wire carries it (6), an installed caller's type checker sees it (7), and
``converged`` means "this value answers the question you asked" (8).

ONE failure mode escaped all of that: the call that never returns. Measured:

    integrate 1/(x^5+x+1) dx        did not finish in 500 s
    integrate log(x)/(x^5+x+1) dx   did not finish
    series exp(sin(exp(x))) n 14    did not finish

Bare SymPy does not finish either, so the COST is inherited, not caused here — and
that is exactly why it is Pycodemath's to handle rather than SymPy's to fix. There
was no timeout, no budget, no way to interrupt. Through MCP that is worse than a
wrong answer: a tool call that hangs forever, with no result, no exception and no
payload for an agent to branch on. Module 2 typed every failure that RETURNS. This
is the one that did not.

--- WHY ASYNCHRONOUS EXCEPTION INJECTION -------------------------------------

The mechanism has to be able to interrupt the actual hang. Each alternative was
measured, and each measurement rejected it:

* **A deadline checked between the engine's own STAGES.** There are no stages to
  check between. Profiled over 6 s of the reproducer: pycodemath-owned frames
  entered **47**, every one of them in the PARSE, which finishes in about a
  millisecond; SymPy frames entered in the same slice, **1 253 664**. After
  ``Expr.integrate`` calls ``sp.integrate`` the library is never on the stack
  again, so 100% of an unbounded wall clock falls into a single gap between two
  consecutive checkpoints. The granularity of a stage deadline is the whole hang.

* **SymPy's own interruption facilities.** There are none. (``sympy.cancel`` is
  the algebraic cancellation of a rational expression, not cancellation of work.)

* **``signal.setitimer`` / ``SIGALRM``.** ``hasattr(signal, "SIGALRM")`` is
  ``False`` on Windows and so is ``hasattr(signal, "setitimer")``; the CI matrix is
  Ubuntu AND Windows. Even on POSIX the handler only runs in the main thread, and
  the MCP server does not own the main thread.

* **``sys.settrace``.** It does interrupt the reproducer, promptly. It also costs
  **×6.25** on the fast path (``integrate sin(x)*x``: 15.8 ms bare, 98.6 ms
  traced). A guard that multiplies every answer to bound the answers that never
  come is not a trade this series makes.

* **A subprocess with a hard kill.** A bare interpreter spawn measured 66.8 ms and
  a spawn that imports pycodemath and runs one ``diff`` measured **868.7 ms** — on
  the order of a thousand times the cost of arming the guard chosen here, and
  roughly a thousand times the cheapest command in the table (``det [[1,2],[3,4]]``
  at 0.9 ms). It would also have to carry the IR across a process boundary, where
  the single-IR rule says there is one ``Expr`` and one place it lives.

What is used instead: ``PyThreadState_SetAsyncExc`` — CPython's own facility for
raising an exception in another thread — driven by ONE shared daemon watchdog for
the whole process. Measured: it interrupts the reproducer with an overshoot of
21-29 ms, and it works off the main thread.

--- WHAT IT COSTS THE ANSWERS THAT DO COME -----------------------------------

**3.5 us** for a guarded call that arms the watchdog, and **2.5 us** for one
nested inside it, which inherits instead of arming a second (measured in-process,
minimum of 11 series of 4000 calls). Against the commands in the cost table, which
run from 0.5 ms (``det [[1,2],[3,4]]``) to 27 ms (``code (sin(x)+cos(x))^2``), that
is between 0.01% and 0.7%.

The end-to-end A/B against a HEAD worktree is reported in the journal, but it does
NOT resolve this number and the honest reason is worth keeping here: running the
same measurement over the SAME tree twice, labelled HEAD and AFTER, produced
deltas from -5.5% to +4.8%. That is the noise floor of a cross-process comparison
on this machine, and it is larger than the effect being measured. The real
comparison against the working tree landed inside it (+0.6% to +5.4%, every delta
positive). So the claim made here is the in-process one, which is the same number
measured without a process boundary in the way.

For scale: module 8 measured, argued and accepted +37 to +280 us per call for the
minimum certificate. This is an order of magnitude cheaper, in absolute terms.

Two details are load-bearing, not incidental:

* ``_Deadline`` derives from ``BaseException``, not ``Exception``. SymPy is full of
  broad ``except Exception`` handlers, and so is this package —
  ``engine.linalg._FALLBACK_ERRORS`` catches ``ValueError``, ``TypeError`` and
  ``NotImplementedError`` in order to fall back to NumPy. An ``Exception`` would be
  swallowed by whichever handler the hang happened to be inside, and the guard
  would silently do nothing. Measured with the ``BaseException`` form: ONE
  injection is enough to break the reproducer.
* The watchdog is SHARED. A ``threading.Thread`` per call measured 217 us to start
  and join — four times the whole cost of ``diff sin(x)*x`` — so the per-call work
  here is a dict insert and a condition notify instead.

--- WHAT THE GUARD DOES NOT PROMISE ------------------------------------------

An asynchronous exception is delivered by the interpreter between two bytecodes.
A thread that is inside a single long C-level call (a large integer operation, a
compiled extension such as python-flint) does not check for it until that call
returns. Every hang measured here is pure-Python SymPy and is interrupted; a hang
that is not would run past its budget. The guard bounds what CPython lets it
bound, and this docstring is where that limit is written down rather than implied.
"""

from __future__ import annotations

import ctypes
import functools
import math
import threading
import time
from types import TracebackType
from typing import Callable, Literal, ParamSpec, TypeVar

from .errors import NUMERIC_ROUTES, TimeBudgetError

__all__ = ["time_budget", "DEFAULT_TIME_BUDGET", "TimeBudgetError"]

#: Seconds a symbolic call may take when the caller did not say otherwise.
#:
#: DEFENDED BY THE COST TABLE, not picked for roundness. Measured on this
#: machine, cold (one process per call, so every lazy SymPy import is paid inside
#: the number):
#:
#:   * every README example and its neighbours — the calls a budget must NEVER
#:     touch — cost at most **0.51 s** (``code (sin(x)+cos(x))^2``); the median is
#:     under 0.07 s;
#:   * the slowest call that finishes with a CORRECT ANSWER measured **14.7 s**
#:     (``integrate 1/(x^9+x+1) dx``), and the slowest that finishes with a correct
#:     REFUSAL measured **14.5 s** (``integrate exp(-x^2)*log(x)/(x^2+1) dx`` — "no
#:     closed form", which is an answer about the problem and must not be replaced
#:     by an answer about our patience);
#:   * the hangs do not finish at all — the reproducer was still running at 500 s.
#:
#: The honest finding is that there is NO gap to aim at. Inside one family —
#: ``integrate 1/(x^n+x+1) dx`` — the cost runs 0.18 s, 0.23 s, 0.44 s, 14.7 s and
#: then never; the legitimate tail is continuous and the failure is at infinity, so
#: no threshold separates "slow" from "never" by measurement. A default can only be
#: chosen generous enough that it never fires on work anyone would have waited for.
#:
#: **120 s errs on the generous side, deliberately.** It is 8x the slowest correct
#: call measured and 235x the slowest fast-path call, so on the evidence available
#: it cannot break a working call; and it is finite, which is the entire point,
#: because the alternative is not 500 s but forever. The cost of erring this way is
#: that an agent waits two minutes before being told to try ``nintegrate``. The cost
#: of erring the other way is a correct answer replaced by a false one — this series
#: has refused that trade since module 2, and refuses it here.
DEFAULT_TIME_BUDGET: float = 120.0

#: How long the message may quote of the offending expression. A budget failure
#: must name what it gave up on (house rule), and a runaway integrand must not turn
#: the message into a page.
_EXPR_IN_MESSAGE = 60


class _Deadline(BaseException):
    """The injected interrupt. NEVER seen by a caller — see the module docstring.

    ``BaseException`` on purpose: an ``Exception`` is swallowed by SymPy's own
    broad handlers (and by ``engine.linalg._FALLBACK_ERRORS``) and the guard would
    do nothing. Converted to the public ``TimeBudgetError`` by ``_Budget`` before it
    can leave the package.
    """


# --- the shared watchdog ---------------------------------------------------
# One daemon thread for the process. ``_ARMED`` maps a thread id to its slot:
# [deadline, depth, fired, chosen]. ``chosen`` is the ALLOWANCE THE CALLER ASKED
# FOR, kept beside the absolute deadline because those are the two different
# numbers a message and a timer respectively need: the deadline is a
# ``time.monotonic()`` reading and would make every message machine-dependent, and
# ``chosen`` is a number a test can pin. Everything is touched under ``_LOCK``,
# which is also the condition the watchdog sleeps on — so arming is a dict write
# plus a notify, and an idle process has a thread parked on ``wait()`` with no
# periodic timer at all.
_LOCK = threading.Lock()
_WAKE = threading.Condition(_LOCK)
_ARMED: dict[int, list] = {}
_WATCHDOG: "threading.Thread | None" = None

#: How often the watchdog re-injects into a thread that has passed its deadline
#: but has not yet died. One injection was enough for every hang measured, but a
#: ``finally:`` deep inside SymPy is entitled to run and could in principle absorb
#: the first one; repeating costs nothing on a thread that is already unwinding.
_REINJECT_EVERY = 0.05


def _set_async_exc(tid: int, exc: "type[BaseException] | None") -> None:
    """Raise ``exc`` in thread ``tid`` — or clear a pending one when ``None``."""
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid),
        ctypes.py_object(exc) if exc is not None else ctypes.py_object(),
    )


def _watch() -> None:
    """The watchdog loop: sleep until the nearest deadline, then inject.

    Holds ``_LOCK`` whenever it injects, and ``_Budget.__exit__`` takes the same
    lock before it stops being a candidate — so no injection can be *started* for
    a call that has already left the guard. One that was already in flight is
    cleared by ``__exit__``; see there.

    A slot whose deadline is ``inf`` is a thread that asked for NO budget. It stays
    in the table (so that a nested guard inherits "unlimited" instead of quietly
    falling back to the default) but it never becomes a wake-up time: an infinite
    timeout cannot be handed to ``Condition.wait``, and there is nothing to wait for.
    """
    with _LOCK:
        while True:
            if not _ARMED:
                _WAKE.wait()  # nothing armed — park, with no periodic wakeup
                continue
            now = time.monotonic()
            next_at = math.inf
            for tid, slot in _ARMED.items():
                if slot[0] <= now:
                    slot[2] = True
                    _set_async_exc(tid, _Deadline)
                    slot[0] = now + _REINJECT_EVERY
                next_at = min(next_at, slot[0])
            if math.isinf(next_at):
                _WAKE.wait()  # every armed thread is unlimited — nothing to time
            else:
                _WAKE.wait(max(0.001, next_at - time.monotonic()))


def _ensure_watchdog() -> None:
    """Start the watchdog on first use — never at import time.

    Importing a library must not spawn a thread: a process that only ever calls
    ``diff`` should carry no thread it did not ask for, and a fork-based worker
    should not inherit one. Called with ``_LOCK`` held.
    """
    global _WATCHDOG
    if _WATCHDOG is None:
        _WATCHDOG = threading.Thread(
            target=_watch, name="pycodemath-time-budget", daemon=True
        )
        _WATCHDOG.start()


class _Budget:
    """The armed guard. Built by ``time_budget``; not part of the public surface.

    ``body_completed`` is set by ``guard`` when the guarded call has ALREADY
    produced its answer. It settles the one case where the two halves of this class
    disagree about what a late interrupt means: for a caller's own
    ``with time_budget(...)`` block a deadline that passes is a refusal, full stop;
    but for an engine call whose value is already in hand, throwing that value away
    in favour of a refusal would be a lie about what happened.
    """

    __slots__ = (
        "seconds", "operation", "subject", "body_completed",
        "_tid", "_armed", "_started",
    )

    def __init__(self, seconds: float, operation: str, subject: object) -> None:
        self.seconds = seconds
        self.operation = operation
        self.subject = subject
        self.body_completed = False
        self._tid = 0
        self._armed = False
        self._started = 0.0

    def __enter__(self) -> "_Budget":
        self._tid = threading.get_ident()
        self._started = time.monotonic()
        # ``inf`` is the caller saying "I would rather wait forever than be
        # refused". It still takes a SLOT — with a deadline the watchdog can never
        # reach — because the slot is also what tells every entry point inside the
        # block that a budget is already in force. Arming nothing at all was the
        # first implementation and it was WRONG in the one way that matters:
        # measured, ``with time_budget(inf): series exp(sin(exp(x))) n 10`` was
        # refused after exactly 120 s, because the inner guard found no slot and
        # armed the DEFAULT. "No budget" that silently means "the default budget"
        # is worse than having no escape hatch at all.
        deadline = (
            math.inf
            if not math.isfinite(self.seconds)
            else time.monotonic() + self.seconds
        )
        with _LOCK:
            _ensure_watchdog()
            slot = _ARMED.get(self._tid)
            if slot is None:
                _ARMED[self._tid] = [deadline, 1, False, self.seconds]
            else:
                # RE-ENTRANT, and the TIGHTER deadline wins in both directions. A
                # nested call must never be able to extend the allowance its caller
                # chose, and an inner call that wants less time must get less. When
                # the inner one IS tighter it also becomes the allowance a refusal
                # quotes, because it is the one that will have run out.
                if deadline < slot[0]:
                    slot[0] = deadline
                    slot[3] = self.seconds
                slot[1] += 1
            self._armed = True
            _WAKE.notify()
        return self

    def _release(self) -> bool:
        """Give up this thread's claim on the watchdog. Returns whether it fired."""
        with _LOCK:
            slot = _ARMED.get(self._tid)
            fired = bool(slot and slot[2])
            if slot is not None:
                slot[1] -= 1
                if slot[1] <= 0:
                    del _ARMED[self._tid]
        return fired

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> "Literal[False]":
        """Disarm, and turn a deadline that fired into the public refusal.

        THE CLEANUP RETRIES, AND THAT IS THE WHOLE POINT. An asynchronous exception
        lands at an arbitrary bytecode — including one inside this method. The first
        version let that happen: ``_release`` was interrupted before it removed the
        slot, so the watchdog went on injecting into a thread that had left the
        guard, every 50 ms, forever. One missed cleanup then poisoned every later
        call on that thread. Measured on 40 000 runs timed to collide: **103
        escapes**, arriving in a cascade rather than singly, which is the signature
        of exactly that leak. Retrying until ``_release`` completes ends it — once
        the slot is gone no further injection can be started — and the loop
        terminates for the same reason. Re-measured on the same 40 000 runs after
        the fix: **0 escapes**, with the armed table empty at the end.

        Translating here rather than only in ``guard`` closes the other half of the
        same hole: ``time_budget`` is PUBLIC, so a caller may run any code at all
        inside it, with no ``guard`` anywhere on the stack. Before this, such a
        caller could be hit by a raw ``_Deadline`` — a ``BaseException`` that sails
        through every ``except Exception`` they wrote.
        """
        if not self._armed:
            return False
        interrupted = exc_type is _Deadline
        while True:
            try:
                fired = self._release()
                break
            except _Deadline:  # landed mid-cleanup — redo it, remember why
                interrupted = True
        if fired:
            # An injection may have been delivered between the body's last bytecode
            # and here. Clearing is a no-op if none was sent; if one has already
            # been raised it is on its way out and the loop above has seen it.
            _set_async_exc(self._tid, None)
        if exc_type is not None and not interrupted:
            # Something real is already on its way out — very often the
            # ``TimeBudgetError`` an INNER frame just built, which names the
            # operation and the expression where this frame only knows "this block".
            # A ``__exit__`` that replaced it would throw away the better message,
            # and a ``__exit__`` that replaced an unrelated error would be a bug.
            return False
        if interrupted and not self.body_completed:
            raise self.expired()
        return False

    def expired(self) -> TimeBudgetError:
        """The public refusal for this budget — one wording, built in one place.

        ``route`` (module 10) is looked up from the OPERATION, not from the message:
        the sentence has said "or compute numerically" since module 9 without ever
        naming which command that is, and an agent had to know the grammar to act on
        it. ``NUMERIC_ROUTES`` names it, and yields ``None`` for the operations that
        genuinely have no numerical counterpart rather than guessing one.
        """
        return TimeBudgetError(
            f"{self.operation}: gave up on {_quote(self.subject)} after the "
            f"{_format_seconds(self.seconds)} time budget — SymPy was still "
            f"working and there is no partial result. This is NOT 'no closed "
            f"form': an answer may well exist. Raise the budget "
            f"(pycodemath.time_budget) or compute numerically.",
            operation=self.operation,
            budget=self.seconds,
            spent=max(0.0, time.monotonic() - self._started),
            route=NUMERIC_ROUTES.get(self.operation),
        )


def time_budget(
    seconds: float, operation: str = "call", subject: object = "this block"
) -> _Budget:
    """Bound how long the symbolic engine may spend inside this block.

    The caller-facing half of module 9: "say in advance how long you are willing to
    wait". Every symbolic entry point already runs under
    ``DEFAULT_TIME_BUDGET``; this REPLACES that allowance for everything called
    inside the block, including entry points this module has never heard of.

        >>> from pycodemath import time_budget, symbolic, E
        >>> with time_budget(2.0):                        # doctest: +SKIP
        ...     symbolic.integrate(E("1/(x^5+x+1)"), "x")
        TimeBudgetError: integrate: gave up on 1/(x**5 + x + 1) after the 2s time budget …

    ``math.inf`` means unlimited — the way to opt out for a caller who would rather
    wait forever than be refused. It arms a deadline the watchdog can never reach
    rather than nothing at all; see ``_Budget.__enter__`` for the measurement that
    made the difference. A budget nested inside
    another does not extend it: the TIGHTER of the two deadlines wins, so a library
    call can shorten its own allowance but never lengthen its caller's.

    ``seconds`` must be strictly positive; zero or negative is a ``DomainError``,
    because a budget that has already expired before any work starts is a caller
    mistake and not a mathematical outcome.
    """
    # Local import: ``core.errors`` is imported at module scope for
    # ``TimeBudgetError``; ``DomainError`` is only needed on this refusal path.
    from .errors import DomainError

    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        raise DomainError(f"time_budget: seconds must be a number, not {seconds!r}")
    if math.isnan(seconds) or seconds <= 0:
        raise DomainError(
            f"time_budget: seconds must be positive, got {seconds!r} "
            f"(use math.inf for no budget at all)"
        )
    return _Budget(float(seconds), operation, subject)


T = TypeVar("T")

#: "the body has not produced a value yet" — distinct from any value it could
#: produce, including ``None``. See ``guard`` for the race this settles.
_UNSET: object = object()


def _quote(subject: object) -> str:
    """The offending expression, bounded — a message must name it, not become it."""
    text = str(subject)
    if len(text) > _EXPR_IN_MESSAGE:
        text = text[: _EXPR_IN_MESSAGE - 1] + "…"
    return text


def _format_seconds(seconds: float) -> str:
    """``120s`` / ``2.5s`` — the BUDGET, which is a chosen number, never the time
    actually spent, which is not. See ``guard`` for why that distinction is the
    difference between a pinnable message and a machine-dependent one."""
    return f"{seconds:g}s"


def guard(operation: str, subject: object, fn: Callable[[], T]) -> T:
    """Run ``fn`` under the active time budget and translate a hit into module 2.

    The single seam every symbolic entry point goes through, so the vocabulary and
    the message shape live in one place. When a budget is already active in this
    thread (the caller opened ``time_budget``, or an outer entry point armed one)
    this adds no second deadline; otherwise it arms ``DEFAULT_TIME_BUDGET``.

    ``subject`` is kept as the OBJECT and only turned into text on the refusal path.
    Stringifying it eagerly meant running SymPy's expression printer on every
    guarded call — real work, on the fast path, to build a message almost no call
    will ever need.

    THE MESSAGE CARRIES THE BUDGET, NEVER THE TIME SPENT. The time spent is
    machine-dependent — putting it in the text would make every message that
    mentions it unpinnable, and module 8's runtime snapshot compares exact
    ``(class, message)`` pairs. The evidence is not lost: it rides on the exception
    as ``.budget``, ``.spent`` and ``.operation``, where a caller can read it and a
    test never has to.

    TWO RACES, BOTH HANDLED HERE, AND THE FIRST ONE WAS MEASURED BITING. An
    asynchronous exception can be delivered at ANY bytecode boundary, including the
    ones inside ``_Budget.__exit__`` after the body has already finished:

      * The ``except`` must therefore sit OUTSIDE the ``with``, not inside it. With
        it inside, an injection landing during ``__exit__`` propagated as a raw
        ``_Deadline`` — a ``BaseException`` that would sail past every
        ``except Exception`` in the caller's code and kill the process at a point
        unrelated to the call that caused it. Measured at 2 escapes in 300 runs
        deliberately timed to collide; with this shape, 0 in 20 000.
      * If the injection lands after ``fn()`` has ALREADY RETURNED, the answer is in
        hand and must be honoured. ``_UNSET`` distinguishes "no result yet" from
        "result computed, interrupt arrived late", so a correct answer is never
        thrown away in favour of a refusal.
    """
    active = _budget_for(operation, subject)
    result: "T | object" = _UNSET
    try:
        with active:
            result = fn()
            # Tell ``__exit__`` the answer is already in hand, so an interrupt that
            # lands from here on cannot turn it into a refusal.
            active.body_completed = True
    except _Deadline:
        # Reached only where ``__exit__`` could not translate — it is the belt to
        # its braces, and it must never let the private class out of the package.
        if result is _UNSET:
            raise active.expired() from None
    return result  # type: ignore[return-value]


def _budget_for(operation: str, subject: object) -> _Budget:
    """The default allowance — unless this thread is already inside one.

    Returning an already-satisfied budget (rather than arming a second one) is what
    makes ``guard`` composable: ``solve`` calling ``simplify`` calling ``integrate``
    is ONE allowance, the caller's, not three stacked ones.
    """
    tid = threading.get_ident()
    with _LOCK:
        slot = _ARMED.get(tid)
        if slot is not None:
            # Already inside a budget: adopt the ALLOWANCE THAT WAS CHOSEN for the
            # message, and arm nothing (``_Inherited`` is a no-op context manager).
            return _Inherited(slot[3], operation, subject)
    return _Budget(DEFAULT_TIME_BUDGET, operation, subject)


class _Inherited(_Budget):
    """A budget that is already running — enters and exits without touching state.

    ``seconds`` reports the ALLOWANCE THE CALLER CHOSE, not the time left, so a
    nested entry point's refusal quotes the number the caller actually named rather
    than the default it did not — and quotes the same number every run, which is
    what keeps the message pinnable (see ``guard``).

    It still goes through the base ``__exit__``: the OUTER budget owns the watchdog
    slot, but this inner frame is the one that knows which OPERATION was running, so
    it is the one that can name it in the refusal.
    """

    __slots__ = ()

    def __enter__(self) -> "_Budget":
        self._tid = threading.get_ident()
        self._started = time.monotonic()
        self._armed = True  # armed enough to translate; owns no slot to release
        return self

    def _release(self) -> bool:
        return False  # the outer budget owns the slot and will release it


P = ParamSpec("P")


def under_budget(operation: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator: run this entry point under the time budget.

    ``ParamSpec``, not ``Callable[..., T]``, and that is not a style preference.
    Module 7 shipped ``py.typed`` so that an INSTALLED caller's type checker sees
    the real signatures; a decorator that collapsed them to ``(*args, **kwargs)``
    would quietly hand every guarded entry point back to ``Any`` and undo exactly
    the guarantee module 7 measured (14 of 14 revealed types were ``Any`` before it,
    0 after). With ``ParamSpec`` the wrapper's signature IS the wrapped one.

    Marks the function with ``__pcm_time_budget__`` so the test suite can enumerate
    the engine's public surface and prove that NONE of it is unguarded — the same
    mechanical completeness this series uses for statuses and exception subclasses,
    rather than trusting that the next function added will remember to opt in.

    The FIRST positional argument is quoted as the subject of a refusal: in every
    guarded signature that is the expression or the matrix being worked on.
    """

    def decorate(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> T:
            subject = args[0] if args else ""
            return guard(operation, subject, lambda: fn(*args, **kwargs))

        wrapper.__pcm_time_budget__ = operation  # type: ignore[attr-defined]
        return wrapper

    return decorate
