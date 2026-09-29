"""The subprocess backstop — the hard half of the time budget (module B, 29.09.2026).

``core.budget`` bounds what CPython lets it bound, and says where that ends: an
asynchronous interrupt is delivered between two bytecodes, so a thread inside ONE
long C-level call that holds the GIL does not see it until the call returns.
Measured for ``7 ** (10**7)`` under a 2 s budget: 7.5-8.9 s, "because nothing can
stop it sooner". This module is the thing that can.

--- WHY IT CANNOT LIVE IN THE SAME PROCESS -----------------------------------

The first question was whether anything in-process could end such a call, and
the answer is no, for a reason that is not an implementation detail. While the
call runs, the GIL is held and no other thread executes a single bytecode — the
watchdog's ``wait`` times out and cannot get the GIL back (that is finding A).
So the thread that is stuck is the caller's own thread, and the only code that
could hand the caller an answer cannot run. Killing the thread is not an option
either: a thread terminated while holding the GIL takes the GIL with it, and the
process deadlocks instead of hanging. The work has to be somewhere the CALLER is
not, and be killable from where the caller is. That is a second process.

--- WHY A WARM WORKER, AND WHY OPT-IN ----------------------------------------

A process per call was measured and rejected in module 9: **868.7 ms** for a
spawn that imports pycodemath and runs one ``diff``, against **3.5 us** for the
in-process guard. Nothing here changes that trade, so the in-process guard stays
the default for every entry point and this module does not touch its fast path.

What this module adds is ``isolated(fn, *args, budget=...)``: ONE worker process,
started on first use and then kept, which runs ``fn`` under the ordinary
``time_budget``. Measured on Windows / 3.12.10 (``symbolic.diff`` of
``sin(x)*x``, 3 x 200 calls): **0.57-0.73 ms** through the worker against
**0.16-0.22 ms** in-process, i.e. about +0.4-0.5 ms per call for the pickling and
the pipe — which is why this is opt-in and not the default. The spawn itself
(**1.2-1.3 s**, interpreter plus the pycodemath import) is paid once, and again
only after a kill — the respawn is ARMED BY an expired budget, never by a call
that finished. Kill-and-reap of a stuck worker measured **10 ms**.

THE CRITERION, measured (``7 ** (10**7)``, 2 s budget, same machine): in-process
``time_budget`` refused after **7.88 s and 8.66 s** — when the C call happened to
return; ``isolated(pow, 7, 10**7, budget=2.0)`` refused after **2.515-2.531 s**
in 5 of 5 runs, i.e. budget + ``_GRACE`` + the kill.

Two layers, each doing what it is good at:

* inside the worker the call runs under ``time_budget(budget)``, so every hang
  the in-process guard CAN interrupt (all the pure-Python SymPy ones) comes back
  as the usual ``TimeBudgetError`` at budget + tens of ms, and the worker lives
  on — no respawn;
* the parent waits ``budget + _GRACE`` for a reply. No reply means the worker is
  stuck where the in-process guard cannot reach; the parent kills it and raises
  ``TimeBudgetError`` itself. THIS is the hard guarantee: the caller's thread
  never runs the stuck code, so nothing can hold it past the deadline.

--- WHAT IT DOES NOT PROMISE -------------------------------------------------

* The budget clock starts when the request is SENT, after the worker is ready.
  A cold first call waits for the spawn (bounded by ``_SPAWN_TIMEOUT``) before
  its budget starts; ``warm()`` moves that cost to a moment of the caller's
  choosing.
* ONE worker, one call at a time: concurrent callers queue on a lock, and time
  spent queuing is not charged to their budget.
* ``fn``, its arguments and its result must cross a process boundary, so they
  must pickle — a module-level function (every pycodemath entry point does), not
  a lambda. The ``Expr``/``Matrix`` IR and the whole ``PycodemathError`` hierarchy
  pickle; checked by the tests, not assumed.
* Workers are started with ``spawn`` on every platform (``fork`` with the
  watchdog thread alive is not safe), which re-imports the caller's ``__main__``:
  a script that calls ``isolated`` needs the usual ``if __name__ == "__main__":``
  guard, exactly as with ``multiprocessing``.
* A worker that DIES without answering (killed from outside, out of memory, a
  crash inside a compiled extension) is ``IsolationError`` — not a budget: the
  clock did not run out, the process did.
"""

from __future__ import annotations

import math
import multiprocessing
import multiprocessing.connection
import pickle
import threading
import time
from typing import Any, Callable, TypeVar

from .budget import DEFAULT_TIME_BUDGET, _format_seconds, _quote, time_budget
from .errors import NUMERIC_ROUTES, DomainError, IsolationError, TimeBudgetError

__all__ = ["isolated", "warm"]

T = TypeVar("T")

#: How long past the budget the parent waits before it kills the worker.
#:
#: It must cover the SOFT refusal's own lateness, or the parent would kill a
#: worker that was about to answer and pay a 1.3 s respawn for nothing. That
#: lateness is the in-process guard's overshoot (21-29 ms, module 9), plus up to
#: one re-injection interval (50 ms), plus the reply's trip through the pipe.
#: Measured through this module (``integrate 1/(x^5+x+1)`` under 1 s, the
#: pure-Python hang, 20 runs on Windows / 3.12.10): the soft refusal arrived
#: **1.015-1.156 s** after the request, i.e. at most 156 ms late, and all 20 came
#: from the same worker (no kill). 0.5 s is three times that tail — generous on
#: purpose, because a premature kill is the expensive mistake here (a 1.2 s
#: respawn) — and small against any budget worth setting.
_GRACE: float = 0.5

#: How long a cold start may take before it is reported as a failure. Measured
#: 1.3 s on this machine; 60 s leaves room for a loaded CI runner and still ends.
_SPAWN_TIMEOUT: float = 60.0

_CTX = multiprocessing.get_context("spawn")


def _serve(conn: "multiprocessing.connection.Connection") -> None:
    """The worker loop: receive one request, answer it, repeat until EOF."""
    import pycodemath  # noqa: F401  (the import is the warm-up; pay it before "ready")

    conn.send(("ready", None))
    while True:
        try:
            payload = conn.recv_bytes()
        except EOFError:
            return
        fn, args, kwargs, budget, operation = pickle.loads(payload)
        try:
            subject = args[0] if args else "this call"
            with time_budget(budget, operation, subject):
                result = fn(*args, **kwargs)
            reply: "tuple[str, object]" = ("ok", result)
        except Exception as exc:  # crossed back to the caller, re-raised there
            reply = ("err", exc)
        try:
            conn.send_bytes(pickle.dumps(reply))
        except Exception as exc:
            # The answer (or the exception) exists but cannot cross the boundary.
            what = "result" if reply[0] == "ok" else f"{type(reply[1]).__name__}"
            conn.send_bytes(
                pickle.dumps(
                    (
                        "err",
                        IsolationError(
                            f"{operation}: the worker finished but its {what} "
                            f"could not be sent back ({type(exc).__name__}: {exc})"
                        ),
                    )
                )
            )


class _Worker:
    """One live worker process and the parent's end of its pipe."""

    def __init__(self) -> None:
        parent, child = _CTX.Pipe()
        self.conn = parent
        self.proc = _CTX.Process(
            target=_serve, args=(child,), name="pycodemath-backstop", daemon=True
        )
        self.proc.start()
        child.close()
        if not parent.poll(_SPAWN_TIMEOUT):
            self.kill()
            raise IsolationError(
                f"isolated: the worker process did not start within "
                f"{_format_seconds(_SPAWN_TIMEOUT)}"
            )
        parent.recv()  # ("ready", None)

    def kill(self) -> None:
        self.proc.kill()
        self.proc.join()
        self.conn.close()


_LOCK = threading.Lock()
_WORKER: "_Worker | None" = None


def _ensure_worker() -> _Worker:
    """The live worker, started if there is none. Called with ``_LOCK`` held."""
    global _WORKER
    if _WORKER is None or not _WORKER.proc.is_alive():
        _WORKER = _Worker()
    return _WORKER


def _drop_worker() -> "int | None":
    """Kill and forget the worker; return its exit code. ``_LOCK`` held."""
    global _WORKER
    worker, _WORKER = _WORKER, None
    if worker is None:
        return None
    worker.kill()
    return worker.proc.exitcode


def warm() -> None:
    """Start the worker now, so the first ``isolated`` call does not wait for it."""
    with _LOCK:
        _ensure_worker()


def isolated(
    fn: Callable[..., T],
    /,
    *args: Any,
    budget: float = DEFAULT_TIME_BUDGET,
    **kwargs: Any,
) -> T:
    """``fn(*args, **kwargs)`` in the worker process, with a HARD deadline.

    Returns what ``fn`` returns and raises what ``fn`` raises (the same class and
    message — the exception is pickled across), with one addition: a call still
    running at ``budget + _GRACE`` is stopped by killing the worker, and the caller
    gets ``TimeBudgetError`` at that moment instead of whenever the call would have
    returned.

        >>> from pycodemath import isolated, symbolic, E        # doctest: +SKIP
        >>> isolated(symbolic.integrate, E("x*sin(x)"), "x", budget=5.0)
        -x*cos(x) + sin(x)
        >>> isolated(pow, 7, 10**8, budget=1.0)
        TimeBudgetError: pow: gave up on 7 after the 1s time budget …

    ``budget`` follows ``time_budget``'s rules: strictly positive, and ``math.inf``
    means unlimited (then there is no deadline to enforce and the worker is never
    killed). See the module docstring for the measured costs and for what this does
    not promise.
    """
    time_budget(budget)  # the same validation, the same DomainError
    operation: str = str(
        getattr(fn, "__pcm_time_budget__", None) or getattr(fn, "__name__", "call")
    )
    try:
        payload = pickle.dumps((fn, args, kwargs, float(budget), operation))
    except Exception as exc:
        raise DomainError(
            f"isolated: {operation} and its arguments must be picklable to cross "
            f"into the worker process — pass a module-level function, not a lambda "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    wait = math.inf if not math.isfinite(budget) else budget + _GRACE
    with _LOCK:
        worker = _ensure_worker()
        started = time.monotonic()
        done = False
        try:
            worker.conn.send_bytes(payload)
            reply = None
            if worker.conn.poll(None if math.isinf(wait) else wait):
                reply = pickle.loads(worker.conn.recv_bytes())
                done = True
        except (EOFError, OSError) as exc:
            code = _drop_worker()
            raise IsolationError(
                f"{operation}: the worker process died without answering "
                f"(exit code {code})",
                exitcode=code,
            ) from exc
        finally:
            # A worker whose exchange did not close cleanly is never reused: after a
            # timeout it is still computing, and after an interrupt of THIS thread
            # (Ctrl+C, or an outer ``time_budget`` around ``isolated`` firing while
            # we waited) its reply would sit in the pipe and be read as the answer
            # to the NEXT, unrelated call. ``finally`` rather than ``except``,
            # because such an interrupt can also land just after the ``try``.
            if not done:
                _drop_worker()
        if reply is None:
            raise TimeBudgetError(
                f"{operation}: gave up on {_quote(args[0] if args else 'this call')} "
                f"after the {_format_seconds(budget)} time budget — the call was "
                f"stuck where the in-process guard cannot reach (one long C-level "
                f"call), so its worker process was stopped and there is no partial "
                f"result. This is NOT 'no closed form': an answer may well exist. "
                f"Raise the budget or compute numerically.",
                operation=operation,
                budget=float(budget),
                spent=time.monotonic() - started,
                route=NUMERIC_ROUTES.get(operation),
            )
    status, value = reply
    if status == "err":
        raise value  # type: ignore[misc]
    return value  # type: ignore[return-value]
