"""Moduł B: backstop podprocesem — twarda granica czasu, której wątek nie da.

POMIAR, KTÓRY TEN PLIK POWTARZA. ``7 ** (10**7)`` pod ``time_budget(2)`` w tym
samym procesie odmówiło po 7,88 s i 8,66 s — wtedy, kiedy wywołanie C samo
wróciło, bo póki trzyma GIL, watchdog nie wykona ani jednego bajtkodu.
``isolated(pow, 7, 10**7, budget=2.0)`` odmówiło po 2,515-2,531 s (5/5): rodzic
zabił proces roboczy. Szczegóły i liczby: docstring ``core.backstop``.

Reguły zegara jak w ``test_budget.py``: górna granica czasu tylko jako
WIELOKROTNOŚĆ budżetu i tylko po to, żeby udowodnić przerwanie wywołania, które
samo trwa minuty (``7 ** (10**8)`` biegło 281 s); dolne granice są bezpieczne.
"""

from __future__ import annotations

import math
import os
import time

import pytest

from pycodemath import E, isolated, parse
from pycodemath.core import backstop
from pycodemath.core.errors import (
    DomainError,
    IsolationError,
    NoClosedFormError,
    ParseError,
    TimeBudgetError,
)
from pycodemath.engine import symbolic


def _pid() -> int:
    backstop.warm()
    assert backstop._WORKER is not None
    return backstop._WORKER.proc.pid


# --- przezroczystość: ta sama odpowiedź, ten sam wyjątek ---------------------
def test_isolated_returns_what_the_call_returns():
    e = E("sin(x)*x")
    assert isolated(symbolic.integrate, e, "x", budget=30.0) == symbolic.integrate(
        e, "x"
    )


def test_isolated_reraises_the_same_class_and_message():
    # wyjątek przechodzi przez pickle — klasa i komunikat co do bajtu jak w procesie
    with pytest.raises(ParseError) as here:
        parse("sin(x")
    with pytest.raises(ParseError) as there:
        isolated(parse, "sin(x")
    assert str(there.value) == str(here.value)

    with pytest.raises(NoClosedFormError) as nc:
        isolated(symbolic.integrate, E("exp(sin(x))"), "x", budget=60.0)
    assert nc.value.route == "nintegrate"


def test_unlimited_budget_is_allowed():
    assert isolated(symbolic.diff, E("x^3"), "x", budget=math.inf) == E("3*x^2")


# --- twarda granica ----------------------------------------------------------
def test_hard_deadline_stops_a_c_level_call_and_the_next_call_works():
    # 7**(10**8) trzyma GIL przez minuty; bez zabicia procesu test by na to czekał
    backstop.warm()
    started = time.monotonic()
    with pytest.raises(TimeBudgetError) as exc:
        isolated(pow, 7, 10**8, budget=1.0)
    elapsed = time.monotonic() - started
    assert elapsed >= 1.0
    assert elapsed < 30.0  # przerwane, nie doczekane (patrz docstring pliku)
    assert exc.value.budget == 1.0
    assert exc.value.spent >= 1.0
    assert exc.value.operation == "pow"
    assert exc.value.route is None
    assert "worker process was stopped" in str(exc.value)
    # nowy proces roboczy wstaje sam przy następnym wywołaniu
    assert isolated(symbolic.diff, E("x^2"), "x") == E("2*x")


def test_soft_refusal_comes_from_the_worker_and_keeps_it_alive():
    # czysto-pythonowa zawieszka SymPy: przerywa ją zwykły strażnik W procesie
    # roboczym, więc odmowa jest ta sama co w procesie, a proces nie ginie
    pid = _pid()
    with pytest.raises(TimeBudgetError) as exc:
        isolated(symbolic.integrate, E("1/(x^5+x+1)"), "x", budget=1.0)
    assert exc.value.operation == "integrate"
    assert exc.value.route == "nintegrate"
    assert "worker process was stopped" not in str(exc.value)
    assert _pid() == pid


# --- porażki procesu, nie matematyki ------------------------------------------
def test_a_dead_worker_is_an_isolation_error_not_a_budget():
    with pytest.raises(IsolationError) as exc:
        isolated(os._exit, 3)
    assert not isinstance(exc.value, TimeBudgetError)
    assert exc.value.exitcode == 3
    assert exc.value.route is None
    assert isolated(symbolic.diff, E("x^2"), "x") == E("2*x")


def test_a_result_that_cannot_cross_back_is_an_isolation_error():
    pid = _pid()
    with pytest.raises(IsolationError) as exc:
        isolated(open, os.devnull)  # obiekt pliku nie przechodzi przez pickle
    assert exc.value.exitcode is None
    assert _pid() == pid  # proces żyje — zawiodła tylko droga powrotna


def test_an_interrupted_exchange_never_hands_a_stale_reply_to_the_next_call(
    monkeypatch,
):
    # Wątek rodzica przerwany W TRAKCIE wymiany (Ctrl+C, zewnętrzny time_budget):
    # worker liczy dalej i jego odpowiedź (tu: None ze sleep) zostałaby w potoku —
    # następne, niezwiązane wywołanie odebrałoby ją jako swoją. Worker ma zginąć.
    _pid()
    worker = backstop._WORKER
    assert worker is not None

    def interrupted(timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker.conn, "poll", interrupted)
    with pytest.raises(KeyboardInterrupt):
        isolated(time.sleep, 0.3)
    assert backstop._WORKER is None
    assert isolated(symbolic.diff, E("x^2"), "x") == E("2*x")


def test_an_unpicklable_function_is_refused_before_anything_runs():
    with pytest.raises(DomainError, match="picklable"):
        isolated(lambda: 1)


@pytest.mark.parametrize("bad", [0, -1.0, math.nan, True, "2"])
def test_budget_is_validated_like_time_budget(bad):
    with pytest.raises(DomainError):
        isolated(symbolic.diff, E("x"), "x", budget=bad)
