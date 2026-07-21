"""Pycodemath exceptions — one lightweight, shared domain error type.

Goal: the user/agent and the REPL get a readable Pycodemath message instead of
a raw SymPy exception (``SympifyError``, ``TypeError`` from ``float``, etc.).
"""

from __future__ import annotations


class PycodemathError(Exception):
    """Pycodemath domain error (parsing, evaluation, generation).

    Always carries a concise, readable message; the original exception is
    attached via ``raise ... from exc`` for inspection while debugging.
    """
