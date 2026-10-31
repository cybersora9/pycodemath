"""``pycodemath lint`` — the mathematics in your code, checked, with a witness for every finding (L1).

    python -m pycodemath.lint src/                  # text, like ruff
    python -m pycodemath.lint src/ --format sarif   # GitHub code scanning
    pytest --pcm -p pycodemath.lint.pytest_plugin   # every :math: / @spec as a test

THE SPECIFICATION CONVENTION — one syntax for lint, pytest and agents. A function
states what it computes in its docstring or with a decorator::

    def sum_to(n: int) -> int:
        \"\"\"Sum of 1..n.

        :math: n*(n+1)/2
        :domain n: integer, n>=0
        \"\"\"

    @spec("sqrt(x + 1) - sqrt(x)", ranges={"x": (1.0, 1e15)})
    def gap(x: float) -> float: ...

``:math: <formula>`` alone on a docstring line is the formula in Python / pycodemath
notation (``**`` or ``^`` for powers), in the function's PARAMETERS — a line whose
names are not parameters (or known functions/constants) is not a spec. Sphinx's
inline role ``:math:`...``` is documentation (LaTeX) and is never read as a spec.
``:domain <name>: <text>`` uses C1's domain words (``integer real positive nonnegative`` and comparisons).
``:range <name>: [low, high]`` (``[low, high, int]`` for an integer-valued float)
bounds the inputs for the precision (C2) and range (C3) rules. ``spec`` is the
same as a decorator; it returns the function unchanged (it only records the spec
in ``__pcm_spec__``), so it costs nothing at run time and importing it does not
import SymPy.

Every name except ``spec`` loads lazily (PEP 562).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable, Mapping, TypeVar

if TYPE_CHECKING:
    from .config import Config, ConfigError, load_config
    from .engine import Finding, LintResult, check_spec, lint_paths, lint_source, spec_functions
    from .rules import RULES, Rule
    from .witness import Fix, Witness

__all__ = ["spec", "lint_source", "lint_paths", "check_spec", "spec_functions", "Finding", "LintResult",
           "Witness", "Fix", "RULES", "Rule", "Config", "ConfigError", "load_config"]

F = TypeVar("F", bound=Callable[..., Any])

_LAZY = {
    "lint_source": "engine", "lint_paths": "engine", "check_spec": "engine", "spec_functions": "engine",
    "Finding": "engine", "LintResult": "engine", "Witness": "witness", "Fix": "witness",
    "RULES": "rules", "Rule": "rules", "Config": "config", "ConfigError": "config", "load_config": "config",
}


def spec(formula: str, /, *, domain: "Mapping[str, str] | None" = None,
         ranges: "Mapping[str, tuple] | None" = None) -> Callable[[F], F]:
    """Declare what a function computes. No effect at run time.

        >>> @spec("n*(n+1)/2", domain={"n": "integer, n>=0"})
        ... def tri(n: int) -> int:
        ...     return n * (n + 1) // 2
        >>> tri(4), tri.__pcm_spec__["formula"]
        (10, 'n*(n+1)/2')

    The lint reads the decorator from the SOURCE (``ast``), so the arguments must
    be literals: a string, and dicts of strings / ``(low, high)`` tuples.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise TypeError("spec(formula): the formula must be a non-empty string")
    record = {"formula": formula, "domain": dict(domain or {}), "ranges": dict(ranges or {})}

    def mark(fn: F) -> F:
        try:
            fn.__pcm_spec__ = record  # type: ignore[attr-defined]
        except (AttributeError, TypeError):  # builtins, slotted callables: nothing to record on
            pass
        return fn

    return mark


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'pycodemath.lint' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
