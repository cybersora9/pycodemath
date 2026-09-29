"""VERIFY — check mathematics instead of computing it.

Three answers and never a fourth: ``VERIFIED`` (proved), ``REFUTED`` (a concrete
counterexample, confirmed at two precisions) and ``UNDECIDED`` (neither — a
first-class result, not a failure). See ``verify.verdict`` for why agreement at
every sample point is still UNDECIDED, and ``verify.equal`` for what equality
means (principal branch, where both sides are defined, decimals as written), and
``verify.certify`` for checking an engine RESULT by a route that did not produce it.

    >>> from pycodemath.verify import check_equal
    >>> check_equal("sqrt(x^2)", "x").counterexample
    {'x': '-1'}

``verify.steps`` checks a whole derivation, pairwise, and names the first wrong
step: ``check_steps("(x+1)^2 - (x-1)^2 = x^2+2x+1 - x^2+2x+1 = 4x+2")`` refutes
step 2 at ``x = 0``.
"""

from __future__ import annotations

from .certify import (
    CERTIFIERS,
    certify,
    certify_diff,
    certify_dsolve,
    certify_integrate,
    certify_limit,
    certify_nintegrate,
    certify_solve,
)
from .equal import check_equal
from .steps import StepCheck, StepsResult, check_steps
from .verdict import Verdict, VerdictStatus

__all__ = [
    "Verdict",
    "VerdictStatus",
    "check_equal",
    "certify",
    "certify_integrate",
    "certify_diff",
    "certify_solve",
    "certify_limit",
    "certify_dsolve",
    "certify_nintegrate",
    "CERTIFIERS",
    "check_steps",
    "StepCheck",
    "StepsResult",
]
