"""The verdict type — the only three answers a verifier is allowed to give.

``VERIFIED`` means PROVED (a symbolic argument closed), ``REFUTED`` means a
concrete point was found where the claim is false, and ``UNDECIDED`` means
neither happened. There is deliberately no fourth word such as "probably" or
"likely": a sampling run that agrees everywhere it looked is ``UNDECIDED`` with
the evidence in ``detail``, never ``VERIFIED``. A verifier that rounds agreement
up to proof is worse than no verifier, because it is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..core.errors import DomainError

__all__ = ["VerdictStatus", "Verdict"]


class VerdictStatus(Enum):
    """What was established about the claim — proof, disproof, or neither."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class Verdict:
    """The outcome of one check, with the evidence that decided it.

    ``method`` names HOW the status was reached (``"symbolic"``,
    ``"numeric-sampling"``), or what stopped the run (``"time-budget"``), so a
    caller can weigh the verdict without parsing ``detail``.

    ``counterexample`` is set exactly when ``status`` is ``REFUTED``: variable
    name -> exact value, as text the parser reads back (``{"x": "-2"}``,
    ``{"x": "7/3"}``). It is an empty dict when the claim has no variables at
    all (``pi`` vs ``3.14159`` is refuted at no point in particular). Every
    counterexample was confirmed at a second, higher precision before it was
    reported, so it is not rounding noise.

    ``detail`` is one sentence for a human; nothing should branch on it.
    """

    status: VerdictStatus
    method: str
    counterexample: "dict[str, str] | None"
    detail: str

    def __post_init__(self) -> None:
        # The invariant is structural, so it is enforced where the object is
        # built rather than trusted to every producer: a REFUTED without a point
        # is an accusation without evidence, and a point on anything else is a
        # contradiction a caller could act on.
        refuted = self.status is VerdictStatus.REFUTED
        if refuted != (self.counterexample is not None):
            raise DomainError(
                "Verdict: counterexample must be set exactly when status is REFUTED"
            )

    @property
    def verified(self) -> bool:
        return self.status is VerdictStatus.VERIFIED

    @property
    def refuted(self) -> bool:
        return self.status is VerdictStatus.REFUTED

    @property
    def undecided(self) -> bool:
        return self.status is VerdictStatus.UNDECIDED
