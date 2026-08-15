"""One absence type, for every layer that has to say "not there, and here is why".

The no-null law holds at the schema layer (`logschema` refuses `None` outright),
at the ring layer (`ReplayRing.get` returns :class:`Absent` rather than a zero),
and it has to hold at the league layer too — a fallback par's ``par_asof`` is not
missing, it is *inapplicable*, and those are different facts.

This lives in its own module rather than in the first place that needed it,
because a second copy of an absence type is exactly the two-implementations
hazard the one-formatter and one-legality-oracle laws exist to prevent. Whoever
adds the fourth layer imports this.
"""

from __future__ import annotations

from dataclasses import dataclass


class AbsenceError(RuntimeError):
    """An absent value used as though it were present."""


@dataclass(frozen=True, slots=True)
class Absent:
    """A value that is not there, with the reason it is not there.

    Returned instead of a zero or a ``None``, always. ``__bool__`` raises because
    **truthiness is precisely how an absence becomes a zero**: a ring handing back
    ``0.0`` for a missing ``root_q`` teaches the blend the position was a draw,
    and a ``par_asof`` of ``None`` read as ``0`` dates a snapshot to the
    beginning of time.
    """

    field: str
    reason: str
    kind: str  # "era" | "runtime" | "inapplicable"

    def __bool__(self) -> bool:  # pragma: no cover - guarding a misuse
        raise AbsenceError(
            f"{self.field} is absent ({self.kind}: {self.reason}) — test for Absent "
            "explicitly rather than relying on truthiness, which is how an absence "
            "becomes a zero."
        )

    def as_dict(self) -> dict:
        return {"absent": True, "field": self.field, "reason": self.reason, "kind": self.kind}
