"""One declaration of whether the W/D/L head is trusted, and two consumers of it.

Noise-as-signal
---------------
This module exists because of a defect class that is the **evil twin of
absence-becomes-zero**. The system *declared* the value head untrained — loss
weight 0, by design, masked with a reason — and the search read that
declared-absent opinion as evidence anyway. Under the old solved-flat terminal
scale a solve backed up ``+1.0`` and drowned the noise; under the z currency an
at-par solve is ``0.0``, and noise around zero routinely beats it, so the search
preferred a noisily-optimistic unexplored line over a **proven** draw
(`FINDINGS.md` F-14). ``Absent.__bool__`` raises for exactly this reason one layer
down; the same principle applies to an opinion nobody has earned the right to
hold.

The structural fix is **one declaration, two consumers**: while the head is
untrained-on-z, the loss masks it *and* the search contributes zero value — both
read :func:`value_contribution` of the same state, so they cannot disagree. A
config flag consulted by one and forgotten by the other is the two-gates-one-name
hazard again.

The switch-over
---------------
Neither a guessed iteration nor a mid-run human call: a **pre-registered
criterion**, four-tupled per rider (c), on held-out z accuracy. It fires **once**
and **ratchets** — the head does not flicker between trusted and not — and it is
logged as an event row rather than inferred from a behaviour change.

A note on the four-tuple that is a property of this metric rather than a mistake:
the **floor** (what a constant predictor achieves) and the **null** (the
majority-class model, run) *coincide* here, because the constant predictor **is**
the null. They are reported as one number with both names, not padded apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Absolute floor for the switch bar. Without it a degenerate label split — an
#: iteration where nearly every episode draws — would make "beat the majority
#: class by a margin" satisfiable by a near-constant predictor.
MIN_ACCURACY = 0.60

#: Margin the head must clear ABOVE the majority class. A head that only
#: reproduces the base rate has learned the base rate, not z.
MARGIN = 0.10


class ValueGateError(ValueError):
    """A misuse of the value-head declaration."""


@dataclass
class ValueHeadState:
    """Whether the W/D/L head has earned the right to be listened to.

    ``live`` is the single declaration. It starts False — spec §5 trains Phase-1
    value on steps-to-solve and leaves W/D/L at loss weight 0, so at iteration 0
    the head is noise — and it ratchets to True exactly once, when the criterion
    fires.
    """

    live: bool = False
    switched_at_iteration: int | None = None
    switched_accuracy: float | None = None
    switched_threshold: float | None = None

    def as_dict(self) -> dict:
        return {
            "live": self.live,
            "switched_at_iteration": self.switched_at_iteration,
            "switched_accuracy": self.switched_accuracy,
            "switched_threshold": self.switched_threshold,
        }


def value_contribution(state: ValueHeadState) -> float:
    """The multiplier BOTH consumers apply. 1.0 when trusted, 0.0 while not.

    * **search** scales the evaluator's value by this, so an untrained head
      contributes nothing and every unexplored child ties at 0.0 — selection then
      falls back to priors and the Gumbel draw, which is what found the solve in
      F-14's ablation arm.
    * **loss** scales ``train.value_loss_weight`` by this, so the head is not
      trained toward a target it is meanwhile being trusted for.

    One function, two call sites, one state. They cannot disagree.
    """
    return 1.0 if state.live else 0.0


def majority_class_accuracy(labels: Sequence[int]) -> float:
    """What a constant predictor achieves. **Both the floor and the null.**"""
    if not labels:
        raise ValueGateError("no held-out labels — the criterion cannot be evaluated on nothing")
    counts: dict[int, int] = {}
    for z in labels:
        counts[z] = counts.get(z, 0) + 1
    return max(counts.values()) / len(labels)


def switch_criterion(labels: Sequence[int], predictions: Sequence[int]) -> dict:
    """The pre-registered switch bar, as a four-tuple.

    ``floor`` / ``null``   majority-class accuracy on this held-out set. They are
                           the same number because the constant predictor IS the
                           null model — reported once, named twice.
    ``threshold``          ``max(MIN_ACCURACY, floor + MARGIN)``.
    ``measured``           the head's accuracy.

    Returns the whole tuple, always, so a row records what the bar *was* on the
    set it was evaluated against — the bar moves with the label distribution, so
    a bar quoted without its floor is not interpretable later.
    """
    if len(labels) != len(predictions):
        raise ValueGateError(
            f"{len(labels)} labels against {len(predictions)} predictions — the "
            "criterion cannot be evaluated on a misaligned pair"
        )
    floor = majority_class_accuracy(labels)
    threshold = max(MIN_ACCURACY, floor + MARGIN)
    correct = sum(1 for a, b in zip(labels, predictions, strict=True) if a == b)
    measured = correct / len(labels)
    return {
        "metric": "held-out z accuracy",
        "n": len(labels),
        "floor": round(floor, 6),
        "null": round(floor, 6),
        "null_is_the_floor": True,
        "threshold": round(threshold, 6),
        "measured": round(measured, 6),
        "clears": measured >= threshold,
    }


def consider_switch(
    state: ValueHeadState,
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    iteration: int,
) -> tuple[ValueHeadState, dict]:
    """Evaluate the criterion and ratchet if it clears. Returns (state, event).

    **Ratcheting is the point.** A head that flickers between trusted and not
    would make two consecutive iterations incomparable, and the corpus each
    produces would be denominated in different searches.
    """
    result = switch_criterion(labels, predictions)
    event = {"iteration": iteration, "already_live": state.live, **result}
    if state.live or not result["clears"]:
        event["fired"] = False
        return state, event
    event["fired"] = True
    return (
        ValueHeadState(
            live=True,
            switched_at_iteration=iteration,
            switched_accuracy=result["measured"],
            switched_threshold=result["threshold"],
        ),
        event,
    )
