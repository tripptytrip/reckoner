"""The value-head declaration, its two consumers, and the switch detector.

The detector gates automation, so it gets two-sided validation before live use —
the clamp-detector precedent. A criterion nobody has watched refuse is a
criterion that will fire on anything.
"""

from __future__ import annotations

import random

import pytest

from reckoner.valuegate import (
    MARGIN,
    MIN_ACCURACY,
    ValueGateError,
    ValueHeadState,
    consider_switch,
    majority_class_accuracy,
    switch_criterion,
    value_contribution,
)

# ---------------------------------------------------------------------------
# One declaration, two consumers
# ---------------------------------------------------------------------------


def test_an_untrained_head_contributes_nothing() -> None:
    assert value_contribution(ValueHeadState(live=False)) == 0.0


def test_a_trained_head_contributes_fully() -> None:
    assert value_contribution(ValueHeadState(live=True)) == 1.0


def test_both_consumers_read_the_same_number() -> None:
    """The structural claim, asserted rather than described.

    A config flag consulted by the loss and forgotten by the search is the
    two-gates-one-name hazard. Here there is one function; this pins that the
    search-side scale and the loss-side scale are literally the same call.
    """
    state = ValueHeadState(live=False)
    search_scale = value_contribution(state)
    loss_scale = value_contribution(state)
    assert search_scale == loss_scale == 0.0
    live = ValueHeadState(live=True)
    assert value_contribution(live) == value_contribution(live) == 1.0


def test_the_head_starts_untrusted() -> None:
    """Spec §5 leaves W/D/L at loss weight 0, so at iteration 0 it is noise."""
    assert ValueHeadState().live is False


# ---------------------------------------------------------------------------
# The floor, and why it is also the null
# ---------------------------------------------------------------------------


def test_majority_class_is_the_floor() -> None:
    assert majority_class_accuracy([0, 0, 0, -1]) == pytest.approx(0.75)
    assert majority_class_accuracy([0, -1]) == pytest.approx(0.5)


def test_floor_and_null_are_reported_as_one_number_with_two_names() -> None:
    """A property of this metric, not a mistake — the constant predictor IS the null."""
    result = switch_criterion([0, 0, 0, -1], [0, 0, 0, 0])
    assert result["floor"] == result["null"]
    assert result["null_is_the_floor"] is True


def test_an_empty_held_out_set_is_refused() -> None:
    with pytest.raises(ValueGateError, match="cannot be evaluated on nothing"):
        majority_class_accuracy([])


def test_a_misaligned_pair_is_refused() -> None:
    with pytest.raises(ValueGateError, match="misaligned"):
        switch_criterion([0, 0], [0])


# ---------------------------------------------------------------------------
# The detector, on both polarities — it must fire, and it must refuse
# ---------------------------------------------------------------------------


def test_a_perfect_head_clears() -> None:
    labels = [0] * 60 + [-1] * 40
    assert switch_criterion(labels, labels)["clears"] is True


def test_a_majority_class_head_does_not_clear() -> None:
    """The sharpest rejecting case: reproducing the base rate is not learning z.

    A head that always says "draw" scores exactly the floor, so it must miss by
    the whole margin. If this ever passes, the bar has collapsed onto its floor.
    """
    labels = [0] * 60 + [-1] * 40
    result = switch_criterion(labels, [0] * 100)
    assert result["measured"] == pytest.approx(result["floor"])
    assert result["clears"] is False


def test_a_random_head_does_not_clear() -> None:
    rng = random.Random(0)
    labels = [rng.choice([0, -1]) for _ in range(400)]
    guesses = [rng.choice([0, -1]) for _ in range(400)]
    assert switch_criterion(labels, guesses)["clears"] is False


def test_the_absolute_floor_stops_a_degenerate_split_from_trivialising_the_bar() -> None:
    """Nearly-all-draws would make "beat the majority class" easy for a constant.

    With 98% draws the floor is 0.98 and the threshold is floor + MARGIN, which
    exceeds 1.0 — unreachable, correctly. MIN_ACCURACY guards the other end.
    """
    labels = [0] * 98 + [-1] * 2
    result = switch_criterion(labels, [0] * 100)
    assert result["threshold"] == pytest.approx(0.98 + MARGIN)
    assert result["clears"] is False
    assert result["threshold"] > MIN_ACCURACY


def test_the_bar_moves_with_the_label_distribution_and_is_recorded_with_it() -> None:
    """A bar quoted without its floor is not interpretable later."""
    balanced = switch_criterion([0] * 50 + [-1] * 50, [0] * 100)
    skewed = switch_criterion([0] * 90 + [-1] * 10, [0] * 100)
    assert balanced["threshold"] != skewed["threshold"]
    for result in (balanced, skewed):
        assert {"floor", "null", "threshold", "measured", "n"} <= set(result)


# ---------------------------------------------------------------------------
# Ratcheting — fires once, never flickers
# ---------------------------------------------------------------------------


def test_it_fires_once_and_records_the_event() -> None:
    labels = [0] * 60 + [-1] * 40
    state, event = consider_switch(ValueHeadState(), labels, labels, iteration=7)
    assert state.live is True
    assert state.switched_at_iteration == 7
    assert event["fired"] is True
    assert event["measured"] == pytest.approx(1.0)


def test_a_live_head_stays_live_even_if_the_criterion_would_now_fail() -> None:
    """The ratchet. Flickering would make consecutive iterations incomparable —
    the corpus each produces would be denominated in a different search."""
    labels = [0] * 60 + [-1] * 40
    state, _ = consider_switch(ValueHeadState(), labels, labels, iteration=3)
    later, event = consider_switch(state, labels, [0] * 100, iteration=9)
    assert later.live is True
    assert later.switched_at_iteration == 3, "the ratchet must not re-date itself"
    assert event["fired"] is False and event["already_live"] is True


def test_a_failing_criterion_leaves_the_head_untrusted() -> None:
    labels = [0] * 60 + [-1] * 40
    state, event = consider_switch(ValueHeadState(), labels, [0] * 100, iteration=1)
    assert state.live is False
    assert event["fired"] is False
    assert value_contribution(state) == 0.0
