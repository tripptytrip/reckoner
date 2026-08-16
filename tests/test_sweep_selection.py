"""The sweep's selection rule and the union merge, after P11B-A3's reconciliation.

Two things are asserted here that a comment could not hold.

The all-above node has **two polarities** and only one of them was ever
implemented. Above the domain floor it asks for a downward extension; *at* the
floor that request is for points below ``sims = 1``, which do not exist, so the
node is terminal and names the successor instead. "Succession fires here" is only
meaningful beside "and the extension still fires there", so both are tested.

The union merge refuses disagreeing collisions. A rung measured twice that
returns two different counts is a finding; the merge must not be the thing that
decides which of the two history keeps.
"""

from __future__ import annotations

import importlib

import pytest

sweep = importlib.import_module("chunk11_part0b")


def point(sims: int, at_par_rate: float, **over: object) -> dict:
    """A sweep point with the fields the rule and the merge actually read."""
    at_par = round(at_par_rate * 1200)
    base = {
        "sims": sims,
        "gumbel_m": min(16, sims),
        "at_par": at_par,
        "at_par_rate": at_par_rate,
        "beat_par": 0,
        "over_par": 1200 - at_par,
        "episodes": 1200,
        "solved": 1200,
        "capped": 0,
        "stuck": 0,
        "steps_minus_par": {"<0": 0, "0": at_par, "1": 1200 - at_par},
        "seconds": 100.0 + sims,
    }
    base.update(over)
    return base


# --------------------------------------------------------------- branch (c)

def test_all_above_at_the_domain_floor_fires_succession() -> None:
    """P11B-A3 §2(c). At sims=1 there is nothing below to extend into.

    This is the branch Part-0b's own extension walked into: every rung from 1 to
    48 pars above the window, so the suite-economy primary is saturated at the
    floor of its own domain and succession fires at iteration 0.
    """
    verdict = sweep.select([point(s, 0.98) for s in (1, 2, 3, 4, 6, 48)])

    assert verdict["s_star"] is None
    assert verdict["needs"] == "succession"
    fired = [b for b in verdict["branches"] if b.get("fired")]
    assert [b["branch"] for b in fired] == ["all_above_window"]
    assert fired[0]["at_domain_floor"] is True
    assert fired[0]["successor_strata"] == [7, 8, 10]
    # 9 is the saturated stratum and must not be carried into the successor set.
    assert 9 not in fired[0]["successor_strata"]


def test_all_above_off_the_floor_still_asks_for_the_extension() -> None:
    """The other polarity, unchanged from the node as first written.

    Without this the succession clause could be a rule that always fires, which
    would make the extension unreachable and P11B-A3's own domain move a fiction.
    """
    verdict = sweep.select([point(s, 0.99) for s in (6, 8, 12, 16, 48)])

    assert verdict["needs"] == "downward_extension"
    fired = [b for b in verdict["branches"] if b.get("fired")][0]
    assert fired["at_domain_floor"] is False
    assert fired["extension"] == [1, 2, 3, 4]


# --------------------------------------------------------------- branch (a)

def test_in_window_selects_nearest_target() -> None:
    verdict = sweep.select([point(4, 0.45), point(8, 0.60), point(16, 0.99)])

    assert verdict["s_star"] == 8
    assert verdict["s_star_rate"] == 0.60


def test_equal_rates_tie_and_the_smaller_sims_wins() -> None:
    """The realistic tie on this instrument, and the one the rule declares.

    Four rungs returned a byte-identical 1189/1200 in Part-0b, so two in-window
    rungs carrying the *same* rate is the tie that actually happens here. The
    declared break is toward smaller sims, and the branch says so on the row.
    """
    verdict = sweep.select([point(4, 0.60), point(8, 0.60), point(16, 0.99)])

    assert verdict["s_star"] == 4
    fired = [b for b in verdict["branches"] if b.get("fired")][0]
    assert fired["tie_broken_toward_smaller_sims"] is True


def test_decimal_symmetric_rates_are_not_ties_under_float_comparison() -> None:
    """Recorded so the tie-break's reach is not overstated.

    0.50 and 0.60 look equidistant from 0.55 on paper, but the criterion compares
    ``abs(rate - TARGET)`` as floats: 0.05000000000000004 against
    0.049999999999999996. The nearer-in-float point wins outright and no tie is
    declared. The break is live for equal rates and silent for symmetric ones —
    which is the honest description of the frozen rule, not a defect in it.
    """
    verdict = sweep.select([point(4, 0.50), point(8, 0.60), point(16, 0.99)])

    assert verdict["s_star"] == 8
    fired = [b for b in verdict["branches"] if b.get("fired")][0]
    assert fired["tie_broken_toward_smaller_sims"] is False


# ------------------------------------------------- branch (b), as withdrawn

def test_a_straddle_bisects_rather_than_selecting_out_of_window() -> None:
    """P11B-A3's (b) was withdrawn: the committed criterion governs.

    Selecting the nearest-0.55 point while unmeasured in-window points still
    exist is the fishing room the window was frozen to remove. Bisection is
    domain completion pointed inward — the same move as the downward extension.
    """
    verdict = sweep.select([point(4, 0.20), point(16, 0.99)])

    assert verdict["s_star"] is None
    assert verdict["needs"] == "bisection"
    assert verdict["bisect_at"] == 10


def test_an_adjacent_straddle_is_terminal_and_does_not_widen_the_window() -> None:
    """The honest terminal: a ruling on an exhausted domain carries no freedom."""
    verdict = sweep.select([point(4, 0.20), point(5, 0.99)])

    assert verdict["needs"] == "ruling"
    assert verdict["bisect_at"] is None
    fired = [b for b in verdict["branches"] if b.get("fired")][0]
    assert fired["adjacent_integers"] is True
    assert "NOT widened" in fired["action"]


# --------------------------------------------------------- the union merge

def test_the_union_is_by_rung_key_and_sorted() -> None:
    merged = sweep.merge_points(
        [point(6, 0.99), point(48, 0.99)], [point(1, 0.98), point(4, 0.98)]
    )
    assert [p["sims"] for p in merged] == [1, 4, 6, 48]


def test_a_rung_measured_twice_in_agreement_merges() -> None:
    """Agreement is not a collision, and wall-clock is not a measurement."""
    prior = [point(6, 0.99)]
    again = [point(6, 0.99, seconds=999.9)]
    merged = sweep.merge_points(prior, again)

    assert [p["sims"] for p in merged] == [6]


def test_a_rung_measured_twice_in_disagreement_refuses() -> None:
    """The whole point of the key: never let the merge decide this quietly."""
    prior = [point(6, 0.99)]
    conflicting = [point(6, 0.55)]

    with pytest.raises(sweep.RungCollision) as excinfo:
        sweep.merge_points(prior, conflicting)

    message = str(excinfo.value)
    assert "sims=6" in message
    assert "at_par" in message
