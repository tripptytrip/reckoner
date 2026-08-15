"""The ladder's arithmetic: pairing, the bootstrap, the self-match null.

Every claim here is one a campaign will lean on, so each gets its rejecting case
beside its accepting one — including the null that must score exactly 0.5.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reckoner.arms import GreedyHeuristic, RandomRewriter
from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.ladder import (
    CURRENCY_BUDGET,
    CURRENCY_Z,
    LadderError,
    PairScore,
    pair,
    paired_bootstrap,
    problem_key_of,
    rigged_null,
    self_match,
    synthetic_elo,
)

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()

needs_suites = pytest.mark.skipif(
    not (SUITES / "solve_in_3.jsonl").exists(), reason="suites not generated"
)


def problems(n: int = 12):
    return [suite_problem(r) for r in read_suite(SUITES / "solve_in_3.jsonl")[:n]]


def scores(arm: str, values: list[float], currency: str = CURRENCY_Z) -> list[PairScore]:
    return [PairScore(f"p{i}", arm, currency, v, steps=1, seed=i) for i, v in enumerate(values)]


# ---------------------------------------------------------------------------
# Pairing — and the refusals that make it honest
# ---------------------------------------------------------------------------


def test_pairing_differences_per_problem() -> None:
    comparison = pair(scores("a", [1.0, 0.0, -1.0]), scores("b", [0.0, 0.0, 0.0]))
    assert comparison.differences == [1.0, 0.0, -1.0]
    assert comparison.mean() == pytest.approx(0.0)


def test_pairing_across_currencies_is_refused() -> None:
    """A z minus a solve-rate is a number with no units — the one the currency
    ruling exists to make unconstructable."""
    with pytest.raises(LadderError, match="cannot pair across currencies"):
        pair(scores("model", [1.0]), scores("sympy", [1.0], CURRENCY_BUDGET))


def test_a_partial_overlap_is_refused() -> None:
    """A 'paired' comparison over a partial overlap is an unpaired comparison
    with a paired name — the arms would be scored on different problem sets."""
    with pytest.raises(LadderError, match="partial overlap|not by"):
        pair(scores("a", [1.0, 1.0, 1.0]), scores("b", [1.0, 1.0]))


def test_pairing_an_empty_arm_is_refused() -> None:
    with pytest.raises(LadderError, match="empty arm"):
        pair([], scores("b", [1.0]))


# ---------------------------------------------------------------------------
# The bootstrap — the test of record
# ---------------------------------------------------------------------------


def test_a_real_difference_excludes_zero() -> None:
    result = paired_bootstrap([1.0] * 30 + [0.0] * 5, resamples=2000, seed=0)
    assert result["mean_difference"] > 0
    assert result["excludes_zero"] is True


def test_no_difference_does_not_exclude_zero() -> None:
    """The rejecting case. A test that always finds an effect finds nothing."""
    result = paired_bootstrap(rigged_null(200), resamples=2000, seed=0)
    assert result["excludes_zero"] is False


def test_a_saturated_interval_says_so_rather_than_reading_as_precision() -> None:
    """Every difference identical gives a zero-width interval BY CONSTRUCTION.

    Rendering that as a narrow CI would read as an extremely precise estimate
    when it is a statistic that has run out of range.
    """
    result = paired_bootstrap([0.0] * 50, resamples=500, seed=0)
    assert result["saturated"] is True
    assert result["ci_low"] == result["ci_high"] == 0.0
    assert "SATURATED" in result["rendering_note"]


def test_an_unsaturated_interval_is_not_flagged() -> None:
    result = paired_bootstrap([1.0, 0.0, -1.0] * 20, resamples=500, seed=0)
    assert result["saturated"] is False


def test_the_bootstrap_refuses_an_empty_comparison() -> None:
    with pytest.raises(LadderError, match="nothing to resample"):
        paired_bootstrap([], resamples=100)


# ---------------------------------------------------------------------------
# Synthetic Elo and its rigged-50% null
# ---------------------------------------------------------------------------


def test_the_rigged_null_scores_exactly_one_half() -> None:
    """If the scoring arithmetic has a bias, every later number inherits it."""
    assert synthetic_elo(rigged_null(200)) == pytest.approx(0.5)
    assert synthetic_elo(rigged_null(201)) == pytest.approx(0.5)


def test_a_dominant_arm_scores_one_and_a_dominated_arm_zero() -> None:
    assert synthetic_elo([1.0] * 40) == pytest.approx(1.0)
    assert synthetic_elo([-1.0] * 40) == pytest.approx(0.0)


def test_all_ties_score_one_half() -> None:
    assert synthetic_elo([0.0] * 40) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The self-match null, in race form — and its contrast case
# ---------------------------------------------------------------------------


@needs_suites
def test_self_match_under_the_eval_profile_is_exactly_zero() -> None:
    """Deterministic under eval ⇒ identical z vectors ⇒ every difference EXACTLY 0.

    Not "small". Any drift means state is leaking between episodes that should be
    independent.
    """
    arm = GreedyHeuristic()

    def play(problem, cfg, seed):
        result = arm.play(problem, cfg, seed)
        return (1.0 if result.solved else -1.0), result.steps

    comparison = self_match(play, problems(), CFG, profile="eval")
    assert comparison.differences == [0.0] * len(comparison.differences)
    assert all(d == 0.0 for d in comparison.differences)


@needs_suites
def test_the_self_play_profile_breaks_the_identity() -> None:
    """The contrast case. A null that reports zero for every configuration is
    measuring nothing."""
    arm = RandomRewriter()

    def play(problem, cfg, seed):
        result = arm.play(problem, cfg, seed)
        return (1.0 if result.solved else -1.0), result.steps

    comparison = self_match(play, problems(24), CFG, profile="self_play")
    assert any(d != 0.0 for d in comparison.differences), (
        "different seeds produced identical outcomes — the contrast case is vacuous "
        "and the eval-profile zero proves nothing"
    )


@needs_suites
def test_a_self_match_bootstrap_is_saturated_and_says_so() -> None:
    """The two facts belong together: exactly zero, and flagged as saturated."""
    arm = GreedyHeuristic()

    def play(problem, cfg, seed):
        result = arm.play(problem, cfg, seed)
        return (1.0 if result.solved else -1.0), result.steps

    comparison = self_match(play, problems(), CFG, profile="eval")
    result = paired_bootstrap(comparison.differences, resamples=500, seed=0)
    assert result["mean_difference"] == 0.0
    assert result["saturated"] is True


def test_an_unknown_profile_is_refused() -> None:
    with pytest.raises(LadderError, match="unknown profile"):
        self_match(lambda p, c, s: (0.0, 0), [], CFG, profile="vibes")


@needs_suites
def test_problem_keys_go_through_the_shared_normalizer() -> None:
    keys = {problem_key_of(p) for p in problems()}
    assert len(keys) == len(problems()), "distinct problems must get distinct keys"
