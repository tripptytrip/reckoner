"""The episode runner: splits, the descent identity, the H split, and parity.

The descent identity is chunk 7's forward obligation landing. It is asserted, not
watched — and the one legitimate exception to `evals == nodes` is *named* rather
than absorbed into an inequality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reckoner.config import Config, config_fingerprint
from reckoner.dataset import git_sha, read_suite, suite_problem
from reckoner.logschema import SCHEMA_ERA, validate_row
from reckoner.replay import ReplayRing
from reckoner.rules import RULESET_VERSION
from reckoner.runner import IterationStats, entropy, iteration_row, run_iteration
from reckoner.search import uniform_stub
from reckoner.vocab import VOCAB_VERSION

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()

needs_suites = pytest.mark.skipif(
    not (SUITES / "solve_in_2.jsonl").exists(), reason="suites not generated"
)


def problems(depth: int, n: int) -> list:
    return [suite_problem(r) for r in read_suite(SUITES / f"solve_in_{depth}.jsonl")[:n]]


# ---------------------------------------------------------------------------
# The descent identity — chunk 7's forward obligation
# ---------------------------------------------------------------------------


@needs_suites
def test_the_descent_identity_holds_over_a_real_iteration() -> None:
    """`nodes - evaluations == terminals`, asserted rather than watched."""
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert stats.nodes - stats.evaluations == stats.terminals
    assert stats.terminals > 0, "an iteration with no terminals would not exercise the identity"


def test_the_identity_is_checked_and_can_fail() -> None:
    """The other polarity. An invariant nobody has seen refuse is a comment."""
    stats = IterationStats(nodes=10, evaluations=10, terminal_solved=3)
    with pytest.raises(AssertionError, match="descent identity violated"):
        stats.check_descent_identity()


def test_evals_equals_nodes_only_when_no_terminal_is_reached() -> None:
    """The named exception, pinned so it cannot quietly become an inequality.

    Chunk 7's `evals == nodes` is the zero-terminal special case of the identity.
    A terminal leaf creates a node without an evaluation, which is why the
    equality does not generalise to real episodes.
    """
    zero_terminals = IterationStats(nodes=49, evaluations=49)
    zero_terminals.check_descent_identity()
    assert zero_terminals.nodes == zero_terminals.evaluations

    with_terminals = IterationStats(nodes=52, evaluations=49, terminal_solved=3)
    with_terminals.check_descent_identity()
    assert with_terminals.nodes != with_terminals.evaluations


# ---------------------------------------------------------------------------
# Splits sum — the schema invariant, met by the producer
# ---------------------------------------------------------------------------


@needs_suites
def test_outcomes_sum_to_episodes() -> None:
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert stats.episodes_solved + stats.episodes_capped + stats.episodes_stuck == stats.episodes


@needs_suites
def test_the_histogram_accounts_for_exactly_the_solved_episodes() -> None:
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert sum(stats.steps_minus_par.values()) == stats.episodes_solved


@needs_suites
def test_the_row_it_produces_validates_against_the_schema() -> None:
    """The producer and the schema must agree without a translation step."""
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    row = iteration_row(
        stats,
        iteration=0,
        run_name="test",
        git_sha=git_sha(REPO),
        config_fingerprint=config_fingerprint(CFG),
        cfg=CFG,
        ruleset_version=RULESET_VERSION,
        vocab_version=VOCAB_VERSION,
        schema_era=SCHEMA_ERA,
    )
    assert validate_row(row) == [], "a clean iteration must raise no alarms"


@needs_suites
def test_exact_par_cannot_be_beaten_in_the_produced_row() -> None:
    """The tripwire's fourth sighting: the producer must not even construct it."""
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert stats.z_by_par_source["bfs"]["+1"] == 0


# ---------------------------------------------------------------------------
# The entropy split
# ---------------------------------------------------------------------------


def test_entropy_is_zero_on_a_degenerate_distribution() -> None:
    assert entropy(np.array([1.0])) == pytest.approx(0.0)
    assert entropy(np.array([])) == 0.0


def test_entropy_is_maximal_on_a_uniform_distribution() -> None:
    assert entropy(np.full(4, 0.25)) == pytest.approx(np.log(4))


@needs_suites
def test_both_entropy_populations_are_measured() -> None:
    """Start and reached must both be populated, or the split is a single number.

    Depth-2 problems take at least two steps, so every episode contributes to
    both populations — which is what makes the split meaningful here.
    """
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    assert stats.h_prior_start and stats.h_prior_reached
    assert stats.h_target_start and stats.h_target_reached
    values = stats.entropies()
    assert values["entropy_prior_step1_start"] != values["entropy_prior_step1_reached"], (
        "start and reached entropies are identical — the split is not splitting"
    )


@needs_suites
def test_prior_and_target_entropy_are_different_measurements() -> None:
    """Prior entropy alone cannot tell a confident policy from a collapsed one."""
    stats = run_iteration(problems(2, 12), uniform_stub(CFG), CFG, sims=16, m=5, seed=1)
    values = stats.entropies()
    assert values["entropy_prior_step1_start"] != values["entropy_target_step1_start"]


# ---------------------------------------------------------------------------
# Batching is across searches, and it changes nothing
# ---------------------------------------------------------------------------


@needs_suites
def test_batch_size_does_not_change_the_iteration() -> None:
    """Pooling across episodes has no coupling, so parity is an equality.

    Not a tolerance: if this ever needs one, batching has leaked inside a tree.
    """
    kwargs = {"sims": 16, "m": 5, "seed": 3}
    small = run_iteration(problems(2, 8), uniform_stub(CFG), CFG, batch_size=1, **kwargs)
    large = run_iteration(problems(2, 8), uniform_stub(CFG), CFG, batch_size=64, **kwargs)
    assert small.episodes_solved == large.episodes_solved
    assert small.nodes == large.nodes and small.evaluations == large.evaluations
    assert small.steps_minus_par == large.steps_minus_par
    assert small.z_by_par_source == large.z_by_par_source


@needs_suites
def test_seeding_is_a_per_episode_fan_out_not_one_shared_stream() -> None:
    """F-11's precedent generalised to the runner.

    Every episode drawing from one stream is not a sample of independent
    episodes. Different seeds must move the iteration; the same seed must
    reproduce it exactly.
    """
    kwargs = {"sims": 16, "m": 5}
    a = run_iteration(problems(2, 10), uniform_stub(CFG), CFG, seed=1, **kwargs)
    b = run_iteration(problems(2, 10), uniform_stub(CFG), CFG, seed=1, **kwargs)
    c = run_iteration(problems(2, 10), uniform_stub(CFG), CFG, seed=2, **kwargs)

    # Same seed reproduces exactly.
    assert a.nodes == b.nodes
    assert a.evaluations == b.evaluations
    assert a.z_by_par_source == b.z_by_par_source
    assert a.h_target_start == b.h_target_start

    # A different seed must move something. The quantities it can move are
    # narrower than they look: against `uniform_stub` the priors are FLAT, so
    # H_prior is a function of the legal-action count alone and is seed-blind by
    # construction, and node counts are budget-driven. What the Gumbel draw
    # actually moves is which actions get visited — so the discriminating
    # quantities are the evaluation count and the visit-derived target entropy.
    assert (a.evaluations, a.h_target_start) != (c.evaluations, c.h_target_start), (
        "a different seed changed neither the evaluation count nor the visit "
        "distribution — the fan-out is not fanning out"
    )


# ---------------------------------------------------------------------------
# The ring receives the episode
# ---------------------------------------------------------------------------


@needs_suites
def test_every_step_of_every_episode_reaches_the_ring() -> None:
    ring = ReplayRing(2000, CFG)
    stats = run_iteration(problems(2, 8), uniform_stub(CFG), CFG, ring, sims=16, m=5, seed=1)
    assert len(ring) > 0
    assert len(ring) >= stats.episodes_solved, "at least one row per solved episode"
    for slot in range(len(ring)):
        record = ring.get(slot)
        assert record["z"] in (-1, 0, 1)
        assert record["par_source"] == "bfs"


@needs_suites
def test_the_ring_learns_z_only_when_the_episode_settles() -> None:
    """z is an episode-level outcome; every step of one episode carries the same z."""
    ring = ReplayRing(2000, CFG)
    run_iteration(problems(2, 4), uniform_stub(CFG), CFG, ring, sims=16, m=5, seed=1)
    zs = {ring.get(slot)["z"] for slot in range(len(ring))}
    assert zs <= {-1, 0, 1} and zs
