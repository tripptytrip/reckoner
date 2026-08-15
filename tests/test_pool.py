"""The snapshot league, on both polarities for each of its three load-bearing rules.

Each rule has a failure this project has already met one layer down, so each gets
its accepting case beside its rejecting one.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.model import Reckoner, save_checkpoint
from reckoner.pool import CheckpointPool, PoolError
from reckoner.valuegate import ValueHeadState

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()

needs_suites = pytest.mark.skipif(
    not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated"
)


def a_snapshot(tmp_path: Path, name: str, *, step: int, live: bool = False) -> Path:
    torch.manual_seed(step)
    path = tmp_path / f"{name}.pt"
    save_checkpoint(
        path,
        Reckoner(CFG),
        CFG,
        step,
        value_head=ValueHeadState(live=live, switched_at_iteration=0 if live else None).as_dict(),
    )
    return path


def recording_factory(seen: list[float]):
    """An evaluator factory that records the value scale it was handed."""

    def factory(model, value_scale: float):
        seen.append(value_scale)
        width = 7 * CFG.model.max_sites

        def evaluate(leaves):
            return [(np.zeros(width, dtype=np.float32), value_scale * 0.5) for _ in leaves]

        return evaluate

    return factory


# ---------------------------------------------------------------------------
# Rule 1 — version refusal, counted, both polarities
# ---------------------------------------------------------------------------


def test_a_matching_snapshot_loads(tmp_path: Path) -> None:
    pool = CheckpointPool(CFG)
    assert pool.add(a_snapshot(tmp_path, "ok", step=100)) is not None
    assert len(pool) == 1
    assert pool.stats.refusals == 0


def test_a_version_mismatch_is_refused_and_counted(tmp_path: Path) -> None:
    """Never a silently smaller pool — the chunk-6 registration, discharged."""
    path = a_snapshot(tmp_path, "stale", step=50)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["meta"]["ruleset_version"] = 999
    torch.save(blob, path)

    pool = CheckpointPool(CFG)
    with pytest.raises(PoolError, match="refusal is counted"):
        pool.add(path)
    assert pool.stats.refusals == 1
    assert pool.stats.refused_paths == [str(path)]
    assert len(pool) == 0


def test_a_vocab_mismatch_is_refused_too(tmp_path: Path) -> None:
    path = a_snapshot(tmp_path, "othervocab", step=50)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["meta"]["vocab_version"] = 999
    torch.save(blob, path)
    pool = CheckpointPool(CFG)
    assert pool.try_add(path) is False
    assert pool.stats.refusals == 1


# ---------------------------------------------------------------------------
# Rule 2 — a snapshot solves under ITS OWN declaration
# ---------------------------------------------------------------------------


@needs_suites
def test_a_pre_switch_snapshot_plays_value_silent_inside_a_post_switch_run(
    tmp_path: Path,
) -> None:
    """The running run has switched; this member has not. It plays silent.

    Re-solving a pre-switch snapshot with post-switch wiring produces a par that
    snapshot never achieved — a wrong number under a provenance tag true in its
    own terms, which is F-02 one layer up.
    """
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "old", step=10, live=False))
    assert member is not None
    assert member.value_head.live is False

    seen: list[float] = []
    problem = suite_problem(read_suite(SUITES / "solve_in_1.jsonl")[0])
    pool.par_for(problem, member, recording_factory(seen), random.Random(0), sims=8, m=5)
    assert seen and all(scale == 0.0 for scale in seen), (
        "a pre-switch member was handed a live value scale — it solved under the "
        "run's wiring, not its own"
    )


@needs_suites
def test_a_post_switch_snapshot_plays_value_live(tmp_path: Path) -> None:
    """The other polarity — otherwise "honours its own declaration" would just
    mean "always silent"."""
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "new", step=90, live=True))
    assert member is not None
    assert member.value_head.live is True

    seen: list[float] = []
    problem = suite_problem(read_suite(SUITES / "solve_in_1.jsonl")[0])
    pool.par_for(problem, member, recording_factory(seen), random.Random(0), sims=8, m=5)
    assert seen and all(scale == 1.0 for scale in seen)


def test_the_declaration_rides_in_the_checkpoint(tmp_path: Path) -> None:
    """It is part of the snapshot's identity, so it must be IN the snapshot."""
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "live", step=7, live=True))
    assert member is not None
    assert member.meta["value_head"]["live"] is True
    assert member.meta["value_head"]["switched_at_iteration"] == 0


# ---------------------------------------------------------------------------
# Rule 3 — unavailability is a COUNTED fallback, never a silent substitution
# ---------------------------------------------------------------------------


@needs_suites
def test_an_unsolved_problem_falls_back_with_the_provenance_flipped(
    tmp_path: Path,
) -> None:
    """The fallback is allowed. The silence is not.

    A par tagged `pool` that no pool member produced is the defect this module
    exists to avoid.
    """
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "weak", step=1))
    assert member is not None

    seen: list[float] = []
    problem = suite_problem(read_suite(SUITES / "solve_in_6.jsonl")[0])
    result = pool.par_for(
        problem, member, recording_factory(seen), random.Random(0), sims=2, m=1, budget=1
    )
    assert result.fell_back is True
    assert result.par_source == problem.par_source != "pool"
    assert result.par_asof is None, "a fallback par has no snapshot to be as-of"
    assert result.par == problem.par
    assert pool.stats.pool_par_unavailable == 1
    assert pool.stats.pool_par_solved == 0


@needs_suites
def test_a_solved_problem_is_tagged_pool_and_dated(tmp_path: Path) -> None:
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "able", step=42))
    assert member is not None
    seen: list[float] = []
    problem = suite_problem(read_suite(SUITES / "solve_in_1.jsonl")[0])
    result = pool.par_for(problem, member, recording_factory(seen), random.Random(0), sims=16, m=16)
    assert result.fell_back is False
    assert result.par_source == "pool"
    assert result.par_asof == 42, "pool par is as-of the snapshot that produced it"
    assert result.par >= 1
    assert pool.stats.pool_par_solved == 1


def test_a_problem_with_no_own_par_has_no_honest_fallback(tmp_path: Path) -> None:
    """Absence carries a reason: there is no label for this pair, so it raises."""
    from reckoner.episode import Problem
    from reckoner.expr import num
    from reckoner.vocab import GOAL_EVALUATE

    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "x", step=1))
    assert member is not None
    unlabelled = Problem(goal=GOAL_EVALUATE, expr=num(5), par=None, par_source="unverified")
    with pytest.raises(PoolError, match="no honest label"):
        pool.par_for(
            unlabelled, member, recording_factory([]), random.Random(0), sims=2, m=1, budget=1
        )


# ---------------------------------------------------------------------------
# Membership and sampling are DECLARED
# ---------------------------------------------------------------------------


def test_the_pool_is_bounded_and_keeps_the_most_recent(tmp_path: Path) -> None:
    cfg = Config()
    cfg.league.pool_size = 3
    pool = CheckpointPool(cfg)
    for step in (10, 20, 30, 40, 50):
        pool.add(a_snapshot(tmp_path, f"s{step}", step=step))
    assert pool.composition()["steps"] == [30, 40, 50]


def test_composition_is_loggable_and_names_the_live_members(tmp_path: Path) -> None:
    pool = CheckpointPool(CFG)
    pool.add(a_snapshot(tmp_path, "a", step=1, live=False))
    pool.add(a_snapshot(tmp_path, "b", step=2, live=True))
    assert pool.composition() == {"size": 2, "steps": [1, 2], "value_head_live": [2]}


def test_sampling_an_empty_pool_returns_nothing_rather_than_raising() -> None:
    """An empty pool is a state, not an error — par_from_pool_frac simply cannot
    be honoured yet, and the caller falls back with that recorded."""
    assert CheckpointPool(CFG).sample(random.Random(0)) is None


def test_an_unknown_sampling_policy_is_refused_by_validate() -> None:
    from reckoner.config import validate

    cfg = Config()
    cfg.league.pool_sample = "recency"
    with pytest.raises(ValueError, match="one-lever round"):
        validate(cfg)


def test_the_solve_budget_is_accounted(tmp_path: Path) -> None:
    """Pool par costs wall clock, and the cost shows up where costs show up."""
    pool = CheckpointPool(CFG)
    member = pool.add(a_snapshot(tmp_path, "t", step=1))
    assert member is not None
    from reckoner.episode import Problem
    from reckoner.expr import num
    from reckoner.vocab import GOAL_EVALUATE

    problem = Problem(goal=GOAL_EVALUATE, expr=num(5), par=1, par_source="bfs")
    pool.par_for(problem, member, recording_factory([]), random.Random(0), sims=2, m=1, budget=1)
    assert pool.stats.as_dict()["seconds_solving"] >= 0.0
    assert "pool_par_unavailable" in pool.stats.as_dict()
