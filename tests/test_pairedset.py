"""Paired sets: frozen at birth, verified at read, censused at both levels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reckoner.dataset import InstrumentAsTrainingSource, assert_training_source, read_suite
from reckoner.dataset import problem_key as strict_key
from reckoner.episode import Problem
from reckoner.expr import add, eq, mul, num, var
from reckoner.pairedset import (
    PairedSetError,
    census,
    census_key,
    freeze,
    load,
    read_anchors,
    source_census_keys,
)
from reckoner.vocab import GOAL_SIMPLIFY, GOAL_SOLVE, VAR_X

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"


def a_problem(rhs: int, par: int = 2) -> Problem:
    """``3x + rhs = 21``-shaped, labelled honestly enough to ship."""
    expr = eq(add(mul(num(3), var(VAR_X)), num(rhs)), num(21))
    return Problem(goal=GOAL_SOLVE, expr=expr, par=par, target=VAR_X, par_source="scripted")


def a_simplify(k: int) -> Problem:
    return Problem(
        goal=GOAL_SIMPLIFY, expr=add(mul(num(k), var(VAR_X)), num(1)), par=1, par_source="scripted"
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "runs").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Frozen at birth
# ---------------------------------------------------------------------------


def test_freezing_writes_the_rows_and_the_anchor(repo: Path) -> None:
    path = repo / "runs" / "paired" / "smoke.jsonl"
    digest = freeze(path, [a_problem(6), a_problem(9)], repo=repo)
    assert len(read_suite(path)) == 2
    assert read_anchors(repo)["runs/paired/smoke.jsonl"] == digest


def test_a_second_freeze_is_refused(repo: Path) -> None:
    """The refusal IS the freeze. Without it, "frozen at birth" is a description
    of intent that a re-run with a different seed quietly falsifies."""
    path = repo / "runs" / "paired" / "smoke.jsonl"
    freeze(path, [a_problem(6)], repo=repo)
    with pytest.raises(PairedSetError, match="already anchored"):
        freeze(path, [a_problem(9)], repo=repo)


def test_an_empty_paired_set_is_refused(repo: Path) -> None:
    with pytest.raises(PairedSetError, match="empty paired set"):
        freeze(repo / "runs" / "paired" / "empty.jsonl", [], repo=repo)


def test_a_duplicated_problem_is_refused_at_the_freeze(repo: Path) -> None:
    """Caught here rather than at pairing: a duplicate in the instrument makes
    every pass ever run on it mis-pair, not just the one that notices."""
    with pytest.raises(PairedSetError, match="duplicate problem"):
        freeze(repo / "runs" / "paired" / "dup.jsonl", [a_problem(6), a_problem(6)], repo=repo)


# ---------------------------------------------------------------------------
# Verified at read — an anchor nobody consults is a comment
# ---------------------------------------------------------------------------


def test_loading_returns_the_problems(repo: Path) -> None:
    path = repo / "runs" / "paired" / "smoke.jsonl"
    originals = [a_problem(6), a_problem(9)]
    freeze(path, originals, repo=repo)
    assert [strict_key(p) for p in load(path, repo=repo)] == [strict_key(p) for p in originals]


def test_a_drifted_file_raises(repo: Path) -> None:
    path = repo / "runs" / "paired" / "smoke.jsonl"
    freeze(path, [a_problem(6), a_problem(9)], repo=repo)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(json.dumps(rows[0], sort_keys=True) + "\n")
    with pytest.raises(PairedSetError, match="has drifted"):
        load(path, repo=repo)


def test_an_unanchored_file_raises(repo: Path) -> None:
    path = repo / "runs" / "paired" / "loose.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("")
    with pytest.raises(PairedSetError, match="not in runs/ANCHORS"):
        load(path, repo=repo)


def test_paired_sets_are_refused_as_a_training_source() -> None:
    """Both polarities of the runtime guard, at the new directory."""
    with pytest.raises(InstrumentAsTrainingSource, match="instrument"):
        assert_training_source(REPO / "runs" / "paired" / "anything.jsonl", REPO)
    assert_training_source(REPO / "runs" / "data" / "train_100k", REPO)


# ---------------------------------------------------------------------------
# Censused at BOTH levels
# ---------------------------------------------------------------------------


def test_the_census_reports_both_levels_separately() -> None:
    candidates = [a_problem(6), a_problem(9), a_simplify(4)]
    result = census(
        candidates,
        problem_sources={"train": {census_key(candidates[0])}},
        state_sources={"supervision": {census_key(candidates[1])}},
    )
    assert result.problem_level == {"train": 1}
    assert result.state_level == {"supervision": 1}
    assert result.clean_indices == [2]


def test_the_state_level_catches_what_the_problem_level_cannot() -> None:
    """F-08's mechanism, at the paired set's boundary.

    A candidate that is nobody's training *problem* can still be an intermediate
    *state* of a training derivation. If the census only asked the first question
    the instrument would come back clean and the model would have trained on it.
    """
    candidates = [a_problem(6)]
    result = census(
        candidates,
        problem_sources={"train": set()},
        state_sources={"supervision": {census_key(candidates[0])}},
    )
    assert result.problem_level == {"train": 0}
    assert result.as_dict()["state_level_beyond_problem_level"] == 1
    assert result.clean_indices == []


def test_a_clean_candidate_survives_both_levels() -> None:
    """The rejecting case: a census that condemns everything measures nothing."""
    result = census(
        [a_problem(6)], problem_sources={"train": set()}, state_sources={"supervision": set()}
    )
    assert result.clean_indices == [0]


def test_the_census_key_is_looser_than_the_pairing_key() -> None:
    """Deliberate, and the direction matters.

    Contamination is reported under the loose key — a model that saw one
    canonicalisation has effectively seen the other. Pairing uses the strict key,
    because merging two distinct rows of an instrument mis-scores an arm.
    """
    left = Problem(
        goal=GOAL_SOLVE,
        expr=eq(add(mul(num(3), var(VAR_X)), num(6)), num(21)),
        par=2,
        target=VAR_X,
        par_source="scripted",
    )
    right = Problem(
        goal=GOAL_SOLVE,
        expr=eq(add(num(6), mul(num(3), var(VAR_X))), num(21)),
        par=2,
        target=VAR_X,
        par_source="scripted",
    )
    assert census_key(left) == census_key(right)
    assert strict_key(left) == strict_key(right), (
        "canonicalisation already merges these; if it ever stops, the strict key "
        "must be the one that separates them"
    )


def test_source_census_keys_reads_a_real_set() -> None:
    data = REPO / "runs" / "data" / "eval_held_out"
    if not data.exists():
        pytest.skip("datasets not generated")
    keys = source_census_keys(data)
    assert len(keys) > 0
    assert all(isinstance(k[0], tuple) and isinstance(k[1], int) for k in keys)
