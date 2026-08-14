"""Datasets, frozen suites, contamination, and reproducibility from seed.

The suites are the measuring stick. Every check here exists because a suite that
is wrong, or that leaked into training, does not fail loudly — it produces
plausible numbers for a year.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from reckoner.config import Config
from reckoner.dataset import (
    FIELDS,
    dataset_keys,
    problem_key,
    read_dataset,
    read_suite,
    sha256_file,
    state_keys,
    suite_problem,
    write_dataset,
    write_suite,
)
from reckoner.episode import Problem, bfs_par, bfs_solution
from reckoner.expr import eq, mul, num, var
from reckoner.generator import check_emission
from reckoner.rules import RULESET_VERSION
from reckoner.vocab import GOAL_SOLVE, VAR_X, VOCAB_VERSION

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
DATA = REPO / "runs" / "data"
CFG = Config()
X = var(VAR_X)

DEPTHS = (1, 2, 3, 4, 5, 6)

pytestmark = pytest.mark.skipif(
    not (SUITES / "solve_in_1.jsonl").exists(),
    reason="suites not generated yet — run `python scripts/generate.py`",
)


def suite(depth: int) -> list[dict]:
    return read_suite(SUITES / f"solve_in_{depth}.jsonl")


# ---------------------------------------------------------------------------
# Suites: shape and provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", DEPTHS)
def test_suite_exists_and_is_the_declared_size(depth: int) -> None:
    rows = suite(depth)
    assert len(rows) == 200, f"solve_in_{depth} has {len(rows)} problems"


@pytest.mark.parametrize("depth", DEPTHS)
def test_every_suite_row_carries_its_versions(depth: int) -> None:
    for row in suite(depth):
        assert row["ruleset_version"] == RULESET_VERSION
        assert row["vocab_version"] == VOCAB_VERSION
        assert row["par_source"] == "bfs"
        assert row["par"] == row["depth"] == depth


@pytest.mark.parametrize("depth", DEPTHS)
def test_every_suite_row_obeys_the_emission_grammar(depth: int) -> None:
    """The grammar is re-checked from disk, not trusted from generation."""
    for row in suite(depth):
        check_emission(suite_problem(row))


def test_suites_are_goal_diverse_where_the_rule_set_allows() -> None:
    """A stratum whose goal mix is an accident of which templates exist is not
    an instrument.

    The first generation produced `solve_in_2` with no SOLVE at all and
    `solve_in_6` with 200/200 EVALUATE — the deepest suite, the one that matters
    most for measuring *solving* rather than *computing*, contained no equations.
    Two templates were added to close both ends.
    """
    for depth in DEPTHS:
        goals = {row["goal"] for row in suite(depth)}
        assert GOAL_SOLVE in goals, f"solve_in_{depth} contains no SOLVE problem"
        assert len(goals) >= 2, f"solve_in_{depth} is a single-goal suite: {goals}"


# ---------------------------------------------------------------------------
# The depth label, re-verified independently of generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", DEPTHS)
def test_suite_depth_labels_are_reverified_by_bfs(depth: int) -> None:
    """**Chunk 5 gate.** Every label re-derived from disk, not trusted.

    Two-sided, because "a solution of length N exists" and "no shorter one
    exists" are different claims and only the pair means *minimum*:

      * a derivation of exactly ``depth`` steps exists, and
      * BFS at cap ``depth - 1`` finds nothing.
    """
    rows = suite(depth)
    # A *sample* here; the exhaustive pass lives in scripts/verify_suites.py and
    # its artifact is checked below. Depth-6 SOLVE labelling is ~4.5 s per
    # problem times three BFS runs, so verifying all 1,200 inline would make
    # `make test` a ten-minute command — and a ten-minute `make test` is a
    # command people stop running, which costs more than it buys.
    sample = rows[:25] if depth <= 4 else rows[:6]
    for row in sample:
        problem = suite_problem(row)
        assert bfs_par(problem, CFG) == depth, f"label {depth} disagrees with BFS"
        path = bfs_solution(problem, CFG)
        assert path is not None and len(path) == depth
        assert bfs_solution(problem, CFG, cap=depth - 1) is None, (
            f"a shorter derivation exists — {depth} is not the minimum"
        )


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (DATA / "train_100k").exists(), reason="training set not generated")
def test_no_suite_contamination() -> None:
    """**Chunk 5 gate.** No suite problem may appear in any training set.

    Ported with its config-field guard: the check is run against *every* dataset
    directory that exists, so a training set added later cannot quietly escape
    the test by not being named in it.
    """
    checked = []
    suite_all: set[tuple[int, ...]] = set()
    for depth in DEPTHS:
        suite_all |= state_keys(suite(depth))

    for path in sorted(DATA.iterdir()):
        if not (path / "meta.json").exists():
            continue
        dataset = read_dataset(path)
        overlap = dataset_keys(dataset) & suite_all
        assert not overlap, f"{path.name} contains {len(overlap)} suite problems"
        checked.append(path.name)

    assert checked, "no dataset directories were checked — the guard is vacuous"
    assert "train_100k" in checked, f"the training set was not among {checked}"


@pytest.mark.skipif(not (DATA / "eval_held_out").exists(), reason="eval set not generated")
def test_eval_is_disjoint_from_train() -> None:
    train = dataset_keys(read_dataset(DATA / "train_100k"))
    held = dataset_keys(read_dataset(DATA / "eval_held_out"))
    assert not (train & held), f"{len(train & held)} problems appear in both"


def test_suites_are_disjoint_from_each_other() -> None:
    """Different depths cannot share a problem: depth is a function of the state."""
    seen: dict[tuple[int, ...], int] = {}
    for depth in DEPTHS:
        for key in state_keys(suite(depth)):
            assert key not in seen, f"a problem is in both solve_in_{seen[key]} and _{depth}"
            seen[key] = depth


@pytest.mark.parametrize("depth", DEPTHS)
def test_a_suite_has_no_internal_duplicates(depth: int) -> None:
    rows = suite(depth)
    assert len(state_keys(rows)) == len(rows), f"solve_in_{depth} repeats a problem"


def test_the_exhaustive_verification_artifact_covers_every_suite() -> None:
    """**Chunk 5 gate.** All 1,200 labels re-verified, and the record proves it.

    The artifact carries each suite's sha256 *as verified*, so a suite edited
    afterwards cannot inherit an old pass — the check is against the file that
    exists now, not the file that existed then.
    """
    path = REPO / "runs" / "suite_verification.json"
    assert path.exists(), "run `python scripts/verify_suites.py`"
    record = json.loads(path.read_text())

    assert not record["failures"], f"verification recorded failures: {record['failures'][:3]}"
    assert record["total"] == 1200, f"only {record['total']} problems verified"
    for depth in DEPTHS:
        name = f"solve_in_{depth}.jsonl"
        entry = record["suites"].get(name)
        assert entry, f"{name} was never verified"
        assert entry["verified"] == entry["problems"] == 200
        assert entry["sha256"] == sha256_file(SUITES / name), (
            f"{name} changed after it was verified — the pass does not carry over"
        )


# ---------------------------------------------------------------------------
# Reproducibility from seed
# ---------------------------------------------------------------------------


def test_suite_3_regenerates_byte_identical(tmp_path: Path) -> None:
    """**Chunk 5 gate.** The spot-check the plan names, in a fresh process.

    A frozen instrument that cannot be rebuilt from its seed is not frozen, it
    is merely old.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "generate.py"),
            "--suite-only",
            "3",
            "--out",
            str(tmp_path),
            "--workers",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rebuilt = tmp_path / "suites" / "solve_in_3.jsonl"
    original = SUITES / "solve_in_3.jsonl"
    assert sha256_file(rebuilt) == sha256_file(original), (
        "solve_in_3 did not regenerate byte-identically from its seed"
    )


# ---------------------------------------------------------------------------
# The memmap writer
# ---------------------------------------------------------------------------


def _labelled(par: int = 1) -> Problem:
    return Problem(
        goal=GOAL_SOLVE, expr=eq(mul(num(3), X), num(15)), target=VAR_X, par=par, par_source="bfs"
    )


def test_dataset_round_trips(tmp_path: Path) -> None:
    problems = [
        _labelled(),
        Problem(
            goal=GOAL_SOLVE,
            expr=eq(mul(num(4), X), num(20)),
            target=VAR_X,
            par=1,
            par_source="bfs",
        ),
    ]
    meta = write_dataset(tmp_path / "d", problems, CFG, mode="test", seed=1, repo=REPO)
    dataset = read_dataset(tmp_path / "d")
    assert len(dataset) == 2
    assert meta["ruleset_version"] == RULESET_VERSION
    assert meta["vocab_version"] == VOCAB_VERSION
    assert set(meta["digests"]) == set(FIELDS)
    for i, problem in enumerate(problems):
        assert dataset.state(i) == problem_key(problem)
        assert int(dataset.par[i]) == problem.par


def test_dataset_refuses_unshippable_rows(tmp_path: Path) -> None:
    """The shipping boundary, at the writer — the last place it can be enforced."""
    from reckoner.dataset import ShippingError

    with pytest.raises(ShippingError):
        write_dataset(
            tmp_path / "a",
            [Problem(goal=GOAL_SOLVE, expr=eq(mul(num(3), X), num(15)), target=VAR_X)],
            CFG,
            mode="test",
            seed=1,
            repo=REPO,
        )
    with pytest.raises(ShippingError, match="does not ship"):
        write_dataset(
            tmp_path / "b",
            [
                Problem(
                    goal=GOAL_SOLVE,
                    expr=eq(mul(num(3), X), num(15)),
                    target=VAR_X,
                    par=1,
                    par_source="unverified",
                )
            ],
            CFG,
            mode="test",
            seed=1,
            repo=REPO,
        )


def test_empty_dataset_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        write_dataset(tmp_path / "e", [], CFG, mode="test", seed=1, repo=REPO)


def test_suite_writer_refuses_unshippable_rows(tmp_path: Path) -> None:
    from reckoner.dataset import ShippingError

    with pytest.raises(ShippingError):
        write_suite(
            tmp_path / "s.jsonl",
            [Problem(goal=GOAL_SOLVE, expr=eq(mul(num(3), X), num(15)), target=VAR_X)],
        )


# ---------------------------------------------------------------------------
# ANCHORS
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "runs" / "ANCHORS.sha256").exists(), reason="not generated")
def test_anchors_cover_every_artifact_and_still_match() -> None:
    """**Chunk 5 gate.** Digests on disk, and they are the digests of what is there."""
    lines = (REPO / "runs" / "ANCHORS.sha256").read_text().splitlines()
    entries = [line.split("  ", 1) for line in lines if line.strip()]
    assert entries, "ANCHORS.sha256 is empty"

    for digest, rel in entries:
        path = REPO / rel
        assert path.exists(), f"{rel} is anchored but missing"
        assert sha256_file(path) == digest, f"{rel} has changed since it was anchored"

    anchored = {rel for _, rel in entries}
    for depth in DEPTHS:
        assert f"runs/suites/solve_in_{depth}.jsonl" in anchored, f"suite {depth} unanchored"
    assert any(rel.endswith("meta.json") for rel in anchored), "no dataset meta.json anchored"


@pytest.mark.skipif(not (DATA / "train_100k").exists(), reason="training set not generated")
def test_training_set_meta_is_honest() -> None:
    meta = json.loads((DATA / "train_100k" / "meta.json").read_text())
    assert meta["par_source_histogram"] == {"bfs": meta["n"]}, "a non-BFS par shipped"
    assert set(meta["depth_histogram"]) <= {str(d) for d in DEPTHS} | set(DEPTHS)
    assert min(int(d) for d in meta["depth_histogram"]) >= 1, "a par-0 problem shipped"
    assert meta["git_sha"] != "unknown"
    assert len(meta["config_fingerprint"]) == 64


def test_the_generator_is_seeded_not_ambient() -> None:
    """Two plans from one seed are the same plan; from different seeds, not."""
    sys.path.insert(0, str(REPO / "scripts"))
    from generate import _plan

    assert _plan(60, 11) == _plan(60, 11)
    assert _plan(60, 11) != _plan(60, 12)
    random.seed(999)  # ambient state must not reach it
    assert _plan(60, 11) == _plan(60, 11)
