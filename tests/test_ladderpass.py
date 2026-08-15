"""A ladder pass: rows from row one, resumable at a kill point, marker-complete."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reckoner.arms import GreedyHeuristic, RandomRewriter, SympySolver
from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.ladder import LadderError, paired_bootstrap
from reckoner.ladderpass import (
    DONE_MARKER,
    PassError,
    PassPaths,
    comparison_from_pass,
    is_complete,
    read_pair_scores,
    repair_torn_tail,
    run_pass,
)
from reckoner.vocab import GOAL_SOLVE

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()
ROLES = {"greedy": "baseline", "random": "baseline", "sympy": "rung"}

needs_suites = pytest.mark.skipif(
    not (SUITES / "solve_in_3.jsonl").exists(), reason="suites not generated"
)


class Kill(RuntimeError):
    """The simulated kill. Distinct from any error the pass could raise itself."""


def problems(n: int = 16):
    return [suite_problem(r) for r in read_suite(SUITES / "solve_in_3.jsonl")[:n]]


def two_arms():
    return [GreedyHeuristic(), RandomRewriter()]


def a_pass(root: Path, index: int = 0, **kw):
    return run_pass(
        root, index, two_arms(), problems(), CFG, roles=ROLES, calibration_note="test", **kw
    )


# ---------------------------------------------------------------------------
# From row one
# ---------------------------------------------------------------------------


@needs_suites
def test_rows_land_as_they_happen_not_at_the_end(tmp_path: Path) -> None:
    """A pass killed at 90% must leave 90% of its rows, not nothing."""
    seen: list[int] = []

    def watch(arm: str, i: int) -> None:
        if arm == "greedy" and i == 5:
            seen.append(len(read_pair_scores(tmp_path, 0)))

    a_pass(tmp_path, on_unit=watch)
    assert seen == [5], f"expected 5 rows already on disk at unit 5, saw {seen}"


@needs_suites
def test_a_complete_pass_writes_its_marker_and_every_row(tmp_path: Path) -> None:
    record = a_pass(tmp_path)
    assert record["rows"] == 2 * len(problems())
    assert record["rows_resumed"] == 0
    assert is_complete(tmp_path, 0)
    assert json.loads(PassPaths(tmp_path, 0).marker.read_text())["rows"] == record["rows"]


@needs_suites
def test_completion_is_a_marker_not_an_absent_process(tmp_path: Path) -> None:
    """A waiter that infers 'done' from 'not running' also infers it from 'crashed'."""
    assert not is_complete(tmp_path, 0)
    with pytest.raises(Kill):
        a_pass(tmp_path, on_unit=lambda arm, i: (_ for _ in ()).throw(Kill()) if i == 3 else None)
    assert read_pair_scores(tmp_path, 0), "rows landed"
    assert not is_complete(tmp_path, 0), "a partial pass must not read as complete"


@needs_suites
def test_rerunning_a_completed_pass_is_refused(tmp_path: Path) -> None:
    a_pass(tmp_path)
    with pytest.raises(PassError, match=DONE_MARKER):
        a_pass(tmp_path)


# ---------------------------------------------------------------------------
# Resume, at a kill point
# ---------------------------------------------------------------------------


@needs_suites
def test_a_pass_killed_mid_way_resumes_to_the_identical_result(tmp_path: Path) -> None:
    """The proof: interrupted-then-resumed and uninterrupted agree row for row."""
    uninterrupted = tmp_path / "whole"
    a_pass(uninterrupted)
    reference = read_pair_scores(uninterrupted, 0)

    killed = tmp_path / "killed"
    with pytest.raises(Kill):
        a_pass(
            killed,
            on_unit=lambda arm, i: (
                (_ for _ in ()).throw(Kill()) if (arm == "random" and i == 7) else None
            ),
        )
    partial = read_pair_scores(killed, 0)
    assert 0 < len(partial) < len(reference), f"kill point produced {len(partial)} rows"

    resumed_record = a_pass(killed)
    resumed = read_pair_scores(killed, 0)

    assert resumed_record["rows_resumed"] == len(partial)
    assert len(resumed) == len(reference)
    assert [(r["arm"], r["problem_key"]) for r in resumed] == [
        (r["arm"], r["problem_key"]) for r in reference
    ]
    assert resumed == reference, "a resumed pass must be the pass, not a similar one"


@needs_suites
def test_resume_writes_no_duplicate_rows(tmp_path: Path) -> None:
    with pytest.raises(Kill):
        a_pass(tmp_path, on_unit=lambda arm, i: (_ for _ in ()).throw(Kill()) if i == 9 else None)
    a_pass(tmp_path)
    units = [(r["arm"], r["problem_key"]) for r in read_pair_scores(tmp_path, 0)]
    assert len(units) == len(set(units))


def test_a_torn_final_line_is_truncated(tmp_path: Path) -> None:
    """What a kill mid-append actually leaves behind."""
    path = tmp_path / "rows.jsonl"
    torn = '{"c":'
    path.write_text(f'{{"a": 1}}\n{{"b": 2}}\n{torn}')
    assert repair_torn_tail(path) == len(torn)
    assert path.read_text() == '{"a": 1}\n{"b": 2}\n'


def test_a_torn_middle_line_raises(tmp_path: Path) -> None:
    """The rejecting case. Dropping it silently turns corruption into a shorter
    pass that looks complete."""
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"b":\n{"c": 3}\n')
    with pytest.raises(PassError, match="not a complete row"):
        repair_torn_tail(path)


def test_an_intact_file_is_left_alone(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n')
    assert repair_torn_tail(path) == 0
    assert path.read_text() == '{"a": 1}\n'


# ---------------------------------------------------------------------------
# Roles, currencies, and the arms that decline a question
# ---------------------------------------------------------------------------


@needs_suites
def test_an_undeclared_role_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PassError, match="no role declared"):
        run_pass(
            tmp_path,
            0,
            two_arms(),
            problems(),
            CFG,
            roles={"greedy": "baseline"},
            calibration_note="test",
        )


@needs_suites
def test_an_unknown_role_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PassError, match="unknown ladder role"):
        run_pass(
            tmp_path,
            0,
            two_arms(),
            problems(),
            CFG,
            roles={"greedy": "champion", "random": "baseline"},
            calibration_note="test",
        )


@needs_suites
def test_a_declining_arm_is_short_by_a_counted_skip_not_a_missing_row(tmp_path: Path) -> None:
    """Sympy plays SOLVE and EVALUATE; SIMPLIFY is a declared skip.

    The pass must record the skip and still complete — scoring an unasked question
    as a failure would report the rung as weak for a reason unrelated to its skill.
    """
    with SympySolver(CFG) as sympy:
        if not sympy.available:
            pytest.skip("sympy absent — the clean-skip path, exercised elsewhere")
        sympy.probe()
        problem_set = problems()
        record = run_pass(
            tmp_path,
            0,
            [GreedyHeuristic(), sympy],
            problem_set,
            CFG,
            roles=ROLES,
            calibration_note="test",
        )
    declined = sum(1 for p in problem_set if not sympy.plays(p))
    assert record["skipped_by_arm"].get("sympy", 0) == declined
    assert record["rows_by_arm"]["sympy"] == len(problem_set) - declined
    assert record["rows_by_arm"]["greedy"] == len(problem_set)


@needs_suites
def test_pairing_two_arms_with_different_playable_subsets_is_refused(tmp_path: Path) -> None:
    """The currency refusal fires first, and the overlap refusal is behind it."""
    with SympySolver(CFG) as sympy:
        if not sympy.available:
            pytest.skip("sympy absent")
        run_pass(
            tmp_path,
            0,
            [GreedyHeuristic(), sympy],
            problems(),
            CFG,
            roles=ROLES,
            calibration_note="test",
        )
    with pytest.raises(LadderError, match="cannot pair across currencies"):
        comparison_from_pass(tmp_path, 0, "greedy", "sympy")


@needs_suites
def test_a_pass_feeds_the_bootstrap_it_was_written_for(tmp_path: Path) -> None:
    """The end-to-end claim: rows on disk become a paired comparison and a CI,
    with no step that needed an aggregate nobody kept."""
    a_pass(tmp_path)
    comparison = comparison_from_pass(tmp_path, 0, "greedy", "random")
    assert len(comparison.differences) == len(problems())
    result = paired_bootstrap(comparison.differences, resamples=500, seed=0)
    assert result["n_pairs"] == len(problems())
    assert result["ci_low"] <= result["mean_difference"] <= result["ci_high"]


@needs_suites
def test_z_rows_carry_no_budget_fields_and_the_schema_enforces_it(tmp_path: Path) -> None:
    a_pass(tmp_path)
    for row in read_pair_scores(tmp_path, 0):
        assert row["currency"] == "z_vs_par"
        assert not {"solved", "steps_used", "budget", "cas_version"} & set(row)
        assert row["calibration_note"], "a score without its caveat gets quoted without it"


@needs_suites
def test_sympy_rows_carry_the_cas_version_as_part_of_the_rungs_identity(tmp_path: Path) -> None:
    with SympySolver(CFG) as sympy:
        if not sympy.available:
            pytest.skip("sympy absent")
        run_pass(tmp_path, 0, [sympy], problems(), CFG, roles=ROLES, calibration_note="test")
        expected = sympy.version
    rows = read_pair_scores(tmp_path, 0)
    assert rows and all(r["cas_version"] == expected for r in rows)
    assert all(not {"z", "par", "par_source"} & set(r) for r in rows)


def test_sympy_refuses_a_goal_it_declared_out_of_scope() -> None:
    from reckoner.arms import ArmError

    with SympySolver(CFG) as sympy:
        if not sympy.available:
            pytest.skip("sympy absent")
        simplify = next(
            (p for p in problems(64) if not sympy.plays(p)),
            None,
        )
        if simplify is None:
            pytest.skip("no declined goal in this slice")
        with pytest.raises(ArmError, match="does not play"):
            sympy.play(simplify, CFG, seed=0)


def test_sympy_solves_a_solve_problem_through_our_own_arbiter() -> None:
    """The accepting polarity: the rung is not merely well-behaved, it works."""
    with SympySolver(CFG) as sympy:
        if not sympy.available:
            pytest.skip("sympy absent")
        solvable = [p for p in problems(64) if p.goal == GOAL_SOLVE]
        assert solvable, "no SOLVE problems in the slice — this test would be vacuous"
        results = [sympy.play(p, CFG, seed=0) for p in solvable]
    assert all(r.solved for r in results), "sympy failed a linear equation"
    assert all(r.currency == "solve_vs_budget" for r in results)
