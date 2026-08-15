"""Ladder rungs and the currency ruling, on both polarities.

The currency guard is the piece that must be structural rather than remembered:
two currencies never silently mix, and the schema is what stops them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reckoner.arms import ArmError, GreedyHeuristic, RandomRewriter, SympySolver
from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.logschema import (
    CURRENCY_BUDGET,
    CURRENCY_Z,
    SCHEMA_ERA,
    SchemaError,
    validate_ladder_row,
)

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()

needs_suites = pytest.mark.skipif(
    not (SUITES / "solve_in_3.jsonl").exists(), reason="suites not generated"
)


def a_problem(depth: int = 3):
    return suite_problem(read_suite(SUITES / f"solve_in_{depth}.jsonl")[0])


def common_row(**over) -> dict:
    row = {
        "pass_index": 0,
        "schema_era": SCHEMA_ERA,
        "arm": "greedy",
        "problem_key": "abc",
        "role": "rung",
        "nondeterministic": False,
        "seed": 7,
        "calibration_note": "smoke pass; not a campaign claim",
    }
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# The currency ruling — structural, not remembered
# ---------------------------------------------------------------------------


def test_a_rule_denominated_row_validates() -> None:
    row = common_row(currency=CURRENCY_Z, z=0, steps=3, par=3, par_source="bfs")
    assert validate_ladder_row(row) == []


def test_an_external_row_validates() -> None:
    row = common_row(
        arm="sympy",
        currency=CURRENCY_BUDGET,
        solved=True,
        steps_used=4,
        budget=16,
        cas_version="1.14.0",
    )
    assert validate_ladder_row(row) == []


def test_a_row_carrying_both_currencies_is_refused() -> None:
    """**The ruling made structural.** sympy's steps are not our steps, and a row
    that carries both invites exactly the average nobody should compute."""
    row = common_row(currency=CURRENCY_Z, z=0, steps=3, par=3, par_source="bfs", solved=True)
    with pytest.raises(SchemaError, match="belong to the other currency"):
        validate_ladder_row(row)


def test_an_external_row_may_not_carry_a_z() -> None:
    row = common_row(
        arm="sympy",
        currency=CURRENCY_BUDGET,
        solved=True,
        steps_used=4,
        budget=16,
        cas_version="1.14.0",
        z=0,
    )
    with pytest.raises(SchemaError, match="belong to the other currency"):
        validate_ladder_row(row)


def test_a_row_without_a_currency_is_refused() -> None:
    """A number whose units are a guess is not a measurement."""
    with pytest.raises(SchemaError, match="Every ladder row states its currency"):
        validate_ladder_row(common_row(z=0, steps=3, par=3, par_source="bfs"))


def test_an_unknown_currency_is_refused() -> None:
    with pytest.raises(SchemaError, match="declares currency"):
        validate_ladder_row(common_row(currency="elo"))


def test_the_calibration_note_is_required() -> None:
    """A score without its caveat gets quoted without its caveat."""
    row = common_row(currency=CURRENCY_Z, z=0, steps=3, par=3, par_source="bfs")
    del row["calibration_note"]
    with pytest.raises(SchemaError, match="required field missing"):
        validate_ladder_row(row)


# ---------------------------------------------------------------------------
# Determinism probes as CONSTRUCTION gates
# ---------------------------------------------------------------------------


@needs_suites
def test_the_deterministic_arm_probes_deterministic() -> None:
    arm = GreedyHeuristic()
    arm.probe(a_problem(), CFG)
    assert arm.nondeterministic is False
    assert arm.play(a_problem(), CFG, 0) == arm.play(a_problem(), CFG, 999)


@needs_suites
def test_the_stochastic_arm_probes_STOCHASTIC() -> None:
    """The opposite gate. A stochastic arm that never varies is a deterministic
    arm wearing a seed parameter — and its rows would carry a label that lies."""
    arm = RandomRewriter()
    arm.probe(a_problem(), CFG)
    assert arm.nondeterministic is True
    outcomes = {(arm.play(a_problem(), CFG, s).steps) for s in range(24)}
    assert len(outcomes) > 1


@needs_suites
def test_a_stochastic_arm_still_reproduces_on_one_seed() -> None:
    """Or the repetitions are not poolable."""
    arm = RandomRewriter()
    assert arm.play(a_problem(), CFG, 5) == arm.play(a_problem(), CFG, 5)


@needs_suites
def test_a_deterministic_probe_would_catch_a_liar() -> None:
    """The probe must be able to fail, or it is a comment that happens to run."""

    class Liar(GreedyHeuristic):
        def play(self, problem, cfg, seed):
            return type(super().play(problem, cfg, seed))(True, seed, CURRENCY_Z)

    with pytest.raises(ArmError, match="declares itself deterministic and is not"):
        Liar().probe(a_problem(), CFG)


# ---------------------------------------------------------------------------
# The external rung: version-pinned, context-managed, clean-skip
# ---------------------------------------------------------------------------


def test_sympy_is_context_managed_and_reports_its_version() -> None:
    with SympySolver(CFG) as solver:
        solver.probe()
        if solver.available:
            assert solver.version != "absent"
            assert solver.currency == CURRENCY_BUDGET
        else:
            assert solver.version == "absent"


def test_sympy_scores_in_the_other_currency_by_declaration() -> None:
    """Spec: sympy is a rung, never par. Its steps are denominated elsewhere."""
    assert SympySolver(CFG).currency == CURRENCY_BUDGET
    assert GreedyHeuristic().currency == CURRENCY_Z
    assert RandomRewriter().currency == CURRENCY_Z


def test_an_absent_cas_is_a_smaller_ladder_not_a_failed_pass() -> None:
    solver = SympySolver(CFG)
    assert solver.available is False, "available must be False before entering"
    solver.probe()  # must not raise when unavailable
