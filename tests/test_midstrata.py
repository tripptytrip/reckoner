"""The mid-strata instrument: scripted par is a floor, so the +1 cell is live.

This is the property the whole instrument exists for, so it gets an assertion
rather than a comment — and it gets both polarities, because "+1 is allowed here"
is only meaningful beside "+1 is still refused there".
"""

from __future__ import annotations

import collections
import random
from pathlib import Path

import pytest

from reckoner.config import Config
from reckoner.dataset import problem_key, read_suite, suite_problem
from reckoner.episode import EpisodeResult, outcome_z
from reckoner.generator import MID_TEMPLATES, emit_mid
from reckoner.solver import scripted_par

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()
STRATA = (7, 8, 9, 10)

needs_mint = pytest.mark.skipif(
    not (SUITES / "scripted_in_7.jsonl").exists(), reason="mid-strata not minted"
)


@needs_mint
@pytest.mark.parametrize("k", STRATA)
def test_every_row_carries_the_label_its_stratum_is_named_for(k: int) -> None:
    """Stratum identity is the LABEL, never the template's intention (chunk-5)."""
    rows = read_suite(SUITES / f"scripted_in_{k}.jsonl")
    assert len(rows) == 200
    assert {r["par"] for r in rows} == {k}
    assert {r["par_source"] for r in rows} == {"scripted"}


@needs_mint
@pytest.mark.parametrize("k", STRATA)
def test_the_plus_one_cell_is_live_under_scripted_par(k: int) -> None:
    """The accepting polarity. Beating a provisional floor is the whole point."""
    problem = suite_problem(read_suite(SUITES / f"scripted_in_{k}.jsonl")[0])
    result = EpisodeResult(
        ruleset_version=1,
        vocab_version=1,
        goal=problem.goal,
        solved=True,
        steps=problem.par - 1,
        par=problem.par,
        par_source="scripted",
        z=outcome_z(solved=True, steps=problem.par - 1, par=problem.par),
        terminal_reason="solved",
    )
    assert result.z == 1


@needs_mint
def test_the_exact_par_tripwire_is_untouched() -> None:
    """The rejecting polarity, on the same construction. Minting an instrument
    where +1 is legal must not make it legal where par is exact."""
    problem = suite_problem(read_suite(SUITES / "scripted_in_7.jsonl")[0])
    with pytest.raises(ValueError, match="z = \\+1 against par_source='bfs'"):
        EpisodeResult(
            ruleset_version=1,
            vocab_version=1,
            goal=problem.goal,
            solved=True,
            steps=problem.par - 1,
            par=problem.par,
            par_source="bfs",
            z=1,
            terminal_reason="solved",
        )


@needs_mint
def test_the_mid_strata_are_not_in_the_bfs_exact_series() -> None:
    """`solve_in_*` is a live glob in three scripts and sizes the instrument P1's
    no-regress floor is computed against. A new file must not join it."""
    assert sorted(p.stem for p in SUITES.glob("solve_in_*.jsonl")) == [
        f"solve_in_{k}" for k in range(1, 7)
    ]
    assert sum(len(read_suite(p)) for p in SUITES.glob("solve_in_*.jsonl")) == 1200


@needs_mint
def test_no_problem_appears_in_two_strata() -> None:
    keys = [
        problem_key(suite_problem(r))
        for k in STRATA
        for r in read_suite(SUITES / f"scripted_in_{k}.jsonl")
    ]
    assert len(keys) == len(set(keys))


def test_the_mid_templates_land_where_their_stratum_needs_them() -> None:
    """Bounded in test, exhaustive in script — the standing precedent.

    Asserts the templates reach the 7..10 band at all; the mint script measures
    the full distribution and buckets by the measured label.
    """
    for name in MID_TEMPLATES:
        pars = collections.Counter(
            scripted_par(emit_mid(name, random.Random(s)), CFG) for s in range(40)
        )
        reached = {p for p in pars if p is not None and p in STRATA}
        assert reached, f"{name} reached {dict(pars)}, none of it in 7..10"


def test_the_mid_templates_are_kept_out_of_the_bfs_template_registry() -> None:
    """Two dicts, because the two carry different epistemic status: the z=+1
    tripwire fires against one family's labels and not the other's."""
    from reckoner.generator import TEMPLATES

    assert not set(TEMPLATES) & set(MID_TEMPLATES)
