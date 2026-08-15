"""The replay ring: capacity, absence semantics, era handling, and root_q's sign.

The sign test is the inherited p2_c obligation taken whole. Chess specified
`root_q` and never stored it, and the gap surfaced only when the z/q blend needed
the field — so here it is stored from field one AND its sign is proven before it
feeds anything. A value whose sign is wrong does not fail loudly; it trains
confidently in the wrong direction.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.episode import Problem
from reckoner.replay import (
    RING_FORMAT,
    Absent,
    ReplayRing,
    RingError,
    bytes_per_record,
)
from reckoner.search import search, uniform_stub

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()


def a_record(**overrides) -> dict:
    record = {
        "tokens": np.array([1, 2, 3], dtype=np.int16),
        "site_positions": np.array([0, 1], dtype=np.int16),
        "visit_actions": np.array([5, 9], dtype=np.int32),
        "visit_counts": np.array([12, 4], dtype=np.int32),
        "root_q": 0.5,
        "z": 0,
        "par_source": "bfs",
        "par": 3,
        "steps_remaining": 2,
        "depth": 3,
        "goal": 16,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# The arithmetic, asserted rather than quoted
# ---------------------------------------------------------------------------


def test_bytes_per_record_matches_the_declared_footprint() -> None:
    """The docstring's table and the code must not be two mental models.

    Fourth-instance discipline: a summary of a derivation drifts from it, so the
    number is computed and the declared value is checked against the computation.
    """
    assert bytes_per_record(CFG) == 1558


def test_the_declared_capacity_fits_the_declared_budget() -> None:
    """0.72 GiB at 500,000 against ~15 GiB realistically free (AGENTS.md §3)."""
    gib = CFG.train.replay_capacity * bytes_per_record(CFG) / 1024**3
    assert gib == pytest.approx(0.72, abs=0.01)
    assert gib < 15.0


# ---------------------------------------------------------------------------
# Ring behaviour
# ---------------------------------------------------------------------------


def test_it_stores_and_returns_a_record() -> None:
    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    got = ring.get(0)
    assert got["par_source"] == "bfs"
    assert got["root_q"] == pytest.approx(0.5)
    assert list(got["tokens"]) == [1, 2, 3]


def test_it_overwrites_oldest_first_and_length_saturates() -> None:
    ring = ReplayRing(3, CFG)
    for i in range(5):
        ring.append(**a_record(par=i))
    assert len(ring) == 3
    assert {int(ring.get(s)["par"]) for s in range(3)} == {2, 3, 4}


def test_sampling_uses_the_blessed_sampler_and_is_never_a_prefix() -> None:
    ring = ReplayRing(1000, CFG)
    for i in range(1000):
        ring.append(**a_record(par=i % 6 + 1))
    picked = ring.sample(50, seed=0)
    assert len(picked) == 50
    assert picked != list(range(50))


def test_reading_outside_the_valid_range_raises() -> None:
    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    with pytest.raises(RingError, match="outside the 1 valid records"):
        ring.get(1)


# ---------------------------------------------------------------------------
# The third layer of the exact-par tripwire
# ---------------------------------------------------------------------------


def test_beating_exact_par_is_refused_at_the_ring_boundary() -> None:
    """EpisodeResult, logschema, and now here — checked wherever it crosses."""
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="impossible by construction"):
        ring.append(**a_record(par_source="bfs", z=1))


def test_beating_pool_par_is_allowed() -> None:
    """Pool par is not exact; beating it is the escalation mechanism."""
    ring = ReplayRing(4, CFG)
    ring.append(**a_record(par_source="pool", z=1))
    assert ring.get(0)["z"] == 1


def test_an_unknown_par_source_is_refused() -> None:
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="unknown par_source"):
        ring.append(**a_record(par_source="vibes"))


def test_an_out_of_range_z_is_refused() -> None:
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="z must be"):
        ring.append(**a_record(z=2))


# ---------------------------------------------------------------------------
# One absence semantics — and absence is never a zero
# ---------------------------------------------------------------------------


def test_an_undeclared_absence_is_refused() -> None:
    """A hole nobody can interpret is worse than a missing value."""
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="does not declare that it can be absent"):
        ring.append(**a_record(absent={"root_q": "search did not run"}))


def test_naming_an_unknown_field_absent_is_refused() -> None:
    ring = ReplayRing(4, CFG)
    with pytest.raises(RingError, match="unknown field"):
        ring.append(**a_record(absent={"not_a_field": "because"}))


def test_era_absence_is_computed_not_stored() -> None:
    """A record written before a field existed cannot testify about it.

    The ring is written under format 0; every field arrived in format 1, so every
    field reads Absent with kind 'era' — computed from (record format, since).
    """
    ring = ReplayRing(4, CFG, ring_format=0)
    ring.append(**a_record())
    got = ring.get(0)
    assert isinstance(got["root_q"], Absent)
    assert got["root_q"].kind == "era"
    assert "predates_field" in got["root_q"].reason


def test_a_current_format_record_reports_no_era_absence() -> None:
    """The other polarity — otherwise era handling degenerates to 'always absent'."""
    ring = ReplayRing(4, CFG, ring_format=RING_FORMAT)
    ring.append(**a_record())
    assert not isinstance(ring.get(0)["root_q"], Absent)


def test_an_absent_value_refuses_to_be_read_as_a_zero() -> None:
    """`Absent.__bool__` raises: truthiness is how an absence becomes a zero."""
    ring = ReplayRing(4, CFG, ring_format=0)
    ring.append(**a_record())
    absent = ring.get(0)["root_q"]
    with pytest.raises(RingError, match="test for Absent"):
        bool(absent)


# ---------------------------------------------------------------------------
# root_q's SIGN and SCALE — re-pinned on the z currency (F-13)
# ---------------------------------------------------------------------------
#
# The first version of this test asserted root_value == 1.0 on a forced win and
# PASSED — which is how it exposed the defect. A forced win *at par* is z = 0, a
# draw, so a +1.0 reading meant the tree scored solved-flat while training scored
# z-vs-par: two currencies in one loop. The contrast that mattered was never
# win-versus-loss; it was at-par-versus-under-par.


def _pessimistic(cfg: Config):
    """Flat priors, value -1.0.

    `root_value` is `max(the root's own evaluation, best line)`, so a NEUTRAL
    evaluator floors it at 0.0 and an over-par line can never show through. The
    fixture is about the terminal scale, so the root's prior must not mask it.
    """
    width = 7 * cfg.model.max_sites

    def evaluate(leaves):
        return [(np.zeros(width, dtype=np.float32), -1.0) for _ in leaves]

    return evaluate


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_root_q_is_plus_one_when_the_solve_beats_par() -> None:
    """Under par pays +1. Fixture par, since nothing beats real BFS par."""
    row = read_suite(SUITES / "solve_in_1.jsonl")[0]
    base = suite_problem(row)
    under = Problem(goal=base.goal, expr=base.expr, par=5, target=base.target, par_source="pool")
    result = search(under, under.expr, uniform_stub(CFG), CFG, random.Random(0), sims=16, m=5)
    assert result.root_value == pytest.approx(1.0)


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_root_q_is_zero_when_the_solve_only_matches_par() -> None:
    """**The fixture the old test was missing.** At par is a DRAW, not a win.

    A depth-1 problem solved in one step ties BFS-exact par. Reading +1.0 here is
    the solved-flat defect; reading 0.0 is the par game.
    """
    row = read_suite(SUITES / "solve_in_1.jsonl")[0]
    problem = suite_problem(row)
    result = search(problem, problem.expr, uniform_stub(CFG), CFG, random.Random(0), sims=16, m=5)
    assert result.root_value == pytest.approx(0.0)


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_root_q_is_minus_one_when_the_line_runs_over_par() -> None:
    """Over par pays -1, identically to the cap (plan chunk 3)."""
    row = read_suite(SUITES / "solve_in_1.jsonl")[0]
    problem = suite_problem(row)
    result = search(
        problem,
        problem.expr,
        _pessimistic(CFG),
        CFG,
        random.Random(0),
        sims=16,
        m=5,
        steps_taken=problem.par,  # the episode already spent its whole budget
    )
    assert result.root_value == pytest.approx(-1.0)


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_the_stored_root_q_is_the_searched_one() -> None:
    """The sign must survive the ring, not just the search."""
    row = read_suite(SUITES / "solve_in_1.jsonl")[0]
    problem = suite_problem(row)
    result = search(problem, problem.expr, uniform_stub(CFG), CFG, random.Random(0), sims=16, m=5)
    ring = ReplayRing(4, CFG)
    ring.append(**a_record(root_q=result.root_value, z=0, par_source="bfs"))
    assert ring.get(0)["root_q"] == pytest.approx(result.root_value)
    assert ring.get(0)["root_q"] == pytest.approx(0.0), "at par is a draw, and the ring stores it"


# ---------------------------------------------------------------------------
# The parked m -> 32 lever must trip loudly, not truncate
# ---------------------------------------------------------------------------


def test_raising_m_past_the_ring_slots_is_refused() -> None:
    """m is CONFIG; the visit layout is FORMAT. load()'s layout check cannot see
    a config that outgrew it, so the ring asserts at open."""
    cfg = Config()
    cfg.search.gumbel_m = 32
    with pytest.raises(RingError, match="exceeds the ring's 16 visit slots"):
        ReplayRing(4, cfg)


def test_m_at_the_slot_count_is_accepted() -> None:
    """The other polarity — the guard must not refuse the layout it was sized for."""
    cfg = Config()
    cfg.search.gumbel_m = 16
    assert ReplayRing(4, cfg).visit_actions.shape[1] == 16


# ---------------------------------------------------------------------------
# Persistence — the resume contract
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    ring = ReplayRing(8, CFG)
    for i in range(5):
        ring.append(**a_record(par=i + 1, root_q=i / 10))
    ring.save(tmp_path / "ring")
    back = ReplayRing.load(tmp_path / "ring", CFG)
    assert len(back) == 5
    assert back.cursor == ring.cursor
    for slot in range(5):
        assert back.get(slot)["par"] == ring.get(slot)["par"]
        assert back.get(slot)["root_q"] == pytest.approx(ring.get(slot)["root_q"])


def test_meta_is_written_last_so_its_presence_means_completeness(tmp_path: Path) -> None:
    """The write-ordering contract, asserted rather than described."""
    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    ring.save(tmp_path / "ring")
    meta = tmp_path / "ring" / "meta.json"
    assert meta.exists()
    newest = max((p.stat().st_mtime_ns for p in (tmp_path / "ring").iterdir()), default=0)
    assert meta.stat().st_mtime_ns == newest, "meta.json must be the last file written"


def test_loading_under_a_different_layout_is_refused(tmp_path: Path) -> None:
    """The record layout is denominated in these shapes; reinterpreting is silent."""
    ring = ReplayRing(4, CFG)
    ring.append(**a_record())
    ring.save(tmp_path / "ring")
    other = Config()
    other.model.max_sites = 64
    with pytest.raises(RingError, match="record layout is denominated"):
        ReplayRing.load(tmp_path / "ring", other)
