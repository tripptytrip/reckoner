"""The resume gate: killed-and-resumed is indistinguishable from uninterrupted.

`Harness` in `test_resume.py` probes the commit contract in isolation with
synthetic payloads, and that isolation is a feature — a kill test that can fail
for reasons which are not the commit contract is a worse test. This is the other
half, on the real composition: the driver, killed by a real SIGKILL at each of
the two boundaries `resume.py` names provisional, then resumed.

**The comparator names what it ignores, per field.** A comparator that quietly
skips whatever it cannot explain is a comparator that cannot fail — so the
excluded set is exactly three columns, each justified below, and *everything
else must match*, provenance included. That the evaluator digest must match is
the sharpest of these: a resumed run has to load the same checkpoint and play
the same weights, or the two runs were not the same run.

Four claims at each kill point:

1. rows equal, excluding only wall-clock
2. **ring content identity** — F-13's duplication signature is precisely what
   this disproves: a ring that redid a killed iteration carries its steps twice.
   This assertion also found F-23, the pool that resume rebuilt with the anchor
   alone: every row column agreed while the ring differed, so nothing weaker
   than ring content would have seen it
3. no orphan ring survives for an iteration that never committed
4. ``LATEST`` names the last iteration that actually completed

**Every append-only log is compared, not just `iterations.jsonl`.** F-26 lived in
`value_switch.jsonl` for exactly as long as this gate compared one file while
describing itself as comparing the run: a comparator that names its exclusions
field-by-field within one artifact, and omits a whole artifact in silence, is
precise about the wrong boundary.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reckoner.campaign import ANCHOR, golden_config
from reckoner.replay import ReplayRing

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "tests" / "subprocess_probes" / "campaign_kill.py"
ITERATIONS = 3

#: Excluded from row equality, and NOTHING ELSE IS. Each is wall-clock: it
#: measures how long the machine took, not what the loop did, and a resumed run
#: legitimately spends different time reaching an identical state.
#:
#:   seconds_self_play — episode generation, varies with scheduler and cache
#:   seconds_train     — optimiser wall-clock, same
#:   seconds_total     — their sum plus commit I/O
#:
#: Every other column — counters, outcomes, entropies, health, and all seven
#: provenance fields including evaluator_checkpoint_sha256 — must be identical.
WALL_CLOCK = ("seconds_self_play", "seconds_train", "seconds_total")

needs_anchor = pytest.mark.skipif(not ANCHOR.exists(), reason="phase-1 anchor not present")
pytestmark = [pytest.mark.slow, needs_anchor]


def drive_probe(run_dir: Path, kill_at: tuple[int, str] | None = None) -> int:
    argv = [sys.executable, str(PROBE), str(run_dir)]
    if kill_at is not None:
        argv += [str(kill_at[0]), kill_at[1]]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    return completed.returncode


#: Every append-only log the driver writes. Named as a set so that adding a log
#: without adding it here is a visible omission rather than a silent one.
LOGS = ("iterations.jsonl", "value_switch.jsonl", "instruments.jsonl")


def rows_of(run_dir: Path, name: str = "iterations.jsonl") -> list[dict]:
    path = run_dir / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def comparable(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in WALL_CLOCK} for row in rows]


def latest(run_dir: Path) -> int | None:
    marker = run_dir / "LATEST"
    return int(marker.read_text().strip()) if marker.exists() else None


def ring_signature(run_dir: Path) -> tuple[int, str]:
    """Length and a content digest of the committed ring.

    Length alone is the F-13 signature — a redone iteration doubles its steps —
    and the digest catches the subtler case where the count survives but the
    contents drifted.
    """
    committed = latest(run_dir)
    ring = ReplayRing.load(run_dir / f"ring-{committed}", golden_config())
    blob = "|".join(
        f"{ring.get(slot)['par']}:{ring.get(slot)['z']}:{ring.get(slot)['par_source']}"
        for slot in range(len(ring))
    )
    return len(ring), hashlib.sha256(blob.encode()).hexdigest()


def orphan_rings(run_dir: Path) -> list[str]:
    committed = latest(run_dir)
    return sorted(
        p.name for p in run_dir.glob("ring-*") if int(p.name.split("-")[1]) > (committed or -1)
    )


@pytest.fixture(scope="session")
def uninterrupted(tmp_path_factory) -> Path:
    """The baseline, produced by the SAME program the killed runs use.

    Same program, so the comparison cannot drift into comparing two different
    loops — which is the failure mode a hand-written baseline invites.
    """
    run_dir = tmp_path_factory.mktemp("clean") / "run"
    assert drive_probe(run_dir) == 0
    assert latest(run_dir) == ITERATIONS - 1
    return run_dir


@pytest.mark.parametrize(
    "phase",
    [
        pytest.param("before_row", id="kill-A-ring-and-state-written-row-not"),
        pytest.param("before_latest", id="kill-B-row-written-LATEST-not"),
    ],
)
def test_killed_and_resumed_is_indistinguishable_from_uninterrupted(
    uninterrupted: Path, tmp_path: Path, phase: str
) -> None:
    run_dir = tmp_path / "run"

    killed_rc = drive_probe(run_dir, kill_at=(1, phase))
    assert killed_rc == -9, f"the probe did not die by SIGKILL (rc={killed_rc})"
    assert latest(run_dir) == 0, "a kill inside iteration 1 must leave LATEST at 0"

    assert drive_probe(run_dir) == 0, "the resumed run did not complete"

    # 1. rows, excluding only wall-clock — and EVERY log, not just this one
    assert comparable(rows_of(run_dir)) == comparable(rows_of(uninterrupted))
    for name in LOGS:
        got, want = rows_of(run_dir, name), rows_of(uninterrupted, name)
        assert [r.get("iteration") for r in got] == [r.get("iteration") for r in want], (
            f"{name}: a resumed run must not duplicate or drop a row (F-26)"
        )
    # AND SAY WHICH OF THOSE COMPARISONS WAS REAL. `instruments.jsonl` does not
    # exist at golden config — `ladder_every = 99`, so the cadence never fires —
    # so its comparison above passes on two empty lists. An assertion that cannot
    # fail must not be left looking like one that did: the door-level cover is
    # `test_resume.py::test_resume_truncates_the_instrument_log_by_iteration...`,
    # and this line is what would break if golden ever gained a cadence and this
    # comment silently stopped being true.
    exercised = [name for name in LOGS if (uninterrupted / name).exists()]
    assert exercised == ["iterations.jsonl", "value_switch.jsonl"], (
        f"the set of logs this gate actually compares has changed: {exercised}"
    )

    # 2. ring content identity — F-13's duplication signature disproved
    assert ring_signature(run_dir) == ring_signature(uninterrupted)

    # 3. the pool was rebuilt, not restarted (F-23). Asserted through the
    #    snapshots on disk, since pool composition reaches no row column — which
    #    is the observability gap the finding registers.
    assert sorted(p.name for p in run_dir.glob("snap-*.pt")) == sorted(
        p.name for p in uninterrupted.glob("snap-*.pt")
    ), "the resumed run did not enrol the same snapshots"

    # 4. no orphan ring for an iteration that never committed
    assert orphan_rings(run_dir) == []

    # 5. LATEST names the last iteration that actually completed
    assert latest(run_dir) == latest(uninterrupted) == ITERATIONS - 1


@pytest.mark.parametrize("phase", ["before_row", "before_latest"])
def test_the_kill_leaves_exactly_the_debris_resume_promises(tmp_path: Path, phase: str) -> None:
    """The provisional-artifact contract, asserted before resume cleans it.

    `resume.py` says steps 1–3 are provisional and only the rename commits. This
    is that claim observed rather than trusted: iteration 1's ring exists on disk
    while `LATEST` still names iteration 0.
    """
    run_dir = tmp_path / "run"
    assert drive_probe(run_dir, kill_at=(1, phase)) == -9

    assert latest(run_dir) == 0
    assert "ring-1" in orphan_rings(run_dir), "the provisional ring was never written"
    rows = rows_of(run_dir)
    if phase == "before_row":
        assert len(rows) == 1, "the row was written despite the kill preceding it"
    else:
        assert len(rows) == 2, "the row was not written despite the kill following it"


@needs_anchor
def test_resume_refuses_when_a_committed_iteration_s_snapshot_is_missing() -> None:
    """F-23's refusal polarity. A thinner pool is not a smaller inconvenience —
    it is the defect the rebuild exists to prevent, arriving quietly by the path
    meant to prevent it. So resume raises rather than continuing."""
    import shutil
    import tempfile

    from reckoner.pool import PoolError

    run_dir = Path(tempfile.mkdtemp()) / "run"
    try:
        assert drive_probe(run_dir, kill_at=(2, "before_row")) == -9
        assert (run_dir / "snap-0.pt").exists(), "nothing to delete; the premise is wrong"
        (run_dir / "snap-0.pt").unlink()

        with pytest.raises(PoolError, match="snap-0.pt"):
            from reckoner.campaign import run

            run(run_dir, golden_config(), run_name="fixture", anchor=ANCHOR)
    finally:
        shutil.rmtree(run_dir.parent, ignore_errors=True)
