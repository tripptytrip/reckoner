"""Running a ladder pass: `pair_scores` from row one, resumable mid-pass.

Row one, not row last
---------------------
Every arm-problem outcome is appended **as it happens**. The alternative —
accumulate in memory, write the aggregate at the end — loses two things that
cannot be recovered: a pass killed at 90% leaves nothing, and a pass that
finishes leaves means that cannot be un-aggregated. The paired-difference
bootstrap consumes *pairs*; a run that stored only means has thrown away the
input to its own test of record.

Resume, and what a resume is allowed to assume
-----------------------------------------------
Restart reads the rows already on disk, rebuilds the set of finished
``(arm, problem_key)`` units, and skips them. Two details make that safe rather
than merely usual:

* **A torn final line is truncated, not parsed.** An append interrupted by a kill
  leaves a partial line; the file is repaired to its last complete row before
  anything reads it. Only the *final* line may be torn — a break anywhere else is
  corruption and raises.
* **Completion is a marker file, never a process check.** ``PASS-DONE`` is written
  after the last row lands. A waiter that greps for a running process concludes
  "done" from "not running", which is also what a crash looks like.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from reckoner.arms import plays
from reckoner.config import Config
from reckoner.episode import Problem, outcome_z
from reckoner.ladder import (
    CURRENCY_BUDGET,
    CURRENCY_Z,
    LadderError,
    PairScore,
    pair,
    problem_key_of,
)
from reckoner.logschema import (
    SCHEMA_ERA,
    SchemaError,
    append_row,
    ladder_fields,
    validate_ladder_row,
)

#: Written after the final row. Its presence is the only completion signal.
DONE_MARKER = "PASS-DONE"

#: What an arm is FOR. Distinct from its currency: a baseline and the subject can
#: share a currency and must never share an interpretation.
LADDER_ROLES = frozenset({"baseline", "rung", "subject"})


class PassError(RuntimeError):
    """A pass that cannot be run or resumed honestly."""


@dataclass
class PassPaths:
    root: Path
    index: int

    @property
    def directory(self) -> Path:
        return self.root / f"pass-{self.index:04d}"

    @property
    def scores(self) -> Path:
        return self.directory / "pair_scores.jsonl"

    @property
    def marker(self) -> Path:
        return self.directory / DONE_MARKER


def is_complete(root: Path, index: int) -> bool:
    """**The** completion test. A marker on disk, not an absent process."""
    return PassPaths(root, index).marker.exists()


def repair_torn_tail(path: Path) -> int:
    """Truncate an interrupted final line. Returns the number of bytes dropped.

    A mid-file break is *not* repaired: it means something other than a kill
    happened, and silently dropping the rest would turn corruption into a shorter
    pass that looks complete.
    """
    if not path.exists():
        return 0
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        for i, line in enumerate(raw.splitlines()):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PassError(f"{path}:{i + 1} is not a complete row: {exc}") from exc
        return 0
    keep = raw.rfind(b"\n") + 1
    dropped = len(raw) - keep
    with path.open("rb+") as fh:
        fh.truncate(keep)
        fh.flush()
        os.fsync(fh.fileno())
    return dropped


def read_pair_scores(root: Path, index: int) -> list[dict]:
    """Rows already written for this pass, each validated against its OWN currency."""
    paths = PassPaths(root, index)
    if not paths.scores.exists():
        return []
    repair_torn_tail(paths.scores)
    rows = [json.loads(line) for line in paths.scores.read_text().splitlines() if line.strip()]
    # Validated per-row against the currency the ROW declares, not per-file against
    # one field set: a pass legitimately mixes z rows and solve-vs-budget rows, and
    # validating the file under either one would refuse half of a correct pass.
    for i, row in enumerate(rows):
        try:
            validate_ladder_row(row)
        except SchemaError as exc:
            raise PassError(f"{paths.scores}:{i + 1}: {exc}") from exc
    return rows


def finished_units(rows: list[dict]) -> set[tuple[str, str]]:
    return {(row["arm"], row["problem_key"]) for row in rows}


def pair_scores_of(rows: list[dict], arm: str) -> list[PairScore]:
    """Re-hydrate one arm's rows into the arithmetic layer's type.

    z rows score their ``z``; budget rows score ``solved`` as 1.0/0.0. The
    currency travels with the score, so :func:`reckoner.ladder.pair` can refuse a
    cross-currency comparison rather than compute a unitless difference.
    """
    out = []
    for row in rows:
        if row["arm"] != arm:
            continue
        if row["currency"] == CURRENCY_Z:
            score, steps = float(row["z"]), int(row["steps"])
        else:
            score, steps = (1.0 if row["solved"] else 0.0), int(row["steps_used"])
        out.append(
            PairScore(row["problem_key"], arm, row["currency"], score, steps, int(row["seed"]))
        )
    return out


def _row_for(
    *,
    arm,
    problem: Problem,
    result,
    index: int,
    seed: int,
    role: str,
    calibration_note: str,
    budget: int,
) -> dict:
    common = {
        "pass_index": index,
        "schema_era": SCHEMA_ERA,
        "arm": arm.name,
        "problem_key": problem_key_of(problem),
        "currency": arm.currency,
        "role": role,
        "nondeterministic": bool(arm.nondeterministic),
        "seed": seed,
        "calibration_note": calibration_note,
    }
    if arm.currency == CURRENCY_Z:
        return common | {
            "z": outcome_z(solved=result.solved, steps=result.steps, par=problem.par),
            "steps": int(result.steps),
            "par": int(problem.par),
            "par_source": problem.par_source,
        }
    if arm.currency == CURRENCY_BUDGET:
        return common | {
            "solved": bool(result.solved),
            "steps_used": int(result.steps),
            "budget": int(budget),
            "cas_version": getattr(arm, "version", "absent"),
        }
    raise PassError(f"{arm.name} declares currency {arm.currency!r}, which has no field set")


def run_pass(
    root: Path,
    index: int,
    arms: list,
    problems: list[Problem],
    cfg: Config,
    *,
    roles: dict[str, str],
    calibration_note: str,
    seed: int = 0,
    on_unit=None,
) -> dict:
    """Play every arm over every problem, writing each outcome as it lands.

    ``on_unit(arm_name, i)`` is called before each unit — the kill point the
    resume test uses, and the only reason it exists.
    """
    unknown = sorted(set(roles.values()) - LADDER_ROLES)
    if unknown:
        raise PassError(f"unknown ladder role(s) {unknown}; known: {sorted(LADDER_ROLES)}")
    missing = sorted({a.name for a in arms} - set(roles))
    if missing:
        raise PassError(
            f"no role declared for {missing}. What an arm is FOR is not inferable "
            "from what it scores — a baseline and the subject can share a currency."
        )
    if is_complete(root, index):
        raise PassError(
            f"pass {index} is already marked {DONE_MARKER}. Re-running would append "
            "a second copy of every row to a file the bootstrap reads as pairs."
        )

    paths = PassPaths(root, index)
    paths.directory.mkdir(parents=True, exist_ok=True)
    existing = read_pair_scores(root, index)
    done = finished_units(existing)
    started = time.perf_counter()
    seconds_by_arm: dict[str, float] = {}
    written = 0

    skipped: dict[str, int] = {}
    for arm in arms:
        arm_started = time.perf_counter()
        for i, problem in enumerate(problems):
            if not plays(arm, problem):
                # A DECLARED skip, counted. Playing a goal the rung cannot express
                # and recording the failure would report it as weak for a reason
                # unrelated to its skill — and would quietly make the paired set
                # two different sets.
                skipped[arm.name] = skipped.get(arm.name, 0) + 1
                continue
            key = problem_key_of(problem)
            if (arm.name, key) in done:
                continue
            if on_unit is not None:
                on_unit(arm.name, i)
            derived = seed * 1_000_003 + i
            result = arm.play(problem, cfg, derived)
            append_row(
                paths.scores,
                _row_for(
                    arm=arm,
                    problem=problem,
                    result=result,
                    index=index,
                    seed=derived,
                    role=roles[arm.name],
                    calibration_note=calibration_note,
                    budget=cfg.episode.step_cap,
                ),
                ladder_fields(arm.currency),
            )
            written += 1
        seconds_by_arm[arm.name] = round(time.perf_counter() - arm_started, 3)

    rows = read_pair_scores(root, index)
    # Completeness is per-arm against that arm's OWN playable subset. A single
    # arms x problems product would count a declared skip as a missing row and a
    # missing row as a declared skip — the two facts this pass keeps apart.
    counted = {a.name: 0 for a in arms}
    for row in rows:
        counted[row["arm"]] = counted.get(row["arm"], 0) + 1
    for arm in arms:
        expected = sum(1 for p in problems if plays(arm, p))
        if counted[arm.name] != expected:
            raise PassError(
                f"pass {index}: {arm.name} has {counted[arm.name]} rows for "
                f"{expected} playable problems. A short arm is an arm scored on a "
                "different problem set than the one it is being compared against."
            )
    paths.marker.write_text(
        json.dumps(
            {
                "pass_index": index,
                "rows": len(rows),
                "arms": [a.name for a in arms],
                "problems": len(problems),
                "rows_by_arm": counted,
                "skipped_by_arm": skipped,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "pass_index": index,
        "rows": len(rows),
        "rows_by_arm": counted,
        "skipped_by_arm": skipped,
        "rows_written_this_attempt": written,
        "rows_resumed": len(rows) - written,
        "seconds_total": round(time.perf_counter() - started, 3),
        "seconds_by_arm": seconds_by_arm,
    }


def comparison_from_pass(root: Path, index: int, arm_a: str, arm_b: str):
    """The pass's rows, paired. Refuses across currencies via :func:`ladder.pair`."""
    rows = read_pair_scores(root, index)
    a, b = pair_scores_of(rows, arm_a), pair_scores_of(rows, arm_b)
    if not a or not b:
        raise LadderError(f"pass {index} has no rows for {arm_a if not a else arm_b}")
    return pair(a, b)


__all__ = [
    "DONE_MARKER",
    "LADDER_ROLES",
    "PassError",
    "PassPaths",
    "comparison_from_pass",
    "finished_units",
    "is_complete",
    "pair_scores_of",
    "read_pair_scores",
    "repair_torn_tail",
    "run_pass",
]
