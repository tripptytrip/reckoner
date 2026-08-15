"""Crash-resume: one atomic commit point, and everything else is provisional.

The write-ordering contract
---------------------------
**The iteration is the atomic unit, and `LATEST` is the only thing that makes one
real.** Per iteration *n*, in this order:

1. ``ring-<n>/``      — the replay ring, written to a *fresh* directory (never
                        overwriting a committed one), its own ``meta.json`` last
2. ``state-<n>.json`` — iteration counter, value-head declaration, seed
3. one row appended to ``iterations.jsonl``
4. ``LATEST``         — written via temp-file + ``os.replace``, which is atomic

Steps 1–3 are **provisional**: a process killed anywhere in them leaves artifacts
for an iteration that never happened, and resume ignores or removes them. Only
step 4 commits, and it is a single-file rename, which the filesystem gives us
atomically. There is no window in which `LATEST` names a half-written iteration.

Why not "the row is the commit point"
--------------------------------------
Because the ring is written before the row and would then contain the killed
iteration's steps. Resuming would redo the iteration and append those steps
again, so the ring would carry each of them twice — silently, and only in runs
that crashed. A duplicated replay corpus is the kind of defect that shows up as a
mildly odd loss curve six iterations later.

The two kill points this is tested through
-------------------------------------------
* **A — after the ring and state, before the row.** Resume finds no committed
  row and redoes *n*; the orphaned ``ring-<n>``/``state-<n>`` are ignored.
* **B — after the row, before ``LATEST``.** Resume finds a row for an
  uncommitted iteration and **truncates it**, then redoes *n*.

Both must land on state identical to an uninterrupted run — asserted, not
asserted-about.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from reckoner.config import Config
from reckoner.replay import ReplayRing
from reckoner.valuegate import ValueHeadState


class ResumeError(RuntimeError):
    """A run directory that cannot be interpreted."""


@dataclass
class RunState:
    """What a resumed run needs to continue as though it had not stopped."""

    iteration: int
    value_head: ValueHeadState
    seed: int

    def as_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "value_head": self.value_head.as_dict(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> RunState:
        head = payload["value_head"]
        return cls(
            iteration=payload["iteration"],
            value_head=ValueHeadState(**head),
            seed=payload["seed"],
        )


def latest_committed(run: Path) -> int | None:
    """The last iteration that actually happened, or None for a fresh run."""
    marker = run / "LATEST"
    if not marker.exists():
        return None
    text = marker.read_text().strip()
    if not text.isdigit():
        raise ResumeError(f"{marker} is not an iteration number: {text!r}")
    return int(text)


def commit_iteration(
    run: Path,
    ring: ReplayRing,
    state: RunState,
    row_writer,
) -> None:
    """Write an iteration's artifacts, then commit. Order is the contract.

    ``row_writer`` is called to append the iteration row *before* the commit, so
    a crash between the two leaves a row that resume will truncate — which is
    kill point B, and is why the truncation exists.
    """
    n = state.iteration
    ring_dir = run / f"ring-{n}"
    if ring_dir.exists():
        shutil.rmtree(ring_dir)  # a provisional leftover from a killed attempt
    ring.save(ring_dir)
    (run / f"state-{n}.json").write_text(
        json.dumps(state.as_dict(), indent=2, sort_keys=True) + "\n"
    )

    row_writer()

    # The commit. Temp-file + os.replace is atomic: LATEST never names a
    # half-written iteration, and there is no ordering after it to get wrong.
    marker_tmp = run / "LATEST.tmp"
    marker_tmp.write_text(f"{n}\n")
    os.replace(marker_tmp, run / "LATEST")


def resume(run: Path, cfg: Config) -> tuple[int, ReplayRing | None, RunState | None]:
    """Return ``(next_iteration, ring, state)``, cleaning provisional artifacts.

    Truncates ``iterations.jsonl`` to the committed prefix. A row for an
    uncommitted iteration is not evidence of anything except a crash, and leaving
    it would make the log claim an iteration the ring does not contain.
    """
    committed = latest_committed(run)
    rows_path = run / "iterations.jsonl"

    if committed is None:
        if rows_path.exists():
            rows_path.unlink()
        _drop_provisional(run, keep=None)
        return 0, None, None

    ring = ReplayRing.load(run / f"ring-{committed}", cfg)
    state = RunState.from_dict(json.loads((run / f"state-{committed}.json").read_text()))

    if rows_path.exists():
        lines = [line for line in rows_path.read_text().splitlines() if line.strip()]
        keep = lines[: committed + 1]
        if len(lines) != len(keep):
            rows_path.write_text("".join(line + "\n" for line in keep))

    _drop_provisional(run, keep=committed)
    return committed + 1, ring, state


def _drop_provisional(run: Path, *, keep: int | None) -> None:
    """Remove artifacts for iterations that never committed.

    Left in place they are not merely clutter: a ``ring-7`` beside a ``LATEST``
    of 6 is an artifact that looks like history and is not.
    """
    for path in run.glob("ring-*"):
        n = path.name.removeprefix("ring-")
        if n.isdigit() and (keep is None or int(n) > keep):
            shutil.rmtree(path)
    for path in run.glob("state-*.json"):
        n = path.name.removeprefix("state-").removesuffix(".json")
        if n.isdigit() and (keep is None or int(n) > keep):
            path.unlink()
    (run / "LATEST.tmp").unlink(missing_ok=True)
