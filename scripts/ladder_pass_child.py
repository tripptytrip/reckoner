"""A ladder pass in its own process, so the kill point can be a **real** SIGKILL.

An exception raised inside `run_pass` leaves every line on disk complete, which
tests the resume logic and not the repair logic. A killed process can be stopped
mid-`write`, leaving a torn final line — the thing that actually happens, and the
only way to exercise the path that handles it.

A separate file rather than a `-c` string: twice in this project a blind
`str.replace` over embedded subprocess source silently changed the program being
run. Helpers prevent, documentation warns.

Prints `READY <n>` after each row so the parent can kill at a known offset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from reckoner.arms import GreedyHeuristic, RandomRewriter  # noqa: E402
from reckoner.config import Config  # noqa: E402
from reckoner.ladderpass import run_pass  # noqa: E402
from reckoner.pairedset import load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    problems = load(args.paired, repo=REPO)
    written = [0]

    def announce(arm: str, i: int) -> None:
        written[0] += 1
        print(f"READY {written[0]}", flush=True)

    run_pass(
        args.root,
        args.index,
        [GreedyHeuristic(), RandomRewriter()],
        problems,
        cfg,
        roles={"greedy": "baseline", "random": "baseline"},
        calibration_note="smoke pass, kill-point child",
        on_unit=announce,
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
