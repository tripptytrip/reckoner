"""BFS-exact vs scripted par, where both labels exist.

Calibration for the "provisional floor" language in spec §3: scripted par is a
floor, and this is how far above the true minimum that floor sits. Free to
compute on the depth <= 6 overlap, because both labels are available there — and
it is the only evidence that will exist for the mid-strata until a round adds
one, since above the BFS horizon there is nothing to compare against.

`FINDINGS.md` F-01 was the first data point, found by hand in a proofread: one
step at depth 3. This is the distribution.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.solver import scripted_par_delta
from reckoner.vocab import GOAL_EVALUATE, GOAL_SIMPLIFY, GOAL_SOLVE

REPO = Path(__file__).resolve().parents[1]
CFG = Config()
NAMES = {GOAL_SOLVE: "SOLVE", GOAL_EVALUATE: "EVALUATE", GOAL_SIMPLIFY: "SIMPLIFY"}


def _delta(row: dict) -> tuple[int, str, int | None]:
    problem = suite_problem(row)
    return row["depth"], NAMES[row["goal"]], scripted_par_delta(problem, CFG)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "par_delta.json")
    args = parser.parse_args()

    rows = []
    for depth in range(1, 7):
        path = REPO / "runs" / "suites" / f"solve_in_{depth}.jsonl"
        if path.exists():
            rows += read_suite(path)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_delta, rows, chunksize=8))

    by_depth: dict[int, Counter] = {}
    by_goal: dict[str, Counter] = {}
    unsolved: Counter = Counter()
    for depth, goal, delta in results:
        if delta is None:
            unsolved[(depth, goal)] += 1
            continue
        by_depth.setdefault(depth, Counter())[delta] += 1
        by_goal.setdefault(goal, Counter())[delta] += 1

    print(f"  {'depth':>6} {'n':>5} {'delta -> count':<28} {'optimal':>9} {'mean':>7}")
    print("  " + "-" * 62)
    for depth in sorted(by_depth):
        c = by_depth[depth]
        n = sum(c.values())
        mean = sum(d * k for d, k in c.items()) / n
        print(
            f"  {depth:>6} {n:>5} {str(dict(sorted(c.items()))):<28} "
            f"{100 * c[0] / n:>8.1f}% {mean:>7.2f}"
        )
    print()
    for goal in sorted(by_goal):
        c = by_goal[goal]
        n = sum(c.values())
        print(
            f"  {goal:<10} n={n:<5} optimal {100 * c[0] / n:>5.1f}%  deltas {dict(sorted(c.items()))}"
        )
    if unsolved:
        print(f"\n  scripted solver failed to solve: {dict(unsolved)}")

    args.out.write_text(
        json.dumps(
            {
                "by_depth": {str(d): dict(c) for d, c in by_depth.items()},
                "by_goal": {g: dict(c) for g, c in by_goal.items()},
                "unsolved": {f"{d}:{g}": n for (d, g), n in unsolved.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\n  wrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
