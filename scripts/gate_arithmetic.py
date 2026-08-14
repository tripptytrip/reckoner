"""Chunk 7's depth-1 gate arithmetic, on the real suite.

**The pre-registered re-check.** Chunk 2 measured branching on disclosed
stand-in samplers, because no generator existed; chunk 5's real problems now do.
This is where that promise is kept — the institutionalised chess lesson (a gate
whose sims-vs-branching arithmetic was never checked before it shipped),
executed rather than remembered.

The question the gate turns on: with a *uniform stub* policy, Gumbel-AlphaZero's
root considers `m` sampled actions out of `B` legal ones. To be **certain** of
considering the single winning action, `m >= B`. Below that the gate is
probabilistic, and the bound has to be stated rather than hoped for: with a
uniform prior and Gumbel top-m sampling without replacement, the winning action
is considered with probability `m / B`, and a suite of `n` problems all solved
requires that to hold `n` times.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from reckoner.dataset import read_suite, suite_problem
from reckoner.rules import legal_actions
from reckoner.vocab import GOAL_EVALUATE, GOAL_SIMPLIFY, GOAL_SOLVE

REPO = Path(__file__).resolve().parents[1]
NAMES = {GOAL_SOLVE: "SOLVE", GOAL_EVALUATE: "EVALUATE", GOAL_SIMPLIFY: "SIMPLIFY"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "gate_arithmetic.json")
    args = parser.parse_args()

    rows = read_suite(REPO / "runs" / "suites" / f"solve_in_{args.depth}.jsonl")
    by_goal: dict[str, list[int]] = {}
    for row in rows:
        problem = suite_problem(row)
        by_goal.setdefault(NAMES[problem.goal], []).append(len(legal_actions(problem.expr)))

    print(f"  legal-action count on solve_in_{args.depth} ({len(rows)} problems)\n")
    print(f"  {'goal':<10} {'n':>5} {'min':>5} {'median':>7} {'p99':>5} {'B_max':>6}")
    print("  " + "-" * 46)
    overall: list[int] = []
    record: dict = {"depth": args.depth, "problems": len(rows), "by_goal": {}}
    for goal, counts in sorted(by_goal.items()):
        counts.sort()
        overall += counts
        entry = {
            "n": len(counts),
            "min": counts[0],
            "median": counts[len(counts) // 2],
            "p99": counts[min(len(counts) - 1, int(0.99 * len(counts)))],
            "max": counts[-1],
        }
        record["by_goal"][goal] = entry
        print(
            f"  {goal:<10} {entry['n']:>5} {entry['min']:>5} {entry['median']:>7} "
            f"{entry['p99']:>5} {entry['max']:>6}"
        )
    overall.sort()
    b_max = overall[-1]
    record["B_max"] = b_max
    record["histogram"] = dict(sorted(Counter(overall).items()))
    print("  " + "-" * 46)
    print(
        f"  {'ALL':<10} {len(overall):>5} {overall[0]:>5} {overall[len(overall) // 2]:>7} "
        f"{overall[min(len(overall) - 1, int(0.99 * len(overall)))]:>5} {b_max:>6}"
    )

    print(f"\n  B_max = {b_max}\n")
    # P(sweep) is the PRODUCT over problems of min(1, m/B), so the exponent is
    # the count of problems where B > m — NOT the suite size. Most depth-1
    # problems have exactly one legal action and are solved with certainty at any
    # m; raising the worst-case probability to the suite size counts them as
    # trials they never were. (Erratum, corrected in chunk 8.)
    print(f"  {'m':>5} {'certain?':<10} {'worst-case P':<14} {'nontrivial':<12} {'P(sweep)'}")
    print("  " + "-" * 68)
    for m in (3, 5, 12, 16, 24, b_max):
        certain = m >= b_max
        nontrivial = sum(1 for b in overall if b > m)
        p_sweep = 1.0
        for b in overall:
            p_sweep *= min(1.0, m / b)
        worst = 1.0 if certain else m / b_max
        record.setdefault("m_table", {})[str(m)] = {
            "certain": certain,
            "p_worst": worst,
            "nontrivial_trials": nontrivial,
            "p_sweep": p_sweep,
        }
        print(
            f"  {m:>5} {'YES' if certain else 'no':<10} {worst:<14.4f} "
            f"{nontrivial:<12} {p_sweep:.3e}"
        )

    print(f"\n  ==> the depth-{args.depth} gate is reachable with certainty only at m >= {b_max}.")
    print("      Below that the gate is probabilistic and its bound is stated above;")
    print("      a 100% gate at m < B_max would be a gate the arithmetic cannot support.")
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    try:
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out
    print(f"  wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
