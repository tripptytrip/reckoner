"""Per-rule participation over BFS-optimal derivations.

Three instruments in one, which is why it is worth the BFS:

1. **Rule liveness.** A rule near 0% is dead weight regardless of how green its
   soundness fuzz is — the fuzz proves a rule is *correct*, never that it is
   *used*. `eval_sub` exists only because the generator emits numeric SUB on
   purpose; this is what checks that the purpose was served.
2. **Evidence for ROUND-01.** `add_both_sides` is claimed reachability-redundant.
   If it never appears in an optimal derivation across the whole suite set, that
   is the suite-level evidence the round asks for — three fixtures are not a
   universal.
3. **What the policy has to learn.** The distribution over (rule, depth) is the
   action distribution a warm start is fitting, before any training exists.

**Participation is measured on the derivation BFS returned**, which is *an*
optimal derivation, not all of them. So these counts are a lower bound on "could
appear in some optimal derivation" — stated because the difference matters for
instrument 2: a rule at 0 here has not been proven unusable, only unused by the
paths BFS happened to find first.

    python scripts/rule_participation.py --suites
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from reckoner.config import Config
from reckoner.dataset import read_dataset, read_suite, suite_problem
from reckoner.episode import Problem, bfs_solution, decode_state
from reckoner.rules import RULES
from reckoner.vocab import GOAL_EVALUATE, GOAL_SIMPLIFY, GOAL_SOLVE

REPO = Path(__file__).resolve().parents[1]
CFG = Config()
GOAL_NAMES = {GOAL_SOLVE: "SOLVE", GOAL_EVALUATE: "EVALUATE", GOAL_SIMPLIFY: "SIMPLIFY"}


def participation(problems: list[Problem]) -> tuple[Counter, Counter, int]:
    """``(rule -> problems using it, (depth, rule) -> count, problems measured)``."""
    per_rule: Counter = Counter()
    per_depth_rule: Counter = Counter()
    measured = 0
    for problem in problems:
        path = bfs_solution(problem, CFG)
        if path is None:
            continue
        measured += 1
        used = {action[0] for action, _state in path}
        for rule_id in used:
            per_rule[RULES[rule_id].name] += 1
            per_depth_rule[(problem.par, RULES[rule_id].name)] += 1
    return per_rule, per_depth_rule, measured


def report(per_rule: Counter, per_depth_rule: Counter, measured: int, depths: list[int]) -> None:
    print(f"\n  rule participation over {measured} BFS-optimal derivations")
    print(f"  {'rule':<20} {'problems':>9} {'share':>8}   by depth")
    print("  " + "-" * 78)
    for rule in RULES:
        count = per_rule[rule.name]
        by_depth = {
            d: per_depth_rule[(d, rule.name)] for d in depths if per_depth_rule[(d, rule.name)]
        }
        flag = "   <-- NEVER USED" if count == 0 else ""
        print(
            f"  {rule.name:<20} {count:>9} {100 * count / max(1, measured):>7.1f}%   {by_depth}{flag}"
        )

    dead = [r.name for r in RULES if per_rule[r.name] == 0]
    print(f"\n  rules never appearing in an optimal derivation: {dead or 'none'}")
    print("  (a lower bound: participation is measured on the derivation BFS returned,")
    print("   which is *an* optimal derivation, not all of them)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", action="store_true")
    parser.add_argument("--train-sample", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    problems: list[Problem] = []
    if args.suites:
        for depth in range(1, 7):
            path = REPO / "runs" / "suites" / f"solve_in_{depth}.jsonl"
            if path.exists():
                rows = read_suite(path)
                problems += [suite_problem(r) for r in (rows if depth <= 4 else rows[:60])]
    if args.train_sample:
        dataset = read_dataset(REPO / "runs" / "data" / "train_100k")
        rng = random.Random(20260814)
        for i in rng.sample(range(len(dataset)), min(args.train_sample, len(dataset))):
            goal, target, expr = decode_state(dataset.state(i))
            problems.append(
                Problem(
                    goal=goal,
                    expr=expr,
                    par=int(dataset.par[i]),
                    target=target,
                    par_source="bfs",
                )
            )

    if not problems:
        raise SystemExit("nothing to measure — pass --suites and/or --train-sample")

    goals = Counter(GOAL_NAMES[p.goal] for p in problems)
    depths = sorted({p.par for p in problems if p.par is not None})
    print(f"  measuring {len(problems)} problems, goals {dict(goals)}, depths {depths}")

    per_rule, per_depth_rule, measured = participation(problems)
    report(per_rule, per_depth_rule, measured, depths)

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "measured": measured,
                    "per_rule": dict(per_rule),
                    "per_depth_rule": {f"{d}:{r}": c for (d, r), c in per_depth_rule.items()},
                    "note": "lower bound — one optimal derivation per problem, not all",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
