"""How large does a *reachable* state get? The measurement that sizes the model.

**Start states are the wrong distribution.** Episodes grow before they shrink:
`sub_both_sides` adds a negation term that `combine_like_terms` only removes a
step later, and `add_both_sides` inflates both sides without bound. The longest
state inside an episode strictly exceeds the longest problem, so sizing
`seq_len` and `max_sites` from the dataset would guarantee overflow the first
time search explored.

Three populations, because each answers a different question:

  * **optimal derivations** — every intermediate state of all 1,200 suite
    derivations. What a well-played episode actually visits.
  * **random walks** — capped random legal play from the deepest stratum. What
    *search* visits, which is worse: search explores badly before it explores
    well, and `add_both_sides` is always available.
  * **start states** — reported only to show the gap.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.episode import bfs_solution, encode_state
from reckoner.rules import enumerate_sites, successors

REPO = Path(__file__).resolve().parents[1]
CFG = Config()


def _extent(problem, expr) -> tuple[int, int]:
    return len(encode_state(problem.goal, expr, problem.target)), len(enumerate_sites(expr))


def _optimal(row: dict) -> list[tuple[int, int]]:
    problem = suite_problem(row)
    path = bfs_solution(problem, CFG)
    states = [problem.expr, *(s for _a, s in path or [])]
    return [_extent(problem, s) for s in states]


def _walk(args: tuple[dict, int, int]) -> list[tuple[int, int]]:
    row, seed, steps = args
    problem = suite_problem(row)
    rng = random.Random(seed)
    expr = problem.expr
    out = [_extent(problem, expr)]
    for _ in range(steps):
        options = successors(expr)
        if not options:
            break
        expr = rng.choice(options)[1]
        out.append(_extent(problem, expr))
    return out


def summarise(name: str, samples: list[tuple[int, int]]) -> dict:
    toks = sorted(s[0] for s in samples)
    sites = sorted(s[1] for s in samples)
    pick = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    row = {
        "population": name,
        "states": len(samples),
        "tokens_p50": pick(toks, 0.5),
        "tokens_p99": pick(toks, 0.99),
        "tokens_p100": toks[-1],
        "sites_p50": pick(sites, 0.5),
        "sites_p99": pick(sites, 0.99),
        "sites_p100": sites[-1],
    }
    print(
        f"  {name:<24} {row['states']:>8}  tokens p50/p99/p100 "
        f"{row['tokens_p50']:>4}/{row['tokens_p99']:>4}/{row['tokens_p100']:>4}   "
        f"sites {row['sites_p50']:>3}/{row['sites_p99']:>3}/{row['sites_p100']:>3}"
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--walk-steps", type=int, default=None)
    parser.add_argument("--walks-per-problem", type=int, default=4)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "state_extent.json")
    args = parser.parse_args()
    steps = args.walk_steps or CFG.episode.step_cap

    rows = []
    for depth in range(1, 7):
        path = REPO / "runs" / "suites" / f"solve_in_{depth}.jsonl"
        if path.exists():
            rows += read_suite(path)

    print(f"  {'population':<24} {'states':>8}  extents\n  " + "-" * 84)
    starts = [_extent(suite_problem(r), suite_problem(r).expr) for r in rows]
    record = {"start_states": summarise("start states", starts)}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        optimal = [e for group in pool.map(_optimal, rows, chunksize=8) for e in group]
    record["optimal_derivations"] = summarise("optimal derivations", optimal)

    deep = [r for r in rows if r["depth"] >= 5]
    work = [
        (r, 9000 + i * 31 + j, steps)
        for i, r in enumerate(deep)
        for j in range(args.walks_per_problem)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        walks = [e for group in pool.map(_walk, work, chunksize=8) for e in group]
    record["random_walks_deep"] = summarise(f"random walks (cap {steps})", walks)

    worst_tokens = max(r["tokens_p100"] for r in record.values())
    worst_sites = max(r["sites_p100"] for r in record.values())
    print(f"\n  measured p100 over all populations: {worst_tokens} tokens, {worst_sites} sites")
    record["measured_max"] = {"tokens": worst_tokens, "sites": worst_sites}
    record["walk_steps"] = steps

    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
