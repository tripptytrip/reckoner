"""Measure the branching factor of the rule engine.

Chunk 7's depth-1 gate arithmetic depends on this number, and the plan is
explicit that it gets measured now rather than then. Two numbers are reported,
because they answer different questions:

**raw** — ``len(legal_actions(state))``. This is what the policy head masks
over, so it sizes the action space the network has to score.

**effective** — the number of *distinct canonical successors*. This is what
search actually branches into, and it is smaller: canonicalisation can merge two
different ``(rule, site)`` actions into one state (``add_both_sides`` at a
duplicated addend, a rewrite that re-sorts into an existing form), and two
actions leading to the same node are one branch, not two.

**The distribution matters as much as the number, so it is disclosed.** There is
no problem generator before chunk 5, so every sampler here is a stand-in and is
named as one. If chunk 5's real distribution differs, the gate arithmetic gets
re-checked against it — the number below is provisional in exactly that sense.

    python scripts/measure_branching.py --states 4000
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from reckoner.expr import Expr, add, eq, identity_key, make_op, mul, num, var
from reckoner.rules import RULES, legal_actions, successors
from reckoner.vocab import ADD, DIV, MUL, SUB, VAR_X

REPO = Path(__file__).resolve().parents[1]
X = var(VAR_X)


# ---------------------------------------------------------------------------
# Samplers — each one disclosed, none of them the real thing
# ---------------------------------------------------------------------------


def sampler_arith(rng: random.Random, depth: int = 0) -> Expr:
    """Random arithmetic trees. Broad, but nothing like a problem."""
    if depth >= 3 or rng.random() < 0.4:
        return X if rng.random() < 0.25 else num(rng.randrange(-30, 31))
    kind = rng.choice((ADD, ADD, MUL, MUL, SUB, DIV))
    n = rng.randint(2, 4) if kind in (ADD, MUL) else 2
    return make_op(kind, [sampler_arith(rng, depth + 1) for _ in range(n)])


def sampler_linear(rng: random.Random) -> Expr:
    """Linear equations in the shapes chunk 5 is expected to generate."""
    a = rng.choice([c for c in range(-6, 7) if c != 0])
    shape = rng.choice(("ax_eq_c", "ax_b_eq_c", "ax_b_eq_cx_d"))
    if shape == "ax_eq_c":
        return eq(mul(num(a), X), num(a * rng.randrange(-9, 10)))
    if shape == "ax_b_eq_c":
        return eq(add(mul(num(a), X), num(rng.randrange(-20, 21))), num(rng.randrange(-40, 41)))
    b = rng.choice([c for c in range(-6, 7) if c != a])
    return eq(
        add(mul(num(a), X), num(rng.randrange(-20, 21))),
        add(mul(num(b), X), num(rng.randrange(-20, 21))),
    )


def sampler_derivation(rng: random.Random) -> Expr:
    """States reached *mid-derivation*, by walking a random legal path.

    The closest stand-in available before chunk 5: search spends most of its
    time on states that are several rewrites deep, not on pristine problems,
    and those look different — more addends, more numerals, more sites.
    """
    state = sampler_linear(rng)
    for _ in range(rng.randint(0, 4)):
        options = successors(state)
        if not options:
            break
        state = rng.choice(options)[1]
    return state


SAMPLERS: dict[str, Callable[[random.Random], Expr]] = {
    "arith": sampler_arith,
    "linear": sampler_linear,
    "derivation": sampler_derivation,
}


# ---------------------------------------------------------------------------


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def measure(name: str, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    sample = SAMPLERS[name]
    raw: list[int] = []
    effective: list[int] = []
    per_rule = dict.fromkeys((r.name for r in RULES), 0)
    terminal = 0
    t0 = time.perf_counter()

    for _ in range(n):
        state = sample(rng)
        actions = legal_actions(state)
        raw.append(len(actions))
        for rule_id, _ in actions:
            per_rule[RULES[rule_id].name] += 1
        if not actions:
            terminal += 1
            effective.append(0)
            continue
        effective.append(len({identity_key(s) for _, s in successors(state)}))

    live_raw = [v for v in raw if v]
    live_eff = [v for v in effective if v]
    return {
        "sampler": name,
        "states": n,
        "terminal_states": terminal,
        "raw_median": percentile(live_raw, 0.5),
        "raw_p90": percentile(live_raw, 0.9),
        "raw_max": max(raw) if raw else 0,
        "raw_mean": round(sum(raw) / len(raw), 2),
        "effective_median": percentile(live_eff, 0.5),
        "effective_p90": percentile(live_eff, 0.9),
        "effective_max": max(effective) if effective else 0,
        "effective_mean": round(sum(effective) / len(effective), 2),
        "merge_rate": round(1 - sum(effective) / max(1, sum(raw)), 4),
        "actions_by_rule": per_rule,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--record",
        action="store_true",
        help="append the rows to benchmarks/results.jsonl with the git SHA",
    )
    args = parser.parse_args()

    rows = [measure(name, args.states, args.seed) for name in SAMPLERS]

    header = f"{'sampler':<12} {'raw med':>8} {'raw p90':>8} {'raw max':>8} {'eff med':>8} {'eff p90':>8} {'eff max':>8} {'merged':>7} {'term':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['sampler']:<12} {row['raw_median']:>8} {row['raw_p90']:>8} {row['raw_max']:>8} "
            f"{row['effective_median']:>8} {row['effective_p90']:>8} {row['effective_max']:>8} "
            f"{row['merge_rate'] * 100:>6.1f}% {row['terminal_states']:>6}"
        )
    print("\nactions by rule:")
    for row in rows:
        print(f"  {row['sampler']:<12} {row['actions_by_rule']}")
    print(
        "\nSamplers are stand-ins: there is no problem generator before chunk 5.\n"
        "Re-check chunk 7's gate arithmetic against chunk 5's real distribution."
    )

    if args.record:
        path = REPO / "benchmarks" / "results.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        sha = git_sha()
        with path.open("a") as fh:
            for row in rows:
                fh.write(json.dumps({"bench": "branching", "git_sha": sha, **row}) + "\n")
        print(f"\nappended {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
