"""Unroll BFS-optimal derivations into per-state supervision examples.

One training example per *state* along the optimal derivation, not one per
problem:

  * **policy target** — the action the derivation took from that state
  * **steps target** — remaining steps to the solve (so the last state before
    the solve has target 1, and the start state has target = par)
  * **depth** — the problem's par, carried so every metric can be stratified

**Recorded caveat, accepted for a warm start.** A single BFS path imprints BFS's
tie-break and site-enumeration order onto the policy: where several optimal
derivations exist, the network is taught the one BFS happened to reach first, as
if the others were wrong. Phase 2's improved-policy targets wash this out — they
are visit distributions over the whole legal set, not a one-hot — but until then
the warm start carries it, and a Phase-1 top-k number is partly measuring
agreement with an enumeration order.

The problem-level split is the split. This script never mixes states from one
problem across sets, because two states from one derivation are not independent
samples and a state-level split would leak the answer.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from reckoner.config import Config
from reckoner.dataset import git_sha, sha256_file
from reckoner.episode import Problem, bfs_solution, decode_state, encode_state
from reckoner.model import action_index
from reckoner.rules import RULESET_VERSION
from reckoner.vocab import PAD, VOCAB_VERSION

REPO = Path(__file__).resolve().parents[1]
CFG = Config()


def _unroll(
    args: tuple[list[int], int, int | None, int],
) -> list[tuple[tuple[int, ...], int, int, int, int]]:
    """``(state_tokens, action_index, remaining_steps, depth, goal)`` per state."""
    tokens, goal, target, par = args
    _goal, _target, expr = decode_state(tuple(tokens))
    problem = Problem(goal=goal, expr=expr, par=par, target=target, par_source="bfs")
    path = bfs_solution(problem, CFG)
    if path is None or len(path) != par:
        return []
    out = []
    state = problem.expr
    for i, ((rule_id, site_id), successor) in enumerate(path):
        out.append(
            (
                encode_state(goal, state, target),
                action_index(rule_id, site_id, CFG.model.max_sites),
                par - i,  # remaining steps, so the last state before solving is 1
                par,
                goal,
            )
        )
        state = successor
    return out


def build(source: Path, out: Path, workers: int, limit: int | None, seed: int = 0) -> dict:
    """Build the supervision set. ``limit`` **samples**, it does not take a prefix.

    The generated dataset is laid out stratum by stratum, so a prefix of 2,000
    rows is 2,000 depth-1-and-2 problems — which is exactly the mistake F-03
    records: a pilot that measures a distribution the real run will not see. A
    seeded sample across the whole set costs nothing and is representative.
    """
    import random as _random

    from reckoner.dataset import read_dataset

    dataset = read_dataset(source)
    indices = list(range(len(dataset)))
    if limit is not None and limit < len(indices):
        indices = sorted(_random.Random(seed).sample(indices, limit))
    n = len(indices)
    work = [
        (
            list(dataset.state(i)),
            int(dataset.goal[i]),
            None if int(dataset.target[i]) < 0 else int(dataset.target[i]),
            int(dataset.par[i]),
        )
        for i in indices
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        groups = list(pool.map(_unroll, work, chunksize=32))

    examples = [e for group in groups for e in group]
    if not examples:
        raise SystemExit("no examples produced")
    dropped = sum(1 for g in groups if not g)

    max_len = max(len(e[0]) for e in examples)
    count = len(examples)
    arrays = {
        "tokens": np.full((count, max_len), PAD, dtype=np.int32),
        "lengths": np.zeros(count, dtype=np.int32),
        "action": np.zeros(count, dtype=np.int32),
        "steps_remaining": np.zeros(count, dtype=np.int32),
        "depth": np.zeros(count, dtype=np.int32),
        "goal": np.zeros(count, dtype=np.int32),
    }
    for i, (seq, action, remaining, depth, goal) in enumerate(examples):
        arrays["tokens"][i, : len(seq)] = seq
        arrays["lengths"][i] = len(seq)
        arrays["action"][i] = action
        arrays["steps_remaining"][i] = remaining
        arrays["depth"][i] = depth
        arrays["goal"][i] = goal

    out.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        array.tofile(out / f"{name}.i32")

    meta = {
        "mode": "phase1_supervision",
        "source": str(source.relative_to(REPO)),
        "source_digests": json.loads((source / "meta.json").read_text())["digests"],
        "problems": n,
        "problems_dropped": dropped,
        "n": count,
        "max_len": max_len,
        "ruleset_version": RULESET_VERSION,
        "vocab_version": VOCAB_VERSION,
        "git_sha": git_sha(REPO),
        "depth_histogram": {
            int(d): int(c)
            for d, c in zip(*np.unique(arrays["depth"], return_counts=True), strict=True)
        },
        "goal_histogram": {
            int(g): int(c)
            for g, c in zip(*np.unique(arrays["goal"], return_counts=True), strict=True)
        },
        "digests": {name: sha256_file(out / f"{name}.i32") for name in arrays},
        "seconds": round(time.perf_counter() - started, 1),
        "caveat": (
            "policy targets are single BFS paths; where several optimal derivations "
            "exist this imprints BFS tie-break and site-enumeration order. Accepted "
            "for the warm start; Phase 2 improved-policy targets wash it out."
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "runs" / "data" / "train_100k")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "data" / "phase1_train")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="sample this many problems")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    meta = build(args.source, args.out, args.workers, args.limit, args.seed)
    print(f"  {meta['n']} examples from {meta['problems']} problems in {meta['seconds']}s")
    print(f"  dropped: {meta['problems_dropped']}")
    print(f"  depths:  {meta['depth_histogram']}")
    print(f"  goals:   {meta['goal_histogram']}")
    print(f"  max_len: {meta['max_len']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
