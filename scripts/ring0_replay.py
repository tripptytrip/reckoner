"""Job 1 — replay iteration 0 deterministically. **The queue's halt gate.**

Iteration 0 is exactly reproducible: the anchor in eval mode, a pool of exactly
one member, `training_problems(..., 400, seed=0)` and `Random(0 * 7919 + 13)`.
Nothing about it depends on anything the rehearsal did afterwards.

Three jobs in one:

1. **Regression check on the whole restoration round.** 398/400 solved, 99
   pool-par and 1,305 ring rows can only reproduce if none of the five fixes
   touched the episode path. A mismatch halts the queue — the remaining jobs
   would be measuring a loop that had quietly moved.
2. **F-28 answered exactly.** The pool's unavailability counters never reached a
   row, so whether `pool_par_fraction = 0.2475` was sampling or structure could
   only be bounded. Replaying the draw sequence counts it.
3. **ring-0 for the sweep**, so every arm trains on one fixed ring and `f` is the
   only thing that varies.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from reckoner.campaign import ANCHOR, campaign_problems
from reckoner.config import Config, validate
from reckoner.dataset import anchored_data, training_problems
from reckoner.evaluate import model_evaluator
from reckoner.model import load_checkpoint
from reckoner.pool import CheckpointPool
from reckoner.replay import ReplayRing
from reckoner.runner import run_iteration

REPO = Path(__file__).resolve().parents[1]

#: What iteration 0 recorded, from the rehearsal's own row.
EXPECTED = {"solved": 398, "episodes": 400, "from_pool": 99, "ring": 1305}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "ring0_replay")
    args = parser.parse_args()

    cfg = Config()
    validate(cfg)
    torch.set_num_threads(cfg.campaign.intra_op_threads)

    model, _ = load_checkpoint(ANCHOR, cfg)
    model.eval()

    pool = CheckpointPool(cfg)
    if cfg.league.seed_pool_with_anchor:
        pool.add(ANCHOR)

    problems, from_pool = campaign_problems(
        cfg, pool, model, cfg.campaign.episodes_per_iteration, seed=0
    )

    # F-28: the counters that never reached a row.
    stats_before = dict(pool.stats.as_dict())

    ring = ReplayRing(cfg.train.replay_capacity, cfg)
    stats = run_iteration(problems, model_evaluator(model, cfg, 0.0), cfg, ring, seed=0)
    stats.check_descent_identity()

    got = {
        "solved": stats.episodes_solved,
        "episodes": stats.episodes,
        "from_pool": from_pool,
        "ring": len(ring),
    }
    ok = got == EXPECTED

    # The draw sequence, recounted independently of the pool's own bookkeeping.
    rng = random.Random(0 * 7919 + 13)
    reference = training_problems(
        anchored_data("train_100k"), cfg.campaign.episodes_per_iteration, seed=0
    )
    drawn = sum(rng.random() < cfg.league.par_from_pool_frac for _ in reference)

    print("\n  RING-0 REPLAY — the queue's halt gate\n")
    for key, want in EXPECTED.items():
        mark = "ok  " if got[key] == want else "FAIL"
        print(f"    {mark} {key:>10}: {got[key]:>5}  (rehearsal recorded {want})")

    print("\n  F-28 — was pool_par_fraction = 0.2475 sampling or structure?\n")
    print(f"    draws attempted (Bernoulli 0.2 over 400) : {drawn}")
    print(f"    par_source == 'pool'                     : {from_pool}")
    print(f"    fallbacks (drawn but not pool par)       : {drawn - from_pool}")
    for name in ("pool_par_unavailable_empty", "pool_par_unavailable_capped", "pool_par_solved"):
        print(f"    {name:<41}: {pool.stats.as_dict()[name]}")

    args.out.mkdir(parents=True, exist_ok=True)
    ring.save(args.out / "ring-0")
    (args.out / "replay.json").write_text(
        json.dumps(
            {
                "expected": EXPECTED,
                "got": got,
                "reproduced": ok,
                "draws_attempted": drawn,
                "fallbacks": drawn - from_pool,
                "pool_stats": pool.stats.as_dict(),
                "pool_stats_at_draw": stats_before,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\n  ring-0 saved to {args.out / 'ring-0'}")
    if not ok:
        print("\n  HALT — iteration 0 did not reproduce. One of the five fixes reached")
        print("  the episode path, and every downstream job would measure a moved loop.")
        return 1
    print("\n  REPRODUCED — the episode path is untouched by the restoration round.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
