"""Gate 10b's top-1, across the rehearsal's checkpoints. **The discriminator.**

The rehearsal's cadence unit found at-par on the suites down from 99.4% to 75.8%
at ``sims = 48``. Two stories fit that, and they have opposite consequences:

* **the model forgot** — the supervised warm start has been trained away, and the
  training schedule is the subject;
* **the search path degraded** — the policy is intact and something between the
  weights and the played move is losing it.

Gate 10b separates them, because it reads the policy *directly*: top-1 rule-site
on depth ≤ 3 of the F-09 unseen held-out subset, **no search involved.** The
anchor measured **0.9699** (`GATE-chunk8-VERDICT.md`, `phase1_result.json`).

If top-1 has collapsed, the first story is true and nothing about the search is
implicated. If top-1 is intact at 0.97 while at-par fell to 75.8%, the second is
true and nothing about learning is.

Reuses `train_phase1`'s own `top_k_by_depth` and `shared_state_indices` rather
than reimplementing them: one keying serves the gate, the census and this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reckoner.config import Config, validate
from reckoner.dataset import anchored_data
from reckoner.model import load_checkpoint
from reckoner.train import SupervisionSet

REPO = Path(__file__).resolve().parents[1]
ANCHOR_TOP1 = 0.9699  # GATE-chunk8-VERDICT.md, the frozen reading


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "gate10b_trajectory.json")
    args = parser.parse_args()

    from scripts.train_phase1 import shared_state_indices, top_k_by_depth

    cfg = Config()
    validate(cfg)
    torch.set_num_threads(cfg.campaign.intra_op_threads)

    eval_path = anchored_data("phase1_eval")
    held = SupervisionSet(eval_path)
    shared = shared_state_indices(eval_path)
    print(f"\n  gate 10b instrument: {len(held):,} states, {len(shared):,} shared excluded")
    print(f"  anchor's frozen reading: top-1 = {ANCHOR_TOP1}\n")

    rows = []
    for path in args.checkpoints:
        if not path.exists():
            print(f"    {path.name:>14}  ABSENT")
            continue
        model, meta = load_checkpoint(path, cfg)
        model.eval()  # F-22: the instrument reads the policy, not dropout
        clean = top_k_by_depth(model, held, cfg, device="cpu", exclude=shared)
        top1 = clean["depth_le_3_top1"]
        top8 = clean["depth_le_3_top8"]
        rows.append(
            {
                "checkpoint": path.name,
                "step": meta.get("step"),
                "depth_le_3_top1": top1,
                "depth_le_3_top8": top8,
                "n": clean["depth_le_3_n"],
                "delta_vs_anchor": round(top1 - ANCHOR_TOP1, 4),
            }
        )
        print(
            f"    {path.name:>14}  top1 {top1:.4f}  top8 {top8:.4f}  "
            f"delta {top1 - ANCHOR_TOP1:+.4f}"
        )

    args.out.write_text(
        json.dumps({"anchor_top1": ANCHOR_TOP1, "trajectory": rows}, indent=2) + "\n"
    )
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
