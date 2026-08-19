"""Job 4 — the rehearsal-fraction sweep. M1-A4's blocking input.

Selection rule, band, and both branches were frozen in `SWEEP-m1a4.md` **before
this script existed**, because top-1 is monotone in `f` by construction: more
supervised data means a better supervised metric, so scoring arms by top-1 alone
drives `f` toward 1 — maximum preservation, zero learning, a winner that has
stopped doing the experiment.

* **Rule:** the *smallest* `f` whose gate-10b top-1 holds the band.
* **Band:** top-1 >= 0.968, derived from §8's floor slack rather than chosen.
* **Control:** `f = 0.00` must reproduce `ckpt-0`'s **0.8942**, or the ring-0
  replay is not the same measurement and the sweep is void.

Every arm trains from the anchor on the **same fixed ring-0**, same seeds, so `f`
is the only thing that varies. The band SCREENS; the cadence DECIDES — a passing
arm is a candidate, and M1-A4's value stays provisional until rehearsal attempt
2's cadence unit confirms it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from reckoner.campaign import ANCHOR
from reckoner.config import Config, validate
from reckoner.model import load_checkpoint
from reckoner.replay import ReplayRing
from reckoner.train import SupervisionSet, train_on_ring

REPO = Path(__file__).resolve().parents[1]

#: (rehearsal_frac, steps, seed, note). Selection considers ONLY the registered
#: protocol — steps == 400 at seed 0. The variance and mechanism arms are
#: diagnostics and are excluded from candidacy by construction, so extra arms can
#: find an answer and can never move the criterion.
ARMS = (
    (0.00, 400, 0, "control — must reproduce ckpt-0"),
    (0.00, 400, 1, "variance: control, second seed"),
    (0.10, 400, 0, ""),
    (0.15, 400, 0, ""),
    (0.25, 400, 0, ""),
    (0.35, 400, 0, ""),
    (0.50, 400, 0, ""),
    (0.50, 400, 1, "variance: best arm, second seed"),
    (0.65, 400, 0, ""),
    (0.75, 400, 0, ""),
    (0.00, 200, 0, "MECHANISM: 19.6 ring-epochs, matched to f=0.50, NO supervision"),
)
REGISTERED_STEPS = 400
BAND = 0.968  # SWEEP-m1a4.md §3
CONTROL = 0.8942  # ckpt-0, from the rehearsal
ANCHOR_TOP1 = 0.9699
MECHANISM_TWIN = 0.50  # the arm whose ring-epochs the mechanism arm matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring", type=Path, default=REPO / "runs" / "ring0_replay" / "ring-0")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "sweep_rehearsal_frac.json")
    args = parser.parse_args()

    from reckoner.dataset import anchored_data
    from scripts.train_phase1 import shared_state_indices, top_k_by_depth

    cfg = Config()
    validate(cfg)
    torch.set_num_threads(cfg.campaign.intra_op_threads)

    eval_path = anchored_data("phase1_eval")
    held = SupervisionSet(eval_path)
    shared = shared_state_indices(eval_path)
    ring = ReplayRing.load(args.ring, cfg)

    print(f"\n  SWEEP — rehearsal_frac, on ring-0 ({len(ring)} rows)")
    print(f"  band: top-1 >= {BAND}   control: f=0.00 must reproduce {CONTROL}")
    print(f"  anchor: {ANCHOR_TOP1}\n")
    print(
        f"    {'f':>5} {'steps':>6} {'seed':>5} {'top-1':>8} {'delta':>8} "
        f"{'ring-ep':>8} {'vh/step':>8}  band note"
    )

    rows = []
    for frac, steps, seed, note in ARMS:
        arm_cfg = replace(cfg, train=replace(cfg.train, rehearsal_frac=frac))
        model, _ = load_checkpoint(ANCHOR, arm_cfg)
        stats = train_on_ring(model, ring, arm_cfg, steps=steps, seed=seed)
        model.eval()
        clean = top_k_by_depth(model, held, arm_cfg, device="cpu", exclude=shared)
        top1 = clean["depth_le_3_top1"]
        batch = arm_cfg.train.batch_size
        ring_ep = steps * round(batch * (1 - frac)) / len(ring)
        candidate = steps == REGISTERED_STEPS and seed == 0
        rows.append(
            {
                "rehearsal_frac": frac,
                "steps": steps,
                "seed": seed,
                "note": note,
                "depth_le_3_top1": top1,
                "delta_vs_anchor": round(top1 - ANCHOR_TOP1, 4),
                "ring_epochs": round(ring_ep, 1),
                "value_head_examples_per_step": round(batch * (1 - frac)),
                "rehearsal_batches": stats.rehearsal_batches,
                "candidate": candidate,
                "holds_band": top1 >= BAND,
            }
        )
        flag = "YES" if top1 >= BAND else "no"
        print(
            f"    {frac:>5.2f} {steps:>6} {seed:>5} {top1:>8.4f} {top1 - ANCHOR_TOP1:>+8.4f} "
            f"{ring_ep:>8.1f} {round(batch * (1 - frac)):>8}  {flag:<4} {note}"
        )

    control = rows[0]["depth_le_3_top1"]
    valid = abs(control - CONTROL) < 5e-4
    print(
        f"\n  control f=0.00 -> {control:.4f} against ckpt-0's {CONTROL}: "
        f"{'REPRODUCED' if valid else 'DIVERGED — THE SWEEP IS VOID'}"
    )

    # --- variance, before any arm-to-arm difference is read as signal --------
    print("\n  VARIANCE — the noise floor the 0.0062 shortfall is measured against\n")
    for frac in (0.00, MECHANISM_TWIN):
        pair = [r for r in rows if r["rehearsal_frac"] == frac and r["steps"] == REGISTERED_STEPS]
        if len(pair) == 2:
            spread = abs(pair[0]["depth_le_3_top1"] - pair[1]["depth_le_3_top1"])
            print(
                f"    f={frac:.2f}: {pair[0]['depth_le_3_top1']:.4f} vs "
                f"{pair[1]['depth_le_3_top1']:.4f}   seed spread {spread:.4f}"
            )

    # --- the mechanism arm: epochs or supervision? ---------------------------
    mech = next((r for r in rows if r["steps"] != REGISTERED_STEPS), None)
    twin = next((r for r in rows if r["rehearsal_frac"] == MECHANISM_TWIN and r["seed"] == 0), None)
    if mech and twin:
        print("\n  MECHANISM — same ring-epochs, supervision present vs absent\n")
        print(
            f"    f=0.50, 400 steps : {twin['depth_le_3_top1']:.4f}  "
            f"({twin['ring_epochs']} epochs, supervision ON)"
        )
        print(
            f"    f=0.00, 200 steps : {mech['depth_le_3_top1']:.4f}  "
            f"({mech['ring_epochs']} epochs, supervision OFF)"
        )
        gap = twin["depth_le_3_top1"] - mech["depth_le_3_top1"]
        print(f"    gap attributable to SUPERVISION: {gap:+.4f}")
        print("    near zero -> epoch scaling is the mechanism, rehearsal redundant")
        print("    large     -> supervised anchoring is the mechanism")

    holding = [r for r in rows if r["holds_band"] and r["candidate"]]
    verdict = min(holding, key=lambda r: r["rehearsal_frac"]) if holding else None
    if verdict:
        print(
            f"\n  SELECTED: f = {verdict['rehearsal_frac']:.2f} — the smallest f holding the band."
        )
        print("  Candidate only: the band SCREENS, the cadence DECIDES (SWEEP-m1a4 §3.1).")
    else:
        print("\n  NO ARM HOLDS THE BAND — SWEEP-m1a4 §5's second branch.")
        print("  A finding, not a failed round: the frozen page admits no viable training")
        print("  configuration at 400 steps per iteration. Epoch scaling becomes the NEXT")
        print("  round's single lever, never a simultaneous change.")

    args.out.write_text(
        json.dumps(
            {
                "band": BAND,
                "control_expected": CONTROL,
                "control_valid": valid,
                "selected": verdict,
                "arms": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\n  wrote {args.out}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
