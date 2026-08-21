"""Sweep verification: is the steps knob live, is the predicate established, and
does the selected `f` leave an experiment?

Three questions the sweep raised and could not answer from its own table.

**1. The mechanism arm's digest.** `f = 0.00` at 200 steps returned *exactly*
0.8942 — the 400-step control's value to four places, from half the training.
An implausible intermediate is worth checking before it is believed (F-28's
lesson, where "minus fourteen fallbacks" announced its own defect). If the two
runs' parameter digests are identical, **the steps knob did not take** and the
arm is void along with the falsification it appeared to deliver.

**2. Seeds at the selected f.** A single reading 0.0019 above the band, against a
seed spread of 0.0090 measured elsewhere in the table, does not establish
"holds". This is not weakening the frozen rule — it is refusing to evaluate a
frozen predicate on an estimate too noisy to evaluate it with.

**3. Does the selected `f` leave an experiment?** Two fractions landing dead level
with the anchor is consistent with the model barely moving. `‖θ_f − θ_anchor‖ /
‖θ_anchor‖`, reported relative to `f = 0.00`'s movement, puts evidence under that.

Top-1 is reported as a **raw count** as well as a ratio, so a coincidence at four
decimal places is visible rather than spooky.
"""

from __future__ import annotations

import argparse
import hashlib
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

#: (rehearsal_frac, steps, seed). The mechanism pair first, so the digest answer
#: lands early; then seeds at the two band-holding arms.
ARMS = (
    (0.00, 400, 0),
    (0.00, 200, 0),
    (0.65, 400, 0),
    (0.65, 400, 1),
    (0.65, 400, 2),
    (0.75, 400, 0),
    (0.75, 400, 1),
    (0.50, 400, 0),
)
BAND = 0.968


def weight_digest(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def flat(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for _, p in sorted(model.state_dict().items())])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring", type=Path, default=REPO / "runs" / "ring0_replay" / "ring-0")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "sweep_verify.json")
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

    anchor_model, _ = load_checkpoint(ANCHOR, cfg)
    anchor_flat = flat(anchor_model)
    anchor_norm = float(torch.linalg.vector_norm(anchor_flat))

    print(f"\n  SWEEP VERIFICATION — ring-0 ({len(ring)} rows), band {BAND}\n")
    print(
        f"    {'f':>5} {'steps':>6} {'seed':>5} {'top-1':>8} {'hits/n':>12} "
        f"{'movement':>9} {'rel':>6}  digest"
    )

    rows = []
    for frac, steps, seed in ARMS:
        arm_cfg = replace(cfg, train=replace(cfg.train, rehearsal_frac=frac))
        model, _ = load_checkpoint(ANCHOR, arm_cfg)
        train_on_ring(model, ring, arm_cfg, steps=steps, seed=seed)
        model.eval()
        clean = top_k_by_depth(model, held, arm_cfg, device="cpu", exclude=shared)
        top1, n = clean["depth_le_3_top1"], clean["depth_le_3_n"]
        hits = round(top1 * n)
        digest = weight_digest(model)
        moved = float(torch.linalg.vector_norm(flat(model) - anchor_flat)) / anchor_norm
        rows.append(
            {
                "rehearsal_frac": frac,
                "steps": steps,
                "seed": seed,
                "depth_le_3_top1": top1,
                "hits": hits,
                "n": n,
                "weight_digest": digest,
                "movement_vs_anchor": round(moved, 6),
                "holds_band": top1 >= BAND,
            }
        )
        print(
            f"    {frac:>5.2f} {steps:>6} {seed:>5} {top1:>8.4f} {hits:>6}/{n:<5} "
            f"{moved:>9.5f} {'':>6}  {digest[:12]}"
        )

    # --- 1. did the steps knob take? ---------------------------------------
    a = next(r for r in rows if r["steps"] == 400 and r["rehearsal_frac"] == 0.0 and r["seed"] == 0)
    b = next(r for r in rows if r["steps"] == 200)
    same = a["weight_digest"] == b["weight_digest"]
    print("\n  1. THE STEPS KNOB\n")
    print(f"    f=0.00 400 steps : {a['weight_digest'][:16]}  {a['hits']}/{a['n']}")
    print(f"    f=0.00 200 steps : {b['weight_digest'][:16]}  {b['hits']}/{b['n']}")
    if same:
        print("\n    IDENTICAL DIGESTS — the steps argument did not take. The mechanism")
        print("    arm is VOID, and so is the falsification it appeared to deliver.")
    else:
        print(f"\n    digests differ; raw counts {a['hits']} vs {b['hits']} of {a['n']}")
        print("    -> the 4dp match is coincidence, and the arm stands.")

    # --- 2. is the predicate established at the selected f? -----------------
    print("\n  2. THE PREDICATE, on an adequate estimate\n")
    for frac in (0.65, 0.75):
        seeds = [r for r in rows if r["rehearsal_frac"] == frac]
        vals = [r["depth_le_3_top1"] for r in seeds]
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            holds = all(v >= BAND for v in vals)
            print(
                f"    f={frac:.2f}: {[f'{v:.4f}' for v in vals]}  spread {spread:.4f}  "
                f"all hold band: {holds}"
            )

    # --- 3. does the selected f leave an experiment? ------------------------
    base = a["movement_vs_anchor"]
    print("\n  3. DOES THE SELECTED f LEAVE AN EXPERIMENT?\n")
    print(
        f"    movement is ||theta_f - theta_anchor|| / ||theta_anchor||, "
        f"relative to f=0.00's {base:.5f}\n"
    )
    for r in rows:
        if r["steps"] == 400:
            rel = r["movement_vs_anchor"] / base if base else float("nan")
            print(
                f"    f={r['rehearsal_frac']:.2f} seed {r['seed']}: "
                f"{r['movement_vs_anchor']:.5f}  = {rel:6.1%} of the control's movement"
            )

    args.out.write_text(json.dumps({"band": BAND, "arms": rows}, indent=2, sort_keys=True) + "\n")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
