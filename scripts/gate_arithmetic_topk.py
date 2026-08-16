"""Gate arithmetic for the Phase-1 top-k gate, run the way chunk 7 ran its own.

Chunk 7 established the rule and this script applies it to chunk 8: **compute
what a gate can and cannot distinguish before declaring it.** There it showed a
100% depth-1 gate was arithmetically *impossible* at m=3. Here it shows the
registered top-8 gate is arithmetically *guaranteed* — the same instrument
catching the opposite failure.

The mechanism: top-k is ranked over the **legal** action set (an unmasked top-k
would credit actions the movegen refuses). So when a state has ``<= k`` legal
actions, every legal action is inside the top k and the target is a certain hit,
whatever the network says. A top-k gate therefore has a **floor** equal to the
fraction of states with ``<= k`` legal actions, and any threshold beneath that
floor is passed by random initialisation.

Both polarities, as always: the floor is computed from the legal-action
distribution *and* measured by running the metric on an untrained network.

Writes ``runs/gate_arithmetic_topk.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from reckoner.config import Config
from reckoner.dataset import anchored_data, git_sha, write_record
from reckoner.model import Reckoner
from reckoner.train import SupervisionSet, make_batch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

KS = (1, 2, 3, 4, 5, 8)


def legal_counts(data: SupervisionSet, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """``legal_mask.sum()`` per row — the set the metric actually ranks over."""
    counts, depths = [], []
    for start in range(0, len(data), 256):
        idx = list(range(start, min(start + 256, len(data))))
        batch = make_batch(data, idx, cfg)
        counts.extend(batch.legal_mask.sum(dim=1).tolist())
        depths.extend(batch.depth.tolist())
    return np.array(counts), np.array(depths)


def measured_topk(model: Reckoner, data: SupervisionSet, cfg: Config) -> dict:
    model.eval()
    hits = dict.fromkeys(KS, 0)
    shallow_hits = dict.fromkeys(KS, 0)
    total = shallow_total = 0
    with torch.no_grad():
        for start in range(0, len(data), 256):
            idx = list(range(start, min(start + 256, len(data))))
            batch = make_batch(data, idx, cfg)
            policy, _v, _s = model(batch.tokens, batch.site_positions)
            masked = policy.masked_fill(~batch.legal_mask, float("-inf"))
            target = batch.policy_target.argmax(dim=1)
            score = masked.gather(1, target.unsqueeze(1))
            rank = (masked > score).sum(dim=1) + 1
            shallow = batch.depth <= 3
            for k in KS:
                ok = rank <= k
                hits[k] += int(ok.sum())
                shallow_hits[k] += int((ok & shallow).sum())
            total += len(idx)
            shallow_total += int(shallow.sum())
    return {
        "overall": {k: round(hits[k] / total, 4) for k in KS},
        "depth_le_3": {k: round(shallow_hits[k] / shallow_total, 4) for k in KS},
        "n": total,
        "n_depth_le_3": shallow_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=anchored_data("phase1_eval"))
    parser.add_argument("--checkpoint", type=Path, default=REPO / "runs" / "phase1" / "phase1.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    data = SupervisionSet(args.data)
    counts, depths = legal_counts(data, cfg)
    shallow = depths <= 3

    print("  GATE ARITHMETIC — Phase-1 top-k, computed before the gate is read\n")
    print(f"  held-out states: {len(counts):,}  (depth <= 3: {int(shallow.sum()):,})")
    print(
        f"  legal actions  : min {counts.min()}  median {int(np.median(counts))}  max {counts.max()}\n"
    )

    torch.manual_seed(args.seed)
    untrained = measured_topk(Reckoner(cfg), data, cfg)
    trained = None
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = Reckoner(cfg)
        model.load_state_dict(state["state_dict"])
        trained = measured_topk(model, data, cfg)

    print(
        f"  {'k':>3} {'floor d<=3':>11} {'untrained':>10} {'trained':>9} {'headroom':>9}  verdict"
    )
    rows = []
    for k in KS:
        floor = float((counts[shallow] <= k).mean())
        un = untrained["depth_le_3"][k]
        tr = trained["depth_le_3"][k] if trained else None
        head = round(tr - floor, 4) if tr is not None else None
        verdict = "VACUOUS at 0.90" if floor >= 0.90 else "discriminating"
        rows.append(
            {
                "k": k,
                "floor_depth_le_3": round(floor, 4),
                "untrained": un,
                "trained": tr,
                "headroom": head,
                "verdict": verdict,
            }
        )
        print(
            f"  {k:>3} {floor:>11.4f} {un:>10.4f} "
            f"{'—' if tr is None else f'{tr:>9.4f}'} "
            f"{'—' if head is None else f'{head:>9.4f}'}  {verdict}"
        )

    print(
        "\n  floor = fraction of depth<=3 states with <= k legal actions. A threshold\n"
        "  at or below the floor is passed by random initialisation."
    )

    out = {
        "n": len(counts),
        "n_depth_le_3": int(shallow.sum()),
        "legal_action_histogram": {
            int(v): int(c) for v, c in zip(*np.unique(counts, return_counts=True), strict=True)
        },
        "rows": rows,
        "untrained": untrained,
        "trained": trained,
        "registered_gate": "top-8 >= 0.90 on depth <= 3",
        "git_sha": git_sha(REPO),
        "data_digests": json.loads((args.data / "meta.json").read_text())["digests"],
    }
    write_record(REPO / "runs" / "gate_arithmetic_topk.json", out)
    print("\n  wrote runs/gate_arithmetic_topk.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
