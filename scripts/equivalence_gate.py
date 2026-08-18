"""Job 3 — the equivalence gate, and the watchlist's first reading.

**What licenses the routing.** F-33 replaced the cadence's aggregate
implementation with the ladder's per-problem one. If the thin implementation was
merely thin, the numbers reproduce exactly; if they move, it was *wrong*, and
that is a finding to diagnose before anything proceeds.

Run on **two** checkpoints, not one:

* **`ckpt-4`** — reproduces the rehearsal's recorded cadence: 910/1200 at 48,
  722/1200 at 1, 108/600 primary, 22/50/36 per stratum.
* **the anchor** — reproduces 1193/1200 at 48 with **the same seven misses by
  key**. Equivalence at one checkpoint proves equivalence at one checkpoint; the
  reference is where every future number is read against, and the pod's licence
  was measured under the old execution path. This re-earns it at the reference,
  and produces the watchlist's reference rows under the same path that will
  produce every campaign reading compared against them — like-for-like, which
  this project applies everywhere else.

It also yields the **first watchlist reading**, the question PREREG §5 was written
for and could not answer when it mattered (F-35).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reckoner.campaign import ANCHOR, ANCHOR_BEAT, frozen_watchlist, run_instruments
from reckoner.config import Config, validate
from reckoner.ladderpass import read_pair_scores
from reckoner.model import load_checkpoint

REPO = Path(__file__).resolve().parents[1]
DIAGNOSTIC = REPO / "runs" / "chunk11_misses_diagnostic.json"

#: The rehearsal's recorded cadence unit — instruments.jsonl, iteration 4.
CKPT4 = {
    "no_regress_sims_48": 910,
    "no_regress_sims_1": 722,
    "primary": "108/600",
    "per_stratum": {"scripted_in_7": 22, "scripted_in_8": 50, "scripted_in_10": 36},
}
#: Part-0d / the misses diagnostic, on the anchor.
ANCHOR_EXPECTED = {"no_regress_sims_48": 1193, "no_regress_sims_1": 1176}
#: Registered in SWEEP-m1a4.md before this ran.
PREDICTED_WATCHLIST = {"family_remaining": 24, "novel_misses": 266}


def measure(label: str, checkpoint: Path, cfg: Config, root: Path, index: int) -> dict:
    model, _ = load_checkpoint(checkpoint, cfg)
    model.eval()
    print(f"\n  measuring {label} ({checkpoint.name}) ...")
    out = run_instruments(model, cfg, iteration=index, anchor_beat=ANCHOR_BEAT, run_dir=root)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt4", type=Path, default=REPO / "runs" / "m1_rehearsal" / "ckpt-4.pt")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "equivalence_gate")
    args = parser.parse_args()

    cfg = Config()
    validate(cfg)
    torch.set_num_threads(cfg.campaign.intra_op_threads)
    args.out.mkdir(parents=True, exist_ok=True)

    checks: list[tuple[str, bool, str]] = []
    results: dict = {}

    # --- ckpt-4: the equivalence gate proper --------------------------------
    if args.ckpt4.exists():
        got = measure("ckpt-4", args.ckpt4, cfg, args.out / "ckpt4", 4)
        results["ckpt4"] = got
        for budget, want in (
            ("48", CKPT4["no_regress_sims_48"]),
            ("1", CKPT4["no_regress_sims_1"]),
        ):
            at_par = got[f"no_regress_sims_{budget}"]["at_par"]
            checks.append((f"ckpt-4 no-regress @{budget} == {want}", at_par == want, str(at_par)))
        pooled = got["primary"]["pooled_beat_par"]
        checks.append((f"ckpt-4 primary == {CKPT4['primary']}", pooled == CKPT4["primary"], pooled))
        for stratum, want in CKPT4["per_stratum"].items():
            beat = got["primary"]["per_stratum"][stratum]["beat"]
            checks.append((f"ckpt-4 {stratum} == {want}", beat == want, str(beat)))
        watch = got["watchlist"]
        results["watchlist"] = watch
    else:
        checks.append(("ckpt-4 present", False, f"absent at {args.ckpt4}"))
        watch = None

    # --- the anchor: re-licensing the path at the reference ------------------
    got = measure("anchor", ANCHOR, cfg, args.out / "anchor", 0)
    results["anchor"] = got
    for budget, want in ANCHOR_EXPECTED.items():
        at_par = got[budget]["at_par"]  # keys are already no_regress_sims_*
        checks.append((f"anchor {budget} == {want}", at_par == want, str(at_par)))

    # the seven misses, BY KEY rather than by count
    diag = json.loads(DIAGNOSTIC.read_text())
    scores = read_pair_scores(args.out / "anchor" / "ladder", 0)
    for budget in ("48", "1"):
        recorded = set(diag["per_budget"][budget]["misses"])
        measured = {
            r["problem_key"] for r in scores if r["arm"] == f"model@{budget}" and int(r["z"]) != 0
        }
        checks.append(
            (
                f"anchor's {len(recorded)} misses @{budget} match BY KEY",
                measured == recorded,
                f"{len(measured)} measured, {len(measured & recorded)} shared",
            )
        )
    anchor_watch = got["watchlist"]
    checks.append(
        (
            "anchor's watchlist reference row is (24, 0)",
            (anchor_watch["family_remaining"], anchor_watch["novel_misses"]) == (24, 0),
            f"({anchor_watch['family_remaining']}, {anchor_watch['novel_misses']})",
        )
    )

    print("\n  EQUIVALENCE GATE\n")
    for name, ok, detail in checks:
        print(f"    {'ok  ' if ok else 'FAIL'} {name}{'  — ' + detail if detail else ''}")

    if watch is not None:
        print("\n  WATCHLIST — first reading, against the registered prediction\n")
        print(f"    frozen family        : {len(frozen_watchlist())}")
        print("    anchor reference     : (24, 0) @1 sim, (7, 0) @48")
        for column, predicted in PREDICTED_WATCHLIST.items():
            actual = watch[column]
            mark = "as predicted" if actual == predicted else f"PREDICTED {predicted}"
            print(f"    {column:<20} : {actual:>4}   {mark}")
        print(
            f"    partition            : {watch['family_remaining']} + {watch['novel_misses']}"
            f" == {watch['pass_misses']} -> "
            f"{watch['family_remaining'] + watch['novel_misses'] == watch['pass_misses']}"
        )

    (args.out / "gate.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    failed = [n for n, ok, _ in checks if not ok]
    if failed:
        print(f"\n  FAIL — {len(failed)} of {len(checks)}. The thin implementation was not")
        print("  merely thin. HALT the queue and diagnose before anything else spends.")
        return 1
    print(f"\n  PASS — {len(checks)}/{len(checks)}. The routing is licensed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
