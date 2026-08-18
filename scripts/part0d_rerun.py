"""Job 5 — Part-0d re-run, capturing per-problem outcomes. Closes F-30's baseline arm.

PREREG-m1 §2 makes the primary's test of record a **paired-difference bootstrap**,
paired **per problem** against the anchor's Part-0d outcomes. Part-0d stored
aggregates — 3,391 bytes, no problem identifiers — so the test has been
unsatisfiable since the freeze (F-30). The campaign arm now carries pairing
(F-33's routing); this is the other arm.

**The re-run's aggregates ARE the verification.** Part-0d's protocol is printed
verbatim in §3 and its inputs are frozen, so the same measurement must reproduce
`43 / 26 / 32` and `101/600` exactly. A mismatch is itself a finding: it would
mean the per-problem capture is a *new* measurement rather than the recorded one,
and pairing against it would be pairing against something else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reckoner.campaign import ANCHOR, SUCCESSOR_STRATA, SUITES, eval_profile
from reckoner.config import Config, validate
from reckoner.dataset import read_suite, suite_problem
from reckoner.evaluate import ModelArm
from reckoner.ladder import problem_key_of
from reckoner.ladderpass import is_complete, read_pair_scores, run_pass
from reckoner.model import load_checkpoint

REPO = Path(__file__).resolve().parents[1]

#: runs/chunk11_part0d_scripted_strata.json — what the record says.
RECORDED = {"scripted_in_7": 43, "scripted_in_8": 26, "scripted_in_10": 32}
POOLED = 101


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "part0d_paired")
    args = parser.parse_args()

    cfg = Config()
    validate(cfg)
    ev = eval_profile(cfg)
    torch.set_num_threads(cfg.campaign.intra_op_threads)

    model, _ = load_checkpoint(ANCHOR, cfg)
    model.eval()
    arm = ModelArm(model, sims=48, m=16)

    problems, stratum_of = [], {}
    for k in SUCCESSOR_STRATA:
        for row in read_suite(SUITES / f"scripted_in_{k}.jsonl"):
            problem = suite_problem(row)
            problems.append(problem)
            stratum_of[problem_key_of(problem)] = f"scripted_in_{k}"

    print(f"\n  PART-0D RE-RUN — {len(problems)} problems, per-problem capture\n")
    if not is_complete(args.out, 0):
        run_pass(
            args.out,
            0,
            [arm],
            problems,
            ev,
            roles={arm.name: "baseline"},
            calibration_note=(
                "the anchor's Part-0d outcomes, recaptured per problem so the "
                "primary's paired-difference bootstrap has both arms (F-30)"
            ),
            seed=0,
        )
    scores = [r for r in read_pair_scores(args.out, 0) if r["arm"] == arm.name]

    per = {f"scripted_in_{k}": {"beat": 0, "of": 0} for k in SUCCESSOR_STRATA}
    for row in scores:
        cell = per[stratum_of[row["problem_key"]]]
        cell["of"] += 1
        cell["beat"] += int(row["z"]) == 1

    ok = True
    for stratum, want in RECORDED.items():
        got = per[stratum]["beat"]
        ok &= got == want
        print(
            f"    {'ok  ' if got == want else 'FAIL'} {stratum:<16} beat {got:>3}  "
            f"(Part-0d recorded {want})"
        )
    pooled = sum(c["beat"] for c in per.values())
    ok &= pooled == POOLED
    print(
        f"    {'ok  ' if pooled == POOLED else 'FAIL'} pooled           {pooled}/600  "
        f"(Part-0d recorded {POOLED}/600)"
    )

    (args.out / "verification.json").write_text(
        json.dumps(
            {
                "recorded": RECORDED,
                "measured": per,
                "pooled": pooled,
                "reproduced": bool(ok),
                "rows": len(scores),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if not ok:
        print("\n  DIVERGED — the re-run is not the recorded measurement, so pairing")
        print("  against it would pair against something else. That is a finding.")
        return 1
    print(f"\n  REPRODUCED — {len(scores)} per-problem rows. F-30's baseline arm is closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
