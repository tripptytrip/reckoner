"""Part 0d: the anchor's z-composition on the scripted strata — rider (c) on the
second instrument, the identical treatment the first received.

F-20's lesson recursed onto its own remedy. Part 0 measured the anchor's headroom
on the BFS-exact suites and found it exhausted. `scripted_in_7..10` was minted as
the successor, and the floor certificate then showed those problems have true BFS
par 5-6 under a 7-step scripted label — which are lengths the anchor already
reaches on the neighbouring suites. So the successor may be ceiling'd **from
above** before its first campaign pass, and nobody had asked.

**The prediction is registered before the run** (`RULING-chunk11-scripted-strata.md`,
committed at `48556a3`), so it can be scored rather than remembered:

> scripted_in_7 mean z comes back strongly positive; liveness, if anywhere, sits
> at strata 9-10, where true par is unmeasured and plausibly >= 7.

Same protocol as Part 0 and Part 0b: sims 48, m 16, eval profile, value-silent.
The `<0` bin is the live one here — on these strata it means the anchor **beat**
a provisional floor, which is exactly what the instrument was minted to allow.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from reckoner.config import Config, config_fingerprint, validate
from reckoner.dataset import git_sha, read_suite, suite_problem, write_record
from reckoner.evaluate import model_evaluator
from reckoner.model import load_checkpoint
from reckoner.runner import run_iteration

REPO = Path(__file__).resolve().parents[1]
RULING_SHA = "48556a3"
SUITES = REPO / "runs" / "suites"
STRATA = (7, 8, 9, 10)

#: Registered before the run, from the ruling. Scored in the record, not recalled.
PREDICTION = {
    "source": "RULING-chunk11-scripted-strata.md @ 48556a3",
    "claim_1": "scripted_in_7 mean z comes back strongly positive",
    "claim_2": "liveness, if anywhere, sits at strata 9-10",
    "consequence_if_all_saturated": (
        "mint at TRUE difficulty beyond the anchor's reach — genuinely harder "
        "problems wearing scripted floors. The measurement, not the framing, decides"
    ),
}

#: A stratum is BORN SATURATED FROM ABOVE when the anchor already beats the
#: provisional floor on most of it: there is little room left to reward beating
#: par, because beating par is already the default outcome.
SATURATED_FROM_ABOVE = 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    cfg = Config()
    validate(cfg)
    cfg = replace(cfg, search=replace(cfg.search, root_noise=False))
    model, _ = load_checkpoint(REPO / "runs" / "phase1" / "phase1.pt", cfg)
    model.eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    evaluator = model_evaluator(model, cfg, 0.0)

    started = time.perf_counter()
    per_stratum = {}
    for k in STRATA:
        problems = [suite_problem(r) for r in read_suite(SUITES / f"scripted_in_{k}.jsonl")]
        stats = run_iteration(problems, evaluator, cfg, None, sims=48, m=16, seed=0)
        stats.check_descent_identity()
        bins = dict(stats.steps_minus_par)
        n = stats.episodes
        beat = bins["<0"]
        at = bins["0"]
        over = sum(v for key, v in bins.items() if key not in ("<0", "0"))
        # z is +1 / 0 / -1; unsolved episodes are -1 and are already outside the
        # solved histogram, so they are added explicitly rather than assumed away.
        unsolved = n - stats.episodes_solved
        mean_z = (beat - (over + unsolved)) / n
        per_stratum[f"scripted_in_{k}"] = {
            "par": k,
            "par_source": "scripted",
            "problems": n,
            "solved": stats.episodes_solved,
            "capped": stats.episodes_capped,
            "stuck": stats.episodes_stuck,
            "beat_par_z_plus_1": beat,
            "at_par_z_0": at,
            "over_par_or_unsolved_z_minus_1": over + unsolved,
            "beat_par_rate": round(beat / n, 6),
            "mean_z": round(mean_z, 6),
            "steps_minus_par": bins,
            "born_saturated_from_above": beat / n >= SATURATED_FROM_ABOVE,
            "seconds": round(stats.seconds, 2),
        }
        print(
            f"    scripted_in_{k}: beat {beat:>3} at {at:>3} over/unsolved "
            f"{over + unsolved:>3} of {n}  mean z {mean_z:+.4f}  {stats.seconds:>7.1f}s",
            flush=True,
        )

    live = [name for name, v in per_stratum.items() if not v["born_saturated_from_above"]]
    scoring = {
        "claim_1_scripted_in_7_strongly_positive": per_stratum["scripted_in_7"]["mean_z"] > 0.5,
        "claim_1_measured_mean_z": per_stratum["scripted_in_7"]["mean_z"],
        "claim_2_liveness_at_9_or_10": bool({"scripted_in_9", "scripted_in_10"} & set(live)),
        "strata_not_saturated_from_above": live,
        "all_four_born_saturated": not live,
    }
    report = {
        "ruling_frozen_at": RULING_SHA,
        "git_sha": git_sha(REPO),
        "protocol": {
            "model": "runs/phase1/phase1.pt",
            "sims": 48,
            "gumbel_m": 16,
            "root_noise": cfg.search.root_noise,
            "step_cap": cfg.episode.step_cap,
            "value_scale": 0.0,
            "seed": 0,
            "config_fingerprint": config_fingerprint(cfg),
            "device": "cpu",
        },
        "saturated_from_above_threshold": SATURATED_FROM_ABOVE,
        "registered_prediction": PREDICTION,
        "per_stratum": per_stratum,
        "prediction_scoring": scoring,
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
    }
    write_record(REPO / "runs" / "chunk11_part0d_scripted_strata.json", report)
    print("\n  PREDICTION SCORING\n")
    print(json.dumps(scoring, indent=2))
    print(f"\n  wall clock {report['wall_clock_seconds']}s")
    print("  wrote runs/chunk11_part0d_scripted_strata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
