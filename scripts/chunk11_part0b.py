"""Part 0b: sweep the anchor's at-par rate over sims, and apply the frozen rule.

Protocol and selection rule are frozen in `PREREG-chunk11-part0bc.md` at
`5872ca8`, before this script existed. Nothing here chooses s-star; it measures the
sweep and **applies** a rule that was already written, including the branches
that find nothing.

`m = min(16, sims)` is the declared clamp: `search` clamps `m` to the legal root
actions but not to `sims`, so an unclamped low-sims run would nominate sixteen
candidates with six simulations to separate them — a different algorithm, not a
cheaper one.
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
from reckoner.logschema import STEPS_MINUS_PAR_BINS
from reckoner.model import load_checkpoint
from reckoner.runner import run_iteration

REPO = Path(__file__).resolve().parents[1]
EXPECTATIONS_SHA = "5872ca8"
SUITES = REPO / "runs" / "suites"

SWEEP = (6, 8, 12, 16, 48)
TARGET = 0.55
WINDOW = (0.4, 0.7)
DOWNWARD_EXTENSION = (1, 2, 3, 4)


def eval_config() -> Config:
    cfg = Config()
    validate(cfg)
    return replace(cfg, search=replace(cfg.search, root_noise=False))


def measure(model, cfg: Config, sims: int, problems_by_suite: dict) -> dict:
    """One sweep point: the whole 1,200-problem instrument at this budget."""
    m = min(cfg.search.gumbel_m, sims)  # declared clamp, PREREG-chunk11-part0bc
    evaluator = model_evaluator(model, cfg, 0.0)
    pooled = dict.fromkeys(STEPS_MINUS_PAR_BINS, 0)
    totals = {"episodes": 0, "solved": 0, "capped": 0, "stuck": 0}
    started = time.perf_counter()
    for problems in problems_by_suite.values():
        stats = run_iteration(problems, evaluator, cfg, None, sims=sims, m=m, seed=0)
        stats.check_descent_identity()
        for k, v in stats.steps_minus_par.items():
            pooled[k] += v
        totals["episodes"] += stats.episodes
        totals["solved"] += stats.episodes_solved
        totals["capped"] += stats.episodes_capped
        totals["stuck"] += stats.episodes_stuck
    n = totals["episodes"]
    return {
        "sims": sims,
        "gumbel_m": m,
        "at_par": pooled["0"],
        "beat_par": pooled["<0"],
        "over_par": sum(v for k, v in pooled.items() if k not in ("<0", "0")),
        "at_par_rate": round(pooled["0"] / n, 6) if n else 0.0,
        "steps_minus_par": pooled,
        **totals,
        "seconds": round(time.perf_counter() - started, 2),
    }


def select(points: list[dict]) -> dict:
    """Apply the frozen rule. Every branch reports whether it fired and why."""
    low, high = WINDOW
    in_window = [p for p in points if low <= p["at_par_rate"] <= high]
    branches = []

    if in_window:
        best = min(in_window, key=lambda p: (abs(p["at_par_rate"] - TARGET), p["sims"]))
        tied = [
            p
            for p in in_window
            if abs(p["at_par_rate"] - TARGET) == abs(best["at_par_rate"] - TARGET)
        ]
        branches.append(
            {
                "branch": "in_window",
                "fired": True,
                "candidates": [p["sims"] for p in in_window],
                "tie_broken_toward_smaller_sims": len(tied) > 1,
            }
        )
        return {"s_star": best["sims"], "s_star_rate": best["at_par_rate"], "branches": branches}

    ordered = sorted(points, key=lambda p: p["sims"])
    below = [p for p in ordered if p["at_par_rate"] < low]
    above = [p for p in ordered if p["at_par_rate"] > high]
    branches.append(
        {
            "branch": "in_window",
            "fired": False,
            "reason": "no swept value landed in [0.4, 0.7]",
        }
    )

    if not below:
        branches.append(
            {
                "branch": "all_above_window",
                "fired": True,
                "action": f"extend downward over {DOWNWARD_EXTENSION} and re-apply",
                "extension": list(DOWNWARD_EXTENSION),
            }
        )
        return {"s_star": None, "needs": "downward_extension", "branches": branches}
    if not above:
        branches.append(
            {
                "branch": "all_below_window",
                "fired": True,
                "note": "impossible against Part 0's measured 1193/1200 at sims=48; "
                "if observed, the sweep is not measuring what Part 0 measured",
            }
        )
        return {"s_star": None, "needs": "discrepancy_finding", "branches": branches}

    bracket = (below[-1]["sims"], above[0]["sims"])
    adjacent = bracket[1] - bracket[0] <= 1
    branches.append(
        {
            "branch": "straddle",
            "fired": True,
            "bracket": list(bracket),
            "adjacent_integers": adjacent,
            "action": "FINDING: the primary cannot be sited mid-scale; a ruling is "
            "required and the window is NOT widened"
            if adjacent
            else f"bisect at sims={(bracket[0] + bracket[1]) // 2} and re-apply",
        }
    )
    return {
        "s_star": None,
        "needs": "ruling" if adjacent else "bisection",
        "bisect_at": None if adjacent else (bracket[0] + bracket[1]) // 2,
        "branches": branches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, nargs="*", default=list(SWEEP))
    args = parser.parse_args()

    cfg = eval_config()
    model, _ = load_checkpoint(REPO / "runs" / "phase1" / "phase1.pt", cfg)
    model.eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    problems_by_suite = {
        path.stem: [suite_problem(r) for r in read_suite(path)]
        for path in sorted(SUITES.glob("solve_in_*.jsonl"))
    }
    started = time.perf_counter()
    points = []
    for sims in sorted(args.sims):
        point = measure(model, cfg, sims, problems_by_suite)
        points.append(point)
        print(
            f"    sims={point['sims']:>2} m={point['gumbel_m']:>2}  "
            f"at-par {point['at_par']:>4}/{point['episodes']} = {point['at_par_rate']:.4f}  "
            f"over {point['over_par']:>4}  capped {point['capped']:>4}  "
            f"stuck {point['stuck']:>3}  {point['seconds']:>7.1f}s",
            flush=True,
        )

    selection = select(points)
    report = {
        "expectations_frozen_at": EXPECTATIONS_SHA,
        "git_sha": git_sha(REPO),
        "protocol": {
            "model": "runs/phase1/phase1.pt",
            "gumbel_m_rule": "min(16, sims), declared in PREREG-chunk11-part0bc",
            "root_noise": cfg.search.root_noise,
            "step_cap": cfg.episode.step_cap,
            "value_scale": 0.0,
            "seed": 0,
            "config_fingerprint": config_fingerprint(cfg),
            "device": "cpu",
        },
        "target": TARGET,
        "window": list(WINDOW),
        "sweep": points,
        "selection": selection,
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
    }
    write_record(REPO / "runs" / "chunk11_part0b_sweep.json", report)
    print("\n  SELECTION\n")
    print(json.dumps(selection, indent=2))
    print(f"\n  wall clock {report['wall_clock_seconds']}s")
    print("  wrote runs/chunk11_part0b_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
