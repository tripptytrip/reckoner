"""Chunk 11 Part 0: the anchor's headroom, measured before the primary is chosen.

Protocol and decision rule are frozen in `PREREG-chunk11-part0.md` at `7af32f4`,
before this script existed. Nothing here chooses a threshold; it computes T and
applies the rule that was already written.

M-A — the anchor's `steps − par` histogram over all six frozen suites, under M1's
exact eval protocol.
M-B — the anchor against the rungs on the frozen paired set, `pair_scores` from
row one, each currency in its own lane.
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch

from reckoner.arms import GreedyHeuristic, SympySolver
from reckoner.config import Config, config_fingerprint, validate
from reckoner.dataset import git_sha, read_suite, suite_problem, write_record
from reckoner.evaluate import ModelArm, model_evaluator
from reckoner.ladder import paired_bootstrap, synthetic_elo
from reckoner.ladderpass import comparison_from_pass, pair_scores_of, read_pair_scores, run_pass
from reckoner.logschema import STEPS_MINUS_PAR_BINS
from reckoner.model import load_checkpoint
from reckoner.pairedset import load
from reckoner.runner import run_iteration

REPO = Path(__file__).resolve().parents[1]
EXPECTATIONS_SHA = "7af32f4"
SUITES = REPO / "runs" / "suites"
PAIRED = REPO / "runs" / "paired" / "smoke_v1.jsonl"

#: PREREG-chunk11-part0.md, computed against no data. T >= 36 selects
#: tail-reduction; below it the primary moves to the rung trajectory.
TAIL_THRESHOLD = 36


def eval_config() -> Config:
    """M1's eval protocol. `root_noise=False` is SET, not inherited — the config
    default is the self-play value on purpose."""
    cfg = Config()
    validate(cfg)
    return replace(cfg, search=replace(cfg.search, root_noise=False))


def suite_composition(model, cfg: Config, sims: int, m: int) -> dict:
    """M-A. Batched through `run_iteration`, which already bins `steps − par`."""
    evaluator = model_evaluator(model, cfg, 0.0)
    pooled = dict.fromkeys(STEPS_MINUS_PAR_BINS, 0)
    per_suite = {}
    totals = {"episodes": 0, "solved": 0, "capped": 0, "stuck": 0}
    for path in sorted(SUITES.glob("solve_in_*.jsonl")):
        problems = [suite_problem(r) for r in read_suite(path)]
        stats = run_iteration(problems, evaluator, cfg, None, sims=sims, m=m, seed=0)
        stats.check_descent_identity()
        per_suite[path.stem] = {
            "problems": stats.episodes,
            "solved": stats.episodes_solved,
            "capped": stats.episodes_capped,
            "stuck": stats.episodes_stuck,
            "steps_minus_par": dict(stats.steps_minus_par),
            "seconds": round(stats.seconds, 2),
        }
        for k, v in stats.steps_minus_par.items():
            pooled[k] += v
        totals["episodes"] += stats.episodes
        totals["solved"] += stats.episodes_solved
        totals["capped"] += stats.episodes_capped
        totals["stuck"] += stats.episodes_stuck
        # flush: a measurement that takes tens of minutes and shows nothing is
        # indistinguishable from one that has hung, and the first thing a reader
        # does about that is kill it.
        print(
            f"    {path.stem}: {stats.episodes_solved}/{stats.episodes} solved, "
            f"steps-par {dict(stats.steps_minus_par)}",
            flush=True,
        )
    return {"per_suite": per_suite, "pooled": pooled, "totals": totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="chunk11_part0")
    args = parser.parse_args()

    cfg = eval_config()
    run = REPO / "runs" / args.name
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)

    anchor = REPO / "runs" / "phase1" / "phase1.pt"
    model, _ = load_checkpoint(anchor, cfg)
    model.eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    started = time.perf_counter()

    print("  M-A — suite z-composition under the eval protocol", flush=True)
    ma_started = time.perf_counter()
    composition = suite_composition(model, cfg, cfg.search.sims, cfg.search.gumbel_m)
    ma_seconds = round(time.perf_counter() - ma_started, 2)

    pooled = composition["pooled"]
    tail = sum(v for k, v in pooled.items() if k not in ("<0", "0"))
    total = composition["totals"]["episodes"]
    primary = "tail_reduction" if tail >= TAIL_THRESHOLD else "rung_trajectory"

    print("\n  M-B — rung baselines on the frozen paired set", flush=True)
    mb_started = time.perf_counter()
    problems = load(PAIRED, repo=REPO)
    arm = ModelArm(model)
    arm.probe(problems[0], cfg)
    greedy = GreedyHeuristic()
    greedy.probe(problems[0], cfg)

    with SympySolver(cfg) as sympy:
        if sympy.available:
            sympy.probe()
        arms = [arm, greedy] + ([sympy] if sympy.available else [])
        record = run_pass(
            run,
            0,
            arms,
            problems,
            cfg,
            roles={"model": "subject", "greedy": "baseline", "sympy": "rung"},
            calibration_note=(
                "chunk 11 Part 0 baseline: the ANCHOR, value-silent, eval profile, "
                "before any Phase-2 training. Licenses nothing about a trained model"
            ),
            seed=0,
        )
        cas_version, sympy_available = sympy.version, sympy.available
    mb_seconds = round(time.perf_counter() - mb_started, 2)

    rows = read_pair_scores(run, 0)
    model_vs_greedy = comparison_from_pass(run, 0, "model", "greedy")
    bootstrap = paired_bootstrap(model_vs_greedy.differences, resamples=10_000, seed=0)

    z_scores = pair_scores_of(rows, "model")
    z_hist = {str(z): sum(1 for s in z_scores if s.score == z) for z in (1.0, 0.0, -1.0)}
    budget_scores = pair_scores_of(rows, "sympy") if sympy_available else []

    report = {
        "expectations_frozen_at": EXPECTATIONS_SHA,
        "git_sha": git_sha(REPO),
        "protocol": {
            "model": "runs/phase1/phase1.pt",
            "sims": cfg.search.sims,
            "gumbel_m": cfg.search.gumbel_m,
            "root_noise": cfg.search.root_noise,
            "perspective": cfg.search.perspective,
            "step_cap": cfg.episode.step_cap,
            "value_scale": arm.value_scale,
            "measure_dtype": cfg.numerics.measure_dtype,
            "config_fingerprint": config_fingerprint(cfg),
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "M_A_suite_composition": composition,
        "M_A_headroom": {
            "problems": total,
            "at_par": pooled["0"],
            "beat_par": pooled["<0"],
            "beat_par_note": (
                "structurally empty: the suites carry BFS-exact par, nothing beats "
                "exact par, and three layers refuse a row claiming otherwise. "
                "Reported so its absence cannot read as nobody having looked"
            ),
            "over_par_T": tail,
            "over_par_fraction": round(tail / total, 6) if total else 0.0,
            "threshold_T": TAIL_THRESHOLD,
            "threshold_source": "PREREG-chunk11-part0.md, computed against no data",
            "primary_selected": primary,
        },
        "M_B_rung_baselines": {
            "paired_set": "runs/paired/smoke_v1.jsonl",
            "problems": len(problems),
            "rows_by_arm": record["rows_by_arm"],
            "skipped_by_arm": record["skipped_by_arm"],
            "seconds_by_arm": record["seconds_by_arm"],
            "z_lane": {
                "currency": "z_vs_par",
                "model_z_histogram": z_hist,
                "model_minus_greedy_mean": bootstrap["mean_difference"],
                "ci": [bootstrap["ci_low"], bootstrap["ci_high"]],
                "excludes_zero": bootstrap["excludes_zero"],
                "n_pairs": bootstrap["n_pairs"],
                "saturated": bootstrap["saturated"],
                "rendering_note": bootstrap["rendering_note"],
                "synthetic_elo_model_vs_greedy": round(
                    synthetic_elo(model_vs_greedy.differences), 6
                ),
            },
            "budget_lane": {
                "currency": "solve_vs_budget",
                "available": sympy_available,
                "cas_version": cas_version,
                "problems_played": len(budget_scores),
                "solved": sum(1 for s in budget_scores if s.score == 1.0),
                "note": (
                    "never differenced against the z lane; pair() refuses the "
                    "construction, so this is a separate reported baseline"
                ),
            },
        },
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
        "wall_clock_split": {"M_A_suites": ma_seconds, "M_B_paired_set": mb_seconds},
    }
    write_record(REPO / "runs" / "chunk11_part0_result.json", report)

    print("\n  HEADROOM\n")
    print(f"    pooled steps-minus-par: {pooled}")
    print(f"    at par {pooled['0']}/{total}, beat par {pooled['<0']} (structurally empty)")
    print(f"    T (over par) = {tail}  ({tail / total:.2%})   threshold {TAIL_THRESHOLD}")
    print(f"    PRIMARY SELECTED: {primary}")
    print(f"\n    model z on the paired set: {z_hist}")
    print(
        f"    model - greedy: {bootstrap['mean_difference']} "
        f"CI [{bootstrap['ci_low']}, {bootstrap['ci_high']}] "
        f"excludes_zero={bootstrap['excludes_zero']}"
    )
    print(f"\n  wall clock {report['wall_clock_seconds']}s {report['wall_clock_split']}")
    print("  wrote runs/chunk11_part0_result.json")
    shutil.rmtree(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
