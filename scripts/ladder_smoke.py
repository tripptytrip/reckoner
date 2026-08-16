"""The chunk-10 smoke pass: one full ladder pass, every number verbatim.

Expectations S1-S7 are read from `PREREG-chunk10-smoke.md`, frozen at `8f30cb7`
before this script existed. **An expectation that fails is a finding with a
verdict, never an adjustment.**

Unlike the chunk-9 shakedown, the paired set is **kept**. It is an instrument;
its whole purpose is to be the same set next time. What is deleted is the pass's
run directory, which is evidence of plumbing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from reckoner.arms import GreedyHeuristic, RandomRewriter, SympySolver
from reckoner.config import Config, validate
from reckoner.dataset import anchored_data, git_sha, read_dataset, sample_indices, write_record
from reckoner.episode import Problem, decode_state
from reckoner.ladder import paired_bootstrap, rigged_null, self_match, synthetic_elo
from reckoner.ladderpass import comparison_from_pass, is_complete, read_pair_scores, run_pass
from reckoner.pairedset import PairedSetError, census, freeze, load, source_census_keys

REPO = Path(__file__).resolve().parents[1]
EXPECTATIONS_SHA = "8f30cb7"
PAIRED = REPO / "runs" / "paired" / "smoke_v1.jsonl"

#: Floor for a paired difference in z. z in {-1, 0, +1}, so the difference of two
#: z values is bounded below by -2. Computed, not chosen — rider (c).
Z_DIFFERENCE_FLOOR = -2.0


def candidates(count: int, seed: int) -> list[Problem]:
    """Drawn from the held-out set, through the blessed subsampler.

    `range(count)` on a stratum-ordered set is the whole shallowest stratum and
    none of the rest (F-03, F-10). `sample_indices` is the one subsampler.
    """
    dataset = read_dataset(anchored_data("eval_held_out"))
    out = []
    for i in sample_indices(len(dataset), count, seed):
        goal, target, expr = decode_state(dataset.state(i))
        out.append(
            Problem(
                goal=goal,
                expr=expr,
                par=int(dataset.par[i]),
                target=target,
                par_source="bfs",
            )
        )
    return out


def build_paired_set(count: int, seed: int, verdicts: dict) -> tuple[list[Problem], float]:
    """S1 + S2. Censused at both levels, then frozen once."""
    started = time.perf_counter()
    if PAIRED.exists():
        problems = load(PAIRED, repo=REPO)
        verdicts["S1"] = {
            "expectation": "frozen at birth, verified at read",
            "note": "already frozen; re-loaded and digest-verified",
            "problems": len(problems),
            "verdict": "PASS",
        }
        verdicts["S2"] = {
            "expectation": "censused at both levels before the freeze",
            "note": "census ran at the freeze; see runs/paired_census.json",
            "verdict": "PASS" if (REPO / "runs" / "paired_census.json").exists() else "FAIL",
        }
        return problems, round(time.perf_counter() - started, 2)

    pool = candidates(count, seed)
    print(f"  censusing {len(pool)} candidates at both levels…")
    problem_level = {"train_100k": source_census_keys(anchored_data("train_100k"))}
    print(f"    problem-level reference: {len(problem_level['train_100k'])} keys")
    state_level = {"phase1_train": source_census_keys(anchored_data("phase1_train"))}
    print(f"    state-level reference:   {len(state_level['phase1_train'])} keys")

    result = census(pool, problem_sources=problem_level, state_sources=state_level)
    clean = [pool[i] for i in result.clean_indices]
    write_record(REPO / "runs" / "paired_census.json", result.as_dict() | {"seed": seed})

    if not clean:
        raise SystemExit("every candidate collided — that is a finding, not a smaller set")
    digest = freeze(PAIRED, clean, repo=REPO)

    verdicts["S1"] = {
        "expectation": "frozen at birth, verified at read",
        "digest": digest,
        "reloads_verified": len(load(PAIRED, repo=REPO)) == len(clean),
        "second_freeze_refused": _second_freeze_refused(clean),
        "verdict": "PASS" if _second_freeze_refused(clean) else "FAIL",
    }
    verdicts["S2"] = {
        "expectation": "censused at both levels; rule pre-stated; drop before the freeze",
        **result.as_dict(),
        "verdict": "PASS",
    }
    return clean, round(time.perf_counter() - started, 2)


def _second_freeze_refused(problems: list[Problem]) -> bool:
    try:
        freeze(PAIRED, problems, repo=REPO)
    except PairedSetError:
        return True
    return False


def kill_point_resume(root: Path, paired: Path, verdicts: dict) -> float:
    """S5. A **real** SIGKILL, then a resume that must reproduce the pass exactly."""
    started = time.perf_counter()
    reference_root = root / "reference"
    problems = load(paired, repo=REPO)
    run_pass(
        reference_root,
        0,
        [GreedyHeuristic(), RandomRewriter()],
        problems,
        Config(),
        roles={"greedy": "baseline", "random": "baseline"},
        calibration_note="smoke pass, kill-point child",
    )
    reference = read_pair_scores(reference_root, 0)

    killed_root = root / "killed"
    kill_after = max(3, len(reference) // 3)
    child = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "scripts" / "ladder_pass_child.py"),
            "--root",
            str(killed_root),
            "--paired",
            str(paired),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    seen = 0
    for line in child.stdout:
        if line.startswith("READY"):
            seen = int(line.split()[1])
            if seen >= kill_after:
                child.send_signal(signal.SIGKILL)
                break
    child.wait()

    partial = read_pair_scores(killed_root, 0)
    resumed_record = run_pass(
        killed_root,
        0,
        [GreedyHeuristic(), RandomRewriter()],
        problems,
        Config(),
        roles={"greedy": "baseline", "random": "baseline"},
        calibration_note="smoke pass, kill-point child",
    )
    resumed = read_pair_scores(killed_root, 0)

    identical = resumed == reference
    units = [(r["arm"], r["problem_key"]) for r in resumed]
    verdicts["S5"] = {
        "expectation": "a SIGKILLed pass resumes to an identical row set",
        "signal": "SIGKILL",
        "killed_after_units": seen,
        "rows_surviving_the_kill": len(partial),
        "rows_after_resume": len(resumed),
        "rows_in_uninterrupted_run": len(reference),
        "rows_resumed": resumed_record["rows_resumed"],
        "duplicate_units": len(units) - len(set(units)),
        "identical_to_uninterrupted": identical,
        "marker_present": is_complete(killed_root, 0),
        "verdict": "PASS"
        if identical and len(units) == len(set(units)) and 0 < len(partial) < len(reference)
        else "FAIL",
    }
    return round(time.perf_counter() - started, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="ladder_smoke")
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    validate(cfg)
    run = REPO / "runs" / args.name
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)

    verdicts: dict[str, dict] = {}
    started = time.perf_counter()

    problems, census_seconds = build_paired_set(args.candidates, args.seed, verdicts)
    print(f"  paired set: {len(problems)} problems ({census_seconds}s)")

    # --- S3 + S4: one full pass against all rungs --------------------------
    pass_started = time.perf_counter()
    row_counts_during: list[int] = []

    def watch(arm: str, i: int) -> None:
        if arm == "greedy" and i in (1, 5):
            row_counts_during.append(len(read_pair_scores(run, 0)))

    with SympySolver(cfg) as sympy:
        if sympy.available:
            sympy.probe()
        arms = [GreedyHeuristic(), RandomRewriter()] + ([sympy] if sympy.available else [])
        roles = {"greedy": "baseline", "random": "baseline", "sympy": "rung"}
        record = run_pass(
            run,
            0,
            arms,
            problems,
            cfg,
            roles={a.name: roles[a.name] for a in arms},
            calibration_note=(
                "smoke pass over a frozen paired set; scripted-par rungs only, no "
                "model in this pass — these numbers license nothing about the model"
            ),
            seed=args.seed,
            on_unit=watch,
        )
        cas_version = sympy.version
        sympy_available = sympy.available
    pass_seconds = round(time.perf_counter() - pass_started, 2)

    verdicts["S3"] = {
        "expectation": "each arm scored completely on its OWN declared subset",
        "rows_by_arm": record["rows_by_arm"],
        "skipped_by_arm": record["skipped_by_arm"],
        "problems": len(problems),
        "sympy_available": sympy_available,
        "cas_version": cas_version,
        "verdict": "PASS",  # run_pass raises on a short arm; reaching here is the pass
    }
    verdicts["S4"] = {
        "expectation": "pair_scores land from row one, not at the end",
        "rows_on_disk_at_greedy_units_1_and_5": row_counts_during,
        "verdict": "PASS" if row_counts_during == [1, 5] else "FAIL",
    }

    # --- S6: the NULL RUN, and its contrast --------------------------------
    null_started = time.perf_counter()
    greedy = GreedyHeuristic()
    random_arm = RandomRewriter()

    # The arms' own results go straight through: `self_match` computes z from the
    # problem's par. The caller no longer chooses a currency (F-19).
    null_run = self_match(greedy.play, problems, cfg, profile="eval")
    null_bootstrap = paired_bootstrap(null_run.differences, resamples=2000, seed=args.seed)
    contrast = self_match(random_arm.play, problems, cfg, profile="self_play")
    null_seconds = round(time.perf_counter() - null_started, 2)

    verdicts["S6"] = {
        "expectation": "self-match under eval is EXACTLY zero; self-play is not",
        "floor": Z_DIFFERENCE_FLOOR,
        "null": 0.0,
        "null_is_a_run": True,
        "threshold": "every paired difference exactly 0, bootstrap saturated",
        "measured_max_abs_difference": max(abs(d) for d in null_run.differences),
        "measured_mean": null_bootstrap["mean_difference"],
        "measured_saturated": null_bootstrap["saturated"],
        "contrast_nonzero_differences": sum(1 for d in contrast.differences if d != 0.0),
        "rigged_null_synthetic_elo": synthetic_elo(rigged_null(len(problems))),
        "verdict": "PASS"
        if all(d == 0.0 for d in null_run.differences)
        and null_bootstrap["saturated"]
        and any(d != 0.0 for d in contrast.differences)
        else "FAIL",
    }

    # --- S7: greedy vs random, four-tupled ---------------------------------
    comparison = comparison_from_pass(run, 0, "greedy", "random")
    bootstrap = paired_bootstrap(comparison.differences, resamples=10_000, seed=args.seed)
    clears = bootstrap["excludes_zero"] and bootstrap["mean_difference"] > 0
    verdicts["S7"] = {
        "expectation": "greedy > random, CI excludes zero (direction pre-stated)",
        "metric": "mean paired difference in z, greedy - random",
        "floor": Z_DIFFERENCE_FLOOR,
        "null": 0.0,
        "threshold": "95% CI excludes zero AND mean > 0",
        "measured": bootstrap["mean_difference"],
        "ci": [bootstrap["ci_low"], bootstrap["ci_high"]],
        "n_pairs": bootstrap["n_pairs"],
        "saturated": bootstrap["saturated"],
        "rendering_note": bootstrap["rendering_note"],
        "synthetic_elo_greedy_vs_random": round(synthetic_elo(comparison.differences), 6),
        "verdict": "PASS" if clears else "FINDING",
        "note": (
            "chunk 10's DONE-WHEN does not depend on this direction; a FINDING here "
            "is a fact about the heuristic, pre-stated as such in PREREG-chunk10-smoke.md"
        ),
    }

    kill_seconds = kill_point_resume(run / "resume", PAIRED, verdicts)

    elapsed = round(time.perf_counter() - started, 2)
    report = {
        "expectations_frozen_at": EXPECTATIONS_SHA,
        "git_sha": git_sha(REPO),
        "paired_set": str(PAIRED.relative_to(REPO)),
        "paired_set_size": len(problems),
        "candidates_drawn": args.candidates,
        "seed": args.seed,
        "wall_clock_seconds": elapsed,
        "wall_clock_split": {
            "census_and_freeze": census_seconds,
            "pass": pass_seconds,
            "pass_by_arm": record["seconds_by_arm"],
            "null_run_and_contrast": null_seconds,
            "kill_point_resume": kill_seconds,
        },
        "pass_record": record,
        "verdicts": verdicts,
    }
    write_record(REPO / "runs" / "ladder_smoke_result.json", report)

    print("\n  LADDER SMOKE — expectations frozen at " + EXPECTATIONS_SHA + "\n")
    for name in sorted(verdicts):
        print(f"    {name}: {verdicts[name]['verdict']}")
    print(f"\n  wall clock {elapsed}s  {json.dumps(report['wall_clock_split'])}")
    print("  wrote runs/ladder_smoke_result.json")

    shutil.rmtree(run)
    print(f"  deleted {run} after recording; the paired set is KEPT (it is an instrument)")
    return 0 if all(v["verdict"] in ("PASS", "FINDING") for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
