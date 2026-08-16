"""BRIEF-m1-host item 4: the campaign's measured per-phase cost, on the pod.

**Protocol pinned before this runs**, per the house pattern.

1. **Fingerprint.** Cites the PRE-A2 config — the same arrangement every Part-0
   measurement used — so M1-A2 can truthfully say its derivation's inputs ran
   under the prior config rather than under the one it introduces.

2. **The evaluator is the anchor, by digest, asserted.** These are the first
   model-in-search episodes this project has generated: `golden` has always run
   `uniform_stub`, so the model→search→ring path has never been timed. The
   campaign runs model-in-search, so model-in-search is the timing that matters —
   stub timings would size a campaign for a loop it will not run.

3. **The episode count is MEASUREMENT SIZE, not treatment.** ``PROBE_EPISODES``
   buys per-episode precision and is nobody's campaign value; the campaign's
   ``episodes_per_iteration`` is *derived* from the cost this measures, on
   M1-A2's page.

4. **Per-phase splits**, because the campaign total is a sum of three very
   different terms: per-episode self-play, per-train-step, and the whole
   every-fifth-iteration eval bill — the last of which dominates and is the term
   the thermal cliff used to own.

5. **Threads at the licensed ambients** — intra-op 8, interop 32, OMP family
   unset — which incidentally re-exercises them on this host before M1-A2
   formalises them as pins.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from reckoner.arms import GreedyHeuristic
from reckoner.config import Config, config_fingerprint, validate
from reckoner.dataset import (
    anchored_data,
    git_sha,
    read_suite,
    sha256_file,
    suite_problem,
    training_problems,
    write_record,
)
from reckoner.evaluate import model_evaluator
from reckoner.ladderpass import run_pass
from reckoner.model import Reckoner, load_checkpoint
from reckoner.replay import ReplayRing
from reckoner.runner import run_iteration
from reckoner.train import train_on_ring

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"

#: MEASUREMENT SIZE. Not a campaign value, not a treatment parameter.
PROBE_EPISODES = 100
PROBE_TRAIN_STEPS = 25
PROBE_LADDER_PROBLEMS = 25

#: The anchor these timings are of, asserted rather than assumed.
ANCHOR_SHA256 = "45333caa8a066b0e8a1d3213aca48470d47dbcafaf3053b1bca4dfa54e6e269b"

#: The licensed thread configuration. Recorded here; pinned by M1-A2.
LICENSED_THREADS = {"intra_op": 8, "interop": 32, "omp_family": "unset"}


def thread_report() -> dict:
    return {
        "intra_op": torch.get_num_threads(),
        "interop": torch.get_num_interop_threads(),
        "omp_env": {
            v: os.environ.get(v, "<unset>")
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def main() -> int:
    cfg = Config()
    validate(cfg)
    fingerprint = config_fingerprint(cfg)

    anchor = REPO / "runs" / "phase1" / "phase1.pt"
    digest = sha256_file(anchor)
    if digest != ANCHOR_SHA256:
        raise SystemExit(f"anchor digest {digest} != declared {ANCHOR_SHA256}; refusing to time")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    model, _ = load_checkpoint(anchor, cfg)
    model.eval()
    print(f"  anchor {digest[:16]}  fingerprint {fingerprint[:16]}")
    print(f"  threads {thread_report()['intra_op']}/{thread_report()['interop']}\n")

    out: dict = {
        "purpose": "BRIEF-m1-host item 4 — measured per-phase campaign cost",
        "git_sha": git_sha(REPO),
        "host": os.uname().nodename,
        "protocol": {
            "config_fingerprint": fingerprint,
            "fingerprint_era": "PRE-M1-A2 (the arrangement every Part-0 measurement used)",
            "anchor_sha256": digest,
            "evaluator": "model_evaluator(anchor) — model-in-search, NOT uniform_stub",
            "sims": cfg.search.sims,
            "gumbel_m": cfg.search.gumbel_m,
            "root_noise": cfg.search.root_noise,
            "threads_observed": thread_report(),
            "threads_licensed": LICENSED_THREADS,
            "probe_sizes_are_measurement_not_treatment": {
                "episodes": PROBE_EPISODES,
                "train_steps": PROBE_TRAIN_STEPS,
                "ladder_problems": PROBE_LADDER_PROBLEMS,
            },
        },
    }

    # ---- phase 1: self-play, model-in-search --------------------------------
    print("  [1/3] self-play, model-in-search")
    ring = ReplayRing(cfg.train.replay_capacity, cfg)
    problems = training_problems(anchored_data("train_100k"), PROBE_EPISODES, seed=0)
    evaluator = model_evaluator(model, cfg, 0.0)
    t0 = time.perf_counter()
    stats = run_iteration(problems, evaluator, cfg, ring, seed=0)
    selfplay_s = time.perf_counter() - t0
    stats.check_descent_identity()
    out["self_play"] = {
        "episodes": stats.episodes,
        "solved": stats.episodes_solved,
        "seconds": round(selfplay_s, 2),
        "seconds_per_episode": round(selfplay_s / stats.episodes, 4),
        "ring_rows": len(ring),
        "ring_rows_per_episode": round(len(ring) / stats.episodes, 3),
    }
    print(
        f"        {selfplay_s:.1f}s / {stats.episodes} ep = "
        f"{selfplay_s / stats.episodes:.3f}s per episode, {len(ring)} ring rows"
    )

    # ---- phase 2: training --------------------------------------------------
    print("  [2/3] training")
    fresh = Reckoner(cfg)
    t0 = time.perf_counter()
    train_on_ring(fresh, ring, cfg, steps=PROBE_TRAIN_STEPS, seed=0)
    train_s = time.perf_counter() - t0
    out["train"] = {
        "steps": PROBE_TRAIN_STEPS,
        "seconds": round(train_s, 2),
        "seconds_per_step": round(train_s / PROBE_TRAIN_STEPS, 4),
        "campaign_steps_per_iteration": cfg.train.train_steps_per_iter,
        "projected_seconds_per_iteration": round(
            train_s / PROBE_TRAIN_STEPS * cfg.train.train_steps_per_iter, 1
        ),
    }
    print(
        f"        {train_s:.1f}s / {PROBE_TRAIN_STEPS} steps = "
        f"{train_s / PROBE_TRAIN_STEPS:.3f}s per step"
    )

    # ---- phase 3: the cadence unit — one ladder pass ------------------------
    print("  [3/3] ladder pass (the every-fifth-iteration bill)")
    rows = read_suite(SUITES / "solve_in_3.jsonl")[:PROBE_LADDER_PROBLEMS]
    ladder_problems = [suite_problem(r) for r in rows]
    root = REPO / "runs" / "_pilot_ladder"
    t0 = time.perf_counter()
    run_pass(
        root,
        0,
        [GreedyHeuristic(cfg)],
        ladder_problems,
        cfg,
        roles={"greedy": "rule_denominated"},
        calibration_note="pilot cost probe, not a ladder result",
        seed=0,
    )
    ladder_s = time.perf_counter() - t0
    per_unit = ladder_s / len(ladder_problems)
    out["ladder"] = {
        "arms_timed": ["greedy"],
        "problems": len(ladder_problems),
        "seconds": round(ladder_s, 2),
        "seconds_per_arm_problem": round(per_unit, 4),
        "campaign_problems_per_pass": cfg.ladder.problems_per_pass,
        "note": "one arm timed; the campaign's pass runs every enrolled arm, so "
        "the pass total scales with arm count — stated rather than extrapolated",
    }
    print(
        f"        {ladder_s:.1f}s / {len(ladder_problems)} problems (1 arm) = "
        f"{per_unit:.3f}s per arm-problem"
    )

    write_record(REPO / "runs" / "chunk11_pilot_campaign_cost.json", out)
    print("\n  wrote runs/chunk11_pilot_campaign_cost.json")
    print(json.dumps({k: out[k] for k in ("self_play", "train", "ladder")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
