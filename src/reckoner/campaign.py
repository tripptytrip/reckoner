"""The M1 campaign driver. **The** loop — `golden` is this at golden config.

Chunk 11's unbuilt half. `PREREG-m1.md` governs what it measures and refuses;
`BRIEF-chunk11-driver.md` composes it; D-A1 names the three parts that are built
rather than promoted; D-A2 fixes the rehearsal count and splits the fingerprint
assertion across its two consumers.

**One composition.** `golden.py` had the whole loop in test clothing — episodes,
ring, training, the switch row, the four-step commit — and the worst available
outcome was a fresh driver written beside it: two loops wearing one name, where
`golden` certifies a composition the campaign never runs. So the driver *is* the
loop and `golden` invokes it at golden config.

**Three parts are new, not promoted** (D-A1 §1). `golden` ran `uniform_stub`
every iteration, so the model→search→ring path — the loop's entire subject — had
never executed. Pool par had never been drawn. The criterion had never seen a
real input. Each now carries a central-job assertion in both polarities, because
an assertion that the central job *happened* standing in for one that its
**product is consumed** is the defect that hid all three.

**No behavioural flags.** Extent, treatment size and thread pins live in
`CampaignConfig`, under the fingerprint. A flag is unfingerprinted config — the
caller-choice hazard standing at the command line — and a campaign whose cadence
can be set at invocation is one that can silently diverge from its prereg.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch

from reckoner.config import Config, config_fingerprint
from reckoner.dataset import (
    anchored_data,
    git_sha,
    read_suite,
    suite_problem,
    training_problems,
)
from reckoner.evaluate import model_evaluator
from reckoner.gates import no_regress_floor
from reckoner.logschema import (
    ITERATION_FIELDS,
    SCHEMA_ERA,
    append_row,
)
from reckoner.replay import ReplayRing
from reckoner.resume import RunState
from reckoner.rules import RULESET_VERSION
from reckoner.runner import iteration_row, run_iteration
from reckoner.vocab import VOCAB_VERSION

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "runs" / "suites"

# --- the two fingerprints, recorded in PREREG-m1 §M1-A2 §1 -------------------
# The driver runs TWO profiles and therefore makes TWO assertions, each pinned
# where its profile acts (D-A2 §2). A single startup check would compare one
# profile's value against a run that also uses the other.
CAMPAIGN_FINGERPRINT = "ce41af96ee85f0a29c90db508ef19c21e11946c95318b8957f5800425e61bb0b"
EVAL_FINGERPRINT = "314fbeb99b6640f65fc1bc05082113de1647a01781ed93aadf6ad13e7a35f139"

#: PREREG-m1 §4: both indistinguishability floors, on the 1,200-problem instrument.
NO_REGRESS = {48: 1188, 1: 1167}

#: PREREG-m1 §2: the primary's population.
SUCCESSOR_STRATA = (7, 8, 10)

#: The OMP family was UNSET throughout the licence. Unset is a value (M1-A2 §4).
OMP_FAMILY = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class CampaignRefusal(RuntimeError):
    """The campaign will not start, or will not continue, and says why."""


# --------------------------------------------------------------------- profiles


def eval_profile(cfg: Config) -> Config:
    """The measurement profile: root noise off, everything else identical."""
    return replace(cfg, search=replace(cfg.search, root_noise=False))


def assert_campaign_profile(cfg: Config) -> str:
    """Once, at startup. The frozen page becomes executable authority."""
    got = config_fingerprint(cfg)
    if got != CAMPAIGN_FINGERPRINT:
        raise CampaignRefusal(
            f"campaign config fingerprint {got} != PREREG-m1's recorded "
            f"{CAMPAIGN_FINGERPRINT}. This config is not the one the campaign was "
            "registered against, so the run would not be the registered run."
        )
    return got


def assert_eval_profile(cfg: Config) -> str:
    """At **every** instrument pass, not once at startup.

    The cadence measurements are comparable to their baselines only if they
    provably ran the eval profile, so the check belongs at the boundary where
    that profile acts.
    """
    got = config_fingerprint(cfg)
    if got != EVAL_FINGERPRINT:
        raise CampaignRefusal(
            f"eval profile fingerprint {got} != PREREG-m1's recorded "
            f"{EVAL_FINGERPRINT}; this pass is not comparable to its baseline."
        )
    return got


def assert_threads(cfg: Config) -> dict:
    """The pins, applied and verified — including the recorded absence."""
    torch.set_num_threads(cfg.campaign.intra_op_threads)
    present = {v: os.environ[v] for v in OMP_FAMILY if v in os.environ}
    if present:
        raise CampaignRefusal(
            f"the OMP family was UNSET throughout the licence and is set here: "
            f"{present}. Unset is a value; setting it is a gated configuration "
            "change whose reproduction gate re-runs, not a tidy-up."
        )
    return {
        "intra_op": torch.get_num_threads(),
        "interop": torch.get_num_interop_threads(),
        "omp_family": "unset",
    }


# ------------------------------------------------------------ the instrument seam


def run_instruments(model, cfg: Config, *, iteration: int, anchor_beat: int) -> dict:
    """**The** instrument seam. Every cadence measurement goes through here.

    Mono-instance because the discipline requires it, and because Lever B —
    unit-parallel instrument passes — lands behind this signature later without
    the driver noticing.

    Asserts the eval fingerprint on entry: this is the boundary where the eval
    profile acts.
    """
    ev = eval_profile(cfg)
    fingerprint = assert_eval_profile(ev)
    evaluator = model_evaluator(model, ev, 0.0)
    out: dict = {"iteration": iteration, "config_fingerprint": fingerprint, "profile": "eval"}

    # --- no-regress, both budgets, against the declared floors ---------------
    suites = sorted(SUITES.glob("solve_in_*.jsonl"))
    instrument = [suite_problem(r) for p in suites for r in read_suite(p)]
    for sims, floor in sorted(NO_REGRESS.items(), reverse=True):
        stats = run_iteration(
            instrument, evaluator, ev, None, sims=sims, m=min(ev.search.gumbel_m, sims), seed=0
        )
        stats.check_descent_identity()
        at_par = stats.steps_minus_par["0"]
        out[f"no_regress_sims_{sims}"] = {
            "at_par": at_par,
            "of": stats.episodes,
            "floor": floor,
            "held": at_par >= floor,
            "floor_kind": "indistinguishability",
            "licensed_sentence": "not below the anchor's own one-sided 95% band",
            "floor_recomputed": no_regress_floor(1193 if sims == 48 else 1176, 1200),
        }

    # --- the primary: pooled beat-par on {7, 8, 10} --------------------------
    per_stratum, pooled_beat, pooled_n = {}, 0, 0
    for k in SUCCESSOR_STRATA:
        problems = [suite_problem(r) for r in read_suite(SUITES / f"scripted_in_{k}.jsonl")]
        stats = run_iteration(problems, evaluator, ev, None, sims=48, m=16, seed=0)
        stats.check_descent_identity()
        beat = stats.steps_minus_par["<0"]
        per_stratum[f"scripted_in_{k}"] = {"beat": beat, "of": stats.episodes}
        pooled_beat += beat
        pooled_n += stats.episodes
    out["primary"] = {
        "pooled_beat_par": f"{pooled_beat}/{pooled_n}",
        "pooled_rate": round(pooled_beat / pooled_n, 6),
        "anchor_baseline": f"{anchor_beat}/{pooled_n}",
        "delta_vs_anchor": round((pooled_beat - anchor_beat) / pooled_n, 6),
        "per_stratum": per_stratum,
    }
    return out


# ---------------------------------------------------------------- the pre-flight


def preflight(cfg: Config, model, scratch: Path) -> None:
    """Every row class, at micro scale, before iteration 0 spends anything.

    D-A1 §3. Not a flag: a script that smokes its output path only when asked is
    a script that will be run without asking.
    """
    started = time.perf_counter()
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    ring = ReplayRing(256, cfg)
    problems = training_problems(anchored_data("train_100k"), 2, seed=0)
    stats = run_iteration(
        problems, model_evaluator(model, cfg, 0.0), cfg, ring, sims=1, m=1, seed=0
    )
    row = iteration_row(
        stats,
        iteration=0,
        run_name="preflight",
        git_sha=git_sha(REPO),
        config_fingerprint=config_fingerprint(cfg),
        cfg=cfg,
        ruleset_version=RULESET_VERSION,
        vocab_version=VOCAB_VERSION,
        schema_era=SCHEMA_ERA,
    )
    append_row(scratch / "iterations.jsonl", row, ITERATION_FIELDS)
    head_state = RunState(iteration=0, value_head=None, seed=0)
    del head_state
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"  pre-flight OK ({time.perf_counter() - started:.1f}s)")
