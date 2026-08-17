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
import random
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch

from reckoner.config import Config, config_fingerprint, validate
from reckoner.dataset import (
    anchored_data,
    git_sha,
    read_suite,
    sha256_file,
    suite_problem,
    training_problems,
)
from reckoner.episode import Problem
from reckoner.evaluate import model_evaluator
from reckoner.gates import no_regress_floor
from reckoner.logschema import (
    ITERATION_FIELDS,
    SCHEMA_ERA,
    VALUE_SWITCH_FIELDS,
    alarm_census,
    append_row,
    read_rows,
    switch_event_row,
)
from reckoner.model import load_checkpoint, save_checkpoint
from reckoner.pool import CheckpointPool
from reckoner.replay import ReplayRing
from reckoner.resume import RunState, commit_iteration, latest_committed, resume
from reckoner.rules import RULESET_VERSION
from reckoner.runner import iteration_row, run_iteration
from reckoner.train import train_on_ring
from reckoner.valuegate import (
    ValueHeadState,
    consider_switch,
    evaluate_head,
    value_contribution,
)
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

#: The Phase-1 anchor: rung zero of the pool, and iteration 0's evaluator.
ANCHOR = REPO / "runs" / "phase1" / "phase1.pt"

#: PREREG-m1 §2.1: the anchor's measured baseline on {7, 8, 10}, from Part-0d —
#: 43 + 26 + 32 beats of 600. The primary is CI-separated improvement over this.
ANCHOR_BEAT = 101

#: The OMP family was UNSET throughout the licence. Unset is a value (M1-A2 §4).
OMP_FAMILY = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class CampaignRefusal(RuntimeError):
    """The campaign will not start, or will not continue, and says why."""


def golden_config(**overrides) -> Config:
    """Golden scale: small enough to be a liveness check, **real otherwise**.

    Lives in the package rather than in tests because it is a config, and
    `golden` is a production script. Scale is declared here rather than passed as
    flags — the no-behavioural-flags rule is about where a behaviour is declared,
    not about who is running.

    This is deliberately NOT the campaign's config: the campaign's door refuses
    it, which is that check working rather than failing.
    """
    cfg = Config()
    validate(cfg)
    cfg = replace(
        cfg,
        campaign=replace(cfg.campaign, iterations=2, episodes_per_iteration=12),
        train=replace(cfg.train, train_steps_per_iter=2),
        search=replace(cfg.search, sims=4, gumbel_m=4),
        ladder=replace(cfg.ladder, ladder_every=99),
    )
    for group, changes in overrides.items():
        cfg = replace(cfg, **{group: replace(getattr(cfg, group), **changes)})
    return cfg


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


# ------------------------------------------------------------------ episodes


def campaign_problems(
    cfg: Config, pool: CheckpointPool, model, k: int, seed: int
) -> tuple[list[Problem], int]:
    """Episode sources, with **pool par actually drawn** (D-A1 §1.2).

    `golden` enrolled snapshots and asserted the pool grows, and never once drew
    a par from it — so `par_from_pool_frac` had no exercise in any composition.
    Here it does: `league.par_from_pool_frac` of episodes take their par from a
    snapshot, which is the mechanism par escalation *is*.

    `training_problems` refuses a frozen instrument first thing, so the runtime
    boundary the contamination censuses could not reach is held here too.
    """
    problems = training_problems(anchored_data("train_100k"), k, seed=seed)
    rng = random.Random(seed * 7919 + 13)

    def factory(snapshot_model, value_scale):
        return model_evaluator(snapshot_model, cfg, value_scale)

    out, from_pool = [], 0
    for problem in problems:
        if rng.random() >= cfg.league.par_from_pool_frac:
            out.append(problem)
            continue
        drawn = pool.par_for_episode(problem, factory, rng)
        out.append(replace(problem, par=drawn.par, par_source=drawn.par_source))
        from_pool += drawn.par_source == "pool"
    return out, from_pool


# ----------------------------------------------------------------------- run


def run_campaign(run_dir: Path, cfg: Config, *, anchor: Path | None = None) -> dict:
    """The campaign's door. **This** is where the registered config is enforced.

    The assertion lives here rather than inside :func:`run` because `run` is the
    shared loop and `golden` invokes it at golden config *deliberately* — one
    composition, two callers. Putting the check inside the loop would force
    `golden` to either fail or route around it, and a routed-around check is the
    two-loops outcome arriving by the back door.

    So: the loop is shared, and the campaign's *entrypoint* is what refuses a
    config the frozen page did not bless.
    """
    assert_campaign_profile(cfg)
    return run(run_dir, cfg, run_name="m1", anchor=anchor)


def run(
    run_dir: Path,
    cfg: Config,
    *,
    run_name: str = "m1",
    anchor: Path | None = None,
    on_commit=None,
) -> dict:
    """**The** loop. `golden` is this at golden config; the campaign is
    :func:`run_campaign`, which asserts the registered fingerprint first.

    ``on_commit(iteration, phase)`` is called at ``"before_row"`` and
    ``"before_latest"`` — the two kill points the resume gate uses, and the only
    reason it exists. Same shape and same justification as
    :func:`ladderpass.run_pass`'s ``on_unit``: a real SIGKILL needs a real
    process, and a process cannot be killed at a boundary it does not announce.

    The iteration's artifacts commit through :func:`resume.commit_iteration` in
    the four-step order that module specifies — ring, state, row, then ``LATEST``
    by atomic rename. Only the rename commits; steps 1–3 are provisional, and a
    process killed anywhere in them leaves artifacts resume discards.
    """
    threads = assert_threads(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    anchor = anchor or (REPO / "runs" / "phase1" / "phase1.pt")
    start, ring, state = resume(run_dir, cfg)
    if ring is None:
        ring = ReplayRing(cfg.train.replay_capacity, cfg)
        state = RunState(iteration=0, value_head=ValueHeadState(), seed=cfg.seed)
    model, _ = load_checkpoint(anchor, cfg)

    # The evaluator's provenance: at iteration 0 the weights are the anchor's; on
    # resume they are the last committed checkpoint's. Either way the digest is
    # of the FILE the search evaluator was built from (D-A1 §1.1).
    evaluator_source = anchor
    if start > 0:
        resumed = run_dir / f"ckpt-{start - 1}.pt"
        if resumed.exists():
            model, _ = load_checkpoint(resumed, cfg)
            evaluator_source = resumed

    # SEED THE POOL WITH THE ANCHOR. `league.seed_pool_with_anchor` was honoured
    # in exactly one script and by no driver, so `par_from_pool_frac` would have
    # been dead through iteration 0 and thin after — the enrolment pattern one
    # level down: the class supports seeding, the config declares it, the
    # consumer never called it. The anchor is rung zero; without it there is
    # nothing to escalate FROM, and M1-A2 §6's two-population split would have
    # been 100/0 rather than 80/20 for the campaign's opening iterations.
    pool = CheckpointPool(cfg)
    if cfg.league.seed_pool_with_anchor and anchor.exists():
        pool.add(anchor)
    summary: dict = {"run_name": run_name, "threads": threads, "iterations": []}

    for n in range(start, cfg.campaign.iterations):
        began = time.perf_counter()
        digest = sha256_file(evaluator_source)

        # --- self-play, MODEL-IN-SEARCH ------------------------------------
        scale = value_contribution(state.value_head)
        problems, from_pool = campaign_problems(
            cfg, pool, model, cfg.campaign.episodes_per_iteration, seed=n
        )
        t0 = time.perf_counter()
        stats = run_iteration(problems, model_evaluator(model, cfg, scale), cfg, ring, seed=n)
        seconds_self_play = time.perf_counter() - t0
        stats.check_descent_identity()

        # --- training -------------------------------------------------------
        t0 = time.perf_counter()
        train_stats = train_on_ring(
            model,
            ring,
            cfg,
            steps=cfg.train.train_steps_per_iter,
            seed=n,
            value_head=state.value_head,
        )
        seconds_train = time.perf_counter() - t0

        # --- checkpoint, then enrol: par escalates with the model ------------
        checkpoint = run_dir / f"ckpt-{n}.pt"
        save_checkpoint(checkpoint, model, cfg, n, value_head=state.value_head.as_dict())
        pool.enroll(model, n, state.value_head, run_dir / f"snap-{n}.pt")

        # --- the criterion, on REAL accrued holdout (D-A1 §1.3) -------------
        slots = sorted(ring.holdout(cfg.train.ring_holdout_frac, seed=0))
        labels, predictions = evaluate_head(model, ring, cfg, slots)
        head, event = consider_switch(state.value_head, labels, predictions, iteration=n)
        append_row(
            run_dir / "value_switch.jsonl",
            switch_event_row(event, schema_era=SCHEMA_ERA),
            VALUE_SWITCH_FIELDS,
        )

        # --- the cadence unit ------------------------------------------------
        absent: dict[str, str] = {}
        ladder_index = None
        if (n + 1) % cfg.ladder.ladder_every == 0:
            ladder_index = n // cfg.ladder.ladder_every
            summary.setdefault("instruments", []).append(
                run_instruments(model, cfg, iteration=n, anchor_beat=ANCHOR_BEAT)
            )
        else:
            absent["ladder_pass"] = (
                "not a ladder iteration (ladder runs on ladder.ladder_every cadence)"
            )
        if from_pool == 0 and not pool.members:
            absent["pool_par_fraction"] = (
                "league.par_from_pool_frac is 0, or no snapshot has been taken yet"
            )

        # --- the row, then the four-step commit ------------------------------
        row = iteration_row(
            stats,
            iteration=n,
            run_name=run_name,
            git_sha=git_sha(REPO),
            config_fingerprint=CAMPAIGN_FINGERPRINT,
            cfg=cfg,
            ruleset_version=RULESET_VERSION,
            vocab_version=VOCAB_VERSION,
            schema_era=SCHEMA_ERA,
            evaluator_checkpoint_sha256=digest,
            seconds_train=seconds_train,
            absent=absent or None,
        )
        row["seconds_self_play"] = round(seconds_self_play, 3)
        row["seconds_total"] = round(time.perf_counter() - began, 3)
        row["nan_skips"] = train_stats.nan_skips
        row["pool_refusals"] = pool.stats.refusals
        if ladder_index is not None:
            row["ladder_pass"] = ladder_index
        if "pool_par_fraction" not in absent:
            row["pool_par_fraction"] = round(from_pool / max(1, stats.episodes), 6)

        state = RunState(iteration=n, value_head=head, seed=state.seed)

        def write_row(row=row, n=n):
            if on_commit is not None:
                on_commit(n, "before_row")
            append_row(run_dir / "iterations.jsonl", row, ITERATION_FIELDS)
            if on_commit is not None:
                on_commit(n, "before_latest")

        commit_iteration(run_dir, ring, state, write_row)
        evaluator_source = checkpoint
        summary["iterations"].append(
            {
                "iteration": n,
                "solved": f"{stats.episodes_solved}/{stats.episodes}",
                "pool_par": from_pool,
                "ring": len(ring),
                "switch": "fired"
                if event.get("fired")
                else ("abstained" if event.get("abstained") else "refused"),
                "seconds": round(time.perf_counter() - began, 1),
            }
        )
        print(
            f"  iter {n:>2}: {stats.episodes_solved}/{stats.episodes} solved, "
            f"pool-par {from_pool}, ring {len(ring)}, "
            f"{time.perf_counter() - began:.1f}s"
        )

    rows = read_rows(run_dir / "iterations.jsonl", ITERATION_FIELDS)
    summary["alarms"] = alarm_census(rows)
    summary["committed"] = latest_committed(run_dir)
    return summary
