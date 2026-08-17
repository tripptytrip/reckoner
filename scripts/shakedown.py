"""The chunk-9 shakedown: three iterations, eight expectations, verdicts verbatim.

Every expectation is read from `PREREG-chunk9-shakedown.md`, frozen at `93868dd`
before this script existed. **An expectation that fails is a finding with a
verdict, never an adjustment** — the amendment header governs that file the way it
governed every PREREG before it.

The run directory is deleted after recording, per the plan: its artifacts are
evidence of plumbing, not of performance, and keeping them invites a later reader
to cite a three-iteration run as a result. The numbers in the report survive.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

import torch

from reckoner.config import Config, config_fingerprint, save_config, validate
from reckoner.dataset import (
    anchored_data,
    git_sha,
    sha256_file,
    training_problems,
    write_record,
)
from reckoner.evaluate import model_evaluator
from reckoner.logschema import (
    ITERATION_FIELDS,
    SCHEMA_ERA,
    VALUE_SWITCH_FIELDS,
    abstention_census,
    alarm_census,
    append_row,
    read_rows,
    switch_event_row,
)
from reckoner.model import load_checkpoint, save_checkpoint
from reckoner.pool import CheckpointPool, PoolError
from reckoner.replay import ReplayRing
from reckoner.resume import KEEP_RINGS, RunState, commit_iteration, latest_committed
from reckoner.rules import RULESET_VERSION
from reckoner.runner import iteration_row, run_iteration
from reckoner.valuegate import ValueHeadState, consider_switch, value_contribution
from reckoner.vocab import VOCAB_VERSION

REPO = Path(__file__).resolve().parents[1]
EXPECTATIONS_SHA = "93868dd"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="shakedown")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--keep", action="store_true", help="do not delete the run directory")
    args = parser.parse_args()

    cfg = Config()
    validate(cfg)
    run = REPO / "runs" / args.name
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    save_config(cfg, run / "config.yaml")

    verdicts: dict[str, dict] = {}
    started = time.perf_counter()

    # --- E3: the pool loads a matching snapshot and refuses a mismatch ------
    pool = CheckpointPool(cfg)
    anchor = REPO / "runs" / "phase1" / "phase1.pt"
    seeded = cfg.league.seed_pool_with_anchor and anchor.exists()
    if seeded:
        pool.add(anchor)
    model, _ = load_checkpoint(anchor, cfg)
    model.eval()

    mismatched = run / "mismatched.pt"
    save_checkpoint(mismatched, model, cfg, step=1)
    blob = torch.load(mismatched, map_location="cpu", weights_only=False)
    blob["meta"]["ruleset_version"] = 999
    torch.save(blob, mismatched)
    refused = False
    try:
        pool.add(mismatched)
    except PoolError:
        refused = True
    mismatched.unlink()

    verdicts["E3"] = {
        "expectation": "snapshot loads; a version mismatch raises and is counted",
        "anchor_seeded_pool": seeded,
        "pool_size_after_load": len(pool),
        "mismatch_refused": refused,
        "refusals_counted": pool.stats.refusals,
        "verdict": "PASS"
        if (seeded and len(pool) == 1 and refused and pool.stats.refusals == 1)
        else "FAIL",
    }

    # --- the loop ----------------------------------------------------------
    ring = ReplayRing(min(20_000, cfg.train.replay_capacity), cfg)
    head = ValueHeadState()
    rows_path = run / "iterations.jsonl"
    switch_path = run / "value_switch.jsonl"
    # A TRAINING source, through the guard. The chunk-9 shakedown drew from
    # solve_in_2 — a frozen instrument — which did no harm because nothing
    # trained, but demonstrated that the loop would happily consume its own
    # measuring stick. training_problems() refuses instruments first thing.
    source = anchored_data("train_100k")
    solve_by_depth: list[dict] = []
    pool_sizes: list[int] = []

    for n in range(args.iterations):
        rng = random.Random(1000 + n)
        problems = training_problems(source, args.episodes, seed=n)

        evaluator = model_evaluator(model, cfg, value_contribution(head))
        stats = run_iteration(problems, evaluator, cfg, ring, sims=args.sims, m=args.m, seed=n)

        # E4: pool par for a slice, provenance carried
        pool_pars = []
        for problem in problems[: max(1, len(problems) // 4)]:
            result = pool.par_for_episode(
                problem,
                lambda mdl, scale: model_evaluator(mdl, cfg, scale),
                rng,
                sims=args.sims,
                m=args.m,
                budget=cfg.episode.step_cap,
            )
            pool_pars.append(result)

        # E5/E6: the switch criterion, evaluated and written whatever it says
        labels = [0] * stats.episodes_solved + [-1] * (stats.episodes - stats.episodes_solved)
        predictions = [0] * len(labels)
        head, event = consider_switch(head, labels, predictions, iteration=n)
        append_row(switch_path, switch_event_row(event, schema_era=SCHEMA_ERA), VALUE_SWITCH_FIELDS)

        row = iteration_row(
            stats,
            iteration=n,
            run_name=args.name,
            git_sha=git_sha(REPO),
            config_fingerprint=config_fingerprint(cfg),
            cfg=cfg,
            ruleset_version=RULESET_VERSION,
            vocab_version=VOCAB_VERSION,
            schema_era=SCHEMA_ERA,
            # Constant BY CONSTRUCTION here, and honestly so: the shakedown has
            # no optimiser, so the evaluator plays the anchor's weights in every
            # iteration. In the campaign this column MOVES, and golden asserts
            # that it does — a digest that never changes is the stub defect
            # wearing a provenance field. Here it never changes because the model
            # never does.
            evaluator_checkpoint_sha256=sha256_file(anchor),
            pool_composition=pool.composition(),
        )
        solve_by_depth.append({"iteration": n, "rates": row["solve_rate_by_depth"]})

        # ENROLLMENT — the loop feeds the pool its own checkpoint, which is what
        # makes par escalate. Without this the "league" is a fixed opponent.
        if (n + 1) % cfg.league.snapshot_every == 0:
            pool.enroll(model, 5000 + n + 1, head, run / f"snapshot-{n}.pt")

        pool_sizes.append(len(pool))
        state = RunState(iteration=n, value_head=head, seed=n)
        commit_iteration(
            run, ring, state, lambda row=row: append_row(rows_path, row, ITERATION_FIELDS)
        )
        print(
            f"  iteration {n}: pool {len(pool)} | "
            f"{stats.episodes_solved}/{stats.episodes} solved, "
            f"{stats.nodes} nodes, pool par {len(pool_pars)} sampled, "
            f"switch {'FIRED' if event['fired'] else 'abstained' if event.get('abstained') else 'refused'}"
        )

    # --- verdicts ----------------------------------------------------------
    rows = read_rows(rows_path, ITERATION_FIELDS)
    switch_rows = read_rows(switch_path, VALUE_SWITCH_FIELDS)

    verdicts["E1"] = {
        "expectation": "one validating row per committed iteration",
        "rows": len(rows),
        "committed": latest_committed(run),
        "verdict": "PASS" if len(rows) == args.iterations == latest_committed(run) + 1 else "FAIL",
    }
    census = alarm_census(rows)
    verdicts["E2"] = {
        "expectation": "alarms: 0",
        "alarm_census": census,
        "verdict": "PASS" if census == {} else "FINDING",
    }
    tagged = [p for p in pool_pars if not p.fell_back]
    verdicts["E4"] = {
        "expectation": "pool par carries par_source and par_asof; fallbacks are counted",
        "pool_par_solved": pool.stats.pool_par_solved,
        "unavailable_capped": pool.stats.pool_par_unavailable_capped,
        "unavailable_empty": pool.stats.pool_par_unavailable_empty,
        "sample_tagged_pool": [
            {"par": p.par, "par_source": p.par_source, "par_asof": p.par_asof} for p in tagged[:3]
        ],
        "verdict": "PASS"
        if all(p.par_source == "pool" and isinstance(p.par_asof, int) for p in tagged)
        and pool.stats.pool_par_unavailable_empty == 0
        else "FAIL",
    }
    verdicts["E5"] = {
        "expectation": "criterion four-tupled, evaluated where support allows",
        "rows": [
            {
                k: r.get(k)
                for k in (
                    "iteration",
                    "k_classes_with_support",
                    "floor",
                    "null",
                    "threshold",
                    "measured",
                    "smallest_class_support",
                    "abstained",
                    "abstain_reason",
                    "clears",
                )
            }
            for r in switch_rows
        ],
        "verdict": "PASS" if switch_rows else "FAIL",
    }
    verdicts["E6"] = {
        "expectation": "every evaluation writes a row, abstentions included",
        "abstention_census": abstention_census(switch_rows),
        "verdict": "PASS" if len(switch_rows) == args.iterations else "FAIL",
    }
    verdicts["E7"] = {
        "expectation": "cross-switch watch armed; regression noted, no auto-revert",
        "switch_fired": any(r["fired"] for r in switch_rows),
        "solve_rate_by_depth_per_iteration": solve_by_depth,
        "verdict": "ARMED — no switch occurred, so there is no crossing to watch",
    }
    rings = sorted(int(p.name.removeprefix("ring-")) for p in run.glob("ring-*"))
    verdicts["E8"] = {
        "expectation": f"KEEP_RINGS = {KEEP_RINGS} behind LATEST",
        "rings_on_disk": rings,
        "verdict": "PASS" if len(rings) <= KEEP_RINGS else "FAIL",
    }

    record = {
        "expectations_frozen_at": EXPECTATIONS_SHA,
        "git_sha": git_sha(REPO),
        "iterations": args.iterations,
        "episodes_per_iteration": args.episodes,
        "sims": args.sims,
        "m": args.m,
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
        "wall_clock_split": {
            "self_play": round(sum(r["seconds_self_play"] for r in rows), 3),
            "pool_solving": pool.stats.as_dict()["seconds_solving"],
        },
        "pool": pool.stats.as_dict() | {"composition": pool.composition()},
        "pool_enrollment": {
            "seeded_with_anchor": seeded,
            "snapshot_every": cfg.league.snapshot_every,
            "size_by_iteration": pool_sizes,
        },
        "dormant_levers": {
            "rehearsal_frac": cfg.train.rehearsal_frac,
            "concede_enabled": cfg.par.concede_enabled,
            "concede_k": cfg.par.concede_k,
        },
        "verdicts": verdicts,
    }
    write_record(REPO / "runs" / "shakedown_result.json", record)

    print("\n  SHAKEDOWN VERDICTS — expectations frozen at " + EXPECTATIONS_SHA + "\n")
    for name in sorted(verdicts):
        print(f"    {name}: {verdicts[name]['verdict']}")
    print(
        f"\n  wall clock {record['wall_clock_seconds']}s "
        f"(self-play {record['wall_clock_split']['self_play']}s, "
        f"pool {record['wall_clock_split']['pool_solving']}s)"
    )
    print("  wrote runs/shakedown_result.json")

    if not args.keep:
        shutil.rmtree(run)
        print(f"  deleted {run} after recording, per plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
