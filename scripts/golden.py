"""`make golden` — the loop's fast liveness check. CPU always, under three minutes.

Not a gate on performance and not a shakedown: a golden run answers *does the
loop still turn*, in the time a person will actually wait. The plan requires it
to run **on CPU regardless of training device**, because a check that needs the
accelerator is a check that stops running the moment the accelerator is busy —
and this box's GPU is shared with a project that assumes it owns it.

What it exercises, end to end: episode running with batching across searches, the
descent identity, the replay ring, the schema (rows validated at write), the
value-head declaration reaching both consumers, pool enrollment and pool par, the
commit point, and ring retention. What it does **not** do is judge any of them —
that is the shakedown's job, against pre-registered expectations.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch

from reckoner.config import Config, config_fingerprint, validate
from reckoner.dataset import git_sha, training_problems
from reckoner.logschema import (
    ITERATION_FIELDS,
    SCHEMA_ERA,
    VALUE_SWITCH_FIELDS,
    alarm_census,
    append_row,
    read_rows,
    switch_event_row,
)
from reckoner.model import Reckoner
from reckoner.pool import CheckpointPool
from reckoner.replay import ReplayRing
from reckoner.resume import RunState, commit_iteration, latest_committed, resume
from reckoner.rules import RULESET_VERSION
from reckoner.runner import iteration_row, run_iteration
from reckoner.search import uniform_stub
from reckoner.train import train_on_ring
from reckoner.valuegate import ValueHeadState, consider_switch
from reckoner.vocab import VOCAB_VERSION

REPO = Path(__file__).resolve().parents[1]
BUDGET_SECONDS = 180.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=12)
    args = parser.parse_args()

    # CPU regardless of what is installed. A liveness check that competes for the
    # accelerator is one that stops being run.
    torch.set_num_threads(min(4, torch.get_num_threads()))

    cfg = Config()
    validate(cfg)
    run = REPO / "runs" / "golden"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)

    started = time.perf_counter()
    ring = ReplayRing(4096, cfg)
    pool = CheckpointPool(cfg)
    torch.manual_seed(0)
    model = Reckoner(cfg)
    weights_moved: list[int] = []
    value_head_moved: list[bool] = []
    head = ValueHeadState()
    # A TRAINING source, through the guard. The chunk-9 shakedown drew from
    # solve_in_2 — a frozen instrument — which did no harm because nothing
    # trained, but demonstrated that the loop would happily consume its own
    # measuring stick. training_problems() refuses instruments first thing.
    source = REPO / "runs" / "data" / "train_100k"
    checks: list[tuple[str, bool, str]] = []

    for n in range(args.iterations):
        problems = training_problems(source, args.episodes, seed=n)
        stats = run_iteration(problems, uniform_stub(cfg), cfg, ring, sims=8, m=5, seed=n)
        stats.check_descent_identity()

        # THE TRAINING PHASE. A loop whose central job is improving the model had
        # no assertion that the model moves — the both-riders heuristic applied
        # prospectively (FINDINGS.md F-15).
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        train_on_ring(model, ring, cfg, steps=2, seed=n)
        moved = sum(1 for k, v in model.state_dict().items() if not torch.equal(v, before[k]))
        weights_moved.append(moved)
        value_head_moved.append(
            any(
                k.startswith("value_head") and not torch.equal(model.state_dict()[k], before[k])
                for k in before
            )
        )

        labels = [0] * stats.episodes_solved + [-1] * (stats.episodes - stats.episodes_solved)
        head, event = consider_switch(head, labels, [0] * len(labels), iteration=n)
        append_row(
            run / "value_switch.jsonl",
            switch_event_row(event, schema_era=SCHEMA_ERA),
            VALUE_SWITCH_FIELDS,
        )

        row = iteration_row(
            stats,
            iteration=n,
            run_name="golden",
            git_sha=git_sha(REPO),
            config_fingerprint=config_fingerprint(cfg),
            cfg=cfg,
            ruleset_version=RULESET_VERSION,
            vocab_version=VOCAB_VERSION,
            schema_era=SCHEMA_ERA,
        )
        state = RunState(iteration=n, value_head=head, seed=n)
        commit_iteration(
            run,
            ring,
            state,
            lambda row=row: append_row(run / "iterations.jsonl", row, ITERATION_FIELDS),
        )
        print(f"  iteration {n}: {stats.episodes_solved}/{stats.episodes} solved, ring {len(ring)}")

    rows = read_rows(run / "iterations.jsonl", ITERATION_FIELDS)
    checks.append(("rows validate and match commits", len(rows) == latest_committed(run) + 1, ""))
    checks.append(("no alarms", alarm_census(rows) == {}, str(alarm_census(rows))))
    checks.append(("ring received steps", len(ring) > 0, f"{len(ring)} rows"))
    first = ring.get(0)
    checks.append(
        (
            "ring rows carry a usable STATE, not just an outcome",
            len(first["tokens"]) > 0 and first["n_sites"] > 0,
            f"{len(first['tokens'])} tokens, {first['n_sites']} sites",
        )
    )
    checks.append(
        (
            "training moves the model",
            all(m > 0 for m in weights_moved),
            f"tensors moved per iteration: {weights_moved}",
        )
    )
    checks.append(
        (
            "the W/D/L head trains even while the search distrusts it",
            all(value_head_moved),
            f"{value_head_moved} (F-15: gating both consumers deadlocks the ratchet)",
        )
    )

    # Resume is part of liveness: a loop that cannot restart is not a loop.
    nxt, restored, restored_state = resume(run, cfg)
    checks.append(
        (
            "resume returns the committed state",
            nxt == args.iterations and restored is not None and restored_state is not None,
            f"next={nxt}",
        )
    )

    # Enrollment: the mechanism par escalation depends on.
    # ENROLLMENT REGRESSION joins the class of things a run cannot silently lack.
    # The chunk-9 expectations gated mismatch refusal and provenance — the known
    # hazards — while nothing asserted the pool POPULATES, so composition() sat in
    # the report as a number nobody asserted on. Rider (a) and rider (b) in one
    # miss; this is the fix at the liveness layer.
    before = len(pool)
    for step in (1, 2, 3):
        pool.enroll(model, step, head, run / f"snap-{step}.pt")
    grew = len(pool) == before + 3
    checks.append(("pool composition GROWS with enrollment", grew, str(pool.composition())))

    elapsed = time.perf_counter() - started
    shutil.rmtree(run)

    print("\n  GOLDEN\n")
    for name, ok, detail in checks:
        print(f"    {'ok  ' if ok else 'FAIL'} {name}{'  — ' + detail if detail else ''}")
    print(
        f"\n  {elapsed:.1f}s of a {BUDGET_SECONDS:.0f}s budget, CPU, {torch.get_num_threads()} threads"
    )

    if not all(ok for _, ok, _ in checks):
        return 1
    if elapsed > BUDGET_SECONDS:
        print(
            f"  OVER BUDGET: {elapsed:.1f}s > {BUDGET_SECONDS:.0f}s — a golden run people "
            "will not wait for is one they stop running"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
