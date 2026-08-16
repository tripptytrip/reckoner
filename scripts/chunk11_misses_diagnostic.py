"""Rider (b) item 2: are the 11 misses the same problems at sims 6 and 16?

The sweep returned a byte-identical `1189/1200` at sims 6, 8, 12 and 16. Item 1
closed that as the good kind of identical — per-point wall-clock is strictly
monotone, so those are four distinct and increasingly expensive computations
agreeing. Item 2 asks the sharper question the counts cannot answer:

    Coherent   — the same eleven hard problems miss at every budget, four of
                 which the 48-budget finally recovers.
    Coinciding — different problems miss at 6 and at 16, and the counts merely
                 happen to agree.

No committed artifact can settle it. `IterationStats` accumulates the
`steps_minus_par` histogram and episode counters only, and `run_iteration`'s
fourth positional argument is `ring: ReplayRing | None` — a replay sink, not a
per-episode outcome recorder.

**The frozen instrument is not modified.** This wraps `runner._settle` for the
duration of the diagnostic and restores it, so `measure()` and `run_iteration`
run exactly the committed code.

**This is a re-observation, not a fresh sample.** Seeding is a per-episode,
per-step fan-out from a fixed `seed = 0` over a fixed problem order, so the same
episodes are replayed. That claim is *checked, not asserted*: each budget's
reproduced at-par count is compared against the committed sweep record, and the
diagnostic refuses to report a miss set that came from a different run.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import chunk11_part0b as sweep
import torch

from reckoner import runner
from reckoner.dataset import git_sha, problem_key, read_suite, suite_problem, write_record
from reckoner.evaluate import model_evaluator
from reckoner.model import load_checkpoint
from reckoner.runner import run_iteration

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
BUDGETS = (6, 16)
RECORD = REPO / "runs" / "chunk11_part0b_sweep.json"


class ReproductionFailure(RuntimeError):
    """The replay did not reproduce the recorded run, so it describes nothing."""


@contextmanager
def capture(sink: list[dict]):
    """Record every settled episode, then hand off to the committed scorer."""
    original = runner._settle

    def wrapped(stats, ring, e, trail, cfg, *, solved, capped):
        par = e.problem.par or 0
        sink.append(
            {
                "key": problem_key(e.problem),
                "par": par,
                "steps": e.steps,
                "delta": (e.steps - par) if solved else None,
                "solved": solved,
                "capped": capped,
            }
        )
        return original(stats, ring, e, trail, cfg, solved=solved, capped=capped)

    runner._settle = wrapped
    try:
        yield
    finally:
        runner._settle = original


def episodes_at(model, cfg, sims: int, problems_by_suite: dict) -> list[dict]:
    """One budget over the whole instrument, per-episode outcomes captured."""
    m = min(cfg.search.gumbel_m, sims)
    evaluator = model_evaluator(model, cfg, 0.0)
    sink: list[dict] = []
    with capture(sink):
        for problems in problems_by_suite.values():
            run_iteration(problems, evaluator, cfg, None, sims=sims, m=m, seed=0)
    return sink


def recorded_at_par(sims: int) -> int | None:
    """What the committed sweep says this budget scored, or None if absent."""
    if not RECORD.exists():
        return None
    for point in json.loads(RECORD.read_text()).get("sweep", []):
        if point["sims"] == sims:
            return point["at_par"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="*", default=list(BUDGETS))
    args = parser.parse_args()

    cfg = sweep.eval_config()
    model, _ = load_checkpoint(REPO / "runs" / "phase1" / "phase1.pt", cfg)
    model.eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    problems_by_suite = {
        path.stem: [suite_problem(r) for r in read_suite(path)]
        for path in sorted(SUITES.glob("solve_in_*.jsonl"))
    }

    per_budget: dict[int, dict] = {}
    for sims in sorted(args.budgets):
        episodes = episodes_at(model, cfg, sims, problems_by_suite)
        at_par = sum(1 for e in episodes if e["delta"] == 0)
        expected = recorded_at_par(sims)
        if expected is not None and at_par != expected:
            raise ReproductionFailure(
                f"sims={sims} replayed to at-par {at_par}, the committed record "
                f"says {expected}. The replay is not the recorded run, so its "
                "miss set describes a different measurement. Diagnose before "
                "reading anything below."
            )
        misses = sorted(e["key"] for e in episodes if e["delta"] != 0)
        per_budget[sims] = {"at_par": at_par, "misses": misses, "episodes": len(episodes)}
        print(
            f"    sims={sims:>2}  at-par {at_par}/{len(episodes)}"
            f"  misses {len(misses)}"
            f"  (record says {expected})",
            flush=True,
        )

    budgets = sorted(per_budget)
    sets = {s: set(per_budget[s]["misses"]) for s in budgets}
    lo, hi = budgets[0], budgets[-1]
    shared = sorted(sets[lo] & sets[hi])
    only_lo = sorted(sets[lo] - sets[hi])
    only_hi = sorted(sets[hi] - sets[lo])
    identical = not only_lo and not only_hi

    verdict = (
        "coherent: the same problems miss at both budgets"
        if identical
        else "coinciding: the counts agree but the failing sets differ"
    )

    report = {
        "question": "rider (b) item 2 — are the misses the same problems at each budget?",
        "git_sha": git_sha(REPO),
        "protocol": {
            "model": "runs/phase1/phase1.pt",
            "gumbel_m_rule": "min(16, sims), declared in PREREG-chunk11-part0bc",
            "seed": 0,
            "value_scale": 0.0,
            "device": "cpu",
            "instrument": "frozen; runner._settle wrapped for capture and restored",
            "reproduction_checked_against": "runs/chunk11_part0b_sweep.json",
        },
        "per_budget": {str(s): per_budget[s] for s in budgets},
        "comparison": {
            "budgets": [lo, hi],
            "identical_sets": identical,
            "shared": shared,
            f"only_at_sims_{lo}": only_lo,
            f"only_at_sims_{hi}": only_hi,
        },
        "verdict": verdict,
    }
    write_record(REPO / "runs" / "chunk11_misses_diagnostic.json", report)

    print(f"\n  shared {len(shared)}   only@{lo} {len(only_lo)}   only@{hi} {len(only_hi)}")
    print(f"  VERDICT: {verdict}")
    print("  wrote runs/chunk11_misses_diagnostic.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
