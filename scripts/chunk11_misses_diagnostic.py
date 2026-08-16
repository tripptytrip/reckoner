"""Rider (b) item 2, and the coincidence that walked in wearing a familiar number.

**The original question.** The sweep returned a byte-identical `1189/1200` at sims
6, 8, 12 and 16. Item 1 closed that as the good kind of identical — per-point
wall-clock is strictly monotone, so those are four distinct and increasingly
expensive computations agreeing. Item 2 asks what the counts cannot:

    Coherent   — the same hard problems miss at every budget.
    Coinciding — different problems miss, and the counts merely agree.

**The coincidence.** The anchor at `sims = 1` pars `1176/1200` — misses 24. The
scripted solver scores optimal on `1176/1200` of the same 1,200 problems — misses
24. Two different systems, one number, one population. Either that is a collision
of counts to dispose of in a line, or the same problems are hard for a greedy
symbolic solver and a one-simulation neural policy alike, which would be a
template-family mechanism with a name waiting for it.

**Registered before the run, scored in the record**, from the reviewer's ruling:
the scripted misses concentrate at depths 4–5 (16 of 24, depths 1 and 6 perfect);
the anchor's 48-sim tail sits at par 4–5; the two counts are equal. Three
residuals, three systems, one mid-depth band.

## Two admissibility rules, because both sources are reconstructions

**The anchor's miss sets.** `IterationStats` keeps histogram bins only and
`run_iteration`'s fourth argument is a `ReplayRing`, not an outcome sink, so no
committed artifact carries episode identity. This wraps `runner._settle` for the
duration and restores it — `measure()` and `run_iteration` execute exactly the
committed code. Seeding is a per-episode, per-step fan-out from a fixed seed over
a fixed problem order, which is what makes a re-run a replay; but architecture is
an argument, not evidence, so **each budget's reproduced at-par is checked against
the committed sweep record** and a mismatch raises.

**The scripted solver's miss set.** `runs/par_delta.json` is histograms —
`by_depth`, `by_goal`, an empty `unsolved` — and carries no identity to intersect.
Identity is recovered by recomputing `scripted_par_delta` per problem, the pure
function that produced those histograms. The law that makes it admissible:
**a recomputation is an identity source only if it reproduces the recorded
aggregates exactly.** Rebuilt `by_depth` and `by_goal` are compared against the
artifact and a mismatch raises — the reproduction check, pointed at the other
system.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import chunk11_part0b as sweep
import torch

from reckoner import runner
from reckoner.config import Config
from reckoner.dataset import git_sha, read_suite, suite_problem, write_record
from reckoner.evaluate import model_evaluator
from reckoner.ladder import problem_key_of
from reckoner.model import load_checkpoint
from reckoner.runner import run_iteration
from reckoner.solver import scripted_par_delta
from reckoner.vocab import GOAL_EVALUATE, GOAL_SIMPLIFY, GOAL_SOLVE

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
RECORD = REPO / "runs" / "chunk11_part0b_sweep.json"
PAR_DELTA = REPO / "runs" / "par_delta.json"

# 48 FIRST, DELIBERATELY. sims=48 is the licence question — the host must
# reproduce 1193/1200 — and the reproduction check raises on divergence, so
# running it first answers the licence in the first pass and halts before the
# remaining budgets spend anything. Counts before identity before keys, enforced
# by execution order rather than by a separate counts-only invocation, which
# would only have paid for itself in the failing case.
BUDGETS = (48, 1, 6, 16)
CFG = Config()
NAMES = {GOAL_SOLVE: "SOLVE", GOAL_EVALUATE: "EVALUATE", GOAL_SIMPLIFY: "SIMPLIFY"}

#: Registered before the run, from the reviewer's ruling of 2026-08-16.
REGISTERED = {
    "source": "reviewer ruling, 2026-08-16",
    "obs_1": "the scripted solver's misses concentrate at depths 4-5 (16 of 24; "
    "depths 1 and 6 perfect)",
    "obs_2": "the anchor's 48-sim tail sits entirely at par 4-5",
    "obs_3": "the anchor's sims-1 miss count equals the scripted solver's (24)",
    "question": "do the sims-1 and scripted miss sets coincide BY KEY, or only by count?",
}


class ReproductionFailure(RuntimeError):
    """A reconstruction did not reproduce its record, so it describes nothing."""


# ------------------------------------------------------- the anchor's misses


@contextmanager
def capture(sink: list[dict]):
    """Record every settled episode, then hand off to the committed scorer."""
    original = runner._settle

    def wrapped(stats, ring, e, trail, cfg, *, solved, capped):
        par = e.problem.par or 0
        sink.append(
            {
                "key": problem_key_of(e.problem),
                "par": par,
                "steps": e.steps,
                "delta": (e.steps - par) if solved else None,
                "solved": solved,
            }
        )
        return original(stats, ring, e, trail, cfg, solved=solved, capped=capped)

    runner._settle = wrapped
    try:
        yield
    finally:
        runner._settle = original


def episodes_at(model, cfg, sims: int, problems_by_suite: dict) -> list[dict]:
    m = min(cfg.search.gumbel_m, sims)
    evaluator = model_evaluator(model, cfg, 0.0)
    sink: list[dict] = []
    with capture(sink):
        for problems in problems_by_suite.values():
            run_iteration(problems, evaluator, cfg, None, sims=sims, m=m, seed=0)
    return sink


def recorded_at_par(sims: int) -> int | None:
    if not RECORD.exists():
        return None
    for pt in json.loads(RECORD.read_text()).get("sweep", []):
        if pt["sims"] == sims:
            return pt["at_par"]
    return None


# ---------------------------------------------- the scripted solver's misses


def _scripted(row: dict) -> tuple[str, int, str, int | None]:
    problem = suite_problem(row)
    return (
        problem_key_of(problem),
        row["depth"],
        NAMES[row["goal"]],
        scripted_par_delta(problem, CFG),
    )


def scripted_misses(rows: list[dict], workers: int) -> tuple[dict[str, int], dict]:
    """Recompute per-problem scripted delta, and refuse unless it reproduces.

    Returns ``(miss_key -> delta, rebuilt_aggregates)``.
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_scripted, rows, chunksize=8))

    by_depth: dict[int, Counter] = {}
    by_goal: dict[str, Counter] = {}
    unsolved: Counter = Counter()
    misses: dict[str, int] = {}
    for key, depth, goal, delta in results:
        if delta is None:
            unsolved[f"{depth}:{goal}"] += 1
            continue
        by_depth.setdefault(depth, Counter())[delta] += 1
        by_goal.setdefault(goal, Counter())[delta] += 1
        if delta != 0:
            misses[key] = delta

    rebuilt = {
        "by_depth": {str(d): dict(c) for d, c in by_depth.items()},
        "by_goal": {g: dict(c) for g, c in by_goal.items()},
        "unsolved": dict(unsolved),
    }

    recorded = json.loads(PAR_DELTA.read_text())
    for field in ("by_depth", "by_goal", "unsolved"):
        want = (
            {k: {str(d): n for d, n in v.items()} for k, v in recorded[field].items()}
            if field != "unsolved"
            else recorded[field]
        )
        got = (
            {k: {str(d): n for d, n in v.items()} for k, v in rebuilt[field].items()}
            if field != "unsolved"
            else rebuilt[field]
        )
        if want != got:
            raise ReproductionFailure(
                f"the scripted recomputation does not reproduce par_delta.json "
                f"on {field}:\n  recorded {want}\n  rebuilt  {got}\n"
                "A recomputation is an identity source only if it reproduces the "
                "recorded aggregates exactly. It does not, so its miss set "
                "describes a different population and must not be intersected."
            )
    return misses, rebuilt


# ----------------------------------------------------------------- pre-flight


def preflight(model, cfg, problems_by_suite: dict) -> None:
    """Two problems through the FULL path, including the report writer.

    D-A1 §3, the pilot law extended to output paths. This script has raised
    twice at the report — once on an attribute that did not exist, once on a
    tuple used as a JSON key — each time after the expensive middle had already
    run. Both would have raised here, in seconds.

    It exercises the three things that have actually broken: the sweep-record
    read, the episode path, and ``write_record``'s serialisation. It does NOT
    exercise the scripted admissibility check, which cannot pass on two rows by
    construction.
    """
    started = time.perf_counter()
    if recorded_at_par(48) is None:
        raise ReproductionFailure(
            f"{RECORD} carries no sims=48 rung; the reproduction check would "
            "silently skip, which is the check that licenses this host"
        )
    tiny = {k: v[:2] for k, v in list(problems_by_suite.items())[:1]}
    episodes = episodes_at(model, cfg, 1, tiny)
    probe = {
        "preflight": True,
        "per_budget": {"1": {"misses": sorted(e["key"] for e in episodes)}},
        "misses_by_par": dict(Counter(e["par"] for e in episodes)),
        "scripted": {"misses_by_key": {e["key"]: 1 for e in episodes}},
    }
    # Same .gitignore negation glob as the real output, so the probe exercises
    # write_record's tracked-record guard instead of dodging it.
    scratch = REPO / "runs" / "chunk11_preflight_probe.json"
    write_record(scratch, probe)
    scratch.unlink()
    print(
        f"  pre-flight OK ({len(episodes)} episodes, writer exercised, "
        f"{time.perf_counter() - started:.1f}s)\n"
    )


# --------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="*", default=list(BUDGETS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = sweep.eval_config()
    model, _ = load_checkpoint(REPO / "runs" / "phase1" / "phase1.pt", cfg)
    model.eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    paths = sorted(SUITES.glob("solve_in_*.jsonl"))
    problems_by_suite = {p.stem: [suite_problem(r) for r in read_suite(p)] for p in paths}
    rows = [r for p in paths for r in read_suite(p)]
    depth_of = {problem_key_of(suite_problem(r)): r["depth"] for r in rows}
    goal_of = {problem_key_of(suite_problem(r)): NAMES[r["goal"]] for r in rows}

    preflight(model, cfg, problems_by_suite)

    print("  scripted solver, recomputed for identity")
    scripted, rebuilt = scripted_misses(rows, args.workers)
    print(f"    reproduces par_delta.json exactly; {len(scripted)} misses by key\n")

    print("  anchor, replayed per budget")
    per_budget: dict[int, dict] = {}
    for sims in args.budgets:
        episodes = episodes_at(model, cfg, sims, problems_by_suite)
        at_par = sum(1 for e in episodes if e["delta"] == 0)
        expected = recorded_at_par(sims)
        if expected is not None and at_par != expected:
            raise ReproductionFailure(
                f"sims={sims} replayed to at-par {at_par}, the record says "
                f"{expected}. The replay is not the recorded run, so its miss set "
                "describes a different measurement."
            )
        misses = sorted(e["key"] for e in episodes if e["delta"] != 0)
        per_budget[sims] = {
            "at_par": at_par,
            "episodes": len(episodes),
            "misses": misses,
            "misses_by_par": dict(sorted(Counter(depth_of[k] for k in misses).items())),
        }
        print(
            f"    sims={sims:>2}  at-par {at_par}/{len(episodes)}  misses {len(misses):>2}"
            f"  by par {per_budget[sims]['misses_by_par']}  (record {expected})",
            flush=True,
        )

    budgets = sorted(per_budget)
    sets = {s: set(per_budget[s]["misses"]) for s in budgets}

    # The plateau: is each larger budget's miss set contained in the smaller's?
    plateau = []
    for lo, hi in zip(budgets, budgets[1:], strict=False):
        plateau.append(
            {
                "from_sims": lo,
                "to_sims": hi,
                "nested": sets[hi] <= sets[lo],
                "recovered": sorted(sets[lo] - sets[hi]),
                "newly_missed": sorted(sets[hi] - sets[lo]),
            }
        )

    one = sets.get(1, set())
    scripted_keys = set(scripted)
    shared = sorted(one & scripted_keys)
    cross = {
        "sims_1_misses": len(one),
        "scripted_misses": len(scripted_keys),
        "counts_equal": len(one) == len(scripted_keys),
        "shared": shared,
        "shared_count": len(shared),
        "only_sims_1": sorted(one - scripted_keys),
        "only_scripted": sorted(scripted_keys - one),
        "jaccard": round(len(one & scripted_keys) / len(one | scripted_keys), 6)
        if (one | scripted_keys)
        else None,
        "shared_by_par": dict(sorted(Counter(depth_of[k] for k in shared).items())),
        "shared_by_goal": dict(sorted(Counter(goal_of[k] for k in shared).items())),
    }

    scripted_by_depth = {
        d: sum(n for delta, n in c.items() if int(delta) != 0)
        for d, c in sorted(rebuilt["by_depth"].items())
    }
    tail = per_budget.get(max(budgets), {}).get("misses_by_par", {})
    scoring = {
        "obs_1_scripted_concentrate_at_depth_4_5": (
            scripted_by_depth.get("4", 0) + scripted_by_depth.get("5", 0),
            sum(scripted_by_depth.values()),
        ),
        "obs_2_largest_budget_tail_pars": sorted(tail),
        "obs_3_counts_equal": cross["counts_equal"],
        "answer_sets_coincide_by_key": bool(shared)
        and cross["only_sims_1"] == []
        and cross["only_scripted"] == [],
        "answer_overlap_jaccard": cross["jaccard"],
    }

    report = {
        "question": REGISTERED["question"],
        "git_sha": git_sha(REPO),
        "registered_observations": REGISTERED,
        "protocol": {
            "model": "runs/phase1/phase1.pt",
            "gumbel_m_rule": "min(16, sims), declared in PREREG-chunk11-part0bc",
            "seed": 0,
            "value_scale": 0.0,
            "device": "cpu",
            "instrument": "frozen; runner._settle wrapped for capture and restored",
            "anchor_reproduction_checked_against": "runs/chunk11_part0b_sweep.json",
            "scripted_reproduction_checked_against": "runs/par_delta.json",
        },
        "per_budget": {str(s): per_budget[s] for s in budgets},
        "plateau": plateau,
        "scripted": {
            "misses_by_key": dict(sorted(scripted.items())),
            "misses_by_depth": scripted_by_depth,
        },
        "cross_reference_sims_1_vs_scripted": cross,
        "scoring": scoring,
    }
    write_record(REPO / "runs" / "chunk11_misses_diagnostic.json", report)

    print("\n  PLATEAU")
    for step in plateau:
        print(
            f"    {step['from_sims']:>2} -> {step['to_sims']:>2}   nested "
            f"{str(step['nested']):<5}  recovered {len(step['recovered']):>2}  "
            f"newly missed {len(step['newly_missed'])}"
        )
    print("\n  SIMS-1 vs SCRIPTED\n")
    print(json.dumps({k: v for k, v in cross.items() if k != "shared"}, indent=2))
    print("\n  SCORING\n")
    print(json.dumps(scoring, indent=2))
    print("\n  wrote runs/chunk11_misses_diagnostic.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
