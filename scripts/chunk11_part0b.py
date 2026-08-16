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
from decimal import Decimal
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

# The floor of the sims domain: a search with no simulations is not a search, so
# there is nothing below sims=1 to extend into. Reaching it turns the all-above
# node from "extend downward" into a terminal verdict — see P11B-A3 §2(c).
DOMAIN_FLOOR = 1

# Part-0d's ruling, 2026-08-16: scripted_in_9 is the sole saturated stratum
# (beat-par 164/200 = 0.82, past the >=0.5 definition) and demotes to
# informational; the successor set is the three strata that carry headroom.
SUCCESSOR_STRATA = (7, 8, 10)


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


def _distance(point: dict) -> Decimal:
    """Distance from the target, in exact decimal.

    P11B-A5. Rates are counts over 1,200 rounded to six places and the target is
    0.55, so both are exact decimals — but ``abs(0.50 - 0.55)`` and
    ``abs(0.60 - 0.55)`` differ in binary float (`0.05000000000000004` against
    `0.049999999999999996`), which silently decided a comparison the rule
    intended to call a tie. Going through ``Decimal(str(...))`` removes float
    sensitivity from the criterion entirely.
    """
    return abs(Decimal(str(point["at_par_rate"])) - Decimal(str(TARGET)))


def _rank(point: dict) -> tuple[Decimal, int]:
    """The declared order: nearest the target, then **smaller sims at every tie
    level**.

    The secondary key is economy-motivated and is the primary's own axis — at
    equal informativeness the cheaper rung serves the campaign — so it is stated
    as a key rather than left to sort stability.
    """
    return (_distance(point), point["sims"])


def select(points: list[dict]) -> dict:
    """Apply the frozen rule. Every branch reports whether it fired and why."""
    low, high = WINDOW
    in_window = [p for p in points if low <= p["at_par_rate"] <= high]
    branches = []

    if in_window:
        best = min(in_window, key=_rank)
        tied = [p for p in in_window if _distance(p) == _distance(best)]
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
        # Both polarities of the all-above node, per P11B-A3 §2(c). The node as
        # first written could only ask for a downward extension, which at the
        # domain floor requests points below sims=1 — an impossibility — and it
        # predated Part-0d, so it had no successor to name. Completing it at a
        # node whose input did not yet exist is the amendment policy's allowance.
        if min(p["sims"] for p in ordered) <= DOMAIN_FLOOR:
            branches.append(
                {
                    "branch": "all_above_window",
                    "fired": True,
                    "at_domain_floor": True,
                    "action": "succession fires at iteration 0: the suite-economy "
                    "primary is saturated at the floor of its own domain",
                    "successor": "P1 becomes the scripted "
                    f"{{{', '.join(str(k) for k in SUCCESSOR_STRATA)}}} paired "
                    "beat-par trajectory (Part-0d certified live 2026-08-16)",
                    "successor_strata": list(SUCCESSOR_STRATA),
                }
            )
            return {"s_star": None, "needs": "succession", "branches": branches}
        branches.append(
            {
                "branch": "all_above_window",
                "fired": True,
                "at_domain_floor": False,
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


class RungCollision(ValueError):
    """Two measurements of one rung disagree, or two protocols would be mixed.

    Either is a finding that demands a diagnosis. Neither is an overwrite that
    silently decides which measurement history keeps.
    """


def merge_points(prior: list[dict], fresh: list[dict]) -> list[dict]:
    """Union two invocations by rung key, refusing disagreeing collisions.

    Branches (b) and (c) are predicates over the **whole measured domain**, so a
    selection computed from one invocation's points answers a different question
    than the rule asks. Hence the union, keyed on ``sims``.

    ``seconds`` is excluded from the agreement check: wall-clock legitimately
    differs between invocations and carries no measurement content. Every count
    that does carry content must agree exactly.
    """
    counts = (
        "episodes",
        "gumbel_m",
        "at_par",
        "beat_par",
        "over_par",
        "solved",
        "capped",
        "stuck",
        "steps_minus_par",
    )
    by_sims: dict[int, dict] = {p["sims"]: p for p in prior}
    for point in fresh:
        seen = by_sims.get(point["sims"])
        if seen is not None:
            differing = sorted(k for k in counts if seen.get(k) != point.get(k))
            if differing:
                raise RungCollision(
                    f"sims={point['sims']} is measured twice and the counts "
                    f"disagree on {differing}. Two measurements of one rung "
                    "disagreeing is a finding: diagnose it. It is not an "
                    "overwrite deciding silently which one history keeps."
                )
        by_sims[point["sims"]] = point
    return [by_sims[k] for k in sorted(by_sims)]


def load_prior(path: Path, cfg: Config) -> list[dict]:
    """Points already measured under *this* protocol, or none.

    Checked before any measurement runs, so a mixed-protocol union fails in
    seconds rather than after the sweep has been paid for.
    """
    if not path.exists():
        return []
    record = json.loads(path.read_text())
    prior_fp = record.get("protocol", {}).get("config_fingerprint")
    if prior_fp != config_fingerprint(cfg):
        raise RungCollision(
            f"the existing sweep record was measured under fingerprint "
            f"{prior_fp}, this invocation runs {config_fingerprint(cfg)}; a "
            "union across two protocols is refused"
        )
    return record.get("sweep", [])


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
    record_path = REPO / "runs" / "chunk11_part0b_sweep.json"
    prior = load_prior(record_path, cfg)

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

    measured_here = sorted(p["sims"] for p in points)
    points = merge_points(prior, points)
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
        # One rule, two invocations (P11B-A3). The record states which rungs this
        # invocation measured and which it carried, so the union is never mistaken
        # for a single sitting.
        "invocations": {
            "amendment": "P11B-A3",
            "measured_this_invocation": measured_here,
            "carried_from_prior_record": sorted(
                p["sims"] for p in points if p["sims"] not in measured_here
            ),
        },
        "sweep": points,
        "selection": selection,
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
    }
    write_record(record_path, report)
    print("\n  SELECTION\n")
    print(json.dumps(selection, indent=2))
    print(f"\n  wall clock {report['wall_clock_seconds']}s")
    print("  wrote runs/chunk11_part0b_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
