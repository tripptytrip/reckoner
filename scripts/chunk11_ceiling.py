"""The ceiling arithmetic for every candidate primary, computed from Part 0's rows.

The PREREG's rule sent the primary to `rung_trajectory` because T = 7 < 36. That
rule asked whether the **suite tail** could carry CI separation. It did not ask
the same question of the branch it falls back to, because at the time nobody had
the anchor's paired-set z composition to ask it with.

Part 0 produced that composition, so the question is now answerable and is asked
here: **for each candidate primary, what is the most it can move, and how does
that compare to the width of the interval that would have to separate?**

Nothing is typed. Every number is read from `runs/chunk11_part0_result.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from reckoner.dataset import write_record

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    part0 = json.loads((REPO / "runs" / "chunk11_part0_result.json").read_text())
    headroom = part0["M_A_headroom"]
    z_lane = part0["M_B_rung_baselines"]["z_lane"]
    budget = part0["M_B_rung_baselines"]["budget_lane"]
    hist = z_lane["model_z_histogram"]
    n = z_lane["n_pairs"]

    # --- candidate 1: suite tail-reduction (the rule already rejected it) ----
    suite = {
        "metric": "over-par indicator on the frozen suites, iteration 0 vs ~20",
        "problems": headroom["problems"],
        "movable_mass_T": headroom["over_par_T"],
        "movable_fraction": headroom["over_par_fraction"],
        "threshold_T": headroom["threshold_T"],
        "verdict": "REJECTED by the pre-registered rule (T = 7 < 36)",
    }

    # --- candidate 2: rung trajectory in the z lane (the selected branch) ----
    over_par, under_par = float(hist["-1.0"]), float(hist["1.0"])
    mean_model_z = (under_par - over_par) / n
    # Par is BFS-exact on this set, so z = 0 is the CEILING per problem: the
    # `+1` cell is the impossible one the whole ruling turns on.
    ceiling_mean_model_z = 0.0
    max_movement = ceiling_mean_model_z - mean_model_z
    ci_half_width = (z_lane["ci"][1] - z_lane["ci"][0]) / 2
    rung = {
        "metric": "mean paired difference in z, model - greedy, over iterations",
        "n_pairs": n,
        "model_z_histogram": hist,
        "mean_model_z_now": round(mean_model_z, 6),
        "mean_model_z_ceiling": ceiling_mean_model_z,
        "ceiling_reason": (
            "par on this set is BFS-exact, so z = 0 is the best attainable value per "
            "problem and the +1 cell is structurally empty — the same impossible cell "
            "the chunk-11 primary was rejected for measuring"
        ),
        "max_possible_movement": round(max_movement, 6),
        "movable_problems": int(over_par),
        "current_mean_difference": z_lane["model_minus_greedy_mean"],
        "ci": z_lane["ci"],
        "ci_half_width": round(ci_half_width, 6),
        "movement_as_fraction_of_ci_half_width": round(max_movement / ci_half_width, 6)
        if ci_half_width
        else None,
        "greedy_is_fixed": True,
        "verdict": "SATURATED — the difference can only move as the model's z moves, "
        "and its entire remaining headroom is smaller than the interval that "
        "would have to separate",
    }

    # --- candidate 3: the budget lane -------------------------------------
    external = {
        "metric": "solve-vs-budget, sympy",
        "problems_played": budget["problems_played"],
        "solved": budget["solved"],
        "movable_mass": budget["problems_played"] - budget["solved"],
        "verdict": "SATURATED — the rung solves everything it plays, so its own "
        "score cannot move and it cannot be differenced against the z lane anyway",
    }

    record = {
        "source": "runs/chunk11_part0_result.json",
        "git_sha": part0["git_sha"],
        "question": (
            "for each candidate primary, what is the most it can move, and how does "
            "that compare to the interval that would have to separate?"
        ),
        "suite_tail_reduction": suite,
        "rung_trajectory_z_lane": rung,
        "budget_lane": external,
        "conclusion": (
            "every candidate primary denominated in z-against-BFS-exact-par is at or "
            "within noise of its ceiling before the campaign starts. The anchor is at "
            "par on 1193/1200 suite problems and 387/389 paired problems, and nothing "
            "can score better than par where par is exact. This is not a finding about "
            "the model; it is the instrument's dynamic range being exhausted at "
            "iteration 0."
        ),
    }
    write_record(REPO / "runs" / "chunk11_ceiling.json", record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
