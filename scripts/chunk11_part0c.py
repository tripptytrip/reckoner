"""Part 0c: mint the mid-strata suites — scripted par, and the +1 cell is live.

Protocol frozen in `PREREG-chunk11-part0bc.md` at `5872ca8`.

**Named `scripted_in_k`, not `solve_in_k`, and the reason is a measurement.**
The ruling says `solve_in_7..10`. Two facts found while building say that name
cannot be used, and both are reported rather than worked around:

1. **`solve_in_*` is a live glob.** `census_supervision_contamination.py`,
   `chunk11_part0.py` and `chunk11_part0b.py` all enumerate the instrument with
   `SUITES.glob("solve_in_*.jsonl")`. Minting into that series would silently
   change the 1,200-problem instrument to 2,000 — including the instrument P1's
   no-regress floor of **1188/1200** is computed against. A frozen instrument
   that changes size when a new file lands is not frozen.
2. **`solve_in_k` has meant BFS-exact par k** for k = 1…6. These carry *scripted*
   par, which is a floor: BFS finds a 6-step derivation where the scripted policy
   takes 7. Reusing the name would keep the wording and drop the justification —
   L7, at a filename.

The instrument the ruling asked for is delivered in full: scripted par, a
provisional floor, `EXACT_PAR_SOURCES = {"bfs"}` so the `z = +1` tripwire does not
fire, the `+1` cell live, the full z scale breathing. Only the name changed, and
it changed to keep the thing the primary is measured against intact.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import replace
from pathlib import Path

from reckoner.config import Config
from reckoner.dataset import data_path, git_sha, problem_key, write_record
from reckoner.episode import Problem, bfs_par
from reckoner.generator import MID_TEMPLATES, emit_mid
from reckoner.pairedset import census, freeze, source_census_keys
from reckoner.solver import scripted_par

REPO = Path(__file__).resolve().parents[1]
EXPECTATIONS_SHA = "5872ca8"
SUITES = REPO / "runs" / "suites"
STRATA = (7, 8, 9, 10)
PER_STRATUM = 200

#: Obligation 5. BFS at cap = par-1 costs ~9 s per problem at par 7 and grows with
#: the branch factor, so the certificate is sampled, not exhaustive — and the
#: sample size is stated rather than "a few".
DELTA_SAMPLE = 20


def mint_pool(count: int, seed: int) -> dict[int, list[Problem]]:
    """Emit, label with SCRIPTED par, and bucket by the label.

    Stratum identity is the **label**, never the template's intention — the
    chunk-5 precedent, where templates measured 4/21 across two depths.
    """
    buckets: dict[int, list[Problem]] = {k: [] for k in STRATA}
    seen: set[tuple[int, ...]] = set()
    names = list(MID_TEMPLATES)
    cfg = Config()
    for i in range(count):
        problem = emit_mid(names[i % len(names)], random.Random(seed * 1_000_003 + i))
        par = scripted_par(problem, cfg)
        if par is None or par not in buckets:
            continue
        labelled = replace(problem, par=par, par_source="scripted")
        key = problem_key(labelled)
        if key in seen:
            continue
        seen.add(key)
        buckets[par].append(labelled)
    return buckets


def certify(problems: list[Problem], sample: int, cfg: Config) -> dict:
    """`scripted − bfs` where BFS is affordable. Absence carries its reason."""
    par = problems[0].par
    if par is None or par > 7:
        return {
            "attempted": False,
            "reason": (
                f"BFS at cap {par - 1 if par else '?'} is unaffordable: the branch "
                "factor is ~18 at these roots and a depth-7+ exhaustive search is the "
                "cost this stratum exists to avoid paying. The absence is recorded; "
                "it is NOT a measured gap of zero"
            ),
        }
    deltas, seconds = [], time.perf_counter()
    for problem in problems[:sample]:
        exact = bfs_par(problem, cfg, cap=par - 1)
        deltas.append(None if exact is None else par - exact)
    confirmed = sum(1 for d in deltas if d is None)
    gaps = [d for d in deltas if d is not None]
    return {
        "attempted": True,
        "sample": len(deltas),
        "bfs_cap": par - 1,
        "exact_confirmed": confirmed,
        "floor_with_gap": len(gaps),
        "gap_histogram": {str(g): gaps.count(g) for g in sorted(set(gaps))},
        "seconds": round(time.perf_counter() - seconds, 1),
        "meaning": (
            "a gap of g means BFS found a derivation g steps shorter than the "
            "scripted label, so the label is a FLOOR by exactly that much and the "
            "+1 cell is reachable — which is what this instrument is for"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=int, default=DELTA_SAMPLE)
    args = parser.parse_args()

    cfg = Config()
    started = time.perf_counter()
    buckets = mint_pool(args.pool, args.seed)
    print(
        f"  pool {args.pool} -> " + ", ".join(f"par {k}: {len(v)}" for k, v in buckets.items()),
        flush=True,
    )
    short = [k for k in STRATA if len(buckets[k]) < PER_STRATUM]
    if short:
        raise SystemExit(
            f"strata {short} came up short of {PER_STRATUM}. Raise --pool rather than "
            "shipping uneven strata: a suite whose size is an accident of yield is not "
            "an instrument."
        )

    print("  censusing at both levels…", flush=True)
    problem_level = {"train_100k": source_census_keys(data_path("") / "train_100k")}
    state_level = {"phase1_train": source_census_keys(data_path("") / "phase1_train")}

    strata_records = {}
    for k in STRATA:
        candidates = buckets[k][: PER_STRATUM * 2]
        result = census(candidates, problem_sources=problem_level, state_sources=state_level)
        clean = [candidates[i] for i in result.clean_indices][:PER_STRATUM]
        if len(clean) < PER_STRATUM:
            raise SystemExit(f"par {k}: only {len(clean)} clean of {PER_STRATUM}")
        path = SUITES / f"scripted_in_{k}.jsonl"
        digest = freeze(path, clean, repo=REPO)
        strata_records[f"scripted_in_{k}"] = {
            "problems": len(clean),
            "par": k,
            "par_source": "scripted",
            "digest": digest,
            "census": result.as_dict(),
            "scripted_par_delta": certify(clean, args.sample, cfg),
        }
        print(
            f"    scripted_in_{k}: {len(clean)} frozen, census "
            f"P{result.as_dict()['problem_level_hits']}/S{result.as_dict()['state_level_hits']}, "
            f"{digest[:12]}",
            flush=True,
        )

    report = {
        "expectations_frozen_at": EXPECTATIONS_SHA,
        "git_sha": git_sha(REPO),
        "naming": {
            "ruling_said": "solve_in_7..10",
            "minted_as": "scripted_in_7..10",
            "reason_1": "solve_in_* is a live glob in three scripts; minting into it "
            "would change the frozen 1,200-problem instrument to 2,000, including the "
            "one P1's no-regress floor of 1188/1200 is computed against",
            "reason_2": "solve_in_k has meant BFS-EXACT par k for k=1..6; these carry "
            "scripted par, which is a floor — BFS finds a 6-step derivation where the "
            "scripted policy takes 7. Same wording, different justification (L7)",
            "unchanged": "scripted par, provisional floor, z=+1 tripwire silent, the "
            "+1 cell live — the instrument the ruling asked for, under a name that "
            "does not break the instrument the primary depends on",
        },
        "pool": args.pool,
        "seed": args.seed,
        "per_stratum": PER_STRATUM,
        "strata": strata_records,
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
    }
    write_record(REPO / "runs" / "chunk11_part0c_mint.json", report)
    print(f"\n  wall clock {report['wall_clock_seconds']}s")
    print("  wrote runs/chunk11_part0c_mint.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
