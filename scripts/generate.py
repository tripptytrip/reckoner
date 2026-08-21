"""Generate the training set, the frozen suites, and the held-out eval sets.

**Frozen instruments are generated here and never regenerated.** The suites are
the measuring stick; regenerating them mid-campaign silently changes what every
prior number meant.

Difficulty is *measured*: a candidate comes from a template, BFS computes its
minimum solution depth, and the depth decides which stratum it lands in. The
template's intended depth is never the label — the pre-flight survey found
`solve_two_terms` splitting 4/21 across depths 3 and 4, and `eval_deepest`
splitting 4/56 across 5 and 6.

Labelling is parallel because it is not uniformly priced: a depth-5 SOLVE costs
~1.1 s where a depth-1 EVALUATE costs 0.02 ms, a spread of 50,000×. See
`BRIEF-chunk5.md`'s pre-flight; run `--preflight` to re-measure before a full
generation.

    python scripts/generate.py --preflight          # measure, write nothing
    python scripts/generate.py --train 100000       # the real thing
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from reckoner.config import Config
from reckoner.dataset import DATA_ROOT, problem_key, sha256_file, write_dataset, write_suite
from reckoner.episode import Problem
from reckoner.generator import emit, label

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


def _rel(path: Path) -> str:
    """Anchor key: repo-relative when it can be, absolute otherwise.

    ``--out`` may point outside the repository (the reproducibility spot-check
    regenerates into a tmpdir), and a bare ``relative_to`` raises there — which
    made the seed-reproducibility gate fail for a reason that had nothing to do
    with reproducibility.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


CFG = Config()

#: Which templates may serve which stratum, from the measured survey. A stratum
#: draws from ALL of its templates rather than the cheapest, because a depth-6
#: suite made entirely of one family measures that family, not that depth.
STRATUM_TEMPLATES: dict[int, tuple[str, ...]] = {
    1: ("eval_sum", "eval_sub", "simplify_like", "simplify_two_vars", "solve_coefficient"),
    2: ("eval_product", "simplify_with_constants", "simplify_two_vars", "solve_product_rhs"),
    3: ("eval_mixed", "simplify_with_products", "solve_constant", "solve_two_terms"),
    4: ("simplify_with_products", "solve_two_terms", "eval_deep"),
    5: ("eval_deep", "eval_deepest", "solve_both_sides"),
    6: ("eval_deepest", "solve_both_sides_product"),
}
DEPTHS = tuple(sorted(STRATUM_TEMPLATES))


def _labelled(args: tuple[str, int]) -> Problem | None:
    """Worker: emit one candidate from a seeded rng and label it. Picklable."""
    template, seed = args
    return label(emit(template, random.Random(seed)), CFG)


def _plan(count: int, seed: int, depths: tuple[int, ...] | None = None) -> list[tuple[str, int]]:
    """Deterministic (template, seed) work list — reproducibility lives here.

    The plan is built from one seed before any work happens, so the output does
    not depend on how the pool schedules it. A generator whose result depends on
    worker interleaving is not reproducible from a seed, whatever its docstring
    says.
    """
    rng = random.Random(seed)
    # Collecting for one stratum draws only from that stratum's templates.
    # Drawing from all six and discarding 5/6 would pay the depth-6 price for
    # candidates that were never wanted — and depth 6 is the expensive one.
    depths = depths or DEPTHS
    per_depth = max(1, count // len(depths))
    plan: list[tuple[str, int]] = []
    for depth in depths:
        templates = STRATUM_TEMPLATES[depth]
        for i in range(per_depth):
            plan.append((templates[i % len(templates)], rng.randrange(2**62)))
    while len(plan) < count:  # remainder goes to the first requested stratum
        templates = STRATUM_TEMPLATES[depths[0]]
        plan.append((templates[len(plan) % len(templates)], rng.randrange(2**62)))
    return plan[:count]


def generate_pool(
    count: int, seed: int, workers: int, depths: tuple[int, ...] | None = None
) -> list[Problem]:
    """Label a work plan in parallel, preserving plan order."""
    plan = _plan(count, seed, depths)
    if workers <= 1:
        return [p for p in map(_labelled, plan) if p is not None]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [p for p in pool.map(_labelled, plan, chunksize=64) if p is not None]


def dedup(problems: list[Problem]) -> list[Problem]:
    """First occurrence wins, by the one shared identity normalizer."""
    seen: set[tuple[int, ...]] = set()
    out: list[Problem] = []
    for problem in problems:
        key = problem_key(problem)
        if key in seen:
            continue
        seen.add(key)
        out.append(problem)
    return out


def bucket(problems: list[Problem]) -> dict[int, list[Problem]]:
    out: dict[int, list[Problem]] = {}
    for problem in problems:
        assert problem.par is not None
        out.setdefault(problem.par, []).append(problem)
    return out


# ---------------------------------------------------------------------------


def preflight(sample: int, workers: int) -> None:
    """Project the total bill before spending it, stratified by depth."""
    print(f"  pre-flight: {sample} candidates per stratum, {workers} workers\n")
    print(
        f"  {'depth':>6} {'templates':<44} {'median':>10} {'MEAN':>10} {'max':>11} {'proj/10k':>12}"
    )
    print("  " + "-" * 96)
    per_depth: dict[int, float] = {}
    for depth in DEPTHS:
        times: list[float] = []
        rng = random.Random(1000 + depth)
        for i in range(sample):
            template = STRATUM_TEMPLATES[depth][i % len(STRATUM_TEMPLATES[depth])]
            t0 = time.perf_counter()
            _labelled((template, rng.randrange(2**62)))
            times.append(time.perf_counter() - t0)
        times.sort()
        median = times[len(times) // 2]
        # **Project from the mean, not the median.** Cost spans 50,000x across
        # templates, so a stratum whose rotation is mostly cheap has a median
        # that says nothing about its bill — one solve_both_sides at 1.1 s
        # outweighs a thousand eval_sums, and the median cannot see it. The
        # median is printed beside it only to expose the spread.
        mean = sum(times) / len(times)
        per_depth[depth] = mean
        names = ",".join(STRATUM_TEMPLATES[depth])
        print(
            f"  {depth:>6} {names[:44]:<44} {median * 1000:>8.2f}ms {mean * 1000:>8.2f}ms "
            f"{times[-1] * 1000:>9.2f}ms {mean * 10_000 / 60:>10.1f} min"
        )

    serial = sum(per_depth.values()) / len(per_depth)
    print(f"\n  mean per problem (uniform strata): {serial * 1000:.2f} ms")
    for n in (10_000, 100_000):
        total = serial * n
        print(
            f"  {n:>7,} problems: {total / 3600:>6.2f} h serial   "
            f"{total / workers / 60:>7.1f} min on {workers} workers"
        )
    print("\n  Stratified by depth, and projected from the MEAN within each stratum:")
    print("  cost spans 50,000x across templates, so the median of a stratum whose")
    print("  rotation is mostly cheap says nothing about its bill. Both are printed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # THE DECLARATION LIVES IN FINGERPRINTED CONFIG, not in an argument default
    # (M1-A4 §4). These were hardcoded literals while `--seed` already read the
    # config, so each dataset's meta.json recorded a config_fingerprint claiming
    # that config produced it while the generator fields in that config were
    # never read — a provenance claim true by coincidence rather than by cause
    # (F-33).
    #
    # PROVABLY INERT at the current values: 100_000 == generator.train_set_size
    # and 200 == generator.suite_problems_per_depth, both matching what the
    # existing datasets' meta records. Asserted below so the equality cannot
    # drift back into coincidence.
    assert CFG.generator.train_set_size == 100_000, "the inertness claim moved"
    assert CFG.generator.suite_problems_per_depth == 200, "the inertness claim moved"
    parser.add_argument("--train", type=int, default=CFG.generator.train_set_size)
    parser.add_argument("--suite-size", type=int, default=CFG.generator.suite_problems_per_depth)
    parser.add_argument("--eval-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--suite-only", type=int, default=None, help="regenerate one suite depth")
    args = parser.parse_args()

    if args.preflight:
        preflight(20, args.workers)
        return 0

    started = time.perf_counter()
    anchors: dict[str, str] = {}

    # --- suites first: frozen instruments, and the training set must avoid them
    suite_keys: set[tuple[int, ...]] = set()
    depths = [args.suite_only] if args.suite_only else list(DEPTHS)
    for depth in depths:
        wanted = args.suite_size
        collected: list[Problem] = []
        attempt = 0
        while len(collected) < wanted and attempt < 40:
            batch = generate_pool(
                wanted * 2, args.seed + 7919 * depth + attempt, args.workers, depths=(depth,)
            )
            collected.extend(p for p in bucket(batch).get(depth, []))
            collected = dedup(collected)
            attempt += 1
        if len(collected) < wanted:
            raise SystemExit(f"only {len(collected)}/{wanted} problems at depth {depth}")
        chosen = collected[:wanted]
        path = args.out / "suites" / f"solve_in_{depth}.jsonl"
        anchors[_rel(path)] = write_suite(path, chosen)
        suite_keys |= {problem_key(p) for p in chosen}
        print(f"  suite depth {depth}: {len(chosen)} problems -> {path.name}")

    if args.suite_only:
        print(json.dumps(anchors, indent=2))
        return 0

    # --- training + eval, contamination-screened against the frozen suites
    for mode, count in (("train_100k", args.train), ("eval_held_out", args.eval_size)):
        pool: list[Problem] = []
        attempt = 0
        while len(pool) < count and attempt < 30:
            batch = generate_pool(
                int((count - len(pool)) * 1.6) + 64,
                args.seed + 104_729 * attempt + hash(mode) % 1000,
                args.workers,
            )
            pool = dedup(pool + [p for p in batch if problem_key(p) not in suite_keys])
            attempt += 1
        if len(pool) < count:
            raise SystemExit(f"{mode}: only {len(pool)}/{count}")
        rows = pool[:count]
        path = args.out / DATA_ROOT.name / mode
        meta = write_dataset(path, rows, CFG, mode=mode, seed=args.seed, repo=REPO)
        for name, digest in meta["digests"].items():
            anchors[_rel(path / f"{name}.i32")] = digest
        anchors[_rel(path / "meta.json")] = sha256_file(path / "meta.json")
        print(
            f"  {mode}: {len(rows)} rows, depths {meta['depth_histogram']}, goals {meta['goal_histogram']}"
        )
        if mode == "train_100k":
            suite_keys |= {problem_key(p) for p in rows}  # eval must avoid train too

    anchors_path = args.out / "ANCHORS.sha256"
    anchors_path.parent.mkdir(parents=True, exist_ok=True)
    anchors_path.write_text("".join(f"{d}  {p}\n" for p, d in sorted(anchors.items())))
    print(f"\n  {len(anchors)} digests -> {_rel(anchors_path)}")
    print(f"  elapsed {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
