"""Re-verify every suite depth label, independently of the generation that made it.

**Chunk 5 gate**, and it is expensive on purpose: 1,200 problems, three BFS runs
each. It lives in a script rather than the test suite because a full pass takes
minutes — depth-6 SOLVE labelling alone is ~4.5 s per problem — and a `make test`
that takes ten minutes is a `make test` people stop running.

The artifact it writes is what the test suite checks: `runs/suite_verification.json`
records what was verified and the digest of each suite it verified, so a suite
edited after verification fails loudly rather than inheriting an old pass.

Two-sided, because *minimum* is two claims:

  * a derivation of exactly ``depth`` steps exists, and
  * BFS at cap ``depth − 1`` finds nothing.

Either alone is satisfiable by a wrong label.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from reckoner.config import Config
from reckoner.dataset import read_suite, sha256_file, suite_problem
from reckoner.episode import bfs_par, bfs_solution

REPO = Path(__file__).resolve().parents[1]
CFG = Config()


def _check(args: tuple[dict, int]) -> tuple[bool, str]:
    row, depth = args
    problem = suite_problem(row)
    if bfs_par(problem, CFG) != depth:
        return False, "bfs_par disagrees with the stored label"
    path = bfs_solution(problem, CFG)
    if path is None or len(path) != depth:
        return False, "no derivation of exactly the stored length"
    if bfs_solution(problem, CFG, cap=depth - 1) is not None:
        return False, "a shorter derivation exists — the label is not the minimum"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--suites", type=Path, default=REPO / "runs" / "suites")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "suite_verification.json")
    args = parser.parse_args()

    started = time.perf_counter()
    record: dict = {"suites": {}, "total": 0, "failures": []}
    for depth in range(1, 7):
        path = args.suites / f"solve_in_{depth}.jsonl"
        if not path.exists():
            continue
        rows = read_suite(path)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_check, [(r, depth) for r in rows], chunksize=4))
        bad = [i for i, (ok, _) in enumerate(results) if not ok]
        for i in bad:
            record["failures"].append({"suite": path.name, "row": i, "why": results[i][1]})
        record["suites"][path.name] = {
            "problems": len(rows),
            "verified": len(rows) - len(bad),
            "sha256": sha256_file(path),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        record["total"] += len(rows)
        print(
            f"  {path.name}: {len(rows) - len(bad)}/{len(rows)} verified "
            f"in {time.perf_counter() - t0:.1f}s"
        )

    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    verified = sum(s["verified"] for s in record["suites"].values())
    print(f"\n  {verified}/{record['total']} labels re-verified in {record['elapsed_seconds']}s")
    if record["failures"]:
        print(f"  FAILURES: {record['failures'][:5]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
