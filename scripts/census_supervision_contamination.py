"""A1: does the unroll walk the supervision set through suite start states?

The problem-level contamination test asks whether a *problem* in a training set
is also a suite problem. ``train_100k`` passes it. But Phase-1 supervision holds
the **intermediate states** of those problems' derivations, and an intermediate
state of a deep problem can be, exactly, the start state of a shallow suite
problem — solving ``9x + (-28) = 44`` passes through ``9x = 72``, and if a suite
problem *is* ``9x = 72`` under the same goal, the model trains on the instrument
verbatim.

Inheritance cannot see this. ``train_100k``'s problems are clean; its
*derivations* are a different set of states, created by the unroll, and nothing
had ever asked about them.

**Decision rule, pre-stated in the brief's amendment A1 and encoded here so it
cannot drift:** overlap ≤ 1% of examples → remove the colliding rows, re-digest,
report the removed count. Overlap > 1% → **STOP**, exit non-zero, remove nothing;
that scale means something structural and is a joint ruling.

Keys are ``(identity_key(expr), goal)`` per A1, through the one shared identity
normalizer (inherited law). The stricter ``(identity_key, goal, target)`` count
is reported beside it as a **diagnostic only** — it does not enter the decision,
because the decision rule was fixed before the numbers were seen.

Writes ``runs/supervision_contamination.json``. ``--apply`` performs the removal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reckoner.dataset import git_sha, read_suite, sha256_file, write_record
from reckoner.episode import decode_state
from reckoner.expr import identity_key
from reckoner.rules import RULESET_VERSION
from reckoner.vocab import VOCAB_VERSION

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
THRESHOLD = 0.01  # A1, pre-stated. Not a tunable.

ARRAYS = ("tokens", "lengths", "action", "steps_remaining", "depth", "goal")


def suite_keys() -> dict[str, set[tuple]]:
    """``(identity_key(expr), goal)`` per suite. The instrument's own identities."""
    out = {}
    for path in sorted(SUITES.glob("solve_in_*.jsonl")):
        keys = set()
        for row in read_suite(path):
            goal, _target, expr = decode_state(tuple(row["tokens"]))
            keys.add((identity_key(expr), goal))
        out[path.stem] = keys
    return out


def supervision_keys(data: Path) -> list[tuple[tuple[int, ...], int]]:
    """``(identity_key(expr), goal)`` for every state in a supervision set."""
    meta = json.loads((data / "meta.json").read_text())
    n, width = meta["n"], meta["max_len"]
    tokens = np.memmap(data / "tokens.i32", dtype=np.int32, mode="r", shape=(n, width))
    lengths = np.memmap(data / "lengths.i32", dtype=np.int32, mode="r", shape=(n,))
    out = []
    for i in range(n):
        goal, _target, expr = decode_state(tuple(int(t) for t in tokens[i, : lengths[i]]))
        out.append((identity_key(expr), goal))
    return out


def census_against_supervision(data: Path, reference: Path) -> dict:
    """State-level overlap between two supervision sets.

    A problem-level split does **not** give a state-level split. Different
    problems have different start states and can still pass through the same
    intermediate — which is F-08's mechanism pointed at the train/eval boundary,
    where it does not merely leak an instrument but inflates a DONE-WHEN metric.
    The same pre-stated threshold applies: ≤1% removable, >1% is structural and
    is a joint ruling.
    """
    ref = set(supervision_keys(reference))
    keys = supervision_keys(data)
    n = len(keys)
    hits = [i for i, k in enumerate(keys) if k in ref]

    steps = np.memmap(
        data / "steps_remaining.i32",
        dtype=np.int32,
        mode="r",
        shape=(json.loads((data / "meta.json").read_text())["n"],),
    )
    by_remaining: dict[int, int] = {}
    for i in hits:
        by_remaining[int(steps[i])] = by_remaining.get(int(steps[i]), 0) + 1

    return {
        "data": data.name,
        "reference": reference.name,
        "examples": n,
        "shared_examples": len(hits),
        "shared_fraction": round(len(hits) / n, 6) if n else 0.0,
        "distinct_shared_states": len({keys[i] for i in hits}),
        "by_steps_remaining": dict(sorted(by_remaining.items())),
        "threshold": THRESHOLD,
        "verdict": "REMOVE" if n and len(hits) / n <= THRESHOLD else "STOP",
        "colliding_indices": sorted(hits),
    }


def census(data: Path) -> dict:
    meta = json.loads((data / "meta.json").read_text())
    n, width = meta["n"], meta["max_len"]
    tokens = np.memmap(data / "tokens.i32", dtype=np.int32, mode="r", shape=(n, width))
    lengths = np.memmap(data / "lengths.i32", dtype=np.int32, mode="r", shape=(n,))
    steps_remaining = np.memmap(data / "steps_remaining.i32", dtype=np.int32, mode="r", shape=(n,))
    depth = np.memmap(data / "depth.i32", dtype=np.int32, mode="r", shape=(n,))

    per_suite = suite_keys()
    everywhere = {k for keys in per_suite.values() for k in keys}
    strict_suite = set()
    for path in sorted(SUITES.glob("solve_in_*.jsonl")):
        for row in read_suite(path):
            goal, target, expr = decode_state(tuple(row["tokens"]))
            strict_suite.add((identity_key(expr), goal, target))

    hits: dict[str, list[int]] = {name: [] for name in per_suite}
    colliding_rows: set[int] = set()
    strict_hits = 0
    start_state_hits = 0

    for i in range(n):
        goal, target, expr = decode_state(tuple(int(t) for t in tokens[i, : lengths[i]]))
        key = (identity_key(expr), goal)
        if key not in everywhere:
            continue
        colliding_rows.add(i)
        if (identity_key(expr), goal, target) in strict_suite:
            strict_hits += 1
        # steps_remaining == depth marks the derivation's own start state. Those
        # ARE train_100k problems, already gated by the problem-level test — a hit
        # here would contradict a passing gate, so it is counted separately.
        if int(steps_remaining[i]) == int(depth[i]):
            start_state_hits += 1
        for name, keys in per_suite.items():
            if key in keys:
                hits[name].append(i)

    total = len(colliding_rows)
    return {
        "examples": n,
        "colliding_examples": total,
        "colliding_fraction": round(total / n, 6),
        "per_suite_examples": {name: len(rows) for name, rows in sorted(hits.items())},
        "distinct_colliding_keys": len(
            {
                (identity_key(decode_state(tuple(int(t) for t in tokens[i, : lengths[i]]))[2]))
                for i in sorted(colliding_rows)
            }
        ),
        "start_state_hits": start_state_hits,
        "strict_key_hits_diagnostic": strict_hits,
        "threshold": THRESHOLD,
        "verdict": "REMOVE" if total / n <= THRESHOLD else "STOP",
        "colliding_indices": sorted(colliding_rows),
    }


def apply_removal(data: Path, report: dict) -> dict:
    """Drop the colliding rows, re-digest, and say so in the meta."""
    meta = json.loads((data / "meta.json").read_text())
    n, width = meta["n"], meta["max_len"]
    drop = np.zeros(n, dtype=bool)
    drop[report["colliding_indices"]] = True
    keep = ~drop

    loaded = {}
    for name in ARRAYS:
        shape = (n, width) if name == "tokens" else (n,)
        loaded[name] = np.array(
            np.memmap(data / f"{name}.i32", dtype=np.int32, mode="r", shape=shape)
        )[keep]

    survivors = int(keep.sum())
    new_width = int(loaded["lengths"].max())
    if new_width < width:
        loaded["tokens"] = loaded["tokens"][:, :new_width]
    for name, array in loaded.items():
        array.astype(np.int32).tofile(data / f"{name}.i32")

    meta["n"] = survivors
    meta["max_len"] = new_width
    meta["digests"] = {name: sha256_file(data / f"{name}.i32") for name in ARRAYS}
    meta["depth_histogram"] = {
        int(d): int(c) for d, c in zip(*np.unique(loaded["depth"], return_counts=True), strict=True)
    }
    meta["goal_histogram"] = {
        int(g): int(c) for g, c in zip(*np.unique(loaded["goal"], return_counts=True), strict=True)
    }
    meta["suite_collisions_removed"] = report["colliding_examples"]
    meta["suite_collision_policy"] = (
        "A1 (2026-08-15): supervision states matching a suite start state on "
        "(identity_key, goal) are removed. Inherited problem-level cleanliness does "
        "not cover derivation intermediates — train_100k's problems are clean, its "
        "derivations pass through states the suites also use as starts."
    )
    meta["git_sha"] = git_sha(REPO)
    meta["ruleset_version"] = RULESET_VERSION
    meta["vocab_version"] = VOCAB_VERSION
    (data / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "runs" / "data" / "phase1_train")
    parser.add_argument("--apply", action="store_true", help="remove the collisions")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="also census state-level overlap against another supervision set",
    )
    parser.add_argument(
        "--out-name",
        default="supervision_contamination",
        help="basename for the record under runs/",
    )
    args = parser.parse_args()

    print("  A1 — SUPERVISION CONTAMINATION CENSUS\n")

    if args.reference is not None:
        pair = census_against_supervision(args.data, args.reference)
        print(f"  {pair['data']} vs {pair['reference']} — STATE-level overlap")
        print(f"    examples               : {pair['examples']:,}")
        print(f"    shared with reference  : {pair['shared_examples']:,}")
        print(f"    as a fraction          : {pair['shared_fraction']:.4%}")
        print(f"    distinct shared states : {pair['distinct_shared_states']:,}")
        print(f"    by steps_remaining     : {pair['by_steps_remaining']}")
        print(f"    threshold (pre-stated) : {THRESHOLD:.2%}")
        print(f"    verdict                : {pair['verdict']}\n")
        record = {k: v for k, v in pair.items() if k != "colliding_indices"}
        record["git_sha"] = git_sha(REPO)
        record["dataset_digests"] = json.loads((args.data / "meta.json").read_text())["digests"]
        write_record(REPO / "runs" / f"{args.out_name}.json", record)
        print(f"  wrote runs/{args.out_name}.json")
        if pair["verdict"] == "STOP":
            print(
                f"\n  STOP — {pair['shared_fraction']:.2%} exceeds the {THRESHOLD:.0%} "
                f"threshold. Nothing removed; this is a joint ruling."
            )
            return 2
        return 0

    report = census(args.data)

    print(f"  supervision examples      : {report['examples']:,}")
    print(f"  colliding examples        : {report['colliding_examples']:,}")
    print(f"  as a fraction             : {report['colliding_fraction']:.6f}")
    print(f"  distinct colliding states : {report['distinct_colliding_keys']:,}")
    print(
        f"  of which START states     : {report['start_state_hits']:,}  "
        f"(non-zero would contradict the problem-level gate)"
    )
    print(
        f"  strict (key,goal,target)  : {report['strict_key_hits_diagnostic']:,}  "
        f"[diagnostic only — not the decision]\n"
    )
    print(f"  {'suite':>12} {'colliding examples':>20}")
    for name, count in report["per_suite_examples"].items():
        print(f"  {name:>12} {count:>20,}")

    print(f"\n  threshold (A1, pre-stated): {THRESHOLD:.2%}")
    print(f"  verdict                   : {report['verdict']}")

    def emit(rep: dict, removed: int) -> None:
        """The record pins the digests it was computed against.

        A contamination record that does not name the bytes it censused is a
        record that goes stale silently — the dataset is rebuilt, the JSON still
        says zero, and the gate reading it still passes. Same defect as F-02:
        a claim outliving the computation that earned it.
        """
        record = {k: v for k, v in rep.items() if k != "colliding_indices"}
        record["removed_examples"] = removed
        record["dataset_digests"] = json.loads((args.data / "meta.json").read_text())["digests"]
        record["git_sha"] = git_sha(REPO)
        write_record(REPO / "runs" / f"{args.out_name}.json", record)

    if report["verdict"] == "STOP":
        emit(report, removed=0)
        print(f"\n  wrote runs/{args.out_name}.json")
        print(
            f"\n  STOP — {report['colliding_fraction']:.2%} exceeds the {THRESHOLD:.0%} "
            f"threshold. Nothing removed. This is a joint ruling per A1."
        )
        return 2

    if not args.apply:
        emit(report, removed=0)
        print(f"\n  wrote runs/{args.out_name}.json")
        print("\n  census only — pass --apply to remove")
        return 0

    meta = apply_removal(args.data, report)
    print(
        f"\n  removed {report['colliding_examples']:,} examples; "
        f"{meta['n']:,} remain, max_len {meta['max_len']}, digests rewritten"
    )
    # Re-census the modified data rather than asserting the removal worked. The
    # record that ships is a MEASUREMENT of the shipped bytes, not a deduction
    # from the operation that produced them.
    after = census(args.data)
    emit(after, removed=report["colliding_examples"])
    print(
        f"  re-censused after removal: {after['colliding_examples']} collisions "
        f"in {after['examples']:,} examples"
    )
    print(f"\n  wrote runs/{args.out_name}.json")
    return 0 if after["colliding_examples"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
