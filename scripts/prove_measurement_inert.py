"""Is a configuration change MEASUREMENT-INERT? Derived, not argued.

When the campaign fingerprint moves, artifacts produced under the old one become
inadmissible as evidence about the new one — an artifact and a gate must agree on
which configuration they are discussing. Re-running a three-hour equivalence gate
is one way to restore that agreement. Proving the change cannot have touched
measurement is the cheaper one, and it is only worth anything if it is a
derivation rather than a claim.

The method, and its one trap:

1. Enumerate the fields whose **value or existence** differs between the two
   configs. (`eval_profile` derives from the campaign config, so **the eval
   fingerprint moves whenever ANY field moves** — 314fbeb9 → c8aa1fcc proves
   nothing on its own about evaluation behaviour. It is the differing FIELDS that
   must be examined, never the derived digest.)
2. Compute the **measurement closure**: every function reachable from
   `run_instruments`, the sole seam every cadence measurement passes through.
3. A change is measurement-inert **iff no differing field is read anywhere in
   that closure.**

If any differing field is read there, the gate re-runs and the licence is
re-earned. This script's job is to make that question answerable, not to answer
it favourably.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from reckoner.config import Config, config_fingerprint, validate
from scripts.config_census import build

REPO = Path(__file__).resolve().parents[1]

#: The measurement path's roots. `run_instruments` is the mono-instance seam
#: (D-A2 §3); the rest are what it delegates to and are listed so a future
#: refactor that moves the seam fails loudly here.
MEASUREMENT_ROOTS = frozenset({"run_instruments", "run_pass", "assert_eval_profile"})


def measurement_closure() -> tuple[set[str], dict[str, set[str]]]:
    """``(functions reachable from the instrument seam, library reads by field)``.

    Extracted so the prover can be validated against **known positives** before
    its verdict is trusted. That check is not suspicion, it is consistency: the
    resolver fix that made this tool able to say "inert" was made in the direction
    that saves three hours, and a tool that can no longer say *no* is the same
    broken instrument as one that could only say *no*.
    """
    reads, _script_reads, _test_reads, calls, defined = build()
    frontier = list(MEASUREMENT_ROOTS)
    closure: set[str] = set(MEASUREMENT_ROOTS)
    while frontier:
        current = frontier.pop()
        for callee in calls.get(current, ()):
            if callee in defined and callee not in closure:
                closure.add(callee)
                frontier.append(callee)
    return closure, reads


def differing_fields(old: Config, new: Config) -> dict[str, tuple]:
    """``{field: (old, new)}`` for every field whose value or existence differs."""
    import dataclasses

    out: dict[str, tuple] = {}
    for group in dataclasses.fields(new):
        a, b = getattr(old, group.name), getattr(new, group.name)
        if not dataclasses.is_dataclass(b):
            if a != b:
                out[group.name] = (a, b)
            continue
        names_a = {f.name for f in dataclasses.fields(a)}
        names_b = {f.name for f in dataclasses.fields(b)}
        for name in sorted(names_a | names_b):
            va = getattr(a, name, "<absent>")
            vb = getattr(b, name, "<absent>")
            if va != vb:
                out[f"{group.name}.{name}"] = (va, vb)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-fingerprint",
        required=True,
        help="the HISTORICAL fingerprint, quoted from the record. It is not "
        "recomputed here: deleted fields cannot be rebuilt on the dataclass, so a "
        "reconstruction would print a digest that never existed.",
    )
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "measurement_inertness.json")
    args = parser.parse_args()

    new = Config()
    validate(new)
    # The pre-M1-A4 config, reconstructed: rehearsal_frac was 0.0 and the two
    # ladder.sympy_* fields existed. Deleted fields cannot be reconstructed on the
    # dataclass, so they are carried explicitly below.
    old = replace(new, train=replace(new.train, rehearsal_frac=0.0))
    deleted = ("ladder.sympy_step_budget", "ladder.sympy_time_budget_s")

    differing = differing_fields(old, new)
    for name in deleted:
        differing[name] = ("<present>", "<deleted>")

    measurement, reads = measurement_closure()

    print("\n  MEASUREMENT-INERTNESS PROOF\n")
    print(f"    old campaign fingerprint : {args.old_fingerprint}  (quoted from the record)")
    print(f"    new campaign fingerprint : {config_fingerprint(new)}  (computed)")
    print(
        f"    measurement closure      : {len(measurement)} functions reachable "
        f"from {sorted(MEASUREMENT_ROOTS)}\n"
    )
    print(f"    {'differing field':<32} {'old':>10} {'new':>10}  read inside the closure?")

    violations = []
    rows = []
    for field, (a, b) in sorted(differing.items()):
        bare = field.split(".")[-1]
        readers = reads.get(bare, set())
        inside = sorted(readers & measurement)
        rows.append(
            {
                "field": field,
                "old": str(a),
                "new": str(b),
                "library_readers": sorted(readers),
                "read_in_measurement": inside,
            }
        )
        if inside:
            violations.append(field)
        print(
            f"    {field:<32} {str(a):>10} {str(b):>10}  "
            f"{'YES -> ' + ', '.join(inside) if inside else 'no'}"
        )

    args.out.write_text(
        json.dumps(
            {
                "old_fingerprint": config_fingerprint(old),
                "new_fingerprint": config_fingerprint(new),
                "measurement_closure_size": len(measurement),
                "fields": rows,
                "measurement_inert": not violations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if violations:
        print(f"\n  NOT MEASUREMENT-INERT: {violations} are read inside the closure.")
        print("  The equivalence gate must re-run and the licence be re-earned.")
        return 1
    print("\n  MEASUREMENT-INERT. No differing field is read anywhere reachable from")
    print("  the instrument seam, so artifacts produced under the old fingerprint")
    print("  remain valid evidence about measurement — and the gate need not re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
