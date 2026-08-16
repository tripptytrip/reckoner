"""Reconcile the sweep's two invocations into one stamped union record.

Part-0b ran twice: five rungs on 2026-08-16 at 08:34, then the P11B-A3 downward
extension at 11:13. The extension held the **pre-reconciliation** code in memory
— it started before P11B-A4 landed — so on exit it applied the old ``select()``
to four points and wrote a four-point file over the canonical path, with the old
all-above node degenerately asking for a downward extension below ``sims = 1``.

This performs no measurement. It merges two artifacts and re-judges the union
with the committed rule.

**Quarantine is the first act.** Between the extension's exit and this merge, the
canonical path holds a four-point file carrying an old-``select()`` verdict —
wrong-looking-right to any future reader if a crash lands in that window. So the
exit file is *moved* to ``.4pt.json`` before anything is written, which leaves
the canonical path **absent** rather than misleading. At rest it is then always
one of three things: the five-point record, absent, or the stamped union.

**And the union carries its provenance in-file** — both source digests and the
``select()`` version that judged it — so no future reader has to reconstruct
which rule produced the verdict they are reading.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chunk11_part0b as sweep

from reckoner.dataset import git_sha, write_record

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
CANON = RUNS / "chunk11_part0b_sweep.json"
FIVE = RUNS / "chunk11_part0b_sweep.5pt.json"
FOUR = RUNS / "chunk11_part0b_sweep.4pt.json"

# The amendments that govern the rule applied below, newest last.
GOVERNED_BY = ("P11B-A3", "P11B-A4", "P11B-A5")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quarantine() -> None:
    """Move the extension's exit file aside, leaving canonical absent."""
    if not CANON.exists():
        if FOUR.exists():
            print(f"  already quarantined: {FOUR.name}")
            return
        raise SystemExit("neither the canonical record nor its quarantine exists")
    if FOUR.exists():
        raise SystemExit(f"{FOUR.name} already exists; refusing to overwrite it")
    CANON.rename(FOUR)
    print(f"  quarantined {CANON.name} -> {FOUR.name} (canonical now absent)")


def main() -> int:
    if not FIVE.exists():
        raise SystemExit(f"the five-point record is missing: {FIVE}")

    quarantine()

    five = json.loads(FIVE.read_text())
    four = json.loads(FOUR.read_text())

    fp5 = five.get("protocol", {}).get("config_fingerprint")
    fp4 = four.get("protocol", {}).get("config_fingerprint")
    if fp5 != fp4:
        raise SystemExit(
            f"the two invocations carry different config fingerprints "
            f"({fp5} vs {fp4}); a union across two protocols is refused"
        )

    union = sweep.merge_points(five["sweep"], four["sweep"])
    selection = sweep.select(union)

    report = {
        "expectations_frozen_at": five["expectations_frozen_at"],
        "git_sha": git_sha(REPO),
        "protocol": five["protocol"],
        "target": five["target"],
        "window": five["window"],
        "invocations": {
            "amendment": "P11B-A3",
            "note": "one rule, two invocations; merged by rung key with "
            "collision-refusal and re-judged by the committed select()",
            "sources": [
                {
                    "file": FIVE.name,
                    "sha256": digest(FIVE),
                    "rungs": [p["sims"] for p in five["sweep"]],
                },
                {
                    "file": FOUR.name,
                    "sha256": digest(FOUR),
                    "rungs": [p["sims"] for p in four["sweep"]],
                    "superseded_selection": four["selection"]["needs"],
                },
            ],
        },
        "selector": {
            "git_sha": git_sha(REPO),
            "governed_by": list(GOVERNED_BY),
            "note": "the four-point exit file was judged by the pre-A4 select(), "
            "whose all-above node had one polarity; this union is judged by the "
            "reconciled rule",
        },
        "sweep": union,
        "selection": selection,
    }
    write_record(CANON, report)

    print(f"  union rungs: {[p['sims'] for p in union]}")
    print(f"  at-par:      {[p['at_par'] for p in union]}")
    print("\n  SELECTION\n")
    print(json.dumps(selection, indent=2))
    print(f"\n  wrote {CANON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
