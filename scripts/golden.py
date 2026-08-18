"""`make golden` — the loop's fast liveness check. CPU always, under three minutes.

**This is the driver at golden config.** Not a loop beside it:
:func:`reckoner.campaign.run` is invoked here exactly as the campaign invokes it,
and what follows are golden's own assertions about what it wrote. One
composition, two doors — the campaign's door asserts the registered fingerprint,
golden's door asserts these.

That distinction is the whole point of the repoint. Golden used to *be* a second
loop, and the cost was three chunks of a defect nobody could see: it ran
``uniform_stub`` every iteration, so the model→search→ring path — the loop's
entire subject — was never exercised by the check whose job was exercising the
loop. A liveness check that certifies a composition the campaign never runs is a
liveness check for the wrong program.

A golden run answers *does the loop still turn*, in the time a person will
actually wait. The plan requires it to run **on CPU regardless of training
device**, because a check that needs the accelerator is a check that stops
running the moment the accelerator is busy. It does not judge performance — that
is the shakedown's job, against pre-registered expectations.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from reckoner.campaign import ANCHOR, CAMPAIGN_FINGERPRINT, golden_config, run
from reckoner.config import config_fingerprint
from reckoner.dataset import sha256_file
from reckoner.logschema import ITERATION_FIELDS, read_rows


def read_rows_raw(path: Path) -> list[dict]:
    """Lines, unvalidated — for artifacts the schema does not govern."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


REPO = Path(__file__).resolve().parents[1]
BUDGET_SECONDS = 180.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    # CPU regardless of what is installed. A liveness check that competes for the
    # accelerator is one that stops being run.
    torch.set_num_threads(min(4, torch.get_num_threads()))

    run_dir = REPO / "runs" / "golden"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    started = time.perf_counter()
    summary = run(run_dir, golden_config(), run_name="golden", anchor=ANCHOR)
    elapsed = time.perf_counter() - started

    per_iteration = summary["iterations"]
    written = read_rows(run_dir / "iterations.jsonl", ITERATION_FIELDS)
    switch = (run_dir / "value_switch.jsonl").read_text().strip().splitlines()
    digests = {r["evaluator_checkpoint_sha256"] for r in written}

    checks: list[tuple[str, bool, str]] = [
        ("rows validate and match commits", len(written) == summary["committed"] + 1, ""),
        ("no alarms", not summary["alarms"], str(summary["alarms"])),
        (
            "every iteration wrote a switch row, abstentions included",
            len(switch) == len(written),
            f"{len(switch)} rows for {len(written)} iterations",
        ),
        (
            "the ring received steps",
            all(r["ring"] > 0 for r in per_iteration),
            str([r["ring"] for r in per_iteration]),
        ),
        # D-A1 §1.1 — THE ASSERTION GOLDEN LACKED. It asserted the model moves;
        # it never asserted the moved model is USED, and that gap is why the stub
        # ran for three chunks unnoticed.
        (
            "the evaluator is the checkpoint, NOT the stub",
            written[0]["evaluator_checkpoint_sha256"] == sha256_file(ANCHOR),
            f"iteration 0 played {written[0]['evaluator_checkpoint_sha256'][:12]}",
        ),
        (
            "and its provenance MOVES as the model trains",
            len(digests) > 1,
            "a digest that never changes is the stub defect wearing a provenance field",
        ),
        # D-A1 §1.2 — pool par DRAWN, not merely enrollable.
        (
            "pool par is drawn, so par escalation is live",
            written[0].get("pool_par_fraction", 0.0) > 0.0,
            f"pool_par_fraction={written[0].get('pool_par_fraction')}",
        ),
        # F-25 — the provenance column must name the config that RAN. `run` is
        # shared with the campaign, and this row previously carried the
        # campaign's fingerprint while golden config executed.
        (
            "rows record the config that produced them, not the campaign's",
            {r["config_fingerprint"] for r in written} == {config_fingerprint(golden_config())},
            f"rows claim {written[0]['config_fingerprint'][:12]}",
        ),
        (
            "and that is NOT the campaign fingerprint",
            written[0]["config_fingerprint"] != CAMPAIGN_FINGERPRINT,
            "a golden row wearing the campaign's fingerprint is the defect F-25 names",
        ),
        (
            "and both par populations appear in the draw-inflation watch",
            bool(written[0]["z_by_par_source"].get("pool"))
            and bool(written[0]["z_by_par_source"].get("bfs")),
            str(written[0]["z_by_par_source"]),
        ),
        # THE CADENCE PATH, exercised end to end (F-29 / the run_pass collision).
        # golden ran `ladder_every = 99` for three chunks, so no test could reach
        # a cadence iteration — and both campaign-blocking defects of the
        # restoration round lived exactly there. These assertions are why the
        # path is now walked rather than merely reachable.
        (
            "the cadence FIRED and wrote its instrument row",
            len(read_rows_raw(run_dir / "instruments.jsonl")) == 1,
            f"{len(read_rows_raw(run_dir / 'instruments.jsonl'))} instrument rows",
        ),
        (
            "the two legs held distinct pass identities",
            (run_dir / "ladder" / "no_regress").is_dir()
            and (run_dir / "ladder" / "primary").is_dir(),
            "a shared root silently skips the second leg's pass",
        ),
        (
            "the ladder iteration carries its watchlist columns",
            all(c in written[-1] for c in ("family_remaining", "novel_misses", "pass_misses")),
            str(
                [c for c in ("family_remaining", "novel_misses", "pass_misses") if c in written[-1]]
            ),
        ),
        (
            "and the watchlist partitions the pass's misses",
            written[-1].get("family_remaining", 0) + written[-1].get("novel_misses", 0)
            == written[-1].get("pass_misses", -1),
            f"{written[-1].get('family_remaining')} + {written[-1].get('novel_misses')}"
            f" == {written[-1].get('pass_misses')}",
        ),
        (
            "non-cadence iterations declare the watchlist ABSENT with a reason",
            "family_remaining" in (written[0].get("absent") or {}),
            str(list(written[0].get("absent") or {})),
        ),
    ]

    print("\n  GOLDEN\n")
    for name, ok, detail in checks:
        print(f"    {'ok  ' if ok else 'FAIL'} {name}{'  — ' + detail if detail else ''}")
    print(
        f"\n  {elapsed:.1f}s of a {BUDGET_SECONDS:.0f}s budget, CPU, "
        f"{torch.get_num_threads()} threads"
    )

    shutil.rmtree(run_dir, ignore_errors=True)
    if not all(ok for _, ok, _ in checks):
        return 1
    if elapsed > BUDGET_SECONDS:
        print(
            f"  OVER BUDGET: {elapsed:.1f}s > {BUDGET_SECONDS:.0f}s — a golden run people "
            "will not wait for is one they stop running"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
