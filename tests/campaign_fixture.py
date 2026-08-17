"""The driver harness. **One law: it calls the doors, never the organs.**

A fixture that re-composes self-play, training and the commit for testability is
the second composition the one-composition rule forbids, wearing test clothing —
and it would certify a loop the campaign never runs, which is the exact defect
`golden` carried for three chunks. So everything here invokes
:func:`reckoner.campaign.run` or :func:`reckoner.campaign.run_campaign` and then
*reads what they wrote*.

Two faces, both precedented:

* **in-process** — :func:`drive`, for the D-A1 assertions and row/state
  inspection. Fast, and it can pass the ``on_commit`` hook.
* **subprocess** — ``tests/subprocess_probes/campaign_kill.py``, for the resume
  gate. A real SIGKILL needs a real process; the extracted-probe pattern is the
  one the determinism gate already uses, for the same reason.

`golden` is this fixture at golden config plus golden's own assertion set, which
is what "golden = the driver at golden config" cashes out to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from reckoner.campaign import ANCHOR, golden_config
from reckoner.config import Config
from reckoner.logschema import ITERATION_FIELDS, VALUE_SWITCH_FIELDS, alarm_census, read_rows

REPO = Path(__file__).resolve().parents[1]

__all__ = ["ANCHOR", "DriverRun", "drive", "drive_campaign", "golden_config"]


@dataclass
class DriverRun:
    """A completed driver invocation, and what a test may assert about it.

    Everything here is **read from disk after the fact**. Nothing is captured by
    instrumenting the loop, because a harness that watches the internals is a
    harness that can pass while the artifacts are wrong — and the artifacts are
    the deliverable.
    """

    run_dir: Path
    cfg: Config
    summary: dict

    @property
    def rows(self) -> list[dict]:
        path = self.run_dir / "iterations.jsonl"
        return read_rows(path, ITERATION_FIELDS) if path.exists() else []

    @property
    def switch_rows(self) -> list[dict]:
        path = self.run_dir / "value_switch.jsonl"
        return read_rows(path, VALUE_SWITCH_FIELDS) if path.exists() else []

    @property
    def alarms(self) -> dict:
        return alarm_census(self.rows)

    @property
    def committed(self) -> int | None:
        marker = self.run_dir / "LATEST"
        return int(marker.read_text().strip()) if marker.exists() else None

    def raw_rows(self) -> list[dict]:
        """Rows without schema validation — for asserting on a torn or partial run."""
        path = self.run_dir / "iterations.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def drive(run_dir: Path, cfg: Config | None = None, *, on_commit=None, **kwargs) -> DriverRun:
    """Invoke the driver **through `run`** and collect what it wrote."""
    from reckoner.campaign import run

    cfg = cfg or golden_config()
    summary = run(run_dir, cfg, run_name="fixture", anchor=ANCHOR, on_commit=on_commit, **kwargs)
    return DriverRun(run_dir=run_dir, cfg=cfg, summary=summary)


def drive_campaign(run_dir: Path, cfg: Config) -> DriverRun:
    """Invoke the driver **through `run_campaign`** — the door that asserts.

    Separate from :func:`drive` because the two doors differ in exactly one
    respect, and that difference is what the fingerprint gate tests.
    """
    from reckoner.campaign import run_campaign

    summary = run_campaign(run_dir, cfg, anchor=ANCHOR)
    return DriverRun(run_dir=run_dir, cfg=cfg, summary=summary)
