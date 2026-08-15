"""Session hooks. Currently one: **the test count writes itself.**

F-16 recorded a commit message reporting 786 passed when the run reported 772,
because the message was composed in the same shell invocation as the run and the
number had to be authored before the measurement existed. The correction was a
rule — *compose the message after the run returns* — and it was **broken one
commit later**, by the same ergonomics: message and run are convenient to send
together, and a rule that fights convenience loses.

So the number stops being authored. `pytest` writes `runs/testcount.json` at the
end of every session, and a commit quotes that file or includes it. A count
nobody types cannot be a count nobody measured.

It records **what was selected**, not just what passed: a partial run
(`pytest tests/test_ladder.py`) legitimately reports a small number, and a record
that did not say so would be a new way to write a misleading total.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "runs" / "testcount.json"


def pytest_sessionfinish(session, exitstatus) -> None:
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:  # -p no:terminal, or an embedded run
        return
    counts = {
        outcome: len(reporter.stats.get(outcome, []))
        for outcome in ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
    }
    selected = list(session.config.args)
    whole_suite = selected in ([str(REPO / "tests")], ["tests/"], ["tests"], [])
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(
        json.dumps(
            {
                "collected": session.testscollected,
                **counts,
                "exitstatus": int(exitstatus),
                # The honest qualifier. Without it, a targeted run's number reads
                # exactly like a full-suite number, which is F-16's failure in a
                # new costume.
                "whole_suite": whole_suite,
                "selected": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
