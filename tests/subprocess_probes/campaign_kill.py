"""Drive the campaign and SIGKILL it at a declared commit boundary.

Run as a FILE, never as an embedded string — the extracted-probe pattern the
determinism gate uses, for the same reason: a program in its own file has
nothing for a text edit to corrupt silently, and it lints, imports and reads
like the code it is.

    python campaign_kill.py RUN_DIR [ITERATION PHASE]

With no kill argument it runs to completion — the *uninterrupted* baseline the
killed runs are compared against, produced by the same program so the comparison
cannot drift.

``PHASE`` is ``before_row`` (ring and state on disk, row not yet written) or
``before_latest`` (row written, ``LATEST`` not yet renamed). Those are the two
points `resume.py` names as provisional, and the only two where a kill can
produce artifacts for an iteration that never happened.

**The kill is a real SIGKILL to self**, not an exception: an exception unwinds,
and unwinding is precisely what a crashed process does not do.
"""

import os
import signal
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))

from campaign_fixture import ANCHOR, golden_config  # noqa: E402

from reckoner.campaign import run  # noqa: E402

if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    kill_at = (int(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 3 else None

    def on_commit(iteration: int, phase: str) -> None:
        if kill_at == (iteration, phase):
            sys.stdout.flush()
            os.kill(os.getpid(), signal.SIGKILL)

    run(
        run_dir,
        golden_config(campaign={"iterations": 3}),
        run_name="fixture",
        anchor=ANCHOR,
        on_commit=on_commit,
    )
    print("COMPLETED")
