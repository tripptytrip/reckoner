"""Completion markers: the primitive, and the ergonomics that make it win.

The law is not new. ``ladderpass`` has stated it since chunk 10:

    Completion is a marker file, never a process check. A waiter that greps for
    a running process concludes "done" from "not running", which is also what a
    crash looks like.

It was stated, and then broken five times in one session — every instance a
shell one-liner (``pgrep -f``, ``pkill -f``) typed because it was one word and
the correct thing was a paragraph. Twice it matched the waiter's *own* command
line and killed the shell doing the waiting; twice it reported RUNNING for work
that had finished; once it reported a live install as finished.

Rule and method both failed against ergonomics, so this is the helpers-prevent
rung: **waiting on a marker is now shorter to type than the wrong thing.**

    reckoner-marker run  diag -- python scripts/long_job.py
    reckoner-marker wait diag

The load-bearing property is asymmetric and deliberate:

* a marker **present** means finished, and carries the exit code
* a marker **absent** means *unknown* — never "done"

That asymmetry is the whole point. Process-absence is ambiguous between success,
crash and never-started; marker-presence is not.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

#: Written after the work lands, containing the exit code and nothing else.
SUFFIX = ".done"


class MarkerTimeout(RuntimeError):
    """The marker did not appear. **This is not "done" — it is "unknown".**"""


def marker_path(name: str, root: Path) -> Path:
    return Path(root) / f"{name}{SUFFIX}"


def status(name: str, root: Path) -> int | None:
    """Exit code if the work completed, ``None`` if that is still unknown.

    ``None`` covers running, crashed and never-started alike, because from the
    outside those are the same observation — which is exactly what a process
    check gets wrong by calling all three "not running".
    """
    path = marker_path(name, root)
    if not path.exists():
        return None
    text = path.read_text().strip()
    return int(text) if text.lstrip("-").isdigit() else 0


def clear(name: str, root: Path) -> None:
    """Remove a stale marker. Always call before launching, never after."""
    marker_path(name, root).unlink(missing_ok=True)


def launch(name: str, argv: list[str], root: Path, *, log: Path | None = None) -> None:
    """Run *argv* detached, writing the marker when it exits.

    The marker is written by the same shell that runs the work, so it cannot be
    forgotten by a caller and cannot be written by a caller that has itself died.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    clear(name, root)
    log = log or root / f"{name}.log"
    inner = " ".join(_quote(a) for a in argv)
    script = f"{inner} > {_quote(str(log))} 2>&1; echo $? > {_quote(str(marker_path(name, root)))}"
    subprocess.Popen(
        ["setsid", "nohup", "bash", "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait(name: str, root: Path, *, timeout: float = 86_400.0, poll: float = 5.0) -> int:
    """Block until the marker appears; return its exit code.

    Raises :class:`MarkerTimeout` rather than returning on timeout, because a
    waiter that returns a value on "I never found out" is a waiter whose callers
    will treat not-knowing as success.
    """
    deadline = time.monotonic() + timeout
    while True:
        code = status(name, root)
        if code is not None:
            return code
        if time.monotonic() >= deadline:
            raise MarkerTimeout(
                f"{marker_path(name, root)} did not appear within {timeout:.0f}s. "
                "This is NOT 'done' — the work may be running, crashed, or never "
                "started, and those are indistinguishable from out here. Check the "
                "log before concluding anything."
            )
        time.sleep(poll)


def _quote(arg: str) -> str:
    return "'" + arg.replace("'", "'\\''") + "'"


def main(argv: list[str] | None = None) -> int:
    """``reckoner-marker run NAME -- CMD...`` / ``reckoner-marker wait NAME``."""
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(os.environ.get("MARKER_ROOT", "/tmp/reckoner-markers"))
    if len(args) >= 2 and args[0] == "run":
        name = args[1]
        rest = args[3:] if len(args) > 2 and args[2] == "--" else args[2:]
        if not rest:
            print("usage: reckoner-marker run NAME -- COMMAND...", file=sys.stderr)
            return 2
        launch(name, rest, root)
        print(f"{name}: launched, marker at {marker_path(name, root)}")
        return 0
    if len(args) == 2 and args[0] == "wait":
        try:
            code = wait(args[1], root)
        except MarkerTimeout as exc:
            print(exc, file=sys.stderr)
            return 3
        print(f"{args[1]}: finished, exit {code}")
        return code
    if len(args) == 2 and args[0] == "status":
        code = status(args[1], root)
        print(f"{args[1]}: {'unknown' if code is None else f'finished, exit {code}'}")
        return 0 if code == 0 else 1
    print(__doc__.strip().splitlines()[0], file=sys.stderr)
    print("usage: reckoner-marker {run NAME -- CMD... | wait NAME | status NAME}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
