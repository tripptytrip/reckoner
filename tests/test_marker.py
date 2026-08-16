"""Completion markers, and the asymmetry that is the whole point.

`ladderpass` has stated the law since chunk 10 — completion is a marker file,
never a process check — and it was broken five times in one session, every
instance a shell one-liner typed because it was one word. So the law gets a
mechanism, and the mechanism gets the property the law is about:

    marker present -> finished, with the exit code
    marker absent  -> UNKNOWN, never "done"

The tests below are mostly about the second line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reckoner.marker import MarkerTimeout, launch, marker_path, status, wait


def test_a_finished_command_writes_its_exit_code(tmp_path: Path) -> None:
    launch("ok", ["true"], tmp_path)
    assert wait("ok", tmp_path, timeout=30, poll=0.1) == 0


def test_a_failing_command_reports_its_code_rather_than_its_absence(tmp_path: Path) -> None:
    """A crash is a result, not a silence — the code survives to the waiter."""
    launch("bad", ["bash", "-c", "exit 7"], tmp_path)
    assert wait("bad", tmp_path, timeout=30, poll=0.1) == 7


def test_absence_is_unknown_and_never_done(tmp_path: Path) -> None:
    """The asymmetry, asserted directly. This is the defect being cured.

    A process check reports "not running" for success, crash and never-started
    alike. `status` refuses to collapse them: no marker means no answer.
    """
    assert status("never_launched", tmp_path) is None


def test_wait_raises_on_timeout_rather_than_returning(tmp_path: Path) -> None:
    """A waiter that returns a value on "I never found out" gets believed.

    Returning any int here would let a caller treat not-knowing as an exit code,
    which is the process-check failure wearing a different hat.
    """
    with pytest.raises(MarkerTimeout) as excinfo:
        wait("stalled", tmp_path, timeout=0.3, poll=0.05)
    assert "NOT 'done'" in str(excinfo.value)


def test_a_killed_command_reports_its_signal_rather_than_going_silent(tmp_path: Path) -> None:
    """The case a process check gets exactly backwards, and better than expected.

    The work is SIGKILLed. A grep for the process finds nothing and concludes
    "done" — success and violent death are the same observation to it. The
    wrapper outlives the work, so the marker records **137**: died on signal 9.

    That is a *result*, not a silence, and it is the stronger half of the
    asymmetry. Absence remains reserved for the case where the wrapper itself
    never got to answer — machine death, or the whole process group going down —
    which is the genuinely unknowable one.
    """
    launch("killed", ["bash", "-c", "kill -9 $$"], tmp_path)
    code = wait("killed", tmp_path, timeout=30, poll=0.1)
    assert code == 137, "a SIGKILL should be reported as 128+9, not as silence"
    assert code != 0, "and must never be mistaken for success"


def test_launching_clears_a_stale_marker(tmp_path: Path) -> None:
    """Yesterday's marker answering today's question is the other way to lie."""
    marker_path("reused", tmp_path).parent.mkdir(parents=True, exist_ok=True)
    marker_path("reused", tmp_path).write_text("0\n")
    launch("reused", ["bash", "-c", "sleep 5"], tmp_path)
    assert status("reused", tmp_path) is None


def test_the_log_is_captured_beside_the_marker(tmp_path: Path) -> None:
    """Diagnosis needs the output, and 'unknown' is when you need it most."""
    launch("loud", ["bash", "-c", "echo hello; exit 3"], tmp_path)
    assert wait("loud", tmp_path, timeout=30, poll=0.1) == 3
    assert "hello" in (tmp_path / "loud.log").read_text()


def test_the_ladder_uses_the_same_primitive() -> None:
    """One marker law, one implementation — `is_complete` is a marker test.

    Stated as an assertion so the ladder and this helper cannot drift into two
    notions of "done".
    """
    import inspect

    from reckoner.ladderpass import is_complete

    assert "marker" in inspect.getsource(is_complete)
