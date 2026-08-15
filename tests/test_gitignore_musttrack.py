"""The .gitignore must never swallow a record the project depends on.

This test is inherited, not invented. Four separate incidents across two
sessions of the chess campaign, each caught by hand after the fact:

  BLOCKED.md               ignored outright, contradicting the project's own rule
  bare `data/`             ignored runs/data/ as a *directory*, so git could not
                           descend to reach the meta.json negations
  benchmarks/results.jsonl ignored, making a dated SHA-stamped record unversioned
  runs/**                  swallowed runs/ANCHORS.sha256 on the commit that added it

Whack-a-mole was losing, so this closes the class. Three lists, three questions:

  MUST_TRACK   a record that exists now and must not be ignored
  MUST_REACH   a record that does NOT exist yet and must be reachable when it
               does — `git check-ignore` answers for hypothetical paths, so the
               rule is testable the day it is written, which is the only way to
               beat "we'll remember to add the negation later" (three of the
               four incidents above)
  MUST_IGNORE  the other polarity — a well-meant negation that starts tracking
               75 MB checkpoints is the same defect facing the other way

Add to the lists whenever a new kind of record starts mattering.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Records that exist in the working tree today. A pattern matching nothing is
# itself a failure: a record that has vanished is as bad as one that is ignored.
MUST_TRACK: list[str] = [
    "experiment2_math_base625_spec.md",
    "experiment2_agent_plan.md",
    "AGENTS.md",
    "REGISTERED-ROUNDS.md",  # rounds specified before they are run
    "FINDINGS.md",  # measured facts later chunks must not rediscover
    # BRIEF-*.md as a glob, not one line per chunk: BRIEF-chunk8.md existed only
    # in a conversation until a crash took it, and a per-file list is a list
    # somebody has to remember to extend. The law is "committed on receipt"
    # (AGENTS.md §7); the pattern is what makes forgetting fail the build.
    "BRIEF-*.md",
    "GATE-*.md",  # manual-gate verdicts — the record a chunk closed on
    "ERRATA-*.md",  # corrections to a shipped chunk report, nothing fixed silently
    "Makefile",
    "pyproject.toml",
    "uv.lock",  # without it, "green in a clean clone" tests resolution luck
    "configs/*.yaml",
    "docs/*.md",  # generated reference (docs/vocab.md) — regenerable, but cited
    "src/reckoner/*.py",
    "tests/*.py",
    "scripts/*.py",
    "runs/state_extent.json",  # decision-bearing: it sized seq_len and max_sites
]

# The chunk 0 list, verbatim from the plan, expanded to concrete example paths
# because `git check-ignore` answers about pathnames, not patterns. The pattern
# each one stands for is in the trailing comment.
MUST_REACH: list[str] = [
    "BLOCKED-2026-08-14-example.md",  # BLOCKED*.md — a halt is information
    "RUNLOG-m1.md",  # RUNLOG*.md — the campaign ledger
    "PREREG-m1.md",  # PREREG*.md — pre-registration, with its amendment header
    "runs/m1/iterations.jsonl",  # runs/*/iterations.jsonl — the loop's narrative
    "runs/m1/ladder.jsonl",  # runs/*/ladder.jsonl — the verdict
    "benchmarks/results.jsonl",  # dated, SHA-stamped bench record
    "runs/ANCHORS.sha256",  # dataset + suite digests
    # Beyond the plan's initial list, written now for the same reason: these are
    # named as deliverables by later chunks, and the negation is free today.
    "runs/m1/ladder_pairscores.jsonl",  # pair_scores, persisted from row one
    "runs/m1/config.yaml",  # the RESOLVED config a run's rows are read against
    "runs/m1/provenance.json",  # git_sha + config fingerprint
    "runs/data/train_100k/meta.json",  # dataset provenance sidecar
    "runs/m1/annotations.jsonl",  # why a timing row is anomalous
    # Chunk 5: the frozen instruments and the dataset provenance sidecars. The
    # suites ARE the measuring stick — a suite git silently dropped would make
    # every number measured against it unreproducible.
    "runs/suites/solve_in_3.jsonl",
    "runs/data/train_100k/meta.json",
    "runs/data/eval_held_out/meta.json",
    "runs/rule_participation.json",
    "runs/par_delta.json",
    # Chunks 7-8: gate records. These three EXISTED and were silently ignored —
    # the guard missed them because nobody wrote the rule (ERRATA-chunk7.md §5).
    # Listed as MUST_REACH rather than MUST_TRACK so the glob is tested for
    # names that do not exist yet, which is the point of this list.
    "runs/gate_arithmetic.json",
    "runs/gate_arithmetic_d2.json",
    "runs/chunk7_gate_table.json",
    "runs/gate_a_future_instrument.json",
    # Chunk 8: pre-flight projections, kept so they can be scored against the run
    # they projected. F-03 exists because one was not.
    "runs/pilot_phase1_timing.json",
    "runs/pilot_a_future_preflight.json",
    # A1: contamination censuses are standing evidence, not one-offs.
    "runs/supervision_contamination.json",
    "runs/eval_suite_contamination.json",
    "runs/eval_independence.json",
    "runs/a_future_set_contamination.json",
    # Chunk 8: what a run reported, and the environment it reported it from.
    "runs/phase1/phase1_result.json",
    "runs/m1/check_env.txt",
]

# The counterpart polarity: things that must STAY ignored.
MUST_IGNORE: list[str] = [
    "runs/m1/latest.pt",
    "runs/m1/snapshots/iter0005.pt",
    "runs/data/train_100k/states.npy",
    "runs/data/train_100k/tokens.i32",  # 100K x max_len int32 — build output, not a record
    ".venv/bin/python",
    "src/reckoner/__pycache__/config.cpython-312.pyc",
]


def _is_git_repo() -> bool:
    return (REPO / ".git").exists()


def _check_ignored(path: str) -> bool:
    """True if git ignores ``path``. Uses check-ignore's exit status (0 = ignored).

    ``git -C`` addresses the repo per-command: a ``cd`` persists for a whole
    shell invocation and has already mis-pointed three merge commits on this box.
    """
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", path],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _is_git_repo(), reason="not a git repository — nothing to check"
)


@pytest.mark.parametrize("pattern", MUST_TRACK)
def test_must_track_paths_are_not_ignored(pattern: str) -> None:
    matches = sorted(REPO.glob(pattern))
    assert matches, f"{pattern!r} matched no file — the record it names is missing"

    ignored = [
        str(p.relative_to(REPO)) for p in matches if _check_ignored(str(p.relative_to(REPO)))
    ]
    assert not ignored, (
        f"{pattern!r} is gitignored: {ignored}\n"
        "These are project records, not build output. Add a negation to .gitignore."
    )


@pytest.mark.parametrize("path", MUST_REACH)
def test_future_records_are_already_reachable(path: str) -> None:
    """The negation must be in place before the run that produces the record."""
    assert not _check_ignored(path), (
        f"{path!r} would be gitignored. A run that writes it would produce a record "
        "git silently drops — add a negation to .gitignore."
    )


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_must_ignore_paths_stay_ignored(path: str) -> None:
    """Guard the other direction: negations must not start tracking large blobs."""
    assert _check_ignored(path), (
        f"{path!r} is NOT ignored — a .gitignore negation has gone too wide. "
        "Checkpoints, datasets and the venv must never enter the repository."
    )


def test_the_guard_itself_detects_an_ignored_record() -> None:
    """Validate the detector on both polarities before trusting it.

    Every other test here asserts `_check_ignored` returns the answer we want.
    None of them would fail if `_check_ignored` were stuck returning False —
    which is exactly the shape of a guard that reports green forever. This pins
    both answers against paths whose status is not in question.
    """
    assert _check_ignored(".venv/lib/python3.12/site-packages/torch/__init__.py")
    assert not _check_ignored("pyproject.toml")
