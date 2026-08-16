"""The integrity registry's gate, both polarities.

Written because the scope widening is only as good as its enforcement. During
the campaign-host migration the transfer gate reported "26/26, 0 mismatches"
while 85 MB of supervision data had not crossed at all — the gate cannot fail on
data it does not know about, and the blind spot was shaped exactly like the
scope. The suite caught it one step later by the luck of touching those files,
and "by luck" is the part this file removes.

So: used-but-unregistered fails **at use**, on any host, immediately. A rule that
only holds while everyone remembers it is a rule with a scheduled failure.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from reckoner.dataset import (
    DATA_ROOT,
    UnanchoredPath,
    anchored_path,
    read_anchors,
    register_anchor,
    write_anchors,
)

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ the gate itself


def test_a_registered_file_resolves() -> None:
    """The positive polarity: the anchor itself, which ANCHORS has always held."""
    resolved = anchored_path("runs/phase1/phase1.pt")
    assert resolved.is_file()
    assert str(resolved.relative_to(REPO)) in read_anchors()


def test_an_unregistered_file_is_refused(tmp_path: Path) -> None:
    """The polarity that matters, and the one the migration needed."""
    stray = REPO / DATA_ROOT / "not_a_registered_dataset.i32"
    with pytest.raises(UnanchoredPath) as excinfo:
        anchored_path(stray)
    assert "no ANCHORS entry" in str(excinfo.value)


def test_the_supervision_data_is_now_covered() -> None:
    """The specific gap the widening closed, asserted rather than assumed.

    phase1_train and phase1_eval are read by the test suite and were carrying
    zero registry entries — 85 MB of load-bearing data the gate was blind to.
    """
    for name in ("phase1_train", "phase1_eval"):
        resolved = anchored_path(DATA_ROOT / name)
        assert resolved.is_dir()


def test_a_directory_with_one_unregistered_file_is_refused(tmp_path: Path) -> None:
    """Partial coverage is the failure mode, so partial coverage is refused.

    A dataset whose tokens.i32 is covered and whose meta.json is not looks
    anchored to any check that stops at the first entry it finds.
    """
    repo = tmp_path
    (repo / ".git").mkdir()
    data = repo / DATA_ROOT / "partial"
    data.mkdir(parents=True)
    (data / "covered.i32").write_bytes(b"covered")
    (data / "uncovered.i32").write_bytes(b"uncovered")
    write_anchors({}, repo)
    register_anchor(data / "covered.i32", repo)

    with pytest.raises(UnanchoredPath) as excinfo:
        anchored_path(data, repo)
    assert "uncovered.i32" in str(excinfo.value)


def test_verify_catches_a_digest_that_moved(tmp_path: Path) -> None:
    """`verify=True` is the transfer-time check; existence is the use-time gate."""
    repo = tmp_path
    (repo / ".git").mkdir()
    f = repo / DATA_ROOT / "d" / "x.i32"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"original")
    write_anchors({}, repo)
    register_anchor(f, repo)

    assert anchored_path(f, repo, verify=True) == f  # matches
    f.write_bytes(b"tampered")
    anchored_path(f, repo)  # existence gate still passes — by design
    with pytest.raises(UnanchoredPath) as excinfo:
        anchored_path(f, repo, verify=True)
    assert "do not match ANCHORS" in str(excinfo.value)


# -------------------------------------------------- registration at birth


def test_write_dataset_registers_every_file_it_writes(tmp_path: Path) -> None:
    """Registration is a property of existing, not a chore of remembering.

    The same move as stamping the role tag at birth, and for the same reason: a
    step someone has to remember is a step that eventually is not taken.
    """
    from reckoner.config import Config
    from reckoner.dataset import FIELDS, write_dataset
    from reckoner.episode import Problem
    from reckoner.expr import add, eq, mul, num, var
    from reckoner.vocab import GOAL_SOLVE, VAR_X

    repo = tmp_path
    (repo / ".git").mkdir()
    write_anchors({}, repo)
    x = var(VAR_X)
    problems = [
        Problem(
            goal=GOAL_SOLVE,
            target=VAR_X,
            par=3,
            par_source="bfs",
            expr=eq(add(mul(num(3), x), num(6)), num(21)),
        )
    ]
    out = repo / DATA_ROOT / "born_registered"
    write_dataset(out, problems, Config(), mode="test", seed=0, repo=repo, role="training")

    entries = read_anchors(repo)
    for name in FIELDS:
        assert f"{DATA_ROOT}/born_registered/{name}.i32" in entries
    assert f"{DATA_ROOT}/born_registered/meta.json" in entries
    # and the gate accepts what the writer registered, which is the closed loop
    assert anchored_path(out, repo).is_dir()


def test_the_registry_has_one_writer_and_one_sort_order() -> None:
    """Two writers of one registry is the defect the registry exists to catch.

    `anchor_phase1` sorted whole lines and `generate` sorted by path, so the
    file's order depended on which script last touched it.
    """
    from reckoner import pairedset

    assert pairedset.read_anchors is read_anchors
    lines = (REPO / "runs" / "ANCHORS.sha256").read_text().splitlines()
    assert lines == sorted(lines), "ANCHORS is not in its declared order"


def test_meta_json_is_covered_because_it_is_load_bearing() -> None:
    """meta.json carries the role tag and the digests — evidence, not packaging."""
    entries = read_anchors()
    for name in ("phase1_train", "train_100k"):
        rel = f"{DATA_ROOT}/{name}/meta.json"
        assert rel in entries, f"{rel} unregistered"
        meta = json.loads((REPO / rel).read_text())
        assert "digests" in meta


# ------------------------------------------------------- adoption: no bypass


def test_every_dataset_read_is_gated_at_the_choke_point() -> None:
    """`read_dataset` gates, so a reader cannot route around the registry.

    The loader refusing unregistered paths governs callers that use it. This
    governs the ones that do not: a reader constructing its own path still
    arrives at `read_dataset`, so the gate holds however the path was built.
    """
    import inspect

    from reckoner.dataset import read_dataset

    assert "anchored_path" in inspect.getsource(read_dataset)


def test_no_runs_data_literal_escapes_the_governed_modules() -> None:
    """The bypass dies by being findable — the source-scan pattern.

    The house has killed formatter-bypass and verify()-reimplementation this
    way: a rule that only holds while everyone remembers it is a rule with a
    scheduled failure, so constructing the path any other way fails a test by
    existing.

    The allowlist is justified per entry and is meant to shrink. Anything not
    listed reads `runs/data` through `anchored_path` / `read_dataset` /
    `training_problems`, all of which gate.
    """
    # EMPTY, AND MEANT TO STAY EMPTY. The way an allowlist does not grow is by
    # not existing: dataset.py is the only file in this repository containing the
    # string, so a third ungated loader cannot come into being — it could not
    # name a path to read. The class closes, not just the instances.
    allowed: set[str] = {"src/reckoner/dataset.py"}
    # The needle is composed from the governed constant, so this test contains
    # no literal either — and if DATA_ROOT ever moves, the scan follows it.
    needle = str(DATA_ROOT)
    split_form = '"{}" / "{}"'.format(*DATA_ROOT.parts)
    offenders: dict[str, list[int]] = {}
    for directory in ("src", "tests", "scripts"):
        for path in sorted((REPO / directory).rglob("*.py")):
            rel = str(path.relative_to(REPO))
            if rel in allowed:
                continue
            source = path.read_text()
            hits = []
            # A STRING THAT IS A PATH, not a sentence mentioning one. Prose
            # describing this rule is not a route around it, and a scan that
            # cannot tell them apart trains people to reword docstrings.
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    v = node.value
                    if v == needle or v.startswith(needle + "/"):
                        hits.append(node.lineno)
            # and the composed form, which prose never contains
            hits += [i for i, line in enumerate(source.splitlines(), 1) if split_form in line]
            if hits:
                offenders[rel] = sorted(set(hits))
    assert not offenders, (
        "these construct a runs/data path outside the governed modules, which is "
        f"a route around the integrity registry: {offenders}. Read through "
        "read_dataset()/training_problems(), or add a justified allowlist entry."
    )
