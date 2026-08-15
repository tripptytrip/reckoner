"""Paired problem sets: frozen at birth, digested into ANCHORS, censused twice.

A paired set is the ladder's instrument. Every arm sees exactly these problems,
so its properties are load-bearing in a way a training set's are not:

* **Frozen at birth.** :func:`freeze` writes the rows, digests the file into
  ``runs/ANCHORS.sha256``, and **refuses to overwrite an already-anchored set**.
  A measuring stick that can be rewritten is not frozen, it is merely old.
* **Verified at read.** :func:`load` re-digests and compares. An anchor nobody
  checks is a comment; the digest has to be *consulted* on the path that uses the
  file, not on a path a person remembers to run.
* **Censused at BOTH levels**, per L6 and the F-08/F-09 machinery:

  - **problem-level** — is a paired problem also a *problem* in a training set?
  - **state-level** — is a paired problem also an *intermediate state* of some
    training derivation? This is the level inheritance cannot see. Solving
    ``9x + (-28) = 44`` passes through ``9x = 72``; if a paired-set problem *is*
    ``9x = 72`` under the same goal, the model has trained on the instrument
    verbatim while every problem-level check stayed green.

**The decision rule, pre-stated:** a candidate colliding at *either* level is
**dropped before the freeze**. Contamination discovered *after* the freeze is a
**finding, not an adjustment** — editing a frozen instrument to make it clean
un-freezes it, and the resulting set is one whose digest no longer describes the
thing that was measured. The rule is written here so it cannot be chosen after
the numbers are seen.

Two keys, and the difference is the point
------------------------------------------
:func:`reckoner.dataset.problem_key` is the **strict** identity: the canonical
token sequence including the goal prefix and target. It is what :mod:`ladder`
pairs on, because pairing must never merge two distinct rows.

:func:`census_key` is ``(identity_key(expr), goal)`` — the **census** key, and it
is deliberately looser: it treats ``3x + 6 = 21`` and ``6 + 3x = 21`` as one
identity. Contamination should be reported under the looser key (a model that
saw one has effectively seen the other), and pairing must use the stricter one.
Using either key for both jobs is wrong in one direction or the other.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reckoner.dataset import (
    ShippingError,
    assert_tracked,
    check_shippable,
    problem_key,
    read_suite,
    sha256_file,
    suite_problem,
    write_suite,
)
from reckoner.episode import Problem, decode_state
from reckoner.expr import identity_key

#: Where paired sets live. Declared in ``dataset.SOURCE_ROLES`` as an instrument,
#: because — like ``runs/suites`` — the file is digested and so cannot carry a
#: role tag of its own without changing the digest that freezes it.
PAIRED_DIR = "runs/paired"

CensusKey = tuple[tuple[int, ...], int]


class PairedSetError(ValueError):
    """A paired set that cannot be trusted as an instrument."""


def census_key(problem: Problem) -> CensusKey:
    """``(identity_key(expr), goal)`` — the looser, contamination-facing key."""
    return (identity_key(problem.expr), problem.goal)


def census_key_of_tokens(tokens: tuple[int, ...]) -> CensusKey:
    goal, _target, expr = decode_state(tokens)
    return (identity_key(expr), goal)


def _memmap_states(path: Path) -> Iterator[tuple[int, ...]]:
    """Every state in a memmap set — problem sets and supervision sets alike.

    A problem set's rows are all start states; a supervision set's rows are the
    intermediate states of derivations. The iteration is identical, which is
    exactly why the *distinction* has to live in the caller's naming rather than
    in the reading: it is invisible in the bytes.
    """
    meta = json.loads((path / "meta.json").read_text())
    n, width = int(meta["n"]), int(meta["max_len"])
    tokens = np.memmap(path / "tokens.i32", dtype=np.int32, mode="r", shape=(n, width))
    lengths = np.memmap(path / "lengths.i32", dtype=np.int32, mode="r", shape=(n,))
    for i in range(n):
        yield tuple(int(t) for t in tokens[i, : int(lengths[i])])


def source_census_keys(path: Path) -> set[CensusKey]:
    """The census keys of every row in a memmap set. Exhaustive by design."""
    return {census_key_of_tokens(state) for state in _memmap_states(path)}


@dataclass
class PairedCensus:
    """Both levels, with the per-candidate verdicts that produced them."""

    candidates: int
    problem_level: dict[str, int]
    state_level: dict[str, int]
    #: Indices into ``candidates`` that collided, by level. Kept so the drop is
    #: reproducible from the record rather than only from re-running the census.
    problem_level_hits: list[int]
    state_level_hits: list[int]

    @property
    def clean_indices(self) -> list[int]:
        dirty = set(self.problem_level_hits) | set(self.state_level_hits)
        return [i for i in range(self.candidates) if i not in dirty]

    def as_dict(self) -> dict:
        return {
            "candidates": self.candidates,
            "problem_level": self.problem_level,
            "state_level": self.state_level,
            "problem_level_hits": len(self.problem_level_hits),
            "state_level_hits": len(self.state_level_hits),
            "state_level_beyond_problem_level": len(
                set(self.state_level_hits) - set(self.problem_level_hits)
            ),
            "clean": len(self.clean_indices),
            "rule": (
                "a candidate colliding at either level is dropped BEFORE the freeze; "
                "contamination found after the freeze is a finding, never an edit"
            ),
        }


def census(
    candidates: list[Problem],
    *,
    problem_sources: dict[str, set[CensusKey]],
    state_sources: dict[str, set[CensusKey]],
) -> PairedCensus:
    """Census candidates against problem sets and supervision sets separately.

    The two dicts are passed pre-computed rather than as paths because reading
    311k supervision states is a script's job, not a library call's — the
    exhaustive-in-script / bounded-in-test precedent. Both are required and may
    be empty **only** if the caller means empty: a census against nothing is
    reported as a census against nothing, never as a clean bill.
    """
    keys = [census_key(p) for p in candidates]
    problem_hits: set[int] = set()
    state_hits: set[int] = set()
    per_problem_source: dict[str, int] = {}
    per_state_source: dict[str, int] = {}

    for name, reference in problem_sources.items():
        hit = [i for i, k in enumerate(keys) if k in reference]
        per_problem_source[name] = len(hit)
        problem_hits |= set(hit)
    for name, reference in state_sources.items():
        hit = [i for i, k in enumerate(keys) if k in reference]
        per_state_source[name] = len(hit)
        state_hits |= set(hit)

    return PairedCensus(
        candidates=len(candidates),
        problem_level=per_problem_source,
        state_level=per_state_source,
        problem_level_hits=sorted(problem_hits),
        state_level_hits=sorted(state_hits),
    )


# ---------------------------------------------------------------------------
# Freezing, and reading back a thing that was frozen
# ---------------------------------------------------------------------------


def _anchors_path(repo: Path) -> Path:
    return repo / "runs" / "ANCHORS.sha256"


def read_anchors(repo: Path) -> dict[str, str]:
    path = _anchors_path(repo)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            out[rel.strip()] = digest
    return out


def freeze(path: Path, problems: list[Problem], *, repo: Path) -> str:
    """Write a paired set, digest it into ANCHORS, and refuse a second write.

    The refusal is the freeze. Without it "frozen at birth" is a description of
    intent, and the first time a script is re-run with a different seed the
    instrument changes underneath every number ever measured on it.
    """
    if not problems:
        raise PairedSetError("refusing to freeze an empty paired set")
    # A paired set git will not keep is worse than an untracked record: every
    # number the ladder reports is measured against it, and it cannot be
    # regenerated without changing the instrument.
    assert_tracked(path, repo)
    rel = str(path.resolve().relative_to(repo.resolve()))
    anchors = read_anchors(repo)
    if rel in anchors:
        raise PairedSetError(
            f"{rel} is already anchored at {anchors[rel][:12]}. A paired set is "
            "frozen at birth: re-writing it would change the instrument under every "
            "number already measured on it. Choose a new name, or delete the anchor "
            "deliberately and say so in the record."
        )
    for problem in problems:
        check_shippable(problem)
    seen: set[tuple[int, ...]] = set()
    for problem in problems:
        key = problem_key(problem)
        if key in seen:
            raise PairedSetError(
                "duplicate problem in a paired set: pairing would match a score "
                "against a partner chosen by write order"
            )
        seen.add(key)

    digest = write_suite(path, problems)
    lines = sorted(f"{d}  {p}\n" for p, d in {**anchors, rel: digest}.items())
    _anchors_path(repo).write_text("".join(lines))
    return digest


def load(path: Path, *, repo: Path) -> list[Problem]:
    """Read a paired set, **verifying its anchor**. A drifted digest raises."""
    anchors = read_anchors(repo)
    rel = str(path.resolve().relative_to(repo.resolve()))
    if rel not in anchors:
        raise PairedSetError(
            f"{rel} is not in runs/ANCHORS.sha256. An un-anchored paired set is a "
            "file, not an instrument — nothing says it is the same one that was "
            "measured against."
        )
    actual = sha256_file(path)
    if actual != anchors[rel]:
        raise PairedSetError(
            f"{rel} has drifted: anchored {anchors[rel][:12]}, on disk {actual[:12]}. "
            "Every number measured on this file is measured on an unknown instrument."
        )
    return [suite_problem(row) for row in read_suite(path)]


__all__ = [
    "PAIRED_DIR",
    "CensusKey",
    "PairedCensus",
    "PairedSetError",
    "ShippingError",
    "census",
    "census_key",
    "census_key_of_tokens",
    "freeze",
    "load",
    "read_anchors",
    "source_census_keys",
]
