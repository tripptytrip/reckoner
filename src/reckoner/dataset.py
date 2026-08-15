"""Dataset and suite writers, and the provenance every row carries.

Two formats, for two jobs:

* **Training sets** are ``numpy.memmap`` arrays with a ``meta.json`` sidecar —
  the chess convention. Host RAM is the scarce pool on this box (~30 GiB, half
  of it spoken for), so a training set is mapped, never loaded.
* **Suites** are JSONL, because a frozen instrument has to be readable by a
  human six weeks later without importing this package to see what is in it.

The shipping boundary
---------------------
``unverified`` may exist at construction; it may never *ship*. Every writer here
asserts, per row, that ``par is not None`` and ``par_source != "unverified"`` and
``par >= 1``. Those are three different facts:

* ``par is None`` — the problem was never labelled.
* ``par_source == "unverified"`` — a number exists with nothing vouching for it.
* ``par == 0`` — the problem is already in goal form: terminal at birth, nothing
  to learn from, and a free draw in any average.

None may substitute for another, and none may leave. The weakest honest state is
allowed to be; it is not allowed to ship.

Provenance
----------
Every ``meta.json`` carries ``ruleset_version`` and ``vocab_version`` beside the
git SHA and config fingerprint. Par is denominated in a rule system and states
are denominated in a vocabulary; a dataset without both is a pile of integers
whose meaning cannot be reconstructed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reckoner.config import Config, config_fingerprint
from reckoner.episode import Problem, decode_state, encode_state
from reckoner.expr import identity_key
from reckoner.rules import RULESET_VERSION
from reckoner.vocab import PAD, VOCAB_VERSION

#: Array files a training set is made of. Fixed names, because a reader six
#: weeks out should not have to guess.
FIELDS = ("tokens", "lengths", "goal", "target", "par", "depth")


class ShippingError(ValueError):
    """A row that must not leave the process it was built in."""


def check_shippable(problem: Problem) -> None:
    """The shipping boundary, applied per row by every writer here."""
    if problem.par is None:
        raise ShippingError("unlabelled problem: par is absent, and absence does not ship")
    if problem.par_source == "unverified":
        raise ShippingError(
            "par_source='unverified' does not ship: a number with nothing vouching "
            "for it is an unlabelable problem wearing a label field"
        )
    if problem.par < 1:
        raise ShippingError(
            f"par={problem.par} does not ship: a par-0 problem is already in goal "
            "form — terminal at birth, and a free draw in any average"
        )


def git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class RecordWouldBeUntracked(RuntimeError):
    """A record was about to be written to a path git ignores."""


def write_record(path: Path, payload: dict, *, repo: Path | None = None) -> Path:
    """Write a JSON record, refusing to write one git would ignore.

    Three separate times in chunk 8 a record shipped untracked — the gate
    arithmetic a gate was declared on, the pre-flight a projection came from, the
    result a run reported — each caught afterwards and each fixed by widening a
    `.gitignore` negation to a glob. The glob fixes the *class*; this fixes the
    *mechanism*. **The writer asserts its own trackedness before the bytes land**,
    so an unversioned record is impossible rather than noticed later.

    It is the same shape as the inherited "instrument the trigger" law: the check
    references the actual path being written, resolved against the repo, not a
    pattern that might match something else. Raising is deliberate — library code
    raises and only ``scripts/`` prints, and a record silently landing outside
    version control is exactly the failure that has to be loud.

    Outside a git repository the check is skipped rather than failed: a clean
    checkout used as a library is not the situation this guards.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    path = Path(path)
    if (repo / ".git").exists():
        ignored = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "-q", str(path)],
            capture_output=True,
            check=False,
        )
        if ignored.returncode == 0:
            raise RecordWouldBeUntracked(
                f"{path} is ignored by .gitignore — a record that git will not keep is "
                f"not a record. Add a negation (prefer a glob covering its kind) and a "
                f"MUST_REACH entry in tests/test_gitignore_musttrack.py, then rerun."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def problem_key(problem: Problem) -> tuple[int, ...]:
    """**The** dedup / contamination key for a problem.

    One shared identity normalizer (inherited law): the encoded state, which is
    the canonical token sequence *including the goal prefix*. The prefix matters
    — the same expression under SIMPLIFY and under EVALUATE is two problems, and
    a key that conflated them would report contamination that is not there and
    miss the kind that is.
    """
    return tuple(encode_state(problem.goal, problem.expr, problem.target))


# ---------------------------------------------------------------------------
# Training sets — memmap + meta.json
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dataset:
    """A mapped training set. Arrays are views, not copies."""

    path: Path
    meta: dict
    tokens: np.ndarray
    lengths: np.ndarray
    goal: np.ndarray
    target: np.ndarray
    par: np.ndarray
    depth: np.ndarray

    def __len__(self) -> int:
        return int(self.meta["n"])

    def state(self, index: int) -> tuple[int, ...]:
        return tuple(int(t) for t in self.tokens[index, : self.lengths[index]])


def write_dataset(
    path: Path, problems: list[Problem], cfg: Config, *, mode: str, seed: int, repo: Path
) -> dict:
    """Write a memmap training set with its provenance sidecar."""
    if not problems:
        raise ValueError("refusing to write an empty dataset")
    for problem in problems:
        check_shippable(problem)

    path.mkdir(parents=True, exist_ok=True)
    states = [problem_key(problem) for problem in problems]
    max_len = max(len(s) for s in states)
    n = len(problems)

    arrays = {
        "tokens": np.full((n, max_len), PAD, dtype=np.int32),
        "lengths": np.zeros(n, dtype=np.int32),
        "goal": np.zeros(n, dtype=np.int32),
        "target": np.full(n, -1, dtype=np.int32),
        "par": np.zeros(n, dtype=np.int32),
        "depth": np.zeros(n, dtype=np.int32),
    }
    for i, (problem, state) in enumerate(zip(problems, states, strict=True)):
        arrays["tokens"][i, : len(state)] = state
        arrays["lengths"][i] = len(state)
        arrays["goal"][i] = problem.goal
        arrays["target"][i] = -1 if problem.target is None else problem.target
        arrays["par"][i] = problem.par
        # Under BFS-exact labelling par and depth coincide — both are the minimum
        # step count. They are stored separately because they stop coinciding the
        # moment a scripted or pool par enters, and a column that silently
        # changes meaning is worse than a redundant one.
        arrays["depth"][i] = problem.par

    for name, array in arrays.items():
        array.tofile(path / f"{name}.i32")

    depths: dict[int, int] = {}
    goals: dict[int, int] = {}
    sources: dict[str, int] = {}
    for problem in problems:
        assert problem.par is not None
        depths[problem.par] = depths.get(problem.par, 0) + 1
        goals[problem.goal] = goals.get(problem.goal, 0) + 1
        sources[problem.par_source] = sources.get(problem.par_source, 0) + 1

    meta = {
        "mode": mode,
        "n": n,
        "max_len": max_len,
        "seed": seed,
        "ruleset_version": RULESET_VERSION,
        "vocab_version": VOCAB_VERSION,
        "git_sha": git_sha(repo),
        "config_fingerprint": config_fingerprint(cfg),
        "fields": {name: {"dtype": "int32", "shape": list(arrays[name].shape)} for name in FIELDS},
        "depth_histogram": dict(sorted(depths.items())),
        "goal_histogram": dict(sorted(goals.items())),
        "par_source_histogram": dict(sorted(sources.items())),
        "digests": {name: sha256_file(path / f"{name}.i32") for name in FIELDS},
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def read_dataset(path: Path) -> Dataset:
    meta = json.loads((path / "meta.json").read_text())
    n, max_len = meta["n"], meta["max_len"]
    load = lambda name, shape: np.memmap(  # noqa: E731
        path / f"{name}.i32", dtype=np.int32, mode="r", shape=shape
    )
    return Dataset(
        path=path,
        meta=meta,
        tokens=load("tokens", (n, max_len)),
        lengths=load("lengths", (n,)),
        goal=load("goal", (n,)),
        target=load("target", (n,)),
        par=load("par", (n,)),
        depth=load("depth", (n,)),
    )


# ---------------------------------------------------------------------------
# Suites — JSONL, frozen, human-readable
# ---------------------------------------------------------------------------


def suite_row(problem: Problem) -> dict:
    check_shippable(problem)
    return {
        "ruleset_version": RULESET_VERSION,
        "vocab_version": VOCAB_VERSION,
        "goal": problem.goal,
        "target": problem.target,
        "par": problem.par,
        "depth": problem.par,
        "par_source": problem.par_source,
        "tokens": list(problem_key(problem)),
    }


def write_suite(path: Path, problems: list[Problem]) -> str:
    """Write a frozen suite. Returns its sha256 for ``runs/ANCHORS.sha256``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for problem in problems:
            fh.write(json.dumps(suite_row(problem), sort_keys=True) + "\n")
    return sha256_file(path)


def read_suite(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def suite_problem(row: dict) -> Problem:
    """Rebuild a Problem from a suite row, re-deriving nothing it does not state."""
    goal, target, expr = decode_state(tuple(row["tokens"]))
    return Problem(
        goal=goal,
        expr=expr,
        par=row["par"],
        target=target,
        par_source=row["par_source"],
    )


def state_keys(rows: list[dict]) -> set[tuple[int, ...]]:
    """Contamination keys for a suite, via the one shared normalizer."""
    return {tuple(row["tokens"]) for row in rows}


def dataset_keys(dataset: Dataset) -> set[tuple[int, ...]]:
    return {dataset.state(i) for i in range(len(dataset))}


def expression_keys(rows: list[dict]) -> set[tuple[int, ...]]:
    """Keys ignoring the goal prefix — a *weaker*, deliberately stricter check.

    ``problem_key`` is the identity a problem has. This is the identity its
    *expression* has, and a training row sharing an expression with a suite row
    under a different goal is not contamination by the strict definition but is
    close enough that a frozen instrument should know about it.
    """
    return {identity_key(decode_state(tuple(row["tokens"]))[2]) for row in rows}
