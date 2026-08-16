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
import random
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


class InstrumentAsTrainingSource(ValueError):
    """A frozen measuring stick was about to be used as a training source."""


#: Roles for artifacts that predate the role tag. **Frozen instruments cannot
#: carry it natively**: `runs/suites/*.jsonl` and the dataset `meta.json` files
#: are digested in `runs/ANCHORS.sha256`, so writing a field into them would
#: change the digest and un-freeze the instrument to add a label saying it is
#: frozen. So the role is declared here for what already exists, and stamped in
#: meta from birth for what comes next.
SOURCE_ROLES: dict[str, str] = {
    "runs/suites": "instrument",  # the measuring stick — never a training source
    "runs/paired": "instrument",  # the ladder's paired sets — same species, same rule
    "runs/data/train_100k": "training",
    "runs/data/phase1_train": "training",
    "runs/data/eval_held_out": "instrument",  # held-out: measures, never trains
    "runs/data/phase1_eval": "instrument",
}


def source_role(path: Path, repo: Path | None = None) -> str:
    """Instrument or training source. **An unknown role refuses; it does not default.**

    Defaulting to "training" would be F-02's shape at the runtime boundary: the
    permissive answer given for free to an artifact nobody classified. The
    trusted value is declared or the call fails.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    meta_path = Path(path) / "meta.json"
    if meta_path.exists():
        declared = json.loads(meta_path.read_text()).get("role")
        if declared:
            return str(declared)
    try:
        key = str(Path(path).resolve().relative_to(repo.resolve()))
    except ValueError:
        key = str(path)
    for prefix, role in SOURCE_ROLES.items():
        if key == prefix or key.startswith(prefix + "/"):
            return role
    raise InstrumentAsTrainingSource(
        f"{path} declares no role and matches no entry in SOURCE_ROLES. An "
        "unclassified artifact is not assumed to be a training source — add it to "
        "the registry, or stamp `role` in its meta.json if it is new."
    )


def assert_training_source(path: Path, repo: Path | None = None) -> None:
    """Refuse a frozen instrument as an episode source.

    The contamination censuses guard datasets **at birth**; nothing guarded the
    **runtime** source, and the chunk-9 shakedown demonstrated the hazard by
    happily consuming `solve_in_2` as a problem source. No harm there — nothing
    trained — but a config slip in a real run would train on the suites and
    poison every measurement downstream, including the ones used to detect it.

    Every consumer re-opens the question its producers closed.
    """
    role = source_role(path, repo)
    if role != "training":
        raise InstrumentAsTrainingSource(
            f"{path} has role {role!r} and must not be an episode source. Frozen "
            "instruments measure; training sources train. A run that trains on its "
            "own measuring stick reports improvement it did not make."
        )


def training_problems(path: Path, k: int, seed: int, repo: Path | None = None) -> list[Problem]:
    """Episode sources for the loop. **Refuses a frozen instrument, first thing.**

    This is the runtime boundary the contamination censuses could not reach: they
    prove a dataset is clean at birth, and say nothing about what a running loop
    points at.
    """
    assert_training_source(path, repo)
    dataset = read_dataset(path)
    out: list[Problem] = []
    for i in dataset.sample(k, seed):
        goal, target, expr = decode_state(dataset.state(i))
        out.append(
            Problem(
                goal=goal,
                expr=expr,
                par=int(dataset.par[i]),
                target=target,
                par_source="bfs",
            )
        )
    return out


def sample_indices(total: int, k: int, seed: int) -> list[int]:
    """**The** subsampler for any stratified artifact in this project.

    Every dataset here is laid out **stratum by stratum**, so ``range(k)`` is not
    a small sample of the set — it is the whole of the shallowest stratum and
    none of the rest. That has now caused three separate defects: F-03's pilot
    measured a distribution the real run would not see; ``build_phase1_data.py``
    shipped a ``--limit`` that took a prefix; and F-10's tie-break diagnostic
    sampled ``range(256)`` and got 256 depth-1 states with one legal action each,
    making every statistic degenerate.

    Documentation warns; helpers prevent. The first two were fixed by writing a
    warning, and the third happened anyway, to someone who had read it. So this
    exists and **raw prefix-slicing of a dataset is a review flag** — if a caller
    wants "some rows", it goes through here.

    Returns sorted indices so memmap reads stay sequential. ``k >= total``
    returns everything, which is the honest answer to "sample more than exists"
    rather than an error nobody wants at a call site.
    """
    if k >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), k))


class RecordWouldBeUntracked(RuntimeError):
    """A record was about to be written to a path git ignores."""


def assert_tracked(path: Path, repo: Path | None = None) -> None:
    """Refuse to write something git will not keep. **The mechanism, extracted.**

    It lived inside :func:`write_record`, which meant it protected JSON records
    and nothing else. A frozen paired set is written by :func:`write_suite` and is
    a strictly worse thing to lose than a record: every number the ladder reports
    is measured against it. So the check is a function, and each writer that
    produces evidence calls it.

    Outside a git repository the check is skipped rather than failed: a clean
    checkout used as a library is not the situation this guards.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    if not (repo / ".git").exists():
        return
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


# ---------------------------------------------------------------------------
# ANCHORS: the integrity registry
#
# SCOPE WIDENED 2026-08-16, and the justification is re-derived rather than the
# wording re-applied.
#
# The registry was scoped to *instruments* — frozen suites, paired sets, the
# anchor — because its original job was freeze verification: proving an
# instrument had not moved under the numbers measured on it. That scope has a
# blind spot shaped exactly like itself. During the campaign-host migration the
# transfer gate reported "26/26, 0 mismatches" while 85 MB of supervision data
# had not crossed at all, because *the gate cannot fail on data it does not know
# about*. The suite caught it one step later, by the luck of touching those
# files.
#
# So the scope stops being a category judgement ("is this an instrument?") and
# becomes mechanical: **ANCHORS is the integrity registry of every load-bearing
# artifact.** Anything a test or gate reads from ``runs/data`` carries an entry.
#
# The instrument-versus-training distinction survives untouched where it already
# lives — :data:`SOURCE_ROLES` — because that distinction governs what may train
# on what, which is a different question from what deserves integrity. And
# regenerable-in-principle does not exempt: a silently truncated regenerable file
# poisons a run exactly as thoroughly as a corrupted frozen one. Regenerability
# changes the recovery cost, not the detection need.
# ---------------------------------------------------------------------------


class UnanchoredPath(RuntimeError):
    """A load-bearing path was used without an integrity registry entry."""


#: The one place the string ``runs/data`` appears in this repository.
#:
#: Not stylistic. A path nobody else can name is a path nobody else can read
#: ungoverned: a third loader cannot come into existence bypassing the registry,
#: because it cannot construct an argument. The class closes, not just the
#: instances — the one-formatter precedent, applied to paths.
DATA_ROOT = Path("runs") / "data"


def data_path(name: str, repo: Path | None = None) -> Path:
    """A dataset path by name, **ungated** — for writers creating new artifacts.

    A dataset being written does not yet have a registry entry; it acquires one
    at birth, in :func:`write_dataset`. Readers use :func:`anchored_data`.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    return repo / DATA_ROOT / name


def anchored_data(name: str, repo: Path | None = None, *, verify: bool = False) -> Path:
    """A dataset path by name, **gated** — the reader's door."""
    return anchored_path(DATA_ROOT / name, repo, verify=verify)


def anchors_path(repo: Path | None = None) -> Path:
    repo = repo or Path(__file__).resolve().parents[2]
    return repo / "runs" / "ANCHORS.sha256"


def read_anchors(repo: Path | None = None) -> dict[str, str]:
    """The registry as ``{repo-relative path: sha256}``."""
    path = anchors_path(repo)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            out[rel.strip()] = digest
    return out


def write_anchors(entries: dict[str, str], repo: Path | None = None) -> None:
    """**The** registry writer, with one sort order.

    There were two: ``anchor_phase1`` sorted whole lines (hence by digest) and
    ``generate`` sorted by path, so the file's order depended on which script
    touched it last. Two writers of one registry is the same defect class the
    registry exists to catch, one level up. Line order is not load-bearing, so
    the existing on-disk order (by digest) is kept and the choice is made once.
    """
    path = anchors_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sorted(f"{d}  {rel}\n" for rel, d in entries.items())))


def register_anchor(path: Path, repo: Path | None = None, *, digest: str | None = None) -> str:
    """Enter a file in the registry, or refresh its digest. Returns the digest.

    Called by the writers at the moment of writing, so registration is a property
    of the artifact existing rather than a chore of remembering — the way
    :data:`SOURCE_ROLES` is stamped at birth.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    rel = str(path.resolve().relative_to(repo.resolve()))
    digest = digest or sha256_file(path)
    entries = read_anchors(repo)
    entries[rel] = digest
    write_anchors(entries, repo)
    return digest


def anchored_path(target: Path | str, repo: Path | None = None, *, verify: bool = False) -> Path:
    """Resolve a load-bearing path, refusing one the registry does not know.

    **The mono-instance gate.** Every test and gate that reads from ``runs/data``
    resolves through here, so used-but-unregistered fails at use, immediately, on
    any host — instead of surviving until something happens to notice.

    A directory is anchored only when *every file in it* is registered: a dataset
    whose ``tokens.i32`` is covered and whose ``meta.json`` is not is exactly the
    partial coverage this exists to refuse.

    ``verify=True`` additionally re-digests. Off by default because the entry's
    existence is the gate — verification is the transfer-time check, and paying
    20 MB of hashing on every open would buy a guarantee the caller did not ask
    for.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    path = (repo / target) if not Path(target).is_absolute() else Path(target)
    rel_root = str(path.resolve().relative_to(repo.resolve()))
    entries = read_anchors(repo)

    wanted = (
        sorted(str(p.resolve().relative_to(repo.resolve())) for p in path.rglob("*") if p.is_file())
        if path.is_dir()
        else [rel_root]
    )
    if not wanted:
        raise UnanchoredPath(f"{rel_root} contains no files to anchor")

    missing = [rel for rel in wanted if rel not in entries]
    if missing:
        raise UnanchoredPath(
            f"{rel_root} is load-bearing but {len(missing)} of {len(wanted)} of its "
            f"files carry no ANCHORS entry: {missing[:4]}. A gate cannot fail on "
            "data it does not know about — register it (write_dataset does this at "
            "birth) or stop reading it from a gate."
        )
    if verify:
        wrong = [rel for rel in wanted if sha256_file(repo / rel) != entries[rel]]
        if wrong:
            raise UnanchoredPath(
                f"{rel_root}: {len(wrong)} digests do not match ANCHORS: {wrong[:4]}"
            )
    return path


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
    path = Path(path)
    assert_tracked(path, repo)
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

    def sample(self, k: int, seed: int) -> list[int]:
        """Indices sampled across the whole set. Never a prefix — see
        :func:`sample_indices`."""
        return sample_indices(len(self), k, seed)

    def __len__(self) -> int:
        return int(self.meta["n"])

    def state(self, index: int) -> tuple[int, ...]:
        return tuple(int(t) for t in self.tokens[index, : self.lengths[index]])


def write_dataset(
    path: Path,
    problems: list[Problem],
    cfg: Config,
    *,
    mode: str,
    seed: int,
    repo: Path,
    role: str = "training",
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
        # Stamped from birth for every artifact written after the role tag
        # existed. `source_role` falls back to SOURCE_ROLES only for the frozen
        # artifacts that predate it and cannot be rewritten.
        "role": role,
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

    # REGISTRATION AT BIRTH. A dataset that exists is a dataset the integrity
    # registry knows about — the same move as stamping its role, and for the same
    # reason: a property of existing rather than a chore of remembering. The
    # per-field digests are already computed above, so the field files cost
    # nothing; meta.json is digested here because it carries the role tag and the
    # digests themselves, which makes it load-bearing in its own right.
    # A dataset written outside the repo is a fixture, not a load-bearing
    # artifact, and has nothing to be load-bearing *to*. Skipped rather than
    # failed, on the same reasoning as assert_tracked outside a git checkout.
    try:
        rels = {
            **{
                str((path / f"{name}.i32").resolve().relative_to(repo.resolve())): meta["digests"][
                    name
                ]
                for name in FIELDS
            },
            str((path / "meta.json").resolve().relative_to(repo.resolve())): sha256_file(
                path / "meta.json"
            ),
        }
    except ValueError:
        return meta
    write_anchors({**read_anchors(repo), **rels}, repo)
    return meta


def read_dataset(path: Path, repo: Path | None = None) -> Dataset:
    """Open a training set. **Every read is gated by the integrity registry.**

    The gate is here rather than at the call sites because a rule enforced at
    call sites is a rule that holds until someone writes a new call site. A
    reader that constructs its own path still arrives at this function, so
    used-but-unregistered fails at use however the path was built.

    Datasets outside the repo are fixtures — skipped, per the precedent in
    :func:`assert_tracked`.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        Path(path).resolve().relative_to(repo.resolve())
    except ValueError:
        pass
    else:
        anchored_path(path, repo)
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
