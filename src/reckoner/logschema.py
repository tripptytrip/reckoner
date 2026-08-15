"""The single schema definition for run logs. Inherited law, and it comes first.

`logschema.py as single schema definition with role fields and
absence-carries-a-reason notes` is in the chunk-0 inherited block. It leads the
chunk-9 build because **"from field one" is only achievable if the schema exists
before any writer does** — a column added after the first row is a column with a
hole in it, and the hole is exactly where the interesting iteration was.

Three properties, each of which has a failure this project has already met:

**1. One definition, imported by writers and readers alike.** A dashboard that
knows the column names independently is a second schema that can disagree with
the first. Analysis code imports `ITERATION_FIELDS` and nothing else.

**2. Every field carries a role.** A number's role is what it is *for*, and it is
not inferable from its name: `solve_rate_by_depth` is an outcome the campaign is
judged on, `entropy_prior_step1_start` is a diagnostic that explains an outcome,
and `nan_skips` is health. A reader six weeks out needs to know which numbers can
move the verdict and which only explain it — and mixing them is how a diagnostic
becomes a target.

**3. Absence carries a reason.** A missing field is never silently null. Either
it is required and its absence is an error, or it is optional and the row must
say *why* it is absent, in `absent: {field: reason}`. F-02 is the ancestor here:
a value that says nothing about its own provenance gets read as though it said
the strongest thing. A `null` in a JSONL column reads as zero to the next
histogram that touches it.

The schema is data, not code that runs at write time — `validate_row` is called
by the writer, and by tests, and by anything that reads a log it did not write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Closed vocabulary. A role outside this set is a schema bug, not a new kind of
#: number — the point of the list is that adding a role is a decision.
ROLES = frozenset(
    {
        "identity",  # which run, which iteration — how a row is addressed
        "provenance",  # what produced it: shas, fingerprints, versions, dtype
        "outcome",  # what the campaign is judged on
        "diagnostic",  # instrumentation that explains an outcome
        "counter",  # event tallies
        "timing",  # wall clock
        "health",  # skips, refusals, aborts — the run's own vital signs
    }
)


class SchemaError(ValueError):
    """A row that does not match the schema. Raised, never warned."""


@dataclass(frozen=True, slots=True)
class Field:
    """One column. ``absence`` is required exactly when the field is optional."""

    name: str
    kind: type | tuple[type, ...]
    role: str
    doc: str
    required: bool = True
    absence: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise SchemaError(f"{self.name}: role {self.role!r} is not in {sorted(ROLES)}")
        if self.required and self.absence is not None:
            raise SchemaError(f"{self.name}: a required field cannot declare an absence reason")
        if not self.required and not self.absence:
            raise SchemaError(
                f"{self.name}: an optional field MUST declare why it may be absent. "
                "Absence carries a reason — a column that is sometimes missing and "
                "never says why is a column that reads as zero."
            )


#: The Phase-2 loop's per-iteration row: ``runs/<name>/iterations.jsonl``.
ITERATION_FIELDS: tuple[Field, ...] = (
    # --- identity ---------------------------------------------------------
    Field("iteration", int, "identity", "0-based loop index."),
    Field("run_name", str, "identity", "Run directory name; rows are portable without it."),
    # --- provenance -------------------------------------------------------
    Field("git_sha", str, "provenance", "Repo SHA at the moment the row was written."),
    Field("config_fingerprint", str, "provenance", "sha256 of the resolved config."),
    Field("ruleset_version", int, "provenance", "Par and actions are denominated in this."),
    Field("vocab_version", int, "provenance", "States are denominated in this."),
    Field(
        "measure_dtype",
        str,
        "provenance",
        "Regime that produced every measured number in this row. fp32 is the "
        "regime the chunk-9 equivalence gate licensed (F-12).",
    ),
    Field("train_dtype", str, "provenance", "Regime the optimiser step ran in."),
    # --- outcome ----------------------------------------------------------
    Field(
        "solve_rate_by_depth",
        dict,
        "outcome",
        "depth -> fraction solved. Stratified because an aggregate hides the "
        "deep tail, and the deep tail is the experiment.",
    ),
    Field(
        "z_by_par_source",
        dict,
        "outcome",
        "par_source -> {'+1': n, '0': n, '-1': n}. **The draw-inflation watch.** "
        "Splitting by par_source is what makes a rising draw rate readable: "
        "draws against exact BFS par mean the policy reached par, draws against "
        "pool par mean the pool stopped improving, and one aggregate number "
        "cannot tell those apart.",
    ),
    Field(
        "steps_minus_par_histogram",
        dict,
        "outcome",
        "(steps - par) -> count, over solved episodes. Registered at chunk-8 "
        "close: the median alone says at least half drew; this is the loss tail.",
    ),
    # --- diagnostic -------------------------------------------------------
    Field(
        "entropy_prior_step1_start",
        float,
        "diagnostic",
        "H of the network prior at step 1, on **problem start states**.",
    ),
    Field(
        "entropy_prior_step1_reached",
        float,
        "diagnostic",
        "H of the network prior at step 1, on states reached after >= 1 rewrite. "
        "The chess split was start-position vs book-position; this is the "
        "faithful analog here, and the mapping is an interpretation rather than "
        "a transplant — stated so a reader does not assume an openings book.",
    ),
    Field(
        "entropy_target_step1_start",
        float,
        "diagnostic",
        "H of the **search-improved** target at step 1 on start states. Prior "
        "and target must both be logged: the chess lesson is that prior entropy "
        "alone cannot distinguish a confident policy from a collapsed one.",
    ),
    Field(
        "entropy_target_step1_reached",
        float,
        "diagnostic",
        "H of the search-improved target at step 1 on reached states.",
    ),
    # --- counters ---------------------------------------------------------
    Field("episodes", int, "counter", "Episodes completed this iteration."),
    Field(
        "search_nodes_total",
        int,
        "counter",
        "Summed SearchStats.nodes — the descent gate's number, carried per iteration.",
    ),
    Field(
        "search_evaluations_total",
        int,
        "counter",
        "Summed SearchStats.evaluations; equals nodes when every node is evaluated once.",
    ),
    Field("terminal_no_actions", int, "counter", "States with no legal action."),
    # --- health -----------------------------------------------------------
    Field(
        "state_too_large",
        int,
        "health",
        "Counted terminal losses from encode overflow. Never cropped (chunk 6).",
    ),
    Field("nan_skips", int, "health", "Optimiser steps skipped on non-finite gradients."),
    Field(
        "pool_refusals",
        int,
        "health",
        "Snapshots the CheckpointPool refused on a version mismatch. A counted "
        "event, never a silently smaller pool (BRIEF-chunk9 registration).",
    ),
    # --- timing -----------------------------------------------------------
    Field("seconds_self_play", float, "timing", "Wall clock generating episodes."),
    Field("seconds_train", float, "timing", "Wall clock in the optimiser."),
    Field("seconds_total", float, "timing", "Wall clock for the whole iteration."),
    # --- optional, each with its reason -----------------------------------
    Field(
        "pool_par_fraction",
        float,
        "diagnostic",
        "Fraction of episodes whose par came from a pool snapshot.",
        required=False,
        absence="league.par_from_pool_frac is 0, or no snapshot has been taken yet",
    ),
    Field(
        "ladder_pass",
        int,
        "outcome",
        "Ladder pass index, when this iteration ran one.",
        required=False,
        absence="not a ladder iteration (ladder runs on ladder.ladder_every cadence)",
    ),
)


def field_map(fields: tuple[Field, ...] = ITERATION_FIELDS) -> dict[str, Field]:
    return {f.name: f for f in fields}


def validate_row(row: dict[str, Any], fields: tuple[Field, ...] = ITERATION_FIELDS) -> None:
    """Raise :class:`SchemaError` unless ``row`` matches the schema exactly.

    Four checks, and the third is the one that earns this module's place:

    1. no unknown keys — a mistyped column is a column nobody reads
    2. every required field present, with the declared type
    3. **every absent optional field named in ``absent`` with a reason** — and
       conversely, nothing named in ``absent`` that is actually present
    4. no ``None`` values: absence is expressed in ``absent``, never as null,
       because a null reads as zero to the next thing that aggregates it
    """
    known = field_map(fields)
    absent = row.get("absent", {})
    if not isinstance(absent, dict):
        raise SchemaError("'absent' must be a mapping of field -> reason")

    unknown = set(row) - set(known) - {"absent"}
    if unknown:
        raise SchemaError(f"unknown columns: {sorted(unknown)}")

    for name, spec in known.items():
        present = name in row
        if present and row[name] is None:
            raise SchemaError(
                f"{name}: null is not how absence is expressed — put it in "
                f"'absent' with a reason. A null reads as zero downstream."
            )
        if present and not isinstance(row[name], spec.kind):
            raise SchemaError(f"{name}: expected {spec.kind}, got {type(row[name])}")
        if not present:
            if spec.required:
                raise SchemaError(f"{name}: required field missing")
            if name not in absent or not str(absent[name]).strip():
                raise SchemaError(
                    f"{name}: optional field is absent without a reason. "
                    f"Declared reason for absence is: {spec.absence!r}"
                )

    for name in absent:
        if name not in known:
            raise SchemaError(f"'absent' names unknown column {name!r}")
        if name in row:
            raise SchemaError(f"{name}: named in 'absent' but present in the row")


def append_row(
    path: Path, row: dict[str, Any], fields: tuple[Field, ...] = ITERATION_FIELDS
) -> None:
    """Validate, then append one JSONL row. Validation is not optional.

    Writing an invalid row and validating later is how a run produces 50
    iterations of unreadable log and discovers it at analysis time.
    """
    validate_row(row, fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_rows(path: Path, fields: tuple[Field, ...] = ITERATION_FIELDS) -> list[dict[str, Any]]:
    """Read and validate. A reader that does not validate is trusting a writer
    it did not run — including an older era of the same writer."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for i, row in enumerate(rows):
        try:
            validate_row(row, fields)
        except SchemaError as exc:
            raise SchemaError(f"{path}:{i + 1}: {exc}") from exc
    return rows


def describe(fields: tuple[Field, ...] = ITERATION_FIELDS) -> str:
    """The schema as a table — for the run directory, so a log is readable
    without this package."""
    lines = ["| column | role | required | meaning / absence |", "|---|---|---|---|"]
    for f in fields:
        req = "yes" if f.required else f"no — {f.absence}"
        lines.append(f"| `{f.name}` | {f.role} | {req} | {f.doc} |")
    return "\n".join(lines)
