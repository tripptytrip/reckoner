"""The Phase-2 replay ring: fixed capacity, `root_q` from field one, era-aware.

Memory arithmetic, shown before the capacity is declared
--------------------------------------------------------
Rider (c) applied to a resource: compute what the quantity can legitimately
reach *before* siting the alarm. Bytes per stored step, at the pinned shapes
(``seq_len 512``, ``max_sites 192``, ``gumbel_m 16``):

===================  ==========================================  =======
field                layout                                      bytes
===================  ==========================================  =======
``tokens``           512 × int16 (vocab 657 fits int16)             1024
``site_positions``   192 × int16                                     384
``length``           int16                                             2
``n_sites``          int16                                             2
``visit_actions``    16 × int32 — the top-m root actions               64
``visit_counts``     16 × int32                                       64
``root_q``           float32                                           4
``z``                int8                                              1
``par_source``       int8 (enum)                                       1
``par``              int16                                             2
``steps_remaining``  int16                                             2
``depth``            int8                                              1
``goal``             int8                                              1
``ring_format``      int16 — the format version this record was written under   2
``absent_mask``      uint32 — one bit per runtime-absent field           4
**per record**                                                    **1558**
===================  ==========================================  =======

At ``train.replay_capacity`` = 500,000 that is **0.72 GiB**, against ~15 GiB of
realistically free host RAM (AGENTS.md §3, and swap is already 3.4 GiB deep so an
OOM-adjacent allocation thrashes rather than degrades). The provisional capacity
clears with two orders of margin, so it is **decided at 500,000** rather than
re-guessed. 200,000 would be 0.29 GiB and 100,000 would be 0.14 GiB, recorded so a
later reduction is a choice with a number attached.

The visit vector is stored **sparsely** — the top-``gumbel_m`` root actions and
their counts — not as a dense 1,344-wide distribution. Dense would be 5,376 B per
record, 3.5× the whole record, for a vector that is zero everywhere except at most
``m`` entries. The search considers at most ``m`` root actions by construction, so
this is lossless rather than lossy.

Two era systems exist. They are not the same one
-------------------------------------------------
* **``RING_FORMAT`` / ``_FIELDS_SINCE`` (here)** version the *memmap record
  layout*. A bump means a column was added to what a stored step contains.
* **``logschema.SCHEMA_ERA`` (there)** versions *JSONL row columns*. A bump means
  a column was added to what an iteration reports.

Different substrates, different lifetimes, adjacent names, adjacent purposes —
which is the dedup-key hazard inverted and just as fatal to a grep. They are
deliberately spelled differently, and each module says the other exists so the
first person debugging an era mismatch learns there are **two** before learning
which one they are in.

One absence semantics
---------------------
A stored value is absent for exactly one of two reasons, and neither is ever a
raw zero:

1. **era absence** — the field did not exist when the record was written.
   Computed from ``record.ring_format`` against ``_FIELDS_SINCE[name]``, never
   stored, because a record written before a field existed could not have
   explained the absence of something nobody had named. (Same resolution as
   ``logschema``'s, for the same reason.)
2. **runtime absence** — the value genuinely was not available. One bit in
   ``absent_mask``; the *reason* is declared once on the field rather than stored
   per record, so absence is a bit and its meaning is a constant.

:meth:`ReplayRing.get` returns :class:`Absent` for either, carrying the reason.
It never returns the underlying zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reckoner.config import Config
from reckoner.dataset import sample_indices

#: Ring record-layout version. Bump when a FIELD IS ADDED to a stored step.
#: Distinct from :data:`reckoner.logschema.SCHEMA_ERA`, which versions JSONL row
#: columns — see this module's docstring.
RING_FORMAT = 1

#: Known ``par_source`` values, as a stored enum. Order is part of the format:
#: appending is a format bump, reordering is a corruption.
PAR_SOURCES: tuple[str, ...] = ("unverified", "bfs", "scripted", "pool")


@dataclass(frozen=True, slots=True)
class RingField:
    """One stored column. ``absence`` is the reason it may be runtime-absent."""

    name: str
    since: int = 1
    absence: str | None = None


#: Every stored column and the format version it appeared in. The ring's era map.
_FIELDS_SINCE: tuple[RingField, ...] = (
    RingField("tokens"),
    RingField("site_positions"),
    RingField("length"),
    RingField("n_sites"),
    RingField("visit_actions"),
    RingField("visit_counts"),
    # root_q is present from field one, not added later. The chess project
    # specified it and never stored it, and the gap was found only when the z/q
    # blend needed it — so it is in the first layout, with its sign proven by
    # test before it feeds anything (BRIEF-chunk9 §2).
    RingField("root_q"),
    RingField("z"),
    RingField("par_source"),
    RingField("par"),
    RingField("steps_remaining"),
    RingField("depth"),
    RingField("goal"),
)

_FIELD_INDEX = {f.name: i for i, f in enumerate(_FIELDS_SINCE)}
_FIELD_MAP = {f.name: f for f in _FIELDS_SINCE}


class RingError(ValueError):
    """A malformed record or an impossible read."""


@dataclass(frozen=True, slots=True)
class Absent:
    """A value that is not there, with the reason it is not there.

    Returned instead of a zero, always. A ring that hands back 0.0 for a missing
    ``root_q`` teaches the blend that the position was a draw.
    """

    field: str
    reason: str
    kind: str  # "era" | "runtime"

    def __bool__(self) -> bool:  # pragma: no cover - guarding a misuse
        raise RingError(
            f"{self.field} is absent ({self.kind}: {self.reason}) — test for Absent "
            "explicitly rather than relying on truthiness, which is how an absence "
            "becomes a zero."
        )


class ReplayRing:
    """Fixed-capacity ring over episode steps. Overwrites oldest first."""

    def __init__(self, capacity: int, cfg: Config, *, ring_format: int = RING_FORMAT) -> None:
        if capacity < 1:
            raise RingError(f"capacity must be >= 1; got {capacity}")
        m = cfg.model
        self.capacity = int(capacity)
        self.cfg = cfg
        self.ring_format = ring_format
        self.cursor = 0
        self.count = 0

        self.tokens = np.zeros((capacity, m.seq_len), dtype=np.int16)
        self.site_positions = np.zeros((capacity, m.max_sites), dtype=np.int16)
        self.length = np.zeros(capacity, dtype=np.int16)
        self.n_sites = np.zeros(capacity, dtype=np.int16)
        width = cfg.search.gumbel_m
        self.visit_actions = np.zeros((capacity, width), dtype=np.int32)
        self.visit_counts = np.zeros((capacity, width), dtype=np.int32)
        self.root_q = np.zeros(capacity, dtype=np.float32)
        self.z = np.zeros(capacity, dtype=np.int8)
        self.par_source = np.zeros(capacity, dtype=np.int8)
        self.par = np.zeros(capacity, dtype=np.int16)
        self.steps_remaining = np.zeros(capacity, dtype=np.int16)
        self.depth = np.zeros(capacity, dtype=np.int8)
        self.goal = np.zeros(capacity, dtype=np.int8)
        self.record_format = np.zeros(capacity, dtype=np.int16)
        self.absent_mask = np.zeros(capacity, dtype=np.uint32)

    # -- writing ----------------------------------------------------------

    def append(
        self,
        *,
        tokens: np.ndarray,
        site_positions: np.ndarray,
        visit_actions: np.ndarray,
        visit_counts: np.ndarray,
        root_q: float,
        z: int,
        par_source: str,
        par: int,
        steps_remaining: int,
        depth: int,
        goal: int,
        absent: dict[str, str] | None = None,
    ) -> int:
        """Store one step. Returns its slot.

        ``absent`` names runtime-absent fields; each must be a field that
        *declares* it can be absent, because an undeclared absence is a hole
        nobody can interpret.
        """
        if z not in (-1, 0, 1):
            raise RingError(f"z must be -1, 0 or +1; got {z}")
        if par_source not in PAR_SOURCES:
            raise RingError(f"unknown par_source {par_source!r}; known: {list(PAR_SOURCES)}")
        if par_source == "bfs" and z > 0:
            raise RingError(
                "z = +1 against bfs par: beating exact par is impossible by "
                "construction. Third layer of the same tripwire (EpisodeResult, "
                "logschema, here) — a value this load-bearing is checked wherever "
                "it passes a boundary."
            )

        mask = 0
        for name, reason in (absent or {}).items():
            spec = _FIELD_MAP.get(name)
            if spec is None:
                raise RingError(f"unknown field {name!r} named absent")
            if spec.absence is None:
                raise RingError(
                    f"{name} does not declare that it can be absent; an undeclared "
                    "absence is a hole nobody can interpret. Give the field an "
                    "`absence` reason if it really can be missing."
                )
            if not str(reason).strip():
                raise RingError(f"{name}: absence needs a reason")
            mask |= 1 << _FIELD_INDEX[name]

        i = self.cursor
        n = int(len(tokens))
        s = int(len(site_positions))
        self.tokens[i] = 0
        self.tokens[i, :n] = tokens
        self.site_positions[i] = 0
        self.site_positions[i, :s] = site_positions
        self.length[i] = n
        self.n_sites[i] = s
        w = min(len(visit_actions), self.visit_actions.shape[1])
        self.visit_actions[i] = 0
        self.visit_counts[i] = 0
        self.visit_actions[i, :w] = visit_actions[:w]
        self.visit_counts[i, :w] = visit_counts[:w]
        self.root_q[i] = root_q
        self.z[i] = z
        self.par_source[i] = PAR_SOURCES.index(par_source)
        self.par[i] = par
        self.steps_remaining[i] = steps_remaining
        self.depth[i] = depth
        self.goal[i] = goal
        self.record_format[i] = self.ring_format
        self.absent_mask[i] = mask

        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        return i

    # -- reading ----------------------------------------------------------

    def _absence(self, slot: int, name: str) -> Absent | None:
        spec = _FIELD_MAP[name]
        record_format = int(self.record_format[slot])
        if record_format < spec.since:
            return Absent(
                name,
                f"predates_field: record written under ring_format {record_format}, "
                f"{name} arrived in {spec.since}",
                "era",
            )
        if self.absent_mask[slot] & (1 << _FIELD_INDEX[name]):
            return Absent(name, spec.absence or "declared absent", "runtime")
        return None

    def get(self, slot: int) -> dict[str, object]:
        """One record, with absences reported rather than zeroed."""
        if not 0 <= slot < self.count:
            raise RingError(f"slot {slot} outside the {self.count} valid records")
        n, s = int(self.length[slot]), int(self.n_sites[slot])
        raw: dict[str, object] = {
            "tokens": self.tokens[slot, :n].copy(),
            "site_positions": self.site_positions[slot, :s].copy(),
            "length": n,
            "n_sites": s,
            "visit_actions": self.visit_actions[slot].copy(),
            "visit_counts": self.visit_counts[slot].copy(),
            "root_q": float(self.root_q[slot]),
            "z": int(self.z[slot]),
            "par_source": PAR_SOURCES[int(self.par_source[slot])],
            "par": int(self.par[slot]),
            "steps_remaining": int(self.steps_remaining[slot]),
            "depth": int(self.depth[slot]),
            "goal": int(self.goal[slot]),
        }
        for name in raw:
            absent = self._absence(slot, name)
            if absent is not None:
                raw[name] = absent
        raw["ring_format"] = int(self.record_format[slot])
        return raw

    def sample(self, k: int, seed: int) -> list[int]:
        """Slots sampled across the ring — never a prefix (the blessed sampler)."""
        return sample_indices(self.count, k, seed)

    def __len__(self) -> int:
        return self.count

    # -- persistence, for crash-resume ------------------------------------

    ARRAYS = (
        "tokens",
        "site_positions",
        "length",
        "n_sites",
        "visit_actions",
        "visit_counts",
        "root_q",
        "z",
        "par_source",
        "par",
        "steps_remaining",
        "depth",
        "goal",
        "record_format",
        "absent_mask",
    )

    def save(self, path: Path) -> dict:
        """Arrays plus a meta sidecar. Meta is written LAST.

        Write ordering is the resume contract: a meta that exists means the
        arrays beside it are complete. A crash between the two leaves a ring
        without meta, which reads as "no ring" rather than as a truncated one.
        """
        path.mkdir(parents=True, exist_ok=True)
        for name in self.ARRAYS:
            getattr(self, name).tofile(path / f"{name}.bin")
        meta = {
            "ring_format": self.ring_format,
            "capacity": self.capacity,
            "cursor": self.cursor,
            "count": self.count,
            "seq_len": self.cfg.model.seq_len,
            "max_sites": self.cfg.model.max_sites,
            "gumbel_m": self.cfg.search.gumbel_m,
            "par_sources": list(PAR_SOURCES),
            "bytes_per_record": bytes_per_record(self.cfg),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return meta

    @classmethod
    def load(cls, path: Path, cfg: Config) -> ReplayRing:
        meta = json.loads((path / "meta.json").read_text())
        for key, got in (
            ("seq_len", cfg.model.seq_len),
            ("max_sites", cfg.model.max_sites),
            ("gumbel_m", cfg.search.gumbel_m),
        ):
            if meta[key] != got:
                raise RingError(
                    f"ring was written with {key}={meta[key]} but the config says {got}. "
                    "The record layout is denominated in these; loading anyway would "
                    "reinterpret every stored step."
                )
        if meta["par_sources"] != list(PAR_SOURCES):
            raise RingError(
                f"par_source enum changed: stored {meta['par_sources']}, now "
                f"{list(PAR_SOURCES)}. Order is part of the format — appending is a "
                "format bump, reordering is a corruption."
            )
        ring = cls(meta["capacity"], cfg, ring_format=meta["ring_format"])
        for name in cls.ARRAYS:
            target = getattr(ring, name)
            data = np.fromfile(path / f"{name}.bin", dtype=target.dtype)
            target[...] = data.reshape(target.shape)
        ring.cursor = meta["cursor"]
        ring.count = meta["count"]
        return ring


def bytes_per_record(cfg: Config) -> int:
    """Per-record footprint, computed rather than quoted from the docstring."""
    m = cfg.model
    return (
        m.seq_len * 2
        + m.max_sites * 2
        + 2  # length
        + 2  # n_sites
        + cfg.search.gumbel_m * 4  # visit_actions
        + cfg.search.gumbel_m * 4  # visit_counts
        + 4  # root_q
        + 1  # z
        + 1  # par_source
        + 2  # par
        + 2  # steps_remaining
        + 1  # depth
        + 1  # goal
        + 2  # record_format
        + 4  # absent_mask
    )
