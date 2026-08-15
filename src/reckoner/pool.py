"""The snapshot league: pool par, re-solved by an old model under its own wiring.

Pool par is the escalation mechanism — a par produced by a *snapshot* solving the
problem at episode time, so the bar rises as the model does. Three things about
it are load-bearing, and each has a failure this project has already met one
layer down.

**1. A snapshot is denominated in a rule system.** Loading one whose
``ruleset_version`` or ``vocab_version`` differs would produce a par measured in a
rule system that is not the one being played — F-02's shape, with a provenance tag
that says ``pool`` and is, in its own terms, true. Refusal is the default and the
refusal is a **counted event**, never a silently smaller pool (chunk-6
registration).

**2. A snapshot solves under its OWN declaration.** The value-head state rides in
the checkpoint meta and the pool honours it, not the running run's. A pre-switch
snapshot re-solved with post-switch wiring produces a par that snapshot never
achieved — a wrong number under a true-in-its-own-terms tag, F-02 for the third
time. This is why `checkpoint_meta` carries ``value_head`` at all.

**3. Unavailability is a counted fallback, never a silent substitution.** When the
sampled snapshot cannot solve the problem within its budget, pool par is
*undefined for that pair*. The episode falls back to the problem's own
``bfs``/``scripted`` par **with the provenance flipped to match**, and
``pool_par_unavailable`` increments. The fallback is allowed; what is forbidden is
the fallback happening without the record saying so — a par tagged ``pool`` that
was not produced by a pool member is the defect this whole module is built to
avoid.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from reckoner.config import Config
from reckoner.episode import Problem, verify
from reckoner.model import Reckoner, load_checkpoint
from reckoner.rules import apply, legal_actions
from reckoner.search import Evaluator, search
from reckoner.valuegate import ValueHeadState, value_contribution


class PoolError(ValueError):
    """A pool that cannot be interpreted."""


@dataclass
class Member:
    """One snapshot, with the declaration it was played under."""

    path: Path
    meta: dict
    model: Reckoner

    @property
    def value_head(self) -> ValueHeadState:
        """**The snapshot's own wiring**, not the running run's."""
        declared = self.meta.get("value_head") or {}
        return ValueHeadState(
            live=bool(declared.get("live", False)),
            switched_at_iteration=declared.get("switched_at_iteration"),
        )

    @property
    def step(self) -> int:
        return int(self.meta.get("step", 0))


@dataclass
class PoolStats:
    """What the pool did, surfaced as counted events rather than inferred."""

    refusals: int = 0
    refused_paths: list[str] = field(default_factory=list)
    pool_par_solved: int = 0
    pool_par_unavailable: int = 0
    seconds_solving: float = 0.0

    def as_dict(self) -> dict:
        return {
            "refusals": self.refusals,
            "refused_paths": list(self.refused_paths),
            "pool_par_solved": self.pool_par_solved,
            "pool_par_unavailable": self.pool_par_unavailable,
            "seconds_solving": round(self.seconds_solving, 3),
        }


@dataclass
class PoolPar:
    """A par and the truth about where it came from."""

    par: int
    par_source: str
    par_asof: int | None
    fell_back: bool


class CheckpointPool:
    """Snapshots that supply pool par. Membership and sampling are declared."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.members: list[Member] = []
        self.stats = PoolStats()

    # -- membership --------------------------------------------------------

    def add(self, path: Path) -> Member | None:
        """Load a snapshot, or **refuse and count it**. Never silently skip.

        ``strict_versions`` is not passed as False here and must not be: the
        escape hatch exists for offline inspection, not for the league
        (BRIEF-chunk9 §4).
        """
        try:
            model, meta = load_checkpoint(path, self.cfg)
        except ValueError as exc:
            self.stats.refusals += 1
            self.stats.refused_paths.append(str(path))
            raise PoolError(
                f"pool refused {path.name}: {exc} — the refusal is counted, and the "
                "pool is smaller by one member that a reader can see rather than infer."
            ) from exc
        model.eval()
        member = Member(path=path, meta=meta, model=model)
        self.members.append(member)
        # Bounded, most-recent-first: pool par is re-solved at episode time, so
        # membership is a cost as well as a population.
        if len(self.members) > self.cfg.league.pool_size:
            self.members = sorted(self.members, key=lambda m: m.step)[-self.cfg.league.pool_size :]
        return member

    def try_add(self, path: Path) -> bool:
        """``add`` without raising, for a pool being assembled from a directory."""
        try:
            self.add(path)
        except PoolError:
            return False
        return True

    def sample(self, rng: random.Random) -> Member | None:
        if not self.members:
            return None
        if self.cfg.league.pool_sample != "uniform":  # pragma: no cover - validate() guards
            raise PoolError(f"unknown pool_sample {self.cfg.league.pool_sample!r}")
        return rng.choice(self.members)

    def __len__(self) -> int:
        return len(self.members)

    def composition(self) -> dict:
        """Logged per iteration: who is in the pool, by step."""
        return {
            "size": len(self.members),
            "steps": sorted(m.step for m in self.members),
            "value_head_live": sorted(m.step for m in self.members if m.value_head.live),
        }

    # -- pool par ----------------------------------------------------------

    def par_for(
        self,
        problem: Problem,
        member: Member,
        evaluator_factory,
        rng: random.Random,
        *,
        sims: int | None = None,
        m: int | None = None,
        budget: int | None = None,
    ) -> PoolPar:
        """Solve ``problem`` with ``member``, under **the member's** declaration.

        ``evaluator_factory(model, value_scale)`` builds the evaluator; the scale
        comes from the SNAPSHOT's value-head state, so a pre-switch member plays
        value-silent even inside a post-switch run.

        Falls back — counted — when the member cannot solve within its budget.
        """
        cfg = self.cfg
        sims = sims if sims is not None else cfg.search.sims
        m = m if m is not None else cfg.search.gumbel_m
        budget = budget if budget is not None else cfg.episode.step_cap

        scale = value_contribution(member.value_head)
        evaluator: Evaluator = evaluator_factory(member.model, scale)

        started = time.perf_counter()
        expr = problem.expr
        steps = 0
        checker = random.Random(rng.randrange(1 << 30))
        while steps < budget:  # noqa: SIM113 - `steps` is the solve length, not an index
            if not legal_actions(expr):
                break
            result = search(problem, expr, evaluator, cfg, rng, sims=sims, m=m, steps_taken=steps)
            if result.chosen is None:
                break
            expr = apply(expr, *result.chosen)
            steps += 1
            if verify(problem, expr, cfg, checker):
                self.stats.pool_par_solved += 1
                self.stats.seconds_solving += time.perf_counter() - started
                return PoolPar(par=steps, par_source="pool", par_asof=member.step, fell_back=False)

        # Undefined for this pair. The fallback is allowed; the silence is not.
        self.stats.pool_par_unavailable += 1
        self.stats.seconds_solving += time.perf_counter() - started
        if problem.par is None:
            raise PoolError(
                "the snapshot could not solve the problem and it carries no own par "
                "to fall back to — there is no honest label for this pair."
            )
        return PoolPar(
            par=problem.par,
            par_source=problem.par_source,
            par_asof=None,
            fell_back=True,
        )
