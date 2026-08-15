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

from reckoner.absence import Absent
from reckoner.config import Config
from reckoner.episode import Problem, verify
from reckoner.model import Reckoner, load_league_checkpoint
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
    #: Unavailability has TWO causes and they are different diseases: the pool had
    #: no members to sample, or a sampled member could not solve the problem. The
    #: capped/stuck lesson applies here — pooling them would hide which is
    #: happening, and the fixes differ (wait for snapshots vs raise the budget).
    pool_par_unavailable_empty: int = 0
    pool_par_unavailable_capped: int = 0
    seconds_solving: float = 0.0

    @property
    def pool_par_unavailable(self) -> int:
        return self.pool_par_unavailable_empty + self.pool_par_unavailable_capped

    def as_dict(self) -> dict:
        return {
            "refusals": self.refusals,
            "refused_paths": list(self.refused_paths),
            "pool_par_solved": self.pool_par_solved,
            "pool_par_unavailable": self.pool_par_unavailable,
            "pool_par_unavailable_empty": self.pool_par_unavailable_empty,
            "pool_par_unavailable_capped": self.pool_par_unavailable_capped,
            "seconds_solving": round(self.seconds_solving, 3),
        }


@dataclass
class PoolPar:
    """A par and the truth about where it came from.

    ``par_asof`` is an :class:`Absent` on a fallback, never a raw ``None``. The
    date is not missing, it is **inapplicable** — an exact par is timeless and has
    no snapshot to be as-of — and a ``None`` read as ``0`` would date it to the
    beginning of time. The no-null law holds at this layer too.
    """

    par: int
    par_source: str
    par_asof: int | Absent
    fell_back: bool
    reason: str  # "solved" | "snapshot_capped" | "pool_empty"


class CheckpointPool:
    """Snapshots that supply pool par. Membership and sampling are declared."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.members: list[Member] = []
        self.stats = PoolStats()

    # -- membership --------------------------------------------------------

    def add(self, path: Path) -> Member | None:
        """Load a snapshot, or **refuse and count it**. Never silently skip.

        Uses :func:`load_league_checkpoint`, which has **no** ``strict_versions``
        parameter — the escape hatch exists for offline inspection and is
        unreachable from here by construction, not by convention (BRIEF-chunk9 §4).
        """
        try:
            # The LEAGUE loader: no strict_versions parameter exists to pass,
            # so 'must not pass False' is impossible rather than remembered.
            model, meta = load_league_checkpoint(path, self.cfg)
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

    def enroll(
        self, model: Reckoner, step: int, value_head: ValueHeadState, path: Path
    ) -> Member | None:
        """Snapshot the running model INTO the pool. This is par escalation.

        The amendment's core mechanism is that par rises with the model, and it
        rises only because the loop feeds the pool its own checkpoints. The
        snapshot carries **the declaration it was played under**, so when it is
        later sampled it re-solves under its own wiring rather than whatever the
        run has become (rule 2).
        """
        from reckoner.model import save_checkpoint

        save_checkpoint(path, model, self.cfg, step, value_head=value_head.as_dict())
        return self.add(path)

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
                return PoolPar(
                    par=steps,
                    par_source="pool",
                    par_asof=member.step,
                    fell_back=False,
                    reason="solved",
                )

        # Undefined for this pair. The fallback is allowed; the silence is not.
        self.stats.pool_par_unavailable_capped += 1
        self.stats.seconds_solving += time.perf_counter() - started
        return self._fallback(problem, "snapshot_capped")

    def _fallback(self, problem: Problem, reason: str) -> PoolPar:
        """The problem's own par, with the provenance flipped and the date absent."""
        if problem.par is None:
            raise PoolError(
                "no pool par and no own par to fall back to — there is no honest "
                "label for this pair."
            )
        return PoolPar(
            par=problem.par,
            par_source=problem.par_source,
            par_asof=Absent(
                "par_asof",
                "source-is-exact-and-timeless: an exact par has no snapshot to be as-of",
                "inapplicable",
            ),
            fell_back=True,
            reason=reason,
        )

    def par_for_episode(
        self, problem: Problem, evaluator_factory, rng: random.Random, **kwargs
    ) -> PoolPar:
        """Sample a member and solve, or fall back with the cause NAMED.

        The empty-pool case is counted here rather than at the call site, so the
        two causes of unavailability cannot be conflated by whoever wires it up.
        """
        member = self.sample(rng)
        if member is None:
            self.stats.pool_par_unavailable_empty += 1
            return self._fallback(problem, "pool_empty")
        return self.par_for(problem, member, evaluator_factory, rng, **kwargs)
