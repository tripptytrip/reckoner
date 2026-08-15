"""Ladder rungs, each with its determinism probe as a **construction gate**.

Inherited law: every detector that gates automation is validated on both
polarities before live use. A ladder rung is a measuring instrument, and an
instrument whose repeatability nobody checked produces differences nobody can
attribute — the skill-limiter lesson, where a "stronger opponent" turned out to
be the same opponent with a different seed.

So each arm declares its determinism **and proves it at construction**:

* :class:`GreedyHeuristic` — deterministic. Probed: the same problem twice gives
  the same derivation, and the probe would notice if it did not.
* :class:`RandomRewriter` — **stochastic by design**, which is a different claim
  from "not yet made deterministic". Per-problem derived seeds, a configured
  repetition count, and ``nondeterministic=True`` on every row **from day one**
  rather than after a surprise. Its probe asserts the *opposite* property: the
  same problem under different seeds must differ, or the arm is not sampling.
* :class:`SympySolver` — external, so its **version is part of its identity**
  (`cas_version` on every row), it is a context manager, and it **clean-skips**
  when sympy is absent rather than failing the pass.

Currencies, per the ruling: the first two are **rule-denominated** — they act in
our rules, so their steps are steps and z against par is meaningful. Sympy is
not, so it scores solve-vs-budget and its rows can never be read as z rows.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from reckoner.config import Config
from reckoner.episode import Problem, verify
from reckoner.expr import Expr
from reckoner.logschema import CURRENCY_BUDGET, CURRENCY_Z
from reckoner.rules import apply, legal_actions


class ArmError(RuntimeError):
    """An arm that cannot be trusted as an instrument."""


@dataclass
class ArmResult:
    """What one arm did to one problem, in **its own** currency."""

    solved: bool
    steps: int
    currency: str


class Arm(Protocol):
    name: str
    currency: str
    nondeterministic: bool

    def play(self, problem: Problem, cfg: Config, seed: int) -> ArmResult: ...


def _play_greedy(problem: Problem, cfg: Config, budget: int) -> ArmResult:
    expr: Expr = problem.expr
    checker = random.Random(0)
    for step in range(budget):
        actions = legal_actions(expr)
        if not actions:
            return ArmResult(False, step, CURRENCY_Z)
        # Largest-subtree-first, with the tie broken by action order so the arm
        # is a FUNCTION of the state — a heuristic with an arbitrary tie-break is
        # a stochastic arm that has not admitted it.
        rule_id, site_id = max(actions, key=lambda a: (a[1], a[0]))
        expr = apply(expr, rule_id, site_id)
        if verify(problem, expr, cfg, checker):
            return ArmResult(True, step + 1, CURRENCY_Z)
    return ArmResult(False, budget, CURRENCY_Z)


@dataclass
class GreedyHeuristic:
    """Largest-subtree-first. Deterministic, and the probe proves it."""

    name: str = "greedy"
    currency: str = CURRENCY_Z
    nondeterministic: bool = False

    def play(self, problem: Problem, cfg: Config, seed: int) -> ArmResult:
        return _play_greedy(problem, cfg, cfg.episode.step_cap)

    def probe(self, problem: Problem, cfg: Config) -> None:
        """Construction gate: same problem, different seeds, identical result."""
        a = self.play(problem, cfg, seed=0)
        b = self.play(problem, cfg, seed=99)
        if (a.solved, a.steps) != (b.solved, b.steps):
            raise ArmError(
                f"{self.name} declares itself deterministic and is not: "
                f"{(a.solved, a.steps)} vs {(b.solved, b.steps)}. A rung whose "
                "repeatability nobody checked produces differences nobody can attribute."
            )


@dataclass
class RandomRewriter:
    """Uniform legal play. **Stochastic by design**, and it says so on every row."""

    name: str = "random"
    currency: str = CURRENCY_Z
    nondeterministic: bool = True

    def play(self, problem: Problem, cfg: Config, seed: int) -> ArmResult:
        rng = random.Random(seed)
        checker = random.Random(seed ^ 0x5EED)
        expr: Expr = problem.expr
        for step in range(cfg.episode.step_cap):
            actions = legal_actions(expr)
            if not actions:
                return ArmResult(False, step, CURRENCY_Z)
            expr = apply(expr, *rng.choice(actions))
            if verify(problem, expr, cfg, checker):
                return ArmResult(True, step + 1, CURRENCY_Z)
        return ArmResult(False, cfg.episode.step_cap, CURRENCY_Z)

    def probe(self, problem: Problem, cfg: Config) -> None:
        """The **opposite** gate: it must actually vary.

        A stochastic arm that returns the same answer under every seed is a
        deterministic arm with a seed parameter — and its rows would carry
        ``nondeterministic=True`` while behaving otherwise, which is a label
        that lies. Same seed must also reproduce, or the reps are not poolable.
        """
        if (r := self.play(problem, cfg, 1)) != self.play(problem, cfg, 1):
            raise ArmError(f"{self.name}: the same seed did not reproduce ({r})")
        outcomes = {
            (self.play(problem, cfg, s).solved, self.play(problem, cfg, s).steps) for s in range(24)
        }
        if len(outcomes) == 1:
            raise ArmError(
                f"{self.name} declares itself stochastic and produced one outcome "
                f"across 24 seeds: {outcomes}. A stochastic arm that never varies is "
                "a deterministic arm wearing a seed parameter."
            )


class SympySolver:
    """External CAS rung. Version-pinned, context-managed, clean-skip if absent.

    **Its currency is solve-vs-budget, never z.** A sympy derivation is not a
    sequence of our rewrites, so its step count is denominated in another rule
    system; scoring it against our par would compare two lengths measured in
    different units and call the difference skill. Spec: sympy is a rung, never
    par.
    """

    name = "sympy"
    currency = CURRENCY_BUDGET
    nondeterministic = False

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.available = False
        self.version = "absent"
        self._module = None

    def __enter__(self) -> SympySolver:
        try:
            import sympy
        except ImportError:
            # Clean skip: an absent optional rung is a smaller ladder, not a
            # failed pass. The absence is recorded, never inferred from a gap.
            return self
        self._module = sympy
        self.version = str(sympy.__version__)
        self.available = True
        return self

    def __exit__(self, *exc) -> None:
        self._module = None

    def probe(self) -> None:
        """Construction gate: the CAS answers the same question the same way."""
        if not self.available:
            return
        sympy = self._module
        x = sympy.Symbol("x")
        first = sympy.solve(sympy.Eq(3 * x + 6, 21), x)
        second = sympy.solve(sympy.Eq(3 * x + 6, 21), x)
        if first != second:
            raise ArmError(f"sympy {self.version} is not repeatable: {first} vs {second}")
        if first != [5]:
            raise ArmError(f"sympy {self.version} solved 3x+6=21 as {first}, expected [5]")
