"""The scripted solver: ground-truth derivations for the easy strata.

Its two jobs, and they are not the same job:

1. **Phase 1 supervision.** ``(state → rule, site)`` pairs from a derivation that
   is known-correct, for the warm start chunk 8 trains.
2. **Scripted par** — a *provisional floor*, never a ceiling (spec §3). It is
   what labels the middle strata a BFS cannot reach, and `par_from_pool_frac`
   exists to replace it as soon as the model is better than it.

It is a heuristic policy, not a search, and it is **measurably suboptimal**:
`FINDINGS.md` F-01 is the first instance — preferring `eval_add` over `eval_mul`
costs a step on `7 + (9 − 30) + 12 × 12`, 4 against a BFS-exact 3. That gap is
the whole reason scripted par is a floor. ``scripted_par_delta`` measures it
where both labels exist, which is the calibration the mid-strata tier needs.

The policy, in order:

1. **Finish if you can** — ``div_both_sides`` ends a SOLVE outright.
2. **Compute** — ``eval_mul`` before ``eval_add``, because folding a product
   *into* a pending sum saves the step F-01 lost.
3. **Collect** — ``combine_like_terms``.
4. **Move** — ``sub_both_sides``, choosing its operand by the rule below.

Choosing what to move is where a naive policy loops forever. If both sides carry
the variable, move a *variable* addend so the like terms land together and
``combine_like_terms`` can fire; otherwise move a *numeric* addend to clear the
constant off the variable's side. Getting that backwards shuffles constants
across the equals sign indefinitely — it produced runaway derivations in chunk 4
before the manifest caught them.

``add_both_sides`` is never chosen. It only grows the state (ROUND-01), so a
policy that reaches for it does not terminate.
"""

from __future__ import annotations

import random

from reckoner.config import Config
from reckoner.episode import Episode, Problem, bfs_par
from reckoner.expr import Expr, Num, Op
from reckoner.rules import RULE_BY_NAME, enumerate_sites
from reckoner.semantics import variables
from reckoner.vocab import EQ

#: Preference order. ``eval_mul`` sits above ``eval_add`` on purpose — see F-01.
POLICY: tuple[str, ...] = (
    "div_both_sides",
    "eval_mul",
    "eval_sub",
    "eval_add",
    "combine_like_terms",
    "sub_both_sides",
)


def _choose_operand(state: Expr, candidates: list[tuple[int, int]]) -> tuple[int, int]:
    """Which addend a both-sides rule should move.

    Both sides carry the variable → move a variable term, so the like terms meet.
    Otherwise → move a constant, clearing the variable's side. Backwards, this
    shuffles constants across the equals sign forever.
    """
    sites = enumerate_sites(state)
    both = (
        isinstance(state, Op)
        and state.kind == EQ
        and all(variables(side) for side in state.children)
    )
    for action in candidates:
        site = sites[action[1]]
        if len(site.path) != 2:
            continue
        is_numeric = isinstance(site.node, Num)
        if (variables(site.node) and both) or (is_numeric and not both):
            return action
    return candidates[0]


def scripted_solve(
    problem: Problem, cfg: Config | None = None, cap: int | None = None
) -> list[tuple[int, int]] | None:
    """A derivation, or ``None`` if the policy cannot finish within the cap.

    Runs a real :class:`Episode`, so "solved" means what the checker means. A
    solver with its own idea of done is the labeller defect (F-02) wearing
    different clothes.
    """
    cfg = cfg or Config()
    limit = cap if cap is not None else cfg.episode.step_cap
    episode = Episode(cfg=cfg, rng=random.Random(cfg.seed))
    episode.reset(problem)

    path: list[tuple[int, int]] = []
    while not episode.done and len(path) < limit:
        actions = episode.legal()
        choice = None
        for name in POLICY:
            rule_id = RULE_BY_NAME[name].rule_id
            candidates = [a for a in actions if a[0] == rule_id]
            if not candidates:
                continue
            choice = (
                _choose_operand(episode.expr, candidates)  # type: ignore[arg-type]
                if name == "sub_both_sides"
                else candidates[0]
            )
            break
        if choice is None:
            return None
        episode.step(choice)
        path.append(choice)

    return path if episode.solved else None


def scripted_par(problem: Problem, cfg: Config | None = None) -> int | None:
    """The floor this solver can certify. ``None`` if it cannot solve at all."""
    path = scripted_solve(problem, cfg)
    return None if path is None else len(path)


def scripted_par_delta(problem: Problem, cfg: Config | None = None) -> int | None:
    """``scripted − bfs`` where both exist; ``None`` when either does not.

    Zero means the policy found an optimal derivation. Positive is the gap the
    "provisional floor" language is paying for, quantified. Negative is
    impossible and would mean the BFS label is wrong.
    """
    cfg = cfg or Config()
    exact = bfs_par(problem, cfg)
    scripted = scripted_par(problem, cfg)
    if exact is None or scripted is None:
        return None
    if scripted < exact:
        raise ValueError(
            f"scripted par {scripted} beats BFS-exact par {exact} — exact par is the "
            "minimum, so this means the BFS label is wrong (see FINDINGS.md F-02)"
        )
    return scripted - exact
