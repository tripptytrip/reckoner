"""Search: array-tree MCTS, Gumbel root, Sequential Halving — the careful port.

There is no opponent, and that changes two things
-------------------------------------------------
**Backup does not negate per ply.** This is the sharpest porting hazard in the
project (plan §8 decision 5), and it is a config key (`search.perspective`) that
``config.validate()`` refuses to set to anything else, so no run can silently do
the wrong thing.

**Backup takes the max, not the mean.** With no opponent there is nothing to
average over: every choice below a node is *ours*, so a node is worth the best
thing reachable from it. A mean-backup port would be quietly pessimistic — it
would average a winning line against the losing siblings we would never pick —
and the symptom is a value head that learns to distrust its own best moves.
``tests/test_search.py`` hand-computes a mixed three-level tree and checks the
arithmetic at every node, because "no negation" and "max not mean" are two
distinct mistakes a blind port can make and only one of them is loud.

**Proof-directed breadth is N/A here** and is deliberately not ported. It exists
to force a proof through *every opponent reply*; in a single-agent tree there are
no replies, so there is nothing to sweep. Stated so nobody ports it blindly.

Terminal states
---------------
A terminal leaf is not expanded and its value is its outcome, but **it still
consumes a simulation**. That convention is kept from the chess port: budget
means budget, and a search that got free visits from terminal short-circuits
would report a sim count it did not spend.

``StateTooLarge`` is a **counted terminal loss** — step-cap-equivalent semantics,
a derivation that went somewhere the budget cannot represent. Never a crash
mid-batch, and never an edit to the legality mask: legality stays the engine's
alone (mono-instance law). ``add_both_sides`` makes over-long states reachable
(``FINDINGS.md`` F-05), so the case is live from the first stub rollout and the
counter is in ``SearchStats`` from field one.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from reckoner.config import Config
from reckoner.episode import Problem, verify
from reckoner.expr import Expr
from reckoner.model import StateTooLarge, action_index, encode
from reckoner.rules import apply, legal_actions

#: A leaf evaluation: priors over the action space, and a scalar value in [-1, 1].
Evaluation = tuple[np.ndarray, float]
#: Evaluates a batch of ``(problem, expr)`` leaves. Batched by contract, so the
#: sync path is simply a batch of one — which is what makes the sync/batched
#: equivalence test meaningful rather than a comparison of two code paths.
Evaluator = Callable[[Sequence[tuple[Problem, Expr]]], list[Evaluation]]

TERMINAL_SOLVED = "solved"
TERMINAL_TOO_LARGE = "state_too_large"
TERMINAL_NO_ACTIONS = "no_legal_actions"


@dataclass
class SearchStats:
    """What a search spent and what it met. Absence carries a reason."""

    sims_requested: int = 0
    sims_used: int = 0
    nodes: int = 0
    evaluations: int = 0
    batches: int = 0
    terminal_solved: int = 0
    terminal_no_actions: int = 0
    state_too_large: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class SearchResult:
    visits: np.ndarray  # (n_root_actions,) int32
    values: np.ndarray  # (n_root_actions,) float32 — max-backup Q per root action
    actions: list[tuple[int, int]]
    chosen: tuple[int, int] | None
    root_value: float
    stats: SearchStats = field(default_factory=SearchStats)

    def improved_policy(self) -> np.ndarray:
        total = self.visits.sum()
        if total == 0:
            return np.full(len(self.actions), 1.0 / max(1, len(self.actions)), dtype=np.float32)
        return (self.visits / total).astype(np.float32)


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


class _Tree:
    """Array-backed. Node 0 is the root; children are indices into the arrays."""

    __slots__ = ("visits", "value", "terminal", "expr", "children", "actions", "parent", "size")

    def __init__(self, capacity: int) -> None:
        self.visits = np.zeros(capacity, dtype=np.int32)
        self.value = np.full(capacity, -1.0, dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=bool)
        self.expr: list[Expr | None] = [None] * capacity
        self.children: list[list[int] | None] = [None] * capacity
        self.actions: list[list[tuple[int, int]]] = [[] for _ in range(capacity)]
        self.parent = np.full(capacity, -1, dtype=np.int32)
        self.size = 0

    def add(self, expr: Expr | None, parent: int) -> int:
        index = self.size
        self.size += 1
        self.expr[index] = expr
        self.parent[index] = parent
        return index


def _backup(tree: _Tree, node: int, value: float) -> None:
    """Propagate to the root: visit counts increment, **values take the max**.

    No negation anywhere in this function, and that absence is the point. Every
    choice on the path was ours, so an ancestor is worth the best thing found
    beneath it — not the average over branches we would never take.
    """
    current = node
    while current != -1:
        tree.visits[current] += 1
        if value > tree.value[current]:
            tree.value[current] = value
        current = int(tree.parent[current])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _terminal_value(problem: Problem, expr: Expr, cfg: Config, rng: random.Random) -> float | None:
    """``+1`` for a verified solve, ``None`` if the state is not terminal."""
    if verify(problem, expr, cfg, rng):
        return 1.0
    return None


def _sigma(q: np.ndarray, visits: np.ndarray, cfg: Config) -> np.ndarray:
    """Gumbel-AZ's monotone transform of completed Q values."""
    return (cfg.search.c_visit + int(visits.max(initial=0))) * cfg.search.c_scale * q


def search(
    problem: Problem,
    expr: Expr,
    evaluator: Evaluator,
    cfg: Config,
    rng: random.Random,
    *,
    sims: int | None = None,
    m: int | None = None,
    batch_leaves: int | None = None,
) -> SearchResult:
    """Run one search from ``expr``. Deterministic given ``rng``'s state.

    ``batch_leaves`` only changes *when* the evaluator is called, never what the
    search does with the answers — the sync/batched equivalence test pins that
    across the whole ``m × sims`` grid, odd pairs included.
    """
    profile = cfg.search.resolved() if hasattr(cfg.search, "resolved") else cfg.search
    sims = sims if sims is not None else profile.sims
    m = m if m is not None else profile.gumbel_m
    batch = batch_leaves if batch_leaves is not None else profile.batch_leaves

    stats = SearchStats(sims_requested=sims)
    tree = _Tree(sims + 2)
    root = tree.add(expr, -1)
    stats.nodes = 1

    root_actions = legal_actions(expr)
    if not root_actions:
        stats.terminal_no_actions += 1
        return SearchResult(np.zeros(0, np.int32), np.zeros(0, np.float32), [], None, -1.0, stats)

    # Root priors come from one evaluation of the root itself.
    priors, root_value = _evaluate(evaluator, [(problem, expr)], stats)[0]
    tree.value[root] = root_value
    logits = np.array(
        [priors[action_index(r, s, cfg.model.max_sites)] for r, s in root_actions],
        dtype=np.float64,
    )

    # Gumbel top-m without replacement over the legal set.
    gumbel = (
        np.array([-np.log(-np.log(rng.random())) for _ in root_actions])
        if profile.root_noise
        else np.zeros(len(root_actions))
    )
    considered = list(np.argsort(-(logits + gumbel))[: min(m, len(root_actions))])

    tree.children[root] = [-1] * len(root_actions)
    tree.actions[root] = root_actions

    visits = np.zeros(len(root_actions), dtype=np.int32)
    values = np.full(len(root_actions), -1.0, dtype=np.float32)

    # --- Sequential Halving over the considered set -------------------------
    remaining = list(considered)
    budget = sims
    while budget > 0 and remaining:
        rounds_left = max(1, int(np.ceil(np.log2(max(2, len(remaining))))))
        per_action = max(1, budget // (rounds_left * len(remaining)))
        pending: list[tuple[int, Problem, Expr, int]] = []  # (slot, problem, expr, parent)
        for slot in remaining:
            for _ in range(per_action):
                if budget <= 0:
                    break
                budget -= 1
                stats.sims_used += 1
                rule_id, site_id = root_actions[slot]
                child = tree.children[root][slot]
                if child == -1:
                    successor = apply(expr, rule_id, site_id)
                    child = tree.add(successor, root)
                    tree.children[root][slot] = child
                    stats.nodes += 1
                    outcome = _leaf_outcome(problem, successor, cfg, rng, stats)
                    if outcome is not None:
                        tree.terminal[child] = True
                        _backup(tree, child, outcome)
                        continue
                    pending.append((slot, problem, successor, child))
                elif tree.terminal[child]:
                    _backup(tree, child, float(tree.value[child]))
                else:
                    pending.append((slot, problem, tree.expr[child], child))  # type: ignore[arg-type]

                if len(pending) >= batch:
                    _flush(pending, evaluator, tree, stats)
                    pending = []
        _flush(pending, evaluator, tree, stats)

        for slot in remaining:
            child = tree.children[root][slot]
            if child != -1:
                visits[slot] = int(tree.visits[child])
                values[slot] = float(tree.value[child])

        if len(remaining) == 1:
            break
        keep = max(1, len(remaining) // 2)
        scored = sorted(
            remaining,
            key=lambda s: logits[s] + gumbel[s] + _sigma(values[s : s + 1], visits, cfg)[0],
            reverse=True,
        )
        remaining = scored[:keep]

    chosen_slot = max(
        considered,
        key=lambda s: logits[s] + gumbel[s] + _sigma(values[s : s + 1], visits, cfg)[0],
    )
    return SearchResult(
        visits=visits,
        values=values,
        actions=root_actions,
        chosen=root_actions[chosen_slot],
        root_value=float(tree.value[root]),
        stats=stats,
    )


def _leaf_outcome(
    problem: Problem, expr: Expr, cfg: Config, rng: random.Random, stats: SearchStats
) -> float | None:
    """Terminal value, or ``None`` if the leaf needs evaluating.

    ``StateTooLarge`` is a **counted terminal loss**: the derivation went
    somewhere the budget cannot represent, which is step-cap-equivalent, not a
    crash and not a reason to touch the mask.
    """
    solved = _terminal_value(problem, expr, cfg, rng)
    if solved is not None:
        stats.terminal_solved += 1
        return solved
    try:
        encode(problem, expr, cfg)
    except StateTooLarge:
        stats.state_too_large += 1
        return -1.0
    if not legal_actions(expr):
        stats.terminal_no_actions += 1
        return -1.0
    return None


def _evaluate(
    evaluator: Evaluator, leaves: Sequence[tuple[Problem, Expr]], stats: SearchStats
) -> list[Evaluation]:
    stats.evaluations += len(leaves)
    stats.batches += 1
    return evaluator(leaves)


def _flush(
    pending: list[tuple[int, Problem, Expr, int]],
    evaluator: Evaluator,
    tree: _Tree,
    stats: SearchStats,
) -> None:
    if not pending:
        return
    results = _evaluate(evaluator, [(p, e) for _slot, p, e, _node in pending], stats)
    for (_slot, _problem, _expr, node), (_priors, value) in zip(pending, results, strict=True):
        _backup(tree, node, float(value))


# ---------------------------------------------------------------------------
# A uniform stub, for the gate and the equivalence tests
# ---------------------------------------------------------------------------


def uniform_stub(cfg: Config) -> Evaluator:
    """Flat priors, neutral value. The stub the depth-1 gate runs against.

    With flat priors the Gumbel draw is the only thing ordering the root, so
    whether the gate is reachable is pure arithmetic: the winning action must be
    *considered*, which needs ``m >= B_max``. Measured on the real suite,
    ``B_max = 5`` (``runs/gate_arithmetic.json``).
    """
    width = 7 * cfg.model.max_sites

    def evaluate(leaves: Sequence[tuple[Problem, Expr]]) -> list[Evaluation]:
        return [(np.zeros(width, dtype=np.float32), 0.0) for _ in leaves]

    return evaluate
