"""Search: array-tree MCTS, Gumbel root, Sequential Halving — the careful port.

There is no opponent, and that changes two things
-------------------------------------------------
**Backup does not negate per ply.** The sharpest porting hazard in the project
(plan §8 decision 5), pinned by a config key ``config.validate()`` refuses to set
to anything else.

**Backup takes the max, not the mean.** With no opponent there is nothing to
average over: every choice below a node is ours, so a node is worth the best
thing reachable from it. Mean backup would be quietly pessimistic — averaging a
winning line against siblings we would never pick. ``tests/test_search.py``
hand-computes a mixed three-level tree, because "no negation" and "max not mean"
are two distinct mistakes and only one of them is loud.

**Proof-directed breadth is N/A** and is deliberately not ported: it exists to
force a proof through every opponent reply, and there are no replies here.

The tree must actually deepen
-----------------------------
The first version of this module expanded only the root's children and then
re-backed-up the same values: 48 simulations produced **2 nodes**
(``FINDINGS.md`` F-06). Every chunk-7 gate passed anyway — a depth-1 suite needs
one ply, budget identity counts visits regardless, and sync-vs-batched compared
two equally shallow paths — so the gates were all satisfiable by a search that
does not search. ``stats.max_depth`` and ``stats.nodes`` are now reported, and a
test asserts that ``sims >> m`` builds a tree deeper than one ply.

Where the batching lives, and why it is there
---------------------------------------------
**Batching is across concurrent searches, never within one search.** Inside one
tree, simulation *k+1*'s selection depends on simulation *k*'s backup, so pooling
leaves within a tree would change what the tree does — and the equivalence test
would then be comparing two algorithms and would have to be weakened to a
tolerance. Batching across trees has no such coupling, so sync-vs-batched parity
is **exact by construction**, and it is the shape AGENTS.md §8 prescribes anyway:
many CPU-side actors feeding one process that owns the GPU.

``_simulate`` is therefore a generator that *yields* leaf requests and receives
evaluations. :func:`search` drives one; :func:`run_batched` drives many and pools
their requests.

Terminal states
---------------
A terminal leaf is not expanded and its value is its outcome, but **it still
consumes a simulation** — budget means budget.

``StateTooLarge`` is a **counted terminal loss**, step-cap-equivalent semantics.
Never a crash mid-batch, and never an edit to the legality mask — legality stays
the engine's alone. ``add_both_sides`` makes it reachable from the first stub
rollout (F-05), so the counter is in ``SearchStats`` from field one.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field

import numpy as np

from reckoner.config import Config
from reckoner.episode import Problem, verify
from reckoner.expr import Expr
from reckoner.model import StateTooLarge, action_index, encode
from reckoner.rules import apply, legal_actions

Evaluation = tuple[np.ndarray, float]
Evaluator = Callable[[Sequence[tuple[Problem, Expr]]], list[Evaluation]]


@dataclass
class SearchStats:
    """What a search spent and what it met. Absence carries a reason."""

    sims_requested: int = 0
    sims_used: int = 0
    nodes: int = 0
    max_depth: int = 0
    evaluations: int = 0
    terminal_solved: int = 0
    terminal_no_actions: int = 0
    state_too_large: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class SearchResult:
    visits: np.ndarray
    values: np.ndarray
    actions: list[tuple[int, int]]
    chosen: tuple[int, int] | None
    root_value: float
    stats: SearchStats = field(default_factory=SearchStats)

    def improved_policy(self) -> np.ndarray:
        total = self.visits.sum()
        if total == 0:
            return np.full(len(self.actions), 1.0 / max(1, len(self.actions)), dtype=np.float32)
        return (self.visits / total).astype(np.float32)


class _Tree:
    """Per-node stats in arrays; per-node child stats sized by the legal count."""

    __slots__ = (
        "visits",
        "value",
        "terminal",
        "depth",
        "parent",
        "from_slot",
        "expr",
        "actions",
        "priors",
        "children",
        "child_visits",
        "child_values",
        "size",
    )

    def __init__(self, capacity: int) -> None:
        self.visits = np.zeros(capacity, dtype=np.int32)
        self.value = np.full(capacity, -1.0, dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=bool)
        self.depth = np.zeros(capacity, dtype=np.int32)
        self.parent = np.full(capacity, -1, dtype=np.int32)
        self.from_slot = np.full(capacity, -1, dtype=np.int32)
        self.expr: list[Expr | None] = [None] * capacity
        self.actions: list[list[tuple[int, int]]] = [[] for _ in range(capacity)]
        self.priors: list[np.ndarray | None] = [None] * capacity
        self.children: list[np.ndarray | None] = [None] * capacity
        self.child_visits: list[np.ndarray | None] = [None] * capacity
        self.child_values: list[np.ndarray | None] = [None] * capacity
        self.size = 0

    def add(self, expr: Expr, parent: int, slot: int) -> int:
        index = self.size
        self.size += 1
        self.expr[index] = expr
        self.parent[index] = parent
        self.from_slot[index] = slot
        self.depth[index] = 0 if parent < 0 else int(self.depth[parent]) + 1
        return index

    def open(self, node: int, actions: list[tuple[int, int]], priors: np.ndarray) -> None:
        self.actions[node] = actions
        self.priors[node] = priors
        self.children[node] = np.full(len(actions), -1, dtype=np.int32)
        self.child_visits[node] = np.zeros(len(actions), dtype=np.int32)
        self.child_values[node] = np.full(len(actions), -1.0, dtype=np.float32)


def _backup(tree: _Tree, node: int, value: float) -> None:
    """To the root: visits increment, **values take the max**, nothing negates."""
    current = node
    while current != -1:
        tree.visits[current] += 1
        if value > tree.value[current]:
            tree.value[current] = value
        parent = int(tree.parent[current])
        if parent != -1:
            slot = int(tree.from_slot[current])
            tree.child_visits[parent][slot] += 1  # type: ignore[index]
            if value > tree.child_values[parent][slot]:  # type: ignore[index]
                tree.child_values[parent][slot] = value  # type: ignore[index]
        current = parent


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max())
    return exp / exp.sum()


def _sigma(q: np.ndarray, max_visits: int, cfg: Config) -> np.ndarray:
    return (cfg.search.c_visit + max_visits) * cfg.search.c_scale * q


def _select(tree: _Tree, node: int, cfg: Config) -> int:
    """Gumbel-AZ's non-root rule: follow the improved policy, minus visit share."""
    priors, visits, values = tree.priors[node], tree.child_visits[node], tree.child_values[node]
    assert priors is not None and visits is not None and values is not None
    completed = np.where(visits > 0, values, float(tree.value[node]))
    improved = _softmax(priors + _sigma(completed, int(visits.max(initial=0)), cfg))
    return int(np.argmax(improved - visits / (1 + visits.sum())))


def _leaf_outcome(
    problem: Problem, expr: Expr, cfg: Config, rng: random.Random, stats: SearchStats
) -> float | None:
    """Terminal value, or ``None`` if the leaf needs evaluating."""
    if verify(problem, expr, cfg, rng):
        stats.terminal_solved += 1
        return 1.0
    try:
        encode(problem, expr, cfg)
    except StateTooLarge:
        stats.state_too_large += 1
        return -1.0
    if not legal_actions(expr):
        stats.terminal_no_actions += 1
        return -1.0
    return None


def _priors_for(raw: np.ndarray, actions: list[tuple[int, int]], cfg: Config) -> np.ndarray:
    return np.array(
        [raw[action_index(r, s, cfg.model.max_sites)] for r, s in actions], dtype=np.float32
    )


def _simulate(
    problem: Problem, expr: Expr, cfg: Config, rng: random.Random, sims: int, m: int
) -> Generator[tuple[Problem, Expr], Evaluation, SearchResult]:
    """Yields leaves needing evaluation; returns the finished result."""
    stats = SearchStats(sims_requested=sims)
    tree = _Tree(sims + 4)
    root = tree.add(expr, -1, -1)
    stats.nodes = 1

    root_actions = legal_actions(expr)
    if not root_actions:
        stats.terminal_no_actions += 1
        return SearchResult(np.zeros(0, np.int32), np.zeros(0, np.float32), [], None, -1.0, stats)

    raw, root_value = yield (problem, expr)
    stats.evaluations += 1
    logits = _priors_for(raw, root_actions, cfg).astype(np.float64)
    tree.open(root, root_actions, logits.astype(np.float32))
    tree.value[root] = root_value

    gumbel = (
        np.array([-np.log(-np.log(rng.random())) for _ in root_actions])
        if cfg.search.root_noise
        else np.zeros(len(root_actions))
    )
    considered = list(np.argsort(-(logits + gumbel))[: min(m, len(root_actions))])

    budget, remaining = sims, list(considered)
    while budget > 0 and remaining:
        rounds_left = max(1, int(np.ceil(np.log2(max(2, len(remaining))))))
        per_action = max(1, budget // (rounds_left * len(remaining)))
        for slot in remaining:
            for _ in range(per_action):
                if budget <= 0:
                    break
                budget -= 1
                stats.sims_used += 1

                node, chosen = root, slot
                while True:
                    child = int(tree.children[node][chosen])  # type: ignore[index]
                    if child == -1:
                        rule_id, site_id = tree.actions[node][chosen]
                        successor = apply(tree.expr[node], rule_id, site_id)  # type: ignore[arg-type]
                        child = tree.add(successor, node, chosen)
                        tree.children[node][chosen] = child  # type: ignore[index]
                        stats.nodes += 1
                        stats.max_depth = max(stats.max_depth, int(tree.depth[child]))
                        outcome = _leaf_outcome(problem, successor, cfg, rng, stats)
                        if outcome is not None:
                            tree.terminal[child] = True
                            tree.value[child] = outcome
                            _backup(tree, child, outcome)
                        else:
                            child_raw, child_value = yield (problem, successor)
                            stats.evaluations += 1
                            child_actions = legal_actions(successor)
                            tree.open(
                                child, child_actions, _priors_for(child_raw, child_actions, cfg)
                            )
                            _backup(tree, child, float(child_value))
                        break
                    if tree.terminal[child]:
                        _backup(tree, child, float(tree.value[child]))
                        break
                    node = child
                    chosen = _select(tree, node, cfg)

        visits, values = tree.child_visits[root], tree.child_values[root]
        assert visits is not None and values is not None
        if len(remaining) == 1:
            break
        remaining = sorted(
            remaining,
            key=lambda s: (
                logits[s] + gumbel[s] + _sigma(values[s : s + 1], int(visits.max()), cfg)[0]
            ),
            reverse=True,
        )[: max(1, len(remaining) // 2)]

    visits, values = tree.child_visits[root], tree.child_values[root]
    assert visits is not None and values is not None
    best = max(
        considered,
        key=lambda s: logits[s] + gumbel[s] + _sigma(values[s : s + 1], int(visits.max()), cfg)[0],
    )
    return SearchResult(
        visits.copy(),
        values.copy(),
        root_actions,
        root_actions[best],
        float(tree.value[root]),
        stats,
    )


def search(
    problem: Problem,
    expr: Expr,
    evaluator: Evaluator,
    cfg: Config,
    rng: random.Random,
    *,
    sims: int | None = None,
    m: int | None = None,
) -> SearchResult:
    """One search, evaluating one leaf at a time."""
    generator = _simulate(
        problem,
        expr,
        cfg,
        rng,
        sims if sims is not None else cfg.search.sims,
        m if m is not None else cfg.search.gumbel_m,
    )
    try:
        request = next(generator)
        while True:
            request = generator.send(evaluator([request])[0])
    except StopIteration as stop:
        return stop.value  # type: ignore[no-any-return]


def run_batched(
    items: Sequence[tuple[Problem, Expr, random.Random]],
    evaluator: Evaluator,
    cfg: Config,
    *,
    sims: int | None = None,
    m: int | None = None,
    batch_size: int | None = None,
) -> list[SearchResult]:
    """Many searches concurrently, pooling their leaf evaluations.

    Exact parity with :func:`search` by construction: each tree advances
    sequentially and only the *evaluator calls* are pooled across trees, so there
    is no coupling to approximate away and the equivalence test is an equality.
    """
    sims = sims if sims is not None else cfg.search.sims
    m = m if m is not None else cfg.search.gumbel_m
    size = batch_size if batch_size is not None else cfg.search.batch_leaves

    generators = [_simulate(p, e, cfg, r, sims, m) for p, e, r in items]
    results: list[SearchResult | None] = [None] * len(generators)
    pending: list[tuple[int, tuple[Problem, Expr]]] = []

    for index, generator in enumerate(generators):
        try:
            pending.append((index, next(generator)))
        except StopIteration as stop:
            results[index] = stop.value

    while pending:
        chunk, rest = pending[:size], pending[size:]
        answers = evaluator([leaf for _index, leaf in chunk])
        resumed: list[tuple[int, tuple[Problem, Expr]]] = []
        for (index, _leaf), answer in zip(chunk, answers, strict=True):
            try:
                resumed.append((index, generators[index].send(answer)))
            except StopIteration as stop:
                results[index] = stop.value
        pending = resumed + rest

    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


def uniform_stub(cfg: Config) -> Evaluator:
    """Flat priors, neutral value — the stub the depth-1 gate runs against."""
    width = 7 * cfg.model.max_sites

    def evaluate(leaves: Sequence[tuple[Problem, Expr]]) -> list[Evaluation]:
        return [(np.zeros(width, dtype=np.float32), 0.0) for _ in leaves]

    return evaluate
