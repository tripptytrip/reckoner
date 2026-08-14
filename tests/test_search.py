"""Search: the backup arithmetic first, then budget, equivalence, determinism.

The backup test is written first and stands first, because the port's sharpest
hazard is silent: a tree that negates per ply, or averages where it should
maximise, still returns plausible numbers.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.episode import Problem
from reckoner.expr import add, eq, mul, num, var
from reckoner.search import _backup, _Tree, search, uniform_stub
from reckoner.vocab import GOAL_SOLVE, VAR_X

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()
X = var(VAR_X)

PROBLEM = Problem(
    goal=GOAL_SOLVE,
    target=VAR_X,
    par=3,
    par_source="bfs",
    expr=eq(add(mul(num(3), X), num(6)), num(21)),
)


# ---------------------------------------------------------------------------
# The backup arithmetic — hand-computed, mixed leaf types
# ---------------------------------------------------------------------------


def test_backup_is_max_and_never_negates() -> None:
    """A hand-computed three-level tree with **both** leaf kinds in it.

    Layout (values are what each leaf backs up):

        root
        ├── A          terminal-solved leaf        +1.0
        └── B          value-estimated leaf        −0.5
            ├── B1     value-estimated leaf        +0.25
            └── B2     terminal (too large) leaf   −1.0

    Two distinct mistakes a blind port makes, and this catches both:

    * **negation per ply** would make root = −1 × max(children) — the sign of
      the whole tree flips, and the search prefers its worst line.
    * **mean backup** would make root = mean(+1, −0.5, +0.25, −1) = −0.0625
      instead of +1.0 — quietly pessimistic, averaging a winning line against
      siblings we would never choose.

    Expected, by hand: B1 = +0.25, B2 = −1.0, B = max(−0.5, +0.25, −1.0) = +0.25,
    root = max(root's own, +1.0, +0.25) = +1.0. Visits count every backup that
    passed through a node.
    """
    tree = _Tree(8)
    root = tree.add(None, -1)
    tree.value[root] = -1.0
    a = tree.add(None, root)
    b = tree.add(None, root)
    b1 = tree.add(None, b)
    b2 = tree.add(None, b)

    _backup(tree, a, 1.0)  # terminal-solved
    _backup(tree, b, -0.5)  # value-estimated
    _backup(tree, b1, 0.25)  # value-estimated
    _backup(tree, b2, -1.0)  # terminal, too large

    assert tree.value[b1] == pytest.approx(0.25)
    assert tree.value[b2] == pytest.approx(-1.0)
    assert tree.value[b] == pytest.approx(0.25), "B must be the max under B, not the mean"
    assert tree.value[root] == pytest.approx(1.0), "root must be the max, and unnegated"

    assert tree.visits[a] == 1
    assert tree.visits[b1] == 1 and tree.visits[b2] == 1
    assert tree.visits[b] == 3, "B is on the path of its own backup plus both children"
    assert tree.visits[root] == 4

    # The negation a blind port would introduce would have produced this:
    assert tree.value[root] != pytest.approx(-1.0)
    # ...and the mean a different blind port would have produced, this:
    assert tree.value[root] != pytest.approx(np.mean([1.0, -0.5, 0.25, -1.0]))


def test_backup_never_lowers_a_value() -> None:
    """Max backup is monotone: a later worse line cannot demote an earlier good one."""
    tree = _Tree(4)
    root = tree.add(None, -1)
    child = tree.add(None, root)
    _backup(tree, child, 1.0)
    _backup(tree, child, -1.0)
    assert tree.value[root] == pytest.approx(1.0)
    assert tree.visits[root] == 2


# ---------------------------------------------------------------------------
# The depth-1 gate, at the arithmetic-verified (sims, m)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_the_gate_arithmetic_is_recorded_and_the_gate_matches_it() -> None:
    """**Chunk 7 gate.** 100% on depth 1 at the (sims, m) the arithmetic supports.

    Measured on the real suite, not chunk 2's stand-in samplers: B_max = 5, so
    a uniform stub considers the winning action with certainty only at m >= 5.
    At m = 3 the worst-case per-problem probability is 3/5 and P(all 200) is
    4e-45 — a 100% gate there would be a gate the arithmetic cannot support.
    """
    record = json.loads((REPO / "runs" / "gate_arithmetic.json").read_text())
    b_max = record["B_max"]
    assert b_max == 5
    assert record["by_goal"]["SOLVE"]["min"] == 5
    assert record["by_goal"]["EVALUATE"]["max"] == 1

    rows = read_suite(SUITES / "solve_in_1.jsonl")
    evaluator = uniform_stub(CFG)
    solved = 0
    for i, row in enumerate(rows):
        problem = suite_problem(row)
        result = search(
            problem, problem.expr, evaluator, CFG, random.Random(1000 + i), sims=16, m=b_max
        )
        if result.values.size and result.values.max() >= 1.0:
            solved += 1
    assert solved == len(rows), f"depth-1 gate: {solved}/{len(rows)} at m={b_max}"


@pytest.mark.skipif(not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated")
def test_below_b_max_the_gate_is_not_reachable() -> None:
    """The other polarity: the arithmetic predicts a miss, and there is one.

    Without this, "100% at m=5" would be consistent with the gate being trivially
    satisfiable at any m — which would mean the arithmetic decided nothing.
    """
    rows = read_suite(SUITES / "solve_in_1.jsonl")
    evaluator = uniform_stub(CFG)
    solved = sum(
        1
        for i, row in enumerate(rows)
        if (p := suite_problem(row))
        and (
            r := search(p, p.expr, evaluator, CFG, random.Random(1000 + i), sims=16, m=3)
        ).values.size
        and r.values.max() >= 1.0
    )
    assert solved < len(rows), "m=3 solved everything — B_max is wrong or the stub is not uniform"


# ---------------------------------------------------------------------------
# Budget identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sims", [6, 16, 31, 48])
def test_budget_identity(sims: int) -> None:
    """A terminal short-circuit still consumes a simulation. Budget means budget."""
    evaluator = uniform_stub(CFG)
    result = search(PROBLEM, PROBLEM.expr, evaluator, CFG, random.Random(7), sims=sims, m=5)
    assert result.stats.sims_requested == sims
    assert result.stats.sims_used <= sims
    assert int(result.visits.sum()) == result.stats.sims_used


# ---------------------------------------------------------------------------
# Sync vs batched — the S2-class parity test, full grid, odd pairs included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [3, 5, 12, 16])
@pytest.mark.parametrize("sims", [6, 16, 31, 48])
def test_sync_and_batched_agree_on_visits(m: int, sims: int) -> None:
    """Batching changes *when* the evaluator is called, never what search does.

    Parametrised over the full m x sims grid from day one, odd pairs included —
    the odd-parity lesson is pre-paid in the plan, and this is the test that
    caught it in chess arriving with the port.
    """
    evaluator = uniform_stub(CFG)
    sync = search(
        PROBLEM, PROBLEM.expr, evaluator, CFG, random.Random(11), sims=sims, m=m, batch_leaves=1
    )
    batched = search(
        PROBLEM, PROBLEM.expr, evaluator, CFG, random.Random(11), sims=sims, m=m, batch_leaves=64
    )
    assert np.array_equal(sync.visits, batched.visits), f"visit parity failed at m={m} sims={sims}"
    assert np.allclose(sync.values, batched.values)
    assert sync.chosen == batched.chosen
    assert sync.stats.sims_used == batched.stats.sims_used
    assert sync.stats.evaluations == batched.stats.evaluations
    assert sync.stats.batches >= batched.stats.batches


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_seeds_give_identical_results_in_the_eval_profile() -> None:
    """The plan's 2-seed identity: with root noise off, the draw cannot matter."""
    cfg = Config()
    cfg.search.root_noise = False
    evaluator = uniform_stub(cfg)
    a = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(1), sims=16, m=5)
    b = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(999), sims=16, m=5)
    assert np.array_equal(a.visits, b.visits)
    assert a.chosen == b.chosen


def test_root_noise_actually_varies_the_search() -> None:
    """Both polarities: if the seed never mattered, the identity above is vacuous."""
    cfg = Config()
    assert cfg.search.root_noise is True
    evaluator = uniform_stub(cfg)
    seen = {
        tuple(search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(s), sims=16, m=3).visits)
        for s in range(12)
    }
    assert len(seen) > 1, "root noise changed nothing — the eval-profile test proves nothing"


def test_search_is_byte_identical_across_processes() -> None:
    """Cross-process under varied PYTHONHASHSEED — the construction-gate standard."""
    program = (
        "import random, json;"
        "from reckoner.config import Config;"
        "from reckoner.episode import Problem;"
        "from reckoner.expr import add, eq, mul, num, var;"
        "from reckoner.search import search, uniform_stub;"
        "from reckoner.vocab import GOAL_SOLVE, VAR_X;"
        "X=var(VAR_X); cfg=Config();"
        "p=Problem(goal=GOAL_SOLVE,target=VAR_X,par=3,par_source='bfs',"
        "expr=eq(add(mul(num(3),X),num(6)),num(21)));"
        "r=search(p,p.expr,uniform_stub(cfg),cfg,random.Random(5),sims=16,m=5);"
        "print(json.dumps([r.visits.tolist(), r.chosen, r.stats.sims_used]))"
    )
    outputs = set()
    for seed in ("0", "1", "7777", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.add(
            subprocess.run(
                [sys.executable, "-c", program], capture_output=True, text=True, check=True, env=env
            ).stdout.strip()
        )
    assert len(outputs) == 1, f"search varied across PYTHONHASHSEED: {outputs}"


# ---------------------------------------------------------------------------
# StateTooLarge is a counted terminal loss
# ---------------------------------------------------------------------------


def test_an_over_long_state_is_a_counted_terminal_loss_not_a_crash() -> None:
    """Step-cap-equivalent semantics: the budget cannot represent where it went.

    Never a crash mid-batch, and never an edit to the legality mask — legality
    stays the engine's alone. add_both_sides makes this reachable from the first
    stub rollout (FINDINGS.md F-05), so the counter is in SearchStats from
    field one.
    """
    cfg = Config()
    cfg.model.seq_len = 26  # the start state fits; one add_both_sides does not
    evaluator = uniform_stub(cfg)
    result = search(PROBLEM, PROBLEM.expr, evaluator, cfg, random.Random(3), sims=24, m=5)

    assert result.stats.state_too_large > 0, "the over-long case was never reached"
    assert result.stats.sims_used > 0
    assert np.isfinite(result.values).all()
    assert result.values.min() >= -1.0
    # It is a loss, not a discard: the budget was spent and the visits recorded.
    assert int(result.visits.sum()) == result.stats.sims_used


def test_search_stats_name_every_terminal_kind() -> None:
    stats = search(
        PROBLEM, PROBLEM.expr, uniform_stub(CFG), CFG, random.Random(1), sims=16, m=5
    ).stats
    for field_name in ("terminal_solved", "terminal_no_actions", "state_too_large"):
        assert field_name in stats.as_dict()
    assert stats.as_dict()["sims_requested"] == 16


def test_a_root_with_no_legal_action_returns_cleanly() -> None:
    """Empty action set, not "solved" — the two differ and only one is here.

    A *solved* equation still has legal rewrites: `x = 5` is goal form, and the
    both-sides rules remain legal on it. Whether to stop there is the episode's
    decision, not search's — search's contract is that its root is non-terminal.
    The genuinely empty case is a SIMPLIFY normal form.
    """
    from reckoner.rules import legal_actions
    from reckoner.vocab import GOAL_SIMPLIFY

    assert legal_actions(eq(X, num(5))), "a solved equation still has legal rewrites"

    stuck = Problem(goal=GOAL_SIMPLIFY, par=1, par_source="bfs", expr=X)
    assert legal_actions(stuck.expr) == []
    result = search(stuck, stuck.expr, uniform_stub(CFG), CFG, random.Random(1), sims=16, m=5)
    assert result.chosen is None and result.actions == []
    assert result.stats.terminal_no_actions == 1
    assert result.stats.sims_used == 0, "an empty root spends nothing"
