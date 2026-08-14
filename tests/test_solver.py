"""The scripted solver: a provisional floor, and how far above the floor sits."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from reckoner.config import Config
from reckoner.episode import Episode, Problem, bfs_par
from reckoner.expr import add, eq, mul, num, sub, var
from reckoner.rules import RULE_BY_NAME
from reckoner.solver import POLICY, scripted_par, scripted_par_delta, scripted_solve
from reckoner.vocab import GOAL_EVALUATE, GOAL_SIMPLIFY, GOAL_SOLVE, VAR_X

REPO = Path(__file__).resolve().parents[1]
CFG = Config()
X = var(VAR_X)


def test_add_both_sides_is_never_in_the_policy() -> None:
    """It only grows the state, so a policy that reaches for it does not terminate.

    Chunk 4's fixture script proved this the expensive way: six runaway
    derivations from a policy that merely *preferred* it.
    """
    assert "add_both_sides" not in POLICY
    assert set(POLICY) | {"add_both_sides"} == set(RULE_BY_NAME)


def test_f01s_own_case_is_now_solved_optimally() -> None:
    """FINDINGS.md F-01: preferring eval_add over eval_mul cost a step.

    The ordering in POLICY is that finding, applied. This is the regression.
    """
    problem = Problem(
        goal=GOAL_EVALUATE, expr=add(mul(num(12), num(12)), sub(num(9), num(30)), num(7))
    )
    assert bfs_par(problem, CFG) == 3
    assert scripted_par(problem, CFG) == 3
    assert scripted_par_delta(problem, CFG) == 0


@pytest.mark.parametrize(
    ("problem", "par"),
    [
        (Problem(goal=GOAL_SOLVE, target=VAR_X, expr=eq(mul(num(3), X), num(15))), 1),
        (
            Problem(goal=GOAL_SOLVE, target=VAR_X, expr=eq(add(mul(num(3), X), num(6)), num(21))),
            3,
        ),
        (
            Problem(
                goal=GOAL_SOLVE,
                target=VAR_X,
                expr=eq(add(mul(num(5), X), num(3)), add(mul(num(2), X), num(18))),
            ),
            5,
        ),
        (Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(3), X), mul(num(2), X))), 1),
    ],
)
def test_scripted_solve_reaches_par_on_the_worked_cases(problem: Problem, par: int) -> None:
    path = scripted_solve(problem, CFG)
    assert path is not None
    assert len(path) == par


def test_a_scripted_derivation_is_accepted_by_the_checker() -> None:
    """The solver runs a real Episode, so "solved" means what the checker means."""
    rng = random.Random(17)
    checked = 0
    for _ in range(120):
        a = rng.choice([c for c in range(-9, 10) if c != 0])
        k, b = rng.randrange(-12, 13), rng.randrange(-20, 21)
        raw = Problem(
            goal=GOAL_SOLVE, target=VAR_X, expr=eq(add(mul(num(a), X), num(b)), num(a * k + b))
        )
        path = scripted_solve(raw, CFG)
        assert path is not None
        exact = bfs_par(raw, CFG)
        assert exact is not None
        problem = Problem(
            goal=raw.goal, expr=raw.expr, target=raw.target, par=exact, par_source="bfs"
        )
        episode = Episode(cfg=CFG, rng=random.Random(0))
        episode.reset(problem)
        for action in path:
            episode.step(action)
        assert episode.solved
        assert episode.result().z <= 0  # cannot beat an exact par
        checked += 1
    assert checked == 120


def test_a_scripted_par_can_never_beat_an_exact_one() -> None:
    """Negative delta is impossible; if it happened the BFS label would be wrong."""
    rng = random.Random(31)
    deltas = []
    for _ in range(80):
        a = rng.choice([c for c in range(-9, 10) if c != 0])
        problem = Problem(
            goal=GOAL_SIMPLIFY,
            expr=add(
                mul(num(a), X), mul(num(rng.randrange(-9, 10) or 2), X), num(rng.randrange(-9, 10))
            ),
        )
        delta = scripted_par_delta(problem, CFG)
        if delta is not None:
            deltas.append(delta)
    assert deltas and min(deltas) >= 0


def test_scripted_solve_gives_up_rather_than_looping() -> None:
    """A cap is not a policy, but an unbounded policy is not a solver."""
    problem = Problem(
        goal=GOAL_SOLVE,
        target=VAR_X,
        expr=eq(add(mul(num(5), X), num(3)), add(mul(num(2), X), num(18))),
    )
    assert scripted_solve(problem, CFG, cap=2) is None
    assert scripted_solve(problem, CFG, cap=5) is not None


@pytest.mark.skipif(
    not (REPO / "runs" / "par_delta.json").exists(), reason="run scripts/par_delta.py"
)
def test_the_measured_floor_is_recorded_and_tight() -> None:
    """The calibration spec §3's "provisional floor" language is paying for.

    Measured over all 1,200 suite problems: the floor is optimal 97.7% of the
    time and never more than one step above the true minimum — on the depth <= 6
    band where BFS exists. Above that band there is no evidence at all, which is
    exactly why par_from_pool_frac exists.
    """
    record = json.loads((REPO / "runs" / "par_delta.json").read_text())
    total = sum(sum(c.values()) for c in record["by_depth"].values())
    optimal = sum(c.get("0", 0) for c in record["by_depth"].values())
    worst = max(int(d) for c in record["by_depth"].values() for d in c)
    assert total == 1200, f"only {total} problems measured"
    assert optimal / total >= 0.9, f"scripted par is optimal on only {optimal}/{total}"
    assert worst <= 1, f"scripted par is {worst} steps above exact somewhere"
    assert not record["unsolved"], f"the scripted solver failed on {record['unsolved']}"


@pytest.mark.skipif(
    not (REPO / "runs" / "rule_participation.json").exists(),
    reason="run scripts/rule_participation.py",
)
def test_every_rule_but_add_both_sides_is_live() -> None:
    """Rule liveness. A green soundness fuzz proves a rule is correct, not used.

    `eval_sub` is the one this exists for: no rule constructs a SUB, so it is
    live only because the generator emits numeric SUB deliberately.
    """
    record = json.loads((REPO / "runs" / "rule_participation.json").read_text())
    per_rule = record["per_rule"]
    assert per_rule.get("eval_sub", 0) > 0, "eval_sub is dead — the SUB emission stopped"
    for name in ("eval_add", "eval_mul", "combine_like_terms", "sub_both_sides", "div_both_sides"):
        assert per_rule.get(name, 0) > 0, f"{name} appears in no optimal derivation"
    assert per_rule.get("add_both_sides", 0) == 0, (
        "add_both_sides appeared in an optimal derivation — ROUND-01's claim that "
        "it is reachability-redundant is false and the round must be revised"
    )
