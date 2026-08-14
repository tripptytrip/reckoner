"""The emission grammar, and the labels the generator is allowed to attach.

Every ban here names a construct the rule set cannot reduce. A problem that
violates one is unsolvable *by construction*, and a suite of unsolvable problems
is indistinguishable from a suite of hard ones — which is why these are checked
before labelling rather than discovered by a BFS that returns None.
"""

from __future__ import annotations

import random

import pytest

from reckoner.config import Config
from reckoner.dataset import ShippingError, check_shippable
from reckoner.episode import Problem, bfs_par
from reckoner.expr import add, div, eq, mul, num, sub, var
from reckoner.generator import (
    CHEAP_TEMPLATES,
    COSTLY_TEMPLATES,
    TEMPLATES,
    EmissionError,
    check_emission,
    emit,
    generate,
    has_variable_sub,
    label,
    numerals,
    variable_tokens,
)
from reckoner.semantics import variables
from reckoner.vocab import DIV, GOAL_EVALUATE, GOAL_SIMPLIFY, GOAL_SOLVE, SUB, VAR_X, VAR_Y

X = var(VAR_X)
Y = var(VAR_Y)
CFG = Config()


# ---------------------------------------------------------------------------
# The three bans
# ---------------------------------------------------------------------------


def test_div_is_refused() -> None:
    """Refused at Problem construction already; restated here as an emission rule."""
    with pytest.raises(ValueError, match="may not contain DIV"):
        Problem(goal=GOAL_EVALUATE, expr=div(num(6), num(2)))


@pytest.mark.parametrize(
    "expr",
    [
        sub(num(21), mul(num(2), X)),  # 21 − 2x
        sub(mul(num(3), X), X),  # 3x − x
        add(num(1), sub(num(5), X)),  # buried one level down
        add(num(1), mul(num(2), sub(num(5), X))),  # buried two levels down
    ],
)
def test_variable_bearing_sub_is_refused(expr) -> None:
    """`eval_sub` needs numeric operands; the movers ignore SUB; combine cannot
    see through one. So a variable under a SUB is a dead end exactly like DIV."""
    assert has_variable_sub(expr)
    with pytest.raises(EmissionError, match="dead end"):
        check_emission(Problem(goal=GOAL_SIMPLIFY, expr=expr))


@pytest.mark.parametrize("expr", [sub(num(21), num(6)), add(sub(num(4), num(9)), num(2))])
def test_numeric_only_sub_is_allowed(expr) -> None:
    """The other polarity — and it must be *emitted*, not merely allowed."""
    assert not has_variable_sub(expr)
    check_emission(Problem(goal=GOAL_EVALUATE, expr=expr))


def test_a_sub_bearing_template_exists_and_fires() -> None:
    """Otherwise `eval_sub` is a dead rule whose soundness fuzz passes in a vacuum.

    No rule *constructs* a SUB. If the generator never emits one either, nothing
    in the whole system ever exercises eval_sub on real data.
    """
    rng = random.Random(5)
    with_sub = 0
    for name in TEMPLATES:
        for _ in range(20):
            if SUB in _tokens(emit(name, rng)):
                with_sub += 1
    assert with_sub >= 20, f"only {with_sub} emitted candidates contain a SUB node"


def _tokens(problem: Problem):
    from reckoner.expr import tokens

    return tokens(problem.expr)


# ---------------------------------------------------------------------------
# Variable policy, per goal
# ---------------------------------------------------------------------------


def test_evaluate_must_be_ground() -> None:
    with pytest.raises(ValueError, match="must be ground"):
        Problem(goal=GOAL_EVALUATE, expr=add(X, num(1)))


def test_solve_admits_exactly_one_variable() -> None:
    """A second unknown is outside v1: nothing in the rule set eliminates one."""
    two_unknowns = Problem(goal=GOAL_SOLVE, expr=eq(add(mul(num(2), X), Y), num(7)), target=VAR_X)
    with pytest.raises(EmissionError, match="exactly one variable"):
        check_emission(two_unknowns)
    # ...and it really is unsolvable, not merely disallowed:
    assert bfs_par(two_unknowns, CFG) is None


def test_simplify_admits_two_variables() -> None:
    """The only goal that exercises combine_like_terms' unlike-variable case."""
    problem = Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(3), X), mul(num(2), Y)))
    check_emission(problem)
    with pytest.raises(EmissionError, match="at most two variables"):
        check_emission(
            Problem(
                goal=GOAL_SIMPLIFY,
                expr=add(mul(num(3), X), mul(num(2), Y), mul(num(4), var(13))),
            )
        )


def test_a_two_variable_simplify_is_actually_generated() -> None:
    rng = random.Random(9)
    seen = {len(variable_tokens(emit("simplify_two_vars", rng).expr)) for _ in range(20)}
    assert 2 in seen, "the unlike-variable case is never emitted"


# ---------------------------------------------------------------------------
# Every template obeys the grammar, and its label is measured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_emits_legally(name: str) -> None:
    rng = random.Random(hash(name) % 10_000)
    for _ in range(25):
        problem = emit(name, rng)  # raises if the grammar is violated
        assert DIV not in _tokens(problem)
        assert not has_variable_sub(problem.expr)
        if problem.goal == GOAL_SOLVE:
            assert variables(problem.expr) == (problem.target,)
        if problem.goal == GOAL_EVALUATE:
            assert not variables(problem.expr)


@pytest.mark.parametrize("name", sorted(CHEAP_TEMPLATES))
def test_cheap_templates_label_within_the_horizon(name: str) -> None:
    rng = random.Random(4242)
    labelled = [generate(name, rng, CFG) for _ in range(12)]
    assert all(p is not None for p in labelled), f"{name} produced an unlabelable candidate"
    assert all(p.par is not None and p.par >= 1 for p in labelled)  # type: ignore[union-attr]
    assert all(p.par_source == "bfs" for p in labelled)  # type: ignore[union-attr]


def test_the_cheap_costly_split_is_real_and_not_decoration() -> None:
    """The split drives the generation plan, so it must reflect the state space.

    EQ-rooted states have both both-sides rules firing at every addend, so their
    branching is 6-11 where an expression's is 2. That is the whole reason
    labelling is parallel.
    """
    assert set(CHEAP_TEMPLATES) | set(COSTLY_TEMPLATES) == set(TEMPLATES)
    assert not set(CHEAP_TEMPLATES) & set(COSTLY_TEMPLATES)
    rng = random.Random(1)
    for name in COSTLY_TEMPLATES:
        assert emit(name, rng).goal == GOAL_SOLVE
    for name in CHEAP_TEMPLATES:
        assert emit(name, rng).goal != GOAL_SOLVE


def test_label_refuses_a_degenerate_problem() -> None:
    """par 0 means already in goal form — a free draw, not a problem."""
    already = Problem(goal=GOAL_SOLVE, expr=eq(X, num(5)), target=VAR_X)
    assert bfs_par(already, CFG) == 0
    assert label(already, CFG) is None


def test_label_refuses_what_it_cannot_label_exactly() -> None:
    """No scripted fallback here: outside the BFS band is outside this generator."""
    hard = Problem(
        goal=GOAL_SOLVE,
        expr=eq(add(mul(num(7), X), num(3)), add(mul(num(2), X), num(18))),
        target=VAR_X,
    )
    assert label(hard, CFG, cap=1) is None


def test_labels_are_measured_not_assumed() -> None:
    """A shape does not know its own depth — the survey found both splits below."""
    rng = random.Random(20260814)
    depths: dict[str, set[int]] = {}
    for name in ("solve_two_terms", "eval_deepest"):
        found = set()
        for _ in range(30):
            problem = generate(name, rng, CFG)
            if problem is not None:
                found.add(problem.par)  # type: ignore[arg-type]
        depths[name] = found
    for name, found in depths.items():
        assert len(found) >= 2, f"{name} always lands on one depth: {found}"


def test_numerals_helper() -> None:
    assert sorted(numerals(add(mul(num(3), X), num(-6)))) == [-6, 3]


# ---------------------------------------------------------------------------
# The shipping boundary
# ---------------------------------------------------------------------------


def test_unlabelled_does_not_ship() -> None:
    with pytest.raises(ShippingError, match="absence does not ship"):
        check_shippable(Problem(goal=GOAL_SIMPLIFY, expr=X))


def test_unverified_does_not_ship() -> None:
    """The weakest honest state may exist; it may not leave."""
    with pytest.raises(ShippingError, match="does not ship"):
        check_shippable(Problem(goal=GOAL_SIMPLIFY, expr=X, par=3, par_source="unverified"))


def test_par_zero_does_not_ship() -> None:
    with pytest.raises(ShippingError, match="already in goal"):
        check_shippable(
            Problem(goal=GOAL_SOLVE, expr=eq(X, num(5)), target=VAR_X, par=0, par_source="bfs")
        )


def test_a_properly_labelled_problem_ships() -> None:
    """Both polarities: a boundary that rejects everything is not a boundary."""
    check_shippable(
        Problem(
            goal=GOAL_SOLVE, expr=eq(mul(num(3), X), num(15)), target=VAR_X, par=1, par_source="bfs"
        )
    )
