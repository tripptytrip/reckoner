"""Episodes, the checker, and the win-condition law.

The checker is the arbiter the whole method rests on, so it is tested against
claims it did not produce: a hand-built adversarial fixture set of answers that
are *nearly* right, and a thousand solutions from a scripted solver that it must
accept. Both directions, as always — a checker that rejects everything passes
the near-miss suite and is worthless.
"""

from __future__ import annotations

import random

import pytest

from reckoner.config import Config
from reckoner.episode import (
    TERMINAL_CONCEDED,
    TERMINAL_NO_ACTIONS,
    TERMINAL_SOLVED,
    TERMINAL_STEP_CAP,
    Episode,
    EpisodeResult,
    Problem,
    decode_state,
    describe_goal,
    encode_state,
    is_goal_form,
    outcome_z,
    verify,
)
from reckoner.expr import Expr, Num, Op, add, canonicalize, div, eq, mul, num, sub, tokens, var
from reckoner.rules import RULE_BY_NAME, RULESET_VERSION, legal_actions
from reckoner.vocab import (
    ADD,
    GOAL_EVALUATE,
    GOAL_SIMPLIFY,
    GOAL_SOLVE,
    SEP,
    VAR_X,
    VAR_Y,
    VOCAB_VERSION,
)

X = var(VAR_X)
Y = var(VAR_Y)
CFG = Config()


def rng() -> random.Random:
    return random.Random(1234)


def solve_problem(expr: Expr, par: int, **kw) -> Problem:
    return Problem(goal=GOAL_SOLVE, expr=expr, par=par, target=VAR_X, **kw)


# ---------------------------------------------------------------------------
# Goal tokens live in the state
# ---------------------------------------------------------------------------


def test_goal_prefix_encoding() -> None:
    """Plan §8 decision 3: prefix tokens in the state, no architecture change."""
    expr = eq(add(mul(num(3), X), num(6)), num(21))
    assert encode_state(GOAL_SOLVE, expr, VAR_X)[:3] == (GOAL_SOLVE, VAR_X, SEP)
    assert encode_state(GOAL_EVALUATE, num(3))[:2] == (GOAL_EVALUATE, SEP)
    assert encode_state(GOAL_SIMPLIFY, expr)[:2] == (GOAL_SIMPLIFY, SEP)


@pytest.mark.parametrize(
    ("goal", "expr", "target"),
    [
        (GOAL_SOLVE, eq(add(mul(num(3), X), num(6)), num(21)), VAR_X),
        (GOAL_EVALUATE, add(num(1), num(2)), None),
        (GOAL_SIMPLIFY, add(mul(num(3), X), mul(num(2), X)), None),
        (GOAL_SOLVE, eq(mul(num(2), Y), num(8)), VAR_Y),
    ],
)
def test_state_round_trip(goal: int, expr: Expr, target: int | None) -> None:
    assert decode_state(encode_state(goal, expr, target)) == (goal, target, expr)


@pytest.mark.parametrize(
    "seq",
    [(), (SEP,), (GOAL_SOLVE,), (GOAL_SOLVE, VAR_X), (VAR_X, SEP), (GOAL_SOLVE, VAR_X, VAR_X, SEP)],
)
def test_malformed_states_are_rejected(seq: tuple) -> None:
    with pytest.raises(ValueError):
        decode_state(seq)


# ---------------------------------------------------------------------------
# Problems validate their own preconditions
# ---------------------------------------------------------------------------


def test_evaluate_problem_must_be_ground() -> None:
    with pytest.raises(ValueError, match="must be ground"):
        Problem(goal=GOAL_EVALUATE, expr=add(X, num(1)), par=1)


def test_evaluate_problem_may_not_be_an_equation() -> None:
    with pytest.raises(ValueError, match="not a value"):
        Problem(goal=GOAL_EVALUATE, expr=eq(num(1), num(1)), par=1)


def test_solve_problem_needs_an_equation_and_a_present_target() -> None:
    with pytest.raises(ValueError, match="must be an equation"):
        solve_problem(add(X, num(1)), par=1)
    with pytest.raises(ValueError, match="does not occur"):
        Problem(goal=GOAL_SOLVE, expr=eq(mul(num(2), Y), num(8)), par=1, target=VAR_X)
    with pytest.raises(ValueError, match="needs a target"):
        Problem(goal=GOAL_SOLVE, expr=eq(X, num(1)), par=1)


def test_non_solve_problems_take_no_target() -> None:
    with pytest.raises(ValueError, match="takes no target"):
        Problem(goal=GOAL_SIMPLIFY, expr=X, par=1, target=VAR_X)


def test_a_problem_may_not_contain_div() -> None:
    """The v1 DIV-free invariant, enforced at the boundary that admits problems."""
    with pytest.raises(ValueError, match="may not contain DIV"):
        Problem(goal=GOAL_EVALUATE, expr=div(num(6), num(2)), par=1)


def test_a_problem_must_be_canonical() -> None:
    """An un-flattened ADD (C3) — legal as a tree, not legal as a state."""
    nested = Op(ADD, (Op(ADD, (num(1), num(2))), num(3)))
    assert canonicalize(nested) != nested
    with pytest.raises(ValueError, match="must be canonical"):
        Problem(goal=GOAL_SIMPLIFY, expr=nested, par=1)


def test_negative_par_is_rejected() -> None:
    with pytest.raises(ValueError, match="par must be non-negative"):
        solve_problem(eq(X, num(1)), par=-1)


# ---------------------------------------------------------------------------
# C7's payoff: the checker never asks which way round
# ---------------------------------------------------------------------------


def test_a_solved_equation_canonicalises_before_terminal_detection() -> None:
    """`<number> = x` is not a form the checker has to recognise — it cannot exist.

    C7 orders an EQ's operands variable-bearing side first, so an answer that
    *arrives* as `5 = x` is already `x = 5` by the time anything looks at it.
    This is the promise chunk 1 made, spent here: the checker has one shape to
    match, not two, and there is no orientation branch to get wrong.
    """
    arrived_backwards = eq(num(5), X)
    assert canonicalize(arrived_backwards) == eq(X, num(5))
    assert tokens(arrived_backwards) == tokens(eq(X, num(5)))

    problem = solve_problem(eq(mul(num(3), X), num(15)), par=1)
    assert is_goal_form(problem, arrived_backwards)
    assert verify(problem, arrived_backwards, CFG, rng())

    # And an episode that *reaches* it that way is solved without special-casing.
    episode = Episode(cfg=CFG, rng=rng())
    episode.reset(problem)
    episode.step((RULE_BY_NAME["div_both_sides"].rule_id, 2))
    assert episode.solved


def test_goal_form_requires_the_right_variable() -> None:
    problem = Problem(goal=GOAL_SOLVE, expr=eq(mul(num(2), Y), num(8)), par=1, target=VAR_Y)
    assert is_goal_form(problem, eq(Y, num(4)))
    assert not is_goal_form(problem, eq(X, num(4)))


# ---------------------------------------------------------------------------
# The adversarial near-miss fixtures — the checker must reject every one
# ---------------------------------------------------------------------------

NEAR_MISSES: list[tuple[str, Problem, Expr]] = [
    (
        "solve: off by one",
        solve_problem(eq(mul(num(3), X), num(15)), par=1),
        eq(X, num(6)),
    ),
    (
        "solve: sign flipped",
        solve_problem(eq(mul(num(3), X), num(15)), par=1),
        eq(X, num(-5)),
    ),
    (
        "solve: answer to the coefficient, not the equation",
        solve_problem(eq(mul(num(3), X), num(15)), par=1),
        eq(X, num(3)),
    ),
    (
        "solve: right answer to a different equation",
        solve_problem(eq(add(mul(num(3), X), num(6)), num(21)), par=3),
        eq(X, num(7)),  # solves 3x = 21, ignoring the +6
    ),
    (
        "solve: x on both sides, cancellation dropped",
        solve_problem(eq(add(mul(num(5), X), num(3)), add(mul(num(2), X), num(18))), par=5),
        eq(X, num(3)),  # would be right if the 2x were not there
    ),
    (
        "solve: correct value but not in goal form",
        solve_problem(eq(mul(num(3), X), num(15)), par=1),
        eq(mul(num(3), X), num(15)),
    ),
    (
        "solve: goal form but unevaluated answer",
        solve_problem(eq(mul(num(3), X), num(15)), par=1),
        eq(X, add(num(2), num(3))),
    ),
    (
        "evaluate: off by one",
        Problem(goal=GOAL_EVALUATE, expr=add(num(17), num(-25)), par=1),
        num(-7),
    ),
    (
        "evaluate: sign error",
        Problem(goal=GOAL_EVALUATE, expr=sub(num(4), num(9)), par=1),
        num(5),
    ),
    (
        "evaluate: not reduced to a numeral",
        Problem(goal=GOAL_EVALUATE, expr=add(num(1), num(2), num(3)), par=1),
        add(num(3), num(3)),
    ),
    (
        "simplify: coefficient off by one",
        Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(3), X), mul(num(2), X)), par=1),
        mul(num(6), X),
    ),
    (
        "simplify: a term silently dropped",
        Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(3), X), num(4)), par=0),
        mul(num(3), X),
    ),
    (
        "simplify: agrees at 0 and 1 but is a different function",
        Problem(goal=GOAL_SIMPLIFY, expr=mul(X, X), par=0),
        X,
    ),
    (
        "simplify: right shape, wrong variable",
        Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(3), X), mul(num(2), X)), par=1),
        mul(num(5), Y),
    ),
]


@pytest.mark.parametrize(
    ("label", "problem", "claim"), NEAR_MISSES, ids=[m[0] for m in NEAR_MISSES]
)
def test_checker_rejects_every_near_miss(label: str, problem: Problem, claim: Expr) -> None:
    assert not verify(problem, claim, CFG, rng()), f"checker accepted a wrong answer: {label}"


def test_the_near_miss_suite_is_not_vacuous() -> None:
    """Both polarities: each fixture's *correct* answer must be accepted.

    Without this, a checker that returned False unconditionally would pass every
    test above — which is the exact shape of a guard that is green forever.
    """
    corrections: list[tuple[Problem, Expr]] = [
        (solve_problem(eq(mul(num(3), X), num(15)), par=1), eq(X, num(5))),
        (solve_problem(eq(add(mul(num(3), X), num(6)), num(21)), par=3), eq(X, num(5))),
        (
            solve_problem(eq(add(mul(num(5), X), num(3)), add(mul(num(2), X), num(18))), par=5),
            eq(X, num(5)),
        ),
        (Problem(goal=GOAL_EVALUATE, expr=add(num(17), num(-25)), par=1), num(-8)),
        (Problem(goal=GOAL_EVALUATE, expr=sub(num(4), num(9)), par=1), num(-5)),
        (
            Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(3), X), mul(num(2), X)), par=1),
            mul(num(5), X),
        ),
        (Problem(goal=GOAL_SIMPLIFY, expr=mul(X, X), par=0), mul(X, X)),
    ]
    for problem, claim in corrections:
        assert verify(problem, claim, CFG, rng()), (
            f"checker rejected a right answer: {tokens(claim)}"
        )


def test_simplify_rejects_a_near_function_with_high_probability() -> None:
    """k=32 draws over a 2^31-sized field. A one-point agreement will not survive.

    `x·x` and `x` agree at 0 and 1 and nowhere else — a checker sampling small
    integers could plausibly be fooled; one sampling the whole field cannot.
    """
    problem = Problem(goal=GOAL_SIMPLIFY, expr=mul(X, X), par=0)
    for seed in range(50):
        assert not verify(problem, X, CFG, random.Random(seed))


# ---------------------------------------------------------------------------
# A scripted solver, and 1K solutions the checker must accept
# ---------------------------------------------------------------------------


def greedy_solve(problem: Problem, cap: int = 24) -> list[tuple[int, int]] | None:
    """A deterministic solver for the fixture set. **Not chunk 5's deliverable.**

    Preference order is the human one: finish if you can, then compute, then
    collect, then move a constant across. Good enough to solve every linear
    problem this test generates; it is a fixture generator, not the scripted
    solver chunk 5 owns.
    """
    order = ["div_both_sides", "eval_add", "eval_sub", "eval_mul", "combine_like_terms"]
    preferred = [RULE_BY_NAME[name].rule_id for name in order]
    episode = Episode(cfg=CFG, rng=random.Random(0))
    episode.reset(problem)
    path: list[tuple[int, int]] = []
    while not episode.done and len(path) < cap:
        actions = episode.legal()
        choice = next((a for rid in preferred for a in actions if a[0] == rid), None)
        if choice is None:
            # Nothing to compute: move a numeric addend across to the other side.
            from reckoner.rules import enumerate_sites

            sub_id = RULE_BY_NAME["sub_both_sides"].rule_id
            sites = enumerate_sites(episode.expr)  # type: ignore[arg-type]
            choice = next(
                (
                    a
                    for a in actions
                    if a[0] == sub_id
                    and isinstance(sites[a[1]].node, Num)
                    and len(sites[a[1]].path) == 2
                ),
                None,
            )
        if choice is None:
            choice = next(
                (a for a in actions if a[0] == RULE_BY_NAME["sub_both_sides"].rule_id), None
            )
        if choice is None:
            return None
        episode.step(choice)
        path.append(choice)
    return path if episode.solved else None


def random_solvable_problem(rng_: random.Random) -> Problem:
    kind = rng_.choice(("solve", "solve", "evaluate", "simplify"))
    if kind == "evaluate":
        expr = add(num(rng_.randrange(-40, 41)), num(rng_.randrange(-40, 41)))
        if rng_.random() < 0.4:
            expr = mul(num(rng_.randrange(-9, 10)), num(rng_.randrange(-9, 10)))
        return Problem(goal=GOAL_EVALUATE, expr=expr, par=1)
    if kind == "simplify":
        a, b = rng_.randrange(-9, 10), rng_.randrange(-9, 10)
        return Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(a), X), mul(num(b), X)), par=1)
    a = rng_.choice([c for c in range(-9, 10) if c != 0])
    answer = rng_.randrange(-12, 13)
    b = rng_.randrange(-20, 21)
    return solve_problem(eq(add(mul(num(a), X), num(b)), num(a * answer + b)), par=3)


def test_checker_accepts_1000_scripted_solutions() -> None:
    """**Chunk 3 gate.** The accepting polarity, at scale."""
    rng_ = random.Random(31337)
    accepted = 0
    by_goal: dict[int, int] = {}
    attempts = 0
    while accepted < 1000 and attempts < 4000:
        attempts += 1
        problem = random_solvable_problem(rng_)
        path = greedy_solve(problem)
        if path is None:
            continue
        episode = Episode(cfg=CFG, rng=random.Random(7))
        episode.reset(problem)
        for action in path:
            episode.step(action)
        assert episode.solved, f"solver produced a path the checker rejects: {tokens(problem.expr)}"
        assert verify(problem, episode.expr, CFG, rng())  # type: ignore[arg-type]
        accepted += 1
        by_goal[problem.goal] = by_goal.get(problem.goal, 0) + 1

    assert accepted == 1000, f"only {accepted} solutions accepted in {attempts} attempts"
    assert len(by_goal) == 3, f"only goals {sorted(by_goal)} exercised — one is untested"
    assert min(by_goal.values()) >= 100, f"a goal is under-covered: {by_goal}"
    print(f"\n  scripted solutions accepted: {accepted}, by goal {by_goal}")


# ---------------------------------------------------------------------------
# The win-condition law, cap edges included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("solved", "steps", "par", "z"),
    [
        (True, 2, 3, 1),  # strictly under par: beat
        (True, 3, 3, 0),  # equal: draw
        (True, 4, 3, -1),  # over par: loss
        (False, 24, 3, -1),  # timed out: loss
        (False, 0, 0, -1),  # dead end: loss
        (True, 0, 0, 0),  # par 0, solved immediately: draw
        (True, 0, 1, 1),  # already solved and par was 1: beat
    ],
)
def test_the_win_condition_law(solved: bool, steps: int, par: int, z: int) -> None:
    """beat = strictly fewer; draw = equal; loss = more or not solved."""
    assert outcome_z(solved=solved, steps=steps, par=par) == z


def test_cap_edge_solved_on_the_capping_step_is_solved() -> None:
    """The cap bounds steps taken, not steps credited.

    `3x + 6 = 21` is par 3 and takes 3 steps. With the cap set to exactly 3, the
    third step both solves the problem and reaches the cap — and it must be
    scored as a solve, not a timeout.
    """
    cfg = Config()
    cfg.episode.step_cap = 3
    problem = solve_problem(eq(add(mul(num(3), X), num(6)), num(21)), par=3)
    episode = Episode(cfg=cfg, rng=rng())
    episode.reset(problem)
    for action in greedy_solve(problem):  # type: ignore[union-attr]
        episode.step(action)

    result = episode.result()
    assert result.steps == 3
    assert result.solved
    assert result.terminal_reason == TERMINAL_SOLVED
    assert result.z == 0  # steps == par


def test_cap_edge_solved_at_the_cap_but_under_par_is_a_win() -> None:
    """Par can exceed the cap. Beating it at the cap is still beating it."""
    cfg = Config()
    cfg.episode.step_cap = 3
    problem = solve_problem(eq(add(mul(num(3), X), num(6)), num(21)), par=5)
    episode = Episode(cfg=cfg, rng=rng())
    episode.reset(problem)
    for action in greedy_solve(problem):  # type: ignore[union-attr]
        episode.step(action)
    result = episode.result()
    assert result.solved and result.steps == 3 and result.z == 1


def test_cap_edge_unsolved_at_the_cap_is_a_loss() -> None:
    cfg = Config()
    cfg.episode.step_cap = 2
    problem = solve_problem(eq(add(mul(num(3), X), num(6)), num(21)), par=3)
    episode = Episode(cfg=cfg, rng=rng())
    episode.reset(problem)
    add_id = RULE_BY_NAME["add_both_sides"].rule_id
    while not episode.done:
        episode.step(next(a for a in episode.legal() if a[0] == add_id))
    result = episode.result()
    assert not result.solved
    assert result.steps == 2
    assert result.terminal_reason == TERMINAL_STEP_CAP
    assert result.z == -1


# ---------------------------------------------------------------------------
# Episode invariants
# ---------------------------------------------------------------------------


def test_result_carries_ruleset_version_first() -> None:
    """Par is denominated in a rule system; the version is part of the label."""
    from dataclasses import fields as dataclass_fields

    assert [f.name for f in dataclass_fields(EpisodeResult)][0] == "ruleset_version"
    problem = solve_problem(eq(mul(num(3), X), num(15)), par=1)
    episode = Episode(cfg=CFG, rng=rng())
    episode.reset(problem)
    episode.step((RULE_BY_NAME["div_both_sides"].rule_id, 2))
    result = episode.result()
    assert result.ruleset_version == RULESET_VERSION
    assert result.vocab_version == VOCAB_VERSION
    assert result.par_source == "bfs"


def test_no_action_is_legal_after_terminal() -> None:
    problem = solve_problem(eq(mul(num(3), X), num(15)), par=1)
    episode = Episode(cfg=CFG, rng=rng())
    episode.reset(problem)
    assert episode.legal()
    episode.step((RULE_BY_NAME["div_both_sides"].rule_id, 2))
    assert episode.done
    assert episode.legal() == []
    # ...and the underlying state still has rewrites available, so this is the
    # episode refusing them, not the rule engine running out.
    assert legal_actions(episode.expr)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="episode is over"):
        episode.step((RULE_BY_NAME["add_both_sides"].rule_id, 1))


def test_illegal_action_is_rejected() -> None:
    problem = solve_problem(eq(mul(num(3), X), num(15)), par=1)
    episode = Episode(cfg=CFG, rng=rng())
    episode.reset(problem)
    with pytest.raises(ValueError, match="illegal action"):
        episode.step((RULE_BY_NAME["eval_add"].rule_id, 0))


def test_result_before_terminal_is_refused() -> None:
    """A result for a running episode would be a guess with a number attached."""
    episode = Episode(cfg=CFG, rng=rng())
    episode.reset(solve_problem(eq(add(mul(num(3), X), num(6)), num(21)), par=3))
    with pytest.raises(ValueError, match="still running"):
        episode.result()


def test_using_an_episode_before_reset_is_refused() -> None:
    episode = Episode(cfg=CFG, rng=rng())
    with pytest.raises(ValueError, match="call reset"):
        episode.legal()


def test_an_already_solved_problem_is_terminal_at_zero_steps() -> None:
    problem = solve_problem(eq(X, num(5)), par=0)
    episode = Episode(cfg=CFG, rng=rng())
    episode.reset(problem)
    assert episode.done and episode.solved
    assert episode.result().steps == 0
    assert episode.result().z == 0


def test_concede_is_implemented_and_off_by_default() -> None:
    """[v1.1] The resign-vs-par analog. Off by default; k is uncalibrated."""
    assert Config().par.concede_enabled is False

    cfg = Config()
    cfg.par.concede_enabled = True
    cfg.par.concede_k = 1
    problem = solve_problem(eq(add(mul(num(3), X), num(6)), num(21)), par=1)
    episode = Episode(cfg=cfg, rng=rng())
    episode.reset(problem)
    add_id = RULE_BY_NAME["add_both_sides"].rule_id
    while not episode.done:
        episode.step(next(a for a in episode.legal() if a[0] == add_id))
    result = episode.result()
    assert result.terminal_reason == TERMINAL_CONCEDED
    assert result.steps == 2  # par 1 + k 1
    assert result.z == -1


def test_step_count_and_cap_invariants_fuzzed() -> None:
    """Random legal play: the step count tracks the steps, and the cap holds."""
    rng_ = random.Random(4242)
    reasons: dict[str, int] = {}
    for _ in range(400):
        cfg = Config()
        cfg.episode.step_cap = rng_.randint(1, 8)
        problem = random_solvable_problem(rng_)
        episode = Episode(cfg=cfg, rng=random.Random(rng_.randrange(10**6)))
        episode.reset(problem)
        taken = 0
        while not episode.done:
            actions = episode.legal()
            assert actions, "not done, yet no legal action — _settle missed a dead end"
            episode.step(rng_.choice(actions))
            taken += 1
            assert episode.steps == taken
            assert episode.steps <= cfg.episode.step_cap
        result = episode.result()
        assert result.steps == taken <= cfg.episode.step_cap
        assert episode.legal() == []
        assert result.z in (-1, 0, 1)
        assert (result.z >= 0) == (result.solved and result.steps <= result.par)
        reasons[result.terminal_reason] = reasons.get(result.terminal_reason, 0) + 1

    assert set(reasons) >= {TERMINAL_SOLVED, TERMINAL_STEP_CAP}, (
        f"the fuzz never reached one of the terminal kinds: {reasons}"
    )
    print(f"\n  episode fuzz terminal reasons: {reasons}")


def test_terminal_reasons_are_distinguishable() -> None:
    """Absence carries a reason: an unsolved episode says which wall it hit."""
    assert len({TERMINAL_SOLVED, TERMINAL_STEP_CAP, TERMINAL_NO_ACTIONS, TERMINAL_CONCEDED}) == 4


def test_describe_goal() -> None:
    assert describe_goal(solve_problem(eq(X, num(1)), par=0)) == "SOLVE for x"
    assert describe_goal(Problem(goal=GOAL_EVALUATE, expr=num(1), par=0)) == "EVALUATE"
    assert describe_goal(Problem(goal=GOAL_SIMPLIFY, expr=X, par=0)) == "SIMPLIFY"
