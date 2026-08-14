"""The movegen oracle.

Soundness here is **two claims, not one**, and the second is the one a field
fuzz cannot make:

  **Equivalence.** For a node rule, the rewritten subtree has the same value as
  the original under every assignment. For an equation rule, the rewritten
  equation has the same *truth value* — solution-set equivalence, not value
  equality, because the two sides of an ``EQ`` are not a number.

  **Closure.** The successor is a well-formed state in this vocabulary, and it
  round-trips through chunk 1's codec. Rule set v1 has no fractions, so a
  rewrite whose honest answer is 16/3 has nowhere to put it.

The separation matters because 𝔽ₚ is structurally blind to closure: every
nonzero constant is invertible mod p, so ``div_both_sides`` with its exactness
guard ripped out passes field-equivalence *perfectly*, forever. The ℚ layer sees
it, and the dedicated guard test states it directly. Both are here, and the
guard test's docstring says exactly what it uniquely carries.

Every fuzz below reports **fired** and **skipped** counts per rule and asserts a
floor on them. A rule that "passed on 10,000 assignments" having actually
evaluated twelve is the vacuity failure this instrumentation exists to expose.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from reckoner.expr import (
    Expr,
    Num,
    Op,
    Var,
    add,
    canonicalize,
    div,
    eq,
    identity_key,
    make_op,
    mul,
    num,
    parse,
    sub,
    tokens,
    var,
)
from reckoner.rules import (
    RULE_BY_NAME,
    RULE_SET_VERSION,
    RULES,
    Site,
    apply,
    enumerate_sites,
    legal_actions,
    negate,
    replace_at,
    rule_set_fingerprint,
    scaled,
    split_coefficient,
    successors,
)
from reckoner.semantics import eval_exact, eval_field, holds_exact, holds_field, variables
from reckoner.vocab import ADD, DIV, EQ, MUL, SUB, VAR_X, VAR_Y

P = 2_147_483_647  # episode.equiv_prime
X = var(VAR_X)
Y = var(VAR_Y)

PINNED_RULE_FINGERPRINT = "f9c61ba15e41f8d3448daddc5cd642217a30914dc830ae11ba4d88e45e93f38b"


# ---------------------------------------------------------------------------
# The rule set is a contract with every dataset that stores an action
# ---------------------------------------------------------------------------


def test_rule_set_v1_is_exactly_the_pinned_seven() -> None:
    """Plan §8 decision 2. Extensions are later one-lever rounds, not edits."""
    assert RULE_SET_VERSION == 1
    assert [r.name for r in RULES] == [
        "eval_add",
        "eval_sub",
        "eval_mul",
        "combine_like_terms",
        "add_both_sides",
        "sub_both_sides",
        "div_both_sides",
    ]
    assert [r.rule_id for r in RULES] == list(range(7))


def test_rule_fingerprint_is_pinned() -> None:
    assert rule_set_fingerprint() == PINNED_RULE_FINGERPRINT, (
        "The rule set changed. If deliberate: bump RULE_SET_VERSION and update "
        "this literal. Datasets store rule ids; renumbering relabels every "
        "recorded action in every dataset already written."
    )


def test_only_div_both_sides_carries_a_guard() -> None:
    """The plan names exactly one guard. A second one appearing silently is news."""
    guarded = [r.name for r in RULES if r.guard is not RULES[0].guard]
    assert guarded == ["div_both_sides"]


# ---------------------------------------------------------------------------
# Site enumeration — the order downstream pins depend on
# ---------------------------------------------------------------------------


def test_site_enumeration_is_preorder_over_the_canonical_tree() -> None:
    state = eq(add(mul(num(3), X), num(6)), num(21))
    sites = enumerate_sites(state)
    assert [s.path for s in sites] == [
        (),  # 0  EQ
        (0,),  # 1    ADD
        (0, 0),  # 2      MUL
        (0, 0, 0),  # 3        3
        (0, 0, 1),  # 4        x
        (0, 1),  # 5      6
        (1,),  # 6    21
    ]
    assert [s.site_id for s in sites] == list(range(7))
    assert sites[3].node == num(3)
    assert sites[4].node == X


def test_site_ids_are_not_stable_across_a_rewrite() -> None:
    """Documented, and pinned, because assuming otherwise is a silent bug.

    An action names a rule and a position *in the current state*. ``apply``
    re-canonicalises, and canonicalisation reorders children and collapses
    singletons, so a site id is not a durable handle on a subtree.
    """
    # (a) A collapse removes sites outright: C4 folds ADD(3x) down to 3x.
    collapsing = add(mul(num(3), X), num(6), num(-6))
    assert len(enumerate_sites(collapsing)) == 6
    assert enumerate_sites(collapsing)[3].node == X
    collapsed = apply(collapsing, RULE_BY_NAME["eval_add"].rule_id, 0)
    assert len(enumerate_sites(collapsed)) == 3  # site 3 no longer exists at all
    with pytest.raises(ValueError, match="does not exist"):
        apply(collapsed, RULE_BY_NAME["eval_add"].rule_id, 3)

    # (b) A rewrite reuses an id for a different node: site 6 was 21, now −21.
    state = eq(add(mul(num(3), X), num(6)), num(21))
    assert enumerate_sites(state)[6].node == num(21)
    moved = apply(state, RULE_BY_NAME["sub_both_sides"].rule_id, 6)
    assert enumerate_sites(moved)[6].node == num(-21)


def test_replace_at_rebuilds_and_recanonicalises() -> None:
    state = add(mul(num(3), X), num(6))
    # Replacing the coefficient with a bigger number must not disturb C5/C6.
    out = replace_at(state, (0, 0), num(9))
    assert out == add(mul(num(9), X), num(6))
    assert canonicalize(out) == out


def test_node_at_root_is_the_whole_state() -> None:
    state = eq(X, num(1))
    assert enumerate_sites(state)[0].node == state


# ---------------------------------------------------------------------------
# Term helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (num(7), (7, ())),
        (X, (1, (X,))),
        (mul(num(3), X), (3, (X,))),
        (mul(num(-3), X), (-3, (X,))),
        (sub(num(1), X), (1, (sub(num(1), X),))),
    ],
)
def test_split_coefficient(expr: Expr, expected: tuple) -> None:
    assert split_coefficient(expr) == expected


def test_scaled_applies_the_arithmetic_identities() -> None:
    """These identities are what make rule set v1 closed without an identity rule."""
    assert scaled(1, (X,)) == X  # not 1·x
    assert scaled(0, (X,)) == num(0)  # not 0·x
    assert scaled(3, (X,)) == mul(num(3), X)
    assert scaled(5, ()) == num(5)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [(num(6), num(-6)), (num(-6), num(6)), (X, mul(num(-1), X)), (mul(num(3), X), mul(num(-3), X))],
)
def test_negate(expr: Expr, expected: Expr) -> None:
    assert negate(expr) == expected


# ---------------------------------------------------------------------------
# Each rule, by hand, on both polarities
# ---------------------------------------------------------------------------


def fire(name: str, state: Expr, site_id: int) -> Expr:
    return apply(state, RULE_BY_NAME[name].rule_id, site_id)


def legal_sites(name: str, state: Expr) -> list[int]:
    rule = RULE_BY_NAME[name]
    return [s for r, s in legal_actions(state) if r == rule.rule_id]


def test_eval_add_sums_all_numeric_children() -> None:
    assert fire("eval_add", add(num(1), num(2), num(3)), 0) == num(6)
    assert fire("eval_add", add(X, num(2), num(3)), 0) == add(X, num(5))


def test_eval_add_drops_a_zero_sum_when_other_terms_remain() -> None:
    """The additive identity, and the reason ``3x + 6 + (−6)`` is not a dead end."""
    assert fire("eval_add", add(mul(num(3), X), num(6), num(-6)), 0) == mul(num(3), X)
    # With nothing else left, zero is the answer rather than an omission.
    assert fire("eval_add", add(num(6), num(-6)), 0) == num(0)


def test_eval_add_needs_two_numerals() -> None:
    assert legal_sites("eval_add", add(X, num(2))) == []
    assert legal_sites("eval_add", add(X, num(2), num(3))) == [0]


def test_eval_sub_and_its_polarity() -> None:
    assert fire("eval_sub", sub(num(9), num(4)), 0) == num(5)
    assert fire("eval_sub", sub(num(4), num(9)), 0) == num(-5)
    assert legal_sites("eval_sub", sub(X, num(4))) == []


def test_eval_mul_and_the_multiplicative_identity() -> None:
    assert fire("eval_mul", mul(num(2), num(3)), 0) == num(6)
    assert fire("eval_mul", mul(num(2), num(3), X), 0) == mul(num(6), X)
    # product 1 with factors left over collapses to the factors
    assert fire("eval_mul", mul(num(-1), num(-1), X), 0) == X
    assert legal_sites("eval_mul", mul(num(2), X)) == []


def test_combine_like_terms() -> None:
    assert fire("combine_like_terms", add(mul(num(3), X), mul(num(2), X)), 0) == mul(num(5), X)
    assert fire("combine_like_terms", add(X, X), 0) == mul(num(2), X)
    assert fire("combine_like_terms", add(mul(num(3), X), mul(num(-3), X)), 0) == num(0)
    # Different variables are not like terms.
    assert legal_sites("combine_like_terms", add(mul(num(3), X), mul(num(2), Y))) == []


def test_combine_like_terms_leaves_numerals_to_eval_add() -> None:
    """Deliberate non-overlap: two rules producing one successor is a duplicate action."""
    state = add(num(1), num(2))
    assert legal_sites("combine_like_terms", state) == []
    assert legal_sites("eval_add", state) == [0]


def test_combine_like_terms_handles_a_non_atomic_term() -> None:
    """`e + e = 2e` holds for any e, so the grouping key is structural, not variable-based."""
    inner = sub(X, num(1))
    assert fire("combine_like_terms", add(inner, inner), 0) == mul(num(2), inner)


def test_both_sides_rules_only_fire_on_top_level_addends() -> None:
    state = eq(add(mul(num(3), X), num(6)), num(21))
    # sites 1 (the ADD), 3 (the 3) and 4 (the x) are not addends of a side
    assert legal_sites("sub_both_sides", state) == [2, 5, 6]
    assert legal_sites("add_both_sides", state) == [2, 5, 6]


def test_both_sides_rules_need_an_equation() -> None:
    assert legal_sites("sub_both_sides", add(mul(num(3), X), num(6))) == []
    assert legal_sites("div_both_sides", mul(num(3), X)) == []


def test_sub_both_sides_moves_the_addend_across() -> None:
    """`A + e = B` becomes `A = B + (−e)`.

    The cancellation on e's own side is structural — the addend is removed from
    the sum, not computed away — so this stays one rule with no arithmetic in
    it. And never a SUB node: `(3x + 6) − 6` is a dead end no v1 rule reduces.
    """
    state = eq(add(mul(num(3), X), num(6)), num(21))
    out = fire("sub_both_sides", state, 5)
    assert out == eq(mul(num(3), X), add(num(21), num(-6)))
    assert SUB not in tokens(out)


def test_sub_both_sides_clearing_a_whole_side_leaves_zero() -> None:
    state = eq(mul(num(3), X), num(15))
    assert fire("sub_both_sides", state, 1) == eq(add(mul(num(-3), X), num(15)), num(0))


def test_sub_both_sides_removes_one_occurrence_not_both() -> None:
    """A side may hold the same addend twice; cancelling both is not subtraction."""
    state = eq(add(X, X, num(2)), num(9))
    out = fire("sub_both_sides", state, 2)
    assert out == eq(add(X, num(2)), add(num(9), mul(num(-1), X)))


def test_add_both_sides_adds_the_operand() -> None:
    state = eq(add(mul(num(3), X), num(6)), num(21))
    out = fire("add_both_sides", state, 5)
    assert out == eq(add(mul(num(3), X), num(6), num(6)), add(num(21), num(6)))


def test_div_both_sides_divides_by_the_coefficient() -> None:
    assert fire("div_both_sides", eq(mul(num(3), X), num(15)), 2) == eq(X, num(5))
    assert fire("div_both_sides", eq(mul(num(-3), X), num(15)), 2) == eq(X, num(-5))


# ---------------------------------------------------------------------------
# The exactness guard — the claim nothing else carries directly
# ---------------------------------------------------------------------------


def test_div_both_sides_guard_blocks_inexact_division() -> None:
    """**This test carries the exactness claim.**

    It is stated here rather than left to the fuzz because of what the fuzz can
    and cannot see:

    * The 𝔽ₚ layer is *structurally blind* to it. Every nonzero constant is
      invertible mod p, so `3x = 16 ⇒ x = 16·3⁻¹` is a valid field rewrite and
      field-equivalence passes it forever, at any number of draws.
    * The ℚ layer detects the *consequence* — `3x = 16` and `x = 5` disagree at
      x = 5 — but only after a broken guard has already fired.
    * This test asserts the *condition*: the rule must not be legal at all.

    The distinction is the one in the guard's own docstring: exactness is a
    closure condition on the vocabulary (rule set v1 has no fractions), not a
    soundness condition over a field.
    """
    inexact = eq(mul(num(3), X), num(16))
    rule = RULE_BY_NAME["div_both_sides"]
    site = enumerate_sites(inexact)[2]

    assert legal_sites("div_both_sides", inexact) == []
    assert not rule.legal(inexact, site)
    # Both polarities on the guard itself, and proof that the *guard* is what
    # blocked it — the LHS pattern matched fine.
    assert rule.lhs(inexact, site), "the shape matches; only the guard should object"
    assert not rule.guard(inexact, site)

    exact = eq(mul(num(3), X), num(15))
    exact_site = enumerate_sites(exact)[2]
    assert rule.lhs(exact, exact_site)
    assert rule.guard(exact, exact_site)
    assert legal_sites("div_both_sides", exact) == [2]


def test_div_both_sides_guard_blocks_a_zero_coefficient() -> None:
    zero = eq(mul(num(0), X), num(0))
    assert legal_sites("div_both_sides", zero) == []


@pytest.mark.parametrize(
    ("coefficient", "rhs", "fires"),
    [(3, 15, True), (3, 16, False), (-3, 15, True), (3, -15, True), (3, -16, False), (7, 0, True)],
)
def test_div_both_sides_guard_polarity_table(coefficient: int, rhs: int, fires: bool) -> None:
    state = eq(mul(num(coefficient), X), num(rhs))
    assert bool(legal_sites("div_both_sides", state)) is fires


def test_no_legal_div_both_sides_ever_invents_a_number() -> None:
    """Scan the whole small-coefficient space: when it fires, the answer is exact."""
    checked = 0
    for coefficient in range(-12, 13):
        for rhs in range(-40, 41):
            state = eq(mul(num(coefficient), X), num(rhs))
            for site_id in legal_sites("div_both_sides", state):
                out = fire("div_both_sides", state, site_id)
                assert isinstance(out, Op) and out.children[0] == X
                answer = out.children[1]
                assert isinstance(answer, Num)
                assert answer.value * coefficient == rhs, "the division was not exact"
                checked += 1
    assert checked >= 200, f"only {checked} firings scanned — the sweep went vacuous"


# ---------------------------------------------------------------------------
# Random state generators for the fuzz (seeded; disclosed in the report)
# ---------------------------------------------------------------------------


def random_arith(rng: random.Random, depth: int = 0) -> Expr:
    """Arithmetic-shaped states: what makes the eval_* rules fire."""
    if depth >= 3 or rng.random() < 0.4:
        return X if rng.random() < 0.25 else num(rng.randrange(-30, 31))
    kind = rng.choice((ADD, ADD, MUL, MUL, SUB, DIV))
    n = rng.randint(2, 4) if kind in (ADD, MUL) else 2
    return make_op(kind, [random_arith(rng, depth + 1) for _ in range(n)])


def random_equation(rng: random.Random) -> Expr:
    """Linear-equation-shaped states: what makes the both-sides rules fire.

    Coefficients and constants are drawn so that ``div_both_sides``'s guard is
    exercised on *both* polarities — roughly half the draws are exactly
    divisible and half are not.
    """
    coefficient = rng.choice([c for c in range(-6, 7) if c != 0])
    root = rng.choice(("solved_shape", "with_constant", "both_sides", "messy"))
    if root == "solved_shape":
        rhs = coefficient * rng.randrange(-9, 10) if rng.random() < 0.5 else rng.randrange(-40, 41)
        return eq(mul(num(coefficient), X), num(rhs))
    if root == "with_constant":
        return eq(
            add(mul(num(coefficient), X), num(rng.randrange(-20, 21))),
            num(rng.randrange(-40, 41)),
        )
    if root == "both_sides":
        other = rng.choice([c for c in range(-6, 7) if c != coefficient])
        return eq(
            add(mul(num(coefficient), X), num(rng.randrange(-20, 21))),
            add(mul(num(other), X), num(rng.randrange(-20, 21))),
        )
    return eq(random_arith(rng), random_arith(rng))


def random_state(rng: random.Random) -> Expr:
    return random_equation(rng) if rng.random() < 0.6 else random_arith(rng)


def _assignment_field(rng: random.Random, vars_: tuple[int, ...]) -> dict[int, int]:
    return {v: rng.randrange(P) for v in vars_}


def _assignment_exact(rng: random.Random, vars_: tuple[int, ...]) -> dict[int, Fraction | int]:
    return {v: rng.randrange(-50, 51) for v in vars_}


# ---------------------------------------------------------------------------
# Layer 1 — equivalence
# ---------------------------------------------------------------------------

INSTANCES_PER_RULE = 200
ASSIGNMENTS_PER_INSTANCE = 50  # 200 x 50 = 10_000 assignments per rule (plan chunk 2)


def _collect_instances(
    rule_name: str, rng: random.Random, wanted: int, budget: int = 200_000
) -> list[tuple[Expr, Site]]:
    """Random states where ``rule_name`` fires, with the site it fires at."""
    rule = RULE_BY_NAME[rule_name]
    found: list[tuple[Expr, Site]] = []
    for _ in range(budget):
        if len(found) >= wanted:
            break
        state = random_state(rng)
        hits = [s for s in enumerate_sites(state) if rule.legal(state, s)]
        if hits:
            found.append((state, rng.choice(hits)))
    return found


@pytest.mark.parametrize("rule_name", [r.name for r in RULES])
def test_soundness_fuzz_equivalence(rule_name: str, request: pytest.FixtureRequest) -> None:
    """**Chunk 2 gate.** 10,000 assignments per rule, over 𝔽ₚ *and* over ℚ.

    Node rules are checked by value equality: the rewritten subtree must equal
    the original under every assignment. Equation rules are checked by truth
    value: `L = R` and `L' = R'` must be true or false *together*. Using value
    equality on an equation would be a category error — an equation is not a
    number — and using truth-value agreement on an expression would be vacuous.

    Undefined evaluations (a denominator that vanished) are skipped and counted,
    never passed. The floors below are what stop "10,000 assignments" from
    meaning "twelve assignments and 9,988 shrugs".
    """
    rule = RULE_BY_NAME[rule_name]
    rng = random.Random(0xC0FFEE + rule.rule_id)
    instances = _collect_instances(rule_name, rng, INSTANCES_PER_RULE)

    stats = {
        "instances": len(instances),
        "field_ok": 0,
        "field_skip": 0,
        "exact_ok": 0,
        "exact_skip": 0,
    }

    for state, site in instances:
        if rule.scope == "node":
            # Compare the rewritten *subtree* against the original subtree: the
            # rest of the state is untouched by construction, and including it
            # would dilute the check with terms the rule never saw.
            before_expr, after_expr = site.node, rule.rhs(state, site)
        else:
            before_expr, after_expr = state, rule.apply(state, site)
        vars_ = tuple(sorted(set(variables(before_expr)) | set(variables(after_expr))))

        for _ in range(ASSIGNMENTS_PER_INSTANCE):
            env_f = _assignment_field(rng, vars_)
            if rule.scope == "node":
                left = eval_field(before_expr, env_f, P)
                right = eval_field(after_expr, env_f, P)
            else:
                left = holds_field(before_expr, env_f, P)
                right = holds_field(after_expr, env_f, P)
            if left is None or right is None:
                stats["field_skip"] += 1
            else:
                assert left == right, (
                    f"{rule_name}: 𝔽ₚ mismatch on {tokens(state)} at {site.site_id}"
                )
                stats["field_ok"] += 1

            env_q = _assignment_exact(rng, vars_)
            if rule.scope == "node":
                left_q = eval_exact(before_expr, env_q)
                right_q = eval_exact(after_expr, env_q)
            else:
                left_q = holds_exact(before_expr, env_q)
                right_q = holds_exact(after_expr, env_q)
            if left_q is None or right_q is None:
                stats["exact_skip"] += 1
            else:
                assert left_q == right_q, f"{rule_name}: ℚ mismatch on {tokens(state)}"
                stats["exact_ok"] += 1

    # --- anti-vacuity floors ------------------------------------------------
    assert stats["instances"] == INSTANCES_PER_RULE, (
        f"{rule_name} fired on only {stats['instances']} random states — the "
        "generator stopped reaching it, so this gate covers nothing."
    )
    total = INSTANCES_PER_RULE * ASSIGNMENTS_PER_INSTANCE
    assert stats["field_ok"] + stats["field_skip"] == total
    assert stats["field_ok"] >= total * 0.5, (
        f"{rule_name}: only {stats['field_ok']}/{total} 𝔽ₚ assignments actually "
        f"evaluated ({stats['field_skip']} skipped on an undefined value)."
    )
    assert stats["exact_ok"] >= total * 0.5, (
        f"{rule_name}: only {stats['exact_ok']}/{total} ℚ assignments actually evaluated."
    )
    request.node.stash  # noqa: B018 — keep the stats visible in a failure report
    print(f"\n  {rule_name:<20} {stats}")


# ---------------------------------------------------------------------------
# Layer 2 — closure
# ---------------------------------------------------------------------------


def test_soundness_closure_every_successor_is_a_well_formed_state() -> None:
    """**Chunk 2 gate, second layer.** Equivalence is not enough on its own.

    A rewrite can be perfectly truth-preserving over a field and still leave the
    vocabulary — that is exactly what an inexact division would do. So every
    successor of every legal action must be canonical and must round-trip
    through chunk 1's codec byte-exactly.
    """
    rng = random.Random(0x5EED)
    per_rule: dict[str, int] = {r.name: 0 for r in RULES}
    checked = 0
    for _ in range(12_000):
        state = random_state(rng)
        for (rule_id, _site_id), successor in successors(state):
            assert canonicalize(successor) == successor, "successor is not canonical"
            printed = tokens(successor)
            assert identity_key(parse(printed)) == printed, "successor does not round-trip"
            for value in _numerals(successor):
                assert isinstance(value, int), "a non-integer entered the state"
            per_rule[RULES[rule_id].name] += 1
            checked += 1

    assert checked >= 40_000, f"only {checked} successors checked"
    for name, count in per_rule.items():
        assert count >= 100, f"{name} produced only {count} successors — closure untested for it"
    print(f"\n  closure: {checked} successors, per rule {per_rule}")


def _numerals(expr: Expr) -> list[int]:
    out: list[int] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Num):
            out.append(node.value)
        elif isinstance(node, Op):
            stack.extend(node.children)
    return out


# ---------------------------------------------------------------------------
# The independent reference matcher
# ---------------------------------------------------------------------------
#
# Independence is in the *code path*, not the mathematics: its own recursive
# traversal (the production matcher is iterative), its own path arithmetic, its
# own predicates, and no import of any production predicate. It shares the rule
# *specification* — which is the point — so it catches implementation divergence
# (traversal order, off-by-one site ids, a guard wired to the wrong rule), not a
# misreading of the spec. The hand-written per-rule tests above and the
# soundness fuzz are what cover that.


def ref_sites(node: Expr, path: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], Expr]]:
    """Recursive pre-order walk. Deliberately not the production traversal."""
    out = [(path, node)]
    if isinstance(node, Op):
        for index, child in enumerate(node.children):
            out.extend(ref_sites(child, (*path, index)))
    return out


def ref_addend_paths(state: Expr) -> set[tuple[int, ...]]:
    if not (isinstance(state, Op) and state.kind == EQ):
        return set()
    out = set()
    for side_index in (0, 1):
        side = state.children[side_index]
        if isinstance(side, Op) and side.kind == ADD:
            out |= {(side_index, i) for i in range(len(side.children))}
        else:
            out.add((side_index,))
    return out


def ref_legal_actions(state: Expr) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    walk = ref_sites(state)
    addends = ref_addend_paths(state)
    is_equation = isinstance(state, Op) and state.kind == EQ

    for site_id, (path, node) in enumerate(walk):
        op = node if isinstance(node, Op) else None
        nums = [c for c in op.children if isinstance(c, Num)] if op else []

        if op is not None and op.kind == ADD and len(nums) > 1:
            out.append((0, site_id))
        if op is not None and op.kind == SUB and len(nums) == 2:
            out.append((1, site_id))
        if op is not None and op.kind == MUL and len(nums) > 1:
            out.append((2, site_id))
        if op is not None and op.kind == ADD:
            keys: dict[tuple, int] = {}
            for child in op.children:
                if isinstance(child, Num):
                    continue
                if (
                    isinstance(child, Op)
                    and child.kind == MUL
                    and isinstance(child.children[0], Num)
                ):
                    rest = (
                        tokens(make_op(MUL, child.children[1:]))
                        if len(child.children) > 2
                        else tokens(child.children[1])
                    )
                else:
                    rest = tokens(child)
                keys[rest] = keys.get(rest, 0) + 1
            if any(count > 1 for count in keys.values()):
                out.append((3, site_id))
        if path in addends:
            out.append((4, site_id))
            out.append((5, site_id))
        if is_equation and path == (0, 0):
            assert isinstance(state, Op)
            left, right = state.children
            if (
                isinstance(left, Op)
                and left.kind == MUL
                and isinstance(left.children[0], Num)
                and left.children[0].value != 0
                and isinstance(right, Num)
                and right.value % left.children[0].value == 0
            ):
                out.append((6, site_id))
    return sorted(out)


def test_matcher_agrees_with_an_independent_reference_on_5k_trees() -> None:
    """**Chunk 2 gate.** 5,000 random trees, exact agreement on the action set."""
    rng = random.Random(0xA11CE)
    total_actions = 0
    nonempty = 0
    seen_rules: set[int] = set()
    for i in range(5000):
        state = random_state(rng)
        mine = sorted(legal_actions(state))
        theirs = ref_legal_actions(state)
        assert mine == theirs, f"disagreement on tree {i}: {tokens(state)}\n  {mine}\n  {theirs}"
        total_actions += len(mine)
        nonempty += bool(mine)
        seen_rules |= {r for r, _ in mine}

    assert nonempty >= 4000, f"only {nonempty}/5000 trees had any legal action"
    assert seen_rules == set(range(7)), f"rules never exercised: {set(range(7)) - seen_rules}"
    print(f"\n  reference agreement: 5000 trees, {total_actions} actions, {nonempty} non-terminal")


# ---------------------------------------------------------------------------
# Closure of the rule set: the spec's own example must actually solve
# ---------------------------------------------------------------------------


def bfs_solve(
    start: Expr, cap: int = 6, exclude: frozenset[int] | set[int] = frozenset()
) -> list[tuple[tuple[int, int], Expr]] | None:
    """Shortest derivation to a solved form, or None within ``cap`` steps.

    ``exclude`` suppresses rule ids, which is how the redundancy of
    ``add_both_sides`` is measured rather than asserted.
    """
    from collections import deque

    def solved(state: Expr) -> bool:
        return (
            isinstance(state, Op)
            and state.kind == EQ
            and isinstance(state.children[0], Var)
            and isinstance(state.children[1], Num)
        )

    seen = {identity_key(start)}
    queue = deque([(start, [])])
    while queue:
        state, path = queue.popleft()
        if solved(state):
            return path
        if len(path) >= cap:
            continue
        for action, successor in successors(state):
            if action[0] in exclude:
                continue
            key = identity_key(successor)
            if key in seen:
                continue
            seen.add(key)
            queue.append((successor, [*path, (action, successor)]))
    return None


def test_the_spec_example_solves_at_bfs_par_4() -> None:
    """The closure demonstration for rule set v1, pinned.

    `3x + 6 = 21` is the spec's own interpreter example (§2). If the minimal
    closed set does not close on it, the set is wrong — so this is a gate, not
    an illustration. The exact derivation is pinned because it is also what
    chunk 4's renderer will be proofread against.
    """
    start = eq(add(mul(num(3), X), num(6)), num(21))
    path = bfs_solve(start)
    assert path is not None, "rule set v1 does not close on the spec's own example"
    assert len(path) == 3
    names = [RULES[rule_id].name for (rule_id, _), _ in path]
    assert names == ["sub_both_sides", "eval_add", "div_both_sides"]
    assert path[-1][1] == eq(X, num(5))


def test_add_both_sides_is_reachability_redundant() -> None:
    """A measured finding, pinned so it stays true or stops quietly being claimed.

    ``add_both_sides`` only ever grows both sides, so it cannot shorten a
    derivation — every problem is solvable without it, at exactly the same par.
    It is in the pinned rule set (plan §8 decision 2) and it is sound, so it
    stays; but it is pure branching tax, and this test is the evidence for
    proposing its removal as a one-lever round.
    """
    problems = [
        eq(add(mul(num(3), X), num(6)), num(21)),
        eq(add(mul(num(2), X), num(-4)), num(10)),
        eq(add(mul(num(5), X), num(3)), add(mul(num(2), X), num(18))),
    ]
    add_id = RULE_BY_NAME["add_both_sides"].rule_id
    for problem in problems:
        with_it = bfs_solve(problem, cap=6)
        without = bfs_solve(problem, cap=6, exclude={add_id})
        assert with_it is not None and without is not None
        assert len(with_it) == len(without), (
            "add_both_sides changed par — it is not redundant after all, and the "
            "docstring in rules.py claiming so must be corrected."
        )


@pytest.mark.parametrize(
    ("equation", "answer", "par"),
    [
        (eq(mul(num(3), X), num(15)), 5, 1),
        (eq(add(mul(num(2), X), num(-4)), num(10)), 7, 3),
        (eq(add(mul(num(5), X), num(3)), add(mul(num(2), X), num(18))), 5, 5),
    ],
)
def test_rule_set_closes_on_representative_problems(equation: Expr, answer: int, par: int) -> None:
    """Including x on both sides, which chunk 3's near-miss fixtures depend on."""
    path = bfs_solve(equation, cap=6)
    assert path is not None, f"no solution within 6 steps for {tokens(equation)}"
    assert len(path) == par
    assert path[-1][1] == eq(X, num(answer))


# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------


def test_apply_rejects_an_illegal_action() -> None:
    state = add(X, num(1))
    with pytest.raises(ValueError, match="not legal"):
        apply(state, RULE_BY_NAME["eval_add"].rule_id, 0)


def test_apply_rejects_an_out_of_range_site() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        apply(add(X, num(1)), 0, 99)


def test_apply_rejects_an_unknown_rule() -> None:
    with pytest.raises(ValueError, match="no rule with id"):
        apply(add(X, num(1)), 99, 0)


def test_legal_actions_is_sorted_and_deterministic() -> None:
    rng = random.Random(11)
    for _ in range(200):
        state = random_state(rng)
        actions = legal_actions(state)
        assert actions == sorted(actions)
        assert actions == legal_actions(state)


def test_successors_agrees_with_legal_actions() -> None:
    rng = random.Random(12)
    for _ in range(200):
        state = random_state(rng)
        assert [a for a, _ in successors(state)] == legal_actions(state)


def test_effective_branching_is_below_raw_branching() -> None:
    """Canonicalisation merges distinct actions into one successor. Pinned.

    Chunk 7's gate arithmetic wants the number search actually branches into,
    which is the count of *distinct canonical successors* — not the count of
    legal actions, which is what the policy head masks over. The two are not the
    same number, and assuming they are inflates every search-budget estimate.
    Measured at ~22% merged on mid-derivation states; see
    ``scripts/measure_branching.py``.
    """
    rng = random.Random(0xB4)
    raw = merged = 0
    for _ in range(400):
        state = random_state(rng)
        for _ in range(rng.randint(1, 3)):  # walk into mid-derivation territory
            options = successors(state)
            if not options:
                break
            state = rng.choice(options)[1]
        actions = successors(state)
        raw += len(actions)
        merged += len(actions) - len({identity_key(s) for _, s in actions})

    assert raw > 1000, f"only {raw} actions sampled — the measurement went vacuous"
    assert merged > 0, (
        "no two actions ever produced the same successor. Either canonicalisation "
        "stopped merging, or this sample never reached a state where it can — "
        "both make the raw/effective distinction untestable here."
    )


def test_no_action_is_legal_on_an_atom() -> None:
    assert legal_actions(num(3)) == []
    assert legal_actions(X) == []


def test_div_node_states_have_no_eval_rule() -> None:
    """The recorded closure gap: rule set v1 has no ``eval_div``.

    ``6 ÷ 2`` is inert. This is pinned so chunk 5 cannot quietly emit DIV in an
    EVALUATE problem and produce something unsolvable by construction.
    """
    assert legal_actions(div(num(6), num(2))) == []
