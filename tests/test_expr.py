"""Parser, printer, and the canonical-form claims.

The module docstring of ``reckoner.expr`` states eight canonicalisation claims,
C1–C8. A docstring is a claim, so each one has a test named after it here. The
two round-trip gates (200K random trees byte-exact; hypothesis fuzz rejecting
malformed input cleanly) are at the bottom.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reckoner.expr import (
    Num,
    Op,
    ParseError,
    Var,
    add,
    canonicalize,
    div,
    eq,
    from_digits,
    has_var,
    identity_key,
    make_op,
    mul,
    num,
    parse,
    sub,
    to_digits,
    tokens,
    var,
)
from reckoner.vocab import (
    ADD,
    DIV,
    EQ,
    GOAL_SOLVE,
    LPAREN,
    MUL,
    NUM,
    NUM_NEG,
    PAD,
    RPAREN,
    SEP,
    SUB,
    VAR_X,
    VAR_Y,
    VOCAB_SIZE,
    digit_token,
)

X = var(VAR_X)
Y = var(VAR_Y)


def T(*names: int) -> tuple[int, ...]:
    return tuple(names)


# ---------------------------------------------------------------------------
# Digits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "digits"),
    [
        (0, [0]),
        (1, [1]),
        (624, [624]),
        (625, [1, 0]),
        (626, [1, 1]),
        (1887, [3, 12]),
        (625**2, [1, 0, 0]),
        (625**3 + 7, [1, 0, 0, 7]),
    ],
)
def test_digit_conversion(value: int, digits: list[int]) -> None:
    assert to_digits(value) == digits
    assert from_digits(digits) == value


def test_to_digits_rejects_negative() -> None:
    """Sign lives in the marker, so the digit layer never sees one (C2)."""
    with pytest.raises(ValueError, match="non-negative"):
        to_digits(-1)


# ---------------------------------------------------------------------------
# The canonical-form claims, C1 .. C8
# ---------------------------------------------------------------------------


def test_c1_numerals_are_msb_first_with_no_leading_zeros() -> None:
    assert tokens(num(1887)) == T(NUM, LPAREN, digit_token(3), digit_token(12), RPAREN)
    # A leading zero is a legal spelling of a legal value: it parses, and prints
    # canonically. Rejecting it would narrow `print ∘ parse = canonical`.
    assert tokens(parse(T(NUM, LPAREN, digit_token(0), digit_token(5), RPAREN))) == tokens(num(5))
    assert tokens(num(0)) == T(NUM, LPAREN, digit_token(0), RPAREN)


def test_c2_sign_lives_in_the_marker_and_zero_has_one_spelling() -> None:
    assert tokens(num(-7))[0] == NUM_NEG
    assert tokens(num(7))[0] == NUM
    # Negative zero is not a second zero.
    assert parse(T(NUM_NEG, LPAREN, digit_token(0), RPAREN)) == num(0)
    assert tokens(parse(T(NUM_NEG, LPAREN, digit_token(0), RPAREN)))[0] == NUM


def test_c3_variadic_nodes_are_flattened() -> None:
    nested = add(add(num(1), num(2)), num(3))
    assert isinstance(nested, Op)
    assert len(nested.children) == 3
    assert not any(isinstance(c, Op) and c.kind == ADD for c in nested.children)

    nested_mul = mul(num(2), mul(num(3), X))
    assert isinstance(nested_mul, Op)
    assert len(nested_mul.children) == 3


def test_c3_flattening_does_not_cross_operators() -> None:
    """A MUL inside an ADD stays a MUL — flattening is same-kind only."""
    tree = add(mul(num(3), X), num(6))
    assert isinstance(tree, Op)
    assert len(tree.children) == 2
    assert isinstance(tree.children[0], Op)
    assert tree.children[0].kind == MUL


def test_c4_singleton_variadic_collapses() -> None:
    assert add(X) == X
    assert mul(num(5)) == num(5)
    # And through the parser, which is where a rewrite's output arrives.
    assert parse(T(ADD, LPAREN, VAR_X, RPAREN)) == X


def test_c5_add_puts_variable_terms_first() -> None:
    """`3x + 6`, never `6 + 3x` — the interpreter's readability depends on it."""
    tree = add(num(6), mul(num(3), X))
    assert isinstance(tree, Op)
    assert has_var(tree.children[0])
    assert tree.children[1] == num(6)
    # Order of the arguments cannot matter.
    assert add(num(6), mul(num(3), X)) == add(mul(num(3), X), num(6))


def test_c6_mul_puts_the_coefficient_first() -> None:
    """`3x`, never `x3`."""
    tree = mul(X, num(3))
    assert isinstance(tree, Op)
    assert tree.children[0] == num(3)
    assert tree.children[1] == X
    assert mul(X, num(3)) == mul(num(3), X)


def test_c7_eq_orients_the_variable_side_first() -> None:
    """Equality is symmetric, so orientation carries no information.

    Ordering it makes the SOLVE goal form `x = <number>` canonical by
    construction rather than by convention — the checker never has to ask which
    way round a solved equation was written.
    """
    assert eq(num(15), mul(num(3), X)) == eq(mul(num(3), X), num(15))
    solved = eq(num(5), X)
    assert isinstance(solved, Op)
    assert solved.children == (X, num(5))


def test_c7_eq_with_variables_on_both_sides_is_still_deterministic() -> None:
    left, right = add(mul(num(2), X), num(3)), mul(num(5), X)
    assert eq(left, right) == eq(right, left)


def test_c8_sub_and_div_keep_their_operand_order() -> None:
    """Ordering a non-commutative operator is not canonicalisation, it is a wrong answer."""
    assert sub(num(3), num(5)) != sub(num(5), num(3))
    assert div(num(6), num(2)) != div(num(2), num(6))
    tree = sub(num(3), X)
    assert isinstance(tree, Op)
    assert tree.children == (num(3), X)  # variable second, unlike ADD


def test_canonicalize_is_idempotent_and_agrees_with_the_parser() -> None:
    raw = Op(ADD, (num(6), Op(ADD, (mul(num(3), X), num(1)))))
    once = canonicalize(raw)
    assert canonicalize(once) == once
    assert once == parse(tokens(once))


def test_canonicalize_fixes_a_hand_built_non_canonical_tree() -> None:
    """The constructors canonicalise; a tree built by hand around them need not be."""
    raw = Op(MUL, (X, Op(MUL, (num(3), Y))))
    fixed = canonicalize(raw)
    assert isinstance(fixed, Op)
    assert fixed.children == (num(3), X, Y)


# ---------------------------------------------------------------------------
# Printing and parsing, by hand
# ---------------------------------------------------------------------------


def test_the_docstring_example_prints_exactly() -> None:
    """`3x + 6 = 21` — the sequence quoted in reckoner.expr's docstring."""
    tree = eq(add(mul(num(3), X), num(6)), num(21))
    assert tokens(tree) == T(
        EQ, LPAREN,
        ADD, LPAREN,
        MUL, LPAREN, NUM, LPAREN, digit_token(3), RPAREN, VAR_X, RPAREN,
        NUM, LPAREN, digit_token(6), RPAREN,
        RPAREN,
        NUM, LPAREN, digit_token(21), RPAREN,
        RPAREN,
    )  # fmt: skip
    assert len(tokens(tree)) == 22
    assert parse(tokens(tree)) == tree


def test_atoms_round_trip() -> None:
    for atom in (num(0), num(-1), num(624), num(625), num(10**12), X, Y):
        assert parse(tokens(atom)) == atom


def test_large_magnitude_survives() -> None:
    """Arbitrary magnitude, finite vocabulary — the compositional claim."""
    value = 625**7 + 12345
    assert parse(tokens(num(value))) == num(value)
    assert len(to_digits(value)) == 8


def test_identity_key_is_the_canonical_token_sequence() -> None:
    a = add(num(6), mul(num(3), X))
    b = add(mul(X, num(3)), num(6))
    assert identity_key(a) == identity_key(b) == tokens(a)
    assert identity_key(a) != identity_key(add(num(7), mul(num(3), X)))


def test_has_var() -> None:
    assert not has_var(num(3))
    assert has_var(X)
    assert has_var(eq(add(mul(num(3), X), num(6)), num(21)))
    assert not has_var(add(num(1), num(2), num(3)))


def test_nodes_are_hashable_and_usable_as_keys() -> None:
    """Dedup and contamination machinery downstream put these in sets."""
    assert len({add(num(1), X), add(X, num(1))}) == 1


# ---------------------------------------------------------------------------
# Rejection — malformed input, by hand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seq", "match"),
    [
        ((), "empty token sequence"),
        ((LPAREN,), "not part of the expression grammar"),
        ((RPAREN,), "unmatched"),
        ((NUM,), "expected '\\('"),
        ((NUM, VAR_X), "expected '\\('"),
        ((NUM, LPAREN, RPAREN), "empty numeral"),
        ((NUM, LPAREN, digit_token(1)), "unclosed group"),
        ((ADD, LPAREN, RPAREN), "empty group"),
        ((ADD, LPAREN, VAR_X, VAR_Y), "unclosed group"),
        ((SUB, LPAREN, VAR_X, RPAREN), "exactly 2 operands"),
        ((DIV, LPAREN, VAR_X, VAR_Y, VAR_X, RPAREN), "exactly 2 operands"),
        ((EQ, LPAREN, VAR_X, RPAREN), "exactly 2 operands"),
        ((digit_token(3),), "outside a numeral"),
        ((ADD, LPAREN, digit_token(3), VAR_X, RPAREN), "outside a numeral"),
        ((NUM, LPAREN, VAR_X, RPAREN), "inside a numeral"),
        ((VAR_X, VAR_Y), "more than one top-level expression"),
        ((PAD,), "not part of the expression grammar"),
        ((SEP,), "not part of the expression grammar"),
        ((GOAL_SOLVE, VAR_X), "not part of the expression grammar"),
        ((VOCAB_SIZE,), "out of vocabulary range"),
        ((-1,), "out of vocabulary range"),
        ((1.5,), "is not an int"),
    ],
)
def test_malformed_sequences_are_rejected(seq: tuple, match: str) -> None:
    with pytest.raises(ParseError, match=match):
        parse(seq)


def test_padding_is_the_callers_business() -> None:
    """A parser that skips PAD cannot tell a padded sequence from a corrupt one."""
    padded = tokens(X) + (PAD, PAD)
    with pytest.raises(ParseError):
        parse(padded)
    assert parse(padded[: padded.index(PAD)]) == X


def test_make_op_arity_errors_are_caller_bugs_not_parse_errors() -> None:
    """A bad call is a ValueError; malformed *input* is a ParseError. Not the same fault."""
    with pytest.raises(ValueError, match="exactly 2 children"):
        make_op(SUB, (X,))
    with pytest.raises(ValueError, match="at least one child"):
        make_op(ADD, ())
    with pytest.raises(ValueError, match="is not an operator"):
        make_op(VAR_X, (X, X))


def test_var_constructor_rejects_non_variables() -> None:
    with pytest.raises(ValueError, match="is not a variable token"):
        var(NUM)


# ---------------------------------------------------------------------------
# Random trees — the generator both gates use
# ---------------------------------------------------------------------------


def random_expr(rng: random.Random, depth: int = 0, max_depth: int = 4) -> object:
    """A random canonical expression. Seeded; used by the 200K gate."""
    if depth >= max_depth or rng.random() < 0.35:
        if rng.random() < 0.3:
            return var(rng.choice((VAR_X, VAR_Y)))
        magnitude = rng.choice((10, 625, 625**2, 625**3))
        value = rng.randrange(-magnitude, magnitude)
        return num(value)

    kind = rng.choice((ADD, MUL, SUB, DIV, EQ))
    n = rng.randint(2, 4) if kind in (ADD, MUL) else 2
    kids = [random_expr(rng, depth + 1, max_depth) for _ in range(n)]
    return make_op(kind, kids)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_parse_print_is_identity_on_random_trees(seed: int) -> None:
    """parse ∘ print = id."""
    tree = random_expr(random.Random(seed))
    assert parse(tokens(tree)) == tree


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_print_parse_is_canonical_and_idempotent(seed: int) -> None:
    """print ∘ parse = canonical: reparsing a printed tree changes nothing further."""
    once = tokens(random_expr(random.Random(seed)))
    twice = tokens(parse(once))
    assert twice == once
    assert tokens(parse(twice)) == twice
    assert canonicalize(parse(once)) == parse(once)


def test_round_trip_200k_byte_exact() -> None:
    """**Chunk 1 gate:** 200K random-tree round-trips, byte-exact.

    Token sequences are tuples of ints; tuple equality over them is byte
    equality of the sequence, and it is checked on both directions of the
    round-trip — the printed form must survive a parse, and the tree must
    survive a print. A single mismatch fails with the offending seed.
    """
    rng = random.Random(20260814)
    for i in range(200_000):
        tree = random_expr(rng)
        printed = tokens(tree)
        reparsed = parse(printed)
        assert reparsed == tree, f"tree mismatch at iteration {i}: {printed}"
        assert tokens(reparsed) == printed, f"token mismatch at iteration {i}: {printed}"


# ---------------------------------------------------------------------------
# Fuzz — malformed input must be rejected, never survived
# ---------------------------------------------------------------------------


def _assert_clean(seq: tuple[int, ...]) -> None:
    """Either it parses and round-trips, or it raises ParseError. No third outcome."""
    try:
        tree = parse(seq)
    except ParseError:
        return
    printed = tokens(tree)
    assert parse(printed) == tree
    assert tokens(parse(printed)) == printed


@settings(max_examples=2000, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=VOCAB_SIZE - 1), max_size=40))
def test_fuzz_random_token_sequences(seq: list[int]) -> None:
    _assert_clean(tuple(seq))


@settings(max_examples=1000, deadline=None)
@given(st.lists(st.integers(min_value=-1000, max_value=VOCAB_SIZE + 1000), max_size=40))
def test_fuzz_out_of_range_tokens(seq: list[int]) -> None:
    _assert_clean(tuple(seq))


@settings(max_examples=1500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.lists(
        st.tuples(
            st.sampled_from(("delete", "insert", "replace", "swap")),
            st.integers(min_value=0, max_value=10_000),
            st.integers(min_value=0, max_value=VOCAB_SIZE - 1),
        ),
        min_size=1,
        max_size=4,
    ),
)
def test_fuzz_mutations_of_valid_sequences(seed: int, edits: list[tuple[str, int, int]]) -> None:
    """The fuzz that actually finds things.

    A uniformly random token list is rejected on its first or second token
    almost every time, so it exercises one branch of the parser. Mutating a
    *valid* sequence lands the malformation deep inside a well-formed prefix —
    an unbalanced paren at depth 3, a digit where a subtree belongs, a numeral
    with an expression in it — which is where a parser actually breaks.
    """
    seq = list(tokens(random_expr(random.Random(seed))))
    for kind, raw_pos, token in edits:
        if not seq:
            break
        pos = raw_pos % len(seq)
        if kind == "delete":
            del seq[pos]
        elif kind == "insert":
            seq.insert(pos, token)
        elif kind == "replace":
            seq[pos] = token
        else:
            other = (pos + 1) % len(seq)
            seq[pos], seq[other] = seq[other], seq[pos]
    _assert_clean(tuple(seq))


@settings(max_examples=200, deadline=None)
@given(st.integers(min_value=1, max_value=400))
def test_deep_nesting_never_raises_recursionerror(depth: int) -> None:
    """Parser and printer are iterative; depth is an input, not a limit.

    A deep sequence must parse or be rejected — a RecursionError is neither, and
    on a fuzz-shaped input it is a crash in the middle of self-play.
    """
    seq = tuple([ADD, LPAREN] * depth + [VAR_X, VAR_X] + [RPAREN] * depth)
    tree = parse(seq)
    assert tokens(tree) == tokens(parse(tokens(tree)))

    unbalanced = tuple([ADD, LPAREN] * depth + [VAR_X])
    with pytest.raises(ParseError, match="unclosed group"):
        parse(unbalanced)


def test_deeply_nested_tree_prints_without_recursion() -> None:
    """Depth 2000 through print and parse.

    Compared by ``identity_key``, not by ``==``: node equality is Python's
    recursive dataclass ``__eq__`` and would blow the interpreter stack here.
    That is the documented limit, and the reason ``identity_key`` is the
    project's comparison of record for trees of unknown depth.
    """
    tree: object = num(1)
    for _ in range(2000):
        tree = make_op(SUB, (tree, num(1)))
    printed = tokens(tree)
    assert identity_key(parse(printed)) == printed


# ---------------------------------------------------------------------------
# Hypothesis-generated trees (structure, not seeds)
# ---------------------------------------------------------------------------

atoms = st.one_of(
    st.integers(min_value=-(625**3), max_value=625**3).map(num),
    st.sampled_from((VAR_X, VAR_Y)).map(var),
)


def _op(children: st.SearchStrategy) -> st.SearchStrategy:
    return st.one_of(
        st.lists(children, min_size=2, max_size=3).map(lambda cs: add(*cs)),
        st.lists(children, min_size=2, max_size=3).map(lambda cs: mul(*cs)),
        st.tuples(children, children).map(lambda cs: sub(*cs)),
        st.tuples(children, children).map(lambda cs: div(*cs)),
        st.tuples(children, children).map(lambda cs: eq(*cs)),
    )


trees = st.recursive(atoms, _op, max_leaves=12)


@settings(max_examples=500, deadline=None)
@given(trees)
def test_hypothesis_trees_round_trip(tree: object) -> None:
    assert parse(tokens(tree)) == tree
    assert canonicalize(tree) == tree


@settings(max_examples=500, deadline=None)
@given(trees)
def test_hypothesis_trees_are_canonical_by_construction(tree: object) -> None:
    """Every claim, checked structurally on every generated tree."""
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, Num):
            digits = to_digits(abs(node.value))
            assert digits == [0] or digits[0] != 0  # C1
            continue
        if isinstance(node, Var):
            continue
        assert isinstance(node, Op)
        if node.kind in (ADD, MUL):
            assert len(node.children) >= 2  # C4
            assert not any(  # C3
                isinstance(c, Op) and c.kind == node.kind for c in node.children
            )
            vars_first = node.kind == ADD  # C5 / C6
            flags = [has_var(c) for c in node.children]
            assert flags == sorted(flags, reverse=vars_first)
        else:
            assert len(node.children) == 2
            if node.kind == EQ:  # C7
                flags = [has_var(c) for c in node.children]
                assert flags == sorted(flags, reverse=True)
        stack.extend(node.children)
