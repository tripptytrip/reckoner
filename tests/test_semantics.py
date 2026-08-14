"""Evaluation over 𝔽ₚ and ℚ — including what each domain is blind to.

Two domains exist because neither is sufficient. These tests pin the specific
things each one catches that the other cannot, so a later simplification that
drops one has to argue with a named failure rather than with a preference.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from reckoner.expr import add, div, eq, mul, num, sub, var
from reckoner.semantics import (
    eval_exact,
    eval_field,
    holds_exact,
    holds_field,
    linear_root,
    variables,
)
from reckoner.vocab import VAR_X, VAR_Y

P = 2_147_483_647
X = var(VAR_X)
Y = var(VAR_Y)


# ---------------------------------------------------------------------------
# ℚ is exact — and that is a requirement, not an aspiration
# ---------------------------------------------------------------------------


def test_exact_division_stays_exact() -> None:
    """Regression: ``DIV`` on two integer leaves used to be *float* division.

    Numerals were pushed onto the evaluation stack as raw Python ints, so
    ``-1 ÷ -24`` evaluated as ``-1 / -24`` — a float — and every value above it
    became a binary rational. The symptom was a soundness fuzz "failure" on
    ``eval_add`` where the two sides differed by 2⁻⁵⁰: the rule was fine, the
    evaluator was lying.

    𝔽ₚ could not have caught this. Its arithmetic is integral end to end, so
    the contamination has nowhere to enter — which is exactly the argument for
    keeping both domains.
    """
    value = eval_exact(div(num(-1), num(-24)), {})
    assert value == Fraction(1, 24)
    assert value.denominator == 24, f"denominator {value.denominator} — float contamination"

    nested = eval_exact(add(div(num(1), num(3)), div(num(1), num(6))), {})
    assert nested == Fraction(1, 2)
    assert nested.denominator == 2


def test_exact_evaluation_of_a_bare_numeral() -> None:
    """A leaf never passes through an operator, so it needs its own coercion."""
    assert eval_exact(num(-6), {}) == Fraction(-6)
    assert isinstance(eval_exact(num(-6), {}), Fraction)


def test_field_evaluation_of_a_bare_numeral_is_reduced() -> None:
    """Otherwise ``-6`` and its equivalent ``p-6`` compare unequal in Python."""
    assert eval_field(num(-6), {}, P) == P - 6
    assert eval_field(sub(num(0), num(6)), {}, P) == P - 6


# ---------------------------------------------------------------------------
# The blindness that motivates the second domain
# ---------------------------------------------------------------------------


def test_field_cannot_distinguish_an_inexact_division_but_exact_can() -> None:
    """The whole reason ``div_both_sides`` needs a guard rather than a fuzz.

    Over 𝔽ₚ, 3 is invertible, so ``3x = 16`` and ``x = 5`` agree on almost every
    draw and disagree only by coincidence. Over ℚ they disagree at x = 5, which
    is the one assignment that matters.
    """
    before, after = eq(mul(num(3), X), num(16)), eq(X, num(5))

    assert holds_exact(before, {VAR_X: 5}) is False
    assert holds_exact(after, {VAR_X: 5}) is True

    # 𝔽ₚ: the "solution" exists, it is just not an integer.
    witness = 16 * pow(3, -1, P) % P
    assert holds_field(before, {VAR_X: witness}, P) is True
    assert holds_field(after, {VAR_X: witness}, P) is False

    # And the honest rewrite agrees in both domains.
    assert holds_exact(eq(mul(num(3), X), num(15)), {VAR_X: 5}) is True
    assert holds_exact(eq(X, num(5)), {VAR_X: 5}) is True


# ---------------------------------------------------------------------------
# Undefined is a third answer, never a silent pass
# ---------------------------------------------------------------------------


def test_division_by_zero_is_none_in_both_domains() -> None:
    assert eval_exact(div(num(1), num(0)), {}) is None
    assert eval_field(div(num(1), num(0)), {}, P) is None
    assert eval_exact(div(num(1), X), {VAR_X: 0}) is None
    assert eval_field(div(num(1), X), {VAR_X: 0}, P) is None


def test_a_vanishing_denominator_mod_p_is_undefined_even_when_nonzero_in_q() -> None:
    """p itself is not zero, but it is zero mod p. The field says so; ℚ does not."""
    assert eval_field(div(num(1), num(P)), {}, P) is None
    assert eval_exact(div(num(1), num(P)), {}) == Fraction(1, P)


def test_undefined_propagates_to_the_equation() -> None:
    assert holds_exact(eq(div(num(1), num(0)), num(1)), {}) is None
    assert holds_field(eq(div(num(1), num(0)), num(1)), {}, P) is None


# ---------------------------------------------------------------------------
# Equations are not values
# ---------------------------------------------------------------------------


def test_evaluating_an_equation_is_a_category_error() -> None:
    equation = eq(X, num(1))
    with pytest.raises(ValueError, match="not a value"):
        eval_exact(equation, {VAR_X: 1})
    with pytest.raises(ValueError, match="not a value"):
        eval_field(equation, {VAR_X: 1}, P)


def test_holds_requires_an_equation() -> None:
    with pytest.raises(ValueError, match="expects an EQ"):
        holds_exact(num(1), {})
    with pytest.raises(ValueError, match="expects an EQ"):
        holds_field(num(1), {}, P)


def test_a_missing_assignment_is_an_error_not_a_default() -> None:
    """A variable silently defaulting to 0 would make a fuzz agree about nothing."""
    with pytest.raises(ValueError, match="no assignment"):
        eval_exact(X, {})


# ---------------------------------------------------------------------------
# Arithmetic, by hand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "env", "expected"),
    [
        (add(num(1), num(2), num(3)), {}, 6),
        (mul(num(2), num(3), num(4)), {}, 24),
        (sub(num(3), num(10)), {}, -7),
        (div(num(15), num(3)), {}, 5),
        (add(mul(num(3), X), num(6)), {VAR_X: 5}, 21),
        (add(mul(num(3), X), mul(num(2), Y)), {VAR_X: 1, VAR_Y: 10}, 23),
    ],
)
def test_arithmetic_agrees_across_domains(expr, env: dict, expected: int) -> None:
    assert eval_exact(expr, env) == expected
    assert eval_field(expr, env, P) == expected % P


# ---------------------------------------------------------------------------
# linear_root — what turns solution-set testing from statistical to deterministic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("equation", "root"),
    [
        (eq(mul(num(3), X), num(15)), Fraction(5)),
        (eq(mul(num(3), X), num(16)), Fraction(16, 3)),  # not an integer, still exact
        (eq(add(mul(num(2), X), num(-4)), num(10)), Fraction(7)),
        (eq(add(mul(num(5), X), num(3)), add(mul(num(2), X), num(18))), Fraction(5)),
        (eq(X, num(0)), Fraction(0)),
    ],
)
def test_linear_root_finds_the_solution(equation, root: Fraction) -> None:
    found = linear_root(equation)
    assert found == root
    assert holds_exact(equation, {VAR_X: found}) is True


@pytest.mark.parametrize(
    "equation",
    [
        eq(mul(num(3), X), mul(num(3), X)),  # every point is a solution
        eq(add(mul(num(3), X), num(1)), add(mul(num(3), X), num(2))),  # parallel, none
        eq(mul(X, Y), num(1)),  # two variables
        eq(mul(X, X), num(4)),  # not linear
        eq(div(num(1), X), num(1)),  # not linear
    ],
)
def test_linear_root_refuses_what_it_cannot_solve(equation) -> None:
    """It is a probe, not a solver. Refusing is correct; guessing would be worse."""
    assert linear_root(equation) is None


def test_linear_root_requires_an_equation() -> None:
    with pytest.raises(ValueError, match="expects an EQ"):
        linear_root(num(1))


def test_the_root_is_where_an_unsound_rewrite_shows_itself() -> None:
    """The whole point, in one case: `3x = 16` vs a floor-divided `x = 5`.

    These differ at exactly one rational point out of infinitely many. Random
    draws find it with probability P(draw hits 5) — measured at 25% over the
    fuzz's ±50 range and **0%** at field width. Evaluating at the root finds it
    every time.
    """
    broken_before, broken_after = eq(mul(num(3), X), num(16)), eq(X, num(5))
    root = linear_root(broken_after)
    assert root == 5
    assert holds_exact(broken_before, {VAR_X: root}) != holds_exact(broken_after, {VAR_X: root})

    honest_before, honest_after = eq(mul(num(3), X), num(15)), eq(X, num(5))
    for probe in (linear_root(honest_before), linear_root(honest_after)):
        assert holds_exact(honest_before, {VAR_X: probe}) == holds_exact(
            honest_after, {VAR_X: probe}
        )


def test_variables_are_reported_in_token_order() -> None:
    assert variables(num(3)) == ()
    assert variables(add(mul(num(2), Y), X)) == (VAR_X, VAR_Y)
