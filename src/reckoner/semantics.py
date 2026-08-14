"""What an expression *means*: evaluation over 𝔽ₚ and over ℚ.

Two domains, because one of them is blind to something the project depends on.

**𝔽ₚ** is the domain the spec pins (§3: random-assignment equivalence, k=32
draws over a prime field) and the one chunk 3's SIMPLIFY checker uses. It is
cheap, and probabilistic equivalence over a large prime is about as strong a
same-function test as exists.

**ℚ** is exact, and it sees one thing 𝔽ₚ structurally cannot: **every nonzero
constant is invertible mod p**. So `3x = 16` and `x = 5` are *not* equivalent
over ℚ — they disagree at x = 5 — while over 𝔽ₚ the first has the perfectly
good solution x = 16·3⁻¹ and a broken exactness guard on ``div_both_sides``
sails through field-equivalence forever. Anything asserting that a rewrite
preserved an *integer* problem has to ask ℚ, not 𝔽ₚ.

Undefined is a third answer, not an exception. Division by zero — actual zero in
ℚ, or a denominator ≡ 0 (mod p) in the field — returns ``None``. Callers must
**skip and count** those assignments rather than pass them: a fuzz that silently
treats "couldn't evaluate" as "agreed" reports 10,000 passing draws when it
evaluated twelve. That is the vacuity failure this whole module is shaped to
make visible.

Equations are not values. ``eval_*`` raises on an ``EQ`` node; ask ``holds_*``
for an equation's truth value under an assignment instead.
"""

from __future__ import annotations

from fractions import Fraction

from reckoner.expr import Expr, Num, Op, Var
from reckoner.vocab import ADD, DIV, EQ, MUL, SUB, token_name


class Undefined(Exception):
    """Raised internally when a subexpression has no value; surfaced as ``None``."""


def _fold(expr: Expr, env: dict[int, object], combine, leaf) -> object | None:
    """Post-order evaluation with an explicit stack (no recursion depth limit).

    ``combine(kind, values)`` does the arithmetic for one operator node and may
    raise :class:`Undefined`. ``leaf`` coerces a numeral into the domain's own
    number type **before** any arithmetic touches it — which is not a nicety:
    left as raw Python ints, ``DIV`` evaluates ``-1 / -24`` with *float*
    division and silently contaminates the whole expression with binary
    rationals. 𝔽ₚ is immune to that by construction (all integer arithmetic),
    so it is precisely the kind of defect only the exact domain can see.
    """
    stack: list[tuple[Expr, bool]] = [(expr, False)]
    done: list[object] = []
    while stack:
        node, expanded = stack.pop()
        if isinstance(node, Op):
            if node.kind == EQ:
                raise ValueError(
                    "an equation is not a value: EQ has a truth value, not a number. "
                    "Use holds_field/holds_exact."
                )
            if not expanded:
                stack.append((node, True))
                for child in reversed(node.children):
                    stack.append((child, False))
            else:
                n = len(node.children)
                values = done[len(done) - n :]
                del done[len(done) - n :]
                try:
                    done.append(combine(node.kind, values))
                except Undefined:
                    return None
        elif isinstance(node, Num):
            done.append(leaf(node.value))
        elif isinstance(node, Var):
            if node.token not in env:
                raise ValueError(f"no assignment for {token_name(node.token)}")
            done.append(env[node.token])
        else:
            raise TypeError(f"not an expression node: {node!r}")
    return done[0]


# ---------------------------------------------------------------------------
# 𝔽ₚ
# ---------------------------------------------------------------------------


def eval_field(expr: Expr, env: dict[int, int], p: int) -> int | None:
    """Value of ``expr`` in 𝔽ₚ, or ``None`` if a denominator vanished mod p."""

    def combine(kind: int, values: list[int]) -> int:
        if kind == ADD:
            return sum(values) % p
        if kind == MUL:
            out = 1
            for v in values:
                out = out * v % p
            return out
        if kind == SUB:
            return (values[0] - values[1]) % p
        if kind == DIV:
            if values[1] % p == 0:
                raise Undefined
            return values[0] * pow(values[1], -1, p) % p
        raise ValueError(f"cannot evaluate {token_name(kind)}")

    # Leaves are reduced on the way in, so a bare ``Num`` expression comes back
    # as p-6 rather than -6 — equal in 𝔽ₚ, unequal in Python, and a spurious
    # fuzz failure at best.
    value = _fold(expr, {k: v % p for k, v in env.items()}, combine, lambda v: v % p)
    return None if value is None else value % p  # type: ignore[operator,return-value]


def holds_field(equation: Expr, env: dict[int, int], p: int) -> bool | None:
    """Truth value of an equation in 𝔽ₚ; ``None`` if either side is undefined."""
    if not (isinstance(equation, Op) and equation.kind == EQ):
        raise ValueError("holds_field expects an EQ node")
    left = eval_field(equation.children[0], env, p)
    right = eval_field(equation.children[1], env, p)
    if left is None or right is None:
        return None
    return left == right


# ---------------------------------------------------------------------------
# ℚ — exact
# ---------------------------------------------------------------------------


def eval_exact(expr: Expr, env: dict[int, Fraction | int]) -> Fraction | None:
    """Exact rational value of ``expr``, or ``None`` on division by zero."""

    def combine(kind: int, values: list[Fraction]) -> Fraction:
        if kind == ADD:
            return sum(values, Fraction(0))
        if kind == MUL:
            out = Fraction(1)
            for v in values:
                out *= v
            return out
        if kind == SUB:
            return values[0] - values[1]
        if kind == DIV:
            if values[1] == 0:
                raise Undefined
            return values[0] / values[1]
        raise ValueError(f"cannot evaluate {token_name(kind)}")

    value = _fold(expr, {k: Fraction(v) for k, v in env.items()}, combine, Fraction)
    return None if value is None else Fraction(value)  # type: ignore[arg-type]


def holds_exact(equation: Expr, env: dict[int, Fraction | int]) -> bool | None:
    """Truth value of an equation over ℚ; ``None`` if either side is undefined.

    This is the layer that can see an inexact division. Over ℚ, ``3x = 16`` is
    false at every integer x, while ``x = 5`` is true at x = 5 — so a
    ``div_both_sides`` that fired without its exactness guard disagrees here,
    and only here.
    """
    if not (isinstance(equation, Op) and equation.kind == EQ):
        raise ValueError("holds_exact expects an EQ node")
    left = eval_exact(equation.children[0], env)
    right = eval_exact(equation.children[1], env)
    if left is None or right is None:
        return None
    return left == right


def variables(expr: Expr) -> tuple[int, ...]:
    """Variable tokens occurring in ``expr``, in ascending token order."""
    seen: set[int] = set()
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Var):
            seen.add(node.token)
        elif isinstance(node, Op):
            stack.extend(node.children)
    return tuple(sorted(seen))
