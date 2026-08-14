"""The rule engine — this domain's movegen.

Actions are ``(rule_id, site_id)``: a rewrite rule applied at a subtree
position, the direct analog of ``(from, to)`` squares. Legality is a pattern
match, and the rule set is closed, versioned, and sound by construction.

Rule set v1 (plan §8 decision 2, pinned — extensions are later one-lever rounds)::

    0  eval_add            1  eval_sub            2  eval_mul
    3  combine_like_terms  4  add_both_sides      5  sub_both_sides
    6  div_both_sides

Site enumeration order
----------------------
**Pre-order depth-first over the canonical tree, children in canonical order,
root = site 0.** Documented because downstream pins depend on it: chunk 6's
policy head masks over site ids, chunk 5's datasets store them, and a change to
this order silently relabels every action in every dataset written before it.

**Site ids are valid only for the state they were enumerated from.** ``apply``
re-canonicalises, and canonicalisation reorders children (C5–C7) and collapses
singletons (C4), so the node at site 7 before a rewrite is generally not the
node at site 7 after one. An action is a pair naming a rule and a position *in
the current state*; it is not a durable handle on a subtree.

Shape of a rule
---------------
``Rule = (name, lhs, rhs, guard)`` as the plan specifies, with each part a
callable over ``(state, site)``:

* ``lhs`` — the left-hand-side pattern: does this rule's shape occur here?
* ``guard`` — the side condition, kept separate so it can be tested alone.
* ``rhs`` — the right-hand-side template, instantiated at the match.

The patterns are predicates rather than term-patterns with sequence variables,
and that is a deliberate trade. ``ADD`` and ``MUL`` are variadic and canonically
sorted, so ``eval_add``'s LHS is "two or more numeric children among any
others" — a term-pattern language expressive enough to say that would be more
machinery, and more surface to be wrong in, than seven explicit predicates.
The independent brute-force reference matcher in the tests is what makes that
trade safe: it re-derives every legal action from the rule descriptions with its
own traversal and its own predicates, and the two must agree exactly.

No SUB node ever enters a rewrite
---------------------------------
``sub_both_sides`` moves an addend across as its negation; it never builds
``L − e = R − e``. That is what makes the set *closed*: ``eval_sub`` needs both
operands numeric, so ``(3x + 6) − 6`` would be a dead end that no rule in v1 can
reduce — there is neither distribution nor an identity rule to rescue it. A
negated addend lands inside the same flattened ``ADD``, where the existing rules
reach it.

    3x + 6 = 21
      ──[sub_both_sides, operand 6]──►   3x = 21 + (−6)
      ──[eval_add, right side]──►        3x = 15
      ──[div_both_sides, operand 3]──►   x = 5

That is BFS-exact par 3, and ``tests/test_rules.py`` pins it — it is the closure
demonstration for the whole rule set, not an illustration.

``add_both_sides`` is reachability-redundant
---------------------------------------------
It is in the pinned set (plan §8 decision 2) and it is sound, but it only ever
*grows* both sides, so no problem needs it: any derivation using it has a
shorter one without it. It is kept because the rule set is pinned, and its cost
is measured rather than assumed — see the chunk report's branching table. It is
the obvious candidate for a one-lever removal round.

Structural rules defer arithmetic; ``div_both_sides`` computes
---------------------------------------------------------------
The asymmetry is deliberate and it is *forced*, not stylistic. The movers
(``add_both_sides``, ``sub_both_sides``) rearrange and leave every sum for
``eval_add`` to compute, so each rendered line is semantically atomic.
``div_both_sides`` cannot do the same: deferring its quotient would mean
emitting a ``DIV`` node, and there is no ``eval_div`` in v1 to reduce it — the
state would be a dead end. So it computes ``c // a`` inside its exactness guard.

Every rendered derivation line is therefore one of two kinds — a structural move
or one arithmetic step — with exactly one exception, named here so chunk 4's
interpreter does not have to rediscover it.

No reachable v1 state contains a DIV node
------------------------------------------
Follows from the above: no rule's RHS template ever constructs a ``DIV``, and
``div_both_sides`` consumes its division rather than emitting one. So if a
*problem* contains no ``DIV``, no state reachable from it does either. Chunk 5's
generator is therefore barred from emitting ``DIV`` in v1 problems — and with
that ban, the invariant holds over the whole reachable state space.

The invariant is a **license**: division is what makes field evaluation
partial (a denominator vanishing mod p), so a state space with no ``DIV`` in it
can be compared by field-only equivalence with no undefined draws to skip.
Chunk 3's SIMPLIFY checker depends on that, so the invariant and the licence are
tested as a pair rather than assumed.

``eval_div`` plus a structural ``div_both_sides`` is registered as its own
candidate one-lever round — see ``REGISTERED-ROUNDS.md`` ROUND-02. Adding an eighth rule
is a spec change, not an implementation detail.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

from reckoner.expr import Expr, Num, Op, canonicalize, identity_key, make_op
from reckoner.vocab import ADD, EQ, MUL, SUB

#: Bump when a rule id changes meaning or a rule is removed. Datasets store rule
#: ids; a silent renumbering relabels every recorded action.
RULESET_VERSION = 1


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Site:
    """A position in a canonical tree: its pre-order id, path, and node."""

    site_id: int
    path: tuple[int, ...]
    node: Expr


def enumerate_sites(state: Expr) -> list[Site]:
    """Every subtree position, in pre-order with children in canonical order.

    Iterative, so tree depth is an input rather than a limit.
    """
    sites: list[Site] = []
    stack: list[tuple[Expr, tuple[int, ...]]] = [(state, ())]
    while stack:
        node, path = stack.pop()
        sites.append(Site(len(sites), path, node))
        if isinstance(node, Op):
            for index in range(len(node.children) - 1, -1, -1):
                stack.append((node.children[index], (*path, index)))
    return sites


def node_at(state: Expr, path: tuple[int, ...]) -> Expr:
    node = state
    for index in path:
        assert isinstance(node, Op)
        node = node.children[index]
    return node


def replace_at(state: Expr, path: tuple[int, ...], new: Expr) -> Expr:
    """Rebuild ``state`` with the subtree at ``path`` replaced, re-canonicalising."""
    if not path:
        return canonicalize(new)
    ancestors: list[Op] = []
    node = state
    for index in path:
        assert isinstance(node, Op)
        ancestors.append(node)
        node = node.children[index]
    current = new
    for depth in range(len(path) - 1, -1, -1):
        parent = ancestors[depth]
        kids = list(parent.children)
        kids[path[depth]] = current
        current = make_op(parent.kind, kids)
    return current


# ---------------------------------------------------------------------------
# Term helpers
# ---------------------------------------------------------------------------


def split_coefficient(expr: Expr) -> tuple[int, tuple[Expr, ...]]:
    """Split a term into ``(coefficient, remaining factors)``.

    ``3x`` → ``(3, (x,))``; ``x`` → ``(1, (x,))``; ``7`` → ``(7, ())``;
    ``a − b`` → ``(1, (a − b,))``. The remaining factors identify the term for
    like-term grouping; the coefficient is what gets summed.
    """
    if isinstance(expr, Num):
        return expr.value, ()
    if isinstance(expr, Op) and expr.kind == MUL and isinstance(expr.children[0], Num):
        return expr.children[0].value, expr.children[1:]
    return 1, (expr,)


def scaled(coefficient: int, factors: tuple[Expr, ...]) -> Expr:
    """Rebuild ``coefficient × factors``, applying the arithmetic identities.

    ``0 × f`` is ``0``, ``1 × f`` is ``f``. These are the identities that make
    rule set v1 closed without a separate identity rule: without them,
    ``3x + (−3x)`` would stall at ``0x`` and ``x`` would stall at ``1x``.
    """
    if not factors:
        return Num(coefficient)
    if coefficient == 0:
        return Num(0)
    if coefficient == 1:
        return make_op(MUL, factors) if len(factors) > 1 else factors[0]
    return make_op(MUL, (Num(coefficient), *factors))


def negate(expr: Expr) -> Expr:
    """``−expr``, expressed inside the vocabulary (there is no unary minus)."""
    coefficient, factors = split_coefficient(expr)
    return scaled(-coefficient, factors)


def _numeric_children(node: Expr) -> list[Num]:
    if not isinstance(node, Op):
        return []
    return [c for c in node.children if isinstance(c, Num)]


def _equation_sides(state: Expr) -> tuple[Expr, Expr] | None:
    if isinstance(state, Op) and state.kind == EQ:
        return state.children[0], state.children[1]
    return None


def _addend_paths(state: Expr) -> set[tuple[int, ...]]:
    """Paths of the top-level addends of each side of an equation.

    A side that is an ``ADD`` contributes its children; any other side
    contributes itself. These are the only operands the both-sides rules accept:
    adding an arbitrary deep subterm to both sides is sound but is not a move
    anyone makes, and every extra legal action is branching the search pays for.
    """
    if _equation_sides(state) is None:
        return set()
    assert isinstance(state, Op)
    paths: set[tuple[int, ...]] = set()
    for side_index, side in enumerate(state.children):
        if isinstance(side, Op) and side.kind == ADD:
            for child_index in range(len(side.children)):
                paths.add((side_index, child_index))
        else:
            paths.add((side_index,))
    return paths


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

Predicate = Callable[[Expr, Site], bool]
Template = Callable[[Expr, Site], Expr]


def _always(state: Expr, site: Site) -> bool:  # noqa: ARG001
    return True


@dataclass(frozen=True)
class Rule:
    """``(name, lhs pattern, rhs template, guard)``.

    ``scope`` says what the site means: ``"node"`` rules rewrite the subtree at
    the site; ``"equation"`` rules rewrite the whole equation, using the site to
    name their operand.
    """

    rule_id: int
    name: str
    scope: str
    lhs: Predicate
    rhs: Template
    guard: Predicate = field(default=_always)

    def legal(self, state: Expr, site: Site) -> bool:
        return self.lhs(state, site) and self.guard(state, site)

    def apply(self, state: Expr, site: Site) -> Expr:
        if not self.legal(state, site):
            raise ValueError(f"{self.name} is not legal at site {site.site_id}")
        rewritten = self.rhs(state, site)
        if self.scope == "equation":
            return canonicalize(rewritten)
        return replace_at(state, site.path, rewritten)


# --- eval_add ---------------------------------------------------------------


def _lhs_eval_add(state: Expr, site: Site) -> bool:  # noqa: ARG001
    node = site.node
    return isinstance(node, Op) and node.kind == ADD and len(_numeric_children(node)) >= 2


def _rhs_eval_add(state: Expr, site: Site) -> Expr:  # noqa: ARG001
    node = site.node
    assert isinstance(node, Op)
    others = [c for c in node.children if not isinstance(c, Num)]
    total = sum(c.value for c in _numeric_children(node))
    if total == 0 and others:  # additive identity
        return make_op(ADD, others)
    return make_op(ADD, [*others, Num(total)])


# --- eval_sub ---------------------------------------------------------------


def _lhs_eval_sub(state: Expr, site: Site) -> bool:  # noqa: ARG001
    node = site.node
    return (
        isinstance(node, Op) and node.kind == SUB and all(isinstance(c, Num) for c in node.children)
    )


def _rhs_eval_sub(state: Expr, site: Site) -> Expr:  # noqa: ARG001
    node = site.node
    assert isinstance(node, Op)
    left, right = node.children
    assert isinstance(left, Num) and isinstance(right, Num)
    return Num(left.value - right.value)


# --- eval_mul ---------------------------------------------------------------


def _lhs_eval_mul(state: Expr, site: Site) -> bool:  # noqa: ARG001
    node = site.node
    return isinstance(node, Op) and node.kind == MUL and len(_numeric_children(node)) >= 2


def _rhs_eval_mul(state: Expr, site: Site) -> Expr:  # noqa: ARG001
    node = site.node
    assert isinstance(node, Op)
    others = tuple(c for c in node.children if not isinstance(c, Num))
    product = 1
    for child in _numeric_children(node):
        product *= child.value
    return scaled(product, others)


# --- combine_like_terms -----------------------------------------------------


def _like_groups(node: Expr) -> dict[tuple, list[Expr]]:
    """Group an ADD's non-numeric children by their variable part.

    Pure numerals are excluded on purpose: summing those is ``eval_add``'s job,
    and two rules that produce the same successor from the same site would be
    duplicate actions in the policy's action space.
    """
    groups: dict[tuple, list[Expr]] = {}
    if not (isinstance(node, Op) and node.kind == ADD):
        return groups
    for child in node.children:
        if isinstance(child, Num):
            continue
        _, factors = split_coefficient(child)
        key = tuple(identity_key(f) for f in factors)
        groups.setdefault(key, []).append(child)
    return groups


def _lhs_combine_like_terms(state: Expr, site: Site) -> bool:  # noqa: ARG001
    return any(len(members) >= 2 for members in _like_groups(site.node).values())


def _rhs_combine_like_terms(state: Expr, site: Site) -> Expr:  # noqa: ARG001
    node = site.node
    assert isinstance(node, Op)
    groups = _like_groups(node)
    out: list[Expr] = [c for c in node.children if isinstance(c, Num)]
    for members in groups.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        total = 0
        factors: tuple[Expr, ...] = ()
        for member in members:
            coefficient, factors = split_coefficient(member)
            total += coefficient
        out.append(scaled(total, factors))
    return make_op(ADD, out)


# --- add_both_sides / sub_both_sides ---------------------------------------


def _lhs_both_sides(state: Expr, site: Site) -> bool:
    return site.path in _addend_paths(state)


def _rhs_add_both_sides(state: Expr, site: Site) -> Expr:
    assert isinstance(state, Op)
    operand = site.node
    return make_op(EQ, [make_op(ADD, [side, operand]) for side in state.children])


def _rhs_sub_both_sides(state: Expr, site: Site) -> Expr:
    """``A + e = B``  ⟹  ``A = B + (−e)``. Subtract the addend from both sides.

    The cancellation on ``e``'s own side is **structural** — the addend is
    removed from the sum, not computed away. That is what makes this one rule
    rather than three: no arithmetic is performed, only the two identities that
    define what subtracting does (``e − e = 0`` and ``A + 0 = A``), and neither
    of them adds two unequal numbers. Leaving ``A + e + (−e)`` for ``eval_add``
    to notice instead would be *finer* than the spec's own worked example, and
    measurably worse: it costs two extra plies on every cancellation, which put
    ``5x + 3 = 2x + 18`` at par 7 — outside the depth-6 BFS-exact band that
    canonical par depends on.
    """
    assert isinstance(state, Op)
    side_index = site.path[0]
    sides = list(state.children)

    if len(site.path) == 1:  # the whole side is the addend
        sides[side_index] = Num(0)
    else:
        side = sides[side_index]
        assert isinstance(side, Op)
        child_index = site.path[1]
        # Remove this occurrence by index, not by value: a side may hold the
        # same addend twice, and cancelling both would not be subtraction.
        remaining = [c for i, c in enumerate(side.children) if i != child_index]
        sides[side_index] = make_op(ADD, remaining)

    other = 1 - side_index
    sides[other] = make_op(ADD, [sides[other], negate(site.node)])
    return make_op(EQ, sides)


# --- div_both_sides ---------------------------------------------------------


def _div_parts(state: Expr, site: Site) -> tuple[int, tuple[Expr, ...], int] | None:
    """``(coefficient, remaining factors, rhs value)`` if the shape matches here.

    The site must be the coefficient numeral of the variable-bearing side. C7
    puts that side first, so "the variable side" is a structural fact, not a
    convention the rule has to re-derive.
    """
    sides = _equation_sides(state)
    if sides is None or site.path != (0, 0):
        return None
    left, right = sides
    if not (isinstance(left, Op) and left.kind == MUL and isinstance(left.children[0], Num)):
        return None
    if not isinstance(right, Num):
        return None
    return left.children[0].value, left.children[1:], right.value


def _lhs_div_both_sides(state: Expr, site: Site) -> bool:
    parts = _div_parts(state, site)
    return parts is not None and parts[0] != 0


def _guard_div_both_sides(state: Expr, site: Site) -> bool:
    """Exact integer division only — the closure condition.

    This guard is **not** a soundness condition over a field: every nonzero
    constant is invertible mod p, so ``3x = 16 ⇒ x = 16·3⁻¹`` is a perfectly
    valid field rewrite and the 𝔽ₚ fuzz would wave it through. What the guard
    protects is *closure of the vocabulary*: there are no fractions in rule set
    v1, so a rewrite whose honest answer is 16/3 has no representation, and
    firing anyway would mean writing down some other number instead.
    """
    parts = _div_parts(state, site)
    if parts is None:
        return False
    coefficient, _, rhs_value = parts
    return coefficient != 0 and rhs_value % coefficient == 0


def _rhs_div_both_sides(state: Expr, site: Site) -> Expr:
    parts = _div_parts(state, site)
    assert parts is not None
    coefficient, factors, rhs_value = parts
    return make_op(EQ, [scaled(1, factors), Num(rhs_value // coefficient)])


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------

RULES: tuple[Rule, ...] = (
    Rule(0, "eval_add", "node", _lhs_eval_add, _rhs_eval_add),
    Rule(1, "eval_sub", "node", _lhs_eval_sub, _rhs_eval_sub),
    Rule(2, "eval_mul", "node", _lhs_eval_mul, _rhs_eval_mul),
    Rule(3, "combine_like_terms", "node", _lhs_combine_like_terms, _rhs_combine_like_terms),
    Rule(4, "add_both_sides", "equation", _lhs_both_sides, _rhs_add_both_sides),
    Rule(5, "sub_both_sides", "equation", _lhs_both_sides, _rhs_sub_both_sides),
    Rule(
        6,
        "div_both_sides",
        "equation",
        _lhs_div_both_sides,
        _rhs_div_both_sides,
        _guard_div_both_sides,
    ),
)

RULE_BY_ID: dict[int, Rule] = {rule.rule_id: rule for rule in RULES}
RULE_BY_NAME: dict[str, Rule] = {rule.name: rule for rule in RULES}


def rule_set_fingerprint() -> str:
    """SHA-256 over ``(id, name, scope)`` rows. A renumbering changes this."""
    canonical = "\n".join(f"{r.rule_id}\t{r.name}\t{r.scope}" for r in RULES)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# The movegen API
# ---------------------------------------------------------------------------


def legal_actions(state: Expr) -> list[tuple[int, int]]:
    """Every legal ``(rule_id, site_id)``, sorted by rule then site.

    Precondition: ``state`` is canonical. The order is part of the interface —
    masks, datasets and pinned expectations downstream are indexed by it.
    """
    sites = enumerate_sites(state)
    return [
        (rule.rule_id, site.site_id) for rule in RULES for site in sites if rule.legal(state, site)
    ]


def apply(state: Expr, rule_id: int, site_id: int) -> Expr:
    """Apply an action, returning the canonical successor state."""
    sites = enumerate_sites(state)
    if not 0 <= site_id < len(sites):
        raise ValueError(f"site {site_id} does not exist ({len(sites)} sites)")
    if rule_id not in RULE_BY_ID:
        raise ValueError(f"no rule with id {rule_id}")
    return RULE_BY_ID[rule_id].apply(state, sites[site_id])


def successors(state: Expr) -> list[tuple[tuple[int, int], Expr]]:
    """Every legal action paired with the canonical state it produces."""
    sites = enumerate_sites(state)
    out: list[tuple[tuple[int, int], Expr]] = []
    for rule in RULES:
        for site in sites:
            if rule.legal(state, site):
                out.append(((rule.rule_id, site.site_id), rule.apply(state, site)))
    return out
