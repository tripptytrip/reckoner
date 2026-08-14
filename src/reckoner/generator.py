"""The procedural problem generator, and the emission grammar it must obey.

Difficulty is parameterised by **verified minimum solution depth**: a candidate
is emitted by a template, and its depth is then *measured* by BFS and used to
bucket it. The template's intended depth is never the label — the survey that
sized this module found `S:ax+bx+c=d` landing on depth 3 two times in 27 and
depth 4 the rest, and `S:ax+b=dx+e` splitting 7/23 between 4 and 5. A shape does
not know its own depth.

The emission grammar
--------------------
Three bans, and each one names a construct the rule set cannot reduce. Emitting
one produces a problem that is unsolvable *by construction* — and a suite of
unsolvable problems is indistinguishable from a suite of hard ones.

1. **No ``DIV``.** There is no ``eval_div`` in v1 (ROUND-02). Already refused by
   ``Problem.__post_init__``.
2. **No variable-containing ``SUB``.** ``eval_sub`` needs both operands numeric,
   the movers ignore ``SUB``, and ``combine_like_terms`` cannot see through one,
   so ``21 − 2x = 3`` and ``3x − x`` are dead ends in the same way a ``DIV`` is.
3. **Numeric-only ``SUB`` is emitted deliberately.** No rule *constructs* a
   ``SUB``; if the generator never emits one either, ``eval_sub`` becomes a dead
   rule whose soundness fuzz passes in a vacuum forever.

Variable policy, per goal
-------------------------
* **EVALUATE** — ground. No variables at all; it has no value otherwise.
* **SOLVE** — exactly one variable, the target. A second unknown is outside v1:
  the rule set has no way to eliminate one, so ``2x + y = 7`` is unsolvable for
  the same reason a ``DIV`` is. Enforced, not assumed.
* **SIMPLIFY** — one or two variables. Two is deliberate: it is the only goal
  that exercises ``combine_like_terms``' unlike-variable case, where terms in
  ``x`` and terms in ``y`` must *not* merge.

Cost, and why the mix matters
-----------------------------
BFS-exact labelling is not uniformly priced. Measured medians on this box:

    EVALUATE / SIMPLIFY, depths 1–6      0.04 ms – 6.6 ms
    SOLVE depth 3                       10 ms
    SOLVE depth 4                       40 ms
    SOLVE depth 5                     1060 ms

The gap is structural rather than incidental: an ``EQ``-rooted state has both
both-sides rules firing at every addend of both sides, so its branching is 6–11
where an expression's is 2. Labelling cost is therefore dominated almost
entirely by deep SOLVE problems, which is why the projection has to be
stratified and why generation is parallel.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace

from reckoner.config import Config
from reckoner.episode import Problem, bfs_par
from reckoner.expr import Expr, Num, Op, Var, add, eq, mul, num, sub, var
from reckoner.semantics import variables
from reckoner.vocab import DIV, GOAL_EVALUATE, GOAL_SIMPLIFY, GOAL_SOLVE, SUB, VAR_X, VAR_Y

X = var(VAR_X)
Y = var(VAR_Y)


class EmissionError(ValueError):
    """A candidate violates the emission grammar and must never be labelled."""


# ---------------------------------------------------------------------------
# The grammar, checked rather than trusted
# ---------------------------------------------------------------------------


def has_variable_sub(expr: Expr) -> bool:
    """True if any ``SUB`` node has a variable anywhere beneath it."""
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Op):
            if node.kind == SUB and variables(node):
                return True
            stack.extend(node.children)
    return False


def check_emission(problem: Problem) -> None:
    """Refuse a candidate the rule set cannot solve. Raises :class:`EmissionError`.

    Called on every candidate *before* it is labelled, because BFS on an
    unsolvable problem is a second, expensive way of finding out.
    """
    from reckoner.expr import tokens

    seq = tokens(problem.expr)
    if DIV in seq:
        raise EmissionError("DIV is irreducible in v1 (ROUND-02)")
    if has_variable_sub(problem.expr):
        raise EmissionError(
            "a SUB node containing a variable is a dead end: eval_sub needs numeric "
            "operands, the movers ignore SUB, and combine_like_terms cannot see "
            "through it"
        )

    names = variables(problem.expr)
    if problem.goal == GOAL_EVALUATE and names:
        raise EmissionError("EVALUATE must be ground")
    if problem.goal == GOAL_SOLVE and names != (problem.target,):
        raise EmissionError(
            f"SOLVE admits exactly one variable, the target: v1 has no way to "
            f"eliminate a second unknown (found {names})"
        )
    if problem.goal == GOAL_SIMPLIFY and len(names) > 2:
        raise EmissionError(f"SIMPLIFY admits at most two variables (found {names})")


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

Template = Callable[[random.Random], Problem]


def _nz(rng: random.Random, lo: int = -9, hi: int = 10, *, exclude: tuple[int, ...] = ()) -> int:
    return rng.choice([c for c in range(lo, hi) if c != 0 and c not in exclude])


def _n(rng: random.Random, lo: int = -9, hi: int = 10) -> int:
    return rng.randrange(lo, hi)


def _solve(expr: Expr) -> Problem:
    return Problem(goal=GOAL_SOLVE, expr=expr, target=VAR_X)


# --- EVALUATE: cheap to label, and the only family that reaches depth 6 -----


def t_eval_sum(rng: random.Random) -> Problem:
    return Problem(goal=GOAL_EVALUATE, expr=add(num(_n(rng, -99, 100)), num(_n(rng, -99, 100))))


def t_eval_sub(rng: random.Random) -> Problem:
    """Numeric-only SUB, emitted on purpose — otherwise eval_sub is a dead rule."""
    return Problem(goal=GOAL_EVALUATE, expr=sub(num(_n(rng, -99, 100)), num(_n(rng, -99, 100))))


def t_eval_product(rng: random.Random) -> Problem:
    return Problem(
        goal=GOAL_EVALUATE, expr=add(mul(num(_nz(rng)), num(_n(rng))), num(_n(rng, -40, 41)))
    )


def t_eval_mixed(rng: random.Random) -> Problem:
    return Problem(
        goal=GOAL_EVALUATE,
        expr=add(mul(num(_nz(rng)), num(_n(rng))), sub(num(_n(rng)), num(_n(rng))), num(_n(rng))),
    )


def t_eval_deep(rng: random.Random) -> Problem:
    return Problem(
        goal=GOAL_EVALUATE,
        expr=add(
            mul(num(_nz(rng)), num(_n(rng))),
            mul(num(_nz(rng)), num(_n(rng))),
            mul(num(_nz(rng)), num(_n(rng))),
            sub(num(_n(rng)), num(_n(rng))),
            num(_n(rng)),
        ),
    )


def t_eval_deepest(rng: random.Random) -> Problem:
    return Problem(
        goal=GOAL_EVALUATE,
        expr=add(
            *(mul(num(_nz(rng)), num(_n(rng))) for _ in range(4)),
            sub(num(_n(rng)), num(_n(rng))),
            num(_n(rng)),
        ),
    )


# --- SIMPLIFY: the only goal that exercises the unlike-variable case --------


def t_simplify_like(rng: random.Random) -> Problem:
    return Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(_nz(rng)), X), mul(num(_nz(rng)), X)))


def t_simplify_with_constants(rng: random.Random) -> Problem:
    return Problem(
        goal=GOAL_SIMPLIFY,
        expr=add(mul(num(_nz(rng)), X), mul(num(_nz(rng)), X), num(_n(rng)), num(_n(rng))),
    )


def t_simplify_two_vars(rng: random.Random) -> Problem:
    """x-terms and y-terms must NOT merge — combine_like_terms' negative case."""
    return Problem(
        goal=GOAL_SIMPLIFY,
        expr=add(
            mul(num(_nz(rng)), X),
            mul(num(_nz(rng)), X),
            mul(num(_nz(rng)), Y),
            mul(num(_nz(rng)), Y),
            num(_n(rng)),
        ),
    )


def t_simplify_with_products(rng: random.Random) -> Problem:
    return Problem(
        goal=GOAL_SIMPLIFY,
        expr=add(
            mul(num(_nz(rng)), X),
            mul(num(_nz(rng)), X),
            mul(num(_nz(rng)), num(_n(rng))),
            mul(num(_nz(rng)), num(_n(rng))),
            num(_n(rng)),
        ),
    )


# --- SOLVE: expensive to label, and the point of the domain ----------------


def t_solve_coefficient(rng: random.Random) -> Problem:
    a, k = _nz(rng), _n(rng, -12, 13)
    return _solve(eq(mul(num(a), X), num(a * k)))


def t_solve_constant(rng: random.Random) -> Problem:
    a, k, b = _nz(rng), _n(rng, -12, 13), _n(rng, -20, 21)
    return _solve(eq(add(mul(num(a), X), num(b)), num(a * k + b)))


def t_solve_two_terms(rng: random.Random) -> Problem:
    a = _nz(rng)
    d = _nz(rng, exclude=(a, -a))
    k, b = _n(rng, -12, 13), _n(rng, -20, 21)
    return _solve(eq(add(mul(num(a), X), mul(num(d), X), num(b)), num((a + d) * k + b)))


def t_solve_product_rhs(rng: random.Random) -> Problem:
    """``(ap)x = p·q`` — one evaluation, then one division. Reaches depth 2.

    Added because the first suite generation produced a `solve_in_2` with **no
    SOLVE problems at all**: no v1 template landed a SOLVE on depth 2, so the
    stratum measured arithmetic only. A depth suite whose goal mix is an accident
    of which templates happened to exist is not an instrument.
    """
    a, p, q = _nz(rng), _nz(rng), _n(rng)
    return _solve(eq(mul(num(a * p), X), mul(num(p), num(a * q))))


def t_solve_both_sides_product(rng: random.Random) -> Problem:
    """x on both sides with an unevaluated right-hand constant. Reaches depth 6.

    Same reason as above, at the other end: `solve_in_6` came out 200/200
    EVALUATE, so the deepest suite — the one that matters most for measuring
    whether the model can *solve* rather than *compute* — contained no equations
    at all. This is the most expensive template in the set (~4.5 s median to
    label, 8.8 s worst); that cost is the price of the stratum being real.
    """
    a = _nz(rng)
    d = _nz(rng, exclude=(a,))
    k, p, q = _n(rng, -9, 10), _nz(rng), _n(rng)
    e = p * q
    return _solve(
        eq(add(mul(num(a), X), num(e + (a - d) * k)), add(mul(num(d), X), mul(num(p), num(q))))
    )


def t_solve_both_sides(rng: random.Random) -> Problem:
    a = _nz(rng)
    d = _nz(rng, exclude=(a,))
    k, e = _n(rng, -12, 13), _n(rng, -20, 21)
    return _solve(eq(add(mul(num(a), X), num(e + (a - d) * k)), add(mul(num(d), X), num(e))))


#: Every template, with the depths the pre-flight survey measured for it. The
#: comment is a *record of a measurement*, not an instruction to the labeller —
#: the label always comes from BFS.
TEMPLATES: dict[str, Template] = {
    "eval_sum": t_eval_sum,  # 1
    "eval_sub": t_eval_sub,  # 1
    "eval_product": t_eval_product,  # 2
    "eval_mixed": t_eval_mixed,  # 3
    "eval_deep": t_eval_deep,  # 4-5
    "eval_deepest": t_eval_deepest,  # 5-6
    "simplify_like": t_simplify_like,  # 1
    "simplify_with_constants": t_simplify_with_constants,  # 2
    "simplify_two_vars": t_simplify_two_vars,  # 2-3
    "simplify_with_products": t_simplify_with_products,  # 3-4
    "solve_coefficient": t_solve_coefficient,  # 1
    "solve_constant": t_solve_constant,  # 3
    "solve_two_terms": t_solve_two_terms,  # 3-4
    "solve_both_sides": t_solve_both_sides,  # 4-5
    "solve_product_rhs": t_solve_product_rhs,  # 2
    "solve_both_sides_product": t_solve_both_sides_product,  # 6
}

#: Templates whose labelling is cheap (no ``EQ`` root, so branching ~2). Kept
#: separate because a generation plan that ignores the split is a generation
#: plan whose runtime is decided by accident.
CHEAP_TEMPLATES = tuple(name for name in TEMPLATES if not name.startswith("solve"))
COSTLY_TEMPLATES = tuple(name for name in TEMPLATES if name.startswith("solve"))


# ---------------------------------------------------------------------------
# Emission + labelling
# ---------------------------------------------------------------------------


def emit(template: str, rng: random.Random) -> Problem:
    """One unlabelled candidate, grammar-checked. Par is absent, not zero."""
    problem = TEMPLATES[template](rng)
    check_emission(problem)
    return problem


def label(problem: Problem, cfg: Config | None = None, cap: int | None = None) -> Problem | None:
    """Attach a BFS-exact par, or return ``None`` beyond the horizon.

    ``None`` rather than a scripted fallback: this generator emits only within
    the BFS-exact band, and a problem it cannot label exactly is a problem it
    does not emit. Scripted par is for the middle strata a later round adds.
    """
    par = bfs_par(problem, cfg or Config(), cap)
    if par is None or par < 1:
        # par 0 means the candidate was already in goal form: a degenerate
        # problem, not a hard one. Rejected here rather than shipped as a free +0.
        return None
    return replace(problem, par=par, par_source="bfs")


def generate(
    template: str, rng: random.Random, cfg: Config | None = None, cap: int | None = None
) -> Problem | None:
    """Emit one candidate and label it. ``None`` if it is not usable."""
    return label(emit(template, rng), cfg, cap)


def numerals(expr: Expr) -> list[int]:
    out: list[int] = []
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Num):
            out.append(node.value)
        elif isinstance(node, Op):
            stack.extend(node.children)
    return out


def variable_tokens(expr: Expr) -> tuple[int, ...]:
    seen: set[int] = set()
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Var):
            seen.add(node.token)
        elif isinstance(node, Op):
            stack.extend(node.children)
    return tuple(sorted(seen))
