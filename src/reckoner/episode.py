"""Episodes, the checker, and the par game's win condition.

A position is ``(expression, goal)``. The goal is carried as **prefix tokens in
the state** (plan §8 decision 3) — no architecture change, no conditioning
channel::

    EVALUATE   [GOAL_EVALUATE,          SEP, …expression…]
    SOLVE(x)   [GOAL_SOLVE,     VAR_X,  SEP, …expression…]
    SIMPLIFY   [GOAL_SIMPLIFY,          SEP, …expression…]

The checker is the only source of truth
---------------------------------------
Rules are sound by construction, so nothing reachable by legal play can be
wrong. That is exactly why the checker must be written and tested **against
claims it did not produce** — its job is not to audit our rules, it is to be the
independent arbiter the whole method rests on. It verifies a claimed final state
against the *original problem*, never against the derivation.

  * **EVALUATE** — exact evaluation. The problem must be ground; the claim must
    be a numeral equal to it.
  * **SOLVE(x)** — substitution into the original equation. Not "does it look
    like an answer", but "does it make the original true".
  * **SIMPLIFY** — random-assignment equivalence, k = 32 draws over a prime
    field (spec §3).

Why SIMPLIFY can use field-only equivalence
-------------------------------------------
Division is the only partial operation — it is the sole reason ``eval_field``
ever returns "undefined". Chunk 2 established that **no reachable v1 state
contains a DIV node** (no rule's template constructs one, and the generator is
barred from emitting one), so over this state space every draw is informative
and there are no undefined results to skip and count. The invariant is the
licence; ``verify`` raises rather than skips if it ever meets an undefined draw,
because that would mean the invariant broke, not that this draw was unlucky.

The win condition (amendment v1.1)
----------------------------------
``z ∈ {+1 strictly under par, 0 equal, −1 over par or step cap}``. Par is the
step count of a reference solution **denominated in this rule system**, so
``ruleset_version`` is part of the label and is the first field of every result.
A granularity change bumps it and invalidates recorded pars loudly.

The cap is a *loss*, not an abstention — an episode does not get to end in a
shrug. But reaching the cap and solving on that very step is a solve: the cap
bounds how many steps you may take, not how many you may be credited with.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction

from reckoner.config import Config
from reckoner.expr import Expr, Num, Op, Var, canonicalize, identity_key, parse, tokens
from reckoner.rules import RULESET_VERSION, apply, legal_actions, successors
from reckoner.semantics import eval_exact, eval_field, holds_exact, variables
from reckoner.vocab import (
    DIV,
    EQ,
    GOAL_EVALUATE,
    GOAL_SIMPLIFY,
    GOAL_SOLVE,
    GOAL_TOKENS,
    SEP,
    VAR_NAME,
    VARIABLE_TOKENS,
    VOCAB_VERSION,
    token_name,
)

#: Why an episode stopped. Absence carries a reason: every terminal state names
#: the condition that produced it, so a run's rows never leave a reader guessing
#: whether an unsolved episode ran out of steps or ran out of moves.
TERMINAL_SOLVED = "solved"
TERMINAL_STEP_CAP = "step_cap"
TERMINAL_NO_ACTIONS = "no_legal_actions"
TERMINAL_CONCEDED = "conceded"

#: Where a par label came from. Only ``bfs`` is exact; the rest are floors or
#: moving targets, and the difference is what ``z`` is allowed to mean.
PAR_SOURCES = frozenset({"bfs", "scripted", "pool", "unverified"})
EXACT_PAR_SOURCES = frozenset({"bfs"})


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Problem:
    """A position plus its par label, with the provenance of that label."""

    goal: int
    expr: Expr
    #: ``None`` means **not labelled**, and it is None rather than 0 because 0 is
    #: a legitimate par — a problem already in goal form is terminal at birth. A
    #: sentinel that shares the domain of real values is the untested-equivalence
    #: -class pattern, and it is exactly what the provenance law forbids one
    #: level down: absent must be *absent*, not a number that reads as absent.
    par: int | None = None
    target: int | None = None  # the variable to solve for; SOLVE only

    # [v1.1] Par labels are provenance-tagged so a re-solved pool par can never
    # silently overwrite a BFS-exact one. Fields carry their epistemic status.
    #
    # **The default is `unverified`, and that is load-bearing.** It was `"bfs"`,
    # and chunk 4's proofread caught what that costs: fifty derivations shipped
    # claiming BFS-exact provenance for pars that were hand-written literals,
    # because the most-trusted value was the one you got by saying nothing. A
    # provenance field whose default is its strongest claim is not a provenance
    # field. Exactness must be asserted, never inherited.
    par_source: str = "unverified"  # "bfs" | "scripted" | "pool" | "unverified"
    par_asof: str = ""

    def __post_init__(self) -> None:
        if self.goal not in GOAL_TOKENS:
            raise ValueError(f"{token_name(self.goal)} is not a goal token")
        if self.par is not None and self.par < 0:
            raise ValueError(f"par must be non-negative; got {self.par}")
        if self.par_source not in PAR_SOURCES:
            raise ValueError(
                f"unknown par_source {self.par_source!r}; expected one of {sorted(PAR_SOURCES)}"
            )
        if self.par is None and self.par_source != "unverified":
            raise ValueError(
                f"par_source={self.par_source!r} on a problem with no par. Provenance "
                "describes a label; there is no label here."
            )
        if canonicalize(self.expr) != self.expr:
            raise ValueError("Problem.expr must be canonical")
        if DIV in tokens(self.expr):
            raise ValueError(
                "a v1 problem may not contain DIV: rule set v1 has no eval_div, so "
                "the node is irreducible and the problem is unsolvable by "
                "construction (see REGISTERED-ROUNDS.md ROUND-02)"
            )

        if self.goal == GOAL_SOLVE:
            if self.target not in VARIABLE_TOKENS:
                raise ValueError("a SOLVE problem needs a target variable")
            if not (isinstance(self.expr, Op) and self.expr.kind == EQ):
                raise ValueError("a SOLVE problem must be an equation")
            if self.target not in variables(self.expr):
                raise ValueError(
                    f"SOLVE target {token_name(self.target)} does not occur in the problem"
                )
        else:
            if self.target is not None:
                raise ValueError(f"{token_name(self.goal)} takes no target variable")
            if self.goal == GOAL_EVALUATE and variables(self.expr):
                raise ValueError("an EVALUATE problem must be ground — it has no value otherwise")
            if self.goal == GOAL_EVALUATE and isinstance(self.expr, Op) and self.expr.kind == EQ:
                raise ValueError("an equation is not a value; EVALUATE needs an expression")


# ---------------------------------------------------------------------------
# State encoding — the goal lives in the token stream
# ---------------------------------------------------------------------------


def encode_state(goal: int, expr: Expr, target: int | None = None) -> tuple[int, ...]:
    """``[goal, (target,) SEP, …expression…]``."""
    prefix: tuple[int, ...] = (goal,) if target is None else (goal, target)
    return (*prefix, SEP, *tokens(expr))


def decode_state(seq: tuple[int, ...] | list[int]) -> tuple[int, int | None, Expr]:
    """Inverse of :func:`encode_state`. Raises on anything malformed."""
    seq = tuple(seq)
    if not seq or seq[0] not in GOAL_TOKENS:
        raise ValueError("state does not begin with a goal token")
    goal = seq[0]
    index = 1
    target: int | None = None
    if index < len(seq) and seq[index] in VARIABLE_TOKENS:
        target = seq[index]
        index += 1
    if index >= len(seq) or seq[index] != SEP:
        raise ValueError("state prefix is not terminated by SEP")
    return goal, target, parse(seq[index + 1 :])


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------


def is_goal_form(problem: Problem, expr: Expr) -> bool:
    """Has ``expr`` reached the *shape* the goal asks for? Structure only.

    SOLVE's shape is ``x = <number>`` and nothing else. It does not have to ask
    which way round, because C7 orders an ``EQ``'s operands with the
    variable-bearing side first — so ``5 = x`` is not a form the checker has to
    recognise, it is a form that cannot exist in a canonical state. That is the
    payoff chunk 1 promised, spent here.
    """
    if problem.goal == GOAL_EVALUATE:
        return isinstance(expr, Num)
    if problem.goal == GOAL_SOLVE:
        return (
            isinstance(expr, Op)
            and expr.kind == EQ
            and isinstance(expr.children[0], Var)
            and expr.children[0].token == problem.target
            and isinstance(expr.children[1], Num)
        )
    # SIMPLIFY: a normal form under the rule set — nothing left to rewrite.
    return not legal_actions(expr)


def verify(problem: Problem, expr: Expr, cfg: Config, rng: random.Random) -> bool:
    """Is ``expr`` a *correct* answer to ``problem``? The arbiter.

    Checked against the original problem, never against the derivation that
    claims to have produced it.
    """
    if not is_goal_form(problem, expr):
        return False

    if problem.goal == GOAL_EVALUATE:
        assert isinstance(expr, Num)
        return eval_exact(problem.expr, {}) == Fraction(expr.value)

    if problem.goal == GOAL_SOLVE:
        assert isinstance(expr, Op)
        answer = expr.children[1]
        assert isinstance(answer, Num)
        assert problem.target is not None
        return holds_exact(problem.expr, {problem.target: answer.value}) is True

    return _equivalent_over_field(problem.expr, expr, cfg, rng)


def _equivalent_over_field(left: Expr, right: Expr, cfg: Config, rng: random.Random) -> bool:
    """k random assignments over 𝔽ₚ. Every draw is informative — see module docs."""
    prime = cfg.episode.equiv_prime
    names = tuple(sorted(set(variables(left)) | set(variables(right))))
    for _ in range(cfg.episode.simplify_equiv_k):
        env = {name: rng.randrange(prime) for name in names}
        a = eval_field(left, env, prime)
        b = eval_field(right, env, prime)
        if a is None or b is None:
            raise ValueError(
                "field evaluation was undefined on a v1 state. No reachable v1 state "
                "contains a DIV node, so this means that invariant is broken — not "
                "that this draw was unlucky. Do not silently skip it."
            )
        if a != b:
            return False
    return True


# ---------------------------------------------------------------------------
# The win condition
# ---------------------------------------------------------------------------


def outcome_z(*, solved: bool, steps: int, par: int) -> int:
    """**The win-condition law** (amendment v1.1).

    beat = strictly fewer steps than par; draw = equal; loss = more, or not
    solved at all. Not solving is a loss whatever the reason — the cap, a dead
    end, or a concession — because an episode does not get to end in a shrug.
    """
    if not solved:
        return -1
    if steps < par:
        return 1
    if steps == par:
        return 0
    return -1


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """What an episode produced. ``ruleset_version`` is deliberately first.

    Par is the step count of a reference solution *in this rule system*, so a
    par without its rule-system version is not a label, it is a number. Chunk
    2's granularity freeze is what makes that concrete: the same problem is par
    3 under v1 and would be par 4 under the granularity considered and rejected.
    """

    ruleset_version: int
    vocab_version: int
    goal: int
    solved: bool
    steps: int
    par: int
    par_source: str
    z: int
    terminal_reason: str

    def __post_init__(self) -> None:
        """**z = +1 implies par_source is not exact.** Structural, not proofread.

        BFS-exact par *is* the minimum step count in this rule system, so
        beating it is not a good result — it is a contradiction, and it means
        the label is wrong. Chunk 4's document shipped six of them before a
        human counted; nothing in the code objected, because the win-condition
        tests pinned what z means and nothing pinned z against the provenance of
        the number it was computed from.

        This is loud on purpose. A par off-by-one is not a cosmetic defect: par
        is the game's currency, and chunk 5 mints 100K of them.
        """
        if self.z > 0 and self.par_source in EXACT_PAR_SOURCES:
            raise ValueError(
                f"z = +1 against par_source={self.par_source!r}: solved in {self.steps} "
                f"steps against an exact par of {self.par}. Exact par is the minimum, "
                "so this is not a win, it is a mislabelled problem."
            )


# ---------------------------------------------------------------------------
# The episode
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """``reset(problem) / legal / step(action) / result``.

    One seed in the config fans out; the rng is explicit and never global.
    """

    cfg: Config = field(default_factory=Config)
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    problem: Problem | None = field(default=None, init=False)
    expr: Expr | None = field(default=None, init=False)
    steps: int = field(default=0, init=False)
    terminal_reason: str | None = field(default=None, init=False)

    def reset(self, problem: Problem) -> None:
        self.problem = problem
        self.expr = problem.expr
        self.steps = 0
        self.terminal_reason = None
        self._settle()

    # --- observation ----------------------------------------------------

    @property
    def state_tokens(self) -> tuple[int, ...]:
        self._require_started()
        assert self.problem is not None and self.expr is not None
        return encode_state(self.problem.goal, self.expr, self.problem.target)

    @property
    def done(self) -> bool:
        return self.terminal_reason is not None

    @property
    def solved(self) -> bool:
        return self.terminal_reason == TERMINAL_SOLVED

    def legal(self) -> list[tuple[int, int]]:
        """No action is legal once the episode is over. Terminal means terminal."""
        self._require_started()
        if self.done:
            return []
        assert self.expr is not None
        return legal_actions(self.expr)

    # --- transition -----------------------------------------------------

    def step(self, action: tuple[int, int]) -> None:
        self._require_started()
        if self.done:
            raise ValueError(f"episode is over ({self.terminal_reason}); no action can be taken")
        rule_id, site_id = action
        if action not in self.legal():
            raise ValueError(f"illegal action {action}")
        assert self.expr is not None
        self.expr = apply(self.expr, rule_id, site_id)
        self.steps += 1
        self._settle()

    def _settle(self) -> None:
        """Decide whether the episode has ended, and why. Checked after every step.

        Order matters and is the win-condition law's cap edge: **solving is
        tested before the cap**, so an episode that solves on the very step that
        reaches the cap is solved and scores from steps-vs-par. The cap bounds
        how many steps may be taken, not how many may be credited.
        """
        assert self.problem is not None and self.expr is not None
        if verify(self.problem, self.expr, self.cfg, self.rng):
            self.terminal_reason = TERMINAL_SOLVED
            return
        if self.steps >= self.cfg.episode.step_cap:
            self.terminal_reason = TERMINAL_STEP_CAP
            return
        par_cfg = self.cfg.par
        if par_cfg.concede_enabled and self.steps >= self.problem.par + par_cfg.concede_k:
            # [v1.1] The resign-vs-par analog. Default off; k is uncalibrated,
            # and calibration is campaign evidence, not a default.
            self.terminal_reason = TERMINAL_CONCEDED
            return
        if not legal_actions(self.expr):
            self.terminal_reason = TERMINAL_NO_ACTIONS

    # --- outcome --------------------------------------------------------

    def result(self) -> EpisodeResult:
        self._require_started()
        if not self.done:
            raise ValueError("episode is still running; result() would be a guess")
        assert self.problem is not None
        if self.problem.par is None:
            raise ValueError(
                "this problem carries no par, so z is undefined. Label it "
                "(episode.bfs_par) before scoring an episode against it — a "
                "result with an invented par is the F-02 defect wearing a z."
            )
        return EpisodeResult(
            ruleset_version=RULESET_VERSION,
            vocab_version=VOCAB_VERSION,
            goal=self.problem.goal,
            solved=self.solved,
            steps=self.steps,
            par=self.problem.par,
            par_source=self.problem.par_source,
            z=outcome_z(solved=self.solved, steps=self.steps, par=self.problem.par),
            terminal_reason=self.terminal_reason or "",
        )

    def _require_started(self) -> None:
        if self.problem is None:
            raise ValueError("call reset(problem) before using the episode")


# ---------------------------------------------------------------------------
# BFS-exact par
# ---------------------------------------------------------------------------


def bfs_solution(
    problem: Problem, cfg: Config | None = None, cap: int | None = None
) -> list[tuple[tuple[int, int], Expr]] | None:
    """A shortest derivation, or ``None`` if none exists within ``cap`` steps.

    **It calls** :func:`verify` **— the episode's own acceptance test.** That is
    not a convenience, it is the point: a labeller with its own terminal test is
    a second definition of "solved", and the first time the two drift, every par
    in every dataset becomes a number whose meaning nobody can reconstruct. One
    definition, one implementation, and a par that disagrees with an episode
    outcome is then impossible rather than merely unlikely.

    The full v1 rule set is used, including ``add_both_sides``. Its removal is
    ROUND-01 and has not fired; computing par against a reduced set early would
    make every recorded par a label for a system that does not exist.
    """
    cfg = cfg or Config()
    cap = cap if cap is not None else cfg.par.bfs_exact_max_depth
    rng = random.Random(cfg.seed)

    if verify(problem, problem.expr, cfg, rng):
        return []

    seen = {identity_key(problem.expr)}
    frontier: list[tuple[Expr, list[tuple[tuple[int, int], Expr]]]] = [(problem.expr, [])]
    for _ in range(cap):
        nxt: list[tuple[Expr, list[tuple[tuple[int, int], Expr]]]] = []
        for state, path in frontier:
            for action, successor in successors(state):
                key = identity_key(successor)
                if key in seen:
                    continue
                seen.add(key)
                extended = [*path, (action, successor)]
                if verify(problem, successor, cfg, rng):
                    return extended
                nxt.append((successor, extended))
        if not nxt:
            return None
        frontier = nxt
    return None


def bfs_par(problem: Problem, cfg: Config | None = None, cap: int | None = None) -> int | None:
    """The exact minimum step count, or ``None`` beyond the BFS horizon."""
    path = bfs_solution(problem, cfg, cap)
    return None if path is None else len(path)


def describe_goal(problem: Problem) -> str:
    """A short human label, for the interpreter and for error messages."""
    if problem.goal == GOAL_SOLVE:
        assert problem.target is not None
        return f"SOLVE for {VAR_NAME[problem.target]}"
    return {GOAL_EVALUATE: "EVALUATE", GOAL_SIMPLIFY: "SIMPLIFY"}[problem.goal]
