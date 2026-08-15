"""Gate predicates, defined once and imported by every consumer.

**A gate expressed in a test and separately in a script is two gates wearing one
name.** Proof, and the reason this module exists: F-13 changed the terminal value
scale, the depth-1 predicate was re-expressed in `tests/test_search.py`, and
`scripts/chunk7_gate_table.py` kept the old expression — so the first re-run
reported **0/200 at every m** for a search that was working correctly. One edit
had to be two, and only one was made.

Same family as the one-formatter law (a caption describes or it calls
`render_expr()`; there is no third path) and the one-legality-oracle rule. A
predicate that decides whether a gate passed is exactly the kind of thing that
must have one implementation, because the two copies fail apart silently and the
failure looks like a result.

Predicates here are **scale-independent by construction** wherever they can be:
they read what the search *did*, not what a value happened to equal. That is what
survived F-13 — the chunk-7 gates were never scale-dependent in what they meant,
only in how one of them was written.
"""

from __future__ import annotations

from reckoner.search import SearchResult


def search_found_a_solve(result: SearchResult) -> bool:
    """Did the tree reach a solved terminal?

    **The depth-1 / B_max gate predicate.** Its arithmetic was always about
    whether the winning action is *considered and found* at ``m >= B_max``, never
    about a value crossing a threshold — and reading it from `terminal_solved`
    makes that explicit and immune to the value scale.

    The superseded expression was ``result.values.max() >= 1.0``. Under F-13's z
    currency an at-par solve scores ``0.0``, which is also what a neutral
    evaluator predicts everywhere else, so a solve stopped standing out by value
    and the old predicate read False on every correct solve.
    """
    return result.stats.terminal_solved > 0


def descent_identity_holds(result: SearchResult) -> bool:
    """``nodes - evaluations == terminals``, for a single search.

    Every node is evaluated exactly once unless it is terminal. Chunk 7's
    ``evals == nodes`` is the zero-terminal special case; the identity is the form
    that generalises, and the runner asserts it per iteration.
    """
    s = result.stats
    terminals = s.terminal_solved + s.terminal_no_actions + s.state_too_large + s.step_cap_reached
    return s.nodes - s.evaluations == terminals
