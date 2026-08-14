# Registered rounds

One-lever rounds that are **specified before they are run**, so the evidence
standard is fixed while nobody yet knows the answer. A round registered here is
not a plan to change something — it is a plan to *test* changing it, with the
kill condition written down first.

The plan's chunk 12 registers the programme-level rounds (Phase-3 adversarial
generator, grid-bit vs token embeddings, rule-set extensions, the FOL
specialist). This file registers the ones that came out of implementation, with
the measurements that motivated them.

---

## R1 — Remove `add_both_sides`

**Registered:** 2026-08-14, from chunk 2.
**Status:** registered, not run. `add_both_sides` stays in rule set v1.

**Claim.** `add_both_sides` is *reachability-redundant*: it only ever grows both
sides of an equation, so no problem needs it and par is identical without it. It
is pure branching tax.

**Motivating measurement** (chunk 2, three problems, BFS both ways):

| problem | par with | par without | BFS states with | without |
|---|---|---|---|---|
| `3x + 6 = 21` | 3 | 3 | 243 | 35 |
| `2x − 4 = 10` | 3 | 3 | 243 | 35 |
| `5x + 3 = 2x + 18` | 5 | 5 | 9,378 | 167 |

**Why it is not simply removed.** Plan §8 decision 2 pins the rule set. Removal
is a spec change and therefore a registered round, not an edit.

**Evidence standard — three fixtures do not prove a universal.** The set-growth
law applies: the round's evidence is **par-with vs par-without over the full
chunk-5 suite**, BFS computed both ways on every problem. Prediction: identical
par on every problem. **Any single counterexample kills the round** — it would
mean `add_both_sides` is reachability-relevant and the claim above is false.

**Declared dependency.** Removing the rule changes branching, so this round must
also re-measure branching (`scripts/measure_branching.py`) and re-derive chunk
7's `(sims, m)` gate arithmetic against the new numbers. It cannot be run as an
isolated deletion.

**Do not let the claim leak early.** BFS-for-par runs the **full v1 rule set**
until this round fires and passes. Par is denominated in a rule system; quietly
computing it against a reduced set would make every recorded par a label for a
system that does not exist.

---

## R2 — `eval_div` plus a structural `div_both_sides`

**Registered:** 2026-08-14, from chunk 2.
**Status:** registered, not run. Separate from R1; do not bundle them.

**Claim.** Adding `eval_div` (evaluate a `DIV` node with exact integer operands)
would let `div_both_sides` become structural like the other movers — emitting
`DIV(lhs, a)` / `DIV(rhs, a)` and deferring the arithmetic — restoring the
"structural rules move, `eval_*` rules compute" symmetry exactly.

**Why v1 does not do this.** With no `eval_div`, a deferred quotient is a dead
end: nothing in v1 reduces a `DIV` node. So `div_both_sides` computes `c // a`
inside its exactness guard, and it is the one asymmetric rule in the set.

**What the round would cost.** Three things it would break, all of which must be
re-established as part of the round rather than assumed:

1. **The v1 DIV-free invariant.** Today no rule constructs a `DIV`, so a
   `DIV`-free problem generates a `DIV`-free reachable state space
   (`test_no_rule_ever_constructs_a_div_node`). R2 breaks that by construction.
2. **The field-only licence that invariant buys.** Division is the only partial
   operation — it is what makes `eval_field` return `None`. Chunk 3's SIMPLIFY
   checker compares by field equivalence alone *because* no state contains a
   `DIV` and so no draw is ever undefined
   (`test_div_free_states_never_evaluate_to_undefined`). R2 reintroduces
   undefined draws, and the checker would have to skip-and-count them.
3. **Par everywhere.** Every solve gains a ply. `RULESET_VERSION` bumps, and
   every recorded par is invalidated.

**Evidence standard.** Par and branching re-measured over the full chunk-5
suite; the SIMPLIFY checker re-specified with skip-and-count and its own
anti-vacuity floors; the interpreter's rendering re-proofread. The round is
justified only if the granularity symmetry is worth a ply on every problem plus
a partial checker.

**Generator ban stands meanwhile.** Chunk 5's v1 generator must not emit `DIV`
in problems. The `DIV` token itself stays in the vocabulary — chunk 1's
id-stability principle: ids never move, so an unused token is cheaper than a
renumbering.
