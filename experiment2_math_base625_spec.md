# Experiment Two — Math in Base-625, Trained by Search Against a Checker
## Spec v0.2 (2026-08-13 — v0.1 + the par game, Tom's design; supersedes the solitaire formulation and resolves §8 Q1)

**One-line thesis:** a mini model can learn multi-step mathematics the way TinyAlphaZero learned chess — search over discrete actions, an external verifier as the only source of truth, improved-policy targets, no brute-force pretraining — with base-625 as the native substrate and every "thought" a checked rewrite a human can read.

**Why math first (ratifying Tom's ordering):** three reasons, each load-bearing. (1) **Verification is total and nearly free** — arithmetic evaluates; equation-solving checks by substitution; algebraic equivalence checks probabilistically by random assignment. Cheaper even than a FOL proof checker, and utterly unflatterable. (2) **base-625 is a numeral system** — math isn't *encoded into* the substrate, it lives there natively; the grid glyphs, digit encoders, and expression calculator already exist. (3) **The interpretability claim is structural, not aspirational** — unlike chain-of-thought, the model's reasoning trace *is* its move sequence, every step a verified rewrite, rendered to human notation by a deterministic interpreter (the base-625 rendering tools, plus one derivation formatter). The thinking is readable because the thinking is the derivation.

---

## 1. The program (staged; only Stage A is specced here)

- **A — this spec:** the math solver. Single specialist, search-trained, verifier-gated.
- **B — the logic specialist:** FOL derivations against a proof checker; the st2 Phase-5 machine. Ports Stage A's environment pattern with a new rule set and checker; the 662-symbol vocabulary already exists.
- **C — the overseer:** a *router*, and deliberately nothing more — it classifies a task and dispatches to a specialist, so the system's verification property survives (an overseer that generates rather than routes is where checkability dies). Meaningful only once ≥2 specialists exist.
- **D — the dictionary-language model:** language as a relational construct — word-meaning relationships as first-class structure. **This is the research edge with no verifier**, and it is deliberately last: weak verification exists (definition round-trips, relational consistency, translation-cycle checks) but nothing like a checkmate. It inherits whatever Stages A–C teach about training against partial truth. Not specced; not promised.

The one-lever, one-round, pre-registered campaign discipline is inherited wholesale. Stage A is "experiment two" in the mini-models series; B–D are its successors, not its scope.

## 2. Domain and encoding (Stage A)

- **Vocabulary:** base-625 numerals (0–624) + operator glyphs (`+ − × ÷ = ( ) x` …) + a small set of structural/goal tokens. Compositional numbers exactly as Symbolic-Transformers does it: `[NUM, d₁, d₀]` in base-625 digits — arbitrary magnitude, finite vocabulary, ~650–700 symbols total.
- **Expression state:** a token sequence with a canonical tree form (parser and printer are the same module — one implementation, round-trip fuzz-tested at 200K expressions, the chess encoding gate transplanted verbatim).
- **External interpreter:** `render(derivation) →` human notation, one line per rewrite with the rule named — e.g. `3x + 6 = 21  ──[sub 6 both sides]──►  3x = 15`. Deterministic, tested against the parser round-trip. This ships in Phase 0, not at the end: the humans read its thinking from day one.

## 3. Environment contract (the game)

- **A position:** (expression, goal), where goal ∈ {EVALUATE → ground value; SOLVE(x) → `x = <number>`; SIMPLIFY → canonical form}.
- **Actions = (rule, site):** a rewrite rule applied at a subtree position — the direct analog of (from, to) squares, and the bilinear policy-head factorization transfers: P(rule) × P(site | rule) over a masked space. **Legality mask = pattern match**: a rule is legal at a site iff its left-hand side matches there. The rule set is closed, versioned, and *sound by construction* (every rule is an equivalence, property-tested by random-assignment equivalence on 10K instances per rule — the movegen oracle of this domain).
- **Terminal + checker:** goal-form reached ⇒ verify (exact evaluation for ground arithmetic; substitution for SOLVE; random-assignment equivalence, k=32 draws over a prime field, for SIMPLIFY). An *unsound* rewrite is impossible by construction — the checker decides *solved*, and par decides *won*.
- **The par game (v0.2):** every problem carries a **par** — the step count of a reference solution **denominated in this rule system** — and the outcome is **z ∈ {+1 strictly under par, 0 equal, −1 over par or step cap}** (the win-condition law, pinned by test). Par sources, provenance-tagged: BFS-exact for depth ≤ 6 (canonical); scripted-solver par for middle strata (provisional floor); **own-snapshot par** beyond (`par_from_pool_frac` ≈ 0.20) — the league returns, and par escalates automatically with the model, which builds half the funnel treatment into the game's constitution. External solvers (sympy) are never par — ladder rungs only — unless their derivations compile into our rules.
- **The tree stays single-agent:** the opponent enters the *label*, never the search — backup does not negate, and the hazard flagged in the plan stands. (True adversarial rewriting is structurally degenerate: equivalence rules are reversible, so an obstructor stalls forever, and terminating rules abolish the game — the race is the correct two-sided form.) No color symmetry; the openings book becomes a frozen problem suite storing depth *and* par, both BFS-verified.

## 4. Model

Reuse the v2 architecture family with the vocabulary swapped and heads adapted: ~2–7M parameters, the compositional embedding table from Symbolic-Transformers, policy = rule × site (masked), value = **3-class W/D/L vs par** + steps-to-solve auxiliary regression (§8 Q1 resolved by the par game). **base-625 grid-bit embeddings vs learned token embeddings is NOT assumed** — it is st2's registered question, and it gets its own one-lever round here (identical model, two embedding front-ends, pre-registered) rather than a founding assumption. The substrate thesis is measured, not smuggled.

## 5. Training loop and phases

- **Phase 0 — environment:** parser/printer, rule set + soundness fuzz, checker, interpreter, masks. Gate: 200K round-trips clean; every rule passes equivalence fuzz; interpreter output proofread on 50 hand-checked derivations.
- **Phase 1 — supervised warm start:** procedural problem generator (seeded, difficulty-parameterized — the random-games analog) + a scripted solver producing ground-truth derivations for *easy strata only*. Train policy on (state → rule,site), value on steps-to-solve. Gates: held-out top-k rule-site accuracy; solve rate at depth ≤ 2 with 16-sim search ≥ 95% (the mate-in-1 analog — and this time the gate's sims-vs-branching arithmetic gets checked *before* it ships).
- **Phase 2 — the loop:** search-generated derivations on generator-drawn problems **raced against par** (generator emits BFS-verified par ≤ depth 6, scripted par above; snapshot-par league at `par_from_pool_frac`), improved-policy targets, replay ring, z/q value blend from day one (its case was proven in `p2_c`; its self-referential hazard is *structurally absent here* — the checker, not the model's own Q, is the source of solved/not — which is exactly why this domain suits the method).
- **Phase 3 — the diversity engine, pre-registered as the funnel treatment:** `p2_d`'s central finding — *the loop collapses when left alone* — arrives here as a design input, not a surprise. Snapshot-par already makes the reward non-stationary in the healthy direction — the built-in half of the treatment; this is the other half. The lever, named in advance: a **generator trained adversarially at the solver's frontier** (propose problems the solver solves ~50% of the time). Curriculum emerges; diversity is manufactured, not hoped for. Phase 2 runs with the procedural generator and the entropy instrumentation armed; Phase 3 fires when the funnel signature appears — as it will.

## 6. Instruments (inherited, renamed)

Suites: solvable-in-N-steps (N = 1…6), frozen, contamination-tested against all training data — mate-in-N reborn. Watchlists with named problems. Decompositions: coverage (right rule-site in top-m) × conversion — the same factorization. **The ladder:** rung 1 random-rewriter, rung 2 greedy-heuristic (always simplify largest subtree), rungs 3+ **sympy under fixed step/time budgets** — the CAS is this domain's Stockfish: external, calibrated, and impossible to flatter. Ladder scoring uses the **same z-vs-par currency as training** — one unit end to end, no translation layer between what the loop optimizes and what the ladder measures. Elo-style paired scoring, `pair_scores` persisted from row one. ECE on the W/D/L-vs-par probabilities. Pre-registration with amendment headers; RUNLOGs; the gitignore guard; all of it.

## 7. Hardware and cost

CPU-heavy again (pattern-matching movegen), GPU light — the box suffices; the multiprocess self-play chunk, if built for chess, transfers. No cloud required for Stage A.

## 8. Open questions for the spec review (answer before the plan is chunked)

1. ~~Value-head shape~~ — **RESOLVED (v0.2):** 3-class W/D/L vs par + steps auxiliary, per the par game.
2. Rule-set v1 contents: integer arithmetic + linear-equation rules only, or fractions/distribution from the start? (My lean: the smallest closed set that makes depth-6 problems non-trivial — extend by round, one lever.)
3. SOLVE goal representation: goal tokens in the state, or a separate conditioning channel?
4. Does the Phase-3 generator share weights with the solver or stand alone? (My lean: stand alone — the adversarial dynamic stays legible.)
5. Naming. "Experiment two" needs a name that isn't a sentence.

---
*Next step on Tom's word: resolve §8, then the chunked agent plan with DONE-WHEN gates, same shape as the chess rebuild — after `p2_d` reports.*

---

# Erratum E1 — §2's interpreter example (2026-08-14, from chunk 2)

**Superseded, not reinterpreted.** §2 illustrates the interpreter with

> `3x + 6 = 21  ──[sub 6 both sides]──►  3x = 15`

Under rule set v1 as built and frozen in chunk 2, that is **two** rewrites, because
`sub_both_sides` is a structural rule and does not compute:

```
3x + 6 = 21  ──[sub_both_sides, operand 6]──►  3x = 21 + (−6)
             ──[eval_add, right side]──►       3x = 15
```

Two corrections to the erratum as first drafted in review: the intermediate is
`21 + (−6)` and the second rule is **`eval_add`**, not `21 − 6` / `eval_sub`.
`sub_both_sides` moves the addend across *as its negation* rather than building a
`SUB` node, and that is load-bearing rather than cosmetic: `eval_sub` requires
both operands numeric, so on a two-sided problem `SUB(2x + 18, 3)` would be a
dead end that no v1 rule can reduce. The negation lands inside the same
flattened `ADD`, where `eval_add` reaches it.

**Why the finer granularity was adopted** (measured in chunk 2, ratified in
review): structural rules move, `eval_*` rules compute, so every rendered line is
semantically atomic — the interpretability thesis expressed in the rule system
itself. The spec's original one-line form hides `21 − 6 = 15` inside a
structural step. The alternative (deferring the cancellation too) was measured
and rejected: it put `5x + 3 = 2x + 18` at par 7, *outside* the depth-6
BFS-exact band that canonical par depends on, and cost 234,627 BFS states
against 9,378.

One exception, forced not stylistic: **`div_both_sides` computes its quotient**
under its exactness guard. Deferring it would emit a `DIV` node, and v1 has no
`eval_div` to reduce one. See `REGISTERED-ROUNDS.md` R2.

**Consequence for par.** Granularity is now frozen as `RULESET_VERSION = 1`. Par
is denominated in a rule system, so the version is part of the label: it is
carried in episode results, par provenance, and every dataset and suite
`meta.json` from chunk 5 on. A future granularity change bumps it and invalidates
recorded pars loudly rather than silently.
