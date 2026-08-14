# Experiment Two — Agent Implementation Plan v1.0 (2026-08-13)
## Working name: `reckoner` (rename freely; it just can't be a sentence)

Companion to `experiment2_math_base625_spec.md`. Execute chunks in order, one chunk per session unless stated. Every chunk ends with its DONE-WHEN gates green, committed, **pushed**, and reported with verbatim numbers and a tree-state block (merged? pushed? test count?). If a gate can't pass after 3 distinct attempts: stop, write `BLOCKED-<date>-<topic>.md` (cause, attempts, options), commit it, halt. Never weaken a gate to pass it.

**§8 decisions baked in (Tom may override any before chunk 4):**
1. Value head = 2-class solved/timed-out CE + steps-to-solve regression (masked to solved episodes). "Stuck-but-progressing" is unmeasurable mid-episode; rejected.
2. Rule set v1 = the minimal closed set: `eval_add, eval_sub, eval_mul, combine_like_terms, add_both_sides, sub_both_sides, div_both_sides` (exact integer division only; the generator guarantees integer solutions). No fractions, no distribution — extensions are later one-lever rounds.
3. Goal representation = prefix tokens in the state (`[GOAL_SOLVE, VAR, x, SEP, …]`) — no architecture change.
4. Phase-3 generator = standalone model, not weight-shared. Not built in this plan; registered as the funnel treatment.
5. Single-agent search: **there is no opponent, so backup does NOT negate per ply.** The ported tree gets a `perspective="single"` mode — this is the sharpest porting hazard in the plan and chunk 7 exists partly to prove it.

**Inherited law (in force from chunk 0, not adopted incident-by-incident):** strict config loader with unknown-key hard errors; `.gitignore` MUST_REACH/MUST_IGNORE guard test; `logschema.py` as single schema definition with `role` fields and absence-carries-a-reason notes; `pair_scores` persisted from the first ladder row; PREREG files carry the amendment-policy header from day one; paired-difference bootstrap is the test of record for pass-vs-pass comparisons; every detector that gates automation is validated on both polarities before live use; instrument the trigger, never trust identical numbers; external processes are context managers; one shared identity normalizer for any dedup key; `git -C` in multi-command shells; waiters reference PIDs, never patterns; provenance (`git_sha` via `git -C <repo_root>`, config fingerprint) in every checkpoint and dataset `meta.json`.

---

## Chunk 0 — Scaffold
`uv`-managed repo, `pyproject.toml` (torch CPU-first; ROCm variant documented but optional until chunk 7), `src/reckoner/` package, ruff + pytest wired, `make lint test`. AGENTS.md carrying the inherited-law block above verbatim plus box facts. GitHub repo created, remote + tracking configured, first push **in this chunk** — the four-session push gap does not get re-learned. `configs/default.yaml` + strict loader + its tests. Gitignore guard test with initial MUST_REACH list (`BLOCKED*.md`, `RUNLOG*.md`, `PREREG*.md`, `runs/*/iterations.jsonl`, `runs/*/ladder.jsonl`, `benchmarks/results.jsonl`, `runs/ANCHORS.sha256`).
**DONE-WHEN:** `make lint test` green in a clean clone; push verified with `git log --branches --tags --not --remotes` empty.

## Chunk 1 — Vocabulary, parser, printer
Token spec in one module: base-625 numerals via compositional `[NUM, d…]`, operators, structural, goal tokens (~650–700 ids, versioned `VOCAB_VERSION`). Parser (tokens → canonical expression tree) and printer (tree → tokens) as **one module, one round-trip**. Canonicalization rules stated in the docstring (child ordering, flattening) — the docstring is a claim; test it.
**DONE-WHEN:** 200K random-tree round-trips byte-exact; property tests (parse∘print = id, print∘parse = canonical); hypothesis fuzz on malformed sequences rejects cleanly; vocab table dumped to `docs/vocab.md`.

## Chunk 2 — Rule engine (the movegen)
Rule = (name, LHS pattern, RHS template, guard). Pattern matcher over tree sites; `legal_actions(state) → [(rule_id, site_id)]` with **deterministic site enumeration order** (documented — downstream pins depend on it); `apply(state, rule, site) → state'`. Soundness fuzz: every rule, 10K random variable assignments over a prime field, LHS-value == RHS-value — this is the movegen oracle and it runs in CI forever. Guard test: `div_both_sides` only fires when division is exact.
**DONE-WHEN:** all v1 rules pass soundness fuzz; matcher agrees with a brute-force reference matcher on 5K random trees; median/max branching factor measured on random states and **recorded in the chunk report** (chunk 7's gate arithmetic depends on this number — measure it now, not then).

## Chunk 3 — Episodes and the checker
`(expression, goal)` state; goals EVALUATE / SOLVE(x) / SIMPLIFY as prefix tokens. Terminal detection per goal; verification: exact evaluation (EVALUATE), substitution (SOLVE), random-assignment equivalence k=32 over a prime field (SIMPLIFY). Step cap (config, default 24). Episode API: `reset(problem) / legal / step(action) / result`.
**DONE-WHEN:** an adversarial near-miss fixture set (hand-built: off-by-one answers, unsimplified-but-close forms, x on both sides) — checker rejects every one; checker accepts 1K scripted-solver solutions; episode invariants fuzzed (step count, cap behavior, no action legal after terminal).

## Chunk 4 — The interpreter (ships now, not later)
`render(derivation) → text`: one line per rewrite, rule named, before → after in human notation; optional base-625 glyph panel output reusing the existing renderer (vendored minimal or optional dependency — agent's call, documented). This is the "external interpreter for the humans to read its thinking" and it is load-bearing for the whole program's interpretability claim.
**DONE-WHEN:** 50 scripted derivations rendered and **proofread by Tom** (a manual gate, flagged as such — the report pauses here for his eyes); renderer round-trips with the parser on every intermediate state; deterministic output byte-identical across runs.

## Chunk 5 — Generator, scripted solver, datasets, suites
Procedural problem generator: seeded, difficulty parameterized by **verified minimum solution depth** (BFS over the rule graph for depths ≤ 6 — the depth label is measured, not assumed). Scripted solver for easy strata producing ground-truth derivations. Dataset writer with the chess conventions (memmap fields, `meta.json` mode/config/sha). **Frozen instruments generated here and never regenerated:** suites `solve_in_N.jsonl` N=1…6 (200 each, BFS-verified), held-out eval sets, all contamination-tested against every training set — `test_no_suite_contamination` ports with its config-field-guard pattern.
**DONE-WHEN:** 100K-problem training set + suites + eval sets on disk with digests in `runs/ANCHORS.sha256`; every suite problem's depth label re-verified by independent BFS; contamination tests green; generation fully reproducible from seed (spot-check: regenerate suite N=3 from seed, byte-identical).

## Chunk 6 — Model
Port the v2 transformer family: compositional embedding table (learned token embeddings — **the grid-bit front-end is a registered later experiment, not the default**), policy head factorized rule × site with legality masking, value head per §8 decision 1. Target 2–7M params, exact count in the report. Config-is-spec guard test.
**DONE-WHEN:** forward/backward on synthetic batches all goals; **masked-loss invariance test** (values under masked-off actions cannot affect loss — the fabricated-target detector, ported); param count recorded; checkpoint save/load with fingerprint round-trip.

## Chunk 7 — Search core (the careful port)
Port array-tree MCTS + Gumbel root + Sequential Halving + batched runner from v2, with: `perspective="single"` (no per-ply negation — **new backup tests written first**: a hand-computed 3-level tree's values must match max-backup arithmetic exactly); terminal-solved short-circuits consume budget (convention kept); eval/self-play profiles kept; batched equivalence parametrized over m ∈ {3, 5, 12, 16} × sims ∈ {6, 16, 31, 48} **from day one** (the odd-parity lesson is pre-paid); proof-directed breadth is N/A (no adversary — replies don't exist), noted in the docstring so nobody ports it blindly.
**Gate arithmetic check is a required step before setting the gate:** using chunk 2's measured branching, verify the depth-1 stub gate is reachable at the chosen sims (the chunk-6-of-chess lesson, institutionalized).
**DONE-WHEN:** depth-1 suite 100% at the arithmetic-verified (sims, m) with uniform stub; budget identity; equivalence green across the full parametrization; determinism in eval profile (2-seed identity).

## Chunk 8 — Phase 1: supervised warm start
Train script with the full inherited kit (strict config, `--init-weights`, NaN-skip guard + abort, constant/cosine LR from config, mode audit, timing slice as required pre-flight). Metrics: top-k (rule,site) accuracy depth-stratified; `eval_phase1` analog. Run Phase 1 on the box (CPU or GPU, whichever the timing slice recommends).
**DONE-WHEN (provisional thresholds, revision only via BLOCKED):** held-out top-8 rule-site ≥ 0.90 on depth ≤ 3; depth-≤2 solve with 16-sim search ≥ 0.95 using the trained net; training reproducible from seed at the loss-curve level.

## Chunk 9 — Phase 2: the loop
Episode runner (batched leaves, the v2 runner adapted), replay ring **with `root_q` from field one and `_FIELDS_SINCE` era handling**, z/q value blend on by default (`value_q_mse_weight: 0.5` — the checker makes its hazard structurally absent, say so in the docstring), rehearsal machinery ported (dormant, `rehearsal_frac: 0.0` default — the lever exists before it's needed), `logschema.py` (policy entropy at step 1, solve rates by depth, wall-clock split, skips), crash-resume with two SIGKILL points, golden mini-run < 3 min in `make golden`.
**DONE-WHEN:** golden green; both kill-points resume clean; 3-iteration shakedown at default config with pre-registered plumbing expectations (rows populate, splits sum, snapshot loads), shakedown deleted after recording.

## Chunk 10 — The ladder
Opponents: `RandomRewriter`, `GreedyHeuristic` (largest-subtree-first), `SympySolver(step_budget, time_budget)` rungs — external CAS as the unflatterable reference, context-managed, clean-skip if absent. Paired problem sets (frozen, like the openings book), Elo-style paired scoring, `pair_scores` persisted from row one, paired-difference bootstrap implemented **and used** for pass-vs-pass, self-match null (deterministic eval profile ⇒ exact identity — the no-hidden-state detector, with the contrast case), `role` field, `calibration_note`, saturated-CI rendering. Suites on ladder cadence; watchlist mechanism ported (live verdicts + `baseline_asof`).
**DONE-WHEN:** synthetic Elo tests incl. rigged-50% null; self-match identity exact; one full smoke pass against all rungs with verbatim numbers; ladder resumable mid-pass.

## Chunk 11 — Campaign M1 (the first pre-registered run)
`PREREG-m1.md` (amendment header) before launch: primary = CI-separated improvement on the sympy-rung or depth-suite trajectory vs the Phase-1 anchor over ~20 iterations at `--ladder-every 5`; entropy instrumentation armed with the funnel signature named; no-regress on Phase-1 gates; decision rules incl. the BLOCKED branch. Run it. `RUNLOG-m1.md` with the full ledger, literal verdicts, and errata as needed.
**DONE-WHEN:** the run completed or BLOCKED per rules; RUNLOG committed with every number verbatim; anchor promoted + Release uploaded if passed.

## Chunk 12 — Registered, not built
Phase-3 adversarial generator (fires on the funnel signature); grid-bit vs token embedding round (st2's question, one lever); rule-set extension rounds; the FOL specialist (Stage B) inherits this repo's environment pattern.

---
*Report format, cadence, and escalation identical to the chess campaign. The first message back is chunk 0's tree-state block — with the push already done.*

---

# Amendment v1.1 — The Par Game (2026-08-13, pre-chunk-0, Tom's design)

Experiment two is a **race, not a solitaire**: every problem carries a **par** — the step count of a reference solution *in this rule system* — and the episode outcome is **z ∈ {+1 strictly under par, 0 equal, −1 over par or timed out}**. The opponent never enters the search tree (the single-agent backup rule and its bold hazard are unchanged); it enters the *label*. This restores the full outcome structure of the chess project without its adversarial tree.

**Plan deltas:**
- **§8 decision 1 REVERSED:** value head returns to **3-class W/D/L** (vs par) + steps-to-solve auxiliary regression. Chunk 6 builds this shape.
- **Chunk 3:** episode result gains `par` and the z-mapping above; the win-condition law is pinned in a test (beat = strictly fewer; draw = equal; loss = more or cap).
- **Chunk 5:** the generator emits par per problem, **denominated in our rules only**: BFS-exact par for depth ≤ 6 (canonical); scripted-solver par for middle strata (marked `par_source: scripted`, provisional); suites store par alongside depth, both BFS-verified. External solvers (sympy) are NEVER par — they remain ladder rungs — unless a future chunk compiles their derivations into our rule vocabulary.
- **Chunk 9:** the **snapshot league returns** — `CheckpointPool` ported; a fraction of training problems (config `par_from_pool_frac`, default 0.20) take par from a pool snapshot's recorded solution on that problem (solved fresh at episode time, same seed fan-out), so par escalates automatically with the model. This is the built-in half of the funnel treatment; the Phase-3 generator remains the other half, registered as before. Resign-vs-par analog (`concede if best-found ≥ par + k`) implemented but default off, calibration deferred to campaign evidence, per the resignation lesson.
- **Chunk 10:** unchanged in structure — the ladder was already paired — but ladder scoring adopts the same z-vs-par currency so training outcomes and evaluation outcomes are one unit.
- **Chunk 11:** PREREG's primary criterion restated in par terms: CI-separated improvement in beat-par rate on the frozen suites (paired, `pair_scores` from row one).

**Hazards on the record:** weak scripted par is a floor, not a ceiling — `par_from_pool_frac` exists to replace it quickly; draw inflation versus snapshot par at convergence is expected and instrumented (composition columns in the schema, per `p2_c`); par labels are provenance-tagged (`par_source`, `par_asof`) so a re-solved pool par can never silently overwrite a BFS-exact one — fields carry their epistemic status, as always.
