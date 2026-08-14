# Findings

Measured facts that are not rounds. A round is a plan to test a change; a finding
is something already true that later chunks must not rediscover. Errata culture:
every defect named, including the trivial ones, and nothing fixed silently.

---

## F-01 — Scripted-solver suboptimality, first measured instance

**Found:** 2026-08-14, chunk 4 proofread (derivation 19 of `docs/derivations.md`).

The greedy scripted policy prefers `eval_add` over `eval_mul`, and on
`7 + (9 − 30) + 12 × 12` that costs a step: it evaluates the subtraction, folds
the two loose constants, then multiplies, then folds again — **4 steps against a
BFS-exact par of 3**. The optimal order evaluates the product first so a single
`eval_add` folds all three numerals at once.

**Why it matters beyond one derivation.** This is the first *measured* instance
of the gap that scripted par's "provisional floor" status (spec §3) is priced
for. It is small (one step at depth 3) and it is real, and until chunk 5 measures
the delta distribution it is the only data point.

**Consequence, already in chunk 5's brief:** on the depth ≤ 6 overlap where both
labels exist, compute BFS *and* scripted par and report the delta distribution —
free calibration of exactly how provisional the mid-strata floor is.

**Status:** recorded. The document now renders BFS-optimal derivations, so the
exhibit is preserved here rather than on the page.

---

## F-02 — Provenance acquired by silence

**Found:** 2026-08-14, chunk 4 proofread (derivations 33, 36, 39, 42, 45, 48).

Six derivations carried `par 2 (par_source=bfs)` beside a one-step solve, giving
`z = +1`. That row is impossible by construction: BFS-exact par **is** the
minimum step count in this rule system, so nothing can beat it.

**Root cause — neither of the two branches the review named.** BFS's terminal
test did not diverge from the episode's, and the par pipeline did not diverge
from BFS's output. **There was no BFS labeller at all.** Every par in the fixture
set was a hand-written literal, and `Problem.par_source` *defaulted* to `"bfs"`.
The document claimed exact provenance for numbers that had never been computed.

Audit of all fifty: **6 mislabelled, 44 correct by luck** — the hand-written
values happened to be right everywhere else, which is precisely why nothing
caught it.

**The structural fault, and the general lesson:** a provenance field whose
default is its *strongest* claim is not a provenance field. The most-trusted
value must be the one that costs something to say. Exactness is asserted, never
inherited.

**Fixes, all landed:**

| | |
|---|---|
| `Problem.par_source` default | `"bfs"` → `"unverified"`, validated against a known set |
| `EpisodeResult.__post_init__` | raises on `z > 0` with an exact `par_source` |
| `episode.bfs_solution` / `bfs_par` | a real labeller, calling `verify()` so a label and an outcome cannot disagree about what "solved" means |
| `scripts/render_derivations.py` | fixtures state *problems*; `label()` computes par and its true provenance |
| `docs/derivations.md` | regenerated: 0 × `z = +1`, 47 × `z = 0`, 3 × `z = −1` (the labelled ILLUSTRATIVE entries) |

**Status:** closed, with the invariant pinned. The reason it is pinned in the
result path rather than in a test is scale: chunk 5 mints 100K par labels and
1,200 suite pars, and a par off-by-one at that scale poisons the game's currency
at birth.

---

## F-03 — The pre-flight underestimated by 5×

**Found:** 2026-08-14, chunk 5 generation.

The stratified pre-flight projected **18.3 min** for 100K problems on 24
workers. Actual: **5,636.9 s — 94 min**, 5.1× over.

**Why.** Three compounding causes, none of them the stratification the
projection got right:

1. **Sample too small for a heavy tail.** 20 candidates per stratum, against a
   template (`solve_both_sides_product`) whose cost ranges 4.5 s median to 8.8 s
   worst. Twenty draws does not estimate the mean of that distribution.
2. **The projection priced labelling, not generation.** The driver emits ~1.6×
   the requested count and deduplicates, so the real work is ~1.6× the projected
   work before any retry.
3. **Retries are unpriced.** Strata that under-fill trigger another pass.

**The correction, for the next projection that matters:** sample per *template*
rather than per stratum, use the mean of ≥100 draws where the tail is heavy, and
multiply by the over-generation factor. The stratification instinct was right —
projecting from a flat average would have been worse — but a stratified estimate
built on 20 samples of a long-tailed distribution is still an estimate of the
wrong number.

**Not a blocker:** 94 min ran unattended and the artifacts are correct. Recorded
because the next pre-flight will be for something that cannot be re-run casually.

---

## F-04 — Depth histograms are outcomes, not requests

**Found:** 2026-08-14, chunk 5 generation.

The generation plan asks for equal counts per stratum. The training set came out
`{1: 20751, 2: 16582, 3: 14068, 4: 29036, 5: 16013, 6: 3550}` — depth 4 nearly
double its share, depth 6 at a fifth of it.

**This is the design working, not failing.** Difficulty is parameterised by
*measured* depth, so a candidate lands in the stratum BFS says it belongs to, not
the one its template was aimed at. `solve_two_terms` splits 4/21 across depths 3
and 4; `eval_deepest` splits 4/56 across 5 and 6. The plan is a request; the
histogram is the answer.

**Consequence for chunk 8:** the warm start sees this distribution, not a uniform
one, and depth 6 is thin (3.6%). Either the sampler rebalances at training time
or the generation plan over-requests the deep strata. Recorded here so it is a
decision rather than a surprise in a loss curve.

---

## F-05 — `add_both_sides` is the entire source of state growth

**Found:** 2026-08-14, chunk 6 sizing measurement.

Random legal play from the depth-5/6 suites, 24 steps, 1,600 walks:

| | tokens p99 | tokens p100 | sites p99 | sites p100 |
|---|---:|---:|---:|---:|
| with `add_both_sides` | 247 | 295 | 87 | 104 |
| **without** | **57** | **57** | **15** | **15** |

Without it, a reachable state never exceeds the scale of a *start* state
(p100 64 tokens / 17 sites). With it, states reach 5× the tokens and 7× the
sites — and since attention is O(L²), that is roughly a 25× cost multiplier on
the trunk, plus a policy head 7× wider.

**This is the third independent argument for ROUND-01**, and the first that
costs money rather than elegance:

1. reachability-redundant — par is identical without it (chunk 2)
2. an active trap — any policy that prefers it fails to terminate (chunk 4)
3. **it alone forces the model's sequence and action bounds** (here)

**What chunk 6 did with it:** nothing. The rule is in the pinned v1 set, so the
model is sized for the rule set that exists, not the one a round might produce —
the same discipline as computing par against the full set until ROUND-01 fires.
The measurement is recorded so the round's cost/benefit is a number rather than
a preference.

---

## F-06 — A search that did not search, and four gates that passed anyway

**Found:** 2026-08-14, while building chunk 8's depth-≤2 gate.

Chunk 7 shipped what its docstring called an array-tree MCTS. It expanded only
the root's children and then re-backed-up their already-computed values:

    sims=48, m=5  ->  nodes in tree: 2

Forty-seven of forty-eight simulations did no work. It was a one-ply lookahead.

**Every chunk-7 gate passed on it**, and each for a reason that had nothing to do
with the defect:

| gate | why it passed anyway |
|---|---|
| depth-1 suite 100% at m ≥ 5 | a depth-1 problem needs exactly one ply |
| budget identity | visits are counted whether or not they do work |
| sync vs batched, 16 cells | both paths were equally shallow |
| 2-seed + cross-process determinism | unaffected by depth |

That is the shape of the problem: the gates were chosen to catch specific
hazards — negation, budget drift, batching skew, hash-order leakage — and a
search that never descends is none of those. **No gate asked whether the search
searched.**

**Fixes:**

* Real descent. A simulation now walks from the chosen root action down through
  expanded nodes by the Gumbel-AZ non-root rule until it reaches an unexpanded
  action, expands it, evaluates and backs up. Measured after: `sims=48` on a
  branchy depth-5 SOLVE gives **49 nodes, max_depth 3**.
* `SearchStats` gains `nodes` and `max_depth`, so the shape of the tree is in
  every row rather than inferable from nothing.
* **The missing gate exists now** — `test_the_tree_deepens_past_one_ply` asserts
  more simulations build more tree, that the tree exceeds root-plus-children,
  and that node count scales with sims rather than with `m`.
* Batching moved to where it belongs: **across concurrent searches, not within
  one**. Pooling leaves inside one tree would change what the tree does, because
  simulation *k+1*'s selection depends on *k*'s backup — the old equivalence test
  was exact only because the old search had no selection to corrupt. Across
  trees there is no coupling, so parity stays an equality rather than a
  tolerance, and it matches AGENTS.md §8's prescribed shape.

**The general lesson, and it is not "add a test":** a gate suite assembled from
known hazards has a blind spot exactly the shape of *the component doing its job
at all*. Every chunk here has gates for how a thing can be subtly wrong; this is
the first case where the thing was grossly absent and every subtle-wrongness
check still went green.
