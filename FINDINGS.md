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
