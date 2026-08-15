# GATE-chunk8-VERDICT.md — Phase 1: supervised warm start

Verdicts for chunk 8's DONE-WHEN gates, recorded under the principal's ruling of
2026-08-15. Every row carries rider (c)'s **four-tuple** — floor, null-model
baseline, threshold, measured value — because a threshold nobody computed the
floor of is not a gate.

Two words are used precisely throughout. **PASSED** means the measured value met
the registered threshold. **VACUOUSLY** means it would have done so for any
model, including an untrained one. They are recorded together where both apply,
per literal-verdict law: the literal result stands, and the finding files beside
it rather than replacing it.

---

## Gate 10 — held-out policy accuracy

### 10a. As registered: `top-8 rule-site ≥ 0.90 on depth ≤ 3`

| | |
|---|---:|
| floor (fraction of depth ≤ 3 states with ≤ 8 legal actions) | **0.9897** |
| null model (untrained network, random init, seed 0) | **1.0000** |
| threshold | 0.9000 |
| measured (trained) | 1.0000 |

**Verdict: PASSED — VACUOUSLY.** The threshold sits below the floor. Top-k is
ranked over the legal action set, 98.97% of depth ≤ 3 held-out states have ≤ 8
legal actions, and a randomly initialised network scores 1.0000. The gate cannot
distinguish 5,000 steps of training from none. Filed as `FINDINGS.md` F-10.

### 10b. As amended, under the principal's signature via the plan's revision channel

**`top-1 rule-site ≥ 0.90 on depth ≤ 3, evaluated on the F-09 unseen subset.`**
The amendment folds F-09 and F-10 into one instrument: the metric that
discriminates, measured on the states the model has not seen. Its arithmetic is
stated **at declaration**, not afterwards:

| | |
|---|---:|
| floor (fraction of depth ≤ 3 states with ≤ 1 legal action) | **0.5241** |
| null model — uniform-random, E[1/B] | **0.6803** |
| null model — untrained network, measured | **0.6950** |
| threshold | 0.9000 |
| headroom demanded above floor | **0.3759** |
| **measured (trained, unseen subset)** | **0.9699** |

**Verdict: PASS. Discriminating by construction.**

### Depth-stratified top-1, unseen subset (n = 5,172)

| depth | n | top-1 |
|---:|---:|---:|
| 1 | 368 | 1.0000 |
| 2 | 345 | 0.9652 |
| 3 | 516 | 0.9516 |
| 4 | 1,606 | 0.9782 |
| 5 | 1,725 | 0.9913 |
| 6 | 612 | 1.0000 |
| **≤ 3 (the gate)** | **1,229** | **0.9699** |

### All three numbers, side by side, per the F-09 ruling's both-numbers form

| measurement | n | depth ≤ 3 top-1 |
|---|---:|---:|
| all held-out states | 1,744 | 0.9782 |
| **unseen subset (the gate)** | **1,229** | **0.9699** |
| inflation delta | | **+0.0083** |

The contaminated instrument reads 0.83 points high. Unseen n = 1,229 ≥ 1,000, so
`eval_held_out_v2` is **not triggered** and v1 stands unmodified. The exclusion is
applied at measurement time, not by deleting data, so one artifact serves both
numbers permanently.

---

## Gate 11 — depth ≤ 2 solve rate under 16-sim search

Registered config, `m ≥ 5` per item 9's B_max arithmetic. Solving means *playing
the episode* — search, apply, repeat — not "a solve appeared somewhere in the
tree". Step budget `min(step_cap, 2 × par + 2)`, declared.

| | |
|---|---:|
| floor (a-priori: no state solves itself) | 0.0000 |
| **null model — uniform-prior stub, same budget, same problems** | **0.9175** |
| threshold | 0.9500 |
| **measured (trained)** | **1.0000** |
| null-to-threshold margin | 0.0325 |

**Verdict: PASS. Discriminating, but thinly** — the null sits 0.0325 below the
threshold, and a slightly stronger null would erase the gap. Per suite: trained
200/200 on both `solve_in_1` and `solve_in_2`; stub 200/200 and 167/200.

**The first null measured for this gate was wrong** and said 1.0000, which would
have made gate 11 vacuous like gate 10. The cause was a constant-seeded search
rng; see `FINDINGS.md` F-11. The correction changed the verdict, not just the
number.

At `m = 1` the same gate reads null 0.7725 against measured 1.0000 — a far wider
margin, because at `m = 1` the prior does the work the search does at `m = 5`.
The tension between item 9's reachability requirement and rider (c)'s
discrimination requirement is recorded in F-11.

---

## Gate 12 — zero-value ablation (diagnostic, not a threshold)

The W/D/L head is trained at loss weight **0** by design (spec §5; Phase-1 value
is the steps-to-solve regression), so its output at inference is an untrained
head's noise. The ablation forces value to 0 and re-runs.

| arm | depth ≤ 2 solve rate |
|---|---:|
| trained (value head live) | 1.0000 |
| **zero-value ablation** | **1.0000** |

**Result: the untrained value head is inert.** Identical to the digit at both
`m = 1` and `m = 5`. It neither helps nor hurts backup at this budget, which is
the outcome the weight-0 decision predicted and the first evidence for it. Chunk
9 wakes the head; this is the before-reading.

---

## Gate 13 — reproducible from seed

**Declared tolerance: exact.** Not "close" — bit-identical loss curves. On CPU
with every RNG seeded from config there is no source of nondeterminism to
tolerate, and a tolerance would be a place for real drift to hide.

**Declared scope:** 5 steps at batch 16, exercising sampling, batch construction,
the crop, both losses, clipping and the optimiser step. It does not cover
long-run accumulation; a 2 × 83-minute repeat is not a `make test` cost, and
saying so is cheaper than implying coverage that does not exist.

**Verdict: PASS.** Seed-identical runs produce identical loss lists; a different
seed produces a different one, so the assertion cannot pass on a loop that
ignores its inputs. Pinned permanently as
`tests/test_train.py::test_training_is_reproducible_from_seed`.

---

## The six informational baselines — `solve_in_1..6` @ 16 sims, m = 5

Not gates. Recorded so chunk 9 opens against numbers rather than impressions.
Record: `runs/gate_phase1_baselines.json`.

| suite | trained | stub null | zero-value | trained median steps | par |
|---|---:|---:|---:|---:|---:|
| `solve_in_1` | **1.0000** | 1.0000 | 1.0000 | 1 | 1 |
| `solve_in_2` | **1.0000** | 0.8350 | 1.0000 | 2 | 2 |
| `solve_in_3` | **1.0000** | 0.5350 | 1.0000 | 3 | 3 |
| `solve_in_4` | **1.0000** | 0.5200 | 1.0000 | 4 | 4 |
| `solve_in_5` | **1.0000** | 0.5450 | 1.0000 | 5 | 5 |
| `solve_in_6` | **1.0000** | 0.5350 | 1.0000 | 6 | 6 |

**1,200 of 1,200**, and the median step count equals par at every depth — the
warm start is not merely solving, it is solving at par on at least half of every
suite. The stub null collapses to ~0.53 from depth 3 onward, so the separation
is 0.465 at the deep end where it matters. The zero-value ablation is identical
at every depth: the untrained W/D/L head is inert throughout, not just at depth 2.

**What these numbers do not say, said here rather than left to be assumed.**

1. **The step distribution against par is not captured — only its median.** In
   this project that distribution *is* z, the game's own currency: equal to par
   is a draw, over par is a loss. A median at par tells us at least half draw; it
   does not give the loss tail. The instrument records `median_steps_when_solved`
   and should record the full histogram. **Registered for chunk 9** rather than
   re-run here — it is a 30-minute measurement and it was not what the brief
   asked for, but the closing report should not imply coverage it does not have.
2. **The step budget is `2 × par + 2`, declared.** A solve at depth 6 had up to
   14 steps available, so "solved" here permits recovery from a wrong move. That
   is the right budget for a *solve rate* and the wrong one for a *par rate*,
   which is the same gap as (1) seen from the other side.
3. **Suite start states are verified absent from training (F-08); their
   intermediate states are not.** A suite problem's solution path may pass through
   states that other problems' derivations contributed. That is not contamination
   under the registered definition — the problem is unseen — but it is the
   honest boundary of what "unseen" covers here.

---

## Gate 14 — anchor by digest

See `runs/phase1/anchor.json` and the `runs/phase1/phase1.pt` line in
`runs/ANCHORS.sha256`. The anchor refuses to write if `phase1_train`'s digests no
longer match what the run recorded — an anchor whose training data was rebuilt
under it is not the anchor anyone thinks it is.

---

## The run

| | |
|---|---:|
| steps | 5,000 (A4 bound 10,000, **one extension unspent**) |
| wall clock | 82.8 min (pilot projected 83.0 — 0.3% error) |
| device / batch | CPU / 128, cosine + 200 warmup |
| NaN skips / encode skips | 0 / 0 |
| loss, first → last windowed mean | 1.4901 → 0.0673 |
| overall top-1, untrained → trained | 0.4791 → 0.9800 |

**No extension**, by a rule fixed before the numbers landed: loss improvement
over the final 500 steps was 1.8% against a 5% bar, and the registered gate was
not short. F-10 supplies a third reason — extending toward a gate that random
initialisation already passes is meaningless.
