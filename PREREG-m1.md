# PREREG-m1.md — the M1 campaign, frozen 2026-08-16

**Amendment policy.** Pre-registration. Anything below may be amended *only* by
appending a dated block that states what changed, why, and what was already known
when it changed. No line above an amendment is edited. An amendment written after
seeing a measurement it affects must say so in its first sentence. Amendments to
this file are labelled `M1-A<n>`.

**Status at commit: M1 has not run.** No number below is a campaign result. Every
number below is either a pre-launch measurement with its own record, or a
threshold derived from one on this page. Per standing law — *rule before
measurement, all the way down* — this file is committed first and the loop is run
against it.

---

## 1. How the primary was chosen, by mechanism rather than choice

`RULING-chunk11-primary.md` (2026-08-15) named search economy as the primary and
froze the rule that would site it: sweep the anchor's at-par rate over
`sims ∈ {6, 8, 12, 16, 48}`, take `s* =` the budget landing nearest 0.55 within
`[0.4, 0.7]`. It also pre-registered the succession in advance: *if the economy
primary saturates, the mid-strata suite is M2's primary, declared now, in
PREREG-m1's own text.*

The rule ran and returned nothing to select:

| sims | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 48 |
|---|---|---|---|---|---|---|---|---|---|
| at-par | 1176 | 1176 | 1176 | 1176 | 1189 | 1189 | 1189 | 1189 | 1193 |
| rate | .9800 | .9800 | .9800 | .9800 | .9908 | .9908 | .9908 | .9908 | .9942 |

Every rung sits above the window's upper edge of 0.70. The rule fired
`all_above_window` and extended downward over `(1, 2, 3, 4)` under `P11B-A3`; at
`sims = 1` the domain has no floor left to extend into, and `P11B-A4`'s
reconciled `select()` returned:

```
s_star: null      needs: succession      at_domain_floor: true
successor_strata: [7, 8, 10]
```

**The anchor pars 98.00% of the 1,200-problem instrument at one simulation.**
Phase-1 imitation of BFS-optimal derivations already amortized the search into the
net — the AlphaZero effect arrived before the loop ran — so the suite-economy axis
is saturated at the floor of its own domain. Succession fires at iteration 0.

The record of that chain: `runs/chunk11_part0b_sweep.json` (union of two
invocations, both source digests and the judging `select()` version stamped
in-file), `PREREG-chunk11-part0bc.md` amendments A3/A4/A5.

---

## 2. P1 — the primary

**CI-separated improvement in pooled beat-par rate on the scripted strata
`{7, 8, 10}`, paired against the Phase-1 anchor.**

| slot | value |
|---|---|
| population | `scripted_in_7`, `scripted_in_8`, `scripted_in_10` — 200 each, **600 problems** |
| metric | pooled beat-par rate (the `<0` bin: strictly under scripted par) |
| pairing | **per problem**, against the anchor's Part-0d outcomes |
| test of record | **paired-difference bootstrap**, criterion **CI excludes zero** |
| reporting | pooled rate **and** per-stratum rates, every ladder pass |
| campaign | ~20 iterations at `--ladder-every 5` |

`scripted_in_9` is **informational only** and is not in the primary. Part-0d found
it the sole stratum born saturated from above — beat-par 164/200 = 0.82, past the
≥0.50 definition — so its trajectory is ceiling-bound at birth and would decorate
the RUNLOG without informing it.

**Why beat-par is the live cell here.** Scripted par is a *provisional floor*, not
an exact minimum, so beating it is legal on these strata — which is precisely what
the instrument was minted to allow, and what the exact-par suites cannot offer.

### 2.1 The four-tuple, filled by measurement

| slot | value | source |
|---|---:|---|
| **floor** | **0.000000** | a-priori: a beat-par rate has no value below zero |
| **null** | **0.000000** | **measured**, Part-0e: uniform stub, `0/600` |
| **baseline** | **0.168333** | measured, Part-0d: anchor `101/600` |
| **ceiling** | **1.000000** | distance from baseline: **0.831667** |

The null does not merely sit low on the scale — **it sits on the scale's floor.**
At `sims = 48` the uniform stub solves nothing at all on these strata: 600 of 600
unsolved, every episode `z = −1`, mean z exactly `−1.0000` on each of the three.
So every point of beat-par rate the loop earns is signal over a reference
contributing literally zero, and the anchor's 101 beats and 12 outright failures
demonstrate both ends of the instrument's working range before the campaign
touches it.

Record: `runs/chunk11_part0e_null.json`, `runs/chunk11_part0d_scripted_strata.json`.

---

## 3. The evaluation protocol, printed verbatim

Inherited from Part-0d without restatement, because **the baseline's protocol is
the primary's protocol** — a baseline measured under one protocol and a primary
measured under another are not paired, whatever the pairing code does. Printed
rather than referenced, because a protocol referenced is a protocol someone will
one day reconstruct differently.

```
sims                48
gumbel_m            16          via the declared clamp m = min(16, sims)
step_cap            24
root_noise          false
value_scale         0.0         value-silent until the switch criterion fires
measure_dtype       fp32
seed                0
device              cpu
model               runs/phase1/phase1.pt
config_fingerprint  f1d258a1161c5ba17f031a3ad89fae0e688ab135f8ced44c1dcd01b1e89491c1
```

The m-clamp is `min(16, sims)`, the frozen rule — **not** `min(5, sims)`. Over
the extended domain it binds trivially (`m = sims` for `sims ≤ 4`); at the
protocol budget of 48 it binds at 16.

---

## 4. No-regress

**The loop may not buy low-budget competence by selling high-budget competence** —
and, per the sweep's own gift, it may not sell the one-simulation competence
either. Two at-par floors on the 1,200-problem exact-par instrument, both rows
carrying their derivation.

### 4.1 The integerization convention, declared once for both rows

**Ceiling on the full-precision bound. No intermediate rounding anywhere.**

The justification is semantic, not numeric, which is what lets it travel to every
future floor without re-litigation: **a floor is an inequality.** The gate is
`count ≥ b` where `b` is a real number and counts are integers. Ceiling is the
only integerization that never admits a count the bound itself excludes.
Round-half-up would admit 1166 against a bound of 1166.4945 — a count the declared
construction rejects — and would do so or not depending on where the decimals
happened to fall.

Two notes kept because the record is the argument:

- **The convention had never been chosen.** The existing `sims = 48` row
  (`1187.8295 → 1188`) rounds to 1188 under nearest *and* under ceiling, so it
  never discriminated. The `sims = 1` row is the first case that does.
- **Intermediate rounding flips this row.** At four decimals,
  `0.9721 × 1200 = 1166.52`; at full precision, `1166.4945`. The full-precision
  bound is authoritative — `P11B-A2`'s lesson, on the row `P11B-A2`'s own
  arithmetic sits beside.

### 4.2 Row one — the sims = 48 floor

```
p            = 1193 / 1200                    = 0.9941666667
1 - p                                         = 0.0058333333
p(1-p)/n     = 0.9941666667 x 0.0058333333 / 1200
             = 0.0000048327546296
SE                                            = 0.0021983527
1.96 x SE                                     = 0.0043087713
lower bound  = p - 1.96 SE                    = 0.9898578954
as a count   = lower x 1200                   = 1187.8294744303
CEILING                                       = 1188
```

**Floor: 1188 / 1200 at sims = 48.** Reconfirmed unchanged under the
now-declared rule; this is not a retro-edit of an existing threshold.

### 4.3 Row two — the sims = 1 floor

```
p            = 1176 / 1200                    = 0.9800000000
1 - p                                         = 0.0200000000
p(1-p)/n     = 0.9800000000 x 0.0200000000 / 1200
             = 0.0000163333333333
SE                                            = 0.0040414519
1.96 x SE                                     = 0.0079212457
lower bound  = p - 1.96 SE                    = 0.9720787543
as a count   = lower x 1200                   = 1166.4945051681
CEILING                                       = 1167
```

**Floor: 1167 / 1200 at sims = 1.**

### 4.4 Both floors name their kind

Per the standing law countersigned at `P11B-A2`: every no-regress floor names its
kind where it is declared. **Both rows above are indistinguishability floors.**

> Holding 1188 at `sims = 48`, or 1167 at `sims = 1`, means **not below the
> anchor's own one-sided 95% band** at that budget. It does **not** mean *at least
> as good on every problem*, and it does **not** mean *at least as good on
> average*. Those are three different gates licensing three different sentences,
> and the derivations above make "distinguishable" mean exactly one computable
> thing.

### 4.5 Gates 10b and 11, re-declared on the ruled wiring

| gate | floor | null | threshold | measured |
|---|---:|---:|---:|---:|
| **10b** top-1 rule-site, depth ≤ 3, F-09 unseen subset | 0.5241 | 0.6803 uniform / 0.6950 untrained | 0.9000 | **0.9699** |
| **11** depth ≤ 2 solve rate, 16-sim search, value-silent-until-criterion (row 3) | 0.0000 | 0.9175 | 0.9500 | **1.0000** |

Gate 11 is carried at **row 3** — the ruled permanent wiring — because rows 1 and
2 are records of what was true on wiring since ruled non-permanent, and verdicts
do not retro-edit.

---

## 5. Informational rungs

At matched `s*` the ladder rungs report but do not gate: greedy and sympy against
the model, **both currencies in their lanes**. sympy is never par — a ladder rung
only, unless its derivations compile into our rule vocabulary.

`scripted_in_9` joins them as informational, per §2.

---

## 6. The entropy funnel signature, named

`p2_d`'s central finding — *the loop collapses when left alone* — enters here as a
design input rather than a surprise. Phase 3's diversity treatment fires when the
signature appears, so the signature is named before it can be recognised
retrospectively.

**Which H, on which population, by how much.** Four columns are already logged:

```
entropy_prior_step1_start      entropy_prior_step1_reached
entropy_target_step1_start     entropy_target_step1_reached
```

"Start" is an episode's first search; "reached" is every search after at least one
rewrite — the faithful analog of chess's start-vs-book split, an interpretation
and not a transplant, since there is no openings book here.

**The signature is the closing gap on the reached population, not a falling
entropy.** Prior entropy alone cannot distinguish a confident policy from a
collapsed one — that is the chess lesson, and it is why the search-improved target
is logged beside the prior. A *confident* policy has low prior-H while the search
still finds improvement, so the prior→target gap stays open. A *collapsed* policy
has low prior-H **and** low target-H: the search has stopped disagreeing with the
policy because there is nothing left to explore.

**Declared trigger** — both conditions, on the reached population, sustained over
**3 consecutive ladder passes**:

1. `entropy_target_step1_reached` ≤ **50%** of its iteration-0 value, **and**
2. the gap `entropy_target_step1_reached − entropy_prior_step1_reached` ≤ **25%**
   of its iteration-0 value.

`entropy_prior_step1_start` on the fixed start population is the **control**: if
start-H holds while the reached population collapses, the collapse is in the
trajectory the loop generates, not in the policy as such. Sustained over three
passes because Phase 3 is a one-way door and a single noisy pass must not open it.

A `0.0` in any column is a **premise-dependent reading, not a measured entropy** —
an iteration with no reached states has no reached entropy — and the population
sizes are logged beside it so the row's reader can tell.

> **First declared here.** The ruling asked for the signature to be named with its
> arithmetic; these two thresholds and the three-pass persistence are new on this
> page and have never been measured against. They are a trigger for a treatment,
> not a gate on the primary.

---

## 7. The switch criterion, its trajectory, and the watches

### 7.1 The criterion, carried unchanged from `PREREG-chunk9-shakedown.md` E5

| slot | value |
|---|---|
| metric | held-out z **balanced** accuracy |
| floor | **0.0**, marked uninformative — an anti-correlated head scores *below* the trivial model |
| null | **1/K**, K = classes with support, census on the row |
| threshold | **1/K + 0.15** |
| min class support | **100** in the rarest class, else **abstain** |
| behaviour | fires **once**, **ratchets**, every evaluation writes a row |

The margin is priced as a one-way door's error rate over campaign length:
at support 100, `se = 0.0354`, `z = 4.24`, `P(fire | null) = 0.001%` per
evaluation and **0.02% over 20 iterations**.

**Expected trajectory over M1.** The head is value-silent until the criterion
fires (`value_scale = 0.0`), so early iterations are expected to **abstain** on
class support rather than refuse on accuracy: the `+1` cell is the rare class on
the scripted strata, and it is the anchor's 101/600 that must grow before the
rarest class reaches 100. Abstention on support early, then either a fire or a
recorded refusal — and **an abstention is data, not silence**.

### 7.2 Abstention rows

Every evaluation writes to `runs/m1/value_switch.jsonl`, **including the ones
that abstain**, each carrying `class_census` and `smallest_class_support`.
`abstention_census` reports fired / refused / abstained / idle. An abstention
nobody records is indistinguishable from a criterion nobody ran.

### 7.3 The cross-switch watch, armed

Solve-rate-by-depth is compared across the switch iteration. A regression is
**noted and diagnosed**; there is **no auto-revert**. The ratchet stands — a
detector that undoes itself on a bad reading is a detector with a feedback loop,
and the diagnosis is the deliverable.

### 7.4 Pool composition growth

Reported every ladder pass: pool membership and its growth, with sampling
declared (`league.pool_sample: uniform`). A pool that grows without its
composition reported is a par source nobody can audit.

### 7.5 z-by-par_source — the draw-inflation watch

`z_by_par_source` is reported every pass, `+1 / 0 / −1` per source. **Par sources
do not pool.** The watch is for the failure mode where a rising z is bought by
shifting mass between par sources of different strictness rather than by solving
better — scripted par being a beatable provisional floor and BFS-exact par not
being beatable at all. A z that improves while its `par_source` mix moves is a
finding to diagnose before it is a result to report.

---

## 8. The BLOCKED branches

Three failed attempts at any gate → **BLOCKED**, never a weakened gate. A halt is
information; `BLOCKED-<date>-<slug>.md` is committed and tracked like any other
record.

Declared in advance:

- **P1 shows no CI separation over ~20 iterations.** Not a failure to hide: the
  successor axis was certified live by Part-0d — three strata with real headroom
  and 12 outright anchor failures — so a null result on a live instrument is a
  finding about the loop, and it is reported as one.
- **Either no-regress floor breaks** (at-par < 1188 at `sims = 48`, or < 1167 at
  `sims = 1`). BLOCKED, diagnosed before any further iteration. The loop does not
  trade one budget's competence for another's in either direction.
- **Gate 10b or gate 11 regresses** on the ruled wiring. BLOCKED.
- **The draw-inflation watch fires** — z improving while the `par_source` mix
  moves. BLOCKED pending diagnosis; a result bought by mix-shift is not a result.
- **The funnel signature fires.** Not BLOCKED: this is Phase 3's declared trigger,
  and it fires the diversity treatment, which is the pre-registered response.

---

## 9. What is open at the freeze

**Rider (b) item 2 — the failing-set identity — is an open finding and does not
gate this freeze.** Its bar was on the suite-economy axis, which succession has
made moot as a primary, and the no-regress floors it would inform (1188, 1167) are
identity-free aggregates. The diagnostic runs in parallel;
`runs/chunk11_misses_diagnostic.json` is its record.

Its question, registered before its run: the anchor at `sims = 1` misses 24 of
1,200 and the scripted solver misses 24 of the same 1,200. Two systems, one count,
one population — coincidence to dispose of in a line, or the same problems hard
for a greedy symbolic solver and a one-simulation neural policy alike.

---

*The primary was chosen by a rule. The rule was amended only where its inputs did
not exist. The successor was certified live by measurement, not by expectation.
Both floors carry their derivation, their kind, and the convention that
integerized them. Every branch that fired was written before its number was.*
