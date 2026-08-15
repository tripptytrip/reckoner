# PREREG-chunk11-part0bc.md — the s\* rule and the mid-strata protocol, before either runs

**Amendment policy.** Pre-registration. Anything below may be amended *only* by
appending a dated block that states what changed, why, and what was already known
when it changed. No line above an amendment is edited. An amendment written after
seeing a measurement it affects must say so in its first sentence. Amendments to
this file are labelled `P11B-A<n>`.

**Status at commit: neither Part-0b nor Part-0c has run.** No number below is a
result. Per the ruling's order of operations — *rule before measurement, all the
way down* — this file is committed first and the sweep is run against it.

---

## Inherited from F-20's general form

> Branch rules need four-tuples on **every** branch, including the ones they fall
> back to. A pre-registered rule that can only interrogate one of its own arms is
> half a rule.

`PREREG-chunk11-part0.md`'s rule failed exactly there, so every branch below is
given its own disposition — including the branches where the rule finds nothing.

---

# Part 0b — the s\* sweep

## The protocol

The chunk-11 Part 0 protocol, unchanged, with `sims` varying:

| slot | value |
|---|---|
| model | `runs/phase1/phase1.pt`, the anchor |
| `sims` | swept over **{6, 8, 12, 16, 48}** |
| `gumbel_m` | **`min(16, sims)`** — see below |
| `root_noise` | **False** (eval profile), set explicitly |
| `step_cap` | 24 |
| value wiring | value-silent, `value_scale = 0.0` |
| instrument | the six frozen suites, 1,200 problems |
| seed | 0 |

**`m = min(16, sims)` is stated because the default is incoherent below sims 16.**
Gumbel sequential halving considers `m` root actions and halves them under a
simulation budget; `search` clamps `m` to the number of legal root actions but
**not** to `sims`, so `sims=6, m=16` would nominate sixteen candidates and have
six simulations to separate them. That is not a low-budget search, it is a
different algorithm. Clamping is declared here rather than discovered in the
numbers.

## The metric

**At-par rate** = (problems finishing with `steps == par`) / 1,200, i.e. the `0`
bin of the pinned `STEPS_MINUS_PAR_BINS`. The `<0` bin remains structurally
empty on these suites and is reported at every `sims`.

## The s\* selection rule

**Target 0.55, window [0.4, 0.7], nearest wins.** Formally, over the swept
values whose at-par rate lands in `[0.4, 0.7]`, s\* is the one minimising
`|rate − 0.55|`.

Every branch gets its disposition, now:

| branch | disposition |
|---|---|
| **exactly one** swept value in the window | it is s\* |
| **several** in the window | nearest 0.55 wins |
| **a tie** on `\|rate − 0.55\|` | **the smaller `sims` wins** — it is cheaper, and cheapness is the co-benefit the ruling names |
| **none** in the window, but two adjacent swept values **straddle** it (one below 0.4, one above 0.7) | **bisect**: measure the integer midpoint of the bracket and re-apply this table. Repeat until a value lands in the window or the bracket is adjacent integers |
| bracket reaches **adjacent integers** without landing in the window | **the primary cannot be sited mid-scale.** A finding, filed with its numbers, requiring a ruling. **No post-hoc widening of the window** |
| **every** swept rate is **above 0.7**, including `sims = 6` | extend the sweep **downward** over `{1, 2, 3, 4}` and re-apply. `sims = 1` is the no-search limit — the network's raw policy — and if even that is above 0.7 the anchor cannot be de-saturated by budget, which is a finding, not a smaller window |
| **every** swept rate is **below 0.4**, including `sims = 48` | impossible against Part 0's measured 1193/1200 at `sims = 48`; if observed it means the sweep is not measuring what Part 0 measured, and the discrepancy is the finding |

The window and target are fixed here and are not re-openable by a measurement.

## P1's four-tuple — the slots fixed now, the values filled by the runs

| slot | value |
|---|---|
| metric | at-par rate on the frozen suites at s\*, **paired per problem**, model − anchor |
| **floor** | **the uniform stub at s\***, *measured by running it* — `search.uniform_stub`, no network, same sims, same suites. Rider (c): the floor is a run |
| **null** | **the anchor's own paired baseline** — the anchor against itself at s\*, which is **exactly 0 by the self-match identity** and is *run*, not assumed |
| **ceiling** | **1.0**, with `1.0 − anchor_rate` stated as the distance available |
| threshold | the paired-difference bootstrap's **95% CI excludes zero** and the mean is **positive** |
| measured | filled at each ladder pass (`--ladder-every 5`, ~20 iterations) |

## No-regress at the top of the budget

The loop may not buy low-budget competence by selling high-budget competence.
At `sims = 48`, both conditions, and the first is the binding one:

1. **Hard floor: at-par count ≥ 1188 of 1200.** Computed, not chosen: the anchor
   measured 1193/1200 = 0.994167; the binomial standard error at n = 1200 is
   `sqrt(0.994167 × 0.005833 / 1200) = 0.002192`; the lower end of a two-sided
   95% band is `0.994167 − 1.96 × 0.002192 = 0.989871`, i.e. **1187.8 → 1188**.
2. The paired-difference bootstrap at `sims = 48` must **not** have its 95% CI
   lying entirely below zero.

Plus gates 10b and 11 on the ruled wiring, as planned.

---

# Part 0c — the mid-strata suites

## What is minted

`runs/suites/solve_in_7.jsonl` … `solve_in_10.jsonl`, **200 problems each**,
`par_source = "scripted"`.

**Scripted par is a provisional floor and therefore beatable.** That is the
point, not a caveat: `EXACT_PAR_SOURCES = {"bfs"}`, so the `z = +1` tripwire does
**not** fire against a scripted label, the `+1` cell is **live**, and the par
game's escalation architecture finally has a frozen instrument denominated where
racing is possible. The existing suites cannot host it — nothing beats exact par,
which is F-20.

## The obligations, stated before minting

1. **Deeper emission.** New templates in `generator.py` reaching scripted par
   7–10. BFS labelling is not available here: `solve_both_sides_product` already
   costs ~4.5 s median at depth 6, and the branch factor makes depth ≥ 7
   unaffordable. This is *why* the label is scripted, and the reason is recorded
   rather than left as an omission.
2. **Stratum identity is the label.** A problem lands in `solve_in_k` because its
   scripted par **is** k, never because a template intended k. The chunk-5
   precedent: the template's intended depth is never the label.
3. **Censused at BOTH levels** against `train_100k` (problem) and `phase1_train`
   (state), through `pairedset.census`, which now refuses a single-level call.
   The decision rule is the one already fixed: **collisions dropped before the
   freeze**; contamination found after is a finding, never an edit.
4. **Frozen at birth, digested into ANCHORS**, and refused as a training source —
   `runs/suites` is already registered as an instrument.
5. **`scripted_par_delta` reported where BFS is affordable.** For any minted
   problem whose BFS par can be computed under a stated cap, the gap
   `scripted − bfs` is measured and reported. Where BFS is unaffordable the
   absence is recorded **with its reason**, never as a zero gap. The floor
   language is being paid for; this is the payment.

## Succession, declared now rather than when it is needed

**If the economy primary (P1) saturates, the mid-strata suite is M2's primary.**
Written into `PREREG-m1`'s own text at freeze time, before any pass runs — the
chess lesson (an anchor rung saturating while the real signal sat elsewhere)
arriving as a pre-registration instead of as a surprise. "Saturates" means the
same thing it means everywhere else here: the measured headroom falls below the
CI half-width of the test of record, computed the way `runs/chunk11_ceiling.json`
computes it.

## The rungs go informational

`greedy` and `sympy` against the model **at matched s\***, both currencies in
their own lanes, reported every ladder pass and gating nothing. A rung that
cannot move (sympy solved 251/251) is a reference point, not a criterion.

---

## Reported verbatim

The full sweep table (sims × at-par × over-par × capped × stuck × seconds); the
selection arithmetic showing every branch of the rule that was tested and why it
did or did not fire; s\* and its rate; the four-tuple's floor and null **as runs**;
the ceiling distance; the mint's per-stratum counts, censuses, digests, and
`scripted_par_delta` where computable, with its absence reasoned where not.

---

# Amendment P11B-A1 — 2026-08-15, two record items from the mint

**Written after the mint ran**, and it affects no threshold: both items are
transcriptions of arithmetic and sampling that were already fixed. Neither
changes a rule above.

## 1. The no-regress floor's arithmetic, restated on the page

Ordered so the floor is readable rather than inferable. It appears in the
no-regress section above and is repeated here in one place:

```
anchor at-par, measured (Part 0)    1193 / 1200 = 0.994167
binomial SE at n = 1200             sqrt(0.994167 × 0.005833 / 1200) = 0.002192
two-sided 95% lower bound           0.994167 − 1.96 × 0.002192 = 0.989871
as a count                          0.989871 × 1200 = 1187.8  →  1188
```

**1188 = 1193 minus a 95% binomial noise band**, rounded up to a whole problem.
The floor is the bottom of the band the anchor's own measurement could have
landed in by chance, so holding it means "not distinguishably worse than the
anchor", not "at least as good on every problem".

## 2. `scripted_in_9`'s sampling margin

Recorded for any future re-mint. From a 6,000-candidate pool the strata yielded:

| stratum | candidates yielded | frozen | margin |
|---|---|---|---|
| `scripted_in_7` | 2044 | 200 | 10.2× |
| `scripted_in_8` | 1849 | 200 | 9.2× |
| **`scripted_in_9`** | **251** | **200** | **1.26×** |
| `scripted_in_10` | 1707 | 200 | 8.5× |

**`scripted_in_9` is a near-census of its own yield: 200 of 251, 79.7%.** The
other three strata are samples; this one is very nearly the population the pool
produced. A re-mint must not assume diversity headroom exists at par 9 — it does
not, at this pool size and template set. `solve_mid_four_terms` lands on 10 far
more often than 9, and par 9 is the thin cell between two thick ones.

The census at par 9 therefore ran against 251 candidates rather than the 400 the
other strata got; it returned 0 collisions at both levels, as they all did.

---

# Amendment P11B-A2 — 2026-08-15, a fourth-decimal correction and a floor taxonomy

**Written after P11B-A1 and after the mint**, correcting an arithmetic slip in
P11B-A1 itself. It changes no threshold: the count of **1188 is unchanged**.

## 1. The standard error, recomputed

P11B-A1 and the no-regress section print `SE = 0.002192`. Recomputed:

```
p              = 1193 / 1200      = 0.994166667
p(1 − p) / n   = 0.994166667 × 0.005833333 / 1200
SE             = 0.002198353      (printed: 0.002192)
1.96 × SE      = 0.004308771
lower bound    = 0.989857895
as a count     = 1187.8295        →  1188
```

The printed SE was low by `6.4e-6`. **The lower bound and the floor of 1188 are
unchanged**, which is why this is a note rather than an amendment to the gate.
Caught by the principal recomputing rather than reading — the same move that
produced E5-1 and E5-2 on the same day.

## 2. Every no-regress floor states which kind it is

Countersigned into the record, because it will matter every time a floor is
quoted:

> A **noise-band floor is an indistinguishability gate.** Holding 1188 means
> **not distinguishably worse than the anchor** — it does not mean **at least as
> good on every problem**, and it does not mean **at least as good on average**.

These are three different gates and they license three different sentences. From
here, **every no-regress floor in this project names its kind** at the point it
is declared. The 1188 floor is an **indistinguishability** floor.
