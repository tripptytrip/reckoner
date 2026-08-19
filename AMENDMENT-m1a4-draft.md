# Amendment M1-A4 — DRAFT, four decisions settled, one slot open

**Status: DRAFT.** Four decisions below are settled and carry their evidence.
The fifth — `train.rehearsal_frac`'s value — is **held open** pending the
extended sweep, and this page lands as **one commit** when that returns.

Batched deliberately: every item here moves the config fingerprint, and the
batching rule exists so a fingerprint moves once with a page explaining all of
it, rather than five times with five partial explanations.

---

## 1. The rung passes are dropped from M1 (F-34)

**Decided. Both options priced; (b) taken on measurement, not cost.**

M1-A2 §2 prices a cadence term — model rung 200 @48 at 5.0 min, greedy rung at
0.0 — that corresponds to no code. Building it needs a population and a budget,
and **both referents evaporated**: PREREG calls the rungs *informational at
matched `s*`*, and `s*` resolved to **null** when branch (c) fired.

| option | verdict |
|---|---|
| **(a)** `smoke_v1` entire — 389 problems, born clean of F-09, ANCHORS-registered, purpose-built for the ladder, ~10–17 min | **rejected** |
| **(b)** drop the rungs from M1, carry to M2 | **taken** |

(a) is rejected because M-B already measured the instrument **saturated**: par is
BFS-exact so `+1` is impossible, the model sits at 387/389, and the entire
remaining headroom is **12% of one CI half-width**. A rung pinned at its ceiling
reports the ceiling. The suites are saturated too (F-20), and the scripted strata
are the only live population — already the primary, where a rung would duplicate
rather than inform.

Nothing gates on their absence; PREREG calls them informational. M2 mints a rung
population with **headroom demonstrated before freezing**, which is the lesson the
succession to the scripted strata already taught once.

## 2. `league.snapshot_every` is honoured by the driver (F-36)

**Decided, built, and proved inert.**

`shakedown.py` honoured it; the driver enrolled unconditionally. They agreed only
because the default is **1** — correct by coincidence, the fourth appearance of
that shape, and the first caught before it could diverge. At 5 the two
compositions would differ in pool growth and therefore **par escalation**, the
mechanism the primary is denominated against.

Proved inert at the current value: **pool composition bit-identical** across the
change at `snapshot_every = 1`. Tested at both values, because "agrees at the
default" is precisely the evidence that let the divergence sit. One predicate
serves the enrolment site and resume's pool rebuild, since a rebuild replaying a
different cadence would reconstruct a pool the run never had.

## 3. The watchlist emits a pair per leg, on the pass record (F-35)

**Decided and built.**

Per pass, not at `min(budgets)`: each leg has its own miss set, and the 48-sim
reading is the one **commensurate with the primary's budget**. Reading only at
`min(budgets)` is what let a prediction registered at 48 sims be scored against a
1-sim number — 24/266 predicted, 20/458 measured at 1 sim, while the 48-sim
reading was **7/283** and told a different story entirely.

It rides on the **pass record**, not as budget-named iteration columns: a budget
in a column name puts the budget in two homes — config and schema — where they
can drift. The partition identity holds where its inputs live, asserted at
construction.

**Era 4 is redefined away.** No committed artifact carried an era-4 row, and the
triple moved off the iteration row entirely, so the iteration schema returns to
its era-3 shape and `SCHEMA_ERA` returns to 3. An era is frozen only once
something durable claims it.

### 3.1 The first reading, and what it establishes

| | misses | `family_remaining` | `novel_misses` |
|---|---|---|---|
| ckpt-4 @48 | 290 | **7** | 283 |
| ckpt-4 @1 | 478 | 20 | 458 |
| anchor @48 | 7 | 7 | **0** |
| anchor @1 | 24 | 24 | **0** |

ckpt-4's seven family misses at 48 sims are the **identical set** to the anchor's
seven; both rescue the same 17 of 24.

**So the frozen family is not where the degradation lives.** All 283 novel misses
are problems the anchor solved. Top-1 ordering carries the easy population;
search carries the hard family; forgetting destroys the former and leaves the
latter untouched.

**Consequence for how the campaign's own numbers are read:** the primary is a
**low-sensitivity instrument for this failure mode.** Its population is
search-carried, so it *rose by 7* while the model lost 283 easy problems. §8's
"either direction" clause names budgets; this is the **population** version of the
same trade, measured rather than assumed. The floors caught it only because they
happen to sit on the sensitive population.

## 4. Dead-key dispositions (F-31's census)

**Decided.** Batched here because each deletion moves the fingerprint.

| field | disposition |
|---|---|
| `train.rehearsal_frac` | **wired** (F-31) — value in §5 |
| `campaign.interop_threads` | **kept, now verified** — OBSERVED per M1-A2 §4, asserted against the runtime |
| `ladder.problems_per_pass` | **wired** as the per-file population cap; inert at campaign config, where every file is exactly 200 |
| `league.snapshot_every` | **wired** (§2) |
| `ladder.bootstrap_resamples` | **kept** — live once F-30's baseline arm is consumed by the bootstrap |
| `search.perspective` | **kept, exempt** — pinned invariant, legal range is one value |
| `generator.{train_set_size, suite_depths, suite_problems_per_depth, max_bfs_depth}` | **DELETE** — unbacked and CLI-shadowed (F-33) |
| `model.{param_budget_min, param_budget_max}` | **DELETE** — unbacked sizing guides, never consulted |
| `ladder.{sympy_step_budget, sympy_time_budget_s}` | **DELETE** — unbacked; the sympy arm carries its own |

### 4.1 Ratified: the `assert_eval_profile` split (touches D-A2 §2)

The registered eval fingerprint was asserted unconditionally, which made the
shared seam **unrunnable at golden config** — and that is how the cadence path
came to have no test at all, and how two campaign-blocking defects reached the
pod.

Split: **the profile must ACT** (root noise off) is asserted on every pass,
whoever calls; the **registered pin** is asserted when the pass is a campaign
measurement. Every campaign pass still asserts the registered value at the
boundary where the profile acts, so §2 is preserved rather than weakened.

The predicate is **derived from the config**, never passed — a `campaign=`
argument would be a caller-supplied claim about evidential status, which is
F-19's currency tag wearing a keyword. A test asserts the *signature*, so
reintroducing a caller flag fails rather than being caught in review.

---

## 5. `train.rehearsal_frac` — SLOT HELD OPEN

Pending the extended sweep (`SWEEP-m1a4.md`, amendment of 2026-08-19). The first
sweep returned **no arm holding the band**, with the control reproducing
`ckpt-0`'s 0.8942 exactly — but it carried a shared-RNG confound, had no variance
estimate against a 0.0062 shortfall, and stopped at `f = 0.50`.

The re-run fixes the confound, adds a second seed on the control and the best
arm, extends to 0.65 and 0.75, and adds the **mechanism arm** — `f = 0.00` at 200
steps, 19.6 ring-epochs matched to `f = 0.50`, no supervision — which decides
whether round two's lever is rehearsal or epoch scaling.

**This page lands when that returns, not before.**
