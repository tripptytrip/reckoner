# The rehearsal-fraction sweep — selection rule, frozen before the sweep runs

**Written 2026-08-17, before any arm has been run.** F-31 wired the rehearsal
path and proved it inert at `f = 0.0`. Choosing `f > 0` is the treatment
decision, and it lands as M1-A4 carrying this sweep's numbers.

The rule below is registered **first** for a reason specific to this metric.

---

## 1. Why the rule cannot be chosen after the numbers

**Top-1 is monotone in `f` by construction.** More supervised data per batch
means a better supervised metric. So scoring arms by top-1 alone drives `f`
toward 1: maximum preservation, zero learning, and a "winner" that has stopped
doing the experiment.

An objective that a knob trivially maximises is not an objective, and the moment
to notice that is before the table exists — the same reason `s*` was frozen
before its sweep rather than read off it.

## 2. The selection rule

> **The smallest `f` whose gate-10b top-1 holds the band.**

Smallest, not best: the sweep is buying the *minimum* supervision that protects
the warm start, because every supervised row is a self-play row not trained on,
and the campaign's subject is what self-play teaches.

## 3. The band, derived from §8 rather than chosen

§8 makes a no-regress breach BLOCKED, so the tolerable top-1 loss is whatever
keeps at-par above the floor — not a number anyone picks.

**Measured sensitivity**, from the rehearsal's own checkpoints:

| | anchor | `ckpt-4` | Δ |
|---|---:|---:|---:|
| gate-10b top-1 | 0.9699 | 0.8845 | −0.0854 |
| at-par @48 | 1193/1200 = 0.99417 | 910/1200 = 0.75833 | −0.23583 |

**2.7615** at-par points per top-1 point.

The floor's slack is `1193 → 1188` = **5 problems** = 0.42 points, so the
tolerated top-1 loss is `0.004167 / 2.7615` = **0.151 points**:

> ### BAND: gate-10b top-1 ≥ **0.968**

**Caveat, registered rather than buried:** this is a linear extrapolation across
a wide interval. An independent multiplicative model — at-par ≈ mean of `tᵏ` over
suite depths 1–3, whose local slope at `t = 0.9699` is 1.9206 — gives a band of
0.9677. The chord is the steeper slope and therefore the tighter band, and
**0.968 is the conservative of the two.**

The direction is unambiguous whichever model is used: **§8 permits essentially no
forgetting at all.**

## 3.1 The band SCREENS; the cadence DECIDES

Stated here so a passing arm cannot be over-read later. The sweep measures **one
training iteration on ring-0** — the right probe, because F-27 established the
damage is one-shot at iteration 0 — but the actual gate is `at-par ≥ 1188`
measured at **iteration 4**, which only rehearsal attempt 2 can produce.

So a passing arm is a **candidate**, and M1-A4's value stays **provisional** until
the rehearsal's cadence unit confirms it.

Conservative banding is right here for an asymmetric reason, not a general one:
§8 is a BLOCKED branch. A band too tight costs one rejected arm and perhaps a
second round; a band too loose costs a 110-minute cadence unit, a breached floor,
and a halted campaign.

## 4. Arms

`f ∈ {0.00, 0.10, 0.15, 0.25, 0.35, 0.50}`, each trained from the anchor on the
**same fixed ring-0**, same seeds — so `f` is the only variable — and scored on
gate 10b, top-1 depth ≤ 3, F-09 unseen subset.

`f = 0.00` is the control and must reproduce `ckpt-0`'s **0.8942**. If it does
not, the ring-0 replay is not the same measurement and the sweep is void.

### The table M1-A4 will carry

Three columns, so the amendment shows preservation, mechanism and cost together
rather than asserting the trade-off in prose:

| column | meaning |
|---|---|
| **gate-10b top-1** | preservation — does the warm start survive |
| **ring-epochs** | the mechanism variable, `(1 − f) × 39.2` at iteration 0 |
| **value-head examples/step** | the cost the head pays, `(1 − f) × 128` |

The third column is why this is a trade rather than a free win: Phase-1 rows
carry no `z`, so every point of `f` is a point off the head whose criterion
accrual is already the binding constraint.

## 5. Both branches, registered now

**Some `f` clears the band** → that `f` is M1-A4's value, with the sweep table as
its evidence.

**No `f` clears the band** → **a finding, not a failed round.** It says the
frozen page admits no viable training configuration at 400 steps per iteration,
and lever 2 — epoch scaling, steps proportional to ring size — becomes the *next*
round's single lever. Never a simultaneous change: two levers at once is a
confound, and the whole point of registering this branch is that discovering it
must not become a licence to turn both.

## 6. A consequence to carry onto M1-A4's page

At `f > 0` the value head sees `(1 − f)` of its examples per step: Phase-1 rows
carry no `z` by construction, so they contribute policy and steps loss only.
That slows precisely the head whose criterion accrual is already the binding
constraint — the rehearsal's switch criterion abstained at every iteration with
the rarest class at 1, 6, 7, 12, 17 against a floor of 100.

Not a blocker, and not an argument against rehearsal. It is a cost that belongs
on the amendment's page rather than in a footnote discovered later.

---

**Three failed attempts at any gate → BLOCKED, never a weakened gate.**

---

# Registered prediction — the watchlist's first reading

**Written 2026-08-17, before the equivalence gate runs.**

Fix 2 gives the `ckpt-4` equivalence gate a second job it was not built for: it
is also the **first watchlist measurement**, and it retroactively attributes the
rehearsal's misses — the exact question PREREG §5 was written for and could not
answer when it mattered (F-35).

At `sims = 48`, `ckpt-4` missed **290** of 1200 (at-par 910). The frozen family
is 24, certified at `sims = 1`.

| quantity | registered call |
|---|---|
| `family_remaining` | **24** — the anchor's entire hard family still missed by a degraded model |
| `novel_misses` | **≈ 266** — the complement |

**The informative branch is the low one.** `family_remaining < 24` would mean the
degraded model *solved* some of the anchor's hardest problems while losing 283
easier ones — genuinely strange, and worth its own look rather than a shrug.

Registered because a prediction written after the number arrives is a
description, and because this is the first reading of an instrument that has
never produced one.
