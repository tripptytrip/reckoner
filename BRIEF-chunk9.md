# Chunk 9 brief — registered obligations

Chunk 9 is *Phase 2: the loop*. Its DONE-WHEN is in `experiment2_agent_plan.md`.
This file carries obligations registered by earlier chunks so none has to be
remembered.

---

## CheckpointPool must refuse a version mismatch

**Registered:** 2026-08-14, chunk 6.

`CheckpointPool` loads snapshots to supply **pool par** (`league.par_from_pool_frac`,
default 0.20). A snapshot whose `ruleset_version` or `vocab_version` differs from
the running environment must be **refused, not loaded**.

**Why, precisely.** Pool par is a par *re-solved by an old snapshot at episode
time*. If that snapshot indexes a different action space or a different symbol
space, the par it produces is denominated in a rule system that is not the one
being played — which is `FINDINGS.md` F-02 reborn one layer up, and harder to
see: F-02 shipped a wrong number with a provenance tag that said `bfs`, and this
would ship a wrong number with a provenance tag that says `pool` and is, in its
own terms, true.

**Already in place:** `model.load_checkpoint` refuses on mismatch by default and
`checkpoint_meta` stamps both versions. Chunk 9 must not pass
`strict_versions=False` in the pool path — the escape hatch exists for offline
inspection, not for the league.

**Test it on both polarities**, as always: a matching snapshot loads, a mismatched
one raises, and the pool surfaces the refusal as a counted event rather than a
silently smaller pool.

---

## The z/q blend's hazard is structurally absent — say so where it is used

`train.value_q_mse_weight` is 0.5 from day one. The self-referential hazard that
makes blending dangerous elsewhere does not exist here because the **checker**,
not the model's own Q, decides solved. State that in the loss docstring so a
later reader does not "fix" it by turning the blend off.

---

## Snapshot par and the win-condition invariant

`EpisodeResult` raises when `z > 0` against an exact `par_source`. Pool par is
*not* exact — beating it is the whole escalation mechanism — so pool-labelled
episodes will legitimately produce `z = +1`. Do not widen the invariant to
silence them; that is the difference the invariant exists to preserve.
