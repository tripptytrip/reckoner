# BRIEF-chunk11-driver.md — the M1 campaign driver (2026-08-16)
# Chunk 11's unbuilt half. PREREG-m1 (eaa347b) governs; this file composes.

## Part 0
0a. Inventory the existing compositions (golden's loop; the chunk-9
    kill-point harness). The driver PROMOTES the tested composition;
    divergences named, not discovered.

## Design (decided; deviations via BLOCKED)
1. ONE composition: driver = the loop. golden = driver at golden config.
   Kill-point identity tests re-pointed through the driver.
2. Per iteration, in the four-step commit order resume.py specifies —
   ring, state (incl. value-gate declaration), row, LATEST atomic —
   with resume.py's prose re-pointed at the implementation: seed ring
   from anchor → self-play run_iteration (pool-sampled problems,
   frozen-instrument refusal in force) → ring append → train_on_ring →
   checkpoint + enroll → every 5th iteration ladderpass.run_pass →
   switch criterion evaluated, row written incl. abstentions → funnel
   trigger row WITH beat-par delta (M1-A1) → no-regress checks against
   gates.no_regress_floor (1188 @48, 1167 @1, licensed sentences) →
   alarms census → telemetry sidecar running throughout.
3. No behavioral flags. Config fingerprint asserted == PREREG-m1's
   recorded fingerprint at startup; refusal both polarities tested.
4. Breach handling per PREREG-m1's decision rules, implemented as
   written there; driver behavior at breach and at no-breach both
   pinned by test. Structural failures (refused rows, version
   mismatch) halt; statistical breaches surface and follow the
   prereg's branch.

## DONE-WHEN
- make golden green via the driver, budget unchanged.
- Both kill points through the driver: killed-and-resumed identical
  to uninterrupted, row for row.
- Fingerprint refusal: mismatched config refused; matched accepted.
- 3-iteration dress rehearsal at CAMPAIGN config, expectations
  committed first: every row class present (iteration, ladder,
  switch, funnel, pool composition, no-regress, alarms: 0), deleted
  after recording.
Three failed attempts at any gate → BLOCKED, never a weakened gate.

---

# Amendment D-A1 — 2026-08-16, three items reclassified, and the seam pinned

**Written after Part 0a ran** (`RUNLOG-chunk11-driver-part0.md`, `f207b33`) and
because of what it found. A brief's premise meeting measurement and losing is
exactly what this channel records; the premise that lost is "the driver promotes
the tested composition", which is true of most of it and false of three parts.

## 1. Three items are build-not-promote

Part 0a's evidence: `golden.py` runs
`run_iteration(problems, uniform_stub(cfg), ...)` every iteration. Golden asserts
**that the model moves** — the F-14-era fix — and never **that the moved model is
used**. The weights train; the episodes they train on came from flat priors, by
construction, every time. *That path is the loop's entire subject.* The same shape
holds twice more: pool composition grows and no par has ever come from it; the
switch criterion's row-writing is exercised and its real inputs never are.

This is the L5 recursion at its terminal level — an assertion that the central job
*happened* standing in for an assertion that the central job's **product is
consumed**. So each of the three carries its own central-job assertion, **both
polarities**, so the class cannot recur one level higher.

### 1.1 Model-in-search

The evaluator is **the current checkpoint**, and the iteration row asserts it **by
digest** — provenance, not configuration, because a configuration field records
what was asked for and a digest records what ran.

- **Positive:** golden asserts the evaluator is not the stub.
- **Contrast:** a stub-configured invocation **must fail** that assertion.

### 1.2 Pool par exercised

Golden-scale config makes pool-par episodes **expected**, not incidental.

- **Positive:** assert ≥ 1 episode occurred with `par_source="pool"`, its
  provenance intact on the row.
- **Contrast:** the empty-pool case keeps its counted-state behaviour — zero
  pool-par episodes is correct there and must stay correct.

### 1.3 The criterion fed real inputs

`consider_switch` consumes `evaluate_head` on the **held-out ring slice**, and the
row records **its inputs' source**.

- Golden's synthetic-label path is **removed, not bypassed** — a path that still
  exists is a path something can fall back to.
- The criterion's **first real exercise is the dress rehearsal**, where its
  abstention reasons become data exactly as PREREG-m1 §7.2 pre-registers.

## 2. Finding 2 adopted, with two additions

The mono-instance law binds the **loop**. `Harness` sits a level below, and its
synthetic payloads are the feature rather than a shortcut: a kill test that can
fail for reasons which are not the commit contract is a worse test. That is the
reference-matcher independence argument, pointed at a filesystem.

Adopted as proposed, plus:

**(a) A seam-identity pin.** The driver **provably calls `commit_iteration`**,
asserted at source level, in the shape of the one-legality-oracle test. *"Probes
the seam the driver also uses"* is only true while nothing reimplements the seam,
and that is an assertion, not a hope.

**(b) The driver-level kill test covers both points** at rehearsal scale on the
real composition — cheap at three iterations, and it is what the DONE-WHEN's
sentence actually claims. `Harness` stays as the contract's fast isolation probe,
now provably probing the driver's own path.

## 3. The pilot law, extended to output paths

Three raises in one day — `cfg.model.measure_dtype` (178s lost),
`-((-x) // 1)` as a ceiling (caught only by independent derivation),
`problem_key` used as a JSON key (≈47 min lost) — all shared one shape: an API
asserted from memory, raising **after the expensive middle**. Combined across both
parties the class is past eleven, and at that count the remedy is not
resolve-harder; it is mechanical.

**F-03's pilot law extends: a long-running script smokes its own output path
before it earns the right to spend.** A micro-invocation through the *full* path —
including report-writing — at two problems and a few seconds. `cfg.numerics` would
have cost 5 seconds instead of 178; the JSON key 5 seconds instead of 47 minutes.

In the driver this is **not a flag**: a mandatory pre-flight before iteration 0
that runs one micro-iteration through every row class into a scratch directory,
validates each row, and discards it. Retrofitting the chunk-11 measurement
scripts follows.
