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
