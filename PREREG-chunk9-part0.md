# PREREG-chunk9-part0.md — ROCm migration: tolerances declared before measuring

**Amendment policy.** This file is pre-registration. Anything below may be
amended *only* by appending a dated amendment block that states what changed,
why, and what was already known when it changed. No line above an amendment is
edited. An amendment written after seeing a measurement it affects must say so in
its first sentence — that is the whole point of the header, and a pre-registration
that can be quietly edited is a lab notebook written in pencil.

**Status at commit: nothing has been measured.** The venv is still
`torch 2.13.0+cpu`. Every number below is a threshold, not a result. This file is
committed *before* `rm -rf .venv`, so the declaration cannot be retrofitted to
whatever the hardware turns out to do.

---

## 0c. Box-occupancy declaration — made first, because it constrains the rest

Read 2026-08-15, before any GPU work, `rocm-smi`:

| | |
|---|---:|
| VRAM total | 103,079,215,104 B (96.0 GiB) |
| VRAM in use | 1,811,087,360 B (1.69 GiB) |
| KFD processes | **none** |
| Host RAM free / available | 14 GiB / 20 GiB of 30 GiB |
| Swap in use | 3.4 GiB of 8.0 GiB |

**Declaration.** The chess project is **not resident** — no KFD processes, 1.69 GiB
held by the desktop compositor. reckoner may therefore take the GPU, and this
chunk claims:

* **VRAM ceiling: 16 GiB.** A 5.07 M-parameter model with batch 128 at width ≤ 64
  cannot legitimately need more; anything approaching the ceiling is a bug, not a
  capacity need, and the ceiling exists to make it show up as one.
* **reckoner yields to a resident chess process.** Before any GPU run, `rocm-smi
  --showpids` is checked and a non-empty KFD list means reckoner runs CPU or
  waits. It does not co-tenant a 96 GiB pool with a project that assumes it owns
  it.
* **Host RAM stays the scarce pool** (AGENTS.md §3). Swap is already 3.4 GiB deep;
  moving training to the GPU must not be taken as licence to load datasets that
  are currently mapped.

## 0a. The blessed stack

| | |
|---|---|
| index | `https://download.pytorch.org/whl/rocm7.2` (≥ rocm7.1 required for gfx1151) |
| fallback | `https://repo.amd.com/rocm/whl/gfx1151/` |
| install | `rm -rf .venv && make install` — rebuild, never surgery (§4 step 2) |
| env | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` |
| `torch.compile` | **OFF**, and stays off this chunk (gfx1151) |
| `HSA_OVERRIDE_GFX_VERSION` | **never** — needing it means an old wheel snuck in |

`uv.lock` changes, so the clean-clone gate is re-run and `make lint` (`uv lock
--check`) must be green. `check_env.py` output lands in the run directory, and it
**must not** report a CUDA wheel — it exits non-zero on one.

## 0b. GPU/CPU equivalence — the tolerances, declared

**Bit-identity across devices is not expected and is not the gate.** Different
reduction orders in matmul and attention produce different last bits in fp32.
What must hold is that the *decisions the system consumes* are unchanged.

Fixed input: **512 states sampled via `dataset.sample_indices(n, 512, seed=0)`**
from `phase1_eval` — the blessed subsampler, not a prefix. Same
`runs/phase1/phase1.pt` weights, `model.eval()`, fp32, both devices.

### Tier 1 — decision equivalence. This is the gate.

| | |
|---|---:|
| legal-masked **argmax action** identical, CPU vs GPU | **512 / 512 (1.0000)** |
| legal-masked **top-8 set** identical (as a set) | **512 / 512 (1.0000)** |

Threshold is exact agreement. These are the quantities search and the top-k
metric actually read; if they agree, the device change is invisible to every
gate this project has.

### Tier 2 — numeric divergence. Diagnostic, with bounds.

| quantity | declared bound (max abs) |
|---|---:|
| policy logits | ≤ **1e-3** |
| value probabilities (post-softmax) | ≤ **1e-4** |
| steps head | ≤ **1e-3** |

Logits measured on this model span ~10.3, so 1e-3 absolute is ~1e-4 relative.
**If tier 2 is exceeded while tier 1 holds, that is a finding, not a failure** —
it means the margins are wider than the noise and the gate still stands, and the
number gets recorded rather than the bound quietly widened.

### Both polarities — the null

A comparison that cannot fail is not a comparison. **Perturbation: add `1e-2` to
`rule_embedding.weight[0, 0]`** on the GPU copy only, then re-run tier 1.

| | |
|---|---:|
| perturbed-arm argmax agreement | **must be < 1.0000** |

If a perturbation that size cannot move a single argmax across 512 states, the
comparison is measuring nothing and the smoke is void regardless of what the
unperturbed arm says.

### Four-tuple for 0b (rider (c))

| | |
|---|---|
| floor | not applicable a-priori; two identical computations agree trivially, which is why the null is a *run* |
| null | perturbed-weight arm's argmax agreement, **measured**, must be < 1.0000 |
| threshold | 1.0000 argmax agreement, unperturbed |
| measured | — *(not yet measured; this file is the declaration)* |

### bf16, stated separately and not gated here

AGENTS.md §4.5 prescribes bf16 autocast with fp32 master weights for GPU
training. **Tier 1 is not expected to hold under bf16** and is not asserted
there. The bf16 argmax agreement rate is recorded as an informational number in
the same artifact, because "we did not test it" and "we tested it and it was
0.97" are different states and only one of them is knowledge.

## 0d. Scope

Four-tuples on every gate this chunk declares. Three failed attempts at any gate
⇒ `BLOCKED-<date>-<topic>.md`, never a weakened gate — and for this file
specifically, never a widened tolerance.

## Registered risk, stated before it can be claimed as foresight

`rm -rf .venv` destroys the environment every chunk-8 number was measured in. The
`uv.lock` currently committed pins the CPU stack, so the rebuild is reversible by
`git checkout` of `pyproject.toml` + `uv.lock` followed by `make install` — that
is the rollback, and it is written here rather than discovered under pressure.
The chunk-8 artifacts are already committed and pushed (`ff53afa`), so nothing
measured depends on the venv surviving.

---

# Amendment A1 — 2026-08-15, before any measurement

**Nothing has been measured.** This amendment is written while the venv is still
`torch 2.13.0+cpu`; it is not a response to a result. It answers three defects the
principal found in the declaration above, and it tightens rather than widens every
one of them.

## A1.1 — The null was falsifiable in the wrong direction. **Form (a) is chosen.**

The declared null — "a 1e-2 nudge to `rule_embedding.weight[0,0]` must flip at
least one argmax across 512 states" — can fail while the pipeline is perfectly
correct. Post-training decision margins are wide (top-1 is 0.98), logit spreads
are ~10.3, and a single-scalar 1e-2 nudge is plausibly sub-margin everywhere. The
null would then read "the comparison measures nothing" about a comparison that was
measuring fine. **A null that can fail spuriously is the mirror of a null that
reads high: both misattribute, one in each direction** (F-11 was the other
direction).

**The null's actual job** is to prove the harness is wired to two different
things rather than comparing a tensor to itself. It is therefore asserted on the
*numeric* channel, which responds to any nonzero change, not on the *decision*
channel, which responds only above margin. Two parts, both required:

**(i) Comparator self-check — exact, and guaranteed by construction.** Given a
tensor `T` and `T + 1.0`, the comparison harness must report `max|Δ| == 1.0`
exactly. This cannot pass on a harness that compares a tensor to itself, and its
result does not depend on the model, the device, or any margin.

**(ii) End-to-end perturbation — sized to clear the tier-2 bound, not to flip a
decision.** Perturbation is enlarged from one scalar to **`rule_embedding.weight[0] += 1.0`**,
the whole 128-dim row. Since `logit[b,0,s] = Σ_d site_vec[b,s,d] · rule_vec[0,d]`,
this shifts every rule-0 logit by `Σ_d site_vec[b,s,d]` — a sum of 128 terms of
O(0.1–1), so a change of order 1–10 against a 1e-3 bound.

| | |
|---|---:|
| comparator self-check, `max|Δ|` on `T` vs `T + 1.0` | **== 1.0000 exactly** |
| perturbed-arm `max|Δ|` on policy logits | **> 1e-3** (the tier-2 bound) |
| perturbed-arm argmax agreement | **recorded, not asserted** |

The argmax figure is kept as an **informational** row precisely because it may
legitimately read 1.0000. If it does, that is a measurement of how wide the
trained margins are, and it is a finding about the model rather than a failure of
the smoke.

## A1.2 — The tier-1 sample must span the widths chunk 9 will actually reach

512 states drawn from Phase-1-flavoured data sit almost entirely at width ≤ 64,
because optimal derivations never apply `add_both_sides` (F-05, ROUND-01). Chunk
9's self-play does apply it, and L = 64 versus L = 300 exercise different kernel
paths. **Equivalence proven on the narrow set is a property proven for a set that
is about to grow** — and the set grows the moment the loop starts.

The tier-1 sample is therefore **stratified by width bucket**:

| bucket | source | minimum states |
|---|---|---:|
| 1–64 | `phase1_eval`, blessed subsampler | 128 |
| 65–128 | random legal walks, depth-5/6 suites | 128 |
| 129–256 | random legal walks | 128 |
| **≥ 257** | random legal walks | **128** |

512 total, ≥ 128 per bucket. The wide buckets come from the same capped random
legal play that produced `runs/state_extent.json`, where `add_both_sides` is
always available.

**Measured ceiling, declared so the span is not overclaimed.** `state_extent.json`
puts reachable tokens at p99 216, **p100 332**. Widths 333–512 are not reachable
under the v1 rule set within the step cap, so the top bucket is ≥257 rather than
"up to 512", and equivalence at literal 512 columns is untestable with real states.
That gap is covered from the other side: `_crop_to_content` is already pinned
bit-exact against full 512-width padding (F-07), so the columns beyond 332 are
proven inert rather than proven equivalent.

**The test asserts its own contrast premise**, as the subsampler test does: the
stratified sample must span all four buckets, **and** a `phase1_eval`-only sample
must fail the span check. If Phase-1 data ever starts producing wide states, that
assertion fails loudly rather than the guarantee quietly weakening.

## A1.3 — The 16 GiB ceiling, restated at L ≤ 512 instead of L ≤ 64

The ceiling's framing survives — a ceiling is a bug detector — but its arithmetic
was computed on the Phase-1 premise. Restated at chunk-9 shapes, `B = 128`,
`L = 512`, `H = 8`, 6 layers, `d_model = 256`, `d_ff = 1024`:

| term | fp32 | bf16 |
|---|---:|---:|
| attention scores, `B·H·L²` per layer × 6 | 6.0 GiB | 3.0 GiB |
| per-token activations saved for backward | ~1.1 GiB | ~0.6 GiB |
| params + grads + AdamW moments (5.07 M) | 0.08 GiB | 0.08 GiB |
| **subtotal** | **~7.2 GiB** | **~3.7 GiB** |
| pessimistic (softmax input *and* output retained) | ~13.2 GiB | ~6.7 GiB |

**16 GiB is kept**, because the pessimistic fp32 accounting still clears it. What
changes is that a reading is now classifiable instead of merely alarming:

| observed peak | reading |
|---|---|
| ≤ 7 GiB | expected under bf16 at full width |
| 7–14 GiB | **premise change, not a bug** — most likely fp32 where bf16 was intended, or a wider batch than declared. Check dtype before checking for leaks |
| > 16 GiB | alarm: exceeds what these shapes can legitimately need |

Rider (c) applied to a resource: the envelope is computed before the alarm is
sited, so the middle band reads as "check the premise" rather than "something
leaked".

---

**Frozen.** Amendments A1.1–A1.3 close the declaration. Execution of 0a follows.
