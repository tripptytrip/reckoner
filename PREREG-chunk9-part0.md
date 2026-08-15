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
