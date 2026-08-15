# Architecture card — `Reckoner`, chunk 6 trunk

Generated facts, reconciled against `Reckoner.parameter_breakdown()`. This card
exists because the VRAM envelope's retention constant and the parameter table
were being quoted from two mental models. They are one model, and this is the
derivation both now trace to.

## Shape

| | |
|---|---:|
| `d_model` | 256 |
| `n_heads` | 8 (head dim 32) |
| `d_ff` | **1024** |
| `n_layers` | 6 |
| `dropout` | 0.1 |
| norm placement | `norm_first` (pre-norm) |
| activation | gelu |
| `seq_len` (envelope) | 512 |
| `max_sites` | 192 |
| `d_policy` | 128 |
| vocabulary | 657 symbols |

The trunk is a **standard** `nn.TransformerEncoder`. Nothing is shared,
factorised, or reshaped — the only non-stock parts are the heads.

## Per-layer parameter derivation

| term | expression | count |
|---|---|---:|
| attention `in_proj` | `3d² + 3d` | 197,376 |
| attention `out_proj` | `d² + d` | 65,792 |
| `linear1` | `d_ff·d + d_ff` | 263,168 |
| `linear2` | `d·d_ff + d` | 262,400 |
| `norm1`, `norm2` | `2 · 2d` | 1,024 |
| **per layer** | `4d² + 2·d·d_ff + biases` | **789,760** |

`789,760 × 6 = 4,738,560`, plus the trunk's own final `LayerNorm` (`2d = 512`)
= **4,739,072**, which is `parameter_breakdown()["trunk"]` exactly.

## Full breakdown

| component | count | |
|---|---:|---|
| embeddings | 299,264 | token (657 × 256) + position (512 × 256) — **verified against `token_embedding.weight.shape` and `position_embedding.weight.shape`**, not factored out of the total |
| trunk | 4,739,072 | 6 layers + final norm |
| policy head | 33,792 | `rule_embedding` (7 × 128) + `site_projection` (256 → 128) |
| value head | 771 | 256 → 3 (W/D/L vs par) |
| steps head | 257 | 256 → 1 |
| **total** | **5,073,156** | inside the spec's 2–7 M envelope |

## Why this card exists: a factorisation is not a verification

299,264 factors as 657 × 256 + 512 × 256 — and also as 668 × 448, and as
1,169 × 256, and others. **Multiple factorisations always fit one number**, so
picking one and calling it verified is the same defect as F-10's floor-versus-
chance: correct arithmetic against the wrong reference. A decomposition is
verified only when its *factors* are independently confirmed — here, against
`VOCAB_SIZE = 657`, `seq_len = 512`, and the tensor shapes themselves.

That is the fourth instance of the pattern this project keeps meeting:
**summaries drift; derivations don't.** A card is the confirmation channel, and
the numbers on it are read off the built model rather than reconstructed from a
total.

## The activation retention constant

The VRAM envelope (`PREREG-chunk9-part0.md` A4) uses **3,584 floats per token per
layer**:

```
6·d_model + 2·d_ff  =  6·256 + 2·1024  =  1536 + 2048  =  3584
```

`d_ff` is **1024**. A reading of `6·448` would imply `d_ff = 448`, which is not
this model — that decomposition does not correspond to any term above. The `6·d_model`
term counts the residual-stream tensors a pre-norm layer retains for backward
(layer input, two norm outputs, attention output, and the two residual adds); the
`2·d_ff` term counts `linear1`'s output and its post-activation. It is an
engineering estimate of *retention*, deliberately conservative, and it is not
derived from the parameter count — but it is now stated in the same units, from
the same shape table, on the same page.

**Why the envelope's conclusion is robust to this constant.** At `L = 512` the
attention-score term is `B·H·L²·layers` = 6.00 GiB fp32 against 5.25 GiB for
retention: scores dominate and grow as `L²` while retention grows as `L`. Halving
or doubling the retention constant moves the subtotal by ±2.6 GiB and does not
change which side of the 16 GiB ceiling any *reachable* width falls on. The band
edges inherit it; the verdict does not.
