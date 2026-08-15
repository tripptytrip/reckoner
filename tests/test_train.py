"""Phase-1 warm start: the batch pipeline and its one optimisation.

The first gate here measures the component doing its central job (AGENTS.md §5,
the F-06 rider): the crop must *fire*, and it must change nothing. A crop test
that only checked the outputs would go green on a crop that never cropped —
which is F-06's shape exactly, one layer down.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from reckoner.config import Config
from reckoner.model import Reckoner
from reckoner.train import (
    SupervisionSet,
    _crop_to_content,
    make_batch,
    rehearsal_split,
    train,
)
from reckoner.vocab import PAD

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "runs" / "data" / "phase1_train"
CFG = Config()

needs_data = pytest.mark.skipif(
    not (DATA / "meta.json").exists(), reason="phase-1 supervision not built"
)


# ---------------------------------------------------------------------------
# The crop: exact, and it must actually crop
# ---------------------------------------------------------------------------


@needs_data
def test_cropping_padding_is_exact() -> None:
    """Bit-identical outputs, and the crop demonstrably fired.

    ``encode`` pads to ``model.seq_len`` (512, sized for Phase-2 reachable
    states); Phase-1 supervision states top out at ``max_len 64``. Cropping the
    all-PAD tail is free because the trunk masks PAD, the pooling masks PAD, and
    positions are indexed from zero. "Free" is asserted here, not argued.
    """
    data = SupervisionSet(DATA)
    rng = random.Random(0)
    batch = make_batch(data, [rng.randrange(len(data)) for _ in range(16)], CFG)

    # Polarity one: the crop fired. Without this the equality below is vacuous.
    assert batch.width < CFG.model.seq_len, (
        f"crop did not fire: width {batch.width} == seq_len {CFG.model.seq_len}"
    )
    assert batch.tokens.shape[1] == batch.width

    padded = torch.full((batch.tokens.shape[0], CFG.model.seq_len), PAD, dtype=batch.tokens.dtype)
    padded[:, : batch.width] = batch.tokens

    model = Reckoner(CFG).eval()  # eval: dropout off, so this compares the maths
    with torch.no_grad():
        cropped_out = model(batch.tokens, batch.site_positions)
        full_out = model(padded, batch.site_positions)

    names = ("policy", "value", "steps")
    for name, cropped, full in zip(names, cropped_out, full_out, strict=True):
        assert torch.equal(cropped, full), f"{name} differs after cropping padding"


def test_crop_keeps_every_column_that_holds_content() -> None:
    """The other polarity: a batch that needs the full width keeps it."""
    tokens = torch.full((2, CFG.model.seq_len), PAD, dtype=torch.long)
    tokens[0, 0] = 5
    tokens[1, CFG.model.seq_len - 1] = 5  # content in the very last column
    sites = torch.zeros((2, CFG.model.max_sites), dtype=torch.long)
    assert _crop_to_content(tokens, sites).shape[1] == CFG.model.seq_len


def test_crop_never_cuts_below_a_site_position() -> None:
    """A site indexes into the trunk output; cropping past one would gather PAD.

    Sites cannot outrun their own tokens in practice, so this guards the
    invariant rather than an observed failure — which is the point of writing it
    while the invariant is cheap to state.
    """
    tokens = torch.full((1, CFG.model.seq_len), PAD, dtype=torch.long)
    tokens[0, :4] = 5
    sites = torch.zeros((1, CFG.model.max_sites), dtype=torch.long)
    sites[0, 0] = 300
    assert _crop_to_content(tokens, sites).shape[1] == 301


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


@needs_data
def test_batch_targets_are_legal_actions() -> None:
    """A one-hot policy target under a masked-off action trains toward nothing.

    ``policy_loss`` scores only legal actions, so a target on an illegal one is
    silently dropped rather than loudly rejected — the mask hides the defect it
    is meant to expose. This asserts the two agree on the data as built.
    """
    data = SupervisionSet(DATA)
    rng = random.Random(1)
    batch = make_batch(data, [rng.randrange(len(data)) for _ in range(64)], CFG)
    chosen = batch.policy_target.argmax(dim=1)
    legal_at_target = batch.legal_mask.gather(1, chosen.unsqueeze(1)).squeeze(1)
    assert bool(legal_at_target.all()), (
        f"{int((~legal_at_target).sum())} of {len(chosen)} policy targets are illegal actions"
    )


@needs_data
def test_encode_failures_are_counted_never_cropped() -> None:
    """``StateTooLarge`` rows are skipped and counted, per chunk 6's semantics."""
    data = SupervisionSet(DATA)
    rng = random.Random(2)
    indices = [rng.randrange(len(data)) for _ in range(32)]
    batch = make_batch(data, indices, CFG)
    assert batch.skipped == len(indices) - int(batch.tokens.shape[0])


# ---------------------------------------------------------------------------
# Gate 13 — reproducible from seed, at a declared tolerance
# ---------------------------------------------------------------------------


@needs_data
def test_training_is_reproducible_from_seed() -> None:
    """**Declared tolerance: exact.** Not "close" — bit-identical loss curves.

    On CPU with every RNG seeded from config there is no source of nondeterminism
    to tolerate, so a tolerance would be a place for real drift to hide. If this
    ever needs one, that is a finding about the environment and gets written down
    rather than absorbed into an epsilon.

    Scope is declared too, because a gate must report what it covered: **5 steps
    at batch 16**, which exercises sampling, batch construction, the crop, both
    losses, clipping and the optimiser step. It does not cover long-run
    accumulation; the full run's reproducibility rests on this plus the seeded
    sampler, and a 2 x 83-minute repeat is not a `make test` cost.
    """
    data = SupervisionSet(DATA)
    cfg = Config()
    cfg.train.batch_size = 16

    def run() -> list[float]:
        torch.manual_seed(0)
        model = Reckoner(cfg)
        stats = train(model, data, cfg, steps=5, seed=0, log_every=0)
        return stats.losses

    first, second = run(), run()
    assert first == second, f"seed-identical runs diverged:\n  {first}\n  {second}"
    assert len(first) == 5
    # Both polarities: a different seed must NOT reproduce it, or the assertion
    # above would pass on a loop that ignores its inputs.
    torch.manual_seed(0)
    other = train(Reckoner(cfg), data, cfg, steps=5, seed=1, log_every=0).losses
    assert other != first, "a different seed produced an identical curve"


# ---------------------------------------------------------------------------
# Rehearsal — ported dormant, and "dormant" is asserted rather than assumed
# ---------------------------------------------------------------------------


def test_rehearsal_is_dormant_at_the_default() -> None:
    """`rehearsal_frac: 0.0` must change nothing in the training path.

    "Dormant" untested means "untested" — the accepting case is that the split
    is a no-op, and it has to be asserted like any other.
    """
    assert CFG.train.rehearsal_frac == 0.0
    assert rehearsal_split(128, CFG) == (0, 128)


def test_rehearsal_splits_when_armed() -> None:
    """The other polarity: the lever must do something when pulled, or it is not
    a lever that exists before it is needed — it is a key that does nothing."""
    cfg = Config()
    cfg.train.rehearsal_frac = 0.25
    assert rehearsal_split(128, cfg) == (32, 96)
    cfg.train.rehearsal_frac = 0.5
    assert rehearsal_split(10, cfg) == (5, 5)


def test_a_full_rehearsal_fraction_is_refused() -> None:
    """At 1.0 a Phase-2 iteration trains on no Phase-2 data, which is not rehearsal."""
    cfg = Config()
    cfg.train.rehearsal_frac = 1.0
    with pytest.raises(ValueError, match="not rehearsal"):
        rehearsal_split(128, cfg)
