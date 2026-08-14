"""Phase 1: supervised warm start.

Targets, per the chunk-8 brief:

* **policy** — the action the BFS-optimal derivation took, one-hot over the legal
  set, masked by ``legal_actions``.
* **steps** — remaining steps to the solve, regressed, masked to solved rows
  (every Phase-1 row is on a solved derivation, so the mask is all-true here and
  is kept anyway: the machinery must be exercised before Phase 2 relies on it).
* **W/D/L** — **loss weight 0, deliberately.** Spec §5 trains Phase-1 value on
  steps-to-solve. Imitation data has degenerate z: every row is on an optimal
  derivation, so every episode would be a draw, and training the head on that
  manufactures a confidently-wrong prior rather than an ignorant one.
  Untrained-but-uninformative beats miswired. Phase 2 wakes it.

The single-BFS-path caveat is recorded in the dataset's own meta: where several
optimal derivations exist, the network is taught the one BFS reached first.
Phase 2's improved-policy targets are visit distributions over the whole legal
set and wash it out.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from reckoner.config import Config
from reckoner.episode import Problem, decode_state
from reckoner.model import N_RULES, Reckoner, StateTooLarge, encode, policy_loss, steps_loss
from reckoner.vocab import PAD


@dataclass
class Batch:
    tokens: torch.Tensor
    site_positions: torch.Tensor
    legal_mask: torch.Tensor
    policy_target: torch.Tensor
    steps_target: torch.Tensor
    solved_mask: torch.Tensor
    depth: torch.Tensor
    skipped: int = 0
    width: int = 0  # columns after the crop; `tokens.shape[1]`, recorded not inferred


@dataclass
class TrainStats:
    steps: int = 0
    examples: int = 0
    nan_skips: int = 0
    encode_skips: int = 0
    losses: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "examples": self.examples,
            "nan_skips": self.nan_skips,
            "encode_skips": self.encode_skips,
            "final_loss": self.losses[-1] if self.losses else None,
        }


class SupervisionSet:
    """Mapped Phase-1 examples. Host RAM is the scarce pool; nothing is loaded."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text())
        n, width = self.meta["n"], self.meta["max_len"]
        load = lambda name, shape: np.memmap(  # noqa: E731
            path / f"{name}.i32", dtype=np.int32, mode="r", shape=shape
        )
        self.tokens = load("tokens", (n, width))
        self.lengths = load("lengths", (n,))
        self.action = load("action", (n,))
        self.steps_remaining = load("steps_remaining", (n,))
        self.depth = load("depth", (n,))
        self.goal = load("goal", (n,))

    def __len__(self) -> int:
        return int(self.meta["n"])


def _crop_to_content(tokens: torch.Tensor, site_positions: torch.Tensor) -> torch.Tensor:
    """Drop trailing all-PAD columns. **Exact, not approximate** — and measured.

    ``encode`` pads every state to ``model.seq_len`` (512), sized in chunk 6 from
    the *reachable*-state distribution, because Phase 2 self-play can grow states
    past 295 tokens (F-05). Phase 1 supervision states are all on BFS-optimal
    derivations, and optimal derivations never apply ``add_both_sides``
    (ROUND-01), so the whole 313,628-example set has ``max_len 64``. Eight times
    the width, and attention is O(L²).

    Cropping changes nothing the network computes: the trunk already ignores PAD
    via ``src_key_padding_mask``, ``_masked_mean`` pools over ``ne(PAD)``, and
    positions are indexed from 0 by ``arange``, so rows 0..width-1 of the
    position table are the same rows either way. That equality is pinned as an
    exact tensor identity by ``test_cropping_padding_is_exact``, both polarities:
    the crop must fire *and* the outputs must be bit-identical.

    **Do not "fix" this by lowering ``model.seq_len``.** The envelope is sized
    for Phase 2, where the states really are that long; the width is a property
    of a batch, not of the model.
    """
    filled = tokens.ne(PAD).nonzero()
    # An all-PAD batch cannot occur (a state has at least a goal token), but a
    # width of 0 would be silently catastrophic, so the floor is stated.
    width = int(filled[:, 1].max()) + 1 if filled.numel() else 1
    width = max(width, int(site_positions.max()) + 1)
    return tokens[:, :width]


def make_batch(data: SupervisionSet, indices: list[int], cfg: Config) -> Batch:
    """Encode a batch. Site positions and legality are derived, never stored.

    Storing them would cost ~660 MB for a 313K set at `max_sites=192`; deriving
    them costs a decode plus one `legal_actions` call per row, and keeps the
    single legality oracle intact — a stored mask is a second copy that can
    disagree with the engine.
    """
    tokens, positions, masks, targets, steps, depths = [], [], [], [], [], []
    skipped = 0
    for i in indices:
        seq = tuple(int(t) for t in data.tokens[i, : data.lengths[i]])
        goal, target, expr = decode_state(seq)
        problem = Problem(
            goal=goal, expr=expr, par=int(data.depth[i]), target=target, par_source="bfs"
        )
        try:
            state = encode(problem, expr, cfg)
        except StateTooLarge:
            skipped += 1  # counted, never cropped
            continue
        one_hot = torch.zeros(N_RULES * cfg.model.max_sites)
        one_hot[int(data.action[i])] = 1.0
        tokens.append(state.tokens)
        positions.append(state.site_positions)
        masks.append(state.legal_mask)
        targets.append(one_hot)
        steps.append(float(data.steps_remaining[i]))
        depths.append(int(data.depth[i]))

    if not tokens:
        raise ValueError("every row in the batch failed to encode")
    steps_tensor = torch.tensor(steps)
    stacked = _crop_to_content(torch.stack(tokens), torch.stack(positions))
    return Batch(
        tokens=stacked,
        site_positions=torch.stack(positions),
        width=int(stacked.shape[1]),
        legal_mask=torch.stack(masks),
        policy_target=torch.stack(targets),
        steps_target=steps_tensor,
        solved_mask=torch.ones_like(steps_tensor, dtype=torch.bool),
        depth=torch.tensor(depths),
        skipped=skipped,
    )


def train(
    model: Reckoner,
    data: SupervisionSet,
    cfg: Config,
    *,
    steps: int,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 50,
    on_log: Callable[[int, float], None] | None = None,
) -> TrainStats:
    """Run the warm start. NaN-skip guard with a 1% abort, per the inherited kit.

    ``on_log(step, mean_loss_over_window)`` is called every ``log_every`` steps.
    Library code does not print (AGENTS.md §6) — the caller in ``scripts/`` owns
    the terminal, and a training loop that printed would also be unusable from a
    dashboard or a test.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    stats = TrainStats()

    for step in range(steps):
        indices = [rng.randrange(len(data)) for _ in range(cfg.train.batch_size)]
        batch = make_batch(data, indices, cfg)
        stats.encode_skips += batch.skipped

        policy, _value, steps_out = model(batch.tokens.to(device), batch.site_positions.to(device))
        loss = cfg.train.policy_loss_weight * policy_loss(
            policy, batch.policy_target.to(device), batch.legal_mask.to(device)
        )
        assert steps_out is not None
        loss = loss + cfg.train.steps_loss_weight * steps_loss(
            steps_out, batch.steps_target.to(device), batch.solved_mask.to(device)
        )
        # W/D/L head: loss weight 0 by design (see module docstring). Its
        # parameters simply do not appear in this graph.

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not finite:
            stats.nan_skips += 1
            optimiser.zero_grad(set_to_none=True)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimiser.step()

        stats.steps += 1
        stats.examples += int(batch.tokens.shape[0])
        stats.losses.append(float(loss.detach()))

        if stats.steps >= cfg.train.nan_abort_min_steps:
            rate = stats.nan_skips / stats.steps
            if rate > cfg.train.nan_abort_frac:
                raise RuntimeError(
                    f"non-finite gradients on {rate:.1%} of steps "
                    f"({stats.nan_skips}/{stats.steps}) — a rate, not a transient"
                )
        if on_log is not None and log_every and (step + 1) % log_every == 0:
            window = stats.losses[-log_every:]
            on_log(step + 1, sum(window) / len(window))

    return stats
