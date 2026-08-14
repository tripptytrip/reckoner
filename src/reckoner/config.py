"""Single source of truth for all reckoner configuration.

Two properties this module exists to guarantee, both inherited law:

**1. Unknown keys are a hard error.** Silence on a mistyped key is the same
failure family as a wrong git SHA — the run looks fine and the number is wrong.
A near-miss on a real field is *worse* than a typo, because the default it
leaves in place is plausible: ``par_from_pool_frac`` (not a field) sitting next
to ``from_pool_frac`` (a field) would read as a request and change nothing.

**2. Every value is traceable.** Each field below is tagged with where its value
comes from:

  ``[spec §N]``        pinned by ``experiment2_math_base625_spec.md``
  ``[plan chunk N]``   pinned by ``experiment2_agent_plan.md``
  ``[v1.1]``           pinned by the par-game amendment
  ``[provisional]``    a placeholder the named chunk owns and will set

A ``[provisional]`` number is not a decision. It exists so the config tree has a
shape to load, round-trip, and fingerprint from chunk 0 — the chunk that owns it
sets it and pins it with a guard test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class EpisodeConfig:
    """The environment contract: positions, the step cap, and the checker."""

    # [plan chunk 3] "Step cap (config, default 24)". Hitting the cap is a loss
    # (z = -1) under the par game, identically to going over par — the episode
    # does not get to end in a shrug.
    step_cap: int = 24

    # [spec §3] SIMPLIFY is verified by random-assignment equivalence, k=32 draws
    # over a prime field. EVALUATE checks by exact evaluation and SOLVE by
    # substitution; neither needs a draw count.
    simplify_equiv_k: int = 32

    # [plan chunk 2/3] "a prime field" is specified; *which* prime is not, so it
    # is a decision recorded here rather than buried in the checker. 2**31 - 1 is
    # a Mersenne prime and the largest that keeps a product of two residues
    # inside int64 without intermediate overflow ((2**31)**2 = 2**62 < 2**63),
    # which is what makes the equivalence fuzz safe to vectorise in numpy.
    equiv_prime: int = 2_147_483_647


@dataclass
class ParConfig:
    """The par game [v1.1] — the opponent enters the label, never the search."""

    # [spec §3] BFS-exact par is canonical for depth <= 6; scripted-solver par is
    # a provisional *floor* above that, and is provenance-tagged `par_source`.
    bfs_exact_max_depth: int = 6

    # [spec §3, v1.1] The plan spells this key ``par_from_pool_frac``; it is
    # ``par.from_pool_frac`` here to avoid the stutter. Fraction of training
    # problems whose par comes from a pool snapshot's own solution, solved fresh
    # at episode time — so par escalates with the model. This is the built-in
    # half of the funnel treatment; the Phase-3 generator is the other half.
    from_pool_frac: float = 0.20

    # [v1.1] The resign-vs-par analog: concede when the best solution found is
    # already >= par + k steps. "Implemented but default off, calibration
    # deferred to campaign evidence, per the resignation lesson." k is therefore
    # NOT a calibrated number — it is inert while ``concede_enabled`` is False,
    # and calibrating it is a campaign decision, not a default.
    concede_enabled: bool = False
    concede_k: int = 2


@dataclass
class ModelConfig:
    """[plan chunk 6] owns every ``[provisional]`` number here.

    What *is* pinned: the parameter envelope and the head shapes. The value head
    is 3-class W/D/L vs par plus a steps-to-solve auxiliary regression — §8
    decision 1 was reversed by the par-game amendment, and a config that quietly
    sets ``value_classes: 2`` is reverting that decision without saying so, which
    is why ``validate()`` rejects it.
    """

    # [spec §4] "~2-7M parameters". Recorded as config so chunk 6's guard test
    # has a pinned envelope to assert the built model against, rather than a
    # number in prose.
    param_budget_min: int = 2_000_000
    param_budget_max: int = 7_000_000

    # [spec §4, v1.1] Value head = 3-class W/D/L vs par + steps-to-solve aux.
    value_classes: int = 3
    steps_aux_head: bool = True

    # [provisional — chunk 6] Trunk shape. Ported from the v2 transformer family
    # as a starting point; chunk 6 sizes it against the envelope above.
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    # [provisional — chunk 1/6] Sequence length depends on the token spec chunk 1
    # writes (goal prefix tokens + compositional [NUM, d1, d0] numerals).
    seq_len: int = 128
    # [provisional — chunk 6] Bilinear projection dim for the factorized
    # rule x site policy head — the (from, to) factorization transplanted.
    d_policy: int = 128


@dataclass
class SearchConfig:
    """[plan chunk 7] the careful port. Array-tree MCTS + Gumbel + Seq. Halving."""

    # [plan §8 decision 5] THE porting hazard, and it is a config key so that a
    # run cannot silently be doing the wrong backup: "there is no opponent, so
    # backup does NOT negate per ply." ``validate()`` rejects any other value —
    # an alternating-perspective backup in this domain is not a variant, it is a
    # bug, and chunk 7 exists partly to prove the single-agent arithmetic.
    perspective: str = "single"

    # [provisional — chunk 7] Chunk 7's batched-equivalence parametrization is
    # m in {3, 5, 12, 16} x sims in {6, 16, 31, 48} "from day one" (the
    # odd-parity lesson, pre-paid). These defaults are the top of that grid; the
    # depth-1 gate's (sims, m) is set only after chunk 2's measured branching
    # factor makes the gate arithmetic checkable.
    sims: int = 48
    gumbel_m: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    batch_leaves: int = 512

    # [plan chunk 7] eval/self-play profiles kept. The default is the *self-play*
    # value, so anything that forgets to choose keeps generating diverse data
    # rather than silently freezing it. Suites and the ladder run root_noise off.
    root_noise: bool = True


@dataclass
class TrainConfig:
    """Optimizer and loss parameters [plan chunks 8, 9]."""

    # [plan chunk 9] "z/q value blend on by default (`value_q_mse_weight: 0.5`)".
    # Its self-referential hazard is *structurally absent here*: the checker, not
    # the model's own Q, is the source of solved/not-solved. Blending Q into the
    # value target cannot bootstrap a false solve, because solved is decided
    # outside the network. That is the reason this domain suits the method, and
    # it is why the weight is on from day one rather than earned by an ablation.
    value_q_mse_weight: float = 0.5

    # [plan chunk 9] "rehearsal machinery ported (dormant, `rehearsal_frac: 0.0`
    # default — the lever exists before it's needed)".
    rehearsal_frac: float = 0.0

    # [provisional — chunk 8/9] Optimizer. Ported defaults; the timing slice that
    # chunk 8 runs as a required pre-flight is what sets the real batch size.
    batch_size: int = 512
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_steps_per_iter: int = 400
    replay_capacity: int = 500_000

    # [provisional — chunk 6/8] Loss weights. ``steps`` is the auxiliary
    # regression, masked to solved episodes.
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    steps_loss_weight: float = 1.0

    # [plan chunk 8] NaN-skip guard, carried over verbatim: skip the update on a
    # non-finite gradient, count the skips, abort if they become a rate rather
    # than a transient.
    nan_abort_frac: float = 0.01
    nan_abort_min_steps: int = 20


@dataclass
class GeneratorConfig:
    """[plan chunk 5] the procedural problem generator and the frozen instruments."""

    # [plan chunk 5] Difficulty is parameterized by *verified minimum solution
    # depth* — BFS over the rule graph for depths <= 6. The depth label is
    # measured, not assumed, and the suites re-verify it independently.
    max_bfs_depth: int = 6

    # [plan chunk 5] "100K-problem training set + suites + eval sets on disk".
    train_set_size: int = 100_000

    # [plan chunk 5] Frozen instruments: `solve_in_N.jsonl` N = 1...6, 200 each,
    # BFS-verified, contamination-tested against every training set, and never
    # regenerated. Mate-in-N reborn [spec §6].
    suite_problems_per_depth: int = 200
    suite_depths: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])


@dataclass
class LadderConfig:
    """[plan chunk 10] the ladder — external, calibrated, impossible to flatter."""

    # [plan chunk 11] Campaign M1 runs at `--ladder-every 5` over ~20 iterations.
    ladder_every: int = 5

    # [inherited law] Paired-difference bootstrap is the test of record for
    # pass-vs-pass comparisons.
    bootstrap_resamples: int = 10_000

    # [provisional — chunk 10] Paired problem set size per ladder pass (the
    # frozen-openings-book analog) and the sympy rung's budgets. sympy is a
    # *rung*, never par [spec §3] — a CAS derivation that does not compile into
    # our rule vocabulary cannot denominate a step count in our rules.
    problems_per_pass: int = 200
    sympy_step_budget: int = 16
    sympy_time_budget_s: float = 1.0


@dataclass
class Config:
    """Root configuration container."""

    # [inherited law] One seed in config fans out to all RNGs; library code takes
    # explicit rng/Generator parameters, never global random state.
    seed: int = 42
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    par: ParConfig = field(default_factory=ParConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    ladder: LadderConfig = field(default_factory=LadderConfig)


# ---------------------------------------------------------------------------
# Validation — the laws a config file is not allowed to break
# ---------------------------------------------------------------------------


def validate(cfg: Config) -> None:
    """Reject configs that violate a pinned decision.

    Checked at load, not at first use: an overnight campaign that ran the wrong
    backup rule is not recoverable by noticing it in iteration 3's log.
    """
    if cfg.search.perspective != "single":
        raise ValueError(
            f"search.perspective must be 'single'; got {cfg.search.perspective!r}. "
            "This game has no opponent in the tree (plan §8 decision 5): the par "
            "opponent enters the label, never the search, so backup does NOT "
            "negate per ply. An alternating-perspective backup here is a bug, "
            "not a variant."
        )
    if cfg.model.value_classes != 3:
        raise ValueError(
            f"model.value_classes must be 3; got {cfg.model.value_classes}. "
            "The par game (amendment v1.1) reversed §8 decision 1: the value head "
            "is 3-class W/D/L vs par, not 2-class solved/timed-out. Changing it "
            "here would revert a recorded decision silently."
        )
    if cfg.model.param_budget_min > cfg.model.param_budget_max:
        raise ValueError(
            f"model.param_budget_min ({cfg.model.param_budget_min}) exceeds "
            f"param_budget_max ({cfg.model.param_budget_max})."
        )
    if not 0.0 <= cfg.par.from_pool_frac <= 1.0:
        raise ValueError(f"par.from_pool_frac must be in [0, 1]; got {cfg.par.from_pool_frac}")
    if not 0.0 <= cfg.train.rehearsal_frac < 1.0:
        raise ValueError(f"train.rehearsal_frac must be in [0, 1); got {cfg.train.rehearsal_frac}")
    if cfg.episode.step_cap < 1:
        raise ValueError(f"episode.step_cap must be >= 1; got {cfg.episode.step_cap}")
    if cfg.episode.simplify_equiv_k < 1:
        raise ValueError(
            f"episode.simplify_equiv_k must be >= 1; got {cfg.episode.simplify_equiv_k}"
        )

    # A suite depth beyond the BFS horizon cannot have a verified depth label or
    # a canonical par — both are BFS-exact by definition [plan chunk 5, v1.1] —
    # so the instrument would be frozen around an unverified number.
    over = [d for d in cfg.generator.suite_depths if d > cfg.generator.max_bfs_depth]
    if over:
        raise ValueError(
            f"generator.suite_depths {over} exceed max_bfs_depth "
            f"({cfg.generator.max_bfs_depth}). Suite depth labels and par are "
            "BFS-verified; beyond the BFS horizon neither can be canonical."
        )
    if any(d < 1 for d in cfg.generator.suite_depths):
        raise ValueError(
            f"generator.suite_depths must all be >= 1; got {cfg.generator.suite_depths}"
        )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts (handles nesting)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _from_dict(raw: dict, cls: type) -> Any:
    """Rebuild a dataclass tree from nested plain dicts.

    ``typing.get_type_hints`` is used rather than ``field.type`` because
    ``from __future__ import annotations`` makes every annotation a *string* —
    with the raw strings, no nested dataclass is ever recognised, every nested
    section silently falls through as a dict, and the config that loads is not
    the config that was written.
    """
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        nested = hints.get(f.name)
        if isinstance(value, dict) and dataclasses.is_dataclass(nested):
            value = _from_dict(value, nested)
        kwargs[f.name] = value
    return cls(**kwargs)


def _assert_known_keys(raw: dict, cls: type, source: str, prefix: str = "") -> None:
    """Reject config keys that do not exist on the dataclass tree."""
    valid = {f.name for f in fields(cls)}
    hints = typing.get_type_hints(cls)
    for key, value in raw.items():
        dotted = f"{prefix}{key}"
        if key not in valid:
            near = sorted(n for n in valid if n.startswith(key[:6]) or key.startswith(n[:6]))
            hint = f" Did you mean {near[0]!r}?" if near else ""
            raise ValueError(
                f"{source}: unknown config key {dotted!r} for {cls.__name__}.{hint} "
                f"Valid: {sorted(valid)}"
            )
        nested = hints.get(key)
        if isinstance(value, dict) and dataclasses.is_dataclass(nested):
            _assert_known_keys(value, nested, source, prefix=f"{dotted}.")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load a Config from YAML, optionally applying dot-notation overrides.

    ``path=None`` returns the defaults. ``overrides`` keys are dotted, e.g.
    ``{"search.sims": 16}``, and are checked against the tree exactly as file
    keys are — a CLI flag mistyped on the command line is the same failure as a
    YAML key mistyped in a file.
    """
    if path is not None:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: config root must be a mapping, got {type(raw).__name__}")
        _assert_known_keys(raw, Config, str(path))
        cfg = _from_dict(raw, Config)
    else:
        cfg = Config()

    if overrides:
        cfg_dict = _dataclass_to_dict(cfg)
        for key, value in overrides.items():
            parts = key.split(".")
            target = cfg_dict
            for i, part in enumerate(parts[:-1]):
                if part not in target or not isinstance(target[part], dict):
                    raise ValueError(
                        f"override {key!r}: {'.'.join(parts[: i + 1])!r} is not a config section"
                    )
                target = target[part]
            if parts[-1] not in target:
                raise ValueError(f"override {key!r}: unknown config key {parts[-1]!r}")
            target[parts[-1]] = value
        _assert_known_keys(cfg_dict, Config, "overrides")
        cfg = _from_dict(cfg_dict, Config)

    validate(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Serialise a Config to YAML. Round-trips exactly with load_config."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(_dataclass_to_dict(cfg), fh, default_flow_style=False, sort_keys=True)


def config_fingerprint(cfg: Config) -> str:
    """SHA-256 of the canonical (sorted-key) YAML representation.

    [inherited law] Goes into every checkpoint and dataset ``meta.json`` beside
    the git SHA, so a run's rows can always be read against the config that
    produced them.
    """
    canonical = yaml.safe_dump(_dataclass_to_dict(cfg), default_flow_style=False, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
