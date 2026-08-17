"""The model as a rung, and the one evaluator that decides what the search trusts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from reckoner.arms import ArmError
from reckoner.config import Config
from reckoner.dataset import read_suite, suite_problem
from reckoner.evaluate import EvaluatorModeError, ModelArm, model_evaluator
from reckoner.model import Reckoner
from reckoner.valuegate import ValueHeadState, value_contribution

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "runs" / "suites"
CFG = Config()

needs_suites = pytest.mark.skipif(
    not (SUITES / "solve_in_1.jsonl").exists(), reason="suites not generated"
)


def a_model() -> Reckoner:
    torch.manual_seed(0)
    return Reckoner(CFG).eval()


def problems(suite: str = "solve_in_1", n: int = 4):
    return [suite_problem(r) for r in read_suite(SUITES / f"{suite}.jsonl")[:n]]


def eval_cfg() -> Config:
    return replace(CFG, search=replace(CFG.search, root_noise=False, sims=6, gumbel_m=3))


# ---------------------------------------------------------------------------
# The evaluator's one job: apply the value declaration, never invent it
# ---------------------------------------------------------------------------


@needs_suites
def test_a_zero_value_scale_silences_the_value_and_leaves_priors_alone() -> None:
    """F-14's closure, at the only place it is applied.

    The search must read no opinion the head has not earned — and the policy
    prior must still arrive, because it is trained on real targets from the first
    iteration. Silencing both would be a different (and wrong) declaration.
    """
    model = a_model()
    problem = problems()[0]
    silent = model_evaluator(model, CFG, 0.0)([(problem, problem.expr)])
    trusting = model_evaluator(model, CFG, 1.0)([(problem, problem.expr)])
    assert silent[0][1] == 0.0
    assert (silent[0][0] == trusting[0][0]).all(), "value_scale must not touch the prior"


@needs_suites
def test_the_value_scale_comes_from_the_gate_not_from_a_literal() -> None:
    """One declaration, two consumers: the number the search uses is the number
    `valuegate` produces, not a constant that happens to match it today."""
    assert value_contribution(ValueHeadState()) == 0.0
    model = a_model()
    problem = problems()[0]
    out = model_evaluator(model, CFG, value_contribution(ValueHeadState()))(
        [(problem, problem.expr)]
    )
    assert out[0][1] == 0.0


# ---------------------------------------------------------------------------
# The rung, and its construction gate
# ---------------------------------------------------------------------------


@needs_suites
def test_the_model_arm_plays_and_scores_in_our_currency() -> None:
    arm = ModelArm(a_model(), sims=6, m=3)
    result = arm.play(problems()[0], eval_cfg(), seed=0)
    assert result.currency == "z_vs_par"
    assert result.steps >= 0


@needs_suites
def test_the_eval_profile_is_deterministic_and_the_probe_proves_it() -> None:
    """With root noise off the search's rng must not reach the outcome."""
    arm = ModelArm(a_model(), sims=6, m=3)
    arm.probe(problems()[0], eval_cfg())
    a = arm.play(problems()[0], eval_cfg(), seed=0)
    b = arm.play(problems()[0], eval_cfg(), seed=12345)
    assert (a.solved, a.steps) == (b.solved, b.steps)


@needs_suites
def test_probing_determinism_under_root_noise_is_refused() -> None:
    """The rejecting case, and it is the branching-premise lesson again: under
    self-play, varying IS correct, so this probe would fail on a correct arm."""
    arm = ModelArm(a_model(), sims=6, m=3, root_noise=True)
    with pytest.raises(ArmError, match="root_noise on"):
        arm.probe(problems()[0], eval_cfg())


@needs_suites
def test_the_arm_sets_its_profile_rather_than_inheriting_the_self_play_default() -> None:
    """The config default is `root_noise=True` on purpose. A measuring rung that
    silently took it would report a distribution the eval passes never see."""
    assert CFG.search.root_noise is True, "if this default changes, the guard below moves"
    arm = ModelArm(a_model())
    assert arm.profile_config(CFG).search.root_noise is False
    assert CFG.search.root_noise is True, "profile_config must not mutate the caller's config"


@needs_suites
def test_a_self_play_arm_keeps_the_profile_it_declares() -> None:
    """Both polarities: the override is toward what the ARM declares, not toward
    a hard-coded False."""
    arm = ModelArm(a_model(), root_noise=True)
    assert arm.profile_config(CFG).search.root_noise is True


# ------------------------------------------------- the evaluator's mode (F-22)


def test_the_evaluator_refuses_a_model_in_train_mode() -> None:
    """F-22, first polarity. **The mode is a property of the evaluator**, so it
    is checked where the evaluator is built rather than trusted to an
    ``.eval()`` somewhere up the call chain — "we call eval somewhere" is
    exactly the claim that rots, and it rotted: `train_on_ring` set train mode
    and never restored it, so every subsequent search ran with dropout live.
    """
    model = Reckoner(CFG)
    model.train()
    with pytest.raises(EvaluatorModeError, match="TRAIN mode"):
        model_evaluator(model, CFG, 0.0)


def test_the_evaluator_accepts_a_model_in_eval_mode() -> None:
    """Second polarity — without it the guard could be an unconditional raise."""
    model = Reckoner(CFG)
    model.eval()
    assert model_evaluator(model, CFG, 0.0) is not None


def test_dropout_is_live_in_train_mode_so_the_guard_has_something_to_guard() -> None:
    """The guard is only worth its line if the two modes actually differ.

    With ``dropout > 0`` a train-mode forward pass is stochastic, which is what
    made the driver's headline measurement swing in the third decimal place
    across identical processes. Asserted rather than assumed, because a config
    that drifted to ``dropout=0`` would leave the two tests above passing while
    guarding nothing.
    """
    cfg = replace(CFG, model=replace(CFG.model, dropout=0.1))
    assert cfg.model.dropout > 0.0
    model = Reckoner(cfg)
    tokens = torch.ones((1, cfg.model.seq_len), dtype=torch.long)
    sites = torch.zeros((1, cfg.model.max_sites), dtype=torch.long)

    model.train()
    with torch.no_grad():
        a, b = model(tokens, sites)[0], model(tokens, sites)[0]
    assert not torch.equal(a, b), "dropout is not live in train mode; the guard guards nothing"

    model.eval()
    with torch.no_grad():
        c, d = model(tokens, sites)[0], model(tokens, sites)[0]
    assert torch.equal(c, d), "eval mode is not deterministic"
