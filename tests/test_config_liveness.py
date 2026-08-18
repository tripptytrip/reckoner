"""**The live-config gate.** Every fingerprinted field either changes what runs,
or names why it does not — with evidence.

Chunk 8 declared this class closed: *"a dead config key is the `batch_leaves`
hazard and is not repeated."* At that moment `value_q_mse_weight` was dead at the
loop's core and `rehearsal_frac` was dead too (F-31). A named class, declared
closed, recurring twice underneath the declaration — which is why the remedy is
mechanical rather than another resolution.

**The rule, in its adopted form:** the question is not whether a field is *read*,
but whether **varying it across its legal range changes what runs.**

That formulation replaced an earlier two-tier version, because the two-tier
version passes `rehearsal_frac`: it was read by `validate()` and copied into a
results JSON, and neither is acting on it. It also makes "observable only at
campaign scale" unsayable without a demonstration.

The worked pair, because it is the whole distinction:

* `search.perspective` — **exempt.** Legal range is exactly one value, enforced
  by `validate`. Varying it within legal values is impossible: *vacuously
  behavioural.*
* `train.rehearsal_frac` — **failed** (until this round wired it). Legal range
  `[0, 1)`, every value passes validation, and none of them changed anything:
  *vacuously inert.*

The registry has now pruned itself twice under real pressure: `rehearsal_frac`
left it when fix 1 wired the lever, and `league.snapshot_every` left it when
F-36's fix made the driver honour the cadence. Both times the staleness test
caught it rather than a reader noticing — which is the point of having the
second polarity at all.
"""

from __future__ import annotations

import ast
from dataclasses import replace

import pytest

from reckoner.config import Config, validate
from scripts.config_census import Walker, build, config_fields, reachable

#: Fields that do not change what runs, each with the reason and the evidence.
#: An entry is a claim, and the test below checks the claims it can check —
#: `search.perspective`'s refusal is executed, not asserted.
EXEMPT: dict[str, str] = {
    # --- vacuously behavioural: the legal range is a single value -------------
    "search.perspective": (
        "pinned invariant — validate() admits only 'single', so there is no legal "
        "variation to observe. Evidence: test_a_pinned_invariant_refuses_variation."
    ),
    # --- spec-backed, blocked on work registered elsewhere --------------------
    "ladder.problems_per_pass": (
        "F-34 — the rung pass is dropped from M1: both candidate populations are "
        "measured-saturated (smoke_v1 headroom is 12% of one CI half-width; the "
        "suites are F-20). Carried to M2, where a population gets minted with "
        "headroom demonstrated before freezing."
    ),
    "ladder.bootstrap_resamples": (
        "F-30 — the paired bootstrap needs per-problem outcomes on BOTH arms. The "
        "campaign arm now has them (F-33 routing); the baseline arm does not until "
        "Part-0d is re-run deterministically. Dead until that lands, then live."
    ),
    # --- unbacked: no page asked for these, so the disposition is -------------
    # --- delete-or-amend at M1-A4, never wire-by-default ---------------------
    "generator.train_set_size": "unbacked; CLI-shadowed in generate.py (F-33). Delete or amend at M1-A4.",
    "generator.suite_depths": "unbacked; CLI-shadowed in generate.py (F-33). Delete or amend at M1-A4.",
    "generator.suite_problems_per_depth": "unbacked; CLI-shadowed (F-33). Delete or amend at M1-A4.",
    "generator.max_bfs_depth": "unbacked; the datasets are built and frozen. Delete or amend at M1-A4.",
    "model.param_budget_min": "unbacked; a sizing guide, never consulted at runtime. Delete or amend at M1-A4.",
    "model.param_budget_max": "unbacked; a sizing guide, never consulted at runtime. Delete or amend at M1-A4.",
    "ladder.sympy_step_budget": "unbacked; the sympy arm carries its own budget. Delete or amend at M1-A4.",
    "ladder.sympy_time_budget_s": "unbacked; the sympy arm carries its own budget. Delete or amend at M1-A4.",
}


def _census() -> dict[str, str]:
    """``{qualified field: status}`` — the census, as the gate consumes it."""
    fields = config_fields()
    reads, script_reads, calls, defined = build()
    live = reachable(calls, defined)
    out = {}
    for field, group in fields.items():
        scopes = reads.get(field, set())
        if scopes & live:
            status = "LIVE"
        elif scopes:
            status = "DEAD"
        elif script_reads.get(field):
            status = "REPORTED"
        else:
            status = "UNREAD"
        out[f"{group}.{field}"] = status
    return out


def test_every_fingerprinted_field_is_live_or_registered() -> None:
    """The gate. A new dead key fails at introduction, which is the only version
    of "not repeated" that is true going forward rather than retroactively."""
    census = _census()
    unexplained = sorted(f for f, s in census.items() if s != "LIVE" and f not in EXEMPT)
    assert not unexplained, (
        "fingerprinted fields that change nothing and name no reason: "
        f"{unexplained}. Either wire the field, or register it in EXEMPT with the "
        "evidence — a config key that moves the fingerprint and moves nothing else "
        "is the class chunk 8 declared closed while two of them were live."
    )


def test_the_registry_does_not_outlive_its_entries() -> None:
    """The other polarity. An exemption for a field that has since been wired is
    a stale claim, and a registry nobody prunes becomes a place to hide things.

    This fired for real: the census's reference vector was `rehearsal_frac`, and
    wiring it in this round made the assertion fail on its own success."""
    census = _census()
    stale = sorted(f for f in EXEMPT if census.get(f) == "LIVE")
    assert not stale, f"registered as inert, but now live: {stale}. Remove the entry."


def test_a_pinned_invariant_refuses_variation() -> None:
    """`search.perspective`'s exemption, executed rather than asserted.

    Its legal range is one value, so it cannot vary within legality — which is
    what makes it vacuously behavioural rather than inert. The proof is that the
    only other value is refused."""
    cfg = Config()
    assert cfg.search.perspective == "single"
    with pytest.raises(ValueError, match="perspective"):
        validate(replace(cfg, search=replace(cfg.search, perspective="alternating")))


def test_the_detector_flags_a_synthetic_dead_key() -> None:
    """**The reference vector, synthetic on purpose.**

    This check used to run inside the census against `train.rehearsal_frac`, the
    field F-31 proved dead by hand. Then this round wired that field and the
    assertion fired on its own success: a detector validated against a live repo
    field is validated against a moving target.

    So the known positive is constructed here instead — a field read only inside
    a function nothing calls, which is exactly `rehearsal_split`'s shape and
    exactly what reference-counting would miss.
    """
    source = """
def unreachable_helper(cfg):
    return cfg.train.rehearsal_frac

def reachable_helper(cfg):
    return cfg.train.lr
"""
    walker = Walker(config_fields())
    walker.visit(ast.parse(source))
    assert walker.reads["rehearsal_frac"] == {"unreachable_helper"}
    assert walker.reads["lr"] == {"reachable_helper"}

    # `unreachable_helper` is called by nothing, so a reachability pass from any
    # entry point excludes it — while a grep for the field name would find it.
    live = reachable({}, {"unreachable_helper", "reachable_helper"})
    assert "unreachable_helper" not in live, (
        "the detector reached a function nothing calls; it is counting references "
        "rather than consumption, which is the failure that hid F-31"
    )
