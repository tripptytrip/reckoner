"""The interpreter: faithful over pretty, and enforced rather than intended.

The central test here is a round-trip through **human notation** —
``read_expr(render_expr(e)) == e``. That is what stops the renderer becoming a
second, undocumented model: any beautification that changes the structure fails
to read back. Prettifying ``21 + (−6)`` to ``21 − 6`` produces a ``SUB`` node,
which is a different state, and the build fails.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

from reckoner.config import Config
from reckoner.episode import Episode, Problem
from reckoner.expr import Expr, add, div, eq, mul, num, sub, tokens, var
from reckoner.interpreter import (
    MINUS,
    RenderError,
    Step,
    glyph_cells,
    glyph_panel,
    read_expr,
    render_derivation,
    render_expr,
    render_step,
    state_tokens_line,
)
from reckoner.rules import RULE_BY_NAME, RULESET_VERSION, successors
from reckoner.vocab import (
    GOAL_SOLVE,
    SUB,
    VAR_X,
    VAR_Y,
    VOCAB_VERSION,
)

REPO = Path(__file__).resolve().parents[1]
DERIVATIONS_MD = REPO / "docs" / "derivations.md"

X = var(VAR_X)
Y = var(VAR_Y)
CFG = Config()


# ---------------------------------------------------------------------------
# The faithfulness rule, stated as a case
# ---------------------------------------------------------------------------


def test_the_true_intermediate_is_rendered_honestly() -> None:
    """`3x = 21 + (−6)`, not `3x = 21 − 6`. The state has no SUB node in it.

    This is the whole ruling in one test. The pretty form would read back as a
    ``SUB``, which is a different state; the honest form reads back as itself.
    """
    state = eq(mul(num(3), X), add(num(21), num(-6)))
    assert SUB not in tokens(state)

    text = render_expr(state)
    assert text == "3x = 21 + (−6)"
    assert read_expr(text) == state

    # And the beautified form is a different state — which is why it is banned.
    pretty = "3x = 21 − 6"
    assert read_expr(pretty) != state
    assert SUB in tokens(read_expr(pretty))


def test_a_genuine_sub_node_renders_as_a_subtraction() -> None:
    """The other polarity: `−` in the text must mean, and only mean, a SUB node."""
    state = sub(num(21), num(6))
    assert render_expr(state) == "21 − 6"
    assert read_expr("21 − 6") == state


@pytest.mark.parametrize(
    ("expr", "text"),
    [
        (num(0), "0"),
        (num(-6), "−6"),
        (num(1887), "1887"),
        (X, "x"),
        (mul(num(3), X), "3x"),
        (mul(num(-3), X), "−3x"),
        (mul(num(2), num(3)), "2 × 3"),
        (mul(num(3), X, Y), "3xy"),
        (add(mul(num(3), X), num(6)), "3x + 6"),
        (add(num(21), num(-6)), "21 + (−6)"),
        (add(mul(num(3), X), mul(num(-3), X)), "3x + (−3x)"),
        (sub(num(21), num(6)), "21 − 6"),
        (sub(num(21), num(-6)), "21 − (−6)"),
        (div(num(6), num(2)), "6 ÷ 2"),
        (eq(mul(num(3), X), num(15)), "3x = 15"),
        (eq(X, num(-5)), "x = −5"),
        (mul(num(3), add(X, num(1))), "3(x + 1)"),
        (sub(add(num(1), num(2)), num(3)), "1 + 2 − 3"),
    ],
)
def test_rendering_is_exact(expr: Expr, text: str) -> None:
    assert render_expr(expr) == text
    assert read_expr(text) == expr


def test_minus_is_the_real_minus_sign() -> None:
    """U+2212, not a hyphen. A reader copying it back must get the same state."""
    assert MINUS == "−"
    assert "-" not in render_expr(num(-6))


def test_parentheses_are_never_decorative() -> None:
    """A paren is a claim about structure, so an unnecessary one is a false claim."""
    assert render_expr(add(mul(num(3), X), num(6))) == "3x + 6"  # not "(3x) + 6"
    assert render_expr(eq(add(X, num(1)), num(2))) == "x + 1 = 2"  # not "(x + 1) = 2"


# ---------------------------------------------------------------------------
# Round-trip, at scale
# ---------------------------------------------------------------------------


def random_state(rng: random.Random) -> Expr:
    """States drawn the way search meets them: problems and mid-derivation forms."""
    a = rng.choice([c for c in range(-9, 10) if c != 0])
    shape = rng.choice(("solve", "solve2", "arith"))
    if shape == "solve":
        state: Expr = eq(
            add(mul(num(a), X), num(rng.randrange(-999, 1000))), num(rng.randrange(-999, 1000))
        )
    elif shape == "solve2":
        b = rng.choice([c for c in range(-9, 10) if c != a])
        state = eq(
            add(mul(num(a), X), num(rng.randrange(-99, 100))),
            add(mul(num(b), X), num(rng.randrange(-99, 100))),
        )
    else:
        state = add(num(rng.randrange(-999, 1000)), mul(num(a), num(rng.randrange(-99, 100))))
    for _ in range(rng.randint(0, 4)):
        options = successors(state)
        if not options:
            break
        state = rng.choice(options)[1]
    return state


def test_human_notation_round_trips_on_5000_states() -> None:
    """**Chunk 4 gate.** The renderer round-trips with the parser on every state.

    Not the token codec — the *human* notation. If a rendering cannot be read
    back into the state it depicts, the rendering is a claim about a state that
    does not exist.
    """
    rng = random.Random(0x4E7)
    widest = 0
    with_negatives = with_parens = 0
    for index in range(5000):
        state = random_state(rng)
        text = render_expr(state)
        assert read_expr(text) == state, f"round-trip failed at {index}: {text!r}"
        widest = max(widest, len(text))
        with_negatives += MINUS in text
        with_parens += "(" in text

    assert with_negatives >= 1000, f"only {with_negatives}/5000 rendered a negative"
    assert with_parens >= 500, f"only {with_parens}/5000 needed a parenthesis"
    print(
        f"\n  round-trip: 5000 states, widest {widest} chars, {with_negatives} with −, {with_parens} with ()"
    )


def test_malformed_notation_is_rejected_cleanly() -> None:
    for text in ("", "3 +", "(3", "3)", "= 5", "3 5", "x +"):
        with pytest.raises((RenderError, ValueError)):
            read_expr(text)


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


def spec_example() -> tuple[Problem, list[Step]]:
    problem = Problem(
        goal=GOAL_SOLVE, expr=eq(add(mul(num(3), X), num(6)), num(21)), par=3, target=VAR_X
    )
    episode = Episode(cfg=CFG, rng=random.Random(0))
    episode.reset(problem)
    steps: list[Step] = []
    for name, site in (("sub_both_sides", 5), ("eval_add", 4), ("div_both_sides", 2)):
        before = episode.expr
        episode.step((RULE_BY_NAME[name].rule_id, site))
        assert before is not None and episode.expr is not None
        steps.append(Step(RULE_BY_NAME[name].rule_id, site, before, episode.expr))
    return problem, steps


def test_a_step_names_its_rule_by_id_and_name() -> None:
    """Grep unity: a line of derivation text leads straight to the code."""
    _, steps = spec_example()
    line = render_step(steps[0])
    assert "rule 5 sub_both_sides" in line
    assert "site 5" in line
    assert "3x + 6 = 21" in line and "3x = 21 + (−6)" in line


def test_a_derivation_is_stamped_with_its_versions() -> None:
    """Par is denominated in a rule system; so is the derivation that achieves it."""
    problem, steps = spec_example()
    text = render_derivation(problem, steps)
    assert f"ruleset_version={RULESET_VERSION}" in text
    assert f"vocab_version={VOCAB_VERSION}" in text
    assert "par    3  (par_source=bfs)" in text


def test_every_intermediate_state_in_a_derivation_round_trips() -> None:
    problem, steps = spec_example()
    for step in (steps[0], *steps):
        assert read_expr(render_expr(step.before)) == step.before
        assert read_expr(render_expr(step.after)) == step.after
    assert read_expr(render_expr(problem.expr)) == problem.expr


def test_rendering_is_deterministic_across_processes() -> None:
    """Byte-identical across runs *and* interpreters, under four hash seeds."""
    import os

    program = (
        "from reckoner.expr import add, eq, mul, num, var;"
        "from reckoner.interpreter import render_expr;"
        "from reckoner.vocab import VAR_X, VAR_Y;"
        "X=var(VAR_X); Y=var(VAR_Y);"
        "print(render_expr(eq(add(mul(num(3),X),mul(num(-2),Y),num(-6)),add(num(21),num(-6)))))"
    )
    outputs = set()
    for seed in ("0", "1", "99", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.add(
            subprocess.run(
                [sys.executable, "-c", program], capture_output=True, text=True, check=True, env=env
            ).stdout
        )
    assert len(outputs) == 1, f"rendering varied across PYTHONHASHSEED: {outputs}"


# ---------------------------------------------------------------------------
# The glyph panel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("digit", "cells"),
    [(0, (0, 0, 0, 0)), (3, (0, 0, 0, 3)), (12, (0, 0, 2, 2)), (624, (4, 4, 4, 4))],
)
def test_glyph_cells(digit: int, cells: tuple) -> None:
    assert glyph_cells(digit) == cells
    a, b, c, d = cells
    assert a * 125 + b * 25 + c * 5 + d == digit  # the panel is lossless


def test_glyph_cells_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        glyph_cells(625)


def test_glyph_panel_shows_the_base_625_digits() -> None:
    panel = glyph_panel(1887)  # 3 × 625 + 12
    assert "1887 = D3 D12" in panel
    assert glyph_panel(-6).startswith(f"{MINUS}6 = D6")


def test_state_tokens_line_is_the_audit_trail() -> None:
    state = eq(X, num(5))
    assert state_tokens_line(state) == " ".join(str(t) for t in tokens(state))


# ---------------------------------------------------------------------------
# The 50-derivation document — the manual gate's artifact
# ---------------------------------------------------------------------------


def test_derivations_md_exists() -> None:
    assert DERIVATIONS_MD.exists(), "docs/derivations.md is chunk 4's manual gate — run `make docs`"


def test_derivations_md_is_current() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "render_derivations.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_derivations_md_is_byte_identical_across_runs() -> None:
    """Deterministic output. A proofread of a document that drifts proves nothing."""
    import importlib

    module = importlib.import_module("render_derivations")
    assert module.document() == module.document()


def test_the_manifest_meets_every_declared_stratum() -> None:
    """The coverage claim, asserted rather than eyeballed.

    A gate reports what it covered. These are the strata the brief named: all
    seven rules at least three times each, all three goals, multi-digit and
    negative base-625 numerals, and x on both sides.
    """
    import importlib

    module = importlib.import_module("render_derivations")
    _, manifest = module.build()

    assert manifest["derivations"] == 50
    thin = {name: count for name, count in manifest["rules"].items() if count < 3}
    assert not thin, f"rules under-covered in the proofread set: {thin}"
    assert len(manifest["rules"]) == 7, f"a rule never appears: {sorted(manifest['rules'])}"
    assert set(manifest["goals"]) == {"SOLVE", "EVALUATE", "SIMPLIFY"}
    assert manifest["with_multi_digit_numerals"] >= 5
    assert manifest["with_negative_numerals"] >= 10
    assert manifest["x_on_both_sides"] >= 1
    assert max(manifest["steps_histogram"]) >= 5, "no derivation is long enough to be interesting"


def test_no_derivation_ran_away() -> None:
    """A step-capped derivation in a proofread set is an exhibit of a bug.

    The first draft of the fixture script produced six of them — `add_both_sides`
    preferred by a policy that never terminates — and the manifest is what
    caught it.
    """
    import importlib

    module = importlib.import_module("render_derivations")
    for note, problem, opening in module.fixtures():
        steps = module.derive(problem, opening)
        assert len(steps) < 12, f"derivation '{note}' hit the step cap"


def test_every_state_in_the_document_round_trips() -> None:
    """The gate's own claim, applied to the artifact Tom proofreads."""
    import importlib

    module = importlib.import_module("render_derivations")
    checked = 0
    for _note, problem, opening in module.fixtures():
        steps = module.derive(problem, opening)
        for state in (problem.expr, *(s.after for s in steps)):
            assert read_expr(render_expr(state)) == state, render_expr(state)
            checked += 1
    assert checked >= 100, f"only {checked} states checked"
    print(f"\n  document states round-tripped: {checked}")
