"""Render the 50 proofread derivations and their coverage manifest.

This is the manual gate of chunk 4: the report stops here for Tom's eyes. What
the proofread certifies is not "the code ran" — it is that **a reader can follow
every line without the code**, that the rule named matches the transformation
shown every time, that base-625 numerals (multi-digit and negative especially)
render correctly, and that nothing was prettified into unfaithfulness.

Expect ``21 + (−6)`` on the page. That is not a wart; it is the state.

The manifest ships with the renders because a gate must report what it covered,
not only that it passed — 50 derivations that all happened to be one-step SOLVEs
would look identical to 50 that swept the space.

    make docs      # regenerates docs/derivations.md
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from reckoner.config import Config
from reckoner.episode import Episode, Problem
from reckoner.expr import Expr, Num, Op, add, eq, has_var, mul, num, sub, var
from reckoner.interpreter import Step, glyph_panel, render_derivation, state_tokens_line
from reckoner.rules import RULE_BY_NAME, RULES, enumerate_sites
from reckoner.vocab import EQ, GOAL_EVALUATE, GOAL_SIMPLIFY, GOAL_SOLVE, VAR_X, VAR_Y

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "docs" / "derivations.md"

X = var(VAR_X)
Y = var(VAR_Y)
CFG = Config()


# ---------------------------------------------------------------------------
# Driving an episode along a chosen policy
# ---------------------------------------------------------------------------


def _move_site(
    episode: Episode, rule_id: int, candidates: list[tuple[int, int]]
) -> tuple[int, int]:
    """Choose which addend a both-sides rule moves.

    Two cases, and getting them the wrong way round is what made the first draft
    of this script loop on every x-on-both-sides problem:

    * **Both sides carry the variable** — move a *variable* addend, so the two
      like terms land on one side and ``combine_like_terms`` can fire. Moving a
      constant here just shuffles constants back and forth forever.
    * **Otherwise** — move a *numeric* addend, clearing the constant off the
      variable's side.
    """
    sites = enumerate_sites(episode.expr)  # type: ignore[arg-type]
    state = episode.expr
    both_sides = (
        isinstance(state, Op) and state.kind == EQ and all(has_var(side) for side in state.children)
    )
    wanted = (lambda node: has_var(node)) if both_sides else (lambda node: isinstance(node, Num))
    for action in candidates:
        node = sites[action[1]].node
        if len(sites[action[1]].path) == 2 and wanted(node):
            return action
    return candidates[0]


def _pick(episode: Episode, prefer: list[str]) -> tuple[int, int] | None:
    actions = episode.legal()
    for name in prefer:
        rule_id = RULE_BY_NAME[name].rule_id
        candidates = [a for a in actions if a[0] == rule_id]
        if not candidates:
            continue
        if name in ("sub_both_sides", "add_both_sides"):
            return _move_site(episode, rule_id, candidates)
        return candidates[0]
    return None


def derive(problem: Problem, opening: list[str] | None = None, cap: int = 12) -> list[Step]:
    """Run a problem to its terminal state, recording every rewrite.

    ``opening`` forces the first k rules, then the default policy takes over.
    That is how ``add_both_sides`` gets exercised at all: it is always legal and
    only ever grows the state, so a policy that merely *prefers* it never
    terminates — the first draft of this script produced six 12-step runaways,
    which the coverage manifest caught and no amount of eyeballing would have.
    """
    order = [
        "div_both_sides",
        "eval_add",
        "eval_sub",
        "eval_mul",
        "combine_like_terms",
        "sub_both_sides",
    ]
    episode = Episode(cfg=CFG, rng=random.Random(0))
    episode.reset(problem)
    steps: list[Step] = []
    while not episode.done and len(steps) < cap:
        stage = [opening[len(steps)]] if opening and len(steps) < len(opening) else order
        action = _pick(episode, stage)
        if action is None:
            break
        before = episode.expr
        episode.step(action)
        assert before is not None and episode.expr is not None
        steps.append(Step(action[0], action[1], before, episode.expr))
    return steps


# ---------------------------------------------------------------------------
# The stratified fixture set
# ---------------------------------------------------------------------------


def solve(expr: Expr, par: int) -> Problem:
    return Problem(goal=GOAL_SOLVE, expr=expr, par=par, target=VAR_X)


def _linear(a: int, b: int, answer: int) -> Expr:
    return eq(add(mul(num(a), X), num(b)), num(a * answer + b))


def fixtures() -> list[tuple[str, Problem, list[str] | None]]:
    """50 derivations, stratified. Each entry is (note, problem, forced opening).

    Notes never restate the equation. Canonicalisation may reorder an EQ's sides
    (C7) and a constant written `− 1250` is really `+ (−1250)` in the state — a
    caption that says otherwise is the same unfaithfulness this chunk exists to
    prevent, just above the fold. The ``start`` line shows the real thing.

    The strata are declared here rather than emerging: all seven rules, all
    three goals, pars 1–6, multi-digit and negative base-625 numerals, and
    x-on-both-sides. ``add_both_sides`` is reachability-redundant (ROUND-01) so
    it never appears in an optimal derivation — the entries that exercise it are
    labelled *illustrative*, because an unlabelled suboptimal derivation in a
    proofread set is a misleading exhibit.
    """
    out: list[tuple[str, Problem, list[str] | None]] = []

    # --- SOLVE, pars 1..6, ordinary coefficients -------------------------
    out.append(("par 1 — one division", solve(eq(mul(num(3), X), num(15)), 1), None))
    out.append(("par 1 — negative coefficient", solve(eq(mul(num(-4), X), num(20)), 1), None))
    out.append(("par 3 — the spec's own example", solve(_linear(3, 6, 5), 3), None))
    out.append(("par 3 — negative constant", solve(_linear(2, -4, 7), 3), None))
    out.append(("par 3 — negative answer", solve(_linear(5, 3, -4), 3), None))
    out.append(("par 3 — coefficient 1 vanishes", solve(_linear(1, 9, 12), 3), None))

    # --- multi-digit base-625 numerals (>= 625 needs two digits) ---------
    out.append(("multi-digit — three-figure constant", solve(_linear(3, 700, 500), 3), None))
    out.append(
        ("multi-digit — four-figure right side", solve(eq(mul(num(7), X), num(8750)), 1), None)
    )
    out.append(("multi-digit negative constant", solve(_linear(4, -1250, 875), 3), None))
    out.append(
        ("three base-625 digits in one numeral", solve(eq(mul(num(2), X), num(781250)), 1), None)
    )

    # --- x on both sides -------------------------------------------------
    out.append(
        (
            "x on both sides",
            solve(eq(add(mul(num(5), X), num(3)), add(mul(num(2), X), num(18))), 5),
            None,
        )
    )
    out.append(
        (
            "x on both sides, negative constant",
            solve(eq(add(mul(num(7), X), num(-2)), add(mul(num(3), X), num(14))), 5),
            None,
        )
    )
    out.append(
        (
            "x on both sides, negative answer",
            solve(eq(add(mul(num(4), X), num(6)), add(mul(num(9), X), num(31))), 5),
            None,
        )
    )

    # --- EVALUATE, exercising eval_add / eval_sub / eval_mul -------------
    out.append(
        ("evaluate — a sum", Problem(goal=GOAL_EVALUATE, expr=add(num(17), num(-25)), par=1), None)
    )
    out.append(
        (
            "evaluate — a numeric SUB node",
            Problem(goal=GOAL_EVALUATE, expr=sub(num(21), num(6)), par=1),
            None,
        )
    )
    out.append(
        (
            "evaluate — SUB inside a sum",
            Problem(goal=GOAL_EVALUATE, expr=add(sub(num(40), num(15)), num(-5)), par=2),
            None,
        )
    )
    out.append(
        (
            "evaluate — a product",
            Problem(goal=GOAL_EVALUATE, expr=mul(num(25), num(25)), par=1),
            None,
        )
    )
    out.append(
        (
            "evaluate — multi-digit product",
            Problem(goal=GOAL_EVALUATE, expr=mul(num(625), num(3)), par=1),
            None,
        )
    )
    out.append(
        (
            "evaluate — mixed, three steps",
            Problem(
                goal=GOAL_EVALUATE,
                expr=add(mul(num(12), num(12)), sub(num(9), num(30)), num(7)),
                par=3,
            ),
            None,
        )
    )
    out.append(
        (
            "evaluate — negative product",
            Problem(goal=GOAL_EVALUATE, expr=mul(num(-7), num(9)), par=1),
            None,
        )
    )

    # --- SIMPLIFY, exercising combine_like_terms -------------------------
    out.append(
        (
            "simplify — like terms",
            Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(3), X), mul(num(2), X)), par=1),
            None,
        )
    )
    out.append(
        (
            "simplify — like terms cancel to zero",
            Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(3), X), mul(num(-3), X)), par=1),
            None,
        )
    )
    out.append(
        (
            "simplify — coefficient falls to 1",
            Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(4), X), mul(num(-3), X)), par=1),
            None,
        )
    )
    out.append(
        (
            "simplify — terms and constants",
            Problem(
                goal=GOAL_SIMPLIFY,
                expr=add(mul(num(3), X), mul(num(2), X), num(4), num(-9)),
                par=2,
            ),
            None,
        )
    )
    out.append(
        (
            "simplify — two variables stay apart",
            Problem(
                goal=GOAL_SIMPLIFY,
                expr=add(mul(num(3), X), mul(num(2), Y), mul(num(4), X)),
                par=1,
            ),
            None,
        )
    )
    out.append(
        (
            "simplify — bare x doubles",
            Problem(goal=GOAL_SIMPLIFY, expr=add(X, X, num(1)), par=1),
            None,
        )
    )
    out.append(
        (
            "simplify — multi-digit coefficients",
            Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(700), X), mul(num(950), X)), par=1),
            None,
        )
    )
    out.append(
        (
            "simplify — a product of numerals inside a sum",
            Problem(goal=GOAL_SIMPLIFY, expr=add(mul(num(6), num(7)), X), par=1),
            None,
        )
    )

    # --- illustrative: add_both_sides (ROUND-01 — never optimal) ---------
    illustrative = ["add_both_sides"]  # fire it once, then recover under the normal policy
    out.append(
        (
            "ILLUSTRATIVE — add_both_sides then recover",
            solve(eq(mul(num(3), X), num(15)), 1),
            illustrative,
        )
    )
    out.append(
        ("ILLUSTRATIVE — add_both_sides on a constant", solve(_linear(2, 5, 6), 3), illustrative)
    )
    out.append(
        ("ILLUSTRATIVE — add_both_sides with negatives", solve(_linear(3, -9, 4), 3), illustrative)
    )

    # --- fill out to 50 with a seeded spread over the same strata --------
    rng = random.Random(20260814)
    while len(out) < 50:
        index = len(out)
        kind = index % 3
        if kind == 0:
            a = rng.choice([c for c in range(-9, 10) if c not in (0, 1)])
            answer = rng.randrange(-15, 16)
            b = rng.randrange(-30, 31)
            out.append((f"solve — spread {index}", solve(_linear(a, b, answer), 3), None))
        elif kind == 1:
            expr = add(num(rng.randrange(-900, 901)), num(rng.randrange(-900, 901)))
            out.append(
                (f"evaluate — spread {index}", Problem(goal=GOAL_EVALUATE, expr=expr, par=1), None)
            )
        else:
            a, b = rng.randrange(-30, 31), rng.randrange(-30, 31)
            out.append(
                (
                    f"simplify — spread {index}",
                    Problem(
                        goal=GOAL_SIMPLIFY, expr=add(mul(num(a), X), mul(num(b), X), num(a)), par=2
                    ),
                    None,
                )
            )
    return out[:50]


# ---------------------------------------------------------------------------
# Manifest + document
# ---------------------------------------------------------------------------


def build() -> tuple[str, dict]:
    entries = fixtures()
    rule_counts: Counter[str] = Counter()
    goal_counts: Counter[str] = Counter()
    step_counts: Counter[int] = Counter()
    multi_digit = negative = both_sides = 0
    body: list[str] = []

    goal_names = {GOAL_SOLVE: "SOLVE", GOAL_EVALUATE: "EVALUATE", GOAL_SIMPLIFY: "SIMPLIFY"}

    for index, (note, problem, opening) in enumerate(entries, start=1):
        steps = derive(problem, opening)
        episode = Episode(cfg=CFG, rng=random.Random(0))
        episode.reset(problem)
        for step in steps:
            episode.step((step.rule_id, step.site_id))
        result = episode.result() if episode.done else None

        for step in steps:
            rule_counts[RULE_BY_NAME_BY_ID[step.rule_id]] += 1
        goal_counts[goal_names[problem.goal]] += 1
        step_counts[len(steps)] += 1

        all_states = [problem.expr, *(s.after for s in steps)]
        numerals = [n for state in all_states for n in _numerals(state)]
        if any(abs(v) >= 625 for v in numerals):
            multi_digit += 1
        if any(v < 0 for v in numerals):
            negative += 1
        if "both sides" in note:
            both_sides += 1

        body.append(f"### {index:02d}. {note}\n")
        body.append("```")
        body.append(render_derivation(problem, steps, result))
        body.append("```")
        big = next((v for v in numerals if abs(v) >= 625), None)
        if big is not None:
            body.append("\nbase-625 panel for the largest numeral in this derivation:\n")
            body.append("```")
            body.append(glyph_panel(big))
            body.append("```")
        body.append(f"\n<sub>start tokens: `{state_tokens_line(problem.expr)}`</sub>\n")

    manifest = {
        "derivations": len(entries),
        "rules": dict(sorted(rule_counts.items())),
        "goals": dict(sorted(goal_counts.items())),
        "steps_histogram": dict(sorted(step_counts.items())),
        "with_multi_digit_numerals": multi_digit,
        "with_negative_numerals": negative,
        "x_on_both_sides": both_sides,
    }
    return "\n".join(body), manifest


RULE_BY_NAME_BY_ID = {rule.rule_id: rule.name for rule in RULES}


def _numerals(expr: Expr) -> list[int]:
    out: list[int] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Num):
            out.append(node.value)
        elif isinstance(node, Op):
            stack.extend(node.children)
    return out


def document() -> str:
    body, manifest = build()
    lines = [
        "# 50 derivations — the chunk 4 manual gate",
        "",
        "*Generated by `make docs` from `scripts/render_derivations.py`. Do not hand-edit —",
        "`tests/test_interpreter.py::test_derivations_md_is_current` regenerates and compares.*",
        "",
        "**What the proofread certifies:** that a reader can follow every line without the",
        "code; that the rule named matches the transformation shown, every time; that",
        "base-625 numerals — multi-digit and negative especially — render correctly; and",
        "that nothing was prettified into unfaithfulness.",
        "",
        "**Expect `21 + (−6)` on the page.** That is not a bug; it is the state.",
        "`sub_both_sides` moves an addend across as its negation, and no `SUB` node exists",
        "there. Rendering it as `21 − 6` would read back as a `SUB` node — a different",
        "state — which is why `read_expr(render_expr(e)) == e` is a test and not a wish.",
        "",
        "## Coverage manifest",
        "",
        "A gate reports what it covered, not only that it passed.",
        "",
        "| stratum | count |",
        "|---|---|",
    ]
    lines.append(f"| derivations | {manifest['derivations']} |")
    for name, count in manifest["rules"].items():
        lines.append(f"| rule `{name}` | {count} |")
    for name, count in manifest["goals"].items():
        lines.append(f"| goal {name} | {count} |")
    lines.append(
        f"| with multi-digit base-625 numerals | {manifest['with_multi_digit_numerals']} |"
    )
    lines.append(f"| with negative numerals | {manifest['with_negative_numerals']} |")
    lines.append(f"| x on both sides | {manifest['x_on_both_sides']} |")
    lines.append(f"| derivation lengths | {dict(manifest['steps_histogram'])} |")
    lines += [
        "",
        "`add_both_sides` is reachability-redundant (`REGISTERED-ROUNDS.md` ROUND-01), so it",
        "never appears in an optimal derivation. The entries that exercise it are labelled",
        "ILLUSTRATIVE — an unlabelled suboptimal derivation in a proofread set is a",
        "misleading exhibit.",
        "",
        "---",
        "",
        body,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = document()
    if args.check:
        current = args.out.read_text() if args.out.exists() else ""
        if current != rendered:
            print(f"{args.out} is STALE — run `make docs`")
            return 1
        print(f"{args.out} is current")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    _, manifest = build()
    print(f"wrote {args.out}")
    for key, value in manifest.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
