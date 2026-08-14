"""The external interpreter: a derivation, rendered for a human to read.

This module is the interpretability claim of the whole programme, made concrete.
The model's reasoning trace *is* its move sequence; this turns that sequence into
notation a person can check, one line per rewrite, with the rule named.

**Faithful over pretty, and it is enforced, not intended.**
``read_expr(render_expr(e)) == e`` for every state — a round-trip through the
*human notation*, not through the token codec. That is what makes the rule
un-cheatable: the true intermediate of ``3x + 6 = 21`` is ``3x = 21 + (−6)``,
because ``sub_both_sides`` moves the addend across as its negation and no ``SUB``
node exists in that state. Beautifying it to ``21 − 6`` would read back as a
``SUB`` node, fail the round-trip, and fail the build. A renderer that improves
on the state is not an interpreter, it is a second, undocumented model — and a
lie with good typography is worse than no rendering at all.

So: expect ``21 + (−6)`` on the page. That is not a wart; it is the state.

Notation
--------
``+ − × ÷ =`` with U+2212 for minus. Precedence is the ordinary one, and
parentheses appear exactly where precedence or a leading minus requires them —
never decoratively, because a decorative paren is a claim about structure.

  * ``3x`` — a numeral juxtaposed with a non-numeral factor
  * ``2 × 3`` — two numerals, where juxtaposition would be ambiguous
  * ``21 + (−6)`` — an ``ADD`` whose second addend is negative
  * ``21 − 6``  — a genuine ``SUB`` node, and *only* that

One formatter of states, ever
-----------------------------
**A caption describes, or it calls** :func:`render_expr` **— there is no third
path.** Hand-formatting a state anywhere outside this module is the renderer bug
this module exists to prevent, recommitted one line above the fold: a heading
reading ``4x − 1250`` describes a state that is really ``4x + (−1250)``, and a
heading naming an equation's sides in the order they were typed describes a
state that C7 may have reordered. Same family as the shared identity
normalizer — one implementation of "how a state is written down", and every
caller goes through it.

``tests/test_interpreter.py::test_no_caption_hand_formats_a_state`` enforces the
cheap half: no operator glyph and no coefficient-variable juxtaposition in a
fixture caption. The law is wider than the test; the test is what is affordable.

Rule ids are rendered beside rule names so a line of derivation text greps
straight to the code that produced it, and every derivation is stamped with
``ruleset_version`` and ``vocab_version``: par is denominated in a rule system,
and so is a derivation.

The glyph panel
---------------
``glyph_panel`` shows the base-625 digits of a numeral as 2×2 grids of base-5
cells (625 = 5⁴). **This is a reckoner-local convention, not the canonical
base-625 renderer**, and the panel's own output says so on every render — the
panel's purpose was continuity with the real base-625 system, so an unlabelled
local convention would look like the system it is not, which is a subtler lie
than claiming reuse outright.

The upstream renderer lives in a public repository; it was deferred, not
impossible. The vendoring pass is registered as ``REGISTERED-ROUNDS.md``
ROUND-03 so the deferral has a name. Until then the panel is off the main path:
the derivation renderer does not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from reckoner.episode import EpisodeResult, Problem, describe_goal, outcome_z
from reckoner.expr import Expr, Num, Op, Var, make_op, num, tokens, var
from reckoner.rules import RULE_BY_ID, RULESET_VERSION, enumerate_sites
from reckoner.vocab import (
    ADD,
    BASE,
    DIV,
    EQ,
    MUL,
    NAME_VAR,
    SUB,
    VAR_NAME,
    VOCAB_VERSION,
    to_digits_of,
)

MINUS = "−"  # U+2212 MINUS SIGN — not a hyphen
TIMES = "×"
DIVIDE = "÷"
ARROW = "──►"  # ──►

# Precedence bands. An operand is parenthesised when its own band is looser than
# the slot it sits in — and never otherwise.
_REL, _SUM, _PRODUCT, _ATOM = 0, 1, 2, 3

_INFIX = {ADD: " + ", SUB: f" {MINUS} ", DIV: f" {DIVIDE} ", EQ: " = "}
_BAND = {EQ: _REL, ADD: _SUM, SUB: _SUM, MUL: _PRODUCT, DIV: _PRODUCT}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _wrap(text: str, band: int, needed: int, *, first: bool, minus_safe: bool = False) -> str:
    """Parenthesise iff precedence demands it, or a leading minus would mislead.

    ``minus_safe`` marks a slot where a leading ``−`` cannot be misread as a
    binary operator — the right side of ``=``, where ``x = −5`` is unambiguous
    and ``x = (−5)`` would be a decorative paren, i.e. a false claim about
    structure.
    """
    if band < needed or (not first and not minus_safe and text.startswith(MINUS)):
        return f"({text})"
    return text


def _child_need(band: int, *, first: bool) -> int:
    """Precedence a child must clear. Non-first children must *beat* the band.

    The reader is left-associative, so an unparenthesised same-band child in a
    later slot re-associates on the way back in: ``−5 + (40 − 15)`` printed as
    ``−5 + 40 − 15`` reads as ``(−5 + 40) − 15`` — equal in value, a different
    tree, and therefore a failed round-trip. Equal value is not the standard
    here; equal structure is.
    """
    return band if first else band + 1


def _render(node: Expr) -> tuple[str, int]:
    if isinstance(node, Num):
        # A numeral is an atom whatever its sign: the parenthesising of `−6` is
        # governed by the leading-minus rule below, not by precedence. Treating
        # a negative as a loose band instead would render `−3x` as `(−3)x` —
        # still faithful, but not what a human writes, and the point of this
        # module is that a human reads it.
        return str(node.value).replace("-", MINUS), _ATOM
    if isinstance(node, Var):
        return VAR_NAME[node.token], _ATOM
    if not isinstance(node, Op):
        raise TypeError(f"not an expression node: {node!r}")

    parts = [_render(child) for child in node.children]

    if node.kind == MUL:
        pieces = [
            _wrap(text, band, _child_need(_PRODUCT, first=(index == 0)), first=(index == 0))
            for index, (text, band) in enumerate(parts)
        ]
        out = pieces[0]
        for child, piece in zip(node.children[1:], pieces[1:], strict=True):
            # Juxtapose unless the right factor is itself a numeral, where `23`
            # would be one number rather than two factors.
            out += f" {TIMES} {piece}" if isinstance(child, Num) else piece
        return out, _PRODUCT

    if node.kind in (SUB, DIV):
        left, right = parts
        # The right operand of a non-commutative operator needs a tighter band:
        # `a − (b − c)` is not `a − b − c`.
        rendered = (
            _wrap(left[0], left[1], _BAND[node.kind], first=True)
            + _INFIX[node.kind]
            + _wrap(right[0], right[1], _BAND[node.kind] + 1, first=False)
        )
        return rendered, _BAND[node.kind]

    band = _BAND[node.kind]
    # A leading minus after `=` cannot be read as subtraction, so `x = −5` needs
    # no paren; after `+` it can, so `21 + (−6)` does.
    minus_safe = node.kind == EQ
    pieces = [
        _wrap(
            text,
            child_band,
            _child_need(band, first=(index == 0)),
            first=(index == 0),
            minus_safe=minus_safe,
        )
        for index, (text, child_band) in enumerate(parts)
    ]
    return _INFIX[node.kind].join(pieces), band


def render_expr(expr: Expr) -> str:
    """Human notation for a canonical state. The inverse of :func:`read_expr`."""
    return _render(expr)[0]


# ---------------------------------------------------------------------------
# Reading — the arbiter that makes "faithful" testable
# ---------------------------------------------------------------------------


class RenderError(ValueError):
    """The rendered text is not readable back into the state it claims to show."""


@dataclass
class _Reader:
    text: str
    pos: int = 0

    def peek(self) -> str:
        while self.pos < len(self.text) and self.text[self.pos] == " ":
            self.pos += 1
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def take(self, char: str) -> bool:
        if self.peek() == char:
            self.pos += 1
            return True
        return False

    def equation(self) -> Expr:
        left = self.sum()
        if self.take("="):
            return make_op(EQ, (left, self.sum()))
        return left

    def sum(self) -> Expr:
        node = self.product()
        while True:
            if self.take("+"):
                node = make_op(ADD, (node, self.product()))
            elif self.take(MINUS):
                node = make_op(SUB, (node, self.product()))
            else:
                return node

    def product(self) -> Expr:
        node = self.factor()
        while True:
            if self.take(TIMES):
                node = make_op(MUL, (node, self.factor()))
            elif self.take(DIVIDE):
                node = make_op(DIV, (node, self.factor()))
            elif self.peek() and (self.peek() == "(" or self.peek() in NAME_VAR):
                node = make_op(MUL, (node, self.factor()))  # juxtaposition
            else:
                return node

    def factor(self) -> Expr:
        char = self.peek()
        if char == "(":
            self.pos += 1
            inner = self.equation()
            if not self.take(")"):
                raise RenderError(f"unclosed '(' at {self.pos} in {self.text!r}")
            return inner
        if char in NAME_VAR:
            self.pos += 1
            return var(NAME_VAR[char])
        sign = 1
        if char == MINUS:
            self.pos += 1
            sign = -1
            char = self.peek()
        if not char.isdigit():
            raise RenderError(f"expected a factor at {self.pos} in {self.text!r}")
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        return num(sign * int(self.text[start : self.pos]))


def read_expr(text: str) -> Expr:
    """Parse human notation back into a canonical state.

    Exists so "the rendering is the derivation" is a *test*, not a promise.
    """
    reader = _Reader(text)
    node = reader.equation()
    if reader.peek():
        raise RenderError(f"trailing input at {reader.pos} in {text!r}")
    return node


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Step:
    """One rewrite: the action taken, and the state it produced."""

    rule_id: int
    site_id: int
    before: Expr
    after: Expr


def step_label(step: Step) -> str:
    """``[rule N name @ site S (subterm)]`` — the rule id is there for grep unity."""
    rule = RULE_BY_ID[step.rule_id]
    site = enumerate_sites(step.before)[step.site_id]
    return f"[rule {step.rule_id} {rule.name} @ site {step.site_id} ({render_expr(site.node)})]"


def render_step(step: Step, *, before_width: int = 0, label_width: int = 0) -> str:
    """``before  ──[rule N name @ site S (subterm)]──►  after``.

    The rule id is rendered beside the name so a line of derivation text greps
    straight to the rule that produced it; the site's subterm is shown because a
    bare site id is not something a human can check.
    """
    rule = RULE_BY_ID[step.rule_id]
    site = enumerate_sites(step.before)[step.site_id]
    label = f"[rule {step.rule_id} {rule.name} @ site {step.site_id} ({render_expr(site.node)})]"
    before = render_expr(step.before).ljust(before_width)
    return f"{before}  ──{label}{ARROW}  {render_expr(step.after)}"


def render_derivation(
    problem: Problem, steps: list[Step], result: EpisodeResult | None = None
) -> str:
    """The whole derivation, stamped with the versions it is denominated in."""
    lines = [
        f"ruleset_version={RULESET_VERSION}  vocab_version={VOCAB_VERSION}",
        f"goal   {describe_goal(problem)}",
        f"par    {problem.par}  (par_source={problem.par_source})",
        f"start  {render_expr(problem.expr)}",
        "",
    ]
    # Both columns padded to the widest entry, so the arrows line up and a
    # reader's eye can run straight down the "after" column.
    width = max((len(render_expr(s.before)) for s in steps), default=0)
    label_width = max((len(step_label(s)) for s in steps), default=0)
    lines.extend(render_step(step, before_width=width, label_width=label_width) for step in steps)
    if not steps:
        lines.append("(already in goal form — no rewrites)")

    final = steps[-1].after if steps else problem.expr
    z = result.z if result else outcome_z(solved=True, steps=len(steps), par=problem.par)
    solved = result.solved if result else True
    lines += [
        "",
        f"result {render_expr(final)}",
        f"       {'solved' if solved else 'not solved'} in {len(steps)} step(s), "
        f"par {problem.par}, z = {'+1' if z > 0 else '0' if z == 0 else '−1'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The base-625 glyph panel (independent construction — see module docstring)
# ---------------------------------------------------------------------------


def glyph_cells(digit: int) -> tuple[int, int, int, int]:
    """A base-625 digit as four base-5 cells, most significant first (625 = 5⁴)."""
    if not 0 <= digit < BASE:
        raise ValueError(f"digit {digit} out of range for base {BASE}")
    return (digit // 125, (digit // 25) % 5, (digit // 5) % 5, digit % 5)


def glyph_panel(value: int) -> str:
    """Render an integer's base-625 digits as 2×2 grids of base-5 cells."""
    digits = to_digits_of(abs(value))
    sign = MINUS if value < 0 else ""
    header = f"{sign}{abs(value)} = " + " ".join(f"D{d}" for d in digits)
    tag = (
        "glyph convention: reckoner-local placeholder, NOT base-625-canonical "
        "(vendoring registered as ROUND-03)"
    )
    tops, bottoms = [], []
    for digit in digits:
        a, b, c, d = glyph_cells(digit)
        tops.append(f"│{a} {b}│")
        bottoms.append(f"│{c} {d}│")
    rule_top = " ".join("┌───┐" for _ in digits)
    rule_bottom = " ".join("└───┘" for _ in digits)
    return "\n".join([tag, header, rule_top, " ".join(tops), " ".join(bottoms), rule_bottom])


def state_tokens_line(expr: Expr) -> str:
    """The literal token ids behind a rendered state — the audit trail."""
    return " ".join(str(t) for t in tokens(expr))
