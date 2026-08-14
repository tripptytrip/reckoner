"""The token spec: base-625 numerals, operators, structure, goals.

One module, one table, one version. Everything downstream — the parser, the
model's embedding table, every dataset on disk — is denominated in these ids, so
**an id may never move**. A dataset written under ``VOCAB_VERSION = 1`` is
readable only by a vocabulary whose ids mean the same things.

Layout, and why it is shaped this way::

    id 0                    PAD
    id 1 .. 16              structural, operator, variable and goal tokens
    id 17 .. 31             RESERVED — unused, deliberately
    id 32 .. 656            D0 .. D624, the base-625 digits

Two decisions are load-bearing:

**PAD is 0.** A zero-initialised array is therefore padding, not the digit
``D0``. The alternative — digits at 0..624 — makes an uninitialised or
short-copied row decode as the perfectly legal number 0, which is the shape of
bug that survives a whole campaign looking like data.

**The 15 reserved ids are a gap on purpose.** Digits sit at a fixed offset above
the special block, so a later chunk that needs a new control token (chunk 3's
state assembly, chunk 6's heads) appends into the gap without moving a single
digit id — and without invalidating every dataset already on disk. The cost is
15 dead embedding rows, about 4K parameters at ``d_model = 256``. That is the
cheapest insurance in the project.

Numbers are compositional, exactly as Symbolic-Transformers does it: a numeral
is a marker, a bracketed run of base-625 digits most-significant-first, and a
close. Arbitrary magnitude, finite vocabulary. The sign lives in the marker
(``NUM`` / ``NUM_NEG``) rather than in a separate token, which costs one id and
saves one position on every negative number in every sequence.
"""

from __future__ import annotations

import hashlib

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

#: Bump whenever an id changes meaning, a token is removed, or the digit offset
#: moves. Appending a token into the reserved block does NOT require a bump —
#: that is what the block is for — but it does change VOCAB_FINGERPRINT, which
#: tests/test_vocab.py pins.
VOCAB_VERSION = 1

#: The numeral base. This is the substrate, not a tuning knob (spec §2).
BASE = 625

# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

PAD = 0

LPAREN = 1
RPAREN = 2
SEP = 3

NUM = 4  # non-negative numeral marker
NUM_NEG = 5  # negative numeral marker

ADD = 6
SUB = 7
MUL = 8
DIV = 9
EQ = 10

VAR_X = 11
VAR_Y = 12
VAR_Z = 13

GOAL_EVALUATE = 14
GOAL_SOLVE = 15
GOAL_SIMPLIFY = 16

#: First id of the reserved gap; ids in [RESERVED_START, DIGIT_OFFSET) are
#: unused and are named RESERVED_n so the table stays total.
RESERVED_START = 17

#: Digit d has id ``DIGIT_OFFSET + d``. Fixed forever under VOCAB_VERSION 1.
DIGIT_OFFSET = 32

VOCAB_SIZE = DIGIT_OFFSET + BASE  # 657

# --- groupings (tuples: these are contracts, not scratch lists) -------------

STRUCTURAL_TOKENS = (PAD, LPAREN, RPAREN, SEP)
NUMERAL_MARKERS = (NUM, NUM_NEG)
OPERATOR_TOKENS = (ADD, SUB, MUL, DIV, EQ)
VARIABLE_TOKENS = (VAR_X, VAR_Y, VAR_Z)
GOAL_TOKENS = (GOAL_EVALUATE, GOAL_SOLVE, GOAL_SIMPLIFY)

#: Tokens that open a bracketed group: ``HEAD ( ... )``. Every non-atomic
#: construct in the language has this one shape (see ``expr`` for the grammar).
HEAD_TOKENS = NUMERAL_MARKERS + OPERATOR_TOKENS

#: Operator arity. ``None`` means variadic (>= 1 accepted, >= 2 after
#: canonicalisation collapses singletons).
ARITY: dict[int, int | None] = {ADD: None, MUL: None, SUB: 2, DIV: 2, EQ: 2}

#: Human glyphs, for chunk 4's interpreter and for error messages. The renderer
#: is a separate module; this is only the per-token half of the mapping.
GLYPH: dict[int, str] = {
    ADD: "+",
    SUB: "−",
    MUL: "×",
    DIV: "÷",
    EQ: "=",
    LPAREN: "(",
    RPAREN: ")",
    VAR_X: "x",
    VAR_Y: "y",
    VAR_Z: "z",
}

#: Variable token -> its name in human notation.
VAR_NAME: dict[int, str] = {VAR_X: "x", VAR_Y: "y", VAR_Z: "z"}
NAME_VAR: dict[str, int] = {v: k for k, v in VAR_NAME.items()}

_SPECIAL_NAMES: dict[int, str] = {
    PAD: "PAD",
    LPAREN: "LPAREN",
    RPAREN: "RPAREN",
    SEP: "SEP",
    NUM: "NUM",
    NUM_NEG: "NUM_NEG",
    ADD: "ADD",
    SUB: "SUB",
    MUL: "MUL",
    DIV: "DIV",
    EQ: "EQ",
    VAR_X: "VAR_X",
    VAR_Y: "VAR_Y",
    VAR_Z: "VAR_Z",
    GOAL_EVALUATE: "GOAL_EVALUATE",
    GOAL_SOLVE: "GOAL_SOLVE",
    GOAL_SIMPLIFY: "GOAL_SIMPLIFY",
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def digit_token(d: int) -> int:
    """Token id for base-625 digit ``d``."""
    if not 0 <= d < BASE:
        raise ValueError(f"digit {d} out of range for base {BASE}")
    return DIGIT_OFFSET + d


def digit_value(token: int) -> int:
    """The base-625 digit a digit token stands for."""
    if not is_digit(token):
        raise ValueError(f"token {token} ({token_name(token)}) is not a digit")
    return token - DIGIT_OFFSET


def is_digit(token: int) -> bool:
    return DIGIT_OFFSET <= token < VOCAB_SIZE


def is_reserved(token: int) -> bool:
    return RESERVED_START <= token < DIGIT_OFFSET


def token_name(token: int) -> str:
    """A readable name for any id, including out-of-range ones.

    Error messages that name the offending token are the difference between a
    five-second and a fifty-minute debugging session, so this never raises.
    """
    if token in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[token]
    if is_reserved(token):
        return f"RESERVED_{token}"
    if is_digit(token):
        return f"D{token - DIGIT_OFFSET}"
    return f"<invalid:{token}>"


def to_digits_of(value: int) -> list[int]:
    """Base-625 digits of a non-negative int, most significant first.

    A thin re-export of the numeral layout so the glyph panel does not have to
    import the expression module to answer a question about digits.
    """
    from reckoner.expr import to_digits

    return to_digits(value)


def vocab_table() -> list[tuple[int, str, str]]:
    """The whole vocabulary as ``(id, name, kind)`` rows, ascending by id.

    This is the single source for both ``docs/vocab.md`` and
    ``vocab_fingerprint()`` — the document and the digest cannot disagree with
    the table because they are the same table.
    """
    rows: list[tuple[int, str, str]] = []
    for token in range(VOCAB_SIZE):
        name = token_name(token)
        if token == PAD:
            kind = "padding"
        elif token in STRUCTURAL_TOKENS:
            kind = "structural"
        elif token in NUMERAL_MARKERS:
            kind = "numeral marker"
        elif token in OPERATOR_TOKENS:
            kind = "operator"
        elif token in VARIABLE_TOKENS:
            kind = "variable"
        elif token in GOAL_TOKENS:
            kind = "goal"
        elif is_reserved(token):
            kind = "reserved"
        else:
            kind = "digit"
        rows.append((token, name, kind))
    return rows


def vocab_fingerprint() -> str:
    """SHA-256 over the whole table. A moved id changes this; a test pins it."""
    canonical = "\n".join(f"{i}\t{name}\t{kind}" for i, name, kind in vocab_table())
    return hashlib.sha256(canonical.encode()).hexdigest()


def vocab_markdown() -> str:
    """Render ``docs/vocab.md``.

    Generated from :func:`vocab_table`, never hand-edited, and
    ``tests/test_vocab.py`` re-renders it and compares — a reference document
    that silently rots is worse than no document, because it is still believed.
    Regenerate with ``make docs``.
    """
    rows = vocab_table()
    specials = [r for r in rows if r[2] != "digit"]

    lines = [
        "# Vocabulary",
        "",
        "*Generated by `make docs` from `src/reckoner/vocab.py`. Do not hand-edit —",
        "`tests/test_vocab.py::test_vocab_md_is_current` regenerates this file and",
        "compares.*",
        "",
        f"- `VOCAB_VERSION` — **{VOCAB_VERSION}**",
        f"- `VOCAB_SIZE` — **{VOCAB_SIZE}** ids (`0 .. {VOCAB_SIZE - 1}`)",
        f"- `BASE` — **{BASE}**",
        f"- `DIGIT_OFFSET` — **{DIGIT_OFFSET}**, so digit *d* is id `{DIGIT_OFFSET} + d`",
        f"- fingerprint — `{vocab_fingerprint()}`",
        "",
        "An id may never move. A dataset written under one `VOCAB_VERSION` is",
        "readable only by a vocabulary whose ids mean the same things.",
        "",
        f"## Control block (ids 0–{DIGIT_OFFSET - 1})",
        "",
        "| id | name | kind | glyph |",
        "|---:|------|------|-------|",
    ]
    for token, name, kind in specials:
        glyph = GLYPH.get(token, "")
        glyph = f"`{glyph}`" if glyph else ""
        lines.append(f"| {token} | `{name}` | {kind} | {glyph} |")

    lines += [
        "",
        f"Ids {RESERVED_START}–{DIGIT_OFFSET - 1} are reserved and unused. They are a gap on",
        "purpose: a later chunk that needs a new control token appends into the gap",
        "without moving a single digit id, and so without invalidating datasets",
        "already on disk.",
        "",
        f"## Digit block (ids {DIGIT_OFFSET}–{VOCAB_SIZE - 1})",
        "",
        f"`D0 .. D{BASE - 1}`, the base-{BASE} digits, at `id = {DIGIT_OFFSET} + d`.",
        "Numerals are most-significant-digit first, bracketed, with the sign in the",
        "marker: `NUM ( D3 D12 )` is 3×625 + 12 = 1887, and `NUM_NEG ( D1 )` is −1.",
        "",
        "```",
    ]
    per_line = 10
    for start in range(0, BASE, per_line):
        chunk = range(start, min(start + per_line, BASE))
        lines.append("  ".join(f"D{d}={DIGIT_OFFSET + d}" for d in chunk))
    lines += ["```", ""]
    return "\n".join(lines)
