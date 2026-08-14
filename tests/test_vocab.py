"""The vocabulary is a contract with every dataset ever written under it.

An id that moves does not break loudly — it decodes the old data as something
else. So the layout is pinned by value, and the whole table by digest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reckoner import vocab
from reckoner.vocab import (
    BASE,
    DIGIT_OFFSET,
    GOAL_TOKENS,
    HEAD_TOKENS,
    NUMERAL_MARKERS,
    OPERATOR_TOKENS,
    PAD,
    RESERVED_START,
    STRUCTURAL_TOKENS,
    VARIABLE_TOKENS,
    VOCAB_SIZE,
    VOCAB_VERSION,
    digit_token,
    digit_value,
    is_digit,
    is_reserved,
    token_name,
    vocab_fingerprint,
    vocab_markdown,
    vocab_table,
)

REPO = Path(__file__).resolve().parents[1]
VOCAB_MD = REPO / "docs" / "vocab.md"

# The whole table, digested. Changing the vocabulary deliberately means bumping
# VOCAB_VERSION *and* updating this literal — the two-step is the point. Every
# dataset, checkpoint and suite on disk is denominated in these ids.
PINNED_FINGERPRINT = "3e61743e8d209ddc9e1e65be91d6711eb71a8f267dbbfe42982c22f231eedbd0"


def test_vocab_version_is_one() -> None:
    assert VOCAB_VERSION == 1


def test_size_is_in_the_spec_envelope() -> None:
    """Spec §2: "~650-700 symbols total"."""
    assert VOCAB_SIZE == DIGIT_OFFSET + BASE == 657
    assert 650 <= VOCAB_SIZE <= 700


def test_pad_is_zero() -> None:
    """A zero-initialised array must be padding, not the number 0."""
    assert PAD == 0
    assert not is_digit(PAD)


def test_digit_ids_are_pinned() -> None:
    assert digit_token(0) == 32
    assert digit_token(624) == 656
    assert digit_token(1) == 33


@pytest.mark.parametrize("d", [0, 1, 42, 624])
def test_digit_round_trip(d: int) -> None:
    assert digit_value(digit_token(d)) == d


@pytest.mark.parametrize("d", [-1, BASE, BASE + 1])
def test_out_of_range_digit_rejected(d: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        digit_token(d)


def test_digit_value_rejects_non_digits() -> None:
    with pytest.raises(ValueError, match="is not a digit"):
        digit_value(PAD)


def test_reserved_gap_is_where_it_says() -> None:
    """The gap is the whole reason digit ids can be promised to stay put."""
    assert RESERVED_START == 17
    assert [t for t in range(VOCAB_SIZE) if is_reserved(t)] == list(range(17, 32))


def test_every_id_has_exactly_one_name_and_kind() -> None:
    rows = vocab_table()
    assert len(rows) == VOCAB_SIZE
    assert [r[0] for r in rows] == list(range(VOCAB_SIZE))
    names = [r[1] for r in rows]
    assert len(set(names)) == VOCAB_SIZE, "duplicate token names"
    assert all(not n.startswith("<invalid") for n in names)


def test_groupings_are_disjoint_and_in_range() -> None:
    groups = [STRUCTURAL_TOKENS, NUMERAL_MARKERS, OPERATOR_TOKENS, VARIABLE_TOKENS, GOAL_TOKENS]
    seen: set[int] = set()
    for group in groups:
        assert not seen & set(group), "a token appears in two groupings"
        seen |= set(group)
    assert all(0 <= t < DIGIT_OFFSET for t in seen)
    assert not any(is_reserved(t) for t in seen), "a live token sits in the reserved gap"


def test_head_tokens_are_markers_plus_operators() -> None:
    assert set(HEAD_TOKENS) == set(NUMERAL_MARKERS) | set(OPERATOR_TOKENS)


def test_token_name_never_raises() -> None:
    """Error paths call this; a namer that raises turns a message into a crash."""
    for token in (-1, 0, 17, 656, VOCAB_SIZE, 10**9):
        assert isinstance(token_name(token), str)
    assert token_name(-1) == "<invalid:-1>"
    assert token_name(17) == "RESERVED_17"
    assert token_name(32) == "D0"


def test_fingerprint_is_pinned() -> None:
    assert vocab_fingerprint() == PINNED_FINGERPRINT, (
        "The vocabulary table changed. If that was deliberate: bump VOCAB_VERSION "
        "and update PINNED_FINGERPRINT here. Every dataset, suite and checkpoint "
        "written under the old table decodes to different symbols now."
    )


def test_fingerprint_moves_when_the_table_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both polarities: a digest that never changes is not a digest."""
    before = vocab_fingerprint()
    monkeypatch.setitem(vocab._SPECIAL_NAMES, PAD, "PADDING")
    assert vocab_fingerprint() != before


def test_vocab_md_exists() -> None:
    assert VOCAB_MD.exists(), "docs/vocab.md is a chunk 1 deliverable — run `make docs`"


def test_vocab_md_is_current() -> None:
    """The document is generated; a stale one is worse than none, being believed."""
    assert VOCAB_MD.read_text() == vocab_markdown(), (
        "docs/vocab.md is stale — run `make docs` and commit the result."
    )


def test_vocab_md_carries_its_own_provenance() -> None:
    text = VOCAB_MD.read_text()
    assert vocab_fingerprint() in text
    assert f"**{VOCAB_SIZE}**" in text
