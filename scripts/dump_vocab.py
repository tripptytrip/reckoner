"""Write ``docs/vocab.md`` from the vocabulary table.

The document is generated, not maintained. ``tests/test_vocab.py`` re-renders it
and compares against the file on disk, so a vocabulary change that forgets this
script fails the build rather than leaving a plausible, wrong reference behind.

    make docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reckoner.vocab import VOCAB_VERSION, vocab_fingerprint, vocab_markdown

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "docs" / "vocab.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the file on disk differs, and write nothing",
    )
    args = parser.parse_args()

    rendered = vocab_markdown()

    if args.check:
        current = args.out.read_text() if args.out.exists() else ""
        if current != rendered:
            print(f"{args.out} is STALE — run `make docs`")
            return 1
        print(f"{args.out} is current")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(f"wrote {args.out} (VOCAB_VERSION {VOCAB_VERSION}, {vocab_fingerprint()[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
