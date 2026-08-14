"""Expression trees: one parser, one printer, one canonical form.

The parser and the printer are in this module together, deliberately. They are
inverses, and inverses written in separate files drift — the round-trip gate
(200K trees, byte-exact) is only meaningful because there is exactly one place
where the surface syntax is defined.

Grammar
-------
Bracketed prefix. Every non-atomic construct has the same shape, ``HEAD ( … )``,
which is what makes the parser a single loop and the fuzz test worth running::

    expr    := numeral | variable | opnode
    numeral := (NUM | NUM_NEG) LPAREN digit+ RPAREN     -- MSB first
    variable:= VAR_X | VAR_Y | VAR_Z
    opnode  := (ADD | MUL) LPAREN expr+ RPAREN          -- variadic
             | (SUB | DIV | EQ) LPAREN expr expr RPAREN -- binary

``3x + 6 = 21`` is 22 tokens::

    EQ ( ADD ( MUL ( NUM ( D3 ) VAR_X ) NUM ( D6 ) ) NUM ( D21 ) )

Canonical form — the claims
---------------------------
Each is a claim, and each has a test named after it in ``tests/test_expr.py``:

  **C1** Numerals are base-625, most-significant digit first, with no leading
  zeros. Zero is the single digit ``D0``.

  **C2** Sign lives in the marker. ``NUM_NEG`` is never used for zero, so zero
  has exactly one spelling.

  **C3** ``ADD`` and ``MUL`` are variadic and fully flattened: no ``ADD`` is a
  direct child of an ``ADD``, no ``MUL`` a direct child of a ``MUL``.

  **C4** A variadic node with one child collapses to that child. ``ADD(x)`` is
  ``x``; there is no unary sum.

  **C5** ``ADD`` children are ordered variable-bearing terms first, then by the
  child's own canonical token sequence — ``3x + 6``, never ``6 + 3x``.

  **C6** ``MUL`` children are ordered constants first, then variable-bearing
  factors, then by token sequence — ``3x``, never ``x3``.

  **C7** ``EQ`` operands are ordered variable-bearing side first, then by token
  sequence. Equality is symmetric, so orientation carries no information, and
  ordering it makes the SOLVE goal form ``x = <number>`` canonical by
  construction rather than by convention.

  **C8** ``SUB`` and ``DIV`` are binary and their operand order is preserved.
  Neither is commutative, so ordering them would not be canonicalisation, it
  would be a wrong answer.

C5 and C6 are asymmetric on purpose: together they reproduce the ordering a
human writes, which is load-bearing for chunk 4's interpreter — the
interpretability claim of the whole program is that a person can read the
derivation, and ``6 + x3`` is not what a person reads.

C5/C6/C7 rank by *presence* of a variable, not by degree, because rule set v1 is
linear (no powers). If a later round adds exponentiation, this ordering key is
the thing that has to change, and this paragraph is the reason it will be found.

Round-trip contract
-------------------
``parse`` accepts any well-formed token sequence, canonical or not, and returns
a canonical tree; ``tokens`` prints a canonical tree. Hence:

  * ``parse(tokens(t)) == t``            for canonical ``t``  (parse ∘ print = id)
  * ``tokens(parse(s))`` is canonical    for any valid ``s``  (print ∘ parse = canon)

Anything else raises :class:`ParseError` — never ``IndexError``, never
``RecursionError``. Parser and printer are both iterative with explicit stacks,
so there is no input depth that turns a rejection into a crash.

One limit, stated rather than discovered: node ``==`` and ``hash()`` are
Python's recursive dataclass implementations, so comparing two trees of
unbounded depth can exhaust the interpreter stack even though printing and
parsing them cannot. Real expressions here are shallow — a depth-6 problem is a
handful of levels — but anything comparing trees of *unknown* depth should use
:func:`identity_key`, which is iterative and is the project's comparison of
record anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from reckoner.vocab import (
    ADD,
    BASE,
    DIGIT_OFFSET,
    DIV,
    EQ,
    HEAD_TOKENS,
    LPAREN,
    MUL,
    NUM,
    NUM_NEG,
    RPAREN,
    SUB,
    VARIABLE_TOKENS,
    VOCAB_SIZE,
    is_digit,
    token_name,
)


class ParseError(ValueError):
    """A token sequence is not a well-formed expression.

    The only exception ``parse`` raises. The fuzz gate asserts exactly this:
    malformed input is *rejected*, not survived, not crashed on.
    """


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
#
# Frozen and hashable: canonical trees are used as dict keys and set members by
# the dedup and contamination machinery downstream, and a mutable node would
# make those keys lie.


@dataclass(frozen=True, slots=True)
class Num:
    """An integer literal. Arbitrary magnitude; the encoding is base-625."""

    value: int


@dataclass(frozen=True, slots=True)
class Var:
    """A variable, identified by its vocabulary token (``VAR_X`` …)."""

    token: int


@dataclass(frozen=True, slots=True)
class Op:
    """An operator node. ``kind`` is the operator's vocabulary token."""

    kind: int
    children: tuple[Expr, ...]


Expr = Union[Num, Var, "Op"]

_VARIADIC = (ADD, MUL)
_BINARY = (SUB, DIV, EQ)


# ---------------------------------------------------------------------------
# Predicates and keys
# ---------------------------------------------------------------------------


def has_var(expr: Expr) -> bool:
    """True if any leaf below ``expr`` is a variable. Iterative: no depth limit."""
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Var):
            return True
        if isinstance(node, Op):
            stack.extend(node.children)
    return False


def identity_key(expr: Expr) -> tuple[int, ...]:
    """**The** identity normalizer for expressions.

    Inherited law: one shared identity normalizer for any dedup key. Every
    place that needs to ask "is this the same expression?" — replay dedup,
    suite contamination tests, the openings-book analog, par lookup — asks it
    here and nowhere else. Two expressions are the same iff their canonical
    token sequences are equal, which is exactly what canonical form is for.

    Deliberately not ``hash()``: a hash collides, and a contamination test that
    silently tolerates collisions is a contamination test that passes.
    """
    return tokens(expr)


# ---------------------------------------------------------------------------
# Construction — the canonical form lives here
# ---------------------------------------------------------------------------


def _sort_key(child: Expr, *, vars_first: bool) -> tuple[bool, tuple[int, ...]]:
    contains = has_var(child)
    primary = not contains if vars_first else contains
    return (primary, tokens(child))


def make_op(kind: int, children: tuple[Expr, ...] | list[Expr]) -> Expr:
    """Build a canonical operator node from **already-canonical** children.

    Bottom-up constructor: it enforces C3–C8 at this node and assumes the
    children below it are already canonical. The parser builds bottom-up so this
    always holds there; a caller rewriting an existing tree (chunk 2's rule
    engine) either rebuilds bottom-up too, or calls :func:`canonicalize`.

    Raises ``ValueError`` on an arity that cannot exist — that is a caller bug,
    not malformed input, so it is not a ``ParseError``.
    """
    kids = tuple(children)
    if kind in _VARIADIC:
        flat: list[Expr] = []
        for child in kids:  # C3: flatten same-kind children
            if isinstance(child, Op) and child.kind == kind:
                flat.extend(child.children)
            else:
                flat.append(child)
        if not flat:
            raise ValueError(f"{token_name(kind)} needs at least one child")
        if len(flat) == 1:  # C4: no unary sum or product
            return flat[0]
        vars_first = kind == ADD  # C5 / C6
        flat.sort(key=lambda c: _sort_key(c, vars_first=vars_first))
        return Op(kind, tuple(flat))

    if kind in _BINARY:
        if len(kids) != 2:
            raise ValueError(f"{token_name(kind)} takes exactly 2 children, got {len(kids)}")
        if kind == EQ:  # C7: equality is symmetric, so order it
            left, right = sorted(kids, key=lambda c: _sort_key(c, vars_first=True))
            return Op(EQ, (left, right))
        return Op(kind, kids)  # C8: SUB and DIV keep their order

    raise ValueError(f"{token_name(kind)} is not an operator")


def canonicalize(expr: Expr) -> Expr:
    """Rebuild ``expr`` bottom-up into canonical form. Idempotent.

    ``canonicalize(canonicalize(e)) == canonicalize(e)``, and for anything
    ``parse`` produced it is already the identity. Iterative post-order, so a
    deep tree cannot blow the stack.
    """
    #: work stack of (node, expanded?) — expanded nodes have their children on
    #: `done` and are ready to be rebuilt.
    stack: list[tuple[Expr, bool]] = [(expr, False)]
    done: list[Expr] = []
    while stack:
        node, expanded = stack.pop()
        if isinstance(node, Op):
            if not expanded:
                stack.append((node, True))
                # Reversed: the stack is LIFO, so pushing children forwards pops
                # them backwards and `done` collects them in reverse. ADD, MUL
                # and EQ re-sort and would hide it; SUB and DIV do not, and a
                # silently flipped divisor is a wrong answer that type-checks.
                for child in reversed(node.children):
                    stack.append((child, False))
            else:
                n = len(node.children)
                kids = done[len(done) - n :]
                del done[len(done) - n :]
                done.append(make_op(node.kind, kids))
        else:
            done.append(node)
    return done[0]


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------


def to_digits(value: int) -> list[int]:
    """Base-625 digits of a non-negative int, most significant first (C1)."""
    if value < 0:
        raise ValueError(f"to_digits expects a non-negative value, got {value}")
    if value == 0:
        return [0]
    digits: list[int] = []
    while value:
        value, rem = divmod(value, BASE)
        digits.append(rem)
    digits.reverse()
    return digits


def from_digits(digits: list[int]) -> int:
    """Fold base-625 digits (most significant first) into an int."""
    value = 0
    for digit in digits:
        value = value * BASE + digit
    return value


def tokens(expr: Expr) -> tuple[int, ...]:
    """Print a canonical tree to its canonical token sequence.

    Iterative. The stack holds either subtrees still to print or bare token ids
    already scheduled to be emitted (the close parens), which is what lets one
    loop handle arbitrary nesting.
    """
    out: list[int] = []
    stack: list[Expr | int] = [expr]
    while stack:
        item = stack.pop()
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, Num):
            marker = NUM_NEG if item.value < 0 else NUM  # C2
            out.append(marker)
            out.append(LPAREN)
            for digit in to_digits(abs(item.value)):
                out.append(DIGIT_OFFSET + digit)
            out.append(RPAREN)
        elif isinstance(item, Var):
            out.append(item.token)
        elif isinstance(item, Op):
            out.append(item.kind)
            out.append(LPAREN)
            stack.append(RPAREN)
            stack.extend(reversed(item.children))
        else:
            raise TypeError(f"not an expression node: {item!r}")
    return tuple(out)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Frame:
    """One open ``HEAD ( … )`` group."""

    __slots__ = ("head", "digits", "items")

    def __init__(self, head: int) -> None:
        self.head = head
        self.digits: list[int] = []
        self.items: list[Expr] = []

    @property
    def is_numeral(self) -> bool:
        return self.head in (NUM, NUM_NEG)

    def build(self, position: int) -> Expr:
        if self.is_numeral:
            if not self.digits:
                raise ParseError(
                    f"empty numeral at token {position}: {token_name(self.head)} ( ) has no digits"
                )
            value = from_digits(self.digits)
            # C1/C2 are enforced by *construction*, not by rejection: leading
            # zeros and a negative zero are legal spellings of a legal value, so
            # they parse and then print canonically. Rejecting them would make
            # `print ∘ parse = canonical` a narrower claim than it should be.
            return Num(-value if self.head == NUM_NEG else value)

        if not self.items:
            raise ParseError(
                f"empty group at token {position}: {token_name(self.head)} ( ) has no operands"
            )
        expected = 2 if self.head in _BINARY else None
        if expected is not None and len(self.items) != expected:
            raise ParseError(
                f"{token_name(self.head)} takes exactly {expected} operands, "
                f"got {len(self.items)} (group closing at token {position})"
            )
        return make_op(self.head, self.items)


def parse(seq: tuple[int, ...] | list[int]) -> Expr:
    """Parse a token sequence into a canonical tree.

    Accepts non-canonical but well-formed input (nested sums, unsorted terms,
    leading zeros) and returns the canonical tree for it. Raises
    :class:`ParseError` — and only :class:`ParseError` — on anything else.

    Padding is the caller's business: ``PAD`` inside an expression is an error,
    because a parser that silently skips it cannot tell a padded sequence from a
    corrupted one.
    """
    stack: list[_Frame] = []
    pending_head: int | None = None
    result: Expr | None = None

    for position, token in enumerate(seq):
        if not isinstance(token, int) or isinstance(token, bool):
            raise ParseError(f"token {position} is not an int: {token!r}")
        if not 0 <= token < VOCAB_SIZE:
            raise ParseError(f"token {position} is out of vocabulary range: {token}")

        if pending_head is not None:
            if token != LPAREN:
                raise ParseError(
                    f"expected '(' after {token_name(pending_head)} at token {position}, "
                    f"got {token_name(token)}"
                )
            stack.append(_Frame(pending_head))
            pending_head = None
            continue

        if token in HEAD_TOKENS:
            pending_head = token
            continue

        if token == RPAREN:
            if not stack:
                raise ParseError(f"unmatched ')' at token {position}")
            frame = stack.pop()
            node = frame.build(position)
        elif token in VARIABLE_TOKENS:
            node = Var(token)
        elif is_digit(token):
            if not stack or not stack[-1].is_numeral:
                raise ParseError(
                    f"digit {token_name(token)} at token {position} is outside a numeral"
                )
            stack[-1].digits.append(token - DIGIT_OFFSET)
            continue
        else:
            raise ParseError(
                f"{token_name(token)} at token {position} is not part of the expression grammar"
            )

        # Attach the completed node to its parent, or accept it as the result.
        if stack:
            parent = stack[-1]
            if parent.is_numeral:
                raise ParseError(
                    f"expression inside a numeral at token {position}: "
                    f"{token_name(parent.head)} accepts digits only"
                )
            parent.items.append(node)
        else:
            if result is not None:
                raise ParseError(
                    f"more than one top-level expression (second starts before token {position})"
                )
            result = node

    if pending_head is not None:
        raise ParseError(f"input ends after {token_name(pending_head)}, expected '('")
    if stack:
        raise ParseError(
            f"{len(stack)} unclosed group(s) at end of input: "
            + ", ".join(token_name(f.head) for f in stack)
        )
    if result is None:
        raise ParseError("empty token sequence")
    return result


# ---------------------------------------------------------------------------
# Small constructors, for tests and for chunk 2's rule templates
# ---------------------------------------------------------------------------


def num(value: int) -> Num:
    return Num(value)


def var(token: int) -> Var:
    if token not in VARIABLE_TOKENS:
        raise ValueError(f"{token_name(token)} is not a variable token")
    return Var(token)


def add(*children: Expr) -> Expr:
    return make_op(ADD, children)


def mul(*children: Expr) -> Expr:
    return make_op(MUL, children)


def sub(left: Expr, right: Expr) -> Expr:
    return make_op(SUB, (left, right))


def div(left: Expr, right: Expr) -> Expr:
    return make_op(DIV, (left, right))


def eq(left: Expr, right: Expr) -> Expr:
    return make_op(EQ, (left, right))
