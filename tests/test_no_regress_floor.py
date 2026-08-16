"""The no-regress floor construction, pinned to reference vectors.

Written because the convention was verified once by agreement and not at all by
execution. Two of the record's own rulings composed into a defect at their seam:
`P11B-A5` migrated the selection arithmetic to `Decimal`, and a ceiling written
as ``-((-x) // 1)`` — correct for ``int`` and ``float`` — silently computes
**floor** for ``Decimal``, because ``Decimal.__floordiv__`` truncates toward zero
rather than flooring. It produced 1187 and 1166 where the declared construction
gives 1188 and 1167.

It was caught by exactly one mechanism: the numbers had to agree with an
independent derivation. No test existed, and review would have read `ceiling` in
the name and moved on. So the vectors are pinned here, and the convention is
executable-verified forever rather than agreement-verified once.

`PREREG-m1.md` §4.2 and §4.3 carry these derivations to ten places. If this file
and that page ever disagree, one of them is wrong and the disagreement is the
finding.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from reckoner.gates import Z_95, ceil_count, no_regress_floor, one_sided_lower_bound

#: The two floors PREREG-m1 froze, as (successes, trials, bound_to_10dp, floor).
REFERENCE_VECTORS = [
    pytest.param(1193, 1200, "1187.8294744303", 1188, id="sims-48-anchor-1193"),
    pytest.param(1176, 1200, "1166.4945051681", 1167, id="sims-1-anchor-1176"),
]


@pytest.mark.parametrize(("successes", "trials", "bound", "floor"), REFERENCE_VECTORS)
def test_the_frozen_floors_reproduce(successes: int, trials: int, bound: str, floor: int) -> None:
    """Both rows of PREREG-m1 §4, bound and integerization, to ten places."""
    computed = one_sided_lower_bound(successes, trials)
    assert computed.quantize(Decimal("0.0000000001")) == Decimal(bound)
    assert no_regress_floor(successes, trials) == floor


def test_an_exact_integer_is_its_own_ceiling() -> None:
    """The boundary every hand-rolled ceiling gets wrong.

    A floor that lands exactly on an integer must not be pushed to the next one:
    the gate is ``count >= b``, and at ``b = 1188`` a count of 1188 satisfies it.
    """
    assert ceil_count(Decimal("1188")) == 1188
    assert ceil_count(Decimal("1188.0000000000")) == 1188
    assert ceil_count(Decimal("0")) == 0


def test_ceiling_never_admits_a_count_the_bound_excludes() -> None:
    """The property the convention exists for, asserted as the property.

    Round-half-up would return 1166 against a bound of 1166.4945 — a count the
    declared construction rejects. Ceiling cannot, for any fractional bound.
    """
    for bound in ("1166.4945051681", "1187.8294744303", "0.0000000001", "1199.5"):
        b = Decimal(bound)
        assert ceil_count(b) >= b


def test_the_ceiling_is_not_secretly_floor() -> None:
    """The exact defect that reached the page: Decimal // truncates toward zero.

    ``-((-x) // 1)`` reads as a ceiling and is one for int and float. For Decimal
    it is floor, and it is off by one on precisely the values that matter.
    """
    x = Decimal("1166.4945051681")
    assert ceil_count(x) == 1167
    assert int(-((-x) // 1)) == 1166  # the trap, pinned so nobody re-enters it


def test_a_lower_bound_is_below_the_point_estimate() -> None:
    """Sanity in the direction of the construction, both rows."""
    for successes, trials, _bound, _floor in [p.values for p in REFERENCE_VECTORS]:
        assert one_sided_lower_bound(successes, trials) < Decimal(successes)


def test_a_wider_z_gives_a_lower_floor() -> None:
    """Monotone in the band's width, so the 1.96 is doing visible work."""
    tight = no_regress_floor(1176, 1200, z=Decimal("1.0"))
    declared = no_regress_floor(1176, 1200, z=Z_95)
    wide = no_regress_floor(1176, 1200, z=Decimal("3.0"))
    assert wide < declared < tight
