"""The ladder: paired problem sets, `pair_scores`, the bootstrap, the self-match null.

Paired, because unpaired is a worse instrument for free
--------------------------------------------------------
Every arm sees the **same problems**. The comparison is then a paired difference
per problem rather than a difference of two averages over different draws, which
removes problem difficulty from the variance entirely. Two arms scored on
different samples differ partly because one got easier problems, and no amount of
sample size fixes that — it makes the wrong estimate more precise.

`pair_scores` is persisted **from row one**, not derived at analysis time.
Inherited law, and the reason is that a per-pair record can be re-analysed while
an aggregate cannot be un-aggregated: a bootstrap needs the pairs, and a pass
that stored only means has thrown away the only thing the test of record consumes.

The self-match null, in race form
----------------------------------
The strongest no-hidden-state detector available here. Under the **eval profile**
the model is deterministic, so running it twice on the paired set must produce
**identical z vectors**, and therefore every paired difference is **exactly 0** —
not "small", exactly 0. Any drift means state is leaking between episodes that
should be independent.

Its contrast case is what makes it non-vacuous: under the **self-play profile**
(root noise on) the same comparison must NOT be identical. A null that reports
zero for every configuration is measuring nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import fmean

import numpy as np

from reckoner.config import Config
from reckoner.dataset import problem_key
from reckoner.episode import Problem, outcome_z
from reckoner.logschema import CURRENCY_BUDGET, CURRENCY_Z


class LadderError(ValueError):
    """A comparison that cannot be made honestly."""


@dataclass
class PairScore:
    """One arm's outcome on one problem, kept per-pair from row one."""

    problem_key: str
    arm: str
    currency: str
    #: z for rule-denominated arms; solved-as-0/1 for external ones. The field is
    #: deliberately generic and the CURRENCY says how to read it — the alternative
    #: is two columns where one is always absent, which is a null by another name.
    score: float
    steps: int
    seed: int


@dataclass
class PairedComparison:
    """Two arms over the same problems, and the differences that follow."""

    arm_a: str
    arm_b: str
    currency: str
    differences: list[float] = field(default_factory=list)
    problem_keys: list[str] = field(default_factory=list)

    def mean(self) -> float:
        return fmean(self.differences) if self.differences else 0.0


def problem_key_of(problem: Problem) -> str:
    """The paired set's identity, through **the** shared normalizer.

    Delegates to :func:`reckoner.dataset.problem_key` — the project's one dedup
    and contamination key — and only renders it as a string for the JSONL column.

    The first version of this built its own key from ``(identity_key(expr), goal)``,
    which is the *census* key and is deliberately looser: canonicalisation makes
    ``3x + 6 = 21`` and ``6 + 3x = 21`` a single identity, so two distinct rows of
    a paired set could share one key and :func:`pair` would match a score against
    the wrong partner without saying so. FINDINGS.md F-17.
    """
    return ",".join(str(t) for t in problem_key(problem))


def pair(scores_a: list[PairScore], scores_b: list[PairScore]) -> PairedComparison:
    """Difference per problem. **Refuses to pair across currencies.**

    A z minus a solve-rate is a number with no units, and it is exactly the
    number the currency ruling exists to make unconstructable.
    """
    if not scores_a or not scores_b:
        raise LadderError("cannot pair an empty arm")
    currencies = {s.currency for s in scores_a} | {s.currency for s in scores_b}
    if len(currencies) != 1:
        raise LadderError(
            f"cannot pair across currencies {sorted(currencies)}: sympy's steps are "
            "not our steps, and their difference has no units. Compare within a "
            "currency, or compare the arms on a statistic both can carry."
        )
    for side in (scores_a, scores_b):
        keys = [s.problem_key for s in side]
        if len(set(keys)) != len(keys):
            raise LadderError(
                f"{side[0].arm} scored {len(keys) - len(set(keys))} problem(s) more "
                "than once. A duplicate key makes pairing choose a partner by write "
                "order while the count check still balances — a silent mis-pairing, "
                "which is the failure mode this key's strictness exists to prevent."
            )
    by_key_b = {s.problem_key: s for s in scores_b}
    missing = [s.problem_key for s in scores_a if s.problem_key not in by_key_b]
    if missing:
        raise LadderError(
            f"{len(missing)} problems scored by {scores_a[0].arm} and not by "
            f"{scores_b[0].arm}. A 'paired' comparison over a partial overlap is an "
            "unpaired comparison with a paired name — the arms would be scored on "
            "different problem sets."
        )
    comparison = PairedComparison(scores_a[0].arm, scores_b[0].arm, currencies.pop())
    for s in scores_a:
        comparison.problem_keys.append(s.problem_key)
        comparison.differences.append(s.score - by_key_b[s.problem_key].score)
    return comparison


def paired_bootstrap(
    differences: list[float], *, resamples: int, seed: int = 0, alpha: float = 0.05
) -> dict:
    """**The test of record** for any pass-vs-pass claim (inherited law).

    Resamples the *paired differences*, not the arms separately — which is the
    whole point: the pairing is what removed problem difficulty from the
    variance, and resampling the arms independently would put it back.

    Reports whether the interval excludes zero, and **says when it is saturated**:
    an interval whose bounds are both at the extreme of what the statistic can
    take is not a tight estimate, it is a statistic that has run out of range, and
    rendering it as a narrow CI would read as precision.
    """
    if not differences:
        raise LadderError("no paired differences — nothing to resample")
    rng = np.random.default_rng(seed)
    array = np.asarray(differences, dtype=np.float64)
    n = len(array)
    means = np.array(
        [array[rng.integers(0, n, n)].mean() for _ in range(resamples)], dtype=np.float64
    )
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    observed = float(array.mean())
    saturated = bool(np.all(array == array[0]))
    return {
        "n_pairs": n,
        "mean_difference": round(observed, 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "alpha": alpha,
        "resamples": resamples,
        "excludes_zero": bool(low > 0 or high < 0),
        "saturated": saturated,
        "rendering_note": (
            "SATURATED: every paired difference is identical, so the interval has "
            "zero width by construction and is not evidence of precision"
            if saturated
            else "ordinary percentile interval over resampled paired differences"
        ),
    }


def self_match(
    play,
    problems: list[Problem],
    cfg: Config,
    *,
    profile: str,
    seed: int = 0,
) -> PairedComparison:
    """The model against itself over the paired set, **scored in z**.

    ``play(problem, cfg, seed)`` returns anything with ``.solved`` and ``.steps``
    — an :class:`reckoner.arms.ArmResult`. **z is computed here, from the
    problem's own par**, and the caller does not get to supply a score.

    That is deliberate and it is a fix, not a convenience. The first version took
    ``(score, steps)`` from the caller, and the smoke pass duly handed it
    ``solved * 2 - 1`` — a solved-flat score, wearing ``CURRENCY_Z``, standing as
    the null for a z-denominated comparison. F-13's exact shape: the null and the
    metric it is a null for, denominated differently, with nothing able to tell
    because both land in {-1, 0, +1}. No validation can catch that downstream, so
    the choice is removed from the caller instead. See FINDINGS.md F-19.

    Under the **eval** profile the two passes must be identical and every
    difference exactly 0; under **self_play** they must not be, or the detector
    is vacuous.
    """
    if profile not in ("eval", "self_play"):
        raise LadderError(f"unknown profile {profile!r}")
    left, right = [], []
    for i, problem in enumerate(problems):
        key = problem_key_of(problem)
        if problem.par is None:
            raise LadderError(
                "a self-match null needs par to score z, and this problem has none. "
                "Absence does not ship, and it does not become a zero here either."
            )
        # Under the eval profile BOTH passes get the same derived seed, because
        # the claim is that the model is a function of the state. Under self-play
        # they differ, because the claim there is that it is not.
        seed_a = seed * 1_000_003 + i
        seed_b = seed_a if profile == "eval" else seed_a + 500_009
        a = play(problem, cfg, seed_a)
        b = play(problem, cfg, seed_b)
        za = outcome_z(solved=a.solved, steps=a.steps, par=problem.par)
        zb = outcome_z(solved=b.solved, steps=b.steps, par=problem.par)
        left.append(PairScore(key, "self_a", CURRENCY_Z, float(za), a.steps, seed_a))
        right.append(PairScore(key, "self_b", CURRENCY_Z, float(zb), b.steps, seed_b))
    return pair(left, right)


def synthetic_elo(differences: list[float]) -> float:
    """A crude score in [0, 1]: the fraction of pairs arm A did better on, ties half.

    Deliberately simple and deliberately named *synthetic*: it exists so the
    ladder's arithmetic has a rigged-50% null to be tested against, not so a
    campaign can quote an Elo it did not earn.
    """
    if not differences:
        raise LadderError("no differences to score")
    wins = sum(1 for d in differences if d > 0)
    ties = sum(1 for d in differences if d == 0)
    return (wins + 0.5 * ties) / len(differences)


def rigged_null(n: int, seed: int = 0) -> list[float]:
    """Differences with no signal: exactly balanced wins and losses.

    The null the synthetic-Elo test must score at 0.5 — if it does not, the
    scoring arithmetic has a bias and every later number inherits it.
    """
    rng = random.Random(seed)
    values = [1.0] * (n // 2) + [-1.0] * (n // 2) + ([0.0] if n % 2 else [])
    rng.shuffle(values)
    return values


__all__ = [
    "CURRENCY_BUDGET",
    "CURRENCY_Z",
    "LadderError",
    "PairScore",
    "PairedComparison",
    "pair",
    "paired_bootstrap",
    "problem_key_of",
    "rigged_null",
    "self_match",
    "synthetic_elo",
]
