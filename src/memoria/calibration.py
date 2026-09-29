"""Confidence calibration, recalibration and selective prediction (Phases 11-12).

A belief's confidence is a *score* (:mod:`memoria.beliefs`) until it is shown to behave like
a probability. This module tests exactly that against ground-truth outcomes:

- :func:`reliability`: equal-width reliability diagram with Wilson intervals per bin,
  expected and maximum calibration error, Brier score;
- :func:`log_loss` is defined only for values declared probabilistic (a fitted
  :class:`Isotonic` recalibration), never for a raw score;
- :func:`fit_isotonic`: pool-adjacent-violators regression of outcome on score, fitted on a
  calibration split and evaluated on a disjoint test split;
- :func:`risk_coverage`, :func:`selective_accuracy_at`: selective prediction;
- :func:`auroc`, :func:`abstention_quality`, :func:`by_subgroup`.

Everything is closed form and deterministic (no resampling).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from memoria.core import Digest, Record, content_hash, quantize
from memoria.statistics import Proportion

Pair = tuple[float, bool]  # (score or probability, outcome)


class Prediction(Record):
    """One probe's decision and its outcome against ground truth."""

    probe: str
    key: str
    mode: Literal["abstain", "answer", "answer_with_uncertainty", "competing", "require_evidence"]
    value: str | None  # the answered value (None unless answered)
    score: float | None
    calibrated: float | None
    correct: bool | None  # the answered value equals the truth (None: not answered)
    would_be_correct: bool | None  # the best-supported value equals the truth (None: no belief)
    candidate_hit: bool | None  # competing: the truth is among the presented values
    tags: tuple[str, ...] = ()
    uncertainty: tuple[tuple[str, float], ...] = ()  # the top belief's decomposition

    @property
    def answered(self) -> bool:
        return self.mode in ("answer", "answer_with_uncertainty")


class Bin(Record):
    lo: float
    hi: float
    n: int = Field(ge=0)
    mean_score: float | None
    accuracy: Proportion | None
    gap: float | None  # |accuracy - mean_score|


class Reliability(Record):
    n: int = Field(ge=0)
    bins: tuple[Bin, ...]
    ece: float | None
    mce: float | None
    brier: float | None
    mean_score: float | None
    accuracy: float | None
    overconfidence: float | None  # mean_score - accuracy: positive = overconfident


def brier(pairs: Sequence[Pair]) -> float | None:
    if not pairs:
        return None
    return quantize(math.fsum((p - float(y)) ** 2 for p, y in pairs) / len(pairs))


def log_loss(pairs: Sequence[Pair], *, probabilistic: bool, eps: float = 1e-6) -> float | None:
    """Mean negative log-likelihood. Refused for a raw score: only values declared to be
    probabilities (a fitted recalibration) have likelihood semantics."""
    if not probabilistic:
        raise ValueError("log loss needs probabilistic semantics; a raw score has none")
    if not pairs:
        return None
    total = 0.0
    for p, y in pairs:
        q = min(max(p, eps), 1 - eps)
        total += -math.log(q if y else 1 - q)
    return quantize(total / len(pairs))


def reliability(pairs: Sequence[Pair], bins: int = 10, confidence: float = 0.95) -> Reliability:
    """Equal-width reliability diagram over [0, 1]; the last bin includes 1."""
    if bins < 1:
        raise ValueError("bins must be positive")
    buckets: list[list[Pair]] = [[] for _ in range(bins)]
    for p, y in pairs:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"score {p} outside [0, 1]")
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    n = len(pairs)
    out: list[Bin] = []
    ece = mce = 0.0
    for i, b in enumerate(buckets):
        lo, hi = i / bins, (i + 1) / bins
        if not b:
            out.append(Bin(lo=lo, hi=hi, n=0, mean_score=None, accuracy=None, gap=None))
            continue
        mean = math.fsum(p for p, _ in b) / len(b)
        acc = Proportion.of(sum(y for _, y in b), len(b), confidence)
        assert acc.estimate is not None
        gap = abs(acc.estimate - mean)
        ece += len(b) / n * gap
        mce = max(mce, gap)
        out.append(
            Bin(lo=lo, hi=hi, n=len(b), mean_score=quantize(mean), accuracy=acc, gap=quantize(gap))
        )
    if not n:
        return Reliability(
            n=0,
            bins=tuple(out),
            ece=None,
            mce=None,
            brier=None,
            mean_score=None,
            accuracy=None,
            overconfidence=None,
        )
    mean_score = math.fsum(p for p, _ in pairs) / n
    accuracy = sum(y for _, y in pairs) / n
    return Reliability(
        n=n,
        bins=tuple(out),
        ece=quantize(ece),
        mce=quantize(mce),
        brier=brier(pairs),
        mean_score=quantize(mean_score),
        accuracy=quantize(accuracy),
        overconfidence=quantize(mean_score - accuracy),
    )


class Isotonic(Record):
    """A monotone step function score -> probability, fitted by pool-adjacent-violators.

    ``uppers[i]`` is the largest fitted score of block i; a score maps to the first block
    whose upper bound is at least the score (the last block above all of them). ``fit``
    hashes the calibration data, so the recalibration's provenance is auditable.
    """

    uppers: tuple[float, ...]
    values: tuple[float, ...]
    n: int = Field(ge=0)
    fit: Digest | None

    @model_validator(mode="after")
    def _check_invariants(self) -> Isotonic:
        if len(self.uppers) != len(self.values):
            raise ValueError("one value per block")
        if list(self.values) != sorted(self.values) or list(self.uppers) != sorted(self.uppers):
            raise ValueError("blocks must be monotone")
        return self

    def __call__(self, score: float) -> float:
        if not self.values:
            return score
        lo, hi = 0, len(self.uppers) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.uppers[mid] >= score:
                hi = mid
            else:
                lo = mid + 1
        return self.values[lo]


def fit_isotonic(pairs: Sequence[Pair]) -> Isotonic:
    """Pool-adjacent-violators isotonic regression of the outcome on the score."""
    data = sorted(pairs, key=lambda x: x[0])
    blocks: list[list[float]] = []  # [sum of y, weight, upper score]
    for score, y in data:
        if blocks and blocks[-1][2] == score:
            blocks[-1][0] += float(y)
            blocks[-1][1] += 1
        else:
            blocks.append([float(y), 1.0, score])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1] = [blocks[-1][0] + top[0], blocks[-1][1] + top[1], top[2]]
    return Isotonic(
        uppers=tuple(quantize(b[2]) for b in blocks),
        values=tuple(quantize(b[0] / b[1]) for b in blocks),
        n=len(data),
        fit=content_hash(sorted((round(p, 9), y) for p, y in pairs)) if data else None,
    )


def calibrated_pairs(pairs: Sequence[Pair], model: Isotonic) -> list[Pair]:
    return [(model(p), y) for p, y in pairs]


class RiskCoveragePoint(Record):
    threshold: float
    coverage: float
    selective_accuracy: float
    answered: int


class RiskCoverage(Record):
    n: int
    points: tuple[RiskCoveragePoint, ...]  # threshold descending, one per distinct score
    aurc: float | None  # area under the risk-coverage curve (lower is better)
    oracle_aurc: float | None  # the same for a perfect ranking of these outcomes
    accuracy: float | None  # accuracy at full coverage


def risk_coverage(pairs: Sequence[Pair]) -> RiskCoverage:
    """Answer the most confident first; at each distinct score threshold record coverage and
    selective accuracy. AURC is the mean risk over all n coverage levels k/n; inside a group
    of tied scores the order is random, so its expected right answers are spread evenly."""
    n = len(pairs)
    if not n:
        return RiskCoverage(n=0, points=(), aurc=None, oracle_aurc=None, accuracy=None)
    ordered = sorted(pairs, key=lambda x: -x[0])
    points: list[RiskCoveragePoint] = []
    right = 0
    risk_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and ordered[j][0] == ordered[i][0]:
            j += 1
        group_right = sum(y for _, y in ordered[i:j])
        for k in range(i + 1, j + 1):
            risk_sum += 1 - (right + group_right * (k - i) / (j - i)) / k
        right += group_right
        points.append(
            RiskCoveragePoint(
                threshold=ordered[i][0],
                coverage=quantize(j / n),
                selective_accuracy=quantize(right / j),
                answered=j,
            )
        )
        i = j
    total_right = right
    oracle = math.fsum(1 - min(k, total_right) / k for k in range(1, n + 1)) / n
    return RiskCoverage(
        n=n,
        points=tuple(points),
        aurc=quantize(risk_sum / n),
        oracle_aurc=quantize(oracle),
        accuracy=quantize(total_right / n),
    )


def selective_accuracy_at(rc: RiskCoverage, coverage: float) -> float | None:
    """Selective accuracy at the smallest coverage that reaches ``coverage``."""
    for pt in rc.points:
        if pt.coverage >= coverage - 1e-12:
            return pt.selective_accuracy
    return None


def auroc(pairs: Sequence[Pair]) -> float | None:
    """P(score of a correct outcome > score of an incorrect one), ties counting half: how
    well the score ranks right above wrong. ``None`` when one class is empty."""
    pos = [p for p, y in pairs if y]
    neg = [p for p, y in pairs if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    neg_sorted = sorted(neg)
    for p in pos:
        lo = bisect.bisect_left(neg_sorted, p)
        hi = bisect.bisect_right(neg_sorted, p)
        wins += lo + 0.5 * (hi - lo)
    return quantize(wins / (len(pos) * len(neg)))


class AbstentionQuality(Record):
    """What abstaining bought: abstained probes whose best guess would have been wrong
    (errors prevented) against right (correct answers given up)."""

    n: int
    answered: int
    abstained: int
    prevented: int
    lost: int
    undecided: int  # abstained with no belief to judge
    precision: Proportion | None  # prevented / judged abstentions
    net: int  # prevented - lost


def abstention_quality(preds: Sequence[Prediction], confidence: float = 0.95) -> AbstentionQuality:
    abstained = [p for p in preds if not p.answered]
    judged = [p for p in abstained if p.would_be_correct is not None]
    prevented = sum(1 for p in judged if p.would_be_correct is False)
    lost = sum(1 for p in judged if p.would_be_correct is True)
    return AbstentionQuality(
        n=len(preds),
        answered=len(preds) - len(abstained),
        abstained=len(abstained),
        prevented=prevented,
        lost=lost,
        undecided=len(abstained) - len(judged),
        precision=Proportion.of(prevented, len(judged), confidence) if judged else None,
        net=prevented - lost,
    )


def answered_pairs(preds: Sequence[Prediction], calibrated: bool = False) -> list[Pair]:
    out: list[Pair] = []
    for p in preds:
        s = p.calibrated if calibrated else p.score
        if p.answered and s is not None and p.correct is not None:
            out.append((s, p.correct))
    return out


def by_subgroup(
    preds: Sequence[Prediction], tags: Sequence[str], calibrated: bool = False, bins: int = 10
) -> dict[str, Reliability]:
    """Reliability per tag over the answered predictions carrying it."""
    return {
        t: reliability(answered_pairs([p for p in preds if t in p.tags], calibrated), bins)
        for t in tags
    }


def apply(model: Isotonic, preds: Sequence[Prediction]) -> list[Prediction]:
    """The predictions with ``calibrated`` set from the recalibration."""
    return [
        p.model_copy(
            update={"calibrated": quantize(model(p.score)) if p.score is not None else None}
        )
        for p in preds
    ]
