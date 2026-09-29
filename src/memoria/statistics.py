"""Statistical foundations for evaluation: proportions, paired comparisons, multiplicity.

Methods (chosen for small samples, closed form, and determinism; no resampling):

- Proportions: Wilson score interval. Never leaves [0, 1] and behaves at k = 0 or n.
- Paired difference of proportions (same probes, two conditions): Newcombe (1998)
  method 10: Wilson intervals of the two marginals combined through the table's phi
  correlation, with phi continuity-corrected (Newcombe's recommendation; the uncorrected
  method 8 has marked coverage dips at small n).
- Paired test: exact McNemar test (two-sided binomial test on the discordant pairs).
- Multiplicity: Holm step-down adjustment within a family of tests.

Nothing is reported without its counts. Undefined quantities (n = 0) are ``None``, never
0. A test is flagged *underpowered* when even the most extreme possible outcome with its
number of discordant pairs could not reach ``alpha``.

References: Wilson (1927) JASA 22:209; Newcombe (1998) Stat Med 17:2635 (method 10);
McNemar (1947) Psychometrika 12:153; Holm (1979) Scand J Stat 6:65.
"""

from __future__ import annotations

import math
from fractions import Fraction
from statistics import NormalDist
from typing import Self

from pydantic import Field, model_validator

from memoria.core import Record, quantize

Probability = float


def significant(x: float) -> float:
    """Round to 12 significant figures: deterministic, and never rounds a tiny p-value to 0."""
    return float(f"{x:.12g}")


def z_value(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def wilson(k: int, n: int, confidence: float) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials; ``None`` when n = 0."""
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    if n == 0:
        return None
    z = z_value(confidence)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # The bounds are exactly 0 at k = 0 and exactly 1 at k = n; do not leave that to
    # floating-point cancellation, which differs in the last bit across platforms.
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c (1.0 when b + c = 0)."""
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)), 2**n)
    return float(min(Fraction(1), 2 * tail))


def min_achievable_p(discordant: int) -> float:
    """The smallest two-sided exact McNemar p-value possible with this many discordant pairs."""
    return mcnemar_exact(0, discordant)


def newcombe_paired(
    e: int, f: int, g: int, h: int, confidence: float
) -> tuple[float, float, float] | None:
    """Paired difference of proportions with Newcombe's (1998) method 10 interval.

    Newcombe's notation: of n pairs, ``e`` are positive under both conditions, ``f``
    under the first only, ``g`` under the second only, ``h`` under neither. The
    difference is theta = (f - g) / n = p(first) - p(second). The interval combines the
    Wilson intervals of the two marginal proportions with the phi correlation of the
    table, continuity-corrected in its numerator: max(eh - fg - n/2, 0) when eh > fg.
    Returns (theta, low, high), or ``None`` when n = 0.
    """
    if min(e, f, g, h) < 0:
        raise ValueError("counts must be non-negative")
    n = e + f + g + h
    if n == 0:
        return None
    p1, p2 = (e + f) / n, (e + g) / n
    l1, u1 = wilson(e + f, n, confidence) or (0.0, 0.0)
    l2, u2 = wilson(e + g, n, confidence) or (0.0, 0.0)
    marginals = (e + f) * (g + h) * (e + g) * (f + h)
    numerator = max(e * h - f * g - n / 2, 0.0) if e * h > f * g else float(e * h - f * g)
    phi = numerator / math.sqrt(marginals) if marginals else 0.0
    dl1, du1, dl2, du2 = p1 - l1, u1 - p1, p2 - l2, u2 - p2
    theta = p1 - p2
    delta = math.sqrt(max(0.0, dl1**2 - 2 * phi * dl1 * du2 + du2**2))
    epsilon = math.sqrt(max(0.0, du1**2 - 2 * phi * du1 * dl2 + dl2**2))
    return theta, max(-1.0, theta - delta), min(1.0, theta + epsilon)


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _derived(k: int, n: int, confidence: float) -> tuple[float | None, float | None, float | None]:
    interval = wilson(k, n, confidence)
    if interval is None:
        return None, None, None
    return quantize(k / n), quantize(interval[0]), quantize(interval[1])


class Proportion(Record):
    """k of n, with a Wilson interval. Estimate and interval are ``None`` when n = 0.

    The estimate and interval are validated to follow from the counts, so a stored
    proportion cannot disagree with its own numerator and denominator.
    """

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    confidence: float = Field(gt=0, lt=1)
    estimate: Probability | None
    low: Probability | None
    high: Probability | None
    method: str = "wilson"

    @classmethod
    def of(cls, k: int, n: int, confidence: float) -> Proportion:
        estimate, low, high = _derived(k, n, confidence)
        return cls(
            numerator=k, denominator=n, confidence=confidence, estimate=estimate, low=low, high=high
        )

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if self.numerator > self.denominator:
            raise ValueError("numerator exceeds denominator")
        if self.method != "wilson":
            raise ValueError(f"unknown interval method {self.method!r}")
        derived = _derived(self.numerator, self.denominator, self.confidence)
        if (self.estimate, self.low, self.high) != derived:
            raise ValueError("estimate or interval does not follow from the counts")
        return self


# --- paired continuous differences ---------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta (modified Lentz)."""
    tiny, eps = 1e-300, 3e-16
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for num in (
            m * (b - m) * x / ((a + m2 - 1) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1)),
        ):
            d = 1 + num * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < eps:
            return h
    raise ArithmeticError("incomplete beta did not converge")


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x)
    ln += b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(ln) * _betacf(a, b, x) / a
    return 1 - math.exp(ln) * _betacf(b, a, 1 - x) / b


def t_cdf(t: float, df: int) -> float:
    """Student's t distribution function (via the regularised incomplete beta)."""
    tail = 0.5 * _betainc(df / 2, 0.5, df / (df + t * t))
    return 1 - tail if t >= 0 else tail


def t_quantile(p: float, df: int) -> float:
    """Inverse of :func:`t_cdf` by bisection (deterministic; |error| < 1e-12)."""
    if not 0 < p < 1 or df < 1:
        raise ValueError("need 0 < p < 1 and df >= 1")
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class PairedMean(Record):
    """Mean of paired differences (treatment - baseline) over n independent units, with a
    Student-t interval, an exact two-sided sign test on the non-zero differences, and the
    standardised effect d_z = mean / sd. ``None`` when undefined (n < 2 or sd = 0)."""

    n: int = Field(ge=0)
    mean: float | None
    low: float | None
    high: float | None
    positive: int = Field(ge=0)
    negative: int = Field(ge=0)
    sign_p: float
    d_z: float | None
    underpowered: bool  # the sign test could not reach alpha with this many non-zero pairs


def paired_mean(diffs: list[float], confidence: float, alpha: float = 0.05) -> PairedMean:
    n = len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    p = significant(mcnemar_exact(pos, neg))
    under = min_achievable_p(pos + neg) > alpha
    if n < 2:
        mean = quantize(math.fsum(diffs) / n) if n else None
        return PairedMean(
            n=n,
            mean=mean,
            low=None,
            high=None,
            positive=pos,
            negative=neg,
            sign_p=p,
            d_z=None,
            underpowered=under,
        )
    mean = math.fsum(diffs) / n
    sd = math.sqrt(math.fsum((d - mean) ** 2 for d in diffs) / (n - 1))
    if sd == 0:
        return PairedMean(
            n=n,
            mean=quantize(mean),
            low=quantize(mean),
            high=quantize(mean),
            positive=pos,
            negative=neg,
            sign_p=p,
            d_z=None,
            underpowered=under,
        )
    half = t_quantile(1 - (1 - confidence) / 2, n - 1) * sd / math.sqrt(n)
    return PairedMean(
        n=n,
        mean=quantize(mean),
        low=quantize(mean - half),
        high=quantize(mean + half),
        positive=pos,
        negative=neg,
        sign_p=p,
        d_z=quantize(mean / sd),
        underpowered=under,
    )
