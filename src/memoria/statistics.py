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
    return max(0.0, centre - half), min(1.0, centre + half)


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
