"""Statistical methods, checked against published reference values."""

import math

import pytest
from pydantic import ValidationError

from memoria.statistics import (
    Proportion,
    holm,
    mcnemar_exact,
    min_achievable_p,
    newcombe_paired,
    significant,
    wilson,
    z_value,
)


def close(a: float, b: float, tol: float = 5e-5) -> bool:
    return abs(a - b) <= tol


# Wilson (1927) score intervals; values as tabulated by Newcombe (1998a) / Brown, Cai &
# DasGupta (2001).
@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (0, 10, 0.0, 0.2775),
        (5, 10, 0.2366, 0.7634),
        (3, 10, 0.1078, 0.6032),
        (10, 10, 0.7225, 1.0),
        (81, 263, 0.2553, 0.3662),
    ],
)
def test_wilson_reference_values(k: int, n: int, low: float, high: float) -> None:
    interval = wilson(k, n, 0.95)
    assert interval is not None
    assert close(interval[0], low)
    assert close(interval[1], high)


def test_wilson_edges() -> None:
    assert wilson(0, 0, 0.95) is None
    for n in (1, 2, 7, 1000):
        for k in (0, n // 2, n):
            interval = wilson(k, n, 0.95)
            assert interval is not None
            assert 0 <= interval[0] <= k / n <= interval[1] <= 1
            assert (interval[0] == 0.0) == (k == 0)  # exact, on every platform
            assert (interval[1] == 1.0) == (k == n)
    wide, narrow = wilson(5, 10, 0.99), wilson(5, 10, 0.80)
    assert wide is not None
    assert narrow is not None
    assert wide[0] < narrow[0]
    for bad in ((-1, 5), (6, 5)):
        with pytest.raises(ValueError, match="k <= n"):
            wilson(*bad, 0.95)
    for bad_confidence in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="confidence"):
            z_value(bad_confidence)
    assert close(z_value(0.95), 1.959964, 1e-6)


# Exact McNemar: two-sided binomial test on discordant pairs.
@pytest.mark.parametrize(
    ("b", "c", "p"),
    [
        (0, 0, 1.0),
        (0, 5, 0.0625),
        (5, 0, 0.0625),
        (1, 9, 22 / 1024),
        (3, 3, 1.0),
        (0, 1, 1.0),
        (2, 10, 2 * (1 + 12 + 66) / 4096),
    ],
)
def test_mcnemar_exact(b: int, c: int, p: float) -> None:
    assert math.isclose(mcnemar_exact(b, c), p, rel_tol=1e-12)


def test_mcnemar_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_exact(-1, 3)


def test_min_achievable_p_defines_underpowered() -> None:
    assert min_achievable_p(0) == 1.0
    assert min_achievable_p(5) == 0.0625  # cannot reach 0.05 with 5 discordant pairs
    assert min_achievable_p(6) == 0.03125  # can
    assert all(min_achievable_p(n + 1) < min_achievable_p(n) for n in range(1, 30))


# Newcombe (1998) Stat Med 17:2635, Table III, method 10 (95 per cent), cells (e, f, g, h).
NEWCOMBE_TABLE_III = [
    ((36, 12, 2, 0), (0.0569, 0.3404)),
    ((20, 12, 2, 16), (0.0562, 0.3292)),
    ((18, 12, 2, 18), (0.0562, 0.3290)),
    ((36, 14, 0, 0), (0.1528, 0.4167)),
    ((35, 14, 0, 1), (0.1461, 0.4175)),
    ((18, 14, 0, 18), (0.1441, 0.3963)),
    ((2, 97, 1, 0), (0.8721, 0.9854)),
    ((1, 97, 1, 1), (0.8736, 0.9850)),
    ((0, 29, 1, 0), (0.6666, 0.9882)),
    ((2, 98, 0, 0), (0.9178, 0.9945)),
    ((1, 98, 0, 1), (0.9171, 0.9916)),
    ((0, 30, 0, 0), (0.8395, 1.0)),
    ((54, 0, 0, 0), (-0.0664, 0.0664)),
]


@pytest.mark.parametrize(("cells", "interval"), NEWCOMBE_TABLE_III)
def test_newcombe_method_10_reproduces_published_table(
    cells: tuple[int, int, int, int], interval: tuple[float, float]
) -> None:
    result = newcombe_paired(*cells, 0.95)
    assert result is not None
    theta, low, high = result
    e, f, g, h = cells
    assert math.isclose(theta, (f - g) / (e + f + g + h))
    assert close(low, interval[0], 1.1e-4)  # the paper prints four decimals
    assert close(high, interval[1], 1.1e-4)


def test_newcombe_properties() -> None:
    assert newcombe_paired(0, 0, 0, 0, 0.95) is None
    with pytest.raises(ValueError, match="non-negative"):
        newcombe_paired(1, -1, 0, 0, 0.95)
    for cells in [(5, 3, 1, 4), (0, 0, 10, 0), (10, 0, 0, 0), (2, 7, 7, 2), (0, 1, 0, 0)]:
        forward = newcombe_paired(*cells, 0.95)
        e, f, g, h = cells
        backward = newcombe_paired(e, g, f, h, 0.95)
        assert forward is not None
        assert backward is not None
        theta, low, high = forward
        assert -1 <= low <= theta <= high <= 1
        # Swapping the conditions negates the difference and mirrors the interval.
        assert math.isclose(backward[0], -theta, abs_tol=1e-12)
        assert math.isclose(backward[1], -high, abs_tol=1e-12)
        assert math.isclose(backward[2], -low, abs_tol=1e-12)


def test_holm_reference_and_properties() -> None:
    assert holm([]) == []
    assert holm([0.04]) == [0.04]
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert all(math.isclose(a, b) for a, b in zip(adjusted, [0.03, 0.06, 0.06, 0.02], strict=True))
    assert holm([0.5, 0.9]) == [1.0, 1.0]  # capped at 1
    raw = [0.001, 0.2, 0.03, 0.04, 0.9]
    adj = holm(raw)
    assert all(a >= r for a, r in zip(adj, raw, strict=True))
    order = sorted(range(len(raw)), key=raw.__getitem__)
    assert [adj[i] for i in order] == sorted(adj)  # monotone in the raw ordering


def test_significant_never_rounds_small_values_to_zero() -> None:
    assert significant(1.23456789012345e-13) == 1.23456789012e-13
    assert significant(0.0625) == 0.0625
    assert significant(1 / 3) == 0.333333333333


def test_proportion_follows_from_its_counts() -> None:
    p = Proportion.of(3, 10, 0.95)
    assert (p.estimate, p.low, p.high) == (0.3, 0.107791267406, 0.603221852539)
    empty = Proportion.of(0, 0, 0.95)
    assert (empty.estimate, empty.low, empty.high) == (None, None, None)
    assert Proportion.model_validate_json(p.canonical()) == p
    for tampered in (
        {"estimate": 0.9},
        {"low": 0.0},
        {"numerator": 11},
        {"denominator": 11},
        {"method": "wald"},
        {"estimate": None, "low": None, "high": None},
    ):
        with pytest.raises(ValidationError):
            Proportion.model_validate(p.model_dump() | tampered)
