import math

import pytest
from pydantic import ValidationError

from memoria.calibration import (
    Isotonic,
    Prediction,
    abstention_quality,
    answered_pairs,
    apply,
    auroc,
    brier,
    by_subgroup,
    calibrated_pairs,
    fit_isotonic,
    log_loss,
    reliability,
    risk_coverage,
    selective_accuracy_at,
)

# 10 answers at 0.9 (9 right), 6 at 0.5 (3 right), 10 at 0.1 (2 right)
PAIRS = (
    [(0.9, True)] * 9 + [(0.9, False)] + [(0.5, True)] * 3 + [(0.5, False)] * 3
    + [(0.1, False)] * 8 + [(0.1, True)] * 2
)  # fmt: skip


def test_reliability_is_computed_by_hand() -> None:
    r = reliability(PAIRS)
    assert r.n == 26
    assert r.ece == pytest.approx(10 / 26 * 0.1)  # only the 0.1 bin is off: |0.2 - 0.1|
    assert r.mce == pytest.approx(0.1)
    assert r.brier == pytest.approx(
        (9 * 0.01 + 1 * 0.81 + 6 * 0.25 + 8 * 0.01 + 2 * 0.81) / 26, abs=1e-9
    )
    assert r.overconfidence == pytest.approx(r.mean_score - r.accuracy)  # type: ignore[operator]
    used = [b for b in r.bins if b.n]
    assert [b.n for b in used] == [10, 6, 10]
    assert used[0].accuracy.estimate == 0.2  # type: ignore[union-attr]
    assert used[0].accuracy is not None
    assert used[0].accuracy.low is not None  # every bin carries an interval (Wilson)
    assert sum(b.n for b in r.bins) == r.n
    assert reliability([]).ece is None


def test_perfectly_calibrated_and_maximally_wrong_extremes() -> None:
    assert reliability([(0.75, True)] * 3 + [(0.75, False)]).ece == 0.0
    assert reliability([(1.0, False)] * 5).ece == 1.0
    assert reliability([(1.0, True)]).bins[-1].n == 1  # the last bin includes 1.0
    with pytest.raises(ValueError, match="outside"):
        reliability([(1.2, True)])


def test_brier_and_log_loss_only_with_probabilistic_semantics() -> None:
    assert brier([(1.0, True), (0.0, False)]) == 0.0
    assert brier([]) is None
    with pytest.raises(ValueError, match="probabilistic"):
        log_loss(PAIRS, probabilistic=False)
    assert log_loss([(0.5, True), (0.5, False)], probabilistic=True) == pytest.approx(math.log(2))
    assert log_loss([(1.0, False)], probabilistic=True) == pytest.approx(-math.log(1e-6), abs=1e-6)


def test_isotonic_is_monotone_and_reproduces_pav() -> None:
    m = fit_isotonic(PAIRS)
    assert m.values == (0.2, 0.5, 0.9)
    assert m.uppers == (0.1, 0.5, 0.9)
    assert [m(x) for x in (0.0, 0.1, 0.3, 0.5, 0.7, 0.95)] == [0.2, 0.2, 0.5, 0.5, 0.9, 0.9]
    assert list(m.values) == sorted(m.values)
    fitted = calibrated_pairs(PAIRS, m)
    assert reliability(fitted).ece == pytest.approx(0.0, abs=1e-9)  # on its own fit data
    # a violation is pooled: high scores that are less accurate than low ones
    inverted = fit_isotonic([(0.2, True)] * 4 + [(0.8, False)] * 4)
    assert inverted.values == (0.5,)
    assert m.fit is not None
    assert m.fit == fit_isotonic(list(reversed(PAIRS))).fit  # order-free provenance of the data
    assert fit_isotonic([]) == Isotonic(uppers=(), values=(), n=0, fit=None)
    assert Isotonic(uppers=(), values=(), n=0, fit=None)(0.7) == 0.7
    with pytest.raises(ValidationError, match="monotone"):
        Isotonic(uppers=(0.1, 0.5), values=(0.9, 0.1), n=2, fit=None)


def test_recalibration_helps_on_held_out_data_when_the_score_is_biased() -> None:
    train = [(0.9, i % 2 == 0) for i in range(40)] + [(0.3, i % 5 == 0) for i in range(40)]
    test = [(0.9, i % 2 == 0) for i in range(60)] + [(0.3, i % 5 == 0) for i in range(60)]
    m = fit_isotonic(train)
    raw, fixed = reliability(test), reliability(calibrated_pairs(test, m))
    assert raw.ece == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)
    assert fixed.ece == pytest.approx(0.0, abs=1e-9)
    assert log_loss(calibrated_pairs(test, m), probabilistic=True) is not None


def test_risk_coverage_and_the_oracle() -> None:
    rc = risk_coverage(PAIRS)
    assert rc.n == 26
    assert rc.points[0].threshold == 0.9
    assert rc.points[0].coverage == pytest.approx(10 / 26, abs=1e-9)
    assert rc.points[0].selective_accuracy == 0.9
    assert rc.points[-1].coverage == 1.0
    assert rc.points[-1].selective_accuracy == pytest.approx(14 / 26, abs=1e-9)
    assert rc.oracle_aurc is not None
    assert rc.aurc is not None
    assert rc.oracle_aurc < rc.aurc
    assert selective_accuracy_at(rc, 0.3) == 0.9
    assert selective_accuracy_at(rc, 1.0) == pytest.approx(14 / 26, abs=1e-9)
    perfect = risk_coverage([(0.9, True)] * 3 + [(0.1, False)] * 3)
    assert perfect.aurc == pytest.approx(perfect.oracle_aurc)  # a perfect ranking meets the oracle
    assert perfect.oracle_aurc == pytest.approx((3 * 0 + 0.25 + 0.4 + 0.5) / 6, abs=1e-9)
    tied = risk_coverage([(0.5, True), (0.5, False)])  # ties are broken at random, in expectation
    assert tied.aurc == pytest.approx((0.5 + 0.5) / 2)
    assert risk_coverage([]).aurc is None


def test_auroc_ties_count_half() -> None:
    assert auroc([(0.9, True), (0.1, False)]) == 1.0
    assert auroc([(0.1, True), (0.9, False)]) == 0.0
    assert auroc([(0.5, True), (0.5, False)]) == 0.5
    assert auroc([(0.5, True)]) is None


def prediction(probe: str, mode: str, score: float | None, correct: bool | None,
               would: bool | None, tags: tuple[str, ...] = ()) -> Prediction:  # fmt: skip
    return Prediction(
        probe=probe, key="a.b", mode=mode, value="x" if correct is not None else None,  # type: ignore[arg-type]
        score=score, calibrated=None, correct=correct, would_be_correct=would, candidate_hit=None,
        tags=tags,
    )  # fmt: skip


def test_abstention_quality_separates_errors_prevented_from_answers_given_up() -> None:
    preds = [
        prediction("1", "answer", 0.9, True, True), prediction("2", "answer", 0.8, False, False),
        prediction("3", "abstain", 0.2, None, False), prediction("4", "abstain", 0.3, None, True),
        prediction("5", "competing", 0.5, None, False),
        prediction("6", "abstain", None, None, None),
    ]  # fmt: skip
    q = abstention_quality(preds)
    assert (q.answered, q.abstained, q.prevented, q.lost, q.undecided, q.net) == (2, 4, 2, 1, 1, 1)
    assert q.precision is not None
    assert (q.precision.numerator, q.precision.denominator) == (2, 3)
    assert answered_pairs(preds) == [(0.9, True), (0.8, False)]


def test_subgroups_expose_what_the_aggregate_hides() -> None:
    good = [prediction(f"g{i}", "answer", 0.9, i != 0, True, ("easy",)) for i in range(10)]
    bad = [prediction(f"b{i}", "answer", 0.9, i < 2, True, ("hard",)) for i in range(10)]
    groups = by_subgroup([*good, *bad], ("easy", "hard", "absent"))
    assert groups["easy"].ece == pytest.approx(0.0)
    assert groups["hard"].ece == pytest.approx(0.7)
    assert reliability(answered_pairs([*good, *bad])).ece == pytest.approx(0.35)  # hides both
    assert groups["absent"].n == 0


def test_apply_sets_calibrated_values_without_changing_scores() -> None:
    m = fit_isotonic(PAIRS)
    out = apply(
        m,
        [prediction("1", "answer", 0.9, True, True), prediction("2", "abstain", None, None, None)],
    )
    assert out[0].calibrated == 0.9
    assert out[0].score == 0.9
    assert out[1].calibrated is None
