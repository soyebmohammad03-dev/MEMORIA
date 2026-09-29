import pytest
from pydantic import ValidationError

from memoria.adaptive import (
    COMPONENTS,
    DEFAULT,
    STATIC,
    CircularEvidenceError,
    Event,
    EventLedger,
    ImportanceSpec,
    LeakageError,
    ablate,
    compute,
    score,
    verify_importance,
)


def used(item: str, t: float, cause: str = "q") -> Event:
    return Event(kind="used", item=item, at=t, recorded_at=t, origin="system_observed", cause=cause)


def test_events_cannot_be_recorded_before_they_happen() -> None:
    with pytest.raises(ValidationError):
        Event(kind="correction", item="i", at=5, recorded_at=4, origin="user", cause="c")
    with pytest.raises(ValidationError):  # access is observed by the system, not asserted
        Event(kind="used", item="i", at=1, recorded_at=1, origin="user", cause="q")


def test_ground_truth_evaluation_is_refused_as_evidence() -> None:
    ledger = EventLedger()
    e = Event(kind="correction", item="i", at=1, recorded_at=1, origin="evaluation", cause="c")
    with pytest.raises(CircularEvidenceError):
        ledger.append(e)
    assert ledger.count == 0


def test_visibility_is_strictly_before_the_decision() -> None:
    ledger = EventLedger()
    ledger.append(used("i", 1.0))
    ledger.append(used("i", 2.0, "q2"))
    assert [e.at for e in ledger.visible("i", 2.0)] == [1.0]  # recorded at 2.0 is not before 2.0
    assert [e.at for e in ledger.visible("i", 2.5)] == [1.0, 2.0]


def test_a_decision_freezes_the_past() -> None:
    ledger = EventLedger()
    ledger.visible("i", 10.0)
    with pytest.raises(LeakageError):
        ledger.append(used("i", 9.0))
    ledger.append(used("i", 10.0))  # recorded at the decision time: not visible to it, allowed
    assert ledger.visible("i", 10.0) == []


def test_late_recorded_feedback_is_invisible_until_recorded() -> None:
    ledger = EventLedger()
    u = used("i", 1.0)
    ledger.append(u)
    late = Event(
        kind="user_confirmed_relevance", item="i", at=2, recorded_at=9, origin="user", cause=u.id
    )
    ledger.append(late)
    early = compute(ledger, "i", 0, 0.5, 8.0, DEFAULT)
    assert late.id not in early.events
    assert late.id in compute(ledger, "i", 0, 0.5, 20.0, DEFAULT).events  # the use has matured


def test_importance_is_the_declared_weighted_sum_and_traceable() -> None:
    ledger = EventLedger()
    for k in range(4):
        u = used("i", float(k), f"q{k}")
        ledger.append(u)
        ledger.append(
            Event(
                kind="retrieval_success",
                item="i",
                at=k + 1,
                recorded_at=k + 1,
                origin="system_observed",
                cause=u.id,
            )
        )
    rec = compute(ledger, "i", 0.0, 0.8, 40.0, DEFAULT)
    assert rec.score == score(ledger, "i", 0.0, 0.8, 40.0, DEFAULT)
    assert [c.name for c in rec.components] == sorted(COMPONENTS)
    assert sum(c.weight for c in rec.components) == pytest.approx(1)
    assert verify_importance(rec, ledger, 0.0, 0.8, DEFAULT) == []
    freq = next(c for c in rec.components if c.name == "frequency")
    assert freq.raw == (("uses", 4.0),)
    assert set(rec.events) <= {e.id for e in ledger.all()}


def test_later_information_cannot_change_an_earlier_importance() -> None:
    ledger = EventLedger()
    ledger.append(used("i", 1.0))
    before = compute(ledger, "i", 0.0, 0.8, 5.0, DEFAULT)
    ledger.append(used("i", 6.0, "q2"))
    ledger.append(Event(kind="correction", item="i", at=6, recorded_at=6, origin="user", cause="c"))
    assert compute(ledger, "i", 0.0, 0.8, 5.0, DEFAULT).digest == before.digest
    assert verify_importance(before, ledger, 0.0, 0.8, DEFAULT) == []


def test_verification_detects_a_record_that_no_longer_reproduces() -> None:
    ledger = EventLedger()
    ledger.append(used("i", 1.0))
    rec = compute(ledger, "i", 0.0, 0.8, 5.0, DEFAULT)
    other = EventLedger()  # a ledger missing the event the record cites
    problems = verify_importance(rec, other, 0.0, 0.8, DEFAULT)
    assert "re-derivation differs" in problems
    assert any(p.startswith("unknown event") for p in problems)


def test_ablation_removes_one_component_and_renormalises() -> None:
    for c in COMPONENTS:
        spec = ablate(DEFAULT, c)
        assert c not in dict(spec.weights)
        assert sum(w for _, w in spec.weights) == pytest.approx(1)
        assert len(spec.weights) == len(COMPONENTS) - 1
    ledger = EventLedger()
    for k in range(3):
        ledger.append(used("i", float(k), f"q{k}"))
    full = score(ledger, "i", 0.0, 0.8, 10.0, DEFAULT)
    assert score(ledger, "i", 0.0, 0.8, 10.0, ablate(DEFAULT, "frequency")) != full
    with pytest.raises(ValueError, match="cannot ablate"):
        ablate(STATIC, "frequency")  # not a component of that model


def test_static_importance_reads_no_events() -> None:
    a, b = EventLedger(), EventLedger()
    b.append(used("i", 1.0))
    b.append(Event(kind="correction", item="i", at=2, recorded_at=2, origin="user", cause="c"))
    assert score(a, "i", 0.0, 0.8, 10.0, STATIC) == score(b, "i", 0.0, 0.8, 10.0, STATIC)
    assert not STATIC.uses_events
    assert DEFAULT.uses_events


def test_spec_weights_must_be_declared_positive_and_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        ImportanceSpec(name="x", weights=(("frequency", 0.5),))
    with pytest.raises(ValidationError):
        ImportanceSpec(name="x", weights=(("nonsense", 1.0),))
    with pytest.raises(ValidationError):
        ImportanceSpec(name="x", weights=(("frequency", 0.0), ("recency", 1.0)))


def test_correction_lowers_and_use_raises_importance_without_any_truth_input() -> None:
    ledger = EventLedger()
    ledger.append(used("used", 1.0))
    ledger.append(
        Event(kind="correction", item="corrected", at=1, recorded_at=1, origin="user", cause="c")
    )
    base = score(EventLedger(), "plain", 0.0, 0.8, 5.0, DEFAULT)
    assert score(ledger, "used", 0.0, 0.8, 5.0, DEFAULT) > base
    assert score(ledger, "corrected", 0.0, 0.8, 5.0, DEFAULT) < base
