from collections import Counter

import pytest

from memoria.adaptive import Event, EventLedger
from memoria.adaptive_cases import extraction_stats
from memoria.adaptive_worlds import (
    WORLD_NAMES,
    World,
    build_world,
    truth_at,
    truth_index,
    world_specs,
)
from memoria.claims import Claim
from memoria.identity import CONSERVATIVE, SIMILARITY
from memoria.retention import (
    KEEP,
    SimResult,
    System,
    age_window,
    gate,
    precompute_claims,
    simulate,
)

_CACHE: dict[str, World] = {}


def world(name: str) -> World:
    if name not in _CACHE:
        _CACHE[name] = build_world(next(s for s in world_specs(1, 0) if s.name == name))
    return _CACHE[name]


def claims(w: World, policy: object = CONSERVATIVE) -> dict[str, tuple[Claim, ...]]:
    return precompute_claims(w)


def stale(res: SimResult) -> float:
    a = [x for x in res.audits if x.t > 30]
    return sum(x.status == "stale" for x in a) / len(a)


@pytest.mark.parametrize("name", WORLD_NAMES)
def test_worlds_are_deterministic_and_well_formed(name: str) -> None:
    spec = next(s for s in world_specs(1, 0) if s.name == name)
    w = build_world(spec)
    assert w.digest == build_world(spec).digest
    assert [r.recorded for r in w.reports] == sorted(r.recorded for r in w.reports)
    assert [q.at for q in w.queries] == sorted(q.at for q in w.queries)
    assert all(r.recorded >= r.occurred for r in w.reports)
    index = truth_index(w)
    assert set(index) == set(w.keys)
    assert w.digest != build_world(next(s for s in world_specs(2, 0) if s.name == name)).digest


def test_probes_and_expectations_do_not_depend_on_the_system() -> None:
    # I20: the instrument is fixed; truth is a function of the world alone
    w = world("changing")
    index = truth_index(w)
    a = simulate(w, System(name="a", schedule=KEEP), claims(w))
    b = simulate(w, System(name="b", schedule=age_window(30)), claims(w))
    assert [(x.t, x.key, x.truth) for x in a.audits] == [(x.t, x.key, x.truth) for x in b.audits]
    assert all(x.truth == truth_at(index, x.key, x.t)[0] for x in a.audits)


@pytest.mark.parametrize("name", WORLD_NAMES)
def test_conservative_extraction_never_turns_uncertainty_into_fact(name: str) -> None:
    w = world(name)
    stats = extraction_stats(w, precompute_claims(w, CONSERVATIVE), "conservative")
    assert stats.uncertain_as_fact == 0
    assert stats.wrong_fact == 0
    assert stats.claims_verified.numerator == stats.claims_verified.denominator
    res = simulate(w, System(name="a", schedule=KEEP), precompute_claims(w, CONSERVATIVE))
    assert res.misattributed == 0


def test_unresolved_claims_are_counted_not_dropped() -> None:
    w = world("noisy")
    res = simulate(w, System(name="a", schedule=KEEP), claims(w))
    uncertain = sum(not r.certain for r in w.reports)
    assert uncertain > 0
    hedged = res.unresolved["hedged"] + res.unresolved["pronoun_subject"]
    assert hedged >= uncertain  # every uncertain report is an unresolved claim
    retractions = sum(r.kind == "negate" for r in w.reports)  # corrections, not items
    assert res.items + sum(res.unresolved.values()) + retractions == len(w.reports)


def test_simulation_is_deterministic() -> None:
    w = world("changing")
    s = System(name="f", schedule=gate(0.5), rank_weight=0.5)
    a, b = simulate(w, s, claims(w)), simulate(w, s, claims(w))
    assert a.audits == b.audits
    assert a.events == b.events


def test_forgetting_is_availability_never_deletion() -> None:
    w = world("changing")
    base = simulate(w, System(name="a", schedule=KEEP), claims(w))
    for schedule in (age_window(30), gate(0.5), gate(0.5, stale_aware=True)):
        res = simulate(w, System(name="x", schedule=schedule), claims(w))
        assert res.items == base.items  # every item was ingested and kept
        assert res.catalog.keys() == base.catalog.keys()
    short = simulate(w, System(name="c", schedule=age_window(30)), claims(w))
    assert sum(x.status == "none" for x in short.audits) > sum(
        x.status == "none" for x in base.audits
    )
    assert all(x.known >= x.available for x in short.audits)
    assert sum(x.known for x in short.audits) == sum(x.known for x in base.audits)


def test_audit_probes_create_no_events() -> None:
    w = world("changing").model_copy(update={"queries": ()})
    res = simulate(w, System(name="f", schedule=gate(0.5), rank_weight=0.5), claims(w))
    assert res.ledger is not None
    kinds = Counter(e.kind for e in res.ledger.all())
    assert kinds["used"] == 0
    assert set(kinds) <= {"correction", "contradiction_discovered"}  # from reports only
    assert len(res.audits) == 36 * len(w.keys)


def test_the_memory_never_sees_ground_truth_evaluation() -> None:
    w = world("adversarial")
    res = simulate(w, System(name="f", schedule=gate(0.5), rank_weight=0.5), claims(w))
    assert res.ledger is not None
    assert {e.origin for e in res.ledger.all()} <= {"system_observed", "user", "source_claim"}
    assert Counter(e.kind for e in res.ledger.all())["used"] > 0


def test_decisions_and_importances_carry_cutoffs_and_reproduce() -> None:
    from memoria.adaptive_study import trace_audit

    for name in ("changing", "adversarial"):
        w = world(name)
        for sysname, sched in (("f", gate(0.5)), ("g", gate(0.5, stale_aware=True))):
            audit = trace_audit(w, System(name=sysname, schedule=sched, rank_weight=0.5), claims(w))
            assert audit.decisions > 0
            assert audit.cutoff_violations == audit.reproduction_violations == 0
            assert audit.unresolved_importance_links == 0
    assert dict(audit.reasons).keys() >= {"newest_exempt"}


def test_age_window_archives_only_beyond_its_age() -> None:
    w = world("changing")
    res = simulate(w, System(name="c", schedule=age_window(30)), claims(w), trace=True)
    assert res.decisions
    for d in res.decisions:
        age = d.at - res.catalog[d.item].recorded
        assert (d.state == "archived") == (age > 30)
        assert d.cutoff == d.at


def test_late_feedback_is_recorded_late() -> None:
    w = world("changing")
    lag = System(name="k", schedule=gate(0.5), rank_weight=0.5, feedback_lag_days=7)
    res = simulate(w, lag, claims(w))
    assert res.ledger is not None
    user = [e for e in res.ledger.all() if e.origin == "user"]
    assert user
    assert all(e.recorded_at - e.at >= 7 - 1e-9 for e in user)


def test_a_dose_of_importance_ranking_entrenches_stale_values_in_a_changing_world() -> None:
    # observed in the generated 'changing' world, not a universal law
    w = world("changing")
    latest = simulate(w, System(name="a", schedule=KEEP), claims(w))
    heavy = simulate(w, System(name="r", schedule=KEEP, rank_weight=1.0), claims(w))
    assert stale(heavy) > stale(latest) + 0.2


def test_the_ledger_refuses_events_before_a_decision_even_inside_a_run() -> None:
    ledger = EventLedger()
    ledger.visible("i", 30.0)
    with pytest.raises(ValueError, match="after a decision"):
        ledger.append(
            Event(kind="correction", item="i", at=1, recorded_at=1, origin="user", cause="c")
        )


def test_similarity_identity_resolves_typos_the_conservative_policy_leaves_open() -> None:
    w = world("adversarial")
    stats = extraction_stats(w, precompute_claims(w, SIMILARITY), "similarity")
    conservative = extraction_stats(w, precompute_claims(w, CONSERVATIVE), "conservative")
    assert stats.recall is not None
    assert conservative.recall is not None
    assert stats.recall.numerator >= conservative.recall.numerator  # typos resolve
    assert stats.uncertain_as_fact >= conservative.uncertain_as_fact == 0
