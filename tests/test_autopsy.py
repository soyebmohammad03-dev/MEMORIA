from collections.abc import Sequence
from functools import cache

import pytest

from memoria.adaptive_worlds import World, build_world, world_specs
from memoria.autopsy import (
    INTERPRETATION,
    MEMORY_LINKS,
    counterfactual,
    historical_mismatches,
    ledger_digest,
    log_digest,
    memory_autopsy,
    replay_memory,
    snapshot,
    substrate_hierarchy,
    substrate_state,
    verify_autopsy,
)
from memoria.claims import Claim
from memoria.consolidation import ConsolidationPolicy
from memoria.core import Experience
from memoria.embeddings import HashedNgramEmbedder
from memoria.formation import EpisodicPolicy, StatementPolicy, form
from memoria.retention import (
    KEEP,
    SimResult,
    System,
    age_window,
    gate,
    precompute_claims,
    simulate,
)
from memoria.scenarios import day
from memoria.store import MemoryLog

F = System(name="f", schedule=gate(0.5), rank_weight=0.5)
G = System(name="g", schedule=gate(0.5, stale_aware=True), rank_weight=0.5)
A = System(name="a", schedule=KEEP)


@cache
def world(name: str = "changing") -> World:
    return build_world(next(s for s in world_specs(1, 0) if s.name == name))


@cache
def claims(name: str = "changing") -> dict[str, tuple[Claim, ...]]:
    return precompute_claims(world(name))


@cache
def run(name: str, which: str) -> SimResult:
    return simulate(world(name), {"f": F, "g": G, "a": A}[which], claims(name))


def test_replay_reproduces_every_sampled_live_decision() -> None:
    for which, sysm in (("f", F), ("g", G), ("a", A)):
        checked, bad = historical_mismatches(run("changing", which), sysm, every=7)
        assert checked > 100
        assert bad == []


def test_replay_never_mutates_the_live_run() -> None:
    res = run("changing", "f")
    assert res.ledger is not None
    before, frozen = ledger_digest(res), res.ledger.frozen
    replay_memory(res, F, 100.0)
    snapshot(res, F, 200.0)
    a = res.audits[100]
    memory_autopsy(world(), F, res, claims(), a.key, a.t)
    assert ledger_digest(res) == before
    assert res.ledger.frozen == frozen


def test_truncated_and_bulk_replay_agree_so_nothing_future_is_used() -> None:
    res = run("changing", "f")
    for a in res.audits[50::60]:
        cut = replay_memory(res, F, a.t, truncate=True)
        full = replay_memory(res, F, a.t, truncate=False)
        x = cut.choose(cut.available(a.key, a.t), a.t)
        y = full.choose(full.available(a.key, a.t), a.t)
        assert (x.id if x else None) == (y.id if y else None) == a.item


def test_before_and_after_a_source_correction_the_item_is_archived() -> None:
    res = run("changing", "g")
    assert res.ledger is not None
    newest = {}
    for e in res.ledger.all():
        if e.kind == "correction" and e.origin == "source_claim":
            it = res.catalog[e.item]
            known = (
                i for i in res.catalog.values() if i.key == it.key and i.recorded <= e.recorded_at
            )
            top = max(known, key=lambda i: (i.occurred, i.recorded, i.id))
            if top.id != it.id:
                newest[e.item] = e.recorded_at
    assert newest
    item, t = next(iter(newest.items()))
    before = dict((i, (s, r)) for i, s, r in snapshot(res, G, t, [res.catalog[item].key]).states)
    after = dict(
        (i, (s, r)) for i, s, r in snapshot(res, G, t + 0.001, [res.catalog[item].key]).states
    )
    assert before[item][1] != "source_corrected"
    assert after[item] == ("archived", "source_corrected")


def test_before_and_after_forgetting_by_age() -> None:
    sysm = System(name="c", schedule=age_window(30))
    res = simulate(world(), sysm, claims())
    it = min(res.catalog.values(), key=lambda i: i.recorded)
    key = [it.key]
    young = dict((i, s) for i, s, _ in snapshot(res, sysm, it.recorded + 29, key).states)
    old = dict((i, s) for i, s, _ in snapshot(res, sysm, it.recorded + 31, key).states)
    assert young[it.id] == "active"
    assert old[it.id] == "archived"
    assert it.id in res.catalog  # nothing was deleted


def test_before_and_after_feedback_importance_changes_and_the_earlier_view_does_not() -> None:
    res = run("changing", "f")
    assert res.ledger is not None
    e = next(e for e in res.ledger.all() if e.kind == "used")
    it = res.catalog[e.item]

    def importance(cutoff: float) -> str:
        a = memory_autopsy(world(), F, res, claims(), it.key, cutoff)
        signals = next(link for link in a.links if link.step == "signals")
        return dict(signals.detail)["importance"]

    assert importance(e.recorded_at) != importance(e.recorded_at + 0.5)  # the use is visible after
    assert importance(e.recorded_at) == importance(e.recorded_at)


def test_contradictory_evidence_arriving_later_is_absent_before_and_present_after() -> None:
    res = run("changing", "f")
    assert res.ledger is not None
    e = next(e for e in res.ledger.all() if e.kind == "contradiction_discovered")
    it = res.catalog[e.item]
    before = memory_autopsy(world(), A, run("changing", "a"), claims(), it.key, e.recorded_at)
    after = memory_autopsy(world(), A, run("changing", "a"), claims(), it.key, e.recorded_at + 0.01)
    con = {link.step: link for link in after.links}["contradictions"]
    assert con.refs
    if before.answer is not None and any(c.item == it.id for c in before.candidates):
        assert (
            not {link.step: link for link in before.links}["contradictions"].refs
            or before.answer != it.value
        )


def test_stale_exposure_is_reported_from_evidence_only() -> None:
    heavy = System(name="r", schedule=KEEP, rank_weight=1.0)  # importance outweighs freshness
    res = simulate(world(), heavy, claims())
    late = snapshot(res, heavy, 300.5)
    early = snapshot(res, heavy, 5.0)
    assert late.stale_exposed  # a superseded value is still what the system answers
    latest = snapshot(run("changing", "a"), A, 300.5)
    assert not latest.stale_exposed  # "latest wins" cannot answer with a superseded item
    assert not early.stale_exposed
    assert late.digest != early.digest


def test_autopsy_is_complete_verified_and_deterministic() -> None:
    res = run("changing", "f")
    a = res.audits[300]
    one = memory_autopsy(world(), F, res, claims(), a.key, a.t, "audit")
    assert one.digest == memory_autopsy(world(), F, res, claims(), a.key, a.t, "audit").digest
    assert [x.step for x in one.links] == list(MEMORY_LINKS)
    assert one.complete
    assert one.missing == ()
    assert verify_autopsy(one, world(), F, res, claims()) == []
    ranked = [c for c in one.candidates if c.rank]
    assert ranked
    assert min(ranked, key=lambda c: c.rank or 0).value == one.answer
    assert dict(one.links[0].detail)["item"] == a.item
    assert all("truth" not in link.step for link in one.links)  # hidden truth stays in analysis
    assert dict(one.analysis)["truth"] == a.truth


def test_missing_provenance_is_reported_not_guessed() -> None:
    res = run("changing", "f")
    a = next(x for x in res.audits if x.item)
    reports = {r.id: r for r in world().reports if r.id != res.catalog[a.item or ""].report}
    one = memory_autopsy(world(), F, res, claims(), a.key, a.t, "audit", reports)
    assert not one.complete
    assert any(m.startswith("experience:") for m in one.missing)
    assert {x.step: x for x in one.links}["experience"].status == "missing"


def test_a_key_with_no_retrievable_item_is_explained_as_such() -> None:
    sysm = System(name="c", schedule=age_window(30))
    res = simulate(world("stable"), sysm, claims("stable"))
    a = next(x for x in res.audits if x.item is None and x.t > 100)
    one = memory_autopsy(world("stable"), sysm, res, claims("stable"), a.key, a.t)
    assert one.answer is None
    assert not one.complete
    assert any("no retrievable item" in m for m in one.missing)
    assert all(c.state == "archived" for c in one.candidates)


def test_gold_ingestion_marks_resolution_not_applicable() -> None:
    sysm = System(name="i", schedule=KEEP, ingest="gold")
    res = simulate(world(), sysm)
    a = next(x for x in res.audits if x.item)
    one = memory_autopsy(world(), sysm, res, None, a.key, a.t)
    assert {x.step: x for x in one.links}["resolution"].status == "not_applicable"
    assert one.complete


def test_verification_detects_a_tampered_autopsy() -> None:
    res = run("changing", "f")
    a = next(x for x in res.audits if x.item)
    one = memory_autopsy(world(), F, res, claims(), a.key, a.t)
    links = list(one.links)
    links[8] = links[8].model_copy(update={"refs": ("r:does-not-exist",)})
    bad = one.model_copy(update={"links": tuple(links)})
    problems = verify_autopsy(bad, world(), F, res, claims())
    assert any("does not resolve" in p for p in problems)
    assert any("different record" in p for p in problems)


def test_after_cutoff_evidence_is_shown_but_not_used() -> None:
    res = run("changing", "a")
    a = res.audits[40]
    early = memory_autopsy(world(), A, res, claims(), a.key, a.t)
    later = {x.step: x for x in early.after_cutoff}["later_evidence"]
    assert later.refs  # something was recorded afterwards
    assert set(later.refs).isdisjoint({c.item for c in early.candidates})


def test_counterfactual_holds_evidence_fixed_and_explains_each_change() -> None:
    res = run("changing", "f")
    before = ledger_digest(res)
    alt = F.model_copy(update={"rank_weight": 1.0})
    cf = counterfactual(world(), F, alt, res, claims())
    assert cf.evidence == before == ledger_digest(res)
    assert cf.changed_field == "rank_weight"
    assert 0 < cf.changed < cf.decisions
    assert len(cf.changes) == cf.changed
    assert all(c.why.detail for c in cf.changes)
    assert sum(n for _, n in cf.by_why) == cf.changed
    assert "not a causal" in INTERPRETATION
    assert cf.interpretation == INTERPRETATION
    assert cf.gained + cf.lost <= cf.changed
    assert cf.fixed_vs_resimulated > 0  # the feedback loop makes re-simulation differ


def test_counterfactual_of_a_schedule_reports_retention_reasons() -> None:
    res = run("changing", "f")
    b = System(name="b", schedule=KEEP, rank_weight=0.5)
    keep = simulate(world(), b, claims())
    cf = counterfactual(world(), b, F, keep, claims())
    assert cf.changed_field == "schedule"
    assert dict(cf.by_why).get("retention", 0) > 0
    assert res.items == keep.items


def test_counterfactual_of_an_importance_component_is_one_change() -> None:
    from memoria.adaptive import ablate

    res = run("changing", "f")
    alt = F.model_copy(update={"importance": ablate(F.importance, "correction")})
    cf = counterfactual(world(), F, alt, res, claims())
    assert cf.changed_field == "importance"
    assert cf.changed > 0
    assert {c.why.kind for c in cf.changes} <= {"importance", "retention"}


def test_a_counterfactual_changes_exactly_one_thing() -> None:
    res = run("changing", "f")
    with pytest.raises(ValueError, match="exactly one"):
        counterfactual(
            world(), F, F.model_copy(update={"rank_weight": 1.0, "schedule": KEEP}), res, claims()
        )
    with pytest.raises(ValueError, match="exactly one"):
        counterfactual(world(), F, F.model_copy(update={"ingest": "gold"}), res, claims())
    with pytest.raises(ValueError, match="exactly one"):
        counterfactual(world(), F, F, res, claims())


# --- substrate replay (Phases 1-7) -------------------------------------------------------------

EMB = HashedNgramEmbedder()


def _log(items: Sequence[tuple[float, str, str]], statement: bool = False) -> MemoryLog:
    log = MemoryLog(":memory:")
    for d, source, content in sorted(items):
        e = Experience(source=source, content=content, occurred_at=day(d))
        form(log, e, StatementPolicy() if statement else EpisodicPolicy(), recorded_at=day(d))
    return log


def test_substrate_state_before_and_after_a_correction_and_a_forget() -> None:
    with _log(
        [(1, "s", "set a.x = 1"), (5, "s", "correct a.x = 2"), (9, "s", "forget a.x")], True
    ) as log:
        digest = log_digest(log)
        contents = [
            [m.content for m in substrate_state(log, day(k), day(k)).memories] for k in (2, 6, 10)
        ]
        assert contents == [["a.x = 1"], ["a.x = 2"], []]
        assert log_digest(log) == digest  # replay does not touch the log


def test_substrate_consolidation_replays_before_and_after_new_evidence() -> None:
    items = [
        (1, "s1", "Ana lives in Paris"),
        (2, "s2", "Ana lives in Paris"),
        (40, "s3", "Ben works at Acme"),
    ]
    pol = ConsolidationPolicy.model_validate(
        {"name": "t", "version": "1", "every_days": 30, "regime": "exact"}
    )
    with _log(items) as log:
        digest = log_digest(log)
        early = substrate_hierarchy(log, pol, day(10), EMB)
        late = substrate_hierarchy(log, pol, day(60), EMB)
        assert early.digest != late.digest
        assert len(late.memories) > len(early.memories)
        assert log_digest(log) == digest
    with _log(items[:2]) as short:  # only what was known at day 10
        assert substrate_hierarchy(short, pol, day(10), EMB).digest == early.digest
