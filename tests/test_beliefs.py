import hashlib
import random
from datetime import datetime

import pytest
from pydantic import ValidationError

from memoria.belief_lab import with_signals
from memoria.beliefs import (
    POLICIES,
    UNCERTAINTY_DEFS,
    BeliefPolicy,
    BeliefState,
    EvidenceItem,
    KeyBeliefs,
    Uncertainty,
    arrival_hash,
    check_arithmetic,
    evidence_from_statement,
    evidence_set_hash,
)
from memoria.comparison import normalize
from memoria.core import EpistemicStatus
from memoria.revision import (
    EvidenceLedger,
    LedgerRun,
    SelectivePolicy,
    decide,
    historical_mismatches,
    reconstruct,
    replay,
    run_ledger,
    violations,
)
from memoria.scenarios import day
from memoria.sources import SourceClass, SourceModel

SM = SourceModel(
    name="s",
    version="1",
    classes=(
        SourceClass(name="clinic", alpha=9, beta=1),
        SourceClass(name="email", alpha=8, beta=2),
        SourceClass(name="forum", alpha=3, beta=7),
    ),
    copies=(("forum:c1", "forum:o"), ("forum:c2", "forum:o"), ("forum:c3", "forum:o")),
)
KEY = "ana.home"
SEL = SelectivePolicy(name="t", answer_at=0.6, uncertain_at=0.35)


def ev(n: int, value: str | None, occ: float, rec: float | None, src: str, *, verb: str = "set",
       key: str = KEY) -> EvidenceItem:  # fmt: skip
    return EvidenceItem(
        id="sha256:" + hashlib.sha256(f"{n}".encode()).hexdigest(), key=key, verb=verb,  # type: ignore[arg-type]
        value=None if value is None else " ".join(normalize(value)), raw=value,
        occurred_at=day(occ), recorded_at=day(occ if rec is None else rec), source=src,
    )  # fmt: skip


def run(items: list[EvidenceItem], policy: str = "A:evidence-count", **kw: object) -> LedgerRun:
    return run_ledger(items, POLICIES[policy], SM, **kw)  # type: ignore[arg-type]


def states(r: LedgerRun, key: str = KEY) -> dict[str, str]:
    kb = r.final(key)
    assert kb is not None
    return {b.value: b.state.value for b in kb.beliefs}


PARIS_BERLIN = [
    ev(1, "Paris", 0, None, "clinic:a"),
    ev(2, "Paris", 5, None, "email:b"),
    ev(3, "Berlin", 60, None, "clinic:a"),
]


# --- states and transitions ----------------------------------------------------------------------


def test_temporal_change_is_supersession_not_contradiction() -> None:
    r = run(PARIS_BERLIN)
    assert states(r) == {"paris": "superseded", "berlin": "supported"}
    kb = r.final(KEY)
    assert kb is not None
    paris = next(b for b in kb.beliefs if b.value == "paris")
    berlin = next(b for b in kb.beliefs if b.value == "berlin")
    assert paris.valid_to == berlin.valid_from == day(60)  # Paris held for the earlier interval
    assert paris.supporting  # history kept ...
    assert not paris.contradicting  # ... and no conflict recorded
    assert berlin.predecessors == (paris.id,)
    assert [e.reason for e in r.ledger.events] == [
        "first_evidence",
        "reinforced",
        "change_accepted",
    ]
    # An answer about the earlier period is still Paris, with the belief that was valid then.
    d = decide(KEY, r.at(KEY, day(70)), day(20), day(70), SEL)
    assert d.values == ("paris",)
    assert d.state is BeliefState.SUPERSEDED


def test_unknown_before_evidence_and_after_withdrawal() -> None:
    r = run(PARIS_BERLIN[:1])
    assert decide(KEY, r.at(KEY, day(-1)), day(1), day(-1), SEL).reason == "no_belief"
    withdrawn = run(PARIS_BERLIN[:1], withdrawals=[(day(10), PARIS_BERLIN[0].id)])
    kb = withdrawn.final(KEY)
    assert kb is not None
    assert not kb.beliefs
    (tomb,) = kb.tombstones
    assert tomb.state is BeliefState.UNKNOWN
    assert tomb.reason == "evidence_withdrawn"
    assert PARIS_BERLIN[0].id in tomb.withdrawn  # what it lost is recorded
    assert withdrawn.ledger.evidence[0].id in {i.id for i in PARIS_BERLIN}  # never deleted
    assert violations(withdrawn) == []
    assert withdrawn.ledger.events[-1].reason == "evidence_withdrawn"


def test_concurrent_disagreement_is_contested_and_keeps_both_sides() -> None:
    items = [ev(1, "Paris", 0, None, "email:a"), ev(2, "Rome", 0.3, None, "email:b")]
    r = run(items)
    assert states(r) == {"paris": "contested", "rome": "contested"}
    kb = r.final(KEY)
    assert kb is not None
    for b in kb.beliefs:
        assert b.contradicting  # neither side is called false
        assert b.reason == "conflicting_evidence"
    assert {i for b in kb.beliefs for i in b.supporting} == {i.id for i in items}
    assert decide(KEY, r.at(KEY, day(5)), day(2), day(5), SEL).mode == "competing"


def test_weighted_vote_rejects_without_declaring_false() -> None:
    items = [ev(1, "Paris", 0, None, "email:a"), ev(2, "Paris", 0.1, None, "email:b"),
             ev(3, "Rome", 0.3, None, "email:c")]  # fmt: skip
    r = run(items)
    assert states(r) == {"paris": "supported", "rome": "rejected"}
    kb = r.final(KEY)
    assert kb is not None
    rome = next(b for b in kb.beliefs if b.value == "rome")
    assert rome.reason == "lost_weighted_vote"
    assert rome.supporting == (items[2].id,)  # its evidence is intact


def test_correction_replaces_retroactively_and_keeps_the_corrected_belief() -> None:
    items = [
        ev(1, "Paris", 0, None, "forum:f"),
        ev(2, "Berlin", 6, None, "clinic:a", verb="correct"),
    ]
    r = run(items)
    assert states(r) == {"paris": "corrected", "berlin": "supported"}
    kb = r.final(KEY)
    assert kb is not None
    paris = next(b for b in kb.beliefs if b.value == "paris")
    berlin = next(b for b in kb.beliefs if b.value == "berlin")
    assert berlin.valid_from == paris.valid_from == day(0)  # retroactive over the same interval
    assert paris.supporting == (items[0].id,)
    assert r.ledger.events[-1].reason == "correction_applied"
    # A corrected belief never answers.
    assert decide(KEY, r.at(KEY, day(9)), day(3), day(9), SEL).values == ("berlin",)


def test_a_retraction_closes_the_belief_without_erasing_it() -> None:
    items = [ev(1, "Paris", 0, None, "clinic:a"), ev(2, None, 30, None, "clinic:a", verb="forget")]
    r = run(items)
    kb = r.final(KEY)
    assert kb is not None
    (b,) = kb.beliefs
    assert (b.state, b.reason, b.valid_to) == (BeliefState.SUPERSEDED, "retracted", day(30))
    assert kb.retractions == (items[1].id,)
    assert violations(r) == []


def test_corroboration_policy_declines_to_adopt_thin_support() -> None:
    r = run(PARIS_BERLIN[2:], "E:corroboration")
    assert states(r) == {"berlin": "unresolved"}
    assert decide(KEY, r.final(KEY), day(70), day(70), SEL).mode == "require_evidence"


def test_unconfirmed_change_stays_contested_until_a_second_root_confirms() -> None:
    items = [ev(1, "Paris", 0, None, "clinic:a"), ev(2, "Paris", 1, None, "email:b"),
             ev(3, "Berlin", 60, None, "clinic:a")]  # fmt: skip
    r = run(items, "H:conservative")
    assert states(r) == {"paris": "contested", "berlin": "contested"}
    kb = r.final(KEY)
    assert kb is not None
    paris = next(b for b in kb.beliefs if b.value == "paris")
    assert paris.valid_to is None  # the policy does not yet accept that the value ended
    assert paris.observed_to == day(60)  # the evidence's own end is kept
    confirmed = run([*items, ev(4, "Berlin", 61, None, "email:c")], "H:conservative")
    assert states(confirmed)["berlin"] in ("supported", "unresolved")
    assert states(confirmed)["paris"] == "superseded"


def test_hysteresis_makes_the_final_belief_depend_on_arrival_order() -> None:
    policy = POLICIES["A:evidence-count"].model_copy(update={"hysteresis": 0.5})
    x = [ev(1, "Paris", 0.0, 0.0, "email:a"), ev(2, "Paris", 0.1, 0.1, "email:b")]
    y = [ev(3, "Rome", 0.2, 0.2, "email:c"), ev(4, "Rome", 0.3, 0.3, "email:d"),
         ev(5, "Rome", 0.4, 0.4, "email:e")]  # fmt: skip

    def final(first: list[EvidenceItem], second: list[EvidenceItem]) -> dict[str, str]:
        late = [i.model_copy(update={"recorded_at": day(10 + n)}) for n, i in enumerate(second)]
        r = run_ledger([*first, *late], policy, SM)
        kb = r.final(KEY)
        assert kb is not None
        return {b.value: b.state.value for b in kb.beliefs}

    incumbent_x, incumbent_y = final(x, y), final(y, x)
    assert incumbent_x == {"paris": "supported", "rome": "rejected"}
    assert incumbent_y == {"paris": "rejected", "rome": "supported"}
    # Without hysteresis the same evidence converges to the same answer in either order.
    plain = POLICIES["A:evidence-count"]
    a = run_ledger([*x, *[i.model_copy(update={"recorded_at": day(10)}) for i in y]], plain, SM)
    b = run_ledger([*y, *[i.model_copy(update={"recorded_at": day(10)}) for i in x]], plain, SM)
    assert a.final(KEY).content_fingerprint() == b.final(KEY).content_fingerprint()  # type: ignore[union-attr]


def test_persistent_conflict_becomes_unresolved() -> None:
    policy = POLICIES["A:evidence-count"].model_copy(update={"unresolve_after": 3, "margin": 1.0})
    items = [ev(1, "Paris", 0, 0, "email:a"), ev(2, "Rome", 0.2, 1, "email:b"),
             ev(3, "Paris", 0.3, 2, "chat:c"), ev(4, "Rome", 0.4, 3, "chat:d")]  # fmt: skip
    r = run_ledger(items, policy, SM)
    reasons = [b.reason for b in r.final(KEY).beliefs]  # type: ignore[union-attr]
    assert set(reasons) == {"persistent_conflict"}
    assert set(states(r).values()) == {"unresolved"}


# --- copies and independence ---------------------------------------------------------------------


def test_copies_masquerade_as_corroboration_only_for_policies_that_count_them() -> None:
    items = (
        [ev(1, "Rome", 0, None, "forum:o")]
        + [ev(10 + i, "Rome", 0.01 * i, None, f"forum:c{i}") for i in (1, 2, 3)]
        + [ev(2, "Paris", 0.2, None, "clinic:a")]
    )
    naive, careful = run(items, "A:evidence-count"), run(items, "E:corroboration")
    nk, ck = naive.final(KEY), careful.final(KEY)
    assert nk is not None
    assert ck is not None
    rome_n = next(b for b in nk.beliefs if b.value == "rome")
    rome_c = next(b for b in ck.beliefs if b.value == "rome")
    assert (rome_n.naive_count, rome_n.independent_count) == (4, 1)
    assert states(naive)["rome"] == "supported"  # four copies outvote one independent source
    assert rome_c.weight == pytest.approx(1.0)  # one root counts once
    assert [c.reason for c in rome_c.contributions].count("duplicate_lineage") == 3
    assert states(careful)["rome"] != "supported"


def test_provenance_depth_discounts_copies() -> None:
    items = [ev(1, "Rome", 0, None, "forum:o"), ev(2, "Rome", 0.1, None, "forum:c1")]
    kb = run(items, "G:provenance-depth").final(KEY)
    assert kb is not None
    (b,) = kb.beliefs
    depths = sorted(c.depth for c in b.contributions)
    assert depths == [0.7, 1.0]
    assert b.raw_weight == pytest.approx(1.7)


# --- derived evidence, lineage and epistemic status ----------------------------------------------


def derived(n: int, value: str, lineage: list[EvidenceItem]) -> EvidenceItem:
    return EvidenceItem(
        id="sha256:" + hashlib.sha256(f"d{n}".encode()).hexdigest(), key=KEY, verb="set",
        value=" ".join(normalize(value)), raw=value, occurred_at=day(0), recorded_at=day(20),
        source=f"derived:m{n}", status=EpistemicStatus.DERIVED, multiplicity=len(lineage),
        lineage=tuple(sorted((i.id, i.source) for i in lineage)),
    )  # fmt: skip


def test_a_belief_from_derived_evidence_is_never_observed() -> None:
    raw = [ev(1, "Paris", 0, None, "clinic:a"), ev(2, "Paris", 0.1, None, "email:b")]
    d = derived(1, "Paris", raw)
    only = run([d])
    (b,) = only.final(KEY).beliefs  # type: ignore[union-attr]
    assert b.epistemic is EpistemicStatus.DERIVED
    mixed = run([d, raw[0]])
    (m,) = mixed.final(KEY).beliefs  # type: ignore[union-attr]
    assert m.epistemic is EpistemicStatus.OBSERVED  # the strongest admissible evidence permits it
    assert violations(only) == []
    assert violations(mixed) == []
    with pytest.raises(ValidationError, match="only derived evidence has lineage"):
        EvidenceItem(**{**raw[0].model_dump(), "lineage": ((raw[1].id, "email:b"),)})


def test_lineage_awareness_stops_a_summary_outliving_its_evidence() -> None:
    raw = [ev(1, "Paris", 0, None, "clinic:a"), ev(2, "Paris", 0.1, None, "email:b"),
           ev(3, "Paris", 0.2, None, "chat:c")]  # fmt: skip
    d = derived(1, "Paris", raw)
    gone = [(day(30), i.id) for i in raw]
    naive = run([*raw, d], withdrawals=gone)
    aware_policy = with_signals(POLICIES["A:evidence-count"], "lineage")
    aware = run_ledger([*raw, d], aware_policy, SM, withdrawals=gone)
    (nb,) = naive.final(KEY).beliefs  # type: ignore[union-attr]
    (ab,) = aware.final(KEY).beliefs  # type: ignore[union-attr]
    assert nb.state is BeliefState.SUPPORTED
    assert nb.weight == 3.0  # the summary still counts its three withdrawn experiences
    assert ab.weight == 0.0
    assert [c.reason for c in ab.contributions] == ["lineage_unavailable"]
    assert ab.state is BeliefState.UNRESOLVED


# --- the ledger: replay, reconstruction, audit ---------------------------------------------------


def test_replay_reproduces_every_event_and_reconstruction_never_sees_the_future() -> None:
    items = [
        ev(1, "Paris", 0, 0, "clinic:a"), ev(2, "Rome", 0.3, 40, "forum:f"),  # arrives late
        ev(3, "Berlin", 60, 61, "clinic:a"), ev(4, "Paris", 0.2, 65, "email:b"),
    ]  # fmt: skip
    r = run(items, "B:source-weighted")
    assert replay(r.ledger, POLICIES["B:source-weighted"], SM) == (True, None)
    assert historical_mismatches(r, [day(1), day(45), day(100)]) == 0
    early = reconstruct(r.ledger, POLICIES["B:source-weighted"], SM, day(3))
    assert {i.id for i in early.ledger.evidence} == {items[0].id}
    assert r.at(KEY, day(3)).fingerprint() == early.final(KEY).fingerprint()  # type: ignore[union-attr]
    late = [e for e in r.ledger.events if e.late]
    assert [e.evidence for e in late] == [items[3].id]  # it occurred before evidence already held
    assert EvidenceLedger.model_validate_json(r.ledger.canonical()) == r.ledger


def test_replay_needs_the_policy_the_ledger_names() -> None:
    r = run(PARIS_BERLIN)
    with pytest.raises(ValueError, match="names"):
        replay(r.ledger, POLICIES["B:source-weighted"], SM)
    tampered = r.ledger.model_copy(
        update={
            "events": (
                r.ledger.events[0].model_copy(update={"reason": "no_change"}),
                *r.ledger.events[1:],
            )
        }
    )
    assert replay(tampered, POLICIES["A:evidence-count"], SM) == (False, 0)


def test_the_audit_detects_each_kind_of_tampering() -> None:
    items = [ev(1, "Paris", 0, None, "clinic:a"), ev(2, "Paris", 1, None, "email:b")]
    r = run(items, "B:source-weighted")
    assert violations(r) == []
    kb = r.history[KEY][0]
    (b,) = kb.beliefs
    # confidence up with no linked evidence event
    boosted = b.model_copy(update={"score": 0.99})
    bad = kb.model_copy(update={"beliefs": (boosted,)})
    r.history[KEY][0] = bad
    codes = {v.code for v in violations(r)}
    assert "V-ARITH" in codes
    # a belief stronger than its evidence permits
    r2 = run([derived(1, "Paris", items)])
    (d,) = r2.history[KEY][0].beliefs
    r2.history[KEY][0] = r2.history[KEY][0].model_copy(
        update={"beliefs": (d.model_copy(update={"epistemic": EpistemicStatus.OBSERVED}),)}
    )
    assert "V-EPISTEMIC" in {v.code for v in violations(r2)}
    # evidence dropped from the beliefs
    r3 = run(items)
    r3.history[KEY][1] = r3.history[KEY][1].model_copy(
        update={
            "beliefs": (
                r3.history[KEY][1].beliefs[0].model_copy(update={"supporting": (items[0].id,)}),
            )
        }
    )
    assert "V-EVIDENCE" in {v.code for v in violations(r3)}
    # a belief that cites evidence which has not arrived yet
    r4 = run([ev(1, "Paris", 0, 0, "clinic:a"), ev(2, "Paris", 1, 50, "email:b")])
    first = r4.history[KEY][0]
    leaky = first.beliefs[0].model_copy(
        update={"supporting": tuple(sorted(i.id for i in r4.ledger.evidence))}
    )
    r4.history[KEY][0] = first.model_copy(update={"beliefs": (leaky,)})
    assert "V-FUTURE" in {v.code for v in violations(r4)}
    # sole support withdrawn but still SUPPORTED
    r5 = run(items[:1], withdrawals=[(day(5), items[0].id)])
    kb5 = r5.history[KEY][1]
    ghost = r5.history[KEY][0].beliefs[0]
    r5.history[KEY][1] = kb5.model_copy(update={"beliefs": (ghost,), "tombstones": ()})
    assert {"V-WITHDRAW", "V-EVIDENCE"} & {v.code for v in violations(r5)}


def test_confidence_rises_only_with_a_recorded_evidence_event() -> None:
    r = run([ev(1, "Paris", 0, None, "email:a"), ev(2, "Paris", 1, None, "email:b"),
             ev(3, "Paris", 2, None, "email:c")])  # fmt: skip
    scores = [e.transitions[-1].score_after for e in r.ledger.events]
    assert scores == sorted(
        x for x in scores if x is not None
    )  # each step up is an arrival that supports the belief
    assert all(e.kind == "arrive" for e in r.ledger.events)
    assert violations(r) == []


def test_arithmetic_of_every_belief_follows_from_its_contributions() -> None:
    r = run([*PARIS_BERLIN, ev(9, "Rome", 60.2, None, "forum:f")], "B:source-weighted")
    for kb in r.history[KEY]:
        for b in kb.beliefs:
            check_arithmetic(b)
            assert b.raw_weight == pytest.approx(
                sum(c.weight for c in b.contributions if c.counted), abs=1e-9
            )
            for c in b.contributions:
                assert c.weight == pytest.approx(
                    c.multiplicity * c.trust * c.recency * c.depth * c.staleness, abs=1e-9
                )


# --- policies, uncertainty, decisions ------------------------------------------------------------


def test_policies_are_content_addressed_and_expose_their_signals() -> None:
    assert len({p.digest for p in POLICIES.values()}) == 8
    assert POLICIES["A:evidence-count"].digest == POLICIES["A:evidence-count"].model_copy().digest
    assert [s.name for s in POLICIES["H:conservative"].signals] == [
        "count",
        "independence",
        "trust",
    ]
    assert POLICIES["B:source-weighted"].signal("trust") == {"mode": "learned"}
    assert POLICIES["A:evidence-count"].signal("trust") is None
    with pytest.raises(ValidationError, match="include count"):
        BeliefPolicy(**{**POLICIES["A:evidence-count"].model_dump(), "signals": ()})


def test_every_uncertainty_component_is_defined_and_bounded() -> None:
    assert set(UNCERTAINTY_DEFS) == set(Uncertainty.model_fields)
    for means, computed, evidence, not_ in UNCERTAINTY_DEFS.values():
        assert all([means, computed, evidence])
        assert not_.startswith("not")
    r = run([*PARIS_BERLIN, ev(9, "Rome", 60.2, None, "forum:f")], "B:source-weighted")
    for kb in r.history[KEY]:
        for b in kb.beliefs:
            assert all(0.0 <= v <= 1.0 for v in b.uncertainty.model_dump().values())
    (single,) = run(PARIS_BERLIN[:1]).final(KEY).beliefs  # type: ignore[union-attr]
    assert single.uncertainty.contradiction == 0.0
    assert single.uncertainty.retrieval == 0.0


def beliefs_of(r: LedgerRun, key: str = KEY):  # type: ignore[no-untyped-def]
    kb = r.final(key)
    assert kb is not None
    return kb.beliefs


def test_uncertainty_tracks_what_it_is_defined_to_track() -> None:
    conflict = run([ev(1, "Paris", 0, None, "email:a"), ev(2, "Rome", 0.2, None, "email:b")])
    assert all(b.uncertainty.contradiction == pytest.approx(0.5) for b in beliefs_of(conflict))
    withdrawn = ev(2, "Paris", 0.2, None, "email:b")
    lost = run(
        [ev(1, "Paris", 0, None, "email:a"), withdrawn], withdrawals=[(day(5), withdrawn.id)]
    )
    (b,) = beliefs_of(lost)
    assert b.uncertainty.retrieval == 0.5
    (fresh,) = beliefs_of(run([ev(1, "Paris", 0, None, "email:a")]))
    assert fresh.uncertainty.temporal == 0.0  # an event at the evidence's own moment
    assert fresh.uncertainty.identity == 0.0
    assert fresh.uncertainty.epistemic == pytest.approx(0.5)  # one unit of evidence, prior mass 1


def test_decisions_cover_every_mode() -> None:
    strong = run([ev(i, "Paris", 0.1 * i, None, f"email:{i}") for i in range(5)])
    assert decide(KEY, strong.final(KEY), day(3), day(9), SEL).mode == "answer"
    weak = run(PARIS_BERLIN[:1])
    assert decide(KEY, weak.final(KEY), day(3), day(9), SEL).mode == "answer_with_uncertainty"
    barely = SelectivePolicy(name="t", answer_at=0.9, uncertain_at=0.8)
    assert decide(KEY, weak.final(KEY), day(3), day(9), barely).mode == "abstain"
    need = SelectivePolicy(name="t", answer_at=0.3, uncertain_at=0.1, min_roots=2)
    assert decide(KEY, weak.final(KEY), day(3), day(9), need).mode == "require_evidence"
    assert decide(KEY, weak.final(KEY), day(-3), day(9), SEL).reason == "no_belief"
    with pytest.raises(ValidationError):
        SelectivePolicy(name="t", answer_at=0.3, uncertain_at=0.5)
    # A calibrated value moves the belief across a threshold; the recorded score stays raw.
    b = beliefs_of(weak)[0]
    d = decide(KEY, weak.final(KEY), day(3), day(9), SEL, {b.id: 0.99})
    assert d.mode == "answer"
    assert d.score == b.score


# --- evidence and identity -----------------------------------------------------------------------


def test_evidence_is_validated_and_free_text_carries_none() -> None:
    assert (
        evidence_from_statement("sha256:" + "1" * 64, "Ana lives in Paris.", day(0), day(0), "x:y")
        is None
    )
    item = evidence_from_statement(
        "sha256:" + "1" * 64, "set ana.home = Paris", day(0), day(1), "x:y"
    )
    assert item is not None
    assert (item.key, item.value, item.raw) == ("ana.home", "paris", "Paris")
    with pytest.raises(ValidationError, match="recorded before"):
        ev(1, "Paris", 5, 1, "x:y")
    with pytest.raises(ValidationError, match="forget"):
        ev(1, "Paris", 0, 0, "x:y", verb="forget")


def test_evidence_and_arrival_identity_are_separate() -> None:
    a = [ev(1, "Paris", 0, 0, "x:a"), ev(2, "Rome", 1, 1, "x:b")]
    b = [a[0], a[1].model_copy(update={"recorded_at": day(30)})]
    assert evidence_set_hash(a) == evidence_set_hash(b)
    assert arrival_hash(a) == arrival_hash(b)  # same order (1 then 2)
    c = [a[0].model_copy(update={"recorded_at": day(40)}), a[1]]
    assert evidence_set_hash(a) == evidence_set_hash(c)
    assert arrival_hash(a) != arrival_hash(c)


def test_ledger_identity_names_everything_that_determines_the_run() -> None:
    r = run(PARIS_BERLIN, "A:evidence-count")
    ident = r.ledger.identity
    assert ident.policy == POLICIES["A:evidence-count"].digest
    assert ident.temporal == POLICIES["A:evidence-count"].temporal.digest
    assert ident.sources == SM.digest
    assert ident.evidence == evidence_set_hash(PARIS_BERLIN)
    other = run(PARIS_BERLIN, "B:source-weighted")
    assert other.ledger.digest != r.ledger.digest
    assert run(PARIS_BERLIN).ledger.digest == r.ledger.digest  # deterministic


# --- invariants over random streams --------------------------------------------------------------


def random_stream(seed: int) -> list[EvidenceItem]:
    rng = random.Random(seed)
    sources = ["clinic:a", "email:b", "email:c", "forum:f", "forum:o", "forum:c1", "chat:d"]
    values = ["Paris", "Rome", "Berlin"]
    items = []
    for n in range(rng.randint(3, 14)):
        occ = rng.choice([0, 0.2, 0.5, 3, 10, 40, 41, 90]) + rng.random() * 0.3
        rec = occ + rng.choice([0, 0, 1, 5, 30])
        verb = rng.choice(["set"] * 9 + ["correct"])
        items.append(
            ev(1000 * seed + n, rng.choice(values), occ, rec, rng.choice(sources), verb=verb)
        )
    return items


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_invariants_hold_on_random_streams(name: str) -> None:
    for seed in range(12):
        items = random_stream(seed)
        r = run(items, name)
        assert violations(r) == [], (name, seed)
        assert replay(r.ledger, POLICIES[name], SM) == (True, None)
        last = max(i.recorded_at for i in items)
        assert (
            historical_mismatches(
                r, [day(0.1), datetime.fromtimestamp(last.timestamp(), tz=last.tzinfo)]
            )
            == 0
        )
        for kb in r.history[KEY]:
            assert {i for b in kb.beliefs for i in b.supporting} | set(kb.retractions) <= {
                i.id for i in items
            }


@pytest.mark.parametrize(
    "name",
    ["A:evidence-count", "E:corroboration", "F:contradiction-penalty", "G:provenance-depth"],
)
def test_policies_without_learning_or_hysteresis_converge_across_arrival_orders(name: str) -> None:
    for seed in range(8):
        items = random_stream(seed)
        rng = random.Random(seed)
        finals = set()
        for _ in range(4):
            shuffled = [
                i.model_copy(
                    update={
                        "recorded_at": i.occurred_at.replace()
                        if False
                        else day(rng.random() * 200 + 100)
                    }
                )
                for i in items
            ]  # every order arrives after everything occurred
            kb = run(shuffled, name).final(KEY)
            finals.add(kb.content_fingerprint() if kb else None)
        assert len(finals) == 1, (name, seed)


def test_key_beliefs_fingerprint_ignores_nothing_it_should_keep() -> None:
    r = run(PARIS_BERLIN)
    kb = r.final(KEY)
    assert isinstance(kb, KeyBeliefs)
    changed = kb.model_copy(update={"conflict_events": 7})
    assert changed.fingerprint() != kb.fingerprint()
    assert changed.content_fingerprint() == kb.content_fingerprint()
