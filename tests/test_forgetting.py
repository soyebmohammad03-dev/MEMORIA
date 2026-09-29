from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from memoria.consolidation import ConsolidationPolicy, consolidate, invalidated
from memoria.consolidation_eval import importance
from memoria.core import Experience, Level
from memoria.embeddings import HashedNgramEmbedder
from memoria.forgetting import (
    ForgettingPolicy,
    ForgettingRecord,
    Selector,
    apply,
    forget,
    measure_forgetting,
    policy,
)
from memoria.formation import EpisodicPolicy, form
from memoria.hybrid import (
    RETRIEVABLE,
    Corpus,
    Engine,
    Exclusion,
    HybridQuery,
    HybridTrace,
    equal_weights,
)
from memoria.hybrid import policy as retrieval_policy
from memoria.scenarios import day
from memoria.store import MemoryLog

EMB = HashedNgramEmbedder()
ITEMS = (
    (0, "chat:a0", "set ana.home = Paris"),
    (2, "email:a2", "Ana lives in Paris."),
    (3, "email:a3", "set ana.home = Paris"),
    (40, "chat:a40", "set ana.home = Berlin"),
    (40, "forum:a40", "set ana.home = Rome"),
    (60, "chat:b60", "set ben.home = Oslo"),
    (61, "email:b61", "Ben lives in Oslo."),
    (70, "clinic:a70", "correct ana.home = Munich"),
    (90, "chat:b90", "set ben.home = Rome"),
)
HIER = ConsolidationPolicy(name="t", version="1", regime="temporal", every_days=30,
                           abstractions=("entity", "timeline"))  # fmt: skip


def build(items: Sequence[tuple[float, str, str]] = ITEMS, at: float = 100) -> tuple[Corpus, str]:
    with MemoryLog(":memory:") as log:
        for d, source, content in sorted(items, key=lambda x: (x[0], x[1])):
            form(log, Experience(source=source, content=content, occurred_at=day(d)),
                 EpisodicPolicy(), recorded_at=day(d))  # fmt: skip
        return Corpus.from_log(log, day(at)), log.export().decode()


def consolidated(at: float = 100) -> Corpus:
    base, _ = build(at=at)
    h = consolidate(base, HIER, day(at), EMB)
    return base.with_derived(h.memories, invalidated(h, base))


def digest(c: Corpus, source: str) -> str:
    return next(e.digest for e in c.entries if e.level is Level.L1
                and e.sources[0].source == source)  # fmt: skip


def state(r: ForgettingRecord, entry: str) -> str:
    return next(d.state for d in r.decisions if d.entry == entry)


QUERY = HybridQuery(text="Where does Ana live?", valid_at=day(95), known_at=day(100), limit=5,
                    key="ana.home")  # fmt: skip
POLICY = retrieval_policy("p", equal_weights(["attribute", "lexical"]), semantic_generator=False)


def test_policy_identity_and_closed_parameters() -> None:
    a = policy("a", "age", max_age_days=30)
    assert a.digest != policy("a", "age", max_age_days=31).digest
    assert a.digest == policy("a", "age", max_age_days=30).digest
    with pytest.raises(ValidationError, match="takes parameters"):
        policy("a", "age", capacity=3)
    with pytest.raises(ValidationError, match="soft suppression"):
        policy("a", "age", "suppress", max_age_days=3)
    with pytest.raises(ValidationError, match="selector"):
        ForgettingPolicy(name="s", version="1", rule="selective", action="forget")
    with pytest.raises(ValidationError, match="importance"):
        policy("i", "importance")


def test_forgetting_never_deletes_history() -> None:
    corpus, export = build()
    r = forget(corpus, policy("age", "age", max_age_days=50), day(100))
    after = apply(corpus, r)
    assert len(after.entries) == len(corpus.entries)
    assert [e.version for e in after.entries] == [e.version for e in corpus.entries]
    assert after.digest != corpus.digest  # the corpus identity names the intervention
    assert after.identity.forgetting == r.digest
    assert build()[1] == export  # the log is untouched (and rebuilds identically)
    assert {state(r, digest(corpus, "chat:a0")), state(r, digest(corpus, "chat:b90"))} == {
        "forgotten", "active"}  # fmt: skip
    assert len(r.decisions) == len(corpus.entries)  # every memory has a recorded decision
    assert r == forget(corpus, policy("age", "age", max_age_days=50), day(100))  # replay
    with pytest.raises(ValueError, match="once"):
        after.with_availability(r.states(), r.digest)
    with pytest.raises(ValueError, match="another corpus"):
        apply(after, r)


def test_hard_exclusion_is_recorded_and_cannot_be_bypassed() -> None:
    corpus, _ = build()
    target = digest(corpus, "chat:a40")
    r = forget(corpus, policy("s", "selective", selector=Selector(source="chat")), day(100))
    trace = Engine(apply(corpus, r), EMB).retrieve(POLICY, QUERY)
    c = trace.candidate(target)
    assert c.excluded is Exclusion.FORGOTTEN_BY_POLICY
    assert c.availability == "forgotten"
    assert target not in {x.version for x in trace.ranking}
    bypass = [x.model_copy(update={"availability": "active"}) if x.version == target else x
              for x in trace.candidates]  # fmt: skip
    with pytest.raises(ValidationError, match="exclusion does not follow"):
        HybridTrace.model_validate(trace.model_dump() | {
            "candidates": [x.model_dump() for x in bypass]})  # fmt: skip


def test_soft_suppression_is_a_recorded_signal() -> None:
    corpus, _ = build()
    r = forget(corpus, policy("r", "recency", "suppress", half_life_days=20, threshold=0.5),
               day(100))  # fmt: skip
    old = digest(corpus, "chat:a0")
    assert state(r, old) == "suppressed"
    d = next(x for x in r.decisions if x.entry == old)
    assert 0 < d.suppression <= 1
    assert float(dict(d.signals)["recency"]) < 0.5
    pol = retrieval_policy("p", equal_weights(["attribute", "lexical", "suppression"]),
                           semantic_generator=False)  # fmt: skip
    trace = Engine(apply(corpus, r), EMB).retrieve(pol, QUERY)
    ranked = trace.result(old)
    s = next(x for x in ranked.signals if x.signal == "suppression")
    assert s.raw == d.suppression
    assert dict(s.inputs)["availability"] == "suppressed"
    assert ranked.contribution("suppression").value < 0


def test_selective_forgetting_targets() -> None:
    corpus, _ = build()
    ana = forget(corpus, policy("e", "selective", selector=Selector(entity="ana")), day(100))
    hidden = {d.entry for d in ana.decisions if d.state == "forgotten"}
    assert digest(corpus, "email:a2") in hidden  # free text naming Ana
    assert digest(corpus, "chat:b60") not in hidden
    window = forget(
        corpus, policy("w", "selective", selector=Selector(start=day(30), end=day(65))), day(100)
    )
    assert {d.entry for d in window.decisions if d.state == "forgotten"} == {
        digest(corpus, s) for s in ("chat:a40", "forum:a40", "chat:b60", "email:b61")
    }
    notes = forget(corpus, policy("n", "selective", selector=Selector(kind="note")), day(100))
    assert {d.entry for d in notes.decisions if d.state == "forgotten"} == {
        digest(corpus, "email:a2"),
        digest(corpus, "email:b61"),
    }
    contested = forget(corpus, policy("c", "selective", selector=Selector(contradictory=True)),
                       day(100))  # fmt: skip
    assert digest(corpus, "forum:a40") in contested.unavailable()


def test_derived_memories_cascade_unless_retained() -> None:
    corpus = consolidated()
    sel = Selector(source="email")
    cascade = forget(corpus, policy("c", "selective", selector=sel), day(100))
    retain = forget(corpus, policy("r", "selective", selector=sel, derived="retain"), day(100))
    hit = [d for d in cascade.decisions if d.level is not Level.L1 and d.state == "forgotten"]
    assert hit
    assert all(d.reason == "cascade:evidence_unavailable" for d in hit)
    assert not [d for d in retain.decisions if d.level is not Level.L1 and d.state != "active"]
    assert set(retain.evidence_retained) == {d.memory_id for d in hit}
    m = measure_forgetting(corpus, retain)
    assert m.provenance_complete.numerator < m.provenance_complete.denominator
    assert measure_forgetting(corpus, cascade).provenance_complete.estimate == 1.0
    only = forget(corpus, policy("d", "selective",
                                 selector=Selector(levels=(Level.L2, Level.L3, Level.L4))),
                  day(100))  # fmt: skip
    assert not [d for d in only.decisions if d.level is Level.L1 and d.state != "active"]


def test_access_history_must_precede_the_intervention() -> None:
    corpus, _ = build()
    early, _ = build(at=95)
    t = Engine(early, EMB).retrieve(POLICY, QUERY.model_copy(update={"known_at": day(95)}))
    p = policy("a", "access", grace_days=10, min_access=1)
    r = forget(corpus, p, day(100), [t])
    assert r.accesses == (t.digest,)
    for d in r.decisions:
        accessed = d.entry in t.selected
        assert dict(d.signals)["accesses"] == int(accessed)
        old = float(dict(d.signals)["age_days"]) > 10
        assert (d.state == "forgotten") == (old and not accessed)
    later = Engine(corpus, EMB).retrieve(POLICY, QUERY)
    with pytest.raises(ValueError, match="precede"):
        forget(corpus, p, day(100), [later])


def test_rules_expose_their_signals() -> None:
    corpus, _ = build()
    rules = {
        "fifo": policy("f", "fifo", capacity=3),
        "validity": policy("v", "validity", grace_days=0),
        "contradiction": policy("c", "contradiction", forget_contested=0),
        "provenance": policy("p", "provenance", keep=1),
        "importance": policy("i", "importance", importance=importance()),
        "hybrid": policy("h", "hybrid", importance=importance(), grace_days=10,
                         max_age_days=50, min_access=1, min_votes=3),
    }  # fmt: skip
    out = {k: forget(corpus, p, day(100)) for k, p in rules.items()}
    assert sum(d.state != "active" for d in out["fifo"].decisions) == len(corpus.entries) - 3
    corrected = digest(corpus, "chat:a40")
    assert state(out["contradiction"], corrected) == "forgotten"  # replaced by the correction
    assert state(out["contradiction"], digest(corpus, "forum:a40")) == "forgotten"
    assert state(out["validity"], digest(corpus, "email:a2")) == "active"  # free text: unknown
    assert "validity:unknown_without_claim" in {d.reason for d in out["validity"].decisions}
    repeats = {d.entry: d for d in out["provenance"].decisions}
    assert repeats[digest(corpus, "chat:a0")].state == "active"  # the first report is kept
    assert repeats[digest(corpus, "email:a3")].state == "forgotten"  # a redundant copy
    votes = dict(out["hybrid"].decisions[0].signals)
    assert {"vote.age", "vote.access", "vote.contradiction", "vote.importance",
            "vote.validity"} <= set(votes)  # fmt: skip


def test_contested_claims_stay_visible_unless_the_policy_says_otherwise() -> None:
    early, _ = build(at=50)
    kept = forget(early, policy("c", "contradiction", forget_contested=0), day(50))
    lost = forget(early, policy("c", "contradiction", forget_contested=1), day(50))
    assert measure_forgetting(early, kept).contradiction_visibility.estimate == 1.0
    assert measure_forgetting(early, lost).contradiction_visibility.estimate == 0.0


def test_aggregates_survive_forgetting_raw_evidence() -> None:
    corpus, _ = build()
    r = forget(corpus, policy("s", "selective", preserve_aggregates=True,
                              selector=Selector(source="chat")), day(100))  # fmt: skip
    assert dict((k, (n, v)) for k, n, v in r.aggregates)["ana.home"] == (2, 2)


def test_precision_recall_and_collateral_against_a_target() -> None:
    corpus, _ = build()
    r = forget(corpus, policy("e", "selective", selector=Selector(entity="ana")), day(100))
    target = {e.digest for e in corpus.entries if e.claim and e.claim.key == "ana.home"}
    m = measure_forgetting(corpus, r, target)
    assert m.recall is not None
    assert m.recall.estimate == 1.0
    assert m.precision is not None
    assert m.precision.numerator == len(target)
    assert m.precision.denominator == len(target) + 1  # the free-text note about Ana
    assert m.collateral is not None
    assert m.collateral.numerator == 1
    assert m.accidental_retention is not None
    assert m.accidental_retention.numerator == 0
    assert {d.state for d in r.decisions} <= {"active", "forgotten"}
    assert {"active", "suppressed"} == set(RETRIEVABLE)
