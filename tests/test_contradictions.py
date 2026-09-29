import hashlib

import pytest

from memoria.belief_cases import ONTOLOGY, SOURCES, taxonomy_cases
from memoria.belief_demo import GRAPH_POLICY, world_ontology
from memoria.contradictions import (
    CONFIDENCE,
    GENUINE,
    TYPES,
    Assertion,
    ClaimOntology,
    classify,
    contradiction_graph,
    parse_assertion,
)
from memoria.core import Experience
from memoria.formation import EpisodicPolicy, form
from memoria.graph import build_graph
from memoria.hybrid import Corpus
from memoria.scenarios import day
from memoria.store import MemoryLog


def digest(n: str) -> str:
    return "sha256:" + hashlib.sha256(n.encode()).hexdigest()


def a(
    id_: str, key: str, raw: str, when: float, source: str = "chat:x", verb: str = "set"
) -> Assertion:
    return parse_assertion(digest(id_), key, verb, raw, day(when), day(when), source)


def kind(x: Assertion, y: Assertion, ont: ClaimOntology = ONTOLOGY) -> str | None:
    r = classify(x, y, ont, SOURCES)
    return r.type if r else None


def test_the_designed_cases_are_classified_as_designed() -> None:
    cases = taxonomy_cases()
    assert [c.name for c in cases if not c.ok] == []
    assert {c.expected for c in cases if c.expected} == set(TYPES)  # all twelve are exercised
    assert sum(c.expected is None for c in cases) >= 5  # and compatible pairs stay unrelated


def test_time_separates_change_from_contradiction() -> None:
    def at(gap: float) -> str | None:
        return kind(
            a("1", "ana.home", "Paris", 0, "chat:x"), a("2", "ana.home", "Rome", gap, "chat:y")
        )

    assert [at(g) for g in (0, 0.5, 1.0)] == ["direct_value"] * 3
    assert [at(g) for g in (1.01, 3, 60)] == ["supersession"] * 3
    tight = ClaimOntology(name="t", version="1", concurrency_days=0.1)
    assert (
        kind(a("1", "ana.home", "Paris", 0), a("2", "ana.home", "Rome", 0.5), tight)
        == "supersession"
    )


def test_relations_are_directional_and_never_declare_falsity() -> None:
    x, y = a("1", "ana.home", "Paris", 0, "chat:x"), a("2", "ana.home", "Berlin", 30, "chat:y")
    r = classify(x, y, ONTOLOGY, SOURCES)
    assert r is not None
    assert (r.relation, r.source, r.target) == (
        "supersedes",
        digest("2"),
        digest("1"),
    )  # later supersedes earlier
    assert not r.genuine
    assert classify(y, x, ONTOLOGY, SOURCES) == r  # symmetric input, same relation
    fix = classify(x, a("3", "ana.home", "Berlin", 5, "clinic:a", "correct"), ONTOLOGY, SOURCES)
    assert fix is not None
    assert (fix.relation, fix.resolution) == ("corrects", "resolved_correction")
    narrow = classify(x, a("4", "ana.home", "France", 0.1), ONTOLOGY, SOURCES)
    assert narrow is not None
    assert (narrow.relation, narrow.source) == ("narrows", digest("1"))  # Paris narrows France
    qual = classify(
        a("5", "ana.dose", "10 mg (morning)", 0),
        a("6", "ana.dose", "20 mg", 0.2),
        ONTOLOGY,
        SOURCES,
    )
    assert qual is not None
    assert (qual.relation, qual.resolution, qual.genuine) == ("qualifies", "apparent", False)
    for r in (fix, narrow, qual):
        assert r.relation in ("contradicts", "corrects", "narrows", "qualifies", "supersedes")
        assert 0 <= r.confidence <= 1


def test_every_relation_carries_type_confidence_evidence_scope_and_time() -> None:
    r = classify(
        a("1", "ana.home", "Paris", 0, "clinic:a"),
        a("2", "ana.home", "Rome", 0.2, "email:b"),
        ONTOLOGY,
        SOURCES,
    )
    assert r is not None
    assert r.type == "source_disagreement"
    assert r.confidence == CONFIDENCE["source_disagreement"]
    assert r.evidence == tuple(sorted((digest("1"), digest("2"))))
    assert (r.scope, r.temporal, r.resolution, r.genuine) == (
        "ana.home",
        "concurrent",
        "open",
        True,
    )
    assert {"direct_value", "numeric", "negation"} <= GENUINE
    assert not GENUINE & {"supersession", "granularity", "scope", "missing_qualifier"}


def test_same_source_lineage_is_a_direct_contradiction_not_a_disagreement() -> None:
    x, y = a("1", "ana.home", "Paris", 0, "forum:o"), a("2", "ana.home", "Rome", 0.2, "forum:f")
    assert kind(x, y) == "direct_value"  # one class: no independent classes disagree


def test_numbers_within_tolerance_agree_and_units_matter() -> None:
    assert kind(a("1", "ana.dose", "100 mg", 0), a("2", "ana.dose", "101 mg", 0)) is None
    assert kind(a("1", "ana.dose", "100 mg", 0), a("2", "ana.dose", "150 mg", 0)) == "numeric"
    assert kind(a("1", "ana.dose", "100 mg", 0), a("2", "ana.dose", "0.1 g", 0)) == "direct_value"


def test_windows_decide_temporal_relations() -> None:
    w1 = "Paris (during=2026-01-01..2026-02-01)"
    assert (
        kind(
            a("1", "ana.home", w1, 0), a("2", "ana.home", "Rome (during=2026-03-01..2026-04-01)", 0)
        )
        == "supersession"
    )
    assert (
        kind(
            a("1", "ana.home", w1, 0), a("2", "ana.home", "Rome (during=2026-01-15..2026-02-15)", 0)
        )
        == "temporal"
    )


def test_parsing_reads_only_the_declared_grammar() -> None:
    p = a("1", "ana.home", "not Paris; Rome (during=2026-01-01..2026-02-01)", 0)
    assert p.negated
    assert p.members == ("paris", "rome")
    assert p.window is not None
    plain = a("2", "ana.dose", "12.5 mg", 0)
    assert (plain.number, plain.unit) == (12.5, "mg")
    assert a("3", "ana.home", "Paris (morning)", 0).qualifier == "morning"


def test_the_contradiction_graph_clusters_and_keeps_every_claim() -> None:
    steps = [
        (0, "chat:x1", "set ana.home = Paris"), (0.3, "email:x2", "set ana.home = Rome"),
        (40, "clinic:x3", "set ana.home = Berlin"), (41, "email:x4", "set ana.home = Berlin"),
        (5, "chat:x5", "set ben.home = Oslo"),
    ]  # fmt: skip
    with MemoryLog(":memory:") as log:
        for d, src, text in sorted(steps):
            form(log, Experience(source=src, content=text, occurred_at=day(d)), EpisodicPolicy(),
                 recorded_at=day(d))  # fmt: skip
        snap = build_graph(Corpus.from_log(log, day(100)), GRAPH_POLICY)
    cg = contradiction_graph(snap, world_ontology(), SOURCES)
    assert cg.graph == snap.digest
    types = dict(cg.counts)
    assert types["categorical"] == 1  # Paris against Rome, concurrent
    assert types["supersession"] >= 2  # later Berlin reports supersede
    (cluster,) = [c for c in cg.clusters if "ana.home" in c.keys]
    assert "ben.home" not in cluster.keys  # unrelated claims do not join
    assert cluster.open  # a conflict left open ...
    assert cluster.resolved  # ... and changes resolved
    assert len(cluster.chain) >= 2
    assert cluster.evidence_volume == 4
    assert set(cluster.roots) == {"chat:x1", "email:x2", "clinic:x3", "email:x4"}
    ids = [c.id for c in cg.contradictions]
    assert ids == sorted(set(ids))
    # both sides of every relation are claims of the graph: nothing is dropped or invented
    claims = {c.id for c in snap.claims}
    assert all(c.a in claims and c.b in claims for c in cg.contradictions)


def test_near_collision_entities_are_a_collision_only_while_unmerged() -> None:
    from memoria.entities import AGGRESSIVE, CONSERVATIVE, resolve

    d = "sha256:" + "0" * 64
    x, y = a("1", "ana.home", "Paris", 0), a("2", "anna.home", "Rome", 0.2)
    unmerged = resolve({"ana": [d], "anna": [d]}, {}, CONSERVATIVE)
    r = classify(x, y, ONTOLOGY, SOURCES, unmerged)
    assert r is not None
    assert (r.type, r.resolution, r.genuine) == ("entity_collision", "identity_unresolved", False)
    assert classify(x, y, ONTOLOGY, SOURCES, None) is None  # no resolution: other entities
    merged = resolve({"ana": [d], "anna": [d]}, {}, AGGRESSIVE)
    assert (
        classify(x, y, ONTOLOGY, SOURCES, merged) is None
    )  # accepted decisions are not candidates


def test_ontology_is_a_closed_sorted_record() -> None:
    with pytest.raises(ValueError, match="unique and sorted"):
        ClaimOntology(name="t", version="1", hierarchy=(("b", "c"), ("a", "b")))
    assert ONTOLOGY.broader("paris") == {"france"}
    assert ONTOLOGY.digest != ClaimOntology(name="t", version="1").digest
