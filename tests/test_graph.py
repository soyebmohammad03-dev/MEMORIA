from collections.abc import Sequence
from functools import cache
from pathlib import Path

import pytest
from pydantic import ValidationError

from memoria.artifacts import ArtifactStore
from memoria.consolidation import ConsolidationPolicy, Hierarchy, consolidate, invalidated
from memoria.core import EpistemicStatus, Experience, Level
from memoria.embeddings import HashedNgramEmbedder
from memoria.entities import AGGRESSIVE, CONSERVATIVE
from memoria.forgetting import Selector, apply, forget, policy
from memoria.formation import EpisodicPolicy, form
from memoria.graph import (
    ENDPOINTS,
    OBSERVED_RULES,
    Edge,
    GraphIntegrityError,
    GraphPolicy,
    GraphSnapshot,
    MemoryGraph,
    build_graph,
    diff_graphs,
    measure_graph,
    temporal_diagnostics,
    trace_provenance,
    verify_snapshot,
)
from memoria.hybrid import (
    Corpus,
    Engine,
    GeneratorSpec,
    HybridQuery,
    HybridTrace,
    PolicyError,
    RetrievalPolicy,
    equal_weights,
)
from memoria.hybrid import policy as retrieval_policy
from memoria.scenarios import day
from memoria.store import MemoryLog

EMB = HashedNgramEmbedder()
G = GraphPolicy(name="g", version="1", resolution=CONSERVATIVE)
ITEMS = (
    (0, "chat:a0", "set ana.home = Paris"),
    (2, "email:a2", "Ana lives in Paris."),
    (3, "email:a3", "set ana.home = Paris"),
    (40, "chat:a40", "set ana.home = Berlin"),
    (40, "forum:a40", "set ana.home = Rome"),
    (45, "email:a45", "Ana does not live in Paris anymore."),
    (60, "chat:b60", "set ben.home = Oslo"),
    (61, "chat:n61", "Anna called about the weekend."),
    (70, "clinic:a70", "correct ana.home = Munich"),
    (80, "chat:e80", "set ana.employer = Acme"),
    (81, "chat:e81", "set ana.employer = Acme"),
)
HIER = ConsolidationPolicy(name="t", version="1", regime="temporal", every_days=30,
                           abstractions=("co_change", "entity", "timeline"))  # fmt: skip


def corpus_at(at: float, items: Sequence[tuple[float, str, str]] = ITEMS) -> Corpus:
    with MemoryLog(":memory:") as log:
        for d, source, content in sorted(items, key=lambda x: (x[0], x[1])):
            e = Experience(source=source, content=content, occurred_at=day(d))
            form(log, e, EpisodicPolicy(), recorded_at=day(d))
        return Corpus.from_log(log, day(at))


@cache
def world() -> tuple[Corpus, Hierarchy, GraphSnapshot]:
    base = corpus_at(100)
    h = consolidate(base, HIER, day(100), EMB)
    corpus = base.with_derived(h.memories, invalidated(h, base))
    return corpus, h, build_graph(corpus, G, [h])


def by_source(c: Corpus, source: str) -> str:
    return next(e.digest for e in c.entries if e.level is Level.L1
                and e.sources[0].source == source)  # fmt: skip


# --- ontology and statuses -----------------------------------------------------------------------


def test_graph_has_every_node_kind_and_valid_endpoints() -> None:
    _, _, s = world()
    kinds = {n.kind for n in s.nodes}
    assert kinds >= {"experience", "memory", "derived", "entity", "claim", "event", "source",
                     "interval", "consolidation"}  # fmt: skip
    for e in s.edges:
        pair = (s.node(e.source).kind, s.node(e.target).kind)
        assert pair in ENDPOINTS[e.relation]
        assert e.evidence
    relations = {e.relation for e in s.edges}
    assert relations >= {"supports", "derived-from", "mentions", "about", "asserts",
                         "contradicts", "supersedes", "temporally-precedes", "same-event",
                         "related-to", "summarized-by", "consolidated-from", "sourced-from",
                         "valid-during"}  # fmt: skip


def test_no_edge_upgrades_inferred_or_computed_relations() -> None:
    _, _, s = world()
    for e in s.edges:
        if e.status is EpistemicStatus.OBSERVED:
            assert (e.relation, e.rule) in OBSERVED_RULES
        if e.rule == "value-mention":
            assert e.status is EpistemicStatus.INFERRED
    node = {n.id: n for n in s.nodes}
    rank = {"observed": 0, "derived": 1, "abstracted": 2, "inferred": 3}
    for e in s.edges:
        weakest = max(rank[node[e.source].status.value], rank[node[e.target].status.value])
        assert rank[e.status.value] >= weakest


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"status": EpistemicStatus.OBSERVED, "rule": "value-mention"}, "cannot be observed"),
        ({"relation": "about"}, "cannot join"),
        ({"status": EpistemicStatus.OBSERVED}, "computed relation"),
    ],
)
def test_tampered_edges_are_refused(change: dict[str, object], message: str) -> None:
    _, _, s = world()
    victim = next(e for e in s.edges if e.rule == "value-mention")
    forged = Edge.model_validate(victim.model_dump() | change)
    edges = sorted([*[e for e in s.edges if e != victim], forged],
                   key=lambda e: (e.source, e.relation, e.target, e.rule))  # fmt: skip
    with pytest.raises(ValidationError, match=message):
        GraphSnapshot.model_validate(s.model_dump() | {"edges": [e.model_dump() for e in edges]})


# --- claims --------------------------------------------------------------------------------------


def test_claims_are_structured_only_where_the_evidence_is() -> None:
    corpus, _, s = world()
    free = {f"memory:{by_source(corpus, x)}" for x in ("email:a2", "email:a45", "chat:n61")}
    unavailable = dict(s.unavailable)
    assert free <= set(unavailable)
    assert {unavailable[f] for f in free} == {"unstructured"}
    inferred = [n.id for n in s.nodes if n.kind == "derived"
                and n.status is EpistemicStatus.INFERRED]  # fmt: skip
    assert all(unavailable[n] == "inferred pattern asserts no claim" for n in inferred)
    observed = [c for c in s.claims if c.status is EpistemicStatus.OBSERVED]
    berlin = next(c for c in observed if c.object == "berlin")
    assert (berlin.subject, berlin.predicate, berlin.qualifier) == ("entity:ana", "home", "set")
    assert berlin.contradicting
    assert berlin.disputed == 1  # Rome at the same instant
    assert berlin.support == 1
    assert berlin.supporting == (f"memory:{by_source(corpus, 'chat:a40')}",)
    for c in s.claims:
        if c.status is not EpistemicStatus.OBSERVED:
            assert c.evidence  # derived claims still resolve to experiences
            assert all(x.startswith("derived:") for x in c.supporting)


def test_free_text_links_to_claims_only_as_inferred_co_mentions() -> None:
    corpus, _, s = world()
    note = f"memory:{by_source(corpus, 'email:a45')}"  # a negation: "does not live in Paris"
    links = [e for e in s.edges if e.source == note and e.rule == "value-mention"]
    assert links  # co-mention is not assertion: kept, and labelled inferred
    assert {e.status for e in links} == {EpistemicStatus.INFERRED}
    anna = f"memory:{by_source(corpus, 'chat:n61')}"
    assert not [e for e in s.edges if e.source == anna and e.relation == "related-to"]


def test_contradictions_corrections_and_events() -> None:
    _, _, s = world()
    rules = {(e.relation, e.rule) for e in s.edges}
    assert ("contradicts", "same-instant-disagreement") in rules
    assert ("contradicts", "contested-period") in rules
    assert ("supersedes", "correction") in rules
    assert ("temporally-precedes", "occurrence-order") in rules
    assert not [d for d in temporal_diagnostics(s) if d.severity == "error"]


# --- entity resolution in the graph --------------------------------------------------------------


def test_near_collision_names_stay_distinct_unless_resolution_is_aggressive() -> None:
    corpus, h, s = world()
    assert s.resolution.false_merges == ()
    anna = s.resolution.entity_of("surface:anna")
    assert anna != s.resolution.entity_of("structured:ana")
    loose = build_graph(corpus, G.model_copy(update={"resolution": AGGRESSIVE}), [h])
    assert loose.resolution.entity_of("surface:anna") == loose.resolution.entity_of(
        "structured:ana"
    )
    same = [e for e in loose.edges if e.relation == "same-entity" and e.rule == "similar"]
    assert same
    assert {e.status for e in same} == {EpistemicStatus.INFERRED}
    candidates = [e for e in s.edges if e.rule == "resolution-candidate"]
    assert {e.status for e in candidates} == {EpistemicStatus.INFERRED}


# --- snapshots, verification, diffs --------------------------------------------------------------


def test_snapshot_round_trip_and_rebuild(tmp_path: Path) -> None:
    corpus, h, s = world()
    store = ArtifactStore(tmp_path)
    loaded = store.get_record(GraphSnapshot, store.put_record(s))
    assert loaded == s
    verify_snapshot(loaded, corpus, [h])
    assert build_graph(corpus, G, [h]).digest == s.digest
    assert (s.node_count, s.edge_count) == (len(s.nodes), len(s.edges))
    assert s.schema_version == "memoria-graph-v1"
    assert s.policy.resolution == CONSERVATIVE


def test_false_support_edge_is_detected_by_rebuild() -> None:
    corpus, h, s = world()
    memory = next(n.id for n in s.nodes if n.kind == "memory")
    claim = next(c for c in s.claims if memory not in c.supporting and c.status.value == "observed")
    forged = Edge(
        source=memory,
        target=claim.id,
        relation="supports",
        rule="cites-assertion",
        status=EpistemicStatus.OBSERVED,
        evidence=(memory.removeprefix("memory:"),),
    )
    edges = sorted([*s.edges, forged], key=lambda e: (e.source, e.relation, e.target, e.rule))
    tampered = GraphSnapshot.model_validate(
        s.model_dump() | {"edges": [e.model_dump() for e in edges], "edge_count": len(edges)}
    )  # structurally valid, re-hashed: only a rebuild can tell
    with pytest.raises(GraphIntegrityError, match="does not match a rebuild"):
        verify_snapshot(tampered, corpus, [h])
    with pytest.raises(GraphIntegrityError, match="other sources"):
        verify_snapshot(s, corpus_at(90), [h])


def test_diff_separates_forgotten_from_removed_and_finds_new_contradictions() -> None:
    corpus, h, s = world()
    rec = forget(corpus, policy("f", "selective", selector=Selector(source="forum")), day(100))
    after = build_graph(apply(corpus, rec), G, [h])
    d = diff_graphs(s, after)
    assert d.forgotten_nodes
    assert not d.removed_nodes
    assert all(after.node(n).availability == "forgotten" for n in d.forgotten_nodes)
    early = build_graph(corpus_at(30), G)
    later = build_graph(corpus_at(50), G)
    grown = diff_graphs(early, later)
    assert grown.added_contradictions
    assert grown.added_nodes
    assert not grown.resolved_contradictions
    assert grown.changed_provenance


# --- temporal diagnostics ------------------------------------------------------------------------


def _with_node(s: GraphSnapshot, node_id: str, **change: object) -> GraphSnapshot:
    nodes = [n.model_copy(update=change) if n.id == node_id else n for n in s.nodes]
    return GraphSnapshot.model_validate(s.model_dump() | {"nodes": [n.model_dump() for n in nodes]})


def test_temporal_spoofing_is_diagnosed() -> None:
    _, _, s = world()
    cites = next(e for e in s.edges if e.rule == "cites")
    spoofed = _with_node(s, cites.source, recorded_at=day(-5))
    assert "evidence-after-record" in {d.check for d in temporal_diagnostics(spoofed)}
    change = next(e for e in s.edges if e.rule == "temporal-change")
    reversed_ = _with_node(s, change.source, valid_from=day(-50))
    errors = {d.check for d in temporal_diagnostics(reversed_) if d.severity == "error"}
    assert {"supersession-direction", "temporal-order"} <= errors


def test_future_evidence_in_a_retrieval_is_an_error() -> None:
    corpus, h, _ = world()
    q = HybridQuery(text="Where does Ana live?", valid_at=day(90), known_at=day(100), limit=2,
                    key="ana.home")  # fmt: skip
    pol = retrieval_policy("p", equal_weights(["attribute", "lexical"]), semantic_generator=False)
    trace = Engine(corpus, EMB).retrieve(pol, q)
    s = build_graph(corpus, G, [h], [trace], "sha256:" + "0" * 64)
    assert not [d for d in temporal_diagnostics(s) if d.severity == "error"]
    rid = f"retrieval:{trace.digest}"
    bad = _with_node(s, rid, recorded_at=day(1))
    assert "future-evidence-in-retrieval" in {d.check for d in temporal_diagnostics(bad)}


# --- provenance traversal ------------------------------------------------------------------------


def test_autopsy_reaches_evidence_and_is_bounded() -> None:
    corpus, h, _ = world()
    q = HybridQuery(text="Where does Ana live?", valid_at=day(90), known_at=day(100), limit=3,
                    key="ana.home")  # fmt: skip
    pol = RetrievalPolicy.model_validate(
        retrieval_policy("p", equal_weights(["attribute", "lexical"]),
                         semantic_generator=False).model_dump()
        | {"levels": (Level.L1, Level.L2, Level.L3, Level.L4)}
    )  # fmt: skip
    trace = Engine(corpus, EMB).retrieve(pol, q)
    s = build_graph(corpus, G, [h], [trace], "sha256:" + "0" * 64)
    t = trace_provenance(s, f"retrieval:{trace.digest}")
    assert t.experiences
    assert t.sources
    assert t.claims
    assert t.broken == ()
    assert not t.truncated
    assert t == trace_provenance(s, f"retrieval:{trace.digest}")  # deterministic
    small = trace_provenance(s, f"retrieval:{trace.digest}", max_nodes=3)
    assert small.truncated
    assert len(small.steps) == 3
    derived = next(n.id for n in s.nodes if n.kind == "derived" and n.attr("level") == "L3")
    deep = trace_provenance(s, derived)
    assert deep.experiences
    assert deep.consolidations


def test_broken_provenance_is_reported_not_repaired() -> None:
    corpus, _, _ = world()
    orphaned = build_graph(corpus, G, [])  # derived memories without their hierarchy
    assert any(why == "no consolidation record produced it" for _, why in orphaned.broken)
    t = trace_provenance(orphaned, next(n.id for n in orphaned.nodes if n.kind == "derived"))
    assert any("no consolidation record" in why for _, why in t.broken)


# --- analytics -----------------------------------------------------------------------------------


def test_metrics_are_decomposed_and_views_differ_after_forgetting() -> None:
    corpus, h, s = world()
    m = measure_graph(s)
    assert m.unreachable_evidence == 0
    assert m.orphans.numerator == 0
    assert m.contradiction_density is not None
    assert m.contradiction_density > 0
    assert m.evidence_coverage.estimate == 1.0
    assert sum(n for _, n in m.kinds) == m.nodes
    rec = forget(corpus, policy("f", "selective", selector=Selector(entity="ana")), day(100))
    after = build_graph(apply(corpus, rec), G, [h])
    full, active = measure_graph(after, "all"), measure_graph(after, "active")
    assert active.unavailable_nodes > 0
    assert active.unsupported_claims.numerator > full.unsupported_claims.numerator


# --- graph-aware retrieval -----------------------------------------------------------------------


def graph_policy() -> RetrievalPolicy:
    base = retrieval_policy("g", equal_weights(["attribute", "graph_claim", "graph_entity",
                                                "lexical"]), semantic_generator=False)  # fmt: skip
    gens = sorted([*base.generators, GeneratorSpec(name="graph", limit=10, params=(("hops", 1),))],
                  key=lambda g: g.name)  # fmt: skip
    return RetrievalPolicy.model_validate(
        base.model_dump() | {"generators": [g.model_dump() for g in gens]}
    )


def test_graph_signals_explain_their_edges() -> None:
    corpus, _, s = world()
    q = HybridQuery(text="Where does Ana live?", valid_at=day(90), known_at=day(100), limit=5,
                    key="ana.home")  # fmt: skip
    trace = Engine(corpus, EMB, graph=MemoryGraph(s)).retrieve(graph_policy(), q)
    assert trace.graph == s.digest
    assert HybridTrace.model_validate(trace.model_dump()) == trace
    graph_run = next(g for g in trace.generators if g.generator == "graph")
    assert graph_run.status == "ok"
    assert graph_run.proposals
    for r in trace.ranking:
        v = next(x for x in r.signals if x.signal == "graph_claim")
        inputs = dict(v.inputs)
        assert "reason" in inputs
        assert "target" in inputs
        if v.raw == 1.0:
            assert str(inputs["edge"]).startswith(("supports|", "related-to|", "asserts|"))
    note = by_source(corpus, "email:a2")
    ranked = trace.result(note) if note in {r.version for r in trace.ranking} else None
    if ranked is not None:
        v = next(x for x in ranked.signals if x.signal == "graph_claim")
        assert "value-mention (inferred)" in str(dict(v.inputs)["reason"])
    with pytest.raises(PolicyError, match="graph"):
        Engine(corpus, EMB).retrieve(graph_policy(), q)
    with pytest.raises(ValidationError, match="names its graph"):
        HybridTrace.model_validate(trace.model_dump() | {"graph": None})


def test_active_view_does_not_traverse_forgotten_memories() -> None:
    corpus, h, _ = world()
    rec = forget(corpus, policy("f", "selective", selector=Selector(source="chat")), day(100))
    forgotten = apply(corpus, rec)
    s = build_graph(forgotten, G, [h])
    hidden = {d.entry for d in rec.decisions if d.state == "forgotten"}
    reached = {v for v, _ in MemoryGraph(s).expand("ana.home", 3)}
    assert not reached & hidden
    historical = MemoryGraph(build_graph(forgotten, G.model_copy(update={"view": "historical"}),
                                         [h]))  # fmt: skip
    assert {v for v, _ in historical.expand("ana.home", 3)} & hidden
