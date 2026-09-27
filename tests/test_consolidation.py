from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from memoria.artifacts import ArtifactIntegrityError, ArtifactStore
from memoria.consolidation import (
    GUARDS,
    ConsolidationPolicy,
    Hierarchy,
    ImportanceComponent,
    ImportanceName,
    ImportanceSpec,
    consolidate,
    invalidated,
    lineage,
    replay,
    timeline,
    violated,
)
from memoria.core import DerivedMemory, EpistemicStatus, Experience, Level
from memoria.embeddings import HashedNgramEmbedder
from memoria.formation import EpisodicPolicy, StatementPolicy, form
from memoria.hybrid import Corpus, Entry
from memoria.scenarios import PHASE7_WORLDS, consolidation_world, day
from memoria.store import MemoryLog

EMB = HashedNgramEmbedder()


def corpus(items: Sequence[tuple[float, str, str]], at: float = 300,
           formation: EpisodicPolicy | StatementPolicy | None = None) -> Corpus:  # fmt: skip
    with MemoryLog(":memory:") as log:
        for d, source, content in sorted(items, key=lambda x: (x[0], x[1])):
            e = Experience(source=source, content=content, occurred_at=day(d))
            form(log, e, formation or EpisodicPolicy(), recorded_at=day(d))
        return Corpus.from_log(log, day(at))


def cpol(**kw: object) -> ConsolidationPolicy:
    return ConsolidationPolicy.model_validate({"name": "t", "version": "1", "every_days": 30} | kw)


def run(items: Sequence[tuple[float, str, str]], at: float = 300, **kw: object) -> Hierarchy:
    return consolidate(corpus(items, at), cpol(**kw), day(at), EMB)


def entry(c: Corpus, source: str) -> Entry:
    return next(e for e in c.entries if e.sources[0].source == source)


def l2(h: Hierarchy) -> list[DerivedMemory]:
    return [m for m in h.memories if m.level is Level.L2]


def merged_contents(h: Hierarchy) -> list[set[str]]:
    return [set(m.versions) for m in l2(h) if len(m.versions) > 1]


# --- lineage records -------------------------------------------------------------------------


def test_derived_memory_status_follows_its_operation() -> None:
    h = run([(0, "chat:1", "set ana.home = Paris"), (1, "chat:2", "set ana.home = Paris")],
            regime="claim")  # fmt: skip
    (m,) = h.memories
    assert (m.level, m.status, m.operation) == (Level.L2, EpistemicStatus.DERIVED, "merge")
    data = m.model_dump()
    with pytest.raises(ValidationError, match="produces"):
        DerivedMemory.model_validate(data | {"status": "observed"})
    with pytest.raises(ValidationError, match="produces level"):
        DerivedMemory.model_validate(data | {"level": "L3"})
    with pytest.raises(ValidationError, match="never asserts"):
        DerivedMemory.model_validate(
            data | {"operation": "infer_co_change", "status": "inferred", "level": "L4"}
        )
    with pytest.raises(ValidationError):
        DerivedMemory.model_validate(data | {"versions": ()})


def test_lineage_reaches_versions_and_experiences() -> None:
    items = [(0, "chat:1", "set ana.home = Paris"), (5, "chat:2", "set ana.work = Acme"),
             (6, "chat:3", "set ana.home = Paris")]  # fmt: skip
    c = corpus(items)
    h = consolidate(c, cpol(regime="claim", abstractions=("entity",)), day(300))
    (profile,) = [m for m in h.memories if m.level is Level.L3]
    chain = lineage(h, profile.memory_id)
    assert chain[0] == profile.memory_id
    assert set(chain[-len(profile.versions):]) == set(profile.versions)  # fmt: skip
    assert set(profile.versions) == {e.digest for e in c.entries}
    assert set(profile.derived_from) == {s.digest for e in c.entries for s in e.sources}
    assert profile.status is EpistemicStatus.ABSTRACTED
    assert profile.key is None


# --- deduplication regimes and merge boundaries ----------------------------------------------


def test_exact_and_canonical_identity() -> None:
    items = [(0, "chat:1", "Ana lives in Berlin."), (1, "chat:2", "Ana lives in Berlin."),
             (2, "chat:3", "ana lives in berlin")]  # fmt: skip
    assert [len(g) for g in merged_contents(run(items, regime="exact"))] == [2]
    assert [len(g) for g in merged_contents(run(items, regime="canonical"))] == [3]


def test_claim_equivalence_ignores_wording_but_not_value() -> None:
    items = [(0, "chat:1", "set ana.home = Berlin"), (1, "clinic:2", "correct ana.home = Berlin"),
             (2, "chat:3", "set ana.home = Munich")]  # fmt: skip
    h = run(items, regime="claim")
    assert [len(g) for g in merged_contents(h)] == [2]
    assert {m.value for m in l2(h)} == {"berlin", "munich"}


@pytest.mark.parametrize(
    ("a", "b", "guard"),
    [
        ("Alex lives in Berlin.", "Alex lived in Berlin in 2022.", "temporal"),
        ("Alex moved to Berlin.", "Alex moved to Munich.", "entity"),
        ("Leo takes 20 mg daily.", "Leo takes 40 mg daily.", "numeric"),
        ("Ana lives in Paris.", "Ana does not live in Paris.", "negation"),
        ("set ana.home = Berlin", "set ana.home = Munich", "claim_conflict"),
        ("set sam.dog = Biscuit", "set sam.cat = Biscuit", "claim_conflict"),
    ],
)
def test_similar_but_different_statements_never_merge(a: str, b: str, guard: str) -> None:
    items = [(0, "chat:1", a), (1, "chat:2", b)]
    guarded = run(items, regime="semantic", threshold=0.01)  # any similarity is "similar"
    assert merged_contents(guarded) == []
    blocked = {g for d in guarded.decisions for _, g in d.blocked}
    assert guard in blocked  # the unsafe merge is detected and reported
    assert guarded.blocked_merges >= 1
    unguarded = run(items, regime="semantic", threshold=0.01, guards=())
    assert len(merged_contents(unguarded)) == 1  # without guards, similarity merges them


def test_guards_are_symmetric_and_empty_for_identical_text() -> None:
    c = corpus([(0, "chat:1", "Alex moved to Berlin."), (1, "chat:2", "Alex moved to Munich."),
                (2, "chat:3", "Alex moved to Berlin.")])  # fmt: skip
    a, b, a2 = (entry(c, s) for s in ("chat:1", "chat:2", "chat:3"))
    assert violated(a, b, GUARDS) == violated(b, a, GUARDS) == ("entity",)
    assert violated(a, a2, GUARDS) == ()


def test_merges_are_complete_linkage() -> None:
    # b is similar to a and to c, but a and c conflict: they never share a group.
    items = [(0, "chat:1", "set k.v = 1"), (1, "chat:2", "note k.v"), (2, "chat:3", "set k.v = 2")]
    h = run(items, regime="semantic", threshold=0.01)
    for group in merged_contents(h):
        assert len(group) < 3


# --- temporal structure ----------------------------------------------------------------------


def claims_of(items: Sequence[tuple[float, str, str]]) -> Corpus:
    return corpus(items)


def test_timeline_changes_corrections_contests_and_forgets() -> None:
    c = claims_of([
        (0, "chat:1", "set k.a = x"), (10, "chat:2", "set k.a = y"),
        (20, "chat:3", "correct k.a = z"), (30, "chat:4", "set k.a = x"),
        (40, "chat:5", "set k.a = w"), (40, "forum:6", "set k.a = v"),
        (50, "chat:7", "forget k.a"),
    ])  # fmt: skip
    periods = timeline(c.claims, "k.a")
    assert [(p.value, p.corrected, p.contested) for p in periods] == [
        ("x", False, False), ("y", True, False), ("z", False, False), ("x", False, False),
        ("v", False, True), ("w", False, True),  # same instant: ordered by value
    ]  # fmt: skip
    assert periods[2].start == day(10)  # the correction replaced y from y's start
    assert periods[3].start == day(30)  # a recurring value is a new period
    assert periods[4].end == periods[5].end == day(50)  # forget closes both


def test_late_reports_are_ordered_by_occurrence_not_ingestion() -> None:
    with MemoryLog(":memory:") as log:
        for occurred, recorded, src, text in [(0, 0, "chat:1", "set k.a = x"),
                                              (20, 20, "chat:2", "set k.a = z"),
                                              (10, 30, "chat:3", "set k.a = y")]:  # fmt: skip
            e = Experience(source=src, content=text, occurred_at=day(occurred))
            form(log, e, EpisodicPolicy(), recorded_at=day(recorded))
        c = Corpus.from_log(log, day(40))
    assert [p.value for p in timeline(c.claims, "k.a")] == ["x", "y", "z"]


def test_temporal_consolidation_keeps_historical_truth() -> None:
    items = [
        (0, "chat:1", "set ana.home = Paris"),
        (5, "chat:2", "set ana.home = Paris"),
        (50, "chat:3", "set ana.home = Berlin"),
        (100, "chat:4", "set ana.home = Paris"),
    ]
    temporal = run(items, regime="temporal")
    facts = sorted((m.value, m.valid_from, m.valid_to) for m in l2(temporal))
    assert facts == [("berlin", day(50), day(100)), ("paris", day(0), day(50)),
                     ("paris", day(100), None)]  # fmt: skip
    claim = run(items, regime="claim")  # no temporal constraint: recurring value merged
    paris = next(m for m in l2(claim) if m.value == "paris")
    assert (len(paris.versions), paris.valid_from, paris.valid_to) == (3, day(0), None)


def test_temporal_consolidation_excludes_corrected_evidence_with_a_reason() -> None:
    items = [(0, "chat:1", "set k.a = x"), (2, "clinic:2", "correct k.a = y")]
    c = corpus(items)
    h = consolidate(c, cpol(regime="temporal"), day(300))
    assert dict(h.not_promoted) == {entry(c, "chat:1").digest: "corrected"}
    (m,) = l2(h)
    assert (m.value, m.valid_from) == ("y", day(0))  # retroactive


def test_same_instant_contradictions_are_preserved_separately() -> None:
    items = [(10, "clinic:1", "set leo.dose = 20 mg"), (10, "forum:2", "set leo.dose = 40 mg"),
             (10, "forum:3", "set leo.dose = 40 mg")]  # fmt: skip
    h = run(items, regime="temporal", abstractions=("entity", "timeline"), min_support=2)
    values = sorted(m.value or "" for m in l2(h))
    assert values == ["20 mg", "40 mg"]
    rival = next(m for m in l2(h) if m.value == "20 mg")
    assert rival.disputed == 2
    assert len(rival.conflicts) == 2
    (tl,) = [m for m in h.memories if m.operation == "abstract_timeline"]
    assert tl.rule == "timeline-contested"


# --- abstraction -----------------------------------------------------------------------------


def test_entity_profile_states_only_current_supported_values() -> None:
    items = [(0, "chat:1", "set ana.home = Paris"), (50, "chat:2", "set ana.home = Berlin"),
             (0, "chat:3", "set ana.employer = Acme")]  # fmt: skip
    h = run(items, regime="temporal", abstractions=("entity",))
    (p,) = [m for m in h.memories if m.level is Level.L3]
    assert p.content == "ana: employer = acme; home = berlin"
    assert [r for _, r in p.excluded] == ["not_current"]  # Paris: historical, not current
    loss = h.losses[[m.memory_id for m in h.memories].index(p.memory_id)]
    assert loss.unsupported == ()


def test_timeline_labels_and_inferred_patterns() -> None:
    items = [(0, "chat:1", "set k.a = x"), (10, "chat:2", "set k.a = y"),
             (20, "chat:3", "set k.a = x"), (0, "chat:4", "set k.b = p"),
             (10, "chat:5", "set k.b = q"), (20, "chat:6", "set k.b = r")]  # fmt: skip
    h = run(items, regime="temporal", abstractions=("co_change", "timeline"))
    labels = {
        m.content.split(":")[0]: m.rule for m in h.memories if m.operation == "abstract_timeline"
    }
    assert labels == {"k.a": "timeline-recurring", "k.b": "timeline-changing"}
    (inferred,) = [m for m in h.memories if m.status is EpistemicStatus.INFERRED]
    assert inferred.key is None
    assert inferred.content.startswith("inferred pattern:")
    assert "3 times" in inferred.content


# --- information loss ------------------------------------------------------------------------


def test_loss_report_catches_a_dropped_qualifier_and_numbers() -> None:
    items = [(0, "chat:1", "Leo takes 20 mg."), (1, "chat:2", "Leo takes 20 mg each morning.")]
    h = run(items, regime="semantic", threshold=0.01)
    (m,) = [m for m in l2(h) if len(m.versions) == 2]
    report = h.losses[[x.memory_id for x in h.memories].index(m.memory_id)]
    assert "token:morning" in report.lost  # the representative omits the qualifier
    assert m.lost == len(report.lost)
    assert report.experience_coverage.estimate == 1.0  # still reachable through lineage
    unguarded = run([(0, "chat:1", "Leo takes 20 mg."), (1, "chat:2", "Leo takes 40 mg.")],
                    regime="semantic", threshold=0.01, guards=())  # fmt: skip
    lost = [f for r in unguarded.losses for f in r.lost]
    assert "number:40" in lost


def test_exact_consolidation_loses_nothing() -> None:
    world, _ = consolidation_world(PHASE7_WORLDS[0])
    with MemoryLog(":memory:") as log:
        for s in world.steps:
            form(log, s.experience, EpisodicPolicy(), recorded_at=s.recorded_at)
        c = Corpus.from_log(log, day(360))
    h = consolidate(c, cpol(regime="exact"), day(360))
    assert all(not r.lost and not r.altered and not r.unsupported for r in h.losses)


# --- importance ------------------------------------------------------------------------------


def importance(
    threshold: float, names: Sequence[ImportanceName] = ("repetition", "source")
) -> ImportanceSpec:
    return ImportanceSpec(
        components=tuple(
            ImportanceComponent(
                name=n,
                weight=1 / len(names),
                normalization="minmax-v1" if n == "repetition" else "bounded-v1",
            )
            for n in names
        ),
        threshold=threshold,
        half_life_days=30,
        priors=(("chat", 0.7), ("forum", 0.3)),
    )


def test_importance_is_decomposed_and_drives_promotion() -> None:
    items = [(0, "forum:1", "set k.a = x"), (0, "forum:2", "set k.a = x"),
             (0, "forum:3", "set k.a = x"), (0, "chat:4", "set k.a = y"),
             (1, "import-5", "set k.b = z")]  # fmt: skip
    h = run(items, regime="claim", promote="important", importance=importance(0.5))
    records = {r.version: r for r in h.importance}
    c = corpus(items)
    poison = records[entry(c, "forum:1").digest]
    assert [v.name for v in poison.components] == ["repetition", "source"]
    assert dict((v.name, v.raw) for v in poison.components) == {"repetition": 3.0, "source": 0.3}
    unstructured = records[entry(c, "import-5").digest]
    assert unstructured.components[1].missing == "unstructured_source"
    assert all(r.promoted == (r.total >= 0.5) for r in h.importance)
    # Repetition by unreliable sources outweighs the reliable report: a measurable exploit.
    assert poison.promoted
    assert not records[entry(c, "chat:4").digest].promoted
    assert dict(h.not_promoted)[entry(c, "chat:4").digest] == "below_importance"
    without = run(items, regime="claim", promote="important",
                  importance=importance(0.5, ("source",)))  # fmt: skip
    assert {r.version for r in without.importance if r.promoted} == {entry(c, "chat:4").digest}


# --- policy identity and validation ----------------------------------------------------------


def test_policy_identity_changes_with_every_parameter() -> None:
    base = cpol(regime="semantic", threshold=0.5)
    variants = [
        cpol(regime="semantic", threshold=0.6),
        cpol(regime="semantic", threshold=0.5, guards=("numeric",)),
        cpol(regime="semantic", threshold=0.5, every_days=31),
        cpol(regime="semantic", threshold=0.5, abstractions=("entity",)),
        cpol(regime="semantic", threshold=0.5, min_support=3),
        cpol(regime="semantic", threshold=0.5, promote="recent", window_days=9),
        base.model_copy(update={"version": "2"}),
    ]
    assert len({base.digest, *(v.digest for v in variants)}) == len(variants) + 1


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"regime": "semantic"}, "threshold"),
        ({"regime": "claim", "threshold": 0.5}, "threshold"),
        ({"regime": "claim", "promote": "recent"}, "window_days"),
        ({"regime": "claim", "promote": "important"}, "importance"),
        ({"regime": "none", "abstractions": ("entity",)}, "no-op"),
        ({"regime": "exact", "abstractions": ("entity",)}, "claim-bearing"),
        ({"regime": "claim", "guards": ("numeric", "entity")}, "sorted"),
    ],
)
def test_invalid_policies_are_rejected(params: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        cpol(**params)


def test_noop_produces_nothing() -> None:
    h = run([(0, "chat:1", "set k.a = x")], regime="none")
    assert (h.memories, h.decisions, h.losses) == ((), (), ())


# --- invalidation, replay, tampering ---------------------------------------------------------


def test_new_conflicting_evidence_invalidates_derived_memory_and_its_descendants() -> None:
    items = [(0, "chat:1", "set ana.home = Paris"), (0, "chat:2", "set ana.work = Acme"),
             (60, "chat:3", "set ana.home = Berlin")]  # fmt: skip
    early = corpus(items, at=30)
    h = consolidate(early, cpol(regime="temporal", abstractions=("entity",)), day(30))
    stale = invalidated(h, corpus(items, at=100))
    paris = next(m for m in l2(h) if m.value == "paris")
    profile = next(m for m in h.memories if m.level is Level.L3)
    assert stale[paris.memory_id] == "new_conflicting_evidence"
    assert stale[profile.memory_id] == "parent_invalidated"
    assert invalidated(h, early) == {}


def test_retracted_evidence_invalidates() -> None:
    items = [(0, "chat:1", "set home = Paris"), (60, "chat:2", "forget home")]
    early = corpus(items, at=30, formation=StatementPolicy())
    h = consolidate(early, cpol(regime="claim"), day(30))
    later = corpus(items, at=100, formation=StatementPolicy())
    assert set(invalidated(h, later).values()) == {"evidence_retracted"}


def test_replay_is_deterministic_and_detects_tampering(tmp_path: Path) -> None:
    world, _ = consolidation_world(PHASE7_WORLDS[0])
    with MemoryLog(":memory:") as log:
        for s in world.steps:
            form(log, s.experience, EpisodicPolicy(), recorded_at=s.recorded_at)
        c = Corpus.from_log(log, day(200))
    policy = cpol(regime="semantic", threshold=0.5, abstractions=("entity", "timeline"))
    h = consolidate(c, policy, day(200), EMB)
    assert replay(h, c, policy, EMB)
    assert consolidate(c, policy, day(200), EMB).digest == h.digest
    forged = h.memories[0].model_copy(update={"content": "set ana.home = Atlantis"})
    assert not replay(h.model_copy(update={"memories": (forged, *h.memories[1:])}), c, policy, EMB)
    store = ArtifactStore(tmp_path / "a")
    digest = store.put_record(h)
    path = store._path(digest)
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b"L2", b"L3", 1))
    with pytest.raises(ArtifactIntegrityError):
        store.get_record(Hierarchy, digest)


def test_hierarchy_validation_rejects_inconsistent_lineage() -> None:
    h = run([(0, "chat:1", "set k.a = x"), (0, "chat:2", "set k.b = y")], regime="claim",
            abstractions=("entity",))  # fmt: skip
    data = h.model_dump()
    bad = [dict(m) for m in data["memories"]]
    bad[-1]["parents"] = ["L2:missing"]
    with pytest.raises(ValidationError, match="not a lower level"):
        Hierarchy.model_validate(data | {"memories": bad})
    wrong_loss = [dict(r) for r in data["losses"]]
    wrong_loss[0]["lost"] = ["token:invented"]
    with pytest.raises(ValidationError, match="loss count"):
        Hierarchy.model_validate(data | {"losses": wrong_loss})


def test_hierarchy_digest_is_pinned() -> None:
    # Hashed embeddings and integer-derived data only: identical on every platform.
    world, _ = consolidation_world(PHASE7_WORLDS[0])
    with MemoryLog(":memory:") as log:
        for s in world.steps:
            form(log, s.experience, EpisodicPolicy(), recorded_at=s.recorded_at)
        c = Corpus.from_log(log, day(360))
    policy = cpol(
        regime="semantic", threshold=0.5, abstractions=("co_change", "entity", "timeline")
    )
    h = consolidate(c, policy, day(360), EMB)
    assert h.digest == PINNED


PINNED = "sha256:f9a6ddd0658ed0c482d9efe5c426ed10d07973893fc74780c7e57b6a09c98d9d"


# --- invariants over generated worlds (property style) ---------------------------------------


@pytest.mark.parametrize("world", PHASE7_WORLDS, ids=lambda w: w.name)
def test_every_derived_memory_is_distinguishable_and_fully_traceable(world: object) -> None:
    from memoria.scenarios import WorldSpec

    assert isinstance(world, WorldSpec)
    dataset, _ = consolidation_world(world)
    with MemoryLog(":memory:") as log:
        for s in dataset.steps:
            form(log, s.experience, EpisodicPolicy(), recorded_at=s.recorded_at)
        end = dataset.steps[-1].recorded_at
        c = Corpus.from_log(log, end)
    policy = cpol(regime="temporal", abstractions=("co_change", "entity", "timeline"))
    h = consolidate(c, policy, end)
    by_digest = {e.digest: e for e in c.entries}
    for m, r in zip(h.memories, h.losses, strict=True):
        assert m.status is not EpistemicStatus.OBSERVED
        assert set(m.versions) <= set(by_digest)
        assert set(m.derived_from) == {s.digest for v in m.versions for s in by_digest[v].sources}
        assert r.experience_coverage.estimate == 1.0
        assert r.unsupported == ()
        if m.status is EpistemicStatus.INFERRED:
            assert m.key is None
    derived = c.with_derived(h.memories, ())
    assert all(
        e.status is not EpistemicStatus.OBSERVED for e in derived.entries if e.level is not Level.L1
    )
