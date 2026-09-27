import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from memoria.artifacts import ArtifactIntegrityError, ArtifactStore
from memoria.core import Experience, MemoryVersion, Operation, Scalar, quantize
from memoria.embeddings import HashedNgramEmbedder
from memoria.formation import EpisodicPolicy, StatementPolicy, form
from memoria.hybrid import (
    CONFLICTED,
    DEFAULT_EXCLUDE,
    SIGNALS,
    STATE_EXCLUDE,
    ConflictStatus,
    Corpus,
    DiversitySpec,
    Engine,
    Exclusion,
    GeneratorError,
    GeneratorSpec,
    HybridQuery,
    HybridTrace,
    PolicyError,
    RetrievalPolicy,
    ScoreError,
    SignalError,
    SignalValue,
    TemporalStatus,
    equal_weights,
    explain,
    generators,
    normalize,
    policy,
    signal,
    tie_key,
)
from memoria.scenarios import day
from memoria.semantic import IndexCompatibilityError, SemanticIndex
from memoria.store import MemoryLog
from memoria.vectors import hnsw_spec

EMBEDDER = HashedNgramEmbedder()
RECENCY: dict[str, Scalar] = {"axis": "recorded", "family": "exponential", "half_life_days": 30.0}
PRIORS: dict[str, Scalar] = {"prior:chat": 0.7, "prior:clinic": 0.9, "prior:forum": 0.3}


def corpus_of(
    items: Sequence[tuple[float, str, str]],
    known: float = 300,
    formation: EpisodicPolicy | StatementPolicy | None = None,
) -> Corpus:
    """(day, source, content) experiences ingested as they occur."""
    with MemoryLog(":memory:") as log:
        for d, source, content in sorted(items, key=lambda x: (x[0], x[1])):
            e = Experience(source=source, content=content, occurred_at=day(d))
            form(log, e, formation or EpisodicPolicy(), recorded_at=day(d))
        return Corpus.from_log(log, day(known))


def query(text: str, valid: float = 100, key: str | None = None, limit: int = 3) -> HybridQuery:
    return HybridQuery(text=text, valid_at=day(valid), known_at=day(300), limit=limit, key=key)


def lexical_only(**kw: object) -> RetrievalPolicy:
    return policy(
        "lexical",
        [signal("lexical", 1.0)],
        semantic_generator=False,
        contradiction="neutral",
        **kw,  # type: ignore[arg-type]
    )


def full(**kw: object) -> RetrievalPolicy:
    configs = equal_weights(
        ["attribute", "lexical", "provenance", "recency", "semantic", "source"],
        {"recency": RECENCY, "source": PRIORS},
    )
    return policy("full", configs, **kw)  # type: ignore[arg-type]


def engine(corpus: Corpus, store: ArtifactStore | None = None) -> Engine:
    index = SemanticIndex.build(corpus.present, EMBEDDER, store) if store else None
    return Engine(corpus, EMBEDDER, index)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


ANA = (
    (0, "chat:a0", "set ana.home = Paris"),
    (2, "chat:a2", "Ana lives in Paris."),
    (55, "chat:a55", "set ana.home = Berlin"),
    (56, "clinic:a56", "Ana lives in Berlin now."),
    (57, "forum:a57", "Ana lives in Berlin now."),
    (58, "chat:b58", "set ben.home = Berlin"),
    (60, "chat:a60", "set ana.work_city = Paris"),
    (120, "chat:a120", "Paris has excellent restaurants."),
    (200, "chat:a200", "set ana.home = Munich"),
    (299, "chat:a299", "Ana bought a new bicycle."),
)


def by_source(corpus: Corpus, source: str) -> str:
    return next(e.digest for e in corpus.entries if e.sources[0].source == source)


# --- signal contracts -------------------------------------------------------------------------


def test_signal_registry_declares_every_contract_field() -> None:
    assert set(SIGNALS) == {
        "attribute",
        "contradiction",
        "lexical",
        "provenance",
        "recency",
        "semantic",
        "source",
        "temporal",
        "type",
    }
    for d in SIGNALS.values():
        assert d.version
        assert d.description
        assert d.direction in (1, -1)
    assert SIGNALS["contradiction"].direction == -1
    assert SIGNALS["lexical"].bounds is None
    assert SIGNALS["semantic"].bounds == (-1.0, 1.0)


def test_signal_value_distinguishes_missing_from_zero() -> None:
    zero = SignalValue(signal="x", version="1", direction=1, raw=0.0)
    missing = SignalValue(signal="x", version="1", direction=1, missing="no_claim")
    assert zero.available
    assert not missing.available
    with pytest.raises(ValidationError, match="raw value or a reason"):
        SignalValue(signal="x", version="1", direction=1)
    with pytest.raises(ValidationError, match="raw value or a reason"):
        SignalValue(signal="x", version="1", direction=1, raw=0.0, missing="why")
    with pytest.raises(ValidationError):
        SignalValue(signal="x", version="1", direction=1, raw=math.nan)
    with pytest.raises(ValidationError, match="no normalised value"):
        SignalValue(signal="x", version="1", direction=1, missing="m", normalized=0.0)


# --- normalisation ------------------------------------------------------------------------------


def test_normalizations_on_ordinary_values() -> None:
    raws = [2.0, 4.0, None, 6.0]
    assert normalize("minmax-v1", raws, None) == [0.0, 0.5, None, 1.0]
    assert normalize("rank-v1", raws, None) == [quantize(1 / 3), quantize(2 / 3), None, 1.0]
    sd = math.sqrt(8 / 3)
    assert normalize("zscore-v1", raws, None) == [
        quantize(-2 / sd),
        0.0,
        None,
        quantize(2 / sd),
    ]
    assert normalize("bounded-v1", [-1.0, 0.0, 1.0, None], (-1.0, 1.0)) == [0.0, 0.5, 1.0, None]
    assert normalize("bounded-v1", [0.25], (0.0, 1.0)) == [0.25]  # declared normalised


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("minmax-v1", 0.0), ("rank-v1", 1.0), ("zscore-v1", 0.0)],
)
def test_degenerate_pools_map_to_documented_constants(strategy: str, expected: float) -> None:
    assert normalize(strategy, [3.5], None) == [expected]  # type: ignore[arg-type]
    assert normalize(strategy, [7.0, 7.0, None], None) == [expected, expected, None]  # type: ignore[arg-type]
    assert normalize(strategy, [None, None], None) == [None, None]  # type: ignore[arg-type]


def test_rank_normalization_shares_ranks_between_ties() -> None:
    assert normalize("rank-v1", [5.0, 5.0, 1.0], None) == [1.0, 1.0, quantize(1 / 3)]


def test_negative_scores_and_outliers() -> None:
    assert normalize("minmax-v1", [-3.0, -1.0, 1.0], None) == [0.0, 0.5, 1.0]
    squashed = normalize("minmax-v1", [0.0, 1.0, 2.0, 1e6], None)
    assert squashed[1] == quantize(1e-6)
    assert squashed[-1] == 1.0
    ranked = normalize("rank-v1", [0.0, 1.0, 2.0, 1e6], None)
    assert ranked == [0.25, 0.5, 0.75, 1.0]  # rank normalisation is robust to it


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("strategy", ["minmax-v1", "rank-v1", "zscore-v1", "bounded-v1"])
def test_non_finite_raw_values_are_rejected(strategy: str, bad: float) -> None:
    with pytest.raises(ScoreError, match="not finite"):
        normalize(strategy, [0.5, bad], (0.0, 1.0))  # type: ignore[arg-type]


def test_bounded_normalization_rejects_out_of_range_and_unbounded_signals() -> None:
    with pytest.raises(SignalError, match="outside the declared range"):
        normalize("bounded-v1", [1.5], (0.0, 1.0))
    with pytest.raises(PolicyError, match="declared range"):
        normalize("bounded-v1", [1.0], None)


def test_overflow_is_an_error_not_a_silent_infinity() -> None:
    with pytest.raises(ScoreError):
        normalize("zscore-v1", [1e308, 1e308, -1e308], None)
    with pytest.raises(ScoreError):
        normalize("minmax-v1", [-1e308, 1e308], None)


# --- corpus: temporal semantics -----------------------------------------------------------------


def statement_corpus() -> Corpus:
    return corpus_of(
        [
            (0, "chat:0", "set home = Paris"),
            (50, "chat:50", "set home = Berlin"),  # ends Paris at day 50
            (10, "chat:10", "set employer = Acme"),
            (20, "chat:20", "correct employer = Globex"),  # retracts Acme entirely
            (5, "chat:5", "set pet = dog"),
            (90, "chat:90", "forget pet"),
            (200, "chat:200", "set car = red"),
        ],
        formation=StatementPolicy(),
    )


def status(corpus: Corpus, content: str, valid: float) -> TemporalStatus:
    entry = next(e for e in corpus.entries if e.version.content == content)
    return entry.temporal(day(valid))


def test_point_in_time_and_adjacent_intervals() -> None:
    c = statement_corpus()
    assert status(c, "home = Paris", 0) is TemporalStatus.VALID  # starts at valid_from
    assert status(c, "home = Paris", 49.9) is TemporalStatus.VALID
    assert status(c, "home = Paris", 50) is TemporalStatus.EXPIRED  # adjacent: end is open
    assert status(c, "home = Berlin", 50) is TemporalStatus.VALID
    assert status(c, "home = Berlin", 49) is TemporalStatus.FUTURE
    assert status(c, "car = red", 100) is TemporalStatus.FUTURE
    assert status(c, "car = red", 250) is TemporalStatus.VALID


def test_corrections_and_forgetting_are_history_statuses_not_time() -> None:
    c = statement_corpus()
    # A corrected-away version asserts nothing at any valid time.
    assert status(c, "employer = Acme", 15) is TemporalStatus.SUPERSEDED
    assert status(c, "employer = Globex", 15) is TemporalStatus.VALID  # correction is retroactive
    assert status(c, "pet = dog", 50) is TemporalStatus.FORGOTTEN
    assert [e.version.operation for e in c.entries if e.version.memory_id == "pet"] == [
        Operation.CREATE  # the tombstone itself is not retrievable content
    ]


def test_overlapping_validity_across_memories_and_same_fact_at_different_times() -> None:
    c = corpus_of(
        [
            (0, "chat:0", "set ana.home = Paris"),
            (10, "chat:10", "set ana.home = Paris"),  # same fact, later report
            (5, "chat:5", "set ana.city = Paris"),
        ]
    )
    statuses = {e.sources[0].source: e.temporal(day(20)) for e in c.entries}
    assert set(statuses.values()) == {TemporalStatus.VALID}  # episodes overlap freely
    engine_ = Engine(c)
    first = next(e for e in c.entries if e.sources[0].source == "chat:0")
    assert engine_.conflict(first, day(20)) is ConflictStatus.SUPPORTED


def test_corpus_identity_is_the_known_universe() -> None:
    a = statement_corpus()
    b = statement_corpus()
    assert a.digest == b.digest
    earlier = corpus_of([(0, "chat:0", "set home = Paris")], known=0)
    assert earlier.digest != a.digest
    assert earlier.identity.versions


# --- claims and conflicts ---------------------------------------------------------------------


def conflict_of(corpus: Corpus, source: str, valid: float) -> ConflictStatus | None:
    entry = next(e for e in corpus.entries if e.sources[0].source == source)
    return Engine(corpus).conflict(entry, day(valid))


def test_conflict_statuses() -> None:
    c = corpus_of(
        [
            (0, "chat:a", "set k = 1"),
            (10, "chat:b", "set k = 2"),
            (20, "chat:c", "set k = 2"),
            (30, "clinic:d", "set m = x"),
            (30, "forum:e", "set m = y"),
            (40, "chat:f", "set n = old"),
            (50, "chat:g", "correct n = new"),
            (60, "chat:h", "set p = v"),
            (70, "chat:i", "forget p"),
            (5, "chat:j", "a note without structure"),
        ]
    )
    assert conflict_of(c, "chat:a", 5) is ConflictStatus.UNDISPUTED  # later change not yet
    assert conflict_of(c, "chat:a", 15) is ConflictStatus.SUPERSEDED
    assert conflict_of(c, "chat:b", 15) is ConflictStatus.SUPPORTED
    assert conflict_of(c, "clinic:d", 100) is ConflictStatus.CONTRADICTED
    assert conflict_of(c, "chat:f", 45) is ConflictStatus.CORRECTED  # retroactive
    assert conflict_of(c, "chat:g", 100) is ConflictStatus.UNDISPUTED
    assert conflict_of(c, "chat:h", 65) is ConflictStatus.UNDISPUTED
    assert conflict_of(c, "chat:h", 75) is ConflictStatus.FORGOTTEN
    assert conflict_of(c, "chat:i", 75) is ConflictStatus.RETRACTION
    assert conflict_of(c, "chat:j", 75) is None  # no claim: unavailable, not "undisputed"
    assert ConflictStatus.SUPPORTED not in CONFLICTED


def test_contradictory_metadata_yields_no_claim() -> None:
    e1 = Experience(source="chat:1", content="set k = 1", occurred_at=day(0))
    e2 = Experience(source="chat:2", content="set k = 2", occurred_at=day(0))
    v = MemoryVersion(
        memory_id="m",
        version=1,
        operation=Operation.CREATE,
        content="k = 1 or 2",
        valid_from=day(0),
        recorded_at=day(0),
        derived_from=(e1.digest, e2.digest),
    )
    c = Corpus.from_versions([v], [e1, e2], day(10))
    assert c.entries[0].claim is None
    assert c.entries[0].claim_issue == "ambiguous"


# --- candidate generation, union, filters ------------------------------------------------------


def test_lexical_generator_ranks_limits_and_records_truncation() -> None:
    c = corpus_of(ANA)
    p = lexical_only().model_copy(
        update={
            "generators": (GeneratorSpec(name="lexical", limit=2, params=generators(1)[0].params),)
        }
    )
    trace = Engine(c).retrieve(p, query("Ana lives in Berlin"))
    (run,) = trace.generators
    assert run.status == "ok"
    assert run.considered == len(c.entries)
    assert [pr.rank for pr in run.proposals] == [1, 2]
    assert run.truncated > 0
    assert run.proposals[0].score >= run.proposals[1].score


def test_union_merges_duplicates_and_keeps_generator_provenance(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    trace = engine(c, store).retrieve(full(), query("Where does Ana live?", key="ana.home"))
    assert len({x.version for x in trace.candidates}) == len(trace.candidates)
    proposals = {(g.generator, p.version) for g in trace.generators for p in g.proposals}
    recorded = {(h.generator, x.version) for x in trace.candidates for h in x.generators}
    assert proposals == recorded
    berlin = trace.candidate(by_source(c, "chat:a55"))
    assert {h.generator for h in berlin.generators} == {"lexical", "metadata", "semantic"}
    order = [(x.memory_id, x.number, x.version) for x in trace.candidates]
    assert order == sorted(order)


def test_metadata_generator_is_not_applicable_without_a_key(store: ArtifactStore) -> None:
    trace = engine(corpus_of(ANA), store).retrieve(full(), query("Where does Ana live?"))
    meta = next(g for g in trace.generators if g.generator == "metadata")
    assert (meta.status, meta.proposals) == ("not_applicable", ())


def test_semantic_generator_scores_equal_feature_similarities(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    trace = engine(c, store).retrieve(full(), query("Ana lives in Berlin", key="ana.home"))
    sem = next(g for g in trace.generators if g.generator == "semantic")
    for pr in sem.proposals:
        r = next((r for r in trace.ranking if r.version == pr.version), None)
        if r is not None:
            assert (
                r.signals[[s.name for s in trace.policy.signals].index("semantic")].raw == pr.score
            )


def test_hard_filters_exclude_with_reasons_and_never_rank(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    trace = engine(c, store).retrieve(full(), query("Where does Ana live?", 100, "ana.home"))
    munich = trace.candidate(by_source(c, "chat:a200"))
    assert (munich.temporal, munich.excluded) == (TemporalStatus.FUTURE, Exclusion.FUTURE)
    assert munich.version not in {r.version for r in trace.ranking}
    relaxed = engine(c, store).retrieve(full(exclude=STATE_EXCLUDE), trace.query)
    assert munich.version in {r.version for r in relaxed.ranking}


def test_malformed_provenance_is_always_excluded() -> None:
    e = Experience(source="chat:1", content="set k = v", occurred_at=day(0))
    v = MemoryVersion(
        memory_id="m",
        version=1,
        operation=Operation.CREATE,
        content="k = v",
        valid_from=day(0),
        recorded_at=day(0),
        derived_from=(e.digest,),
    )
    c = Corpus.from_versions([v], [], day(300))  # the cited experience is unavailable
    trace = Engine(c).retrieve(lexical_only(), query("k"))
    (cand,) = trace.candidates
    assert cand.excluded is Exclusion.MALFORMED_PROVENANCE
    assert trace.ranking == ()
    with pytest.raises(ValidationError, match="always a hard exclusion"):
        RetrievalPolicy.model_validate(full().model_dump() | {"exclude": ("future",)})


def test_empty_candidate_set_and_all_filtered_are_explicit() -> None:
    c = corpus_of(ANA)
    empty = Engine(c).retrieve(lexical_only(), query("zebra"))
    assert (empty.candidates, empty.ranking, empty.selected) == ((), (), ())
    filtered = Engine(c).retrieve(lexical_only(), query("Munich", valid=100))
    assert filtered.candidates
    assert all(x.excluded for x in filtered.candidates)
    assert filtered.selected == ()


# --- generator and resource failures -------------------------------------------------------------


def test_missing_semantic_index_fails_or_is_recorded_as_declared() -> None:
    c = corpus_of(ANA)
    with pytest.raises(GeneratorError, match="no semantic index"):
        Engine(c, EMBEDDER).retrieve(full(), query("Ana"))
    trace = Engine(c, EMBEDDER).retrieve(full(on_generator_failure="record"), query("Ana"))
    sem = next(g for g in trace.generators if g.generator == "semantic")
    assert sem.status == "failed"
    assert sem.error
    assert sem.proposals == ()


def test_stale_or_mismatched_index_is_a_generator_failure(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    stale = SemanticIndex.build(corpus_of(ANA[:3]).present, EMBEDDER, store)
    trace = Engine(c, EMBEDDER, stale).retrieve(full(on_generator_failure="record"), query("Ana"))
    assert "stale index" in (
        next(g for g in trace.generators if g.generator == "semantic").error or ""
    )
    hnsw = full(on_generator_failure="record").model_copy(
        update={
            "generators": tuple(
                g.model_copy(update={"params": (("index", "hnsw"),)}) if g.name == "semantic" else g
                for g in full().generators
            )
        }
    )
    trace = engine(c, store).retrieve(hnsw, query("Ana"))
    assert "expects a hnsw index" in (
        next(g for g in trace.generators if g.generator == "semantic").error or ""
    )


def test_engine_refuses_foreign_index_and_missing_embedder(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    other = HashedNgramEmbedder(dimensions=64)
    index = SemanticIndex.build(c.present, other, store)
    with pytest.raises(IndexCompatibilityError):
        Engine(c, EMBEDDER, index)
    with pytest.raises(PolicyError, match="needs the embedder"):
        Engine(c, None, index)
    with pytest.raises(PolicyError, match="no embedder"):
        Engine(c).retrieve(full(), query("Ana"))


def test_corrupted_vector_artifact_is_refused(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    index = SemanticIndex.build(c.present, EMBEDDER, store)
    path = store._path(index.manifest.vectors)
    path.chmod(0o644)
    path.write_bytes(b"\x00" * len(path.read_bytes()))
    with pytest.raises(ArtifactIntegrityError):
        SemanticIndex.load(store, index.digest, EMBEDDER)


def test_query_must_match_the_corpus_record_time() -> None:
    c = corpus_of(ANA)
    q = HybridQuery(text="Ana", valid_at=day(10), known_at=day(10), limit=1)
    with pytest.raises(PolicyError, match="known_at"):
        Engine(c).retrieve(lexical_only(), q)


@pytest.mark.ann
def test_approximate_index_only_proposes_candidates(store: ArtifactStore) -> None:
    pytest.importorskip("faiss")
    c = corpus_of(ANA)
    approx = SemanticIndex.build(c.present, EMBEDDER, store, hnsw_spec())
    p = full().model_copy(
        update={
            "generators": tuple(
                g.model_copy(update={"params": (("index", "hnsw"),)}) if g.name == "semantic" else g
                for g in full().generators
            )
        }
    )
    exact = engine(c, store).retrieve(full(), query("Ana lives in Berlin", key="ana.home"))
    ann = Engine(c, EMBEDDER, approx).retrieve(p, exact.query)
    for r in ann.ranking:  # every score is the exact one, whichever generator proposed it
        e = next((x for x in exact.ranking if x.version == r.version), None)
        if e is not None:
            assert e.signals == r.signals


# --- policy validation and identity -------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"signals": (signal("semantic", 0.5), signal("lexical", 0.5))}, "sorted by name"),
        ({"signals": (signal("lexical", 0.5), signal("semantic", 0.4))}, "sum to 1"),
        ({"scoring": "rrf"}, "rrf_k"),
        ({"rrf_k": 60}, "rrf_k"),
        ({"contradiction": "penalize"}, "contradiction"),
        ({"exclude": (Exclusion.FUTURE, Exclusion.EXPIRED)}, "unique and sorted"),
        ({"generators": tuple(reversed(generators(5)))}, "sorted by name"),
    ],
)
def test_invalid_policies_are_rejected(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RetrievalPolicy.model_validate(full().model_dump() | change)


@pytest.mark.parametrize("weight", [math.nan, math.inf, -0.5, 0.0])
def test_invalid_weights_are_rejected(weight: float) -> None:
    with pytest.raises(ValidationError):
        signal("lexical", weight)


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("recency", {**RECENCY, "half_life_days": 0.0}, "half_life_days"),
        ("recency", {**RECENCY, "half_life_days": math.nan}, "half_life_days"),
        ("recency", {**RECENCY, "family": "linear"}, "family"),
        ("recency", {"axis": "recorded"}, "missing"),
        ("source", {"prior:chat": 1.5}, r"\[0, 1\]"),
        ("source", {}, "at least one"),
        ("source", {"chat": 0.5}, "prior:<class>"),
        ("lexical", {"k1": -1.0, "b": 0.75}, "BM25"),
        ("temporal", {"x": 1}, "unknown"),
    ],
)
def test_signal_parameters_are_validated(name: str, params: dict[str, float], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        policy("p", [signal(name, 1.0, params=params)])


def test_unknown_signal_and_bounded_on_unbounded_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown signal"):
        policy("p", [signal("lexical", 1.0).model_copy(update={"name": "vibes"})])
    with pytest.raises(ValidationError, match="unbounded"):
        policy("p", [signal("lexical", 1.0, normalization="bounded-v1")])


def test_policy_identity_changes_with_every_meaningful_parameter() -> None:
    base = full()
    variants = [
        full(diversity=DiversitySpec(beta=1.0, similarity="embedding")),
        full(contradiction="paired"),
        full(exclude=STATE_EXCLUDE),
        full(limit=5),
        full(on_generator_failure="record"),
        base.model_copy(update={"version": "2"}),
        policy(
            "full",
            equal_weights(
                ["attribute", "lexical", "provenance", "recency", "semantic", "source"],
                {"recency": {**RECENCY, "half_life_days": 31.0}, "source": PRIORS},
            ),
        ),
        RetrievalPolicy.model_validate(
            base.model_dump()
            | {
                "signals": tuple(
                    s.model_copy(update={"normalization": "rank-v1"}) if s.name == "lexical" else s
                    for s in base.signals
                )
            }
        ),
        RetrievalPolicy.model_validate(base.model_dump() | {"scoring": "rrf", "rrf_k": 60}),
    ]
    digests = {base.digest, *(v.digest for v in variants)}
    assert len(digests) == len(variants) + 1
    assert full().digest == base.digest  # same parameters, same identity


def test_same_policy_same_inputs_same_trace(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    q = query("Where does Ana live?", key="ana.home")
    a = engine(c, store).retrieve(full(), q)
    b = engine(corpus_of(ANA), store).retrieve(full(), q)
    assert a.digest == b.digest


# --- scoring, decomposition, explanations -----------------------------------------------------


def test_score_decomposition_reconstructs_every_final_score(store: ArtifactStore) -> None:
    p = full(diversity=DiversitySpec(beta=1.0, similarity="embedding"))
    trace = engine(corpus_of(ANA), store).retrieve(p, query("Where does Ana live?", key="ana.home"))
    weights = {s.name: s.weight for s in p.signals}
    for r in trace.ranking:
        for c, sv in zip(r.contributions, r.signals, strict=True):
            direction = sv.direction
            expected = (
                0.0
                if sv.normalized is None
                else quantize(weights[c.signal] * direction * sv.normalized)
            )
            assert c.value == expected
        assert r.relevance == quantize(math.fsum(c.value for c in r.contributions))
        assert r.final == quantize(r.relevance - r.penalty)
        if r.similarity is not None:
            assert r.penalty == quantize(max(r.similarity, 0.0))  # beta = 1


def test_rank_fusion_decomposes_into_reciprocal_ranks(store: ArtifactStore) -> None:
    p = RetrievalPolicy.model_validate(full().model_dump() | {"scoring": "rrf", "rrf_k": 60})
    trace = engine(corpus_of(ANA), store).retrieve(p, query("Where does Ana live?", key="ana.home"))
    for r in trace.ranking:
        for c in r.contributions:
            if c.omitted:
                assert (c.value, c.rank) == (0.0, None)
            else:
                assert c.rank is not None
                assert c.value == quantize(c.weight / (60 + c.rank))
        assert r.relevance == quantize(math.fsum(c.value for c in r.contributions))


def tamper(trace: HybridTrace, **changes: object) -> dict[str, object]:
    return trace.model_dump() | changes


def test_inconsistent_traces_and_explanations_are_rejected(store: ArtifactStore) -> None:
    trace = engine(corpus_of(ANA), store).retrieve(
        full(), query("Where does Ana live?", key="ana.home")
    )
    assert HybridTrace.model_validate(trace.model_dump()) == trace
    top = trace.ranking[0]
    cases = {
        "contributions not reproducible": top.model_copy(
            update={
                "contributions": (
                    top.contributions[0].model_copy(update={"value": 0.99}),
                    *top.contributions[1:],
                )
            }
        ),
        "relevance is not the sum": top.model_copy(update={"relevance": top.relevance + 0.1}),
        "final score is not relevance - penalty": top.model_copy(update={"final": 5.0}),
        "explanation is not derived": top.model_copy(
            update={"explanation": top.explanation.model_copy(update={"text": "trust me"})}
        ),
        "normalised values not reproducible": top.model_copy(
            update={
                "signals": (
                    top.signals[0].model_copy(update={"normalized": 0.123}),
                    *top.signals[1:],
                )
            }
        ),
    }
    for message, bad in cases.items():
        with pytest.raises(ValidationError, match=message):
            HybridTrace.model_validate(
                tamper(
                    trace, ranking=(bad.model_dump(), *[r.model_dump() for r in trace.ranking[1:]])
                )
            )
    with pytest.raises(ValidationError, match="selected"):
        HybridTrace.model_validate(tamper(trace, selected=trace.selected[::-1]))
    bad_candidates = [c.model_dump() for c in trace.candidates]
    target = next(i for i, c in enumerate(trace.candidates) if c.excluded is not None)
    bad_candidates[target]["excluded"] = None
    with pytest.raises(ValidationError, match="exclusion"):
        HybridTrace.model_validate(tamper(trace, candidates=bad_candidates))
    swapped = [r.model_dump() for r in trace.ranking]
    swapped[0]["stage_rank"], swapped[1]["stage_rank"] = 2, 1
    with pytest.raises(ValidationError, match="tie-break"):
        HybridTrace.model_validate(tamper(trace, ranking=swapped))


def test_explanations_are_derived_from_recorded_signals(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    q = query("Where does Ana live in Paris?", valid=10, key="ana.home", limit=20)
    trace = engine(c, store).retrieve(full(), q)
    for r in trace.ranking:
        assert r.explanation == explain(r, trace.candidate(r.version))
        top2 = sorted(
            (x for x in r.contributions if x.value > 0), key=lambda x: (-x.value, x.signal)
        )[:2]
        assert [f"supports:{x.signal}" for x in top2] == [
            code for code in r.explanation.reasons if code.startswith("supports:")
        ]
        assert f"temporal:{trace.candidate(r.version).temporal.value}" in r.explanation.reasons
    note = trace.result(by_source(c, "chat:a2"))
    assert "missing:attribute" in note.explanation.reasons  # free text: no claim key
    assert "conflict:" not in " ".join(note.explanation.reasons)  # no claim, no status
    fact = trace.result(by_source(c, "chat:a0"))
    assert "conflict:undisputed" in fact.explanation.reasons  # Berlin is later than valid_at


# --- ties and ordering --------------------------------------------------------------------------


def test_exact_score_ties_break_by_memory_id_then_version() -> None:
    c = corpus_of([(0, "chat:1", "same words"), (1, "chat:2", "same words")])
    trace = Engine(c).retrieve(lexical_only(), query("same words"))
    a, b = trace.ranking
    assert (a.final, a.relevance) == (b.final, b.relevance)
    assert a.memory_id < b.memory_id
    assert tie_key(a) < tie_key(b)


# --- missing signals ----------------------------------------------------------------------------


def test_missing_signal_fails_or_is_omitted_as_declared() -> None:
    c = corpus_of(ANA)
    strict = policy(
        "strict",
        [signal("attribute", 0.5, on_missing="fail"), signal("lexical", 0.5)],
        semantic_generator=False,
        contradiction="neutral",
    )
    with pytest.raises(SignalError, match="attribute"):
        Engine(c).retrieve(strict, query("Ana lives"))  # no query key
    lenient = strict.model_copy(
        update={
            "signals": tuple(s.model_copy(update={"on_missing": "omit"}) for s in strict.signals)
        }
    )
    trace = Engine(c).retrieve(lenient, query("Ana lives"))
    for r in trace.ranking:
        attr = r.contribution("attribute")
        assert attr.omitted
        assert attr.value == 0.0
        assert r.signals[0].missing == "query_has_no_key"


# --- individual signals ----------------------------------------------------------------------


def test_attribute_match_is_structural_not_mention(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    trace = engine(c, store).retrieve(
        full(exclude=STATE_EXCLUDE), query("Paris home", key="ana.home", limit=20)
    )
    attr = {r.version: r.signals[0] for r in trace.ranking}  # attribute sorts first
    assert attr[by_source(c, "chat:a0")].raw == 1.0
    wrong_attribute = attr[by_source(c, "chat:a60")]
    assert wrong_attribute.raw == 0.0
    assert dict(wrong_attribute.inputs)["entity_match"] == 1
    wrong_entity = attr[by_source(c, "chat:b58")]
    assert wrong_entity.raw == 0.0
    assert dict(wrong_entity.inputs)["entity_match"] == 0
    assert attr[by_source(c, "chat:a120")].missing == "memory_unstructured"


def test_source_reliability_is_separate_from_similarity(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    trace = engine(c, store).retrieve(full(), query("Ana lives in Berlin now", key="ana.home"))
    names = [s.name for s in trace.policy.signals]
    clinic = trace.result(by_source(c, "clinic:a56"))
    forum = trace.result(by_source(c, "forum:a57"))
    sem, src = names.index("semantic"), names.index("source")
    assert clinic.signals[sem].raw == forum.signals[sem].raw  # identical text
    assert (clinic.signals[src].raw, forum.signals[src].raw) == (0.9, 0.3)


def test_undeclared_and_unstructured_sources_are_missing_not_zero() -> None:
    c = corpus_of([(0, "import-1", "Ana lives in Berlin."), (1, "email:2", "Ana lives in Rome.")])
    p = policy(
        "src",
        [signal("lexical", 0.5), signal("source", 0.5, params={"prior:chat": 0.7})],
        semantic_generator=False,
        contradiction="neutral",
    )
    trace = Engine(c).retrieve(p, query("Ana lives"))
    missing = {r.signals[1].missing for r in trace.ranking}
    assert missing == {"unstructured_source", "undeclared_source_class"}
    prov = policy(
        "prov", [signal("provenance", 1.0)], semantic_generator=False, contradiction="neutral"
    )
    # provenance has no generator of its own; lexical proposes, provenance ranks
    prov = prov.model_copy(update={"generators": generators(5, semantic=False)})
    ranked = Engine(c).retrieve(prov, query("Ana lives"))
    assert sorted(r.signals[0].raw or 0.0 for r in ranked.ranking) == [0.0, 1.0]


def test_recency_is_not_temporal_validity(store: ArtifactStore) -> None:
    c = corpus_of(ANA)
    p = policy(
        "time",
        equal_weights(["lexical", "recency", "temporal"], {"recency": RECENCY}),
        semantic_generator=False,
        exclude=STATE_EXCLUDE,
        contradiction="neutral",
    )
    trace = Engine(c).retrieve(p, query("Ana Paris bicycle", valid=5, limit=20))
    names = [s.name for s in p.signals]
    old = trace.result(by_source(c, "chat:a2"))
    recent = trace.result(by_source(c, "chat:a299"))
    rec, tmp = names.index("recency"), names.index("temporal")
    old_age, recent_age = old.signals[rec].raw, recent.signals[rec].raw
    assert old_age is not None
    assert recent_age is not None
    assert old_age < recent_age  # older
    assert (old.signals[tmp].raw, recent.signals[tmp].raw) == (1.0, 0.0)  # but valid then
    assert dict(recent.signals[tmp].inputs)["status"] == "future"


def test_recency_on_the_valid_axis_is_missing_for_future_memories() -> None:
    c = corpus_of(ANA)
    valid_axis: dict[str, Scalar] = {**RECENCY, "axis": "valid"}
    p = policy(
        "r",
        [signal("lexical", 0.5), signal("recency", 0.5, params=valid_axis)],
        semantic_generator=False,
        exclude=STATE_EXCLUDE,
        contradiction="neutral",
    )
    trace = Engine(c).retrieve(p, query("Ana bicycle", valid=100, limit=20))
    future = trace.result(by_source(c, "chat:a299"))
    assert future.signals[1].missing == "future_on_valid_axis"


def test_type_signal_requires_a_requested_kind() -> None:
    c = corpus_of(ANA)
    p = policy(
        "t",
        [signal("lexical", 0.5), signal("type", 0.5)],
        semantic_generator=False,
        contradiction="neutral",
    )
    q = query("Ana lives", limit=20)
    none = Engine(c).retrieve(p, q)
    assert {r.signals[1].missing for r in none.ranking} == {"query_requests_no_kind"}
    facts = Engine(c).retrieve(p, q.model_copy(update={"kinds": ("fact",)}))
    kinds = {dict(r.signals[1].inputs)["kind"]: r.signals[1].raw for r in facts.ranking}
    assert kinds == {"fact": 1.0, "note": 0.0}


# --- contradiction handling ---------------------------------------------------------------------

CONFLICT = (
    (10, "clinic:1", "set leo.dose = 20 mg"),
    (10, "forum:1", "set leo.dose = 40 mg"),
    (11, "clinic:2", "Leo takes 20 mg of lisinopril."),
    (12, "chat:3", "Leo likes lisinopril jokes."),
    (13, "chat:4", "Leo dose notes"),
)


def test_surface_mode_attaches_counter_evidence_without_reordering(store: ArtifactStore) -> None:
    c = corpus_of(CONFLICT)
    q = query("Leo dose", key="leo.dose", limit=1)
    neutral = engine(c, store).retrieve(full(contradiction="neutral"), q)
    surface = engine(c, store).retrieve(full(contradiction="surface"), q)
    assert [r.version for r in neutral.ranking] == [r.version for r in surface.ranking]
    top = surface.ranking[0]
    assert top.counter_evidence
    assert all(v in surface.candidate(top.version).conflicts_with for v in top.counter_evidence)
    assert "counter_evidence" in top.explanation.reasons
    assert all(not r.counter_evidence for r in neutral.ranking)


def test_paired_mode_places_disputing_evidence_next_to_its_primary(store: ArtifactStore) -> None:
    c = corpus_of(CONFLICT)
    trace = engine(c, store).retrieve(
        full(contradiction="paired"), query("Leo dose", key="leo.dose", limit=2)
    )
    first, second = trace.ranking[:2]
    assert second.placement == "paired"
    assert second.paired_with == first.version
    assert second.version in trace.candidate(first.version).conflicts_with
    assert trace.selected == (first.version, second.version)


def test_penalize_mode_scores_the_contradiction_signal(store: ArtifactStore) -> None:
    c = corpus_of(CONFLICT)
    configs = equal_weights(["contradiction", "lexical", "semantic"])
    p = policy("pen", configs, contradiction="penalize")
    trace = engine(c, store).retrieve(p, query("Leo dose", key="leo.dose", limit=5))
    for r in trace.ranking:
        contra = r.contribution("contradiction")
        if trace.candidate(r.version).conflict in CONFLICTED:
            assert contra.value == quantize(-1 / 3)
        elif trace.candidate(r.version).conflict is None:
            assert contra.omitted


# --- diversity ---------------------------------------------------------------------------------

REDUNDANT = (
    (0, "chat:1", "Noah is allergic to peanuts."),
    (1, "chat:2", "Noah is allergic to peanuts."),
    (2, "chat:3", "Noah is allergic to peanuts."),
    (3, "chat:4", "Peanut allergy: Noah reacts badly."),
    (4, "chat:5", "Noah enjoys football."),
)


def test_diversity_disabled_reproduces_pure_relevance_order(store: ArtifactStore) -> None:
    trace = engine(corpus_of(REDUNDANT), store).retrieve(full(), query("Noah allergic peanuts"))
    assert [r.version for r in trace.ranking] == [
        r.version for r in sorted(trace.ranking, key=tie_key)
    ]
    assert all((r.penalty, r.similarity) == (0.0, None) for r in trace.ranking)


def test_mmr_penalises_redundancy_without_touching_relevance(store: ArtifactStore) -> None:
    c = corpus_of(REDUNDANT)
    q = query("Noah allergic peanuts", limit=2)
    plain = engine(c, store).retrieve(full(), q)
    mmr = engine(c, store).retrieve(
        full(diversity=DiversitySpec(beta=1.0, similarity="embedding")), q
    )
    rel = {r.version: r.relevance for r in plain.ranking}
    assert {r.version: r.relevance for r in mmr.ranking} == rel  # primary score unchanged
    second = mmr.ranking[1]
    assert second.similar_to == mmr.ranking[0].version
    assert second.similarity is not None
    assert second.penalty == quantize(max(second.similarity, 0.0))
    duplicates = {by_source(c, s) for s in ("chat:1", "chat:2", "chat:3")}
    assert len(duplicates & set(plain.selected)) == 2
    assert len(duplicates & set(mmr.selected)) == 1  # the copy was displaced
    copy = next(r for r in mmr.ranking if r.version in duplicates - set(mmr.selected))
    assert copy.penalty > 0
    assert "redundant" in copy.explanation.reasons


def test_token_jaccard_diversity_needs_no_embedder() -> None:
    p = policy(
        "lex-div",
        [signal("lexical", 1.0)],
        semantic_generator=False,
        contradiction="neutral",
        diversity=DiversitySpec(beta=1.0, similarity="token-jaccard"),
    )
    trace = Engine(corpus_of(REDUNDANT)).retrieve(p, query("Noah allergic peanuts"))
    assert trace.embedder is None
    assert trace.ranking[1].similarity == 1.0


# --- timings ---------------------------------------------------------------------------------


def test_stage_timings_are_recorded_outside_the_trace(store: ArtifactStore) -> None:
    timings: dict[str, float] = {}
    c = corpus_of(ANA)
    a = engine(c, store).retrieve(full(), query("Ana"), timings)
    b = engine(c, store).retrieve(full(), query("Ana"))
    assert a == b
    assert set(timings) == {"generate", "filter", "features", "score", "rerank", "explain"}
    assert all(v >= 0 for v in timings.values())


def test_default_exclusions_cover_every_non_valid_status() -> None:
    assert {x.value for x in DEFAULT_EXCLUDE} == {s.value for s in TemporalStatus} - {"valid"} | {
        "malformed_provenance"
    }


def test_generator_runs_reject_duplicate_or_inconsistent_proposals() -> None:
    from memoria.hybrid import GeneratorRun, Proposal

    d = "sha256:" + "1" * 64
    base = {"generator": "lexical", "version": "1", "limit": 5, "considered": 1, "truncated": 0}
    with pytest.raises(ValidationError, match="at most once"):
        GeneratorRun(
            **base,  # type: ignore[arg-type]
            status="ok",
            proposals=(
                Proposal(version=d, rank=1, score=1.0),
                Proposal(version=d, rank=2, score=1.0),
            ),
        )
    with pytest.raises(ValidationError, match="iff the generator failed"):
        GeneratorRun(**base, status="failed")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match=r"1\.\.n"):
        GeneratorRun(**base, status="ok", proposals=(Proposal(version=d, rank=2, score=1.0),))  # type: ignore[arg-type]


def test_corpus_identity_ignores_experiences_not_cited_by_known_versions() -> None:
    items = [(0, "chat:0", "set home = Paris"), (50, "chat:50", "set home = Berlin")]
    with MemoryLog(":memory:") as log:
        for d, source, content in items:
            e = Experience(source=source, content=content, occurred_at=day(d))
            form(log, e, EpisodicPolicy(), recorded_at=day(d))
        early = Corpus.from_log(log, day(10))
        late_experience = Experience(source="chat:99", content="later", occurred_at=day(99))
        form(log, late_experience, EpisodicPolicy(), recorded_at=day(99))
        assert Corpus.from_log(log, day(10)).digest == early.digest
        assert len(early.identity.experiences) == 1
