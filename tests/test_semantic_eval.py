"""Representation experiments, exact-vs-approximate search, and index integrity."""

import importlib.util
import math
import sys
from array import array
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from memoria.artifacts import ArtifactIntegrityError, ArtifactStore
from memoria.core import (
    EmbedderSpec,
    MemoryState,
    ModelFile,
    ModelIdentity,
    RepresentationSpec,
    Tolerance,
)
from memoria.embeddings import Agreement, HashedNgramEmbedder, Preprocessing, embed
from memoria.experiments import DEFAULT_REGISTRY, Registry
from memoria.neural import MINILM, ModelUnavailableError, minilm_spec, neural_embedders
from memoria.scenarios import semantic_diagnostic
from memoria.semantic import (
    IndexCompatibilityError,
    IndexIntegrityError,
    IndexManifest,
    SemanticIndex,
)
from memoria.semantic_eval import (
    UNRELATED,
    PerformanceRecord,
    Relevance,
    SemanticExperiment,
    SemanticExperimentSpec,
    compare_indexes,
    max_rss_bytes,
    profile_scaling,
    reproduce_semantic,
    run_semantic_experiment,
    similarity_tolerance,
    synthetic_state,
)
from memoria.store import MemoryLog
from memoria.vectors import hnsw_spec

DATASET, BENCHMARK = semantic_diagnostic()
HASHED = HashedNgramEmbedder()
has_faiss = importlib.util.find_spec("faiss") is not None
needs_faiss = pytest.mark.skipif(
    not has_faiss, reason="the 'ann' extra (faiss-cpu) is not installed"
)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    s = ArtifactStore(tmp_path / "var")
    s.put_record(DATASET)
    s.put_record(BENCHMARK)
    return s


def experiment_spec(*embedders: EmbedderSpec, name: str = "test") -> SemanticExperimentSpec:
    return SemanticExperimentSpec(
        name=name,
        benchmark=BENCHMARK.digest,
        representations=tuple(RepresentationSpec(embedder=e) for e in embedders),
    )


# --- the diagnostic fixture --------------------------------------------------------------------


def test_fixture_covers_every_designed_relationship() -> None:
    labels = Counter(j.relevance for q in BENCHMARK.queries for j in q.judgements)
    assert set(labels) == set(Relevance)
    assert len(BENCHMARK.queries) == 14
    assert len(DATASET.steps) == 51
    sources = [s.experience.source for s in DATASET.steps]
    assert len(set(sources)) == len(sources)  # candidate ids are unique
    texts = Counter(s.experience.content for s in DATASET.steps)
    assert texts["Ana lives in Berlin."] == 2  # the same text from two sources
    for q in BENCHMARK.queries:
        assert {j.candidate for j in q.judgements} <= set(sources)
    assert semantic_diagnostic() == (DATASET, BENCHMARK)


def test_relevance_semantics() -> None:
    assert {r for r in Relevance if not r.relevant} == {
        Relevance.ENTITY_SUBSTITUTION,
        Relevance.LEXICAL_DISTRACTOR,
        Relevance.SHARED_VOCABULARY,
    }
    assert {r for r in Relevance if r.meaning_preserving} == {
        Relevance.PARAPHRASE,
        Relevance.EQUIVALENT,
        Relevance.OTHER_SOURCE,
    }


# --- the experiment (reference embedder: deterministic, CI-safe) -------------------------------


def test_experiment_record_is_complete_and_consistent(store: ArtifactStore) -> None:
    spec = experiment_spec(HASHED.spec, HashedNgramEmbedder(n_min=2, n_max=3).spec)
    record, performance = run_semantic_experiment(spec, store, DEFAULT_REGISTRY)
    assert store.get_record(SemanticExperiment, record.digest) == record
    assert (record.spec, record.benchmark, record.dataset) == (
        spec.digest,
        BENCHMARK.digest,
        DATASET.digest,
    )
    relevant_total = sum(1 for q in BENCHMARK.queries for j in q.judgements if j.relevance.relevant)
    for result in record.results:
        manifest = store.get_record(IndexManifest, result.index)
        assert manifest.source == record.state
        for m in result.at:
            assert m.recall.denominator == relevant_total
            assert m.recall.numerator <= m.attainable <= relevant_total
            assert m.precision.denominator == m.k * len(BENCHMARK.queries)
        for q in result.queries:
            assert all(h.candidate not in q.relevant for h in q.false_matches)
            assert set(q.misses) == set(q.relevant) - {h.candidate for h in q.retrieved}
            assert [h.rank for h in q.retrieved] == list(range(1, len(q.retrieved) + 1))
        labels = {p.label for p in result.profiles}
        assert labels == {*(r.value for r in Relevance), UNRELATED}
    assert {p.k for p in record.paired} == set(spec.ks)
    for p in record.paired:
        assert p.both + p.first_only + p.second_only + p.neither == p.pairs == relevant_total
    assert performance.experiment == record.digest
    assert store.get_record(PerformanceRecord, performance.digest) == performance


def test_hits_trace_to_experiences(store: ArtifactStore) -> None:
    record, _ = run_semantic_experiment(experiment_spec(HASHED.spec), store, DEFAULT_REGISTRY)
    with MemoryLog.load(store.get(record.log)) as log:
        for q in record.results[0].queries:
            for hit in q.retrieved:
                version = log.version(hit.version)
                assert version is not None
                (source,) = version.derived_from
                experience = log.experience(source)
                assert experience is not None
                assert experience.source == hit.candidate  # candidate id is the input's source


def test_reference_experiment_reproduces_identically(store: ArtifactStore) -> None:
    record, _ = run_semantic_experiment(experiment_spec(HASHED.spec), store, DEFAULT_REGISTRY)
    result = reproduce_semantic(record.digest, store, DEFAULT_REGISTRY)
    assert (result.outcome, result.rerun) == (Agreement.IDENTICAL, record.digest)
    assert len(store.digests("PerformanceRecord")) >= 1  # timings stored, never in the identity


# Pinned: a change means stored experiment results would differ (semantic change or drift).
GOLDEN_EXPERIMENT = "sha256:3302c070b2f92ed765d4b8acc05c92f1e08e717c79ad18a3483b29d5728fb320"


def test_reference_experiment_matches_golden_digest(store: ArtifactStore) -> None:
    record, _ = run_semantic_experiment(experiment_spec(HASHED.spec), store, DEFAULT_REGISTRY)
    assert record.digest == GOLDEN_EXPERIMENT


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="distinct"):
        experiment_spec(HASHED.spec, HASHED.spec)
    with pytest.raises(ValueError, match="increasing"):
        SemanticExperimentSpec.model_validate(
            experiment_spec(HASHED.spec).model_dump() | {"ks": (5, 1)}
        )
    with pytest.raises(ValueError, match="at least 1"):
        SemanticExperimentSpec.model_validate(
            experiment_spec(HASHED.spec).model_dump() | {"representations": ()}
        )


# --- reproduction under a tolerance ------------------------------------------------------------


class Jittery:
    """Test embedder with a declared tolerance whose output drifts by ``noise`` per call."""

    calls: ClassVar[list[int]] = [0]

    def __init__(self, spec: EmbedderSpec) -> None:
        self._spec = spec
        self._base = HashedNgramEmbedder(dimensions=spec.dimensions)

    @property
    def spec(self) -> EmbedderSpec:
        return self._spec

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        self.calls[0] += 1
        noise = float(dict(self._spec.params)["noise"]) * (self.calls[0] % 2)
        out: list[Sequence[float]] = []
        for v in self._base.encode(texts):
            w = [x + noise for x in v]
            n = math.sqrt(math.fsum(x * x for x in w))
            out.append([x / n for x in w] if n else w)
        return out


def jittery_spec(noise: float) -> EmbedderSpec:
    return EmbedderSpec(
        name="jittery-test",
        version="1",
        dimensions=64,
        normalized=True,
        params=(("noise", noise),),
        model=ModelIdentity(
            provider="test",
            id="memoria/jitter",
            revision="1",
            files=(ModelFile(path="none", sha256="0" * 64, size=0),),
        ),
        pooling="mean",
        max_tokens=16,
        truncation="right",
        runtime="test",
        batch_size=1,
        tolerance=Tolerance(max_abs=1e-4, min_cosine=0.9999),
    )


def jittery_registry() -> Registry:
    return DEFAULT_REGISTRY.extend(embedders={"jittery-test": Jittery})


@pytest.mark.parametrize(
    ("noise", "outcome"), [(1e-7, Agreement.EQUIVALENT), (5e-3, Agreement.DIFFERENT)]
)
def test_reproduction_classifies_numerical_drift(
    store: ArtifactStore, noise: float, outcome: Agreement
) -> None:
    registry = jittery_registry()
    record, _ = run_semantic_experiment(experiment_spec(jittery_spec(noise)), store, registry)
    result = reproduce_semantic(record.digest, store, registry)
    assert result.outcome is outcome
    assert result.details  # says what differed


def test_similarity_tolerance_bound() -> None:
    assert similarity_tolerance(Tolerance(max_abs=1e-4, min_cosine=0.9999), 384) == pytest.approx(
        2 * math.sqrt(384) * 1e-4
    )


# --- index integrity ------------------------------------------------------------------------------


def _state() -> MemoryState:
    return synthetic_state(40)


def _forge(store: ArtifactStore, manifest: IndexManifest, **changes: object) -> str:
    return store.put_record(IndexManifest.model_validate(manifest.model_dump() | changes))


def test_integrity_failures_are_typed(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(_state(), HASHED, store)
    m = ix.manifest
    blob = store.get(m.vectors)
    truncated = store.put(blob[:-4])
    with pytest.raises(IndexIntegrityError, match="bytes"):
        SemanticIndex.load(store, _forge(store, m, vectors=truncated), HASHED)
    wide = store.put(array("d", array("f", blob)).tobytes())  # float64: wrong dtype
    with pytest.raises(IndexIntegrityError, match="bytes"):
        SemanticIndex.load(store, _forge(store, m, vectors=wide), HASHED)
    other = SemanticIndex.build(synthetic_state(40, seed=3), HASHED, store)
    with pytest.raises(IndexIntegrityError, match="entries"):
        SemanticIndex.load(store, _forge(store, m, source=other.manifest.source), HASHED).verify(
            store, HASHED
        )
    with pytest.raises(ValueError, match="each memory once"):
        IndexManifest.model_validate(m.model_dump() | {"memory_ids": (m.memory_ids[0],) * 40})


def test_modified_stored_manifest_is_detected(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(_state(), HASHED, store)
    h = ix.digest.removeprefix("sha256:")
    path = store.root / "objects" / h[:2] / h[2:]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b'"cosine"', b'"cosinE"'))
    with pytest.raises(ArtifactIntegrityError):
        SemanticIndex.load(store, ix.digest, HASHED)


def test_incompatible_representations_are_refused(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(_state(), HASHED, store)
    unnormalised = EmbedderSpec.model_validate(HASHED.spec.model_dump() | {"normalized": False})
    other_prep = HashedNgramEmbedder(preprocessing=Preprocessing(casefold=False))
    for embedder_spec in (unnormalised, other_prep.spec, minilm_spec()):
        with pytest.raises(IndexCompatibilityError, match="built by"):
            ix.search("x", 1, _Named(embedder_spec))
    revised = minilm_spec().model_copy(
        update={"model": MINILM.model_copy(update={"revision": "0" * 40})}
    )
    assert revised.digest != minilm_spec().digest  # a different revision is a different model


class _Named:
    def __init__(self, spec: EmbedderSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> EmbedderSpec:
        return self._spec

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        raise AssertionError("never reached: compatibility is checked first")


def test_stale_index_is_refused(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(_state(), HASHED, store)
    ix.require_source(_state())
    with pytest.raises(IndexCompatibilityError, match="stale"):
        ix.require_source(synthetic_state(41))


def test_unicode_and_long_text(store: ArtifactStore) -> None:
    capped = HashedNgramEmbedder(preprocessing=Preprocessing(max_chars=20))
    a, b = embed(capped, ["x" * 20 + "tail one", "x" * 20 + "tail two"])
    assert a == b  # truncated identically
    zoe, zoe_nfd = embed(HASHED, ["Zoë's café", "Zoë's café"])
    assert zoe == zoe_nfd  # NFKC before hashing


# --- exact vs approximate ------------------------------------------------------------------------


@needs_faiss
def test_hnsw_is_deterministic_and_rescored_exactly(store: ArtifactStore) -> None:
    state = synthetic_state(300)
    exact = SemanticIndex.build(state, HASHED, store)
    a = SemanticIndex.build(state, HASHED, store, hnsw_spec())
    b = SemanticIndex.build(state, HASHED, store, hnsw_spec())
    assert a.digest == b.digest  # the serialised graph is a function of its inputs
    assert a.manifest.ann is not None
    exact_sims = exact.similarities("Ana lives in Berlin", HASHED)
    for n in a.search("Ana lives in Berlin", 10, HASHED):
        assert n.similarity == exact_sims[n.version]  # approximation never changes a score
        assert n.representation == "hnsw"
    assert a.verify(store, HASHED) is Agreement.IDENTICAL
    reloaded = SemanticIndex.load(store, a.digest, HASHED)
    assert reloaded.search("Ben works at Acme", 5, HASHED) == a.search(
        "Ben works at Acme", 5, HASHED
    )


@needs_faiss
def test_approximation_loss_is_measured(store: ArtifactStore) -> None:
    state = synthetic_state(2000)
    exact = SemanticIndex.build(state, HASHED, store)
    queries = [f"Who reads poetry? {i}" for i in range(20)]
    tight = compare_indexes(
        exact,
        SemanticIndex.build(state, HASHED, store, hnsw_spec(ef_search=128)),
        queries,
        10,
        HASHED,
    )
    loose = compare_indexes(
        exact,
        SemanticIndex.build(state, HASHED, store, hnsw_spec(m=2, ef_construction=2, ef_search=1)),
        queries,
        10,
        HASHED,
    )
    assert tight.recall.estimate is not None
    assert loose.recall.estimate is not None
    assert loose.recall.estimate < tight.recall.estimate
    assert any(q.lost for q in loose.queries)  # what was lost is listed
    assert all(len(q.lost) == round((1 - q.recall) * 10) for q in loose.queries)
    with pytest.raises(ValueError, match="exact index with an approximate"):
        compare_indexes(exact, exact, queries, 10, HASHED)
    other = SemanticIndex.build(synthetic_state(2000, seed=1), HASHED, store, hnsw_spec())
    with pytest.raises(ValueError, match="same vectors"):
        compare_indexes(exact, other, queries, 10, HASHED)


@needs_faiss
def test_corrupted_or_mismatched_ann_artifacts(store: ArtifactStore) -> None:
    state = synthetic_state(50)
    a = SemanticIndex.build(state, HASHED, store, hnsw_spec())
    small = SemanticIndex.build(synthetic_state(49), HASHED, store, hnsw_spec())
    with pytest.raises(IndexIntegrityError, match="approximate index"):
        SemanticIndex.load(store, _forge(store, a.manifest, ann=small.manifest.ann), HASHED)
    junk = store.put(b"not a faiss index")
    with pytest.raises(IndexIntegrityError, match="approximate index"):
        SemanticIndex.load(store, _forge(store, a.manifest, ann=junk), HASHED)
    empty = SemanticIndex.build(
        MemoryState(valid_at=state.valid_at, known_at=state.known_at, memories=()),
        HASHED,
        store,
        hnsw_spec(),
    )
    assert empty.search("anything", 3, HASHED) == ()


def test_ann_unavailable_is_diagnostic(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memoria.vectors import AnnUnavailableError

    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(AnnUnavailableError, match="'ann' extra"):
        SemanticIndex.build(_state(), HASHED, store, hnsw_spec())


# --- the real neural model ------------------------------------------------------------------------


@pytest.mark.model
def test_lexical_versus_neural_on_the_diagnostic(store: ArtifactStore) -> None:
    registry = DEFAULT_REGISTRY.extend(embedders=neural_embedders())
    spec = experiment_spec(HASHED.spec, minilm_spec(), name="phase5")
    try:
        record, _ = run_semantic_experiment(spec, store, registry)
    except ModelUnavailableError as e:
        pytest.skip(f"pinned model unavailable: {e}")
    lexical, neural = record.results
    at5 = {r.representation: next(m for m in r.at if m.k == 5) for r in (lexical, neural)}
    assert (
        at5[neural.representation].recall.numerator > at5[lexical.representation].recall.numerator
    )
    profile = {p.label: p.mean for p in neural.profiles}
    assert (profile["paraphrase"] or 0) > (profile["entity_substitution"] or 0)
    assert reproduce_semantic(record.digest, store, registry).outcome is Agreement.IDENTICAL


# --- performance characterisation ----------------------------------------------------------------


def test_scaling_profile_measures_every_size(store: ArtifactStore) -> None:
    ann = RepresentationSpec(embedder=HASHED.spec, index=hnsw_spec()) if has_faiss else None
    points = profile_scaling(HASHED, [10, 40], store, ann=ann, queries=3)
    assert [p.size for p in points] == [10, 40]
    for p in points:
        assert p.vector_bytes == p.size * HASHED.spec.dimensions * 4
        assert min(p.embed_s, p.exact_build_s, p.exact_query_median_s) > 0
        assert p.python_peak_bytes > 0
        assert (p.ann_recall_at_10 is None) == (ann is None)
    assert max_rss_bytes() > 0
    assert synthetic_state(40) == synthetic_state(40)  # the corpus is deterministic
