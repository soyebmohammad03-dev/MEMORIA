"""Phase 5, slice 1: embedding contract, deterministic embedder, semantic index."""

import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from memoria.artifacts import ArtifactIntegrityError, ArtifactStore
from memoria.core import MemoryState, MemoryVersion, Operation, content_hash, state_as_of
from memoria.embeddings import (
    EmbedderSpec,
    EmbeddingError,
    HashedNgramEmbedder,
    Preprocessing,
    embed,
    preprocess,
)
from memoria.formation import EpisodicPolicy, form
from memoria.scenarios import day, relocation_year
from memoria.semantic import (
    IndexCompatibilityError,
    IndexIntegrityError,
    IndexManifest,
    SemanticIndex,
)
from memoria.store import MemoryLog

E = HashedNgramEmbedder()
DIGEST = "sha256:" + "0" * 64


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def mem(memory_id: str, content: str) -> MemoryVersion:
    return MemoryVersion(
        memory_id=memory_id,
        version=1,
        operation=Operation.CREATE,
        content=content,
        valid_from=day(0),
        recorded_at=day(0),
        derived_from=(DIGEST,),
    )


def state(*memories: MemoryVersion) -> MemoryState:
    return state_as_of(memories, valid_at=day(1), known_at=day(1))


CORPUS = state(
    mem("home", "home = Berlin"),
    mem("pet", "pet = dog"),
    mem("tea", "drink = green tea"),
    mem("office", "home.office = Nice"),
)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "var")


# --- preprocessing and the embedding contract -------------------------------------------------


def test_preprocessing_is_explicit() -> None:
    p = Preprocessing()
    assert preprocess("  \uff28\uff4f\uff4d\uff45 \n\t = BERLIN ", p) == "home = berlin"
    assert preprocess("Straße", p) == "strasse"
    raw = Preprocessing(unicode="none", casefold=False, collapse_whitespace=False)
    assert preprocess(" A  b ", raw) == " A  b "
    assert preprocess("abcdef", Preprocessing(max_chars=3)) == "abc"


def test_embeddings_are_deterministic_and_normalised() -> None:
    texts = ["home = Berlin", "pet = dog", ""]
    first = embed(E, texts)
    assert first == embed(HashedNgramEmbedder(), texts)  # a fresh instance agrees
    assert all(len(v) == 256 for v in first)
    assert math.isclose(dot(first[0], first[0]), 1.0, rel_tol=1e-12)
    assert first[2] == (0.0,) * 256  # empty text is the zero vector
    # Pinned: identical on every platform (integer counts, one correctly-rounded division).
    assert (
        content_hash(list(first[0]))
        == "sha256:f0ced7a5f6773847e6f93abb7294a61b1b1c4740042515ad11a06719d080d2e5"
    )


def test_embedding_reflects_spelling_not_meaning() -> None:
    home, shouted, dotted, sofa, couch = embed(
        E, ["home = Berlin", "  HOME =  Berlin", "home = Berlin.", "sofa", "couch"]
    )
    assert dot(home, shouted) == pytest.approx(1.0)  # identical after preprocessing
    assert 0.8 < dot(home, dotted) < 1.0  # punctuation is kept: close, not identical
    assert dot(sofa, couch) < 0.3  # synonyms are not close: this is not a semantic model


def test_identity_covers_every_output_affecting_setting() -> None:
    base = HashedNgramEmbedder().spec
    assert HashedNgramEmbedder().spec == base
    variants = [
        HashedNgramEmbedder(dimensions=128).spec,
        HashedNgramEmbedder(seed=1).spec,
        HashedNgramEmbedder(n_min=2).spec,
        HashedNgramEmbedder(n_max=4).spec,
        HashedNgramEmbedder(preprocessing=Preprocessing(casefold=False)).spec,
    ]
    assert len({base.digest, *(v.digest for v in variants)}) == 6
    assert EmbedderSpec.model_validate_json(base.canonical()) == base
    with pytest.raises(ValueError, match="n_min"):
        HashedNgramEmbedder(n_min=4, n_max=3)


class Faulty:
    """An embedder that violates its contract in a chosen way."""

    def __init__(self, fault: str, normalized: bool = False) -> None:
        self.fault = fault
        self._spec = EmbedderSpec(name="faulty", version="1", dimensions=3, normalized=normalized)

    @property
    def spec(self) -> EmbedderSpec:
        return self._spec

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        vectors: list[Sequence[float]] = [[1.0, 0.0, 0.0] for _ in texts]
        if self.fault == "count":
            return vectors[1:]
        if self.fault == "dimensions":
            return [[1.0, 0.0] for _ in texts]
        if self.fault == "nan":
            return [[math.nan, 0.0, 0.0] for _ in texts]
        if self.fault == "unnormalised":
            return [[2.0, 0.0, 0.0] for _ in texts]
        return vectors


@pytest.mark.parametrize(
    ("fault", "normalized", "message"),
    [
        ("count", False, "vectors for"),
        ("dimensions", False, "dimensions"),
        ("nan", False, "not finite"),
        ("unnormalised", True, "not L2-normalised"),
    ],
)
def test_contract_violations_are_rejected(fault: str, normalized: bool, message: str) -> None:
    with pytest.raises(EmbeddingError, match=message):
        embed(Faulty(fault, normalized), ["a", "b"])


def test_no_texts_no_call() -> None:
    assert embed(Faulty("count"), []) == []


# --- index construction, search, persistence ----------------------------------------------


def test_index_binds_embedder_state_entries_texts_and_vectors(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(CORPUS, E, store)
    m = ix.manifest
    assert m.embedder == E.spec
    assert store.get_record(MemoryState, m.source) == CORPUS
    assert m.entries == tuple(v.digest for v in CORPUS.memories)
    assert m.memory_ids == ("home", "office", "pet", "tea")  # the state's order
    assert m.texts == tuple(
        content_hash(preprocess(v.content or "", E.spec.preprocessing)) for v in CORPUS.memories
    )
    blob = store.get(m.vectors)
    assert len(blob) == 4 * 256 * 4  # little-endian float32, row-major
    assert store.get_record(IndexManifest, ix.digest) == m


def test_search_ranks_by_cosine_with_decomposable_results(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(CORPUS, E, store)
    hits = ix.search("home", 4, E)
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert hits[0].memory_id in {"home", "office"}
    assert [h.similarity for h in hits] == sorted((h.similarity for h in hits), reverse=True)
    assert all(-1 <= h.similarity <= 1 for h in hits)
    assert len(ix.search("home", 2, E)) == 2
    assert len(ix.search("home", 99, E)) == 4
    with pytest.raises(ValueError, match="k must"):
        ix.search("home", 0, E)


def test_duplicate_contents_are_separate_entries_in_deterministic_order(
    store: ArtifactStore,
) -> None:
    twins = state(mem("b", "same text"), mem("a", "same text"), mem("c", "other"))
    ix = SemanticIndex.build(twins, E, store)
    assert ix.vectors[0] == ix.vectors[1]
    hits = ix.search("same text", 3, E)
    assert [h.memory_id for h in hits] == ["a", "b", "c"]  # ties broken by memory_id
    assert hits[0].similarity == hits[1].similarity == 1.0


def test_empty_state(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(state(), E, store)
    assert ix.manifest.entries == ()
    assert store.get(ix.manifest.vectors) == b""
    assert ix.search("anything", 5, E) == ()
    again = SemanticIndex.load(store, ix.digest, E)
    again.verify(store, E)


def test_zero_vector_has_zero_similarity(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(state(mem("blank", ""), mem("x", "home")), E, store)
    blank = next(h for h in ix.search("home", 2, E) if h.memory_id == "blank")
    assert blank.similarity == 0.0
    assert all(h.similarity == 0.0 for h in ix.search("", 2, E))


def test_persistence_reload_and_reproducibility(tmp_path: Path) -> None:
    stores = [ArtifactStore(tmp_path / n) for n in ("a", "b")]
    built = [SemanticIndex.build(CORPUS, E, s) for s in stores]
    assert built[0].digest == built[1].digest  # same state + embedder = same artifact
    assert stores[0].get(built[0].manifest.vectors) == stores[1].get(built[1].manifest.vectors)
    reloaded = SemanticIndex.load(ArtifactStore(tmp_path / "a"), built[0].digest, E)
    assert reloaded == built[0]
    assert reloaded.search("pet dog", 4, E) == built[0].search("pet dog", 4, E)
    reloaded.verify(stores[0], E)


# Pinned: a change means stored indexes would differ (semantic change or platform drift).
GOLDEN_INDEX = "sha256:2a20d20da2f7cf34ec9ee499d9c7d72b2a0faa812111646f52d1b83d1452dac6"


def test_index_matches_golden_digest(store: ArtifactStore) -> None:
    assert SemanticIndex.build(CORPUS, E, store).digest == GOLDEN_INDEX


# --- integrity and compatibility -------------------------------------------------------------


def _path(store: ArtifactStore, digest: str) -> Path:
    h = digest.removeprefix("sha256:")
    return store.root / "objects" / h[:2] / h[2:]


def test_corrupted_vectors_are_detected(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(CORPUS, E, store)
    path = _path(store, ix.manifest.vectors)
    path.chmod(0o644)
    data = bytearray(path.read_bytes())
    data[5] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ArtifactIntegrityError, match="does not match"):
        SemanticIndex.load(store, ix.digest, E)


def test_forged_manifests_are_detected(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(CORPUS, E, store)
    m = ix.manifest
    # Claims one entry fewer than the vector artifact holds.
    short = IndexManifest.model_validate(
        m.model_dump()
        | {"entries": m.entries[:3], "memory_ids": m.memory_ids[:3], "texts": m.texts[:3]}
    )
    store.put_record(short)
    with pytest.raises(IndexIntegrityError, match="bytes"):
        SemanticIndex.load(store, short.digest, E)
    # Points at vectors of another state: loads, but verification against its source fails.
    other = SemanticIndex.build(
        state(mem("x", "a"), mem("y", "b"), mem("z", "c"), mem("w", "d")), E, store
    )
    swapped = IndexManifest.model_validate(m.model_dump() | {"vectors": other.manifest.vectors})
    store.put_record(swapped)
    with pytest.raises(IndexIntegrityError, match="re-embedding"):
        SemanticIndex.load(store, swapped.digest, E).verify(store, E)
    # Claims texts that are not the source state's contents.
    lying = IndexManifest.model_validate(m.model_dump() | {"texts": tuple(reversed(m.texts))})
    store.put_record(lying)
    with pytest.raises(IndexIntegrityError, match="embedded texts"):
        SemanticIndex.load(store, lying.digest, E).verify(store, E)
    with pytest.raises(ValueError, match="align"):
        IndexManifest.model_validate(m.model_dump() | {"texts": m.texts[:1]})
    with pytest.raises(ValueError, match="once"):
        IndexManifest.model_validate(m.model_dump() | {"entries": (m.entries[0],) * 4})


def test_incompatible_embedders_are_refused(store: ArtifactStore) -> None:
    ix = SemanticIndex.build(CORPUS, E, store)
    other = HashedNgramEmbedder(seed=1)
    with pytest.raises(IndexCompatibilityError, match="built by"):
        SemanticIndex.load(store, ix.digest, other)
    with pytest.raises(IndexCompatibilityError, match="built by"):
        ix.search("home", 1, other)
    with pytest.raises(IndexCompatibilityError, match="built by"):
        ix.verify(store, other)
    with pytest.raises(IndexCompatibilityError):
        SemanticIndex.load(store, ix.digest, HashedNgramEmbedder(dimensions=128))


# --- provenance ---------------------------------------------------------------------------------


def test_neighbours_trace_to_versions_and_experiences(store: ArtifactStore) -> None:
    with MemoryLog(":memory:") as log:
        for s in relocation_year().steps:
            form(log, s.experience, EpisodicPolicy(), recorded_at=s.recorded_at)
        st = log.state_as_of(valid_at=day(345), known_at=day(345))
        ix = SemanticIndex.build(st, E, store)
        for hit in ix.search("employer Initech", 5, E):
            version = log.version(hit.version)
            assert version is not None
            assert version.memory_id == hit.memory_id
            (source,) = version.derived_from
            assert log.experience(source) is not None


# --- no neural dependency ------------------------------------------------------------------------


def test_core_runs_without_neural_dependencies() -> None:
    code = (
        "import sys\n"
        "import memoria.embeddings, memoria.semantic, memoria.evaluation, memoria.experiments\n"
        "heavy = {'numpy', 'torch', 'sentence_transformers', 'faiss', 'transformers'}\n"
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
