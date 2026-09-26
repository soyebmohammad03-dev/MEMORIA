"""Semantic memory index: embeddings of a memory state as a reproducible artifact.

A :class:`SemanticIndex` embeds the content of every memory in a
:class:`~memoria.core.MemoryState`. Its :class:`IndexManifest` binds, by digest:

- the embedder identity (:class:`~memoria.core.EmbedderSpec`) and index spec,
- the source state (stored as an artifact, so each entry traces to a memory version,
  and through the log to its experiences),
- each entry's version digest, memory id and the digest of the exact preprocessed text
  embedded, in the state's order (the insertion order of the index),
- the vectors, stored separately as little-endian float32, row-major,
- for approximate indexes, the serialised approximate index.

Search scores candidates by exact cosine over the stored float32 vectors, so an index
answers identically before and after a save/load round trip. The exact index considers
every entry and is the reference. An approximate index only proposes candidates; each
is re-scored exactly and ordered deterministically.
"""

from __future__ import annotations

import sys
from array import array
from dataclasses import dataclass, field
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.artifacts import ArtifactStore
from memoria.core import (
    Digest,
    EmbedderSpec,
    ExtensibleRecord,
    IndexSpec,
    MemoryState,
    Record,
    content_hash,
    quantize,
)
from memoria.embeddings import (
    Agreement,
    Embedder,
    Vector,
    agreement,
    cosine,
    embed,
    preprocess,
)
from memoria.vectors import HnswIndex


class IndexIntegrityError(RuntimeError):
    """A stored index is inconsistent with itself or with the state it claims to index."""


class IndexCompatibilityError(ValueError):
    """An index is used with an embedder or state other than the one it was built for."""


class IndexManifest(ExtensibleRecord):
    _evolved = frozenset({"index", "ann"})

    embedder: EmbedderSpec
    source: Digest  # the MemoryState artifact that was indexed
    entries: tuple[Digest, ...]  # memory version digests, in the state's order
    memory_ids: tuple[str, ...]
    texts: tuple[Digest, ...]  # content_hash of each preprocessed text embedded
    vectors: Digest  # artifact: len(entries) x dimensions little-endian float32
    metric: Literal["cosine"] = "cosine"
    index: IndexSpec = IndexSpec()
    ann: Digest | None = None  # serialised approximate index, for non-exact kinds

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if not len(self.entries) == len(self.memory_ids) == len(self.texts):
            raise ValueError("entries, memory_ids and texts must align")
        if len(set(self.entries)) != len(self.entries):
            raise ValueError("an index lists each memory version once")
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("an index lists each memory once (a state holds one version each)")
        if (self.index.kind == "exact") != (self.ann is None):
            raise ValueError("an approximate index artifact is present iff the kind is not exact")
        return self


class Neighbor(Record):
    """One auditable search result.

    ``version`` leads to the memory version, its source experiences and their inputs
    (through the log); ``source`` is the indexed state; ``index`` and ``embedder`` name
    the representation that produced the similarity.
    """

    rank: int = Field(ge=1)
    version: Digest
    memory_id: str
    similarity: float  # exact cosine in [-1, 1], quantised
    text: Digest  # the preprocessed text that was embedded
    source: Digest  # MemoryState digest
    index: Digest  # IndexManifest digest
    embedder: Digest  # EmbedderSpec digest
    representation: Literal["exact", "hnsw"]


def _pack(vectors: list[Vector]) -> bytes:
    flat = array("f", (x for v in vectors for x in v))
    if sys.byteorder == "big":
        flat.byteswap()
    return flat.tobytes()


def _unpack(data: bytes, rows: int, dimensions: int) -> list[Vector]:
    if len(data) != rows * dimensions * 4:
        raise IndexIntegrityError(
            f"vector artifact has {len(data)} bytes, expected {rows} x {dimensions} x 4"
        )
    flat = array("f")
    flat.frombytes(data)
    if sys.byteorder == "big":
        flat.byteswap()
    return [tuple(flat[i * dimensions : (i + 1) * dimensions]) for i in range(rows)]


def _texts(state: MemoryState) -> list[str]:
    return [m.content or "" for m in state.memories]  # states never hold tombstones


@dataclass(frozen=True)
class SemanticIndex:
    manifest: IndexManifest
    vectors: tuple[Vector, ...]  # float32-rounded, as stored
    ann: HnswIndex | None = field(default=None, compare=False, repr=False)

    @classmethod
    def build(
        cls,
        state: MemoryState,
        embedder: Embedder,
        store: ArtifactStore,
        index: IndexSpec | None = None,
    ) -> SemanticIndex:
        """Embed every memory of ``state``; store the state, vectors, any approximate
        index, and the manifest."""
        spec = embedder.spec
        index = index or IndexSpec()
        texts = _texts(state)
        data = _pack(embed(embedder, texts))
        vectors = _unpack(data, len(texts), spec.dimensions)
        ann_digest = None
        if index.kind != "exact":
            ann_digest = store.put(
                HnswIndex.build(vectors, spec.dimensions, index), kind="HnswIndex"
            )
        manifest = IndexManifest(
            embedder=spec,
            source=store.put_record(state),
            entries=tuple(m.digest for m in state.memories),
            memory_ids=tuple(m.memory_id for m in state.memories),
            texts=tuple(content_hash(preprocess(t, spec.preprocessing)) for t in texts),
            vectors=store.put(data, kind="IndexVectors"),
            index=index,
            ann=ann_digest,
        )
        store.put_record(manifest)
        return cls._open(manifest, vectors, store)

    @classmethod
    def _open(
        cls, manifest: IndexManifest, vectors: list[Vector], store: ArtifactStore
    ) -> SemanticIndex:
        ann = None
        if manifest.ann is not None:
            try:
                ann = HnswIndex.load(
                    store.get(manifest.ann),
                    len(manifest.entries),
                    manifest.embedder.dimensions,
                    manifest.index,
                )
            except ValueError as e:
                raise IndexIntegrityError(f"approximate index: {e}") from e
        return cls(manifest, tuple(vectors), ann)

    @property
    def digest(self) -> str:
        return self.manifest.digest

    @classmethod
    def load(cls, store: ArtifactStore, digest: str, embedder: Embedder) -> SemanticIndex:
        """Load a stored index for use with ``embedder``, which must be the one that built it."""
        manifest = store.get_record(IndexManifest, digest)
        _require_embedder(manifest, embedder)
        vectors = _unpack(
            store.get(manifest.vectors), len(manifest.entries), manifest.embedder.dimensions
        )
        return cls._open(manifest, vectors, store)

    def require_source(self, state: MemoryState) -> None:
        """Refuse to answer for a state other than the one indexed (a stale index)."""
        if state.digest != self.manifest.source:
            raise IndexCompatibilityError(
                f"stale index: built from state {self.manifest.source}, asked about {state.digest}"
            )

    def query_vector(self, text: str, embedder: Embedder) -> Vector:
        _require_embedder(self.manifest, embedder)
        (vector,) = _unpack(_pack(embed(embedder, [text])), 1, self.manifest.embedder.dimensions)
        return vector

    def similarities(self, text: str, embedder: Embedder) -> dict[str, float]:
        """Exact cosine of ``text`` with every entry, keyed by version digest."""
        q = self.query_vector(text, embedder)
        return {
            d: quantize(cosine(q, v))
            for d, v in zip(self.manifest.entries, self.vectors, strict=True)
        }

    def search(self, text: str, k: int, embedder: Embedder) -> tuple[Neighbor, ...]:
        """The ``k`` most similar memories by exact cosine; ties by memory_id, then digest.

        An exact index scores every entry. An approximate index scores only the
        candidates its graph proposes; what that loses is measured, not hidden.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        q = self.query_vector(text, embedder)
        m = self.manifest
        positions = range(len(m.entries)) if self.ann is None else self.ann.candidates(q, k)
        rows = sorted(
            (
                (quantize(cosine(q, self.vectors[i])), m.memory_ids[i], m.entries[i], i)
                for i in positions
            ),
            key=lambda r: (-r[0], r[1], r[2]),
        )
        return tuple(
            Neighbor(
                rank=rank,
                version=d,
                memory_id=mid,
                similarity=similarity,
                text=m.texts[i],
                source=m.source,
                index=self.digest,
                embedder=m.embedder.digest,
                representation=m.index.kind,
            )
            for rank, (similarity, mid, d, i) in enumerate(rows[:k], 1)
        )

    def verify(self, store: ArtifactStore, embedder: Embedder) -> Agreement:
        """Rebuild from the stored source state: texts must match exactly; vectors must be
        identical, or equivalent within the embedder's declared tolerance."""
        state = store.get_record(MemoryState, self.manifest.source)
        spec = self.manifest.embedder
        texts = _texts(state)
        if tuple(m.digest for m in state.memories) != self.manifest.entries:
            raise IndexIntegrityError("index entries are not the source state's memories")
        if (
            tuple(content_hash(preprocess(t, spec.preprocessing)) for t in texts)
            != self.manifest.texts
        ):
            raise IndexIntegrityError("embedded texts differ from the source state's contents")
        _require_embedder(self.manifest, embedder)
        rebuilt = _unpack(_pack(embed(embedder, texts)), len(texts), spec.dimensions)
        result = agreement(rebuilt, self.vectors, spec.tolerance)
        if result is Agreement.DIFFERENT:
            raise IndexIntegrityError("re-embedding does not reproduce the stored vectors")
        if self.manifest.ann is not None and result is Agreement.IDENTICAL:
            again = HnswIndex.build(rebuilt, spec.dimensions, self.manifest.index)
            if again != store.get(self.manifest.ann):
                raise IndexIntegrityError("rebuilding does not reproduce the approximate index")
        return result


def _require_embedder(manifest: IndexManifest, embedder: Embedder) -> None:
    if manifest.embedder != embedder.spec:
        e, g = manifest.embedder, embedder.spec
        raise IndexCompatibilityError(
            f"index was built by {e.name} v{e.version} ({e.digest}); "
            f"got {g.name} v{g.version} ({g.digest})"
        )
