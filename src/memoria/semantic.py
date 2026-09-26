"""Semantic memory index: embeddings of a memory state as a reproducible artifact.

A :class:`SemanticIndex` embeds the content of every memory in a
:class:`~memoria.core.MemoryState`. Its :class:`IndexManifest` binds, by digest:

- the embedder identity (:class:`~memoria.embeddings.EmbedderSpec`),
- the source state (stored as an artifact, so each entry traces to a memory version,
  and through the log to its experiences),
- each entry's version digest and the digest of the exact preprocessed text embedded,
- the vectors, stored separately as little-endian float32, row-major.

Search is exact cosine similarity over the stored float32 vectors, so an index answers
identically before and after a save/load round trip. Approximate indexes (e.g. FAISS)
are deferred; this exact index is the reference they must be checked against.
"""

from __future__ import annotations

import math
import sys
from array import array
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.artifacts import ArtifactStore
from memoria.core import Digest, MemoryState, Record, content_hash, quantize
from memoria.embeddings import Embedder, EmbedderSpec, Vector, embed, preprocess


class IndexIntegrityError(RuntimeError):
    """A stored index is inconsistent with itself or with the state it claims to index."""


class IndexCompatibilityError(ValueError):
    """An index is used with an embedder other than the one that built it."""


class IndexManifest(Record):
    embedder: EmbedderSpec
    source: Digest  # the MemoryState artifact that was indexed
    entries: tuple[Digest, ...]  # memory version digests, in the state's order
    memory_ids: tuple[str, ...]
    texts: tuple[Digest, ...]  # content_hash of each preprocessed text embedded
    vectors: Digest  # artifact: len(entries) x dimensions little-endian float32
    metric: Literal["cosine"] = "cosine"

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if not len(self.entries) == len(self.memory_ids) == len(self.texts):
            raise ValueError("entries, memory_ids and texts must align")
        if len(set(self.entries)) != len(self.entries):
            raise ValueError("an index lists each memory version once")
        return self


class Neighbor(Record):
    """One search result: a memory version and its similarity to the query."""

    rank: int = Field(ge=1)
    version: Digest
    memory_id: str
    similarity: float  # cosine in [-1, 1], quantised


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


def _cosine(a: Vector, b: Vector) -> float:
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)))


@dataclass(frozen=True)
class SemanticIndex:
    manifest: IndexManifest
    vectors: tuple[Vector, ...]  # float32-rounded, as stored

    @classmethod
    def build(cls, state: MemoryState, embedder: Embedder, store: ArtifactStore) -> SemanticIndex:
        """Embed every memory of ``state``; store the state, vectors and manifest."""
        spec = embedder.spec
        texts = _texts(state)
        data = _pack(embed(embedder, texts))
        manifest = IndexManifest(
            embedder=spec,
            source=store.put_record(state),
            entries=tuple(m.digest for m in state.memories),
            memory_ids=tuple(m.memory_id for m in state.memories),
            texts=tuple(content_hash(preprocess(t, spec.preprocessing)) for t in texts),
            vectors=store.put(data, kind="IndexVectors"),
        )
        store.put_record(manifest)
        return cls(manifest, tuple(_unpack(data, len(texts), spec.dimensions)))

    @property
    def digest(self) -> str:
        return self.manifest.digest

    @classmethod
    def load(cls, store: ArtifactStore, digest: str, embedder: Embedder) -> SemanticIndex:
        """Load a stored index for use with ``embedder``, which must be the one that built it."""
        manifest = store.get_record(IndexManifest, digest)
        if manifest.embedder != embedder.spec:
            raise IndexCompatibilityError(
                f"index {digest} was built by {manifest.embedder.name} "
                f"v{manifest.embedder.version} ({manifest.embedder.digest}); "
                f"got {embedder.spec.name} v{embedder.spec.version} ({embedder.spec.digest})"
            )
        vectors = _unpack(
            store.get(manifest.vectors), len(manifest.entries), manifest.embedder.dimensions
        )
        return cls(manifest, tuple(vectors))

    def search(self, text: str, k: int, embedder: Embedder) -> tuple[Neighbor, ...]:
        """The ``k`` most similar memories, by cosine; ties by memory_id, then digest."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if embedder.spec != self.manifest.embedder:
            raise IndexCompatibilityError("query embedder differs from the index's embedder")
        (query,) = _unpack(_pack(embed(embedder, [text])), 1, self.manifest.embedder.dimensions)
        rows = sorted(
            (
                (quantize(_cosine(query, v)), mid, d)
                for v, mid, d in zip(
                    self.vectors, self.manifest.memory_ids, self.manifest.entries, strict=True
                )
            ),
            key=lambda r: (-r[0], r[1], r[2]),
        )
        return tuple(
            Neighbor(rank=i, version=d, memory_id=mid, similarity=similarity)
            for i, (similarity, mid, d) in enumerate(rows[:k], 1)
        )

    def verify(self, store: ArtifactStore, embedder: Embedder) -> None:
        """Rebuild from the stored source state and require identical texts and vectors."""
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
        if embedder.spec != spec:
            raise IndexCompatibilityError("verification needs the embedder that built the index")
        if _pack(embed(embedder, texts)) != store.get(self.manifest.vectors):
            raise IndexIntegrityError("re-embedding does not reproduce the stored vectors")
