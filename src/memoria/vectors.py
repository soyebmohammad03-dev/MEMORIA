"""Approximate nearest-neighbour candidate generation (optional extra: ``memoria[ann]``).

The exact index in :mod:`memoria.semantic` is the scientific reference. An approximate
index only proposes candidates: :mod:`memoria.semantic` re-scores every candidate with
exact cosine similarity and orders them deterministically, so approximation can change
*which* memories are returned, never the scores they report. What it loses is measured
by :func:`memoria.semantic_eval.compare_indexes`.

:class:`HnswIndex` wraps FAISS ``IndexHNSWFlat`` (Malkov & Yashunin, 2018) over
inner product on L2-normalised vectors, i.e. cosine. It is built and searched with one
thread; HNSW level assignment uses FAISS's fixed internal seed, so a build is a
deterministic function of the vectors, their order and the parameters. The default
parameters are FAISS's own defaults (M = 16 is also the HNSW paper's typical setting).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memoria.core import IndexSpec

BACKEND = "faiss"


class AnnUnavailableError(RuntimeError):
    """The approximate-index backend is not installed. Nothing falls back silently."""


def hnsw_spec(m: int = 16, ef_construction: int = 40, ef_search: int = 16) -> IndexSpec:
    if min(m, ef_construction, ef_search) < 1:
        raise ValueError("HNSW parameters must be positive")
    return IndexSpec(
        kind="hnsw",
        backend=BACKEND,
        params=(("ef_construction", ef_construction), ("ef_search", ef_search), ("m", m)),
    )


def _backend() -> tuple[Any, Any]:
    """FAISS (an untyped C extension) and numpy, or a diagnostic error if either is missing."""
    try:
        import faiss
        import numpy
    except ImportError as e:
        raise AnnUnavailableError(
            "approximate indexes need the 'ann' extra (faiss-cpu, numpy)"
        ) from e
    faiss.omp_set_num_threads(1)
    return faiss, numpy


class HnswIndex:
    """A FAISS HNSW graph over row vectors; returns candidate row positions."""

    def __init__(self, index: Any, spec: IndexSpec) -> None:
        self._index = index
        self._ef_search = int(dict(spec.params)["ef_search"])

    @staticmethod
    def build(vectors: Sequence[Sequence[float]], dimensions: int, spec: IndexSpec) -> bytes:
        """Build and serialise. The bytes are stored as an artifact."""
        if spec.kind != "hnsw" or spec.backend != BACKEND:
            raise ValueError(f"not an HNSW/{BACKEND} spec: {spec}")
        faiss, np = _backend()
        params = dict(spec.params)
        index = faiss.IndexHNSWFlat(dimensions, int(params["m"]), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(params["ef_construction"])
        if vectors:
            index.add(np.asarray(vectors, dtype=np.float32).reshape(len(vectors), dimensions))
        return bytes(faiss.serialize_index(index).tobytes())

    @classmethod
    def load(cls, data: bytes, rows: int, dimensions: int, spec: IndexSpec) -> HnswIndex:
        faiss, np = _backend()
        try:
            index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
        except RuntimeError as e:
            raise ValueError(f"not a FAISS index: {e}") from e
        if (index.ntotal, index.d) != (rows, dimensions):
            raise ValueError(
                f"FAISS index holds {index.ntotal} x {index.d}, expected {rows} x {dimensions}"
            )
        return cls(index, spec)

    def candidates(self, query: Sequence[float], k: int) -> list[int]:
        """Up to ``k`` row positions proposed by the graph search (unordered for callers)."""
        _, np = _backend()
        index = self._index
        total = int(index.ntotal)
        if total == 0:
            return []
        index.hnsw.efSearch = self._ef_search  # FAISS itself searches with max(efSearch, k)
        _, found = index.search(np.asarray([query], dtype=np.float32), min(k, total))
        return [int(i) for i in found[0] if i >= 0]
