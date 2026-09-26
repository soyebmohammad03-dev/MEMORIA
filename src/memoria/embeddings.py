"""Embedding contract: text -> fixed-dimension vectors, with a recorded identity.

An :class:`Embedder` is identified by its :class:`EmbedderSpec` — name, version,
dimensionality, preprocessing and every parameter that affects its output. Two
embedders with equal specs must produce equal vectors for equal input; vectors from
different specs are never compared. Embeddings are experimental artifacts, not an
implementation detail: :func:`embed` validates every vector it returns.

:class:`HashedNgramEmbedder` is the deterministic, dependency-free reference
implementation. It is **not** a semantic or neural model: it hashes character n-grams
into signed buckets (the "hashing trick", Weinberger et al., ICML 2009) and L2-normalises
the counts. Similar spellings get similar vectors; synonyms do not. It exists so the
semantic-memory infrastructure can be built and tested without downloading a model, and
as a lexical-subword baseline. Neural embedders plug in behind the same contract.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from array import array
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from memoria.core import EmbedderSpec, Preprocessing, Scalar, Tolerance

__all__ = [
    "Agreement",
    "Embedder",
    "EmbedderSpec",
    "EmbeddingError",
    "HashedNgramEmbedder",
    "Preprocessing",
    "Vector",
    "agreement",
    "cosine",
    "embed",
    "float32",
    "preprocess",
]

Vector = tuple[float, ...]


class EmbeddingError(ValueError):
    """An embedder produced output that violates its contract."""


def preprocess(text: str, p: Preprocessing) -> str:
    if p.unicode == "NFKC":
        text = unicodedata.normalize("NFKC", text)
    if p.casefold:
        text = text.casefold()
    if p.collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    if p.max_chars is not None:
        text = text[: p.max_chars]
    return text


class Embedder(Protocol):
    """Maps texts to vectors of ``spec.dimensions`` floats. Must be deterministic."""

    @property
    def spec(self) -> EmbedderSpec: ...

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        """One vector per text, in order. ``texts`` are already preprocessed."""
        ...


def embed(embedder: Embedder, texts: Sequence[str]) -> list[Vector]:
    """Preprocess, encode and validate: count, dimensionality, finiteness, normalisation."""
    spec = embedder.spec
    prepared = [preprocess(t, spec.preprocessing) for t in texts]
    raw = embedder.encode(prepared) if prepared else []
    if len(raw) != len(prepared):
        raise EmbeddingError(f"{spec.name}: {len(raw)} vectors for {len(prepared)} texts")
    out = []
    for i, v in enumerate(raw):
        vector = tuple(float(x) for x in v)
        if len(vector) != spec.dimensions:
            raise EmbeddingError(
                f"{spec.name}: vector {i} has {len(vector)} dimensions, spec says {spec.dimensions}"
            )
        if not all(math.isfinite(x) for x in vector):
            raise EmbeddingError(f"{spec.name}: vector {i} is not finite")
        if spec.normalized:
            norm = math.sqrt(math.fsum(x * x for x in vector))
            if norm and abs(norm - 1) > 1e-6:
                raise EmbeddingError(f"{spec.name}: vector {i} is not L2-normalised (norm {norm})")
        out.append(vector)
    return out


def float32(vector: Sequence[float]) -> Vector:
    """The vector as stored: each component rounded to IEEE float32."""
    return tuple(array("f", vector))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 0 when either vector is zero."""
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)))


class HashedNgramEmbedder:
    """Signed feature hashing of character n-grams; deterministic on every platform.

    Each n-gram of ``" " + text + " "`` (for n in [n_min, n_max]) is hashed with SHA-256
    together with ``seed``: 8 bytes choose a bucket, one bit chooses a sign. Bucket counts
    are integers, so the vector is exact up to one correctly-rounded division by the
    L2 norm. Empty text maps to the zero vector.
    """

    NAME = "hashed-char-ngrams"
    VERSION = "1"

    def __init__(
        self,
        dimensions: int = 256,
        n_min: int = 3,
        n_max: int = 5,
        seed: int = 0,
        preprocessing: Preprocessing | None = None,
    ) -> None:
        if not 1 <= n_min <= n_max:
            raise ValueError("need 1 <= n_min <= n_max")
        self._n = (n_min, n_max)
        self._seed = seed
        params: tuple[tuple[str, Scalar], ...] = (
            ("n_max", n_max),
            ("n_min", n_min),
            ("seed", seed),
        )
        self._spec = EmbedderSpec(
            name=self.NAME,
            version=self.VERSION,
            dimensions=dimensions,
            preprocessing=preprocessing or Preprocessing(),
            normalized=True,
            params=params,
        )

    @classmethod
    def from_spec(cls, spec: EmbedderSpec) -> HashedNgramEmbedder:
        params = dict(spec.params)
        built = cls(
            dimensions=spec.dimensions,
            n_min=int(params.get("n_min", 0)),
            n_max=int(params.get("n_max", 0)),
            seed=int(params.get("seed", 0)),
            preprocessing=spec.preprocessing,
        )
        if built.spec != spec:
            raise ValueError(f"{spec.name}: spec does not describe this embedder")
        return built

    @property
    def spec(self) -> EmbedderSpec:
        return self._spec

    def encode(self, texts: Sequence[str]) -> list[Sequence[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        counts = [0] * self._spec.dimensions
        if text:
            padded = f" {text} "
            for n in range(self._n[0], self._n[1] + 1):
                for i in range(len(padded) - n + 1):
                    h = hashlib.sha256(f"{self._seed}\x00{padded[i : i + n]}".encode()).digest()
                    bucket = int.from_bytes(h[:8], "big") % self._spec.dimensions
                    counts[bucket] += 1 if h[8] & 1 else -1
        norm = math.sqrt(sum(c * c for c in counts))
        return [c / norm for c in counts] if norm else [0.0] * self._spec.dimensions


class Agreement(StrEnum):
    """How two sets of vectors from the same spec compare (ARCHITECTURE.md §4.8)."""

    IDENTICAL = "identical"  # byte-for-byte the same float32 values
    EQUIVALENT = "equivalent"  # within the spec's declared tolerance
    DIFFERENT = "different"  # outside it (or any difference, with no tolerance declared)


def agreement(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], tolerance: Tolerance | None
) -> Agreement:
    """Classify two equally-shaped vector sets. Without a tolerance only identity passes."""
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b, strict=True)):
        return Agreement.DIFFERENT
    if all(tuple(x) == tuple(y) for x, y in zip(a, b, strict=True)):
        return Agreement.IDENTICAL
    if tolerance is None:
        return Agreement.DIFFERENT
    for x, y in zip(a, b, strict=True):
        if max((abs(p - q) for p, q in zip(x, y, strict=True)), default=0.0) > tolerance.max_abs:
            return Agreement.DIFFERENT
        nx = math.sqrt(math.fsum(p * p for p in x))
        ny = math.sqrt(math.fsum(q * q for q in y))
        if nx and ny:
            if (
                math.fsum(p * q for p, q in zip(x, y, strict=True)) / (nx * ny)
                < tolerance.min_cosine
            ):
                return Agreement.DIFFERENT
        elif nx or ny:
            return Agreement.DIFFERENT
    return Agreement.EQUIVALENT
