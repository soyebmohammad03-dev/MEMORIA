"""Retrieval over a memory state, with complete, reproducible evidence.

A :class:`Retriever` scores every memory in a :class:`~memoria.core.MemoryState` with
each of its signals, gates eligibility on one signal, ranks by the weighted sum, and
returns a :class:`~memoria.core.RetrievalTrace` that records every signal's score and
raw inputs for every candidate, eligible or not. Nothing here performs I/O except
:func:`answer`, which runs the whole pipeline against a log and records the result.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import ClassVar, Literal, Protocol

from memoria.core import (
    Candidate,
    MemoryState,
    MemoryVersion,
    Query,
    Response,
    RetrievalTrace,
    RetrieverSpec,
    Scalar,
    SignalEvidence,
    SignalSpec,
    combine,
    quantize,
    rank_key,
)
from memoria.embeddings import Embedder, Vector, cosine, embed, float32
from memoria.store import MemoryLog

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Unicode word tokens, case-folded. No stemming or stop-word removal (see BM25 idf)."""
    return _TOKEN.findall(text.casefold())


class Signal(Protocol):
    """Scores every memory of a state for a query. Must be deterministic."""

    name: ClassVar[str]

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]: ...

    def score(self, query: Query, memories: Sequence[MemoryVersion]) -> list[SignalEvidence]:
        """One evidence record per memory, in the given order."""
        ...


@dataclass(frozen=True)
class BM25:
    """Okapi BM25 over memory content, with corpus statistics from the queried state.

    idf = ln(1 + (N - df + 0.5) / (df + 0.5)), which is always positive. Query terms
    are de-duplicated. Per matched term the evidence records tf, df, idf and the
    term's contribution; per memory, its length and the state's average length.
    """

    k1: float = 1.2
    b: float = 0.75
    name: ClassVar[str] = "bm25"

    def __post_init__(self) -> None:
        if not (self.k1 >= 0 and 0 <= self.b <= 1):
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return (("b", self.b), ("k1", self.k1))

    def score(self, query: Query, memories: Sequence[MemoryVersion]) -> list[SignalEvidence]:
        docs = [tokenize(m.content or "") for m in memories]
        n = len(docs)
        avg_len = sum(map(len, docs)) / n if n else 0.0
        df = Counter(t for d in docs for t in set(d))
        terms = sorted(set(tokenize(query.text)))
        evidence = []
        for doc in docs:
            tf = Counter(doc)
            inputs: list[tuple[str, Scalar]] = [
                ("avg_doc_len", quantize(avg_len)),
                ("doc_len", len(doc)),
                ("n_docs", n),
                ("query_terms", " ".join(terms)),
            ]
            total = 0.0
            for t in terms:
                if not tf[t]:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                norm = (
                    tf[t]
                    * (self.k1 + 1)
                    / (tf[t] + self.k1 * (1 - self.b + self.b * len(doc) / avg_len))
                )
                total += idf * norm
                inputs += [
                    (f"tf:{t}", tf[t]),
                    (f"df:{t}", df[t]),
                    (f"idf:{t}", quantize(idf)),
                    (f"contribution:{t}", quantize(idf * norm)),
                ]
            evidence.append(
                SignalEvidence(signal=self.name, score=quantize(total), inputs=tuple(inputs))
            )
        return evidence


@dataclass(frozen=True)
class Recency:
    """Exponential decay 0.5 ** (age / half_life), in days.

    ``axis="recorded"``: age since the version was recorded, as of ``query.known_at``
    (how recently the system learned it). ``axis="valid"``: age since the belief began,
    as of ``query.valid_at`` (how recently it became true).
    """

    half_life_days: float = 30.0
    axis: Literal["recorded", "valid"] = "recorded"
    name: ClassVar[str] = "recency"

    def __post_init__(self) -> None:
        if not self.half_life_days > 0 or self.axis not in ("recorded", "valid"):
            raise ValueError("Recency requires half_life_days > 0 and axis 'recorded'|'valid'")

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return (("axis", self.axis), ("half_life_days", self.half_life_days))

    def score(self, query: Query, memories: Sequence[MemoryVersion]) -> list[SignalEvidence]:
        evidence = []
        for m in memories:
            if self.axis == "recorded":
                age = query.known_at - m.recorded_at
            else:
                assert m.valid_from is not None  # states never contain tombstones
                age = query.valid_at - m.valid_from
            days = age / timedelta(days=1)
            evidence.append(
                SignalEvidence(
                    signal=self.name,
                    score=quantize(0.5 ** (days / self.half_life_days)),
                    inputs=(("age_days", quantize(days)),),
                )
            )
        return evidence


class SemanticSignal:
    """Exact cosine between the query and each memory's content, under one embedder.

    Score = max(cosine, 0); the raw cosine is recorded as an input. As a gate, a memory
    is eligible iff its cosine is positive (positively aligned with the query) — the
    natural boundary, not a tuned threshold. Vectors are rounded to float32, as stored
    in a :class:`~memoria.semantic.SemanticIndex`, so both report the same similarity.
    Embeddings are memoised by text within one signal instance (a pure function).
    """

    name: ClassVar[str] = "semantic"

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._memo: dict[str, Vector] = {}

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return (("embedder", self._embedder.spec.digest),)

    def _vectors(self, texts: Sequence[str]) -> list[Vector]:
        missing = sorted({t for t in texts if t not in self._memo})
        for t, v in zip(missing, embed(self._embedder, missing), strict=True):
            self._memo[t] = float32(v)
        return [self._memo[t] for t in texts]

    def score(self, query: Query, memories: Sequence[MemoryVersion]) -> list[SignalEvidence]:
        (q,) = self._vectors([query.text])
        evidence = []
        for v in self._vectors([m.content or "" for m in memories]):
            c = quantize(cosine(q, v))
            evidence.append(
                SignalEvidence(signal=self.name, score=max(c, 0.0), inputs=(("cosine", c),))
            )
        return evidence


@dataclass(frozen=True)
class Retriever:
    """A named, fully specified ranking configuration."""

    name: str
    signals: tuple[tuple[Signal, float], ...]  # (signal, weight)
    gate: str

    @property
    def spec(self) -> RetrieverSpec:
        return RetrieverSpec(
            name=self.name,
            gate=self.gate,
            signals=tuple(
                SignalSpec(name=s.name, weight=w, params=s.params) for s, w in self.signals
            ),
        )

    def retrieve(self, state: MemoryState, query: Query) -> RetrievalTrace:
        if (state.valid_at, state.known_at) != (query.valid_at, query.known_at):
            raise ValueError("state is not at the query's time coordinates")
        spec = self.spec
        per_signal = [s.score(query, state.memories) for s, _ in self.signals]
        for (signal, _), scores in zip(self.signals, per_signal, strict=True):
            if len(scores) != len(state.memories) or {e.signal for e in scores} - {signal.name}:
                raise ValueError(f"signal {signal.name!r} must score each memory exactly once")
        candidates = []
        for i, m in enumerate(state.memories):
            evidence = tuple(scores[i] for scores in per_signal)
            eligible, total = combine(spec, evidence)
            candidates.append(
                Candidate(
                    version=m.digest,
                    memory_id=m.memory_id,
                    eligible=eligible,
                    total=total,
                    signals=evidence,
                )
            )
        candidates.sort(key=rank_key)
        selected = tuple(c.version for c in candidates if c.eligible)[: query.limit]
        return RetrievalTrace(
            query=query,
            retriever=spec,
            state=state.digest,
            candidates=tuple(candidates),
            selected=selected,
        )


def lexical(k1: float = 1.2, b: float = 0.75) -> Retriever:
    """BM25 only: the lexical baseline."""
    return Retriever("bm25-v1", ((BM25(k1, b), 1.0),), gate=BM25.name)


def lexical_recency(half_life_days: float = 30.0, recency_weight: float = 0.5) -> Retriever:
    """BM25-gated, ranked by BM25 plus a recorded-time recency term."""
    return Retriever(
        "bm25+recency-v1",
        ((BM25(), 1.0), (Recency(half_life_days), recency_weight)),
        gate=BM25.name,
    )


def semantic(embedder: Embedder) -> Retriever:
    """Semantic similarity only: the vector-representation counterpart of ``lexical``."""
    return Retriever("semantic-v1", ((SemanticSignal(embedder), 1.0),), gate=SemanticSignal.name)


EXTRACTIVE = "extractive-top1-v1"


def extractive(trace: RetrievalTrace, state: MemoryState) -> Response:
    """Answer with the top selected memory's content verbatim; abstain if none."""
    if not trace.selected:
        return Response(trace=trace.digest, responder=EXTRACTIVE, output=None)
    top = next(m for m in state.memories if m.digest == trace.selected[0])
    return Response(
        trace=trace.digest, responder=EXTRACTIVE, output=top.content, cited=(top.digest,)
    )


Responder = Callable[[RetrievalTrace, MemoryState], Response]


def answer(
    log: MemoryLog, retriever: Retriever, query: Query, responder: Responder = extractive
) -> tuple[RetrievalTrace, Response]:
    """Retrieve from the log's state at the query's coordinates, respond, record both."""
    with log.transaction():
        state = log.state_as_of(valid_at=query.valid_at, known_at=query.known_at)
        trace = retriever.retrieve(state, query)
        response = responder(trace, state)
        log.append_record(trace)
        log.append_record(response)
    return trace, response
