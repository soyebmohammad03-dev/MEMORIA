"""Representation experiments: how vector representations of memory behave.

Two questions are kept apart:

- **Representation quality** — how similar each kind of variant is to a query under a
  representation (per-label similarity profiles, and whether meaning-preserving
  variants are separated from meaning-changing ones). No retrieval policy involved.
- **Retrieval quality** — what top-k cosine retrieval over an index returns: recall,
  precision, reciprocal rank, false matches and misses by label, and paired differences
  between representations. This uses the simplest policy (rank by similarity); hybrid
  policies are Phase 6.

:func:`run_semantic_experiment` turns a :class:`SemanticExperimentSpec` into a
content-addressed :class:`SemanticExperiment`; timings go to a separate
:class:`PerformanceRecord`, because they describe the machine, not the result.
:func:`reproduce_semantic` re-runs a stored experiment and classifies the outcome as
identical, equivalent within the embedders' declared tolerance, or different.
:func:`compare_indexes` measures what an approximate index loses against the exact one.
"""

from __future__ import annotations

import math
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Protocol, Self

from pydantic import Field, model_validator

from memoria.artifacts import ArtifactStore
from memoria.core import (
    Dataset,
    Digest,
    EmbedderSpec,
    MemoryState,
    MemoryVersion,
    Operation,
    Record,
    RepresentationSpec,
    Tolerance,
    content_hash,
    quantize,
)
from memoria.embeddings import Agreement, Embedder, agreement, embed
from memoria.formation import EpisodicPolicy, form
from memoria.semantic import SemanticIndex
from memoria.statistics import Proportion, mcnemar_exact, newcombe_paired, significant
from memoria.store import MemoryLog


class Relevance(StrEnum):
    """A candidate's designed relationship to a query's subject and attribute."""

    PARAPHRASE = "paraphrase"  # the same fact, reworded with shared words
    EQUIVALENT = "equivalent"  # the same fact with little shared wording
    OTHER_SOURCE = "other_source"  # the same fact, attributed to another source
    TEMPORAL_VARIANT = "temporal_variant"  # same subject and attribute, another time
    CONTRADICTION = "contradiction"  # same subject and attribute, incompatible value
    NEGATION = "negation"  # the fact negated
    NUMERIC_CHANGE = "numeric_change"  # the fact with a different number
    ENTITY_SUBSTITUTION = "entity_substitution"  # the same statement about another entity
    LEXICAL_DISTRACTOR = "lexical_distractor"  # shares most words, a different fact
    SHARED_VOCABULARY = "shared_vocabulary"  # shares topic words, an unrelated fact

    @property
    def relevant(self) -> bool:
        """About the query's subject and attribute — evidence a memory system should
        surface, including contradictions and old values (resolving them comes later)."""
        return self not in _NOT_RELEVANT

    @property
    def meaning_preserving(self) -> bool:
        return self in (Relevance.PARAPHRASE, Relevance.EQUIVALENT, Relevance.OTHER_SOURCE)


_NOT_RELEVANT = frozenset(
    {Relevance.ENTITY_SUBSTITUTION, Relevance.LEXICAL_DISTRACTOR, Relevance.SHARED_VOCABULARY}
)
UNRELATED = "unrelated"  # label reported for unjudged candidates


class Judgement(Record):
    candidate: str  # the candidate experience's source id
    relevance: Relevance


class SemanticQuery(Record):
    id: str = Field(min_length=1)
    text: str
    judgements: tuple[Judgement, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        ids = [j.candidate for j in self.judgements]
        if len(set(ids)) != len(ids):
            raise ValueError("each candidate is judged at most once per query")
        return self


class SemanticBenchmark(Record):
    """Queries with relevance judgements over the experiences of a dataset."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dataset: Digest
    queries: tuple[SemanticQuery, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        ids = [q.id for q in self.queries]
        if len(set(ids)) != len(ids):
            raise ValueError("query ids must be unique")
        return self


class SemanticExperimentSpec(Record):
    """Everything that determines a representation experiment. Its digest is its ID."""

    name: str = Field(min_length=1)
    benchmark: Digest
    representations: tuple[RepresentationSpec, ...] = Field(min_length=1)
    ks: tuple[int, ...] = (1, 3, 5)
    confidence: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if not self.ks or list(self.ks) != sorted(set(self.ks)) or self.ks[0] < 1:
            raise ValueError("ks must be distinct positive integers in increasing order")
        digests = [r.digest for r in self.representations]
        if len(set(digests)) != len(digests):
            raise ValueError("representations must be distinct")
        return self


# --- results --------------------------------------------------------------------------------


class Hit(Record):
    candidate: str
    label: str  # a Relevance value, or "unrelated"
    rank: int = Field(ge=1)
    similarity: float
    version: Digest


class QueryResult(Record):
    query: str
    retrieved: tuple[Hit, ...]  # top max(ks), exact ranking
    relevant: tuple[str, ...]  # candidate ids judged relevant
    first_relevant_rank: int | None
    false_matches: tuple[Hit, ...]  # not relevant, within the top max(ks)
    misses: tuple[str, ...]  # relevant, outside the top max(ks)


class LabelProfile(Record):
    """Similarity to the query of every judged candidate with one label."""

    label: str
    n: int = Field(ge=0)
    mean: float | None
    minimum: float | None
    maximum: float | None


class KMetrics(Record):
    k: int
    recall: Proportion  # relevant retrieved / relevant, pooled over queries
    precision: Proportion  # relevant retrieved / retrieved, pooled over queries
    attainable: int  # the most relevant hits possible at k: sum of min(k, |relevant|)


class RepresentationResult(Record):
    representation: Digest  # RepresentationSpec digest
    index: Digest  # IndexManifest digest
    queries: tuple[QueryResult, ...]
    at: tuple[KMetrics, ...]
    mean_reciprocal_rank: float | None
    profiles: tuple[LabelProfile, ...]
    # Queries whose least similar meaning-preserving variant is more similar than every
    # meaning-changing one (negation, number, entity, contradiction, temporal).
    separation: Proportion


class PairedRecall(Record):
    """Hit or miss of each (query, relevant candidate) at k, paired across representations.

    Pairs share queries, so they are clustered; the interval treats them as independent
    and may be optimistic (ARCHITECTURE.md §4.5).
    """

    k: int
    first: Digest
    second: Digest
    pairs: int
    both: int
    first_only: int
    second_only: int
    neither: int
    difference: float | None  # second minus first
    low: float | None
    high: float | None
    p_value: float | None


class Overlap(Record):
    """Top-k agreement between two representations on one query."""

    query: str
    k: int
    shared: tuple[str, ...]
    only_first: tuple[str, ...]
    only_second: tuple[str, ...]


class SemanticExperiment(Record):
    spec: Digest
    benchmark: Digest
    dataset: Digest
    log: Digest  # canonical export of the memory log built from the dataset
    state: Digest
    results: tuple[RepresentationResult, ...]
    paired: tuple[PairedRecall, ...]
    overlaps: tuple[Overlap, ...]


class EmbedderResolver(Protocol):
    """Anything that builds the embedder a spec describes (``experiments.Registry``)."""

    def embedder(self, spec: EmbedderSpec) -> Embedder: ...


# Time coordinates for an empty dataset's (empty) state; any fixed instant would do.
_EMPTY_AT = datetime(1970, 1, 1, tzinfo=UTC)


# --- the experiment ----------------------------------------------------------------------------


def _memory(dataset: Dataset, store: ArtifactStore) -> tuple[MemoryState, str, dict[str, str]]:
    """Ingest the dataset verbatim (one episodic memory per experience)."""
    with MemoryLog(":memory:") as log:
        for step in dataset.steps:
            form(log, step.experience, EpisodicPolicy(), recorded_at=step.recorded_at)
        last = dataset.steps[-1].recorded_at if dataset.steps else None
        at = last or _EMPTY_AT
        state = log.state_as_of(valid_at=at, known_at=at)
        exported = store.put(log.export(), kind="MemoryLogExport")
        candidate_of: dict[str, str] = {}
        for m in state.memories:
            (source,) = m.derived_from
            experience = log.experience(source)
            assert experience is not None  # the log verified every citation
            candidate_of[m.digest] = experience.source
    return state, exported, candidate_of


def _label(query: SemanticQuery) -> dict[str, str]:
    return {j.candidate: j.relevance.value for j in query.judgements}


def _profiles(sims: dict[str, list[float]]) -> tuple[LabelProfile, ...]:
    out = []
    for label in [*(r.value for r in Relevance), UNRELATED]:
        values = sims.get(label, [])
        out.append(
            LabelProfile(
                label=label,
                n=len(values),
                mean=quantize(math.fsum(values) / len(values)) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
            )
        )
    return tuple(out)


def _evaluate(
    index: SemanticIndex,
    embedder: Embedder,
    benchmark: SemanticBenchmark,
    candidate_of: dict[str, str],
    spec: SemanticExperimentSpec,
    representation: RepresentationSpec,
) -> RepresentationResult:
    depth = spec.ks[-1]
    results = []
    sims_by_label: dict[str, list[float]] = {}
    separated = 0
    reciprocal = []
    for q in benchmark.queries:
        labels = _label(q)
        relevant = tuple(c for c, lab in labels.items() if Relevance(lab).relevant)
        hits = tuple(
            Hit(
                candidate=candidate_of[n.version],
                label=labels.get(candidate_of[n.version], UNRELATED),
                rank=n.rank,
                similarity=n.similarity,
                version=n.version,
            )
            for n in index.search(q.text, depth, embedder)
        )
        found = [h.rank for h in hits if h.candidate in relevant]
        first = min(found) if found else None
        reciprocal.append(1 / first if first else 0.0)
        similarity = {candidate_of[d]: s for d, s in index.similarities(q.text, embedder).items()}
        preserving: list[float] = []
        changing: list[float] = []
        for cand, sim in similarity.items():
            label = labels.get(cand, UNRELATED)
            sims_by_label.setdefault(label, []).append(sim)
            if label != UNRELATED:
                (preserving if Relevance(label).meaning_preserving else changing).append(sim)
        if preserving and changing and min(preserving) > max(changing):
            separated += 1
        results.append(
            QueryResult(
                query=q.id,
                retrieved=hits,
                relevant=relevant,
                first_relevant_rank=first,
                false_matches=tuple(h for h in hits if h.candidate not in relevant),
                misses=tuple(c for c in relevant if c not in {h.candidate for h in hits}),
            )
        )
    at = []
    for k in spec.ks:
        got = sum(1 for r in results for h in r.retrieved[:k] if h.candidate in r.relevant)
        wanted = sum(len(r.relevant) for r in results)
        returned = sum(len(r.retrieved[:k]) for r in results)
        at.append(
            KMetrics(
                k=k,
                recall=Proportion.of(got, wanted, spec.confidence),
                precision=Proportion.of(got, returned, spec.confidence),
                attainable=sum(min(k, len(r.relevant)) for r in results),
            )
        )
    return RepresentationResult(
        representation=representation.digest,
        index=index.digest,
        queries=tuple(results),
        at=tuple(at),
        mean_reciprocal_rank=quantize(math.fsum(reciprocal) / len(reciprocal))
        if reciprocal
        else None,
        profiles=_profiles(sims_by_label),
        separation=Proportion.of(separated, len(benchmark.queries), spec.confidence),
    )


def _paired(
    k: int, a: RepresentationResult, b: RepresentationResult, confidence: float
) -> PairedRecall:
    cells = {"both": 0, "first_only": 0, "second_only": 0, "neither": 0}
    for qa, qb in zip(a.queries, b.queries, strict=True):
        top_a = {h.candidate for h in qa.retrieved[:k]}
        top_b = {h.candidate for h in qb.retrieved[:k]}
        for c in qa.relevant:
            key = {
                (True, True): "both",
                (True, False): "first_only",
                (False, True): "second_only",
            }.get((c in top_a, c in top_b), "neither")
            cells[key] += 1
    n = sum(cells.values())
    # Newcombe's cells: second positive in both / second only / first only / neither.
    diff = newcombe_paired(
        cells["both"], cells["second_only"], cells["first_only"], cells["neither"], confidence
    )
    return PairedRecall(
        k=k,
        first=a.representation,
        second=b.representation,
        pairs=n,
        **cells,
        difference=quantize(diff[0]) if diff else None,
        low=quantize(diff[1]) if diff else None,
        high=quantize(diff[2]) if diff else None,
        p_value=significant(mcnemar_exact(cells["first_only"], cells["second_only"]))
        if n
        else None,
    )


def _overlaps(k: int, a: RepresentationResult, b: RepresentationResult) -> list[Overlap]:
    out = []
    for qa, qb in zip(a.queries, b.queries, strict=True):
        top_a = [h.candidate for h in qa.retrieved[:k]]
        top_b = [h.candidate for h in qb.retrieved[:k]]
        out.append(
            Overlap(
                query=qa.query,
                k=k,
                shared=tuple(sorted(set(top_a) & set(top_b))),
                only_first=tuple(c for c in top_a if c not in top_b),
                only_second=tuple(c for c in top_b if c not in top_a),
            )
        )
    return out


class PerformanceRecord(Record):
    """Environment-bound measurements of one execution. Not part of any result identity."""

    experiment: Digest
    environment: tuple[tuple[str, str], ...]
    timings: tuple[tuple[str, float], ...]  # name -> seconds (or bytes for *_bytes)


def environment() -> tuple[tuple[str, str], ...]:
    info = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": f"{sys.platform}-{platform.machine()}",
    }
    for package in ("memoria", "pydantic", "numpy", "onnxruntime", "tokenizers", "faiss-cpu"):
        try:
            info[package] = package_version(package)
        except PackageNotFoundError:
            info[package] = "not installed"
    return tuple(sorted(info.items()))


def run_semantic_experiment(
    spec: SemanticExperimentSpec, store: ArtifactStore, resolver: EmbedderResolver
) -> tuple[SemanticExperiment, PerformanceRecord]:
    """Run the experiment; store and return its deterministic record and its timings."""
    benchmark = store.get_record(SemanticBenchmark, spec.benchmark)
    dataset = store.get_record(Dataset, benchmark.dataset)
    embedders = [resolver.embedder(r.embedder) for r in spec.representations]
    state, log, candidate_of = _memory(dataset, store)
    timings: list[tuple[str, float]] = []
    results = []
    for representation, embedder in zip(spec.representations, embedders, strict=True):
        tag = representation.embedder.name
        start = time.perf_counter()
        index = SemanticIndex.build(state, embedder, store, representation.index)
        timings.append((f"{tag}:index_build_s", time.perf_counter() - start))
        latencies = []
        for q in benchmark.queries:
            start = time.perf_counter()
            index.search(q.text, spec.ks[-1], embedder)
            latencies.append(time.perf_counter() - start)
        if latencies:
            timings.append((f"{tag}:query_median_s", statistics.median(latencies)))
        timings.append((f"{tag}:vector_bytes", float(len(store.get(index.manifest.vectors)))))
        results.append(_evaluate(index, embedder, benchmark, candidate_of, spec, representation))
    paired, overlaps = [], []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            paired += [_paired(k, results[i], results[j], spec.confidence) for k in spec.ks]
            overlaps += _overlaps(spec.ks[-1], results[i], results[j])
    store.put_record(spec)
    record = SemanticExperiment(
        spec=spec.digest,
        benchmark=benchmark.digest,
        dataset=dataset.digest,
        log=log,
        state=state.digest,
        results=tuple(results),
        paired=tuple(paired),
        overlaps=tuple(overlaps),
    )
    store.put_record(record)
    performance = PerformanceRecord(
        experiment=record.digest,
        environment=environment(),
        timings=tuple((name, significant(value)) for name, value in timings),
    )
    store.put_record(performance)
    return record, performance


class SemanticReproduction(Record):
    original: Digest
    rerun: Digest
    outcome: Agreement
    details: tuple[str, ...]  # what differed, per representation


def similarity_tolerance(tolerance: Tolerance, dimensions: int) -> float:
    """How far a cosine may move when both vectors move within ``tolerance``.

    For unit vectors |cos(q, a) - cos(q', a')| <= ||q - q'|| + ||a - a'||, and each
    distance is at most sqrt(dimensions) * max_abs.
    """
    return 2 * math.sqrt(dimensions) * tolerance.max_abs


def reproduce_semantic(
    experiment: str, store: ArtifactStore, resolver: EmbedderResolver
) -> SemanticReproduction:
    """Re-run a stored experiment. Identical if the record reproduces byte for byte;
    equivalent if every difference is within the embedders' declared tolerance (vectors
    within tolerance, similarities within the implied bound, and retrieved sets equal
    except for swaps among near-ties); different otherwise."""
    original = store.get_record(SemanticExperiment, experiment)
    spec = store.get_record(SemanticExperimentSpec, original.spec)
    rerun, _ = run_semantic_experiment(spec, store, resolver)
    if rerun == original:
        return _reproduction(original, rerun, Agreement.IDENTICAL, ())
    details = []
    outcome = Agreement.EQUIVALENT
    for rep, a, b in zip(spec.representations, original.results, rerun.results, strict=True):
        if a == b:
            continue
        embedder = resolver.embedder(rep.embedder)
        vectors = agreement(
            SemanticIndex.load(store, a.index, embedder).vectors,
            SemanticIndex.load(store, b.index, embedder).vectors,
            rep.embedder.tolerance,
        )
        tolerance = rep.embedder.tolerance
        bound = similarity_tolerance(tolerance, rep.embedder.dimensions) if tolerance else 0.0
        within = vectors is not Agreement.DIFFERENT and _results_within(a, b, bound)
        details.append(
            f"{rep.embedder.name}: vectors {vectors.value}; results within {bound:.3g}: {within}"
        )
        if not within:
            outcome = Agreement.DIFFERENT
    if not details:  # the records differ, but no representation's results explain it
        outcome = Agreement.DIFFERENT
        details.append("records differ outside the per-representation results")
    return _reproduction(original, rerun, outcome, tuple(details))


def _results_within(a: RepresentationResult, b: RepresentationResult, bound: float) -> bool:
    for qa, qb in zip(a.queries, b.queries, strict=True):
        sim_a = {h.candidate: h.similarity for h in qa.retrieved}
        sim_b = {h.candidate: h.similarity for h in qb.retrieved}
        for c in sim_a.keys() & sim_b.keys():
            if abs(sim_a[c] - sim_b[c]) > bound:
                return False
        swapped = sim_a.keys() ^ sim_b.keys()
        cutoff = min(min(sim_a.values(), default=0.0), min(sim_b.values(), default=0.0))
        if any(abs((sim_a | sim_b)[c] - cutoff) > 2 * bound for c in swapped):
            return False
    return True


def _reproduction(
    a: SemanticExperiment, b: SemanticExperiment, outcome: Agreement, details: tuple[str, ...]
) -> SemanticReproduction:
    return SemanticReproduction(original=a.digest, rerun=b.digest, outcome=outcome, details=details)


# --- exact vs approximate ----------------------------------------------------------------------


class AnnQuery(Record):
    query: str
    recall: float  # |approx top-k ∩ exact top-k| / |exact top-k|
    top1_agrees: bool
    displacement: float | None  # mean |rank change| of items in both lists
    lost: tuple[Digest, ...]  # exact top-k items the approximate index missed


class AnnComparison(Record):
    """What an approximate index lost relative to the exact index, on the same vectors."""

    exact: Digest
    approximate: Digest
    k: int
    queries: tuple[AnnQuery, ...]
    recall: Proportion  # pooled over queries
    top1: Proportion


def compare_indexes(
    exact: SemanticIndex,
    approximate: SemanticIndex,
    queries: Sequence[str],
    k: int,
    embedder: Embedder,
    confidence: float = 0.95,
) -> AnnComparison:
    if exact.manifest.index.kind != "exact" or approximate.manifest.index.kind == "exact":
        raise ValueError("compare an exact index with an approximate one")
    same = ("source", "entries", "vectors", "embedder")
    if any(getattr(exact.manifest, f) != getattr(approximate.manifest, f) for f in same):
        raise ValueError("indexes must hold the same vectors of the same state")
    rows = []
    kept = total = top1 = answered = 0
    for text in queries:
        truth = [n.version for n in exact.search(text, k, embedder)]
        found = [n.version for n in approximate.search(text, k, embedder)]
        common = [d for d in truth if d in found]
        kept += len(common)
        total += len(truth)
        agrees = bool(truth) and bool(found) and truth[0] == found[0]
        top1 += agrees
        answered += bool(truth)
        rows.append(
            AnnQuery(
                query=text,
                recall=quantize(len(common) / len(truth)) if truth else 1.0,
                top1_agrees=agrees,
                displacement=quantize(
                    math.fsum(abs(truth.index(d) - found.index(d)) for d in common) / len(common)
                )
                if common
                else None,
                lost=tuple(d for d in truth if d not in found),
            )
        )
    return AnnComparison(
        exact=exact.digest,
        approximate=approximate.digest,
        k=k,
        queries=tuple(rows),
        recall=Proportion.of(kept, total, confidence),
        top1=Proportion.of(top1, answered, confidence),
    )


# --- performance characterisation ------------------------------------------------------------


def synthetic_state(size: int, seed: int = 0) -> MemoryState:
    """A deterministic state of ``size`` short factual sentences, for scaling runs only."""
    people = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun"]
    things = ["lives in", "works at", "studies", "prefers", "owns", "visited", "plays", "reads"]
    objects = ["Berlin", "Acme", "physics", "tea", "a bicycle", "Tokyo", "chess", "poetry"]
    memories = []
    for i in range(size):
        text = (
            f"{people[(i + seed) % 10]} {things[(i // 10) % 8]} "
            f"{objects[(i // 80) % 8]} (note {i})."
        )
        memories.append(
            MemoryVersion(
                memory_id=f"m{i:06d}",
                version=1,
                operation=Operation.CREATE,
                content=text,
                valid_from=_EMPTY_AT,
                recorded_at=_EMPTY_AT,
                derived_from=(content_hash(text),),
            )
        )
    return MemoryState(valid_at=_EMPTY_AT, known_at=_EMPTY_AT, memories=tuple(memories))


SCALING_QUERIES = [("lives in", "Berlin"), ("works at", "Acme"), ("reads", "poetry")]


class ScalingPoint(Record):
    size: int
    embed_s: float
    exact_build_s: float
    exact_query_median_s: float
    ann_build_s: float | None
    ann_query_median_s: float | None
    ann_recall_at_10: float | None
    vector_bytes: int
    ann_bytes: int | None
    python_peak_bytes: int  # tracemalloc peak during embedding (excludes native runtimes)


def profile_scaling(
    embedder: Embedder,
    sizes: Sequence[int],
    store: ArtifactStore,
    ann: RepresentationSpec | None = None,
    queries: int = 20,
) -> list[ScalingPoint]:
    """Measure embedding, build and query costs over corpus sizes. Environment-bound."""
    points = []
    for size in sizes:
        state = synthetic_state(size)
        tracemalloc.start()
        start = time.perf_counter()
        embed(embedder, [m.content or "" for m in state.memories])
        embed_s = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        start = time.perf_counter()
        exact = SemanticIndex.build(state, embedder, store)
        exact_build = time.perf_counter() - start
        texts = [f"Who {v} {o}? ({i})" for i, (v, o) in enumerate(SCALING_QUERIES * queries)]
        texts = texts[:queries]
        exact_q = _median_latency(exact, texts, embedder)
        ann_build = ann_q = ann_recall = None
        ann_bytes = None
        if ann is not None:
            start = time.perf_counter()
            approx = SemanticIndex.build(state, embedder, store, ann.index)
            ann_build = time.perf_counter() - start
            ann_q = _median_latency(approx, texts, embedder)
            ann_recall = compare_indexes(exact, approx, texts, 10, embedder).recall.estimate
            ann_bytes = len(store.get(approx.manifest.ann or approx.manifest.vectors))
        points.append(
            ScalingPoint(
                size=size,
                embed_s=significant(embed_s),
                exact_build_s=significant(exact_build),
                exact_query_median_s=significant(exact_q),
                ann_build_s=significant(ann_build) if ann_build is not None else None,
                ann_query_median_s=significant(ann_q) if ann_q is not None else None,
                ann_recall_at_10=ann_recall,
                vector_bytes=len(store.get(exact.manifest.vectors)),
                ann_bytes=ann_bytes,
                python_peak_bytes=peak,
            )
        )
    return points


def _median_latency(index: SemanticIndex, texts: Sequence[str], embedder: Embedder) -> float:
    latencies = []
    for text in texts:
        start = time.perf_counter()
        index.search(text, 10, embedder)
        latencies.append(time.perf_counter() - start)
    return statistics.median(latencies)


def max_rss_bytes() -> int:
    """Peak resident set size of this process so far (whole process, including runtimes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


# --- the Phase 5 study ---------------------------------------------------------------------------


def phase5_spec(benchmark: str) -> SemanticExperimentSpec:
    """The reference (hashed character n-grams) versus the pinned MiniLM encoder, both
    with the exact index, on the semantic diagnostic."""
    from memoria.embeddings import HashedNgramEmbedder
    from memoria.neural import minilm_spec

    return SemanticExperimentSpec(
        name="phase5-lexical-vs-neural",
        benchmark=benchmark,
        representations=(
            RepresentationSpec(embedder=HashedNgramEmbedder().spec),
            RepresentationSpec(embedder=minilm_spec()),
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m memoria.semantic_eval <store dir>``: run and reproduce the Phase 5 study.

    Needs the 'neural' extra and the pinned model files (see memoria.neural.fetch_model).
    """
    from memoria.experiments import DEFAULT_REGISTRY
    from memoria.neural import MINILM_CARD, neural_embedders
    from memoria.scenarios import semantic_diagnostic

    args = list(sys.argv[1:] if argv is None else argv)
    store = ArtifactStore(args[0] if args else "var/artifacts")
    registry = DEFAULT_REGISTRY.extend(embedders=neural_embedders())
    dataset, benchmark = semantic_diagnostic()
    store.put_record(dataset)
    store.put_record(benchmark)
    card = store.put_record(MINILM_CARD)  # licence and provenance metadata, not identity
    record, performance = run_semantic_experiment(phase5_spec(benchmark.digest), store, registry)
    reproduction = reproduce_semantic(record.digest, store, registry)
    store.put_record(reproduction)
    print(f"spec        {record.spec}")
    print(f"experiment  {record.digest}")
    print(f"performance {performance.digest}")
    print(f"model card  {card}")
    print(f"reproduced  {reproduction.outcome.value}")


if __name__ == "__main__":
    main()
