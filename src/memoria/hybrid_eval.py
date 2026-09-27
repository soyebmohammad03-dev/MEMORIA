"""Hybrid retrieval experiments: ablations, leave-one-out, counterfactuals, adversarial cases.

:func:`run_hybrid_experiment` runs every :class:`~memoria.hybrid.RetrievalPolicy` of a
:class:`HybridExperimentSpec` on every query of a :class:`HybridBenchmark`, stores every
:class:`~memoria.hybrid.HybridTrace`, and writes one content-addressed
:class:`HybridExperiment` with:

- per-policy metrics (recall/precision@k with Wilson intervals, MRR, nDCG@k,
  contradiction exposure, redundant-result rate, temporal-validity rate);
- per-query diagnostics (candidate counts per generator, union size, exclusions by
  reason, signal distributions, final ranking, target ranks, exposure, diversity moves);
- paired comparisons along the ablation ladder and for leave-one-signal-out, reusing
  :mod:`memoria.statistics` (Newcombe method 10, exact McNemar, Holm within a family,
  an underpowered flag);
- leave-one-out deltas (metric changes, affected queries and memories, rank changes);
- counterfactual stability (paraphrase, reordering and irrelevant wording, versus entity,
  attribute, time, negation and numeric changes);
- adversarial-case measurements.

Timings go to a separate :class:`~memoria.semantic_eval.PerformanceRecord`: they describe
the machine, not the result. Metrics measure retrieval against the benchmark's designed
relevance labels under a stated policy; they are not claims about truth.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
import tracemalloc
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.artifacts import ArtifactStore
from memoria.core import (
    Dataset,
    Digest,
    Record,
    RepresentationSpec,
    UTCDatetime,
    content_hash,
    quantize,
)
from memoria.embeddings import Embedder
from memoria.formation import EpisodicPolicy, form
from memoria.hybrid import (
    DEFAULT_EXCLUDE,
    RRF_K,
    STATE_EXCLUDE,
    Corpus,
    DiversitySpec,
    Engine,
    Exclusion,
    GeneratorSpec,
    HybridQuery,
    HybridTrace,
    MemoryKind,
    RetrievalPolicy,
    SignalConfig,
    TemporalStatus,
    equal_weights,
    generators,
    policy,
    tie_key,
)
from memoria.semantic import SemanticIndex
from memoria.semantic_eval import EmbedderResolver, PerformanceRecord, environment, max_rss_bytes
from memoria.statistics import (
    Proportion,
    holm,
    mcnemar_exact,
    min_achievable_p,
    newcombe_paired,
    significant,
)
from memoria.store import MemoryLog

# --- benchmark -------------------------------------------------------------------------------


class Role(StrEnum):
    """A candidate's designed role for one query."""

    TARGET = "target"  # asserts the answer the evidence supports at (valid_at, known_at)
    RELATED = "related"  # same subject and attribute, not the answer (old or disputed value)
    TRAP = "trap"  # designed to attract a signal while being irrelevant

    @property
    def grade(self) -> int:
        return {Role.TARGET: 2, Role.RELATED: 1, Role.TRAP: 0}[self]


class Perturbation(StrEnum):
    PARAPHRASE = "paraphrase"
    REORDER = "reorder"
    IRRELEVANT = "irrelevant_wording"
    ENTITY = "entity"
    ATTRIBUTE = "attribute"
    TIME = "time"
    NEGATION = "negation"
    NUMERIC = "numeric"

    @property
    def invariant(self) -> bool:
        """Whether the perturbation preserves the information need (retrieval should not
        change); the others may legitimately change it."""
        return self in (Perturbation.PARAPHRASE, Perturbation.REORDER, Perturbation.IRRELEVANT)


class Judgement(Record):
    candidate: str  # the candidate experience's source id
    role: Role
    cases: tuple[str, ...] = ()  # adversarial cases this candidate is the focus of


class BenchmarkQuery(Record):
    id: str = Field(min_length=1)
    text: str
    valid_at: UTCDatetime
    known_at: UTCDatetime
    key: str | None = None
    kinds: tuple[MemoryKind, ...] = ()
    judgements: tuple[Judgement, ...] = ()
    cases: tuple[str, ...] = ()
    base: str | None = None  # the query this one perturbs
    perturbation: Perturbation | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        ids = [j.candidate for j in self.judgements]
        if len(set(ids)) != len(ids):
            raise ValueError("each candidate is judged at most once per query")
        if (self.base is None) != (self.perturbation is None):
            raise ValueError("a variant names both its base query and its perturbation")
        return self

    def hybrid(self, limit: int) -> HybridQuery:
        return HybridQuery(
            text=self.text,
            valid_at=self.valid_at,
            known_at=self.known_at,
            limit=limit,
            key=self.key,
            kinds=self.kinds,
        )


class CandidateFact(Record):
    """The fact a candidate expresses (``key=value``), used to measure redundancy."""

    candidate: str
    fact: str


class HybridBenchmark(Record):
    """Queries with designed roles over the experiences of a dataset."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dataset: Digest
    queries: tuple[BenchmarkQuery, ...]
    facts: tuple[CandidateFact, ...]
    cases: tuple[str, ...]  # the adversarial cases the benchmark declares

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        ids = [q.id for q in self.queries]
        if len(set(ids)) != len(ids):
            raise ValueError("query ids must be unique")
        primary = {q.id for q in self.queries if q.base is None}
        for q in self.queries:
            if q.base is not None and q.base not in primary:
                raise ValueError(f"variant {q.id} perturbs unknown primary query {q.base}")
            undeclared = (set(q.cases) | {c for j in q.judgements for c in j.cases}) - set(
                self.cases
            )
            if undeclared:
                raise ValueError(f"query {q.id} uses undeclared cases {sorted(undeclared)}")
        return self


# --- spec ------------------------------------------------------------------------------------


class HybridExperimentSpec(Record):
    """Everything that determines a hybrid-retrieval experiment. Its digest is its ID."""

    name: str = Field(min_length=1)
    benchmark: Digest
    representation: RepresentationSpec | None  # embedder and index for semantic stages
    policies: tuple[RetrievalPolicy, ...] = Field(min_length=1)
    reference: str  # the policy others are compared with (rank changes, leave-one-out)
    ladder: tuple[str, ...] = ()  # ablation order: each step adds one component
    # (removed component, reference policy, policy without the component)
    leave_one_out: tuple[tuple[str, str, str], ...] = ()
    ks: tuple[int, ...] = (1, 3, 5)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        names = [p.name for p in self.policies]
        if len(set(names)) != len(names):
            raise ValueError("policy names must be unique")
        known = set(names)
        used = {self.reference, *self.ladder, *(n for t in self.leave_one_out for n in t[1:])}
        if used - known:
            raise ValueError(f"unknown policies referenced: {sorted(used - known)}")
        if not self.ks or list(self.ks) != sorted(set(self.ks)) or self.ks[0] < 1:
            raise ValueError("ks must be distinct positive integers in increasing order")
        return self


# --- results ---------------------------------------------------------------------------------


class SignalDistribution(Record):
    signal: str
    available: int
    missing: int
    minimum: float | None
    mean: float | None
    maximum: float | None


class QueryDiagnostics(Record):
    """Everything needed to inspect one retrieval without re-running it."""

    query: str
    text_hash: Digest
    trace: Digest
    proposed: tuple[tuple[str, int], ...]  # generator -> proposals
    generator_status: tuple[tuple[str, str], ...]
    union: int
    excluded: tuple[tuple[str, int], ...]  # exclusion reason -> count
    survivors: int
    signals: tuple[SignalDistribution, ...]
    ranking: tuple[str, ...]  # candidate ids of the selected results, in order
    target_ranks: tuple[int, ...]  # final ranks of targets among survivors
    disputed_top: bool  # a survivor asserts another value for the top result's key
    exposed: bool  # ... and such a memory is visible (selected or surfaced)
    diversity_moves: int  # selected positions that differ from pure relevance order
    rank_changes: tuple[tuple[str, int], ...]  # candidate -> reference rank - this rank


class KMetric(Record):
    k: int
    recall: Proportion  # targets retrieved / targets, pooled over queries
    fact_recall: Proportion  # distinct target facts retrieved / distinct target facts
    precision: Proportion
    ndcg: float | None  # mean over queries with a target


class PolicyResult(Record):
    policy: Digest
    name: str
    queries: tuple[QueryDiagnostics, ...]  # every benchmark query, in benchmark order
    at: tuple[KMetric, ...]  # primary queries
    mrr: float | None  # primary queries with a target; full ranking
    exposure: Proportion  # of primary queries whose top result has other-value evidence
    redundancy: Proportion  # selected results repeating a higher-ranked result's fact
    validity: Proportion  # selected results valid at the query's valid time


class PairedTest(Record):
    """One paired comparison of an indicator (statistics as in evaluation.PairedMeasure)."""

    family: str
    measure: str  # "recall@k" over (query, target) pairs, or "success@1" over queries
    first: str  # policy names; difference is second - first
    second: str
    pairs: int
    both: int
    first_only: int
    second_only: int
    neither: int
    difference: float | None
    low: float | None
    high: float | None
    p_value: float | None
    p_adjusted: float | None
    min_achievable_p: float | None
    underpowered: bool


class Delta(Record):
    """What removing one component from a reference policy changed."""

    removed: str
    reference: str
    policy: str
    recall: tuple[tuple[int, float | None], ...]  # k -> recall(removed) - recall(reference)
    fact_recall: tuple[tuple[int, float | None], ...]
    mrr: float | None
    ndcg: tuple[tuple[int, float | None], ...]
    exposure: float | None
    redundancy: float | None
    validity: float | None
    affected_queries: tuple[str, ...]  # top-k set or order changed
    entered: tuple[str, ...]  # candidates newly in some top-k (query:candidate)
    left: tuple[str, ...]
    mean_displacement: float | None  # mean |rank change| of candidates in both top-k


class Counterfactual(Record):
    policy: str
    base: str
    variant: str
    perturbation: Perturbation
    invariant: bool
    overlap: float  # |top-k(base) ∩ top-k(variant)| / k
    top1_same: bool
    displacement: float | None  # mean |rank change| of shared top-k candidates
    variant_recall: Proportion | None  # the variant's own targets in its top-k, if judged


class CaseResult(Record):
    case: str
    policy: str
    queries: tuple[str, ...]
    focus_visible: Proportion  # (query, focus) pairs in the top-k or surfaced
    target_above_focus: Proportion  # queries with a non-target focus: best target ranks higher
    redundancy: Proportion  # selected results of these queries repeating a fact
    focus_ranks: tuple[tuple[str, int | None], ...]  # query:candidate -> final rank


class HybridExperiment(Record):
    spec: Digest
    benchmark: Digest
    dataset: Digest
    log: Digest
    corpus: Digest
    index: Digest | None
    embedder: Digest | None
    results: tuple[PolicyResult, ...]
    paired: tuple[PairedTest, ...]
    deltas: tuple[Delta, ...]
    counterfactuals: tuple[Counterfactual, ...]
    cases: tuple[CaseResult, ...]


# --- running ---------------------------------------------------------------------------------


def build_log(dataset: Dataset) -> MemoryLog:
    """Ingest the dataset verbatim (one episodic memory per experience)."""
    log = MemoryLog(":memory:")
    for step in dataset.steps:
        form(log, step.experience, EpisodicPolicy(), recorded_at=step.recorded_at)
    return log


def _single_known_at(benchmark: HybridBenchmark) -> UTCDatetime:
    known = {q.known_at for q in benchmark.queries}
    if len(known) != 1:
        raise ValueError("the benchmark's queries must share one known_at (one corpus)")
    return known.pop()


def _ndcg(grades: Sequence[int], ideal: Sequence[int], k: int) -> float | None:
    """nDCG@k with exponential gain (2^g - 1) and log2 discount (Jarvelin & Kekalainen)."""

    def dcg(gs: Sequence[int]) -> float:
        return math.fsum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))

    best = dcg(sorted(ideal, reverse=True))
    return quantize(dcg(grades) / best) if best else None


def _mean(xs: Sequence[float]) -> float | None:
    return quantize(math.fsum(xs) / len(xs)) if xs else None


def _diagnose(
    q: BenchmarkQuery,
    trace: HybridTrace,
    candidate_of: dict[str, str],
    reference: HybridTrace | None,
) -> QueryDiagnostics:
    roles = {j.candidate: j.role for j in q.judgements}
    proposed = tuple((g.generator, len(g.proposals)) for g in trace.generators)
    excluded = Counter(c.excluded.value for c in trace.candidates if c.excluded is not None)
    distributions = []
    for j, cfg in enumerate(trace.policy.signals):
        values = [r.signals[j].normalized for r in trace.ranking]
        present = [v for v in values if v is not None]
        distributions.append(
            SignalDistribution(
                signal=cfg.name,
                available=len(present),
                missing=len(values) - len(present),
                minimum=min(present) if present else None,
                mean=_mean(present),
                maximum=max(present) if present else None,
            )
        )
    selected = [candidate_of[v] for v in trace.selected]
    top = trace.ranking[0] if trace.ranking else None
    survivors = {r.version for r in trace.ranking}
    disputes = (
        [v for v in trace.candidate(top.version).conflicts_with if v in survivors] if top else []
    )
    visible = set(trace.selected) | {v for r in trace.ranking for v in r.counter_evidence}
    limit = trace.query.limit
    pure = sorted(trace.ranking, key=lambda r: (-r.relevance, *tie_key(r)[2:]))
    moves = sum(
        1
        for a, b in zip(pure[:limit], trace.ranking[:limit], strict=True)
        if a.version != b.version
    )
    changes: list[tuple[str, int]] = []
    if reference is not None:
        ref_rank = {candidate_of[r.version]: r.rank for r in reference.ranking}
        for r in trace.ranking[:limit]:
            c = candidate_of[r.version]
            if c in ref_rank and ref_rank[c] != r.rank:
                changes.append((c, ref_rank[c] - r.rank))
    return QueryDiagnostics(
        query=q.id,
        text_hash=content_hash(q.text),
        trace=trace.digest,
        proposed=proposed,
        generator_status=tuple((g.generator, g.status) for g in trace.generators),
        union=len(trace.candidates),
        excluded=tuple(sorted(excluded.items())),
        survivors=len(trace.ranking),
        signals=tuple(distributions),
        ranking=tuple(selected),
        target_ranks=tuple(
            r.rank for r in trace.ranking if roles.get(candidate_of[r.version]) is Role.TARGET
        ),
        disputed_top=bool(disputes),
        exposed=bool(disputes) and any(v in visible for v in disputes),
        diversity_moves=moves,
        rank_changes=tuple(sorted(changes)),
    )


def _metrics(
    name: str,
    pol: RetrievalPolicy,
    queries: Sequence[BenchmarkQuery],
    traces: dict[str, HybridTrace],
    diagnostics: tuple[QueryDiagnostics, ...],
    facts: dict[str, str],
    candidate_of: dict[str, str],
    spec: HybridExperimentSpec,
) -> PolicyResult:
    primary = [q for q in queries if q.base is None]
    diag = {d.query: d for d in diagnostics}
    at = []
    for k in spec.ks:
        got = wanted = returned = facts_got = facts_wanted = 0
        ndcgs = []
        for q in primary:
            roles = {j.candidate: j.role for j in q.judgements}
            targets = {c for c, r in roles.items() if r is Role.TARGET}
            top = [candidate_of[v] for v in traces[q.id].selected[:k]]
            got += sum(1 for c in top if c in targets)
            wanted += len(targets)
            returned += len(top)
            target_facts = {facts.get(c, c) for c in targets}
            facts_got += len(target_facts & {facts.get(c, c) for c in top if c in targets})
            facts_wanted += len(target_facts)
            if targets:
                value = _ndcg(
                    [roles[c].grade if c in roles else 0 for c in top],
                    [r.grade for r in roles.values()],
                    k,
                )
                if value is not None:
                    ndcgs.append(value)
        at.append(
            KMetric(
                k=k,
                recall=Proportion.of(got, wanted, spec.confidence),
                fact_recall=Proportion.of(facts_got, facts_wanted, spec.confidence),
                precision=Proportion.of(got, returned, spec.confidence),
                ndcg=_mean(ndcgs),
            )
        )
    reciprocal = []
    disputed = exposed = 0
    repeats = selected = valid = 0
    for q in primary:
        d = diag[q.id]
        if any(j.role is Role.TARGET for j in q.judgements):
            reciprocal.append(1 / min(d.target_ranks) if d.target_ranks else 0.0)
        disputed += d.disputed_top
        exposed += d.exposed
        trace = traces[q.id]
        seen: set[str] = set()
        for v in trace.selected:
            fact = facts.get(candidate_of[v])
            repeats += fact is not None and fact in seen
            if fact is not None:
                seen.add(fact)
            valid += trace.candidate(v).temporal is TemporalStatus.VALID
        selected += len(trace.selected)
    return PolicyResult(
        policy=pol.digest,
        name=name,
        queries=diagnostics,
        at=tuple(at),
        mrr=_mean(reciprocal),
        exposure=Proportion.of(exposed, disputed, spec.confidence),
        redundancy=Proportion.of(repeats, selected, spec.confidence),
        validity=Proportion.of(valid, selected, spec.confidence),
    )


def _paired(
    family: str,
    measure: str,
    first: str,
    second: str,
    outcomes: Sequence[tuple[bool, bool]],
    spec: HybridExperimentSpec,
) -> PairedTest:
    cells = {"both": 0, "first_only": 0, "second_only": 0, "neither": 0}
    for a, b in outcomes:
        cells[
            {(True, True): "both", (True, False): "first_only", (False, True): "second_only"}.get(
                (a, b), "neither"
            )
        ] += 1
    n = len(outcomes)
    diff = newcombe_paired(
        cells["both"], cells["second_only"], cells["first_only"], cells["neither"], spec.confidence
    )
    discordant = cells["first_only"] + cells["second_only"]
    floor = min_achievable_p(discordant) if n else None
    return PairedTest(
        family=family,
        measure=measure,
        first=first,
        second=second,
        pairs=n,
        **cells,
        difference=quantize(diff[0]) if diff else None,
        low=quantize(diff[1]) if diff else None,
        high=quantize(diff[2]) if diff else None,
        p_value=significant(mcnemar_exact(cells["first_only"], cells["second_only"]))
        if n
        else None,
        p_adjusted=None,
        min_achievable_p=significant(floor) if floor is not None else None,
        underpowered=floor is None or floor > spec.alpha,
    )


def _indicators(
    queries: Sequence[BenchmarkQuery],
    traces: dict[str, HybridTrace],
    candidate_of: dict[str, str],
    facts: dict[str, str],
    k: int,
) -> dict[str, list[bool]]:
    """Paired units per measure, in benchmark order: recall@k per (query, target),
    fact_recall@k per (query, distinct target fact), success@1 per query with a target."""
    out: dict[str, list[bool]] = {f"recall@{k}": [], f"fact_recall@{k}": [], "success@1": []}
    for q in queries:
        if q.base is not None:
            continue
        targets = [j.candidate for j in q.judgements if j.role is Role.TARGET]
        top = [candidate_of[v] for v in traces[q.id].selected[:k]]
        out[f"recall@{k}"] += [t in top for t in targets]
        found = {facts.get(c, c) for c in top if c in targets}
        out[f"fact_recall@{k}"] += [f in found for f in sorted({facts.get(t, t) for t in targets})]
        if targets:
            out["success@1"].append(bool(top) and top[0] in targets)
    return out


def _family(
    name: str,
    pairs: Sequence[tuple[str, str]],
    indicators: dict[str, dict[str, list[bool]]],
    spec: HybridExperimentSpec,
) -> list[PairedTest]:
    """Paired tests per measure; Holm adjustment within (family, measure)."""
    adjusted = []
    for measure in next(iter(indicators.values()), {}):
        tests = [
            _paired(
                name,
                measure,
                first,
                second,
                list(zip(indicators[first][measure], indicators[second][measure], strict=True)),
                spec,
            )
            for first, second in pairs
        ]
        values = holm([t.p_value if t.p_value is not None else 1.0 for t in tests])
        adjusted += [
            t.model_copy(update={"p_adjusted": significant(v) if t.p_value is not None else None})
            for t, v in zip(tests, values, strict=True)
        ]
    return adjusted


def _delta(
    removed: str,
    reference: str,
    name: str,
    ref: PolicyResult,
    other: PolicyResult,
    queries: Sequence[BenchmarkQuery],
    k: int,
) -> Delta:
    def diff(a: float | None, b: float | None) -> float | None:
        return quantize(b - a) if a is not None and b is not None else None

    affected, entered, left, moves = [], [], [], []
    for qa, qb in zip(ref.queries, other.queries, strict=True):
        a, b = list(qa.ranking[:k]), list(qb.ranking[:k])
        if a != b:
            affected.append(qa.query)
        entered += [f"{qa.query}:{c}" for c in b if c not in a]
        left += [f"{qa.query}:{c}" for c in a if c not in b]
        moves += [abs(a.index(c) - b.index(c)) for c in a if c in b]
    primary = {q.id for q in queries if q.base is None}
    return Delta(
        removed=removed,
        reference=reference,
        policy=name,
        recall=tuple(
            (x.k, diff(x.recall.estimate, y.recall.estimate))
            for x, y in zip(ref.at, other.at, strict=True)
        ),
        fact_recall=tuple(
            (x.k, diff(x.fact_recall.estimate, y.fact_recall.estimate))
            for x, y in zip(ref.at, other.at, strict=True)
        ),
        mrr=diff(ref.mrr, other.mrr),
        ndcg=tuple((x.k, diff(x.ndcg, y.ndcg)) for x, y in zip(ref.at, other.at, strict=True)),
        exposure=diff(ref.exposure.estimate, other.exposure.estimate),
        redundancy=diff(ref.redundancy.estimate, other.redundancy.estimate),
        validity=diff(ref.validity.estimate, other.validity.estimate),
        affected_queries=tuple(q for q in affected if q in primary),
        entered=tuple(e for e in entered if e.split(":", 1)[0] in primary),
        left=tuple(e for e in left if e.split(":", 1)[0] in primary),
        mean_displacement=_mean(moves),
    )


def _counterfactuals(
    name: str,
    queries: Sequence[BenchmarkQuery],
    result: PolicyResult,
    traces: dict[str, HybridTrace],
    candidate_of: dict[str, str],
    k: int,
    confidence: float,
) -> list[Counterfactual]:
    diag = {d.query: d for d in result.queries}
    out = []
    for q in queries:
        if q.base is None or q.perturbation is None:
            continue
        a, b = list(diag[q.base].ranking[:k]), list(diag[q.id].ranking[:k])
        targets = [j.candidate for j in q.judgements if j.role is Role.TARGET]
        top = [candidate_of[v] for v in traces[q.id].selected[:k]]
        shared = [c for c in a if c in b]
        out.append(
            Counterfactual(
                policy=name,
                base=q.base,
                variant=q.id,
                perturbation=q.perturbation,
                invariant=q.perturbation.invariant,
                overlap=quantize(len(shared) / k),
                top1_same=bool(a) and bool(b) and a[0] == b[0],
                displacement=_mean([abs(a.index(c) - b.index(c)) for c in shared]),
                variant_recall=Proportion.of(
                    sum(t in top for t in targets), len(targets), confidence
                )
                if targets
                else None,
            )
        )
    return out


def _cases(
    name: str,
    benchmark: HybridBenchmark,
    traces: dict[str, HybridTrace],
    candidate_of: dict[str, str],
    facts: dict[str, str],
    k: int,
    confidence: float,
) -> list[CaseResult]:
    out = []
    for case in benchmark.cases:
        tagged = [
            q
            for q in benchmark.queries
            if q.base is None and (case in q.cases or any(case in j.cases for j in q.judgements))
        ]
        visible = pairs = above = with_trap = repeats = selected = 0
        ranks: list[tuple[str, int | None]] = []
        for q in tagged:
            trace = traces[q.id]
            rank_of = {candidate_of[r.version]: r.rank for r in trace.ranking}
            shown = {candidate_of[v] for v in trace.selected[:k]} | {
                candidate_of[v] for r in trace.ranking[:k] for v in r.counter_evidence
            }
            focus = [j for j in q.judgements if case in j.cases]
            for j in focus:
                pairs += 1
                visible += j.candidate in shown
                ranks.append((f"{q.id}:{j.candidate}", rank_of.get(j.candidate)))
            traps = [j.candidate for j in focus if j.role is not Role.TARGET]
            targets = [j.candidate for j in q.judgements if j.role is Role.TARGET]
            if traps and targets:
                with_trap += 1
                best_target = min((rank_of[t] for t in targets if t in rank_of), default=None)
                best_trap = min((rank_of[t] for t in traps if t in rank_of), default=None)
                above += best_target is not None and (best_trap is None or best_target < best_trap)
            seen: set[str] = set()
            for v in trace.selected[:k]:
                fact = facts.get(candidate_of[v])
                repeats += fact is not None and fact in seen
                if fact is not None:
                    seen.add(fact)
            selected += len(trace.selected[:k])
        out.append(
            CaseResult(
                case=case,
                policy=name,
                queries=tuple(q.id for q in tagged),
                focus_visible=Proportion.of(visible, pairs, confidence),
                target_above_focus=Proportion.of(above, with_trap, confidence),
                redundancy=Proportion.of(repeats, selected, confidence),
                focus_ranks=tuple(ranks),
            )
        )
    return out


PROFILED = ("lexical", "semantic", "lexical+semantic", "full-diversity", "full")


def run_hybrid_experiment(
    spec: HybridExperimentSpec, store: ArtifactStore, resolver: EmbedderResolver
) -> tuple[HybridExperiment, PerformanceRecord]:
    """Run every policy on every query; store traces, the experiment and its timings.

    Components are resolved before any retrieval: an unavailable embedder is an error,
    never a substitute (I36). A policy whose generator fails raises unless the policy
    says to record the failure.
    """
    benchmark = store.get_record(HybridBenchmark, spec.benchmark)
    dataset = store.get_record(Dataset, benchmark.dataset)
    embedder: Embedder | None = None
    if spec.representation is not None:
        embedder = resolver.embedder(spec.representation.embedder)
    known_at = _single_known_at(benchmark)
    depth = spec.ks[-1]
    with build_log(dataset) as log:
        log_digest = store.put(log.export(), kind="MemoryLogExport")
        corpus = Corpus.from_log(log, known_at)
        candidate_of = {
            e.digest: log.experience(e.version.derived_from[0]).source  # type: ignore[union-attr]
            for e in corpus.entries
        }
    index = None
    if spec.representation is not None and embedder is not None:
        index = SemanticIndex.build(corpus.present, embedder, store, spec.representation.index)
    engine = Engine(corpus, embedder, index)
    facts = {f.candidate: f.fact for f in benchmark.facts}
    by_name = {p.name: p for p in spec.policies}

    timings: dict[str, list[float]] = {}
    all_traces: dict[str, dict[str, HybridTrace]] = {}
    for pol in spec.policies:
        traces = {}
        for q in benchmark.queries:
            stages: dict[str, float] = {}
            start = time.perf_counter()
            trace = engine.retrieve(pol, q.hybrid(depth), stages)
            stages["total"] = time.perf_counter() - start
            for stage, seconds in stages.items():
                timings.setdefault(f"{pol.name}:{stage}", []).append(seconds)
            timings.setdefault(f"{pol.name}:candidates", []).append(float(len(trace.candidates)))
            store.put_record(trace)
            traces[q.id] = trace
        all_traces[pol.name] = traces

    # Memory: traced Python allocations of one pass (warm caches) per compared policy.
    for name in PROFILED:
        if name in by_name:
            tracemalloc.start()
            for q in benchmark.queries:
                engine.retrieve(by_name[name], q.hybrid(depth))
            timings[f"{name}:python_peak_bytes"] = [float(tracemalloc.get_traced_memory()[1])]
            tracemalloc.stop()
    timings["process:max_rss_bytes"] = [float(max_rss_bytes())]

    reference = all_traces[spec.reference]
    results: dict[str, PolicyResult] = {}
    for pol in spec.policies:
        traces = all_traces[pol.name]
        diagnostics = tuple(
            _diagnose(q, traces[q.id], candidate_of, reference[q.id]) for q in benchmark.queries
        )
        results[pol.name] = _metrics(
            pol.name, pol, benchmark.queries, traces, diagnostics, facts, candidate_of, spec
        )

    indicators = {
        name: _indicators(benchmark.queries, all_traces[name], candidate_of, facts, depth)
        for name in by_name
    }
    paired = _family("ladder", list(pairwise(spec.ladder)), indicators, spec)
    for ref in dict.fromkeys(r for _, r, _ in spec.leave_one_out):
        pairs = [(r, name) for _, r, name in spec.leave_one_out if r == ref]
        paired += _family(f"leave_one_out:{ref}", pairs, indicators, spec)
    deltas = [
        _delta(removed, ref, name, results[ref], results[name], benchmark.queries, depth)
        for removed, ref, name in spec.leave_one_out
    ]
    counterfactuals = [
        c
        for pol in spec.policies
        for c in _counterfactuals(
            pol.name,
            benchmark.queries,
            results[pol.name],
            all_traces[pol.name],
            candidate_of,
            depth,
            spec.confidence,
        )
    ]
    cases = [
        c
        for pol in spec.policies
        for c in _cases(
            pol.name, benchmark, all_traces[pol.name], candidate_of, facts, depth, spec.confidence
        )
    ]
    store.put_record(spec)
    record = HybridExperiment(
        spec=spec.digest,
        benchmark=benchmark.digest,
        dataset=dataset.digest,
        log=log_digest,
        corpus=store.put_record(corpus.identity),
        index=index.digest if index is not None else None,
        embedder=embedder.spec.digest if embedder is not None else None,
        results=tuple(results[p.name] for p in spec.policies),
        paired=tuple(paired),
        deltas=tuple(deltas),
        counterfactuals=tuple(counterfactuals),
        cases=tuple(cases),
    )
    store.put_record(record)
    performance = PerformanceRecord(
        experiment=record.digest,
        environment=environment(),
        timings=tuple(
            (f"{name}_median", significant(statistics.median(values)))
            for name, values in sorted(timings.items())
        ),
    )
    store.put_record(performance)
    return record, performance


class HybridReproduction(Record):
    original: Digest
    rerun: Digest
    identical: bool
    differing_policies: tuple[str, ...]


def reproduce_hybrid(
    experiment: str, store: ArtifactStore, resolver: EmbedderResolver
) -> HybridReproduction:
    """Re-run a stored experiment and compare it byte for byte (same-machine guarantee;
    across platforms, neural similarities may move within the embedder's tolerance)."""
    original = store.get_record(HybridExperiment, experiment)
    spec = store.get_record(HybridExperimentSpec, original.spec)
    rerun, _ = run_hybrid_experiment(spec, store, resolver)
    differing = tuple(
        a.name for a, b in zip(original.results, rerun.results, strict=True) if a != b
    )
    result = HybridReproduction(
        original=original.digest,
        rerun=rerun.digest,
        identical=original == rerun,
        differing_policies=differing,
    )
    store.put_record(result)
    return result


def verify_experiment(experiment: HybridExperiment, store: ArtifactStore) -> int:
    """Re-read every trace the experiment cites (re-hashed, re-validated on load) and check
    that each diagnostic's ranking follows from its trace. Returns the traces checked."""
    spec = store.get_record(HybridExperimentSpec, experiment.spec)
    policies = {p.digest: p for p in spec.policies}
    checked = 0
    for result in experiment.results:
        for d in result.queries:
            trace = store.get_record(HybridTrace, d.trace)  # validators re-derive everything
            if trace.policy != policies[result.policy] or trace.corpus != experiment.corpus:
                raise ValueError(f"{result.name}/{d.query}: trace is from another policy/corpus")
            if len(d.ranking) != len(trace.selected) or d.survivors != len(trace.ranking):
                raise ValueError(f"{result.name}/{d.query}: diagnostics do not match the trace")
            checked += 1
    return checked


# --- the Phase 6 study -----------------------------------------------------------------------

# Declared metadata assumptions of the benchmark's source classes, not measurements.
PHASE6_PRIORS = {"clinic": 0.9, "email": 0.7, "chat": 0.7, "forum": 0.3}
RECENCY: dict[str, str | float] = {
    "axis": "recorded",
    "family": "exponential",
    "half_life_days": 30.0,
}
LIMIT = 20  # per generator; the benchmark corpus has fewer memories per subject


def _params(half_life: float = 30.0) -> dict[str, dict[str, str | float]]:
    return {
        "recency": {**RECENCY, "half_life_days": half_life},
        "source": {f"prior:{c}": p for c, p in sorted(PHASE6_PRIORS.items())},
    }


LADDER = (
    ("semantic", ("semantic",)),
    ("lexical", ("lexical",)),
    ("lexical+semantic", ("lexical", "semantic")),
    ("+temporal", ("lexical", "semantic")),
    ("+recency", ("lexical", "recency", "semantic")),
    ("+source", ("lexical", "recency", "semantic", "source")),
    ("+provenance", ("lexical", "provenance", "recency", "semantic", "source")),
    ("+attribute", ("attribute", "lexical", "provenance", "recency", "semantic", "source")),
    ("+contradiction", ("attribute", "lexical", "provenance", "recency", "semantic", "source")),
    ("full", ("attribute", "lexical", "provenance", "recency", "semantic", "source")),
)
FULL_SIGNALS = LADDER[-1][1]
DIVERSITY = DiversitySpec(beta=1.0, similarity="embedding")  # lambda = 0.5: symmetric MMR


def _generators_for(step: int) -> tuple[GeneratorSpec, ...]:
    """Single-signal steps use only their own generator; metadata joins with attribute."""
    all_gens = generators(LIMIT)
    if step == 0:
        return tuple(g for g in all_gens if g.name == "semantic")
    if step == 1:
        return tuple(g for g in all_gens if g.name == "lexical")
    if step < 7:
        return tuple(g for g in all_gens if g.name != "metadata")
    return all_gens


def _make(
    name: str,
    signals: Sequence[str] = FULL_SIGNALS,
    gens: tuple[GeneratorSpec, ...] | None = None,
    *,
    configs: tuple[SignalConfig, ...] | None = None,
    exclude: Sequence[Exclusion] = DEFAULT_EXCLUDE,
    contradiction: Literal["neutral", "penalize", "surface", "paired"] = "surface",
    diversity: DiversitySpec | None = DIVERSITY,
    scoring: Literal["weighted", "rrf"] = "weighted",
    half_life: float = 30.0,
) -> RetrievalPolicy:
    made = policy(
        name,
        configs or equal_weights(signals, _params(half_life)),
        exclude=exclude,
        scoring=scoring,
        rrf_k=RRF_K if scoring == "rrf" else None,
        contradiction=contradiction,
        diversity=diversity,
    )
    return made.model_copy(update={"generators": gens or generators(LIMIT)})


def _without(signal_name: str) -> tuple[str, ...]:
    return tuple(s for s in FULL_SIGNALS if s != signal_name)


def phase6_policies() -> tuple[list[RetrievalPolicy], list[tuple[str, str, str]]]:
    """The ablation ladder, leave-one-out variants, and sensitivity variants."""
    out = [
        _make(
            name,
            signals,
            _generators_for(step),
            exclude=DEFAULT_EXCLUDE if step >= 3 else STATE_EXCLUDE,
            contradiction="surface" if step >= 8 else "neutral",
            diversity=DIVERSITY if step >= 9 else None,
        )
        for step, (name, signals) in enumerate(LADDER)
    ]
    loo: list[tuple[str, str, RetrievalPolicy]] = []
    for ref, prefix, div in (("full", "full", DIVERSITY), ("+contradiction", "nodiv", None)):
        loo += [
            (s, ref, _make(f"{prefix}-{s}", _without(s), diversity=div))
            for s in ("semantic", "lexical", "recency", "source", "provenance", "attribute")
        ]
        loo += [
            ("temporal", ref, _make(f"{prefix}-temporal", exclude=STATE_EXCLUDE, diversity=div)),
            (
                "contradiction",
                ref,
                _make(f"{prefix}-contradiction", contradiction="neutral", diversity=div),
            ),
        ]
    loo.append(("diversity", "full", _make("full-diversity", diversity=None)))
    out += [p for _, _, p in loo]
    out += [
        _make(
            "contradiction:penalize",
            configs=equal_weights([*FULL_SIGNALS, "contradiction"], _params()),
            contradiction="penalize",
        ),
        _make("contradiction:paired", contradiction="paired"),
        _make("fusion:rrf", scoring="rrf"),
        _make("temporal:soft", (*FULL_SIGNALS, "temporal"), exclude=STATE_EXCLUDE),
        _make("recency:7d", half_life=7.0),
        _make("recency:90d", half_life=90.0),
        _make("recency:365d", half_life=365.0),
        *(
            _make(
                f"diversity:beta={beta:g}",
                diversity=DiversitySpec(beta=beta, similarity="embedding"),
            )
            for beta in (0.1, 0.25, 0.5, 2.0)
        ),
        _make("diversity:jaccard", diversity=DiversitySpec(beta=1.0, similarity="token-jaccard")),
    ]
    for strategy in ("rank-v1", "zscore-v1"):
        configs = tuple(
            s.model_copy(update={"normalization": strategy}) if s.name == "lexical" else s
            for s in equal_weights(FULL_SIGNALS, _params())
        )
        out.append(_make(f"lexical:{strategy}", configs=configs))
    return out, [(removed, ref, p.name) for removed, ref, p in loo]


def phase6_spec(benchmark: str, representation: RepresentationSpec) -> HybridExperimentSpec:
    policies, loo = phase6_policies()
    return HybridExperimentSpec(
        name="phase6-hybrid-retrieval",
        benchmark=benchmark,
        representation=representation,
        policies=tuple(policies),
        reference="full",
        ladder=tuple(name for name, _ in LADDER),
        leave_one_out=tuple(loo),
    )


# --- report ----------------------------------------------------------------------------------


def _p(x: Proportion) -> str:
    if x.estimate is None:
        return f"{x.numerator}/{x.denominator} (undefined)"
    return f"{x.numerator}/{x.denominator} = {x.estimate:.3f} [{x.low:.3f}, {x.high:.3f}]"


def _f(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def _table(title: str, header: Sequence[str]) -> list[str]:
    return ["", f"## {title}", "", "| " + " | ".join(header) + " |", "|---" * len(header) + "|"]


SHOWN = ("semantic", "lexical", "lexical+semantic", "+contradiction", "full")


def report(experiment: HybridExperiment, performance: PerformanceRecord | None = None) -> str:
    """A Markdown summary of the stored experiment (every number read from the record)."""
    lines = _table(
        "Policies",
        ["policy", "R@1", "R@3", "R@5", "fact R@5", "P@5", "MRR", "nDCG@5", "exposure",
         "redundancy", "validity"],
    )  # fmt: skip
    for r in experiment.results:
        at = {m.k: m for m in r.at}
        lines.append(
            f"| {r.name} | {_f(at[1].recall.estimate)} | {_f(at[3].recall.estimate)} | "
            f"{_p(at[5].recall)} | {_p(at[5].fact_recall)} | {_f(at[5].precision.estimate)} | "
            f"{_f(r.mrr)} | "
            f"{_f(at[5].ndcg)} | {_p(r.exposure)} | {_p(r.redundancy)} | {_p(r.validity)} |"
        )
    lines += _table(
        "Paired tests (difference = second - first)",
        ["family", "measure", "first", "second", "n", "gained/lost", "difference [CI]", "p",
         "p (Holm)", "underpowered"],
    )  # fmt: skip
    for t in experiment.paired:
        ci = f"{_f(t.difference, True)} [{_f(t.low, True)}, {_f(t.high, True)}]"
        p_raw = "—" if t.p_value is None else f"{t.p_value:.3g}"
        p_adj = "—" if t.p_adjusted is None else f"{t.p_adjusted:.3g}"
        lines.append(
            f"| {t.family} | {t.measure} | {t.first} | {t.second} | {t.pairs} | "
            f"{t.second_only}/{t.first_only} | {ci} | {p_raw} | {p_adj} | {t.underpowered} |"
        )
    lines += _table(
        "Leave one out (policy without the component, minus its reference)",
        ["reference", "removed", "ΔR@5", "Δfact R@5", "ΔMRR", "ΔnDCG@5", "Δexposure", "Δredundancy",
         "Δvalidity", "affected queries", "entered", "left", "displacement"],
    )  # fmt: skip
    for d in experiment.deltas:
        lines.append(
            f"| {d.reference} | {d.removed} | {_f(dict(d.recall)[5], True)} | "
            f"{_f(dict(d.fact_recall)[5], True)} | {_f(d.mrr, True)} | "
            f"{_f(dict(d.ndcg)[5], True)} | {_f(d.exposure, True)} | {_f(d.redundancy, True)} | "
            f"{_f(d.validity, True)} | {len(d.affected_queries)} | {len(d.entered)} | "
            f"{len(d.left)} | {_f(d.mean_displacement)} |"
        )
    lines += _table(
        "Counterfactuals (per perturbation)",
        ["policy", "perturbation", "invariant", "n", "overlap@5", "top-1 same",
         "variant recall@5"],
    )  # fmt: skip
    groups: dict[tuple[str, Perturbation], list[Counterfactual]] = {}
    for cf in experiment.counterfactuals:
        groups.setdefault((cf.policy, cf.perturbation), []).append(cf)
    for (pol, pert), cfs in groups.items():
        if pol not in SHOWN:
            continue
        judged = [cf.variant_recall for cf in cfs if cf.variant_recall is not None]
        got = sum(x.numerator for x in judged)
        wanted = sum(x.denominator for x in judged)
        lines.append(
            f"| {pol} | {pert.value} | {pert.invariant} | {len(cfs)} | "
            f"{_f(_mean([cf.overlap for cf in cfs]))} | "
            f"{sum(cf.top1_same for cf in cfs)}/{len(cfs)} | "
            f"{f'{got}/{wanted}' if wanted else '—'} |"
        )
    lines += _table(
        "Adversarial cases",
        ["case", "policy", "queries", "focus visible", "target above focus", "redundancy"],
    )
    for case in experiment.cases:
        if case.policy in SHOWN:
            lines.append(
                f"| {case.case} | {case.policy} | {len(case.queries)} | "
                f"{_p(case.focus_visible)} | {_p(case.target_above_focus)} | "
                f"{_p(case.redundancy)} |"
            )
    if performance is not None:
        stages = ("generate", "filter", "features", "score", "rerank", "explain", "total")
        lines += _table(
            "Performance (median per query, seconds; environment-bound)",
            ["policy", *stages, "candidates", "Python peak (KiB)"],
        )
        timing = dict(performance.timings)
        for pol in PROFILED:
            cells = [timing.get(f"{pol}:{stage}_median") for stage in stages]
            peak = timing.get(f"{pol}:python_peak_bytes_median")
            lines.append(
                f"| {pol} | "
                + " | ".join("—" if v is None else f"{v:.2e}" for v in cells)
                + f" | {timing.get(f'{pol}:candidates_median', 0):.0f} | "
                + ("—" if peak is None else f"{peak / 1024:.0f}")
                + " |"
            )
    return "\n".join(lines).lstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m memoria.hybrid_eval <store dir> [hashed|minilm]``: run, reproduce and
    report the Phase 6 study. ``minilm`` needs the 'neural' extra and the pinned model."""
    from memoria.embeddings import HashedNgramEmbedder
    from memoria.experiments import DEFAULT_REGISTRY
    from memoria.scenarios import hybrid_benchmark

    args = list(sys.argv[1:] if argv is None else argv)
    store = ArtifactStore(args[0] if args else "var/artifacts")
    which = args[1] if len(args) > 1 else "hashed"
    registry = DEFAULT_REGISTRY
    if which == "minilm":
        from memoria.neural import minilm_spec, neural_embedders

        registry = registry.extend(embedders=neural_embedders())
        representation = RepresentationSpec(embedder=minilm_spec())
    elif which == "hashed":
        representation = RepresentationSpec(embedder=HashedNgramEmbedder().spec)
    else:
        raise SystemExit(f"unknown representation {which!r} (hashed | minilm)")
    dataset, benchmark = hybrid_benchmark()
    store.put_record(dataset)
    store.put_record(benchmark)
    spec = phase6_spec(benchmark.digest, representation)
    record, performance = run_hybrid_experiment(spec, store, registry)
    reproduction = reproduce_hybrid(record.digest, store, registry)
    traces = verify_experiment(record, store)
    print(f"benchmark    {benchmark.digest}")
    print(f"spec         {record.spec}")
    print(f"experiment   {record.digest}")
    print(f"performance  {performance.digest}")
    print(f"reproduced   {'identical' if reproduction.identical else 'different'}")
    print(f"traces       {traces} verified")
    print()
    print(report(record, performance))


if __name__ == "__main__":
    main()
