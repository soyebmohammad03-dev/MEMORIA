"""The consolidation laboratory: strategies, stability-plasticity, loss and adversaries.

Research question: *how does memory consolidation strategy affect long-horizon retrieval
quality, provenance fidelity, information loss, contradiction handling, computational
cost and downstream memory reliability?*

Every condition is an ordinary run (manifest schema v3: a hybrid retrieval policy and a
consolidation policy, both stored artifacts), evaluated by Phase 4 (:func:`evaluate`)
and compared with its baseline by :func:`evaluation.compare` (paired Newcombe interval
= risk difference as effect size, exact McNemar, Holm within families, underpowered
flags). Consolidation-specific measurements are computed from the run's stored
hierarchies against the benchmark's ground-truth labels, which the consolidation never
sees: merge precision/recall against the true (key, value, period) of every report,
abstraction and inference errors against the epistemic answer, structured feature loss,
lineage completeness, contradiction preservation and staleness. Timings and sizes are a
separate :class:`~memoria.semantic_eval.PerformanceRecord`.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Sequence
from datetime import datetime

from pydantic import Field, model_validator

from memoria.artifacts import ArtifactStore
from memoria.comparison import normalize
from memoria.consolidation import (
    GUARDS,
    ConsolidationPolicy,
    Hierarchy,
    ImportanceComponent,
    ImportanceSpec,
    lineage,
    replay,
)
from memoria.core import (
    Dataset,
    DerivedMemory,
    Digest,
    EpistemicStatus,
    ExpectationStatus,
    InterventionSpec,
    Level,
    Record,
    RepresentationSpec,
    RunManifest,
    RunOutcomes,
    RunRecord,
    UTCDatetime,
    quantize,
)
from memoria.evaluation import DEFAULT_SPEC, Evaluation, RunComparison, compare, evaluate
from memoria.experiments import DEFAULT_REGISTRY, Registry, execute
from memoria.hybrid import (
    DEFAULT_EXCLUDE,
    Corpus,
    Exclusion,
    HybridTrace,
    RetrievalPolicy,
    equal_weights,
    policy,
)
from memoria.retrieval import EXTRACTIVE
from memoria.semantic_eval import PerformanceRecord, environment
from memoria.statistics import (
    Proportion,
    holm,
    mcnemar_exact,
    min_achievable_p,
    newcombe_paired,
    significant,
)
from memoria.store import MemoryLog
from memoria.taxonomy import Claim, claims

# --- benchmark -------------------------------------------------------------------------------


class FactLabel(Record):
    """Ground truth for one report: which key, which value, which true period of the world.

    Two reports are *equivalent* (a correct merge) iff they share (key, value, period).
    """

    candidate: str  # the experience's source id
    key: str
    value: str  # normalised tokens joined by spaces
    period: int = Field(ge=0)


class ConsolidationBenchmark(Record):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dataset: Digest
    labels: tuple[FactLabel, ...]
    coupled: tuple[tuple[str, str], ...]  # key pairs that change together by design

    @model_validator(mode="after")
    def _check(self) -> ConsolidationBenchmark:
        ids = [x.candidate for x in self.labels]
        if len(set(ids)) != len(ids):
            raise ValueError("each report is labelled once")
        return self


# --- the lab ---------------------------------------------------------------------------------


class Condition(Record):
    """One experimental condition: a retrieval policy and an optional consolidation policy
    (both stored artifacts), plus optional interventions (adversarial perturbations)."""

    name: str = Field(min_length=1)
    retrieval: Digest
    consolidation: Digest | None = None
    interventions: tuple[InterventionSpec, ...] = ()
    worlds: tuple[str, ...] = ()  # benchmark names it runs on; empty = every benchmark


class LabSpec(Record):
    """Everything that determines a consolidation study. Its digest is its ID."""

    name: str = Field(min_length=1)
    benchmarks: tuple[Digest, ...] = Field(min_length=1)
    representation: RepresentationSpec | None
    conditions: tuple[Condition, ...] = Field(min_length=1)
    comparisons: tuple[tuple[str, str, str], ...] = ()  # (family, baseline, treatment)

    @model_validator(mode="after")
    def _check(self) -> LabSpec:
        names = [c.name for c in self.conditions]
        if len(set(names)) != len(names):
            raise ValueError("condition names must be unique")
        used = {n for _, a, b in self.comparisons for n in (a, b)}
        if used - set(names):
            raise ValueError(f"unknown conditions: {sorted(used - set(names))}")
        return self


class Growth(Record):
    at: UTCDatetime
    l1: int
    derived: tuple[tuple[str, int], ...]  # level -> count
    blocked: int


class ConsolidationMetrics(Record):
    """Consolidation measurements of one run, pooled over its checkpoints."""

    run: Digest
    hierarchies: int
    compression: float | None  # held L1 versions per L2 memory at the last checkpoint
    merge_precision: Proportion  # merged pairs that are truly equivalent
    merge_recall: Proportion  # truly equivalent promoted pairs that were merged
    blocked_merges: int  # candidate merges refused by a guard (reported, not applied)
    abstraction_errors: Proportion  # abstracted statements the evidence does not support
    inference_errors: Proportion  # inferred co-changes between keys not coupled by design
    feature_loss: tuple[tuple[str, Proportion], ...]  # per feature kind: lost / input
    unsupported: Proportion  # derived memories with a feature no input states
    lineage_complete: Proportion  # derived memories whose lineage resolves to the log
    max_depth: int  # longest downward lineage (memories + versions)
    contradictions_kept: Proportion  # contested periods whose values stay separate at L2
    derived_answers: Proportion  # probes answered from a derived memory
    stale_answers: Proportion  # ... from a derived memory invalidated at query time
    growth: tuple[Growth, ...]
    replayed: bool  # every hierarchy re-consolidated byte-identically


class LabRun(Record):
    benchmark: str
    condition: str
    manifest: Digest
    run: Digest
    evaluation: Digest
    correct: Proportion  # known.correct
    in_state: Proportion  # known.expected_in_state (retention)
    contested: Proportion  # contested.answered
    unknown: Proportion  # unknown.abstained
    consolidation: ConsolidationMetrics | None


class LabComparison(Record):
    benchmark: str
    family: str
    baseline: str
    treatment: str
    comparison: Digest  # the stored RunComparison
    # (difference, low, high, Holm-adjusted p within world x family, underpowered):
    # known.correct from evaluation.compare; retention computed with the same statistics.
    known_correct: tuple[float | None, float | None, float | None, float | None, bool]
    in_state: tuple[float | None, float | None, float | None, float | None, bool]


class LabStudy(Record):
    spec: Digest
    runs: tuple[LabRun, ...]
    comparisons: tuple[LabComparison, ...]


# --- measurements ----------------------------------------------------------------------------


def _epistemic_values(
    claims_: Sequence[Claim], key: str, valid_at: datetime, known_at: datetime
) -> set[str]:
    """The values an ideal system could hold for ``key`` (Phase 3 standard): latest
    report by occurrence up to valid_at; a later ``correct`` replaces it retroactively up
    to the next report; ties are contested; ``forget`` leaves nothing."""
    reports = sorted(
        (c for c in claims_ if c.key == key and c.recorded_at <= known_at),
        key=lambda c: c.occurred_at,
    )
    held = [c for c in reports if c.occurred_at <= valid_at]
    if not held:
        return set()
    latest = max(c.occurred_at for c in held)
    current = [c for c in held if c.occurred_at == latest]
    horizon = min(
        (c.occurred_at for c in reports if c.occurred_at > latest and c.verb != "correct"),
        default=None,
    )
    corrections = [
        c for c in reports if c.verb == "correct" and c.occurred_at > latest
        and (horizon is None or c.occurred_at < horizon)
    ]  # fmt: skip
    if corrections:
        last = max(c.occurred_at for c in corrections)
        current = [c for c in corrections if c.occurred_at == last]
    if any(c.verb == "forget" for c in current):
        return set()
    return {" ".join(normalize(c.value or "")) for c in current}


def measure_consolidation(
    record: RunRecord, store: ArtifactStore, benchmark: ConsolidationBenchmark
) -> ConsolidationMetrics | None:
    if not record.hierarchies:
        return None
    manifest = store.get_record(RunManifest, record.manifest)
    assert manifest.consolidation_policy is not None
    cpolicy = store.get_record(ConsolidationPolicy, manifest.consolidation_policy)
    dataset = store.get_record(Dataset, record.dataset)
    authored = store.get_record(Dataset, benchmark.dataset)
    evidence = list(claims(dataset, authored).values())
    labels = {x.candidate: x for x in benchmark.labels}
    coupled = {tuple(sorted(p)) for p in benchmark.coupled}
    hierarchies = [store.get_record(Hierarchy, h) for h in record.hierarchies]
    registry = DEFAULT_REGISTRY
    embedder = (
        registry.embedder(manifest.representation.embedder)
        if manifest.representation is not None and cpolicy.uses_vectors
        else None
    )
    tp = merged = equivalent = blocked = 0
    abstracted = abstracted_bad = inferred = inferred_bad = 0
    kinds: dict[str, list[int]] = {}
    derived = unsupported = complete = contested_total = contested_kept = 0
    depth = 0
    growth = []
    replayed = True
    with MemoryLog.load(store.get(record.log)) as log:
        source_of = {v.digest: log.experience(v.derived_from[0]).source for v in log.versions()}  # type: ignore[union-attr]
        versions = {v.digest for v in log.versions()}
        for h in hierarchies:
            corpus = Corpus.from_log(log, h.at)
            replayed &= replay(h, corpus, cpolicy, embedder)
            held = [e for e in corpus.entries if e.fate == "held"]
            promoted = [e.digest for e in held if e.digest not in dict(h.not_promoted)]

            def label(d: str) -> tuple[str, str, int] | None:
                x = labels.get(source_of[d])
                return (x.key, x.value, x.period) if x else None

            group_of = {
                v: m.memory_id for m in h.memories if m.level is Level.L2 for v in m.versions
            }
            for i, a in enumerate(promoted):
                for b in promoted[i + 1 :]:
                    same = label(a) is not None and label(a) == label(b)
                    together = a in group_of and group_of.get(a) == group_of.get(b)
                    equivalent += same
                    merged += together
                    tp += same and together
            blocked += h.blocked_merges
            for m in h.memories:
                derived += 1
                r = h.losses[[x.memory_id for x in h.memories].index(m.memory_id)]
                for kind in ("claim", "entity", "number", "temporal", "negation", "token"):
                    kept, total = r.kind_counts(kind)
                    kinds.setdefault(kind, [0, 0])
                    kinds[kind][0] += total - kept
                    kinds[kind][1] += total
                unsupported += bool(r.unsupported)
                chain = lineage(h, m.memory_id)
                depth = max(depth, len(chain))
                complete += all(v in versions for v in m.versions) and (
                    r.experience_coverage.numerator == r.experience_coverage.denominator
                )
                if m.status is EpistemicStatus.INFERRED:
                    inferred += 1
                    keys = tuple(
                        sorted(
                            k
                            for k in {
                                labels[source_of[v]].key
                                for v in m.versions
                                if source_of[v] in labels
                            }
                        )
                    )
                    inferred_bad += not any(tuple(sorted(p)) in coupled for p in _pairs(keys))
                elif m.status is EpistemicStatus.ABSTRACTED:
                    for key, value, when in _statements(m, h):
                        abstracted += 1
                        truth = _epistemic_values(evidence, key, when, h.at)
                        abstracted_bad += value not in truth
            for key in sorted({c.key for c in evidence}):
                visible = [c for c in evidence if c.key == key and c.recorded_at <= h.at]
                instants: dict[datetime, set[str]] = {}
                for c in visible:
                    if c.value is not None:
                        instants.setdefault(c.occurred_at, set()).add(" ".join(normalize(c.value)))
                for t, values in instants.items():
                    if len(values) < 2:
                        continue
                    contested_total += 1
                    separate = {
                        m.value
                        for m in h.memories
                        if m.level is Level.L2 and m.key == key and m.valid_from <= t
                    }
                    contested_kept += values <= separate
            growth.append(
                Growth(
                    at=h.at,
                    l1=len(held),
                    derived=tuple(
                        (lv.value, sum(1 for m in h.memories if m.level is lv))
                        for lv in (Level.L2, Level.L3, Level.L4)
                    ),
                    blocked=h.blocked_merges,
                )
            )
    outcomes_derived = stale = probes = 0
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    for o in outcomes.probes:
        probes += 1
        trace = store.get_record(HybridTrace, o.trace)
        if trace.selected:
            top = trace.candidate(trace.selected[0])
            outcomes_derived += top.level is not Level.L1
            stale += top.stale
    last = hierarchies[-1]
    l2 = sum(1 for m in last.memories if m.level is Level.L2)
    return ConsolidationMetrics(
        run=record.digest,
        hierarchies=len(hierarchies),
        compression=quantize(growth[-1].l1 / l2) if l2 else None,
        merge_precision=Proportion.of(tp, merged, 0.95),
        merge_recall=Proportion.of(tp, equivalent, 0.95),
        blocked_merges=blocked,
        abstraction_errors=Proportion.of(abstracted_bad, abstracted, 0.95),
        inference_errors=Proportion.of(inferred_bad, inferred, 0.95),
        feature_loss=tuple((k, Proportion.of(v[0], v[1], 0.95)) for k, v in sorted(kinds.items())),
        unsupported=Proportion.of(unsupported, derived, 0.95),
        lineage_complete=Proportion.of(complete, derived, 0.95),
        max_depth=depth,
        contradictions_kept=Proportion.of(contested_kept, contested_total, 0.95),
        derived_answers=Proportion.of(outcomes_derived, probes, 0.95),
        stale_answers=Proportion.of(stale, probes, 0.95),
        growth=tuple(growth),
        replayed=replayed,
    )


def _pairs(keys: Sequence[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :]]


def _statements(m: DerivedMemory, h: Hierarchy) -> list[tuple[str, str, datetime]]:
    """(key, value, as-of time) statements an abstracted memory makes: an entity profile
    states each listed value at consolidation time; a timeline states each of its L2
    parents' values from that parent's start (read from the parent, not the rendering)."""
    out = []
    if m.operation == "abstract_entity":
        ent, _, body = m.content.partition(": ")
        for part in body.split("; "):
            attr, _, values = part.partition(" = ")
            for v in values.removesuffix(" (contested)").split(" | "):
                out.append((f"{ent}.{attr}", v, m.recorded_at))
    elif m.operation == "abstract_timeline":
        for parent in m.parents:
            p = h.memory(parent)
            if p.key is not None and p.value is not None:
                out.append((p.key, p.value, p.valid_from))
    return out


# --- the study ----------------------------------------------------------------------------------


def _retrieval(name: str, levels: tuple[Level, ...], *, support: bool = False,
               stale: bool = False) -> RetrievalPolicy:  # fmt: skip
    """The lab's retrieval policies: equal-weight attribute + lexical + semantic (exact
    scan, so derived memories are reachable), temporal filter, surface exposure;
    ``support`` adds the hierarchy-aware coverage signal, ``stale`` excludes invalidated
    derived memories."""
    names = ["attribute", "lexical", "semantic", *(["support"] if support else [])]
    exclude = (*DEFAULT_EXCLUDE, *([Exclusion.STALE] if stale else []))
    made = policy(name, equal_weights(names), exclude=exclude, contradiction="surface")
    gens = tuple(
        g.model_copy(update={"params": (("index", "scan"),)}) if g.name == "semantic" else g
        for g in made.generators
    )
    return RetrievalPolicy.model_validate(
        made.model_dump() | {"generators": [g.model_dump() for g in gens], "levels": levels}
    )


ALL_LEVELS = (Level.L1, Level.L2, Level.L3, Level.L4)
RETRIEVAL = {
    "raw": _retrieval("raw", (Level.L1,)),
    "consolidated": _retrieval("consolidated", (Level.L2, Level.L3, Level.L4)),
    "mixed": _retrieval("mixed", ALL_LEVELS),
    "hierarchy-aware": _retrieval("hierarchy-aware", ALL_LEVELS, support=True, stale=True),
}
EVERY = 30.0  # consolidation frequency of the strategy family (varied in the S-P family)
_IMPORTANCE_NAMES = ("contradiction", "persistence", "provenance", "recency", "repetition",
                     "source")  # fmt: skip
PRIORS = (("chat", 0.7), ("clinic", 0.9), ("email", 0.7), ("forum", 0.3))


def importance(threshold: float = 0.5, drop: str | None = None) -> ImportanceSpec:
    names = [n for n in _IMPORTANCE_NAMES if n != drop]
    bounded = {"contradiction", "provenance", "recency", "source"}
    return ImportanceSpec(
        components=tuple(
            ImportanceComponent(
                name=n,  # type: ignore[arg-type]
                weight=1 / len(names),
                normalization="bounded-v1" if n in bounded else "minmax-v1",
            )
            for n in names
        ),
        threshold=threshold,
        half_life_days=90.0,
        priors=PRIORS,
    )


def consolidation(name: str, **params: object) -> ConsolidationPolicy:
    return ConsolidationPolicy.model_validate(
        {"name": name, "version": "1", "every_days": EVERY} | params
    )


STRATEGIES = {
    "B:lossless": consolidation("lossless", regime="exact"),
    "C:dedup": consolidation("claim-dedup", regime="claim"),
    "C:canonical": consolidation("canonical-dedup", regime="canonical"),
    "D:abstractive": consolidation("abstractive", regime="semantic", threshold=0.5,
                                   abstractions=("entity",)),
    "E:hierarchical": consolidation("hierarchical", regime="temporal",
                                    abstractions=("co_change", "entity", "timeline")),
    "F:recency": consolidation("recency", regime="claim", promote="recent", window_days=90.0),
    "G:importance": consolidation("importance", regime="claim", promote="important",
                                  importance=importance()),
    "H:hybrid": consolidation("hybrid", regime="temporal", promote="important",
                              importance=importance(),
                              abstractions=("co_change", "entity", "timeline")),
    "noop": consolidation("noop", regime="none"),
}  # fmt: skip

ABLATIONS = {
    "semantic:0.3": consolidation("semantic-0.3", regime="semantic", threshold=0.3),
    "semantic:0.5": consolidation("semantic-0.5", regime="semantic", threshold=0.5),
    "semantic:0.7": consolidation("semantic-0.7", regime="semantic", threshold=0.7),
    "semantic:0.3-unguarded": consolidation("semantic-0.3-unguarded", regime="semantic",
                                            threshold=0.3, guards=()),
    "temporal": consolidation("temporal", regime="temporal"),
    "hierarchical-no-abstraction": consolidation("hierarchical-l2", regime="temporal"),
    **{
        f"importance-{n}": consolidation(f"importance-{n}", regime="claim", promote="important",
                                         importance=importance(drop=n))
        for n in _IMPORTANCE_NAMES
    },
}  # fmt: skip

PLASTICITY = {
    **{f"every:{d:g}": consolidation(f"every-{d:g}", regime="temporal", every_days=d,
                                     abstractions=("entity",)) for d in (7.0, 30.0, 90.0)},
    **{f"window:{w:g}": consolidation(f"window-{w:g}", regime="claim", promote="recent",
                                      window_days=w) for w in (30.0, 90.0, 180.0)},
    **{f"importance:{t:g}": consolidation(f"importance-{t:g}", regime="claim",
                                          promote="important", importance=importance(t))
       for t in (0.3, 0.5, 0.7)},
    "depth:L2": consolidation("depth-l2", regime="temporal"),
    "depth:L3": consolidation("depth-l3", regime="temporal", abstractions=("entity",)),
    "depth:L4": consolidation("depth-l4", regime="temporal",
                              abstractions=("co_change", "entity", "timeline")),
    "support:4": consolidation("support-4", regime="temporal",
                               abstractions=("co_change", "entity", "timeline"), min_support=4),
    "unguarded-claims": consolidation("unguarded-claims", regime="semantic", threshold=0.3,
                                      guards=tuple(g for g in GUARDS if g != "claim_conflict")),
}  # fmt: skip


ABLATION_WORLDS = ("adversarial", "baseline", "contradiction")
PLASTICITY_WORLDS = ("baseline", "drift", "long")
RETRIEVAL_WORLDS = ("baseline", "drift", "long", "repetitive")


def _conditions(store: ArtifactStore) -> tuple[list[Condition], list[tuple[str, str, str]]]:
    ids = {name: store.put_record(p) for name, p in RETRIEVAL.items()}
    out = [Condition(name="A:none", retrieval=ids["mixed"])]
    comparisons = []
    families = (
        ("strategy", STRATEGIES, ()),
        ("ablation", ABLATIONS, ABLATION_WORLDS),
        ("plasticity", PLASTICITY, PLASTICITY_WORLDS),
    )
    for family, table, worlds in families:
        for name, cp in table.items():
            if any(c.name == name for c in out):
                continue
            out.append(Condition(name=name, retrieval=ids["mixed"],
                                 consolidation=store.put_record(cp), worlds=worlds))  # fmt: skip
            comparisons.append((family, "A:none", name))
    hier = store.put_record(STRATEGIES["E:hierarchical"])
    for mode in ("raw", "consolidated", "hierarchy-aware"):
        out.append(Condition(name=f"retrieval:{mode}", retrieval=ids[mode], consolidation=hier,
                             worlds=RETRIEVAL_WORLDS))  # fmt: skip
        if mode != "raw":
            comparisons.append(("retrieval", "retrieval:raw", f"retrieval:{mode}"))
    comparisons.append(("retrieval", "retrieval:raw", "E:hierarchical"))
    return out, comparisons


def lab_spec(store: ArtifactStore, benchmarks: Sequence[str],
             representation: RepresentationSpec) -> LabSpec:  # fmt: skip
    conditions, comparisons = _conditions(store)
    return LabSpec(
        name="phase7-consolidation-lab",
        benchmarks=tuple(benchmarks),
        representation=representation,
        conditions=tuple(conditions),
        comparisons=tuple(comparisons),
    )


def run_lab(
    spec: LabSpec, store: ArtifactStore, registry: Registry = DEFAULT_REGISTRY,
    only: Sequence[str] | None = None,
) -> tuple[LabStudy, PerformanceRecord]:  # fmt: skip
    """Run every condition on every benchmark (``only``: a subset of conditions), evaluate,
    measure consolidation, compare, and store the study and its timings."""
    conditions = [c for c in spec.conditions if only is None or c.name in only]
    runs: list[LabRun] = []
    timings: list[tuple[str, float]] = []
    for bdigest in spec.benchmarks:
        bench = store.get_record(ConsolidationBenchmark, bdigest)
        for c in conditions:
            if c.worlds and bench.name not in c.worlds:
                continue
            manifest = RunManifest(
                name=f"{bench.name}/{c.name}",
                dataset=bench.dataset,
                interventions=c.interventions,
                policy="episodic-v1",
                responder=EXTRACTIVE,
                representation=spec.representation,
                retrieval_policy=c.retrieval,
                consolidation_policy=c.consolidation,
            )
            start = time.perf_counter()
            record = execute(manifest, store, registry)
            elapsed = time.perf_counter() - start
            evaluation = evaluate(record.digest, store, DEFAULT_SPEC)
            start = time.perf_counter()
            metrics = measure_consolidation(record, store, bench)
            replay_s = time.perf_counter() - start
            size = sum(len(store.get(h)) for h in record.hierarchies)
            tag = f"{bench.name}/{c.name}"
            timings += [(f"{tag}:run_s", elapsed), (f"{tag}:measure_and_replay_s", replay_s),
                        (f"{tag}:hierarchy_bytes", float(size)),
                        (f"{tag}:log_bytes", float(len(store.get(record.log))))]  # fmt: skip
            runs.append(
                LabRun(
                    benchmark=bench.name,
                    condition=c.name,
                    manifest=manifest.digest,
                    run=record.digest,
                    evaluation=evaluation.digest,
                    correct=evaluation.measurement("known.correct").proportion,
                    in_state=evaluation.measurement("known.expected_in_state").proportion,
                    contested=evaluation.measurement("contested.answered").proportion,
                    unknown=evaluation.measurement("unknown.abstained").proportion,
                    consolidation=metrics,
                )
            )
    by_key = {(r.benchmark, r.condition): r for r in runs}
    comparisons: list[LabComparison] = []
    for bench_name in dict.fromkeys(r.benchmark for r in runs):
        for family in dict.fromkeys(f for f, _, _ in spec.comparisons):
            pairs = [(a, b) for f, a, b in spec.comparisons if f == family
                     and (bench_name, a) in by_key and (bench_name, b) in by_key]  # fmt: skip
            made = [
                compare(by_key[(bench_name, a)].evaluation, by_key[(bench_name, b)].evaluation,
                        store)
                for a, b in pairs
            ]  # fmt: skip
            evals = {
                name: store.get_record(Evaluation, by_key[(bench_name, name)].evaluation)
                for pair in pairs
                for name in pair
            }
            correct = _holm([_paired(m, "known.correct") for m in made])
            retained = _holm([_retention(evals[a], evals[b]) for a, b in pairs])
            for (a, b), m, kc, ret in zip(pairs, made, correct, retained, strict=True):
                comparisons.append(
                    LabComparison(
                        benchmark=bench_name,
                        family=family,
                        baseline=a,
                        treatment=b,
                        comparison=m.digest,
                        known_correct=kc,
                        in_state=ret,
                    )
                )
    study = LabStudy(spec=store.put_record(spec), runs=tuple(runs), comparisons=tuple(comparisons))
    store.put_record(study)
    performance = PerformanceRecord(
        experiment=study.digest,
        environment=environment(),
        timings=tuple((n, significant(v)) for n, v in timings),
    )
    store.put_record(performance)
    return study, performance


Cell = tuple[float | None, float | None, float | None, float | None, bool]


def _holm(cells: list[Cell]) -> list[Cell]:
    """Replace each cell's p-value by its Holm adjustment across the family."""
    adjusted = holm([c[3] if c[3] is not None else 1.0 for c in cells])
    return [
        (c[0], c[1], c[2], significant(q) if c[3] is not None else None, c[4])
        for c, q in zip(cells, adjusted, strict=True)
    ]


def _paired(m: RunComparison, name: str) -> Cell:
    x = m.measure(name)
    return (x.difference, x.low, x.high, x.p_value, x.underpowered)


def _retention(base: Evaluation, treat: Evaluation, confidence: float = 0.95) -> Cell:
    """Paired retention (the expected memory was among the memories retrieval considered)
    on scorable known-value probes of both runs, with the Phase 4 statistics."""
    pairs = [
        (x.retrieval.expected_rank is not None, y.retrieval.expected_rank is not None)
        for x, y in zip(base.probes, treat.probes, strict=True)
        if x.expected.status is ExpectationStatus.KNOWN
    ]
    both = sum(a and b for a, b in pairs)
    t_only = sum(b and not a for a, b in pairs)
    b_only = sum(a and not b for a, b in pairs)
    neither = len(pairs) - both - t_only - b_only
    diff = newcombe_paired(both, t_only, b_only, neither, confidence)
    if diff is None:
        return (None, None, None, None, True)
    floor = min_achievable_p(t_only + b_only)
    return (
        quantize(diff[0]),
        quantize(diff[1]),
        quantize(diff[2]),
        significant(mcnemar_exact(b_only, t_only)),
        floor > 0.05,
    )


class LabReproduction(Record):
    original: Digest
    rerun: Digest
    identical: bool
    differing: tuple[str, ...]


def reproduce_lab(study: str, store: ArtifactStore, registry: Registry = DEFAULT_REGISTRY,
                  only: Sequence[str] | None = None) -> LabReproduction:  # fmt: skip
    original = store.get_record(LabStudy, study)
    spec = store.get_record(LabSpec, original.spec)
    rerun, _ = run_lab(spec, store, registry, only)
    differing = tuple(
        f"{a.benchmark}/{a.condition}"
        for a, b in zip(original.runs, rerun.runs, strict=True)
        if a != b
    )
    result = LabReproduction(original=original.digest, rerun=rerun.digest,
                             identical=original == rerun, differing=differing)  # fmt: skip
    store.put_record(result)
    return result


# --- report --------------------------------------------------------------------------------------


def _p(x: Proportion) -> str:
    if x.estimate is None:
        return f"{x.numerator}/{x.denominator}"
    return f"{x.numerator}/{x.denominator}={x.estimate:.2f} [{x.low:.2f},{x.high:.2f}]"


def _cell(c: tuple[float | None, float | None, float | None, float | None, bool]) -> str:
    d, lo, hi, p, under = c
    if d is None:
        return "—"
    return f"{d:+.3f} [{lo:+.2f},{hi:+.2f}] p={p:.2g}{' u' if under else ''}"


def report(study: LabStudy, performance: PerformanceRecord | None = None) -> str:
    lines = ["## Runs", "",
             "| world | condition | correct | retention | contested ans. | compression | merge P | "
             "merge R | blocked | abstraction err | inference err | number loss | token loss | "
             "unsupported | lineage | contradictions kept | derived ans. | stale ans. |",
             "|---" * 18 + "|"]  # fmt: skip
    for r in study.runs:
        m = r.consolidation
        cells = ["—"] * 13
        if m is not None:
            none = Proportion.of(0, 0, 0.95)
            loss = {k: dict(m.feature_loss).get(k, none) for k in ("number", "token")}
            cells = [
                f"{m.compression:.2f}" if m.compression else "—",
                _p(m.merge_precision), _p(m.merge_recall), str(m.blocked_merges),
                _p(m.abstraction_errors), _p(m.inference_errors), _p(loss["number"]),
                _p(loss["token"]), _p(m.unsupported), _p(m.lineage_complete),
                _p(m.contradictions_kept), _p(m.derived_answers), _p(m.stale_answers),
            ]  # fmt: skip
        lines.append(
            f"| {r.benchmark} | {r.condition} | {_p(r.correct)} | {_p(r.in_state)} | "
            f"{_p(r.contested)} | " + " | ".join(cells) + " |"
        )
    lines += ["", "## Paired comparisons (treatment - baseline; Holm within world x family; "
              "u = underpowered)", "",
              "| world | family | baseline | treatment | known.correct | retention |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for c in study.comparisons:
        lines.append(f"| {c.benchmark} | {c.family} | {c.baseline} | {c.treatment} | "
                     f"{_cell(c.known_correct)} | {_cell(c.in_state)} |")  # fmt: skip
    if performance is not None:
        t = dict(performance.timings)
        lines += ["", "## Cost (environment-bound)", "",
                  "| world | condition | run s | measure+replay s | hierarchy KiB | log KiB |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        for r in study.runs:
            tag = f"{r.benchmark}/{r.condition}"
            lines.append(f"| {r.benchmark} | {r.condition} | {t[tag + ':run_s']:.2f} | "
                         f"{t[tag + ':measure_and_replay_s']:.2f} | "
                         f"{t[tag + ':hierarchy_bytes'] / 1024:.0f} | "
                         f"{t[tag + ':log_bytes'] / 1024:.0f} |")  # fmt: skip
    return "\n".join(lines) + "\n"


# --- end-to-end demonstration -------------------------------------------------------------------


class Traversal(Record):
    """One answer followed down to its evidence."""

    probe: str
    answer: str | None
    memory: str  # the cited memory (derived id or version digest)
    level: Level
    status: EpistemicStatus
    chain: tuple[str, ...]  # derived ancestors, then L1 versions
    sources: tuple[str, ...]  # the L0 experiences' sources


class Efficiency(Record):
    """Memories a query considers; latencies are environment-bound and live in the
    demonstration's separate PerformanceRecord."""

    world: str
    raw_universe: float  # mean memories a raw-only query considers
    consolidated_universe: float


class Demonstration(Record):
    clean: Digest  # run: hierarchical consolidation, mixed retrieval
    perturbed: Digest  # the same with contaminated input, re-consolidated
    comparison: Digest  # evaluation.compare(clean, perturbed): variable "interventions"
    correct_clean: Proportion
    correct_perturbed: Proportion
    traversal: Traversal
    lossy: Digest  # a hierarchy with a lossy merge
    loss: tuple[str, ...]  # that merge's lost features
    efficiency: Efficiency


def demonstration(
    store: ArtifactStore, representation: RepresentationSpec, registry: Registry = DEFAULT_REGISTRY
) -> tuple[Demonstration, PerformanceRecord]:
    from memoria.scenarios import PHASE7_WORLDS, consolidation_world

    worlds = {w.name: w for w in PHASE7_WORLDS}
    dataset, bench = consolidation_world(worlds["baseline"])
    store.put_record(dataset)
    store.put_record(bench)
    mixed = store.put_record(RETRIEVAL["mixed"])
    hier = store.put_record(STRATEGIES["E:hierarchical"])

    def run(cp: str, *interventions: InterventionSpec) -> RunRecord:
        return execute(
            RunManifest(
                name="demonstration",
                dataset=dataset.digest,
                policy="episodic-v1",
                responder=EXTRACTIVE,
                representation=representation,
                retrieval_policy=mixed,
                consolidation_policy=cp,
                interventions=interventions,
            ),
            store,
            registry,
        )

    clean = run(hier)
    contaminate = InterventionSpec(
        name="contaminate",
        params=(("delay_days", 0), ("rate", 0.5), ("seed", 7), ("source", "contaminant")),
    )
    perturbed = run(hier, contaminate)
    ec, ep = evaluate(clean.digest, store), evaluate(perturbed.digest, store)
    comparison = compare(ec.digest, ep.digest, store)
    hierarchies = [store.get_record(Hierarchy, h) for h in clean.hierarchies]
    by_id = {m.digest: (h, m) for h in hierarchies for m in h.memories}
    outcomes = store.get_record(RunOutcomes, clean.outcomes)
    with MemoryLog.load(store.get(clean.log)) as log:
        traversal = None
        for o in outcomes.probes:
            if o.cited and o.cited[0] in by_id:
                h, m = by_id[o.cited[0]]
                chain = lineage(h, m.memory_id)
                sources = sorted(
                    log.experience(d).source  # type: ignore[union-attr]
                    for d in m.derived_from
                )
                traversal = Traversal(probe=o.probe, answer=o.output, memory=m.memory_id,
                                      level=m.level, status=m.status, chain=tuple(chain),
                                      sources=tuple(sources))  # fmt: skip
                break
    assert traversal is not None, "no probe was answered from a derived memory"
    lossy_run = run(store.put_record(STRATEGIES["D:abstractive"]))
    lossy = [store.get_record(Hierarchy, h) for h in lossy_run.hierarchies][-1]
    worst = max(lossy.losses, key=lambda r: (len(r.lost), r.memory))
    efficiency, latencies = _efficiency(worlds["repetitive"], store, representation, registry)
    demo = Demonstration(
        clean=clean.digest,
        perturbed=perturbed.digest,
        comparison=comparison.digest,
        correct_clean=ec.measurement("known.correct").proportion,
        correct_perturbed=ep.measurement("known.correct").proportion,
        traversal=traversal,
        lossy=lossy.digest,
        loss=worst.lost,
        efficiency=efficiency,
    )
    store.put_record(demo)
    performance = PerformanceRecord(experiment=demo.digest, environment=environment(),
                                    timings=latencies)  # fmt: skip
    store.put_record(performance)
    return demo, performance


def _efficiency(
    world: object, store: ArtifactStore, representation: RepresentationSpec, registry: Registry
) -> tuple[Efficiency, tuple[tuple[str, float], ...]]:
    """Per-query cost of raw-only versus consolidated-only retrieval over one world's final
    hierarchy: memories considered and median latency (environment-bound)."""
    import statistics as stats

    from memoria.consolidation import consolidate
    from memoria.hybrid import Engine, HybridQuery
    from memoria.scenarios import WorldSpec, consolidation_world

    assert isinstance(world, WorldSpec)
    dataset, _ = consolidation_world(world)
    embedder = registry.embedder(representation.embedder)
    from memoria.formation import EpisodicPolicy, form

    with MemoryLog(":memory:") as log:
        for s in dataset.steps:
            form(log, s.experience, EpisodicPolicy(), recorded_at=s.recorded_at)
        end = max(p.known_at for p in dataset.probes)
        base = Corpus.from_log(log, end)
        h = consolidate(base, STRATEGIES["E:hierarchical"], end, embedder)
        corpus = base.with_derived(h.memories, ())
    out = {}
    for mode in ("raw", "consolidated"):
        pol = RETRIEVAL[mode]
        engine = Engine(corpus, embedder)
        sizes, latencies = [], []
        for p in dataset.probes:
            q = HybridQuery(text=p.text, valid_at=p.valid_at, known_at=end, limit=1, key=p.key)
            engine.retrieve(pol, q)  # warm the embedding memo; time the second call
            start = time.perf_counter()
            engine.retrieve(pol, q)
            latencies.append(time.perf_counter() - start)
            sizes.append(len(engine.universe(pol.levels)))
        out[mode] = (math.fsum(sizes) / len(sizes), stats.median(latencies))
    efficiency = Efficiency(
        world=world.name,
        raw_universe=quantize(out["raw"][0]),
        consolidated_universe=quantize(out["consolidated"][0]),
    )
    medians = (
        ("raw:retrieve_median_s", significant(out["raw"][1])),
        ("consolidated:retrieve_median_s", significant(out["consolidated"][1])),
    )
    return efficiency, medians


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m memoria.consolidation_eval <store dir> [world ...]``: run the Phase 7
    lab on the named worlds (default: all), reproduce it, run the demonstration, report."""
    from memoria.embeddings import HashedNgramEmbedder
    from memoria.scenarios import PHASE7_WORLDS, consolidation_world

    args = list(sys.argv[1:] if argv is None else argv)
    store = ArtifactStore(args[0] if args else "var/artifacts")
    names = args[1:] or [w.name for w in PHASE7_WORLDS]
    representation = RepresentationSpec(embedder=HashedNgramEmbedder().spec)
    benchmarks = []
    for w in PHASE7_WORLDS:
        if w.name in names:
            dataset, bench = consolidation_world(w)
            store.put_record(dataset)
            benchmarks.append(store.put_record(bench))
    spec = lab_spec(store, benchmarks, representation)
    study, performance = run_lab(spec, store)
    reproduction = reproduce_lab(study.digest, store)
    demo, demo_performance = demonstration(store, representation)
    print(f"spec          {study.spec}")
    print(f"study         {study.digest}")
    print(f"performance   {performance.digest}")
    print(f"reproduced    {'identical' if reproduction.identical else reproduction.differing}")
    print(f"demonstration {demo.digest}")
    print(f"demo timings  {dict(demo_performance.timings)}")
    print()
    print(demo.model_dump_json(indent=1))
    print()
    print(report(study, performance))


if __name__ == "__main__":
    main()
