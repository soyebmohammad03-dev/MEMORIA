"""The Phase 8-10 laboratory: memory graph, forgetting and interference, alone and together.

Research questions (docs/experiments/phase8-10-graph-forgetting-interference.md):

- RQ-G1 Can a provenance-preserving graph represent entities, claims, events, time,
  contradictions and derived memories without collapsing evidence and inference?
- RQ-G2 How does graph structure affect retrieval, contradiction discovery, provenance
  traversal and long-horizon reasoning?
- RQ-F1 What happens to retrieval quality, provenance completeness, contradiction
  visibility and memory integrity under different forgetting policies?
- RQ-F2 Can a system forget operationally while retaining auditable historical evidence?
- RQ-I1 How do memories interfere as the population grows?  RQ-I2 Which mechanisms are
  most damaging?  RQ-I3 Can interference be measured separately from retrieval failure?

Every condition of the world matrix is an ordinary run (manifest schema v4), evaluated by
Phase 4 and compared with :func:`evaluation.compare` when exactly one manifest variable
differs (Holm within world and family). Probes about one key share memory, so a second,
cluster-level analysis averages each key's probes and compares keys (paired t, sign test,
d_z). Comparisons that change several variables at once are *composite*: reported with
the same paired statistics, labelled as not attributable to any one variable. The
interference study runs outside the manifest runner on generated populations (see
:mod:`memoria.interference`), with replicates as independent units.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Literal

from pydantic import Field

from memoria.artifacts import ArtifactStore
from memoria.consolidation import ConsolidationPolicy, Hierarchy, consolidate, invalidated
from memoria.consolidation_eval import (
    RETRIEVAL,
    STRATEGIES,
    Cell,
    ConsolidationBenchmark,
    _holm,
    _paired,
    _retention,
    importance,
    measure_consolidation,
)
from memoria.core import (
    Dataset,
    Digest,
    ExpectationStatus,
    Experience,
    InterventionSpec,
    Level,
    Record,
    RepresentationSpec,
    RunManifest,
    RunOutcomes,
    RunRecord,
    Scalar,
    Step,
    quantize,
    unit_interval,
)
from memoria.embeddings import Embedder, HashedNgramEmbedder
from memoria.entities import AGGRESSIVE, CONSERVATIVE
from memoria.evaluation import DEFAULT_SPEC, Evaluation, RunComparison, compare, evaluate
from memoria.experiments import DEFAULT_REGISTRY, Registry, execute, hybrid_uses_vectors
from memoria.forgetting import (
    ForgettingMetrics,
    ForgettingPolicy,
    ForgettingRecord,
    Selector,
    apply,
    forget,
    measure_forgetting,
)
from memoria.forgetting import policy as forgetting_policy
from memoria.formation import EpisodicPolicy, form
from memoria.graph import (
    GraphDiff,
    GraphMetrics,
    GraphPolicy,
    GraphSnapshot,
    MemoryGraph,
    ProvenanceTrace,
    build_graph,
    diff_graphs,
    measure_graph,
    temporal_diagnostics,
    trace_provenance,
    verify_snapshot,
)
from memoria.hybrid import (
    RETRIEVABLE,
    STATE_EXCLUDE,
    Corpus,
    Exclusion,
    GeneratorSpec,
    HybridTrace,
    RetrievalPolicy,
    equal_weights,
)
from memoria.hybrid_eval import RECENCY
from memoria.interference import (
    ATTRIBUTES,
    CITIES,
    MECHANISMS,
    InterferenceSpec,
    LoadCurve,
    Mechanism,
    Observation,
    false_merges,
    load_curve,
    manipulation,
    run_scenario,
    scenario,
)
from memoria.scenarios import WorldSpec, consolidation_world, day
from memoria.semantic_eval import PerformanceRecord, environment
from memoria.statistics import (
    PairedMean,
    Proportion,
    mcnemar_exact,
    min_achievable_p,
    newcombe_paired,
    paired_mean,
    significant,
)
from memoria.store import MemoryLog

# --- worlds --------------------------------------------------------------------------------------

WORLDS = (
    WorldSpec(name="stable", seed=8, entities=3, horizon_days=360, change_every_days=180),
    WorldSpec(
        name="rapid", seed=8, entities=3, horizon_days=360, change_every_days=20, delay_days=5
    ),
    WorldSpec(
        name="contradictory",
        seed=8,
        entities=3,
        horizon_days=360,
        change_every_days=90,
        contradiction_rate=0.6,
        correction_rate=0.3,
    ),
    WorldSpec(
        name="many-entities",
        seed=8,
        entities=10,
        horizon_days=360,
        change_every_days=90,
        probes_per_key=2,
    ),
    WorldSpec(
        name="overlap",
        seed=8,
        entities=5,
        horizon_days=360,
        change_every_days=90,
        repetition=4,
        note_share=0.9,
    ),
    WorldSpec(
        name="source-noise",
        seed=8,
        entities=3,
        horizon_days=360,
        change_every_days=90,
        contradiction_rate=0.3,
        poison=2,
        noise=40,
    ),
    WorldSpec(name="long", seed=8, entities=3, horizon_days=1080, change_every_days=120),
    WorldSpec(
        name="adversarial",
        seed=8,
        entities=3,
        horizon_days=360,
        change_every_days=60,
        repetition=3,
        contradiction_rate=0.4,
        poison=3,
        negations=True,
        grid_days=30,
    ),
)
ABLATION_WORLDS = ("adversarial", "contradictory", "overlap")
_DISTRACTORS = ("collision", "entity", "rival", "semantic")


def world_distractors(
    world: Dataset, per_key: int = 4, kinds: Sequence[str] = _DISTRACTORS
) -> Dataset:
    """Interference to inject into a world (as the ``inject`` intervention): for each key
    its probes ask about, ``per_key`` distractors cycling over ``kinds``: a free-text report
    about a near-collision name (collision: ``Ana`` -> ``Anai``), the same entity with
    another attribute (entity), the same key with another value at another time (rival:
    older and newer reports compete), another (generated) entity with the same attribute
    (semantic). Seeded by the world's identity; unreliable source class ``forum``. The
    world's probes and expectations are unchanged (I20), so an answer from a rival is
    classified ``contaminated``."""
    keys = sorted({p.key for p in world.probes if p.key is not None})
    times = [s.experience.occurred_at for s in world.steps]
    start, span = min(times), max(times) - min(times)
    seed = world.digest
    steps = []
    for key in keys:
        entity_id, attribute = key.split(".", 1)
        name = entity_id.capitalize()
        for i in range(per_key):
            kind = kinds[i % len(kinds)]
            t = start + span * unit_interval(0, seed, "distractor", key, i)
            v = CITIES[int(unit_interval(0, seed, "value", key, i) * len(CITIES))]
            if kind == "semantic":
                text = f"set x{i}{entity_id}.{attribute} = {v}"
            elif kind == "rival":
                text = f"set {key} = {v}"
            elif kind == "entity":
                other = ATTRIBUTES[int(unit_interval(0, seed, "attribute", key, i) * 24)]
                text = f"set {entity_id}.{other} = {v}"
            else:
                text = f"{name}i's {attribute} is {v}."
            src = f"forum:inject-{key}-{i}"
            steps.append(
                Step(experience=Experience(source=src, content=text, occurred_at=t), recorded_at=t)
            )
    steps.sort(key=lambda s: (s.recorded_at, s.experience.source))
    return Dataset(
        name=f"distractors:{world.name}", version=f"{seed}/{'+'.join(kinds)}", steps=tuple(steps)
    )


# --- policies ------------------------------------------------------------------------------------

GRAPH = {
    "conservative": GraphPolicy(name="graph-conservative", version="1", resolution=CONSERVATIVE),
    "aggressive": GraphPolicy(name="graph-aggressive", version="1", resolution=AGGRESSIVE),
    "no-value-mentions": GraphPolicy(
        name="graph-no-value-mentions", version="1", resolution=CONSERVATIVE, value_mentions=False
    ),
    "historical": GraphPolicy(
        name="graph-historical", version="1", resolution=CONSERVATIVE, view="historical"
    ),
}


def _variant(
    name: str,
    base: RetrievalPolicy,
    signals: Sequence[str],
    *,
    graph_generator: bool = True,
    only_graph: bool = False,
    exclude: Sequence[Exclusion] | None = None,
) -> RetrievalPolicy:
    gens = [g for g in base.generators if not only_graph]
    if graph_generator or only_graph:
        gens.append(GeneratorSpec(name="graph", limit=20, params=(("hops", 1),)))
    lexical = base.signal("lexical")
    params: dict[str, Mapping[str, Scalar]] = {"recency": {**RECENCY, "half_life_days": 30.0}}
    if lexical is not None:
        params["lexical"] = dict(lexical.params)
    return RetrievalPolicy.model_validate(
        base.model_dump()
        | {
            "name": name,
            "generators": [g.model_dump() for g in sorted(gens, key=lambda g: g.name)],
            "signals": [s.model_dump() for s in equal_weights(signals, params)],
            **({"exclude": sorted(exclude)} if exclude is not None else {}),
        }
    )


_MIXED = RETRIEVAL["mixed"]
_BASE_SIGNALS = ("attribute", "lexical", "semantic")
POLICIES = {
    "mixed": _MIXED,
    "graph-assisted": _variant("graph-assisted", _MIXED, _BASE_SIGNALS),
    "graph-only": _variant("graph-only", _MIXED, ("graph_claim", "graph_entity"), only_graph=True),
    "graph-hybrid": _variant(
        "graph-hybrid", _MIXED, (*_BASE_SIGNALS, "graph_claim", "graph_entity")
    ),
    "graph-no-claim": _variant("graph-no-claim", _MIXED, (*_BASE_SIGNALS, "graph_entity")),
    "graph-no-entity": _variant("graph-no-entity", _MIXED, (*_BASE_SIGNALS, "graph_claim")),
    "graph+contradiction": _variant(
        "graph+contradiction",
        _MIXED,
        (*_BASE_SIGNALS, "graph_claim", "graph_contradiction", "graph_entity"),
    ),
    "mixed-no-temporal": _variant(
        "mixed-no-temporal", _MIXED, _BASE_SIGNALS, graph_generator=False, exclude=STATE_EXCLUDE
    ),
    "mixed+suppression": _variant(
        "mixed+suppression", _MIXED, (*_BASE_SIGNALS, "suppression"), graph_generator=False
    ),
    "text": _variant("text", _MIXED, ("lexical", "semantic"), graph_generator=False),
    "mixed+recency": _variant(
        "mixed+recency", _MIXED, (*_BASE_SIGNALS, "recency"), graph_generator=False
    ),
}


def _forgetting() -> dict[str, ForgettingPolicy]:
    imp = importance()
    p = forgetting_policy
    return {
        "age": p("age-120", "age", max_age_days=120),
        "fifo": p("fifo-60", "fifo", capacity=60),
        "recency": p("recency-60", "recency", "suppress", half_life_days=60, threshold=0.5),
        "importance": p("importance-0.5", "importance", importance=imp),
        "access": p("access-1", "access", grace_days=60, min_access=1),
        "validity": p("validity-0", "validity", grace_days=0),
        "contradiction": p("contradiction", "contradiction", forget_contested=0),
        "provenance": p("provenance-keep-1", "provenance", keep=1),
        "hybrid": p(
            "hybrid-3",
            "hybrid",
            importance=imp,
            grace_days=60,
            max_age_days=120,
            min_access=1,
            min_votes=3,
        ),
        "validity-retain": p("validity-0-retain", "validity", derived="retain", grace_days=0),
        "entity": p("forget-entity-ana", "selective", selector=Selector(entity="ana")),
        "source": p(
            "forget-source-forum",
            "selective",
            preserve_aggregates=True,
            selector=Selector(source="forum"),
        ),
        "derived-only": p(
            "forget-derived", "selective", selector=Selector(levels=(Level.L2, Level.L3, Level.L4))
        ),
        "contested": p("contradiction-contested", "contradiction", forget_contested=1),
    }


FORGETTING = _forgetting()
SYSTEM_FORGETTING = "validity"  # the forgetting-enabled system (ablations vary the rule)
HIERARCHICAL = STRATEGIES["E:hierarchical"]
ABSTRACTIVE = STRATEGIES["D:abstractive"]


# --- the world matrix ----------------------------------------------------------------------------


class Condition(Record):
    """One condition: retrieval (and graph) policy, consolidation, forgetting, interference.
    ``inject`` names the distractor kinds injected (empty: none)."""

    name: str = Field(min_length=1)
    retrieval: Digest
    graph: Digest | None = None
    consolidation: Digest | None = None
    forgetting: Digest | None = None
    inject: tuple[str, ...] = ()
    worlds: tuple[str, ...] = ()  # empty = every world


class Comparison(Record):
    family: str
    baseline: str
    treatment: str
    composite: bool = False  # several variables differ: not attributable to one


class LabSpec(Record):
    """Everything that determines the matrix. Its digest is the study identity."""

    name: str = Field(min_length=1)
    benchmarks: tuple[Digest, ...] = Field(min_length=1)
    representation: RepresentationSpec
    conditions: tuple[Condition, ...]
    comparisons: tuple[Comparison, ...]


def lab_spec(
    store: ArtifactStore, benchmarks: Sequence[str], representation: RepresentationSpec
) -> LabSpec:
    put = store.put_record
    pol = {n: put(p) for n, p in POLICIES.items()}
    gr = {n: put(p) for n, p in GRAPH.items()}
    fg = {n: put(p) for n, p in FORGETTING.items()}
    hier, abstr = put(HIERARCHICAL), put(ABSTRACTIVE)
    ab = ABLATION_WORLDS
    sys_f = fg[SYSTEM_FORGETTING]

    def c(
        name: str,
        retrieval: str = "mixed",
        worlds: tuple[str, ...] = (),
        *,
        graph: str | None = None,
        consolidation: str | None = None,
        forgetting: str | None = None,
        inject: tuple[str, ...] = (),
    ) -> Condition:
        if graph is not None and retrieval == "mixed":
            retrieval = "graph-hybrid"  # the graph system: graph signals and generator
        return Condition(
            name=name,
            retrieval=pol[retrieval],
            worlds=worlds,
            graph=gr[graph] if graph else None,
            consolidation=consolidation,
            forgetting=forgetting,
            inject=inject,
        )

    conditions = [
        c("base"),
        c("consolidated", consolidation=hier),
        c("graph", graph="conservative"),
        c("forgetting", forgetting=sys_f),
        c("interference", inject=_DISTRACTORS),
        c("ladder:+consolidation", consolidation=hier, inject=_DISTRACTORS),
        c("ladder:+graph", consolidation=hier, inject=_DISTRACTORS, graph="conservative"),
        c(
            "combined",
            consolidation=hier,
            forgetting=sys_f,
            inject=_DISTRACTORS,
            graph="conservative",
        ),
        # graph signals (A: base, B: assisted, C: graph-only, D: graph = hybrid)
        *[
            c(f"graph:{n}", n, ab, graph="conservative")
            for n in (
                "graph-assisted",
                "graph-only",
                "graph-no-claim",
                "graph-no-entity",
                "graph+contradiction",
            )
        ],
        *[
            c(f"resolution:{n}", "graph-hybrid", (*ab, "many-entities"), graph=n)
            for n in ("aggressive", "no-value-mentions")
        ],
        *[
            c(f"forget:{n}", "mixed", ab, forgetting=fg[n])
            for n in FORGETTING
            if n not in ("recency", "validity-retain")
        ],
        c("mixed+suppression", "mixed+suppression", ab),
        c("forget:recency", "mixed+suppression", ab, forgetting=fg["recency"]),
        c("graph+consolidation", worlds=ab, consolidation=hier, graph="conservative"),
        c("graph+abstractive", worlds=ab, consolidation=abstr, graph="conservative"),
        c("temporal:off", "mixed-no-temporal", ab),
        c("consolidated+forget", worlds=ab, consolidation=hier, forgetting=sys_f),
        c(
            "consolidated+forget-retain",
            worlds=ab,
            consolidation=hier,
            forgetting=fg["validity-retain"],
        ),
        c("graph+forget", worlds=ab, forgetting=sys_f, graph="conservative"),
        c("graph:historical+forget", "graph-hybrid", ab, graph="historical", forgetting=sys_f),
        *[c(f"inject:{k}", "mixed", ab, inject=(k,)) for k in _DISTRACTORS],
    ]
    cmp = Comparison
    comparisons = [
        *[
            cmp(family="system", baseline="base", treatment=t)
            for t in ("consolidated", "graph", "forgetting", "interference")
        ],
        cmp(family="ladder", baseline="interference", treatment="ladder:+consolidation"),
        cmp(family="ladder", baseline="ladder:+consolidation", treatment="ladder:+graph"),
        cmp(family="ladder", baseline="ladder:+graph", treatment="combined"),
        cmp(family="composite", baseline="base", treatment="combined", composite=True),
        cmp(family="composite", baseline="interference", treatment="combined", composite=True),
        *[
            cmp(family="graph", baseline="base", treatment=t)
            for t in ("graph:graph-assisted", "graph:graph-only", "graph")
        ],
        *[
            cmp(family="graph-ablation", baseline="graph", treatment=f"graph:{n}")
            for n in ("graph-no-claim", "graph-no-entity", "graph+contradiction")
        ],
        *[
            cmp(family="resolution", baseline="graph", treatment=f"resolution:{n}")
            for n in ("aggressive", "no-value-mentions")
        ],
        *[
            cmp(family="forgetting", baseline="base", treatment=f"forget:{n}")
            for n in FORGETTING
            if n not in ("recency", "validity-retain")
        ],
        cmp(family="forgetting", baseline="mixed+suppression", treatment="forget:recency"),
        cmp(family="consolidation", baseline="graph", treatment="graph+consolidation"),
        cmp(family="consolidation", baseline="graph", treatment="graph+abstractive"),
        cmp(family="temporal", baseline="base", treatment="temporal:off"),
        cmp(
            family="provenance",
            baseline="consolidated+forget",
            treatment="consolidated+forget-retain",
        ),
        cmp(family="provenance", baseline="graph+forget", treatment="graph:historical+forget"),
        *[
            cmp(family="interference", baseline="base", treatment=f"inject:{k}")
            for k in _DISTRACTORS
        ],
    ]
    return LabSpec(
        name="phase8-10-memory-lab",
        benchmarks=tuple(benchmarks),
        representation=representation,
        conditions=tuple(conditions),
        comparisons=tuple(comparisons),
    )


class RunMetrics(Record):
    """Integrity measurements of one run, from its stored artifacts."""

    answered: int
    integrity: Proportion  # answered from memory whose evidence is available and current
    stale_answers: Proportion  # answered from a stale derived memory
    hidden_evidence_answers: Proportion  # answered from a derived memory whose evidence is hidden
    derived_answers: Proportion
    hidden_share: Proportion | None  # memories unavailable, pooled over probes (descriptive)


class LabRun(Record):
    world: str
    condition: str
    manifest: Digest
    run: Digest
    evaluation: Digest
    correct: Proportion  # known.correct
    in_state: Proportion  # known.expected_in_state (the expected memory was retrievable)
    contested: Proportion
    unknown: Proportion
    wrong_memory: Proportion  # outcome.wrong_memory (another subject's memory answered)
    contaminated: Proportion  # outcome.contaminated (an injected report answered)
    metrics: RunMetrics


class Effect(Record):
    """A paired effect of one comparison in one world."""

    world: str
    family: str
    baseline: str
    treatment: str
    composite: bool
    comparison: Digest | None  # the stored RunComparison (single-variable only)
    known_correct: Cell  # (difference, low, high, Holm p within world x family, underpowered)
    in_state: Cell
    clusters: PairedMean  # key-level mean difference in known.correct (keys independent)
    lost: int  # probes whose expected memory was retrievable before and not after
    gained: int
    retrievable_before: int


class LabStudy(Record):
    spec: Digest
    runs: tuple[LabRun, ...]
    effects: tuple[Effect, ...]


class Reproduction(Record):
    """A re-execution of a stored study (lab or interference) compared digest for digest."""

    original: Digest
    rerun: Digest
    identical: bool
    differing: tuple[str, ...]


def _manifest(
    bench: ConsolidationBenchmark, c: Condition, spec: LabSpec, store: ArtifactStore
) -> RunManifest:
    interventions: tuple[InterventionSpec, ...] = ()
    if c.inject:
        world = store.get_record(Dataset, bench.dataset)
        d = store.put_record(world_distractors(world, kinds=c.inject))
        interventions = (InterventionSpec(name="inject", params=(("dataset", d),)),)
    uses_vectors = hybrid_uses_vectors(store.get_record(RetrievalPolicy, c.retrieval)) or (
        c.consolidation is not None
        and store.get_record(ConsolidationPolicy, c.consolidation).uses_vectors
    )
    return RunManifest(
        name=f"{bench.name}/{c.name}",
        dataset=bench.dataset,
        interventions=interventions,
        policy="episodic-v1",
        responder="extractive-top1-v1",
        representation=spec.representation if uses_vectors else None,
        retrieval_policy=c.retrieval,
        consolidation_policy=c.consolidation,
        graph_policy=c.graph,
        forgetting_policy=c.forgetting,
    )


def run_metrics(record: RunRecord, store: ArtifactStore, confidence: float = 0.95) -> RunMetrics:
    """Answer integrity: an answer is intact if its cited memory is available and, when
    derived, is not stale and none of the L1 evidence it covers is unavailable."""
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    dataset = store.get_record(Dataset, record.dataset)
    hierarchies = [store.get_record(Hierarchy, h) for h in record.hierarchies]
    derived = {m.digest: m for h in hierarchies for m in h.memories}
    records = [store.get_record(ForgettingRecord, d) for d in record.forgetting]
    answered = intact = stale = hidden_ev = from_derived = 0
    hidden = total = 0
    with MemoryLog.load(store.get(record.log)) as log:
        for i, (probe, o) in enumerate(zip(dataset.probes, outcomes.probes, strict=True)):
            state: dict[str, str] = {}
            if records:
                r = records[i]
                state = {d.entry: d.state for d in r.decisions}
                hidden += sum(1 for s in state.values() if s not in RETRIEVABLE)
                total += len(state)
            if not o.cited:
                continue
            answered += 1
            cited = o.cited[0]
            m = derived.get(cited)
            if m is None:
                intact += state.get(cited, "active") in RETRIEVABLE
                continue
            from_derived += 1
            h = next(h for h in hierarchies if any(x.digest == cited for x in h.memories))
            corpus = Corpus.from_log(log, probe.known_at)
            is_stale = m.memory_id in invalidated(h, corpus)
            ev_hidden = any(state.get(v, "active") not in RETRIEVABLE for v in m.versions)
            stale += is_stale
            hidden_ev += ev_hidden
            intact += not is_stale and not ev_hidden
    p = Proportion.of
    return RunMetrics(
        answered=answered,
        integrity=p(intact, answered, confidence),
        stale_answers=p(stale, answered, confidence),
        hidden_evidence_answers=p(hidden_ev, answered, confidence),
        derived_answers=p(from_derived, answered, confidence),
        hidden_share=p(hidden, total, confidence) if records else None,
    )


def _cluster(base: Evaluation, treat: Evaluation, confidence: float) -> PairedMean:
    """Key-level analysis: per key, mean known.correct (treatment) - mean (baseline) over
    its scorable known-value probes; keys are the independent units."""
    per: dict[str, list[float]] = {}
    for x, y in zip(base.probes, treat.probes, strict=True):
        if x.expected.status is not ExpectationStatus.KNOWN:
            continue
        if x.agreement.value == "unscorable" or y.agreement.value == "unscorable":
            continue
        key = x.probe.rsplit("-", 1)[0]
        per.setdefault(key, []).append(
            float(y.outcome.value == "correct") - float(x.outcome.value == "correct")
        )
    return paired_mean([math.fsum(v) / len(v) for _, v in sorted(per.items())], confidence)


def _composite(
    base: Evaluation, treat: Evaluation, confidence: float = 0.95, alpha: float = 0.05
) -> Cell:
    """known.correct paired on scorable known-value probes of both runs (Phase 4 method)."""
    pairs = [
        (x.outcome.value == "correct", y.outcome.value == "correct")
        for x, y in zip(base.probes, treat.probes, strict=True)
        if x.expected.status is ExpectationStatus.KNOWN
        and "unscorable" not in (x.outcome.category.value, y.outcome.category.value)
    ]
    both = sum(a and b for a, b in pairs)
    t_only = sum(b and not a for a, b in pairs)
    b_only = sum(a and not b for a, b in pairs)
    diff = newcombe_paired(both, t_only, b_only, len(pairs) - both - t_only - b_only, confidence)
    if diff is None:
        return (None, None, None, None, True)
    return (
        quantize(diff[0]),
        quantize(diff[1]),
        quantize(diff[2]),
        significant(mcnemar_exact(b_only, t_only)),
        min_achievable_p(t_only + b_only) > alpha,
    )


def run_lab(
    spec: LabSpec,
    store: ArtifactStore,
    registry: Registry = DEFAULT_REGISTRY,
    only: Sequence[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[LabStudy, PerformanceRecord]:
    runs: list[LabRun] = []
    timings: list[tuple[str, float]] = []
    evals: dict[tuple[str, str], Evaluation] = {}
    for bdigest in spec.benchmarks:
        bench = store.get_record(ConsolidationBenchmark, bdigest)
        for c in spec.conditions:
            if (c.worlds and bench.name not in c.worlds) or (only and c.name not in only):
                continue
            manifest = _manifest(bench, c, spec, store)
            start = time.perf_counter()
            record = execute(manifest, store, registry)
            timings.append((f"{bench.name}/{c.name}:run_s", time.perf_counter() - start))
            ev = evaluate(record.digest, store, DEFAULT_SPEC)
            evals[(bench.name, c.name)] = ev
            m = ev.measurement
            runs.append(
                LabRun(
                    world=bench.name,
                    condition=c.name,
                    manifest=manifest.digest,
                    run=record.digest,
                    evaluation=ev.digest,
                    correct=m("known.correct").proportion,
                    in_state=m("known.expected_in_state").proportion,
                    contested=m("contested.answered").proportion,
                    unknown=m("unknown.abstained").proportion,
                    wrong_memory=m("outcome.wrong_memory").proportion,
                    contaminated=m("outcome.contaminated").proportion,
                    metrics=run_metrics(record, store),
                )
            )
            if log is not None:
                log(
                    f"{bench.name}/{c.name}: correct {runs[-1].correct.numerator}/"
                    f"{runs[-1].correct.denominator} ({timings[-1][1]:.1f}s)"
                )
    effects = []
    worlds = list(dict.fromkeys(r.world for r in runs))
    for world in worlds:
        for family in dict.fromkeys(x.family for x in spec.comparisons):
            cmp = [
                x
                for x in spec.comparisons
                if x.family == family
                and (world, x.baseline) in evals
                and (world, x.treatment) in evals
            ]
            if not cmp:
                continue
            made: list[RunComparison | None] = [
                None
                if x.composite
                else compare(
                    evals[(world, x.baseline)].digest, evals[(world, x.treatment)].digest, store
                )
                for x in cmp
            ]
            correct = _holm(
                [
                    _composite(evals[(world, x.baseline)], evals[(world, x.treatment)])
                    if m is None
                    else _paired(m, "known.correct")
                    for x, m in zip(cmp, made, strict=True)
                ]
            )
            retained = _holm(
                [_retention(evals[(world, x.baseline)], evals[(world, x.treatment)]) for x in cmp]
            )
            for x, rc, kc, ret in zip(cmp, made, correct, retained, strict=True):
                b, t = evals[(world, x.baseline)], evals[(world, x.treatment)]
                known = [
                    (p.retrieval.expected_rank is not None, q.retrieval.expected_rank is not None)
                    for p, q in zip(b.probes, t.probes, strict=True)
                    if p.expected.status is ExpectationStatus.KNOWN
                ]
                effects.append(
                    Effect(
                        world=world,
                        family=family,
                        baseline=x.baseline,
                        treatment=x.treatment,
                        composite=x.composite,
                        comparison=rc.digest if rc else None,
                        known_correct=kc,
                        in_state=ret,
                        clusters=_cluster(b, t, 0.95),
                        lost=sum(a and not q for a, q in known),
                        gained=sum(q and not a for a, q in known),
                        retrievable_before=sum(a for a, _ in known),
                    )
                )
    study = LabStudy(spec=store.put_record(spec), runs=tuple(runs), effects=tuple(effects))
    store.put_record(study)
    performance = PerformanceRecord(
        experiment=study.digest,
        environment=environment(),
        timings=tuple((n, significant(v)) for n, v in timings),
    )
    store.put_record(performance)
    return study, performance


def reproduce_lab(study: str, store: ArtifactStore) -> Reproduction:
    original = store.get_record(LabStudy, study)
    rerun, _ = run_lab(store.get_record(LabSpec, original.spec), store)
    differing = tuple(
        f"{a.world}/{a.condition}" for a, b in zip(original.runs, rerun.runs, strict=True) if a != b
    )
    result = Reproduction(
        original=original.digest,
        rerun=rerun.digest,
        identical=original == rerun,
        differing=differing,
    )
    store.put_record(result)
    return result


# --- graph structure study -----------------------------------------------------------------------


class Growth(Record):
    at: datetime
    nodes: int
    edges: int
    claims: int
    contradictions: int


class StructureResult(Record):
    """RQ-G1: the graph of one world's full history with hierarchical consolidation."""

    world: str
    snapshot: Digest  # stored
    verified: bool  # rebuilt byte-identically from its sources
    metrics_all: GraphMetrics
    metrics_active: GraphMetrics  # after the system forgetting policy (active view)
    forgetting: Digest
    diagnostics: tuple[tuple[str, str, int], ...]  # (check, severity, count)
    decisions: tuple[tuple[str, bool, int], ...]  # resolution (rule, accepted, count)
    false_merges: tuple[int, int]  # (accepted merges across true entities, accepted merges)
    unavailable_claims: int  # memories with no structured claim (free text, patterns)
    inferred_edges: int
    observed_edges: int
    growth: tuple[Growth, ...]


def _world_log(world: WorldSpec) -> tuple[Dataset, MemoryLog]:
    dataset, _ = consolidation_world(world)
    log = MemoryLog(":memory:")
    for s in dataset.steps:
        form(log, s.experience, EpisodicPolicy(), recorded_at=s.recorded_at)
    return dataset, log


def structure(
    world: WorldSpec, store: ArtifactStore, embedder: Embedder, resolution: str = "conservative"
) -> StructureResult:
    dataset, log = _world_log(world)
    with log:
        end = max(p.known_at for p in dataset.probes)
        base = Corpus.from_log(log, end)
        h = consolidate(base, HIERARCHICAL, end, embedder)
        corpus = base.with_derived(h.memories, invalidated(h, base))
        snap = build_graph(corpus, GRAPH[resolution], [h])
        store.put_record(h)
        store.put_record(snap)
        verify_snapshot(store.get_record(GraphSnapshot, snap.digest), corpus, [h])
        rec = forget(corpus, FORGETTING[SYSTEM_FORGETTING], end)
        store.put_record(rec)
        after = build_graph(apply(corpus, rec), GRAPH[resolution], [h])
        growth = []
        for t in sorted({day(world.horizon_days * f) for f in (0.25, 0.5, 0.75, 1.0)}):
            g = build_graph(Corpus.from_log(log, min(t, end)), GRAPH[resolution])
            growth.append(
                Growth(
                    at=min(t, end),
                    nodes=g.node_count,
                    edges=g.edge_count,
                    claims=sum(1 for n in g.nodes if n.kind == "claim"),
                    contradictions=sum(1 for e in g.edges if e.relation == "contradicts"),
                )
            )
    diag: dict[tuple[str, str], int] = {}
    for dg in temporal_diagnostics(snap):
        diag[(dg.check, dg.severity)] = diag.get((dg.check, dg.severity), 0) + 1
    rules: dict[tuple[str, bool], int] = {}
    for d in snap.resolution.decisions:
        rules[(d.rule, d.accepted)] = rules.get((d.rule, d.accepted), 0) + 1
    truth = {m.id: m.normalized.split()[0] for m in snap.resolution.mentions}
    merged = [d for d in snap.resolution.decisions if d.accepted]
    wrong = sum(1 for d in merged if truth[d.a] != truth[d.b])
    return StructureResult(
        world=world.name,
        snapshot=snap.digest,
        verified=True,
        metrics_all=measure_graph(snap, "all"),
        metrics_active=measure_graph(after, "active"),
        forgetting=rec.digest,
        diagnostics=tuple(sorted((c, s, n) for (c, s), n in diag.items())),
        decisions=tuple(sorted((r, a, n) for (r, a), n in rules.items())),
        false_merges=(wrong, len(merged)),
        unavailable_claims=len(snap.unavailable),
        inferred_edges=sum(1 for e in snap.edges if e.status.value == "inferred"),
        observed_edges=sum(1 for e in snap.edges if e.status.value == "observed"),
        growth=tuple(growth),
    )


# --- interference study --------------------------------------------------------------------------

LOADS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
INTERFERENCE_POLICIES = ("mixed", "text", "mixed-no-temporal", "graph-hybrid", "mixed+recency")


class Arm(Record):
    """A mechanism with its modifiers (one load curve per policy)."""

    name: str
    mechanism: Mechanism
    frequency: int = 1
    wording: Literal["template", "paraphrase"] = "template"
    recency: Literal["older", "newer"] = "older"


ARMS = (
    *[
        Arm(name=m, mechanism=m, wording="paraphrase" if m == "consolidation" else "template")
        for m in MECHANISMS
    ],
    Arm(name="contradiction-x4", mechanism="contradiction", frequency=4),
    Arm(name="semantic-paraphrase", mechanism="semantic", wording="paraphrase"),
    Arm(name="semantic-newer", mechanism="semantic", recency="newer"),
)


class InterferenceStudySpec(Record):
    name: str
    seed: int
    replicates: int = Field(ge=2)
    loads: tuple[int, ...]
    arms: tuple[Arm, ...]
    policies: tuple[tuple[str, Digest], ...]
    consolidation: Digest
    graph: Digest
    representation: RepresentationSpec
    k: int = 5


class ArmResult(Record):
    arm: str
    curves: tuple[LoadCurve, ...]  # one per policy
    checks: tuple[tuple[str, float], ...]  # manipulation checks at the largest load (means)
    false_merges: tuple[tuple[str, int, int], ...]  # (resolution policy, wrong, merges)


class InterferenceStudy(Record):
    spec: Digest
    observations: tuple[Observation, ...]
    arms: tuple[ArmResult, ...]


def interference_spec(
    store: ArtifactStore, representation: RepresentationSpec, replicates: int = 24
) -> InterferenceStudySpec:
    return InterferenceStudySpec(
        name="phase10-interference",
        seed=10,
        replicates=replicates,
        loads=LOADS,
        arms=ARMS,
        policies=tuple((n, store.put_record(POLICIES[n])) for n in INTERFERENCE_POLICIES),
        consolidation=store.put_record(ABSTRACTIVE),
        graph=store.put_record(GRAPH["conservative"]),
        representation=representation,
    )


def run_interference(
    spec: InterferenceStudySpec,
    store: ArtifactStore,
    embedder: Embedder,
) -> InterferenceStudy:
    if embedder.spec != spec.representation.embedder:
        raise ValueError("the study declares another embedder")
    policies = [(n, store.get_record(RetrievalPolicy, d)) for n, d in spec.policies]
    cons = store.get_record(ConsolidationPolicy, spec.consolidation)
    gpol = store.get_record(GraphPolicy, spec.graph)

    def view(corpus: Corpus, hs: Sequence[Hierarchy]) -> MemoryGraph:
        return MemoryGraph(build_graph(corpus, gpol, hs))

    observations: list[Observation] = []
    arms = []
    for arm in spec.arms:
        obs: list[Observation] = []
        checks: dict[str, list[float]] = {}
        merges = {"conservative": [0, 0], "aggressive": [0, 0]}
        for r in range(spec.replicates):
            for load in spec.loads:
                s = InterferenceSpec(
                    name=arm.name,
                    seed=spec.seed,
                    mechanism=arm.mechanism,
                    replicate=r,
                    load=load,
                    frequency=arm.frequency,
                    wording=arm.wording,
                    recency=arm.recency,
                )
                ds, sc = scenario(s)
                for o in run_scenario(ds, sc, policies, embedder, cons, view, spec.k):
                    obs.append(o.model_copy(update={"mechanism": arm.mechanism}))
                if load == max(spec.loads):
                    for name, value in manipulation(sc, ds, embedder).items():
                        checks.setdefault(name, []).append(value)
                    for rname, rp in (("conservative", CONSERVATIVE), ("aggressive", AGGRESSIVE)):
                        w, n = false_merges(ds, sc, rp)
                        merges[rname][0] += w
                        merges[rname][1] += n
        curves = tuple(load_curve(obs, arm.mechanism, name) for name, _ in spec.policies)
        arms.append(
            ArmResult(
                arm=arm.name,
                curves=curves,
                checks=tuple(
                    sorted((k, quantize(math.fsum(v) / len(v))) for k, v in checks.items())
                ),
                false_merges=tuple((k, w, n) for k, (w, n) in sorted(merges.items())),
            )
        )
        observations += obs
    study = InterferenceStudy(
        spec=store.put_record(spec), observations=tuple(observations), arms=tuple(arms)
    )
    store.put_record(study)
    return study


def reproduce_interference(study: str, store: ArtifactStore, embedder: Embedder) -> Reproduction:
    """Re-run the study: every observation (and its trace digest) must be identical."""
    original = store.get_record(InterferenceStudy, study)
    rerun = run_interference(
        store.get_record(InterferenceStudySpec, original.spec), store, embedder
    )
    differing = tuple(
        f"{a.mechanism}/{a.policy}/{a.replicate}/{a.load}"
        for a, b in zip(original.observations, rerun.observations, strict=True)
        if a != b
    )
    result = Reproduction(
        original=original.digest,
        rerun=rerun.digest,
        identical=original == rerun,
        differing=differing,
    )
    store.put_record(result)
    return result


# --- interaction experiments ---------------------------------------------------------------------


class Interaction(Record):
    """One graph x forgetting x consolidation x interference intervention and its observed
    effect. These are interventions on one designed world, not universal causal claims."""

    name: str
    world: str
    before: Digest  # snapshot or run
    after: Digest
    measures: tuple[tuple[str, float | int | None], ...]
    note: str


def interactions(
    store: ArtifactStore,
    embedder: Embedder,
    study: LabStudy,
    interference: InterferenceStudy | None,
) -> tuple[Interaction, ...]:
    worlds = {w.name: w for w in WORLDS}
    out = []
    # I1 forgetting a source leaves derived graph claims unsupported (retain) or removes them
    # (cascade); the graph diff locates exactly what was lost.
    w = worlds["source-noise"]
    dataset, log = _world_log(w)
    with log:
        end = max(p.known_at for p in dataset.probes)
        base = Corpus.from_log(log, end)
        h = consolidate(base, HIERARCHICAL, end, embedder)
        corpus = base.with_derived(h.memories, invalidated(h, base))
        g0 = build_graph(corpus, GRAPH["conservative"], [h])
        results = {}
        modes: tuple[Literal["cascade", "retain"], ...] = ("cascade", "retain")
        for mode in modes:
            fp = forgetting_policy(
                f"forget-forum-{mode}",
                "selective",
                derived=mode,
                selector=Selector(source="forum"),
            )
            rec = forget(corpus, fp, end)
            g1 = build_graph(apply(corpus, rec), GRAPH["conservative"], [h])
            store.put_record(g1)
            d = diff_graphs(g0, g1)
            store.put_record(d)
            m = measure_graph(g1, "active")
            results[mode] = (g1, d, m, rec)
        store.put_record(g0)
        (_, dc, mc, rc), (gr, dr, mr, rr) = results["cascade"], results["retain"]
        out.append(
            Interaction(
                name="forget-source-derived-claims",
                world=w.name,
                before=g0.digest,
                after=gr.digest,
                measures=(
                    ("forgotten_nodes_cascade", len(dc.forgotten_nodes)),
                    ("forgotten_nodes_retain", len(dr.forgotten_nodes)),
                    ("unsupported_claims_cascade", mc.unsupported_claims.numerator),
                    ("unsupported_claims_retain", mr.unsupported_claims.numerator),
                    ("claims_in_view", mr.unsupported_claims.denominator),
                    ("dangling_derived_retain", len(rr.dangling)),
                    ("retained_with_hidden_evidence", len(rr.evidence_retained)),
                    ("unsupported_abstractions_retain", len(rr.unsupported_abstractions)),
                    ("aggregates_preserved_keys", len(rc.aggregates)),
                ),
                note="retain keeps derived memories whose forum evidence is hidden; cascade hides "
                "them. The diff lists every node made unavailable.",
            )
        )
    # I4 forgetting one entity cluster: graph fragmentation in the active view.
    w = worlds["many-entities"]
    dataset, log = _world_log(w)
    with log:
        end = max(p.known_at for p in dataset.probes)
        corpus = Corpus.from_log(log, end)
        g0 = build_graph(corpus, GRAPH["conservative"])
        rec = forget(corpus, FORGETTING["entity"], end)
        g1 = build_graph(apply(corpus, rec), GRAPH["conservative"])
        m0, m1 = measure_graph(g0, "active"), measure_graph(g1, "active")
        store.put_record(g0)
        store.put_record(g1)
        targets = [
            e.digest
            for e in corpus.entries
            if (e.claim and e.claim.key.startswith("ana."))
            or (
                e.claim is None and any("Ana" in x.content.split("'")[0].split() for x in e.sources)
            )
        ]
        fm = measure_forgetting(corpus, rec, targets)
        store.put_record(fm)
        out.append(
            Interaction(
                name="forget-entity-fragmentation",
                world=w.name,
                before=g0.digest,
                after=g1.digest,
                measures=(
                    ("components_before", m0.components),
                    ("components_after", m1.components),
                    ("orphans_before", m0.orphans.numerator),
                    ("orphans_after", m1.orphans.numerator),
                    ("unsupported_claims_after", m1.unsupported_claims.numerator),
                    ("precision", fm.precision.estimate if fm.precision else None),
                    ("recall", fm.recall.estimate if fm.recall else None),
                    ("collateral_memories", fm.collateral.numerator if fm.collateral else None),
                    ("non_target_memories", fm.collateral.denominator if fm.collateral else None),
                ),
                note="target = memories whose claim key is ana.* or whose free text names 'Ana' "
                "(the ground-truth entity label); near-collision 'Anna' is a distinct entity.",
            )
        )
    # I3 graph neighbourhood exposes contradictions that ordinary retrieval did not surface.
    for world in ("contradictory", "adversarial"):
        run = next((r for r in study.runs if r.world == world and r.condition == "graph"), None)
        if run is None:
            continue
        exposed, hidden, total = _graph_exposure(run.run, store)
        out.append(
            Interaction(
                name="graph-exposes-contradictions",
                world=world,
                before=run.run,
                after=run.run,
                measures=(
                    ("answered", total),
                    ("graph_contradiction_near_answer", exposed),
                    ("graph_shows_unsurfaced_contradicting_memories", hidden),
                ),
                note="answers whose memory the graph links to a contradiction, and those for "
                "which the graph shows contradicting memories retrieval did not surface",
            )
        )
    # I5 stale derived memories keep competing after their evidence is forgotten (retain).
    for world in ABLATION_WORLDS:
        runs = {r.condition: r for r in study.runs if r.world == world}
        a, b = runs.get("consolidated+forget"), runs.get("consolidated+forget-retain")
        if a is None or b is None:
            continue
        out.append(
            Interaction(
                name="stale-derived-compete",
                world=world,
                before=a.run,
                after=b.run,
                measures=(
                    (
                        "hidden_evidence_answers_cascade",
                        a.metrics.hidden_evidence_answers.numerator,
                    ),
                    ("hidden_evidence_answers_retain", b.metrics.hidden_evidence_answers.numerator),
                    ("answered_retain", b.metrics.answered),
                    ("integrity_cascade", a.metrics.integrity.estimate),
                    ("integrity_retain", b.metrics.integrity.estimate),
                    ("correct_cascade", a.correct.numerator),
                    ("correct_retain", b.correct.numerator),
                ),
                note="same forgetting rule; only the derived-memory semantics differ",
            )
        )
    # I2 consolidation reduces competition but carries abstraction risk (interference runs).
    for world in ("overlap", "adversarial"):
        runs = {r.condition: r for r in study.runs if r.world == world}
        a, b = runs.get("interference"), runs.get("ladder:+consolidation")
        if a is None or b is None:
            continue
        bench = next(
            store.get_record(ConsolidationBenchmark, d)
            for d in store.get_record(LabSpec, study.spec).benchmarks
            if store.get_record(ConsolidationBenchmark, d).name == world
        )
        cm = measure_consolidation(store.get_record(RunRecord, b.run), store, bench)
        out.append(
            Interaction(
                name="consolidation-competition-vs-abstraction",
                world=world,
                before=a.run,
                after=b.run,
                measures=(
                    ("wrong_memory_raw", a.wrong_memory.numerator),
                    ("wrong_memory_consolidated", b.wrong_memory.numerator),
                    ("contaminated_raw", a.contaminated.numerator),
                    ("contaminated_consolidated", b.contaminated.numerator),
                    ("abstraction_errors", cm.abstraction_errors.numerator if cm else None),
                    ("abstracted_statements", cm.abstraction_errors.denominator if cm else None),
                    ("inference_errors", cm.inference_errors.numerator if cm else None),
                ),
                note="wrong_memory/contaminated are Phase 4 outcomes (another subject's or an "
                "injected memory answered); abstraction errors are Phase 7 measurements",
            )
        )
    # I6 frequency of contradictory reports distorts ranking (interference study).
    if interference is not None:
        by = {a.arm: a for a in interference.arms}
        for arm in ("contradiction", "contradiction-x4"):
            curve = next(c for c in by[arm].curves if c.policy == "mixed")
            top = curve.points[-1]
            out.append(
                Interaction(
                    name="contradiction-frequency",
                    world=f"interference:{arm}",
                    before=interference.digest,
                    after=interference.digest,
                    measures=(
                        ("load", top.load),
                        ("target_at_1", top.target_at_1.numerator),
                        ("n", top.target_at_1.denominator),
                        ("rival_values_exposed", top.contradiction_exposure.numerator),
                        ("onset", curve.onset),
                    ),
                    note="distinct rival reports per load are equal; x4 repeats each four times",
                )
            )
    return tuple(out)


def _graph_exposure(run: str, store: ArtifactStore) -> tuple[int, int, int]:
    """(answers whose memory the graph links to a contradiction, of which the graph shows
    contradicting memories retrieval did not surface, answers). Rebuilds each probe's graph
    and checks it against the snapshot digest the run recorded."""
    record = store.get_record(RunRecord, run)
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    dataset = store.get_record(Dataset, record.dataset)
    manifest = store.get_record(RunManifest, record.manifest)
    assert manifest.graph_policy is not None
    gpol = store.get_record(GraphPolicy, manifest.graph_policy)
    hierarchies = [store.get_record(Hierarchy, h) for h in record.hierarchies]
    exposed = beyond = total = 0
    with MemoryLog.load(store.get(record.log)) as log:
        for i, (probe, o) in enumerate(zip(dataset.probes, outcomes.probes, strict=True)):
            if not o.cited:
                continue
            total += 1
            snap = _probe_graph(log, probe.known_at, hierarchies, gpol)
            if snap.digest != record.graphs[i]:
                raise ValueError("graph does not reproduce the run's recorded snapshot")
            trace = store.get_record(HybridTrace, o.trace)
            g = MemoryGraph(snap)
            if g.contradiction_edges(o.cited[0]):
                exposed += 1
                shown = set(trace.ranking[0].counter_evidence)
                beyond += bool(set(g.contradicting_memories(o.cited[0])) - shown)
    return exposed, beyond, total


def _probe_graph(
    log: MemoryLog, known_at: datetime, hierarchies: Sequence[Hierarchy], gpol: GraphPolicy
) -> GraphSnapshot:
    corpus = Corpus.from_log(log, known_at)
    latest = next((h for h in reversed(hierarchies) if h.at <= known_at), None)
    if latest is not None:
        corpus = corpus.with_derived(latest.memories, invalidated(latest, corpus))
    return build_graph(corpus, gpol, [latest] if latest else [])


# --- end-to-end demonstration --------------------------------------------------------------------


class Transition(Record):
    probe: str
    before: str | None
    after: str | None
    before_outcome: str
    after_outcome: str


class Demonstration(Record):
    """experiences → formation → consolidation → graph → retrieval → targeted forgetting →
    graph rebuild → interference injection → retrieval → autopsy → statistics, on one world."""

    world: str
    runs: tuple[tuple[str, Digest], ...]  # stage -> run
    graph_before: Digest
    graph_after: Digest
    graph_diff: Digest
    # (probe, answering memory, contradiction edges near it, counter-evidence retrieval
    # surfaced, contradicting memories only the graph shows)
    contradiction: tuple[str, str, int, int, int]
    forgetting_changed: tuple[Transition, ...]  # probes whose answer changed under forgetting
    interference_effect: Cell  # known.correct: consolidated+graph vs + interference
    interference_counts: tuple[int, int]  # wrong_memory + contaminated before, after
    autopsy: ProvenanceTrace
    introduced_failures: tuple[Transition, ...]  # correct -> not correct (any stage)
    reduced_failures: tuple[Transition, ...]  # not correct -> correct (any stage)
    forgetting_metrics: ForgettingMetrics
    diff: GraphDiff


def demonstration(
    store: ArtifactStore, representation: RepresentationSpec, registry: Registry = DEFAULT_REGISTRY
) -> Demonstration:
    world = next(w for w in WORLDS if w.name == "contradictory")
    dataset, bench = consolidation_world(world)
    store.put_record(dataset)
    store.put_record(bench)
    put = store.put_record
    g, hier = put(GRAPH["conservative"]), put(HIERARCHICAL)
    entity_forget = put(FORGETTING["entity"])
    graph_policy = put(POLICIES["graph-hybrid"])
    distractors = put(world_distractors(dataset))

    def run(
        name: str,
        *,
        forgetting: str | None = None,
        inject: bool = False,
        consolidated: bool = True,
    ) -> RunRecord:
        return execute(
            RunManifest(
                name=f"demonstration/{name}",
                dataset=dataset.digest,
                policy="episodic-v1",
                responder="extractive-top1-v1",
                representation=representation,
                retrieval_policy=graph_policy,
                consolidation_policy=hier if consolidated else None,
                graph_policy=g,
                forgetting_policy=forgetting,
                interventions=(InterventionSpec(name="inject", params=(("dataset", distractors),)),)
                if inject
                else (),
            ),
            store,
            registry,
        )

    plain = run("graph-l1", consolidated=False)  # the graph over raw memories only
    stages = {
        "graph": run("graph"),
        "forgetting": run("forget-entity", forgetting=entity_forget),
        "interference": run("forget-entity+interference", forgetting=entity_forget, inject=True),
    }
    evals = {k: evaluate(r.digest, store) for k, r in stages.items()}
    base = stages["graph"]
    hierarchies = [store.get_record(Hierarchy, h) for h in base.hierarchies]
    # 1. a contradiction the graph shows next to an answer, which retrieval did not surface
    #    (searched in the run over raw memories: consolidated answers cite derived memories)
    outcomes = store.get_record(RunOutcomes, plain.outcomes)
    contradiction = ("", "", 0, 0, 0)
    with MemoryLog.load(store.get(plain.log)) as log:
        for i, (probe, o) in enumerate(zip(dataset.probes, outcomes.probes, strict=True)):
            if not o.cited:
                continue
            snap = _probe_graph(log, probe.known_at, [], GRAPH["conservative"])
            assert snap.digest == plain.graphs[i]
            mg = MemoryGraph(snap)
            shown = set(store.get_record(HybridTrace, o.trace).ranking[0].counter_evidence)
            extra = set(mg.contradicting_memories(o.cited[0])) - shown
            if extra:
                edges = mg.contradiction_edges(o.cited[0])
                contradiction = (probe.id, o.cited[0], len(edges), len(shown), len(extra))
                break
    outcomes = store.get_record(RunOutcomes, base.outcomes)
    with MemoryLog.load(store.get(base.log)) as log:
        end = max(p.known_at for p in dataset.probes)
        # graph before and after the targeted forgetting, at the end of the history
        corpus = Corpus.from_log(log, end)
        h_end = hierarchies[-1]
        corpus = corpus.with_derived(h_end.memories, invalidated(h_end, corpus))
        before = build_graph(corpus, GRAPH["conservative"], [h_end])
        rec = forget(corpus, FORGETTING["entity"], end)
        after = build_graph(apply(corpus, rec), GRAPH["conservative"], [h_end])
        targets = [
            e.digest
            for e in corpus.entries
            if e.level is Level.L1
            and (
                (e.claim is not None and e.claim.key.startswith("ana."))
                or (e.claim is None and any(x.content.startswith("Ana") for x in e.sources))
            )
        ]
        fm = measure_forgetting(corpus, rec, targets)
    diff = diff_graphs(before, after)
    for x in (before, after, diff, rec, fm):
        store.put_record(x)
    # 4. autopsy of an answer given from a derived memory, back to experiences and sources
    snap_for_autopsy = None
    autopsy = None
    with MemoryLog.load(store.get(base.log)) as log:
        derived = {m.digest: m for h in hierarchies for m in h.memories}
        for probe, o in zip(dataset.probes, outcomes.probes, strict=True):
            if o.cited and o.cited[0] in derived:
                corpus = Corpus.from_log(log, probe.known_at)
                latest = next(h for h in reversed(hierarchies) if h.at <= probe.known_at)
                corpus = corpus.with_derived(latest.memories, invalidated(latest, corpus))
                trace = store.get_record(HybridTrace, o.trace)
                snap_for_autopsy = build_graph(
                    corpus, GRAPH["conservative"], [latest], [trace], base.digest
                )
                autopsy = trace_provenance(snap_for_autopsy, f"retrieval:{trace.digest}")
                break
    assert autopsy is not None
    assert snap_for_autopsy is not None
    store.put_record(snap_for_autopsy)
    store.put_record(autopsy)

    def transitions(a: Evaluation, b: Evaluation) -> list[Transition]:
        return [
            Transition(
                probe=x.probe,
                before=x.output,
                after=y.output,
                before_outcome=x.outcome.value,
                after_outcome=y.outcome.value,
            )
            for x, y in zip(a.probes, b.probes, strict=True)
            if x.output != y.output
        ]

    changed = transitions(evals["graph"], evals["forgetting"])
    both = changed + transitions(evals["forgetting"], evals["interference"])
    introduced = tuple(
        t for t in both if t.before_outcome == "correct" and t.after_outcome != "correct"
    )
    reduced = tuple(
        t for t in both if t.before_outcome != "correct" and t.after_outcome == "correct"
    )
    m = {k: e.measurement for k, e in evals.items()}
    demo = Demonstration(
        world=world.name,
        runs=(("graph-l1", plain.digest), *((k, r.digest) for k, r in stages.items())),
        graph_before=before.digest,
        graph_after=after.digest,
        graph_diff=diff.digest,
        contradiction=contradiction,
        forgetting_changed=tuple(changed),
        interference_effect=_composite(evals["forgetting"], evals["interference"]),
        interference_counts=(
            m["forgetting"]("outcome.wrong_memory").proportion.numerator
            + m["forgetting"]("outcome.contaminated").proportion.numerator,
            m["interference"]("outcome.wrong_memory").proportion.numerator
            + m["interference"]("outcome.contaminated").proportion.numerator,
        ),
        autopsy=autopsy,
        introduced_failures=introduced,
        reduced_failures=reduced,
        forgetting_metrics=fm,
        diff=diff,
    )
    store.put_record(demo)
    return demo


# --- report --------------------------------------------------------------------------------------


def _p(x: Proportion | None) -> str:
    if x is None:
        return "—"
    if x.estimate is None:
        return f"{x.numerator}/{x.denominator}"
    return f"{x.numerator}/{x.denominator}={x.estimate:.2f} [{x.low:.2f},{x.high:.2f}]"


def _c(c: Cell) -> str:
    d, lo, hi, p, under = c
    if d is None:
        return "—"
    return f"{d:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.2g}{' u' if under else ''}"


def _m(x: PairedMean) -> str:
    if x.mean is None:
        return "—"
    ci = f" [{x.low:+.2f},{x.high:+.2f}]" if x.low is not None else ""
    dz = f" dz={x.d_z:+.2f}" if x.d_z is not None else ""
    return f"{x.mean:+.2f}{ci} n={x.n} sign p={x.sign_p:.2g}{dz}"


def report(
    study: LabStudy,
    structures: Sequence[StructureResult],
    interference: InterferenceStudy | None,
    inter: Sequence[Interaction],
    demo: Demonstration | None,
) -> str:
    out = ["# Phase 8-10 report", "", f"study {study.digest}", ""]
    out += [
        "## Runs",
        "",
        "| world | condition | correct | in state | wrong memory | "
        "contaminated | integrity | stale | hidden-evidence answers | hidden share |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in study.runs:
        out.append(
            f"| {r.world} | {r.condition} | {_p(r.correct)} | {_p(r.in_state)} | "
            f"{r.wrong_memory.numerator} | {r.contaminated.numerator} | "
            f"{_p(r.metrics.integrity)} | {r.metrics.stale_answers.numerator} | "
            f"{r.metrics.hidden_evidence_answers.numerator} | "
            f"{_p(r.metrics.hidden_share)} |"
        )
    out += [
        "",
        "## Effects",
        "",
        "| world | family | baseline → treatment | known.correct | "
        "retention | key clusters | lost/gained (of retrievable) |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in study.effects:
        star = " (composite)" if e.composite else ""
        out.append(
            f"| {e.world} | {e.family} | {e.baseline} → {e.treatment}{star} | "
            f"{_c(e.known_correct)} | {_c(e.in_state)} | {_m(e.clusters)} | "
            f"{e.lost}/{e.gained} (of {e.retrievable_before}) |"
        )
    out += ["", "## Graph structure", ""]
    for s in structures:
        a, b = s.metrics_all, s.metrics_active
        out.append(
            f"- {s.world}: {a.nodes} nodes, {a.edges} edges ({dict(a.statuses)}), components "
            f"{a.components}, contradiction density {a.contradiction_density}, provenance "
            f"depth {dict(a.provenance_depth)}, unreachable {a.unreachable_evidence}, evidence "
            f"coverage {_p(a.evidence_coverage)}, unsupported claims {_p(a.unsupported_claims)} "
            f"→ active view after forgetting {_p(b.unsupported_claims)} "
            f"({b.unavailable_nodes} nodes unavailable), entity ambiguity "
            f"{_p(a.entity_ambiguity)}, orphans {_p(a.orphans)}, temporal consistency "
            f"{_p(a.temporal_consistency)}, source HHI {a.source_concentration}, diagnostics "
            f"{list(s.diagnostics)}, resolution {list(s.decisions)}, false merges "
            f"{s.false_merges}, claims unavailable {s.unavailable_claims}, growth "
            f"{[(g.nodes, g.edges) for g in s.growth]}"
        )
    if interference is not None:
        out += ["", "## Interference", ""]
        for arm in interference.arms:
            out.append(
                f"### {arm.arm}  checks {dict(arm.checks)}  false merges {list(arm.false_merges)}"
            )
            for c in arm.curves:
                pts = "; ".join(
                    f"L{p.load}: {p.target_at_1.numerator}/{p.target_at_1.denominator}"
                    f"{'*' if p.regime == 'degraded' else ''} d{p.distractor_at_1.numerator}"
                    f" i{p.interference_failures}/b{p.baseline_failures}"
                    for p in c.points
                )
                out.append(f"- {c.policy}: onset {c.onset}; {pts}")
    out += ["", "## Interactions", ""]
    for i in inter:
        out.append(f"- {i.name} ({i.world}): {dict(i.measures)} — {i.note}")
    if demo is not None:
        out += ["", "## Demonstration", "", demo.model_dump_json(indent=1)[:20000]]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m memoria.memory_lab <store dir> [world ...]``: run the matrix on the named
    worlds (default: all), the structure and interference studies, the interaction
    experiments and the demonstration; print the report."""
    args = list(sys.argv[1:] if argv is None else argv)
    store = ArtifactStore(args[0] if args else "var/artifacts")
    names = args[1:] or [w.name for w in WORLDS]
    embedder = HashedNgramEmbedder()
    representation = RepresentationSpec(embedder=embedder.spec)
    benchmarks = []
    for w in WORLDS:
        if w.name in names:
            dataset, bench = consolidation_world(w)
            store.put_record(dataset)
            benchmarks.append(store.put_record(bench))

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    spec = lab_spec(store, benchmarks, representation)
    study, performance = run_lab(spec, store, log=say)
    say(f"study {study.digest}")
    reproduced = reproduce_lab(study.digest, store)
    say(f"reproduced {reproduced.identical} {reproduced.differing}")
    structures = [structure(w, store, embedder) for w in WORLDS if w.name in names]
    istudy = run_interference(interference_spec(store, representation), store, embedder)
    say(f"interference {istudy.digest}")
    ireproduced = reproduce_interference(istudy.digest, store, embedder)
    say(f"interference reproduced {ireproduced.identical} {ireproduced.differing[:3]}")
    inter = interactions(store, embedder, study, istudy)
    demo = demonstration(store, representation)
    say(f"demonstration {demo.digest}")
    print(f"spec          {study.spec}")
    print(f"study         {study.digest}")
    print(f"performance   {performance.digest}")
    print(f"reproduced    {'identical' if reproduced.identical else reproduced.differing}")
    print(f"interference reproduced {'identical' if ireproduced.identical else 'different'}")
    print(f"structures    {[s.snapshot for s in structures]}")
    print(f"interference  {istudy.digest}")
    print(f"demonstration {demo.digest}")
    print()
    print(report(study, structures, istudy, inter, demo))


if __name__ == "__main__":
    main()
