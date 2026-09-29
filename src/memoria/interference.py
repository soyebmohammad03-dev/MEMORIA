"""The interference laboratory (Phase 10): controlled memory competition with exact truth.

A *scenario* is one target fact (``set <t>.home = V`` from a reliable source), one probe
asking for it, and a deterministic, ordered sequence of *distractors* produced by one
mechanism. Loads are prefixes of that sequence, so a replicate's population only grows:
the load-0 population is the **controlled baseline** of every comparison. A retrieval
failure counts as *interference* only relative to that baseline, on the same replicate:
the target was retrieved at load 0, is not at load L, and a labelled distractor took its
place. Failures already present at load 0 are retrieval failures, not interference.

Mechanisms (what every distractor shares with the target):

================  =========================================================================
proactive         same key, another value, occurred *before* the target (older memories)
retroactive       same key, another value, occurred *after* the probe's valid time (newer)
temporal          same key, other values, in periods before and after the target's
semantic          a near-collision entity (same first two syllables), same attribute and
                  wording: near-duplicate text about someone else
entity            the same entity, another attribute
contradiction     same key, another value, the same instant, unreliable source
consolidation     the semantic population, consolidated (derived memories compete)
retrieval         a mixture (semantic, entity, proactive, contradiction) competing for top-k
================  =========================================================================

Modifiers: ``frequency`` (copies of each distractor), ``wording`` (statements in the
target's template, or free-text paraphrases), ``recency`` (semantic/entity distractors
occurring before the target or between it and the probe) and ``source`` (the distractors'
source class). Each generated experience is labelled with its role, mechanism, entity,
key and value, so every observation is judged against exact ground truth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.comparison import normalize
from memoria.consolidation import ConsolidationPolicy, consolidate
from memoria.core import (
    Dataset,
    DerivedMemory,
    Digest,
    Experience,
    Level,
    Probe,
    Record,
    Step,
    quantize,
    unit_interval,
)
from memoria.embeddings import Embedder, cosine, embed
from memoria.entities import ResolutionPolicy, resolve, surface_names
from memoria.formation import EpisodicPolicy, form
from memoria.hybrid import Corpus, Engine, Entry, HybridQuery, HybridTrace, RetrievalPolicy
from memoria.scenarios import day, known
from memoria.statistics import (
    PairedMean,
    Proportion,
    holm,
    mcnemar_exact,
    min_achievable_p,
    newcombe_paired,
    paired_mean,
    significant,
)
from memoria.store import MemoryLog

Mechanism = Literal[
    "consolidation",
    "contradiction",
    "entity",
    "proactive",
    "retrieval",
    "retroactive",
    "semantic",
    "temporal",
]
MECHANISMS: tuple[Mechanism, ...] = (
    "proactive",
    "retroactive",
    "semantic",
    "entity",
    "temporal",
    "contradiction",
    "consolidation",
    "retrieval",
)

_SYLLABLES = ("ka", "lo", "mi", "ra", "te", "vo", "zu", "ne", "sa", "di", "po", "fe")
CITIES = ("Paris", "Berlin", "Munich", "Rome", "Oslo", "Lisbon", "Vienna", "Prague")
ATTRIBUTES = (
    "employer",
    "car",
    "pet",
    "team",
    "school",
    "bank",
    "doctor",
    "gym",
    "hobby",
    "phone",
    "club",
    "dentist",
    "airline",
    "language",
    "instrument",
    "sport",
    "coffee",
    "cinema",
    "library",
    "market",
    "bakery",
    "garage",
    "tailor",
    "florist",
)
WORDS = ("Acme", "Globex", "Initech", "Hooli", "Umbrella", "Vandelay", "Stark", "Wayne")
TARGET_DAY, VALID_DAY, KNOWN_DAY = 100.0, 150.0, 300.0


class InterferenceSpec(Record):
    """One replicate of one mechanism at one load. Every choice is seeded (I21)."""

    name: str = Field(min_length=1)
    seed: int
    mechanism: Mechanism
    replicate: int = Field(ge=0)
    load: int = Field(ge=0, le=256)
    frequency: int = Field(default=1, ge=1, le=8)
    wording: Literal["template", "paraphrase"] = "template"
    recency: Literal["older", "newer"] = "older"
    source: str = Field(default="forum", pattern=r"^[a-z]+$")

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if self.mechanism == "consolidation" and self.wording != "paraphrase":
            raise ValueError("the consolidation mechanism consolidates free-text paraphrases")
        return self


class Label(Record):
    """Ground truth for one generated experience (by its source id)."""

    source: str
    role: Literal["target", "distractor"]
    mechanism: Mechanism | None
    entity: str
    key: str
    value: str  # normalised tokens joined by spaces


class Scenario(Record):
    """A generated population and its exact ground truth."""

    spec: InterferenceSpec
    dataset: Digest
    target_key: str
    target_value: str
    target_entity: str
    probe: str
    labels: tuple[Label, ...]  # by source


def _name(seed: int, *labels: str | int) -> str:
    parts = [_SYLLABLES[int(unit_interval(seed, *labels, k) * len(_SYLLABLES))] for k in range(3)]
    return "".join(parts).capitalize()


def _other(options: Sequence[str], avoid: str, seed: int, *labels: str | int) -> str:
    choices = [o for o in options if o != avoid]
    return choices[int(unit_interval(seed, *labels) * len(choices))]


def scenario(spec: InterferenceSpec) -> tuple[Dataset, Scenario]:
    """The population of ``spec``: the target, then the first ``load`` distractors of the
    replicate's sequence (each ``frequency`` times). Deterministic in ``spec``."""
    s, r = spec.seed, spec.replicate
    target = _name(s, spec.name, "target", r)
    t = target.lower()
    value = CITIES[int(unit_interval(s, spec.name, "value", r) * len(CITIES))]
    key = f"{t}.home"
    steps: list[Step] = []
    labels: list[Label] = []

    def emit(occurred: float, content: str, source: str, label: Label) -> None:
        steps.append(
            Step(
                experience=Experience(source=source, content=content, occurred_at=day(occurred)),
                recorded_at=day(occurred),
            )
        )
        labels.append(label)

    tsource = f"clinic:target-{spec.name}-{r}"
    emit(
        TARGET_DAY,
        f"set {key} = {value}",
        tsource,
        Label(
            source=tsource,
            role="target",
            mechanism=None,
            entity=t,
            key=key,
            value=" ".join(normalize(value)),
        ),
    )
    others: list[str] = []
    for i in range(spec.load):
        mech = spec.mechanism
        if mech == "retrieval":
            mech = ("semantic", "entity", "proactive", "contradiction")[i % 4]
        j = i if spec.mechanism != "retrieval" else i // 4
        when_older = TARGET_DAY - 0.5 * (j + 1)
        when_newer = TARGET_DAY + 1 + (VALID_DAY - TARGET_DAY - 2) * (j + 1) / (spec.load + 1)
        if mech in ("semantic", "consolidation"):
            # a near-collision: another entity sharing the target's first two syllables
            other = target[:4] + _SYLLABLES[int(unit_interval(s, spec.name, "o", r, j) * 12)]
            k = 0
            while other == target or other in others:
                k += 1
                other = _name(s, spec.name, "other", r, j, k)
            others.append(other)
            v = CITIES[int(unit_interval(s, spec.name, "other-value", r, j) * len(CITIES))]
            ekey, ent = f"{other.lower()}.home", other.lower()
            text = f"set {ekey} = {v}" if spec.wording == "template" else f"{other} lives in {v}."
            when = when_older if spec.recency == "older" else when_newer
        elif mech == "entity":
            attr = ATTRIBUTES[j % len(ATTRIBUTES)] + ("" if j < len(ATTRIBUTES) else str(j))
            v = WORDS[int(unit_interval(s, spec.name, "entity-value", r, j) * len(WORDS))]
            ekey, ent = f"{t}.{attr}", t
            text = (
                f"set {ekey} = {v}" if spec.wording == "template" else f"{target}'s {attr} is {v}."
            )
            when = when_older if spec.recency == "older" else when_newer
        else:
            v = _other(CITIES, value, s, spec.name, "rival", r, j)
            ekey, ent = key, t
            text = f"set {key} = {v}" if spec.wording == "template" else f"{target} lives in {v}."
            if mech == "proactive":
                when = when_older
            elif mech == "retroactive":
                when = VALID_DAY + 1 + 0.5 * j
            elif mech == "temporal":
                when = (
                    TARGET_DAY - 0.5 * (j // 2 + 1)
                    if j % 2 == 0
                    else VALID_DAY + 1 + 0.5 * (j // 2)
                )
            else:  # contradiction: the same instant as the target
                when = TARGET_DAY
        for copy in range(spec.frequency):
            src = f"{spec.source}:d-{spec.name}-{r}-{i}-{copy}"
            emit(
                when,
                text,
                src,
                Label(
                    source=src,
                    role="distractor",
                    mechanism=mech,
                    entity=ent,
                    key=ekey,
                    value=" ".join(normalize(v)),
                ),
            )
    steps.sort(key=lambda st: (st.recorded_at, st.experience.source))
    probe = Probe(
        id=f"{spec.name}-{r}",
        text=f"Where does {target} live?",
        valid_at=day(VALID_DAY if spec.mechanism != "retroactive" else VALID_DAY - 30),
        known_at=day(KNOWN_DAY),
        expected=known(value),
        key=key,
    )
    dataset = Dataset(
        name=f"interference:{spec.mechanism}",
        version=spec.digest,
        steps=tuple(steps),
        probes=(probe,),
    )
    return dataset, Scenario(
        spec=spec,
        dataset=dataset.digest,
        target_key=key,
        target_value=" ".join(normalize(value)),
        target_entity=t,
        probe=probe.id,
        labels=tuple(sorted(labels, key=lambda x: x.source)),
    )


# --- observing one retrieval --------------------------------------------------------------------

Role = Literal["target", "target-derived", "distractor", "distractor-derived", "other"]


class Observation(Record):
    """One retrieval of the target under one condition, judged against ground truth."""

    mechanism: Mechanism
    policy: str
    replicate: int
    load: int
    trace: Digest  # recomputed by replay, not stored (a pure function of the inputs)
    corpus: Digest
    target_rank: int | None  # best-ranked target (or target-equivalent derived) memory
    source_rank: int | None  # the L1 target itself
    target_excluded: bool  # the target was a candidate but a hard filter removed it
    top: tuple[Role, ...]  # roles of the top-k results
    top1_entity: str | None
    top1_key: str | None
    top1_value: str | None
    rival_values_in_top: int  # distinct other values for the target key in the top-k
    entropy: float | None  # of top-k final scores normalised to a distribution (all > 0)
    derived_merges_across: int  # derived memories covering the target and a distractor
    temporal_confused: bool  # top-1 asserts another value for the target key
    entity_confused: bool  # top-1 is about another entity
    provenance_confused: bool  # top-1 states the target value but is not the target's evidence

    @property
    def hit(self) -> bool:
        return self.target_rank == 1

    @property
    def distractor_first(self) -> bool:
        return self.top[:1] in (("distractor",), ("distractor-derived",))


def _labels_of(e: Entry, by_source: dict[str, Label]) -> list[Label]:
    return [by_source[x.source] for x in e.sources if x.source in by_source]


def _role(e: Entry, sc: Scenario, by_source: dict[str, Label]) -> Role:
    ls = _labels_of(e, by_source)
    if e.level is Level.L1:
        return ls[0].role if ls else "other"
    has_target = any(x.role == "target" for x in ls)
    m = e.version
    asserts_target = (
        isinstance(m, DerivedMemory) and m.key == sc.target_key and m.value == sc.target_value
    )
    return "target-derived" if has_target and asserts_target else "distractor-derived"


def observe(trace: HybridTrace, corpus: Corpus, sc: Scenario, policy: str, k: int) -> Observation:
    by_source = {x.source: x for x in sc.labels}
    by_digest = {e.digest: e for e in corpus.entries}
    ranking = [by_digest[r.version] for r in trace.ranking]
    roles = [_role(e, sc, by_source) for e in ranking]
    target_rank = next(
        (i for i, x in enumerate(roles, 1) if x in ("target", "target-derived")), None
    )
    source_rank = next((i for i, x in enumerate(roles, 1) if x == "target"), None)
    excluded = any(
        c.excluded is not None
        and any(
            by_source.get(x.source) is not None and by_source[x.source].role == "target"
            for x in by_digest[c.version].sources
        )
        and by_digest[c.version].level is Level.L1
        for c in trace.candidates
    )
    top1 = ranking[0] if ranking else None
    ent = key = val = None
    if top1 is not None:
        ls = _labels_of(top1, by_source)
        m = top1.version
        if isinstance(m, DerivedMemory):
            key = m.key
            val = m.value
            ents = {x.entity for x in ls}
            ent = ents.pop() if len(ents) == 1 else None
        elif ls:
            ent, key, val = ls[0].entity, ls[0].key, ls[0].value
    rivals = set()
    for e in ranking[:k]:
        for x in _labels_of(e, by_source):
            if x.key == sc.target_key and x.value != sc.target_value and e.level is Level.L1:
                rivals.add(x.value)
    finals = [r.final for r in trace.ranking[:k]]
    entropy = None
    if len(finals) > 1 and all(f > 0 for f in finals):
        total = math.fsum(finals)
        entropy = quantize(-math.fsum(f / total * math.log(f / total) for f in finals))
    across = 0
    for e in corpus.entries:
        if e.level is not Level.L1:
            facts = {(x.entity, x.key, x.value) for x in _labels_of(e, by_source)}
            if any(x.role == "target" for x in _labels_of(e, by_source)) and len(facts) > 1:
                across += 1
    targetish = bool(roles) and roles[0] in ("target", "target-derived")
    return Observation(
        mechanism=sc.spec.mechanism,
        policy=policy,
        replicate=sc.spec.replicate,
        load=sc.spec.load,
        trace=trace.digest,
        corpus=corpus.digest,
        target_rank=target_rank,
        source_rank=source_rank,
        target_excluded=excluded,
        top=tuple(roles[:k]),
        top1_entity=ent,
        top1_key=key,
        top1_value=val,
        rival_values_in_top=len(rivals),
        entropy=entropy,
        derived_merges_across=across,
        temporal_confused=key == sc.target_key and val is not None and val != sc.target_value,
        entity_confused=ent is not None and ent != sc.target_entity,
        provenance_confused=not targetish and val == sc.target_value,
    )


def run_scenario(
    dataset: Dataset,
    sc: Scenario,
    policies: Sequence[tuple[str, RetrievalPolicy]],
    embedder: Embedder,
    consolidation: ConsolidationPolicy | None,
    graph: object | None = None,
    k: int = 5,
) -> list[Observation]:
    """Ingest the population (episodic formation), consolidate if the mechanism is
    consolidation, and retrieve the probe under each policy. ``graph`` builds a graph
    view for policies that read one: a callable (corpus, hierarchies) -> GraphView."""
    probe = dataset.probes[0]
    with MemoryLog(":memory:") as log:
        for st in dataset.steps:
            form(log, st.experience, EpisodicPolicy(), recorded_at=st.recorded_at)
        corpus = Corpus.from_log(log, probe.known_at)
    hierarchies = []
    if consolidation is not None and sc.spec.mechanism == "consolidation":
        h = consolidate(corpus, consolidation, probe.known_at, embedder)
        hierarchies.append(h)
        corpus = corpus.with_derived(h.memories, ())
    query = HybridQuery(
        text=probe.text, valid_at=probe.valid_at, known_at=probe.known_at, limit=k, key=probe.key
    )
    out = []
    view = None
    for name, pol in policies:
        if pol.uses_graph and view is None:
            assert callable(graph)
            view = graph(corpus, hierarchies)
        engine = Engine(corpus, embedder, graph=view if pol.uses_graph else None)
        trace = engine.retrieve(pol, query)
        out.append(observe(trace, corpus, sc, name, k))
    return out


def false_merges(dataset: Dataset, sc: Scenario, policy: ResolutionPolicy) -> tuple[int, int]:
    """Entity resolution over the population's names: (accepted merges joining the target
    with another true entity, accepted merges involving the target)."""
    by_source = {x.source: x for x in sc.labels}
    structured: dict[str, set[str]] = {}
    surface: dict[str, set[str]] = {}
    truth: dict[str, str] = {}
    for st in dataset.steps:
        x = st.experience
        lab = by_source[x.source]
        structured.setdefault(lab.key.split(".")[0], set()).add(x.digest)
        truth[f"structured:{lab.key.split('.')[0]}"] = lab.entity
        for n in surface_names(x.content):
            surface.setdefault(n, set()).add(x.digest)
            truth[f"surface:{n.casefold()}"] = lab.entity
    r = resolve(structured, surface, policy)
    target = {m for m, e in truth.items() if e == sc.target_entity}
    merged = [d for d in r.decisions if d.accepted and (d.a in target or d.b in target)]
    wrong = sum(1 for d in merged if truth.get(d.a) != truth.get(d.b))
    return wrong, len(merged)


# --- analysis ------------------------------------------------------------------------------------


class LoadPoint(Record):
    """One (mechanism, policy, load) cell over the replicates, paired against load 0."""

    load: int
    n: int
    target_at_1: Proportion
    target_at_k: Proportion
    distractor_at_1: Proportion
    temporal_confusion: Proportion  # top-1 is the target key with another value
    entity_confusion: Proportion  # top-1 is about another entity
    provenance_confusion: Proportion  # top-1 states the target value, not from the target
    contradiction_exposure: Proportion  # a rival value for the target key in the top-k
    excluded: Proportion  # the target was removed by a hard filter
    rank_displacement: PairedMean  # target rank at L - at 0 (both ranked)
    rr_degradation: PairedMean  # reciprocal rank at 0 - at L (unranked = 0)
    entropy: float | None  # mean over replicates where defined
    # target@1 against load 0 on the same replicates: Newcombe difference, exact McNemar
    difference: tuple[float, float, float] | None
    p_value: float
    holm_p: float
    underpowered: bool
    # RQ-I3 decomposition over replicates
    baseline_failures: int  # not retrieved at load 0 (retrieval failure, not interference)
    interference_failures: int  # retrieved at 0, displaced at L by a labelled distractor
    other_failures: int  # retrieved at 0, not at L, and no distractor at rank 1
    regime: Literal["baseline", "no detectable change", "degraded", "improved"]


class LoadCurve(Record):
    mechanism: Mechanism
    policy: str
    points: tuple[LoadPoint, ...]
    onset: int | None  # smallest load significant after Holm and not underpowered


def load_curve(
    observations: Sequence[Observation],
    mechanism: str,
    policy: str,
    confidence: float = 0.95,
    alpha: float = 0.05,
) -> LoadCurve:
    """Aggregate one mechanism and policy over replicates. Replicates are independent
    populations (independent seeds), so pairs are independent units; loads within a
    replicate are nested and each is compared only with load 0 of the same replicate."""
    obs = [o for o in observations if o.mechanism == mechanism and o.policy == policy]
    by = {(o.replicate, o.load): o for o in obs}
    loads = sorted({o.load for o in obs})
    reps = sorted({o.replicate for o in obs})
    p = Proportion.of

    def pairs(load: int) -> list[tuple[Observation, Observation]]:
        return [(by[(r, 0)], by[(r, load)]) for r in reps if (r, 0) in by and (r, load) in by]

    def discordant(load: int) -> tuple[int, int]:
        ps = pairs(load)
        return sum(a.hit and not b.hit for a, b in ps), sum(b.hit and not a.hit for a, b in ps)

    tested = [x for x in loads if x != 0]
    raw = [significant(mcnemar_exact(*discordant(x))) for x in tested]
    holm_p = dict(zip(tested, (significant(q) for q in holm(raw)), strict=True))
    points = []
    onset = None
    for load in loads:
        cur = [by[(r, load)] for r in reps if (r, load) in by]
        ps = pairs(load)
        n = len(cur)
        f, g = discordant(load)
        e = sum(a.hit and b.hit for a, b in ps)
        diff = newcombe_paired(e, g, f, len(ps) - e - f - g, confidence)  # (L) - (0)
        under = min_achievable_p(f + g) > alpha
        q = holm_p.get(load, 1.0)
        interference = sum(a.hit and not b.hit and b.distractor_first for a, b in ps)
        changed = load != 0 and q < alpha and not under
        regime: Literal["baseline", "no detectable change", "degraded", "improved"] = (
            "baseline"
            if load == 0
            else "no detectable change"
            if not changed
            else "degraded"
            if diff is not None and diff[0] < 0
            else "improved"
        )
        if regime == "degraded" and onset is None:
            onset = load
        ents = [o.entropy for o in cur if o.entropy is not None]
        points.append(
            LoadPoint(
                load=load,
                n=n,
                target_at_1=p(sum(o.hit for o in cur), n, confidence),
                target_at_k=p(
                    sum(o.target_rank is not None and o.target_rank <= len(o.top) for o in cur),
                    n,
                    confidence,
                ),
                distractor_at_1=p(sum(o.distractor_first for o in cur), n, confidence),
                temporal_confusion=p(sum(o.temporal_confused for o in cur), n, confidence),
                entity_confusion=p(sum(o.entity_confused for o in cur), n, confidence),
                provenance_confusion=p(sum(o.provenance_confused for o in cur), n, confidence),
                contradiction_exposure=p(
                    sum(o.rival_values_in_top > 0 for o in cur), n, confidence
                ),
                excluded=p(sum(o.target_excluded for o in cur), n, confidence),
                rank_displacement=paired_mean(
                    [
                        float(b.target_rank - a.target_rank)
                        for a, b in ps
                        if a.target_rank is not None and b.target_rank is not None
                    ],
                    confidence,
                    alpha,
                ),
                rr_degradation=paired_mean(
                    [
                        (1 / a.target_rank if a.target_rank else 0.0)
                        - (1 / b.target_rank if b.target_rank else 0.0)
                        for a, b in ps
                    ],
                    confidence,
                    alpha,
                ),
                entropy=quantize(math.fsum(ents) / len(ents)) if ents else None,
                difference=(quantize(diff[0]), quantize(diff[1]), quantize(diff[2]))
                if diff
                else None,
                p_value=significant(mcnemar_exact(f, g)),
                holm_p=q,
                underpowered=under,
                baseline_failures=sum(not a.hit for a, _ in ps),
                interference_failures=interference,
                other_failures=f - interference,
                regime=regime,
            )
        )
    return LoadCurve(mechanism=mechanism, policy=policy, points=tuple(points), onset=onset)  # type: ignore[arg-type]


def manipulation(sc: Scenario, dataset: Dataset, embedder: Embedder) -> dict[str, float]:
    """How much a population's distractors share with the target, per property: the check
    that each generator parameter moves the property it is meant to move."""
    d = [x for x in sc.labels if x.role == "distractor"]
    if not d:
        return {}
    texts = {st.experience.source: st.experience.content for st in dataset.steps}
    times = {st.experience.source: st.experience.occurred_at for st in dataset.steps}
    target = next(x for x in sc.labels if x.role == "target")
    probe = dataset.probes[0]
    vecs = embed(embedder, [probe.text, *[texts[x.source] for x in d]])
    sims = [cosine(vecs[0], v) for v in vecs[1:]]
    n = len(d)
    return {
        "same_entity": quantize(sum(x.entity == sc.target_entity for x in d) / n),
        "same_key": quantize(sum(x.key == sc.target_key for x in d) / n),
        "same_value": quantize(sum(x.value == sc.target_value for x in d) / n),
        "same_instant": quantize(sum(times[x.source] == times[target.source] for x in d) / n),
        "after_valid_time": quantize(sum(times[x.source] > probe.valid_at for x in d) / n),
        "newer_than_target": quantize(sum(times[x.source] > times[target.source] for x in d) / n),
        "mean_query_cosine": quantize(math.fsum(sims) / n),
        "copies": quantize(n / max(1, len({x.source.rsplit("-", 1)[0] for x in d}))),
    }
