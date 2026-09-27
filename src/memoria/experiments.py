"""Experiment runner: a manifest in, content-addressed artifacts out (constitution §8, phase 3).

:func:`execute` resolves every component a :class:`~memoria.core.RunManifest` names,
applies its interventions to the dataset, feeds the resulting stream into a fresh memory
log under the named formation policy, asks every probe, and stores each product as a
write-once artifact. The returned :class:`~memoria.core.RunRecord` names them all by
digest and is itself deterministic: executing the same manifest again yields the same
record. :func:`reproduce` checks exactly that for a stored run.

The execution environment is recorded as a separate :class:`Execution` artifact; it is
evidence about a run, not part of what determines it.
"""

from __future__ import annotations

import platform
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import version as package_version

from memoria.artifacts import ArtifactStore
from memoria.consolidation import ConsolidationPolicy, Hierarchy, consolidate, invalidated
from memoria.core import (
    Dataset,
    Digest,
    EmbedderSpec,
    InterventionRecord,
    InterventionSpec,
    InvalidTransitionError,
    ProbeOutcome,
    Record,
    RepresentationSpec,
    RetrieverSpec,
    RunManifest,
    RunOutcomes,
    RunRecord,
    Scalar,
    Step,
    StepOutcome,
)
from memoria.embeddings import Embedder, HashedNgramEmbedder
from memoria.formation import EpisodicPolicy, FormationPolicy, StatementPolicy, form
from memoria.hybrid import Corpus, Engine, HybridQuery, RetrievalPolicy
from memoria.hybrid import extractive as hybrid_extractive
from memoria.interventions import Contaminate, Delay, Drop, Inject, Intervention, Reorder
from memoria.retrieval import (
    BM25,
    EXTRACTIVE,
    Recency,
    Responder,
    Retriever,
    SemanticSignal,
    Signal,
    answer,
    extractive,
)
from memoria.semantic import SemanticIndex
from memoria.store import MemoryLog


class UnknownComponentError(LookupError):
    """A manifest names a component the registry does not provide."""


InterventionFactory = Callable[[Mapping[str, Scalar], ArtifactStore], Intervention]
EmbedderFactory = Callable[[EmbedderSpec], Embedder]


def _lookup[T](table: Mapping[str, T], name: str, kind: str) -> T:
    try:
        return table[name]
    except KeyError:
        known = ", ".join(sorted(table)) or "none"
        raise UnknownComponentError(f"unknown {kind} {name!r} (registered: {known})") from None


@dataclass(frozen=True)
class Registry:
    """Maps the names used in manifests to implementations.

    Every resolution is checked by round-trip: the built component must describe itself
    exactly as the manifest does, so a manifest can never silently mean something else.
    """

    policies: Mapping[str, Callable[[], FormationPolicy]]
    signals: Mapping[str, Callable[..., Signal]]
    responders: Mapping[str, Responder]
    interventions: Mapping[str, InterventionFactory]
    embedders: Mapping[str, EmbedderFactory] = field(default_factory=dict)

    def extend(
        self,
        *,
        policies: Mapping[str, Callable[[], FormationPolicy]] | None = None,
        signals: Mapping[str, Callable[..., Signal]] | None = None,
        responders: Mapping[str, Responder] | None = None,
        interventions: Mapping[str, InterventionFactory] | None = None,
        embedders: Mapping[str, EmbedderFactory] | None = None,
    ) -> Registry:
        """A new registry with more components. Existing names cannot be redefined."""

        def merged[T](old: Mapping[str, T], new: Mapping[str, T] | None) -> dict[str, T]:
            clash = set(old) & set(new or {})
            if clash:
                raise ValueError(f"already registered: {', '.join(sorted(clash))}")
            return {**old, **(new or {})}

        return Registry(
            policies=merged(self.policies, policies),
            signals=merged(self.signals, signals),
            responders=merged(self.responders, responders),
            interventions=merged(self.interventions, interventions),
            embedders=merged(self.embedders, embedders),
        )

    def policy(self, name: str) -> FormationPolicy:
        policy = _lookup(self.policies, name, "formation policy")()
        if policy.name != name:
            raise ValueError(f"policy registered as {name!r} calls itself {policy.name!r}")
        return policy

    def embedder(self, spec: EmbedderSpec) -> Embedder:
        """Build the embedder a spec describes. There is no fallback: an unregistered or
        unavailable embedder is an error, never a substitute."""
        built = _lookup(self.embedders, spec.name, "embedder")(spec)
        if built.spec != spec:
            raise ValueError(f"embedder {spec.name!r} does not rebuild to its manifest spec")
        return built

    def retriever(
        self, spec: RetrieverSpec, representation: RepresentationSpec | None = None
    ) -> Retriever:
        signals: list[tuple[Signal, float]] = []
        for s in spec.signals:
            if s.name == SemanticSignal.name:
                if representation is None:
                    raise ValueError("a semantic signal needs a declared representation")
                if representation.index.kind != "exact":
                    raise ValueError(
                        "retrieval scores every memory, so it uses the exact index; "
                        "approximate candidate generation belongs to hybrid retrieval"
                    )
                signals.append((SemanticSignal(self.embedder(representation.embedder)), s.weight))
                continue
            factory = _lookup(self.signals, s.name, "signal")
            try:
                signals.append((factory(**dict(s.params)), s.weight))
            except TypeError as e:
                raise ValueError(f"invalid parameters for signal {s.name!r}: {e}") from e
        retriever = Retriever(spec.name, tuple(signals), gate=spec.gate)
        if retriever.spec != spec:
            raise ValueError(f"retriever {spec.name!r} does not rebuild to its manifest spec")
        return retriever

    def responder(self, name: str) -> Responder:
        return _lookup(self.responders, name, "responder")

    def intervention(self, spec: InterventionSpec, store: ArtifactStore) -> Intervention:
        factory = _lookup(self.interventions, spec.name, "intervention")
        try:
            built = factory(dict(spec.params), store)
        except TypeError as e:
            raise ValueError(f"invalid parameters for intervention {spec.name!r}: {e}") from e
        if InterventionSpec(name=built.name, params=built.params) != spec:
            raise ValueError(f"intervention {spec.name!r} does not rebuild to its manifest spec")
        return built


def _from_params(cls: Callable[..., Intervention]) -> InterventionFactory:
    return lambda params, store: cls(**params)


def _inject(params: Mapping[str, Scalar], store: ArtifactStore) -> Intervention:
    if set(params) != {"dataset"}:
        raise TypeError("inject takes exactly one parameter, 'dataset'")
    return Inject(store.get_record(Dataset, str(params["dataset"])))


DEFAULT_REGISTRY = Registry(
    policies={EpisodicPolicy.name: EpisodicPolicy, StatementPolicy.name: StatementPolicy},
    signals={BM25.name: BM25, Recency.name: Recency},
    responders={EXTRACTIVE: extractive},
    embedders={HashedNgramEmbedder.NAME: HashedNgramEmbedder.from_spec},
    interventions={
        Drop.name: _from_params(Drop),
        Delay.name: _from_params(Delay),
        Reorder.name: _from_params(Reorder),
        Contaminate.name: _from_params(Contaminate),
        Inject.name: _inject,
    },
)


def _diff(before: Dataset, after: Dataset) -> tuple[tuple[str, ...], tuple[str, ...]]:
    old = Counter(s.digest for s in before.steps)
    new = Counter(s.digest for s in after.steps)
    return tuple(sorted((old - new).elements())), tuple(sorted((new - old).elements()))


def _ingest(log: MemoryLog, step: Step, policy: FormationPolicy) -> StepOutcome:
    try:
        decision = form(log, step.experience, policy, recorded_at=step.recorded_at)
    except InvalidTransitionError as e:  # the log refused it atomically; record why
        return StepOutcome(step=step.digest, rejected=str(e))
    return StepOutcome(step=step.digest, decision=decision.digest, reason=decision.reason)


def execute(
    manifest: RunManifest, store: ArtifactStore, registry: Registry = DEFAULT_REGISTRY
) -> RunRecord:
    """Run a manifest and store every artifact it produces. Deterministic.

    All components are resolved before any work is done. The dataset must already be
    in ``store`` (manifests reference it by digest).
    """
    policy = registry.policy(manifest.policy)
    if manifest.retriever is None:
        return _execute_hybrid(manifest, store, registry, policy)
    uses_vectors = any(s.name == SemanticSignal.name for s in manifest.retriever.signals)
    if manifest.representation is not None and not uses_vectors:
        raise ValueError("the manifest declares a representation that no signal uses")
    retriever = registry.retriever(manifest.retriever, manifest.representation)
    responder = registry.responder(manifest.responder)
    dataset = store.get_record(Dataset, manifest.dataset)
    dataset, applied = _intervene(manifest, dataset, store, registry)

    with MemoryLog(":memory:") as log:
        steps = tuple(_ingest(log, step, policy) for step in dataset.steps)
        probes = []
        for probe in dataset.probes:
            trace, response = answer(log, retriever, probe.query, responder)
            if response.responder != manifest.responder:
                raise ValueError(
                    f"responder registered as {manifest.responder!r} "
                    f"calls itself {response.responder!r}"
                )
            probes.append(
                ProbeOutcome(
                    probe=probe.id,
                    trace=trace.digest,
                    response=response.digest,
                    output=response.output,
                    cited=response.cited,
                    expected=probe.expected,
                )
            )
        log.verify()
        log_digest = store.put(log.export(), kind="MemoryLogExport")

    record = RunRecord(
        manifest=store.put_record(manifest),
        dataset=dataset.digest,
        interventions=tuple(applied),
        log=log_digest,
        outcomes=store.put_record(RunOutcomes(steps=steps, probes=tuple(probes))),
    )
    store.put_record(record)
    store.put_record(Execution.current(record))
    return record


def _intervene(
    manifest: RunManifest, dataset: Dataset, store: ArtifactStore, registry: Registry
) -> tuple[Dataset, list[str]]:
    """Apply the manifest's interventions in order, storing every derived dataset and record."""
    interventions = [registry.intervention(spec, store) for spec in manifest.interventions]
    applied = []
    for spec, intervention in zip(manifest.interventions, interventions, strict=True):
        derived = intervention.apply(dataset)
        removed, added = _diff(dataset, derived)
        applied.append(
            store.put_record(
                InterventionRecord(
                    spec=spec,
                    input=dataset.digest,
                    output=store.put_record(derived),
                    removed=removed,
                    added=added,
                )
            )
        )
        dataset = derived
    return dataset, applied


def hybrid_uses_vectors(policy: RetrievalPolicy) -> bool:
    return (
        policy.signal("semantic") is not None
        or any(g.name == "semantic" for g in policy.generators)
        or (policy.diversity is not None and policy.diversity.similarity == "embedding")
    )


def _execute_hybrid(
    manifest: RunManifest, store: ArtifactStore, registry: Registry, policy: FormationPolicy
) -> RunRecord:
    """A run whose probes are answered by a hybrid retrieval policy (schema v3), optionally
    over consolidated memory rebuilt at the consolidation policy's checkpoints.

    Traces and responses are stored as artifacts (they are not log records: a hybrid trace
    covers derived memories the log never holds). Consolidation reads the log's history at
    each checkpoint and never writes to it.
    """
    assert manifest.retrieval_policy is not None
    retrieval = store.get_record(RetrievalPolicy, manifest.retrieval_policy)
    consolidation = (
        store.get_record(ConsolidationPolicy, manifest.consolidation_policy)
        if manifest.consolidation_policy is not None
        else None
    )
    uses_vectors = hybrid_uses_vectors(retrieval) or (
        consolidation is not None and consolidation.uses_vectors
    )
    if uses_vectors != (manifest.representation is not None):
        raise ValueError(
            "a representation is declared iff a retrieval or consolidation stage uses vectors"
        )
    embedder = (
        registry.embedder(manifest.representation.embedder)
        if manifest.representation is not None
        else None
    )
    indexed = [dict(g.params)["index"] for g in retrieval.generators if g.name == "semantic"]
    needs_index = manifest.representation is not None and any(k != "scan" for k in indexed)
    if needs_index and indexed != [manifest.representation.index.kind]:  # type: ignore[union-attr]
        raise ValueError("the semantic generator's index kind must match the representation")
    if manifest.responder != EXTRACTIVE:
        raise ValueError(f"hybrid runs support the {EXTRACTIVE!r} responder only")
    dataset, applied = _intervene(
        manifest, store.get_record(Dataset, manifest.dataset), store, registry
    )

    with MemoryLog(":memory:") as log:
        steps = tuple(_ingest(log, step, policy) for step in dataset.steps)
        log.verify()
        log_digest = store.put(log.export(), kind="MemoryLogExport")
        hierarchies: list[Hierarchy] = []
        memo = (
            Engine(Corpus.from_log(log, dataset.steps[0].recorded_at), embedder)
            if (dataset.steps)
            else None
        )  # embedding memos shared by every stage of this run (one embedder)
        if consolidation is not None and dataset.steps and dataset.probes:
            start = dataset.steps[0].recorded_at
            end = max(p.known_at for p in dataset.probes)
            for t in consolidation.checkpoints(start, end):
                h = consolidate(Corpus.from_log(log, t), consolidation, t, embedder, memo)
                store.put_record(h)
                hierarchies.append(h)
        indexes: dict[str, SemanticIndex] = {}
        probes = []
        for probe in dataset.probes:
            corpus = Corpus.from_log(log, probe.known_at)
            latest = next((h for h in reversed(hierarchies) if h.at <= probe.known_at), None)
            if latest is not None:
                corpus = corpus.with_derived(latest.memories, invalidated(latest, corpus))
            index = None
            if needs_index:
                assert embedder is not None
                assert manifest.representation is not None
                key = corpus.present.digest
                if key not in indexes:
                    indexes[key] = SemanticIndex.build(
                        corpus.present, embedder, store, manifest.representation.index
                    )
                index = indexes[key]
            query = HybridQuery(
                text=probe.text,
                valid_at=probe.valid_at,
                known_at=probe.known_at,
                limit=probe.limit,
                key=probe.key,
            )
            vectors = memo.vectors if memo is not None else {}
            similarities = memo.similarities if memo is not None else {}
            engine = Engine(corpus, embedder, index, vectors, similarities)
            trace = engine.retrieve(retrieval, query)
            response = hybrid_extractive(trace, corpus)
            store.put_record(trace)
            store.put_record(response)
            probes.append(
                ProbeOutcome(
                    probe=probe.id,
                    trace=trace.digest,
                    response=response.digest,
                    output=response.output,
                    cited=response.cited,
                    expected=probe.expected,
                )
            )
    record = RunRecord(
        manifest=store.put_record(manifest),
        dataset=dataset.digest,
        interventions=tuple(applied),
        log=log_digest,
        outcomes=store.put_record(RunOutcomes(steps=steps, probes=tuple(probes))),
        hierarchies=tuple(h.digest for h in hierarchies),
    )
    store.put_record(record)
    store.put_record(Execution.current(record))
    return record


class Execution(Record):
    """Where and with what a run was executed. Evidence, not part of the run's identity."""

    run: Digest
    memoria: str
    python: str
    implementation: str
    platform: str
    pydantic: str
    sqlite: str

    @classmethod
    def current(cls, run: RunRecord) -> Execution:
        return cls(
            run=run.digest,
            memoria=package_version("memoria"),
            python=platform.python_version(),
            implementation=platform.python_implementation(),
            platform=f"{sys.platform}-{platform.machine()}",
            pydantic=package_version("pydantic"),
            sqlite=sqlite3.sqlite_version,
        )


class Reproduction(Record):
    """The result of re-executing a stored run's manifest and comparing every artifact."""

    original: Digest
    rerun: Digest
    differences: tuple[str, ...]  # RunRecord fields whose digests differ

    @property
    def reproduced(self) -> bool:
        return not self.differences


def reproduce(
    run: str, store: ArtifactStore, registry: Registry = DEFAULT_REGISTRY
) -> Reproduction:
    """Re-execute a stored run's manifest and compare all artifacts by digest.

    Differing artifacts are stored alongside the originals (nothing is overwritten), so
    a failed reproduction leaves both versions available for inspection.
    """
    original = store.get_record(RunRecord, run)
    rerun = execute(store.get_record(RunManifest, original.manifest), store, registry)
    differences = tuple(
        f for f in RunRecord.model_fields if getattr(original, f) != getattr(rerun, f)
    )
    result = Reproduction(original=original.digest, rerun=rerun.digest, differences=differences)
    store.put_record(result)
    return result
