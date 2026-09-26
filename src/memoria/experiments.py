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
from dataclasses import dataclass
from importlib.metadata import version as package_version

from memoria.artifacts import ArtifactStore
from memoria.core import (
    Dataset,
    Digest,
    InterventionRecord,
    InterventionSpec,
    InvalidTransitionError,
    ProbeOutcome,
    Record,
    RetrieverSpec,
    RunManifest,
    RunOutcomes,
    RunRecord,
    Scalar,
    Step,
    StepOutcome,
)
from memoria.formation import EpisodicPolicy, FormationPolicy, StatementPolicy, form
from memoria.interventions import Contaminate, Delay, Drop, Inject, Intervention, Reorder
from memoria.retrieval import (
    BM25,
    EXTRACTIVE,
    Recency,
    Responder,
    Retriever,
    Signal,
    answer,
    extractive,
)
from memoria.store import MemoryLog


class UnknownComponentError(LookupError):
    """A manifest names a component the registry does not provide."""


InterventionFactory = Callable[[Mapping[str, Scalar], ArtifactStore], Intervention]


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

    def extend(
        self,
        *,
        policies: Mapping[str, Callable[[], FormationPolicy]] | None = None,
        signals: Mapping[str, Callable[..., Signal]] | None = None,
        responders: Mapping[str, Responder] | None = None,
        interventions: Mapping[str, InterventionFactory] | None = None,
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
        )

    def policy(self, name: str) -> FormationPolicy:
        policy = _lookup(self.policies, name, "formation policy")()
        if policy.name != name:
            raise ValueError(f"policy registered as {name!r} calls itself {policy.name!r}")
        return policy

    def retriever(self, spec: RetrieverSpec) -> Retriever:
        signals = []
        for s in spec.signals:
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
    retriever = registry.retriever(manifest.retriever)
    responder = registry.responder(manifest.responder)
    dataset = store.get_record(Dataset, manifest.dataset)
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
