"""The experiment runner. Phase 3 exit criterion: re-running a manifest reproduces its artifacts."""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import pytest

from memoria.artifacts import ArtifactIntegrityError, ArtifactStore
from memoria.core import (
    Dataset,
    Experience,
    FormationDecision,
    FormationReason,
    InterventionRecord,
    InterventionSpec,
    MemoryVersion,
    Operation,
    Query,
    RetrievalTrace,
    RetrieverSpec,
    RunManifest,
    RunOutcomes,
    RunRecord,
    Scalar,
    SignalEvidence,
    SignalSpec,
    Step,
)
from memoria.experiments import (
    DEFAULT_REGISTRY,
    Execution,
    UnknownComponentError,
    execute,
    reproduce,
)
from memoria.formation import HistoryView, Proposal
from memoria.retrieval import EXTRACTIVE, lexical, lexical_recency
from memoria.scenarios import day, drifting_facts, relocation_year
from memoria.store import MemoryLog

RELOCATION = relocation_year()
DRIFT = drifting_facts(seed=11, keys=5, changes=6, probes_per_key=8)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    s = ArtifactStore(tmp_path / "var")
    s.put_record(RELOCATION)
    s.put_record(DRIFT)
    return s


def manifest(
    dataset: Dataset = RELOCATION,
    *interventions: InterventionSpec,
    policy: str = "statement-v1",
    retriever: RetrieverSpec | None = None,
    responder: str = EXTRACTIVE,
) -> RunManifest:
    return RunManifest(
        name="test",
        dataset=dataset.digest,
        interventions=interventions,
        policy=policy,
        retriever=retriever or lexical_recency().spec,
        responder=responder,
    )


def iv(name: str, **params: Scalar) -> InterventionSpec:
    return InterventionSpec(name=name, params=tuple(params.items()))


CONTAMINATE = iv("contaminate", rate=0.5, seed=2, delay_days=1, source="contaminant")
ADVERSE = (iv("drop", rate=0.2, seed=1), iv("delay", rate=0.3, seed=1, days=12), CONTAMINATE)


# --- the exit criterion ----------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["statement-v1", "episodic-v1"])
@pytest.mark.parametrize("dataset", [RELOCATION, DRIFT], ids=["relocation", "drift"])
def test_rerunning_a_manifest_reproduces_its_artifacts(
    tmp_path: Path, dataset: Dataset, policy: str
) -> None:
    m = manifest(dataset, *ADVERSE, policy=policy)
    stores = [ArtifactStore(tmp_path / name) for name in ("a", "b")]
    records = []
    for s in stores:
        s.put_record(dataset)
        records.append(execute(m, s))
    assert records[0] == records[1]
    first = records[0]
    artifacts = [first.dataset, first.log, first.outcomes, *first.interventions, first.digest]
    for digest in artifacts:  # byte-identical in independent stores
        assert stores[0].get(digest) == stores[1].get(digest)
    result = reproduce(first.digest, stores[0])
    assert result.reproduced
    assert result.rerun == first.digest


def test_run_id_is_the_manifest_digest(store: ArtifactStore) -> None:
    m = manifest()
    record = execute(m, store)
    assert record.manifest == m.digest
    assert store.get_record(RunManifest, m.digest) == m
    different = manifest(retriever=lexical().spec)
    assert execute(different, store).manifest != record.manifest


# Pinned run digests: a change means stored results would differ (see test_scenario.GOLDEN).
GOLDEN = {
    "baseline": "sha256:94f401c516a2f8094212f5b5bda34e86a445fcea273cdfe7ceaf14f137f3e969",
    "adverse": "sha256:a01ccb9e075af274ab49dbed697f423889aff6810d0458fdf6f1f221f1a8ff7f",
}


@pytest.mark.parametrize("case", ["baseline", "adverse"])
def test_run_matches_golden_digest(store: ArtifactStore, case: str) -> None:
    m = manifest(DRIFT, *(ADVERSE if case == "adverse" else ()))
    assert execute(m, store).digest == GOLDEN[case]


# --- what a run records ------------------------------------------------------------------------


def test_outcomes_record_every_step_and_probe(store: ArtifactStore) -> None:
    record = execute(manifest(), store)
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    assert [s.step for s in outcomes.steps] == [s.digest for s in RELOCATION.steps]
    assert [o.probe for o in outcomes.probes] == [p.id for p in RELOCATION.probes]
    assert [o.expected for o in outcomes.probes] == [p.expected for p in RELOCATION.probes]
    reasons = [s.reason for s in outcomes.steps]
    assert reasons.count(FormationReason.CONFLICTING) == 1
    assert reasons.count(FormationReason.DUPLICATE_EXPERIENCE) == 1
    assert all(s.rejected is None for s in outcomes.steps)


def test_outcomes_are_backed_by_the_stored_log(store: ArtifactStore) -> None:
    record = execute(manifest(RELOCATION, CONTAMINATE), store)
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    with MemoryLog.load(store.get(record.log)) as log:
        log.verify()
        decisions = {d.digest: d for d in log.records(FormationDecision)}
        for step in outcomes.steps:
            assert step.decision is not None
            assert decisions[step.decision].reason is step.reason
        for probe in outcomes.probes:
            trace = log.record(RetrievalTrace, probe.trace)
            assert trace is not None
            assert trace.query == next(p.query for p in RELOCATION.probes if p.id == probe.probe)
            for cited in probe.cited:
                assert log.version(cited) is not None


def test_intervention_records_form_a_verifiable_chain(store: ArtifactStore) -> None:
    record = execute(manifest(DRIFT, *ADVERSE), store)
    chain = [store.get_record(InterventionRecord, d) for d in record.interventions]
    assert [r.spec for r in chain] == list(ADVERSE)
    assert chain[0].input == DRIFT.digest
    for earlier, later in itertools.pairwise(chain):
        assert later.input == earlier.output
    assert chain[-1].output == record.dataset
    for r in chain:
        before = store.get_record(Dataset, r.input)
        after = store.get_record(Dataset, r.output)
        assert sorted({s.digest for s in before.steps} - {s.digest for s in after.steps}) == sorted(
            set(r.removed)
        )
        assert set(r.added) == {s.digest for s in after.steps} - {s.digest for s in before.steps}
    assert chain[0].added == ()  # drop only removes
    assert chain[2].removed == ()  # contaminate only adds


def test_contamination_is_traceable_into_answers(store: ArtifactStore) -> None:
    """A wrong answer can be followed back to the contaminant experience that caused it."""
    record = execute(
        manifest(DRIFT, iv("contaminate", rate=1, seed=0, delay_days=1, source="contaminant")),
        store,
    )
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    contaminated = [p for p in outcomes.probes if p.output and "(contaminated)" in p.output]
    assert contaminated
    with MemoryLog.load(store.get(record.log)) as log:
        for probe in contaminated:
            version = log.version(probe.cited[0])
            assert version is not None
            (source,) = version.derived_from
            experience = log.experience(source)
            assert experience is not None
            assert experience.source.startswith("contaminant:sha256:")
            original = experience.source.removeprefix("contaminant:")
            assert any(s.experience.digest == original for s in DRIFT.steps)


def test_execution_environment_is_recorded_separately(store: ArtifactStore) -> None:
    record = execute(manifest(), store)
    (digest,) = store.digests("Execution")
    execution = store.get_record(Execution, digest)
    assert execution.run == record.digest
    assert execution.python
    assert execution.sqlite
    # Evidence, not identity: no environment detail is part of the deterministic record.
    assert set(RunRecord.model_fields) == {
        "manifest",
        "dataset",
        "interventions",
        "log",
        "outcomes",
        "hierarchies",  # schema v3: consolidation artifacts, deterministic (not environment)
    }


def test_empty_dataset_runs(store: ArtifactStore) -> None:
    empty = Dataset(name="empty", version="1", steps=())
    store.put_record(empty)
    record = execute(manifest(empty), store)
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    assert (outcomes.steps, outcomes.probes) == ((), ())
    assert store.get(record.log) == b""


# --- rejections and failures are explicit ----------------------------------------------------


@dataclass(frozen=True)
class Collider:
    """An adversarial policy: every experience creates the same memory."""

    name: ClassVar[str] = "collider-v1"

    def decide(
        self, experience: Experience, view: HistoryView, *, recorded_at: datetime
    ) -> Proposal:
        version = MemoryVersion(
            memory_id="m",
            version=1,
            operation=Operation.CREATE,
            content=experience.content,
            valid_from=experience.occurred_at,
            recorded_at=recorded_at,
            derived_from=(experience.digest,),
        )
        decision = FormationDecision(
            experience=experience.digest,
            policy=self.name,
            reason=FormationReason.NEW_EPISODE,
            memory_id="m",
            version=version.digest,
            recorded_at=recorded_at,
        )
        return Proposal(decision, version)


def test_rejected_steps_are_recorded_and_the_run_continues(store: ArtifactStore) -> None:
    registry = DEFAULT_REGISTRY.extend(policies={"collider-v1": Collider})
    record = execute(manifest(policy="collider-v1"), store, registry)
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    first, *rest = outcomes.steps
    assert first.reason is FormationReason.NEW_EPISODE
    duplicate = RELOCATION.steps[7].experience.digest == RELOCATION.steps[0].experience.digest
    assert duplicate
    rejected = [s for s in rest if s.rejected]
    assert len(rejected) == len(rest) - 1  # all but the duplicate-experience step
    assert all("expected version 2" in (s.rejected or "") for s in rejected)
    with MemoryLog.load(store.get(record.log)) as log:
        assert len(log.versions()) == 1  # rejected steps left no trace in memory
        assert len(log.experiences()) == 1


@pytest.mark.parametrize(
    ("m", "error", "message"),
    [
        (manifest(policy="nope-v1"), UnknownComponentError, "formation policy 'nope-v1'"),
        (manifest(responder="oracle"), UnknownComponentError, "responder 'oracle'"),
        (manifest(RELOCATION, iv("teleport")), UnknownComponentError, "intervention 'teleport'"),
        (
            manifest(
                retriever=RetrieverSpec(
                    name="r", gate="dense", signals=(SignalSpec(name="dense", weight=1),)
                )
            ),
            UnknownComponentError,
            "signal 'dense'",
        ),
        (manifest(RELOCATION, iv("drop", rate=0.5)), ValueError, "invalid parameters"),
        (
            manifest(RELOCATION, iv("drop", rate=0.5, seed=1, colour=3)),
            ValueError,
            "invalid parameters",
        ),
        (manifest(RELOCATION, iv("drop", rate=5, seed=1)), ValueError, "rate must be"),
        (
            manifest(
                retriever=RetrieverSpec(
                    name="r",
                    gate="bm25",
                    signals=(SignalSpec(name="bm25", weight=1, params=(("k1", 1.2),)),),
                )
            ),
            ValueError,
            "does not rebuild",
        ),
        (
            manifest(RELOCATION, iv("inject", dataset="sha256:" + "0" * 64)),
            ArtifactIntegrityError,
            "not found",
        ),
    ],
    ids=[
        "policy",
        "responder",
        "intervention",
        "signal",
        "missing-param",
        "extra-param",
        "bad-param",
        "partial-spec",
        "inject-missing",
    ],
)
def test_manifests_that_cannot_be_resolved_fail_before_any_work(
    store: ArtifactStore, m: RunManifest, error: type[Exception], message: str
) -> None:
    before = sorted(store.root.rglob("*"))
    with pytest.raises(error, match=message):
        execute(m, store)
    assert sorted(store.root.rglob("*")) == before  # nothing was written


def test_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(ArtifactIntegrityError, match="not found"):
        execute(manifest(), ArtifactStore(tmp_path / "empty"))


def test_registry_cannot_redefine_components() -> None:
    with pytest.raises(ValueError, match="already registered: statement-v1"):
        DEFAULT_REGISTRY.extend(policies={"statement-v1": Collider})


def test_misnamed_components_are_rejected(store: ArtifactStore) -> None:
    registry = DEFAULT_REGISTRY.extend(policies={"alias-v1": Collider})
    with pytest.raises(ValueError, match="calls itself 'collider-v1'"):
        execute(manifest(policy="alias-v1"), store, registry)


def test_inject_through_a_manifest(store: ArtifactStore) -> None:
    contradiction = Dataset(
        name="contradiction",
        version="1",
        steps=(
            Step(
                experience=Experience(
                    source="adversary", content="set home = Atlantis", occurred_at=day(56)
                ),
                recorded_at=day(80),
            ),
        ),
    )
    store.put_record(contradiction)
    record = execute(manifest(RELOCATION, iv("inject", dataset=contradiction.digest)), store)
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    answers = {o.probe: o.output for o in outcomes.probes}
    assert answers["home-100"] == "home = Atlantis"  # measured: the injected claim wins
    (intervention,) = [store.get_record(InterventionRecord, d) for d in record.interventions]
    assert intervention.added == (contradiction.steps[0].digest,)


# --- reproduction detects what it must ---------------------------------------------------------


@dataclass
class Drifting:
    """A signal that violates determinism: its scores depend on how often it was called."""

    calls: ClassVar[list[int]] = [0]
    name: ClassVar[str] = "drifting"

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return ()

    def score(self, query: Query, memories: Sequence[MemoryVersion]) -> list[SignalEvidence]:
        self.calls[0] += 1
        return [SignalEvidence(signal="drifting", score=float(self.calls[0])) for _ in memories]


def test_reproduce_detects_nondeterminism(store: ArtifactStore) -> None:
    registry = DEFAULT_REGISTRY.extend(signals={"drifting": Drifting})
    spec = RetrieverSpec(
        name="bm25+drifting",
        gate="bm25",
        signals=(
            SignalSpec(name="bm25", weight=1, params=(("b", 0.75), ("k1", 1.2))),
            SignalSpec(name="drifting", weight=1),
        ),
    )
    record = execute(manifest(retriever=spec), store, registry)
    result = reproduce(record.digest, store, registry)
    assert not result.reproduced
    assert set(result.differences) == {"log", "outcomes"}
    # Both versions are kept for inspection; nothing was overwritten.
    rerun = store.get_record(RunRecord, result.rerun)
    assert store.has(record.outcomes)
    assert store.has(rerun.outcomes)


def test_reproduce_detects_corrupted_artifacts(store: ArtifactStore) -> None:
    record = execute(manifest(), store)
    h = record.outcomes.removeprefix("sha256:")
    path = store.root / "objects" / h[:2] / h[2:]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b"Berlin", b"Berlix"))
    with pytest.raises(ArtifactIntegrityError, match="does not match"):
        reproduce(record.digest, store)
