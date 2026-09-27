from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from memoria.artifacts import ArtifactStore
from memoria.consolidation import Hierarchy
from memoria.consolidation_eval import (
    RETRIEVAL,
    STRATEGIES,
    ConsolidationBenchmark,
    LabSpec,
    LabStudy,
    demonstration,
    lab_spec,
    measure_consolidation,
    report,
    reproduce_lab,
    run_lab,
)
from memoria.core import (
    EpistemicStatus,
    Level,
    RepresentationSpec,
    RunManifest,
    RunOutcomes,
    RunRecord,
)
from memoria.embeddings import HashedNgramEmbedder
from memoria.evaluation import EvaluationError, compare, evaluate
from memoria.experiments import execute, reproduce
from memoria.hybrid import HybridTrace, RetrievalPolicy
from memoria.retrieval import EXTRACTIVE, lexical
from memoria.scenarios import PHASE7_WORLDS, WorldSpec, consolidation_world, relocation_year

HASHED = RepresentationSpec(embedder=HashedNgramEmbedder().spec)
SMALL = WorldSpec(name="small", seed=3, entities=2, horizon_days=120, change_every_days=40,
                  contradiction_rate=0.5, correction_rate=0.3, probes_per_key=3)  # fmt: skip


@dataclass(frozen=True)
class World:
    store: ArtifactStore
    bench: ConsolidationBenchmark
    mixed: str
    hier: str


@pytest.fixture
def world(tmp_path: Path) -> World:
    store = ArtifactStore(tmp_path / "a")
    dataset, bench = consolidation_world(SMALL)
    store.put_record(dataset)
    store.put_record(bench)
    return World(store, bench, store.put_record(RETRIEVAL["mixed"]),
                 store.put_record(STRATEGIES["E:hierarchical"]))  # fmt: skip


def manifest(w: World, consolidation: str | None, retrieval: str | None = None) -> RunManifest:
    return RunManifest(
        name="t",
        dataset=w.bench.dataset,
        policy="episodic-v1",
        responder=EXTRACTIVE,
        representation=HASHED,
        retrieval_policy=retrieval or w.mixed,
        consolidation_policy=consolidation,
    )


# --- manifest schema v3 ----------------------------------------------------------------------


def test_v3_fields_are_additive_and_validated(world: World) -> None:
    v1 = RunManifest(name="l", dataset=world.bench.dataset, policy="statement-v1",
                     retriever=lexical().spec, responder=EXTRACTIVE)  # fmt: skip
    for field in ("retrieval_policy", "consolidation_policy"):
        assert field not in v1.canonical()
    with pytest.raises(ValidationError, match="exactly one"):
        RunManifest.model_validate(v1.model_dump() | {"retrieval_policy": world.mixed})
    with pytest.raises(ValidationError, match="exactly one"):
        RunManifest.model_validate(v1.model_dump() | {"retriever": None})
    with pytest.raises(ValidationError, match="hybrid retrieval policy"):
        RunManifest.model_validate(v1.model_dump() | {"consolidation_policy": world.hier})
    loaded = RunManifest.model_validate_json(v1.canonical())
    assert loaded == v1
    assert loaded.digest == v1.digest


def test_representation_must_match_vector_use(world: World) -> None:
    lexical_only = world.store.put_record(
        RetrievalPolicy.model_validate(RETRIEVAL["mixed"].model_dump() | {
            "signals": [s.model_dump() | {"weight": 1.0} for s in RETRIEVAL["mixed"].signals
                        if s.name == "lexical"],
            "generators": [g.model_dump() for g in RETRIEVAL["mixed"].generators
                           if g.name != "semantic"],
        })
    )  # fmt: skip
    with pytest.raises(ValueError, match="representation"):
        execute(manifest(world, None, lexical_only), world.store)


# --- consolidated runs --------------------------------------------------------------------------


def test_consolidated_run_is_evaluated_compared_and_reproduced(world: World) -> None:
    base = execute(manifest(world, None), world.store)
    treat = execute(manifest(world, world.hier), world.store)
    assert base.hierarchies == ()
    assert treat.hierarchies  # one per checkpoint
    hierarchies = [world.store.get_record(Hierarchy, h) for h in treat.hierarchies]
    assert [h.at for h in hierarchies] == sorted(h.at for h in hierarchies)
    eb, et = evaluate(base.digest, world.store), evaluate(treat.digest, world.store)
    comparison = compare(eb.digest, et.digest, world.store)
    assert comparison.variable == "consolidation"
    assert reproduce(treat.digest, world.store).reproduced
    outcomes = world.store.get_record(RunOutcomes, treat.outcomes)
    levels = set()
    for o in outcomes.probes:
        trace = world.store.get_record(HybridTrace, o.trace)  # re-validated on load
        for c in trace.candidates:
            levels.add(c.level)
            assert (c.level is Level.L1) == (c.status is EpistemicStatus.OBSERVED)
    assert Level.L2 in levels


def test_retrieval_levels_are_a_policy_variable(world: World) -> None:
    raw = execute(manifest(world, world.hier, world.store.put_record(RETRIEVAL["raw"])),
                  world.store)  # fmt: skip
    cons = execute(
        manifest(world, world.hier, world.store.put_record(RETRIEVAL["consolidated"])), world.store
    )
    for run, allowed in ((raw, {Level.L1}), (cons, {Level.L2, Level.L3, Level.L4})):
        for o in world.store.get_record(RunOutcomes, run.outcomes).probes:
            trace = world.store.get_record(HybridTrace, o.trace)
            assert {c.level for c in trace.candidates} <= allowed
    er, ec = evaluate(raw.digest, world.store), evaluate(cons.digest, world.store)
    assert compare(er.digest, ec.digest, world.store).variable == "retrieval"


def test_retrieval_policy_rejects_l0_and_unsorted_levels() -> None:
    data = RETRIEVAL["mixed"].model_dump()
    with pytest.raises(ValidationError, match="provenance"):
        RetrievalPolicy.model_validate(data | {"levels": ("L0", "L1")})
    with pytest.raises(ValidationError, match="sorted"):
        RetrievalPolicy.model_validate(data | {"levels": ("L2", "L1")})


def test_tampered_hybrid_trace_level_is_rejected(world: World) -> None:
    run = execute(manifest(world, world.hier), world.store)
    traces = [world.store.get_record(HybridTrace, o.trace)
              for o in world.store.get_record(RunOutcomes, run.outcomes).probes]  # fmt: skip
    trace = next(t for t in traces if any(c.level is not Level.L1 for c in t.candidates))
    data = trace.model_dump()
    idx = next(i for i, c in enumerate(trace.candidates) if c.level is not Level.L1)
    data["candidates"][idx]["status"] = "observed"  # a derived memory posing as evidence
    with pytest.raises(ValidationError, match="only L1 memories are observed"):
        HybridTrace.model_validate(data)


def test_evaluation_refuses_a_foreign_trace(world: World) -> None:
    run = execute(manifest(world, world.hier), world.store)
    outcomes = world.store.get_record(RunOutcomes, run.outcomes)
    swapped = outcomes.model_copy(
        update={
            "probes": (
                outcomes.probes[0].model_copy(update={"trace": outcomes.probes[1].trace}),
                *outcomes.probes[1:],
            )
        }
    )
    forged = run.model_copy(update={"outcomes": world.store.put_record(swapped)})
    world.store.put_record(forged)
    with pytest.raises(EvaluationError, match="different query"):
        evaluate(forged.digest, world.store)


def test_classic_runs_are_unchanged(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "c")
    dataset = relocation_year()
    store.put_record(dataset)
    m = RunManifest(name="lexical", dataset=dataset.digest, policy="statement-v1",
                    retriever=lexical().spec, responder=EXTRACTIVE)  # fmt: skip
    record = execute(m, store)
    assert "hierarchies" not in record.canonical()
    assert evaluate(record.digest, store).probes


# --- consolidation measurements ------------------------------------------------------------------


def test_measurements_replay_and_follow_the_hierarchies(world: World) -> None:
    run = execute(manifest(world, world.hier), world.store)
    metrics = measure_consolidation(run, world.store, world.bench)
    assert metrics is not None
    assert metrics.replayed
    assert metrics.hierarchies == len(run.hierarchies)
    assert metrics.lineage_complete.numerator == metrics.lineage_complete.denominator
    assert metrics.merge_precision.denominator >= metrics.merge_precision.numerator
    assert measure_consolidation(execute(manifest(world, None), world.store), world.store,
                                 world.bench) is None  # fmt: skip


def test_lab_study_reproduces_and_reports(world: World) -> None:
    spec = lab_spec(world.store, [world.store.put_record(world.bench)], HASHED)
    only = ("A:none", "B:lossless", "E:hierarchical", "semantic:0.3-unguarded",
            "retrieval:raw", "retrieval:consolidated")  # fmt: skip
    spec = spec.model_copy(
        update={"conditions": tuple(c.model_copy(update={"worlds": ()}) for c in spec.conditions)}
    )
    spec = LabSpec.model_validate(spec.model_dump())
    study, performance = run_lab(spec, world.store, only=only)
    assert world.store.get_record(LabStudy, study.digest) == study
    assert {r.condition for r in study.runs} == set(only)
    for c in study.comparisons:
        d, lo, hi, p, _ = c.known_correct
        assert d is None or (lo is not None and hi is not None and lo <= d <= hi)
        assert p is None or 0 <= p <= 1
    assert reproduce_lab(study.digest, world.store, only=only).identical
    text = report(study, performance)
    assert "## Runs" in text
    assert "## Paired comparisons" in text


def test_lab_spec_validation(world: World) -> None:
    spec = lab_spec(world.store, [world.store.put_record(world.bench)], HASHED)
    with pytest.raises(ValidationError, match="unknown conditions"):
        LabSpec.model_validate(spec.model_dump() | {"comparisons": [("x", "A:none", "nope")]})


def test_demonstration_observes_gains_and_losses(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "d")
    demo, performance = demonstration(store, HASHED)
    assert performance.experiment == demo.digest  # latencies are not identity
    assert demonstration(store, HASHED)[0] == demo
    assert demo.traversal.status is not EpistemicStatus.OBSERVED
    assert demo.traversal.chain[0] == demo.traversal.memory
    assert demo.traversal.sources
    assert demo.loss  # a measurable information loss
    assert demo.efficiency.consolidated_universe < demo.efficiency.raw_universe
    record = store.get_record(RunRecord, demo.perturbed)
    assert record.interventions  # the adversarial perturbation is recorded
    assert {w.name for w in PHASE7_WORLDS} >= {"adversarial", "baseline"}
