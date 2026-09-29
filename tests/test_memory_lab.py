from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from memoria.artifacts import ArtifactStore
from memoria.consolidation_eval import ConsolidationBenchmark
from memoria.core import Dataset, RepresentationSpec, RunManifest, RunOutcomes, RunRecord
from memoria.embeddings import HashedNgramEmbedder
from memoria.evaluation import MANIFEST_VARIABLES, EvaluationError, compare, evaluate
from memoria.experiments import execute, reproduce
from memoria.forgetting import ForgettingRecord
from memoria.graph import GraphPolicy, GraphSnapshot, verify_snapshot
from memoria.hybrid import Corpus, HybridTrace
from memoria.memory_lab import (
    FORGETTING,
    GRAPH,
    POLICIES,
    LabSpec,
    _manifest,
    _probe_graph,
    lab_spec,
    run_lab,
    run_metrics,
    structure,
    world_distractors,
)
from memoria.retrieval import EXTRACTIVE
from memoria.scenarios import WorldSpec, consolidation_world
from memoria.store import MemoryLog

EMB = HashedNgramEmbedder()
HASHED = RepresentationSpec(embedder=EMB.spec)
SMALL = WorldSpec(name="adversarial", seed=3, entities=2, horizon_days=120, change_every_days=40,
                  contradiction_rate=0.5, correction_rate=0.3, probes_per_key=2)  # fmt: skip


@dataclass(frozen=True)
class Lab:
    store: ArtifactStore
    bench: ConsolidationBenchmark
    spec: LabSpec


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    store = ArtifactStore(tmp_path / "a")
    dataset, bench = consolidation_world(SMALL)
    store.put_record(dataset)
    b = store.put_record(bench)
    return Lab(store, bench, lab_spec(store, [b], HASHED))


def changed(a: RunManifest, b: RunManifest) -> list[str]:
    return [n for n, fields in MANIFEST_VARIABLES.items()
            if any(getattr(a, f) != getattr(b, f) for f in fields)]  # fmt: skip


def test_every_controlled_comparison_changes_exactly_one_variable(lab: Lab) -> None:
    by = {c.name: _manifest(lab.bench, c, lab.spec, lab.store) for c in lab.spec.conditions}
    for x in lab.spec.comparisons:
        diff = changed(by[x.baseline], by[x.treatment])
        assert (len(diff) == 1) != x.composite, (x, diff)
    names = [c.name for c in lab.spec.conditions]
    assert len(set(names)) == len(names)


def test_schema_v4_manifests(lab: Lab) -> None:
    graph = lab.store.put_record(GRAPH["conservative"])
    forgetting = lab.store.put_record(FORGETTING["validity"])
    common = {"name": "m", "dataset": lab.bench.dataset, "policy": "episodic-v1",
              "responder": EXTRACTIVE}  # fmt: skip
    with pytest.raises(ValidationError, match="hybrid retrieval only"):
        RunManifest.model_validate(common | {"retriever": {"name": "r", "gate": "bm25",
                                                            "signals": [{"name": "bm25",
                                                                         "weight": 1}]},
                                              "graph_policy": graph})  # fmt: skip
    mixed = lab.store.put_record(POLICIES["mixed"])
    m = RunManifest.model_validate(common | {"retrieval_policy": mixed, "graph_policy": graph,
                                             "representation": HASHED})  # fmt: skip
    with pytest.raises(ValueError, match="iff the retrieval policy reads a graph"):
        execute(m, lab.store)
    plain = RunManifest.model_validate(common | {"retrieval_policy": mixed,
                                                 "representation": HASHED})  # fmt: skip
    assert "graph_policy" not in plain.canonical()  # absent fields keep v3 digests
    assert "forgetting_policy" not in plain.canonical()
    assert RunManifest.model_validate(plain.model_dump() | {"forgetting_policy": forgetting})


def test_all_systems_together(lab: Lab) -> None:
    only = ["base", "graph", "forgetting", "consolidated", "interference", "combined",
            "ladder:+consolidation", "ladder:+graph"]  # fmt: skip
    study, _ = run_lab(lab.spec, lab.store, only=only)
    runs = {r.condition: r for r in study.runs}
    assert set(runs) == set(only)
    combined = lab.store.get_record(RunRecord, runs["combined"].run)
    dataset_probes = len(lab.store.get_record(RunOutcomes, combined.outcomes).probes)
    assert len(combined.graphs) == len(combined.forgetting) == dataset_probes
    assert combined.hierarchies
    assert reproduce(combined.digest, lab.store).reproduced
    # each probe's graph is rebuilt from the log and matches the recorded digest
    traces = [lab.store.get_record(HybridTrace, o.trace)
              for o in lab.store.get_record(RunOutcomes, combined.outcomes).probes]  # fmt: skip
    assert [t.graph for t in traces] == list(combined.graphs)
    for d in combined.forgetting:
        r = lab.store.get_record(ForgettingRecord, d)
        assert {x.state for x in r.decisions} <= {"active", "forgotten", "excluded"}
    effects = {(e.baseline, e.treatment): e for e in study.effects}
    assert effects[("base", "combined")].composite
    assert effects[("base", "combined")].comparison is None
    assert effects[("base", "graph")].comparison is not None
    for e in study.effects:
        diff, lo, hi, pv, _ = e.known_correct
        if diff is not None:
            assert lo is not None
            assert hi is not None
            assert lo <= diff <= hi
            assert pv is not None
            assert 0 <= pv <= 1
        assert e.lost <= e.retrievable_before
    base_eval = runs["base"].evaluation
    with pytest.raises(EvaluationError, match="exactly one manifest variable"):
        compare(base_eval, runs["combined"].evaluation, lab.store)
    metrics = run_metrics(combined, lab.store)
    assert metrics.hidden_share is not None
    assert metrics.integrity.denominator == metrics.answered


def test_injected_interference_is_classified_not_silent(lab: Lab) -> None:
    study, _ = run_lab(lab.spec, lab.store, only=["base", "inject:rival"])
    runs = {r.condition: r for r in study.runs}
    rival = runs["inject:rival"]
    assert rival.contaminated.numerator > 0  # rival answers are attributed to the injection
    record = lab.store.get_record(RunRecord, rival.run)
    assert record.interventions  # the injected dataset is a recorded intervention
    ev = evaluate(rival.run, lab.store)
    assert any(p.outcome.value == "contaminated" for p in ev.probes)


def test_distractor_generation_is_deterministic_and_leaves_probes_alone() -> None:
    world, _ = consolidation_world(SMALL)
    a, b = world_distractors(world), world_distractors(world)
    assert a == b
    assert not a.probes
    assert {s.experience.source.split(":")[0] for s in a.steps} == {"forum"}
    per_key: dict[str, list[str]] = {}
    for st in a.steps:
        key, _ = st.experience.source.removeprefix("forum:inject-").rsplit("-", 1)
        per_key.setdefault(key, []).append(st.experience.content)
    assert len(per_key) == SMALL.entities * len(SMALL.attributes)
    for key, texts in per_key.items():
        assert len(texts) == 4
        assert sum(t == f"set {key} = {t.split(' = ')[-1]}" for t in texts) >= 1  # a rival
        assert any(not t.startswith("set ") for t in texts)  # a near-collision note


def test_structure_study_verifies_its_snapshot(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    s = structure(SMALL, store, EMB)
    snap = store.get_record(GraphSnapshot, s.snapshot)
    assert s.verified
    assert s.metrics_all.unreachable_evidence == 0
    assert s.metrics_active.unavailable_nodes > 0
    assert s.false_merges[0] == 0
    assert [g.nodes for g in s.growth] == sorted(g.nodes for g in s.growth)
    assert snap.node_count == s.metrics_all.nodes


def test_run_graphs_rebuild_from_the_log(lab: Lab) -> None:
    study, _ = run_lab(lab.spec, lab.store, only=["graph"])
    record = lab.store.get_record(RunRecord, study.runs[0].run)
    dataset = lab.store.get_record(RunOutcomes, record.outcomes)
    manifest = lab.store.get_record(RunManifest, record.manifest)
    assert manifest.graph_policy is not None
    probes = lab.store.get_record(Dataset, record.dataset).probes
    gpol = lab.store.get_record(GraphPolicy, manifest.graph_policy)
    with MemoryLog.load(lab.store.get(record.log)) as log:
        for i, p in enumerate(probes[:3]):
            snap = _probe_graph(log, p.known_at, [], gpol)
            assert snap.digest == record.graphs[i]
            verify_snapshot(snap, Corpus.from_log(log, p.known_at))
    assert len(dataset.probes) == len(record.graphs)
