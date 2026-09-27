from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from memoria.artifacts import ArtifactStore
from memoria.comparison import normalize
from memoria.core import Dataset, EmbedderSpec, RepresentationSpec
from memoria.embeddings import HashedNgramEmbedder
from memoria.experiments import DEFAULT_REGISTRY, UnknownComponentError
from memoria.formation import parse_statement
from memoria.hybrid import HybridTrace
from memoria.hybrid_eval import (
    HybridBenchmark,
    HybridExperiment,
    HybridExperimentSpec,
    Perturbation,
    Role,
    phase6_policies,
    phase6_spec,
    report,
    reproduce_hybrid,
    run_hybrid_experiment,
    verify_experiment,
)
from memoria.scenarios import HYBRID_CASES, day, hybrid_benchmark
from memoria.semantic_eval import PerformanceRecord
from memoria.statistics import min_achievable_p

HASHED = RepresentationSpec(embedder=HashedNgramEmbedder().spec)
SMALL = ("lexical", "lexical+semantic", "full", "full-diversity", "full-contradiction")


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def benchmark(store: ArtifactStore) -> HybridBenchmark:
    dataset, bench = hybrid_benchmark()
    store.put_record(dataset)
    store.put_record(bench)
    return bench


@dataclass(frozen=True)
class Run:
    store: ArtifactStore
    benchmark: HybridBenchmark
    record: HybridExperiment
    performance: PerformanceRecord


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """The reduced study, executed once for the tests that only read its results."""
    store = ArtifactStore(tmp_path_factory.mktemp("hybrid") / "artifacts")
    dataset, bench = hybrid_benchmark()
    store.put_record(dataset)
    store.put_record(bench)
    record, performance = run_hybrid_experiment(small_spec(bench.digest), store, DEFAULT_REGISTRY)
    return Run(store, bench, record, performance)


def small_spec(benchmark: str) -> HybridExperimentSpec:
    policies, _ = phase6_policies()
    chosen = tuple(p for p in policies if p.name in SMALL)
    return HybridExperimentSpec(
        name="phase6-small",
        benchmark=benchmark,
        representation=HASHED,
        policies=chosen,
        reference="full",
        ladder=("lexical", "lexical+semantic", "full"),
        leave_one_out=(
            ("diversity", "full", "full-diversity"),
            ("contradiction", "full", "full-contradiction"),
        ),
    )


# --- the benchmark --------------------------------------------------------------------------


def test_benchmark_shape_and_declared_design() -> None:
    dataset, bench = hybrid_benchmark()
    assert len(dataset.steps) == 45
    primary = [q for q in bench.queries if q.base is None]
    variants = [q for q in bench.queries if q.base is not None]
    assert (len(primary), len(variants)) == (13, 20)
    sources = {s.experience.source for s in dataset.steps}
    assert len(sources) == 45
    for q in bench.queries:
        assert {j.candidate for j in q.judgements} <= sources
    assert {f.candidate for f in bench.facts} <= sources
    used = {
        c for q in bench.queries for c in (*q.cases, *(c for j in q.judgements for c in j.cases))
    }
    assert used == set(HYBRID_CASES) == set(bench.cases)
    assert {v.perturbation for v in variants} == set(Perturbation)
    assert {q.known_at for q in bench.queries} == {day(300)}


def _expected_values(bench: HybridBenchmark, dataset: Dataset, key: str, at: datetime) -> set[str]:
    """The epistemic answer from every report on ``key`` (statements and labelled notes):
    the latest report occurring by ``at`` holds (ties are contested); a later ``correct``
    replaces it retroactively (up to the next report); a ``forget`` leaves nothing."""
    fact = {f.candidate: f.fact for f in bench.facts}
    reports = []  # (occurred, verb, value)
    for s in dataset.steps:
        e = s.experience
        st = parse_statement(e.content)
        if st is not None and st.key == key:
            reports.append((e.occurred_at, st.verb, st.value or ""))
        elif st is None and fact.get(e.source, "").startswith(f"{key}="):
            reports.append((e.occurred_at, "set", fact[e.source].split("=", 1)[1]))
    held = [r for r in reports if r[0] <= at]
    if not held:
        return set()
    latest = max(r[0] for r in held)
    current = [r for r in held if r[0] == latest]
    later_sets = [r[0] for r in reports if r[0] > latest and r[1] != "correct"]
    horizon = min(later_sets, default=None)
    corrections = [
        r
        for r in reports
        if r[1] == "correct" and r[0] > latest and (horizon is None or r[0] < horizon)
    ]
    if corrections:
        last = max(r[0] for r in corrections)
        current = [r for r in corrections if r[0] == last]
    if any(verb == "forget" for _, verb, _ in current):
        return set()
    return {" ".join(normalize(v)) for _, _, v in current}


def test_targets_follow_the_epistemic_standard() -> None:
    dataset, bench = hybrid_benchmark()
    content = {s.experience.source: s.experience.content for s in dataset.steps}
    fact = {f.candidate: f.fact for f in bench.facts}
    checked = 0
    for q in bench.queries:
        if q.key is None or q.perturbation is Perturbation.NEGATION:
            continue  # a negated question has no declared target
        targets = [j.candidate for j in q.judgements if j.role is Role.TARGET]
        values = {" ".join(normalize(fact[c].split("=", 1)[1])) for c in targets}
        assert values == _expected_values(bench, dataset, q.key, q.valid_at), q.id
        for c in targets:
            st = parse_statement(content[c])
            assert st is None or st.key == q.key, (q.id, c)
        checked += 1
    assert checked == 31


def test_benchmark_rejects_inconsistent_designs(benchmark: HybridBenchmark) -> None:
    data = benchmark.model_dump()
    data["queries"] = [*data["queries"], data["queries"][0]]
    with pytest.raises(ValidationError, match="unique"):
        HybridBenchmark.model_validate(data)
    q = benchmark.queries[0].model_dump() | {
        "id": "orphan",
        "base": "nope",
        "perturbation": "reorder",
    }
    with pytest.raises(ValidationError, match="unknown primary"):
        HybridBenchmark.model_validate(
            benchmark.model_dump() | {"queries": [*benchmark.model_dump()["queries"], q]}
        )
    with pytest.raises(ValidationError, match="undeclared cases"):
        HybridBenchmark.model_validate(benchmark.model_dump() | {"cases": ()})


def test_benchmark_and_study_identities_are_pinned() -> None:
    _, bench = hybrid_benchmark()
    # Golden digests: a change here means the benchmark or the study design changed.
    assert bench.digest == "sha256:7aa1b23f696d24bd8c93bf4bb54857cc0fc7d741782e0bd3ec88a770412e571b"
    policies, loo = phase6_policies()
    assert len(policies) == 41
    assert len({p.digest for p in policies}) == 41
    components = ["semantic", "lexical", "recency", "source", "provenance", "attribute",
                  "temporal", "contradiction"]  # fmt: skip
    assert loo == [
        *((c, "full", f"full-{c}") for c in components),
        *((c, "+contradiction", f"nodiv-{c}") for c in components),
        ("diversity", "full", "full-diversity"),
    ]
    spec = phase6_spec(bench.digest, HASHED)
    assert spec.ladder[0] == "semantic"
    assert spec.ladder[-1] == spec.reference == "full"


# --- experiment ------------------------------------------------------------------------------


def test_experiment_runs_reproduces_and_verifies(run: Run) -> None:
    store, benchmark, record, performance = run.store, run.benchmark, run.record, run.performance
    assert store.get_record(HybridExperiment, record.digest) == record
    assert performance.experiment == record.digest
    assert performance.digest not in record.canonical()  # timings are not identity
    names = {name for name, _ in performance.timings}
    for pol in ("lexical", "full"):
        assert {f"{pol}:{s}_median" for s in ("generate", "filter", "features", "score",
                                                "rerank", "explain", "total")} <= names  # fmt: skip
    assert "full:python_peak_bytes_median" in names
    assert verify_experiment(record, store) == len(SMALL) * len(benchmark.queries)
    again = reproduce_hybrid(record.digest, store, DEFAULT_REGISTRY)
    assert again.identical  # timings differed between the runs; the identity did not
    assert again.differing_policies == ()
    assert store.get_record(PerformanceRecord, performance.digest) == performance
    assert record.embedder == HASHED.embedder.digest
    assert record.index is not None


def test_metrics_are_recounts_of_the_stored_traces(run: Run) -> None:
    store, benchmark, record = run.store, run.benchmark, run.record
    facts = {f.candidate: f.fact for f in benchmark.facts}
    primary = [q for q in benchmark.queries if q.base is None]
    for result in record.results:
        diag = {d.query: d for d in result.queries}
        at5 = next(m for m in result.at if m.k == 5)
        got = wanted = fact_got = fact_wanted = 0
        for q in primary:
            trace = store.get_record(HybridTrace, diag[q.id].trace)
            assert len(diag[q.id].ranking) == len(trace.selected)
            targets = {j.candidate for j in q.judgements if j.role is Role.TARGET}
            top = diag[q.id].ranking[:5]
            got += sum(c in targets for c in top)
            wanted += len(targets)
            tf = {facts.get(t, t) for t in targets}
            fact_got += len(tf & {facts.get(c, c) for c in top if c in targets})
            fact_wanted += len(tf)
        assert (at5.recall.numerator, at5.recall.denominator) == (got, wanted)
        assert (at5.fact_recall.numerator, at5.fact_recall.denominator) == (fact_got, fact_wanted)
        assert result.validity.denominator == sum(len(diag[q.id].ranking) for q in primary)


def test_paired_tests_and_deltas_are_consistent(run: Run) -> None:
    benchmark, record = run.benchmark, run.record
    results = {r.name: r for r in record.results}
    targets = sum(
        1
        for q in benchmark.queries
        if q.base is None
        for j in q.judgements
        if j.role is Role.TARGET
    )
    for t in record.paired:
        assert t.both + t.first_only + t.second_only + t.neither == t.pairs
        if t.measure == "recall@5":
            assert t.pairs == targets
        discordant = t.first_only + t.second_only
        assert t.min_achievable_p == pytest.approx(min_achievable_p(discordant))
        assert t.underpowered == (min_achievable_p(discordant) > 0.05)
        assert t.p_value is not None
        assert t.p_adjusted is not None
        assert t.p_adjusted >= t.p_value
        if discordant == 0:
            assert (t.p_value, t.difference) == (1.0, 0.0)
    for d in record.deltas:
        ref, other = results["full"], results[d.policy]
        r5 = next(m for m in ref.at if m.k == 5).recall.estimate
        o5 = next(m for m in other.at if m.k == 5).recall.estimate
        assert r5 is not None
        assert o5 is not None
        assert dict(d.recall)[5] == pytest.approx(o5 - r5)
    contradiction = next(d for d in record.deltas if d.removed == "contradiction")
    assert contradiction.affected_queries == ()  # surfacing never reorders
    assert contradiction.exposure is not None
    assert contradiction.exposure < 0


def test_counterfactuals_and_cases_are_reported(run: Run) -> None:
    benchmark, record, performance = run.benchmark, run.record, run.performance
    variants = [q for q in benchmark.queries if q.base is not None]
    for pol in SMALL:
        cfs = [c for c in record.counterfactuals if c.policy == pol]
        assert len(cfs) == len(variants)
        for c in cfs:
            assert 0 <= c.overlap <= 1
            assert c.invariant == c.perturbation.invariant
        assert {c.case for c in record.cases if c.policy == pol} == set(HYBRID_CASES)
    text = report(record, performance)
    for heading in ("Policies", "Paired tests", "Leave one out", "Counterfactuals",
                    "Adversarial cases", "Performance"):  # fmt: skip
        assert f"## {heading}" in text


def test_undefined_measurements_stay_undefined(run: Run) -> None:
    record = run.record
    proportions = [x for r in record.results for x in (r.exposure, r.redundancy, r.validity)]
    proportions += [x for c in record.cases for x in (c.focus_visible, c.target_above_focus)]
    empty = [x for x in proportions if x.denominator == 0]
    assert empty  # e.g. exposure when no top result is disputed
    assert all((x.estimate, x.low, x.high) == (None, None, None) for x in empty)
    retracted = next(c for c in record.cases if c.case == "retracted" and c.policy == "full")
    assert retracted.target_above_focus.denominator == 0  # no target exists after a forget
    assert retracted.target_above_focus.estimate is None


def test_unavailable_embedder_fails_before_any_result(
    store: ArtifactStore, benchmark: HybridBenchmark
) -> None:
    missing = EmbedderSpec(name="not-installed", version="1", dimensions=8, normalized=True)
    spec = small_spec(benchmark.digest).model_copy(
        update={"representation": RepresentationSpec(embedder=missing)}
    )
    with pytest.raises(UnknownComponentError, match="not-installed"):
        run_hybrid_experiment(spec, store, DEFAULT_REGISTRY)
    assert store.digests("HybridExperiment") == []
    assert store.digests("HybridTrace") == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"reference": "nope"}, "unknown policies"),
        ({"ladder": ("lexical", "nope")}, "unknown policies"),
        ({"ks": (5, 1)}, "increasing"),
    ],
)
def test_spec_validation(
    benchmark: HybridBenchmark, change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        HybridExperimentSpec.model_validate(small_spec(benchmark.digest).model_dump() | change)
    spec = small_spec(benchmark.digest)
    with pytest.raises(ValidationError, match="unique"):
        HybridExperimentSpec.model_validate(
            spec.model_dump()
            | {"policies": [*spec.model_dump()["policies"], spec.policies[0].model_dump()]}
        )


def test_benchmark_must_share_one_known_at(store: ArtifactStore) -> None:
    dataset, bench = hybrid_benchmark()
    shifted = bench.queries[0].model_copy(update={"known_at": day(301)})
    odd = bench.model_copy(update={"queries": (shifted, *bench.queries[1:])})
    store.put_record(dataset)
    store.put_record(odd)
    with pytest.raises(ValueError, match="one known_at"):
        run_hybrid_experiment(small_spec(odd.digest), store, DEFAULT_REGISTRY)
