"""End-to-end evaluation of stored runs. Phase 4 exit criterion: baseline vs intervention
comparison with confidence intervals."""

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from memoria.artifacts import ArtifactIntegrityError, ArtifactStore
from memoria.core import (
    Dataset,
    ExpectationStatus,
    MemoryState,
    Response,
    RetrievalTrace,
    RunManifest,
    RunRecord,
)
from memoria.evaluation import (
    DEFAULT_SPEC,
    Evaluation,
    EvaluationError,
    EvaluationSpec,
    Measurement,
    MissingCause,
    RunComparison,
    compare,
    evaluate,
)
from memoria.experiments import DEFAULT_REGISTRY, Registry, execute
from memoria.scenarios import conditions, drifting_facts, neighbours, relocation_year
from memoria.statistics import Proportion
from memoria.taxonomy import Category, Locus, Outcome

RELOCATION = relocation_year()
DRIFT = drifting_facts(seed=7, keys=8, changes=8, probes_per_key=12)
OC = Outcome


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    s = ArtifactStore(tmp_path / "var")
    for d in (RELOCATION, DRIFT, neighbours()):
        s.put_record(d)
    return s


def run_and_evaluate(
    store: ArtifactStore,
    manifest: RunManifest,
    registry: Registry = DEFAULT_REGISTRY,
    spec: EvaluationSpec = DEFAULT_SPEC,
) -> Evaluation:
    return evaluate(execute(manifest, store, registry).digest, store, spec)


def grid(dataset: Dataset, policy: str) -> dict[str, RunManifest]:
    return conditions(dataset.digest, policy)


# --- per-probe semantics on the hand-built scenario -------------------------------------------

STATEMENT_BASELINE = {
    "home-57-before-report": (OC.CORRECT, Locus.NONE, None),
    "home-57-after-report": (OC.CORRECT, Locus.NONE, None),
    "home-100": (OC.CORRECT, Locus.NONE, None),
    "home-345": (OC.CONTESTED_ANSWERED, Locus.NONE, None),
    "employer-10-before-correction": (OC.CORRECT, Locus.NONE, None),
    "employer-10-after-correction": (OC.CORRECT, Locus.NONE, None),
    "employer-175-forgotten": (OC.CORRECT_ABSTENTION, Locus.NONE, None),
    "employer-345-relearned": (OC.CORRECT, Locus.NONE, None),
    "pet-345": (OC.CORRECT, Locus.NONE, None),
    # statement-v1 cannot correct what it never held: the policy rejected it (unknown_key).
    "pet-260-correct-without-memory": (
        OC.MISSING_MEMORY,
        Locus.MEMORY,
        MissingCause.REJECTED_BY_POLICY,
    ),
    "unrelated": (OC.CORRECT_ABSTENTION, Locus.NONE, None),
    "before-anything": (OC.CORRECT_ABSTENTION, Locus.NONE, None),
}
EPISODIC_BASELINE = STATEMENT_BASELINE | {
    # A late, older report wins a BM25 tie although Berlin was held: retrieval, not memory.
    "home-100": (OC.STALE, Locus.RETRIEVAL, None),
    # Unrelated chatter outranks both contested reports.
    "home-345": (OC.WRONG_MEMORY, Locus.RETRIEVAL, None),
    # Episodes are valid from when they occurred, so the correction does not reach day 10.
    "employer-10-after-correction": (
        OC.CORRECTION_FAILURE,
        Locus.MEMORY,
        MissingCause.OUTSIDE_VALIDITY,
    ),
    # The "forget employer" episode is returned: it asserts no value.
    "employer-345-relearned": (OC.RETRIEVAL_MISS, Locus.RETRIEVAL, None),
    # Episodes keep "correct pet = cat" as a plain assertion, so this one is answered.
    "pet-260-correct-without-memory": (OC.CORRECT, Locus.NONE, None),
}


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("statement-v1", STATEMENT_BASELINE), ("episodic-v1", EPISODIC_BASELINE)],
)
def test_relocation_classifications(
    store: ArtifactStore, policy: str, expected: dict[str, tuple[Outcome, Locus, str | None]]
) -> None:
    ev = run_and_evaluate(store, grid(RELOCATION, policy)["baseline"])
    got = {p.probe: (p.outcome, p.classification.locus, p.missing_cause) for p in ev.probes}
    assert got == expected


def test_probe_evaluation_preserves_raw_observation(store: ArtifactStore) -> None:
    record = execute(grid(RELOCATION, "statement-v1")["baseline"], store)
    ev = evaluate(record.digest, store)
    assert (ev.run, ev.dataset) == (record.digest, RELOCATION.digest)
    p = next(p for p in ev.probes if p.probe == "home-57-after-report")
    assert p.output == "home = Berlin"
    assert p.reading.values == ("Berlin",)
    assert p.reading.reader == "assignment"
    assert p.retrieval.expected_rank == 1
    assert p.retrieval.expected_selected
    assert p.classification.support[0].source == "chat:55"


# --- measurements -------------------------------------------------------------------------------


def check_measurements(ev: Evaluation) -> None:
    ids = {p.probe for p in ev.probes}
    by_id = {p.probe: p for p in ev.probes}
    for m in ev.measurements:
        assert set(m.counted) <= set(m.over) <= ids
        assert m.proportion.denominator == len(m.over)
        assert m.proportion.confidence == ev.spec.confidence
    shares = [m for m in ev.measurements if m.name.startswith("outcome.")]
    assert {m.name for m in shares} == {f"outcome.{o.value}" for o in Outcome}
    assert sum(m.proportion.numerator for m in shares) == len(ev.probes)  # a partition
    known = ev.measurement("known.correct")
    assert set(known.over) == {
        p.probe
        for p in ev.probes
        if p.expected.status is ExpectationStatus.KNOWN
        and p.outcome.category is not Category.UNSCORABLE
    }
    assert all(by_id[i].outcome is OC.CORRECT for i in known.counted)
    failed = [p for p in ev.probes if p.outcome.category is Category.FAILURE]
    loci = [ev.measurement(f"failure.locus.{lo}") for lo in ("retrieval", "memory")]
    assert sum(m.proportion.numerator for m in loci) == len(failed)
    causes = [m for m in ev.measurements if m.name.startswith("failure.cause.")]
    unheld = {
        p.probe
        for p in failed
        if p.expected.status is ExpectationStatus.KNOWN and p.retrieval.expected_rank is None
    }
    assert all(set(m.over) == unheld for m in causes)  # population by definition
    assert sum(m.proportion.numerator for m in causes) <= len(unheld)


@pytest.mark.parametrize("policy", ["statement-v1", "episodic-v1"])
@pytest.mark.parametrize("dataset", [RELOCATION, DRIFT, neighbours()], ids=lambda d: d.name)
def test_every_condition_evaluates_consistently(
    store: ArtifactStore, dataset: Dataset, policy: str
) -> None:
    """The Phase 3 suite, run through the evaluator."""
    for manifest in grid(dataset, policy).values():
        ev = run_and_evaluate(store, manifest)
        assert len(ev.probes) == len(dataset.probes)
        check_measurements(ev)


def test_measurements_reject_inconsistent_counts() -> None:
    good = Measurement(
        name="m",
        population="p",
        over=("a", "b"),
        counted=("a",),
        proportion=Proportion.of(1, 2, 0.95),
    )
    for bad in (
        {"counted": ("c",)},
        {"proportion": Proportion.of(2, 2, 0.95)},
        {"over": ("a",)},
    ):
        with pytest.raises(ValidationError):
            Measurement.model_validate(good.model_dump() | bad)


def test_drop_and_delay_failures_are_attributed(store: ArtifactStore) -> None:
    runs = grid(DRIFT, "statement-v1")
    causes = {
        name: Counter(
            p.missing_cause for p in run_and_evaluate(store, runs[name]).probes if p.missing_cause
        )
        for name in ("drop", "delay")
    }
    assert causes["drop"][MissingCause.NOT_INGESTED] > 0
    assert causes["delay"][MissingCause.NOT_YET_INGESTED] > 0
    assert MissingCause.NOT_INGESTED not in causes["delay"]


def test_interference_is_classified_as_wrong_memory(store: ArtifactStore) -> None:
    ev = run_and_evaluate(store, grid(neighbours(), "statement-v1")["baseline"])
    home = next(p for p in ev.probes if p.probe == "home")
    assert (home.outcome, home.classification.locus) == (OC.WRONG_MEMORY, Locus.RETRIEVAL)
    assert home.classification.observed is not None
    assert home.classification.observed.key == "home.office"
    assert home.retrieval.subject_values >= 1


def test_empty_dataset_has_undefined_not_zero_rates(store: ArtifactStore) -> None:
    empty = Dataset(name="empty", version="1", steps=())
    store.put_record(empty)
    ev = run_and_evaluate(store, grid(empty, "statement-v1")["baseline"])
    assert ev.probes == ()
    for m in ev.measurements:
        assert (m.proportion.denominator, m.proportion.estimate, m.proportion.low) == (
            0,
            None,
            None,
        )


# --- baseline vs intervention (the exit criterion) --------------------------------------------


def comparison(
    store: ArtifactStore, dataset: Dataset, policy: str, treatment: str
) -> RunComparison:
    runs = grid(dataset, policy)
    base = run_and_evaluate(store, runs["baseline"])
    treat = run_and_evaluate(store, runs[treatment])
    return compare(base.digest, treat.digest, store)


def test_baseline_vs_intervention_with_confidence_intervals(store: ArtifactStore) -> None:
    c = comparison(store, DRIFT, "statement-v1", "drop")
    assert c.variable == "interventions"
    m = c.measure("known.correct")
    assert len(m.pairs) == m.baseline.denominator == m.treatment.denominator > 0
    assert m.difference is not None
    assert m.low is not None
    assert m.high is not None
    assert m.low <= m.difference <= m.high
    assert m.high < 0  # dropping 30% of reports clearly degrades accuracy
    assert m.p_value is not None
    assert m.p_value < 0.05
    assert not m.underpowered
    assert (c.measure("outcome.stale").difference or 0) > 0  # ... through stale answers


def test_paired_cells_and_marginals_are_consistent(store: ArtifactStore) -> None:
    c = comparison(store, DRIFT, "statement-v1", "contaminate")
    base = store.get_record(Evaluation, c.baseline)
    treat = store.get_record(Evaluation, c.treatment)
    outcome_of = {
        "b": {p.probe: p.outcome for p in base.probes},
        "t": {p.probe: p.outcome for p in treat.probes},
    }
    for m in c.measures:
        assert m.both + m.treatment_only + m.baseline_only + m.neither == len(m.pairs)
        if m.difference is not None:
            assert m.difference == pytest.approx(
                (m.treatment_only - m.baseline_only) / len(m.pairs), abs=1e-11
            )
        if m.p_value is not None:
            assert m.p_adjusted is not None
            assert m.p_adjusted >= m.p_value
    for t in c.transitions:
        for probe in t.probes:
            assert (outcome_of["b"][probe], outcome_of["t"][probe]) == (t.baseline, t.treatment)
    moved = sum(len(t.probes) for t in c.transitions)
    assert moved == sum(outcome_of["b"][p] is not outcome_of["t"][p] for p in outcome_of["b"])
    contaminated = c.measure("outcome.contaminated")
    assert contaminated.baseline_only == 0
    assert contaminated.treatment_only > 0


def test_no_change_is_reported_as_underpowered_not_as_no_effect(store: ArtifactStore) -> None:
    c = comparison(store, DRIFT, "statement-v1", "reorder")
    m = c.measure("known.correct")
    assert (m.treatment_only, m.baseline_only, m.difference) == (0, 0, 0.0)
    assert m.underpowered  # zero discordant pairs can never reach alpha
    assert m.low is not None
    assert m.low < 0 < (m.high or 0)  # the interval still bounds the possible effect


def test_empty_comparison_is_undefined(store: ArtifactStore) -> None:
    empty = Dataset(name="empty", version="1", steps=())
    store.put_record(empty)
    c = comparison(store, empty, "statement-v1", "drop")
    for m in c.measures:
        assert (m.difference, m.p_value, m.underpowered) == (None, None, True)


def test_comparison_requires_exactly_one_variable(store: ArtifactStore) -> None:
    runs = grid(DRIFT, "statement-v1")
    base = run_and_evaluate(store, runs["baseline"])
    with pytest.raises(EvaluationError, match="changed: none"):
        compare(base.digest, base.digest, store)
    two = conditions(DRIFT.digest, "episodic-v1")["drop"]  # policy and interventions differ
    with pytest.raises(EvaluationError, match="policy"):
        compare(base.digest, run_and_evaluate(store, two).digest, store)
    other_policy = run_and_evaluate(store, conditions(DRIFT.digest, "episodic-v1")["baseline"])
    assert compare(base.digest, other_policy.digest, store).variable == "policy"


def test_comparison_requires_same_spec_and_probes(store: ArtifactStore) -> None:
    runs = grid(DRIFT, "statement-v1")
    base = run_and_evaluate(store, runs["baseline"])
    strict = EvaluationSpec(name="strict", readers=("statement", "assignment"))
    with pytest.raises(EvaluationError, match="specifications"):
        compare(base.digest, run_and_evaluate(store, runs["drop"], spec=strict).digest, store)
    other_dataset = conditions(RELOCATION.digest, "statement-v1")["baseline"]
    with pytest.raises(EvaluationError, match="identical probes"):
        compare(base.digest, run_and_evaluate(store, other_dataset).digest, store)


def test_policy_vulnerability_is_queryable(store: ArtifactStore) -> None:
    """Which policy is vulnerable to which intervention, from structured results only."""
    vulnerable = {}
    for policy in ("statement-v1", "episodic-v1"):
        c = comparison(store, DRIFT, policy, "contaminate")
        m = c.measure("known.correct")
        vulnerable[policy] = m.high is not None and m.high < 0
    assert vulnerable == {"statement-v1": True, "episodic-v1": False}


# --- reproducibility and provenance -------------------------------------------------------------


def test_evaluation_is_deterministic_across_stores(tmp_path: Path) -> None:
    digests = []
    for name in ("a", "b"):
        s = ArtifactStore(tmp_path / name)
        s.put_record(DRIFT)
        runs = grid(DRIFT, "statement-v1")
        base = run_and_evaluate(s, runs["baseline"])
        treat = run_and_evaluate(s, runs["delay"])
        again = evaluate(base.run, s)
        assert again == base  # re-evaluating the same run is idempotent
        digests.append((base.digest, treat.digest, compare(base.digest, treat.digest, s).digest))
    assert digests[0] == digests[1]


# Pinned digests: a change means stored evaluations would differ (semantic change or drift).
GOLDEN = {
    "evaluation": "sha256:16b37d931f4cbb7f66631bbae7129e277274f4516a6275d2408932b874e41824",
    "comparison": "sha256:88240f68377a7ea93519b0f155b4b1e61449f4469569c649f1f8de81ea412079",
}


def test_golden_digests(store: ArtifactStore) -> None:
    c = comparison(store, DRIFT, "statement-v1", "contaminate")
    assert (c.baseline, c.digest) == (GOLDEN["evaluation"], GOLDEN["comparison"])


def test_evaluations_persist_and_reload(store: ArtifactStore) -> None:
    ev = run_and_evaluate(store, grid(RELOCATION, "statement-v1")["baseline"])
    assert store.get_record(Evaluation, ev.digest) == ev
    assert ev.digest in store.digests("Evaluation")
    assert Evaluation.model_validate_json(ev.canonical()) == ev


def test_tampered_evaluation_is_detected(store: ArtifactStore) -> None:
    ev = run_and_evaluate(store, grid(RELOCATION, "statement-v1")["baseline"])
    h = ev.digest.removeprefix("sha256:")
    path = store.root / "objects" / h[:2] / h[2:]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b'"correct"', b'"stale"', 1))
    with pytest.raises(ArtifactIntegrityError):
        store.get_record(Evaluation, ev.digest)


def test_forged_classification_counts_are_rejected(store: ArtifactStore) -> None:
    ev = run_and_evaluate(store, grid(RELOCATION, "statement-v1")["baseline"])
    dumped = ev.model_dump(mode="json")
    m = next(m for m in dumped["measurements"] if m["name"] == "known.correct")
    m["proportion"]["numerator"] += 1
    with pytest.raises(ValidationError):
        Evaluation.model_validate(dumped)


def test_inconsistent_run_artifacts_are_refused(store: ArtifactStore) -> None:
    runs = grid(RELOCATION, "statement-v1")
    a = execute(runs["baseline"], store)
    b = execute(runs["drop"], store)
    franken = RunRecord.model_validate(a.model_dump() | {"outcomes": b.outcomes})
    store.put_record(franken)
    with pytest.raises(EvaluationError):
        evaluate(franken.digest, store)


def test_unsupported_spec_is_refused(store: ArtifactStore) -> None:
    run = execute(grid(RELOCATION, "statement-v1")["baseline"], store)
    for spec in (
        EvaluationSpec(name="x", comparator="fuzzy-v1"),
        EvaluationSpec(name="x", classifier="llm-judge"),
    ):
        with pytest.raises(EvaluationError, match="unsupported"):
            evaluate(run.digest, store, spec)
    for bad in (
        {"readers": ()},
        {"readers": ("mention", "mention")},
        {"readers": ("fuzzy",)},
        {"confidence": 1.0},
    ):
        with pytest.raises(ValidationError):
            EvaluationSpec.model_validate({"name": "x"} | bad)


def test_reader_configuration_changes_results_explicitly(store: ArtifactStore) -> None:
    """Disabling the mention reader makes natural-language answers unscorable."""

    def paraphrase(trace: RetrievalTrace, state: MemoryState) -> Response:
        if not trace.selected:
            return Response(trace=trace.digest, responder="paraphrase-v1", output=None)
        top = next(m for m in state.memories if m.digest == trace.selected[0])
        value = (top.content or "").split(" = ", 1)[-1]
        return Response(
            trace=trace.digest,
            responder="paraphrase-v1",
            output=f"I believe it is {value}.",
            cited=(top.digest,),
        )

    registry = DEFAULT_REGISTRY.extend(responders={"paraphrase-v1": paraphrase})
    manifest = RunManifest.model_validate(
        grid(RELOCATION, "statement-v1")["baseline"].model_dump() | {"responder": "paraphrase-v1"}
    )
    lenient = run_and_evaluate(store, manifest, registry)
    strict = run_and_evaluate(
        store,
        manifest,
        registry,
        EvaluationSpec(name="strict", readers=("statement", "assignment")),
    )
    assert lenient.measurement("known.correct").proportion.numerator == 7
    assert lenient.measurement("unscorable").proportion.numerator == 0
    answered = [p for p in strict.probes if p.output]
    assert len(answered) == 8
    assert strict.measurement("unscorable").proportion.numerator == len(answered)
    assert {p.outcome for p in answered} == {OC.MALFORMED_OUTPUT}


def test_uncited_answers_are_judged_by_visible_claims(store: ArtifactStore) -> None:
    """A responder that cites nothing is judged on what it says; fabrication is unsupported."""

    def oracle(trace: RetrievalTrace, state: MemoryState) -> Response:
        answer = {"where is home": "home = Atlantis", "pet": "pet = dog"}.get(trace.query.text)
        return Response(trace=trace.digest, responder="oracle-v1", output=answer)

    registry = DEFAULT_REGISTRY.extend(responders={"oracle-v1": oracle})
    manifest = RunManifest.model_validate(
        grid(RELOCATION, "statement-v1")["baseline"].model_dump() | {"responder": "oracle-v1"}
    )
    with pytest.raises(ValidationError):
        execute(manifest, store, registry)  # a response cannot answer without citing
