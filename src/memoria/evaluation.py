"""Evaluation: from a stored run to per-probe judgements, measurements and comparisons.

Layers, each consuming only the one before:

1. observation — the run's stored probe outcome, trace, response and memory log;
2. comparison (:mod:`memoria.comparison`) — what the output asserts and whether it
   agrees with the expectation;
3. classification (:mod:`memoria.taxonomy`) — which outcome class, by which rule, with
   which claims as evidence;
4. measurement — proportions whose numerator and denominator are listed probe ids;
5. statistics (:mod:`memoria.statistics`) — intervals and paired tests.

:func:`evaluate` turns a run into an :class:`Evaluation`; :func:`compare` turns two
evaluations that differ in exactly one manifest variable into a :class:`RunComparison`.
Both are deterministic records stored as artifacts; the run's artifacts are only read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from memoria.artifacts import ArtifactStore
from memoria.comparison import READERS, Agreement, Reading, agree, normalize, read
from memoria.consolidation import Hierarchy
from memoria.core import (
    Dataset,
    DerivedMemory,
    Digest,
    Expectation,
    ExpectationStatus,
    FormationDecision,
    FormationReason,
    MemoryVersion,
    Record,
    Response,
    RetrievalTrace,
    RunManifest,
    RunOutcomes,
    RunRecord,
    UTCDatetime,
    beliefs,
    quantize,
)
from memoria.hybrid import HybridTrace
from memoria.statistics import (
    Proportion,
    holm,
    mcnemar_exact,
    min_achievable_p,
    newcombe_paired,
    significant,
)
from memoria.store import MemoryLog
from memoria.taxonomy import (
    Category,
    Claim,
    Classification,
    Locus,
    Outcome,
    ProbeEvidence,
    claims,
    classify,
    subjects,
)

COMPARATOR = "tokens-v1"
CLASSIFIER = "provenance-v1"
# The experimental variables of a manifest. The retriever and the representation it
# embeds with form one variable, "retrieval": a representation has no effect except
# through a retriever signal, and the runner requires them to be declared together. The
# graph policy belongs to it for the same reason (only graph signals and the graph
# generator read the graph).
MANIFEST_VARIABLES: dict[str, tuple[str, ...]] = {
    "dataset": ("dataset",),
    "interventions": ("interventions",),
    "policy": ("policy",),
    "retrieval": ("retriever", "representation", "retrieval_policy", "graph_policy"),
    "consolidation": ("consolidation_policy",),
    "forgetting": ("forgetting_policy",),
    "responder": ("responder",),
}


class EvaluationError(RuntimeError):
    """The evaluation cannot proceed: inconsistent inputs or an unsupported specification."""


class EvaluationSpec(Record):
    """Everything that determines an evaluation of a given run."""

    name: str = Field(min_length=1)
    readers: tuple[str, ...] = READERS
    comparator: str = COMPARATOR
    classifier: str = CLASSIFIER
    confidence: float = Field(default=0.95, gt=0, lt=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if not self.readers or len(set(self.readers)) != len(self.readers):
            raise ValueError("readers must be a non-empty sequence without repeats")
        if set(self.readers) - set(READERS):
            raise ValueError(f"readers must be among {READERS}")
        return self


DEFAULT_SPEC = EvaluationSpec(name="default")


class MissingCause:
    """Why the claim supporting an expected value was not in the queried state."""

    NOT_INGESTED = "not_ingested"  # its experience never reached the system (e.g. dropped)
    NOT_YET_INGESTED = "not_yet_ingested"  # ingested only after known_at (e.g. delayed)
    REJECTED_BY_LOG = "rejected_by_log"  # the log refused the step
    REJECTED_BY_POLICY = "rejected_by_policy"  # the policy decided not to change memory
    SUPERSEDED = "superseded"  # it entered memory but was replaced or retracted by known_at
    OUTSIDE_VALIDITY = "outside_validity"  # still believed, but not valid at valid_at

    ALL = (
        NOT_INGESTED,
        NOT_YET_INGESTED,
        REJECTED_BY_LOG,
        REJECTED_BY_POLICY,
        SUPERSEDED,
        OUTSIDE_VALIDITY,
    )


class RetrievalEvidence(Record):
    candidates: int = Field(ge=0)  # memories in the queried state
    eligible: int = Field(ge=0)
    selected: int = Field(ge=0)
    expected_rank: int | None = Field(default=None, ge=1)  # of a memory asserting a value
    expected_selected: bool = False
    subject_values: int = Field(ge=0)  # distinct values held about the probe's subjects


class ProbeEvaluation(Record):
    """One probe, judged: the raw observation, its reading, and its classification."""

    probe: str
    text: str
    valid_at: UTCDatetime
    known_at: UTCDatetime
    expected: Expectation
    trace: Digest
    response: Digest
    output: str | None
    cited: tuple[Digest, ...]
    subjects: tuple[str, ...]
    reading: Reading
    agreement: Agreement
    classification: Classification
    retrieval: RetrievalEvidence
    missing_cause: str | None = None
    missing_reason: FormationReason | None = None  # the policy's reason, if it rejected

    @property
    def outcome(self) -> Outcome:
        return self.classification.outcome


class Measurement(Record):
    """A proportion with its population definition and the exact probes counted."""

    name: str
    population: str  # what the denominator is, in words
    over: tuple[str, ...]  # probe ids in the denominator
    counted: tuple[str, ...]  # probe ids in the numerator
    proportion: Proportion

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if not set(self.counted) <= set(self.over):
            raise ValueError(f"{self.name}: counted probes are outside the population")
        p = self.proportion
        if (p.numerator, p.denominator) != (len(self.counted), len(self.over)):
            raise ValueError(f"{self.name}: counts do not match the listed probes")
        return self


class Evaluation(Record):
    spec: EvaluationSpec
    run: Digest
    manifest: Digest
    dataset: Digest  # the authored dataset (ground truth)
    effective: Digest  # the dataset the system received
    probes: tuple[ProbeEvaluation, ...]
    measurements: tuple[Measurement, ...]

    def measurement(self, name: str) -> Measurement:
        return next(m for m in self.measurements if m.name == name)


# --- per-probe evaluation --------------------------------------------------------------------


def _missing_cause(
    support: tuple[Claim, ...],
    effective: Dataset,
    decisions: Sequence[FormationDecision],
    log: MemoryLog,
    known_at: datetime,
) -> tuple[str | None, FormationReason | None]:
    if not support:
        return None, None
    s = support[-1]
    steps = [st for st in effective.steps if st.experience.digest == s.experience]
    if not steps:
        return MissingCause.NOT_INGESTED, None
    if all(st.recorded_at > known_at for st in steps):
        return MissingCause.NOT_YET_INGESTED, None
    made = [d for d in decisions if d.experience == s.experience and d.recorded_at <= known_at]
    if not made:
        return MissingCause.REJECTED_BY_LOG, None
    if not any(d.version for d in made):
        return MissingCause.REJECTED_BY_POLICY, made[0].reason
    for d in made:
        if d.version is None or d.memory_id is None:
            continue
        chain = [v for v in log.history(d.memory_id) if v.recorded_at <= known_at]
        if any(b.version.digest == d.version for b in beliefs(chain)):
            return MissingCause.OUTSIDE_VALIDITY, None
    return MissingCause.SUPERSEDED, None


def _evaluate_probe(
    spec: EvaluationSpec,
    outcome_probe: str,
    probe_index: dict[str, int],
    effective: Dataset,
    authored_claims: dict[str, Claim],
    run_claims: dict[str, Claim],
    vocabulary: list[str],
    ranked: Sequence[str],
    selected: Sequence[str],
    trace: str,
    response: Response,
    log: MemoryLog,
    versions: Mapping[str, MemoryVersion | DerivedMemory],
    decisions: Sequence[FormationDecision],
    eligible: int,
) -> ProbeEvaluation:
    """Judge one probe. ``ranked`` is the retrieval's order over the memories it
    considered (a Phase 2 trace: every state memory; a hybrid trace: every survivor)."""
    probe = effective.probes[probe_index[outcome_probe]]
    keys = subjects(probe, authored_claims)
    history = tuple(
        sorted(
            (
                c
                for c in authored_claims.values()
                if c.key in keys and c.recorded_at <= probe.known_at
            ),
            key=lambda c: (c.occurred_at, c.experience),
        )
    )

    def claims_of(version_digest: str) -> list[Claim]:
        version = versions.get(version_digest)
        if version is None:
            raise EvaluationError(f"probe {probe.id}: version {version_digest} is not in the log")
        return [run_claims[x] for x in version.derived_from if x in run_claims]

    wanted = {normalize(v) for v in probe.expected.values}
    rank = None
    held_values = set()
    for i, candidate in enumerate(ranked, 1):
        for c in claims_of(candidate):
            if c.key in keys and c.value:
                held_values.add(normalize(c.value))
                if rank is None and normalize(c.value) in wanted:
                    rank = i
    expected_selected = rank is not None and ranked[rank - 1] in selected

    reading = read(response.output, spec.readers, vocabulary)
    agreement = agree(reading, probe.expected)
    cited = tuple(c for d in response.cited for c in claims_of(d))
    visible = {k: c for k, c in run_claims.items() if c.recorded_at <= probe.known_at}
    classification = classify(
        ProbeEvidence(
            probe=probe,
            reading=reading,
            agreement=agreement,
            subjects=keys,
            history=history,
            cited=cited,
            cited_versions=len(response.cited),
            visible=visible,
            expected_in_state=rank is not None,
        )
    )
    cause, reason = (None, None)
    if (
        probe.expected.status is ExpectationStatus.KNOWN
        and rank is None
        and classification.outcome.category is Category.FAILURE
    ):
        cause, reason = _missing_cause(
            classification.support, effective, decisions, log, probe.known_at
        )
    return ProbeEvaluation(
        probe=probe.id,
        text=probe.text,
        valid_at=probe.valid_at,
        known_at=probe.known_at,
        expected=probe.expected,
        trace=trace,
        response=response.digest,
        output=response.output,
        cited=response.cited,
        subjects=keys,
        reading=reading,
        agreement=agreement,
        classification=classification,
        retrieval=RetrievalEvidence(
            candidates=len(ranked),
            eligible=eligible,
            selected=len(selected),
            expected_rank=rank,
            expected_selected=expected_selected,
            subject_values=len(held_values),
        ),
        missing_cause=cause,
        missing_reason=reason,
    )


# --- measurements ------------------------------------------------------------------------------

Predicate = Callable[[ProbeEvaluation], bool]


def _is(outcome: Outcome) -> Predicate:
    return lambda p: p.outcome is outcome


def _at(locus: Locus) -> Predicate:
    return lambda p: p.classification.locus is locus


def _caused_by(cause: str) -> Predicate:
    return lambda p: p.missing_cause == cause


def measurements(probes: Sequence[ProbeEvaluation], confidence: float) -> tuple[Measurement, ...]:
    """The fixed, documented set of measurements. Every outcome class is reported, even
    at zero, so the taxonomy's shares always add up over all probes."""
    out: list[Measurement] = []

    def measure(
        name: str,
        population: str,
        over: Sequence[ProbeEvaluation],
        hit: Callable[[ProbeEvaluation], bool],
    ) -> None:
        ids = tuple(p.probe for p in over)
        counted = tuple(p.probe for p in over if hit(p))
        out.append(
            Measurement(
                name=name,
                population=population,
                over=ids,
                counted=counted,
                proportion=Proportion.of(len(counted), len(ids), confidence),
            )
        )

    def status(s: ExpectationStatus) -> list[ProbeEvaluation]:
        return [p for p in scorable if p.expected.status is s]

    everything = list(probes)
    scorable = [p for p in everything if p.outcome.category is not Category.UNSCORABLE]
    known = status(ExpectationStatus.KNOWN)
    held = [p for p in known if p.retrieval.expected_rank is not None]
    failures = [p for p in everything if p.outcome.category is Category.FAILURE]
    unheld_known_failures = [
        p
        for p in failures
        if p.expected.status is ExpectationStatus.KNOWN and p.retrieval.expected_rank is None
    ]  # by definition, not by whether a cause was found: undetermined causes stay visible

    measure("coverage", "all probes", everything, lambda p: p.agreement is not Agreement.ABSTAINED)
    measure(
        "unscorable", "all probes", everything, lambda p: p.outcome.category is Category.UNSCORABLE
    )
    measure(
        "known.correct",
        "scorable probes expecting a known value",
        known,
        lambda p: p.outcome is Outcome.CORRECT,
    )
    measure(
        "known.abstained",
        "scorable probes expecting a known value",
        known,
        lambda p: p.agreement is Agreement.ABSTAINED,
    )
    measure(
        "known.expected_in_state",
        "scorable probes expecting a known value",
        known,
        lambda p: p.retrieval.expected_rank is not None,
    )
    measure(
        "known.expected_selected_given_in_state",
        "scorable known-value probes whose expected memory was in the queried state",
        held,
        lambda p: p.retrieval.expected_selected,
    )
    measure(
        "unknown.abstained",
        "scorable probes expecting nothing (unknown)",
        status(ExpectationStatus.UNKNOWN),
        lambda p: p.outcome is Outcome.CORRECT_ABSTENTION,
    )
    measure(
        "contested.answered",
        "scorable probes with contested ground truth",
        status(ExpectationStatus.CONTESTED),
        lambda p: p.outcome is Outcome.CONTESTED_ANSWERED,
    )
    for outcome in Outcome:
        measure(
            f"outcome.{outcome.value}",
            "all probes",
            everything,
            _is(outcome),
        )
    for locus in (Locus.RETRIEVAL, Locus.MEMORY):
        measure(
            f"failure.locus.{locus.value}",
            "failed probes",
            failures,
            _at(locus),
        )
    for cause in MissingCause.ALL:
        measure(
            f"failure.cause.{cause}",
            "failed known-value probes whose expected memory was not in the queried state",
            unheld_known_failures,
            _caused_by(cause),
        )
    return tuple(out)


def evaluate(run: str, store: ArtifactStore, spec: EvaluationSpec = DEFAULT_SPEC) -> Evaluation:
    """Evaluate a stored run. Reads the run's artifacts; stores and returns the evaluation."""
    if (spec.comparator, spec.classifier) != (COMPARATOR, CLASSIFIER):
        raise EvaluationError(
            f"unsupported comparator/classifier {spec.comparator!r}/{spec.classifier!r}; "
            f"this version implements {COMPARATOR!r}/{CLASSIFIER!r}"
        )
    record = store.get_record(RunRecord, run)
    manifest = store.get_record(RunManifest, record.manifest)
    authored = store.get_record(Dataset, manifest.dataset)
    effective = store.get_record(Dataset, record.dataset)
    outcomes = store.get_record(RunOutcomes, record.outcomes)
    if [o.probe for o in outcomes.probes] != [p.id for p in effective.probes]:
        raise EvaluationError("run outcomes do not cover the dataset's probes in order")
    if authored.probes != effective.probes:
        raise EvaluationError("the effective dataset's probes differ from the authored ones")

    authored_claims = claims(authored, authored)
    run_claims = claims(effective, authored)
    vocabulary = sorted({c.value for c in run_claims.values() if c.value})
    index = {p.id: i for i, p in enumerate(effective.probes)}
    evaluated = []
    hybrid = manifest.retrieval_policy is not None
    with MemoryLog.load(store.get(record.log)) as log:
        log.verify()
        decisions = log.records(FormationDecision)
        versions: dict[str, MemoryVersion | DerivedMemory] = {v.digest: v for v in log.versions()}
        for h in record.hierarchies:
            versions |= {m.digest: m for m in store.get_record(Hierarchy, h).memories}
        for o in outcomes.probes:
            probe = effective.probes[index[o.probe]]
            if hybrid:
                hybrid_trace = store.get_record(HybridTrace, o.trace)
                response = store.get_record(Response, o.response)
                q = hybrid_trace.query
                asked = (q.text, q.valid_at, q.known_at, q.limit, q.key)
                if asked != (probe.text, probe.valid_at, probe.known_at, probe.limit, probe.key):
                    raise EvaluationError(f"probe {o.probe}: trace answers a different query")
                ranked = [r.version for r in hybrid_trace.ranking]
                selected: Sequence[str] = hybrid_trace.selected
                eligible = len(ranked)
                trace_digest = hybrid_trace.digest
            else:
                trace = log.record(RetrievalTrace, o.trace)
                logged = log.record(Response, o.response)
                if trace is None or logged is None:
                    raise EvaluationError(
                        f"probe {o.probe}: trace or response missing from the log"
                    )
                response = logged
                if trace.query != probe.query:
                    raise EvaluationError(f"probe {o.probe}: trace answers a different query")
                ranked = [c.version for c in trace.candidates]
                selected = trace.selected
                eligible = sum(c.eligible for c in trace.candidates)
                trace_digest = trace.digest
            if (response.trace, response.output, response.cited) != (
                trace_digest,
                o.output,
                o.cited,
            ):
                raise EvaluationError(f"probe {o.probe}: outcome disagrees with the response")
            evaluated.append(
                _evaluate_probe(
                    spec,
                    o.probe,
                    index,
                    effective,
                    authored_claims,
                    run_claims,
                    vocabulary,
                    ranked,
                    selected,
                    trace_digest,
                    response,
                    log,
                    versions,
                    decisions,
                    eligible,
                )
            )
    evaluation = Evaluation(
        spec=spec,
        run=record.digest,
        manifest=manifest.digest,
        dataset=authored.digest,
        effective=effective.digest,
        probes=tuple(evaluated),
        measurements=measurements(evaluated, spec.confidence),
    )
    store.put_record(evaluation)
    return evaluation


# --- paired comparison ---------------------------------------------------------------------------


class PairedMeasure(Record):
    """One indicator compared on the same probes under baseline and treatment.

    Cells follow Newcombe: ``both``, ``treatment_only``, ``baseline_only``, ``neither``.
    ``difference`` is treatment minus baseline, with Newcombe's method-10 interval. The
    exact McNemar p-value is Holm-adjusted within ``family``. ``underpowered`` means even
    the most one-sided split of the discordant pairs could not reach ``alpha``.
    """

    name: str
    family: str
    population: str
    pairs: tuple[str, ...]
    both: int = Field(ge=0)
    treatment_only: int = Field(ge=0)
    baseline_only: int = Field(ge=0)
    neither: int = Field(ge=0)
    baseline: Proportion
    treatment: Proportion
    difference: float | None
    low: float | None
    high: float | None
    p_value: float | None
    p_adjusted: float | None
    min_achievable_p: float | None
    underpowered: bool

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        cells = self.both + self.treatment_only + self.baseline_only + self.neither
        if cells != len(self.pairs):
            raise ValueError(f"{self.name}: cells do not add up to the number of pairs")
        if (self.baseline.numerator, self.treatment.numerator) != (
            self.both + self.baseline_only,
            self.both + self.treatment_only,
        ):
            raise ValueError(f"{self.name}: marginals do not follow from the cells")
        return self


class Transition(Record):
    """Probes whose outcome moved from ``baseline`` to ``treatment``."""

    baseline: Outcome
    treatment: Outcome
    probes: tuple[str, ...]


class RunComparison(Record):
    spec: EvaluationSpec
    baseline: Digest  # Evaluation digests
    treatment: Digest
    variable: str  # the single manifest field that differs
    measures: tuple[PairedMeasure, ...]
    transitions: tuple[Transition, ...]

    def measure(self, name: str) -> PairedMeasure:
        return next(m for m in self.measures if m.name == name)


def _paired(
    name: str,
    family: str,
    population: str,
    pairs: Sequence[tuple[ProbeEvaluation, ProbeEvaluation]],
    hit: Callable[[ProbeEvaluation], bool],
    spec: EvaluationSpec,
) -> PairedMeasure:
    cells = {"both": 0, "treatment_only": 0, "baseline_only": 0, "neither": 0}
    for b, t in pairs:
        cells[
            {
                (True, True): "both",
                (False, True): "treatment_only",
                (True, False): "baseline_only",
            }.get((hit(b), hit(t)), "neither")
        ] += 1
    n = len(pairs)
    diff = newcombe_paired(
        cells["both"],
        cells["treatment_only"],
        cells["baseline_only"],
        cells["neither"],
        spec.confidence,
    )
    discordant = cells["treatment_only"] + cells["baseline_only"]
    floor = min_achievable_p(discordant) if n else None
    return PairedMeasure(
        name=name,
        family=family,
        population=population,
        pairs=tuple(b.probe for b, _ in pairs),
        **cells,
        baseline=Proportion.of(cells["both"] + cells["baseline_only"], n, spec.confidence),
        treatment=Proportion.of(cells["both"] + cells["treatment_only"], n, spec.confidence),
        difference=quantize(diff[0]) if diff else None,
        low=quantize(diff[1]) if diff else None,
        high=quantize(diff[2]) if diff else None,
        p_value=significant(mcnemar_exact(cells["baseline_only"], cells["treatment_only"]))
        if n
        else None,
        p_adjusted=None,
        min_achievable_p=significant(floor) if floor is not None else None,
        underpowered=floor is None or floor > spec.alpha,
    )


def compare(baseline: str, treatment: str, store: ArtifactStore) -> RunComparison:
    """Paired comparison of two evaluations of runs that differ in one manifest variable."""
    b = store.get_record(Evaluation, baseline)
    t = store.get_record(Evaluation, treatment)
    if b.spec != t.spec:
        raise EvaluationError("evaluations use different specifications")
    mb = store.get_record(RunManifest, b.manifest)
    mt = store.get_record(RunManifest, t.manifest)
    changed = [
        name
        for name, fields in MANIFEST_VARIABLES.items()
        if any(getattr(mb, f) != getattr(mt, f) for f in fields)
    ]
    covered = {f for fields in MANIFEST_VARIABLES.values() for f in fields} | {"name"}
    if set(RunManifest.model_fields) - covered:  # a new manifest field must be classified
        raise EvaluationError("manifest has fields the comparison does not account for")
    if len(changed) != 1:
        raise EvaluationError(
            "a controlled comparison changes exactly one manifest variable; "
            f"changed: {changed or 'none'}"
        )
    key = [(p.probe, p.expected, p.text, p.valid_at, p.known_at) for p in b.probes]
    if key != [(p.probe, p.expected, p.text, p.valid_at, p.known_at) for p in t.probes]:
        raise EvaluationError("paired comparison needs identical probes in both evaluations")
    pairs = list(zip(b.probes, t.probes, strict=True))

    def scorable_with(status: ExpectationStatus) -> list[tuple[ProbeEvaluation, ProbeEvaluation]]:
        return [
            (x, y)
            for x, y in pairs
            if x.expected.status is status
            and Category.UNSCORABLE not in (x.outcome.category, y.outcome.category)
        ]

    spec = b.spec
    measures = [
        _paired(
            "known.correct",
            "primary",
            "known-value probes scorable in both runs",
            scorable_with(ExpectationStatus.KNOWN),
            lambda p: p.outcome is Outcome.CORRECT,
            spec,
        ),
        _paired(
            "unknown.abstained",
            "primary",
            "unknown-value probes scorable in both runs",
            scorable_with(ExpectationStatus.UNKNOWN),
            lambda p: p.outcome is Outcome.CORRECT_ABSTENTION,
            spec,
        ),
        _paired(
            "coverage",
            "primary",
            "all probes",
            pairs,
            lambda p: p.agreement is not Agreement.ABSTAINED,
            spec,
        ),
        *(
            _paired(
                f"outcome.{o.value}",
                "outcome",
                "all probes",
                pairs,
                _is(o),
                spec,
            )
            for o in Outcome
        ),
        *(
            _paired(
                f"failure.locus.{lo.value}",
                "locus",
                "all probes",
                pairs,
                _at(lo),
                spec,
            )
            for lo in (Locus.RETRIEVAL, Locus.MEMORY)
        ),
    ]
    adjusted: list[PairedMeasure] = []
    for family in dict.fromkeys(m.family for m in measures):
        members = [m for m in measures if m.family == family]
        tested = [m for m in members if m.p_value is not None]
        holm_p = dict(
            zip((m.name for m in tested), holm([m.p_value or 0.0 for m in tested]), strict=True)
        )
        adjusted += [
            m.model_copy(
                update={"p_adjusted": significant(holm_p[m.name]) if m.name in holm_p else None}
            )
            for m in members
        ]
    transitions: dict[tuple[Outcome, Outcome], list[str]] = {}
    for x, y in pairs:
        if x.outcome is not y.outcome:
            transitions.setdefault((x.outcome, y.outcome), []).append(x.probe)
    comparison = RunComparison(
        spec=spec,
        baseline=b.digest,
        treatment=t.digest,
        variable=changed[0],
        measures=tuple(adjusted),
        transitions=tuple(
            Transition(baseline=o1, treatment=o2, probes=tuple(ps))
            for (o1, o2), ps in sorted(transitions.items())
        ),
    )
    store.put_record(comparison)
    return comparison
