"""Failure taxonomy: claims, relations between claims, and probe-outcome classification.

The fact model is the statement language (``set``/``correct``/``forget``): each such
experience is a :class:`Claim`. Two claims about the same key are related by *when they
occurred*, never by when they were ingested — a newer report is not more true, and a
temporal change is not a contradiction.

A probe outcome is classified by an ordered rule list (:func:`classify`). Answer text
decides *whether* an answer matches (:mod:`memoria.comparison`); provenance decides
*why* it failed: the cited version's source experience yields the observed claim, which
is related to the claim(s) supporting the expected value. Every classification records
the rule that fired.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Literal, NamedTuple

from memoria.comparison import Agreement, Reading, normalize
from memoria.core import Dataset, Digest, ExpectationStatus, Probe, Record, UTCDatetime
from memoria.formation import parse_statement


class Claim(Record):
    """A statement experience read as a claim about one key.

    ``recorded_at`` is the earliest ingestion of the experience in the dataset it was
    read from; ``introduced`` marks claims absent from the authored dataset, i.e. added
    by an intervention (contamination or injection).
    """

    experience: Digest
    source: str
    verb: Literal["set", "correct", "forget"]
    key: str
    value: str | None
    occurred_at: UTCDatetime
    recorded_at: UTCDatetime
    introduced: bool


def claims(dataset: Dataset, authored: Dataset) -> dict[str, Claim]:
    """The dataset's statement experiences as claims, keyed by experience digest."""
    authored_experiences = {s.experience.digest for s in authored.steps}
    out: dict[str, Claim] = {}
    for step in dataset.steps:  # ingestion order: the first step is the earliest
        e = step.experience
        statement = parse_statement(e.content)
        if statement is None or e.digest in out:
            continue
        out[e.digest] = Claim(
            experience=e.digest,
            source=e.source,
            verb=statement.verb,  # type: ignore[arg-type]
            key=statement.key,
            value=statement.value,
            occurred_at=e.occurred_at,
            recorded_at=step.recorded_at,
            introduced=e.digest not in authored_experiences,
        )
    return out


class Relation(StrEnum):
    """How two claims relate. Symmetric; decided by key, value and occurrence time."""

    INDEPENDENT = "independent"  # different keys: compatible facts
    SAME = "same"  # same key and value: compatible (redundant)
    RETRACTION = "retraction"  # one of them is a ``forget``
    CONTRADICTION = "contradiction"  # same key, same instant, different values
    CORRECTION = "correction"  # different values; the later-occurring one is a ``correct``
    TEMPORAL_CHANGE = "temporal_change"  # different values at different times: a change


def relate(a: Claim, b: Claim) -> Relation:
    if a.key != b.key:
        return Relation.INDEPENDENT
    if "forget" in (a.verb, b.verb):
        return Relation.RETRACTION
    if normalize(a.value or "") == normalize(b.value or ""):  # set/correct carry values
        return Relation.SAME
    if a.occurred_at == b.occurred_at:
        return Relation.CONTRADICTION
    later = a if a.occurred_at > b.occurred_at else b
    return Relation.CORRECTION if later.verb == "correct" else Relation.TEMPORAL_CHANGE


class Category(StrEnum):
    SUCCESS = "success"
    NEUTRAL = "neutral"  # no right answer exists (contested ground truth)
    FAILURE = "failure"
    UNSCORABLE = "unscorable"  # the evaluator cannot judge the output


class Outcome(StrEnum):
    CORRECT = "correct"
    CORRECT_ABSTENTION = "correct_abstention"  # nothing supported, and none claimed
    CONTESTED_ANSWERED = "contested_answered"  # one of several equally supported values
    CONTESTED_ABSTAINED = "contested_abstained"
    MISSING_MEMORY = "missing_memory"  # abstained; the expected memory was not held
    RETRIEVAL_MISS = "retrieval_miss"  # abstained; the expected memory was held
    WRONG_MEMORY = "wrong_memory"  # answered from a memory about another subject
    CONTAMINATED = "contaminated"  # answered an intervention-introduced claim
    FORGOTTEN_RECALLED = "forgotten_recalled"  # answered a claim that was forgotten
    CORRECTION_FAILURE = "correction_failure"  # answered the claim a correction replaced
    STALE = "stale"  # answered an older claim superseded by a later-occurring one
    TEMPORAL_ERROR = "temporal_error"  # answered a claim not yet true at valid_at
    CONTRADICTION = "contradiction"  # answered a claim contradicting the truth at its instant
    UNSUPPORTED = "unsupported"  # the answer is backed by no visible claim or its citation
    AMBIGUOUS_OUTPUT = "ambiguous_output"
    MALFORMED_OUTPUT = "malformed_output"
    UNCLASSIFIABLE = "unclassifiable"  # the evidence needed to judge is missing

    @property
    def category(self) -> Category:
        return _CATEGORY.get(self, Category.FAILURE)


_CATEGORY = {
    Outcome.CORRECT: Category.SUCCESS,
    Outcome.CORRECT_ABSTENTION: Category.SUCCESS,
    Outcome.CONTESTED_ANSWERED: Category.NEUTRAL,
    Outcome.CONTESTED_ABSTAINED: Category.NEUTRAL,
    Outcome.AMBIGUOUS_OUTPUT: Category.UNSCORABLE,
    Outcome.MALFORMED_OUTPUT: Category.UNSCORABLE,
    Outcome.UNCLASSIFIABLE: Category.UNSCORABLE,
}


class Locus(StrEnum):
    """Where a failure arose: the right memory was held but not returned (retrieval), or
    the memory state itself was wrong (memory: formation, history, interventions)."""

    NONE = "none"  # not a failure
    RETRIEVAL = "retrieval"
    MEMORY = "memory"


class Classification(Record):
    outcome: Outcome
    rule: str  # identifier of the rule that fired (see ``classify``)
    locus: Locus
    observed: Claim | None = None  # the claim behind the answer, if identified
    support: tuple[Claim, ...] = ()  # the claim(s) supporting the expected value
    relation: Relation | None = None  # observed vs the latest support claim


class ProbeEvidence(NamedTuple):
    """Everything :func:`classify` may consider, assembled from the run's artifacts."""

    probe: Probe
    reading: Reading
    agreement: Agreement
    subjects: tuple[str, ...]  # keys the probe is about
    history: tuple[Claim, ...]  # authored claims on those keys visible at known_at
    cited: tuple[Claim, ...]  # claims behind the versions the response cited
    cited_versions: int
    visible: Mapping[str, Claim]  # claims the run had ingested by known_at
    expected_in_state: bool  # a queried memory asserted an expected value


def subjects(probe: Probe, authored: Mapping[str, Claim]) -> tuple[str, ...]:
    """Keys a probe is about: keys asserting an expected value, and keys whose tokens
    all occur in the probe text."""
    wanted = {normalize(v) for v in probe.expected.values}
    text = set(normalize(probe.text))
    return tuple(
        sorted(
            {c.key for c in authored.values() if c.value and normalize(c.value) in wanted}
            | {c.key for c in authored.values() if set(normalize(c.key)) <= text}
        )
    )


def support(evidence: ProbeEvidence) -> tuple[Claim, ...]:
    """The latest-occurring visible claims that assert an expected value."""
    wanted = {normalize(v) for v in evidence.probe.expected.values}
    matching = [c for c in evidence.history if c.value and normalize(c.value) in wanted]
    if not matching:
        return ()
    latest = max(c.occurred_at for c in matching)
    return tuple(c for c in matching if c.occurred_at == latest)


def _corrected(correction: Claim, history: tuple[Claim, ...]) -> Claim | None:
    """The claim a ``correct`` replaced: the latest earlier-occurring claim on its key."""
    earlier = [
        c
        for c in history
        if c.key == correction.key and c.verb != "forget" and c.occurred_at < correction.occurred_at
    ]
    return max(earlier, key=lambda c: c.occurred_at) if earlier else None


def classify(e: ProbeEvidence) -> Classification:
    """Classify one probe outcome. Rules are checked in the order written here."""
    status = e.probe.expected.status
    valid_at: datetime = e.probe.valid_at
    backing = support(e)
    held = Locus.RETRIEVAL if e.expected_in_state else Locus.MEMORY

    def result(
        outcome: Outcome,
        rule: str,
        locus: Locus | None = None,
        observed: Claim | None = None,
    ) -> Classification:
        if locus is None:
            locus = held if outcome.category is Category.FAILURE else Locus.NONE
        relation = relate(observed, backing[-1]) if observed and backing else None
        return Classification(
            outcome=outcome,
            rule=rule,
            locus=locus,
            observed=observed,
            support=backing,
            relation=relation,
        )

    a = e.agreement
    if a is Agreement.ABSTAINED:
        if status is ExpectationStatus.UNKNOWN:
            return result(Outcome.CORRECT_ABSTENTION, "abstained.nothing_supported")
        if status is ExpectationStatus.CONTESTED:
            return result(Outcome.CONTESTED_ABSTAINED, "abstained.contested")
        if e.expected_in_state:
            return result(Outcome.RETRIEVAL_MISS, "abstained.expected_held")
        return result(Outcome.MISSING_MEMORY, "abstained.expected_not_held")
    if a is Agreement.MATCH:
        if status is ExpectationStatus.CONTESTED:
            return result(Outcome.CONTESTED_ANSWERED, "match.contested")
        return result(Outcome.CORRECT, "match.expected")
    if a is Agreement.AMBIGUOUS:
        return result(Outcome.AMBIGUOUS_OUTPUT, "output.ambiguous", Locus.NONE)
    if a is Agreement.UNREADABLE:
        about = [c for c in e.cited if c.key in e.subjects]
        if e.cited_versions and not about:
            return result(
                Outcome.WRONG_MEMORY, "unreadable.cites_unrelated_memory", Locus.RETRIEVAL
            )
        if about and all(c.value is None for c in about):
            # The answer is a memory about the subject that asserts no value (a retraction):
            # semantically no value was claimed, so it is judged as a non-answer.
            if status is ExpectationStatus.UNKNOWN:
                return result(Outcome.CORRECT_ABSTENTION, "valueless.nothing_supported")
            if status is ExpectationStatus.CONTESTED:
                return result(Outcome.CONTESTED_ABSTAINED, "valueless.contested")
            if e.expected_in_state:
                return result(Outcome.RETRIEVAL_MISS, "valueless.expected_held")
            return result(Outcome.MISSING_MEMORY, "valueless.expected_not_held")
        return result(Outcome.MALFORMED_OUTPUT, "output.unreadable", Locus.NONE)

    # A mismatch: some value was asserted that is not expected.
    if status is not ExpectationStatus.UNKNOWN and not backing:
        return result(
            Outcome.UNCLASSIFIABLE, "mismatch.expectation_has_no_supporting_claim", Locus.NONE
        )
    answer = e.reading.tokens[0]
    observed = next((c for c in e.cited if c.value and normalize(c.value) == answer), None)
    if e.cited_versions and observed is None:
        return result(Outcome.UNSUPPORTED, "mismatch.answer_not_in_cited_evidence")
    if observed is None:
        candidates = [c for c in e.visible.values() if c.value and normalize(c.value) == answer]
        observed = max(candidates, key=lambda c: c.occurred_at) if candidates else None
    if observed is None:
        return result(Outcome.UNSUPPORTED, "mismatch.no_visible_claim_asserts_answer")
    if observed.key not in e.subjects:
        return result(Outcome.WRONG_MEMORY, "mismatch.other_subject", Locus.RETRIEVAL, observed)
    if observed.introduced:
        return result(Outcome.CONTAMINATED, "mismatch.intervention_introduced", observed=observed)
    if status is ExpectationStatus.UNKNOWN:
        forgotten = any(
            c.verb == "forget" and c.key == observed.key and c.occurred_at >= observed.occurred_at
            for c in e.history
        )
        if forgotten:
            return result(Outcome.FORGOTTEN_RECALLED, "mismatch.forgotten", Locus.MEMORY, observed)
        if observed.verb != "correct" and observed.occurred_at > valid_at:
            return result(Outcome.TEMPORAL_ERROR, "mismatch.not_yet_true", Locus.MEMORY, observed)
        return result(Outcome.UNSUPPORTED, "mismatch.nothing_supported", Locus.MEMORY, observed)
    # Corrections are retroactive, so only plain assertions can be "not yet true".
    if observed.verb != "correct" and observed.occurred_at > valid_at:
        return result(Outcome.TEMPORAL_ERROR, "mismatch.not_yet_true", observed=observed)
    for s in backing:
        replaced = _corrected(s, e.history) if s.verb == "correct" else None
        if (
            replaced is not None
            and replaced.value is not None
            and normalize(replaced.value) == answer
            and observed.occurred_at <= s.occurred_at
        ):
            return result(Outcome.CORRECTION_FAILURE, "mismatch.corrected_value", observed=observed)
    if any(relate(observed, s) is Relation.CONTRADICTION for s in backing):
        return result(Outcome.CONTRADICTION, "mismatch.same_instant", observed=observed)
    if all(observed.occurred_at < s.occurred_at for s in backing):
        return result(Outcome.STALE, "mismatch.superseded", observed=observed)
    return result(Outcome.UNSUPPORTED, "mismatch.newer_than_supported", observed=observed)
