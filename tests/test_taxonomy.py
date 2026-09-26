"""Answer comparison and failure classification, rule by rule."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from memoria.comparison import (
    READERS,
    Agreement,
    Reading,
    ReadingKind,
    agree,
    normalize,
    read,
)
from memoria.core import Dataset, Expectation, Experience, Probe, Step
from memoria.scenarios import UNKNOWN, contested, day, known
from memoria.taxonomy import (
    Category,
    Claim,
    Locus,
    Outcome,
    ProbeEvidence,
    Relation,
    claims,
    classify,
    relate,
    subjects,
)

VOCAB = ["Paris", "Berlin", "Berlin (contaminated)", "New York", "York", "Zoë"]


# --- comparison ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "reader", "key", "value"),
    [
        ("Paris", "mention", None, "Paris"),
        ("home = Paris", "assignment", "home", "Paris"),
        ("home#2 = Paris", "assignment", "home", "Paris"),
        ("set home = Paris", "statement", "home", "Paris"),
        ("correct home = Paris", "statement", "home", "Paris"),
        ("My home is Paris.", "mention", None, "Paris"),
        ("PARIS!", "mention", None, "Paris"),
        ("I moved to New York", "mention", None, "New York"),  # longest match absorbs "York"
        ("Berlin (contaminated)", "mention", None, "Berlin (contaminated)"),
        (
            "\uff3a\uff4f\uff45\u0308",
            "mention",
            None,
            "Zoë",
        ),  # NFKC: fullwidth, combining diaeresis
    ],
)
def test_reading_forms(output: str, reader: str, key: str | None, value: str) -> None:
    r = read(output, READERS, VOCAB)
    assert (r.kind, r.reader, r.key, r.values) == (ReadingKind.VALUE, reader, key, (value,))
    assert r.output == output  # the original is preserved
    assert r.tokens == (normalize(value),)


@pytest.mark.parametrize(
    ("output", "kind"),
    [
        (None, ReadingKind.ABSTAINED),
        ("", ReadingKind.UNREADABLE),
        ("   ", ReadingKind.UNREADABLE),
        ("no idea", ReadingKind.UNREADABLE),
        ("forget home", ReadingKind.UNREADABLE),  # a retraction asserts no value
        ("It was Paris, then Berlin", ReadingKind.AMBIGUOUS),
    ],
)
def test_non_value_readings(output: str | None, kind: ReadingKind) -> None:
    assert read(output, READERS, VOCAB).kind is kind


def test_reader_order_and_subsets_are_explicit() -> None:
    # With only the mention reader, memory content is read by vocabulary, keyless.
    r = read("home = Paris", ("mention",), VOCAB)
    assert (r.reader, r.key) == ("mention", None)
    # Without the mention reader, free text cannot be read at all.
    assert (
        read("My home is Paris", ("statement", "assignment"), VOCAB).kind is ReadingKind.UNREADABLE
    )
    with pytest.raises(ValueError, match="unknown readers"):
        read("x", ("fuzzy",), VOCAB)


def test_equality_is_token_sequence_not_similarity() -> None:
    assert normalize("  Berlin. ") == normalize("berlin")
    assert normalize("Berlin (contaminated)") != normalize("Berlin")
    assert normalize("New York") != normalize("York New")
    assert normalize("Straße") == normalize("STRASSE")  # case folding


@pytest.mark.parametrize(
    ("output", "expected", "agreement"),
    [
        ("home = Paris", known("Paris"), Agreement.MATCH),
        ("home = paris.", known("Paris"), Agreement.MATCH),
        ("home = Berlin", known("Paris"), Agreement.MISMATCH),
        ("home = Paris", UNKNOWN, Agreement.MISMATCH),  # anything asserted against unknown
        ("home = Paris", contested("Paris", "Berlin"), Agreement.MATCH),
        (None, known("Paris"), Agreement.ABSTAINED),
        ("Paris or Berlin", known("Paris"), Agreement.AMBIGUOUS),
        ("", known("Paris"), Agreement.UNREADABLE),
    ],
)
def test_agreement(output: str | None, expected: Expectation, agreement: Agreement) -> None:
    assert agree(read(output, READERS, VOCAB), expected) is agreement


def test_reading_invariants() -> None:
    with pytest.raises(ValidationError):
        Reading(output="x", kind=ReadingKind.VALUE)  # a value reading needs a value
    with pytest.raises(ValidationError):
        Reading(output=None, kind=ReadingKind.UNREADABLE)
    with pytest.raises(ValidationError):
        Reading(output="x", kind=ReadingKind.VALUE, values=("Paris",), tokens=(("london",),))


# --- claims and relations --------------------------------------------------------------------


def claim(
    key: str,
    value: str | None,
    occurred: float,
    verb: str = "set",
    *,
    recorded: float | None = None,
    introduced: bool = False,
) -> Claim:
    content = f"{verb} {key}" if value is None else f"{verb} {key} = {value}"
    e = Experience(source=f"s:{content}:{occurred}", content=content, occurred_at=day(occurred))
    return Claim(
        experience=e.digest,
        source=e.source,
        verb=verb,  # type: ignore[arg-type]
        key=key,
        value=value,
        occurred_at=day(occurred),
        recorded_at=day(occurred if recorded is None else recorded),
        introduced=introduced,
    )


@pytest.mark.parametrize(
    ("a", "b", "relation"),
    [
        (claim("home", "Paris", 0), claim("pet", "dog", 0), Relation.INDEPENDENT),
        (claim("home", "Paris", 0), claim("home", "paris", 5), Relation.SAME),
        (claim("home", "Paris", 0), claim("home", None, 5, "forget"), Relation.RETRACTION),
        (claim("home", "Paris", 5), claim("home", "Rome", 5), Relation.CONTRADICTION),
        (claim("home", "Paris", 0), claim("home", "Rome", 5, "correct"), Relation.CORRECTION),
        (claim("home", "Paris", 0), claim("home", "Rome", 5), Relation.TEMPORAL_CHANGE),
        # A correct that occurred *earlier* does not correct a later claim: that is change.
        (claim("home", "Paris", 0, "correct"), claim("home", "Rome", 5), Relation.TEMPORAL_CHANGE),
    ],
)
def test_relations_are_symmetric_and_time_based(a: Claim, b: Claim, relation: Relation) -> None:
    assert relate(a, b) is relation
    assert relate(b, a) is relation


def test_ingestion_order_does_not_decide_relations() -> None:
    early_but_late_reported = claim("home", "Paris", 0, recorded=100)
    later = claim("home", "Rome", 5, recorded=6)
    assert relate(early_but_late_reported, later) is Relation.TEMPORAL_CHANGE  # not correction


def test_claims_from_datasets() -> None:
    authored = Dataset(
        name="d",
        version="1",
        steps=(
            Step(
                experience=Experience(source="a", content="set home = Paris", occurred_at=day(0)),
                recorded_at=day(2),
            ),
            Step(
                experience=Experience(source="b", content="chatter", occurred_at=day(1)),
                recorded_at=day(3),
            ),
            Step(
                experience=Experience(source="a", content="set home = Paris", occurred_at=day(0)),
                recorded_at=day(9),
            ),
        ),
    )
    extra = Step(
        experience=Experience(source="x", content="forget home", occurred_at=day(4)),
        recorded_at=day(4),
    )
    derived = Dataset(
        name="d",
        version="1+x",
        steps=tuple(sorted((*authored.steps, extra), key=lambda s: s.recorded_at)),
    )
    found = claims(derived, authored)
    assert len(found) == 2  # chatter is not a claim; the duplicate counts once
    (paris, forget) = sorted(found.values(), key=lambda c: c.occurred_at)
    assert (paris.recorded_at, paris.introduced) == (day(2), False)  # earliest ingestion
    assert (forget.verb, forget.value, forget.introduced) == ("forget", None, True)


def test_subjects_come_from_values_and_probe_text() -> None:
    authored = {
        c.experience: c
        for c in [
            claim("home", "Paris", 0),
            claim("home.office", "Lyon", 0),
            claim("pet", "dog", 0),
        ]
    }

    def probe(text: str, expected: Expectation) -> Probe:
        return Probe(id="p", text=text, valid_at=day(1), known_at=day(1), expected=expected)

    assert subjects(probe("where is home", UNKNOWN), authored) == ("home",)
    assert subjects(probe("home office", UNKNOWN), authored) == ("home", "home.office")
    assert subjects(probe("anything", known("dog")), authored) == ("pet",)
    assert subjects(probe("zebra", UNKNOWN), authored) == ()


# --- classification --------------------------------------------------------------------------

PARIS = claim("home", "Paris", 0)
BERLIN = claim("home", "Berlin", 50)
ROME_CORRECTS_PARIS = claim("home", "Rome", 10, "correct")
FORGET_HOME = claim("home", None, 60, "forget")
PET = claim("pet", "dog", 0)
FAKE = claim("home", "Atlantis", 55, introduced=True)
FUTURE = claim("home", "Oslo", 200)
TWIN = claim("home", "Madrid", 50)


def evidence(
    expected: Expectation,
    output: str | None,
    *,
    cited: tuple[Claim, ...] = (),
    history: tuple[Claim, ...] = (PARIS, BERLIN),
    held: bool = False,
    valid: float = 100,
    cited_versions: int | None = None,
) -> ProbeEvidence:
    probe = Probe(id="p", text="home", valid_at=day(valid), known_at=day(300), expected=expected)
    vocab = [c.value for c in (*history, *cited, PET, FAKE, FUTURE) if c.value]
    reading = read(output, READERS, vocab)
    everything = (*history, *cited, PET, FAKE, FUTURE)
    return ProbeEvidence(
        probe=probe,
        reading=reading,
        agreement=agree(reading, expected),
        subjects=("home",),
        history=history,
        cited=cited,
        cited_versions=len(cited) if cited_versions is None else cited_versions,
        visible={c.experience: c for c in everything},
        expected_in_state=held,
    )


CASES = [
    # (id, evidence, outcome, rule, locus)
    (
        "correct",
        evidence(known("Berlin"), "home = Berlin", cited=(BERLIN,)),
        Outcome.CORRECT,
        "match.expected",
        Locus.NONE,
    ),
    (
        "correct-natural-language",
        evidence(known("Berlin"), "Home is Berlin now."),
        Outcome.CORRECT,
        "match.expected",
        Locus.NONE,
    ),
    (
        "abstain-unknown",
        evidence(UNKNOWN, None),
        Outcome.CORRECT_ABSTENTION,
        "abstained.nothing_supported",
        Locus.NONE,
    ),
    (
        "abstain-contested",
        evidence(contested("Berlin", "Madrid"), None, history=(PARIS, BERLIN, TWIN)),
        Outcome.CONTESTED_ABSTAINED,
        "abstained.contested",
        Locus.NONE,
    ),
    (
        "answer-contested",
        evidence(
            contested("Berlin", "Madrid"),
            "home = Madrid",
            cited=(TWIN,),
            history=(PARIS, BERLIN, TWIN),
        ),
        Outcome.CONTESTED_ANSWERED,
        "match.contested",
        Locus.NONE,
    ),
    (
        "missing",
        evidence(known("Berlin"), None),
        Outcome.MISSING_MEMORY,
        "abstained.expected_not_held",
        Locus.MEMORY,
    ),
    (
        "retrieval-miss",
        evidence(known("Berlin"), None, held=True),
        Outcome.RETRIEVAL_MISS,
        "abstained.expected_held",
        Locus.RETRIEVAL,
    ),
    (
        "stale",
        evidence(known("Berlin"), "home = Paris", cited=(PARIS,), held=True),
        Outcome.STALE,
        "mismatch.superseded",
        Locus.RETRIEVAL,
    ),
    (
        "stale-not-held",
        evidence(known("Berlin"), "home = Paris", cited=(PARIS,)),
        Outcome.STALE,
        "mismatch.superseded",
        Locus.MEMORY,
    ),
    (
        "temporal",
        evidence(known("Paris"), "home = Berlin", cited=(BERLIN,), valid=20),
        Outcome.TEMPORAL_ERROR,
        "mismatch.not_yet_true",
        Locus.MEMORY,
    ),
    (
        "correction-failure",
        evidence(
            known("Rome"),
            "home = Paris",
            cited=(PARIS,),
            history=(PARIS, ROME_CORRECTS_PARIS),
            valid=5,
        ),
        Outcome.CORRECTION_FAILURE,
        "mismatch.corrected_value",
        Locus.MEMORY,
    ),
    (
        "contradiction",
        evidence(known("Berlin"), "home = Madrid", cited=(TWIN,), history=(PARIS, BERLIN)),
        Outcome.CONTRADICTION,
        "mismatch.same_instant",
        Locus.MEMORY,
    ),
    (
        "contaminated",
        evidence(known("Berlin"), "home = Atlantis", cited=(FAKE,)),
        Outcome.CONTAMINATED,
        "mismatch.intervention_introduced",
        Locus.MEMORY,
    ),
    (
        "wrong-memory",
        evidence(known("Berlin"), "pet = dog", cited=(PET,), held=True),
        Outcome.WRONG_MEMORY,
        "mismatch.other_subject",
        Locus.RETRIEVAL,
    ),
    (
        "forgotten-recalled",
        evidence(UNKNOWN, "home = Berlin", cited=(BERLIN,), history=(PARIS, BERLIN, FORGET_HOME)),
        Outcome.FORGOTTEN_RECALLED,
        "mismatch.forgotten",
        Locus.MEMORY,
    ),
    (
        "unknown-temporal",
        evidence(UNKNOWN, "home = Oslo", cited=(FUTURE,)),
        Outcome.TEMPORAL_ERROR,
        "mismatch.not_yet_true",
        Locus.MEMORY,
    ),
    (
        "unknown-other-subject",
        evidence(UNKNOWN, "pet = dog", cited=(PET,)),
        Outcome.WRONG_MEMORY,
        "mismatch.other_subject",
        Locus.RETRIEVAL,
    ),
    (
        "not-in-citation",
        evidence(known("Berlin"), "home = Paris", cited=(BERLIN,)),
        Outcome.UNSUPPORTED,
        "mismatch.answer_not_in_cited_evidence",
        Locus.MEMORY,
    ),
    (
        "no-claim-at-all",
        evidence(known("Berlin"), "home = Lisbon", cited_versions=0),
        Outcome.UNSUPPORTED,
        "mismatch.no_visible_claim_asserts_answer",
        Locus.MEMORY,
    ),
    (
        "newer-than-truth",
        evidence(
            known("Paris"), "home = Berlin", cited=(BERLIN,), history=(PARIS, BERLIN), valid=100
        ),
        Outcome.UNSUPPORTED,
        "mismatch.newer_than_supported",
        Locus.MEMORY,
    ),
    (
        "no-supporting-claim",
        evidence(known("Lisbon"), "home = Berlin", cited=(BERLIN,)),
        Outcome.UNCLASSIFIABLE,
        "mismatch.expectation_has_no_supporting_claim",
        Locus.NONE,
    ),
    (
        "ambiguous",
        evidence(known("Berlin"), "Paris or Berlin"),
        Outcome.AMBIGUOUS_OUTPUT,
        "output.ambiguous",
        Locus.NONE,
    ),
    (
        "malformed",
        evidence(known("Berlin"), "", cited=(BERLIN,)),
        Outcome.MALFORMED_OUTPUT,
        "output.unreadable",
        Locus.NONE,
    ),
    (
        "unreadable-unrelated",
        evidence(known("Berlin"), "lovely weather", cited=(PET,), held=True),
        Outcome.WRONG_MEMORY,
        "unreadable.cites_unrelated_memory",
        Locus.RETRIEVAL,
    ),
    (
        "unreadable-no-claim",
        evidence(known("Berlin"), "lovely weather", cited_versions=1),
        Outcome.WRONG_MEMORY,
        "unreadable.cites_unrelated_memory",
        Locus.RETRIEVAL,
    ),
    (
        "retraction-unknown",
        evidence(UNKNOWN, "forget home", cited=(FORGET_HOME,)),
        Outcome.CORRECT_ABSTENTION,
        "valueless.nothing_supported",
        Locus.NONE,
    ),
    (
        "retraction-known-held",
        evidence(known("Berlin"), "forget home", cited=(FORGET_HOME,), held=True),
        Outcome.RETRIEVAL_MISS,
        "valueless.expected_held",
        Locus.RETRIEVAL,
    ),
    (
        "retraction-known-missing",
        evidence(known("Berlin"), "forget home", cited=(FORGET_HOME,)),
        Outcome.MISSING_MEMORY,
        "valueless.expected_not_held",
        Locus.MEMORY,
    ),
    (
        "retraction-contested",
        evidence(
            contested("Berlin", "Madrid"),
            "forget home",
            cited=(FORGET_HOME,),
            history=(PARIS, BERLIN, TWIN),
        ),
        Outcome.CONTESTED_ABSTAINED,
        "valueless.contested",
        Locus.NONE,
    ),
]


@pytest.mark.parametrize(
    ("case", "e", "outcome", "rule", "locus"), CASES, ids=[c[0] for c in CASES]
)
def test_classification_rules(
    case: str, e: ProbeEvidence, outcome: Outcome, rule: str, locus: Locus
) -> None:
    c = classify(e)
    assert (c.outcome, c.rule, c.locus) == (outcome, rule, locus)
    assert (c.locus is Locus.NONE) == (outcome.category is not Category.FAILURE)


def test_every_outcome_is_reachable() -> None:
    assert {c[2] for c in CASES} == set(Outcome)


def test_classification_records_its_evidence() -> None:
    c = classify(evidence(known("Berlin"), "home = Paris", cited=(PARIS,), held=True))
    assert c.observed == PARIS
    assert c.support == (BERLIN,)
    assert c.relation is Relation.TEMPORAL_CHANGE  # a change, not a contradiction


def test_newer_is_not_more_true() -> None:
    """A report that occurred earlier but was ingested later does not win."""
    late_old = claim("home", "Munich", 40, recorded=90)
    c = classify(
        evidence(
            known("Berlin"), "home = Munich", cited=(late_old,), history=(PARIS, late_old, BERLIN)
        )
    )
    assert c.outcome is Outcome.STALE


def test_forgotten_absent_is_correct() -> None:
    c = classify(evidence(UNKNOWN, None, history=(PARIS, BERLIN, FORGET_HOME)))
    assert c.outcome is Outcome.CORRECT_ABSTENTION


def test_retroactive_correction_is_not_a_temporal_error() -> None:
    """A correction occurring after valid_at still applies to it."""
    c = classify(
        evidence(
            known("Paris"),
            "home = Rome",
            cited=(ROME_CORRECTS_PARIS,),
            history=(PARIS, ROME_CORRECTS_PARIS),
            valid=5,
        )
    )
    assert c.outcome is not Outcome.TEMPORAL_ERROR


def test_outcome_categories_partition_the_taxonomy() -> None:
    by = {cat: {o for o in Outcome if o.category is cat} for cat in Category}
    assert by[Category.SUCCESS] == {Outcome.CORRECT, Outcome.CORRECT_ABSTENTION}
    assert by[Category.NEUTRAL] == {Outcome.CONTESTED_ANSWERED, Outcome.CONTESTED_ABSTAINED}
    assert by[Category.UNSCORABLE] == {
        Outcome.AMBIGUOUS_OUTPUT,
        Outcome.MALFORMED_OUTPUT,
        Outcome.UNCLASSIFIABLE,
    }
    assert sum(len(v) for v in by.values()) == len(Outcome)


def test_classification_is_a_closed_record() -> None:
    c = classify(evidence(known("Berlin"), "home = Berlin", cited=(BERLIN,)))
    assert type(c).model_validate_json(c.canonical()) == c
    assert isinstance(c.support[0].occurred_at, datetime)
