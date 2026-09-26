from collections.abc import Iterator

import pytest

from memoria.core import (
    Experience,
    FormationDecision,
    FormationReason,
    InvalidTransitionError,
    Operation,
)
from memoria.formation import (
    EpisodicPolicy,
    FormationPolicy,
    Statement,
    StatementPolicy,
    form,
    incarnations,
    parse_statement,
)
from memoria.scenarios import day, relocation_year
from memoria.store import MemoryLog

R = FormationReason
STATEMENT = StatementPolicy()


@pytest.fixture
def log() -> Iterator[MemoryLog]:
    with MemoryLog(":memory:") as log:
        yield log


def feed(
    log: MemoryLog, *steps: tuple[float, str] | tuple[float, str, float]
) -> list[FormationDecision]:
    """Form (occurred_day, content[, recorded_day]) steps with the statement policy."""
    out = []
    for occurred, content, *recorded in steps:
        e = Experience(source=f"s:{occurred}:{content}", content=content, occurred_at=day(occurred))
        out.append(form(log, e, STATEMENT, recorded_at=day(recorded[0] if recorded else occurred)))
    return out


def reasons(decisions: list[FormationDecision]) -> list[R]:
    return [d.reason for d in decisions]


# --- statement language -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("set home = Paris", Statement("set", "home", "Paris")),
        ("  set home = New York  ", Statement("set", "home", "New York")),
        (
            "correct job.title = Senior Engineer",
            Statement("correct", "job.title", "Senior Engineer"),
        ),
        ("forget home", Statement("forget", "home", None)),
        ("set a = b = c", Statement("set", "a", "b = c")),
        ("set name = Zoë Ñúñez", Statement("set", "name", "Zoë Ñúñez")),
    ],
)
def test_parse_statement(text: str, parsed: Statement) -> None:
    assert parse_statement(text) == parsed


@pytest.mark.parametrize(
    "text",
    [
        "",
        "set home =",
        "set home = ",
        "set  = Paris",
        "Set home = Paris",  # verbs are lowercase
        "set Home = Paris",  # keys are lowercase
        "set home=Paris",  # exact spacing is part of the grammar
        "forget home = Paris",  # forget takes no value
        "update home = Paris",
        "set home = Paris\nset home = Rome",  # one statement per experience
        "I live in Paris.",
    ],
)
def test_parse_rejects(text: str) -> None:
    assert parse_statement(text) is None


# --- the research scenario, as a regression table ------------------------------------------

EXPECTED = [
    (R.NEW_KEY, "home"),
    (R.NEW_KEY, "employer"),
    (R.UNPARSED, None),
    (R.REDUNDANT, "employer"),
    (R.VALUE_CHANGED, "home"),
    (R.EXPLICIT_CORRECTION, "employer"),
    (R.STALE, "home"),
    (R.DUPLICATE_EXPERIENCE, None),
    (R.EXPLICIT_FORGET, "employer"),
    (R.RELEARNED, "employer#2"),
    (R.UNKNOWN_KEY, None),
    (R.VALUE_CHANGED, "home"),
    (R.CONFLICTING, "home"),
    (R.UNPARSED, None),
    (R.NEW_KEY, "pet"),
    (R.UNPARSED, None),
]


def run(log: MemoryLog, policy: FormationPolicy) -> list[FormationDecision]:
    return [form(log, s.experience, policy, recorded_at=s.recorded_at) for s in relocation_year()]


def test_relocation_year_outcomes(log: MemoryLog) -> None:
    decisions = run(log, STATEMENT)
    assert [(d.reason, d.memory_id) for d in decisions] == EXPECTED
    assert log.records(FormationDecision) == decisions
    log.verify()


def test_relocation_year_final_and_historical_state(log: MemoryLog) -> None:
    run(log, STATEMENT)

    def held(valid: float, known: float) -> dict[str, str | None]:
        state = log.state_as_of(valid_at=day(valid), known_at=day(known))
        return {m.memory_id: m.content for m in state.memories}

    assert held(345, 345) == {
        "employer#2": "employer = Initech",
        "home": "home = Hamburg",
        "pet": "pet = dog",
    }
    # The move happened on day 55 but was only recorded on day 60.
    assert held(57, 59)["home"] == "home = Paris"
    assert held(57, 60)["home"] == "home = Berlin"
    # The correction is retroactive: day 10 is Acme as known then, Globex as known later.
    assert held(10, 60)["employer"] == "employer = Acme"
    assert held(10, 61)["employer"] == "employer = Globex"
    # Between forgetting (150) and relearning (200) nothing is known about the employer.
    assert "employer" not in held(175, 175)
    assert "employer#2" not in held(175, 175)
    # The stale Munich report and the conflicting Bremen report never entered memory.
    contents = {v.content for v in log.versions()}
    assert "home = Munich" not in contents
    assert "home = Bremen" not in contents


def test_every_version_is_explained_by_exactly_one_decision(log: MemoryLog) -> None:
    run(log, STATEMENT)
    decisions = log.records(FormationDecision)
    by_version = {d.version: d for d in decisions if d.version}
    assert len(by_version) == len([d for d in decisions if d.version])
    for v in log.versions():
        d = by_version[v.digest]
        assert d.operation is v.operation
        assert d.experience in v.derived_from
        assert log.experience(d.experience) is not None


def test_formation_is_deterministic() -> None:
    def digests() -> list[str]:
        with MemoryLog(":memory:") as log:
            return [d.digest for d in run(log, STATEMENT)] + [v.digest for v in log.versions()]

    assert digests() == digests()


# --- policy rules, one at a time ------------------------------------------------------------


def test_update_is_valid_from_when_it_occurred(log: MemoryLog) -> None:
    feed(log, (0, "set home = Paris"), (20, "set home = Berlin", 25))
    v2 = log.history("home")[1]
    assert (v2.operation, v2.valid_from, v2.recorded_at) == (Operation.UPDATE, day(20), day(25))


def test_correction_takes_over_the_corrected_interval(log: MemoryLog) -> None:
    feed(log, (0, "set home = Paris"), (20, "set home = Berlin"), (30, "correct home = Bonn"))
    v3 = log.history("home")[2]
    assert (v3.operation, v3.valid_from) == (Operation.CORRECT, day(20))
    state = log.state_as_of(valid_at=day(10), known_at=day(30))
    assert state.memories[0].content == "home = Paris"


@pytest.mark.parametrize("verb", ["set home = Rome", "correct home = Rome", "forget home"])
def test_statements_predating_the_current_belief_are_stale(log: MemoryLog, verb: str) -> None:
    decisions = feed(log, (10, "set home = Paris"), (5, verb, 20))
    assert decisions[-1].reason is R.STALE
    assert decisions[-1].basis == log.history("home")[0].digest
    assert len(log.history("home")) == 1


def test_redundant_set_and_correct(log: MemoryLog) -> None:
    decisions = feed(
        log, (0, "set home = Paris"), (5, "set home = Paris"), (6, "correct home = Paris")
    )
    assert reasons(decisions)[1:] == [R.REDUNDANT, R.REDUNDANT]
    assert len(log.history("home")) == 1


def test_same_instant_conflict_keeps_first_recorded(log: MemoryLog) -> None:
    decisions = feed(
        log, (0, "set home = Paris"), (0, "set home = Rome"), (0, "correct home = Oslo")
    )
    assert reasons(decisions) == [R.NEW_KEY, R.CONFLICTING, R.CONFLICTING]
    assert log.history("home")[-1].content == "home = Paris"


@pytest.mark.parametrize("verb", ["correct pet = cat", "forget pet"])
def test_correct_or_forget_unknown_key(log: MemoryLog, verb: str) -> None:
    (d,) = feed(log, (0, verb))
    assert (d.reason, d.memory_id, d.basis) == (R.UNKNOWN_KEY, None, None)
    assert log.versions() == []


def test_unknown_key_after_forget_cites_tombstone(log: MemoryLog) -> None:
    decisions = feed(log, (0, "set pet = cat"), (5, "forget pet"), (6, "forget pet"))
    tomb = log.history("pet")[-1]
    assert (decisions[-1].reason, decisions[-1].memory_id, decisions[-1].basis) == (
        R.UNKNOWN_KEY,
        "pet",
        tomb.digest,
    )


def test_relearning_creates_a_new_incarnation(log: MemoryLog) -> None:
    decisions = feed(
        log,
        (0, "set pet = cat"),
        (5, "forget pet"),
        (9, "set pet = dog"),
        (12, "forget pet"),
        (15, "set pet = fish"),
    )
    assert [d.memory_id for d in decisions] == ["pet", "pet", "pet#2", "pet#2", "pet#3"]
    assert decisions[2].basis == log.history("pet")[-1].digest  # the tombstone
    assert [len(c) for c in incarnations(log, "pet")] == [2, 2, 1]


def test_forget_cites_its_experience(log: MemoryLog) -> None:
    decisions = feed(log, (0, "set pet = cat"), (5, "forget pet"))
    tomb = log.history("pet")[-1]
    assert tomb.operation is Operation.FORGET
    assert tomb.derived_from == (decisions[-1].experience,)


def test_duplicate_experience_is_recorded_not_reapplied(log: MemoryLog) -> None:
    e = Experience(source="s", content="set home = Paris", occurred_at=day(0))
    first = form(log, e, STATEMENT, recorded_at=day(0))
    again = form(log, e, STATEMENT, recorded_at=day(3))
    same = form(log, e, STATEMENT, recorded_at=day(3))
    assert again.reason is R.DUPLICATE_EXPERIENCE
    assert same == again  # identical observation, stored once
    assert [d.reason for d in log.records(FormationDecision)] == [first.reason, again.reason]
    assert len(log.experiences()) == 1


def test_same_words_from_another_source_are_a_new_experience(log: MemoryLog) -> None:
    decisions = feed(log, (0, "set home = Paris"), (1, "set home = Paris"))
    assert reasons(decisions) == [R.NEW_KEY, R.REDUNDANT]


# --- rejection and atomicity ---------------------------------------------------------------


def test_recorded_before_occurred_is_rejected_atomically(log: MemoryLog) -> None:
    e = Experience(source="s", content="set home = Paris", occurred_at=day(5))
    with pytest.raises(InvalidTransitionError, match="occurred after"):
        form(log, e, STATEMENT, recorded_at=day(4))
    skip = Experience(source="s", content="chatter", occurred_at=day(5))
    with pytest.raises(InvalidTransitionError, match="before the experience occurred"):
        form(log, skip, STATEMENT, recorded_at=day(4))
    assert (log.experiences(), log.versions(), log.records(FormationDecision)) == ([], [], [])


def test_one_policy_per_log(log: MemoryLog) -> None:
    feed(log, (0, "set home = Paris"))
    e = Experience(source="s", content="anything", occurred_at=day(1))
    with pytest.raises(InvalidTransitionError, match="formed by 'statement-v1'"):
        form(log, e, EpisodicPolicy(), recorded_at=day(1))
    assert len(log.experiences()) == 1
    assert len(log.versions()) == 1


def test_record_time_cannot_go_backwards_through_skips(log: MemoryLog) -> None:
    feed(log, (0, "chatter", 10))
    with pytest.raises(InvalidTransitionError, match="record time"):
        feed(log, (1, "set home = Paris", 5))


# --- episodic baseline ------------------------------------------------------------------------


def test_episodic_stores_every_distinct_experience(log: MemoryLog) -> None:
    decisions = run(log, EpisodicPolicy())
    distinct = {s.experience.digest for s in relocation_year()}
    created = [d for d in decisions if d.reason is R.NEW_EPISODE]
    assert len(created) == len(distinct) == len(log.versions())
    assert [d.reason for d in decisions].count(R.DUPLICATE_EXPERIENCE) == 1
    assert all(v.operation is Operation.CREATE for v in log.versions())
    # Contradictory reports coexist: nothing is ever superseded.
    held = log.state_as_of(valid_at=day(345), known_at=day(345))
    contents = {m.content for m in held.memories}
    assert {"set home = Paris", "set home = Berlin", "set home = Munich"} <= contents
    log.verify()
