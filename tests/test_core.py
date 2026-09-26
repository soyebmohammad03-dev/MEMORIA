from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from memoria.core import (
    Belief,
    Dataset,
    Expectation,
    ExpectationStatus,
    Experience,
    FormationDecision,
    FormationReason,
    InvalidTransitionError,
    MemoryVersion,
    Operation,
    Query,
    Response,
    RetrievalTrace,
    RetrieverSpec,
    SignalEvidence,
    SignalSpec,
    Step,
    StepOutcome,
    beliefs,
    check_citations,
    check_decision,
    content_hash,
    quantize,
    state_as_of,
    unit_interval,
)
from memoria.retrieval import lexical_recency

T0 = datetime(2026, 1, 1, tzinfo=UTC)
FAKE = "sha256:" + "0" * 64


def day(n: int) -> datetime:
    return T0 + timedelta(days=n)


EXP = Experience(source="dialogue:1", content="I live in Paris.", occurred_at=T0)


def create(memory_id: str = "m1", *, at: int = 0, content: str = "Paris") -> MemoryVersion:
    return MemoryVersion(
        memory_id=memory_id,
        version=1,
        operation=Operation.CREATE,
        content=content,
        valid_from=day(at),
        recorded_at=day(at),
        derived_from=(EXP.digest,),
    )


# --- content identity ---------------------------------------------------------------


def test_content_hash_is_canonical() -> None:
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})
    assert content_hash({"s": "é"}) != content_hash({"s": "e"})


def test_content_hash_rejects_non_json() -> None:
    with pytest.raises(TypeError):
        content_hash({"t": T0})


def test_digest_is_timezone_invariant() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    same_instant = Experience(
        source="dialogue:1", content="I live in Paris.", occurred_at=T0.astimezone(ist)
    )
    assert same_instant.digest == EXP.digest
    assert same_instant.occurred_at.tzinfo is UTC


def test_citations_are_canonicalised() -> None:
    a, b = content_hash("a"), content_hash("b")
    one = MemoryVersion.model_validate(create().model_dump() | {"derived_from": (b, a, b)})
    two = MemoryVersion.model_validate(create().model_dump() | {"derived_from": (a, b)})
    assert one.derived_from == (a, b) == tuple(sorted((a, b)))
    assert one.digest == two.digest


def test_canonical_roundtrip_preserves_digest() -> None:
    v = create()
    assert MemoryVersion.model_validate_json(v.canonical()).digest == v.digest


# --- record invariants ----------------------------------------------------------------


def test_records_are_immutable_and_closed() -> None:
    with pytest.raises(ValidationError):
        EXP.content = "tampered"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Experience(source="s", content="c", occurred_at=T0, extra="x")  # type: ignore[call-arg]


def test_naive_datetimes_rejected() -> None:
    with pytest.raises(ValidationError):
        Experience(source="s", content="c", occurred_at=datetime(2026, 1, 1))  # noqa: DTZ001


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": 2},  # CREATE must be version 1
        {"supersedes": FAKE},  # version 1 supersedes nothing
        {"derived_from": ()},  # CREATE requires provenance
        {"derived_from": ("not-a-digest",)},
        {"valid_to": T0},  # empty validity interval
        {"operation": Operation.UPDATE},  # version 1 must be CREATE
        {"content": None},
        {"valid_from": None},
        {"memory_id": ""},
        {"version": 0},
    ],
)
def test_record_invariants(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MemoryVersion.model_validate(create().model_dump() | overrides)


@pytest.mark.parametrize(
    "fields",
    [
        {"content": "x"},
        {"valid_from": T0},
        {"valid_to": T0},
    ],
)
def test_forget_carries_nothing(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        create().successor(Operation.FORGET, recorded_at=day(1), **fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("op", [Operation.UPDATE, Operation.CORRECT])
def test_assertions_require_content_and_validity(op: Operation) -> None:
    with pytest.raises(ValidationError):
        create().successor(op, recorded_at=day(1), valid_from=day(1))
    with pytest.raises(ValidationError):
        create().successor(op, recorded_at=day(1), content="x")


def test_successor_links_chain() -> None:
    v1 = create()
    v2 = v1.successor(Operation.UPDATE, recorded_at=day(1), content="Berlin", valid_from=day(1))
    assert (v2.memory_id, v2.version, v2.supersedes) == ("m1", 2, v1.digest)


# --- belief timeline semantics ----------------------------------------------------------


def test_update_ends_previous_belief() -> None:
    v1 = create()
    v2 = v1.successor(Operation.UPDATE, recorded_at=day(30), content="Berlin", valid_from=day(20))
    assert beliefs([v1, v2]) == [Belief(v1, day(0), day(20)), Belief(v2, day(20), None)]


def test_update_after_bounded_belief_leaves_gap() -> None:
    v1 = MemoryVersion.model_validate(create().model_dump() | {"valid_to": day(5)})
    v2 = v1.successor(Operation.UPDATE, recorded_at=day(9), content="Berlin", valid_from=day(9))
    assert beliefs([v1, v2]) == [Belief(v1, day(0), day(5)), Belief(v2, day(9), None)]


def test_correct_retracts_and_may_redate() -> None:
    v1 = create(at=5)
    v2 = v1.successor(Operation.CORRECT, recorded_at=day(6), content="Lyon", valid_from=day(1))
    assert beliefs([v1, v2]) == [Belief(v2, day(1), None)]


def test_correct_of_update_restores_prior_interval_end() -> None:
    v1 = MemoryVersion.model_validate(create().model_dump() | {"valid_to": day(50)})
    v2 = v1.successor(Operation.UPDATE, recorded_at=day(10), content="Berlin", valid_from=day(10))
    v3 = v2.successor(Operation.CORRECT, recorded_at=day(11), content="Rome", valid_from=day(40))
    assert beliefs([v1, v2, v3]) == [Belief(v1, day(0), day(40)), Belief(v3, day(40), None)]


def test_forget_retracts_everything() -> None:
    v1 = create()
    v2 = v1.successor(Operation.FORGET, recorded_at=day(3))
    assert beliefs([v1, v2]) == []


def test_empty_chain_has_no_beliefs() -> None:
    assert beliefs([]) == []


def _bad_chains() -> list[tuple[str, list[MemoryVersion]]]:
    v1 = create()
    upd = v1.successor(Operation.UPDATE, recorded_at=day(2), content="Berlin", valid_from=day(2))
    gone = v1.successor(Operation.FORGET, recorded_at=day(2))
    other = create("m2")
    return [
        ("start at version 1", [upd]),
        (
            "terminal",
            [
                v1,
                gone,
                gone.successor(
                    Operation.UPDATE, recorded_at=day(3), content="x", valid_from=day(3)
                ),
            ],
        ),
        (
            "follows a version of memory",
            [v1, MemoryVersion.model_validate(upd.model_dump() | {"memory_id": "m2"})],
        ),
        (
            "expected version 2",
            [
                v1,
                upd.successor(Operation.UPDATE, recorded_at=day(3), content="x", valid_from=day(3)),
            ],
        ),
        (
            "supersedes",
            [v1, MemoryVersion.model_validate(upd.model_dump() | {"supersedes": other.digest})],
        ),
        (
            "recorded before",
            [
                create(at=5),
                create(at=5).successor(
                    Operation.UPDATE, recorded_at=day(4), content="x", valid_from=day(6)
                ),
            ],
        ),
        (
            "valid_from must be after",
            [
                v1,
                v1.successor(Operation.UPDATE, recorded_at=day(2), content="x", valid_from=day(0)),
            ],
        ),
        ("expected version 2 (duplicate)", [v1, v1]),
    ]


@pytest.mark.parametrize(("reason", "chain"), _bad_chains(), ids=[r for r, _ in _bad_chains()])
def test_invalid_transitions(reason: str, chain: list[MemoryVersion]) -> None:
    with pytest.raises(InvalidTransitionError, match=reason.removesuffix(" (duplicate)")):
        beliefs(chain)


# --- citations ----------------------------------------------------------------------------


def test_citations() -> None:
    check_citations(create(), {EXP.digest: EXP})
    with pytest.raises(InvalidTransitionError, match="unknown"):
        check_citations(create(), {})
    future = Experience(source="s", content="c", occurred_at=day(1))
    v = MemoryVersion.model_validate(create().model_dump() | {"derived_from": (future.digest,)})
    with pytest.raises(InvalidTransitionError, match="after"):
        check_citations(v, {future.digest: future})


# --- bitemporal state -----------------------------------------------------------------------


def _moved_to_berlin() -> list[MemoryVersion]:
    """Paris from day 0; on day 30 the system learns the user moved on day 20."""
    v1 = create()
    v2 = v1.successor(Operation.UPDATE, recorded_at=day(30), content="Berlin", valid_from=day(20))
    return [v1, v2]


def _held(valid: int, known: int, versions: list[MemoryVersion]) -> list[str | None]:
    state = state_as_of(versions, valid_at=day(valid), known_at=day(known))
    return [m.content for m in state.memories]


@pytest.mark.parametrize(
    ("valid", "known", "expected"),
    [
        (25, 25, ["Paris"]),  # on day 25 the move was not yet known
        (25, 30, ["Berlin"]),  # known on day 30, retroactive to day 20
        (10, 30, ["Paris"]),  # history before the move is preserved
        (-1, 30, []),  # before anything was valid
        (0, -1, []),  # before anything was recorded
    ],
)
def test_bitemporal_queries(valid: int, known: int, expected: list[str]) -> None:
    assert _held(valid, known, _moved_to_berlin()) == expected


def test_forgotten_memory_absent_only_after_forget_is_known() -> None:
    v1, v2 = _moved_to_berlin()
    v3 = v2.successor(Operation.FORGET, recorded_at=day(40))
    assert _held(25, 39, [v1, v2, v3]) == ["Berlin"]
    assert _held(25, 40, [v1, v2, v3]) == []


def test_state_is_order_independent_and_deterministic() -> None:
    versions = [*_moved_to_berlin(), create("m0", content="likes tea")]
    a = state_as_of(versions, valid_at=day(25), known_at=day(30))
    b = state_as_of(reversed(versions), valid_at=day(25), known_at=day(30))
    assert a == b
    assert a.digest == b.digest
    assert [m.memory_id for m in a.memories] == ["m0", "m1"]


def test_state_rejects_naive_times() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        state_as_of([], valid_at=datetime(2026, 1, 1), known_at=T0)  # noqa: DTZ001


def test_state_validates_chains() -> None:
    _, v2 = _moved_to_berlin()
    with pytest.raises(InvalidTransitionError):
        state_as_of([v2], valid_at=day(25), known_at=day(30))


# --- Phase 2 records: formation decisions ------------------------------------------------


D = FormationReason


def decision(**overrides: object) -> FormationDecision:
    v1 = create()
    fields: dict[str, object] = {
        "experience": EXP.digest,
        "policy": "p",
        "reason": D.NEW_KEY,
        "memory_id": "m1",
        "version": v1.digest,
        "recorded_at": day(0),
    }
    return FormationDecision.model_validate(fields | overrides)


def test_reason_operation_mapping() -> None:
    assert D.NEW_KEY.operation is Operation.CREATE
    assert D.VALUE_CHANGED.operation is Operation.UPDATE
    assert D.EXPLICIT_CORRECTION.operation is Operation.CORRECT
    assert D.EXPLICIT_FORGET.operation is Operation.FORGET
    assert {r for r in D if r.operation is None} == {
        D.DUPLICATE_EXPERIENCE,
        D.REDUNDANT,
        D.STALE,
        D.CONFLICTING,
        D.UNKNOWN_KEY,
        D.UNPARSED,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"version": None}, "version must be present"),
        ({"reason": D.REDUNDANT}, "version must be present"),
        ({"reason": D.VALUE_CHANGED}, "requires a basis"),
        ({"basis": FAKE}, "cannot have a basis"),
        ({"memory_id": None}, "memory_id is required"),
        ({"reason": D.UNPARSED, "version": None}, "memory_id is required"),
        ({"policy": ""}, "at least 1 character"),
    ],
)
def test_decision_invariants(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        decision(**overrides)


def test_skip_decisions_are_valid_records() -> None:
    unparsed = decision(reason=D.UNPARSED, memory_id=None, version=None)
    stale = decision(reason=D.STALE, version=None, basis=create().digest)
    unknown = decision(reason=D.UNKNOWN_KEY, memory_id=None, version=None)
    assert unparsed.operation is stale.operation is unknown.operation is None


def test_check_decision_cross_references() -> None:
    v1 = create()
    check_decision(decision(), EXP, v1, None)
    other = Experience(source="x", content="y", occurred_at=T0)
    bad = [
        ((decision(), other, v1, None), "experience does not match"),
        ((decision(recorded_at=T0 - timedelta(1)), EXP, v1, None), "before the experience"),
        ((decision(), EXP, None, None), "version does not match"),
        ((decision(), EXP, v1, v1), "basis does not match"),
        ((decision(version=create("m2").digest, memory_id="m2"), EXP, v1, None), "does not match"),
        ((decision(memory_id="zz"), EXP, v1, None), "not the recorded operation"),
        ((decision(recorded_at=day(1)), EXP, v1, None), "record times differ"),
    ]
    for args, message in bad:
        with pytest.raises(InvalidTransitionError, match=message):
            check_decision(*args)
    uncited = MemoryVersion.model_validate(v1.model_dump() | {"derived_from": (FAKE,)})
    with pytest.raises(InvalidTransitionError, match="does not cite"):
        check_decision(decision(version=uncited.digest), EXP, uncited, None)
    v2 = v1.successor(
        Operation.UPDATE,
        recorded_at=day(1),
        content="x",
        valid_from=day(1),
        derived_from=(EXP.digest,),
    )
    upd = decision(reason=D.VALUE_CHANGED, version=v2.digest, basis=FAKE, recorded_at=day(1))
    with pytest.raises(InvalidTransitionError, match="basis does not match"):
        check_decision(upd, EXP, v2, v1)


# --- Phase 2 records: retrieval ------------------------------------------------------------


def test_quantize_is_stable() -> None:
    assert quantize(0.1 + 0.2) == quantize(0.3)
    assert quantize(1 / 3) == 0.333333333333


def test_signal_evidence_inputs_are_canonical_and_validated() -> None:
    a = SignalEvidence(signal="s", score=1.0, inputs=(("b", 1), ("a", "x")))
    b = SignalEvidence(signal="s", score=1.0, inputs=(("a", "x"), ("b", 1)))
    assert a.inputs == (("a", "x"), ("b", 1))
    assert a.digest == b.digest
    with pytest.raises(ValidationError, match="unique"):
        SignalEvidence(signal="s", score=1.0, inputs=(("a", 1), ("a", 2)))
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            SignalEvidence(signal="s", score=bad)


@pytest.mark.parametrize(
    ("signals", "gate", "message"),
    [
        ((), "a", "at least 1"),
        (("a", "a"), "a", "unique"),
        (("a",), "b", "not one of the signals"),
    ],
)
def test_retriever_spec_invariants(signals: tuple[str, ...], gate: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RetrieverSpec(
            name="r", gate=gate, signals=tuple(SignalSpec(name=s, weight=1.0) for s in signals)
        )


def test_signal_weight_must_be_positive_and_finite() -> None:
    for w in (0.0, -1.0, float("inf")):
        with pytest.raises(ValidationError):
            SignalSpec(name="s", weight=w)


def test_query_limit_positive() -> None:
    with pytest.raises(ValidationError):
        Query(text="x", valid_at=T0, known_at=T0, limit=0)


def _trace() -> RetrievalTrace:
    home = create("home", content="home = Paris")
    pet = create("pet", content="pet = dog")
    tea = create("tea", content="drink = tea and home cooking")
    state = state_as_of([home, pet, tea], valid_at=day(5), known_at=day(5))
    query = Query(text="home", valid_at=day(5), known_at=day(5), limit=1)
    return lexical_recency().retrieve(state, query)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda d: d["candidates"][0].update(total=d["candidates"][0]["total"] + 1),
        lambda d: d["candidates"][0].update(eligible=not d["candidates"][0]["eligible"]),
        lambda d: d.update(candidates=list(reversed(d["candidates"]))),
        lambda d: d.update(selected=[]),
        lambda d: d.update(selected=[c["version"] for c in d["candidates"][:2]]),
        lambda d: d["candidates"][0].update(signals=d["candidates"][0]["signals"][:1]),
        lambda d: d.update(candidates=[d["candidates"][0], d["candidates"][0]]),
    ],
    ids=["total", "eligible", "order", "unselected", "over-limit", "signals", "duplicate"],
)
def test_trace_cannot_be_internally_inconsistent(tamper: object) -> None:
    dumped = _trace().model_dump(mode="json")
    tamper(dumped)  # type: ignore[operator]
    with pytest.raises(ValidationError):
        RetrievalTrace.model_validate(dumped)


def test_trace_roundtrip() -> None:
    trace = _trace()
    assert RetrievalTrace.model_validate_json(trace.canonical()).digest == trace.digest


def test_response_invariants() -> None:
    Response(trace=FAKE, responder="r", output=None)
    Response(trace=FAKE, responder="r", output="x", cited=(FAKE,))
    with pytest.raises(ValidationError, match="iff it answers"):
        Response(trace=FAKE, responder="r", output="x")
    with pytest.raises(ValidationError, match="iff it answers"):
        Response(trace=FAKE, responder="r", output=None, cited=(FAKE,))
    with pytest.raises(ValidationError, match="unique"):
        Response(trace=FAKE, responder="r", output="x", cited=(FAKE, FAKE))


# --- Phase 3 records -----------------------------------------------------------------------


def test_unit_interval_is_stateless_and_seeded() -> None:
    a = unit_interval(1, "drop", FAKE)
    assert a == unit_interval(1, "drop", FAKE)  # no hidden generator state
    assert 0 <= a < 1
    assert a != unit_interval(2, "drop", FAKE)
    assert a != unit_interval(1, "delay", FAKE)
    # Pinned: the value is a function of the arguments on every platform and version.
    assert unit_interval(0, "x") == 0.7098306351812017
    values = [unit_interval(3, i) for i in range(2000)]
    assert 0.45 < sum(values) / len(values) < 0.55


@pytest.mark.parametrize(
    ("status", "values"),
    [
        (ExpectationStatus.KNOWN, ()),
        (ExpectationStatus.KNOWN, ("a", "b")),
        (ExpectationStatus.UNKNOWN, ("a",)),
        (ExpectationStatus.CONTESTED, ("a",)),
        (ExpectationStatus.CONTESTED, ("a", "a")),  # de-duplicated to one value
    ],
)
def test_expectation_invariants(status: ExpectationStatus, values: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        Expectation(status=status, values=values)


def test_expectation_values_are_canonical() -> None:
    a = Expectation(status=ExpectationStatus.CONTESTED, values=("b", "a"))
    assert a == Expectation(status=ExpectationStatus.CONTESTED, values=("a", "b"))


def _step(at: int) -> Step:
    return Step(experience=EXP, recorded_at=day(at))


def test_dataset_invariants() -> None:
    Dataset(name="d", version="1", steps=(_step(0), _step(0), _step(1)))
    with pytest.raises(ValidationError, match="ingestion"):
        Dataset(name="d", version="1", steps=(_step(1), _step(0)))
    probe = {
        "id": "p",
        "text": "x",
        "valid_at": T0,
        "known_at": T0,
        "expected": {"status": "unknown"},
    }
    with pytest.raises(ValidationError, match="unique"):
        Dataset.model_validate({"name": "d", "version": "1", "steps": [], "probes": [probe, probe]})
    with pytest.raises(ValidationError):
        Dataset(name="", version="1", steps=())


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"decision": FAKE},
        {"reason": FormationReason.NEW_KEY},
        {"decision": FAKE, "reason": FormationReason.NEW_KEY, "rejected": "x"},
        {"rejected": ""},
    ],
)
def test_step_outcome_is_decision_xor_rejection(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        StepOutcome.model_validate({"step": FAKE} | fields)
