from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from memoria.core import (
    Belief,
    Experience,
    InvalidTransitionError,
    MemoryVersion,
    Operation,
    beliefs,
    check_citations,
    content_hash,
    state_as_of,
)

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
