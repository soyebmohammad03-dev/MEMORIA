from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from memoria.core import Experience, MemoryVersion, Operation, content_hash

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _exp() -> Experience:
    return Experience(source="dialogue:1", content="I live in Paris.", occurred_at=T0)


def _v1() -> MemoryVersion:
    return MemoryVersion(
        memory_id="m1",
        version=1,
        operation=Operation.CREATE,
        content="User lives in Paris.",
        valid_from=T0,
        recorded_at=T0,
        derived_from=(_exp().digest,),
    )


def test_content_hash_is_canonical() -> None:
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_content_hash_rejects_non_json() -> None:
    with pytest.raises(TypeError):
        content_hash({"t": T0})


def test_digest_is_timezone_invariant() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    same_instant = Experience(
        source="dialogue:1", content="I live in Paris.", occurred_at=T0.astimezone(ist)
    )
    assert same_instant.digest == _exp().digest


def test_records_are_immutable_and_closed() -> None:
    with pytest.raises(ValidationError):
        _exp().content = "tampered"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Experience(source="s", content="c", occurred_at=T0, extra="x")  # type: ignore[call-arg]


def test_naive_datetimes_rejected() -> None:
    with pytest.raises(ValidationError):
        Experience(source="s", content="c", occurred_at=datetime(2026, 1, 1))  # noqa: DTZ001


def test_version_chain() -> None:
    v1 = _v1()
    v2 = MemoryVersion(
        memory_id="m1",
        version=2,
        operation=Operation.UPDATE,
        content="User lives in Berlin.",
        valid_from=T0 + timedelta(days=30),
        recorded_at=T0 + timedelta(days=30),
        supersedes=v1.digest,
    )
    assert v2.supersedes == v1.digest


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": 2},  # CREATE must be version 1
        {"supersedes": "sha256:x"},  # version 1 supersedes nothing
        {"derived_from": ()},  # CREATE requires provenance
        {"valid_to": T0},  # empty validity interval
        {"operation": Operation.UPDATE},  # version 1 must be CREATE
    ],
)
def test_invariants_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MemoryVersion.model_validate(_v1().model_dump() | overrides)
