"""Core domain: content identity, the foundational records, and memory-history semantics.

This module depends only on the standard library and Pydantic. Everything else in
MEMORIA depends inward on it; it depends on nothing in MEMORIA and performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, NamedTuple, Self

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class InvalidTransitionError(ValueError):
    """A record that is well-formed on its own but cannot extend the history it targets."""


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, no ASCII escaping.

    Raises TypeError for anything not natively JSON-serialisable, so no value is
    ever serialised through an implicit, lossy conversion.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(payload: Any) -> str:
    """SHA-256 of the UTF-8 canonical JSON of ``payload``."""
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _canonical_set(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


# Timezone-aware, normalised to UTC so equal instants always hash identically.
UTCDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class Record(BaseModel):
    """Immutable, closed-schema record identified by the hash of its content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def canonical(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    @property
    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class Experience(Record):
    """A raw source interaction, exactly as observed. The root of all provenance."""

    source: str = Field(min_length=1)
    content: str
    occurred_at: UTCDatetime


class Operation(StrEnum):
    CREATE = "create"
    UPDATE = "update"  # the world changed; the prior belief held until this one begins
    CORRECT = "correct"  # the prior belief was wrong; it is retracted entirely
    FORGET = "forget"  # tombstone: all beliefs retracted, chain ends, history retained


class MemoryVersion(Record):
    """One immutable version of a memory. Versions form a hash-linked chain.

    Bitemporal: ``valid_from``/``valid_to`` is when the content is asserted true in
    the world; ``recorded_at`` is when the system committed this version. A FORGET
    tombstone asserts nothing, so it carries no content and no validity interval.
    """

    memory_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    operation: Operation
    content: str | None = None
    valid_from: UTCDatetime | None = None
    valid_to: UTCDatetime | None = None
    recorded_at: UTCDatetime
    # Experience digests, in canonical (sorted, de-duplicated) form.
    derived_from: Annotated[tuple[Digest, ...], AfterValidator(_canonical_set)] = ()
    supersedes: Digest | None = None  # digest of the previous version

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        first = self.version == 1
        if first != (self.operation is Operation.CREATE):
            raise ValueError("version 1 must be CREATE, and CREATE must be version 1")
        if first != (self.supersedes is None):
            raise ValueError("only version 1 may omit `supersedes`")
        if self.operation is Operation.CREATE and not self.derived_from:
            raise ValueError("CREATE must cite at least one source experience")
        if self.operation is Operation.FORGET:
            if (self.content, self.valid_from, self.valid_to) != (None, None, None):
                raise ValueError("FORGET carries no content or validity interval")
        elif self.content is None or self.valid_from is None:
            raise ValueError(f"{self.operation.value.upper()} requires content and valid_from")
        if self.valid_to and self.valid_from and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        return self

    def successor(
        self,
        operation: Operation,
        *,
        recorded_at: datetime,
        content: str | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        derived_from: tuple[str, ...] = (),
    ) -> MemoryVersion:
        """The next version in this chain, correctly linked to this one."""
        return MemoryVersion(
            memory_id=self.memory_id,
            version=self.version + 1,
            operation=operation,
            content=content,
            valid_from=valid_from,
            valid_to=valid_to,
            recorded_at=recorded_at,
            derived_from=derived_from,
            supersedes=self.digest,
        )


class Belief(NamedTuple):
    """A version together with the validity interval it effectively holds in its chain."""

    version: MemoryVersion
    valid_from: datetime
    valid_to: datetime | None  # None = open-ended

    def holds_at(self, t: datetime) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


def _start(v: MemoryVersion) -> datetime:
    assert v.valid_from is not None  # guaranteed by MemoryVersion for non-FORGET
    return v.valid_from


def _end(v: MemoryVersion, following: MemoryVersion | None) -> datetime | None:
    ends = [t for t in (v.valid_to, following.valid_from if following else None) if t is not None]
    return min(ends, default=None)


def _check_link(head: MemoryVersion | None, v: MemoryVersion) -> None:
    where = f"memory {v.memory_id!r} v{v.version}"
    if head is None:
        if v.version != 1:
            raise InvalidTransitionError(f"{where}: history must start at version 1")
        return
    if head.operation is Operation.FORGET:
        raise InvalidTransitionError(
            f"{where}: memory was forgotten at v{head.version}; FORGET is terminal"
        )
    if v.memory_id != head.memory_id:
        raise InvalidTransitionError(f"{where}: follows a version of memory {head.memory_id!r}")
    if v.version != head.version + 1:
        raise InvalidTransitionError(f"{where}: expected version {head.version + 1}")
    if v.supersedes != head.digest:
        raise InvalidTransitionError(f"{where}: `supersedes` is not the digest of v{head.version}")
    if v.recorded_at < head.recorded_at:
        raise InvalidTransitionError(f"{where}: recorded before the version it supersedes")


def beliefs(chain: Sequence[MemoryVersion]) -> list[Belief]:
    """Validate one memory's version chain and return its effective belief timeline.

    UPDATE ends the previous belief where the new one begins. CORRECT retracts the
    previous belief entirely, then asserts its own interval. FORGET retracts every
    belief and ends the chain. The resulting timeline is ordered and non-overlapping.
    """
    timeline: list[MemoryVersion] = []
    head: MemoryVersion | None = None
    for v in chain:
        _check_link(head, v)
        if v.operation is Operation.FORGET:
            timeline.clear()
        else:
            if v.operation is Operation.CORRECT:
                timeline.pop()
            if timeline and _start(v) <= _start(timeline[-1]):
                raise InvalidTransitionError(
                    f"memory {v.memory_id!r} v{v.version}: valid_from must be after that of "
                    f"the belief it follows (v{timeline[-1].version})"
                )
            timeline.append(v)
        head = v
    following: list[MemoryVersion | None] = [*timeline[1:], None]
    return [Belief(v, _start(v), _end(v, nxt)) for v, nxt in zip(timeline, following, strict=False)]


def check_citations(version: MemoryVersion, cited: Mapping[str, Experience]) -> None:
    """Every cited experience must be known and must have occurred before it was recorded."""
    for d in version.derived_from:
        experience = cited.get(d)
        if experience is None:
            raise InvalidTransitionError(
                f"memory {version.memory_id!r}: cites unknown experience {d}"
            )
        if experience.occurred_at > version.recorded_at:
            raise InvalidTransitionError(
                f"memory {version.memory_id!r}: cites experience {d} that occurred after "
                "the version was recorded"
            )


class MemoryState(Record):
    """The memories believed at record time ``known_at`` to hold at valid time ``valid_at``."""

    valid_at: UTCDatetime
    known_at: UTCDatetime
    memories: tuple[MemoryVersion, ...]  # at most one per memory, ordered by memory_id


def state_as_of(
    versions: Iterable[MemoryVersion], *, valid_at: datetime, known_at: datetime
) -> MemoryState:
    """Pure bitemporal fold: identical versions always yield an identical state and digest.

    Only versions recorded at or before ``known_at`` are considered; each memory's
    chain prefix is validated with :func:`beliefs`.
    """
    if valid_at.tzinfo is None or known_at.tzinfo is None:
        raise ValueError("valid_at and known_at must be timezone-aware")
    chains: dict[str, list[MemoryVersion]] = {}
    for v in versions:
        if v.recorded_at <= known_at:
            chains.setdefault(v.memory_id, []).append(v)
    held = [
        b.version
        for memory_id in sorted(chains)
        for b in beliefs(sorted(chains[memory_id], key=lambda v: v.version))
        if b.holds_at(valid_at)
    ]
    return MemoryState(valid_at=valid_at, known_at=known_at, memories=tuple(held))
