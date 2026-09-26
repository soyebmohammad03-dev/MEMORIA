"""Core provenance primitives: content hashing and the two foundational records.

This module depends only on the standard library and Pydantic. Everything else in
MEMORIA depends inward on it; it depends on nothing in MEMORIA.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator


def content_hash(payload: Any) -> str:
    """SHA-256 of canonical JSON (sorted keys, compact separators, UTF-8).

    Raises TypeError for anything not natively JSON-serialisable, so no value is
    ever hashed through an implicit, lossy conversion.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


# Timezone-aware, normalised to UTC so equal instants always hash identically.
UTCDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]


class Record(BaseModel):
    """Immutable, closed-schema record identified by the hash of its content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    UPDATE = "update"  # the world changed; the prior version was true for its interval
    CORRECT = "correct"  # the prior version was wrong; retroactive
    FORGET = "forget"  # tombstone; history is retained


class MemoryVersion(Record):
    """One immutable version of a memory. Versions form a hash-linked chain.

    Bitemporal: ``valid_from``/``valid_to`` is when the content is believed true in
    the world; ``recorded_at`` is when the system committed this version.
    """

    memory_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    operation: Operation
    content: str
    valid_from: UTCDatetime
    valid_to: UTCDatetime | None = None
    recorded_at: UTCDatetime
    derived_from: tuple[str, ...] = ()  # Experience digests
    supersedes: str | None = None  # digest of the previous MemoryVersion

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        first = self.version == 1
        if first != (self.operation is Operation.CREATE):
            raise ValueError("version 1 must be CREATE, and CREATE must be version 1")
        if first != (self.supersedes is None):
            raise ValueError("only version 1 may omit `supersedes`")
        if self.operation is Operation.CREATE and not self.derived_from:
            raise ValueError("CREATE must cite at least one source experience")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        return self
