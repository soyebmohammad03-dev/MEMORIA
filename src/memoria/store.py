"""Append-only SQLite operation log: the source of truth for memory history.

Storage only. Every domain rule is enforced by :mod:`memoria.core`, so another backend
(e.g. PostgreSQL) has to reproduce storage and append-only enforcement, not semantics.
SQLite-specific parts: the guard triggers and ``PRAGMA user_version``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import ValidationError

from memoria import core
from memoria.core import Experience, InvalidTransitionError, MemoryState, MemoryVersion, Record

SCHEMA_VERSION = 1
_TABLES = ("experiences", "memory_versions")
_SCHEMA = f"""
BEGIN;
CREATE TABLE experiences (
    seq INTEGER PRIMARY KEY,
    digest TEXT NOT NULL UNIQUE,
    body TEXT NOT NULL
);
CREATE TABLE memory_versions (
    seq INTEGER PRIMARY KEY,
    digest TEXT NOT NULL UNIQUE,
    memory_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    body TEXT NOT NULL,
    UNIQUE (memory_id, version)
);
{
    "".join(
        f"CREATE TRIGGER {t}_no_{op.lower()} BEFORE {op} ON {t} "
        f"BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;\n"
        for t in _TABLES
        for op in ("UPDATE", "DELETE")
    )
}
PRAGMA user_version = {SCHEMA_VERSION};
COMMIT;
"""


class LogIntegrityError(RuntimeError):
    """The stored log is not a valid MEMORIA log (corruption, tampering, or wrong schema)."""


def _load[R: Record](model: type[R], rows: Iterable[tuple[str, str]]) -> list[R]:
    records = []
    for digest, body in rows:
        try:
            record = model.model_validate_json(body)
        except ValidationError as e:
            raise LogIntegrityError(f"{model.__name__} {digest}: stored body is invalid") from e
        if record.digest != digest:
            raise LogIntegrityError(f"{model.__name__} {digest}: stored body does not match digest")
        records.append(record)
    return records


class MemoryLog:
    """Append-only log of experiences and memory versions, validated on every append.

    Record time is the log's own axis: versions must be appended in non-decreasing
    ``recorded_at`` order. Nothing is ever updated or deleted.
    """

    # ponytail: single-writer design; appends serialise on SQLite's write lock.

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(path, isolation_level=None)  # transactions are explicit
        found = self._db.execute("PRAGMA user_version").fetchone()[0]
        if found == 0 and not self._db.execute("SELECT 1 FROM sqlite_master").fetchone():
            self._db.executescript(_SCHEMA)
        elif found != SCHEMA_VERSION:
            self._db.close()
            raise LogIntegrityError(f"{path}: not a MEMORIA log with schema v{SCHEMA_VERSION}")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, t: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def append_experience(self, experience: Experience) -> str:
        """Idempotent: an identical experience is the same content-addressed record."""
        digest = experience.digest
        with self._write() as db:
            db.execute(
                "INSERT OR IGNORE INTO experiences (digest, body) VALUES (?, ?)",
                (digest, experience.canonical()),
            )
        return digest

    def append_version(self, version: MemoryVersion) -> str:
        """Append after validating it against the memory's full history and its citations."""
        with self._write():
            core.beliefs([*self.history(version.memory_id), version])
            last = self._versions("ORDER BY seq DESC LIMIT 1")
            if last and version.recorded_at < last[0].recorded_at:
                raise InvalidTransitionError(
                    f"memory {version.memory_id!r} v{version.version}: recorded_at precedes "
                    "the latest entry in the log; record time cannot go backwards"
                )
            core.check_citations(version, self._experiences(version.derived_from))
            self._db.execute(
                "INSERT INTO memory_versions (digest, memory_id, version, body) VALUES (?,?,?,?)",
                (version.digest, version.memory_id, version.version, version.canonical()),
            )
        return version.digest

    def _versions(
        self, clause: str = "ORDER BY seq", args: tuple[object, ...] = ()
    ) -> list[MemoryVersion]:
        rows = self._db.execute(f"SELECT digest, body FROM memory_versions {clause}", args)
        return _load(MemoryVersion, rows)

    def _experiences(self, digests: Iterable[str] | None = None) -> dict[str, Experience]:
        if digests is None:
            rows = self._db.execute("SELECT digest, body FROM experiences ORDER BY seq")
        else:
            wanted = tuple(digests)
            marks = ",".join("?" * len(wanted))
            rows = self._db.execute(
                f"SELECT digest, body FROM experiences WHERE digest IN ({marks}) ORDER BY seq",
                wanted,
            )
        return {e.digest: e for e in _load(Experience, rows)}

    def experiences(self) -> list[Experience]:
        """All experiences, in append order."""
        return list(self._experiences().values())

    def versions(self) -> list[MemoryVersion]:
        """All memory versions, in append (record-time) order."""
        return self._versions()

    def history(self, memory_id: str) -> list[MemoryVersion]:
        """One memory's full version chain, oldest first, including any tombstone."""
        return self._versions("WHERE memory_id = ? ORDER BY version", (memory_id,))

    def state_as_of(self, *, valid_at: datetime, known_at: datetime) -> MemoryState:
        # ponytail: full-log scan per query; add snapshots/indexes when logs outgrow memory.
        return core.state_as_of(self.versions(), valid_at=valid_at, known_at=known_at)

    def verify(self) -> None:
        """Re-check every stored record and invariant, independent of the append path.

        Detects tampering that bypassed the append-only triggers.
        """
        experiences = self._experiences()
        versions = self.versions()
        for earlier, later in pairwise(versions):
            if later.recorded_at < earlier.recorded_at:
                raise LogIntegrityError(f"record time goes backwards at {later.digest}")
        for v in versions:
            try:
                core.check_citations(v, experiences)
            except InvalidTransitionError as e:
                raise LogIntegrityError(str(e)) from e
        for memory_id in sorted({v.memory_id for v in versions}):
            try:
                core.beliefs(self.history(memory_id))
            except InvalidTransitionError as e:
                raise LogIntegrityError(str(e)) from e
