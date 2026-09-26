"""Append-only SQLite operation log: the source of truth for memory history.

Storage only. Every domain rule is enforced by :mod:`memoria.core`, so another backend
(e.g. PostgreSQL) has to reproduce storage and append-only enforcement, not semantics.
SQLite-specific parts: the guard triggers, savepoints and ``PRAGMA user_version``.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import ValidationError

from memoria import core
from memoria.core import (
    Experience,
    FormationDecision,
    InvalidTransitionError,
    MemoryState,
    MemoryVersion,
    Record,
    Response,
    RetrievalTrace,
)

SCHEMA_VERSION = 2


def _guards(table: str) -> str:
    return "".join(
        f"CREATE TRIGGER {table}_no_{op.lower()} BEFORE {op} ON {table} "
        f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;\n"
        for op in ("UPDATE", "DELETE")
    )


_SCHEMA_V1 = f"""
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
{_guards("experiences")}{_guards("memory_versions")}"""

# v2: derived observations (formation decisions, retrieval traces, responses).
# `mark` is the record's time coordinate in integer microseconds since the epoch:
# recorded_at for decisions, the query's known_at for traces, NULL for responses.
_SCHEMA_V2 = f"""
CREATE TABLE records (
    seq INTEGER PRIMARY KEY,
    digest TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    mark INTEGER,
    body TEXT NOT NULL
);
{_guards("records")}"""

LogRecord = FormationDecision | RetrievalTrace | Response
_KINDS: dict[type[Record], str] = {
    FormationDecision: "formation_decision",
    RetrievalTrace: "retrieval_trace",
    Response: "response",
}
_EXPORT_KINDS: dict[type[Record], str] = {
    Experience: "experience",
    MemoryVersion: "memory_version",
    **_KINDS,
}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _mark(t: datetime) -> int:
    return (t - _EPOCH) // timedelta(microseconds=1)


def _unmark(m: int | None) -> datetime | None:
    return None if m is None else _EPOCH + timedelta(microseconds=m)


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
    """Append-only log of experiences, memory versions and derived records.

    Every append is validated. Record time is the log's own axis: versions and
    formation decisions are appended in non-decreasing record time, and strictly
    after the ``known_at`` of any stored retrieval, whose view of the past is thereby
    frozen. Nothing is ever updated or deleted.
    """

    # ponytail: single-writer design; appends serialise on SQLite's write lock.

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(path, isolation_level=None)  # transactions are explicit
        self._depth = 0
        found = self._db.execute("PRAGMA user_version").fetchone()[0]
        if found == 0 and not self._db.execute("SELECT 1 FROM sqlite_master").fetchone():
            script = _SCHEMA_V1 + _SCHEMA_V2
        elif found == 1:
            script = _SCHEMA_V2  # additive migration; existing rows are untouched
        elif found == SCHEMA_VERSION:
            return
        else:
            self._db.close()
            raise LogIntegrityError(f"{path}: not a MEMORIA log with schema v{SCHEMA_VERSION}")
        self._db.executescript(f"BEGIN;\n{script}PRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, t: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """An atomic unit of work. Nests via savepoints; an error undoes its level only."""
        depth = self._depth
        self._db.execute(f"SAVEPOINT s{depth}" if depth else "BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        except BaseException:
            if depth:
                self._db.execute(f"ROLLBACK TO s{depth}")
                self._db.execute(f"RELEASE s{depth}")
            else:
                self._db.execute("ROLLBACK")
            raise
        else:
            self._db.execute(f"RELEASE s{depth}" if depth else "COMMIT")
        finally:
            self._depth = depth

    # --- appends ----------------------------------------------------------------------

    def append_experience(self, experience: Experience) -> str:
        """Idempotent: an identical experience is the same content-addressed record."""
        digest = experience.digest
        with self.transaction():
            self._db.execute(
                "INSERT OR IGNORE INTO experiences (digest, body) VALUES (?, ?)",
                (digest, experience.canonical()),
            )
        return digest

    def append_version(self, version: MemoryVersion) -> str:
        """Append after validating it against the memory's full history and its citations."""
        with self.transaction():
            core.beliefs([*self.history(version.memory_id), version])
            self._check_record_time(version.recorded_at, f"memory {version.memory_id!r}")
            core.check_citations(version, self._experiences(version.derived_from))
            self._db.execute(
                "INSERT INTO memory_versions (digest, memory_id, version, body) VALUES (?,?,?,?)",
                (version.digest, version.memory_id, version.version, version.canonical()),
            )
        return version.digest

    def append_record(self, record: LogRecord) -> str:
        """Append a decision, trace or response after checking every reference it makes.

        Idempotent: an identical record is the same content-addressed observation.
        """
        digest = record.digest
        with self.transaction():
            if self._db.execute("SELECT 1 FROM records WHERE digest = ?", (digest,)).fetchone():
                return digest
            mark: int | None = None
            match record:
                case FormationDecision():
                    self._check_decision(record)
                    self._check_record_time(record.recorded_at, "formation decision")
                    mark = _mark(record.recorded_at)
                case RetrievalTrace():
                    q = record.query
                    core.check_trace(
                        record, self.state_as_of(valid_at=q.valid_at, known_at=q.known_at)
                    )
                    mark = _mark(q.known_at)
                case Response():
                    trace = self.record(RetrievalTrace, record.trace)
                    if trace is None:
                        raise InvalidTransitionError(f"response cites unknown trace {record.trace}")
                    core.check_response(record, trace)
            self._db.execute(
                "INSERT INTO records (digest, kind, mark, body) VALUES (?, ?, ?, ?)",
                (digest, _KINDS[type(record)], mark, record.canonical()),
            )
        return digest

    def _check_record_time(self, at: datetime, what: str) -> None:
        floor = max(
            (t for t in (self._last_version_time(), self._max_mark("formation_decision")) if t),
            default=None,
        )
        if floor and at < floor:
            raise InvalidTransitionError(
                f"{what}: recorded_at precedes the latest entry in the log; "
                "record time cannot go backwards"
            )
        frozen = self._max_mark("retrieval_trace")
        if frozen and at <= frozen:
            raise InvalidTransitionError(
                f"{what}: recorded_at must be after {frozen.isoformat()}, the known_at of a "
                "stored retrieval; the state it observed is frozen"
            )

    def _check_decision(self, d: FormationDecision) -> None:
        experience = self.experience(d.experience)
        if experience is None:
            raise InvalidTransitionError(f"decision cites unknown experience {d.experience}")
        refs = {"version": d.version, "basis": d.basis}
        found = {k: self.version(v) if v else None for k, v in refs.items()}
        for k, v in refs.items():
            if v and found[k] is None:
                raise InvalidTransitionError(f"decision cites unknown {k} {v}")
        core.check_decision(d, experience, found["version"], found["basis"])
        first = self._db.execute(
            "SELECT digest, body FROM records WHERE kind = ? ORDER BY seq LIMIT 1",
            (_KINDS[FormationDecision],),
        ).fetchall()
        if first and (policy := _load(FormationDecision, first)[0].policy) != d.policy:
            raise InvalidTransitionError(
                f"decision by policy {d.policy!r}: this log is formed by {policy!r}"
            )

    # --- reads --------------------------------------------------------------------------

    def _last_version_time(self) -> datetime | None:
        last = self._versions("ORDER BY seq DESC LIMIT 1")
        return last[0].recorded_at if last else None

    def _max_mark(self, kind: str) -> datetime | None:
        row = self._db.execute("SELECT MAX(mark) FROM records WHERE kind = ?", (kind,)).fetchone()
        return _unmark(row[0])

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

    def experience(self, digest: str) -> Experience | None:
        return self._experiences([digest]).get(digest)

    def versions(self) -> list[MemoryVersion]:
        """All memory versions, in append (record-time) order."""
        return self._versions()

    def version(self, digest: str) -> MemoryVersion | None:
        found = self._versions("WHERE digest = ?", (digest,))
        return found[0] if found else None

    def history(self, memory_id: str) -> list[MemoryVersion]:
        """One memory's full version chain, oldest first, including any tombstone."""
        return self._versions("WHERE memory_id = ? ORDER BY version", (memory_id,))

    def records[R: LogRecord](self, model: type[R]) -> list[R]:
        """All records of one kind, in append order."""
        rows = self._db.execute(
            "SELECT digest, body FROM records WHERE kind = ? ORDER BY seq", (_KINDS[model],)
        )
        return _load(model, rows)

    def record[R: LogRecord](self, model: type[R], digest: str) -> R | None:
        rows = self._db.execute(
            "SELECT digest, body FROM records WHERE kind = ? AND digest = ?",
            (_KINDS[model], digest),
        )
        found = _load(model, rows)
        return found[0] if found else None

    def export(self) -> bytes:
        """Canonical JSON Lines serialisation of the whole log: ``{"kind", "record"}`` per line.

        Order: experiences (append order); then versions and decisions merged by record
        time (a version before a decision at the same instant, append order otherwise);
        then traces and responses in append order. Every such order is a valid replay,
        and equal logs export to identical bytes.
        """
        timed: list[tuple[datetime, int, int, Record]] = [
            (v.recorded_at, 0, i, v) for i, v in enumerate(self.versions())
        ]
        timed += [(d.recorded_at, 1, i, d) for i, d in enumerate(self.records(FormationDecision))]
        rows = self._db.execute(
            "SELECT kind, digest, body FROM records WHERE kind IN (?, ?) ORDER BY seq",
            (_KINDS[RetrievalTrace], _KINDS[Response]),
        )
        observations = [
            _load(RetrievalTrace if kind == _KINDS[RetrievalTrace] else Response, [(d, b)])[0]
            for kind, d, b in rows
        ]
        ordered: list[Record] = [
            *self.experiences(),
            *(r for *_, r in sorted(timed, key=lambda t: t[:3])),
            *observations,
        ]
        return "".join(
            core.canonical_json(
                {"kind": _EXPORT_KINDS[type(r)], "record": r.model_dump(mode="json")}
            )
            + "\n"
            for r in ordered
        ).encode("utf-8")

    @classmethod
    def load(cls, data: bytes, path: str | Path = ":memory:") -> MemoryLog:
        """Rebuild a log from :meth:`export` output, re-validating every record on append.

        The target must be empty. Non-canonical, malformed or invalid lines are rejected.
        """
        log = cls(path)
        try:
            if log._db.execute(
                "SELECT (SELECT COUNT(*) FROM experiences) + (SELECT COUNT(*) FROM records)"
            ).fetchone()[0]:
                raise LogIntegrityError(f"{path}: can only load into an empty log")
            models = {kind: model for model, kind in _EXPORT_KINDS.items()}
            with log.transaction():
                for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
                    try:
                        entry = json.loads(line)
                        model = models[entry["kind"]]
                        record = model.model_validate(entry["record"])
                    except (ValueError, KeyError, TypeError) as e:
                        raise LogIntegrityError(f"line {n}: not a valid log entry") from e
                    if core.canonical_json(entry) != line:
                        raise LogIntegrityError(f"line {n}: not in canonical form")
                    try:
                        match record:
                            case Experience():
                                log.append_experience(record)
                            case MemoryVersion():
                                log.append_version(record)
                            case FormationDecision() | RetrievalTrace() | Response():
                                log.append_record(record)
                    except InvalidTransitionError as e:
                        raise LogIntegrityError(f"line {n}: {e}") from e
        except BaseException:
            log.close()
            raise
        return log

    def state_as_of(self, *, valid_at: datetime, known_at: datetime) -> MemoryState:
        # ponytail: full-log scan per query; add snapshots/indexes when logs outgrow memory.
        return core.state_as_of(self.versions(), valid_at=valid_at, known_at=known_at)

    def verify(self) -> None:
        """Re-check every stored record and invariant, independent of the append path.

        Detects tampering that bypassed the append-only triggers.
        """
        experiences = self._experiences()
        versions = self.versions()
        decisions = self.records(FormationDecision)
        for kind, times in (
            ("version", [v.recorded_at for v in versions]),
            ("decision", [d.recorded_at for d in decisions]),
        ):
            for earlier, later in pairwise(times):
                if later < earlier:
                    raise LogIntegrityError(f"{kind} record time goes backwards at {later}")
        try:
            for v in versions:
                core.check_citations(v, experiences)
            for memory_id in sorted({v.memory_id for v in versions}):
                core.beliefs(self.history(memory_id))
            if len({d.policy for d in decisions}) > 1:
                raise InvalidTransitionError("log contains decisions from more than one policy")
            for d in decisions:
                self._check_decision(d)
            traces = {t.digest: t for t in self.records(RetrievalTrace)}
            for t in traces.values():
                q = t.query
                core.check_trace(
                    t, core.state_as_of(versions, valid_at=q.valid_at, known_at=q.known_at)
                )
            for r in self.records(Response):
                if r.trace not in traces:
                    raise InvalidTransitionError(f"response cites unknown trace {r.trace}")
                core.check_response(r, traces[r.trace])
        except InvalidTransitionError as e:
            raise LogIntegrityError(str(e)) from e
