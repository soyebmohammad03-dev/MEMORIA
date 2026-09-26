import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memoria.core import Experience, InvalidTransitionError, MemoryVersion, Operation
from memoria.store import LogIntegrityError, MemoryLog

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def day(n: int) -> datetime:
    return T0 + timedelta(days=n)


PARIS = Experience(source="dialogue:1", content="I live in Paris.", occurred_at=day(0))
MOVED = Experience(source="dialogue:9", content="I moved to Berlin last week.", occurred_at=day(27))
TEA = Experience(source="dialogue:2", content="I like tea.", occurred_at=day(1))


@contextmanager
def raw(path: Path) -> Iterator[sqlite3.Connection]:
    """Direct SQLite access, committed and closed on exit."""
    db = sqlite3.connect(path)
    try:
        with db:
            yield db
    finally:
        db.close()


def create(memory_id: str, content: str, exp: Experience, at: int) -> MemoryVersion:
    return MemoryVersion(
        memory_id=memory_id,
        version=1,
        operation=Operation.CREATE,
        content=content,
        valid_from=day(at),
        recorded_at=day(at),
        derived_from=(exp.digest,),
    )


def populate(log: MemoryLog) -> None:
    """A history exercising every operation."""
    for e in (PARIS, TEA, MOVED):
        log.append_experience(e)
    home = create("home", "Paris", PARIS, 0)
    tea = create("drink", "tea", TEA, 1)
    log.append_version(home)
    log.append_version(tea)
    moved = home.successor(
        Operation.UPDATE,
        recorded_at=day(28),
        content="Berlin",
        valid_from=day(20),
        derived_from=(MOVED.digest,),
    )
    log.append_version(moved)
    log.append_version(
        tea.successor(Operation.CORRECT, recorded_at=day(29), content="coffee", valid_from=day(1))
    )
    log.append_version(moved.successor(Operation.FORGET, recorded_at=day(40)))


QUERIES = [(v, k) for v in (-1, 0, 10, 25, 50) for k in (-1, 0, 1, 28, 29, 40, 60)]


def states(log: MemoryLog) -> list[str]:
    return [log.state_as_of(valid_at=day(v), known_at=day(k)).digest for v, k in QUERIES]


@pytest.fixture
def log(tmp_path: Path) -> Iterator[MemoryLog]:
    with MemoryLog(tmp_path / "log.db") as log:
        yield log


def test_bitemporal_state_from_log(log: MemoryLog) -> None:
    populate(log)

    def held(v: int, k: int) -> dict[str, str | None]:
        state = log.state_as_of(valid_at=day(v), known_at=day(k))
        return {m.memory_id: m.content for m in state.memories}

    assert held(25, 27) == {"home": "Paris", "drink": "tea"}
    assert held(25, 28) == {"home": "Berlin", "drink": "tea"}
    assert held(10, 29) == {"home": "Paris", "drink": "coffee"}
    assert held(25, 40) == {"drink": "coffee"}
    log.verify()


def test_replay_reproduces_state_bit_for_bit(tmp_path: Path) -> None:
    """Phase 1 exit criterion."""
    with MemoryLog(tmp_path / "a.db") as original:
        populate(original)
        expected = states(original)
        experiences, versions = original.experiences(), original.versions()

    with MemoryLog(tmp_path / "a.db") as reopened:
        assert states(reopened) == expected

    with MemoryLog(tmp_path / "b.db") as replayed:
        for e in experiences:
            replayed.append_experience(e)
        for v in versions:
            replayed.append_version(v)
        assert states(replayed) == expected

    def dump(name: str) -> list[tuple[object, ...]]:
        with raw(tmp_path / name) as db:
            return [
                *db.execute("SELECT * FROM experiences ORDER BY seq"),
                *db.execute("SELECT * FROM memory_versions ORDER BY seq"),
            ]

    assert dump("a.db") == dump("b.db")


def test_stored_body_is_canonical_json(log: MemoryLog, tmp_path: Path) -> None:
    populate(log)
    with raw(tmp_path / "log.db") as db:
        for digest, body in db.execute("SELECT digest, body FROM memory_versions"):
            assert MemoryVersion.model_validate_json(body).canonical() == body
            assert MemoryVersion.model_validate_json(body).digest == digest


def test_experiences_are_idempotent(log: MemoryLog) -> None:
    assert log.append_experience(PARIS) == log.append_experience(PARIS) == PARIS.digest
    assert log.experiences() == [PARIS]


def test_history_keeps_every_version(log: MemoryLog) -> None:
    populate(log)
    ops = [v.operation for v in log.history("home")]
    assert ops == [Operation.CREATE, Operation.UPDATE, Operation.FORGET]


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE memory_versions SET body = '{}'",
        "DELETE FROM memory_versions",
        "UPDATE experiences SET body = '{}'",
        "DELETE FROM experiences",
    ],
)
def test_log_tables_are_append_only(log: MemoryLog, sql: str) -> None:
    populate(log)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        log._db.execute(sql)


def test_rejected_append_leaves_log_unchanged(log: MemoryLog) -> None:
    populate(log)
    before = log.versions()
    stale = log.history("drink")[0].successor(
        Operation.UPDATE, recorded_at=day(50), content="water", valid_from=day(50)
    )
    with pytest.raises(InvalidTransitionError, match="expected version 3"):
        log.append_version(stale)
    assert log.versions() == before


def test_forgotten_memory_cannot_be_extended(log: MemoryLog) -> None:
    populate(log)
    tomb = log.history("home")[-1]
    with pytest.raises(InvalidTransitionError, match="terminal"):
        log.append_version(
            tomb.successor(
                Operation.UPDATE, recorded_at=day(50), content="Rome", valid_from=day(50)
            )
        )


def test_duplicate_create_rejected(log: MemoryLog) -> None:
    log.append_experience(PARIS)
    log.append_version(create("home", "Paris", PARIS, 0))
    with pytest.raises(InvalidTransitionError):
        log.append_version(create("home", "Lyon", PARIS, 1))


def test_record_time_cannot_go_backwards(log: MemoryLog) -> None:
    log.append_experience(PARIS)
    log.append_version(create("a", "x", PARIS, 5))
    with pytest.raises(InvalidTransitionError, match="record time"):
        log.append_version(create("b", "y", PARIS, 4))


def test_citations_must_exist_and_precede(log: MemoryLog) -> None:
    with pytest.raises(InvalidTransitionError, match="unknown"):
        log.append_version(create("home", "Paris", PARIS, 0))
    log.append_experience(MOVED)
    with pytest.raises(InvalidTransitionError, match="after"):
        log.append_version(create("home", "Berlin", MOVED, 20))
    assert log.versions() == []


def _tamper(path: Path, sql: str) -> None:
    with raw(path) as db:
        db.execute("DROP TRIGGER memory_versions_no_update")
        db.execute("DROP TRIGGER memory_versions_no_delete")
        db.execute(sql)


def test_tampered_body_detected(tmp_path: Path) -> None:
    with MemoryLog(tmp_path / "t.db") as log:
        populate(log)
    _tamper(tmp_path / "t.db", "UPDATE memory_versions SET body = replace(body, 'Paris', 'Rome')")
    with MemoryLog(tmp_path / "t.db") as log, pytest.raises(LogIntegrityError, match="digest"):
        log.versions()


def test_removed_version_detected_by_verify(tmp_path: Path) -> None:
    with MemoryLog(tmp_path / "t.db") as log:
        populate(log)
    _tamper(
        tmp_path / "t.db", "DELETE FROM memory_versions WHERE memory_id = 'home' AND version = 2"
    )
    with (
        MemoryLog(tmp_path / "t.db") as log,
        pytest.raises(LogIntegrityError, match="expected version 2"),
    ):
        log.verify()


def test_refuses_foreign_or_future_databases(tmp_path: Path) -> None:
    with raw(tmp_path / "other.db") as db:
        db.execute("CREATE TABLE t (x)")
    with pytest.raises(LogIntegrityError):
        MemoryLog(tmp_path / "other.db")
    with raw(tmp_path / "future.db") as db:
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(LogIntegrityError):
        MemoryLog(tmp_path / "future.db")
