"""Deterministic research scenarios: small, realistic long-horizon experience streams.

Each step is an experience plus the record time at which the system ingests it.
Contents follow the statement language of :class:`memoria.formation.StatementPolicy`
interleaved with ordinary chatter, so the same stream exercises every formation
outcome under that policy and serves as unstructured episodes for others.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import NamedTuple

from memoria.core import Experience

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def day(n: float) -> datetime:
    return EPOCH + timedelta(days=n)


class Step(NamedTuple):
    experience: Experience
    recorded_at: datetime


def _step(occurred: float, content: str, source: str, recorded: float | None = None) -> Step:
    return Step(
        Experience(source=source, content=content, occurred_at=day(occurred)),
        day(occurred if recorded is None else recorded),
    )


def relocation_year() -> list[Step]:
    """One user over a year: moves, a job misremembered then corrected, a forgotten and
    relearned employer, late and conflicting reports, duplicates and malformed input.

    The comment on each step is the outcome ``statement-v1`` records for it.
    """
    return [
        _step(0, "set home = Paris", "chat:0"),  # NEW_KEY
        _step(3, "set employer = Acme", "chat:3"),  # NEW_KEY
        _step(5, "We had pasta for lunch near the office.", "chat:5"),  # UNPARSED
        _step(40, "set employer = Acme", "chat:40"),  # REDUNDANT
        _step(55, "set home = Berlin", "chat:55", recorded=60),  # VALUE_CHANGED (late)
        _step(61, "correct employer = Globex", "chat:61"),  # EXPLICIT_CORRECTION
        _step(50, "set home = Munich", "email:50", recorded=90),  # STALE
        _step(0, "set home = Paris", "chat:0", recorded=120),  # DUPLICATE_EXPERIENCE
        _step(150, "forget employer", "chat:150"),  # EXPLICIT_FORGET
        _step(200, "set employer = Initech", "chat:200"),  # RELEARNED -> employer#2
        _step(250, "correct pet = cat", "chat:250"),  # UNKNOWN_KEY
        _step(298, "set home = Hamburg", "chat:298a", recorded=300),  # VALUE_CHANGED
        _step(298, "set home = Bremen", "chat:298b", recorded=300),  # CONFLICTING
        _step(310, "forget home = Hamburg", "chat:310"),  # UNPARSED (forget takes no value)
        _step(330, "set pet = dog", "chat:330"),  # NEW_KEY
        _step(340, "Remind me where I live these days?", "chat:340"),  # UNPARSED
    ]
