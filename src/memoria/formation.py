"""Memory formation: turning experiences into recorded memory operations.

A policy is pure: given an experience and a read-only view of history, it returns a
:class:`~memoria.core.FormationDecision` and, if memory changes, the new version.
:func:`form` applies one experience to a log atomically and records the decision
even when nothing changes, so every outcome is inspectable.

Two deterministic baselines are provided:

- :class:`EpisodicPolicy` stores every experience verbatim as its own memory.
- :class:`StatementPolicy` maintains one memory per key from an explicit statement
  language (``set k = v`` / ``correct k = v`` / ``forget k``). It does not interpret
  natural language; anything else is recorded as UNPARSED.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import NamedTuple, Protocol

from memoria.core import (
    Experience,
    FormationDecision,
    FormationReason,
    MemoryVersion,
    Operation,
)
from memoria.store import MemoryLog


class HistoryView(Protocol):
    def history(self, memory_id: str) -> list[MemoryVersion]: ...


class Proposal(NamedTuple):
    decision: FormationDecision
    version: MemoryVersion | None


class FormationPolicy(Protocol):
    """Pure decision function. ``name`` identifies the policy and its version in records."""

    @property
    def name(self) -> str: ...

    def decide(
        self, experience: Experience, view: HistoryView, *, recorded_at: datetime
    ) -> Proposal: ...


def form(
    log: MemoryLog, experience: Experience, policy: FormationPolicy, *, recorded_at: datetime
) -> FormationDecision:
    """Record ``experience``, apply the policy's decision, and record the decision. Atomic.

    An experience already in the log is not re-decided: the outcome is a
    DUPLICATE_EXPERIENCE decision, so repeated observations remain visible.
    """
    with log.transaction():
        if log.experience(experience.digest) is not None:
            decision = FormationDecision(
                experience=experience.digest,
                policy=policy.name,
                reason=FormationReason.DUPLICATE_EXPERIENCE,
                recorded_at=recorded_at,
            )
        else:
            log.append_experience(experience)
            decision, version = policy.decide(experience, log, recorded_at=recorded_at)
            if version is not None:
                log.append_version(version)
        log.append_record(decision)
    return decision


class EpisodicPolicy:
    """Every experience becomes its own immutable memory, valid from when it occurred.

    The "store raw episodes" baseline: nothing is ever superseded, so later
    contradictions coexist with what they contradict.
    """

    name = "episodic-v1"

    def decide(
        self, experience: Experience, view: HistoryView, *, recorded_at: datetime
    ) -> Proposal:
        memory_id = "episode:" + experience.digest.removeprefix("sha256:")
        version = MemoryVersion(
            memory_id=memory_id,
            version=1,
            operation=Operation.CREATE,
            content=experience.content,
            valid_from=experience.occurred_at,
            recorded_at=recorded_at,
            derived_from=(experience.digest,),
        )
        decision = FormationDecision(
            experience=experience.digest,
            policy=self.name,
            reason=FormationReason.NEW_EPISODE,
            memory_id=memory_id,
            version=version.digest,
            recorded_at=recorded_at,
        )
        return Proposal(decision, version)


_STATEMENT = re.compile(r"(set|correct) ([a-z0-9_.-]+) = (\S(?:.*\S)?)|(forget) ([a-z0-9_.-]+)")


class Statement(NamedTuple):
    verb: str  # "set" | "correct" | "forget"
    key: str
    value: str | None


def parse_statement(text: str) -> Statement | None:
    """Parse one statement; surrounding whitespace is ignored, nothing else is normalised."""
    m = _STATEMENT.fullmatch(text.strip())
    if m is None:
        return None
    if m[4]:
        return Statement("forget", m[5], None)
    return Statement(m[1], m[2], m[3])


def incarnations(view: HistoryView, key: str) -> list[list[MemoryVersion]]:
    """Chains for ``key``, ``key#2``, ...: a forgotten key relearned is a new memory."""
    chains: list[list[MemoryVersion]] = []
    while chain := view.history(key if not chains else f"{key}#{len(chains) + 1}"):
        chains.append(chain)
    return chains


class StatementPolicy:
    """One memory per key, maintained from explicit statements.

    For a live memory whose current belief is ``head`` (valid from ``head.valid_from``),
    a statement that occurred at ``t``:

    - ``t < head.valid_from``                    -> STALE (never rewrites the past)
    - same content as ``head``                   -> REDUNDANT
    - ``t == head.valid_from``, other content    -> CONFLICTING (first recorded wins)
    - ``set``                                    -> UPDATE, valid from ``t``
    - ``correct``                                -> CORRECT over ``head``'s interval
    - ``forget``                                 -> FORGET

    Without a live memory, ``set`` creates one (NEW_KEY or RELEARNED); ``correct`` and
    ``forget`` are UNKNOWN_KEY.
    """

    name = "statement-v1"

    def decide(
        self, experience: Experience, view: HistoryView, *, recorded_at: datetime
    ) -> Proposal:
        def outcome(
            reason: FormationReason,
            memory_id: str | None = None,
            basis: MemoryVersion | None = None,
            version: MemoryVersion | None = None,
        ) -> Proposal:
            decision = FormationDecision(
                experience=experience.digest,
                policy=self.name,
                reason=reason,
                memory_id=memory_id,
                basis=basis.digest if basis else None,
                version=version.digest if version else None,
                recorded_at=recorded_at,
            )
            return Proposal(decision, version)

        statement = parse_statement(experience.content)
        if statement is None:
            return outcome(FormationReason.UNPARSED)
        verb, key, value = statement
        content = None if value is None else f"{key} = {value}"
        t = experience.occurred_at
        chains = incarnations(view, key)
        head = chains[-1][-1] if chains else None

        if head is None or head.operation is Operation.FORGET:
            if verb != "set":
                return outcome(FormationReason.UNKNOWN_KEY, head.memory_id if head else None, head)
            memory_id = key if head is None else f"{key}#{len(chains) + 1}"
            created = MemoryVersion(
                memory_id=memory_id,
                version=1,
                operation=Operation.CREATE,
                content=content,
                valid_from=t,
                recorded_at=recorded_at,
                derived_from=(experience.digest,),
            )
            reason = FormationReason.NEW_KEY if head is None else FormationReason.RELEARNED
            return outcome(reason, memory_id, head, created)

        assert head.valid_from is not None  # live head is never a tombstone
        if t < head.valid_from:
            return outcome(FormationReason.STALE, head.memory_id, head)
        if verb == "forget":
            forgotten = head.successor(
                Operation.FORGET, recorded_at=recorded_at, derived_from=(experience.digest,)
            )
            return outcome(FormationReason.EXPLICIT_FORGET, head.memory_id, head, forgotten)
        if content == head.content:
            return outcome(FormationReason.REDUNDANT, head.memory_id, head)
        if t == head.valid_from:
            return outcome(FormationReason.CONFLICTING, head.memory_id, head)
        if verb == "set":
            operation, reason, start = Operation.UPDATE, FormationReason.VALUE_CHANGED, t
        else:
            operation, reason, start = (
                Operation.CORRECT,
                FormationReason.EXPLICIT_CORRECTION,
                head.valid_from,
            )
        changed = head.successor(
            operation,
            recorded_at=recorded_at,
            content=content,
            valid_from=start,
            derived_from=(experience.digest,),
        )
        return outcome(reason, head.memory_id, head, changed)
