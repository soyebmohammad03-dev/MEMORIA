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
from typing import Annotated, Any, ClassVar, Literal, NamedTuple, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)


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


class ExtensibleRecord(Record):
    """A record whose schema can grow without changing existing digests (I33).

    Fields named in ``_evolved`` are left out of every serialisation — hence of the
    canonical form and the digest — while they hold their default. A field added this
    way must default to the behaviour records had before it existed.
    """

    _evolved: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _omit_defaults(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        fields = type(self).model_fields
        for name in self._evolved:
            if getattr(self, name) == fields[name].default:
                data.pop(name, None)
        return data


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


# --- formation decisions ----------------------------------------------------------------


class FormationReason(StrEnum):
    """Why a formation policy did (or did not) change memory. A closed vocabulary."""

    NEW_KEY = "new_key"  # CREATE: nothing was known about the key
    RELEARNED = "relearned"  # CREATE: the key's previous memory was forgotten
    NEW_EPISODE = "new_episode"  # CREATE: episodic policy stores every experience
    VALUE_CHANGED = "value_changed"  # UPDATE: a newer value for a live memory
    EXPLICIT_CORRECTION = "explicit_correction"  # CORRECT
    EXPLICIT_FORGET = "explicit_forget"  # FORGET
    DUPLICATE_EXPERIENCE = "duplicate_experience"  # skip: this exact experience was seen
    REDUNDANT = "redundant"  # skip: asserts what is already believed
    STALE = "stale"  # skip: predates the current belief
    CONFLICTING = "conflicting"  # skip: contradicts a belief asserted for the same instant
    UNKNOWN_KEY = "unknown_key"  # skip: correct/forget with no live memory
    UNPARSED = "unparsed"  # skip: not a statement the policy understands

    @property
    def operation(self) -> Operation | None:
        return _REASON_OPERATION.get(self)


_REASON_OPERATION = {
    FormationReason.NEW_KEY: Operation.CREATE,
    FormationReason.RELEARNED: Operation.CREATE,
    FormationReason.NEW_EPISODE: Operation.CREATE,
    FormationReason.VALUE_CHANGED: Operation.UPDATE,
    FormationReason.EXPLICIT_CORRECTION: Operation.CORRECT,
    FormationReason.EXPLICIT_FORGET: Operation.FORGET,
}
# Reasons that are only meaningful relative to an existing version (the decision's basis).
_NEEDS_BASIS = frozenset(
    {
        FormationReason.RELEARNED,
        FormationReason.VALUE_CHANGED,
        FormationReason.EXPLICIT_CORRECTION,
        FormationReason.EXPLICIT_FORGET,
        FormationReason.REDUNDANT,
        FormationReason.STALE,
        FormationReason.CONFLICTING,
    }
)
_NO_BASIS = frozenset(
    {
        FormationReason.NEW_KEY,
        FormationReason.NEW_EPISODE,
        FormationReason.DUPLICATE_EXPERIENCE,
        FormationReason.UNPARSED,
    }
)


class FormationDecision(Record):
    """The recorded outcome of processing one experience, including decisions to do nothing.

    ``basis`` is the version the decision was judged against (e.g. the belief an UPDATE
    supersedes, or the tombstone a RELEARNED memory follows); ``version`` is the version
    the decision appended, if any. UNKNOWN_KEY may cite a tombstone as basis.
    """

    experience: Digest
    policy: str = Field(min_length=1)
    reason: FormationReason
    memory_id: str | None = Field(default=None, min_length=1)
    basis: Digest | None = None
    version: Digest | None = None
    recorded_at: UTCDatetime

    @property
    def operation(self) -> Operation | None:
        return self.reason.operation

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if (self.version is None) != (self.operation is None):
            raise ValueError(f"{self.reason.value}: version must be present iff memory changes")
        if self.reason in _NEEDS_BASIS and self.basis is None:
            raise ValueError(f"{self.reason.value} requires a basis version")
        if self.reason in _NO_BASIS and self.basis is not None:
            raise ValueError(f"{self.reason.value} cannot have a basis version")
        if (self.memory_id is None) != (self.basis is None and self.version is None):
            raise ValueError("memory_id is required iff the decision targets a memory")
        return self


def check_decision(
    decision: FormationDecision,
    experience: Experience,
    version: MemoryVersion | None,
    basis: MemoryVersion | None,
) -> None:
    """Cross-record consistency of a decision with the records it references."""
    where = f"decision on {decision.experience}"
    if experience.digest != decision.experience:
        raise InvalidTransitionError(f"{where}: experience does not match")
    if decision.recorded_at < experience.occurred_at:
        raise InvalidTransitionError(f"{where}: recorded before the experience occurred")
    if (basis is None) != (decision.basis is None) or (basis and basis.digest != decision.basis):
        raise InvalidTransitionError(f"{where}: basis does not match")
    if (version is None) != (decision.version is None):
        raise InvalidTransitionError(f"{where}: version does not match")
    if version is None:
        return
    if version.digest != decision.version:
        raise InvalidTransitionError(f"{where}: version does not match")
    if version.operation is not decision.operation or version.memory_id != decision.memory_id:
        raise InvalidTransitionError(f"{where}: version is not the recorded operation")
    if version.recorded_at != decision.recorded_at:
        raise InvalidTransitionError(f"{where}: version and decision record times differ")
    if decision.experience not in version.derived_from:
        raise InvalidTransitionError(f"{where}: version does not cite the experience")
    if version.operation is not Operation.CREATE and version.supersedes != decision.basis:
        raise InvalidTransitionError(f"{where}: version does not supersede the basis")


# --- retrieval traces and responses -------------------------------------------------------

Scalar = int | float | str


def _named_inputs(pairs: tuple[tuple[str, Scalar], ...]) -> tuple[tuple[str, Scalar], ...]:
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError("input names must be unique")
    return tuple(sorted(pairs, key=lambda p: p[0]))


# Named raw inputs in canonical (name-sorted) order.
Inputs = Annotated[tuple[tuple[str, Scalar], ...], AfterValidator(_named_inputs)]
Score = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def quantize(x: float) -> float:
    """Round a derived score so recorded values do not depend on last-bit libm differences."""
    return round(x, 12)


class Query(Record):
    """A retrieval request, pinned to both time axes so it is reproducible."""

    text: str
    valid_at: UTCDatetime
    known_at: UTCDatetime
    limit: int = Field(ge=1)


class SignalSpec(Record):
    name: str = Field(min_length=1)
    weight: float = Field(gt=0, allow_inf_nan=False)
    params: Inputs = ()


class RetrieverSpec(Record):
    """Everything that determines ranking: signals, their parameters and weights, the gate."""

    name: str = Field(min_length=1)
    gate: str  # a candidate is eligible iff this signal scores > 0
    signals: tuple[SignalSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        names = [s.name for s in self.signals]
        if len(set(names)) != len(names):
            raise ValueError("signal names must be unique")
        if self.gate not in names:
            raise ValueError(f"gate {self.gate!r} is not one of the signals")
        return self


class SignalEvidence(Record):
    """One signal's score for one candidate, with the named raw inputs that produced it."""

    signal: str
    score: Score
    inputs: Inputs = ()


class Candidate(Record):
    """A memory from the queried state, with every signal's evidence and its combined total."""

    version: Digest
    memory_id: str
    eligible: bool
    total: Score
    signals: tuple[SignalEvidence, ...]


def combine(spec: RetrieverSpec, signals: Sequence[SignalEvidence]) -> tuple[bool, float]:
    """(eligible, total): the gate decides eligibility; total is the weighted sum."""
    by_name = {s.signal: s.score for s in signals}
    total = quantize(sum(s.weight * by_name[s.name] for s in spec.signals))
    return by_name[spec.gate] > 0, total


def rank_key(c: Candidate) -> tuple[bool, float, str]:
    """Eligible first, then higher total, then memory_id: a total, deterministic order."""
    return (not c.eligible, -c.total, c.memory_id)


class RetrievalTrace(Record):
    """A complete, self-checking record of one retrieval.

    Every memory in the queried state appears as a candidate, so the trace answers
    why each memory was or was not returned. The validator re-derives eligibility,
    totals, ranking and selection, so an internally inconsistent trace cannot exist.
    """

    query: Query
    retriever: RetrieverSpec
    state: Digest  # digest of the MemoryState at (query.valid_at, query.known_at)
    candidates: tuple[Candidate, ...]  # ranked
    selected: tuple[Digest, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        names = [s.name for s in self.retriever.signals]
        for c in self.candidates:
            if [s.signal for s in c.signals] != names:
                raise ValueError(f"candidate {c.memory_id}: signals do not match the retriever")
            if combine(self.retriever, c.signals) != (c.eligible, c.total):
                raise ValueError(f"candidate {c.memory_id}: eligibility/total not reproducible")
        if len({c.memory_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("a memory appears more than once among candidates")
        if list(self.candidates) != sorted(self.candidates, key=rank_key):
            raise ValueError("candidates are not in rank order")
        eligible = [c.version for c in self.candidates if c.eligible]
        if self.selected != tuple(eligible[: self.query.limit]):
            raise ValueError("selected is not the top eligible candidates")
        return self


def check_trace(trace: RetrievalTrace, state: MemoryState) -> None:
    """The trace must describe exactly the state it claims to have queried."""
    if (state.valid_at, state.known_at) != (trace.query.valid_at, trace.query.known_at):
        raise InvalidTransitionError("trace: state is not at the query's time coordinates")
    if state.digest != trace.state:
        raise InvalidTransitionError("trace: state digest does not match the log")
    if sorted(c.version for c in trace.candidates) != sorted(m.digest for m in state.memories):
        raise InvalidTransitionError("trace: candidates are not exactly the state's memories")


class Response(Record):
    """An answer and the exact memory versions it relied on. ``output=None`` = abstained."""

    trace: Digest
    responder: str = Field(min_length=1)
    output: str | None
    cited: tuple[Digest, ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if (self.output is None) != (not self.cited):
            raise ValueError("a response cites evidence iff it answers")
        if len(set(self.cited)) != len(self.cited):
            raise ValueError("cited versions must be unique")
        return self


def check_response(response: Response, trace: RetrievalTrace) -> None:
    if response.trace != trace.digest:
        raise InvalidTransitionError("response: trace does not match")
    if not set(response.cited) <= set(trace.selected):
        raise InvalidTransitionError("response: cites a version the retrieval did not select")


# --- derived memory: lineage of consolidated representations ------------------------------


class Level(StrEnum):
    """The memory hierarchy. Every level above L1 is analysis over immutable history."""

    L0 = "L0"  # raw experiences (evidence)
    L1 = "L1"  # memory versions formed from experiences (the log)
    L2 = "L2"  # consolidated factual memories: groups of L1 versions judged equivalent
    L3 = "L3"  # concept memories: an entity's attributes, abstracted from L2 facts
    L4 = "L4"  # longitudinal patterns: timelines (abstracted) and co-changes (inferred)


class EpistemicStatus(StrEnum):
    """What kind of claim a memory makes about the world."""

    OBSERVED = "observed"  # evidence, or a memory formed directly from it (L0, L1)
    DERIVED = "derived"  # a grouping of observed memories; asserts nothing they do not
    ABSTRACTED = "abstracted"  # a restatement over several memories (profile, timeline)
    INFERRED = "inferred"  # a pattern no single piece of evidence states; never a fact


_OPERATION_STATUS = {
    "promote": EpistemicStatus.DERIVED,
    "merge": EpistemicStatus.DERIVED,
    "abstract_entity": EpistemicStatus.ABSTRACTED,
    "abstract_timeline": EpistemicStatus.ABSTRACTED,
    "infer_co_change": EpistemicStatus.INFERRED,
}
_OPERATION_LEVEL = {
    "promote": Level.L2,
    "merge": Level.L2,
    "abstract_entity": Level.L3,
    "abstract_timeline": Level.L4,
    "infer_co_change": Level.L4,
}


class DerivedMemory(Record):
    """A memory produced by consolidation. It is never evidence and never enters the log.

    ``versions`` are the L1 memory versions it covers and ``derived_from`` their source
    experiences (the same meaning as on a MemoryVersion, so provenance tools apply);
    ``parents`` are the memories it was built from directly (L1 version digests for L2,
    derived memory ids above). ``support`` counts supporting experiences, ``disputed``
    those whose claims assert another value for its key. ``recorded_at`` is the logical
    time the consolidation ran. ``model`` names the embedder whose similarities decided a
    merge, if any. ``lost`` counts input features its content does not carry (see the
    consolidation loss report); lineage still reaches them.
    """

    memory_id: str = Field(min_length=1)
    level: Level
    status: EpistemicStatus
    operation: Literal[
        "promote", "merge", "abstract_entity", "abstract_timeline", "infer_co_change"
    ]
    rule: str = Field(min_length=1)  # the dedup regime or abstraction rule that fired
    policy: Digest
    content: str
    key: str | None = None  # the structured claim it asserts, if any
    value: str | None = None
    valid_from: UTCDatetime
    valid_to: UTCDatetime | None = None
    recorded_at: UTCDatetime
    parents: tuple[str, ...] = Field(min_length=1)
    versions: Annotated[tuple[Digest, ...], AfterValidator(_canonical_set)]
    derived_from: Annotated[tuple[Digest, ...], AfterValidator(_canonical_set)]
    excluded: tuple[tuple[str, str], ...] = ()  # (memory considered, reason), sorted
    conflicts: Annotated[tuple[Digest, ...], AfterValidator(_canonical_set)] = ()
    support: int = Field(ge=1)
    disputed: int = Field(ge=0)
    model: Digest | None = None
    lost: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if self.status is not _OPERATION_STATUS[self.operation]:
            raise ValueError(f"{self.operation} produces {_OPERATION_STATUS[self.operation]}")
        if self.level is not _OPERATION_LEVEL[self.operation]:
            raise ValueError(f"{self.operation} produces level {_OPERATION_LEVEL[self.operation]}")
        if (self.key is None) != (self.value is None):
            raise ValueError("a claim has both a key and a value, or neither")
        if self.status is EpistemicStatus.INFERRED and self.key is not None:
            raise ValueError("an inferred pattern never asserts a fact claim")
        if not self.versions or not self.derived_from:
            raise ValueError("a derived memory cites the versions and experiences it covers")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        if list(self.excluded) != sorted(set(self.excluded)):
            raise ValueError("exclusions must be unique and sorted")
        return self

    @property
    def version(self) -> int:
        """Derived memories are immutable; a re-consolidation yields a new memory_id."""
        return 1


# --- experiments: seeded randomness, datasets, interventions, runs -------------------------


def unit_interval(seed: int, *labels: Scalar) -> float:
    """A reproducible pseudo-random number in [0, 1), derived from ``(seed, *labels)``.

    Stateless and platform-independent (SHA-256 of canonical JSON): the value depends
    only on its arguments, never on call order or a global generator.
    """
    h = hashlib.sha256(canonical_json([seed, *labels]).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


class Step(Record):
    """One experience and the record time at which a system under study ingests it."""

    experience: Experience
    recorded_at: UTCDatetime


class ExpectationStatus(StrEnum):
    KNOWN = "known"  # exactly one value is supported by the evidence
    UNKNOWN = "unknown"  # nothing is supported: never learned, forgotten, or not yet known
    CONTESTED = "contested"  # the evidence supports several values and does not decide


class Expectation(Record):
    """Ground truth for a probe: what an ideal system could hold, given the evidence.

    It is *epistemic*: judged from the experiences recorded by the probe's ``known_at``,
    not from facts the system could not yet have observed.
    """

    status: ExpectationStatus
    values: Annotated[tuple[str, ...], AfterValidator(_canonical_set)] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        needed = {
            ExpectationStatus.KNOWN: len(self.values) == 1,
            ExpectationStatus.UNKNOWN: not self.values,
            ExpectationStatus.CONTESTED: len(self.values) >= 2,
        }
        if not needed[self.status]:
            raise ValueError(
                f"{self.status.value}: needs 1 value (known), 0 (unknown) or >=2 (contested)"
            )
        return self


class Probe(ExtensibleRecord):
    """A question asked of the memory system after ingestion, with its expected answer.

    ``key`` (schema evolution, absent by default) names the claim key the probe asks
    about, for retrieval policies that use structured targets. Nothing infers it from text.
    """

    _evolved = frozenset({"key"})

    id: str = Field(min_length=1)
    text: str
    valid_at: UTCDatetime
    known_at: UTCDatetime
    limit: int = Field(default=1, ge=1)
    expected: Expectation
    key: str | None = Field(default=None, pattern=r"^[a-z0-9_.-]+$")

    @property
    def query(self) -> Query:
        return Query(
            text=self.text, valid_at=self.valid_at, known_at=self.known_at, limit=self.limit
        )


class Dataset(Record):
    """A named, versioned experience stream (in ingestion order) plus probes."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    steps: tuple[Step, ...]
    probes: tuple[Probe, ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        times = [s.recorded_at for s in self.steps]
        if times != sorted(times):
            raise ValueError("steps must be in non-decreasing recorded_at (ingestion) order")
        ids = [p.id for p in self.probes]
        if len(set(ids)) != len(ids):
            raise ValueError("probe ids must be unique")
        return self


class InterventionSpec(Record):
    """A named intervention and every parameter (including seeds) that determines it."""

    name: str = Field(min_length=1)
    params: Inputs = ()


class InterventionRecord(Record):
    """What an intervention did: its spec, input and output datasets, and the step diff."""

    spec: InterventionSpec
    input: Digest
    output: Digest
    removed: tuple[Digest, ...]  # step digests, sorted; multiset difference input - output
    added: tuple[Digest, ...]  # step digests, sorted; multiset difference output - input


# --- representations: embedders and vector indexes --------------------------------------


class Preprocessing(Record):
    """Deterministic text preprocessing applied before embedding. Part of the identity."""

    unicode: Literal["NFKC", "none"] = "NFKC"
    casefold: bool = True
    collapse_whitespace: bool = True
    max_chars: int | None = Field(default=None, ge=1)  # truncate after normalisation


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ModelFile(Record):
    """One file a model's output depends on, pinned by content."""

    path: str = Field(min_length=1)  # relative to the model directory
    sha256: Sha256
    size: int = Field(ge=0)


class ModelIdentity(Record):
    """Where a model comes from and exactly which bytes it is."""

    provider: str = Field(min_length=1)  # e.g. "huggingface"
    id: str = Field(min_length=1)  # e.g. "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = Field(min_length=1)  # an immutable revision (a commit hash)
    files: tuple[ModelFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        paths = [f.path for f in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("model files must be unique and sorted by path")
        return self


class Tolerance(Record):
    """When two vectors from the same spec count as numerically equivalent.

    Equivalent iff every component differs by at most ``max_abs`` and the cosine
    between them is at least ``min_cosine``. Deterministic embedders declare no
    tolerance: their outputs must be byte-identical.
    """

    max_abs: float = Field(gt=0, allow_inf_nan=False)
    min_cosine: float = Field(gt=0, le=1, allow_inf_nan=False)


class EmbedderSpec(ExtensibleRecord):
    """Everything that determines an embedder's output. Its digest is the model identity.

    Execution details that do not change the declared computation (hardware, thread
    counts, library patch versions) are environment, not identity. A spec that names a
    model must declare everything that shapes its output: pooling, truncation, maximum
    tokens, runtime, batching, and the tolerance within which its outputs reproduce.
    """

    _evolved = frozenset(
        {"model", "pooling", "max_tokens", "truncation", "runtime", "batch_size", "tolerance"}
    )

    name: str = Field(min_length=1)  # the adapter, e.g. "hashed-char-ngrams", "onnx-sentence"
    version: str = Field(min_length=1)  # the adapter's version
    dimensions: int = Field(ge=1)
    preprocessing: Preprocessing = Preprocessing()
    normalized: bool  # whether output vectors are L2-normalised (or zero)
    params: Inputs = ()
    model: ModelIdentity | None = None
    pooling: Literal["mean", "cls"] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    truncation: Literal["right"] | None = None
    runtime: str | None = None  # the inference runtime, e.g. "onnxruntime-cpu"
    batch_size: int | None = Field(default=None, ge=1)
    tolerance: Tolerance | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        neural = (
            self.pooling,
            self.max_tokens,
            self.truncation,
            self.runtime,
            self.batch_size,
            self.tolerance,
        )
        if self.model is not None and None in neural:
            raise ValueError(
                "a model spec must declare pooling, max_tokens, truncation, runtime, "
                "batch_size and tolerance"
            )
        if self.model is None and any(x is not None for x in neural):
            raise ValueError("model-inference settings require a model identity")
        return self


class IndexSpec(Record):
    """How vectors are indexed. ``exact`` is the reference; approximate kinds are
    validated against it."""

    kind: Literal["exact", "hnsw"] = "exact"
    metric: Literal["cosine"] = "cosine"
    backend: str | None = None  # e.g. "faiss" for hnsw; None for the built-in exact index
    params: Inputs = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if (self.kind == "exact") != (self.backend is None):
            raise ValueError("the exact index is built in; approximate kinds name a backend")
        if self.kind == "exact" and self.params:
            raise ValueError("the exact index takes no parameters")
        return self


class RepresentationSpec(Record):
    """A retrieval representation: which embedder produces vectors, how they are indexed."""

    embedder: EmbedderSpec
    index: IndexSpec = IndexSpec()


class RunManifest(ExtensibleRecord):
    """Everything that determines a run. Its digest is the run ID.

    Components are named with explicit versions (``statement-v1``, ``bm25``...); those
    names, their recorded parameters, and the dataset digest are the reproducibility
    contract. The execution environment is recorded separately and is not part of it.
    """

    _evolved = frozenset(
        {"representation", "retriever", "retrieval_policy", "consolidation_policy"}
    )

    name: str = Field(min_length=1)
    dataset: Digest
    interventions: tuple[InterventionSpec, ...] = ()
    policy: str = Field(min_length=1)
    # The Phase 2 single-stage retriever. Schema v3 makes it optional (absent iff a hybrid
    # retrieval policy is declared); every v1/v2 manifest carries it, so digests are kept.
    retriever: RetrieverSpec | None = None
    responder: str = Field(min_length=1)
    # Schema v2: the vector representation a semantic stage uses. Absent (v1) means no
    # vectors are involved.
    representation: RepresentationSpec | None = None
    # Schema v3: a hybrid RetrievalPolicy and a ConsolidationPolicy, each a stored
    # artifact referenced by digest (like the dataset). Absent means not used.
    retrieval_policy: Digest | None = None
    consolidation_policy: Digest | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if (self.retriever is None) == (self.retrieval_policy is None):
            raise ValueError("a manifest declares exactly one of retriever / retrieval_policy")
        if self.consolidation_policy is not None and self.retrieval_policy is None:
            raise ValueError("consolidated memory is retrieved only by a hybrid retrieval policy")
        return self


class StepOutcome(Record):
    """How the system under study handled one step: a recorded decision, or a rejection."""

    step: Digest
    decision: Digest | None = None
    reason: FormationReason | None = None
    rejected: str | None = None  # the rejection message, if the log refused the step

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        decided = self.decision is not None and self.reason is not None
        undecided = self.decision is None and self.reason is None
        if not ((decided and self.rejected is None) or (undecided and self.rejected)):
            raise ValueError("a step outcome is either a decision with a reason or a rejection")
        return self


class ProbeOutcome(Record):
    """A probe's recorded answer next to its expectation. Scoring is not done here."""

    probe: str
    trace: Digest
    response: Digest
    output: str | None
    cited: tuple[Digest, ...]
    expected: Expectation


class RunOutcomes(Record):
    steps: tuple[StepOutcome, ...]
    probes: tuple[ProbeOutcome, ...]


class RunRecord(ExtensibleRecord):
    """The deterministic result of executing a manifest: digests of every artifact.

    ``hierarchies`` (schema v3, absent by default) lists the consolidation hierarchies a
    run built, in checkpoint order.
    """

    _evolved = frozenset({"hierarchies"})

    manifest: Digest
    dataset: Digest  # the effective dataset, after interventions
    interventions: tuple[Digest, ...]  # InterventionRecord artifacts, in application order
    log: Digest  # canonical export of the resulting memory log
    outcomes: Digest  # RunOutcomes artifact
    hierarchies: tuple[Digest, ...] = ()
