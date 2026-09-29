"""The evidence ledger: belief revision as a replayable sequence of events (Phases 11-12).

:func:`run_ledger` feeds evidence to a :class:`~memoria.beliefs.BeliefPolicy` in arrival
order (and applies withdrawals, e.g. from a forgetting record). Each event records the
evidence, the key's state before and after (as fingerprints), every belief transition with
its score change, the reason and the supporting and contradicting evidence. The ledger
contains the evidence itself and the withdrawals, so :func:`replay` re-derives every event
and :func:`reconstruct` any earlier state; :func:`violations` audits the ledger against
the revision constraints.

Logical time is the evidence's arrival time (``recorded_at``): a state at time k is built
from evidence that arrived at or before k, and nothing else (future evidence cannot alter
a historical reconstruction; :func:`historical_mismatches` checks exactly that).
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.beliefs import (
    ANSWERING,
    Belief,
    BeliefPolicy,
    BeliefState,
    Context,
    EvidenceItem,
    KeyBeliefs,
    arrival_hash,
    check_arithmetic,
    derive,
    epistemic_rank,
    evidence_set_hash,
)
from memoria.core import Digest, Record, UTCDatetime, content_hash
from memoria.sources import SourceModel, TrustLedger, TrustRecord

EMPTY = "empty"
Reason = Literal[
    "change_accepted", "change_unconfirmed", "conflict_appeared", "conflict_resolved",
    "correction_applied", "evidence_withdrawn", "first_evidence", "no_change", "reinforced",
    "restated", "state_changed",
]  # fmt: skip


class Transition(Record):
    belief: str
    value: str | None
    before: BeliefState | None  # None: the belief did not exist
    after: BeliefState | None  # None: the belief was replaced by a revised one (its id changed)
    score_before: float | None
    score_after: float | None


class RevisionEvent(Record):
    """One evidence event and the revision it caused."""

    seq: int = Field(ge=0)
    logical_time: UTCDatetime
    kind: Literal["arrive", "withdraw"]
    evidence: Digest
    key: str
    reason: Reason
    late: bool  # arrived after evidence that occurred later
    previous: str  # fingerprint of the key's beliefs before ("empty" if none)
    resulting: str
    transitions: tuple[Transition, ...]
    supporting: tuple[Digest, ...]
    contradicting: tuple[Digest, ...]
    trust_changed: bool


class RunIdentity(Record):
    """Everything that determines a belief-revision run (the experiment identity)."""

    world: str  # the dataset the evidence came from
    memory_state: str  # digest of the memory state (corpus) it was drawn from
    evidence: str
    arrival: str
    sources: Digest
    policy: Digest
    temporal: Digest
    confidence_model: str
    seed: int
    forgetting: str  # forgetting record digest, or "none"
    consolidation: str  # hierarchy digest, or "none"
    interference: str  # injected dataset digest, or "none"


class EvidenceLedger(Record):
    """The authoritative record of a belief-revision run."""

    identity: RunIdentity
    evidence: tuple[EvidenceItem, ...]  # every item ever, by id
    withdrawals: tuple[tuple[UTCDatetime, Digest], ...]
    events: tuple[RevisionEvent, ...]
    final: str  # fingerprint of all final key states and the trust counts
    trust: tuple[TrustRecord, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if [e.seq for e in self.events] != list(range(len(self.events))):
            raise ValueError("event sequence numbers must be consecutive from 0")
        times = [e.logical_time for e in self.events]
        if times != sorted(times):
            raise ValueError("logical time must not go backwards")
        ids = [i.id for i in self.evidence]
        if ids != sorted(set(ids)):
            raise ValueError("evidence must be unique and sorted by id")
        arrived = [e.evidence for e in self.events if e.kind == "arrive"]
        if sorted(arrived) != ids:
            raise ValueError("every evidence item arrives exactly once")
        return self


@dataclass
class RunMeta:
    world: str = "unspecified"
    memory_state: str = "none"
    seed: int = 0
    confidence_model: str = "score-v1"
    forgetting: str = "none"
    consolidation: str = "none"
    interference: str = "none"


@dataclass
class LedgerRun:
    """A finished run: the ledger and the in-memory state history it produced."""

    ledger: EvidenceLedger
    policy: BeliefPolicy
    sources: SourceModel
    history: dict[str, list[KeyBeliefs]] = field(default_factory=dict)
    trust: TrustLedger | None = None
    _times: dict[str, list[datetime]] = field(default_factory=dict)

    def keys(self) -> list[str]:
        return sorted(self.history)

    def at(self, key: str, known_at: datetime) -> KeyBeliefs | None:
        """The key's beliefs as of arrival time ``known_at`` (nothing later is visible)."""
        times = self._times.get(key)
        if not times:
            return None
        i = bisect.bisect_right(times, known_at) - 1
        return self.history[key][i] if i >= 0 else None

    def final(self, key: str) -> KeyBeliefs | None:
        h = self.history.get(key)
        return h[-1] if h else None


def _stance(state: BeliefState) -> int:
    if state in (BeliefState.SUPPORTED, BeliefState.SUPERSEDED):
        return 1
    return -1 if state in (BeliefState.REJECTED, BeliefState.CORRECTED) else 0


def _reason(kind: str, transitions: list[Transition], new: KeyBeliefs, prev: KeyBeliefs | None,
            item: EvidenceItem) -> Reason:  # fmt: skip
    if kind == "withdraw":
        return "evidence_withdrawn"
    states = [(t.before, t.after) for t in transitions if t.after is not None]
    if any(a is BeliefState.CORRECTED and b is not BeliefState.CORRECTED for b, a in states):
        return "correction_applied"
    if any(a is BeliefState.CONTESTED and b is not BeliefState.CONTESTED for b, a in states):
        unconfirmed = any(x.reason == "change_unconfirmed" for x in new.beliefs)
        return "change_unconfirmed" if unconfirmed else "conflict_appeared"
    if any(a is BeliefState.UNRESOLVED and b is not BeliefState.UNRESOLVED for b, a in states):
        return "conflict_appeared"
    if any(b in (BeliefState.CONTESTED, BeliefState.UNRESOLVED) and a is BeliefState.SUPPORTED
           for b, a in states):  # fmt: skip
        return "conflict_resolved"
    if any(a is BeliefState.SUPERSEDED and b is not BeliefState.SUPERSEDED for b, a in states):
        return "change_accepted"
    if prev is None or not prev.beliefs:
        return "first_evidence"
    if any(b is None for b, _ in states):
        return "state_changed"
    if any((t.score_after or 0.0) > (t.score_before or 0.0) + 1e-12 for t in transitions):
        return "reinforced"
    return "restated" if not transitions else "state_changed"


def run_ledger(
    evidence: Sequence[EvidenceItem],
    policy: BeliefPolicy,
    sources: SourceModel,
    *,
    identity: Mapping[str, float] | None = None,
    withdrawals: Sequence[tuple[datetime, str]] = (),
    meta: RunMeta | None = None,
) -> LedgerRun:
    """Process evidence in arrival order under ``policy``. Deterministic."""
    meta = meta or RunMeta()
    by_id = {i.id: i for i in evidence}
    if len(by_id) != len(evidence):
        raise ValueError("evidence ids must be unique")
    for _, e in withdrawals:
        if e not in by_id:
            raise ValueError("a withdrawal names evidence that never arrives")
    events: list[tuple[datetime, int, str, str]] = [
        (i.recorded_at, 0, i.id, "arrive") for i in evidence
    ] + [(t, 1, e, "withdraw") for t, e in withdrawals]
    events.sort()
    trust = TrustLedger(sources)
    ident = dict(identity or {})
    items_by_key: dict[str, list[EvidenceItem]] = {}
    withdrawn: set[str] = set()
    current: dict[str, KeyBeliefs] = {}
    run = LedgerRun(
        ledger=None,  # type: ignore[arg-type]
        policy=policy, sources=sources, trust=trust,
    )  # fmt: skip
    out: list[RevisionEvent] = []
    for seq, (t, _, eid, kind) in enumerate(events):
        item = by_id[eid]
        key = item.key
        prev = current.get(key)
        late = any(
            i.occurred_at > item.occurred_at and i.id not in withdrawn
            for i in items_by_key.get(key, [])
        )
        if kind == "arrive":
            items_by_key.setdefault(key, []).append(item)
            trust.register(item.id, item.source)
        else:
            if item.id in withdrawn:
                raise ValueError("evidence is withdrawn twice")
            withdrawn.add(item.id)
            trust.forget(item.id)
        before_trust = trust.state()
        ctx = Context(policy, sources, trust, ident, t, seq, withdrawing=kind == "withdraw")
        kb = derive(key, items_by_key[key], withdrawn, prev, ctx)
        for b in kb.beliefs:
            for e in b.supporting:
                trust.credit(e, _stance(b.state))
        old = prev.by_id() if prev else {}
        transitions = []
        for b in (*kb.beliefs, *kb.tombstones):
            o = old.get(b.id)
            if o is None or o.state is not b.state or abs(o.score - b.score) > 1e-12:
                transitions.append(Transition(
                    belief=b.id, value=b.value, before=o.state if o else None, after=b.state,
                    score_before=o.score if o else None, score_after=b.score,
                ))  # fmt: skip
        now = kb.by_id()
        for o in prev.beliefs if prev else ():
            if o.id not in now:
                transitions.append(Transition(
                    belief=o.id, value=o.value, before=o.state, after=None,
                    score_before=o.score, score_after=None,
                ))  # fmt: skip
        home = next((b for b in kb.beliefs if item.id in b.supporting), None)
        if home is None and kind == "withdraw" and prev is not None:
            home = next((b for b in prev.beliefs if item.id in b.supporting), None)
        out.append(RevisionEvent(
            seq=seq, logical_time=t, kind=kind, evidence=eid, key=key,  # type: ignore[arg-type]
            reason=_reason(kind, transitions, kb, prev, item), late=late,
            previous=prev.fingerprint() if prev else EMPTY, resulting=kb.fingerprint(),
            transitions=tuple(transitions),
            supporting=home.supporting if home else (),
            contradicting=home.contradicting if home else (),
            trust_changed=trust.state() != before_trust,
        ))  # fmt: skip
        current[key] = kb
        run.history.setdefault(key, []).append(kb)
        run._times.setdefault(key, []).append(t)
    final = content_hash([[k, current[k].fingerprint()] for k in sorted(current)]
                         + [list(map(list, trust.state()))])  # fmt: skip
    ident_rec = RunIdentity(
        world=meta.world, memory_state=meta.memory_state, evidence=evidence_set_hash(evidence),
        arrival=arrival_hash(evidence), sources=sources.digest, policy=policy.digest,
        temporal=policy.temporal.digest, confidence_model=meta.confidence_model, seed=meta.seed,
        forgetting=meta.forgetting, consolidation=meta.consolidation,
        interference=meta.interference,
    )  # fmt: skip
    run.ledger = EvidenceLedger(
        identity=ident_rec,
        evidence=tuple(sorted(evidence, key=lambda i: i.id)),
        withdrawals=tuple(sorted(withdrawals, key=lambda w: (w[0], w[1]))),
        events=tuple(out), final=final, trust=trust.snapshot(),
    )  # fmt: skip
    return run


def replay(
    ledger: EvidenceLedger,
    policy: BeliefPolicy,
    sources: SourceModel,
    *,
    identity: Mapping[str, float] | None = None,
) -> tuple[bool, int | None]:
    """Re-derive the ledger from its own evidence and withdrawals: (identical, first event
    that differs). Needs the policy and source model the identity names."""
    if ledger.identity.policy != policy.digest or ledger.identity.sources != sources.digest:
        raise ValueError("replay needs the policy and source model the ledger names")
    again = run_ledger(ledger.evidence, policy, sources, identity=identity,
                       withdrawals=ledger.withdrawals)  # fmt: skip
    for a, b in zip(ledger.events, again.ledger.events, strict=True):
        if a != b:
            return False, a.seq
    return again.ledger.final == ledger.final, None


def reconstruct(
    ledger: EvidenceLedger,
    policy: BeliefPolicy,
    sources: SourceModel,
    known_at: datetime,
    *,
    identity: Mapping[str, float] | None = None,
) -> LedgerRun:
    """The run as it stood at ``known_at``: only evidence that had arrived, only
    withdrawals that had happened."""
    return run_ledger(
        [i for i in ledger.evidence if i.recorded_at <= known_at], policy, sources,
        identity=identity, withdrawals=[w for w in ledger.withdrawals if w[0] <= known_at],
    )  # fmt: skip


def historical_mismatches(
    run: LedgerRun, times: Iterable[datetime], *, identity: Mapping[str, float] | None = None
) -> int:
    """How many (time, key) pairs differ between the live run and a reconstruction from
    the ledger at that time: any number above zero is future evidence leaking into the
    past (or a non-deterministic policy)."""
    bad = 0
    for t in times:
        rebuilt = reconstruct(run.ledger, run.policy, run.sources, t, identity=identity)
        for key in run.history:
            live, again = run.at(key, t), rebuilt.final(key)
            if (live.fingerprint() if live else None) != (again.fingerprint() if again else None):
                bad += 1
    return bad


class Violation(Record):
    code: str
    seq: int
    subject: str
    detail: str


def violations(run: LedgerRun, arithmetic: bool = True) -> list[Violation]:
    """Audit a run against the revision constraints; an empty list means none is violated.

    V-TIME time goes backwards; V-FUTURE evidence used before it arrived; V-CHAIN a state
    does not follow from the previous one; V-EVIDENCE beliefs do not cite exactly the active
    evidence (evidence dropped or invented); V-WITHDRAW withdrawn evidence still supports;
    V-SUPERSEDE a superseded belief lost its interval or evidence; V-EPISTEMIC a belief is
    stronger than its evidence permits; V-CONFIDENCE confidence rose although nothing it
    depends on changed; V-ARITH weight or score does not follow from the contributions.
    """
    ledger = run.ledger
    out: list[Violation] = []

    def bad(code: str, seq: int, subject: str, detail: str) -> None:
        out.append(Violation(code=code, seq=seq, subject=subject, detail=detail))

    items = {i.id: i for i in ledger.evidence}
    arrived = {e.evidence: e.seq for e in ledger.events if e.kind == "arrive"}
    withdrawn = {e.evidence: e.seq for e in ledger.events if e.kind == "withdraw"}
    by_key: dict[str, list[EvidenceItem]] = {}
    for it in ledger.evidence:
        by_key.setdefault(it.key, []).append(it)
    last_time: datetime | None = None
    last_state: dict[str, str] = {}
    for e in ledger.events:
        if last_time is not None and e.logical_time < last_time:
            bad("V-TIME", e.seq, e.key, "logical time went backwards")
        last_time = e.logical_time
        item = items[e.evidence]
        if e.kind == "arrive" and e.logical_time != item.recorded_at:
            bad("V-FUTURE", e.seq, e.evidence, "evidence used at a time other than its arrival")
        if e.kind == "withdraw" and e.logical_time < item.recorded_at:
            bad("V-FUTURE", e.seq, e.evidence, "evidence withdrawn before it arrived")
        if e.previous != last_state.get(e.key, EMPTY):
            bad("V-CHAIN", e.seq, e.key, "previous state is not the prior event's result")
        last_state[e.key] = e.resulting
    for key, states in run.history.items():
        prev: KeyBeliefs | None = None
        for kb in states:
            event = ledger.events[kb.seq]
            live = {
                i.id for i in by_key[key]
                if arrived[i.id] <= kb.seq and withdrawn.get(i.id, 10**9) > kb.seq
            }  # fmt: skip
            cited: set[str] = set(kb.retractions)
            for b in kb.beliefs:
                cited.update(b.supporting)
                for x in (*b.supporting, *b.contradicting):
                    if arrived.get(x, 10**9) > kb.seq:
                        bad("V-FUTURE", kb.seq, b.id, "cites evidence that has not arrived")
                if b.state is BeliefState.SUPERSEDED and (b.valid_to is None or not b.supporting):
                    bad("V-SUPERSEDE", kb.seq, b.id, "lost its interval or its evidence")
                if b.supporting and epistemic_rank(b) < min(
                    items[x].epistemic_rank for x in b.supporting
                ):
                    bad("V-EPISTEMIC", kb.seq, b.id, "stronger than its evidence permits")
                if arithmetic:
                    try:
                        check_arithmetic(b)
                    except ValueError as exc:
                        bad("V-ARITH", kb.seq, b.id, str(exc))
            if cited != live:
                bad("V-EVIDENCE", kb.seq, key, "beliefs do not cite exactly the active evidence")
            if event.kind == "withdraw":
                for b in kb.beliefs:
                    if event.evidence in b.supporting:
                        bad("V-WITHDRAW", kb.seq, b.id, "withdrawn evidence still supports")
                for b in prev.beliefs if prev else ():
                    after = kb.by_id().get(b.id)
                    sole = b.supporting == (event.evidence,)
                    if sole and after is not None and after.state is BeliefState.SUPPORTED:
                        bad(
                            "V-WITHDRAW",
                            kb.seq,
                            b.id,
                            "sole support withdrawn, still supported",
                        )
            if prev is not None:
                before = prev.by_id()
                for b in kb.beliefs:
                    o = before.get(b.id)
                    if o is None or b.score <= o.score + 1e-9:
                        continue
                    unchanged = (
                        b.supporting == o.supporting
                        and b.contradicting == o.contradicting
                        and b.weight == o.weight
                        and b.mass == o.mass
                    )
                    if unchanged and not event.trust_changed:
                        bad(
                            "V-CONFIDENCE",
                            kb.seq,
                            b.id,
                            "rose although its evidence did not change",
                        )
            prev = kb
    return out


# --- answers -------------------------------------------------------------------------------------

Mode = Literal["abstain", "answer", "answer_with_uncertainty", "competing", "require_evidence"]


class SelectivePolicy(Record):
    """When the system answers, hedges, presents competing beliefs, abstains or asks for
    more evidence. Thresholds apply to the confidence *score* (or its calibrated value)."""

    name: str = Field(min_length=1)
    answer_at: float = Field(gt=0, le=1)
    uncertain_at: float = Field(ge=0, le=1)
    min_roots: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if self.uncertain_at > self.answer_at:
            raise ValueError("uncertain_at cannot exceed answer_at")
        return self


class Decision(Record):
    key: str
    valid_at: UTCDatetime
    known_at: UTCDatetime
    mode: Mode
    values: tuple[str, ...]
    score: float | None
    beliefs: tuple[str, ...]
    state: BeliefState | None
    reason: str


def covering(kb: KeyBeliefs | None, valid_at: datetime) -> list[Belief]:
    """The beliefs whose (believed) interval contains ``valid_at`` and may answer."""
    if kb is None:
        return []
    return [
        b
        for b in kb.beliefs
        if b.state in ANSWERING
        and b.valid_from <= valid_at
        and (b.valid_to is None or valid_at < b.valid_to)
    ]


def decide(
    key: str,
    kb: KeyBeliefs | None,
    valid_at: datetime,
    known_at: datetime,
    sel: SelectivePolicy,
    calibrate: Mapping[str, float] | None = None,
) -> Decision:
    """Turn the key's beliefs into a decision. The decision records the belief's raw score;
    ``calibrate`` (belief id -> calibrated value) only changes which side of the thresholds
    the belief falls on."""
    cover = covering(kb, valid_at)

    def make(mode: Mode, why: str, cs: Sequence[Belief], score: float | None = None) -> Decision:
        return Decision(
            key=key, valid_at=valid_at, known_at=known_at, mode=mode,
            values=tuple(sorted({b.value for b in cs})), score=score,
            beliefs=tuple(sorted(b.id for b in cs)), state=cs[0].state if cs else None, reason=why,
        )  # fmt: skip

    if not cover:
        return make("abstain", "no_belief", [])
    open_ = [b for b in cover if b.state in (BeliefState.CONTESTED, BeliefState.UNRESOLVED)]
    if len(cover) > 1 or open_:
        top = max(b.score for b in cover)
        if len(cover) == 1:
            return make("require_evidence", "unresolved_single", cover, top)
        return make("competing", "competing_beliefs", cover, top)
    b = cover[0]
    used = calibrate.get(b.id, b.score) if calibrate else b.score  # what the thresholds read
    if b.independent_count < sel.min_roots:
        return make("require_evidence", "insufficient_independent_evidence", cover, b.score)
    if used >= sel.answer_at:
        return make("answer", "confident", cover, b.score)
    if used >= sel.uncertain_at:
        return make("answer_with_uncertainty", "uncertain", cover, b.score)
    return make("abstain", "low_confidence", cover, b.score)
