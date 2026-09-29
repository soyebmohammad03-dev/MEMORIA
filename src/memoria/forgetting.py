"""The forgetting laboratory (Phase 9): forgetting as a recorded intervention on availability.

Forgetting never deletes. The log, its experiences and every hierarchy stay exactly as
recorded (I1, I7, I48); a :class:`ForgettingRecord` says, for every memory of a corpus at
one time, which *availability* a :class:`ForgettingPolicy` gives it and why:

============  ===============  ===========  =================================================
state         in history       retrievable  meaning
============  ===============  ===========  =================================================
active        yes              yes          unaffected
suppressed    yes              yes          soft: penalised by the ``suppression`` signal
archived      yes              no           kept for audit and aggregates, not retrieved
excluded      yes              no           hard retrieval exclusion only
forgotten     yes              no           forgotten by policy (not retrieved, not traversed)
============  ===============  ===========  =================================================

:func:`apply` returns the corpus with availability set; its identity names the record, so
every retrieval trace over it cites the intervention. A derived memory whose evidence
became unavailable is made unavailable too (``derived="cascade"``) unless the policy
explicitly retains it (``derived="retain"``), in which case the record lists it: forgotten
evidence still able to influence retrieval through it is reported, never silent.

Policies (``rule``) expose every signal they use in each decision; none collapses several
signals into a hidden score. ``hybrid`` votes: every component's vote is recorded.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.comparison import normalize
from memoria.consolidation import ImportanceSpec, Period, _importance, timeline
from memoria.core import (
    DerivedMemory,
    Digest,
    Inputs,
    Level,
    Record,
    Scalar,
    UTCDatetime,
    quantize,
)
from memoria.entities import normalize_name, surface_names
from memoria.hybrid import (
    CONFLICTED,
    RETRIEVABLE,
    Availability,
    ConflictStatus,
    Corpus,
    Entry,
    HybridTrace,
    conflict,
    entity,
)
from memoria.statistics import Proportion
from memoria.taxonomy import Claim

Action = Literal["suppress", "archive", "exclude", "forget"]
STATE: dict[str, Availability] = {
    "suppress": "suppressed",
    "archive": "archived",
    "exclude": "excluded",
    "forget": "forgotten",
}
Rule = Literal[
    "access",
    "age",
    "contradiction",
    "fifo",
    "hybrid",
    "importance",
    "provenance",
    "recency",
    "selective",
    "validity",
]
# Parameters each rule takes (all required).
PARAMS: dict[str, tuple[str, ...]] = {
    "age": ("max_age_days",),
    "fifo": ("capacity",),
    "recency": ("half_life_days", "threshold"),
    "importance": (),
    "access": ("grace_days", "min_access"),
    "validity": ("grace_days",),
    "contradiction": ("forget_contested",),
    "provenance": ("keep",),
    "hybrid": ("grace_days", "max_age_days", "min_access", "min_votes"),
    "selective": (),
}
# Hybrid votes use its shared parameters; its contradiction component never forgets
# contested claims (forget_contested = 0), so a vote cannot hide an open disagreement.
HYBRID_COMPONENTS = ("access", "age", "contradiction", "importance", "validity")


class Selector(Record):
    """What selective forgetting targets. Every set field must match (conjunction)."""

    entity: str | None = None  # a structured id; free text matches by normalised name only
    start: UTCDatetime | None = None  # earliest evidence occurrence in [start, end)
    end: UTCDatetime | None = None
    source: str | None = None  # source class, e.g. "forum"
    kind: Literal["fact", "note"] | None = None  # memory class
    contradictory: bool = False  # claims another known claim disputes
    levels: tuple[Level, ...] = (Level.L1,)  # (L2, L3, L4): derived memories only

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if list(self.levels) != sorted(set(self.levels)) or Level.L0 in self.levels:
            raise ValueError("levels must be unique, sorted, and above L0")
        if self.start and self.end and self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class ForgettingPolicy(Record):
    """Everything that determines a forgetting intervention. Its digest is its identity."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    rule: Rule
    action: Action
    params: Inputs = ()
    importance: ImportanceSpec | None = None  # importance and hybrid rules
    selector: Selector | None = None  # selective rule
    derived: Literal["cascade", "retain"] = "cascade"
    preserve_aggregates: bool = False  # keep per-key counts of what was made unavailable

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        names = sorted(n for n, _ in self.params)
        if names != sorted(PARAMS[self.rule]):
            raise ValueError(f"{self.rule} takes parameters {PARAMS[self.rule]}, got {names}")
        for n, v in self.params:
            if not isinstance(v, int | float) or not math.isfinite(v) or v < 0:
                raise ValueError(f"parameter {n} must be a finite non-negative number")
        needs_importance = self.rule in ("importance", "hybrid")
        if needs_importance != (self.importance is not None):
            raise ValueError("an importance model is required for, and only for, importance")
        if (self.rule == "selective") != (self.selector is not None):
            raise ValueError("a selector is required for, and only for, selective forgetting")
        if self.action == "suppress" and self.rule != "recency":
            raise ValueError("soft suppression is graded by recency; other rules act hard")
        return self

    def param(self, name: str, default: float | None = None) -> float:
        """A declared parameter; ``default`` only for a hybrid component's fixed setting."""
        params = dict(self.params)
        if name not in params and default is not None and self.rule == "hybrid":
            return default
        return float(params[name])


class ForgettingDecision(Record):
    """One memory's availability under the policy, with the signals that decided it."""

    entry: Digest
    memory_id: str
    level: Level
    state: Availability
    suppression: float = Field(ge=0, le=1)
    reason: str
    signals: Inputs = ()


class ForgettingRecord(Record):
    """A forgetting intervention: the policy, the corpus it applied to, the accesses it
    read, one decision per memory, preserved aggregates and semantic diagnostics."""

    policy: Digest
    at: UTCDatetime
    corpus: Digest  # the corpus before the intervention
    accesses: tuple[Digest, ...]  # retrieval traces read as the access history, sorted
    decisions: tuple[ForgettingDecision, ...]  # by (level, memory_id, entry)
    aggregates: tuple[tuple[str, int, int], ...] = ()  # (key, reports hidden, values), sorted
    dangling: tuple[str, ...] = ()  # retained derived memories with no available evidence
    unsupported_abstractions: tuple[str, ...] = ()  # retained L3/L4 with unavailable parents
    evidence_retained: tuple[str, ...] = ()  # retained derived memories citing hidden evidence
    stale: tuple[str, ...] = ()  # available derived memories whose evidence changed (I56)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        order = [(d.level, d.memory_id, d.entry) for d in self.decisions]
        if order != sorted(set(order)):
            raise ValueError("decisions must be unique and in (level, memory_id, entry) order")
        for d in self.decisions:
            if (d.state == "suppressed") != (d.suppression > 0):
                raise ValueError(f"{d.memory_id}: suppression strength iff suppressed")
        return self

    def states(self) -> dict[str, tuple[Availability, float]]:
        return {d.entry: (d.state, d.suppression) for d in self.decisions if d.state != "active"}

    def unavailable(self) -> set[str]:
        return {d.entry for d in self.decisions if d.state not in RETRIEVABLE}


def apply(corpus: Corpus, record: ForgettingRecord) -> Corpus:
    """The corpus with the record's availability applied (nothing removed)."""
    if record.corpus != corpus.digest:
        raise ValueError("the record was decided on another corpus")
    return corpus.with_availability(record.states(), record.digest)


# --- deciding ------------------------------------------------------------------------------------


def _earliest(e: Entry) -> datetime:
    times = [x.occurred_at for x in e.sources]
    return min(times) if times else e.version.recorded_at


def _age(e: Entry, at: datetime) -> float:
    return quantize((at - e.version.recorded_at) / timedelta(days=1))


def _entity_match(e: Entry, name: str) -> bool:
    if e.claim is not None:
        return entity(e.claim.key) == name
    return any(normalize_name(n) == name for x in e.sources for n in surface_names(x.content))


def _select(e: Entry, s: Selector, at: datetime, claims_: Sequence[Claim]) -> bool:
    if e.level not in s.levels:
        return False
    if s.entity is not None and not _entity_match(e, s.entity):
        return False
    t = _earliest(e)
    if (s.start and t < s.start) or (s.end and t >= s.end):
        return False
    if s.source is not None and not any(x.source.partition(":")[0] == s.source for x in e.sources):
        return False
    if s.kind is not None and e.kind != s.kind:
        return False
    if s.contradictory:
        status = conflict(e, claims_, at)
        if status is None or status not in CONFLICTED | {ConflictStatus.CONTRADICTED}:
            return False
    return True


Vote = tuple[bool | None, str, tuple[tuple[str, Scalar], ...]]  # (forget?, reason, signals)


def forget(
    corpus: Corpus,
    policy: ForgettingPolicy,
    at: datetime,
    accesses: Sequence[HybridTrace] = (),
) -> ForgettingRecord:
    """Decide every memory's availability at ``at``. Deterministic in (corpus, policy,
    accesses). ``accesses`` are retrieval traces read as the access history; each must have
    been recorded strictly before ``at`` (no future access influences forgetting)."""
    if corpus.known_at != at:
        raise ValueError("forget over the corpus known at the intervention time")
    if any(t.query.known_at >= at for t in accesses):
        raise ValueError("an access trace must precede the intervention (known_at < at)")
    if corpus.identity.forgetting is not None:
        raise ValueError("the corpus already carries a forgetting intervention")
    selected: dict[str, int] = {}
    for t in accesses:
        for v in t.selected:
            selected[v] = selected.get(v, 0) + 1
    l1 = [e for e in corpus.entries if e.level is Level.L1]
    held = [e for e in l1 if e.fate == "held" and not e.missing]
    importance = (
        _importance(held, policy.importance, at, corpus) if policy.importance is not None else {}
    )
    periods = {k: timeline(corpus.claims, k) for k in sorted({c.key for c in corpus.claims})}
    fifo_rank = {
        e.digest: i
        for i, e in enumerate(
            sorted(
                l1,
                key=lambda e: (e.version.recorded_at, e.version.memory_id, e.digest),
                reverse=True,
            )
        )
    }
    redundant = (
        _redundancy(l1, periods, int(policy.param("keep"))) if policy.rule == "provenance" else {}
    )

    def vote(rule: str, e: Entry) -> Vote:
        age = _age(e, at)
        if rule == "age":
            limit = policy.param("max_age_days")
            return age > limit, f"age:{'older' if age > limit else 'within'}", (("age_days", age),)
        if rule == "fifo":
            rank = fifo_rank[e.digest]
            over = rank >= policy.param("capacity")
            return over, f"fifo:{'over' if over else 'within'}_capacity", (("newest_rank", rank),)
        if rule == "access":
            n = selected.get(e.digest, 0)
            cold = n < policy.param("min_access") and age > policy.param("grace_days")
            return (
                cold,
                f"access:{'cold' if cold else 'kept'}",
                (("accesses", n), ("age_days", age)),
            )
        if rule == "importance":
            r = importance.get(e.digest)
            if r is None:
                return None, "importance:not_scored", ()
            return (
                not r.promoted,
                f"importance:{'below' if not r.promoted else 'at_or_above'}",
                (("importance", r.total),),
            )
        if rule == "validity":
            c = e.claim
            if c is None or c.value is None:
                return None, "validity:unknown_without_claim", ()
            period = next((p for p in periods[c.key] if c.experience in p.experiences), None)
            end = period.end if period is not None else None
            if end is None:
                return False, "validity:open", ()
            expired = end <= at - timedelta(days=policy.param("grace_days"))
            days = quantize((at - end) / timedelta(days=1))
            return (
                expired,
                f"validity:{'expired' if expired else 'recent'}",
                (("ended_days_ago", days),),
            )
        if rule == "contradiction":
            status = conflict(e, corpus.claims, at)
            if status is None:
                return None, "contradiction:unknown_without_claim", ()
            contested = status is ConflictStatus.CONTRADICTED
            out = status in (ConflictStatus.CORRECTED, ConflictStatus.FORGOTTEN) or (
                contested and policy.param("forget_contested", 0.0) > 0
            )
            return out, f"contradiction:{status.value}", (("status", status.value),)
        if rule == "provenance":
            if e.digest not in redundant:
                return None, "provenance:unknown_without_claim", ()
            copies, keep = redundant[e.digest]
            return (
                not keep,
                f"provenance:{'redundant' if not keep else 'kept_evidence'}",
                (("copies", copies),),
            )
        raise AssertionError(rule)

    decisions: dict[str, ForgettingDecision] = {}
    hidden: dict[str, tuple[int, set[str]]] = {}
    for e in l1:
        signals: list[tuple[str, Scalar]] = []
        state: Availability = "active"
        strength = 0.0
        if policy.rule == "selective":
            assert policy.selector is not None
            hit = _select(e, policy.selector, at, corpus.claims)
            reason = "selective:selected" if hit else "selective:not_selected"
            state = STATE[policy.action] if hit else "active"
        elif policy.rule == "recency":
            age = _age(e, at)
            r = quantize(0.5 ** (age / policy.param("half_life_days")))
            signals = [("age_days", age), ("recency", r)]
            weak = r < policy.param("threshold")
            reason = f"recency:{'decayed' if weak else 'fresh'}"
            if weak:
                state, strength = STATE[policy.action], quantize(max(1e-12, 1 - r))
                if policy.action != "suppress":
                    strength = 0.0
        elif policy.rule == "hybrid":
            votes = [(c, *vote(c, e)) for c in HYBRID_COMPONENTS]
            yes = sum(1 for _, verdict, _, _ in votes if verdict)
            for comp, verdict, _why, sig in votes:
                signals.append((f"vote.{comp}", "abstain" if verdict is None else int(verdict)))
                signals += [(f"{comp}.{n}", x) for n, x in sig]
            hit = yes >= policy.param("min_votes")
            reason = f"hybrid:{yes}_of_{len(votes)}_votes"
            state = STATE[policy.action] if hit else "active"
        else:
            verdict, reason, sig = vote(policy.rule, e)
            signals = list(sig)
            state = STATE[policy.action] if verdict else "active"
        if state not in RETRIEVABLE and e.claim is not None and e.claim.value is not None:
            count, values = hidden.get(e.claim.key, (0, set()))
            hidden[e.claim.key] = (count + 1, values | {" ".join(normalize(e.claim.value))})
        decisions[e.digest] = ForgettingDecision(
            entry=e.digest,
            memory_id=e.version.memory_id,
            level=e.level,
            state=state,
            suppression=strength,
            reason=reason,
            signals=tuple(signals),
        )

    # Derived memories, lowest level first: selective targets, then cascade or retention.
    dangling, unsupported, retained, stale = [], [], [], []
    by_id = {e.version.memory_id: e for e in corpus.entries if e.level is not Level.L1}
    for e in sorted(
        (e for e in corpus.entries if e.level is not Level.L1),
        key=lambda e: (e.level, e.version.memory_id),
    ):
        m = e.version
        assert isinstance(m, DerivedMemory)
        hidden_versions = [
            v for v in m.versions if v in decisions and decisions[v].state not in RETRIEVABLE
        ]
        hidden_parents = [
            p
            for p in m.parents
            if p in by_id
            and by_id[p].digest in decisions
            and decisions[by_id[p].digest].state not in RETRIEVABLE
        ]
        signals = [
            ("hidden_evidence", len(hidden_versions)),
            ("evidence", len(m.versions)),
            ("hidden_parents", len(hidden_parents)),
        ]
        state = "active"
        if (
            policy.rule == "selective"
            and policy.selector is not None
            and _select(e, policy.selector, at, corpus.claims)
        ):
            state, reason = STATE[policy.action], "selective:selected"
        elif (hidden_versions or hidden_parents) and policy.derived == "cascade":
            state = STATE[policy.action] if policy.action != "suppress" else "excluded"
            reason = "cascade:evidence_unavailable"
        elif hidden_versions or hidden_parents:
            reason = "retained_despite_unavailable_evidence"
            retained.append(m.memory_id)
            if len(hidden_versions) == len(m.versions):
                dangling.append(m.memory_id)
            if m.level in (Level.L3, Level.L4) and hidden_parents:
                unsupported.append(m.memory_id)
        else:
            reason = "retained"
        if state in RETRIEVABLE and e.stale:
            stale.append(m.memory_id)
        decisions[e.digest] = ForgettingDecision(
            entry=e.digest,
            memory_id=m.memory_id,
            level=e.level,
            state=state,
            suppression=0.0,
            reason=reason,
            signals=tuple(signals),
        )
    return ForgettingRecord(
        policy=policy.digest,
        at=at,
        corpus=corpus.digest,
        accesses=tuple(sorted(t.digest for t in accesses)),
        decisions=tuple(sorted(decisions.values(), key=lambda d: (d.level, d.memory_id, d.entry))),
        aggregates=tuple(sorted((k, n, len(vs)) for k, (n, vs) in hidden.items()))
        if policy.preserve_aggregates
        else (),
        dangling=tuple(sorted(dangling)),
        unsupported_abstractions=tuple(sorted(unsupported)),
        evidence_retained=tuple(sorted(retained)),
        stale=tuple(sorted(stale)),
    )


def _redundancy(
    l1: Sequence[Entry], periods: Mapping[str, list[Period]], keep: int
) -> dict[str, tuple[int, bool]]:
    """For claim-bearing memories: (copies of the same key, value and period, whether this
    one is among the ``keep`` earliest-recorded). Free text is not in the result."""
    groups: dict[tuple[str, str, int], list[Entry]] = {}
    for e in l1:
        c = e.claim
        if c is None or c.value is None:
            continue
        ps = periods[c.key]
        i = next((i for i, p in enumerate(ps) if c.experience in p.experiences), -1)
        groups.setdefault((c.key, " ".join(normalize(c.value)), i), []).append(e)
    out = {}
    for members in groups.values():
        members.sort(key=lambda e: (e.version.recorded_at, e.digest))
        for rank, e in enumerate(members):
            out[e.digest] = (len(members), rank < max(1, keep))
    return out


# --- measuring ------------------------------------------------------------------------------------


class ForgettingMetrics(Record):
    """What an intervention changed, memory by memory. ``target`` metrics need a target set
    (the memories the intervention was meant to make unavailable), from ground truth.

    *Collateral forgetting* is relevant information unintentionally made unavailable: here,
    at memory level, non-target memories made unavailable; at probe level (runs), probes
    whose expected answer was retrievable before the intervention and is not after it,
    although the intervention did not target them.
    """

    record: Digest
    retention: Proportion  # memories retrievable after / retrievable before
    unavailable: int
    suppressed: int
    precision: Proportion | None  # made unavailable and targeted / made unavailable
    recall: Proportion | None  # targeted and made unavailable / targeted
    accidental_retention: Proportion | None  # targeted but still retrievable / targeted
    collateral: Proportion | None  # non-targeted made unavailable / non-targeted
    derived_carrying_targets: int  # available derived memories whose evidence is targeted
    provenance_complete: Proportion  # available derived memories whose evidence is available
    stale_rate: Proportion  # available derived memories stale or citing hidden evidence
    contradiction_visibility: Proportion  # contested keys still showing >= 2 values


def measure_forgetting(
    before: Corpus,
    record: ForgettingRecord,
    targets: Iterable[str] | None = None,
    confidence: float = 0.95,
) -> ForgettingMetrics:
    hidden = record.unavailable()
    decided = {d.entry: d for d in record.decisions}
    was = [e for e in before.entries if e.availability in RETRIEVABLE]
    now = [e for e in was if e.digest not in hidden]
    target = set(targets) if targets is not None else None
    derived = [e for e in before.entries if e.level is not Level.L1 and e.digest not in hidden]

    def evidence_hidden(e: Entry) -> bool:
        return any(v in hidden for v in e.version.versions)  # type: ignore[union-attr]

    contested: dict[str, set[str]] = {}
    for e in was:
        c = e.claim
        if c is not None and c.value is not None and e.level is Level.L1:
            status = conflict(e, before.claims, record.at)
            if status is ConflictStatus.CONTRADICTED:
                contested.setdefault(c.key, set())
    visible = 0
    for key in contested:
        values = {
            " ".join(normalize(e.claim.value or ""))
            for e in now
            if e.claim is not None and e.claim.key == key and e.claim.value is not None
        }
        visible += len(values) >= 2
    p = Proportion.of
    precision = recall = accidental = collateral = None
    carrying = 0
    if target is not None:
        l1 = {e.digest for e in was if e.level is Level.L1}
        t = target & l1
        h = hidden & l1
        precision = p(len(h & t), len(h), confidence)
        recall = p(len(h & t), len(t), confidence)
        accidental = p(len(t - h), len(t), confidence)
        collateral = p(len(h - t), len(l1 - t), confidence)
        carrying = sum(
            1
            for e in derived
            if isinstance(e.version, DerivedMemory) and set(e.version.versions) & t
        )
    return ForgettingMetrics(
        record=record.digest,
        retention=p(len(now), len(was), confidence),
        unavailable=len(hidden),
        suppressed=sum(1 for d in decided.values() if d.state == "suppressed"),
        precision=precision,
        recall=recall,
        accidental_retention=accidental,
        collateral=collateral,
        derived_carrying_targets=carrying,
        provenance_complete=p(
            sum(not evidence_hidden(e) for e in derived), len(derived), confidence
        ),
        stale_rate=p(sum(e.stale or evidence_hidden(e) for e in derived), len(derived), confidence),
        contradiction_visibility=p(visible, len(contested), confidence),
    )


def policy(
    name: str,
    rule: Rule,
    action: Action = "forget",
    *,
    derived: Literal["cascade", "retain"] = "cascade",
    importance: ImportanceSpec | None = None,
    selector: Selector | None = None,
    preserve_aggregates: bool = False,
    **params: float,
) -> ForgettingPolicy:
    return ForgettingPolicy(
        name=name,
        version="1",
        rule=rule,
        action=action,
        params=tuple(sorted(params.items())),
        importance=importance,
        selector=selector,
        derived=derived,
        preserve_aggregates=preserve_aggregates,
    )
