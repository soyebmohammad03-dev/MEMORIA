"""Retention schedules, the feedback loop and the longitudinal simulator (Super-Phase 5).

**Forgetting is availability, never deletion** (I7, I61). Every item ever ingested stays in
the store; a *schedule* decides, at each decision time ``t`` and from history strictly before
it, whether an item is ``active`` (retrievable) or ``archived`` (not retrievable, kept for
audit). An archived item can return if a later decision says so. Schedules:

- ``keep_all``: nothing is archived (the static baseline);
- ``age_window``: archived when older than ``max_age_days`` since it was recorded;
- ``gate``: within ``grace_days`` of being recorded an item is kept; afterwards it is kept iff
  its importance is at least ``min_importance``. No exemption: the newest evidence for a key
  can be archived;
- ``gate_stale_aware``: the gate, but the newest item of a key (by occurrence) is never
  archived (the last evidence of a claim is never forgotten, as in Phase 9), and an item is
  archived when a source retracted its value or a later, different value for its key was
  recorded outside the concurrency window ("superseded"). Superseded by *newer* is not
  *truer*, so a stale-value flood defeats this rule; it is measured, not assumed away.

**Ranking.** The answer is the available item with the highest
``(1 - lambda) * freshness + lambda * importance``, where freshness is ``0.5 ** (age since it
occurred / half-life)`` and ties go to the later occurrence. ``lambda = 0`` is exactly "the
latest occurrence wins". ``lambda > 0`` is the adaptive loop: use raises importance, importance
decides what is used. ``lambda`` is a declared dose, not a tuned value.

**Loop.** A user query is answered from available items; the cited item gets a ``used``
event; a simulated user then may confirm relevance (independent of correctness) or correct
the answer (noticing errors with a declared probability, false alarms with another); the
system emits ``retrieval_success`` after the success window unless a correction arrived.
Ground truth is used only by the simulated *user* (the environment) and by analysis; the
memory sees events. Audit probes ask each key's current value every 10 days *without* creating
events, so the instrument does not change what it measures (I20).

Time is in days. Every decision is a function of (items recorded, events recorded strictly
before ``t``); the ledger refuses later-arriving information about the past.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

from pydantic import Field

from memoria.adaptive import (
    DEFAULT,
    Event,
    EventLedger,
    ImportanceRecord,
    ImportanceSpec,
    Kind,
    Origin,
    compute,
    score,
)
from memoria.adaptive_worlds import (
    AUDIT_EVERY,
    CHECKPOINTS,
    HORIZON,
    Report,
    UserQuery,
    World,
    truth_at,
    truth_index,
)
from memoria.claims import EXTRACTOR, Claim, extract, text_digest, verify_claim
from memoria.core import Digest, Record, quantize, unit_interval
from memoria.identity import CONSERVATIVE, IdentityPolicy

CONCURRENCY_DAYS = 7.0  # reports of one key this close in occurrence disagree; farther: change
WARMUP_DAYS = 30.0
FUTURE_DAYS = 30.0  # importance calibration: was the item used again within this many days?
Rule = Literal["keep_all", "age_window", "gate", "gate_stale_aware"]
Reason = Literal[
    "beyond_age",
    "grace",
    "importance_above",
    "importance_below",
    "kept_all",
    "newest_exempt",
    "source_corrected",
    "superseded",
    "within_age",
]


class Schedule(Record):
    name: str
    version: str = "1"
    rule: Rule
    max_age_days: float = Field(default=0.0, ge=0)
    min_importance: float = Field(default=0.0, ge=0, le=1)
    grace_days: float = Field(default=14.0, ge=0)


class System(Record):
    """One memory configuration under study: ingestion, schedule, ranking, importance model."""

    name: str
    ingest: Literal["extracted", "gold"] = "extracted"
    schedule: Schedule
    rank_weight: float = Field(default=0.0, ge=0, le=1)  # lambda; 0 = latest occurrence wins
    importance: ImportanceSpec = DEFAULT
    identity: Literal["conservative", "similarity"] = "conservative"
    feedback_lag_days: float = Field(default=0.0, ge=0)  # user feedback is recorded this late


KEEP = Schedule(name="keep-all", rule="keep_all")


def age_window(days: float) -> Schedule:
    return Schedule(name=f"age-{days:g}", rule="age_window", max_age_days=days)


def gate(theta: float, *, stale_aware: bool = False) -> Schedule:
    return Schedule(
        name=f"gate-{theta:g}" + ("-stale-aware" if stale_aware else ""),
        rule="gate_stale_aware" if stale_aware else "gate",
        min_importance=theta,
    )


class RetentionDecision(Record):
    """One availability decision, with the history it was allowed to see."""

    item: str
    at: float
    cutoff: float
    schedule: Digest
    state: Literal["active", "archived"]
    reason: Reason
    importance: Digest | None  # the ImportanceRecord it used
    basis: tuple[str, ...]  # superseding items or source-correction events


@dataclass(frozen=True)
class Item:
    id: str
    key: str
    value: str
    occurred: float
    recorded: float
    source: str
    report: str  # lineage: the experience the text came from
    claim: str | None
    prior: float
    kind: str


class Feedback(NamedTuple):
    """Feedback scheduled to be recorded at the heap time it is pushed with."""

    kind: Kind
    item: str
    at: float
    origin: Origin
    cause: str


class QueryTask(NamedTuple):
    number: int
    query: UserQuery


Task = Report | Feedback | QueryTask | str  # str: an audit probe of that key


class AuditProbe(NamedTuple):
    t: float
    key: str
    item: str | None
    value: str | None
    truth: str
    period: int
    status: Literal["correct", "stale", "wrong", "none"]
    available: int
    known: int
    evidence_known: bool
    evidence_available: bool
    lineage_ok: bool | None  # None: no answer


@dataclass
class SimResult:
    audits: list[AuditProbe] = field(default_factory=list)
    user_status: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    unresolved: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    claims: int = 0
    misattributed: int = 0  # extracted claims whose entity is not the report's gold entity
    items: int = 0
    events: int = 0
    scanned: int = 0  # available items considered per decision, summed (cost proxy)
    importance_evals: int = 0
    text_bytes: int = 0  # bytes of text of available items, summed over audit probes
    calibration: list[tuple[float, bool, bool]] = field(default_factory=list)  # score, used, true
    decisions: list[RetentionDecision] = field(default_factory=list)
    importances: list[ImportanceRecord] = field(default_factory=list)
    latency: list[tuple[str, float, bool]] = field(default_factory=list)  # key, days, censored
    ledger: EventLedger | None = None
    catalog: dict[str, Item] = field(default_factory=dict)
    query_log: list[tuple[int, float, str, str | None]] = field(
        default_factory=list
    )  # user queries


def precompute_claims(
    world: World, policy: IdentityPolicy = CONSERVATIVE
) -> dict[str, tuple[Claim, ...]]:
    """Extraction of every report's text, once per world (independent of the system)."""
    return {r.id: extract(r.text, r.id, world.registry, policy) for r in world.reports}


def _gold(r: Report) -> tuple[tuple[str, str, str, str] | None, str]:
    """(entity, attribute, value, kind) a careful reader gets, or None with the reason."""
    if not r.certain:
        return None, "uncertain"
    return (r.entity, r.attribute, r.value, r.kind), "gold"


class Memory:
    """The decision logic of a system over recorded items and a ledger.

    Pure with respect to its inputs: ``available``, ``choose`` and ``ranked`` at time ``t`` read
    items recorded at or before ``t`` and events recorded strictly before ``t`` (the ledger
    freezes the past at ``t``). The simulator drives it live; replay drives it over a stored
    ledger, so both make the same decisions from the same records.
    """

    def __init__(self, system: System, ledger: EventLedger) -> None:
        self.system = system
        self.ledger = ledger
        self.by_key: dict[str, list[Item]] = defaultdict(list)
        self.importance_evals = 0
        self.scanned = 0

    def imp(self, it: Item, t: float) -> float:
        self.importance_evals += 1
        return score(self.ledger, it.id, it.recorded, it.prior, t, self.system.importance)

    def known(self, key: str, t: float) -> list[Item]:
        return [i for i in self.by_key[key] if i.recorded <= t]

    def newest(self, key: str, t: float) -> Item | None:
        return max(self.known(key, t), key=lambda i: (i.occurred, i.recorded, i.id), default=None)

    def decide(self, it: Item, t: float, top: Item | None) -> tuple[bool, Reason, tuple[str, ...]]:
        sch = self.system.schedule
        if sch.rule == "keep_all":
            return True, "kept_all", ()
        age = t - it.recorded
        if sch.rule == "age_window":
            return (
                (True, "within_age", ()) if age <= sch.max_age_days else (False, "beyond_age", ())
            )
        if sch.rule == "gate_stale_aware":
            if top is not None and it.id == top.id:
                return True, "newest_exempt", ()
            cor = [
                e.id
                for e in self.ledger.visible(it.id, t)
                if e.kind == "correction" and e.origin == "source_claim"
            ]
            if cor:
                return False, "source_corrected", tuple(sorted(cor))
            sup = [
                j.id
                for j in self.by_key[it.key]
                if j.recorded < t
                and j.value != it.value
                and j.occurred > it.occurred + CONCURRENCY_DAYS
            ]
            if sup:
                return False, "superseded", tuple(sorted(sup))
        if age <= sch.grace_days:
            return True, "grace", ()
        ok = self.imp(it, t) >= sch.min_importance
        return ok, "importance_above" if ok else "importance_below", ()

    def decisions(self, key: str, t: float) -> list[tuple[Item, bool, Reason, tuple[str, ...]]]:
        """The availability decision of every item of ``key`` known at ``t``."""
        items = self.known(key, t)
        top = self.newest(key, t) if self.system.schedule.rule == "gate_stale_aware" else None
        self.scanned += len(items)
        return [(it, *self.decide(it, t, top)) for it in items]

    def available(self, key: str, t: float) -> list[Item]:
        return [it for it, ok, _, _ in self.decisions(key, t) if ok]

    def rank_key(self, i: Item, t: float) -> tuple[float, float, str]:
        """The ordering key of a candidate: rank score, then later occurrence, then id."""
        lam = self.system.rank_weight
        half = self.system.importance.half_life_days
        fresh = 0.5 ** (max(0.0, t - i.occurred) / half)
        return (quantize((1 - lam) * fresh + lam * self.imp(i, t)), i.occurred, i.id)

    def choose(self, cands: list[Item], t: float) -> Item | None:
        if not cands:
            return None
        if self.system.rank_weight == 0:
            return max(cands, key=lambda i: (i.occurred, i.recorded, i.id))
        return max(cands, key=lambda i: self.rank_key(i, t))


def simulate(
    world: World,
    system: System,
    claims: dict[str, tuple[Claim, ...]] | None = None,
    *,
    shadow: ImportanceSpec | None = None,
    trace: bool = False,
) -> SimResult:
    """Run ``system`` over ``world`` chronologically. ``shadow`` scores importance passively at
    the checkpoints (for calibration) without letting it influence anything; ``trace`` keeps
    every availability decision and importance record at the checkpoints."""
    if system.ingest == "extracted" and claims is None:
        claims = precompute_claims(world)
    declared = dict(world.declared)
    user = dict(world.user)
    ledger = EventLedger()
    res = SimResult(ledger=ledger)
    index = truth_index(world)
    sch, spec = system.schedule, system.importance
    mem = Memory(system, ledger)
    by_key = mem.by_key
    seen_claims: dict[str, Claim] = {}
    texts = {r.id: r.text for r in world.reports}
    corrected_used: set[str] = set()
    lineage_memo: dict[str, bool] = {}
    heap: list[tuple[float, int, int, Task]] = []
    seq = 0
    pending: list[tuple[str, float, float, bool]] = []

    def push(t: float, prio: int, task: Task) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(heap, (t, prio, seq, task))

    for r in world.reports:
        push(r.recorded, 0, r)
    for i, q in enumerate(world.queries):
        push(q.at, 1, QueryTask(i, q))
    d = AUDIT_EVERY
    while d <= HORIZON:
        for k in world.keys:
            push(d + 0.5, 2, k)
        d += AUDIT_EVERY

    def emit(kind: Kind, item: str, at: float, recorded: float, origin: Origin, cause: str) -> None:
        ledger.append(
            Event(kind=kind, item=item, at=at, recorded_at=recorded, origin=origin, cause=cause)
        )

    def available(key: str, t: float, tr: bool = False) -> list[Item]:
        dec = mem.decisions(key, t)
        if tr:
            for it, ok, reason, basis in dec:
                rec = (
                    compute(ledger, it.id, it.recorded, it.prior, t, spec)
                    if reason.startswith("importance")
                    else None
                )
                if rec is not None:
                    res.importances.append(rec)
                res.decisions.append(
                    RetentionDecision(
                        item=it.id,
                        at=t,
                        cutoff=t,
                        schedule=sch.digest,
                        state="active" if ok else "archived",
                        reason=reason,
                        importance=rec.digest if rec else None,
                        basis=basis,
                    )
                )
        return [it for it, ok, _, _ in dec if ok]

    def choose(cands: list[Item], t: float) -> Item | None:
        return mem.choose(cands, t)

    def lineage(it: Item) -> bool:
        if it.id not in lineage_memo:
            ok = it.report in texts
            if ok and it.claim is not None:
                ok = not verify_claim(seen_claims[it.claim], texts[it.report])
            lineage_memo[it.id] = ok
        return lineage_memo[it.id]

    def ingest(r: Report, t: float) -> None:
        parsed: list[tuple[str, str, str, str, str | None]] = []  # entity, attr, value, kind, claim
        if system.ingest == "gold":
            g, _ = _gold(r)
            if g is None:
                res.unresolved["uncertain"] += 1
            else:
                parsed.append((*g, None))
        else:
            assert claims is not None
            for c in claims[r.id]:
                res.claims += 1
                seen_claims[c.id] = c
                if c.status == "unresolved" or c.kind is None:
                    res.unresolved[c.reason] += 1
                    continue
                assert c.entity and c.attribute and c.value  # noqa: PT018
                res.misattributed += c.entity != r.entity
                parsed.append((c.entity, c.attribute, c.value, c.kind, c.id))
        for n, (ent, attr, value, kind, cid) in enumerate(parsed):
            key = f"{ent}.{attr}"
            if key not in index:
                continue
            if kind == "negate":
                for j in by_key[key]:
                    if j.value == value and j.occurred <= r.occurred:
                        emit("correction", j.id, t, t, "source_claim", cid or r.id)
                continue
            it = Item(
                id=f"item:{r.id}" + (f"#{n}" if len(parsed) > 1 else ""),
                key=key,
                value=value,
                occurred=r.occurred,
                recorded=t,
                source=r.source,
                report=r.id,
                claim=cid,
                prior=declared[r.source.partition(":")[0]],
                kind=kind,
            )
            for j in by_key[key]:
                if j.value != value:
                    if kind == "correct":
                        emit("correction", j.id, t, t, "source_claim", it.id)
                    if abs(j.occurred - it.occurred) <= CONCURRENCY_DAYS:
                        emit("contradiction_discovered", j.id, t, t, "system_observed", it.id)
                        emit("contradiction_discovered", it.id, t, t, "system_observed", j.id)
            by_key[key].append(it)
            res.catalog[it.id] = it
            res.items += 1

    def probe(t: float, key: str) -> None:
        day = round(t - 0.5)
        at_checkpoint = day in CHECKPOINTS
        cands = available(key, t, tr=trace and at_checkpoint)
        it = choose(cands, t)
        truth, period = truth_at(index, key, t)
        known = [i for i in by_key[key] if i.recorded <= t]
        past = set(index[key][1][:period])
        status: Literal["correct", "stale", "wrong", "none"]
        if it is None:
            status = "none"
        elif it.value == truth:
            status = "correct"
        elif it.value in past:
            status = "stale"
        else:
            status = "wrong"
        if at_checkpoint:
            res.text_bytes += sum(len(texts[i.report]) for i in cands)
        res.audits.append(
            AuditProbe(
                t=t,
                key=key,
                item=it.id if it else None,
                value=it.value if it else None,
                truth=truth,
                period=period,
                status=status,
                available=len(cands),
                known=len(known),
                evidence_known=any(i.value == truth for i in known),
                evidence_available=any(i.value == truth for i in cands),
                lineage_ok=lineage(it) if it else None,
            )
        )
        if shadow is not None and at_checkpoint and t + FUTURE_DAYS <= HORIZON:
            for i in cands:
                pending.append(
                    (i.id, t, score(ledger, i.id, i.recorded, i.prior, t, shadow), i.value == truth)
                )

    while heap:
        t, _, _, task = heapq.heappop(heap)
        if isinstance(task, Report):
            ingest(task, t)
        elif isinstance(task, Feedback):
            if task.kind == "retrieval_success" and task.cause in corrected_used:
                continue
            if task.kind == "correction":
                corrected_used.add(task.cause)
            emit(task.kind, task.item, task.at, t, task.origin, task.cause)
        elif isinstance(task, QueryTask):
            qi, q = task
            cands = available(q.key, t)
            it = choose(cands, t)
            res.query_log.append((qi, t, q.key, it.id if it else None))
            truth, _ = truth_at(index, q.key, t)
            if it is None:
                res.user_status["none"] += 1
                continue
            res.user_status["correct" if it.value == truth else "wrong"] += 1
            used = Event(
                kind="used", item=it.id, at=t, recorded_at=t, origin="system_observed",
                cause=f"query:{qi}",
            )  # fmt: skip
            ledger.append(used)
            after = t + 1.0 + system.feedback_lag_days  # the user reacts a day later
            u = unit_interval(world.seed, "user", qi)
            if u < (user["notice_wrong"] if it.value != truth else user["false_alarm"]):
                push(after, 3, Feedback("correction", it.id, t + 1.0, "user", used.id))
            if unit_interval(world.seed, "confirm", qi) < user["confirm"]:
                push(
                    after, 3, Feedback("user_confirmed_relevance", it.id, t + 1.0, "user", used.id)
                )
            due = t + spec.success_window_days
            push(due, 3, Feedback("retrieval_success", it.id, due, "system_observed", used.id))
        else:
            probe(t, task)
    for item, t0, sc, true_now in pending:
        again = any(e.kind == "used" for e in ledger.after(item, t0, t0 + FUTURE_DAYS))
        res.calibration.append((sc, again, true_now))
    res.events = ledger.count
    res.importance_evals = mem.importance_evals
    res.scanned = mem.scanned
    _latency(world, res, index)
    return res


def _latency(world: World, res: SimResult, index: dict[str, tuple[list[float], list[str]]]) -> None:
    by_key: dict[str, list[AuditProbe]] = defaultdict(list)
    for a in res.audits:
        by_key[a.key].append(a)
    for key, (starts, _) in sorted(index.items()):
        ends = [*starts[1:], HORIZON]
        for s, e in zip(starts[1:], ends[1:], strict=True):
            in_period = [a for a in by_key[key] if s <= a.t < e]
            hit = next((a for a in in_period if a.status == "correct"), None)
            res.latency.append((key, quantize((hit.t if hit else e) - s), hit is None))


def key_counts(audits: Sequence[AuditProbe], lo: float, hi: float) -> dict[str, int]:
    """Counts over audit probes with ``lo < t <= hi``."""
    c: dict[str, int] = defaultdict(int)
    for a in audits:
        if lo < a.t <= hi:
            c["n"] += 1
            c[a.status] += 1
            c["evidence_known"] += a.evidence_known
            c["evidence_kept"] += a.evidence_known and a.evidence_available
            c["available"] += a.available
            c["known"] += a.known
    return dict(c)


__all__ = [
    "CHECKPOINTS",
    "CONCURRENCY_DAYS",
    "EXTRACTOR",
    "KEEP",
    "AuditProbe",
    "Item",
    "RetentionDecision",
    "Schedule",
    "SimResult",
    "System",
    "age_window",
    "gate",
    "key_counts",
    "precompute_claims",
    "simulate",
    "text_digest",
]
