"""Beliefs, the evidence ledger and belief revision policies (Phases 11-12).

Evidence stays authoritative. A :class:`Belief` is a *derived* record: a stance on one value
of one key over one interval of valid time, with the evidence for it, the evidence against
it, a confidence score, an uncertainty decomposition and a revision lineage. It never
replaces or edits evidence, and a contradicted claim is never called false: the states are

============  ==========================================================================
SUPPORTED     the policy adopts this value for its interval
CONTESTED     concurrent evidence for another value; the policy has not adopted either
CORRECTED     replaced retroactively by a later correction (its evidence is kept)
SUPERSEDED    held for an earlier interval that a later change closed (still valid then)
UNKNOWN       no admissible evidence remains (a tombstone: the belief existed)
UNRESOLVED    the policy declines to adopt: too little corroboration, or a conflict that
              persisted
REJECTED      lost a weighted vote; *not adopted*, which says nothing about truth
============  ==========================================================================

Every change is an event in an :class:`EvidenceLedger`: evidence arrives (or is withdrawn by
a forgetting record), the policy recomputes the key's beliefs from the active evidence, and
the event records the state before and after, the transitions and their reason. Replaying
the evidence reproduces every event (:func:`replay`); :func:`violations` audits the
ledger against the revision constraints.

A policy (:class:`BeliefPolicy`) is a content-addressed record listing the signals it uses
and their parameters. The confidence *score* is ``weight / (competing mass + prior mass)``
where weights come from those signals: an ordering signal until :mod:`memoria.calibration`
shows it can be read as a probability.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.comparison import normalize
from memoria.consolidation import Period
from memoria.core import (
    Digest,
    EpistemicStatus,
    Inputs,
    Record,
    Scalar,
    UTCDatetime,
    canonical_json,
    content_hash,
    quantize,
)
from memoria.formation import parse_statement
from memoria.sources import SourceModel, TrustLedger, beta_variance

DAY = timedelta(days=1)


class BeliefState(StrEnum):
    SUPPORTED = "supported"
    CONTESTED = "contested"
    CORRECTED = "corrected"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


# States a belief may answer from at a valid time inside its interval.
ANSWERING = frozenset(
    {BeliefState.SUPPORTED, BeliefState.SUPERSEDED, BeliefState.CONTESTED, BeliefState.UNRESOLVED}
)
_RANK = {
    EpistemicStatus.OBSERVED: 0,
    EpistemicStatus.DERIVED: 1,
    EpistemicStatus.ABSTRACTED: 2,
    EpistemicStatus.INFERRED: 3,
}

UNCERTAINTY_DEFS: dict[str, tuple[str, str, str, str]] = {
    "aleatoric": (
        "the reports of the supporting sources' classes are noisy even in the limit of more "
        "evidence",
        "1 - the mean trust of the supporting roots (declared or learned Beta mean; 0.5 when "
        "trust is uninformative)",
        "declared class priors and, in learned mode, credits from the system's own beliefs",
        "not a measured error rate: with neutral trust it is a constant 0.5",
    ),
    "epistemic": (
        "how much of the belief's mass comes from the prior rather than from evidence",
        "prior_mass / (competing mass + prior_mass); shrinks as weighted evidence accumulates",
        "the weights of admissible, counted evidence",
        "not the probability the belief is wrong; copies add no independent evidence",
    ),
    "source": (
        "how poorly the reliability of the supporting sources is known",
        "mean Beta posterior variance of the supporting roots' trust, divided by 0.25",
        "declared prior and credited reports of each source",
        "not the sources' unreliability (that is aleatoric)",
    ),
    "temporal": (
        "how long since anyone last asserted this value",
        "1 - 0.5 ** (days from the latest supporting occurrence to the event / half-life)",
        "occurrence times of the supporting evidence",
        "not the probability that the world changed",
    ),
    "identity": (
        "the entity may be confused with a near-collision entity",
        "the highest similarity of an unmerged resolution candidate of the key's entity",
        "entity resolution decisions",
        "not the probability that the resolution is wrong",
    ),
    "retrieval": (
        "supporting evidence for this value has been made unavailable",
        "withdrawn / (active + withdrawn) evidence for the value",
        "withdrawal events from forgetting records",
        "not retrieval noise: ranking is not part of belief confidence",
    ),
    "contradiction": (
        "the share of the competing mass held by other values at the same time",
        "1 - weight / competing mass (0 without a competitor)",
        "weights of the competing hypotheses",
        "not disagreement across time: a temporal change is not a contradiction",
    ),
}


class Uncertainty(Record):
    """Seven operationally defined quantities in [0, 1]; see :data:`UNCERTAINTY_DEFS`."""

    aleatoric: float = Field(ge=0, le=1)
    epistemic: float = Field(ge=0, le=1)
    source: float = Field(ge=0, le=1)
    temporal: float = Field(ge=0, le=1)
    identity: float = Field(ge=0, le=1)
    retrieval: float = Field(ge=0, le=1)
    contradiction: float = Field(ge=0, le=1)


# --- evidence ------------------------------------------------------------------------------------


class EvidenceItem(Record):
    """One piece of evidence: a structured claim with its occurrence and arrival times.

    ``occurred_at`` is valid time; ``recorded_at`` is when it reached the system (the
    ledger's logical clock). A derived item (from a consolidated memory) names the
    experiences it covers in ``lineage`` so its independence can be traced.
    """

    id: Digest
    key: str = Field(pattern=r"^[a-z0-9_.-]+$")
    verb: Literal["set", "correct", "forget"]
    value: str | None  # normalised tokens
    raw: str | None
    occurred_at: UTCDatetime
    recorded_at: UTCDatetime
    source: str = Field(min_length=1)
    status: EpistemicStatus = EpistemicStatus.OBSERVED
    multiplicity: int = Field(default=1, ge=1)
    lineage: tuple[tuple[Digest, str], ...] = ()  # (experience, source), sorted

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if (self.value is None) != (self.verb == "forget"):
            raise ValueError("a forget carries no value; set and correct do")
        if self.recorded_at < self.occurred_at:
            raise ValueError("evidence cannot be recorded before it occurred")
        if (self.status is EpistemicStatus.OBSERVED) != (not self.lineage):
            raise ValueError("only derived evidence has lineage")
        if list(self.lineage) != sorted(set(self.lineage)):
            raise ValueError("lineage must be unique and sorted")
        return self

    @property
    def epistemic_rank(self) -> int:
        return _RANK[self.status]


def evidence_from_statement(
    experience_id: str,
    content: str,
    occurred_at: datetime,
    recorded_at: datetime,
    source: str,
) -> EvidenceItem | None:
    """The evidence a statement experience carries; free text carries none (I59)."""
    s = parse_statement(content)
    if s is None:
        return None
    return EvidenceItem(
        id=experience_id,
        key=s.key,
        verb=s.verb,  # type: ignore[arg-type]
        value=" ".join(normalize(s.value)) if s.value is not None else None,
        raw=s.value,
        occurred_at=occurred_at,
        recorded_at=recorded_at,
        source=source,
    )


def evidence_set_hash(items: Iterable[EvidenceItem]) -> str:
    """Identity of a set of evidence, independent of arrival order and times."""
    return content_hash(sorted({(i.id, i.key, i.verb, i.value, i.source) for i in items}))


def arrival_hash(items: Iterable[EvidenceItem]) -> str:
    """Identity of an arrival order: the evidence ids by (arrival time, id)."""
    return content_hash([i.id for i in sorted(items, key=lambda i: (i.recorded_at, i.id))])


# --- policies ------------------------------------------------------------------------------------

SIGNALS = ("count", "depth", "identity", "independence", "lineage", "penalty", "recency",
           "staleness", "trust")  # fmt: skip
Kind = Literal[
    "contradiction-penalty", "conservative", "corroboration", "evidence-count",
    "provenance-depth", "recency-aware", "source-weighted", "temporal-validity",
]  # fmt: skip


class SignalUse(Record):
    name: Literal["count", "depth", "identity", "independence", "lineage", "penalty", "recency",
                  "staleness", "trust"]  # fmt: skip
    params: Inputs = ()


class TemporalPolicy(Record):
    """How time separates a change from a contradiction: claims within ``concurrency_days``
    of each other are concurrent (a conflict); further apart, a later value is a change."""

    concurrency_days: float = Field(ge=0, allow_inf_nan=False)
    half_life_days: float = Field(default=30.0, gt=0, allow_inf_nan=False)  # temporal uncertainty


class BeliefPolicy(Record):
    """A belief revision policy. Its digest is its identity; it exposes every signal it uses."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    kind: Kind
    signals: tuple[SignalUse, ...]
    temporal: TemporalPolicy
    margin: float = Field(ge=0, allow_inf_nan=False)  # a winner needs (1 + margin) x the runner-up
    min_roots: int = Field(ge=1)  # independent roots needed to adopt a value
    confirm_roots: int = Field(ge=1)  # independent roots needed to accept a later value as a change
    hysteresis: float = Field(ge=0, allow_inf_nan=False)  # extra margin to displace an incumbent
    unresolve_after: int = Field(ge=0)  # events a conflict may persist before UNRESOLVED (0: never)
    prior_mass: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        names = [s.name for s in self.signals]
        if names != sorted(set(names)) or "count" not in names:
            raise ValueError("signals must be unique, sorted, and include count")
        return self

    def signal(self, name: str) -> dict[str, Scalar] | None:
        for s in self.signals:
            if s.name == name:
                return dict(s.params)
        return None


def _sig(name: str, **params: Scalar) -> SignalUse:
    return SignalUse.model_validate({"name": name, "params": tuple(sorted(params.items()))})


def _policy(name: str, kind: Kind, *signals: SignalUse, window: float = 1.0, margin: float = 0.0,
            min_roots: int = 1, confirm: int = 1, hysteresis: float = 0.0,
            unresolve: int = 0) -> BeliefPolicy:  # fmt: skip
    ordered = tuple(sorted([_sig("count"), *signals], key=lambda s: s.name))
    return BeliefPolicy(
        name=name, version="1", kind=kind, signals=ordered,
        temporal=TemporalPolicy(concurrency_days=window), margin=margin, min_roots=min_roots,
        confirm_roots=confirm, hysteresis=hysteresis, unresolve_after=unresolve,
    )  # fmt: skip


POLICIES: dict[str, BeliefPolicy] = {
    p.name: p
    for p in (
        _policy("A:evidence-count", "evidence-count"),
        _policy("B:source-weighted", "source-weighted", _sig("trust", mode="learned")),
        _policy("C:recency-aware", "recency-aware", _sig("recency", half_life_days=5.0)),
        _policy("D:temporal-validity", "temporal-validity",
                _sig("staleness", horizon_days=90.0, half_life_days=60.0), window=2.0),
        _policy("E:corroboration", "corroboration", _sig("independence"), min_roots=2),
        _policy("F:contradiction-penalty", "contradiction-penalty",
                _sig("penalty", strength=1.0), margin=0.25),
        _policy("G:provenance-depth", "provenance-depth", _sig("depth", gamma=0.7)),
        _policy("H:conservative", "conservative", _sig("independence"),
                _sig("trust", mode="declared"), margin=0.5, min_roots=2, confirm=2,
                hysteresis=0.5, unresolve=4),
    )
}  # fmt: skip


# --- beliefs -------------------------------------------------------------------------------------


class Contribution(Record):
    """One unit of evidence's part in a belief's weight, with every factor recorded."""

    evidence: Digest
    source: str
    root: str
    multiplicity: float
    trust: float
    recency: float
    depth: float
    staleness: float
    weight: float  # multiplicity x trust x recency x depth x staleness
    counted: bool  # False: a duplicate of the same root's stronger contribution
    reason: str  # "counted", "duplicate_lineage" or "lineage_unavailable"


class Belief(Record):
    """A stance on one value of one key over an interval of valid time."""

    id: str
    key: str
    value: str
    valid_from: UTCDatetime
    valid_to: UTCDatetime | None  # as believed
    observed_to: UTCDatetime | None  # where the evidence's own timeline ends
    state: BeliefState
    reason: str
    supporting: tuple[Digest, ...]
    contradicting: tuple[Digest, ...]
    withdrawn: tuple[Digest, ...]
    sources: tuple[str, ...]
    roots: tuple[str, ...]
    naive_count: int = Field(ge=0)
    independent_count: int = Field(ge=0)
    epistemic: EpistemicStatus
    raw_weight: float
    penalty: float
    weight: float  # raw_weight - penalty, floored at 0
    mass: float  # weights of the competing hypotheses, this one included
    prior_mass: float
    score: float = Field(ge=0, le=1)
    uncertainty: Uncertainty
    competing: tuple[str, ...]
    contributions: tuple[Contribution, ...]
    predecessors: tuple[str, ...]
    revisions: int = Field(ge=0)
    since_seq: int = Field(ge=0)  # the ledger event that last changed its state
    policy: Digest

    def fingerprint(self) -> tuple[object, ...]:
        end = self.valid_to.isoformat() if self.valid_to else None
        return (self.id, self.state.value, round(self.score, 9), end, len(self.supporting),
                len(self.contradicting), self.naive_count)  # fmt: skip


class KeyBeliefs(Record):
    """Every belief about one key at one logical time (the state a revision produces)."""

    key: str
    at: UTCDatetime
    seq: int = Field(ge=0)
    beliefs: tuple[Belief, ...]  # by (valid_from, value)
    tombstones: tuple[Belief, ...] = ()  # beliefs whose evidence was all withdrawn (UNKNOWN)
    retractions: tuple[Digest, ...] = ()  # forget statements
    conflict_events: int = Field(default=0, ge=0)

    def fingerprint(self) -> str:
        return content_hash([[b.fingerprint() for b in (*self.beliefs, *self.tombstones)],
                             self.retractions, self.conflict_events])  # fmt: skip

    def content_fingerprint(self) -> str:
        """The current beliefs alone (ids, states, scores, intervals, evidence counts), without
        the bookkeeping that depends on arrival order (tombstones, the conflict counter)."""
        return content_hash([b.fingerprint() for b in self.beliefs])

    def by_id(self) -> dict[str, Belief]:
        return {b.id: b for b in (*self.beliefs, *self.tombstones)}


def belief_id(key: str, value: str, start: datetime) -> str:
    h = hashlib.sha256(canonical_json([key, value, start.isoformat()]).encode()).hexdigest()
    return "belief:" + h[:20]


def epistemic_rank(b: Belief) -> int:
    return _RANK[b.epistemic]


def check_arithmetic(b: Belief) -> None:
    """A belief's weight and score follow from its recorded contributions (auditable)."""
    counted = sum(c.weight for c in b.contributions if c.counted)
    if abs(quantize(counted) - b.raw_weight) > 1e-9:
        raise ValueError(f"{b.id}: raw weight is not the sum of counted contributions")
    if abs(max(0.0, b.raw_weight - b.penalty) - b.weight) > 1e-9:
        raise ValueError(f"{b.id}: weight is not raw weight minus penalty")
    if b.state is not BeliefState.CORRECTED and b.uncertainty.identity == 0:
        expected = b.weight / (b.mass + b.prior_mass)
        if abs(expected - b.score) > 1e-9:
            raise ValueError(f"{b.id}: score is not weight / (mass + prior)")


# --- derivation ----------------------------------------------------------------------------------


@dataclass
class Context:
    policy: BeliefPolicy
    sources: SourceModel
    trust: TrustLedger
    identity: Mapping[str, float]  # entity -> identity uncertainty
    at: datetime
    seq: int
    withdrawing: bool = False  # the event withdraws evidence (else evidence arrives)


@dataclass
class _Hyp:
    period: Period
    items: list[EvidenceItem]
    id: str
    atoms: list[Contribution] = field(default_factory=list)
    raw: float = 0.0
    penalty: float = 0.0
    weight: float = 0.0
    naive: int = 0
    roots: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    withdrawn: tuple[str, ...] = ()
    state: BeliefState = BeliefState.SUPPORTED
    reason: str = ""
    valid_to: datetime | None = None
    group: int = 0
    competing: tuple[str, ...] = ()


def _trust(ctx: Context, source: str) -> tuple[float, float]:
    """(weight factor, posterior variance) of a source under the policy's trust mode."""
    sig = ctx.policy.signal("trust")
    if sig is None:
        return 1.0, 0.0
    if sig["mode"] == "neutral":
        return 1.0, beta_variance(1.0, 1.0)
    rec = ctx.trust.record(source, learned=sig["mode"] == "learned")
    return 2 * rec.mean, rec.variance


def _atoms(h: _Hyp, ctx: Context, withdrawn: set[str]) -> list[Contribution]:
    """One contribution per unit of evidence. A derived item under a lineage-aware policy
    contributes once per independent root of the experiences it covers that are still
    available; otherwise it is one report with its declared multiplicity."""
    out: list[Contribution] = []
    for it in h.items:
        if it.lineage and ctx.policy.signal("lineage") is not None:
            pairs = [(e, s) for e, s in it.lineage if e not in withdrawn]
            if not pairs:
                out.append(Contribution(
                    evidence=it.id, source=it.source, root=ctx.sources.root(it.source),
                    multiplicity=0.0, trust=0.0, recency=0.0, depth=0.0, staleness=0.0,
                    weight=0.0, counted=False, reason="lineage_unavailable",
                ))  # fmt: skip
                continue
            for root in sorted({ctx.sources.root(s) for _, s in pairs}):
                src = next(s for _, s in pairs if ctx.sources.root(s) == root)
                out.append(_atom(it, ctx, 1.0, src, root, 1))
        else:
            out.append(_atom(it, ctx, float(it.multiplicity), it.source,
                             ctx.sources.root(it.source), 0))  # fmt: skip
    return out


def _atom(it: EvidenceItem, ctx: Context, mult: float, source: str, root: str,
          extra_depth: int) -> Contribution:  # fmt: skip
    p = ctx.policy
    trust, _ = _trust(ctx, source)
    recency = depth = stale = 1.0
    if (rec := p.signal("recency")) is not None:
        age = max(0.0, (ctx.at - it.recorded_at) / DAY)
        recency = quantize(0.5 ** (age / float(rec["half_life_days"])))
    if (dep := p.signal("depth")) is not None:
        depth = quantize(float(dep["gamma"]) ** (ctx.sources.depth(source) - 1 + extra_depth))
    if (st := p.signal("staleness")) is not None:
        age = max(0.0, (ctx.at - it.occurred_at) / DAY)
        excess = max(0.0, age - float(st["horizon_days"]))
        stale = quantize(0.5 ** (excess / float(st["half_life_days"])))
    return Contribution(
        evidence=it.id, source=source, root=root, multiplicity=mult, trust=quantize(trust),
        recency=recency, depth=depth, staleness=stale,
        weight=quantize(mult * trust * recency * depth * stale), counted=True, reason="counted",
    )  # fmt: skip


def _weigh(h: _Hyp, ctx: Context, withdrawn: set[str]) -> None:
    atoms = _atoms(h, ctx, withdrawn)
    if ctx.policy.signal("independence") is not None:
        best: dict[str, int] = {}
        for i, a in enumerate(atoms):
            if a.counted and (a.root not in best or a.weight > atoms[best[a.root]].weight):
                best[a.root] = i
        atoms = [
            a if (not a.counted or best[a.root] == i)
            else a.model_copy(update={"counted": False, "reason": "duplicate_lineage"})
            for i, a in enumerate(atoms)
        ]  # fmt: skip
    h.atoms = atoms
    h.raw = quantize(sum(a.weight for a in atoms if a.counted))
    h.naive = int(sum(it.multiplicity for it in h.items))
    live = [a for a in atoms if a.reason != "lineage_unavailable"]
    h.roots = tuple(sorted({a.root for a in live}))
    h.sources = tuple(sorted({a.source for a in live}))


@dataclass
class _Open:
    value: str
    start: datetime
    experiences: list[str]
    end: datetime | None = None
    contested: bool = False
    corrected: bool = False


def build_periods(items: Sequence[EvidenceItem], key: str, window_days: float) -> list[Period]:
    """The periods of ``key`` from its evidence, ordered by occurrence (never arrival).

    Reports within ``window_days`` of a cluster's earliest are concurrent: they are one
    moment, and different values in it contest each other whatever their order. A moment with
    a different value closes the open period (a change) and opens one period per value at
    that moment (contested if several; the standing value asserted again is one of them). A
    moment with the standing value alone extends it. A ``correct`` replaces what held,
    retroactively, from that period's start (the replaced period is marked corrected);
    ``forget`` closes.
    """
    ordered = sorted(items, key=lambda i: (i.occurred_at, i.id))
    clusters: list[list[EvidenceItem]] = []
    anchor: datetime | None = None
    for it in ordered:
        if anchor is None or (it.occurred_at - anchor) / DAY > window_days:
            clusters.append([it])
            anchor = it.occurred_at
        else:
            clusters[-1].append(it)
    made: list[_Open] = []
    open_: list[_Open] = []

    def close(at: datetime) -> None:
        for q in open_:
            q.end = at
        open_.clear()

    for cl in clusters:
        t = cl[0].occurred_at
        if any(i.verb == "forget" for i in cl):
            close(t)
            cl = [i for i in cl if i.verb != "forget"]
            if not cl:
                continue
        by_value: dict[str, list[str]] = {}
        for it in cl:
            by_value.setdefault(str(it.value), []).append(it.id)
        if any(i.verb == "correct" for i in cl) and open_:
            begin = min(q.start for q in open_)
            for q in open_:
                q.corrected, q.end = True, begin
            open_.clear()
            start = begin
        elif len(open_) == 1 and set(by_value) == {open_[0].value}:
            open_[0].experiences += by_value[open_[0].value]
            continue
        else:
            close(t)
            start = t
        fresh = [_Open(v, start, ids, contested=len(by_value) > 1) for v, ids in by_value.items()]
        open_.extend(fresh)
        made.extend(fresh)
    return [
        Period(
            key=key, value=q.value, start=q.start,
            end=q.end if q.end is None or q.end > q.start else None,
            experiences=tuple(sorted(q.experiences)), contested=q.contested,
            corrected=q.corrected,
        )
        for q in made
    ]  # fmt: skip


def derive(
    key: str,
    items: Sequence[EvidenceItem],
    withdrawn: set[str],
    prev: KeyBeliefs | None,
    ctx: Context,
) -> KeyBeliefs:
    """The beliefs about ``key`` from its active evidence at ``ctx.at``. Pure given the
    previous state (for hysteresis and persistence) and the trust ledger."""
    p = ctx.policy
    active = [i for i in items if i.id not in withdrawn]
    gone = [i for i in items if i.id in withdrawn]
    by_id = {i.id: i for i in active}
    periods = build_periods(active, key, p.temporal.concurrency_days)
    hyps = [_Hyp(period=pd, items=[by_id[e] for e in pd.experiences if e in by_id],
                 id=belief_id(key, pd.value, pd.start)) for pd in periods]  # fmt: skip
    for h in hyps:
        _weigh(h, ctx, withdrawn)
        h.valid_to = h.period.end
        slack = timedelta(days=p.temporal.concurrency_days)
        h.withdrawn = tuple(
            sorted(
                g.id
                for g in gone
                if " ".join(normalize(g.raw or "")) == h.period.value
                and h.period.start <= g.occurred_at + slack
                and (h.period.end is None or g.occurred_at < h.period.end)
            )
        )
    starts = sorted({h.period.start for h in hyps})
    for h in hyps:
        h.group = starts.index(h.period.start)
    groups: list[list[_Hyp]] = [[h for h in hyps if h.group == g] for g in range(len(starts))]
    incumbents = {b.id for b in (prev.beliefs if prev else ()) if b.state is BeliefState.SUPPORTED}
    n_conflict = (prev.conflict_events if prev else 0) + 1
    any_conflict = False
    lam = p.signal("penalty")
    for gi, group in enumerate(groups):
        live = [h for h in group if not h.period.corrected]
        for h in group:
            if h.period.corrected:
                h.state, h.reason = BeliefState.CORRECTED, "corrected_by_later_evidence"
                h.competing = ()
        ended = gi < len(groups) - 1 or any(h.period.end is not None for h in live)
        for h in live:
            h.penalty = 0.0
            if lam is not None and len(live) > 1:
                h.penalty = quantize(
                    float(lam["strength"]) * sum(o.raw for o in live if o is not h)
                )
            h.weight = quantize(max(0.0, h.raw - h.penalty))
            h.competing = tuple(sorted(o.id for o in live))
        for h in group:
            if h.period.corrected:
                h.weight = h.raw
        if len(live) > 1:
            ranked = sorted(live, key=lambda h: (-h.weight, h.id))
            top, second = ranked[0], ranked[1]
            winner: _Hyp | None = None
            if (
                top.weight > 0 and top.weight > (1 + p.margin) * second.weight + 1e-12
            ):  # ties are not wins
                winner = top
            inc = next((h for h in ranked if h.id in incumbents), None)
            held = False
            if (
                inc is not None
                and inc is not winner
                and p.hysteresis > 0
                and inc.weight * (1 + p.hysteresis) >= top.weight
            ):
                winner, held = inc, True
            if winner is not None and len(winner.roots) < p.min_roots:
                winner = None
            if winner is None:
                any_conflict = True
                stuck = p.unresolve_after > 0 and n_conflict >= p.unresolve_after
                thin = all(len(h.roots) < p.min_roots for h in live)
                state = BeliefState.UNRESOLVED if (stuck or thin) else BeliefState.CONTESTED
                if thin:
                    why = "insufficient_corroboration"
                elif stuck:
                    why = "persistent_conflict"
                else:
                    why = "conflicting_evidence"
                for h in live:
                    h.state, h.reason = state, why
            else:
                for h in live:
                    if h is winner:
                        h.state = BeliefState.SUPERSEDED if ended else BeliefState.SUPPORTED
                        if held:
                            h.reason = "hysteresis_hold"
                        else:
                            h.reason = "temporal_change" if ended else "weighted_majority"
                    else:
                        h.state, h.reason = BeliefState.REJECTED, "lost_weighted_vote"
        elif live:
            h = live[0]
            if len(h.roots) < p.min_roots:
                h.state, h.reason = BeliefState.UNRESOLVED, "insufficient_corroboration"
            elif ended:
                h.state = BeliefState.SUPERSEDED
                retracted = h.period.end is not None and gi == len(groups) - 1
                h.reason = "retracted" if retracted else "temporal_change"
            else:
                h.state, h.reason = BeliefState.SUPPORTED, "single_hypothesis"
    # A later value seen from too few independent roots is not yet a change.
    if p.confirm_roots > 1 and len(groups) > 1:
        last = [h for h in groups[-1] if not h.period.corrected]
        before = [h for h in groups[-2] if h.state is BeliefState.SUPERSEDED]
        if len(last) == 1 and len(before) == 1 and len(last[0].roots) < p.confirm_roots \
                and last[0].period.end is None:  # fmt: skip
            new, earlier = last[0], before[0]
            any_conflict = True
            for h in (new, earlier):
                h.state, h.reason = BeliefState.CONTESTED, "change_unconfirmed"
                h.competing = tuple(sorted((new.id, earlier.id)))
                h.penalty = 0.0
                h.weight = h.raw
            earlier.valid_to = None
    beliefs = [
        _finish(h, hyps, ctx, key)
        for h in sorted(hyps, key=lambda h: (h.period.start, h.period.value))
    ]
    # Predecessors, revisions and tombstones from the previous state.
    prev_by = prev.by_id() if prev else {}
    finished: list[Belief] = []
    prev_group: dict[int, list[str]] = {
        g: [h.id for h in gr if not h.period.corrected] for g, gr in enumerate(groups)
    }
    gid = {h.id: h.group for h in hyps}
    for bel in beliefs:
        was = prev_by.get(bel.id)
        changed = was is None or was.state is not bel.state
        preds = tuple(sorted(prev_group.get(gid[bel.id] - 1, ()))) if gid[bel.id] > 0 else ()
        finished.append(bel.model_copy(update={
            "predecessors": preds,
            "revisions": 0 if was is None else was.revisions + (1 if changed else 0),
            "since_seq": ctx.seq if (changed or was is None) else was.since_seq,
        }))  # fmt: skip
    present = {b.id for b in finished}
    tombs = {t.id: t for t in (prev.tombstones if prev else ())}
    for lost in prev.beliefs if prev else ():
        gone_for_good = ctx.withdrawing and set(lost.supporting) <= withdrawn
        if lost.id not in present and lost.state is not BeliefState.UNKNOWN and gone_for_good:
            tombs[lost.id] = lost.model_copy(update={
                "state": BeliefState.UNKNOWN, "reason": "evidence_withdrawn", "supporting": (),
                "withdrawn": tuple(sorted({*lost.withdrawn, *lost.supporting})), "weight": 0.0,
                "raw_weight": 0.0, "penalty": 0.0, "score": 0.0, "contributions": (),
                "since_seq": ctx.seq, "revisions": lost.revisions + 1,
            })  # fmt: skip
    for tid in list(tombs):
        if tid in present:
            del tombs[tid]
    return KeyBeliefs(
        key=key, at=ctx.at, seq=ctx.seq, beliefs=tuple(finished),
        tombstones=tuple(sorted(tombs.values(), key=lambda b: b.id)),
        retractions=tuple(sorted(i.id for i in active if i.verb == "forget")),
        conflict_events=n_conflict if any_conflict else 0,
    )  # fmt: skip


def _finish(h: _Hyp, hyps: list[_Hyp], ctx: Context, key: str) -> Belief:
    p = ctx.policy
    by = {x.id: x for x in hyps}
    comp = [by[i] for i in h.competing if i in by] if h.competing else []
    mass = quantize(sum(x.weight for x in comp)) if comp else h.weight
    score = quantize(h.weight / (mass + p.prior_mass))
    rank = min((it.epistemic_rank for it in h.items), default=0)
    status = next(s for s, r in _RANK.items() if r == rank)
    ent = key.split(".", 1)[0]
    ident = quantize(ctx.identity.get(ent, 0.0)) if p.signal("identity") is not None else 0.0
    if ident:
        score = quantize(score * (1 - ident))
    variances = [_trust(ctx, s)[1] for s in h.sources]
    trusts = []
    for a in h.atoms:
        if a.counted:
            trusts.append(a.trust / 2 if p.signal("trust") is not None else 0.5)
    last = max((it.occurred_at for it in h.items), default=h.period.start)
    horizon = min(ctx.at, h.valid_to) if h.valid_to else ctx.at
    age = max(0.0, (horizon - last) / DAY)
    active_n, gone_n = len(h.items), len(h.withdrawn)
    unc = Uncertainty(
        aleatoric=quantize(1 - (sum(trusts) / len(trusts) if trusts else 0.5)),
        epistemic=quantize(p.prior_mass / (mass + p.prior_mass)),
        source=quantize(min(1.0, (sum(variances) / len(variances) if variances else 0.25) / 0.25)),
        temporal=quantize(1 - 0.5 ** (age / p.temporal.half_life_days)),
        identity=ident,
        retrieval=quantize(gone_n / (active_n + gone_n)) if active_n + gone_n else 0.0,
        contradiction=quantize(1 - h.weight / mass) if len(comp) > 1 and mass > 0 else 0.0,
    )
    contradicting = tuple(sorted({it.id for x in comp if x is not h for it in x.items}))
    return Belief(
        id=h.id, key=key, value=h.period.value, valid_from=h.period.start, valid_to=h.valid_to,
        observed_to=h.period.end, state=h.state, reason=h.reason or "unassigned",
        supporting=tuple(sorted(it.id for it in h.items)), contradicting=contradicting,
        withdrawn=h.withdrawn, sources=h.sources, roots=h.roots, naive_count=h.naive,
        independent_count=len(h.roots), epistemic=status, raw_weight=h.raw, penalty=h.penalty,
        weight=h.weight, mass=mass, prior_mass=p.prior_mass, score=score, uncertainty=unc,
        competing=h.competing, contributions=tuple(h.atoms), predecessors=(), revisions=0,
        since_seq=0, policy=p.digest,
    )  # fmt: skip
