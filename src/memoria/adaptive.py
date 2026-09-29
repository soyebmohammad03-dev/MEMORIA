"""Access and feedback events, and an auditable importance model (Super-Phase 5).

*Descriptive access statistics are not epistemic truth.* Importance says how much a memory
item has been used, confirmed, contradicted and corrected, and how good its provenance is; it
never says the item is true (I45 extended). Everything here is a pure function of recorded
events.

**Events.** An :class:`Event` is either an *access* (``used``: the item supported an answer) or
*feedback* (``retrieval_success``: no correction followed a use within the window;
``user_confirmed_relevance``; ``correction``; ``contradiction_discovered``). Each has an
``origin`` and a ``cause``. Answer correctness judged against ground truth (origin
``evaluation``) is circular evidence and is refused for the ledger: a memory cannot learn from
the instrument that measures it.

**Temporal cutoff.** An event is *visible* to a decision at time ``t`` only if it was
*recorded* strictly before ``t`` (feedback occurs later than the access it concerns, and may be
recorded later still). A decision *freezes* the past: afterwards the ledger refuses any event
recorded before the freeze, so no decision can be changed by information that arrives later
(the I14 rule for feedback).

**Importance.** A weighted sum of six components in [0, 1], each with raw inputs and the ids of
the events it used (:class:`ImportanceRecord`, re-derivable by :func:`verify_importance`):
``frequency`` (uses in a trailing window, saturating), ``success`` (Beta-smoothed share of
matured uses followed by success feedback), ``contradiction`` (contradiction exposure:
review priority, not truth), ``correction`` (1 / (1 + corrections)), ``recency`` (half-life
since the last use or creation) and ``provenance`` (the declared source prior). Weights are
declared, positive and sum to 1; :func:`ablate` removes one component and renormalises.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Sequence
from functools import cached_property
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.core import Digest, Record, content_hash, quantize

ACCESS_KINDS = ("used",)
FEEDBACK_KINDS = (
    "contradiction_discovered",
    "correction",
    "retrieval_success",
    "user_confirmed_relevance",
)
Kind = Literal[
    "used",
    "retrieval_success",
    "user_confirmed_relevance",
    "correction",
    "contradiction_discovered",
]
Origin = Literal["system_observed", "user", "source_claim", "evaluation"]
CIRCULAR_ORIGINS = ("evaluation",)
COMPONENTS = ("contradiction", "correction", "frequency", "provenance", "recency", "success")
STATIC_COMPONENTS = ("age",)  # event-free: only for the static heuristic baseline


class LeakageError(ValueError):
    """An event recorded before a decision that has already been made."""


class CircularEvidenceError(ValueError):
    """Ground-truth evaluation offered as feedback."""


class Event(Record):
    """One access or feedback event on one memory item."""

    kind: Kind
    item: str
    at: float  # when it happened (days)
    recorded_at: float  # when the system learned of it: >= at
    origin: Origin
    cause: str  # the event, claim or query it responds to

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if self.recorded_at < self.at:
            raise ValueError("an event cannot be recorded before it happens")
        if self.kind == "used" and self.origin != "system_observed":
            raise ValueError("access events are observed by the system")
        return self

    @cached_property
    def id(self) -> str:
        return "event:" + self.digest[7:23]


class EventLedger:
    """Append-only events per item, visible by strict record-time cutoff, frozen by decisions."""

    def __init__(self) -> None:
        self._by_item: dict[str, list[tuple[float, str, Event]]] = defaultdict(list)
        self._ids: set[str] = set()
        self.frozen = float("-inf")  # decisions have been made up to this time
        self.count = 0

    def append(self, event: Event) -> None:
        if event.origin in CIRCULAR_ORIGINS:
            raise CircularEvidenceError("ground-truth evaluation is not evidence for memory")
        if event.recorded_at < self.frozen:
            raise LeakageError(
                f"event recorded at {event.recorded_at} after a decision at {self.frozen}"
            )
        eid = event.id
        if eid in self._ids:
            return
        self._ids.add(eid)
        bisect.insort(
            self._by_item[event.item], (event.recorded_at, eid, event), key=lambda x: x[:2]
        )
        self.count += 1

    def visible(self, item: str, at: float) -> list[Event]:
        """Events on ``item`` recorded strictly before ``at``; freezes the past at ``at``."""
        self.frozen = max(self.frozen, at)
        rows = self._by_item.get(item, [])
        n = bisect.bisect_left(rows, at, key=lambda x: x[0])
        return [e for _, _, e in rows[:n]]

    def after(self, item: str, at: float, until: float) -> list[Event]:
        """Events recorded in [at, until): analysis only (never used by a decision)."""
        rows = self._by_item.get(item, [])
        lo = bisect.bisect_left(rows, at, key=lambda x: x[0])
        hi = bisect.bisect_left(rows, until, key=lambda x: x[0])
        return [e for _, _, e in rows[lo:hi]]

    def copy(self, before: float | None = None) -> EventLedger:
        """A fresh, unfrozen ledger with the same events (only those recorded strictly before
        ``before`` if given). Reading a copy never moves the original's freeze."""
        other = EventLedger()
        for e in self.all():
            if before is None or e.recorded_at < before:
                other.append(e)
        return other

    def all(self) -> list[Event]:
        return sorted(
            (e for rows in self._by_item.values() for _, _, e in rows),
            key=lambda e: (e.recorded_at, e.id),
        )


class ImportanceSpec(Record):
    """The declared importance model. Its digest is its identity."""

    name: str
    version: str = "1"
    weights: tuple[tuple[str, float], ...]  # component -> weight, sorted, positive, sum 1
    freq_window_days: float = Field(default=90.0, gt=0)
    freq_scale: float = Field(default=3.0, gt=0)  # uses for 1 - 1/e
    success_window_days: float = Field(default=14.0, gt=0)  # a use matures after this
    prior_alpha: float = Field(default=1.0, gt=0)
    prior_beta: float = Field(default=1.0, gt=0)
    contradiction_scale: float = Field(default=2.0, gt=0)
    half_life_days: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        names = [n for n, _ in self.weights]
        if (
            names != sorted(set(names))
            or not set(names) <= {*COMPONENTS, *STATIC_COMPONENTS}
            or not names
        ):
            raise ValueError("weights: unique, sorted, known components")
        if any(w <= 0 for _, w in self.weights) or abs(sum(w for _, w in self.weights) - 1) > 1e-9:
            raise ValueError("weights must be positive and sum to 1")
        return self

    @property
    def uses_events(self) -> bool:
        return bool({n for n, _ in self.weights} - {"provenance", *STATIC_COMPONENTS})


DEFAULT = ImportanceSpec(
    name="adaptive",
    weights=(
        ("contradiction", 0.05),
        ("correction", 0.15),
        ("frequency", 0.2),
        ("provenance", 0.2),
        ("recency", 0.2),
        ("success", 0.2),
    ),
)
STATIC = ImportanceSpec(name="static", weights=(("age", 0.5), ("provenance", 0.5)))


def ablate(spec: ImportanceSpec, component: str) -> ImportanceSpec:
    """``spec`` without ``component``; the remaining weights are renormalised."""
    rest = [(n, w) for n, w in spec.weights if n != component]
    if len(rest) == len(spec.weights) or not rest:
        raise ValueError(f"cannot ablate {component!r}")
    total = sum(w for _, w in rest)
    return spec.model_copy(
        update={
            "name": f"{spec.name}-minus-{component}",
            "weights": tuple((n, quantize(w / total)) for n, w in rest),
        }
    )


class ComponentValue(Record):
    name: str
    value: float = Field(ge=0, le=1)
    weight: float
    raw: tuple[tuple[str, float], ...]  # named raw inputs
    events: tuple[str, ...]  # ids of the events used, sorted


class ImportanceRecord(Record):
    """One importance value of one item at one time, with everything that produced it."""

    item: str
    at: float
    spec: Digest
    components: tuple[ComponentValue, ...]  # by name
    score: float = Field(ge=0, le=1)
    events: tuple[str, ...]  # every event used, sorted (the visible ones that mattered)
    cutoff: float  # events recorded strictly before this were visible

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        total = quantize(math.fsum(c.weight * c.value for c in self.components))
        if total != self.score:
            raise ValueError("score is not the weighted sum of its components")
        if self.cutoff != self.at:
            raise ValueError("the cutoff is the decision time")
        return self


Part = tuple[str, float, tuple[tuple[str, float], ...], list[Event]]


def _parts(
    ledger: EventLedger, item: str, created: float, prior: float, at: float, spec: ImportanceSpec
) -> list[Part]:
    """The components of ``spec`` for ``item`` at ``at``: (name, value, raw inputs, events)."""
    ev = ledger.visible(item, at)
    used = [e for e in ev if e.kind == "used"]
    w = {n for n, _ in spec.weights}
    out: list[Part] = []

    def add(name: str, value: float, raw: Sequence[tuple[str, float]], evs: list[Event]) -> None:
        out.append((name, quantize(value), tuple(sorted(raw)), evs))

    if "contradiction" in w:
        contra = [e for e in ev if e.kind == "contradiction_discovered"]
        add(
            "contradiction",
            1 - math.exp(-len(contra) / spec.contradiction_scale),
            [("contradictions", len(contra))],
            contra,
        )
    if "correction" in w:
        corr = [e for e in ev if e.kind == "correction"]
        add("correction", 1 / (1 + len(corr)), [("corrections", len(corr))], corr)
    if "frequency" in w:
        recent = [e for e in used if e.recorded_at > at - spec.freq_window_days]
        add(
            "frequency",
            1 - math.exp(-len(recent) / spec.freq_scale),
            [("uses", len(recent))],
            recent,
        )
    if "provenance" in w:
        add("provenance", prior, [("source_prior", prior)], [])
    if "age" in w:  # freshness since it was recorded, whatever happened to it afterwards
        age = max(0.0, at - created)
        add("age", 0.5 ** (age / spec.half_life_days), [("age_since_recorded", quantize(age))], [])
    if "recency" in w:
        last = max([created, *(e.recorded_at for e in used)])
        add(
            "recency",
            0.5 ** (max(0.0, at - last) / spec.half_life_days),
            [("age_since_touch", quantize(max(0.0, at - last)))],
            [e for e in used if e.recorded_at == last][:1],
        )
    if "success" in w:
        matured = [e for e in used if e.recorded_at + spec.success_window_days <= at]
        ids = {m.id for m in matured}
        won = [
            e
            for e in ev
            if e.kind in ("retrieval_success", "user_confirmed_relevance") and e.cause in ids
        ]
        s = len({e.cause for e in won})
        add(
            "success",
            (s + spec.prior_alpha) / (len(matured) + spec.prior_alpha + spec.prior_beta),
            [("matured", len(matured)), ("successes", s)],
            [*matured, *won],
        )
    return out


def score(
    ledger: EventLedger, item: str, created: float, prior: float, at: float, spec: ImportanceSpec
) -> float:
    """The importance score alone: the same arithmetic as :func:`compute`, without records."""
    weights = dict(spec.weights)
    parts = _parts(ledger, item, created, prior, at, spec)
    return quantize(math.fsum(weights[n] * v for n, v, _, _ in parts))


def compute(
    ledger: EventLedger, item: str, created: float, prior: float, at: float, spec: ImportanceSpec
) -> ImportanceRecord:
    """Importance of ``item`` (recorded at ``created``, declared source prior ``prior``) as of
    ``at``, from events recorded strictly before ``at`` only."""
    weights = dict(spec.weights)
    out = [
        ComponentValue(
            name=n, value=v, weight=weights[n], raw=raw, events=tuple(sorted(e.id for e in evs))
        )
        for n, v, raw, evs in sorted(_parts(ledger, item, created, prior, at, spec))
    ]
    return ImportanceRecord(
        item=item,
        at=at,
        spec=spec.digest,
        components=tuple(out),
        score=quantize(math.fsum(c.weight * c.value for c in out)),
        events=tuple(sorted({i for c in out for i in c.events})),
        cutoff=at,
    )


def verify_importance(
    record: ImportanceRecord,
    ledger: EventLedger,
    created: float,
    prior: float,
    spec: ImportanceSpec,
) -> list[str]:
    """Problems with a stored record: not re-derivable, using a future event, or citing an event
    the ledger does not hold (empty: reproducible and traceable)."""
    bad = []
    again = compute(ledger, record.item, created, prior, record.at, spec)
    if again.digest != record.digest:
        bad.append("re-derivation differs")
    known = {e.id: e for e in ledger.all()}
    for eid in record.events:
        e = known.get(eid)
        if e is None:
            bad.append(f"unknown event {eid}")
        elif e.recorded_at >= record.at:
            bad.append(f"future event {eid}")
    return bad


def content_id(*parts: object) -> str:
    return content_hash(list(parts))[7:23]
