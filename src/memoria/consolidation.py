"""Memory consolidation: derived, hierarchical memory over immutable history (Phase 7).

Consolidation reads a :class:`~memoria.hybrid.Corpus` (every L1 memory version known at a
checkpoint, with its evidence) and writes a :class:`Hierarchy`: a content-addressed
artifact of :class:`~memoria.core.DerivedMemory` records. Nothing is ever written to the
log and no evidence is changed; a hierarchy is a pure function of (history up to the
checkpoint, policy, embedder), so :func:`replay` must reproduce it byte for byte.

.. code-block:: text

    L1 versions ──promotion filter──► promoted ──grouping (regime + guards)──► L2 facts
      (all | recent | important)          │ MergeDecision per version,          │
       not promoted: archived at L1       │ blocked merges recorded             ▼
                                                              L3 entity profiles (abstracted)
                                                              L4 timelines (abstracted)
                                                              L4 co-changes (inferred)
    every derived memory ──► LossReport (preserved / lost / altered / conflicting /
                                         unsupported features, provenance coverage)

Grouping regimes decide which versions *may* merge: ``exact`` (identical text),
``canonical`` (identical tokens), ``claim`` (same structured key and value), ``temporal``
(same key and value within one period of the key's timeline), ``semantic`` (cosine at or
above a threshold to every member). Guards then decide which merges are *unsafe*: a
different value for the same key, different numbers, negation, temporal markers or
surface entities. A blocked merge is recorded, never silently applied or dropped.
Similarity is evidence of topic, not of equivalence (Phase 5).

Every text feature used here (entities, numbers, temporal markers, negation) is a
deterministic surface heuristic, labelled as such; structured claims come from the
statement language through the evidence, never from derived content.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.comparison import normalize
from memoria.core import (
    DerivedMemory,
    Digest,
    EpistemicStatus,
    Level,
    Record,
    UTCDatetime,
    canonical_json,
    quantize,
)
from memoria.embeddings import Embedder
from memoria.hybrid import (
    CONFLICTED,
    Corpus,
    Engine,
    Entry,
    Normalization,
    conflict,
    entity,
)
from memoria.hybrid import (
    normalize as normalize_signal,
)
from memoria.statistics import Proportion
from memoria.taxonomy import Claim

# --- surface features (deterministic heuristics) ------------------------------------------

_NEGATION = frozenset({"not", "no", "never", "none", "nobody", "nothing", "neither", "nor"})
_MONTHS = "january february march april may june july august september october november december"
_WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday"
_TEMPORAL = frozenset(
    {
        *_MONTHS.split(),
        *_WEEKDAYS.split(),
        *[
            "yesterday",
            "today",
            "tomorrow",
            "ago",
            "last",
            "next",
            "until",
            "since",
            "before",
            "after",
            "former",
            "formerly",
            "previously",
            "used",
            "was",
            "were",
            "lived",
            "had",
            "will",
            "now",
            "currently",
            "still",
            "anymore",
        ],
    }
)
_FUNCTION = frozenset(
    [
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "it",
        "he",
        "she",
        "they",
        "we",
        "i",
        "this",
        "that",
        "these",
        "those",
        "correction",
        "note",
    ]
)
_STOP = _FUNCTION | frozenset(
    [
        "is",
        "are",
        "be",
        "to",
        "of",
        "and",
        "or",
        "for",
        "with",
        "by",
        "as",
        "s",
        "t",
        "set",
        "correct",
    ]
)
_CAPITAL = re.compile(r"\b[A-Z][\w'-]*")
_YEAR = re.compile(r"^(1[89]|20)\d\d$")


@dataclass(frozen=True)
class Features:
    """What a text (and its claim, if any) states, as comparable feature strings."""

    claims: frozenset[str]  # "key=value"
    entities: frozenset[str]
    numbers: frozenset[str]
    temporal: frozenset[str]
    negated: bool
    tokens: frozenset[str]  # content words, for qualifier preservation

    def all(self) -> set[str]:
        out = {f"claim:{c}" for c in self.claims}
        out |= {f"entity:{x}" for x in self.entities}
        out |= {f"number:{x}" for x in self.numbers}
        out |= {f"temporal:{x}" for x in self.temporal}
        out |= {f"token:{x}" for x in self.tokens}
        if self.negated:
            out.add("negation")
        return out


def claim_string(key: str, value: str) -> str:
    return f"{key}={' '.join(normalize(value))}"


def features(text: str, claims: Iterable[str] = ()) -> Features:
    tokens = normalize(text)
    folded = text.casefold()
    return Features(
        claims=frozenset(claims),
        entities=frozenset(
            m.casefold() for m in _CAPITAL.findall(text) if m.casefold() not in _FUNCTION
        ),
        numbers=frozenset(t for t in tokens if any(ch.isdigit() for ch in t)),
        temporal=frozenset(t for t in tokens if t in _TEMPORAL or _YEAR.match(t)),
        negated=bool(_NEGATION & set(tokens)) or "n't" in folded,
        tokens=frozenset(t for t in tokens if t not in _STOP),
    )


def _entry_claims(e: Entry) -> list[str]:
    c = e.claim
    return [claim_string(c.key, c.value)] if c is not None and c.value is not None else []


def entry_features(e: Entry) -> Features:
    return features(e.version.content or "", _entry_claims(e))


# --- guards: when a merge would be unsafe --------------------------------------------------

Guard = Literal["claim_conflict", "numeric", "negation", "temporal", "entity"]
GUARDS: tuple[Guard, ...] = ("claim_conflict", "entity", "negation", "numeric", "temporal")


def violated(a: Entry, b: Entry, guards: Sequence[Guard]) -> tuple[Guard, ...]:
    """The guards a merge of ``a`` and ``b`` would violate (empty: safe to merge).
    Texts with identical canonical tokens state the same thing and violate none."""
    if normalize(a.version.content or "") == normalize(b.version.content or ""):
        return ()
    fa, fb = entry_features(a), entry_features(b)
    out: list[Guard] = []
    for g in guards:
        if g == "claim_conflict":
            ca, cb = a.claim, b.claim
            # Two structured claims merge only if they are the same claim: another key is
            # another fact (entity or attribute), another value or a retraction disputes it.
            bad = (
                ca is not None
                and cb is not None
                and (ca.key, normalize(ca.value or "")) != (cb.key, normalize(cb.value or ""))
            )
        elif g == "numeric":
            bad = fa.numbers != fb.numbers
        elif g == "negation":
            bad = fa.negated != fb.negated
        elif g == "temporal":
            bad = fa.temporal != fb.temporal
        else:
            bad = fa.entities != fb.entities
        if bad:
            out.append(g)
    return tuple(out)


# --- the temporal structure of claims ----------------------------------------------------------


class Period(Record):
    """One stretch of a key's timeline during which a value holds (by occurrence time).

    A later report does not rewrite an earlier period: a change closes it. A ``correct``
    replaces the period it follows retroactively (that period is ``corrected``). Two values
    reported for the same instant open parallel, ``contested`` periods. ``forget`` closes.
    """

    key: str
    value: str  # normalised tokens joined by spaces
    start: UTCDatetime
    end: UTCDatetime | None
    experiences: tuple[Digest, ...]
    contested: bool = False
    corrected: bool = False


@dataclass
class _Open:
    value: str
    start: datetime
    experiences: list[str]
    end: datetime | None = None
    contested: bool = False
    corrected: bool = False


def timeline(claims: Sequence[Claim], key: str) -> list[Period]:
    """The periods of ``key`` from its claims, ordered by occurrence (never ingestion)."""
    events = sorted(
        (c for c in claims if c.key == key),
        key=lambda c: (c.occurred_at, c.verb == "forget", normalize(c.value or ""), c.experience),
    )
    periods: list[_Open] = []
    current: list[_Open] = []  # periods open now (several when contested)

    def close(t: datetime) -> None:
        for q in current:
            q.end = t
        current.clear()

    for c in events:
        t = c.occurred_at
        if c.verb == "forget":
            close(t)
            continue
        value = " ".join(normalize(c.value or ""))
        same = next((q for q in current if q.value == value), None)
        if same is not None:  # a repeated report of a value that already holds
            same.experiences.append(c.experience)
            continue
        if c.verb == "correct" and current:  # retroactive: replaces what held
            for q in current:
                q.corrected = True
            begin = min(q.start for q in current)
            close(begin)
            new = _Open(value, begin, [c.experience])
        elif current and all(q.start == t for q in current):  # same instant, another value
            for q in current:
                q.contested = True
            new = _Open(value, t, [c.experience], contested=True)
        else:
            close(t)
            new = _Open(value, t, [c.experience])
        periods.append(new)
        current.append(new)
    return [
        Period(
            key=key,
            value=q.value,
            start=q.start,
            end=q.end if q.end is None or q.end > q.start else None,
            experiences=tuple(sorted(q.experiences)),
            contested=q.contested,
            corrected=q.corrected,
        )
        for q in periods
    ]


def _period_of(experience: str, periods: Sequence[Period]) -> int | None:
    return next((i for i, p in enumerate(periods) if experience in p.experiences), None)


# --- importance ------------------------------------------------------------------------------

ImportanceName = Literal["recency", "source", "repetition", "persistence", "contradiction",
                         "provenance"]  # fmt: skip
_IMPORTANCE_BOUNDS: dict[str, tuple[float, float] | None] = {
    "recency": (0.0, 1.0),
    "source": (0.0, 1.0),
    "repetition": None,
    "persistence": None,
    "contradiction": (0.0, 1.0),
    "provenance": (0.0, 1.0),
}


class ImportanceComponent(Record):
    name: ImportanceName
    weight: float = Field(gt=0, allow_inf_nan=False)
    normalization: Normalization


class ImportanceSpec(Record):
    """A decomposed importance model: named components with declared weights (sum 1),
    normalisation, and the threshold for promotion. Missing components contribute 0 and
    are recorded as missing. Not a universal score: every component stays inspectable."""

    components: tuple[ImportanceComponent, ...] = Field(min_length=1)
    threshold: float = Field(allow_inf_nan=False)
    half_life_days: float = Field(gt=0, allow_inf_nan=False)
    priors: tuple[tuple[str, float], ...] = ()  # source class -> declared prior

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        names = [c.name for c in self.components]
        if names != sorted(set(names)):
            raise ValueError("importance components must be unique and sorted")
        total = math.fsum(c.weight for c in self.components)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"importance weights must sum to 1, got {total!r}")
        for c in self.components:
            if c.normalization == "bounded-v1" and _IMPORTANCE_BOUNDS[c.name] is None:
                raise ValueError(f"{c.name} is unbounded; bounded-v1 does not apply")
        if list(self.priors) != sorted(set(self.priors)):
            raise ValueError("priors must be unique and sorted")
        return self


class ImportanceValue(Record):
    name: str
    raw: float | None
    normalized: float | None
    contribution: float
    missing: str | None = None


class ImportanceRecord(Record):
    version: Digest
    components: tuple[ImportanceValue, ...]
    total: float
    promoted: bool


def _source_class(source: str) -> str | None:
    cls, sep, ident = source.partition(":")
    return cls if sep and cls and ident else None


# --- policy ----------------------------------------------------------------------------------

Regime = Literal["none", "exact", "canonical", "claim", "temporal", "semantic"]
Abstraction = Literal["co_change", "entity", "timeline"]


class ConsolidationPolicy(Record):
    """Everything that determines a consolidation. Its digest is the policy identity.

    ``regime`` none produces no derived memory (the no-op). ``threshold`` is required for,
    and only for, the semantic regime. ``guards`` are the safety checks applied to every
    candidate merge. ``promote`` selects which L1 versions consolidate: all, those recorded
    within ``window_days`` (recency-driven), or those whose importance reaches the
    threshold. ``abstractions`` add L3/L4 memories, each needing ``min_support`` inputs.
    ``every_days`` is the consolidation frequency used by runs.
    """

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    regime: Regime
    threshold: float | None = Field(default=None, gt=0, le=1)
    guards: tuple[Guard, ...] = GUARDS
    promote: Literal["all", "recent", "important"] = "all"
    window_days: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    importance: ImportanceSpec | None = None
    abstractions: tuple[Abstraction, ...] = ()
    min_support: int = Field(default=2, ge=2)
    every_days: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if (self.regime == "semantic") != (self.threshold is not None):
            raise ValueError("a similarity threshold is required for, and only for, semantic")
        if list(self.guards) != sorted(set(self.guards)):
            raise ValueError("guards must be unique and sorted")
        if list(self.abstractions) != sorted(set(self.abstractions)):
            raise ValueError("abstractions must be unique and sorted")
        if (self.promote == "recent") != (self.window_days is not None):
            raise ValueError("window_days is required for, and only for, recency promotion")
        if (self.promote == "important") != (self.importance is not None):
            raise ValueError("an importance model is required for, and only for, it")
        if self.regime == "none" and (self.abstractions or self.promote != "all"):
            raise ValueError("the no-op regime takes no promotion filter or abstraction")
        if self.abstractions and self.regime not in ("claim", "temporal", "semantic"):
            raise ValueError("abstraction needs claim-bearing L2 facts: claim/temporal/semantic")
        return self

    @property
    def uses_vectors(self) -> bool:
        return self.regime == "semantic"

    def checkpoints(self, start: datetime, end: datetime) -> list[datetime]:
        """Consolidation times: start + i * every_days, up to ``end`` inclusive."""
        step = timedelta(days=self.every_days)
        out, t = [], start + step
        while t <= end:
            out.append(t)
            t += step
        return out


# --- records ------------------------------------------------------------------------------------


class MergeDecision(Record):
    """How one L1 version was grouped. ``blocked`` lists groups whose regime matched but a
    guard refused, with the guards: an unsafe merge detected and reported, not applied."""

    version: Digest
    group: Digest  # the group's founding version
    founded: bool
    similarity: float | None = None  # min cosine to the group's members (semantic)
    blocked: tuple[tuple[Digest, str], ...] = ()  # (group, guard), sorted


class LossReport(Record):
    """What a derived memory's content kept of its inputs' content, feature by feature.

    ``preserved``/``lost`` partition the input features; ``altered`` are input claims whose
    key the output restates with another value; ``conflicting`` input claims disagree among
    themselves (``conflicts_kept``: the output keeps both sides); ``unsupported`` are output
    features no input states. Lineage coverage says what remains reachable by reference.
    """

    memory: str
    inputs: int = Field(ge=1)
    preserved: tuple[str, ...]
    lost: tuple[str, ...]
    altered: tuple[str, ...]
    conflicting: tuple[str, ...]
    conflicts_kept: bool
    unsupported: tuple[str, ...]
    experience_coverage: Proportion
    source_coverage: Proportion

    def kind_counts(self, kind: str) -> tuple[int, int]:
        """(preserved, total) input features of one kind: claim, entity, number, ..."""
        kept = sum(1 for f in self.preserved if f.split(":", 1)[0] == kind)
        total = kept + sum(1 for f in self.lost if f.split(":", 1)[0] == kind)
        return kept, total


class Hierarchy(Record):
    """The derived memory of one consolidation: what was built, how, and at what cost."""

    policy: Digest
    at: UTCDatetime
    corpus: Digest  # the L1 corpus consolidated
    model: Digest | None  # embedder spec, if similarities were used
    memories: tuple[DerivedMemory, ...]  # by (level, memory_id)
    decisions: tuple[MergeDecision, ...]
    losses: tuple[LossReport, ...]  # one per derived memory, same order
    importance: tuple[ImportanceRecord, ...] = ()
    not_promoted: tuple[tuple[Digest, str], ...] = ()  # (version, reason): archived at L1

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        ids = [m.memory_id for m in self.memories]
        if len(set(ids)) != len(ids):
            raise ValueError("derived memory ids must be unique")
        if [(m.level, m.memory_id) for m in self.memories] != sorted(
            (m.level, m.memory_id) for m in self.memories
        ):
            raise ValueError("memories must be ordered by (level, memory_id)")
        by_id = {m.memory_id: m for m in self.memories}
        for m in self.memories:
            if m.policy != self.policy or m.recorded_at != self.at or m.model != self.model:
                raise ValueError(f"{m.memory_id}: not produced by this consolidation")
            if m.level is Level.L2:
                if tuple(sorted(m.parents)) != m.versions:
                    raise ValueError(f"{m.memory_id}: an L2 memory's parents are its versions")
                continue
            for parent in m.parents:
                p = by_id.get(parent)
                if p is None or p.level >= m.level:
                    raise ValueError(f"{m.memory_id}: parent {parent} is not a lower level")
                if not set(p.versions) <= set(m.versions):
                    raise ValueError(f"{m.memory_id}: does not cover its parent's versions")
        if [r.memory for r in self.losses] != ids:
            raise ValueError("one loss report per derived memory, in order")
        for m, r in zip(self.memories, self.losses, strict=True):
            if m.lost != len(r.lost) + len(r.altered):
                raise ValueError(f"{m.memory_id}: loss count disagrees with its report")
        return self

    def memory(self, memory_id: str) -> DerivedMemory:
        return next(m for m in self.memories if m.memory_id == memory_id)

    @property
    def blocked_merges(self) -> int:
        return sum(len(d.blocked) for d in self.decisions)


# --- consolidation ----------------------------------------------------------------------------


def _id(level: Level, policy: str, parents: Iterable[str], rule: str) -> str:
    h = hashlib.sha256(canonical_json([policy, rule, sorted(parents)]).encode()).hexdigest()
    return f"{level.value}:{h[:20]}"


def _start(e: Entry) -> datetime:
    assert e.interval is not None  # promoted entries are held
    return e.interval[0]


def _importance(
    held: Sequence[Entry], spec: ImportanceSpec, at: datetime, corpus: Corpus
) -> dict[str, ImportanceRecord]:
    priors = dict(spec.priors)
    counts: dict[object, int] = {}

    def repetition_key(e: Entry) -> object:
        return tuple(_entry_claims(e)) or normalize(e.version.content or "")

    for e in held:
        counts[repetition_key(e)] = counts.get(repetition_key(e), 0) + 1
    periods = {c.key: timeline(corpus.claims, c.key) for e in held if (c := e.claim)}
    raw: dict[str, list[tuple[float | None, str | None]]] = {}
    for comp in spec.components:
        column: list[tuple[float | None, str | None]] = []
        for e in held:
            if comp.name == "recency":
                age = (at - e.version.recorded_at) / timedelta(days=1)
                column.append((quantize(0.5 ** (age / spec.half_life_days)), None))
            elif comp.name == "source":
                classes = [_source_class(s.source) for s in e.sources]
                if not classes or None in classes:
                    column.append((None, "unstructured_source"))
                elif any(c not in priors for c in classes):
                    column.append((None, "undeclared_source_class"))
                else:
                    column.append((min(priors[c] for c in classes if c is not None), None))
            elif comp.name == "repetition":
                column.append((float(counts[repetition_key(e)]), None))
            elif comp.name == "persistence":
                c = e.claim
                i = _period_of(c.experience, periods[c.key]) if c is not None else None
                if c is None or i is None:
                    column.append((None, "no_claim"))
                else:
                    p = periods[c.key][i]
                    end = min(p.end or at, at)
                    column.append((quantize(max(0.0, (end - p.start) / timedelta(days=1))), None))
            elif comp.name == "contradiction":
                status = conflict(e, corpus.claims, at)
                if status is None:
                    column.append((None, "no_claim"))
                else:
                    column.append((1.0 if status in CONFLICTED else 0.0, None))
            else:  # provenance
                cited = len(e.version.derived_from)
                structured = sum(1 for s in e.sources if _source_class(s.source))
                column.append((quantize(structured / cited), None))
        raw[comp.name] = column
    normalized = {
        comp.name: normalize_signal(
            comp.normalization, [r for r, _ in raw[comp.name]], _IMPORTANCE_BOUNDS[comp.name]
        )
        for comp in spec.components
    }
    out = {}
    for i, e in enumerate(held):
        values = []
        for comp in spec.components:
            r, why = raw[comp.name][i]
            n = normalized[comp.name][i]
            values.append(
                ImportanceValue(
                    name=comp.name,
                    raw=r,
                    normalized=n,
                    contribution=0.0 if n is None else quantize(comp.weight * n),
                    missing=why,
                )
            )
        total = quantize(math.fsum(v.contribution for v in values))
        out[e.digest] = ImportanceRecord(
            version=e.digest,
            components=tuple(values),
            total=total,
            promoted=total >= spec.threshold,
        )
    return out


def _group(
    entries: Sequence[Entry],
    policy: ConsolidationPolicy,
    corpus: Corpus,
    engine: Engine | None,
) -> tuple[list[list[Entry]], list[MergeDecision], dict[str, str]]:
    """Complete-linkage grouping: an entry joins the first group it matches under the
    regime *with every member* and that no guard blocks; otherwise it founds a group."""
    periods: dict[str, list[Period]] = {}
    excluded: dict[str, str] = {}

    def key(e: Entry) -> object:
        c = e.claim
        if policy.regime == "exact":
            return e.version.content
        if policy.regime == "canonical":
            return normalize(e.version.content or "")
        if c is None or c.value is None:
            return ("unstructured", e.digest)  # claim regimes cannot compare free text
        if policy.regime == "claim":
            return (c.key, normalize(c.value))
        if c.key not in periods:
            periods[c.key] = timeline(corpus.claims, c.key)
        return (c.key, _period_of(c.experience, periods[c.key]))

    usable = []
    for e in entries:
        c = e.claim
        if policy.regime == "temporal" and c is not None and c.value is not None:
            if c.key not in periods:
                periods[c.key] = timeline(corpus.claims, c.key)
            i = _period_of(c.experience, periods[c.key])
            if i is not None and periods[c.key][i].corrected:
                excluded[e.digest] = "corrected"  # the evidence was replaced retroactively
                continue
        usable.append(e)

    groups: list[list[Entry]] = []
    decisions = []
    for e in usable:
        joined = None
        similarity = None
        blocked: list[tuple[str, str]] = []
        for g in groups:
            if policy.regime == "semantic":
                assert engine is not None
                assert policy.threshold is not None
                sims = [
                    engine.similarity(e.version.content or "", m.version.content or "") for m in g
                ]
                matches = min(sims) >= policy.threshold
            else:
                matches = all(key(e) == key(m) for m in g)
                sims = []
            if not matches:
                continue
            bad = sorted({v for m in g for v in violated(e, m, policy.guards)})
            if bad:
                blocked += [(g[0].digest, v) for v in bad]
                continue
            joined = g
            similarity = min(sims) if sims else None
            break
        if joined is None:
            groups.append([e])
            joined = groups[-1]
        else:
            joined.append(e)
        decisions.append(
            MergeDecision(
                version=e.digest,
                group=joined[0].digest,
                founded=joined[0] is e,
                similarity=similarity,
                blocked=tuple(sorted(blocked)),
            )
        )
    return groups, decisions, excluded


_TEMPLATE = frozenset(
    [
        "contested",
        "inferred",
        "pattern",
        "changed",
        "together",
        "same",
        "day",
        "times",
        "stable",
        "changing",
        "recurring",
        "of",
    ]
)


def _loss(
    memory_id: str,
    inputs: Sequence[Entry],
    content: str,
    out_claims: Sequence[str],
    derived_from: Sequence[str],
    dates: Iterable[datetime],
) -> LossReport:
    per_input = [entry_features(e) for e in inputs]
    wanted = set().union(*(f.all() for f in per_input))
    out = features(content, out_claims)
    have = out.all()
    date_parts = {p for d in dates for p in d.date().isoformat().split("-")}
    date_parts |= {p.lstrip("0") for p in date_parts}
    lost = sorted(wanted - have)
    in_claims = [c for f in per_input for c in f.claims]
    out_keys = {c.split("=", 1)[0]: c for c in out.claims}
    altered = sorted(
        {c for c in in_claims if c.split("=", 1)[0] in out_keys and c not in out.claims}
    )
    by_key: dict[str, set[str]] = {}
    for c in in_claims:
        by_key.setdefault(c.split("=", 1)[0], set()).add(c)
    conflicting = sorted(c for cs in by_key.values() if len(cs) > 1 for c in cs)

    def supported(f: str) -> bool:
        kind, _, x = f.partition(":")
        # Content words and rendering vocabulary are not facts; rendered dates restate
        # the memory's own validity interval (metadata), not new content.
        return (
            kind == "token"
            or x in _TEMPLATE
            or (kind in ("number", "temporal") and x in date_parts)
        )

    unsupported = sorted(f for f in have - wanted if not supported(f))
    experiences = {s.digest for e in inputs for s in e.sources}
    sources = {s.source for e in inputs for s in e.sources}
    covered = experiences & set(derived_from)
    return LossReport(
        memory=memory_id,
        inputs=len(inputs),
        preserved=tuple(sorted(wanted & have)),
        lost=tuple(f for f in lost if f not in altered and f.removeprefix("claim:") not in altered),
        altered=tuple(altered),
        conflicting=tuple(conflicting),
        conflicts_kept=all(c in out.claims for c in conflicting),
        unsupported=tuple(unsupported),
        experience_coverage=Proportion.of(len(covered), len(experiences), 0.95),
        source_coverage=Proportion.of(
            len({s.source for e in inputs for s in e.sources if s.digest in covered}),
            len(sources),
            0.95,
        ),
    )


def consolidate(
    corpus: Corpus,
    policy: ConsolidationPolicy,
    at: datetime,
    embedder: Embedder | None = None,
    memo: Engine | None = None,
) -> Hierarchy:
    """Consolidate every L1 version held at ``at`` (the corpus must be known at ``at``).

    Deterministic: the result is a function of the corpus, the policy and the embedder.
    Retracted versions (superseded, forgotten) are never consolidated.
    """
    if corpus.known_at != at:
        raise ValueError("consolidate the corpus known at the consolidation time")
    if policy.uses_vectors and embedder is None:
        raise ValueError("a semantic consolidation needs its embedder; none was given")
    model = embedder.spec.digest if policy.uses_vectors and embedder is not None else None
    l1 = [e for e in corpus.entries if e.fate == "held" and e.level is Level.L1]
    malformed = {e.digest: "malformed_provenance" for e in l1 if e.missing}
    held = [e for e in l1 if not e.missing]
    held.sort(key=lambda e: (_start(e), e.version.memory_id, e.digest))
    pid = policy.digest
    if policy.regime == "none":
        return Hierarchy(policy=pid, at=at, corpus=corpus.digest, model=None, memories=(),
                         decisions=(), losses=())  # fmt: skip

    not_promoted: dict[str, str] = dict(malformed)
    importance: dict[str, ImportanceRecord] = {}
    if policy.promote == "recent":
        assert policy.window_days is not None
        floor = at - timedelta(days=policy.window_days)
        not_promoted |= {
            e.digest: "older_than_window" for e in held if e.version.recorded_at < floor
        }
    elif policy.promote == "important":
        assert policy.importance is not None
        importance = _importance(held, policy.importance, at, corpus)
        not_promoted |= {d: "below_importance" for d, r in importance.items() if not r.promoted}
    promoted = [e for e in held if e.digest not in not_promoted]
    engine = None
    if policy.uses_vectors:  # share the caller's embedding memos (same embedder) if given
        engine = Engine(corpus, embedder)
        if memo is not None and memo.embedder is embedder:
            engine = Engine(corpus, embedder, None, memo.vectors, memo.similarities)
    groups, decisions, excluded = _group(promoted, policy, corpus, engine)
    not_promoted |= excluded

    by_digest = {e.digest: e for e in corpus.entries}
    periods: dict[str, list[Period]] = {}
    memories: list[DerivedMemory] = []
    losses: list[LossReport] = []

    def add(
        m: DerivedMemory,
        inputs: Sequence[Entry],
        out_claims: Sequence[str],
        parents: Sequence[DerivedMemory] = (),
    ) -> DerivedMemory:
        # Dates a rendering may restate: its own and its parents' interval bounds.
        dates = [m.valid_from, *([m.valid_to] if m.valid_to else [])]
        dates += [_start(e) for e in inputs]
        dates += [d for p in parents for d in (p.valid_from, p.valid_to) if d is not None]
        r = _loss(m.memory_id, inputs, m.content, out_claims, m.derived_from, dates)
        m = m.model_copy(update={"lost": len(r.lost) + len(r.altered)})
        memories.append(DerivedMemory.model_validate(m.model_dump()))
        losses.append(r)
        return memories[-1]

    l2: list[tuple[DerivedMemory, list[Entry]]] = []
    for g in groups:
        # A claim only if every member asserts the same one; the content then states just
        # that claim (its rendering), else the earliest member's text (the representative).
        claims = {tuple(_entry_claims(e)) for e in g}
        claim = g[0].claim if len(claims) == 1 and claims != {()} else None
        key = claim.key if claim is not None else None
        value = " ".join(normalize(claim.value or "")) if claim is not None else None
        content = (
            f"{claim.key} = {claim.value}" if claim is not None else g[0].version.content or ""
        )
        start = min(_start(e) for e in g)
        ends = [e.interval[1] if e.interval else None for e in g]
        end = None if None in ends else max(x for x in ends if x is not None)
        if policy.regime == "temporal" and claim is not None:
            if claim.key not in periods:
                periods[claim.key] = timeline(corpus.claims, claim.key)
            i = _period_of(claim.experience, periods[claim.key])
            if i is not None:
                start, end = periods[claim.key][i].start, periods[claim.key][i].end
        versions = sorted(e.digest for e in g)
        experiences = sorted({s.digest for e in g for s in e.sources})
        rivals = sorted(
            {
                v
                for e in g
                for v in _conflicting(e, corpus.entries)
                if _within(by_digest[v], start, end)
            }
        )
        disputed = len({s.digest for v in rivals for s in by_digest[v].sources})
        mid = _id(Level.L2, pid, versions, policy.regime)
        m = DerivedMemory(
            memory_id=mid,
            level=Level.L2,
            status=EpistemicStatus.DERIVED,
            operation="merge" if len(g) > 1 else "promote",
            rule=policy.regime,
            policy=pid,
            content=content,
            key=key,
            value=value,
            valid_from=start,
            valid_to=end if end is None or end > start else None,
            recorded_at=at,
            parents=tuple(versions),
            versions=tuple(versions),
            derived_from=tuple(experiences),
            conflicts=tuple(rivals),
            support=len(experiences),
            disputed=disputed,
            model=model,
            lost=0,
        )
        out_claims = [claim_string(key, value)] if key is not None and value is not None else []
        l2.append((add(m, g, out_claims), g))

    facts = [(m, g) for m, g in l2 if m.key is not None]
    if "entity" in policy.abstractions:
        by_entity: dict[str, list[tuple[DerivedMemory, list[Entry]]]] = {}
        for m, g in facts:
            if (ent := entity(m.key or "")) is not None:
                by_entity.setdefault(ent, []).append((m, g))
        for ent, members in sorted(by_entity.items()):
            current = [(m, g) for m, g in members if _holds(m, at)]
            if len(current) < policy.min_support:
                continue
            attrs: dict[str, list[str]] = {}
            for m, _ in current:
                attrs.setdefault((m.key or "").split(".", 1)[1], []).append(m.value or "")
            content = f"{ent}: " + "; ".join(
                f"{a} = {' | '.join(sorted(set(vs)))}"
                + (" (contested)" if len(set(vs)) > 1 else "")
                for a, vs in sorted(attrs.items())
            )
            inputs = [e for _, g in current for e in g]
            m = DerivedMemory(
                memory_id=_id(Level.L3, pid, [m.memory_id for m, _ in current], "entity"),
                level=Level.L3,
                status=EpistemicStatus.ABSTRACTED,
                operation="abstract_entity",
                rule="entity-current-profile",
                policy=pid,
                content=content,
                valid_from=max(m.valid_from for m, _ in current),
                valid_to=min((m.valid_to for m, _ in current if m.valid_to), default=None),
                recorded_at=at,
                parents=tuple(sorted(m.memory_id for m, _ in current)),
                versions=tuple(sorted({v for m, _ in current for v in m.versions})),
                derived_from=tuple(sorted({d for m, _ in current for d in m.derived_from})),
                excluded=tuple(
                    sorted((m.memory_id, "not_current") for m, _ in members if not _holds(m, at))
                ),
                conflicts=tuple(sorted({c for m, _ in current for c in m.conflicts})),
                support=len({d for m, _ in current for d in m.derived_from}),
                disputed=sum(m.disputed for m, _ in current),
                model=model,
                lost=0,
            )
            out_claims = [claim_string(m2.key or "", m2.value or "") for m2, _ in current]
            add(m, inputs, out_claims)
    if "timeline" in policy.abstractions:
        by_key: dict[str, list[tuple[DerivedMemory, list[Entry]]]] = {}
        for m, g in facts:
            by_key.setdefault(m.key or "", []).append((m, g))
        for key, members in sorted(by_key.items()):
            if len(members) < policy.min_support:
                continue
            members.sort(key=lambda x: (x[0].valid_from, x[0].memory_id))
            values = [m.value for m, _ in members]
            starts = [m.valid_from for m, _ in members]
            label = (
                "contested"
                if len(set(starts)) < len(starts)
                else "stable"
                if len(set(values)) == 1
                else "recurring"
                if any(values[i] in values[: i - 1] for i in range(2, len(values)))
                else "changing"
            )
            steps = " -> ".join(
                f"{m.value} ({m.valid_from.date().isoformat()}"
                f"..{m.valid_to.date().isoformat() if m.valid_to else ''})"
                for m, _ in members
            )
            m = DerivedMemory(
                memory_id=_id(Level.L4, pid, [m.memory_id for m, _ in members], "timeline"),
                level=Level.L4,
                status=EpistemicStatus.ABSTRACTED,
                operation="abstract_timeline",
                rule=f"timeline-{label}",
                policy=pid,
                content=f"{key}: {steps} [{label}]",
                valid_from=members[0][0].valid_from,
                valid_to=None
                if any(m.valid_to is None for m, _ in members)
                else max(m.valid_to for m, _ in members if m.valid_to),
                recorded_at=at,
                parents=tuple(sorted(m.memory_id for m, _ in members)),
                versions=tuple(sorted({v for m, _ in members for v in m.versions})),
                derived_from=tuple(sorted({d for m, _ in members for d in m.derived_from})),
                conflicts=tuple(sorted({c for m, _ in members for c in m.conflicts})),
                support=len({d for m, _ in members for d in m.derived_from}),
                disputed=sum(m.disputed for m, _ in members),
                model=model,
                lost=0,
            )
            out_claims = [claim_string(key, m2.value or "") for m2, _ in members]
            add(m, [e for _, g in members for e in g], out_claims, [m2 for m2, _ in members])
    if "co_change" in policy.abstractions:
        changes: dict[str, dict[str, set[datetime]]] = {}
        for m, _ in facts:
            key = m.key or ""
            ent = entity(key)
            if ent is not None:
                changes.setdefault(ent, {}).setdefault(key, set()).add(
                    m.valid_from.replace(hour=0, minute=0, second=0, microsecond=0)
                )
        for _, keys in sorted(changes.items()):
            names = sorted(keys)
            for i, k1 in enumerate(names):
                for k2 in names[i + 1 :]:
                    together = sorted(keys[k1] & keys[k2])
                    if len(together) < policy.min_support:
                        continue
                    members = [(m, g) for m, g in facts if m.key in (k1, k2)]
                    m = DerivedMemory(
                        memory_id=_id(Level.L4, pid, [k1, k2], "co_change"),
                        level=Level.L4,
                        status=EpistemicStatus.INFERRED,
                        operation="infer_co_change",
                        rule="same-day-changes",
                        policy=pid,
                        content=f"inferred pattern: {k1} and {k2} changed on the same day "
                        f"{len(together)} times",
                        valid_from=together[0],
                        recorded_at=at,
                        parents=tuple(sorted(m.memory_id for m, _ in members)),
                        versions=tuple(sorted({v for m, _ in members for v in m.versions})),
                        derived_from=tuple(sorted({d for m, _ in members for d in m.derived_from})),
                        support=len({d for m, _ in members for d in m.derived_from}),
                        disputed=0,
                        model=model,
                        lost=0,
                    )
                    add(m, [e for _, g in members for e in g], [])

    order = sorted(range(len(memories)), key=lambda i: (memories[i].level, memories[i].memory_id))
    return Hierarchy(
        policy=pid,
        at=at,
        corpus=corpus.digest,
        model=model,
        memories=tuple(memories[i] for i in order),
        decisions=tuple(decisions),
        losses=tuple(losses[i] for i in order),
        importance=tuple(importance[e.digest] for e in held if e.digest in importance),
        not_promoted=tuple(sorted(not_promoted.items())),
    )


def _conflicting(e: Entry, entries: Sequence[Entry]) -> list[str]:
    c = e.claim
    if c is None or c.value is None:
        return []
    return [
        x.digest
        for x in entries
        if x.level is Level.L1
        and x.claim is not None
        and x.claim.key == c.key
        and x.claim.value is not None
        and normalize(x.claim.value) != normalize(c.value)
    ]


def _within(e: Entry, start: datetime, end: datetime | None) -> bool:
    """Whether an entry's claim occurred inside [start, end)."""
    c = e.claim
    return c is not None and start <= c.occurred_at and (end is None or c.occurred_at < end)


def _holds(m: DerivedMemory, at: datetime) -> bool:
    return m.valid_from <= at and (m.valid_to is None or at < m.valid_to)


# --- lineage traversal, invalidation, replay -----------------------------------------------------


def lineage(h: Hierarchy, memory_id: str) -> list[str]:
    """Downward provenance: the memory, every derived ancestor, then its L1 versions."""
    m = h.memory(memory_id)
    out = [memory_id]
    frontier = [p for p in m.parents if not p.startswith("sha256:")]
    while frontier:
        p = frontier.pop(0)
        if p not in out:
            out.append(p)
            frontier += [q for q in h.memory(p).parents if not q.startswith("sha256:")]
    return out + list(m.versions)


def invalidated(h: Hierarchy, now: Corpus) -> dict[str, str]:
    """Derived memories of ``h`` whose evidence changed by ``now.known_at``, with a reason:
    a covered version was retracted, or a later-recorded claim gives another value inside
    the memory's interval. Anything built on an invalidated memory is invalidated too."""
    if now.known_at < h.at:
        raise ValueError("invalidation is judged at or after the consolidation time")
    by_digest = {e.digest: e for e in now.entries}
    new = [c for c in now.claims if c.recorded_at > h.at and c.value is not None]
    out: dict[str, str] = {}
    for m in h.memories:
        if any(by_digest[v].fate != "held" for v in m.versions if v in by_digest):
            out[m.memory_id] = "evidence_retracted"
        elif m.key is not None and any(
            c.key == m.key
            and " ".join(normalize(c.value or "")) != m.value
            and m.valid_from <= c.occurred_at
            and (m.valid_to is None or c.occurred_at < m.valid_to)
            for c in new
        ):
            out[m.memory_id] = "new_conflicting_evidence"
        elif any(p in out for p in m.parents):
            out[m.memory_id] = "parent_invalidated"
    return out


def replay(
    h: Hierarchy, corpus: Corpus, policy: ConsolidationPolicy, embedder: Embedder | None = None
) -> bool:
    """Whether consolidating ``corpus`` again reproduces ``h`` exactly (byte for byte)."""
    if policy.digest != h.policy or corpus.digest != h.corpus:
        raise ValueError("replay needs the policy and corpus the hierarchy names")
    return consolidate(corpus, policy, h.at, embedder) == h
