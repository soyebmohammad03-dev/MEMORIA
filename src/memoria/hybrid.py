"""Hybrid, explainable retrieval: candidate generation, filtering, signals, policies, traces.

Phase 6 turns retrieval into a sequence of recorded stages, each with its own contract:

.. code-block:: text

    corpus (every content-bearing version known at known_at)
      -> candidate generation   lexical (BM25) | semantic (vector index) | metadata (key)
      -> candidate union        merged by version digest; each generator's rank and score kept
      -> hard filtering         typed exclusion reasons; an excluded memory is never scored
      -> feature extraction     one SignalValue per (candidate, signal): raw value or missing
      -> normalisation          declared, versioned strategy per signal, over the survivors
      -> scoring                weighted sum or reciprocal rank fusion: relevance = sum(parts)
      -> diversity reranking    optional MMR: final = relevance - redundancy penalty
      -> contradiction exposure neutral | penalize (a signal) | surface | paired
      -> explanation            reason codes and text derived from the recorded numbers

A :class:`RetrievalPolicy` declares every choice above and is content-addressed: its
digest is the policy identity. :meth:`Engine.retrieve` returns a :class:`HybridTrace`
whose validator re-derives normalisation, every contribution, relevance, penalties, final
scores, the order, exposure placements, selection and each explanation from the recorded
raw values, so an inconsistent trace (or explanation) cannot be constructed.

Relevance is a retrieval-policy score. It is not truth, confidence or memory quality:
source priors are declared metadata, recency is not correctness, and temporal
compatibility says only whether a memory's validity interval contains the query time.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from memoria.comparison import normalize as normalize_value
from memoria.core import (
    DerivedMemory,
    Digest,
    EpistemicStatus,
    Experience,
    ExtensibleRecord,
    Inputs,
    Level,
    MemoryState,
    MemoryVersion,
    Operation,
    Query,
    Record,
    Response,
    Scalar,
    SignalEvidence,
    UTCDatetime,
    beliefs,
    quantize,
    state_as_of,
)
from memoria.embeddings import Embedder, Vector, cosine, embed, float32
from memoria.formation import parse_statement
from memoria.retrieval import BM25, EXTRACTIVE, tokenize
from memoria.semantic import IndexCompatibilityError, SemanticIndex
from memoria.store import MemoryLog
from memoria.taxonomy import Claim, Relation, relate


class PolicyError(ValueError):
    """A policy is invalid, or cannot run with the resources it was given."""


class SignalError(RuntimeError):
    """A signal violated its contract (non-finite or out-of-range value), or a signal the
    policy declares ``on_missing="fail"`` was unavailable."""


class GeneratorError(RuntimeError):
    """A candidate generator could not run. Recorded or raised, never silently skipped."""


class ScoreError(ArithmeticError):
    """A normalisation or score is not a finite number."""


# --- the corpus: what can be retrieved, with its temporal and structural metadata ----------


class TemporalStatus(StrEnum):
    """A version's relation to the query's valid time, from the bitemporal history."""

    VALID = "valid"  # its effective interval contains valid_at
    EXPIRED = "expired"  # its effective interval ended at or before valid_at
    FUTURE = "future"  # its effective interval starts after valid_at
    SUPERSEDED = "superseded"  # removed from its chain's timeline by a later CORRECT
    FORGOTTEN = "forgotten"  # its chain ends in a FORGET tombstone


class ConflictStatus(StrEnum):
    """A structured claim's relation to the other known claims on its key (Phase 4
    relations). Exposed for retrieval; resolving conflicts is Phase 11."""

    UNDISPUTED = "undisputed"  # no known claim disputes or repeats it
    SUPPORTED = "supported"  # another claim asserts the same value
    SUPERSEDED = "superseded"  # a later claim (occurring by valid_at) changed the value
    CONTRADICTED = "contradicted"  # another claim asserts a different value at the same instant
    CORRECTED = "corrected"  # a later ``correct`` replaced it (corrections are retroactive)
    FORGOTTEN = "forgotten"  # a later ``forget`` (occurring by valid_at) retracted the key
    RETRACTION = "retraction"  # the claim is itself a ``forget``


# Statuses that mean another known claim disputes this one.
CONFLICTED = frozenset(
    {
        ConflictStatus.SUPERSEDED,
        ConflictStatus.CONTRADICTED,
        ConflictStatus.CORRECTED,
        ConflictStatus.FORGOTTEN,
    }
)


class Exclusion(StrEnum):
    """Why a hard filter removed a candidate."""

    MALFORMED_PROVENANCE = "malformed_provenance"  # cites an experience the corpus lacks
    EXPIRED = "expired"
    FUTURE = "future"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"
    STALE = "stale"  # a derived memory whose supporting evidence changed after it was built


MemoryKind = Literal["fact", "note"]  # carries one structured claim / free text


@dataclass(frozen=True)
class Entry:
    """One retrievable memory, with everything filters and signals read.

    A log version is L1 and observed; a consolidated memory carries its own level and
    epistemic status (never observed), and ``stale`` when evidence it rests on changed
    after it was built.
    """

    version: MemoryVersion | DerivedMemory
    fate: Literal["held", "superseded", "forgotten"]
    interval: tuple[datetime, datetime | None] | None  # effective validity, if held
    claim: Claim | None
    claim_issue: Literal["unstructured", "ambiguous"] | None
    sources: tuple[Experience, ...]  # resolved cited experiences, by digest
    missing: int  # cited experiences absent from the corpus
    level: Level = Level.L1
    status: EpistemicStatus = EpistemicStatus.OBSERVED
    stale: bool = False

    @property
    def digest(self) -> str:
        return self.version.digest

    @property
    def kind(self) -> MemoryKind:
        return "fact" if self.claim is not None and self.claim.value is not None else "note"

    def temporal(self, valid_at: datetime) -> TemporalStatus:
        if self.fate == "forgotten":
            return TemporalStatus.FORGOTTEN
        if self.fate == "superseded" or self.interval is None:
            return TemporalStatus.SUPERSEDED
        start, end = self.interval
        if valid_at < start:
            return TemporalStatus.FUTURE
        if end is not None and end <= valid_at:
            return TemporalStatus.EXPIRED
        return TemporalStatus.VALID


def _claim_of(
    version: MemoryVersion, cited: Sequence[Experience]
) -> tuple[Claim | None, Literal["unstructured", "ambiguous"] | None]:
    found: dict[tuple[str, str, str | None], Claim] = {}
    for e in cited:
        s = parse_statement(e.content)
        if s is None:
            continue
        found.setdefault(
            (s.verb, s.key, s.value),
            Claim(
                experience=e.digest,
                source=e.source,
                verb=s.verb,  # type: ignore[arg-type]
                key=s.key,
                value=s.value,
                occurred_at=e.occurred_at,
                recorded_at=version.recorded_at,
                introduced=False,
            ),
        )
    if not found:
        return None, "unstructured"
    if len(found) > 1:
        return None, "ambiguous"  # contradictory metadata: no single claim to rely on
    return next(iter(found.values())), None


class CorpusIdentity(ExtensibleRecord):
    """What a corpus contains. Its digest identifies the retrieval universe."""

    _evolved = frozenset({"derived", "stale"})

    known_at: UTCDatetime
    versions: tuple[Digest, ...]  # every version recorded by known_at, sorted
    experiences: tuple[Digest, ...]  # cited experiences that are available, sorted
    derived: tuple[Digest, ...] = ()  # consolidated memories added, sorted
    stale: tuple[Digest, ...] = ()  # those among them that are invalidated, sorted


@dataclass(frozen=True)
class Corpus:
    """Everything retrievable at record time ``known_at``: each non-tombstone version of
    every memory, with its fate in its chain, effective interval and structured claim.

    Unlike a :class:`~memoria.core.MemoryState`, a corpus keeps versions that are expired,
    not yet valid, corrected away or forgotten, so filters can exclude them *with a
    reason* instead of their absence going unexplained.
    """

    known_at: datetime
    entries: tuple[Entry, ...]  # ordered by (memory_id, version)
    claims: tuple[Claim, ...]  # every claim cited by a known version, including tombstones
    present: MemoryState  # state at (known_at, known_at): what a semantic index covers
    identity: CorpusIdentity
    experiences: Mapping[str, Experience] = field(default_factory=dict, repr=False)

    @property
    def digest(self) -> str:
        return self.identity.digest

    @classmethod
    def from_log(cls, log: MemoryLog, known_at: datetime) -> Corpus:
        return cls.from_versions(log.versions(), log.experiences(), known_at)

    @classmethod
    def from_versions(
        cls,
        versions: Iterable[MemoryVersion],
        experiences: Iterable[Experience],
        known_at: datetime,
    ) -> Corpus:
        known = [v for v in versions if v.recorded_at <= known_at]
        by_digest = {e.digest: e for e in experiences}
        chains: dict[str, list[MemoryVersion]] = {}
        for v in known:
            chains.setdefault(v.memory_id, []).append(v)
        entries: list[Entry] = []
        claims: dict[str, Claim] = {}
        for memory_id in sorted(chains):
            chain = sorted(chains[memory_id], key=lambda v: v.version)
            timeline = {b.version.digest: b for b in beliefs(chain)}  # validates the chain
            forgotten = chain[-1].operation is Operation.FORGET
            for v in chain:
                cited = [by_digest[d] for d in v.derived_from if d in by_digest]
                claim, issue = _claim_of(v, cited)
                if claim is not None:
                    claims.setdefault(claim.experience, claim)
                if v.operation is Operation.FORGET:
                    continue
                belief = timeline.get(v.digest)
                entries.append(
                    Entry(
                        version=v,
                        fate="forgotten"
                        if forgotten
                        else ("held" if belief is not None else "superseded"),
                        interval=(belief.valid_from, belief.valid_to) if belief else None,
                        claim=claim,
                        claim_issue=issue,
                        sources=tuple(sorted(cited, key=lambda e: e.digest)),
                        missing=len(v.derived_from) - len(cited),
                    )
                )
        return cls(
            known_at=known_at,
            entries=tuple(entries),
            claims=tuple(sorted(claims.values(), key=lambda c: c.experience)),
            present=state_as_of(known, valid_at=known_at, known_at=known_at),
            identity=CorpusIdentity(
                known_at=known_at,
                versions=tuple(sorted(v.digest for v in known)),
                experiences=tuple(
                    sorted({d for v in known for d in v.derived_from if d in by_digest})
                ),
            ),
            experiences=by_digest,
        )

    def with_derived(self, memories: Sequence[DerivedMemory], stale: Iterable[str]) -> Corpus:
        """This corpus plus consolidated memories, each marked with its level and status.

        A derived memory's claim is read from its own evidence: the latest-occurring
        evidence claim asserting its key and value (never from its rendered content).
        """
        stale = set(stale)
        added = []
        for m in memories:
            if m.recorded_at > self.known_at:
                raise ValueError(f"{m.memory_id} was consolidated after known_at")
            cited = [self.experiences[d] for d in m.derived_from if d in self.experiences]
            claim = None
            if m.key is not None:
                matching = [
                    c
                    for c in self.claims
                    if c.experience in m.derived_from
                    and c.key == m.key
                    and c.value is not None
                    and normalize_value(c.value) == normalize_value(m.value or "")
                ]
                claim = max(matching, key=lambda c: (c.occurred_at, c.experience), default=None)
            added.append(
                Entry(
                    version=m,
                    fate="held",
                    interval=(m.valid_from, m.valid_to),
                    claim=claim,
                    claim_issue=None if claim is not None else "unstructured",
                    sources=tuple(sorted(cited, key=lambda e: e.digest)),
                    missing=len(m.derived_from) - len(cited),
                    level=m.level,
                    status=m.status,
                    stale=m.memory_id in stale,
                )
            )
        entries = sorted(
            (*self.entries, *added),
            key=lambda e: (e.version.memory_id, e.version.version, e.digest),
        )
        return Corpus(
            known_at=self.known_at,
            entries=tuple(entries),
            claims=self.claims,
            present=self.present,
            identity=self.identity.model_copy(
                update={
                    "derived": tuple(sorted(m.digest for m in memories)),
                    "stale": tuple(sorted(m.digest for m in memories if m.memory_id in stale)),
                }
            ),
            experiences=self.experiences,
        )


def conflict(entry: Entry, claims: Sequence[Claim], valid_at: datetime) -> ConflictStatus | None:
    """The entry's claim against every other known claim on its key, in priority order
    forgotten > corrected > contradicted > superseded > supported > undisputed.
    ``None`` when the entry carries no claim (the signal is then unavailable)."""
    c = entry.claim
    if c is None:
        return None
    if c.verb == "forget":
        return ConflictStatus.RETRACTION
    peers = [p for p in claims if p.key == c.key and p.experience != c.experience]
    relations = [(p, relate(c, p)) for p in peers]
    if any(
        r is Relation.RETRACTION and c.occurred_at <= p.occurred_at <= valid_at
        for p, r in relations
    ):
        return ConflictStatus.FORGOTTEN
    if any(
        r is Relation.CORRECTION and p.verb == "correct" and p.occurred_at > c.occurred_at
        for p, r in relations
    ):
        return ConflictStatus.CORRECTED
    if any(r is Relation.CONTRADICTION for _, r in relations):
        return ConflictStatus.CONTRADICTED
    if any(
        r in (Relation.TEMPORAL_CHANGE, Relation.CORRECTION)
        and c.occurred_at < p.occurred_at <= valid_at
        for p, r in relations
    ):
        return ConflictStatus.SUPERSEDED
    if any(r is Relation.SAME for _, r in relations):
        return ConflictStatus.SUPPORTED
    return ConflictStatus.UNDISPUTED


def conflicting(entry: Entry, entries: Sequence[Entry]) -> tuple[str, ...]:
    """Versions whose claims assert another value for this entry's key, or retract it
    (older or newer). Sorted by digest. Counter-evidence for exposure, not a verdict."""
    c = entry.claim
    if c is None:
        return ()
    return tuple(
        sorted(
            e.digest
            for e in entries
            if e.claim is not None
            and e.digest != entry.digest
            and e.claim.key == c.key
            and relate(c, e.claim) not in (Relation.SAME, Relation.INDEPENDENT)
        )
    )


# --- queries -------------------------------------------------------------------------------------


class HybridQuery(Record):
    """Retrieval text pinned to both time axes, with optional structured targets.

    ``key`` names the claim key asked about (``<entity>.<attribute>``, e.g. ``ana.home``);
    ``kinds`` the memory kinds requested. Both are explicit: nothing is inferred from text.
    """

    text: str
    valid_at: UTCDatetime
    known_at: UTCDatetime
    limit: int = Field(ge=1)
    key: str | None = Field(default=None, pattern=r"^[a-z0-9_.-]+$")
    kinds: tuple[MemoryKind, ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if list(self.kinds) != sorted(set(self.kinds)):
            raise ValueError("kinds must be unique and sorted")
        return self

    @property
    def core(self) -> Query:
        return Query(
            text=self.text, valid_at=self.valid_at, known_at=self.known_at, limit=self.limit
        )


def entity(key: str) -> str | None:
    """The entity of ``<entity>.<attribute>``; keys without a dot name no entity."""
    return key.split(".", 1)[0] if "." in key else None


# --- signals -------------------------------------------------------------------------------------


class SignalValue(Record):
    """One signal for one candidate: the raw value (or why it is missing), its declared
    direction, the normalised value, and the named inputs that produced it."""

    signal: str
    version: str
    direction: Literal[1, -1]  # 1: higher raw favours retrieval; -1: disfavours it
    raw: float | None = Field(default=None, allow_inf_nan=False)
    normalized: float | None = Field(default=None, allow_inf_nan=False)
    missing: str | None = None  # why the signal is unavailable for this candidate
    inputs: Inputs = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if (self.raw is None) != (self.missing is not None):
            raise ValueError("a signal value has a raw value or a reason it is missing")
        if self.raw is None and self.normalized is not None:
            raise ValueError("a missing signal has no normalised value")
        return self

    @property
    def available(self) -> bool:
        return self.raw is not None


Extracted = tuple[float | None, str | None, tuple[tuple[str, Scalar], ...]]


@dataclass(frozen=True)
class Context:
    """What one retrieval gives every signal extractor."""

    engine: Engine
    query: HybridQuery
    params: Mapping[str, Scalar]
    levels: tuple[Level, ...] = (Level.L1,)


def _need(params: Mapping[str, Scalar], allowed: Iterable[str], required: bool = True) -> None:
    unknown = set(params) - set(allowed)
    missing = set(allowed) - set(params) if required else set()
    if unknown or missing:
        raise PolicyError(
            f"parameters: unknown {sorted(unknown)} / missing {sorted(missing)} "
            f"(allowed {sorted(allowed)})"
        )


def _check_bm25(params: Mapping[str, Scalar]) -> None:
    _need(params, ("b", "k1"))
    try:
        BM25(float(params["k1"]), float(params["b"]))
    except ValueError as e:
        raise PolicyError(str(e)) from e


def _check_recency(params: Mapping[str, Scalar]) -> None:
    _need(params, ("axis", "family", "half_life_days"))
    if params["family"] != "exponential":
        raise PolicyError("recency decay family must be 'exponential'")
    if params["axis"] not in ("recorded", "valid"):
        raise PolicyError("recency axis must be 'recorded' or 'valid'")
    hl = params["half_life_days"]
    if isinstance(hl, str) or not (math.isfinite(hl) and hl > 0):
        raise PolicyError("recency half_life_days must be a finite positive number")


def _check_source(params: Mapping[str, Scalar]) -> None:
    if not params:
        raise PolicyError("source reliability needs at least one declared prior")
    for name, value in params.items():
        if not name.startswith("prior:") or name == "prior:":
            raise PolicyError(f"source parameters are 'prior:<class>', got {name!r}")
        if isinstance(value, str) or not (math.isfinite(value) and 0 <= value <= 1):
            raise PolicyError(f"{name} must be a number in [0, 1]")


def _no_params(params: Mapping[str, Scalar]) -> None:
    _need(params, (), required=False)


def _lexical(ctx: Context, e: Entry) -> Extracted:
    k1, b = float(ctx.params["k1"]), float(ctx.params["b"])
    ev = ctx.engine.bm25(ctx.query, k1, b, ctx.levels)[e.digest]
    return ev.score, None, ev.inputs


def _semantic(ctx: Context, e: Entry) -> Extracted:
    c = ctx.engine.similarity(ctx.query.text, e.version.content or "")
    return c, None, (("cosine", c),)


def _recency(ctx: Context, e: Entry) -> Extracted:
    axis = ctx.params["axis"]
    if axis == "recorded":
        ref, at = ctx.query.known_at, e.version.recorded_at
    else:
        ref = ctx.query.valid_at
        at = e.interval[0] if e.interval else e.version.valid_from  # type: ignore[assignment]
    assert at is not None  # every entry has content, hence valid_from
    days = quantize((ref - at) / timedelta(days=1))
    inputs: tuple[tuple[str, Scalar], ...] = (
        ("age_days", days),
        ("reference", ref.isoformat()),
    )
    if days < 0:  # only possible on the valid axis: not yet valid at valid_at
        return None, "future_on_valid_axis", inputs
    hl = float(ctx.params["half_life_days"])
    return quantize(0.5 ** (days / hl)), None, inputs


def _temporal(ctx: Context, e: Entry) -> Extracted:
    status = e.temporal(ctx.query.valid_at)
    gap = 0.0
    if e.interval is not None and status is TemporalStatus.FUTURE:
        gap = (e.interval[0] - ctx.query.valid_at) / timedelta(days=1)
    elif e.interval is not None and e.interval[1] is not None and status is TemporalStatus.EXPIRED:
        gap = (ctx.query.valid_at - e.interval[1]) / timedelta(days=1)
    raw = 1.0 if status is TemporalStatus.VALID else 0.0
    return raw, None, (("gap_days", quantize(gap)), ("status", status.value))


def _source_class(source: str) -> str | None:
    cls, sep, ident = source.partition(":")
    return cls if sep and cls and ident else None


def _source(ctx: Context, e: Entry) -> Extracted:
    classes = [_source_class(s.source) for s in e.sources]
    inputs: tuple[tuple[str, Scalar], ...] = (
        ("classes", " ".join(sorted({c or "?" for c in classes}))),
        ("evidence", len(e.sources)),
        ("sources", " ".join(s.source for s in e.sources)),
    )
    if not classes:
        return None, "no_source", inputs
    if None in classes:
        return None, "unstructured_source", inputs
    priors = [ctx.params.get(f"prior:{c}") for c in classes]
    if any(p is None for p in priors):
        return None, "undeclared_source_class", inputs
    # Several sources: the least reliable declared prior (a conservative, stated choice).
    return min(float(p) for p in priors if p is not None), None, inputs


def _provenance(ctx: Context, e: Entry) -> Extracted:
    cited = len(e.version.derived_from)
    structured = sum(1 for s in e.sources if _source_class(s.source) is not None)
    inputs: tuple[tuple[str, Scalar], ...] = (
        ("cited", cited),
        ("depth", e.version.version),
        ("resolved", len(e.sources)),
        ("structured", structured),
    )
    return quantize(structured / cited), None, inputs


def _kind(ctx: Context, e: Entry) -> Extracted:
    inputs: tuple[tuple[str, Scalar], ...] = (("kind", e.kind),)
    if not ctx.query.kinds:
        return None, "query_requests_no_kind", inputs
    return (1.0 if e.kind in ctx.query.kinds else 0.0), None, inputs


def _attribute(ctx: Context, e: Entry) -> Extracted:
    q = ctx.query.key
    if q is None:
        return None, "query_has_no_key", ()
    if e.claim is None:
        return None, f"memory_{e.claim_issue}", ()
    same_entity = entity(q) is not None and entity(q) == entity(e.claim.key)
    inputs: tuple[tuple[str, Scalar], ...] = (
        ("entity_match", int(same_entity or e.claim.key == q)),
        ("memory_key", e.claim.key),
    )
    return (1.0 if e.claim.key == q else 0.0), None, inputs


def _support(ctx: Context, e: Entry) -> Extracted:
    inputs: tuple[tuple[str, Scalar], ...] = (
        ("level", e.level.value),
        ("status", e.status.value),
    )
    return float(len(e.version.derived_from)), None, inputs


def _contradiction(ctx: Context, e: Entry) -> Extracted:
    status = ctx.engine.conflict(e, ctx.query.valid_at)
    if status is None:
        return None, f"memory_{e.claim_issue}", ()
    peers = len(ctx.engine.conflicting(e))
    raw = 1.0 if status in CONFLICTED else 0.0
    return raw, None, (("other_value_versions", peers), ("status", status.value))


@dataclass(frozen=True)
class SignalDef:
    name: str
    version: str
    direction: Literal[1, -1]
    bounds: tuple[float, float] | None  # declared range of the raw value, if bounded
    check: Callable[[Mapping[str, Scalar]], None]
    extract: Callable[[Context, Entry], Extracted]
    description: str


SIGNALS: dict[str, SignalDef] = {
    d.name: d
    for d in (
        SignalDef(
            "lexical", "1", 1, None, _check_bm25, _lexical,
            "Okapi BM25 of the query over memory content, corpus statistics from the corpus.",
        ),
        SignalDef(
            "semantic", "1", 1, (-1.0, 1.0), _no_params, _semantic,
            "Exact cosine of float32 embeddings under the engine's embedder.",
        ),
        SignalDef(
            "recency", "1", 1, (0.0, 1.0), _check_recency, _recency,
            "0.5 ** (age / half-life) on the recorded or valid axis; not temporal validity.",
        ),
        SignalDef(
            "temporal", "1", 1, (0.0, 1.0), _no_params, _temporal,
            "1 if the version's effective interval contains valid_at, else 0 (compatibility).",
        ),
        SignalDef(
            "source", "1", 1, (0.0, 1.0), _check_source, _source,
            "Declared prior of the cited experiences' source class ('<class>:<id>').",
        ),
        SignalDef(
            "provenance", "1", 1, (0.0, 1.0), _no_params, _provenance,
            "Fraction of cited experiences that resolve and carry a structured source id.",
        ),
        SignalDef(
            "type", "1", 1, (0.0, 1.0), _no_params, _kind,
            "1 if the memory's kind (fact / note) is one the query requests.",
        ),
        SignalDef(
            "attribute", "1", 1, (0.0, 1.0), _no_params, _attribute,
            "1 if the memory's claim key equals the query key; an entity-only match is 0.",
        ),
        SignalDef(
            "support", "1", 1, None, _no_params, _support,
            "Number of source experiences the memory covers (1 for an L1 episode).",
        ),
        SignalDef(
            "contradiction", "1", -1, (0.0, 1.0), _no_params, _contradiction,
            "1 if another known claim disputes the memory's claim (see ConflictStatus).",
        ),
    )
}  # fmt: skip


# --- normalisation -------------------------------------------------------------------------------

Normalization = Literal["bounded-v1", "minmax-v1", "rank-v1", "zscore-v1"]


def _finite(x: float, what: str) -> float:
    if not math.isfinite(x):
        raise ScoreError(f"{what} is not finite: {x}")
    return x


def normalize(
    strategy: Normalization, raws: Sequence[float | None], bounds: tuple[float, float] | None
) -> list[float | None]:
    """Normalise one signal's raw values over the candidates that survived filtering.

    Missing values stay missing and do not enter pool statistics.

    - ``bounded-v1``: (raw - lo) / (hi - lo) over the signal's declared range; for [0, 1]
      signals this is the identity (declared already normalised). Out of range is an error.
    - ``minmax-v1``: (raw - min) / (max - min) over the pool; a degenerate pool (one value,
      or all equal) maps to 0.0, which is rank-neutral.
    - ``rank-v1``: (n - r + 1) / n with competition ranks r (ties share the best rank), in
      (0, 1]; a single value or an all-equal pool maps to 1.0.
    - ``zscore-v1``: (raw - mean) / population sd; a degenerate pool (n < 2 or sd = 0)
      maps to 0.0 (every value is the mean). Unbounded, may be negative.
    """
    values = [x for x in raws if x is not None]
    for x in values:
        _finite(x, "raw signal value")
    if strategy == "bounded-v1":
        if bounds is None:
            raise PolicyError("bounded normalisation needs a signal with a declared range")
        lo, hi = bounds
        for x in values:
            if not lo <= x <= hi:
                raise SignalError(f"raw value {x} outside the declared range [{lo}, {hi}]")
        return [None if x is None else quantize((x - lo) / (hi - lo)) for x in raws]
    if not values:
        return [None] * len(raws)
    try:
        if strategy == "minmax-v1":
            lo, hi = min(values), max(values)
            span = _finite(hi - lo, "min-max range")
            return [
                None if x is None else (quantize((x - lo) / span) if span else 0.0) for x in raws
            ]
        if strategy == "rank-v1":
            n = len(values)
            return [
                None if x is None else quantize((n - sum(1 for y in values if y > x)) / n)
                for x in raws
            ]
        if strategy == "zscore-v1":
            n = len(values)
            mean = _finite(math.fsum(values) / n, "mean")
            sd = _finite(math.sqrt(math.fsum((x - mean) ** 2 for x in values) / n), "sd")
            degenerate = n < 2 or sd == 0
            return [
                None
                if x is None
                else (0.0 if degenerate else quantize(_finite((x - mean) / sd, "z-score")))
                for x in raws
            ]
    except OverflowError as e:
        raise ScoreError(f"{strategy}: {e}") from e
    raise PolicyError(f"unknown normalisation {strategy!r}")


# --- policy ---------------------------------------------------------------------------------------

GENERATORS = ("lexical", "metadata", "semantic")
TIE_BREAK = "final>relevance>memory_id>version>digest"
RRF = "Reciprocal rank fusion, Cormack, Clarke & Buettcher (SIGIR 2009)"


class GeneratorSpec(Record):
    """A candidate generator and its configuration. ``limit`` caps what it proposes."""

    name: Literal["lexical", "metadata", "semantic"]
    limit: int = Field(ge=1)
    params: Inputs = ()


class SignalConfig(Record):
    """How a policy uses one signal: weight, normalisation, and what a missing value does
    (``fail``: the query fails; ``omit``: no contribution, recorded as omitted)."""

    name: str
    weight: float = Field(gt=0, allow_inf_nan=False)
    normalization: Normalization
    on_missing: Literal["fail", "omit"]
    params: Inputs = ()


class DiversitySpec(Record):
    """Greedy MMR (Carbonell & Goldstein, 1998) in penalty form:
    final = relevance - beta * max(0, max similarity to the items placed before).
    Ordering equals MMR with lambda = 1 / (1 + beta)."""

    beta: float = Field(gt=0, allow_inf_nan=False)
    similarity: Literal["embedding", "token-jaccard"]


Exposure = Literal["neutral", "penalize", "surface", "paired"]


class RetrievalPolicy(ExtensibleRecord):
    """A complete, versioned retrieval policy. Its digest is the policy identity.

    Generators and signals are listed in name order, so one behaviour has one identity.
    Weights are positive and sum to 1. ``exclude`` lists the hard filters; malformed
    provenance is always excluded (no provenance-free memory, I4). ``levels`` (schema
    evolution; default L1, the only level before consolidation) are the memory levels the
    policy may retrieve.
    """

    _evolved = frozenset({"levels"})

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    generators: tuple[GeneratorSpec, ...] = Field(min_length=1)
    on_generator_failure: Literal["fail", "record"]
    exclude: tuple[Exclusion, ...]
    scoring: Literal["weighted", "rrf"]
    rrf_k: int | None = Field(default=None, ge=1)
    signals: tuple[SignalConfig, ...] = Field(min_length=1)
    contradiction: Exposure
    diversity: DiversitySpec | None
    tie_break: Literal["final>relevance>memory_id>version>digest"] = (
        "final>relevance>memory_id>version>digest"
    )
    levels: tuple[Level, ...] = (Level.L1,)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if not self.levels or list(self.levels) != sorted(set(self.levels)):
            raise ValueError("levels must be non-empty, unique and sorted")
        if Level.L0 in self.levels:
            raise ValueError("L0 experiences are reached through provenance, not retrieved")
        gens = [g.name for g in self.generators]
        if gens != sorted(set(gens)):
            raise ValueError("generators must be unique and sorted by name")
        for g in self.generators:
            allowed = {"lexical": ("b", "k1"), "semantic": ("index",), "metadata": ()}[g.name]
            _need(dict(g.params), allowed)
            if g.name == "semantic" and dict(g.params)["index"] not in ("exact", "hnsw", "scan"):
                raise ValueError("semantic index must be 'exact', 'hnsw' or 'scan'")
        if list(self.exclude) != sorted(set(self.exclude)):
            raise ValueError("exclusions must be unique and sorted")
        if Exclusion.MALFORMED_PROVENANCE not in self.exclude:
            raise ValueError("malformed provenance is always a hard exclusion")
        names = [s.name for s in self.signals]
        if names != sorted(set(names)):
            raise ValueError("signals must be unique and sorted by name")
        for s in self.signals:
            d = SIGNALS.get(s.name)
            if d is None:
                raise ValueError(f"unknown signal {s.name!r} (registered: {sorted(SIGNALS)})")
            if s.normalization == "bounded-v1" and d.bounds is None:
                raise ValueError(f"signal {s.name!r} is unbounded; bounded-v1 does not apply")
            d.check(dict(s.params))
        total = math.fsum(s.weight for s in self.signals)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"signal weights must sum to 1, got {total!r}")
        if (self.scoring == "rrf") != (self.rrf_k is not None):
            raise ValueError("rrf_k is required for, and only for, rank fusion")
        if (self.contradiction == "penalize") != ("contradiction" in names):
            raise ValueError("the contradiction signal is scored iff contradiction='penalize'")
        return self

    def signal(self, name: str) -> SignalConfig | None:
        return next((s for s in self.signals if s.name == name), None)


# --- trace records -----------------------------------------------------------------------------


class Proposal(Record):
    version: Digest
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)


class GeneratorRun(Record):
    """What one generator did for one query. ``not_applicable``: the query lacks what the
    generator needs (e.g. no key for metadata); ``failed``: it could not run (``error``)."""

    generator: str
    version: str
    params: Inputs = ()
    limit: int
    status: Literal["ok", "not_applicable", "failed"]
    error: str | None = None
    considered: int = Field(ge=0)  # corpus entries it scored
    truncated: int = Field(ge=0)  # qualifying entries dropped by the limit
    proposals: tuple[Proposal, ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("an error is recorded iff the generator failed")
        if self.status != "ok" and self.proposals:
            raise ValueError("only a generator that ran proposes candidates")
        if [p.rank for p in self.proposals] != list(range(1, len(self.proposals) + 1)):
            raise ValueError("proposal ranks must be 1..n")
        if len({p.version for p in self.proposals}) != len(self.proposals):
            raise ValueError("a generator proposes each version at most once")
        if len(self.proposals) > self.limit:
            raise ValueError("a generator proposes at most its limit")
        return self


class GeneratorHit(Record):
    generator: str
    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)


class CandidateRecord(ExtensibleRecord):
    """A member of the candidate union: who proposed it, its metadata, and its exclusion.

    ``level``, ``status`` and ``stale`` (schema evolution; defaults: an observed L1
    version) say what kind of memory it is, so a derived memory is never mistaken for one.
    """

    _evolved = frozenset({"level", "status", "stale"})

    version: Digest
    memory_id: str
    number: int = Field(ge=1)  # the version number in its chain
    generators: tuple[GeneratorHit, ...] = Field(min_length=1)  # by generator name
    temporal: TemporalStatus
    conflict: ConflictStatus | None  # None: the memory carries no structured claim
    conflicts_with: tuple[Digest, ...]  # corpus versions with another value for its key
    excluded: Exclusion | None = None
    level: Level = Level.L1
    status: EpistemicStatus = EpistemicStatus.OBSERVED
    stale: bool = False


class Contribution(Record):
    """A signal's part of the relevance score. ``omitted``: missing, policy said omit."""

    signal: str
    weight: float
    normalized: float | None
    rank: int | None = None  # rank fusion: the candidate's rank under this signal
    value: float = Field(allow_inf_nan=False)
    omitted: bool = False


class Explanation(Record):
    reasons: tuple[str, ...]  # machine-readable codes, derived from the recorded numbers
    text: str  # a deterministic rendering of the codes and numbers


class Ranked(Record):
    """One surviving candidate: its evidence, score decomposition and place.

    relevance = sum(contributions); final = relevance - penalty (the reranking adjustment).
    ``stage_rank`` is its position after scoring and diversity; ``rank`` after exposure.
    """

    rank: int = Field(ge=1)
    stage_rank: int = Field(ge=1)
    version: Digest
    memory_id: str
    number: int = Field(ge=1)
    signals: tuple[SignalValue, ...]
    contributions: tuple[Contribution, ...]
    relevance: float = Field(allow_inf_nan=False)
    similarity: float | None = None  # max similarity to items placed before (diversity)
    similar_to: Digest | None = None
    penalty: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    final: float = Field(allow_inf_nan=False)
    placement: Literal["ordered", "paired"] = "ordered"
    paired_with: Digest | None = None
    counter_evidence: tuple[Digest, ...] = ()  # surfaced other-value versions (surface)
    explanation: Explanation

    def contribution(self, signal: str) -> Contribution:
        return next(c for c in self.contributions if c.signal == signal)


def tie_key(r: Ranked) -> tuple[float, float, str, int, str]:
    """final desc, relevance desc, memory_id, version number, digest: content-based."""
    return (-r.final, -r.relevance, r.memory_id, r.number, r.version)


_TEMPORAL_TEXT = {
    TemporalStatus.VALID: "valid at the query's valid time",
    TemporalStatus.EXPIRED: "no longer valid at the query's valid time",
    TemporalStatus.FUTURE: "not yet valid at the query's valid time",
    TemporalStatus.SUPERSEDED: "corrected away in its own history",
    TemporalStatus.FORGOTTEN: "forgotten in its own history",
}
_CONFLICT_TEXT = {
    ConflictStatus.UNDISPUTED: "no known claim disputes or repeats it",
    ConflictStatus.SUPPORTED: "another claim asserts the same value",
    ConflictStatus.SUPERSEDED: "a later claim changed the value",
    ConflictStatus.CONTRADICTED: "a claim for the same instant disagrees",
    ConflictStatus.CORRECTED: "a later correction replaced it",
    ConflictStatus.FORGOTTEN: "a later forget retracted it",
    ConflictStatus.RETRACTION: "it is itself a retraction",
}


def explain(r: Ranked, c: CandidateRecord) -> Explanation:
    """Reason codes and text, derived only from the recorded numbers of ``r`` and ``c``."""
    supporting = sorted(
        (x for x in r.contributions if x.value > 0), key=lambda x: (-x.value, x.signal)
    )[:2]
    opposing = sorted((x for x in r.contributions if x.value < 0), key=lambda x: x.signal)
    omitted = [x.signal for x in r.contributions if x.omitted]
    reasons = [f"supports:{x.signal}" for x in supporting]
    reasons += [f"opposes:{x.signal}" for x in opposing]
    reasons += [f"missing:{s}" for s in omitted]
    reasons.append(f"temporal:{c.temporal.value}")
    if c.conflict is not None:
        reasons.append(f"conflict:{c.conflict.value}")
    reasons.append("generators:" + "+".join(g.generator for g in c.generators))
    if r.penalty > 0:
        reasons.append("redundant")
    if r.placement == "paired":
        reasons.append("paired")
    if r.counter_evidence:
        reasons.append("counter_evidence")
    parts = []
    if supporting:
        parts.append(
            "largest contributions: " + ", ".join(f"{x.signal} {x.value:+.3f}" for x in supporting)
        )
    if opposing:
        parts.append("against: " + ", ".join(f"{x.signal} {x.value:+.3f}" for x in opposing))
    if omitted:
        parts.append("missing and omitted: " + ", ".join(omitted))
    parts.append(_TEMPORAL_TEXT[c.temporal])
    if c.conflict is not None:
        parts.append(_CONFLICT_TEXT[c.conflict])
    parts.append("proposed by " + ", ".join(g.generator for g in c.generators))
    if r.penalty > 0:
        parts.append(f"redundancy penalty {-r.penalty:+.3f} (similarity {r.similarity:.3f})")
    if r.placement == "paired":
        parts.append("placed next to a result with another value for its key")
    if r.counter_evidence:
        parts.append(f"surfaced {len(r.counter_evidence)} memories with another value for its key")
    text = f"relevance {r.relevance:.3f}, final {r.final:.3f}: " + "; ".join(parts) + "."
    return Explanation(reasons=tuple(reasons), text=text)


def _contribute(
    policy: RetrievalPolicy, ranked_signals: Sequence[Sequence[SignalValue]]
) -> list[tuple[Contribution, ...]]:
    """Every candidate's contributions, recomputed from the (normalised) signal values."""
    out: list[list[Contribution]] = [[] for _ in ranked_signals]
    for j, cfg in enumerate(policy.signals):
        direction = SIGNALS[cfg.name].direction
        values = [sv[j].normalized for sv in ranked_signals]
        ranks: list[int | None] = [None] * len(values)
        if policy.scoring == "rrf":
            keyed = [None if v is None else direction * v for v in values]
            present = [k for k in keyed if k is not None]
            ranks = [None if k is None else 1 + sum(1 for x in present if x > k) for k in keyed]
        for i, v in enumerate(values):
            if v is None:
                out[i].append(
                    Contribution(
                        signal=cfg.name, weight=cfg.weight, normalized=None, value=0.0, omitted=True
                    )
                )
                continue
            if policy.scoring == "rrf":
                rank = ranks[i]
                assert rank is not None
                assert policy.rrf_k is not None
                value = quantize(cfg.weight / (policy.rrf_k + rank))
            else:
                rank = None
                value = quantize(_finite(cfg.weight * direction * v, "contribution"))
            out[i].append(
                Contribution(
                    signal=cfg.name, weight=cfg.weight, normalized=v, rank=rank, value=value
                )
            )
    return [tuple(c) for c in out]


def _relevance(contributions: Sequence[Contribution]) -> float:
    try:
        return quantize(_finite(math.fsum(c.value for c in contributions), "relevance"))
    except OverflowError as e:
        raise ScoreError(f"relevance: {e}") from e


def _penalty(diversity: DiversitySpec | None, similarity: float | None) -> float:
    if diversity is None or similarity is None:
        return 0.0
    return quantize(diversity.beta * max(similarity, 0.0))


def _expose(
    stage: Sequence[str], disputes: Mapping[str, tuple[str, ...]], mode: Exposure, limit: int
) -> list[tuple[str, str | None]]:
    """Final order as (version, paired_with): the stage order, except that in ``paired``
    mode each result within the limit is followed by its best-placed survivor asserting
    another value for its key (if not already placed)."""
    placed: list[tuple[str, str | None]] = []
    used: set[str] = set()
    for v in stage:
        if v in used:
            continue
        placed.append((v, None))
        used.add(v)
        if mode != "paired" or len(placed) >= limit:
            continue
        partner = next((p for p in stage if p in disputes[v] and p not in used), None)
        if partner is not None:
            placed.append((partner, v))
            used.add(partner)
    return placed


class HybridTrace(Record):
    """A complete, self-checking record of one hybrid retrieval.

    Every generator run, every union member (with its exclusion, if any) and every
    surviving candidate (with signals, contributions and placement) is recorded. The
    validator re-derives normalisation, contributions, relevance, penalties, final
    scores, the order, exposure, selection and explanations from the raw values.
    """

    query: HybridQuery
    policy: RetrievalPolicy
    corpus: Digest
    embedder: Digest | None
    index: Digest | None
    generators: tuple[GeneratorRun, ...]
    candidates: tuple[CandidateRecord, ...]
    ranking: tuple[Ranked, ...]
    selected: tuple[Digest, ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        p = self.policy
        # Generators and union.
        if [g.generator for g in self.generators] != [g.name for g in p.generators]:
            raise ValueError("generator runs do not match the policy")
        hits: dict[str, list[GeneratorHit]] = {}
        for g in self.generators:
            for pr in g.proposals:
                hits.setdefault(pr.version, []).append(
                    GeneratorHit(generator=g.generator, rank=pr.rank, score=pr.score)
                )
        if {c.version for c in self.candidates} != set(hits) or len(self.candidates) != len(hits):
            raise ValueError("candidates are not exactly the union of the proposals")
        for c in self.candidates:
            if list(c.generators) != hits[c.version]:
                raise ValueError(f"candidate {c.memory_id}: generator provenance does not match")
        order = [(c.memory_id, c.number, c.version) for c in self.candidates]
        if order != sorted(order):
            raise ValueError("candidates are not in canonical (memory_id, version) order")
        by_version = {c.version: c for c in self.candidates}
        for c in self.candidates:
            if c.level not in p.levels:
                raise ValueError(f"candidate {c.memory_id}: level {c.level} not retrievable")
            if (c.level is Level.L1) != (c.status is EpistemicStatus.OBSERVED):
                raise ValueError(f"candidate {c.memory_id}: only L1 memories are observed")
            expected: Exclusion | None = None
            if c.excluded is Exclusion.MALFORMED_PROVENANCE:
                expected = c.excluded
            elif c.stale and Exclusion.STALE in p.exclude:
                expected = Exclusion.STALE
            elif c.temporal.value in {x.value for x in p.exclude}:
                expected = Exclusion(c.temporal.value)
            if c.excluded is not expected:
                raise ValueError(f"candidate {c.memory_id}: exclusion does not follow the policy")
        survivors = {c.version for c in self.candidates if c.excluded is None}
        if {r.version for r in self.ranking} != survivors or len(self.ranking) != len(survivors):
            raise ValueError("the ranking is not exactly the surviving candidates")
        # Signals, normalisation, contributions, relevance, penalty, final.
        names = [s.name for s in p.signals]
        for r in self.ranking:
            if [s.signal for s in r.signals] != names:
                raise ValueError(f"{r.memory_id}: signals do not match the policy")
            if (r.memory_id, r.number) != (
                by_version[r.version].memory_id,
                by_version[r.version].number,
            ):
                raise ValueError(f"{r.memory_id}: identity does not match its candidate")
        for j, cfg in enumerate(p.signals):
            column = [r.signals[j] for r in self.ranking]
            if cfg.on_missing == "fail" and any(not s.available for s in column):
                raise ValueError(f"signal {cfg.name!r} is missing but the policy says fail")
            renormalized = normalize(
                cfg.normalization, [s.raw for s in column], SIGNALS[cfg.name].bounds
            )
            if [s.normalized for s in column] != renormalized:
                raise ValueError(f"signal {cfg.name!r}: normalised values not reproducible")
        contributions = _contribute(p, [r.signals for r in self.ranking])
        for r, cs in zip(self.ranking, contributions, strict=True):
            if r.contributions != cs:
                raise ValueError(f"{r.memory_id}: contributions not reproducible")
            if r.relevance != _relevance(cs):
                raise ValueError(f"{r.memory_id}: relevance is not the sum of contributions")
            if r.penalty != _penalty(p.diversity, r.similarity):
                raise ValueError(f"{r.memory_id}: redundancy penalty not reproducible")
            if r.final != quantize(r.relevance - r.penalty):
                raise ValueError(f"{r.memory_id}: final score is not relevance - penalty")
        # Order: stage order is the tie-break order of final scores; diversity links back.
        stage = sorted(self.ranking, key=lambda r: r.stage_rank)
        if [r.stage_rank for r in stage] != list(range(1, len(stage) + 1)):
            raise ValueError("stage ranks must be 1..n")
        if [tie_key(r) for r in stage] != sorted(tie_key(r) for r in stage):
            raise ValueError("stage order does not follow the tie-break rule")
        stage_of = {r.version: r.stage_rank for r in self.ranking}
        for r in stage:
            first = r.stage_rank == 1
            if p.diversity is None or first:
                if (r.similarity, r.similar_to) != (None, None):
                    raise ValueError(f"{r.memory_id}: similarity recorded without a predecessor")
            elif r.similar_to is None or stage_of.get(r.similar_to, r.stage_rank) >= r.stage_rank:
                raise ValueError(f"{r.memory_id}: redundancy must refer to an earlier item")
        # Exposure and selection.
        disputes = {
            r.version: tuple(v for v in by_version[r.version].conflicts_with if v in survivors)
            for r in self.ranking
        }
        placed = _expose([r.version for r in stage], disputes, p.contradiction, self.query.limit)
        if placed != [(r.version, r.paired_with) for r in self.ranking]:
            raise ValueError("final order does not follow the exposure rule")
        for i, r in enumerate(self.ranking, 1):
            if r.rank != i or (r.placement == "paired") != (r.paired_with is not None):
                raise ValueError(f"{r.memory_id}: rank or placement inconsistent")
        if self.selected != tuple(r.version for r in self.ranking[: self.query.limit]):
            raise ValueError("selected is not the top of the final ranking")
        chosen = set(self.selected)
        for r in self.ranking:
            expected_ce: tuple[str, ...] = ()
            if p.contradiction == "surface" and r.version in chosen:
                expected_ce = tuple(sorted(disputes[r.version], key=lambda v: stage_of[v]))
            if r.counter_evidence != expected_ce:
                raise ValueError(f"{r.memory_id}: counter-evidence does not follow the policy")
            if r.explanation != explain(r, by_version[r.version]):
                raise ValueError(f"{r.memory_id}: explanation is not derived from the record")
        return self

    def result(self, version: str) -> Ranked:
        return next(r for r in self.ranking if r.version == version)

    def candidate(self, version: str) -> CandidateRecord:
        return next(c for c in self.candidates if c.version == version)


# --- the engine ---------------------------------------------------------------------------------


def _token_jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return quantize(len(x & y) / len(x | y)) if x | y else 0.0


@dataclass
class Engine:
    """Runs policies over one corpus with fixed resources (embedder, semantic index).

    Pure given its inputs; memoises embeddings and BM25 scores by text. Timings, when
    requested, are accumulated separately and never enter a trace.
    """

    corpus: Corpus
    embedder: Embedder | None = None
    index: SemanticIndex | None = None
    # Memos of pure functions of (text, embedder); callers may share them between engines
    # that use the same embedder (e.g. every probe of one run).
    vectors: dict[str, Vector] = field(default_factory=dict, repr=False)
    similarities: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)
    _bm25: dict[tuple[str, float, float, tuple[Level, ...]], dict[str, SignalEvidence]] = field(
        default_factory=dict, repr=False
    )
    _conflicts: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.index is not None:
            if self.embedder is None:
                raise PolicyError("a semantic index needs the embedder that built it")
            if self.index.manifest.embedder != self.embedder.spec:
                raise IndexCompatibilityError("the index was built by another embedder")
        self._by_digest = {e.digest: e for e in self.corpus.entries}

    # Shared, memoised feature sources.
    def universe(self, levels: tuple[Level, ...]) -> list[Entry]:
        """The entries a policy may retrieve: those at its levels."""
        return [e for e in self.corpus.entries if e.level in levels]

    def bm25(
        self, query: HybridQuery, k1: float, b: float, levels: tuple[Level, ...] = (Level.L1,)
    ) -> dict[str, SignalEvidence]:
        """BM25 with corpus statistics over the policy's universe (its levels)."""
        key = (query.text, k1, b, levels)
        if key not in self._bm25:
            entries = self.universe(levels)
            # BM25 reads only ``content``, which derived memories carry too.
            scores = BM25(k1, b).score(query.core, [e.version for e in entries])  # type: ignore[misc]
            self._bm25[key] = {e.digest: s for e, s in zip(entries, scores, strict=True)}
        return self._bm25[key]

    def vector(self, text: str) -> Vector:
        if self.embedder is None:
            raise PolicyError("this policy needs an embedder and none was given")
        if text not in self.vectors:
            (v,) = embed(self.embedder, [text])
            self.vectors[text] = float32(v)
        return self.vectors[text]

    def similarity(self, a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)  # cosine is symmetric
        if key not in self.similarities:
            self.similarities[key] = quantize(cosine(self.vector(a), self.vector(b)))
        return self.similarities[key]

    def conflict(self, e: Entry, valid_at: datetime) -> ConflictStatus | None:
        return conflict(e, self.corpus.claims, valid_at)

    def conflicting(self, e: Entry) -> tuple[str, ...]:
        if e.digest not in self._conflicts:
            self._conflicts[e.digest] = conflicting(e, self.corpus.entries)
        return self._conflicts[e.digest]

    # Stages.
    def _generate(
        self, spec: GeneratorSpec, query: HybridQuery, levels: tuple[Level, ...] = (Level.L1,)
    ) -> GeneratorRun:
        params = dict(spec.params)

        def run(
            status: Literal["ok", "not_applicable", "failed"],
            error: str | None = None,
            considered: int = 0,
            truncated: int = 0,
            proposals: tuple[Proposal, ...] = (),
        ) -> GeneratorRun:
            return GeneratorRun(
                generator=spec.name,
                version="1",
                params=spec.params,
                limit=spec.limit,
                status=status,
                error=error,
                considered=considered,
                truncated=truncated,
                proposals=proposals,
            )

        entries = self.universe(levels)
        rows: list[tuple[float, Entry]]
        if spec.name == "lexical":
            scores = self.bm25(query, float(params["k1"]), float(params["b"]), levels)
            rows = [(scores[e.digest].score, e) for e in entries if scores[e.digest].score > 0]
            considered = len(entries)
        elif spec.name == "metadata":
            if query.key is None:
                return run("not_applicable")
            rows = [(1.0, e) for e in entries if e.claim is not None and e.claim.key == query.key]
            considered = len(entries)
        else:
            try:
                if params["index"] == "scan":
                    # Exact cosine over the policy's universe, no index artifact: the only
                    # way to reach derived memories, which no MemoryState holds.
                    text = query.text
                    rows = [(self.similarity(text, e.version.content or ""), e) for e in entries]
                    considered = len(entries)
                else:
                    rows, considered = self._semantic_rows(
                        query, spec.limit, str(params["index"]), levels
                    )
            except (GeneratorError, IndexCompatibilityError, PolicyError) as e:
                return run("failed", error=str(e))
        rows.sort(key=lambda r: (-r[0], r[1].version.memory_id, r[1].version.version, r[1].digest))
        kept = rows[: spec.limit]
        return run(
            "ok",
            considered=considered,
            truncated=len(rows) - len(kept),
            proposals=tuple(
                Proposal(version=e.digest, rank=i, score=s) for i, (s, e) in enumerate(kept, 1)
            ),
        )

    def _semantic_rows(
        self, query: HybridQuery, limit: int, kind: str, levels: tuple[Level, ...] = (Level.L1,)
    ) -> tuple[list[tuple[float, Entry]], int]:
        if self.index is None or self.embedder is None:
            raise GeneratorError("no semantic index was given")
        if self.index.manifest.index.kind != kind:
            raise GeneratorError(
                f"policy expects a {kind} index, got {self.index.manifest.index.kind}"
            )
        self.index.require_source(self.corpus.present)
        hits = self.index.search(query.text, limit, self.embedder)
        kept_hits = [h for h in hits if self._by_digest[h.version].level in levels]
        return [(h.similarity, self._by_digest[h.version]) for h in kept_hits], len(
            self.index.manifest.entries
        )

    def retrieve(
        self,
        policy: RetrievalPolicy,
        query: HybridQuery,
        timings: dict[str, float] | None = None,
    ) -> HybridTrace:
        """Run ``policy`` for ``query``. Raises on invalid resources, a failing generator
        under ``on_generator_failure="fail"``, or a missing ``on_missing="fail"`` signal."""
        if query.known_at != self.corpus.known_at:
            raise PolicyError("the corpus is not at the query's known_at")
        uses_vectors = policy.signal("semantic") is not None or (
            policy.diversity is not None and policy.diversity.similarity == "embedding"
        )
        if uses_vectors and self.embedder is None:
            raise PolicyError("the policy uses embeddings but the engine has no embedder")
        clock = _Clock(timings)

        with clock("generate"):
            runs = tuple(self._generate(g, query, policy.levels) for g in policy.generators)
        failed = [r for r in runs if r.status == "failed"]
        if failed and policy.on_generator_failure == "fail":
            raise GeneratorError("; ".join(f"{r.generator}: {r.error}" for r in failed))

        with clock("filter"):
            hits: dict[str, list[GeneratorHit]] = {}
            for run in runs:
                for pr in run.proposals:
                    hits.setdefault(pr.version, []).append(
                        GeneratorHit(generator=run.generator, rank=pr.rank, score=pr.score)
                    )
            union = sorted(
                (self._by_digest[d] for d in hits),
                key=lambda e: (e.version.memory_id, e.version.version, e.digest),
            )
            excluded_statuses = {x.value for x in policy.exclude}
            candidates = []
            for e in union:
                status = e.temporal(query.valid_at)
                reason = None
                if e.missing:
                    reason = Exclusion.MALFORMED_PROVENANCE
                elif e.stale and Exclusion.STALE in policy.exclude:
                    reason = Exclusion.STALE
                elif status.value in excluded_statuses:
                    reason = Exclusion(status.value)
                candidates.append(
                    CandidateRecord(
                        version=e.digest,
                        memory_id=e.version.memory_id,
                        number=e.version.version,
                        generators=tuple(hits[e.digest]),
                        temporal=status,
                        conflict=self.conflict(e, query.valid_at),
                        conflicts_with=self.conflicting(e),
                        excluded=reason,
                        level=e.level,
                        status=e.status,
                        stale=e.stale,
                    )
                )
            survivors = [e for e, c in zip(union, candidates, strict=True) if c.excluded is None]

        with clock("features"):
            raw: list[list[SignalValue]] = []
            for e in survivors:
                row = []
                for cfg in policy.signals:
                    d = SIGNALS[cfg.name]
                    ctx = Context(self, query, dict(cfg.params), policy.levels)
                    value, why, inputs = d.extract(ctx, e)
                    if value is not None:
                        _finite(value, f"signal {cfg.name}")
                    elif cfg.on_missing == "fail":
                        raise SignalError(f"signal {cfg.name!r} missing for {e.digest}: {why}")
                    row.append(
                        SignalValue(
                            signal=d.name,
                            version=d.version,
                            direction=d.direction,
                            raw=value,
                            missing=why,
                            inputs=inputs,
                        )
                    )
                raw.append(row)

        with clock("score"):
            columns = [
                normalize(cfg.normalization, [row[j].raw for row in raw], SIGNALS[cfg.name].bounds)
                for j, cfg in enumerate(policy.signals)
            ]
            signals = [
                tuple(
                    sv.model_copy(update={"normalized": columns[j][i]}) for j, sv in enumerate(row)
                )
                for i, row in enumerate(raw)
            ]
            contributions = _contribute(policy, signals)
            relevance = [_relevance(cs) for cs in contributions]

        with clock("rerank"):
            items = list(range(len(survivors)))

            def key(i: int, penalty: float) -> tuple[float, float, str, int, str]:
                e = survivors[i]
                return (
                    -quantize(relevance[i] - penalty),
                    -relevance[i],
                    e.version.memory_id,
                    e.version.version,
                    e.digest,
                )

            order: list[tuple[int, float | None, str | None]] = []  # (item, similarity, to)
            if policy.diversity is None:
                order = [(i, None, None) for i in sorted(items, key=lambda i: key(i, 0.0))]
            else:
                # ponytail: O(n^2) greedy MMR over all survivors; bounded by generator limits.
                sim = self._diversity_similarity(policy.diversity)
                best: dict[int, tuple[float, str] | None] = dict.fromkeys(items)
                remaining = set(items)
                while remaining:
                    choice = min(
                        remaining,
                        key=lambda i: key(i, _penalty(policy.diversity, _sim_of(best[i]))),
                    )
                    remaining.discard(choice)
                    link = best[choice]
                    order.append((choice, _sim_of(link), link[1] if link else None))
                    for i in remaining:
                        s = sim(survivors[i], survivors[choice])
                        current = best[i]
                        if (
                            current is None
                            or s > current[0]
                            or (s == current[0] and survivors[choice].digest < current[1])
                        ):
                            best[i] = (s, survivors[choice].digest)
            info = {
                survivors[i].digest: (stage_rank, i, similarity, to)
                for stage_rank, (i, similarity, to) in enumerate(order, 1)
            }
            alive = set(info)
            disputes = {
                e.digest: tuple(v for v in self.conflicting(e) if v in alive) for e in survivors
            }
            placed = _expose(
                [survivors[i].digest for i, _, _ in order],
                disputes,
                policy.contradiction,
                query.limit,
            )

        with clock("explain"):
            by_version = {c.version: c for c in candidates}
            ranking = []
            for rank, (version, partner) in enumerate(placed, 1):
                stage_rank, i, similarity, to = info[version]
                penalty = _penalty(policy.diversity, similarity)
                surfaced: tuple[str, ...] = ()
                if policy.contradiction == "surface" and rank <= query.limit:
                    surfaced = tuple(sorted(disputes[version], key=lambda v: info[v][0]))
                fields: dict[str, Any] = {
                    "rank": rank,
                    "stage_rank": stage_rank,
                    "version": version,
                    "memory_id": survivors[i].version.memory_id,
                    "number": survivors[i].version.version,
                    "signals": signals[i],
                    "contributions": contributions[i],
                    "relevance": relevance[i],
                    "similarity": similarity,
                    "similar_to": to,
                    "penalty": penalty,
                    "final": quantize(relevance[i] - penalty),
                    "placement": "paired" if partner else "ordered",
                    "paired_with": partner,
                    "counter_evidence": surfaced,
                }
                explained = explain(Ranked.model_construct(**fields), by_version[version])
                ranking.append(Ranked.model_validate(fields | {"explanation": explained}))
            trace = HybridTrace(
                query=query,
                policy=policy,
                corpus=self.corpus.digest,
                embedder=self.embedder.spec.digest if uses_vectors and self.embedder else None,
                index=self.index.digest
                if self.index is not None and any(g.name == "semantic" for g in policy.generators)
                else None,
                generators=runs,
                candidates=tuple(candidates),
                ranking=tuple(ranking),
                selected=tuple(r.version for r in ranking[: query.limit]),
            )
        return trace

    def _diversity_similarity(self, spec: DiversitySpec) -> Callable[[Entry, Entry], float]:
        if spec.similarity == "embedding":
            return lambda a, b: self.similarity(a.version.content or "", b.version.content or "")
        return lambda a, b: _token_jaccard(a.version.content or "", b.version.content or "")


def _sim_of(link: tuple[float, str] | None) -> float | None:
    return None if link is None else link[0]


class _Clock:
    """Accumulates wall time per stage into ``timings`` (if given). Environment-bound."""

    def __init__(self, timings: dict[str, float] | None) -> None:
        self._timings = timings

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._timings is not None:
                self._timings[stage] = self._timings.get(stage, 0.0) + time.perf_counter() - start


# --- policy presets -----------------------------------------------------------------------------

BM25_PARAMS: tuple[tuple[str, Scalar], ...] = (("b", 0.75), ("k1", 1.2))  # Robertson's defaults
DEFAULT_EXCLUDE = (
    Exclusion.EXPIRED,
    Exclusion.FORGOTTEN,
    Exclusion.FUTURE,
    Exclusion.MALFORMED_PROVENANCE,
    Exclusion.SUPERSEDED,
)
# Excludes only what the memory's own history retracted, not temporal incompatibility.
STATE_EXCLUDE = (Exclusion.FORGOTTEN, Exclusion.MALFORMED_PROVENANCE, Exclusion.SUPERSEDED)


def generators(
    limit: int, *, semantic: bool = True, index: str = "exact"
) -> tuple[GeneratorSpec, ...]:
    """Lexical, metadata and (optionally) semantic generators, each capped at ``limit``."""
    specs = [
        GeneratorSpec(name="lexical", limit=limit, params=BM25_PARAMS),
        GeneratorSpec(name="metadata", limit=limit),
    ]
    if semantic:
        specs.append(GeneratorSpec(name="semantic", limit=limit, params=(("index", index),)))
    return tuple(specs)


def signal(
    name: str,
    weight: float,
    normalization: Normalization | None = None,
    on_missing: Literal["fail", "omit"] = "omit",
    params: Mapping[str, Scalar] | None = None,
) -> SignalConfig:
    """A signal configuration; bounded signals default to ``bounded-v1``, lexical to
    ``minmax-v1``. Lexical parameters default to :data:`BM25_PARAMS` (explicit in the spec)."""
    if name == "lexical" and not params:
        params = dict(BM25_PARAMS)
    if normalization is None:
        normalization = "bounded-v1" if SIGNALS[name].bounds is not None else "minmax-v1"
    return SignalConfig(
        name=name,
        weight=weight,
        normalization=normalization,
        on_missing=on_missing,
        params=tuple((params or {}).items()),
    )


def equal_weights(
    names: Sequence[str], params: Mapping[str, Mapping[str, Scalar]] | None = None
) -> tuple[SignalConfig, ...]:
    """Equal weights over the named signals (the uninformed baseline, not tuned values)."""
    params = params or {}
    return tuple(signal(name, 1 / len(names), params=params.get(name)) for name in sorted(names))


def policy(
    name: str,
    signals: Sequence[SignalConfig],
    *,
    limit: int = 20,
    semantic_generator: bool = True,
    exclude: Sequence[Exclusion] = DEFAULT_EXCLUDE,
    scoring: Literal["weighted", "rrf"] = "weighted",
    rrf_k: int | None = None,
    contradiction: Exposure = "surface",
    diversity: DiversitySpec | None = None,
    on_generator_failure: Literal["fail", "record"] = "fail",
    version: str = "1",
) -> RetrievalPolicy:
    return RetrievalPolicy(
        name=name,
        version=version,
        generators=generators(limit, semantic=semantic_generator),
        on_generator_failure=on_generator_failure,
        exclude=tuple(sorted(set(exclude))),
        scoring=scoring,
        rrf_k=rrf_k,
        signals=tuple(sorted(signals, key=lambda s: s.name)),
        contradiction=contradiction,
        diversity=diversity,
    )


RRF_K = 60  # the constant used by Cormack, Clarke & Buettcher (2009)


def extractive(trace: HybridTrace, corpus: Corpus) -> Response:
    """The Phase 2 extractive responder over a hybrid trace: the top selected memory's
    content verbatim (observed or derived), citing it; abstain if nothing is selected."""
    if not trace.selected:
        return Response(trace=trace.digest, responder=EXTRACTIVE, output=None)
    top = next(e for e in corpus.entries if e.digest == trace.selected[0])
    return Response(
        trace=trace.digest,
        responder=EXTRACTIVE,
        output=top.version.content,
        cited=(top.digest,),
    )
