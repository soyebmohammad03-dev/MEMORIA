"""Contradiction taxonomy and the contradiction graph (Phases 11-12).

Two claims that disagree are not thereby wrong. This module classifies *how* a pair of
claims relate, by a fixed, ordered set of rules over structured claims, and keeps every
relation directional where direction matters. No relation says an endpoint is false: the
relation vocabulary has no ``falsifies``, and a supersession or correction keeps both
endpoints and their evidence.

Twelve categories (:data:`TYPES`), each with an operational definition below.

======================  ====================================================================
direct_value            same key, concurrent, different values, one source class
numeric                 same key, concurrent, numbers with one unit differing beyond tolerance
categorical             same key, concurrent, different members of a declared closed domain
temporal                same key, different values, *explicit* validity windows that overlap
negation                same key, concurrent: ``X`` against ``not X``
source_disagreement     same key, concurrent, different values, independent sources of
                        different classes
entity_collision        different entities the resolution keeps as unmerged *candidates*,
                        same attribute, concurrent, different values: possible mis-filing
scope                   parent and child scope of a declared scope pair, different values
partial                 multi-valued claims (``a; b``) that overlap but are not equal
supersession            same key, different values, not concurrent (or disjoint windows): a
                        change over time, or a correction when the later claim is ``correct``
granularity             one value refines the other in a declared hierarchy
missing_qualifier       different values where one claim has a qualifier and the other none
======================  ====================================================================

*Concurrent* means occurring within the ontology's ``concurrency_days`` of each other; it is
what separates a temporal change from a contradiction. Claims with different qualifiers
(``(morning)`` against ``(evening)``) are compatible: each holds under its own. A
``confidence`` is the ontology's declared prior that the relation is a genuine
incompatibility (a table, not a calibrated probability). Resolution status records what
the *taxonomy* can say; whether a belief policy adopts one side is a belief-layer matter.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.comparison import normalize
from memoria.core import Digest, Record, UTCDatetime, canonical_json
from memoria.entities import Resolution
from memoria.graph import GraphSnapshot
from memoria.hybrid import entity as key_entity
from memoria.sources import SourceModel, source_class

ContradictionType = Literal[
    "categorical",
    "direct_value",
    "entity_collision",
    "granularity",
    "missing_qualifier",
    "negation",
    "numeric",
    "partial",
    "scope",
    "source_disagreement",
    "supersession",
    "temporal",
]
TYPES: tuple[ContradictionType, ...] = (
    "direct_value",
    "numeric",
    "categorical",
    "temporal",
    "negation",
    "source_disagreement",
    "entity_collision",
    "scope",
    "partial",
    "supersession",
    "granularity",
    "missing_qualifier",
)
Relation = Literal["contradicts", "corrects", "narrows", "qualifies", "supersedes"]
Resolved = Literal[
    "apparent",
    "identity_unresolved",
    "open",
    "resolved_correction",
    "resolved_narrowing",
    "resolved_temporal",
]

# Types that are genuine incompatibilities if nothing else explains them.
GENUINE = frozenset(
    {
        "direct_value",
        "numeric",
        "categorical",
        "temporal",
        "negation",
        "source_disagreement",
        "partial",
    }
)
# Declared priors that a relation of this type is a genuine incompatibility.
CONFIDENCE: dict[str, float] = {
    "direct_value": 0.9,
    "numeric": 0.9,
    "categorical": 0.9,
    "temporal": 0.8,
    "negation": 0.95,
    "source_disagreement": 0.85,
    "entity_collision": 0.5,
    "scope": 0.2,
    "partial": 0.6,
    "supersession": 0.1,
    "granularity": 0.1,
    "missing_qualifier": 0.3,
}

_NUMBER = re.compile(r"^(-?\d+(?:\.\d+)?)\s*([a-z%]*)$")
_TAIL = re.compile(r"\(([^()]*)\)\s*$")
_WINDOW = re.compile(r"^during=(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$")


class ClaimOntology(Record):
    """Declared background knowledge the classifier may use. Nothing is inferred from text."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    hierarchy: tuple[tuple[str, str], ...] = ()  # (narrower value, broader value)
    categories: tuple[tuple[str, tuple[str, ...]], ...] = ()  # attribute -> closed domain
    scopes: tuple[tuple[str, str], ...] = ()  # (child key, parent key)
    numeric_tolerance: float = Field(default=0.0, ge=0, allow_inf_nan=False)  # relative
    concurrency_days: float = Field(default=1.0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        for name in ("hierarchy", "categories", "scopes"):
            value = list(getattr(self, name))
            if value != sorted(set(value)):
                raise ValueError(f"{name} must be unique and sorted")
        return self

    def broader(self, value: str) -> set[str]:
        parents = dict(self.hierarchy)
        out: set[str] = set()
        while value in parents and parents[value] not in out:
            value = parents[value]
            out.add(value)
        return out


@dataclass(frozen=True)
class Assertion:
    """One claim parsed for classification."""

    id: str
    key: str
    entity: str
    attribute: str
    verb: str
    negated: bool
    members: tuple[str, ...]  # normalised values of a (possibly multi-valued) claim
    number: float | None
    unit: str | None
    qualifier: str | None
    window: tuple[datetime, datetime] | None
    occurred_at: datetime
    recorded_at: datetime
    source: str
    evidence: tuple[str, ...]

    @property
    def base(self) -> str:
        return "; ".join(self.members)


def parse_assertion(
    id_: str,
    key: str,
    verb: str,
    raw: str,
    occurred_at: datetime,
    recorded_at: datetime,
    source: str,
    evidence: tuple[str, ...] = (),
) -> Assertion:
    """Parse ``<value>[ (<qualifier>)]`` with ``not `` negation, ``;``-separated members,
    ``<number> <unit>`` numbers and ``(during=YYYY-MM-DD..YYYY-MM-DD)`` windows."""
    text = raw.strip()
    qualifier: str | None = None
    window = None
    tail = _TAIL.search(text)
    if tail:
        text = text[: tail.start()].strip()
        w = _WINDOW.match(tail.group(1).strip())
        if w:
            window = tuple(datetime.fromisoformat(d).replace(tzinfo=UTC) for d in w.groups())
        else:
            qualifier = " ".join(normalize(tail.group(1))) or None
    negated = text.casefold().startswith("not ")
    if negated:
        text = text[4:]
    parts = [m.strip() for m in text.split(";") if m.strip()]
    members = tuple(sorted(" ".join(normalize(m)) for m in parts))
    number = unit = None
    if len(parts) == 1:
        m = _NUMBER.match(parts[0].casefold())
        if m:
            number, unit = float(m.group(1)), m.group(2)
    ent = key_entity(key)
    return Assertion(
        id=id_,
        key=key,
        entity=ent if ent is not None else key,
        attribute=key.split(".", 1)[1] if "." in key else key,
        verb=verb,
        negated=negated,
        members=members,
        number=number,
        unit=unit,
        qualifier=qualifier,
        window=window,  # type: ignore[arg-type]
        occurred_at=occurred_at,
        recorded_at=recorded_at,
        source=source,
        evidence=evidence or (id_,),
    )


class Contradiction(Record):
    """A classified relation between two claims. Neither endpoint is declared false."""

    id: str
    a: str
    b: str  # a occurs no later than b
    type: ContradictionType
    relation: Relation
    source: str  # the relation runs source -> target (a symmetric contradicts: a -> b)
    target: str
    genuine: bool
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[Digest, ...]
    scope: str
    temporal: str  # "concurrent", "sequential" or "overlapping-windows"
    resolution: Resolved
    reason: str


def _rid(a: str, b: str, kind: str) -> str:
    return "contradiction:" + hashlib.sha256(canonical_json([a, b, kind]).encode()).hexdigest()[:20]


def classify(
    a: Assertion,
    b: Assertion,
    ontology: ClaimOntology,
    sources: SourceModel,
    resolution: Resolution | None = None,
) -> Contradiction | None:
    """The relation between two claims, or ``None`` if they are compatible or unrelated.
    Rules are checked in the order of the module docstring's precedence (see code)."""
    if (b.occurred_at, b.id) < (a.occurred_at, a.id):
        a, b = b, a
    gap = abs((b.occurred_at - a.occurred_at).total_seconds()) / 86400
    concurrent = gap <= ontology.concurrency_days

    def make(
        kind: ContradictionType,
        relation: Relation,
        source: Assertion,
        target: Assertion,
        resolved: Resolved,
        why: str,
        scope: str | None = None,
        when: str | None = None,
    ) -> Contradiction:
        return Contradiction(
            id=_rid(a.id, b.id, kind),
            a=a.id,
            b=b.id,
            type=kind,
            relation=relation,
            source=source.id,
            target=target.id,
            genuine=kind in GENUINE,
            confidence=CONFIDENCE[kind],
            evidence=tuple(sorted({*a.evidence, *b.evidence})),
            scope=scope or a.key,
            temporal=when or ("concurrent" if concurrent else "sequential"),
            resolution=resolved,
            reason=why,
        )

    if a.entity != b.entity:
        if resolution is None or a.attribute != b.attribute or a.base == b.base:
            return None
        ids = {f"structured:{a.entity}", f"structured:{b.entity}"}
        candidate = any(
            {d.a, d.b} == ids and d.rule == "similar" and not d.accepted
            for d in resolution.decisions
        )
        if candidate and concurrent and not a.negated and not b.negated:
            return make(
                "entity_collision",
                "contradicts",
                a,
                b,
                "identity_unresolved",
                "unmerged resolution candidates disagree on one attribute",
                scope=f"{a.entity}~{b.entity}.{a.attribute}",
            )
        return None
    if a.key != b.key:
        scoped = {(a.key, b.key): (a, b), (b.key, a.key): (b, a)}
        for (child_key, parent_key), (child, parent) in scoped.items():
            if (child_key, parent_key) in ontology.scopes and child.base != parent.base:
                return make(
                    "scope",
                    "narrows",
                    child,
                    parent,
                    "resolved_narrowing",
                    "a narrower scope holds a different value",
                    scope=f"{child_key}<{parent_key}",
                )
        return None
    if a.verb == "forget" or b.verb == "forget" or not a.members or not b.members:
        return None
    # Same key. Negation.
    if a.negated != b.negated and a.members == b.members:
        if concurrent:
            return make("negation", "contradicts", a, b, "open", "X against not X")
        return make(
            "supersession",
            "supersedes",
            b,
            a,
            "resolved_temporal",
            "a later report replaces the earlier assertion",
        )
    if a.negated or b.negated:
        return None  # "not X" is compatible with any other value Y
    if a.members == b.members:
        return None
    # Qualifiers: distinct qualifiers each hold under their own.
    if a.qualifier and b.qualifier and a.qualifier != b.qualifier:
        return None
    if (a.qualifier is None) != (b.qualifier is None):
        q, plain = (a, b) if a.qualifier else (b, a)
        return make(
            "missing_qualifier",
            "qualifies",
            q,
            plain,
            "apparent",
            "one claim states a qualifier the other omits",
        )
    # Granularity.
    for fine, coarse in ((a, b), (b, a)):
        if (
            len(fine.members) == 1
            and len(coarse.members) == 1
            and coarse.members[0] in (ontology.broader(fine.members[0]))
        ):
            return make(
                "granularity",
                "narrows",
                fine,
                coarse,
                "resolved_narrowing",
                "one value refines the other in the declared hierarchy",
            )
    # Partial overlap of multi-valued claims.
    overlap = set(a.members) & set(b.members)
    if overlap and (len(a.members) > 1 or len(b.members) > 1) and concurrent:
        return make("partial", "contradicts", a, b, "open", "multi-valued claims overlap")
    # Numeric equivalence within tolerance.
    numeric = a.number is not None and b.number is not None and a.unit == b.unit
    if numeric:
        assert a.number is not None
        assert b.number is not None
        scale = max(abs(a.number), abs(b.number))
        if scale == 0 or abs(a.number - b.number) <= ontology.numeric_tolerance * scale:
            return None
    # Explicit windows decide temporal relations.
    if a.window and b.window:
        if a.window[0] < b.window[1] and b.window[0] < a.window[1]:
            return make(
                "temporal",
                "contradicts",
                a,
                b,
                "open",
                "explicit validity windows overlap with different values",
                when="overlapping-windows",
            )
        return make(
            "supersession",
            "supersedes",
            b,
            a,
            "resolved_temporal",
            "disjoint validity windows",
            when="sequential",
        )
    if not concurrent:
        if b.verb == "correct":
            return make(
                "supersession",
                "corrects",
                b,
                a,
                "resolved_correction",
                "a later correction replaces the earlier value",
            )
        return make(
            "supersession",
            "supersedes",
            b,
            a,
            "resolved_temporal",
            "a later value replaces the earlier one",
        )
    if numeric:
        return make("numeric", "contradicts", a, b, "open", "numbers differ beyond tolerance")
    domain = dict(ontology.categories).get(a.attribute)
    if domain and a.base in domain and b.base in domain:
        return make(
            "categorical", "contradicts", a, b, "open", "distinct members of a closed domain"
        )
    ra, rb = sources.root(a.source), sources.root(b.source)
    if ra != rb and source_class(a.source) != source_class(b.source):
        return make(
            "source_disagreement",
            "contradicts",
            a,
            b,
            "open",
            "independent sources of different classes disagree",
        )
    return make("direct_value", "contradicts", a, b, "open", "different values at one time")


class ContradictionCluster(Record):
    """A connected set of claims joined by classified relations."""

    id: str
    claims: tuple[str, ...]
    keys: tuple[str, ...]
    entities: tuple[str, ...]
    first: UTCDatetime
    last: UTCDatetime
    sources: tuple[str, ...]
    roots: tuple[str, ...]
    evidence_volume: int = Field(ge=0)  # experiences involved
    open: tuple[str, ...]  # relation ids the taxonomy leaves open
    resolved: tuple[str, ...]  # relation ids it resolves structurally
    chain: tuple[str, ...]  # claims joined by supersession or correction, oldest first


class ContradictionGraph(Record):
    """The classified relations over one graph snapshot, and their clusters."""

    graph: Digest
    ontology: Digest
    sources: Digest
    contradictions: tuple[Contradiction, ...]  # by id
    clusters: tuple[ContradictionCluster, ...]  # by id
    counts: tuple[tuple[str, int], ...]

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        ids = [c.id for c in self.contradictions]
        if ids != sorted(set(ids)):
            raise ValueError("contradictions must be unique and sorted")
        known = set(ids)
        for cl in self.clusters:
            if not set(cl.open) | set(cl.resolved) <= known:
                raise ValueError(f"{cl.id}: refers to an unknown relation")
        return self


def contradiction_graph(
    snapshot: GraphSnapshot, ontology: ClaimOntology, sources: SourceModel
) -> ContradictionGraph:
    """Classify every pair of observed claims of one key (and candidate collisions across
    keys). Derived from the snapshot alone; deterministic."""
    nodes = {n.id: n for n in snapshot.nodes}
    assertions: list[Assertion] = []
    for gc in snapshot.claims:
        node = nodes[gc.id]
        if gc.status.value != "observed" or gc.object is None or node.occurred_at is None:
            continue
        raw = node.label.split(" = ", 1)[1] if " = " in node.label else gc.object
        assertions.append(
            parse_assertion(
                gc.id,
                gc.key,
                gc.qualifier,
                raw,
                node.occurred_at,
                node.recorded_at or node.occurred_at,
                gc.sources[0] if gc.sources else "",
                gc.evidence,
            )
        )
    assertions.sort(key=lambda x: (x.occurred_at, x.id))
    by_attr: dict[str, list[Assertion]] = {}
    for x in assertions:
        by_attr.setdefault(x.attribute, []).append(x)
    found: dict[str, Contradiction] = {}
    for group in by_attr.values():
        for i, x in enumerate(group):
            for y in group[i + 1 :]:
                if x.entity != y.entity and x.key != y.key and not snapshot.resolution.decisions:
                    continue
                r = classify(x, y, ontology, sources, snapshot.resolution)
                if r is not None:
                    found[r.id] = r
    scoped = [x for x in assertions if any(x.key in pair for pair in ontology.scopes)]
    for i, x in enumerate(scoped):
        for y in scoped[i + 1 :]:
            r = classify(x, y, ontology, sources, snapshot.resolution)
            if r is not None:
                found[r.id] = r
    contradictions = tuple(sorted(found.values(), key=lambda c: c.id))
    by_id = {x.id: x for x in assertions}
    parent = {x.id: x.id for x in assertions}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for c in contradictions:
        ra, rb = find(c.a), find(c.b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    members: dict[str, list[str]] = {}
    for c in contradictions:
        for n in (c.a, c.b):
            members.setdefault(find(n), [])
            if n not in members[find(n)]:
                members[find(n)].append(n)
    clusters = []
    for root, ids in sorted(members.items()):
        xs = sorted((by_id[i] for i in ids), key=lambda x: (x.occurred_at, x.id))
        rels = [c for c in contradictions if find(c.a) == root]
        chain_ids = sorted(
            {n for c in rels if c.relation in ("supersedes", "corrects") for n in (c.a, c.b)},
            key=lambda n: (by_id[n].occurred_at, n),
        )
        srcs = sorted({x.source for x in xs if x.source})
        clusters.append(
            ContradictionCluster(
                id="cluster:"
                + hashlib.sha256(canonical_json(sorted(ids)).encode()).hexdigest()[:16],
                claims=tuple(sorted(ids)),
                keys=tuple(sorted({x.key for x in xs})),
                entities=tuple(sorted({x.entity for x in xs})),
                first=xs[0].occurred_at,
                last=xs[-1].occurred_at,
                sources=tuple(srcs),
                roots=tuple(sorted({sources.root(s) for s in srcs})),
                evidence_volume=len({e for x in xs for e in x.evidence}),
                open=tuple(
                    sorted(c.id for c in rels if c.resolution in ("open", "identity_unresolved"))
                ),
                resolved=tuple(
                    sorted(
                        c.id for c in rels if c.resolution not in ("open", "identity_unresolved")
                    )
                ),
                chain=tuple(chain_ids),
            )
        )
    counts: dict[str, int] = {}
    for c in contradictions:
        counts[c.type] = counts.get(c.type, 0) + 1
    return ContradictionGraph(
        graph=snapshot.digest,
        ontology=ontology.digest,
        sources=sources.digest,
        contradictions=contradictions,
        clusters=tuple(sorted(clusters, key=lambda c: c.id)),
        counts=tuple(sorted(counts.items())),
    )
