"""The semantic memory graph (Phase 8): a derived, provenance-aware, content-addressed view.

The log and the consolidation hierarchies stay authoritative. A :class:`GraphSnapshot` is
a pure function of (corpus, hierarchies, retrieval traces, run, :class:`GraphPolicy`):
rebuilding it from the same inputs reproduces it byte for byte, and :func:`verify_snapshot`
does exactly that to detect tampering. Nothing is written back.

.. code-block:: text

    source ◄─sourced-from─ experience ─asserts─► claim ─about─► entity ◄─same-entity─ entity
                               ▲                   │  ▲ ─same-event─► event ─valid-during─► interval
                         derived-from     supports │  └─contradicts / supersedes (claims, events)
                               │                   │
                    memory (L1 version) ───────────┘ ─mentions─► entity (surface name)
                      ▲    │ └─related-to[value-mention, inferred]─► claim
       consolidated-from   └─retrieved-by─► retrieval ─related-to[part-of-run]─► run
                      │
    derived (L2..L4) ─asserts─► claim (derived / abstracted) ─derived-from─► observed claim
                      └─related-to[produced-by]─► consolidation   event ─summarized-by─► L4

Every edge names the rule that produced it, the records that justify it (``evidence``) and
an epistemic status. An edge is never stronger than either endpoint, and only edges that
copy a fact from a stored record may be *observed* (:data:`OBSERVED_RULES`): a relation a
rule computes is at best derived, and surface co-mentions and resolution candidates are
*inferred*. Claims exist only where the evidence is structured (the statement language, or
a derived memory's declared key and value); free text is listed as ``unavailable``.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.comparison import normalize
from memoria.consolidation import Hierarchy, Period, timeline
from memoria.core import (
    DerivedMemory,
    Digest,
    EpistemicStatus,
    Inputs,
    Level,
    Record,
    Scalar,
    UTCDatetime,
    canonical_json,
    content_hash,
)
from memoria.entities import Resolution, ResolutionPolicy, normalize_name, resolve, surface_names
from memoria.formation import parse_statement
from memoria.hybrid import RETRIEVABLE, Availability, Corpus, Entry, HybridTrace, entity
from memoria.statistics import Proportion
from memoria.taxonomy import Claim, Relation, relate

GRAPH_SCHEMA: Literal["memoria-graph-v1"] = "memoria-graph-v1"

NodeKind = Literal[
    "claim",
    "consolidation",
    "derived",
    "entity",
    "event",
    "experience",
    "interval",
    "memory",
    "retrieval",
    "run",
    "source",
]
GraphRelation = Literal[
    "about",
    "asserts",
    "consolidated-from",
    "contradicts",
    "derived-from",
    "mentions",
    "related-to",
    "retrieved-by",
    "same-entity",
    "same-event",
    "sourced-from",
    "summarized-by",
    "supersedes",
    "supports",
    "temporally-precedes",
    "valid-during",
]

# Which node kinds each relation connects (source kind, target kind).
ENDPOINTS: dict[str, frozenset[tuple[str, str]]] = {
    "about": frozenset({("claim", "entity"), ("event", "entity")}),
    "asserts": frozenset({("experience", "claim"), ("derived", "claim")}),
    "consolidated-from": frozenset({("derived", "memory")}),
    "contradicts": frozenset({("claim", "claim"), ("event", "event")}),
    "derived-from": frozenset(
        {("memory", "experience"), ("derived", "derived"), ("claim", "claim")}
    ),
    "mentions": frozenset({("memory", "entity")}),
    "related-to": frozenset(
        {
            ("memory", "claim"),
            ("entity", "entity"),
            ("derived", "consolidation"),
            ("retrieval", "run"),
        }
    ),
    "retrieved-by": frozenset({("memory", "retrieval"), ("derived", "retrieval")}),
    "same-entity": frozenset({("entity", "entity")}),
    "same-event": frozenset({("claim", "event")}),
    "sourced-from": frozenset({("experience", "source")}),
    "summarized-by": frozenset({("event", "derived")}),
    "supersedes": frozenset(
        {("memory", "memory"), ("claim", "claim"), ("claim", "event"), ("event", "event")}
    ),
    "supports": frozenset({("memory", "claim")}),
    "temporally-precedes": frozenset({("event", "event")}),
    "valid-during": frozenset(
        {("claim", "interval"), ("event", "interval"), ("derived", "interval")}
    ),
}
# (relation, rule) pairs that copy a fact from a stored record: the only edges that may be
# observed. Everything a rule computes is at best derived.
OBSERVED_RULES = frozenset(
    {
        ("about", "structured-key"),
        ("asserts", "statement"),
        ("derived-from", "cites"),
        ("related-to", "part-of-run"),
        ("retrieved-by", "selected"),
        ("sourced-from", "source-id"),
        ("supersedes", "version-chain"),
        ("supports", "cites-assertion"),
    }
)
# Rules whose edges are inferred (never derived or observed).
INFERRED_RULES = frozenset({"value-mention", "resolution-candidate", "similar"})
_RANK = {
    EpistemicStatus.OBSERVED: 0,
    EpistemicStatus.DERIVED: 1,
    EpistemicStatus.ABSTRACTED: 2,
    EpistemicStatus.INFERRED: 3,
}


def weakest(*statuses: EpistemicStatus) -> EpistemicStatus:
    """The least certain status: observed < derived < abstracted < inferred."""
    return max(statuses, key=lambda s: _RANK[s])


def _h(*parts: object) -> str:
    return hashlib.sha256(canonical_json([str(p) for p in parts]).encode()).hexdigest()[:20]


class GraphIntegrityError(ValueError):
    """A snapshot is inconsistent, or does not match a rebuild from its sources."""


# --- records ------------------------------------------------------------------------------------


class Node(Record):
    id: str = Field(min_length=1)
    kind: NodeKind
    status: EpistemicStatus
    label: str
    occurred_at: UTCDatetime | None = None
    recorded_at: UTCDatetime | None = None
    valid_from: UTCDatetime | None = None
    valid_to: UTCDatetime | None = None
    availability: Availability = "active"
    attrs: Inputs = ()

    def attr(self, name: str) -> Scalar | None:
        return dict(self.attrs).get(name)


class Edge(Record):
    source: str
    target: str
    relation: GraphRelation
    rule: str = Field(min_length=1)
    status: EpistemicStatus
    evidence: tuple[str, ...] = Field(min_length=1)  # records justifying it, sorted

    @property
    def id(self) -> str:
        return f"{self.relation}|{self.source}|{self.target}|{self.rule}"


class GraphClaim(Record):
    """A structured claim: subject, predicate, object, qualifier, interval, sources, status,
    and the memories supporting it. Uncertainty is decomposed into counts of supporting
    and disputing experiences; no scalar confidence is invented (calibration is Phase 12)."""

    id: str
    key: str
    subject: str  # the resolved entity
    predicate: str
    object: str | None  # normalised value; None for a retraction
    qualifier: str  # set | correct | forget | the derived operation
    valid_from: UTCDatetime
    valid_to: UTCDatetime | None
    status: EpistemicStatus
    sources: tuple[str, ...]
    evidence: tuple[Digest, ...]  # experiences asserting it
    supporting: tuple[str, ...]  # memory / derived node ids
    contradicting: tuple[str, ...]  # claim node ids
    support: int = Field(ge=0)
    disputed: int = Field(ge=0)


class GraphPolicy(Record):
    """Everything that determines a graph: entity resolution, claim extraction, temporal
    semantics, inferred links, the view retrieval traverses, and the rebuild frequency."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    resolution: ResolutionPolicy
    claims: Literal["statement-v1"] = "statement-v1"
    temporal: Literal["occurrence-v1"] = "occurrence-v1"
    value_mentions: bool = True  # inferred memory -> claim links from free text
    view: Literal["active", "historical"] = "active"  # what retrieval signals traverse


class GraphSnapshot(Record):
    """An immutable graph artifact. Its digest is the content hash."""

    schema_version: Literal["memoria-graph-v1"] = GRAPH_SCHEMA
    policy: GraphPolicy
    at: UTCDatetime  # the corpus's known_at
    source_state: Digest  # hash of (corpus, hierarchies, forgetting, traces, run)
    corpus: Digest
    hierarchies: tuple[Digest, ...]
    forgetting: Digest | None
    traces: tuple[Digest, ...]
    run: Digest | None
    resolution: Resolution
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    nodes: tuple[Node, ...]  # by id
    edges: tuple[Edge, ...]  # by (source, relation, target, rule)
    claims: tuple[GraphClaim, ...]  # by id
    unavailable: tuple[tuple[str, str], ...]  # (memory node, why no claim), sorted
    broken: tuple[tuple[str, str], ...]  # (node, broken provenance), sorted

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if (self.node_count, self.edge_count) != (len(self.nodes), len(self.edges)):
            raise ValueError("node and edge counts do not match the graph")
        if self.source_state != _source_state(
            self.corpus, self.hierarchies, self.forgetting, self.traces, self.run
        ):
            raise ValueError("source_state does not match the named sources")
        ids = [n.id for n in self.nodes]
        if ids != sorted(set(ids)):
            raise ValueError("nodes must be unique and sorted by id")
        order = [(e.source, e.relation, e.target, e.rule) for e in self.edges]
        if order != sorted(set(order)):
            raise ValueError("edges must be unique and sorted")
        kind = {n.id: n for n in self.nodes}
        for e in self.edges:
            s, t = kind.get(e.source), kind.get(e.target)
            if s is None or t is None:
                raise ValueError(f"edge {e.id}: endpoint is not a node")
            if (s.kind, t.kind) not in ENDPOINTS[e.relation]:
                raise ValueError(f"edge {e.id}: {e.relation} cannot join {s.kind} to {t.kind}")
            if _RANK[e.status] < _RANK[weakest(s.status, t.status)]:
                raise ValueError(f"edge {e.id}: stronger than an endpoint (upgraded status)")
            if e.status is EpistemicStatus.OBSERVED and (e.relation, e.rule) not in OBSERVED_RULES:
                raise ValueError(f"edge {e.id}: a computed relation cannot be observed")
            if e.rule in INFERRED_RULES and e.status is not EpistemicStatus.INFERRED:
                raise ValueError(f"edge {e.id}: {e.rule} links are inferred")
            if list(e.evidence) != sorted(set(e.evidence)):
                raise ValueError(f"edge {e.id}: evidence must be unique and sorted")
        claim_nodes = [n.id for n in self.nodes if n.kind == "claim"]
        if [c.id for c in self.claims] != claim_nodes:
            raise ValueError("one claim record per claim node, in id order")
        for c in self.claims:
            if kind[c.id].status is not c.status:
                raise ValueError(f"{c.id}: claim status differs from its node")
        if list(self.unavailable) != sorted(set(self.unavailable)):
            raise ValueError("unavailable claims must be unique and sorted")
        return self

    def node(self, node_id: str) -> Node:
        return next(n for n in self.nodes if n.id == node_id)


def _source_state(
    corpus: str,
    hierarchies: Sequence[str],
    forgetting: str | None,
    traces: Sequence[str],
    run: str | None,
) -> str:
    return content_hash([corpus, list(hierarchies), forgetting, list(traces), run])


# --- building ------------------------------------------------------------------------------------


def _event_id(key: str, p: Period) -> str:
    """A period's id. A corrected (retracted) period and the period its correction asserts
    can share key, value and start, so the flags and evidence are part of the identity."""
    return "event:" + _h(key, p.value, p.start, p.corrected, p.contested, p.experiences)


def _interval_id(start: datetime, end: datetime | None) -> str:
    return f"interval:{start.isoformat()}/{end.isoformat() if end else 'open'}"


def _attribute(key: str) -> str:
    return key.split(".", 1)[1] if "." in key else key


def _subject_id(key: str) -> str:
    return entity(key) or key


class _Builder:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str, str, str], Edge] = {}
        self.broken: set[tuple[str, str]] = set()

    def node(self, n: Node) -> str:
        known = self.nodes.setdefault(n.id, n)
        if known != n:
            raise GraphIntegrityError(f"node {n.id} built twice with different content")
        return n.id

    def interval(self, start: datetime, end: datetime | None) -> str:
        return self.node(
            Node(
                id=_interval_id(start, end),
                kind="interval",
                status=EpistemicStatus.OBSERVED,
                label=f"{start.date()}..{end.date() if end else ''}",
                valid_from=start,
                valid_to=end,
            )
        )

    def edge(
        self,
        source: str,
        target: str,
        relation: GraphRelation,
        rule: str,
        status: EpistemicStatus,
        evidence: Iterable[str],
    ) -> None:
        if source not in self.nodes or target not in self.nodes:
            self.broken.add((source, f"{relation} to missing {target}"))
            return
        status = weakest(status, self.nodes[source].status, self.nodes[target].status)
        key = (source, relation, target, rule)
        old = self.edges.get(key)
        merged = set(evidence) | (set(old.evidence) if old else set())
        self.edges[key] = Edge(
            source=source,
            target=target,
            relation=relation,
            rule=rule,
            status=status,
            evidence=tuple(sorted(merged)),
        )


def build_graph(
    corpus: Corpus,
    policy: GraphPolicy,
    hierarchies: Sequence[Hierarchy] = (),
    traces: Sequence[HybridTrace] = (),
    run: str | None = None,
) -> GraphSnapshot:
    """Build the graph of everything ``corpus`` holds (L1 and any derived entries), with
    availability as the corpus records it. Pure and deterministic."""
    b = _Builder()
    memo = {id(x): d for d, x in corpus.experiences.items()}

    def dg(r: Record) -> str:  # digests are recomputed on every access; memoise per build
        if id(r) not in memo:
            memo[id(r)] = r.digest
        return memo[id(r)]

    obs = EpistemicStatus.OBSERVED
    der = EpistemicStatus.DERIVED
    l1 = [e for e in corpus.entries if e.level is Level.L1]
    derived = [e for e in corpus.entries if e.level is not Level.L1]

    def node_of(e: Entry) -> str:
        if isinstance(e.version, DerivedMemory):
            return f"derived:{e.version.memory_id}"
        return f"memory:{dg(e.version)}"

    # Evidence: experiences and sources.
    for e in corpus.entries:
        for x in e.sources:
            xid = b.node(
                Node(
                    id=f"experience:{dg(x)}",
                    kind="experience",
                    status=obs,
                    label=x.content[:160],
                    occurred_at=x.occurred_at,
                    attrs=(("source", x.source),),
                )
            )
            cls = x.source.partition(":")[0] if ":" in x.source else ""
            sid = b.node(
                Node(
                    id=f"source:{x.source}",
                    kind="source",
                    status=obs,
                    label=x.source,
                    attrs=(("class", cls),),
                )
            )
            b.edge(xid, sid, "sourced-from", "source-id", obs, [dg(x)])

    # L1 memory versions.
    in_corpus = {dg(e.version) for e in l1}
    for e in l1:
        v = e.version
        start, end = e.interval if e.interval else (v.valid_from, v.valid_to)
        mid = b.node(
            Node(
                id=f"memory:{dg(e.version)}",
                kind="memory",
                status=obs,
                label=v.content or "",
                recorded_at=v.recorded_at,
                valid_from=start,
                valid_to=end,
                availability=e.availability,
                attrs=(("fate", e.fate), ("memory_id", v.memory_id), ("number", v.version)),
            )
        )
        for x in e.sources:
            b.edge(mid, f"experience:{dg(x)}", "derived-from", "cites", obs, [dg(e.version)])
        if e.missing:
            b.broken.add((mid, f"cites {e.missing} experience(s) absent from the corpus"))
        prev = getattr(v, "supersedes", None)
        if prev is not None and prev in in_corpus:
            b.edge(mid, f"memory:{prev}", "supersedes", "version-chain", obs, [dg(e.version)])

    # Observed claims: one per (key, verb, value, instant), from statement experiences.
    groups: dict[str, list[Claim]] = {}
    for c in corpus.claims:
        cid = "claim:" + _h(c.key, c.verb, " ".join(normalize(c.value or "")), c.occurred_at)
        groups.setdefault(cid, []).append(c)
    claim_of_experience = {c.experience: cid for cid, cs in groups.items() for c in cs}

    # Entities: structured identifiers from claim keys, surface names from free text.
    structured: dict[str, set[str]] = {}
    for c in corpus.claims:
        structured.setdefault(_subject_id(c.key), set()).add(c.experience)
    surface: dict[str, set[str]] = {}
    names_of: dict[str, list[str]] = {}
    for e in l1:
        if e.claim is not None:
            continue
        for x in e.sources:
            if parse_statement(x.content) is None:
                for name in surface_names(x.content):
                    surface.setdefault(name, set()).add(dg(x))
                    names_of.setdefault(dg(e.version), []).append(name)
    resolution = resolve(structured, surface, policy.resolution)
    for mention in resolution.mentions:
        b.node(
            Node(
                id=f"entity:{mention.id}",
                kind="entity",
                status=obs if mention.kind == "structured" else der,
                label=mention.normalized,
                attrs=(("entity", resolution.entity_of(mention.id)), ("kind", mention.kind)),
            )
        )
    for d in resolution.decisions:
        if d.accepted:
            status = EpistemicStatus.INFERRED if d.rule == "similar" else der
            b.edge(f"entity:{d.a}", f"entity:{d.b}", "same-entity", d.rule, status, d.evidence)
        elif d.rule == "similar":
            b.edge(
                f"entity:{d.a}",
                f"entity:{d.b}",
                "related-to",
                "resolution-candidate",
                EpistemicStatus.INFERRED,
                d.evidence,
            )

    periods: dict[str, list[Period]] = {}
    for key in sorted({c.key for c in corpus.claims}):
        periods[key] = timeline(corpus.claims, key)
    period_end: dict[str, datetime | None] = {}
    for ps in periods.values():
        for period in ps:
            for xd in period.experiences:
                period_end[xd] = period.end
    for cid, cs in sorted(groups.items()):
        c = cs[0]
        value = " ".join(normalize(c.value or "")) if c.value is not None else None
        b.node(
            Node(
                id=cid,
                kind="claim",
                status=obs,
                label=f"{c.verb} {c.key}" + (f" = {c.value}" if c.value is not None else ""),
                occurred_at=c.occurred_at,
                recorded_at=min(x.recorded_at for x in cs),
                valid_from=c.occurred_at,
                valid_to=period_end.get(c.experience),
                attrs=(("key", c.key), ("value", value or ""), ("verb", c.verb)),
            )
        )
        for rep in cs:
            b.edge(
                f"experience:{rep.experience}", cid, "asserts", "statement", obs, [rep.experience]
            )
        b.edge(
            cid,
            f"entity:structured:{_subject_id(c.key)}",
            "about",
            "structured-key",
            obs,
            [x.experience for x in cs],
        )
        b.edge(
            cid,
            b.interval(c.occurred_at, period_end.get(c.experience)),
            "valid-during",
            "timeline-period",
            der,
            [x.experience for x in cs],
        )
    unavailable: set[tuple[str, str]] = set()
    for e in l1:
        for x in e.sources:
            if dg(x) in claim_of_experience:
                b.edge(
                    f"memory:{dg(e.version)}",
                    claim_of_experience[dg(x)],
                    "supports",
                    "cites-assertion",
                    obs,
                    [dg(e.version)],
                )
        if e.claim is None:
            unavailable.add((f"memory:{dg(e.version)}", e.claim_issue or "unstructured"))
        for name in sorted(set(names_of.get(dg(e.version), []))):
            b.edge(
                f"memory:{dg(e.version)}",
                f"entity:surface:{normalize_name(name)}",
                "mentions",
                "surface-name-v1",
                der,
                [dg(x) for x in e.sources],
            )

    # Claim relations (Phase 4): same-instant disagreement, corrections, retractions.
    by_key: dict[str, list[str]] = {}
    for cid, cs in sorted(groups.items()):
        by_key.setdefault(cs[0].key, []).append(cid)
    for key, cids in by_key.items():
        for i, a in enumerate(cids):
            for bb in cids[i + 1 :]:
                ca, cb = groups[a][0], groups[bb][0]
                if relate(ca, cb) is Relation.CONTRADICTION:
                    ev = [x.experience for x in groups[a] + groups[bb]]
                    b.edge(a, bb, "contradicts", "same-instant-disagreement", der, ev)
        _events(b, key, periods[key], groups, claim_of_experience)

    if policy.value_mentions:
        _value_mentions(b, l1, names_of, resolution, groups, dg)

    # Derived memories: lineage, their declared (or abstracted) claims, their operation.
    by_memory_id = {e.version.memory_id: e for e in derived}
    producing = {dg(m): h for h in hierarchies for m in h.memories}
    derived_claims: dict[str, list[tuple[str, str, str, datetime, datetime | None]]] = {}
    for e in derived:
        m = e.version
        assert isinstance(m, DerivedMemory)
        did = b.node(
            Node(
                id=f"derived:{m.memory_id}",
                kind="derived",
                status=m.status,
                label=m.content,
                recorded_at=m.recorded_at,
                valid_from=m.valid_from,
                valid_to=m.valid_to,
                availability=e.availability,
                attrs=(
                    ("digest", dg(m)),
                    ("level", m.level.value),
                    ("lost", m.lost),
                    ("operation", m.operation),
                    ("rule", m.rule),
                    ("stale", int(e.stale)),
                ),
            )
        )
        h = producing.get(dg(m))
        if h is None:
            b.broken.add((did, "no consolidation record produced it"))
        else:
            hid = b.node(
                Node(
                    id=f"consolidation:{dg(h)}",
                    kind="consolidation",
                    status=obs,
                    label=f"consolidation at {h.at.date()}",
                    recorded_at=h.at,
                    attrs=(("policy", h.policy),),
                )
            )
            b.edge(did, hid, "related-to", "produced-by", m.status, [dg(h)])
        ev = [dg(h) if h else dg(m)]
        if m.level is Level.L2:
            for pv in m.parents:
                b.edge(did, f"memory:{pv}", "consolidated-from", f"grouping:{m.rule}", m.status, ev)
        else:
            for parent_id in m.parents:
                b.edge(did, f"derived:{parent_id}", "derived-from", m.operation, m.status, ev)
        b.edge(
            did,
            b.interval(m.valid_from, m.valid_to),
            "valid-during",
            "declared-interval",
            m.status,
            ev,
        )
        stated = _stated(m, by_memory_id)
        if not stated:
            why = (
                "inferred pattern asserts no claim"
                if m.status is EpistemicStatus.INFERRED
                else "derived from free text: no structured claim"
            )
            unavailable.add((did, why))
        derived_claims[did] = stated

    for did, stated in derived_claims.items():
        dn = b.nodes[did]
        for key, value, parent, start, end in stated:
            cid = "claim:" + _h("derived", did, key, value, start)
            b.node(
                Node(
                    id=cid,
                    kind="claim",
                    status=dn.status,
                    label=f"{key} = {value}",
                    recorded_at=dn.recorded_at,
                    valid_from=start,
                    valid_to=end,
                    attrs=(("key", key), ("value", value), ("verb", str(dn.attr("operation")))),
                )
            )
            rule = "declared-claim" if parent == did else "abstraction"
            b.edge(did, cid, "asserts", rule, dn.status, [str(dn.attr("digest"))])
            b.edge(
                cid,
                f"entity:structured:{_subject_id(key)}",
                "about",
                "structured-key",
                dn.status,
                [str(dn.attr("digest"))],
            )
            b.edge(
                cid,
                b.interval(start, end),
                "valid-during",
                "declared-interval",
                dn.status,
                [str(dn.attr("digest"))],
            )
            for oc in _restated(key, value, parent, by_memory_id, groups, claim_of_experience):
                b.edge(cid, oc, "derived-from", "restates", dn.status, [str(dn.attr("digest"))])
        if dn.attr("operation") == "abstract_timeline":
            keys = {k for k, *_ in stated}
            for key in keys:
                for p in periods.get(key, []):
                    b.edge(
                        _event_id(key, p),
                        did,
                        "summarized-by",
                        "timeline",
                        dn.status,
                        [str(dn.attr("digest"))],
                    )

    # Retrievals and the run.
    by_digest = {dg(e.version): node_of(e) for e in corpus.entries}
    if run is not None:
        b.node(Node(id=f"run:{run}", kind="run", status=obs, label=run))
    for t in traces:
        q = t.query
        rid = b.node(
            Node(
                id=f"retrieval:{dg(t)}",
                kind="retrieval",
                status=obs,
                label=q.text,
                recorded_at=q.known_at,
                valid_from=q.valid_at,
                attrs=(("key", q.key or ""), ("limit", q.limit)),
            )
        )
        for sel in t.selected:
            if sel not in by_digest:
                b.broken.add((rid, f"selected {sel} is not in the corpus"))
                continue
            b.edge(by_digest[sel], rid, "retrieved-by", "selected", obs, [dg(t)])
        if run is not None:
            b.edge(rid, f"run:{run}", "related-to", "part-of-run", obs, [run])

    # Claim records.
    in_edges: dict[str, list[Edge]] = {}
    for edge in b.edges.values():
        in_edges.setdefault(edge.target, []).append(edge)
    contra: dict[str, set[str]] = {}
    for edge in b.edges.values():
        if edge.relation == "contradicts" and b.nodes[edge.source].kind == "claim":
            contra.setdefault(edge.source, set()).add(edge.target)
            contra.setdefault(edge.target, set()).add(edge.source)
    experiences = corpus.experiences
    claims_out = []
    for n in sorted((n for n in b.nodes.values() if n.kind == "claim"), key=lambda n: n.id):
        key, value = str(n.attr("key")), str(n.attr("value"))
        supporters = sorted(
            {
                x.source
                for x in in_edges.get(n.id, [])
                if x.relation in ("supports", "asserts")
                and b.nodes[x.source].kind in ("memory", "derived")
            }
        )
        if n.status is EpistemicStatus.OBSERVED:
            evidence = sorted(c.experience for c in groups[n.id])
        else:
            evidence = sorted(
                {
                    d
                    for x in in_edges.get(n.id, [])
                    if x.relation == "asserts"
                    for d in _lineage_experiences(b, x.source)
                }
            )
        rivals = sorted(contra.get(n.id, ()))
        claims_out.append(
            GraphClaim(
                id=n.id,
                key=key,
                subject=resolution.entity_of(f"structured:{_subject_id(key)}"),
                predicate=_attribute(key),
                object=value or None,
                qualifier=str(n.attr("verb")),
                valid_from=n.valid_from or corpus.known_at,
                valid_to=n.valid_to,
                status=n.status,
                sources=tuple(
                    sorted({experiences[d].source for d in evidence if d in experiences})
                ),
                evidence=tuple(evidence),
                supporting=tuple(supporters),
                contradicting=tuple(rivals),
                support=len(evidence),
                disputed=sum(len(groups[r]) for r in rivals if r in groups),
            )
        )
    hdigests = tuple(dg(h) for h in hierarchies)
    tdigests = tuple(dg(t) for t in traces)
    nodes = tuple(sorted(b.nodes.values(), key=lambda n: n.id))
    edges = tuple(sorted(b.edges.values(), key=lambda e: (e.source, e.relation, e.target, e.rule)))
    return GraphSnapshot(
        policy=policy,
        at=corpus.known_at,
        source_state=_source_state(
            corpus.digest, hdigests, corpus.identity.forgetting, tdigests, run
        ),
        corpus=corpus.digest,
        hierarchies=hdigests,
        forgetting=corpus.identity.forgetting,
        traces=tdigests,
        run=run,
        resolution=resolution,
        node_count=len(nodes),
        edge_count=len(edges),
        nodes=nodes,
        edges=edges,
        claims=tuple(claims_out),
        unavailable=tuple(sorted(unavailable)),
        broken=tuple(sorted(b.broken)),
    )


def _events(
    b: _Builder,
    key: str,
    periods: Sequence[Period],
    groups: Mapping[str, list[Claim]],
    claim_of_experience: Mapping[str, str],
) -> None:
    """Timeline periods as derived events: claims report them; consecutive events are
    ordered; a change or correction supersedes; parallel contested periods contradict."""
    der = EpistemicStatus.DERIVED
    ids = []
    for p in periods:
        eid = b.node(
            Node(
                id=_event_id(key, p),
                kind="event",
                status=der,
                label=f"{key} = {p.value}",
                valid_from=p.start,
                valid_to=p.end,
                attrs=(
                    ("contested", int(p.contested)),
                    ("corrected", int(p.corrected)),
                    ("key", key),
                ),
            )
        )
        ids.append(eid)
        b.edge(
            eid, b.interval(p.start, p.end), "valid-during", "timeline-period", der, p.experiences
        )
        b.edge(
            eid,
            f"entity:structured:{_subject_id(key)}",
            "about",
            "structured-key",
            der,
            p.experiences,
        )
        for x in p.experiences:
            if x in claim_of_experience:
                b.edge(claim_of_experience[x], eid, "same-event", "timeline-period", der, [x])
    order = sorted(range(len(periods)), key=lambda i: (periods[i].start, ids[i]))
    for i, j in itertools.pairwise(order):
        pi, pj = periods[i], periods[j]
        ev = [pi.experiences[0], pj.experiences[0]]
        if pi.start < pj.start:
            b.edge(ids[i], ids[j], "temporally-precedes", "occurrence-order", der, ev)
            if pi.end is not None and pi.end == pj.start:
                b.edge(ids[j], ids[i], "supersedes", "temporal-change", der, ev)
    for i, p in enumerate(periods):
        for j in range(i + 1, len(periods)):
            q = periods[j]
            if p.start == q.start and p.contested and q.contested:
                b.edge(
                    ids[i],
                    ids[j],
                    "contradicts",
                    "contested-period",
                    der,
                    [*p.experiences, *q.experiences],
                )
            if p.corrected != q.corrected and p.start == q.start:
                old, new = (i, j) if p.corrected else (j, i)
                fixes = [
                    x
                    for x in periods[new].experiences
                    if x in claim_of_experience
                    and groups[claim_of_experience[x]][0].verb == "correct"
                ]
                if fixes:
                    b.edge(ids[new], ids[old], "supersedes", "correction", der, fixes)
                    for x in periods[old].experiences:
                        if x in claim_of_experience:
                            b.edge(
                                claim_of_experience[fixes[0]],
                                claim_of_experience[x],
                                "supersedes",
                                "correction",
                                der,
                                [fixes[0], x],
                            )
    for cid, cs in groups.items():
        c = cs[0]
        if c.key == key and c.verb == "forget":
            for i, p in enumerate(periods):
                if p.end == c.occurred_at:
                    b.edge(cid, ids[i], "supersedes", "retraction", der, [c.experience])


def _value_mentions(
    b: _Builder,
    l1: Sequence[Entry],
    names_of: Mapping[str, list[str]],
    resolution: Resolution,
    groups: Mapping[str, list[Claim]],
    dg: Callable[[Record], str],
) -> None:
    """Inferred links from a free-text memory to claims about an entity it names whose
    value it states verbatim (token sequence). Co-mention, not assertion: negation or
    hypotheticals are not detected, so these links are inferred and measured as such."""
    by_entity: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for cid, cs in groups.items():
        c = cs[0]
        if c.value is not None:
            ent = resolution.entity_of(f"structured:{_subject_id(c.key)}")
            by_entity.setdefault(ent, []).append((cid, tuple(normalize(c.value))))
    for e in l1:
        names = names_of.get(dg(e.version))
        if not names:
            continue
        tokens = tuple(normalize(e.version.content or ""))
        ents = {resolution.entity_of(f"surface:{normalize_name(n)}") for n in names}
        for ent in sorted(ents):
            for cid, value in by_entity.get(ent, []):
                if value and any(
                    tokens[i : i + len(value)] == value for i in range(len(tokens) - len(value) + 1)
                ):
                    ev = [dg(x) for x in e.sources] + [x.experience for x in groups[cid]]
                    b.edge(
                        f"memory:{dg(e.version)}",
                        cid,
                        "related-to",
                        "value-mention",
                        EpistemicStatus.INFERRED,
                        ev,
                    )


def _stated(
    m: DerivedMemory, by_memory_id: Mapping[str, Entry]
) -> list[tuple[str, str, str, datetime, datetime | None]]:
    """(key, value, the memory that declares it, start, end) for each claim a derived memory
    asserts: its own declared claim (L2), or its parents' claims restated (L3, L4)."""
    if m.key is not None and m.value is not None:
        return [(m.key, m.value, f"derived:{m.memory_id}", m.valid_from, m.valid_to)]
    if m.status is EpistemicStatus.INFERRED:
        return []
    out = []
    for p in m.parents:
        pe = by_memory_id.get(p)
        pm = pe.version if pe is not None else None
        if isinstance(pm, DerivedMemory) and pm.key is not None and pm.value is not None:
            out.append((pm.key, pm.value, f"derived:{pm.memory_id}", pm.valid_from, pm.valid_to))
    return out


def _restated(
    key: str,
    value: str,
    parent: str,
    by_memory_id: Mapping[str, Entry],
    groups: Mapping[str, list[Claim]],
    claim_of_experience: Mapping[str, str],
) -> list[str]:
    """Observed claims a derived claim restates: its declaring memory's evidence asserting
    the same key and value."""
    e = by_memory_id.get(parent.removeprefix("derived:"))
    if e is None:
        return []
    out = set()
    for d in e.version.derived_from:
        cid = claim_of_experience.get(d)
        if cid is not None:
            c = groups[cid][0]
            if c.key == key and " ".join(normalize(c.value or "")) == value:
                out.add(cid)
    return sorted(out)


def _lineage_experiences(b: _Builder, start: str) -> set[str]:
    out: set[str] = set()
    frontier = [start]
    seen = set(frontier)
    down = ("consolidated-from", "derived-from")
    outgoing: dict[str, list[Edge]] = {}
    for e in b.edges.values():
        if e.relation in down:
            outgoing.setdefault(e.source, []).append(e)
    while frontier:
        n = frontier.pop()
        for e in outgoing.get(n, []):
            if e.target.startswith("experience:"):
                out.add(e.target.removeprefix("experience:"))
            elif e.target not in seen:
                seen.add(e.target)
                frontier.append(e.target)
    return out


def verify_snapshot(
    snapshot: GraphSnapshot,
    corpus: Corpus,
    hierarchies: Sequence[Hierarchy] = (),
    traces: Sequence[HybridTrace] = (),
    run: str | None = None,
) -> None:
    """Rebuild from the named sources and require byte identity; raise with a diagnosis."""
    rebuilt = build_graph(corpus, snapshot.policy, hierarchies, traces, run)
    if rebuilt == snapshot:
        return
    if rebuilt.source_state != snapshot.source_state:
        raise GraphIntegrityError("the snapshot names other sources than those given")
    d = diff_graphs(rebuilt, snapshot)
    raise GraphIntegrityError(
        "snapshot does not match a rebuild from its sources: "
        f"+nodes {list(d.added_nodes)[:3]} -nodes {list(d.removed_nodes)[:3]} "
        f"+edges {list(d.added_edges)[:3]} -edges {list(d.removed_edges)[:3]} "
        f"changed claims {list(d.changed_claims)[:3]}"
    )


# --- diffs --------------------------------------------------------------------------------------

_ENTITY_LINKS = ("about", "mentions", "same-entity")
_PROVENANCE = ("asserts", "consolidated-from", "derived-from", "sourced-from", "supports")


class GraphDiff(Record):
    """What changed between two snapshots. Forgotten nodes still exist (history); they are
    listed apart from removed nodes."""

    before: Digest
    after: Digest
    added_nodes: tuple[str, ...]
    removed_nodes: tuple[str, ...]
    forgotten_nodes: tuple[str, ...]  # retrievable before, not retrievable after
    restored_nodes: tuple[str, ...]
    changed_claims: tuple[str, ...]
    added_contradictions: tuple[str, ...]
    resolved_contradictions: tuple[str, ...]
    changed_entity_links: tuple[str, ...]  # "+edge" / "-edge"
    changed_provenance: tuple[str, ...]  # "+edge" / "-edge"
    added_edges: tuple[str, ...]
    removed_edges: tuple[str, ...]


def diff_graphs(before: GraphSnapshot, after: GraphSnapshot) -> GraphDiff:
    nb, na = {n.id: n for n in before.nodes}, {n.id: n for n in after.nodes}
    eb, ea = {e.id: e for e in before.edges}, {e.id: e for e in after.edges}
    cb, ca = {c.id: c for c in before.claims}, {c.id: c for c in after.claims}
    added, removed = sorted(ea.keys() - eb.keys()), sorted(eb.keys() - ea.keys())

    def rel(ids: Iterable[str], relations: Sequence[str], sign: str) -> list[str]:
        return [sign + i for i in ids if i.split("|", 1)[0] in relations]

    def changed(relations: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted(rel(added, relations, "+") + rel(removed, relations, "-")))

    both = sorted(nb.keys() & na.keys())
    return GraphDiff(
        before=before.digest,
        after=after.digest,
        added_nodes=tuple(sorted(na.keys() - nb.keys())),
        removed_nodes=tuple(sorted(nb.keys() - na.keys())),
        forgotten_nodes=tuple(
            i
            for i in both
            if nb[i].availability in RETRIEVABLE and na[i].availability not in RETRIEVABLE
        ),
        restored_nodes=tuple(
            i
            for i in both
            if nb[i].availability not in RETRIEVABLE and na[i].availability in RETRIEVABLE
        ),
        changed_claims=tuple(sorted(i for i in cb.keys() & ca.keys() if cb[i] != ca[i])),
        added_contradictions=tuple(rel(added, ("contradicts",), "")),
        resolved_contradictions=tuple(rel(removed, ("contradicts",), "")),
        changed_entity_links=changed(_ENTITY_LINKS),
        changed_provenance=changed(_PROVENANCE),
        added_edges=tuple(added),
        removed_edges=tuple(removed),
    )


# --- the retrieval view ---------------------------------------------------------------------------

_CLAIM_LINKS = ("asserts", "related-to", "supports")
_EXPANSION = frozenset(
    {
        "asserts",
        "consolidated-from",
        "contradicts",
        "derived-from",
        "related-to",
        "same-event",
        "supersedes",
        "supports",
    }
)


class MemoryGraph:
    """A snapshot indexed for retrieval (:class:`memoria.hybrid.GraphView`).

    Under the policy's ``active`` view, memories the given availability marks as not
    retrievable are not traversed (so forgotten evidence cannot reach retrieval through
    the graph); the ``historical`` view traverses everything and is the measured
    alternative. ``availability`` maps entry digests to states (the retrieval corpus's).
    """

    def __init__(
        self, snapshot: GraphSnapshot, availability: Mapping[str, Availability] | None = None
    ) -> None:
        self.snapshot = snapshot
        self._node = {n.id: n for n in snapshot.nodes}
        self._edge = {e.id: e for e in snapshot.edges}
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}
        for e in snapshot.edges:
            self._out.setdefault(e.source, []).append(e)
            self._in.setdefault(e.target, []).append(e)
        self._by_version = {
            n.id.removeprefix("memory:"): n.id for n in snapshot.nodes if n.kind == "memory"
        }
        self._by_version |= {
            str(n.attr("digest")): n.id for n in snapshot.nodes if n.kind == "derived"
        }
        state = dict(availability or {})
        self._blocked = (
            {
                nid
                for d, nid in self._by_version.items()
                if state.get(d, self._node[nid].availability) not in RETRIEVABLE
            }
            if snapshot.policy.view == "active"
            else set()
        )
        self._entity = {n.id: str(n.attr("entity")) for n in snapshot.nodes if n.kind == "entity"}

    @property
    def digest(self) -> str:
        return self.snapshot.digest

    def _claim_links(self, version: str) -> list[tuple[Edge, str]]:
        nid = self._by_version.get(version)
        if nid is None:
            return []
        return [
            (e, e.target)
            for e in self._out.get(nid, [])
            if e.relation in _CLAIM_LINKS and self._node[e.target].kind == "claim"
        ]

    def _supported(self, claim: str) -> bool:
        """A claim is visible if some memory supporting it is traversable."""
        return any(
            e.source not in self._blocked
            for e in self._in.get(claim, [])
            if e.relation in _CLAIM_LINKS and self._node[e.source].kind in ("memory", "derived")
        )

    def entity_link(self, version: str, entity_name: str) -> tuple[str, str, str] | None:
        nid = self._by_version.get(version)
        if nid is None:
            return None
        want = next(
            (v for k, v in self._entity.items() if k == f"entity:structured:{entity_name}"), None
        )
        if want is None:
            return None
        found = []
        for e in self._out.get(nid, []):
            if e.relation == "mentions" and self._entity.get(e.target) == want:
                found.append((e.id, e.rule, e.status))
        for link, claim in self._claim_links(version):
            for e in self._out.get(claim, []):
                if e.relation == "about" and self._entity.get(e.target) == want:
                    found.append((e.id, link.rule, weakest(link.status, e.status)))
        best = min(found, key=lambda f: (_RANK[f[2]], f[0]), default=None)
        return None if best is None else (best[0], best[1], best[2].value)

    def claim_link(self, version: str, key: str) -> tuple[str, str, str] | None:
        found = [
            (e.id, e.rule, e.status)
            for e, claim in self._claim_links(version)
            if self._node[claim].attr("key") == key
        ]
        best = min(found, key=lambda f: (_RANK[f[2]], f[0]), default=None)
        return None if best is None else (best[0], best[1], best[2].value)

    def contradiction_edges(self, version: str) -> tuple[str, ...]:
        out = set()
        for _, claim in self._claim_links(version):
            near = [claim] + [
                e.target for e in self._out.get(claim, []) if e.relation == "same-event"
            ]
            for n in near:
                for e in self._out.get(n, []) + self._in.get(n, []):
                    other = e.target if e.source == n else e.source
                    if e.relation == "contradicts" and (
                        self._node[other].kind == "event" or self._supported(other)
                    ):
                        out.add(e.id)
        return tuple(sorted(out))

    def contradicting_memories(self, version: str) -> tuple[str, ...]:
        """Memories (version digests) on the other side of the contradiction edges near
        this memory's claims: the disagreement the graph can show next to it."""
        own = {c for _, c in self._claim_links(version)}
        inverse = {nid: d for d, nid in self._by_version.items()}
        out = set()
        for edge_id in self.contradiction_edges(version):
            e = self._edge[edge_id]
            for side in (e.source, e.target):
                claims = (
                    [side]
                    if self._node[side].kind == "claim"
                    else [x.source for x in self._in.get(side, []) if x.relation == "same-event"]
                )
                for c in claims:
                    if c in own:
                        continue
                    for x in self._in.get(c, []):
                        if (
                            x.relation in _CLAIM_LINKS
                            and x.source in inverse
                            and x.source not in self._blocked
                        ):
                            out.add(inverse[x.source])
        out.discard(version)
        return tuple(sorted(out))

    def expand(self, key: str, hops: int) -> list[tuple[str, int]]:
        seeds = sorted(
            n.id
            for n in self.snapshot.nodes
            if n.kind == "claim" and n.attr("key") == key and self._supported(n.id)
        )
        dist = dict.fromkeys(seeds, 0)
        queue = deque(seeds)
        while queue:
            n = queue.popleft()
            if dist[n] >= hops:
                continue
            edges = self._out.get(n, []) + self._in.get(n, [])
            for e in sorted(edges, key=lambda e: e.id):
                if e.relation not in _EXPANSION:
                    continue
                other = e.target if e.source == n else e.source
                if other in dist or other in self._blocked:
                    continue
                if self._node[other].kind not in ("claim", "event", "memory", "derived"):
                    continue
                dist[other] = dist[n] + 1
                queue.append(other)
        inverse = {nid: d for d, nid in self._by_version.items()}
        return sorted(
            ((inverse[n], d) for n, d in dist.items() if n in inverse), key=lambda x: (x[1], x[0])
        )


# --- analytics ---------------------------------------------------------------------------------


class GraphMetrics(Record):
    """Decomposed structural measurements of one view of a snapshot. There is no single
    quality score; centralities are structural and are not importance."""

    snapshot: Digest
    view: Literal["all", "active"]
    nodes: int
    edges: int
    kinds: tuple[tuple[str, int], ...]
    relations: tuple[tuple[str, int], ...]
    statuses: tuple[tuple[str, int], ...]  # edges by epistemic status
    components: int
    largest_component: int
    degree: tuple[tuple[int, int], ...]  # (degree, nodes)
    contradiction_density: float | None  # contradiction edges per claim
    provenance_depth: tuple[tuple[int, int], ...]  # memories by hops to an experience
    unreachable_evidence: int  # memories / derived memories with no path to an experience
    evidence_coverage: Proportion  # claims asserted by an experience and supported in view
    unsupported_claims: Proportion  # claims no memory in view supports
    entity_ambiguity: Proportion  # surface mentions with unaccepted candidates or no id
    orphans: Proportion  # nodes without edges in view
    temporal_consistency: Proportion  # temporal edges without an error diagnostic
    source_concentration: float | None  # Herfindahl index of claim evidence by source class
    top_source_share: float | None
    memory_centrality: tuple[tuple[str, int], ...]  # top 5 memories by degree
    claim_centrality: tuple[tuple[str, int], ...]  # top 5 claims by degree
    unavailable_nodes: int  # nodes the view drops (forgetting)


def measure_graph(
    s: GraphSnapshot, view: Literal["all", "active"] = "all", confidence: float = 0.95
) -> GraphMetrics:
    drop = {
        n.id
        for n in s.nodes
        if view == "active"
        and n.kind in ("memory", "derived")
        and n.availability not in RETRIEVABLE
    }
    nodes = [n for n in s.nodes if n.id not in drop]
    edges = [e for e in s.edges if e.source not in drop and e.target not in drop]
    kind = {n.id: n.kind for n in nodes}
    deg: dict[str, int] = dict.fromkeys(kind, 0)
    parent = {n: n for n in kind}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in edges:
        deg[e.source] += 1
        deg[e.target] += 1
        ra, rb = find(e.source), find(e.target)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    sizes: dict[str, int] = {}
    for n in kind:
        sizes[find(n)] = sizes.get(find(n), 0) + 1

    def count(xs: Iterable[str]) -> tuple[tuple[str, int], ...]:
        c: dict[str, int] = {}
        for x in xs:
            c[x] = c.get(x, 0) + 1
        return tuple(sorted(c.items()))

    histogram: dict[int, int] = {}
    for d in deg.values():
        histogram[d] = histogram.get(d, 0) + 1
    down: dict[str, list[str]] = {}
    for e in edges:
        if e.relation in ("consolidated-from", "derived-from"):
            down.setdefault(e.source, []).append(e.target)
    depths: dict[int, int] = {}
    unreachable = 0
    for n in (n for n in kind if kind[n] in ("memory", "derived")):
        seen, frontier, depth, found = {n}, [n], 0, None
        while frontier and found is None:
            depth += 1
            nxt = []
            for x in frontier:
                for y in down.get(x, []):
                    if kind.get(y) == "experience":
                        found = depth
                    elif y not in seen:
                        seen.add(y)
                        nxt.append(y)
            frontier = nxt
        if found is None:
            unreachable += 1
        else:
            depths[found] = depths.get(found, 0) + 1
    claims = [n for n in nodes if n.kind == "claim"]
    supported = {
        e.target
        for e in edges
        if e.relation in ("supports", "asserts") and kind.get(e.source) in ("memory", "derived")
    }
    asserted = {
        e.target for e in edges if e.relation == "asserts" and kind.get(e.source) == "experience"
    }
    contradictions = sum(
        1 for e in edges if e.relation == "contradicts" and kind[e.source] == "claim"
    )
    surface = [m for m in s.resolution.mentions if m.kind == "surface"]
    candidates = {x for d in s.resolution.decisions if not d.accepted for x in (d.a, d.b)}
    ambiguous = [
        m
        for m in surface
        if m.id in candidates or s.resolution.entity_of(m.id) in s.resolution.unresolved
    ]
    errors = {d.subject for d in temporal_diagnostics(s) if d.severity == "error"}
    temporal = [
        e for e in edges if e.relation in ("supersedes", "temporally-precedes", "retrieved-by")
    ]
    classes: dict[str, int] = {}
    in_view_claims = {c.id for c in claims}
    source_class = {n.id: str(n.attr("class")) for n in s.nodes if n.kind == "source"}
    for c in s.claims:
        if c.id in in_view_claims:
            for src in c.sources:
                cls = source_class.get(f"source:{src}", "")
                classes[cls] = classes.get(cls, 0) + 1
    total = sum(classes.values())
    hhi = round(sum((v / total) ** 2 for v in classes.values()), 12) if total else None

    def top(k: str) -> tuple[tuple[str, int], ...]:
        return tuple(
            sorted(((n, d) for n, d in deg.items() if kind[n] == k), key=lambda x: (-x[1], x[0]))[
                :5
            ]
        )

    return GraphMetrics(
        snapshot=s.digest,
        view=view,
        nodes=len(nodes),
        edges=len(edges),
        kinds=count(n.kind for n in nodes),
        relations=count(e.relation for e in edges),
        statuses=count(e.status.value for e in edges),
        components=len(sizes),
        largest_component=max(sizes.values(), default=0),
        degree=tuple(sorted(histogram.items())),
        contradiction_density=round(contradictions / len(claims), 12) if claims else None,
        provenance_depth=tuple(sorted(depths.items())),
        unreachable_evidence=unreachable,
        evidence_coverage=Proportion.of(
            sum(1 for c in claims if c.id in asserted and c.id in supported),
            sum(1 for c in claims if c.id in asserted),
            confidence,
        ),
        unsupported_claims=Proportion.of(
            sum(1 for c in claims if c.id not in supported), len(claims), confidence
        ),
        entity_ambiguity=Proportion.of(len(ambiguous), len(surface), confidence),
        orphans=Proportion.of(sum(1 for d in deg.values() if d == 0), len(nodes), confidence),
        temporal_consistency=Proportion.of(
            sum(1 for e in temporal if e.id not in errors), len(temporal), confidence
        ),
        source_concentration=hhi,
        top_source_share=round(max(classes.values()) / total, 12) if total else None,
        memory_centrality=top("memory"),
        claim_centrality=top("claim"),
        unavailable_nodes=len(drop),
    )


# --- temporal and provenance diagnostics ----------------------------------------------------------


class Diagnostic(Record):
    check: str
    severity: Literal["error", "warning", "info"]
    subject: str  # node or edge id
    detail: str


def temporal_diagnostics(s: GraphSnapshot) -> tuple[Diagnostic, ...]:
    """Impossible (error) and suspicious (warning) temporal configurations; by-design
    retroactivity (a correction's period) is reported as info."""
    node = {n.id: n for n in s.nodes}
    out: list[Diagnostic] = []

    def add(
        check: str, severity: Literal["error", "warning", "info"], subject: str, detail: str
    ) -> None:
        out.append(
            Diagnostic(
                check=check,
                severity=severity,
                subject=subject,
                detail=detail,
            )
        )

    down: dict[str, list[str]] = {}
    reported: dict[str, datetime] = {}  # event -> earliest report of it
    correcting: set[str] = set()  # events that replace a corrected period
    for e in s.edges:
        a, b = node[e.source], node[e.target]
        if e.relation == "supersedes" and e.rule == "correction" and b.kind == "event":
            correcting.add(e.source)
        if e.relation == "derived-from" and a.kind == "memory" and b.kind == "experience":
            down.setdefault(a.id, []).append(b.id)
            if a.recorded_at and b.occurred_at and a.recorded_at < b.occurred_at:
                add(
                    "evidence-after-record",
                    "error",
                    e.id,
                    "a memory cites evidence that occurred after the memory was recorded",
                )
        elif e.relation == "supersedes":
            ta = a.recorded_at if a.kind == "memory" else a.valid_from or a.occurred_at
            tb = b.recorded_at if b.kind == "memory" else b.valid_from or b.occurred_at
            if ta and tb and ta < tb and e.rule != "correction":
                add(
                    "supersession-direction",
                    "error",
                    e.id,
                    "the superseding record is earlier than what it supersedes",
                )
            if (
                e.rule in ("correction", "retraction")
                and a.recorded_at
                and b.recorded_at
                and a.recorded_at < b.recorded_at
            ):
                add(
                    "resolution-before-evidence",
                    "warning",
                    e.id,
                    "a correction or retraction was recorded before the claim it resolves",
                )
        elif e.relation == "temporally-precedes":
            if a.valid_from and b.valid_from and a.valid_from >= b.valid_from:
                add("temporal-order", "error", e.id, "precedence contradicts the intervals")
        elif e.relation == "retrieved-by":
            if a.recorded_at and b.recorded_at and a.recorded_at > b.recorded_at:
                add(
                    "future-evidence-in-retrieval",
                    "error",
                    e.id,
                    "a retrieval selected a memory recorded after its known_at",
                )
        elif e.relation == "same-event" and a.valid_from:
            first = reported.get(b.id)
            reported[b.id] = a.valid_from if first is None else min(first, a.valid_from)
        elif e.relation == "contradicts" and a.kind == "claim":
            if a.occurred_at and b.occurred_at and a.occurred_at != b.occurred_at:
                add(
                    "contradiction-timing",
                    "error",
                    e.id,
                    "a same-instant contradiction joins claims from different instants",
                )
    for ev, first in sorted(reported.items()):
        start = node[ev].valid_from
        if start is not None and start < first:
            add(
                "event-before-source",
                "info" if ev in correcting else "warning",
                ev,
                "the event starts before any report of it (retroactive)",
            )
    earliest: dict[str, datetime] = {}
    for m, xs in down.items():
        times = [t for x in xs if (t := node[x].occurred_at) is not None]
        if times:
            earliest[m] = min(times)
    for n in s.nodes:
        if n.kind == "derived" and n.valid_from is not None:
            mems = [
                e.target for e in s.edges if e.source == n.id and e.relation == "consolidated-from"
            ]
            times = [earliest[m] for m in mems if m in earliest]
            if times and n.valid_from < min(times):
                add(
                    "derived-interval-coverage",
                    "warning",
                    n.id,
                    "validity starts before any supporting evidence occurred",
                )
        if (
            n.kind == "claim"
            and n.status is EpistemicStatus.OBSERVED
            and n.recorded_at
            and n.occurred_at
            and n.recorded_at < n.occurred_at
        ):
            add(
                "claim-before-evidence",
                "error",
                n.id,
                "a claim was recorded before the report of it occurred",
            )
    return tuple(sorted(out, key=lambda d: (d.severity, d.check, d.subject)))


# --- provenance traversal (memory autopsy preparation) --------------------------------------------

_DOWN = frozenset(
    {
        "about",
        "consolidated-from",
        "derived-from",
        "mentions",
        "same-event",
        "sourced-from",
        "supports",
        "valid-during",
        "asserts",
    }
)


class ProvenanceStep(Record):
    depth: int = Field(ge=0)
    node: str
    kind: NodeKind
    status: EpistemicStatus
    availability: Availability
    via: str | None  # the edge followed (None for the start)


class ProvenanceTrace(Record):
    """A bounded, deterministic walk from an answer (retrieval) or memory down to evidence:
    retrieval → memory → derived memory → consolidation → claim → entity → experience →
    source → interval. Broken chains are reported, never filled in."""

    graph: Digest
    start: str
    max_depth: int
    max_nodes: int
    steps: tuple[ProvenanceStep, ...]
    experiences: tuple[str, ...]
    sources: tuple[str, ...]
    claims: tuple[str, ...]
    entities: tuple[str, ...]
    intervals: tuple[str, ...]
    consolidations: tuple[str, ...]
    broken: tuple[tuple[str, str], ...]
    inferred_links: int  # inferred edges followed (co-mentions, candidates)
    truncated: bool


def trace_provenance(
    s: GraphSnapshot, start: str, max_depth: int = 10, max_nodes: int = 500
) -> ProvenanceTrace:
    node = {n.id: n for n in s.nodes}
    if start not in node:
        raise GraphIntegrityError(f"{start} is not in the graph")
    out: dict[str, list[Edge]] = {}
    back: dict[str, list[Edge]] = {}
    for e in s.edges:
        out.setdefault(e.source, []).append(e)
        back.setdefault(e.target, []).append(e)

    def follow(n: str) -> list[tuple[Edge, str]]:
        k = node[n].kind
        steps = [(e, e.target) for e in out.get(n, []) if e.relation in _DOWN]
        steps += [
            (e, e.target)
            for e in out.get(n, [])
            if e.relation == "related-to" and e.rule in ("produced-by", "value-mention")
        ]
        if k == "retrieval":
            steps += [(e, e.source) for e in back.get(n, []) if e.relation == "retrieved-by"]
        if k == "claim":
            steps += [
                (e, e.source)
                for e in back.get(n, [])
                if e.relation == "asserts" and node[e.source].kind == "experience"
            ]
        return sorted(steps, key=lambda x: (x[0].relation, x[1]))

    seen = {start: 0}
    steps = [
        ProvenanceStep(
            depth=0,
            node=start,
            kind=node[start].kind,
            status=node[start].status,
            availability=node[start].availability,
            via=None,
        )
    ]
    queue = deque([start])
    truncated = False
    inferred = 0
    while queue:
        n = queue.popleft()
        if seen[n] >= max_depth:
            truncated = truncated or bool(follow(n))
            continue
        for e, m in follow(n):
            if m in seen:
                continue
            if len(seen) >= max_nodes:
                truncated = True
                break
            seen[m] = seen[n] + 1
            inferred += e.status is EpistemicStatus.INFERRED
            nm = node[m]
            steps.append(
                ProvenanceStep(
                    depth=seen[m],
                    node=m,
                    kind=nm.kind,
                    status=nm.status,
                    availability=nm.availability,
                    via=e.id,
                )
            )
            queue.append(m)
    reached = set(seen)
    broken = [(a, why) for a, why in s.broken if a in reached]
    for n in sorted(reached):
        k = node[n].kind
        if k in ("memory", "derived") and not _reaches_experience(n, out, node):
            broken.append((n, "no provenance path to an experience"))
        if node[n].availability not in RETRIEVABLE and k in ("memory", "derived"):
            broken.append((n, f"{node[n].availability} under the forgetting policy"))

    def of(kind: str) -> tuple[str, ...]:
        return tuple(sorted(n for n in reached if node[n].kind == kind))

    return ProvenanceTrace(
        graph=s.digest,
        start=start,
        max_depth=max_depth,
        max_nodes=max_nodes,
        steps=tuple(steps),
        experiences=of("experience"),
        sources=of("source"),
        claims=of("claim"),
        entities=of("entity"),
        intervals=of("interval"),
        consolidations=of("consolidation"),
        broken=tuple(sorted(set(broken))),
        inferred_links=inferred,
        truncated=truncated,
    )


def _reaches_experience(n: str, out: Mapping[str, list[Edge]], node: Mapping[str, Node]) -> bool:
    frontier, seen = [n], {n}
    while frontier:
        x = frontier.pop()
        for e in out.get(x, []):
            if e.relation not in ("consolidated-from", "derived-from"):
                continue
            if node[e.target].kind == "experience":
                return True
            if e.target not in seen:
                seen.add(e.target)
                frontier.append(e.target)
    return False
