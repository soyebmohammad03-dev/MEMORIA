"""Conservative entity identity for free-text claims (Super-Phase 5).

A *registry* declares entities (canonical name, aliases) and pairs declared distinct. A
mention found in text is resolved against it with an explicit outcome:

- ``resolved``: exactly one entity matches by ``exact`` (canonical name) or ``alias``;
- ``ambiguous``: two or more entities match (a shared alias, a canonical name that is also
  another entity's alias, or a similarity tie): never resolved by picking one;
- ``candidates``: no exact or alias match, but near names exist (near-collisions, typos,
  initials). Similarity yields candidates, never identity (I60);
- ``unknown``: nothing matches.

The only baseline that resolves on similarity is the ``similarity`` mode, kept so that false
merges are *measurable* against the conservative rule, not because it is recommended.

Identity assertions (``a`` is the same entity as ``b``, from a source) merge two entities only
with ``min_roots`` independent sources and no declared-distinct pair inside the merged
group; otherwise the decision is ``insufficient`` or ``conflict`` and nothing merges.
Every decision is a record, and a merge that joins truly distinct entities is a *false merge*
(:func:`false_merges`), detectable against a truth partition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.core import Record
from memoria.entities import initials_match, normalize_name, trigram_jaccard

Outcome = Literal["resolved", "ambiguous", "candidates", "unknown"]
Mode = Literal["conservative", "similarity", "exact_only"]
MatchRule = Literal["exact", "alias", "similar"]


class EntityEntry(Record):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()  # normalised, sorted


class Registry(Record):
    """Declared entities and declared-distinct pairs (ids, a < b)."""

    entries: tuple[EntityEntry, ...]
    distinct: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        ids = [e.id for e in self.entries]
        if ids != sorted(set(ids)):
            raise ValueError("entries must be unique and sorted by id")
        if any(
            list(e.aliases) != sorted({normalize_name(a) for a in e.aliases}) for e in self.entries
        ):
            raise ValueError("aliases must be normalised, unique and sorted")
        if list(self.distinct) != sorted(set(self.distinct)) or any(
            a >= b for a, b in self.distinct
        ):
            raise ValueError("distinct pairs must be unique, sorted and ordered (a < b)")
        return self


def make_registry(
    entities: Mapping[str, tuple[str, Iterable[str]]], distinct: Iterable[tuple[str, str]] = ()
) -> Registry:
    """``entities``: id -> (canonical name, aliases). Normalises and sorts."""
    return Registry(
        entries=tuple(
            EntityEntry(id=i, name=c, aliases=tuple(sorted({normalize_name(a) for a in al})))
            for i, (c, al) in sorted(entities.items())
        ),
        distinct=tuple(sorted({(min(a, b), max(a, b)) for a, b in distinct})),
    )


class IdentityPolicy(Record):
    name: str = Field(min_length=1)
    version: str = "1"
    mode: Mode = "conservative"
    near_threshold: float = Field(default=0.5, gt=0, le=1)  # trigram Jaccard for candidates
    min_roots: int = Field(default=2, ge=1)  # independent sources needed to merge entities


CONSERVATIVE = IdentityPolicy(name="conservative")
SIMILARITY = IdentityPolicy(name="similarity", mode="similarity")
EXACT_ONLY = IdentityPolicy(name="exact-only", mode="exact_only")


class Resolved(Record):
    mention: str
    normalized: str
    outcome: Outcome
    entity: str | None  # set iff resolved
    rule: MatchRule | None
    candidates: tuple[str, ...]  # entity ids that matched or were near, sorted
    similarity: float = Field(ge=0, le=1)  # best trigram similarity to any name


def _names(reg: Registry) -> list[tuple[str, str, str]]:
    """(entity id, normalised name, kind) for canonical names and aliases."""
    return sorted(
        (e.id, n, k)
        for e in reg.entries
        for n, k in [(normalize_name(e.name), "exact"), *((a, "alias") for a in e.aliases)]
    )


def resolve_mention(mention: str, reg: Registry, policy: IdentityPolicy = CONSERVATIVE) -> Resolved:
    n = normalize_name(mention)
    names = _names(reg)
    hits = sorted({(i, k) for i, name, k in names if name == n})
    ids = tuple(sorted({i for i, _ in hits}))
    best = max((trigram_jaccard(n, name) for _, name, _ in names), default=0.0)

    def out(
        o: Outcome, entity: str | None, rule: MatchRule | None, cands: tuple[str, ...]
    ) -> Resolved:
        return Resolved(
            mention=mention,
            normalized=n,
            outcome=o,
            entity=entity,
            rule=rule,
            candidates=cands,
            similarity=best,
        )

    if len(ids) == 1:
        return out(
            "resolved", ids[0], "exact" if any(k == "exact" for _, k in hits) else "alias", ids
        )
    if len(ids) > 1:
        return out("ambiguous", None, None, ids)
    if policy.mode == "exact_only":
        return out("unknown", None, None, ())
    near = {
        i
        for i, name, _ in names
        if trigram_jaccard(n, name) >= policy.near_threshold or initials_match(n, name)
    }
    if not near:
        return out("unknown", None, None, ())
    cands = tuple(sorted(near))
    if policy.mode == "similarity":
        scored = sorted(
            {(max(trigram_jaccard(n, name) for j, name, _ in names if j == i), i) for i in near},
            reverse=True,
        )
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            return out("resolved", scored[0][1], "similar", cands)
        return out("ambiguous", None, None, cands)
    return out("candidates", None, None, cands)


class IdentityAssertion(Record):
    a: str
    b: str
    source: str  # a source root: copies must be collapsed by the caller
    evidence: str  # the experience the assertion comes from


class MergeDecision(Record):
    a: str
    b: str  # a < b
    roots: tuple[str, ...]  # distinct independent sources asserting it
    evidence: tuple[str, ...]
    outcome: Literal["merged", "insufficient", "conflict"]
    reason: str


def merge_decisions(
    reg: Registry, assertions: Iterable[IdentityAssertion], policy: IdentityPolicy
) -> tuple[MergeDecision, ...]:
    """One decision per asserted pair. A pair merges with ``min_roots`` independent sources and
    no declared-distinct pair inside the group it would form (single linkage over merged
    pairs, in pair order); a pair that would join distinct entities is a ``conflict``."""
    by_pair: dict[tuple[str, str], list[IdentityAssertion]] = {}
    for x in assertions:
        by_pair.setdefault((min(x.a, x.b), max(x.a, x.b)), []).append(x)
    distinct = set(reg.distinct)
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            a = parent[a]
        return a

    def group(root: str) -> set[str]:
        return {m for m in list(parent) if find(m) == root}

    out: list[MergeDecision] = []
    for (a, b), xs in sorted(by_pair.items()):
        roots = tuple(sorted({x.source for x in xs}))
        ev = tuple(sorted({x.evidence for x in xs}))
        merged = group(find(a)) | group(find(b)) | {a, b}
        clash = any((min(p, q), max(p, q)) in distinct for p in merged for q in merged if p < q)
        if clash:
            oc, why = "conflict", "declared distinct"
        elif len(roots) < policy.min_roots:
            oc, why = "insufficient", f"{len(roots)} of {policy.min_roots} independent sources"
        else:
            oc, why = "merged", "enough independent sources, no declared distinction"
            parent[find(b)] = find(a)
        out.append(MergeDecision(a=a, b=b, roots=roots, evidence=ev, outcome=oc, reason=why))  # type: ignore[arg-type]
    return tuple(out)


def false_merges(decisions: Iterable[MergeDecision], truth: Mapping[str, str]) -> tuple[int, int]:
    """(false, merged): merged decisions joining entities whose true identity differs."""
    merged = [d for d in decisions if d.outcome == "merged"]
    return sum(truth[d.a] != truth[d.b] for d in merged), len(merged)
