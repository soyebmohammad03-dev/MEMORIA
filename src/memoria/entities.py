"""Conservative entity resolution (Phase 8).

A *mention* is a normalised name that evidence uses for something: a structured identifier
(the entity part of a statement key, ``ana`` in ``ana.home``) or a surface name in free
text (a run of capitalised words, ``Alex K.``). Resolution decides which mentions denote
the same entity, one recorded :class:`ResolutionDecision` per candidate pair:

- ``exact``: identical surface strings;
- ``normalized``: identical after NFKC, case folding and punctuation removal (``Ana`` and
  the structured ``ana``);
- ``alias``: an explicit alias table in the policy;
- ``structured``: two mentions of one structured identifier;
- ``similar``: character-trigram Jaccard at or above the policy threshold, or one name's
  tokens being initials/prefixes of the other's (``Alex K.`` / ``Alex Kumar``).

Only the rules a policy lists in ``accept`` merge. By default ``similar`` pairs are
*candidates*: recorded, never merged, because similarity of names is not identity. Every
decision cites the experiences both mentions come from, carries the rule, the similarity
and the policy's declared confidence for that rule (a declared prior, not a calibrated
probability), and is reversible: listing a pair in ``rejected`` removes the merge on the
next resolution. Two distinct structured identifiers in one entity is a *false merge* by
construction (structured identifiers name distinct entities) and is reported.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Literal, Self

from pydantic import Field, model_validator

from memoria.core import Digest, Record

Rule = Literal["alias", "exact", "normalized", "similar", "structured"]
RULES: tuple[Rule, ...] = ("alias", "exact", "normalized", "similar", "structured")

# A run of capitalised words or initials ("Ana", "Alex K.", "Alexander Kumar").
_NAME = re.compile(r"\b[A-Z][\w-]*(?:'s)?(?:\s+(?:[A-Z]\.|[A-Z][\w-]*(?:'s)?))*")
# Capitalised function words that start sentences are not names.
_NOT_NAMES = frozenset(
    {"a", "an", "correction", "he", "i", "it", "note", "she", "that", "the", "these"}
    | {"they", "this", "those", "we"}
)


def normalize_name(text: str) -> str:
    """NFKC, case folding, possessive and punctuation removal, collapsed whitespace."""
    folded = unicodedata.normalize("NFKC", text).casefold().replace("'s", "")
    return " ".join(re.sub(r"[^\w\s]", " ", folded).split())


def surface_names(text: str) -> list[str]:
    """Capitalised name runs in ``text``, possessives stripped (a surface heuristic)."""
    out = []
    for m in _NAME.findall(text):
        words = [w for w in m.replace("'s", "").split() if w]
        while words and words[0].casefold() in _NOT_NAMES:
            words.pop(0)
        if words:
            out.append(" ".join(words))
    return out


def trigram_jaccard(a: str, b: str) -> float:
    def grams(s: str) -> set[str]:
        padded = f"  {s} "
        return {padded[i : i + 3] for i in range(len(padded) - 2)}

    x, y = grams(a), grams(b)
    return round(len(x & y) / len(x | y), 12) if x | y else 0.0


def initials_match(a: str, b: str) -> bool:
    """Whether the shorter name's tokens are prefixes (or initials) of the longer's, in
    order, and they share a first token: ``alex k`` ~ ``alex kumar``."""
    ta, tb = a.split(), b.split()
    if len(ta) > len(tb):
        ta, tb = tb, ta
    if len(ta) < 2 or ta[0] != tb[0] or ta == tb[: len(ta)]:
        return False
    return all(y.startswith(x) for x, y in zip(ta[1:], tb[1:], strict=False))


class Mention(Record):
    """A normalised name and the experiences in which it occurs."""

    id: str  # "<kind>:<normalised>"
    kind: Literal["structured", "surface"]
    normalized: str = Field(min_length=1)
    surfaces: tuple[str, ...]  # the raw strings seen, sorted
    evidence: tuple[Digest, ...] = Field(min_length=1)  # experience digests, sorted


class ResolutionPolicy(Record):
    """Everything that determines a resolution. Its digest is its identity."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    accept: tuple[Rule, ...] = ("alias", "exact", "normalized", "structured")
    threshold: float = Field(default=0.5, gt=0, le=1)  # trigram Jaccard for candidates
    aliases: tuple[tuple[str, str], ...] = ()  # (normalised alias, structured id), sorted
    rejected: tuple[tuple[str, str], ...] = ()  # mention-id pairs never merged, sorted
    confidence: tuple[tuple[Rule, float], ...] = (
        ("alias", 0.95),
        ("exact", 1.0),
        ("normalized", 0.9),
        ("structured", 1.0),
    )  # declared per rule; ``similar`` uses its similarity

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        for name in ("accept", "aliases", "rejected", "confidence"):
            value = list(getattr(self, name))
            if value != sorted(set(value)):
                raise ValueError(f"{name} must be unique and sorted")
        if any(a >= b for a, b in self.rejected):
            raise ValueError("rejected pairs are ordered (a < b)")
        if {r for r, _ in self.confidence} != set(RULES) - {"similar"}:
            raise ValueError("declare a confidence for every identity rule")
        return self


class ResolutionDecision(Record):
    """One candidate pair: which rule matched, the similarity, and whether it merged."""

    a: str
    b: str  # a < b
    rule: Rule
    similarity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)  # the policy's declared confidence for the rule
    accepted: bool
    reason: str  # "accepted" | "candidate_only" | "rejected_by_policy"
    evidence: tuple[Digest, ...]  # experiences of both mentions


class Resolution(Record):
    """Mentions, decisions, and the entities they induce (accepted decisions only)."""

    policy: Digest
    mentions: tuple[Mention, ...]  # by id
    decisions: tuple[ResolutionDecision, ...]  # by (a, b, rule)
    entities: tuple[tuple[str, tuple[str, ...]], ...]  # entity id -> mention ids, sorted
    false_merges: tuple[str, ...]  # entities joining distinct structured identifiers

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        ids = [m.id for m in self.mentions]
        if ids != sorted(set(ids)):
            raise ValueError("mentions must be unique and sorted")
        members = [x for _, ms in self.entities for x in ms]
        if sorted(members) != ids:
            raise ValueError("every mention belongs to exactly one entity")
        return self

    def entity_of(self, mention: str) -> str:
        return next(e for e, ms in self.entities if mention in ms)

    @property
    def unresolved(self) -> tuple[str, ...]:
        """Entities with no structured identifier (known only by surface names)."""
        return tuple(
            e for e, ms in self.entities if not any(m.startswith("structured:") for m in ms)
        )


CONSERVATIVE = ResolutionPolicy(name="conservative", version="1")
AGGRESSIVE = ResolutionPolicy(
    name="aggressive",
    version="1",
    accept=("alias", "exact", "normalized", "similar", "structured"),
)


def resolve(
    structured: Mapping[str, Iterable[str]],
    surface: Mapping[str, Iterable[str]],
    policy: ResolutionPolicy,
) -> Resolution:
    """Resolve mentions. ``structured`` maps identifiers (``ana``) and ``surface`` maps raw
    names (``Ana``) to the experience digests they occur in. Deterministic."""
    found: dict[str, tuple[str, str, set[str], set[str]]] = {}
    for kind, table in (("structured", structured), ("surface", surface)):
        for raw, evidence in table.items():
            norm = raw if kind == "structured" else normalize_name(raw)
            if not norm:
                continue
            mid = f"{kind}:{norm}"
            entry = found.setdefault(mid, (kind, norm, set(), set()))
            entry[2].add(raw)
            entry[3].update(evidence)
    mentions = tuple(
        Mention(
            id=mid,
            kind=k,  # type: ignore[arg-type]
            normalized=n,
            surfaces=tuple(sorted(s)),
            evidence=tuple(sorted(ev)),
        )
        for mid, (k, n, s, ev) in sorted(found.items())
    )
    aliases = dict(policy.aliases)
    confidence = dict(policy.confidence)
    rejected = set(policy.rejected)
    decisions: list[ResolutionDecision] = []
    for i, x in enumerate(mentions):
        for y in mentions[i + 1 :]:
            rule, sim = _match(x, y, aliases, policy.threshold)
            if rule is None:
                continue
            accepted = rule in policy.accept and (x.id, y.id) not in rejected
            reason = (
                "rejected_by_policy"
                if (x.id, y.id) in rejected
                else ("accepted" if accepted else "candidate_only")
            )
            decisions.append(
                ResolutionDecision(
                    a=x.id,
                    b=y.id,
                    rule=rule,
                    similarity=sim,
                    confidence=sim if rule == "similar" else confidence[rule],
                    accepted=accepted,
                    reason=reason,
                    evidence=tuple(sorted(set(x.evidence) | set(y.evidence))),
                )
            )
    parent = {m.id: m.id for m in mentions}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for d in decisions:  # ponytail: single linkage over accepted pairs; the aggressive
        if d.accepted:  # policy's transitive chaining is the failure mode it measures
            ra, rb = find(d.a), find(d.b)
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for m in mentions:
        groups.setdefault(find(m.id), []).append(m.id)
    entities = []
    false_merges = []
    for ms in groups.values():
        ids = sorted(m.removeprefix("structured:") for m in ms if m.startswith("structured:"))
        name = ids[0] if ids else "~" + min(ms).split(":", 1)[1]
        entities.append((f"entity:{name}", tuple(sorted(ms))))
        if len(ids) > 1:
            false_merges.append(f"entity:{name}")
    for a, b in rejected:
        if a in parent and b in parent and find(a) == find(b):
            raise ValueError(f"rejected pair {a} / {b} merged through other decisions")
    return Resolution(
        policy=policy.digest,
        mentions=mentions,
        decisions=tuple(decisions),
        entities=tuple(sorted(entities)),
        false_merges=tuple(sorted(false_merges)),
    )


def _match(
    x: Mention, y: Mention, aliases: Mapping[str, str], threshold: float
) -> tuple[Rule | None, float]:
    if set(x.surfaces) & set(y.surfaces) and x.kind == y.kind == "surface":
        return "exact", 1.0
    if x.normalized == y.normalized:
        return ("structured" if x.kind == y.kind == "structured" else "normalized"), 1.0
    for a, b in ((x, y), (y, x)):
        if b.kind == "structured" and aliases.get(a.normalized) == b.normalized:
            return "alias", 1.0
    sim = trigram_jaccard(x.normalized, y.normalized)
    if sim >= threshold or initials_match(x.normalized, y.normalized):
        return "similar", sim
    return None, sim


def false_merge_rate(resolution: Resolution, truth: Mapping[str, str]) -> tuple[int, int]:
    """(false, merged): accepted decisions joining mentions whose true entities (``truth``:
    mention id -> true entity) differ, over accepted decisions between labelled mentions."""
    merged = [d for d in resolution.decisions if d.accepted and d.a in truth and d.b in truth]
    return sum(truth[d.a] != truth[d.b] for d in merged), len(merged)
