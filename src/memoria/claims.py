"""Conservative free-text claim extraction (Super-Phase 5).

``extract`` reads text with a fixed table of deterministic rules (no model, nothing learned)
and returns :class:`Claim` records. Uncertainty is never turned into fact (I59 extended):

- a claim is ``extracted`` only when exactly one rule matches its sentence, the sentence is not
  hedged, its subject is a name (not a pronoun) that :mod:`memoria.identity` resolves to exactly
  one registered entity, and the attribute is known;
- everything claim-like that fails a condition is an ``unresolved`` claim with a closed
  ``reason`` — a first-class object that keeps its text, sentence span and whatever was
  proposed, but has no entity, attribute or value that anything downstream may use;
- a sentence with no claim-like trigger yields nothing.

Every claim keeps the original text digest, its experience, the sentence and (for extracted
claims) the subject and value spans, so ``text[span]`` reproduces the surface it came from
(:func:`verify_claim`). ``confidence`` is the rule's declared prior, not a calibrated
probability. Values are case-folded; kinds are ``assert``, ``correct`` (statement language) and
``negate`` ("no longer lives in X": a retraction of X, never a new value).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from memoria.core import Digest, Record, content_hash
from memoria.identity import CONSERVATIVE, IdentityPolicy, Registry, resolve_mention

Kind = Literal["assert", "correct", "negate"]
Reason = Literal[
    "rule_match",
    "statement",
    "hedged",
    "pronoun_subject",
    "multiple_candidates",
    "no_pattern",
    "ambiguous_entity",
    "candidate_entity",
    "unknown_entity",
    "unknown_attribute",
]
REASONS: tuple[str, ...] = (
    "ambiguous_entity",
    "candidate_entity",
    "hedged",
    "multiple_candidates",
    "no_pattern",
    "pronoun_subject",
    "rule_match",
    "statement",
    "unknown_attribute",
    "unknown_entity",
)
ATTRIBUTES = ("employer", "home")

_NAME = r"[A-Z][\w'-]*(?:\s+[A-Z][\w.'-]*)?"
_VALUE = r"[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*"
_PRONOUNS = frozenset({"he", "she", "they", "it", "someone", "who", "we", "i", "you"})
_HEDGE = re.compile(
    r"\b(?:think|thinks|maybe|might|perhaps|probably|possibly|apparently|supposedly|heard"
    r"|rumou?red|not sure|seems?|guess|allegedly)\b|\?\s*$",
    re.IGNORECASE,
)
_TRIGGER = re.compile(
    r"\b(?:lives?|living|moved|relocated|works?|working|joined|employed|based|left|home)\b",
    re.IGNORECASE,
)
_STATEMENT = re.compile(r"^\s*(set|correct)\s+([a-z0-9_.-]+)\s*=\s*(\S.*?)\s*$")


class Rule(Record):
    name: str
    attribute: str
    kind: Kind
    pattern: str  # groups: subj, val
    confidence: float = Field(gt=0, le=1)  # declared prior for the rule


def _rule(name: str, attribute: str, kind: Kind, verb: str, confidence: float) -> Rule:
    return Rule(
        name=name,
        attribute=attribute,
        kind=kind,
        confidence=confidence,
        pattern=rf"(?P<subj>{_NAME})(?:'s)?\s+{verb}\s+(?P<val>{_VALUE})",
    )


RULES: tuple[Rule, ...] = (
    _rule(
        "employer_negate",
        "employer",
        "negate",
        r"(?:no longer works (?:at|for)|does not work (?:at|for)|doesn't work (?:at|for)|left)",
        0.8,
    ),
    _rule(
        "home_negate",
        "home",
        "negate",
        r"(?:no longer lives in|does not live in|doesn't live in|moved away from)",
        0.8,
    ),
    _rule(
        "employer_assert", "employer", "assert", r"(?:works (?:at|for)|joined|is employed by)", 0.9
    ),
    _rule(
        "home_assert",
        "home",
        "assert",
        r"(?:lives in|is living in|moved to|relocated to|is based in|has home in)",
        0.9,
    ),
    _rule("home_possessive", "home", "assert", r"home is", 0.85),
)


class ExtractorSpec(Record):
    """Identity of the extractor: its rules and its fixed hedge / trigger vocabularies."""

    name: str
    version: str
    rules: tuple[Rule, ...]
    hedge: str
    trigger: str
    pronouns: tuple[str, ...]


EXTRACTOR = ExtractorSpec(
    name="rules",
    version="1",
    rules=RULES,
    hedge=_HEDGE.pattern,
    trigger=_TRIGGER.pattern,
    pronouns=tuple(sorted(_PRONOUNS)),
)
_ENTITY_REASON: dict[str, Reason] = {
    "ambiguous": "ambiguous_entity",
    "candidates": "candidate_entity",
    "unknown": "unknown_entity",
}
_COMPILED = [(r, re.compile(r.pattern)) for r in RULES]


class Claim(Record):
    """One claim-like sentence, extracted or unresolved."""

    id: str
    experience: str  # lineage: the experience the text came from
    text: Digest  # digest of the original text
    sentence: tuple[int, int]  # span in the original text
    status: Literal["extracted", "unresolved"]
    reason: Reason
    kind: Kind | None
    entity: str | None  # set iff extracted
    attribute: str | None  # set iff extracted
    value: str | None  # set iff extracted (case-folded)
    proposed_subject: str | None = None  # what was read, unresolved or not (never used)
    proposed_value: str | None = None
    proposed_kind: Kind | None = None
    subject_span: tuple[int, int] | None
    value_span: tuple[int, int] | None
    rule: str | None
    confidence: float = Field(ge=0, le=1)  # the rule's declared prior; 0 when unresolved
    candidates: tuple[str, ...] = ()  # entity ids for ambiguous / near-collision subjects
    extractor: Digest

    @property
    def key(self) -> str | None:
        return f"{self.entity}.{self.attribute}" if self.status == "extracted" else None


def text_digest(text: str) -> str:
    return content_hash(text)


def _sentences(text: str) -> list[tuple[int, int]]:
    out = []
    for m in re.finditer(r"[^.!?\n]+[.!?]*", text):
        chunk = m.group()
        lead = len(chunk) - len(chunk.lstrip())
        s, e = m.start() + lead, m.start() + len(chunk.rstrip())
        if e > s:
            out.append((s, e))
    return out


def _make(
    experience: str,
    text: str,
    sent: tuple[int, int],
    status: Literal["extracted", "unresolved"],
    reason: Reason,
    *,
    kind: Kind | None = None,
    entity: str | None = None,
    attribute: str | None = None,
    value: str | None = None,
    subject_span: tuple[int, int] | None = None,
    value_span: tuple[int, int] | None = None,
    rule: str | None = None,
    confidence: float = 0.0,
    proposed_subject: str | None = None,
    proposed_value: str | None = None,
    proposed_kind: Kind | None = None,
    candidates: tuple[str, ...] = (),
) -> Claim:
    ident = content_hash([experience, sent, reason, rule])[7:23]
    return Claim(
        id=f"claim:{ident}", experience=experience, text=text_digest(text), sentence=sent,
        status=status, reason=reason, kind=kind, entity=entity, attribute=attribute, value=value,
        proposed_subject=proposed_subject, proposed_value=proposed_value,
        proposed_kind=proposed_kind, subject_span=subject_span, value_span=value_span,
        rule=rule, confidence=confidence, candidates=candidates, extractor=EXTRACTOR.digest,
    )  # fmt: skip


def extract(
    text: str, experience: str, registry: Registry, policy: IdentityPolicy = CONSERVATIVE
) -> tuple[Claim, ...]:
    """Claims of ``text`` in sentence order. Deterministic; see the module docstring."""
    st = _STATEMENT.match(text)
    if st:
        verb, key, val = st.groups()
        ent, _, attr = key.partition(".")
        span = (0, len(text))
        ids = {e.id for e in registry.entries}
        if ent not in ids:
            return (
                _make(
                    experience,
                    text,
                    span,
                    "unresolved",
                    "unknown_entity",
                    proposed_subject=ent,
                    proposed_value=val.casefold(),
                ),
            )
        if attr not in ATTRIBUTES:
            return (
                _make(
                    experience,
                    text,
                    span,
                    "unresolved",
                    "unknown_attribute",
                    proposed_subject=ent,
                    proposed_value=val.casefold(),
                ),
            )
        v0 = st.start(3)
        return (
            _make(
                experience,
                text,
                span,
                "extracted",
                "statement",
                kind="correct" if verb == "correct" else "assert",
                entity=ent,
                attribute=attr,
                value=val.casefold(),
                subject_span=(st.start(2), st.start(2) + len(ent)),
                value_span=(v0, v0 + len(val)),
                rule="statement",
                confidence=1.0,
                proposed_subject=ent,
                proposed_value=val.casefold(),
            ),
        )
    claims: list[Claim] = []
    for sent in _sentences(text):
        chunk = text[sent[0] : sent[1]]
        hits = [(r, m) for r, rx in _COMPILED for m in rx.finditer(chunk)]
        # A longer/earlier match may contain a shorter one; distinct matches are those with
        # different subject starts or different rules over disjoint spans.
        hits = _distinct(hits)
        if not hits:
            if _TRIGGER.search(chunk):
                claims.append(_make(experience, text, sent, "unresolved", "no_pattern"))
            continue
        if len(hits) > 1:
            for r, m in hits:
                claims.append(_unresolved(experience, text, sent, r, m, "multiple_candidates"))
            continue
        r, m = hits[0]
        subj, val = m.group("subj"), m.group("val")
        if _HEDGE.search(chunk):
            claims.append(_unresolved(experience, text, sent, r, m, "hedged"))
        elif subj.split()[0].casefold() in _PRONOUNS:
            claims.append(_unresolved(experience, text, sent, r, m, "pronoun_subject"))
        else:
            res = resolve_mention(subj, registry, policy)
            if res.outcome != "resolved" or res.entity is None:
                reason = _ENTITY_REASON[res.outcome]
                claims.append(
                    _unresolved(experience, text, sent, r, m, reason, candidates=res.candidates)
                )
            else:
                s0, v0 = sent[0] + m.start("subj"), sent[0] + m.start("val")
                claims.append(
                    _make(
                        experience,
                        text,
                        sent,
                        "extracted",
                        "rule_match",
                        kind=r.kind,
                        entity=res.entity,
                        attribute=r.attribute,
                        value=val.casefold(),
                        proposed_subject=subj,
                        proposed_value=val,
                        subject_span=(s0, s0 + len(subj)),
                        value_span=(v0, v0 + len(val)),
                        rule=r.name,
                        confidence=r.confidence,
                    )
                )
    return tuple(claims)


def _distinct(hits: list[tuple[Rule, re.Match[str]]]) -> list[tuple[Rule, re.Match[str]]]:
    """Drop matches whose span lies inside another match's span (overlapping rules)."""
    keep = []
    for i, (r, m) in enumerate(hits):
        inside = any(
            j != i
            and o.start() <= m.start()
            and m.end() <= o.end()
            and (o.end() - o.start()) > (m.end() - m.start())
            for j, (_, o) in enumerate(hits)
        )
        if not inside:
            keep.append((r, m))
    return sorted(keep, key=lambda h: (h[1].start(), h[0].name))


def _unresolved(
    experience: str,
    text: str,
    sent: tuple[int, int],
    r: Rule,
    m: re.Match[str],
    reason: Reason,
    candidates: tuple[str, ...] = (),
) -> Claim:
    return _make(
        experience,
        text,
        sent,
        "unresolved",
        reason,
        proposed_kind=r.kind,
        rule=r.name,
        proposed_subject=m.group("subj"),
        proposed_value=m.group("val"),
        candidates=candidates,
    )


def verify_claim(claim: Claim, text: str) -> list[str]:
    """Problems with a claim's lineage against its original text (empty: complete)."""
    bad = []
    if claim.text != text_digest(text):
        bad.append("text digest differs")
    if claim.status == "unresolved":
        if claim.entity or claim.attribute or claim.value:
            bad.append("unresolved claim carries a usable fact")
        return bad
    if claim.subject_span is None or claim.value_span is None:
        return [*bad, "extracted claim without spans"]
    s, v = claim.subject_span, claim.value_span
    if claim.rule != "statement" and text[s[0] : s[1]] != claim.proposed_subject:
        bad.append("subject span does not reproduce the subject")
    if (claim.value or "") != text[v[0] : v[1]].casefold():
        bad.append("value span does not reproduce the value")
    return bad
